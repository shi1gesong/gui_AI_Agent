"""
公开GUI任务数据集下载与预处理脚本
覆盖：Mind2Web（HuggingFace）、ScreenAgent（GitHub）、WebArena（GitHub，仅任务配置，非完整环境）

统一预处理目标格式（每条样本）：
{
    "task_id": str,
    "instruction": str,          # 自然语言任务指令
    "screenshot_path": str,      # 对应截图路径（若数据集提供）
    "action_sequence": list,     # [{"action_type": ..., "target": ..., ...}, ...]
    "source": str,               # 数据来源标记
}

运行前提：本脚本需要在有公网访问权限的环境执行（WSL2），Hugging Face数据集下载
可能需要科学上网或配置HF镜像（HF_ENDPOINT=https://hf-mirror.com）。
"""
import json
import subprocess
from pathlib import Path

DATA_ROOT = Path("./data_cache")


def _ensure_root() -> Path:
    """需要写文件时才创建缓存目录（原先在 import 时创建，属于模块级副作用）"""
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    return DATA_ROOT


def download_mind2web(sample_limit: int = 200):
    """
    下载 Mind2Web 数据集（HuggingFace: osunlp/Mind2Web）
    该数据集提供网页任务指令 + 动作序列标注，适合做任务规划训练/评测参考
    """
    from datasets import load_dataset

    print("正在下载 Mind2Web 数据集（可能需要几分钟）...")
    try:
        ds = load_dataset("osunlp/Mind2Web", split="train", streaming=True)
    except Exception as e:
        print(f"下载失败，请检查网络或尝试设置镜像 HF_ENDPOINT=https://hf-mirror.com。错误: {e}")
        return []

    samples = []
    for i, item in enumerate(ds):
        if i >= sample_limit:
            break
        raw_actions = item.get("actions", [])
        # Mind2Web原始actions字段自带完整网页raw_html/cleaned_html及候选元素坐标，
        # 体积巨大且对本项目无用，这里只抽取操作类型、目标元素属性、输入值等关键信息
        simplified_actions = []
        for act in raw_actions:
            op = act.get("operation", {})
            pos_candidates = act.get("pos_candidates", [])
            target_tag = pos_candidates[0].get("tag") if pos_candidates else None
            target_attrs = pos_candidates[0].get("attributes") if pos_candidates else None
            simplified_actions.append({
                "op": op.get("op"),
                "value": op.get("value"),
                "target_tag": target_tag,
                "target_attributes": target_attrs,
            })
        samples.append({
            "task_id": item.get("annotation_id", f"mind2web_{i}"),
            "instruction": item.get("confirmed_task", ""),
            "screenshot_path": None,  # Mind2Web原始数据为HTML快照，非截图，需额外渲染
            "action_sequence": simplified_actions,
            "source": "mind2web",
        })

    out_path = _ensure_root() / "mind2web_processed.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"Mind2Web 处理完成，共 {len(samples)} 条样本，保存至 {out_path}")
    return samples


def clone_screenagent_dataset():
    """
    克隆 ScreenAgent 官方仓库（含数据集与标注格式说明）
    仓库地址: https://github.com/niuzaisheng/ScreenAgent
    """
    target_dir = DATA_ROOT / "ScreenAgent"
    if target_dir.exists():
        print(f"ScreenAgent 仓库已存在于 {target_dir}，跳过克隆")
        return target_dir

    _ensure_root()
    print("正在克隆 ScreenAgent 仓库...")
    result = subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/niuzaisheng/ScreenAgent.git", str(target_dir)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"克隆失败: {result.stderr}")
        return None

    print(f"ScreenAgent 仓库克隆完成，位于 {target_dir}")
    print("请查阅仓库内 README 与 data/ 目录，确认具体数据集下载与标注格式（不同版本仓库结构可能有调整）")
    return target_dir


def clone_webarena_repo():
    """
    克隆 WebArena 官方仓库（含任务配置文件 config_files/，非完整可运行环境，
    完整环境需要额外部署Docker容器，此处仅获取任务描述用于规划能力参考）
    仓库地址: https://github.com/web-arena-x/webarena
    """
    target_dir = DATA_ROOT / "webarena"
    if target_dir.exists():
        print(f"WebArena 仓库已存在于 {target_dir}，跳过克隆")
        return target_dir

    _ensure_root()
    print("正在克隆 WebArena 仓库...")
    result = subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/web-arena-x/webarena.git", str(target_dir)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"克隆失败: {result.stderr}")
        return None

    print(f"WebArena 仓库克隆完成，位于 {target_dir}")
    return target_dir


def preprocess_webarena_tasks():
    """从WebArena仓库的config_files目录中提取任务指令，统一成本项目格式
    实际任务json文件位于 config_files/examples/ 子目录下（如 examples/1.json），
    用rglob递归查找以兼容不同版本仓库可能的目录层级差异"""
    config_dir = DATA_ROOT / "webarena" / "config_files"
    if not config_dir.exists():
        print(f"未找到 {config_dir}，请先运行 clone_webarena_repo()")
        return []

    samples = []
    json_files = [f for f in config_dir.rglob("*.json") if f.name != "test.raw.json"]
    for config_file in json_files:
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            samples.append({
                "task_id": cfg.get("task_id", config_file.stem),
                "instruction": cfg.get("intent", ""),
                "screenshot_path": None,
                "action_sequence": [],  # WebArena是交互式评测环境，不提供预标注动作序列
                "source": "webarena",
            })
        except Exception:
            continue

    out_path = _ensure_root() / "webarena_processed.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"WebArena 任务处理完成，共 {len(samples)} 条，保存至 {out_path}")
    return samples


def _convert_screenagent_action(act: dict, screen_w: int, screen_h: int):
    """把 ScreenAgent 的动作格式转换为本项目统一动作格式；Plan/Evaluate 等非执行动作返回 None"""
    kind = act.get("action_type")
    if kind == "MouseAction":
        m_type = act.get("mouse_action_type")
        pos = act.get("mouse_position") or {}
        out = {"button": act.get("mouse_button")}
        if pos:
            x, y = pos.get("width"), pos.get("height")
            out.update({"x": x, "y": y,
                        "x_ratio": round(x / screen_w, 4) if screen_w else None,
                        "y_ratio": round(y / screen_h, 4) if screen_h else None})
        if m_type in ("scroll_up", "scroll_down"):
            n = act.get("scroll_repeat", 1)
            out.update({"action_type": "scroll", "amount": n if m_type == "scroll_up" else -n})
        elif m_type in ("click", "double_click", "move", "drag"):
            out["action_type"] = m_type
        elif m_type in ("down", "up"):
            out["action_type"] = f"mouse_{m_type}"
        else:
            return None
        return {k: v for k, v in out.items() if v is not None}
    if kind == "KeyboardAction":
        if act.get("keyboard_action_type") == "text":
            return {"action_type": "type", "text": act.get("keyboard_text", "")}
        keys = act.get("keyboard_key") or []
        keys = [keys] if isinstance(keys, str) else list(keys)
        return {"action_type": "key", "keys": keys}
    if kind == "WaitAction":
        return {"action_type": "wait", "seconds": act.get("wait_time", 1)}
    return None


def preprocess_screenagent(include_negative: bool = False):
    """
    解析 ScreenAgent 仓库 data/ScreenAgent/{train,test}/<session>/*.json 标注，统一成本项目格式。
    每个 json 是一步交互：任务指令(task_prompt) + 当前截图(images/<saved_image_name>) + 标注动作(actions)。
    - *_neg_plan.json / *_neg_eval.json 是人为构造的负样本，默认跳过
    - 只保留含可执行动作（鼠标/键盘/等待）的样本；PlanAction / EvaluateSubTaskAction 属于规划与评估阶段
    坐标同时保留原始像素与相对屏幕的比例，便于在不同分辨率下复用。
    """
    base = DATA_ROOT / "ScreenAgent" / "data" / "ScreenAgent"
    if not base.exists():
        print(f"未找到 {base}，请先运行 clone_screenagent_dataset()")
        return []

    samples = []
    for f in sorted(base.rglob("*.json")):
        if not include_negative and (f.stem.endswith("_neg_plan") or f.stem.endswith("_neg_eval")):
            continue
        try:
            with open(f, "r", encoding="utf-8") as fp:
                item = json.load(fp)
            if not isinstance(item, dict):
                continue
            w, h = item.get("video_width") or 0, item.get("video_height") or 0
            actions = [a for a in (_convert_screenagent_action(x, w, h) for x in item.get("actions", [])
                                   if isinstance(x, dict)) if a]
            if not actions:
                continue
            img = item.get("saved_image_name")
            samples.append({
                "task_id": f"{item.get('session_id', f.parent.name)}/{f.stem}",
                "instruction": item.get("task_prompt_zh") or item.get("task_prompt") or item.get("task_prompt_en", ""),
                "instruction_en": item.get("task_prompt_en", ""),
                "sub_task": item.get("current_task", ""),
                "screenshot_path": str(f.parent / "images" / img) if img else None,
                "screen_size": [w, h],
                "action_sequence": actions,
                "split": f.relative_to(base).parts[0],
                "source": "screenagent",
            })
        except Exception:
            continue

    out_path = _ensure_root() / "screenagent_processed.jsonl"
    with open(out_path, "w", encoding="utf-8") as fp:
        for s in samples:
            fp.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"ScreenAgent 处理完成，共 {len(samples)} 条，保存至 {out_path}")
    return samples


if __name__ == "__main__":
    print("===== 第3周：公开GUI数据集下载与预处理 =====\n")

    print("--- 1. Mind2Web ---")
    mind2web_samples = download_mind2web(sample_limit=200)

    print("\n--- 2. ScreenAgent ---")
    clone_screenagent_dataset()
    screenagent_samples = preprocess_screenagent()

    print("\n--- 3. WebArena ---")
    clone_webarena_repo()
    webarena_samples = preprocess_webarena_tasks()

    print("\n===== 汇总 =====")
    print(f"Mind2Web 样本数: {len(mind2web_samples)}")
    print(f"WebArena 样本数: {len(webarena_samples)}")
    print(f"ScreenAgent 样本数: {len(screenagent_samples)}")
