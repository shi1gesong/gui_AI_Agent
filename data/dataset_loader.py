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
DATA_ROOT.mkdir(exist_ok=True)


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

    out_path = DATA_ROOT / "mind2web_processed.jsonl"
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

    out_path = DATA_ROOT / "webarena_processed.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"WebArena 任务处理完成，共 {len(samples)} 条，保存至 {out_path}")
    return samples


if __name__ == "__main__":
    print("===== 第3周：公开GUI数据集下载与预处理 =====\n")

    print("--- 1. Mind2Web ---")
    mind2web_samples = download_mind2web(sample_limit=200)

    print("\n--- 2. ScreenAgent ---")
    clone_screenagent_dataset()

    print("\n--- 3. WebArena ---")
    clone_webarena_repo()
    webarena_samples = preprocess_webarena_tasks()

    print("\n===== 汇总 =====")
    print(f"Mind2Web 样本数: {len(mind2web_samples)}")
    print(f"WebArena 样本数: {len(webarena_samples)}")
    print("ScreenAgent: 已克隆仓库，需人工查阅具体数据集格式后按需扩展预处理逻辑")
