"""
构建 LoRA 微调数据集（第5周）

输入：第3周 data/dataset_loader.preprocess_screenagent() 输出的 data_cache/screenagent_processed.jsonl
     （每条 = 一张 ScreenAgent 截图 + 任务指令 + 子任务 + 标注动作序列）
输出：data_cache/sft/{train,val,test}.jsonl + stats.json

每条样本：
    {"id", "session", "image", "instruction", "sub_task", "screen_size", "actions", "target"}
    actions 为 VLM 动作（坐标 0~1000），target 为训练目标文本（紧凑 JSON 数组）

划分规则：
  - test = ScreenAgent 官方 test 划分（282 条，不参与训练和选模型）
  - 官方 train 按“会话”(session) 再切出 val（默认 10%）：同一会话的截图高度相似，按条随机划分会让验证集虚高
  - 同一会话内 (截图, 子任务, 目标) 完全相同的重复样本只保留一条

用法（WSL2 或 Windows 都可以，只用标准库）：
    python -m finetune.build_dataset                      # 已有 screenagent_processed.jsonl
    python -m finetune.build_dataset --prepare            # 先克隆 ScreenAgent 并预处理
"""
import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional

try:
    from .action_format import convert_sequence, to_target_text
except ImportError:  # 直接运行
    from action_format import convert_sequence, to_target_text

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_jsonl(path) -> List[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path} 第 {i} 行不是合法 JSON: {e}") from e
    return out


def write_jsonl(path, rows: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def resolve_image(path: Optional[str], base_dirs: List[Path]) -> Optional[Path]:
    """screenshot_path 是预处理时的相对路径（相对当时的工作目录），依次在几个候选目录下查找"""
    if not path:
        return None
    p = Path(str(path).replace("\\", "/"))
    if p.is_absolute():
        return p if p.exists() else None
    for base in base_dirs:
        cand = base / p
        if cand.exists():
            return cand.resolve()
    return None


def session_of(sample: dict) -> str:
    tid = str(sample.get("task_id", ""))
    return tid.split("/")[0] if "/" in tid else tid


def _bucket(session: str, seed: int) -> float:
    h = hashlib.md5(f"{seed}:{session}".encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def convert_samples(raw: List[dict], base_dirs: List[Path], check_images: bool = True):
    """返回 (样本列表, 丢弃原因计数)"""
    kept, dropped, seen = [], Counter(), set()
    for s in raw:
        size = s.get("screen_size") or [0, 0]
        if not isinstance(size, (list, tuple)) or len(size) != 2 or not size[0] or not size[1]:
            dropped["屏幕尺寸缺失"] += 1
            continue
        actions = convert_sequence(s.get("action_sequence") or [], tuple(size))
        if not actions:
            dropped["没有可训练的动作"] += 1
            continue
        instruction = (s.get("instruction") or s.get("instruction_en") or "").strip()
        sub_task = (s.get("sub_task") or "").strip()
        if not instruction and not sub_task:
            dropped["指令为空"] += 1
            continue
        if check_images:
            img = resolve_image(s.get("screenshot_path"), base_dirs)
            if img is None:
                dropped["截图文件不存在"] += 1
                continue
            image = str(img)
        else:
            image = s.get("screenshot_path")
            if not image:
                dropped["截图文件不存在"] += 1
                continue
        target = to_target_text(actions)
        session = session_of(s)
        key = (session, Path(str(image)).name, sub_task, target)
        if key in seen:
            dropped["重复样本"] += 1
            continue
        seen.add(key)
        kept.append({"id": s.get("task_id"), "session": session, "image": image, "instruction": instruction,
                     "sub_task": sub_task, "screen_size": list(size), "actions": actions, "target": target,
                     "orig_split": s.get("split", "train")})
    return kept, dropped


def split_samples(samples: List[dict], val_ratio: float = 0.1, seed: int = 42) -> Dict[str, List[dict]]:
    if not 0 <= val_ratio < 1:
        raise ValueError(f"val_ratio 必须在 [0, 1) 内: {val_ratio}")
    out = {"train": [], "val": [], "test": []}
    for s in samples:
        if s.get("orig_split") == "test":
            out["test"].append(s)
        elif _bucket(s["session"], seed) < val_ratio:
            out["val"].append(s)
        else:
            out["train"].append(s)
    # 会话很少时可能一个都分不到 val：至少拿出一个会话
    if val_ratio > 0 and not out["val"] and len({s["session"] for s in out["train"]}) > 1:
        sessions = sorted({s["session"] for s in out["train"]}, key=lambda x: _bucket(x, seed))
        pick = sessions[0]
        out["val"] = [s for s in out["train"] if s["session"] == pick]
        out["train"] = [s for s in out["train"] if s["session"] != pick]
    rng = random.Random(seed)
    rng.shuffle(out["train"])
    return out


def stats_of(splits: Dict[str, List[dict]], dropped: Counter) -> dict:
    st = {"dropped": dict(dropped)}
    for name, rows in splits.items():
        first = Counter(r["actions"][0]["action"] for r in rows)
        allc = Counter(a["action"] for r in rows for a in r["actions"])
        st[name] = {"samples": len(rows), "sessions": len({r["session"] for r in rows}),
                    "first_action": dict(first.most_common()), "all_actions": dict(allc.most_common()),
                    "avg_actions": round(sum(len(r["actions"]) for r in rows) / len(rows), 3) if rows else 0.0,
                    "multi_action_ratio": round(sum(len(r["actions"]) > 1 for r in rows) / len(rows), 4)
                    if rows else 0.0}
    tr, va, te = ({r["session"] for r in splits[k]} for k in ("train", "val", "test"))
    st["session_overlap"] = {"train_val": len(tr & va), "train_test": len(tr & te), "val_test": len(va & te)}
    return st


def build(processed: Path, out_dir: Path, val_ratio: float = 0.1, seed: int = 42,
          base_dirs: Optional[List[Path]] = None, check_images: bool = True) -> dict:
    processed = Path(processed)
    if not processed.exists():
        raise FileNotFoundError(f"找不到 {processed}，请先运行 python -m finetune.build_dataset --prepare")
    raw = load_jsonl(processed)
    bases = base_dirs or [Path.cwd(), processed.parent.parent, processed.parent, REPO_ROOT]
    samples, dropped = convert_samples(raw, bases, check_images=check_images)
    splits = split_samples(samples, val_ratio=val_ratio, seed=seed)
    out_dir = Path(out_dir)
    for name, rows in splits.items():
        write_jsonl(out_dir / f"{name}.jsonl", rows)
    st = stats_of(splits, dropped)
    st.update(source=str(processed), raw=len(raw), kept=len(samples), val_ratio=val_ratio, seed=seed)
    (out_dir / "stats.json").write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")
    return st


def main(argv=None):
    ap = argparse.ArgumentParser(description="构建 LoRA 微调数据集（ScreenAgent → train/val/test）")
    ap.add_argument("--processed", default="data_cache/screenagent_processed.jsonl")
    ap.add_argument("--out", default="data_cache/sft")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prepare", action="store_true", help="先克隆 ScreenAgent 仓库并运行第3周预处理")
    args = ap.parse_args(argv)
    if args.prepare:
        from data import dataset_loader as dl
        dl.clone_screenagent_dataset()
        test_zip = dl.DATA_ROOT / "ScreenAgent" / "data" / "ScreenAgent" / "test.zip"
        if test_zip.exists() and not (test_zip.parent / "test").exists():
            import zipfile
            print("解压 ScreenAgent 测试集 test.zip ...")
            with zipfile.ZipFile(test_zip) as z:
                z.extractall(test_zip.parent)
        dl.preprocess_screenagent()
    st = build(Path(args.processed), Path(args.out), val_ratio=args.val_ratio, seed=args.seed)
    print(json.dumps({k: (v["samples"] if isinstance(v, dict) and "samples" in v else v)
                      for k, v in st.items()}, ensure_ascii=False, indent=2))
    print(f"已写入 {args.out}/train.jsonl、val.jsonl、test.jsonl、stats.json")
    return st


if __name__ == "__main__":
    main()
