"""
微调效果评测（第5周）

指标（对每条测试样本，比较模型输出的动作序列与标注动作序列）：
  format_rate     输出能解析出至少一个合法动作的比例（格式合格率）
  type_acc        第一个动作的类型正确率（click / double_click / type / key ...）
  first_action_acc 第一个动作完全正确（类型 + 参数：坐标距离 ≤ 阈值、文字一致、按键一致、滚动方向一致）
  step_success    整步成功：动作数相同且每个动作都正确（最严格）
  click_hit@25/50/100  标注第一个动作带坐标时，预测坐标与标注坐标的距离 ≤ 25/50/100 像素（按原截图尺寸换算）
  median_dist_px  坐标距离中位数（像素）
  text_acc        标注含输入动作时，输入文字（去首尾空格、忽略大小写）一致的比例
  key_acc         标注含按键动作时，按键集合一致的比例
  sec_mean        每条样本平均推理耗时

用法：
  # 不需要 GPU 的参照基线：总是输出训练集中最常见的动作（点击训练集点击位置的中位数）
  python -m finetune.evaluate --baseline const --out outputs/eval/const
  # 基座模型零样本（三种提示词各跑一次）
  python -m finetune.evaluate --prompt v0_minimal --out outputs/eval/base_v0
  # 微调后
  python -m finetune.evaluate --adapter outputs/lora_qwen2vl_2b/best --out outputs/eval/lora_v1
  # 汇总对比表 + 柱状图
  python -m finetune.evaluate --compare outputs/eval --report outputs/eval/compare.md

评测会逐条把结果追加写入 <out>/predictions.jsonl，中途中断后重跑会跳过已完成的样本。
"""
import argparse
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional

try:
    from .action_format import POINT_ACTIONS, parse_output
    from .build_dataset import load_jsonl
    from .prompts import DEFAULT_VARIANT
except ImportError:
    from action_format import POINT_ACTIONS, parse_output
    from build_dataset import load_jsonl
    from prompts import DEFAULT_VARIANT

DEFAULT_THRESHOLDS = (25, 50, 100)
MAIN_THRESHOLD = 50


# --------------------------------------------------------------------------- #
# 单条比较
# --------------------------------------------------------------------------- #
def point_distance(a: dict, b: dict, screen_size=(1000, 1000), second: bool = False) -> Optional[float]:
    kx, ky = ("x2", "y2") if second else ("x", "y")
    if kx not in a or ky not in a or kx not in b or ky not in b:
        return None
    w, h = screen_size or (1000, 1000)
    dx = (float(a[kx]) - float(b[kx])) / 1000.0 * w
    dy = (float(a[ky]) - float(b[ky])) / 1000.0 * h
    return math.hypot(dx, dy)


def norm_text(s) -> str:
    return " ".join(str(s or "").strip().lower().split())


def action_match(pred: dict, gt: dict, screen_size=(1000, 1000), threshold: float = MAIN_THRESHOLD) -> bool:
    if not isinstance(pred, dict) or pred.get("action") != gt.get("action"):
        return False
    t = gt["action"]
    if t in POINT_ACTIONS:
        d = point_distance(pred, gt, screen_size)
        if d is None or d > threshold:
            return False
        if t == "drag":
            d2 = point_distance(pred, gt, screen_size, second=True)
            return d2 is not None and d2 <= threshold
        return True
    if t == "scroll":
        return (pred.get("amount", 0) > 0) == (gt.get("amount", 0) > 0)
    if t == "type":
        return norm_text(pred.get("text")) == norm_text(gt.get("text"))
    if t == "key":
        return sorted(pred.get("keys") or []) == sorted(gt.get("keys") or [])
    if t == "wait":
        return True
    return False


def _first_of(actions: List[dict], kind: str) -> Optional[dict]:
    return next((a for a in actions if a.get("action") == kind), None)


def score_sample(pred_actions: List[dict], gt_actions: List[dict], screen_size=(1000, 1000), parsed_ok: bool = True,
                 thresholds=DEFAULT_THRESHOLDS, main_threshold: float = MAIN_THRESHOLD) -> dict:
    if not gt_actions:
        raise ValueError("标注动作为空")
    pred_actions = pred_actions or []
    g0 = gt_actions[0]
    p0 = pred_actions[0] if pred_actions else None
    r = {"format_ok": bool(parsed_ok and pred_actions), "gt_type": g0["action"],
         "pred_type": p0.get("action") if p0 else None}
    r["type_ok"] = bool(p0 and p0.get("action") == g0["action"])
    r["first_ok"] = bool(p0 and action_match(p0, g0, screen_size, main_threshold))
    r["step_ok"] = len(pred_actions) == len(gt_actions) and all(
        action_match(p, g, screen_size, main_threshold) for p, g in zip(pred_actions, gt_actions))
    if g0["action"] in POINT_ACTIONS:
        d = point_distance(p0, g0, screen_size) if p0 else None
        r["dist_px"] = None if d is None else round(d, 2)
        for t in thresholds:
            r[f"hit@{t}"] = d is not None and d <= t
    gt_type = _first_of(gt_actions, "type")
    if gt_type is not None:
        pt = _first_of(pred_actions, "type")
        r["text_ok"] = bool(pt and norm_text(pt.get("text")) == norm_text(gt_type.get("text")))
    gt_key = _first_of(gt_actions, "key")
    if gt_key is not None:
        pk = _first_of(pred_actions, "key")
        r["key_ok"] = bool(pk and sorted(pk.get("keys") or []) == sorted(gt_key.get("keys") or []))
    return r


def _rate(vals) -> Optional[float]:
    vals = list(vals)
    return round(sum(bool(v) for v in vals) / len(vals), 4) if vals else None


def summarize(records: List[dict], thresholds=DEFAULT_THRESHOLDS) -> dict:
    n = len(records)
    if n == 0:
        return {"n": 0}
    s = {"n": n,
         "format_rate": _rate(r["format_ok"] for r in records),
         "type_acc": _rate(r["type_ok"] for r in records),
         "first_action_acc": _rate(r["first_ok"] for r in records),
         "step_success": _rate(r["step_ok"] for r in records)}
    pts = [r for r in records if "dist_px" in r]
    s["n_point"] = len(pts)
    for t in thresholds:
        s[f"click_hit@{t}"] = _rate(r.get(f"hit@{t}") for r in pts)
    dists = [r["dist_px"] for r in pts if r.get("dist_px") is not None]
    s["median_dist_px"] = round(statistics.median(dists), 1) if dists else None
    texts = [r["text_ok"] for r in records if "text_ok" in r]
    keys = [r["key_ok"] for r in records if "key_ok" in r]
    s["n_text"], s["text_acc"] = len(texts), _rate(texts)
    s["n_key"], s["key_acc"] = len(keys), _rate(keys)
    secs = [r["sec"] for r in records if isinstance(r.get("sec"), (int, float))]
    s["sec_mean"] = round(sum(secs) / len(secs), 3) if secs else None
    per = defaultdict(list)
    for r in records:
        per[r["gt_type"]].append(r)
    s["per_type"] = {k: {"n": len(v), "type_acc": _rate(x["type_ok"] for x in v),
                         "first_action_acc": _rate(x["first_ok"] for x in v)}
                     for k, v in sorted(per.items(), key=lambda kv: -len(kv[1]))}
    conf = Counter((r["gt_type"], r["pred_type"] or "无效") for r in records if not r["type_ok"])
    s["top_confusions"] = [{"gt": g, "pred": p, "n": c} for (g, p), c in conf.most_common(8)]
    return s


# --------------------------------------------------------------------------- #
# 基线预测器（不需要 GPU）
# --------------------------------------------------------------------------- #
class ConstPredictor:
    """总是输出训练集里最常见的第一个动作；若是点击类，坐标取训练集该类动作坐标的中位数"""
    name = "const"

    def __init__(self, train_rows: List[dict]):
        if not train_rows:
            raise ValueError("训练集为空，无法构建常数基线")
        first = Counter(r["actions"][0]["action"] for r in train_rows)
        kind = first.most_common(1)[0][0]
        same = [r["actions"][0] for r in train_rows if r["actions"][0]["action"] == kind]
        act = dict(same[0])
        if kind in POINT_ACTIONS:
            act["x"] = int(statistics.median(a["x"] for a in same))
            act["y"] = int(statistics.median(a["y"] for a in same))
        self.action = act

    def predict(self, image, instruction, sub_task=None, prompt=None, screen_size=None) -> dict:
        raw = json.dumps([self.action], ensure_ascii=False)
        return {"raw": raw, "actions": [dict(self.action)], "ok": True, "error": None, "dropped": 0, "sec": 0.0}


class ReplayPredictor:
    """从已有 raw 输出回放（用于离线重新打分 / 单元测试）"""
    name = "replay"

    def __init__(self, outputs: Dict[str, str]):
        self.outputs = outputs

    def predict(self, image, instruction, sub_task=None, prompt=None, screen_size=None, sample_id=None) -> dict:
        raw = self.outputs.get(sample_id, "")
        p = parse_output(raw, screen_size)
        return {"raw": raw, **p, "sec": 0.0}


# --------------------------------------------------------------------------- #
# 批量评测
# --------------------------------------------------------------------------- #
def evaluate(predictor, rows: List[dict], out_dir: Optional[Path] = None, prompt: Optional[str] = None,
             limit: int = 0, log: Callable = print, meta: Optional[dict] = None,
             thresholds=DEFAULT_THRESHOLDS) -> dict:
    rows = rows[:limit] if limit and limit > 0 else rows
    done: Dict[str, dict] = {}
    pred_path = None
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        pred_path = out_dir / "predictions.jsonl"
        if pred_path.exists():
            for rec in load_jsonl(pred_path):
                done[rec["id"]] = rec
    records = []
    f = open(pred_path, "a", encoding="utf-8") if pred_path else None
    t0 = time.time()
    try:
        for i, row in enumerate(rows, 1):
            if row["id"] in done:
                records.append(done[row["id"]])
                continue
            size = tuple(row.get("screen_size") or (1000, 1000))
            kw = {"sample_id": row["id"]} if isinstance(predictor, ReplayPredictor) else {}
            try:
                p = predictor.predict(row["image"], row.get("instruction", ""), row.get("sub_task"), prompt=prompt,
                                      screen_size=size, **kw)
            except Exception as e:  # 单条失败（如显存不足）记为格式失败，不中断整批评测
                p = {"raw": "", "actions": [], "ok": False, "error": f"{type(e).__name__}: {e}", "sec": None}
            rec = {"id": row["id"], "raw": p.get("raw", ""), "pred": p.get("actions", []), "gt": row["actions"],
                   "sec": p.get("sec"), "error": p.get("error"),
                   **score_sample(p.get("actions", []), row["actions"], size, bool(p.get("ok")), thresholds)}
            records.append(rec)
            if f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
            if i % 20 == 0 or i == len(rows):
                ok = sum(r["first_ok"] for r in records)
                log(f"  [{i}/{len(rows)}] 第一个动作正确 {ok}/{len(records)}，已用 {time.time() - t0:.0f}s")
    finally:
        if f:
            f.close()
    summary = summarize(records, thresholds)
    summary.update(meta or {})
    if out_dir is not None:
        (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


# --------------------------------------------------------------------------- #
# 汇总对比
# --------------------------------------------------------------------------- #
COMPARE_COLUMNS = [("format_rate", "格式合格率"), ("type_acc", "动作类型准确率"), ("first_action_acc", "首动作正确率"),
                   ("step_success", "整步成功率"), ("click_hit@50", "点击命中@50px"), ("median_dist_px", "坐标距离中位数(px)"),
                   ("text_acc", "输入文字准确率"), ("key_acc", "按键准确率"), ("sec_mean", "平均耗时(s)")]


def _fmt(v, key):
    if v is None:
        return "—"
    if key in ("median_dist_px", "sec_mean"):
        return f"{v}"
    return f"{v * 100:.1f}%"


def load_summaries(paths) -> List[dict]:
    out = []
    for p in paths:
        p = Path(p)
        files = sorted(p.glob("*/summary.json")) if p.is_dir() else [p]
        for fp in files:
            s = json.loads(fp.read_text(encoding="utf-8"))
            s.setdefault("label", fp.parent.name)
            out.append(s)
    return out


def compare_table(summaries: List[dict]) -> str:
    if not summaries:
        raise ValueError("没有可对比的评测结果")
    head = "| 配置 | 样本数 | " + " | ".join(c[1] for c in COMPARE_COLUMNS) + " |"
    sep = "|" + "---|" * (len(COMPARE_COLUMNS) + 2)
    lines = [head, sep]
    for s in summaries:
        lines.append(f"| {s.get('label')} | {s.get('n')} | " +
                     " | ".join(_fmt(s.get(k), k) for k, _ in COMPARE_COLUMNS) + " |")
    return "\n".join(lines)


def plot_compare(summaries: List[dict], path: Path, keys=("format_rate", "type_acc", "first_action_acc",
                                                         "click_hit@50", "step_success")) -> Optional[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    labels = {k: v for k, v in COMPARE_COLUMNS}
    fig, ax = plt.subplots(figsize=(10, 4.5))
    width = 0.8 / max(1, len(summaries))
    for i, s in enumerate(summaries):
        vals = [(s.get(k) or 0) * 100 for k in keys]
        xs = [j + i * width for j in range(len(keys))]
        ax.bar(xs, vals, width, label=s.get("label"))
    ax.set_xticks([j + width * (len(summaries) - 1) / 2 for j in range(len(keys))])
    ax.set_xticklabels([k for k in keys])
    ax.set_ylabel("%")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8)
    ax.set_title("GUI action generation on ScreenAgent test split")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description="GUI 动作生成评测（微调前后 / 提示词对比）")
    ap.add_argument("--data-dir", default="data_cache/sft")
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", default=None, help="基座模型（默认 Qwen/Qwen2-VL-2B-Instruct）")
    ap.add_argument("--adapter", default=None, help="LoRA 适配器目录；不填为基座零样本")
    ap.add_argument("--prompt", default=DEFAULT_VARIANT)
    ap.add_argument("--baseline", default=None, choices=["const"], help="不加载模型的参照基线")
    ap.add_argument("--out", default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--max-pixels", type=int, default=None)
    ap.add_argument("--min-pixels", type=int, default=None)
    ap.add_argument("--compare", nargs="*", default=None, help="汇总这些目录/summary.json")
    ap.add_argument("--report", default=None, help="--compare 时输出 Markdown 表格路径")
    a = ap.parse_args(argv)

    if a.compare is not None:
        sums = load_summaries(a.compare or ["outputs/eval"])
        table = compare_table(sums)
        print(table)
        if a.report:
            rp = Path(a.report)
            rp.parent.mkdir(parents=True, exist_ok=True)
            rp.write_text(table + "\n", encoding="utf-8")
            png = plot_compare(sums, rp.with_suffix(".png"))
            print(f"已写入 {rp}" + (f" 与 {png}" if png else ""))
        return sums

    rows = load_jsonl(Path(a.data_dir) / f"{a.split}.jsonl")
    if a.baseline == "const":
        predictor = ConstPredictor(load_jsonl(Path(a.data_dir) / "train.jsonl"))
        label = a.label or "const_baseline"
    else:
        from .infer import VLMActor
        from .modeling import DEFAULT_MAX_PIXELS, DEFAULT_MIN_PIXELS, DEFAULT_MODEL
        model = a.model or DEFAULT_MODEL
        max_px = a.max_pixels or DEFAULT_MAX_PIXELS
        min_px = a.min_pixels or min(DEFAULT_MIN_PIXELS, max_px)  # 只调小 max 时 min 跟着降，避免 min > max
        predictor = VLMActor.from_pretrained(model, adapter=a.adapter, prompt=a.prompt, load_4bit=a.load_4bit,
                                             min_pixels=min_px, max_pixels=max_px)
        label = a.label or f"{'lora' if a.adapter else 'base'}_{a.prompt}"
    out = Path(a.out or f"outputs/eval/{label}")
    meta = {"label": label, "prompt": None if a.baseline else a.prompt, "adapter": a.adapter, "split": a.split,
            "model": None if a.baseline else (a.model or "Qwen/Qwen2-VL-2B-Instruct")}
    print(f"[评测] {label}：{len(rows) if not a.limit else min(a.limit, len(rows))} 条 → {out}")
    s = evaluate(predictor, rows, out, prompt=None if a.baseline else a.prompt, limit=a.limit, meta=meta)
    print(json.dumps({k: v for k, v in s.items() if k not in ("per_type", "top_confusions")},
                     ensure_ascii=False, indent=2))
    return s


if __name__ == "__main__":
    main()
