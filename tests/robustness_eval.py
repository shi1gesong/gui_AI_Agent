"""
第6周鲁棒性评测：在带故障注入的模拟桌面（tests/fault_desktop.py）上对比 v1.0 与 v2.0

  任务：第4周 5 个基础任务 T1~T5 + 第6周复杂任务 C1~C4（共 9 个）
  故障：none + 11 种（见 fault_desktop.FAULTS）
  种子：每个 (任务, 故障) 跑 seeds 次（OCR 随机故障用不同随机数）
  v1 与 v2 用同一个模拟桌面、同一个规则决策、同一份子任务（v1 直接使用 v2 拆解出的子任务，只比较执行阶段的容错）

运行（仓库根目录，任何机器都可以，不动真实鼠标）：
    python tests/robustness_eval.py                         # 默认 3 个种子，结果写到 report/week6_assets/
    python tests/robustness_eval.py --seeds 5 --faults none,popup_on_select
"""
import argparse
import csv
import json
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for p in (REPO, REPO / "agent", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from agent.agent_v2 import GUIAgentV2  # noqa: E402
from agent.basic_tasks import TaskContext, build_tasks, run_task  # noqa: E402
from agent.complex_tasks import build_complex_tasks  # noqa: E402
from agent.decomposer import Decomposer  # noqa: E402
from agent.executor import ControllerExecutor  # noqa: E402
from agent.gui_agent import GUIAgent  # noqa: E402
from agent.planner import HybridPlanner, RulePlanner  # noqa: E402
from fault_desktop import FAULTS, FaultyDesktop  # noqa: E402
from perception.perceiver_v2 import PerceiverV2  # noqa: E402
from perception.screen_parser import ScreenPerceiver  # noqa: E402

TASK_IDS = ["T1", "T2", "T3", "T4", "T5", "C1", "C2", "C3", "C4"]


class DownLLM:
    """Ollama 没启动：每次调用都连接失败"""

    def __init__(self):
        self.calls = 0

    def generate(self, prompt, image_path=None):
        self.calls += 1
        raise ConnectionError("Ollama 连接失败: Connection refused (localhost:11434)")


def make_brain(llm):
    from base_agent import BaseGUIAgent, LangChainLLMWrapper
    b = BaseGUIAgent(llm_mode="api")
    b.llm = LangChainLLMWrapper(inner_llm=llm)
    return b


def build(engine: str, fault: str, seed: int, tmp: Path, quiet: bool = True):
    faults = () if fault in ("none", "llm_down") else (fault,)
    sim = FaultyDesktop(tmp, faults=faults, seed=seed)
    apps = sim.app_manager()
    log = (lambda *a, **k: None) if quiet else print
    brain = make_brain(DownLLM()) if fault == "llm_down" else None
    rule = RulePlanner(app_resolver=apps.resolve_app)
    executor = ControllerExecutor(sim, app_manager=apps, sleep=sim.sleep)
    if engine == "v1":
        perceiver = ScreenPerceiver(capture=sim, ocr=sim, logical_size_fn=sim.logical_size, region_fn=apps.screen_regions)
        planner = HybridPlanner("auto" if brain else "rule", rule, llm_plan=brain.plan if brain else None)
        agent = GUIAgent(perceiver, executor, planner=planner, brain=brain, act_mode="hybrid" if brain else "rule",
                         sleep=sim.sleep, log=log, trace_dir=None)
    else:
        perceiver = PerceiverV2(capture=sim, ocr=sim, logical_size_fn=sim.logical_size, region_fn=apps.screen_regions,
                                save_frames=False)
        dec = Decomposer(rule, llm_plan=brain.plan if brain else None, mode="auto" if brain else "rule")
        agent = GUIAgentV2(perceiver, executor, decomposer=dec, brain=brain, act_mode="hybrid" if brain else "rule",
                           app_manager=apps, sleep=sim.sleep, log=log, trace_dir=str(tmp / "trace"), settle_sec=0.2)
    ctx = TaskContext(apps, tmp / "work", sim.chat_log, sleep=sim.sleep)
    return agent, sim, ctx


def run_case(engine: str, task_id: str, fault: str, seed: int, tmp: Path, quiet: bool = True) -> dict:
    agent, sim, ctx = build(engine, fault, seed, tmp, quiet)
    tasks = {t.id: t for t in build_tasks(ctx.work_dir) + build_complex_tasks(ctx.work_dir)}
    task = tasks[task_id]
    t0 = time.perf_counter()
    if engine == "v1":
        # v1 的模板规划器不认识复杂任务的写法：使用同一份拆解结果，只比较执行
        plan = Decomposer(RulePlanner(app_resolver=ctx.apps.resolve_app)).decompose(task.instruction).to_plan()
        orig = agent.run
        agent.run = lambda ins, checker=None: orig(ins, checker=checker, plan=plan)
    try:
        res = run_task(agent, task, ctx)
    except Exception as e:  # 评测框架本身不应因单个用例崩溃
        res = {"success": False, "error": f"{type(e).__name__}: {e}", "steps": []}
    steps = res.get("steps", [])
    rb = res.get("robustness") or {}
    return {"engine": engine, "task": task_id, "fault": fault, "seed": seed, "success": bool(res.get("success")),
            "check_ok": bool((res.get("check") or {}).get("ok")), "agent_complete":
            bool((res.get("reflection") or {}).get("is_complete")),
            "steps": len(steps), "attempts": sum(len(s.get("attempts", [])) for s in steps),
            "retries": rb.get("retries", sum(max(0, len(s.get("attempts", [])) - 1) for s in steps)),
            "recoveries": rb.get("recoveries", 0), "goal_replays": rb.get("goal_replays", 0),
            "virtual_sec": round(sim.t, 2), "wall_sec": round(time.perf_counter() - t0, 3),
            "ocr_calls": sim.ocr_calls, "typed_to_nowhere": len(sim.typed_to_nowhere),
            "chat_launches": sum(1 for c in sim.calls if c[0] == "spawn" and "mock_chat.py" in c[1]),
            "false_success": bool((res.get("reflection") or {}).get("is_complete")) and not
            bool((res.get("check") or {}).get("ok")),
            "error": (res.get("error") or next((s.get("error") for s in steps if not s.get("ok")), "")
                      or ("" if (res.get("check") or {}).get("ok") else (res.get("check") or {}).get("message", "")))[:160]}


def evaluate(faults=FAULTS, tasks=TASK_IDS, seeds=3, engines=("v1", "v2"), quiet=True, progress=None):
    rows = []
    with tempfile.TemporaryDirectory() as d:
        n = 0
        for fault in faults:
            for task in tasks:
                for seed in range(seeds):
                    for eng in engines:
                        n += 1
                        tmp = Path(d) / f"{eng}_{task}_{fault}_{seed}"
                        rows.append(run_case(eng, task, fault, seed, tmp, quiet))
                        if progress:
                            progress(n, rows[-1])
    return rows


def aggregate(rows):
    by = defaultdict(lambda: {"n": 0, "ok": 0, "retries": 0, "virtual": 0.0, "false_success": 0})
    for r in rows:
        for key in [(r["engine"], r["fault"]), (r["engine"], "ALL")]:
            a = by[key]
            a["n"] += 1
            a["ok"] += r["success"]
            a["retries"] += r["retries"]
            a["virtual"] += r["virtual_sec"]
            a["false_success"] += r["false_success"]
    return by


def to_markdown(rows, faults) -> str:
    by = aggregate(rows)
    lines = ["| 故障 | v1.0 成功率 | v2.0 成功率 | v1 误报成功 | v2 误报成功 | v2 平均重试 | v1 / v2 平均虚拟耗时(秒) |",
             "|---|---|---|---|---|---|---|"]
    for f in list(faults) + ["ALL"]:
        a, b = by[("v1", f)], by[("v2", f)]
        if not a["n"] and not b["n"]:
            continue
        lines.append(f"| {'**合计**' if f == 'ALL' else f} | {a['ok']}/{a['n']} = {a['ok'] / max(1, a['n']):.0%} | "
                     f"{b['ok']}/{b['n']} = {b['ok'] / max(1, b['n']):.0%} | {a['false_success']} | {b['false_success']} | "
                     f"{b['retries'] / max(1, b['n']):.2f} | {a['virtual'] / max(1, a['n']):.1f} / "
                     f"{b['virtual'] / max(1, b['n']):.1f} |")
    return "\n".join(lines)


def per_task_markdown(rows, faults) -> str:
    cell = defaultdict(lambda: [0, 0])
    for r in rows:
        c = cell[(r["engine"], r["fault"], r["task"])]
        c[0] += r["success"]
        c[1] += 1
    tasks = sorted({r["task"] for r in rows}, key=TASK_IDS.index)
    lines = ["| 故障 \\ 任务 | " + " | ".join(tasks) + " |", "|---|" + "---|" * len(tasks)]
    for f in faults:
        vals = []
        for t in tasks:
            a, b = cell[("v1", f, t)], cell[("v2", f, t)]
            vals.append(f"{a[0]}/{a[1]} → {b[0]}/{b[1]}")
        lines.append(f"| {f} | " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--faults", default=",".join(FAULTS))
    ap.add_argument("--tasks", default=",".join(TASK_IDS))
    ap.add_argument("--out", default=str(REPO / "report" / "week6_assets"))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)
    faults = [f for f in args.faults.split(",") if f]
    tasks = [t for t in args.tasks.split(",") if t]
    total = len(faults) * len(tasks) * args.seeds * 2
    t0 = time.time()

    def progress(n, r):
        if n % 20 == 0 or n == total:
            print(f"  {n}/{total}  {time.time() - t0:.0f}s", flush=True)
        if args.verbose and not r["success"]:
            print(f"    ✗ {r['engine']} {r['task']} {r['fault']} seed={r['seed']}: {r['error']}")

    rows = evaluate(faults, tasks, args.seeds, quiet=True, progress=progress)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "robustness_cases.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    md = ["# 鲁棒性评测（模拟桌面 + 故障注入）", "",
          f"- 任务 {len(tasks)} 个 × 故障 {len(faults)} 种 × 种子 {args.seeds} 个 × 2 个引擎 = {len(rows)} 次运行，"
          f"用时 {time.time() - t0:.0f} 秒", "", "## 按故障汇总", "", to_markdown(rows, faults), "",
          "## 按任务（v1 成功/次数 → v2 成功/次数）", "", per_task_markdown(rows, faults), ""]
    (out / "robustness_summary.md").write_text("\n".join(md), encoding="utf-8")
    (out / "robustness_summary.json").write_text(json.dumps(
        {f"{k[0]}|{k[1]}": v for k, v in aggregate(rows).items()}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
