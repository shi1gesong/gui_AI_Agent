"""
第4周 5 个基础桌面任务的定义、客观检查器与批量测试脚本

  T1 打开浏览器            检查：前台窗口是浏览器（标题含 Edge/Chrome/Firefox/Bing）
  T2 在浏览器中搜索指定内容   检查：有浏览器窗口标题包含搜索词（必应/百度结果页标题 = “搜索词 - 搜索”）
  T3 打开指定文件           检查：出现标题含该文件名的窗口（测试前自动在临时目录生成该文件）
  T4 发送消息              检查：模拟聊天程序日志里有 {"to": 联系人, "text": 消息} 的发送记录
  T5 关闭应用              检查：测试前打开计算器，任务后不再有标题含“计算器/Calculator”的窗口

判定“成功”= Agent 自己认为完成（每个子任务执行成功 + Reflect 判定完成）且 检查器通过。
只看 Agent 自述不看检查器，就会重蹈第3周“什么都没做却判定完成”的覆辙。

运行（Windows，Anaconda Prompt，运行期间不要动鼠标键盘）：
    python -m agent.basic_tasks                 # 5 个任务各跑 1 次，规则 + LLM 混合
    python -m agent.basic_tasks --repeat 3      # 各跑 3 次，统计成功率
    python -m agent.basic_tasks --act llm       # 每个子任务都让 LLM 决策（对比实验）
    python -m agent.basic_tasks --no-llm        # 不连 Ollama，纯规则
    python -m agent.basic_tasks --engine v1     # 第6周起默认 v2，加这个参数用第4周 v1.0 对比
结果写入 report/week4_results/<时间>/：results.csv、summary.md、每次运行的 trace.json 与截图。
"""
import argparse
import csv
import json
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BROWSER_TITLES = ["Edge", "Chrome", "Firefox", "Bing", "必应", "浏览器"]
CALC_TITLES = ["计算器", "Calculator"]
DEMO_FILE_NAME = "gui_agent_week4_demo.txt"
SEARCH_QUERY = "天气预报"
MESSAGE_TO, MESSAGE_TEXT = "张三", "今晚7点开会"


@dataclass
class BasicTask:
    id: str
    name: str
    instruction: str
    check: Callable[["TaskContext"], tuple]
    setup: Optional[Callable[["TaskContext"], None]] = None
    teardown: Optional[Callable[["TaskContext"], None]] = None


class TaskContext:
    def __init__(self, app_manager, work_dir: Path, chat_log: Path, sleep=time.sleep):
        self.apps = app_manager
        self.work_dir = Path(work_dir)
        self.chat_log = Path(chat_log)
        self.sleep = sleep
        self.demo_file = self.work_dir / DEMO_FILE_NAME


def _norm(s):
    return "".join(str(s or "").split()).lower()


# ---------------- 检查器 ----------------
def check_browser_foreground(ctx: TaskContext):
    title = ctx.apps.active_title()
    ok = any(_norm(k) in _norm(title) for k in BROWSER_TITLES)
    return ok, f"前台窗口标题「{title}」" + ("是浏览器" if ok else "不是浏览器")


def make_check_search(query: str):
    def check(ctx: TaskContext):
        titles = [w.title for w in ctx.apps.find_windows(BROWSER_TITLES)]
        hit = [t for t in titles if _norm(query) in _norm(t)]
        return bool(hit), (f"浏览器窗口「{hit[0]}」包含搜索词" if hit else f"没有浏览器窗口标题包含「{query}」: {titles}")
    return check


def check_file_window(ctx: TaskContext):
    wins = ctx.apps.find_windows([ctx.demo_file.name, ctx.demo_file.stem])
    return bool(wins), (f"已打开窗口「{wins[0].title}」" if wins else f"没有标题含 {ctx.demo_file.name} 的窗口")


def read_chat_log(path: Path):
    if not Path(path).exists():
        return []
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def make_check_message(to: str, text: str):
    def check(ctx: TaskContext):
        sends = [r for r in read_chat_log(ctx.chat_log) if r.get("event") == "send"]
        ok = any(r.get("to") == to and _norm(text) in _norm(r.get("text")) for r in sends)
        got = [(r.get("to"), r.get("text")) for r in sends]
        return ok, (f"已向「{to}」发送「{text}」" if ok else f"聊天记录中没有发给「{to}」的「{text}」，实际: {got}")
    return check


def check_calc_closed(ctx: TaskContext):
    left = [w.title for w in ctx.apps.find_windows(CALC_TITLES)]
    return not left, ("计算器窗口已全部关闭" if not left else f"仍有窗口: {left}")


# ---------------- setup / teardown ----------------
def setup_demo_file(ctx: TaskContext):
    ctx.work_dir.mkdir(parents=True, exist_ok=True)
    ctx.demo_file.write_text("GUI Agent 第4周基础任务测试文件\n打开这个文件即表示 T3 成功。\n", encoding="utf-8-sig")


def teardown_demo_file(ctx: TaskContext):
    ctx.apps.close(ctx.demo_file.name)


def setup_chat(ctx: TaskContext):
    ctx.chat_log.parent.mkdir(parents=True, exist_ok=True)
    ctx.chat_log.write_text("", encoding="utf-8")


def teardown_chat(ctx: TaskContext):
    ctx.apps.close("模拟聊天")


def setup_calc(ctx: TaskContext):
    r = ctx.apps.open("计算器")
    if not r.get("ok"):
        raise RuntimeError(f"测试准备失败，无法打开计算器: {r.get('error')}")
    ctx.sleep(1.0)


def build_tasks(work_dir: Optional[Path] = None):
    demo = (Path(work_dir) if work_dir else Path(tempfile.gettempdir()) / "gui_agent_week4") / DEMO_FILE_NAME
    return [
        BasicTask("T1", "打开浏览器", "打开浏览器", check_browser_foreground),
        BasicTask("T2", "搜索指定内容", f"在浏览器中搜索{SEARCH_QUERY}", make_check_search(SEARCH_QUERY)),
        BasicTask("T3", "打开指定文件", f"打开文件 {demo}", check_file_window, setup_demo_file, teardown_demo_file),
        BasicTask("T4", "发送消息", f"给{MESSAGE_TO}发送消息：{MESSAGE_TEXT}", make_check_message(MESSAGE_TO, MESSAGE_TEXT),
                  setup_chat, teardown_chat),
        BasicTask("T5", "关闭应用", "关闭计算器", check_calc_closed, setup_calc),
    ]


def run_task(agent, task: BasicTask, ctx: TaskContext) -> dict:
    setup_error = None
    if task.setup:
        try:
            task.setup(ctx)
        except Exception as e:
            setup_error = f"{type(e).__name__}: {e}"
    if setup_error:
        res = {"instruction": task.instruction, "success": False, "error": f"setup 失败: {setup_error}",
               "steps": [], "reflection": {}, "sec": 0}
    else:
        res = agent.run(task.instruction, checker=lambda: task.check(ctx))
    if task.teardown:
        try:
            task.teardown(ctx)
        except Exception:
            pass
    res["task_id"], res["task_name"] = task.id, task.name
    return res


def summarize(results):
    rows = []
    for r in results:
        steps = r.get("steps", [])
        rows.append({
            "task_id": r.get("task_id"), "task_name": r.get("task_name"), "run": r.get("run", 1),
            "instruction": r.get("instruction"), "plan_source": (r.get("plan") or {}).get("source", ""),
            "subtasks": len((r.get("plan") or {}).get("subtasks", [])), "steps_ok": sum(1 for s in steps if s.get("ok")),
            "steps_run": len(steps), "llm_steps": sum(1 for s in steps if s.get("source") == "llm"),
            "agent_complete": bool((r.get("reflection") or {}).get("is_complete")),
            "check_ok": bool((r.get("check") or {}).get("ok")), "success": bool(r.get("success")),
            "sec": r.get("sec", 0), "error": r.get("error") or next((s.get("error") for s in steps if not s.get("ok")), "")
                                                       or ("" if (r.get("check") or {}).get("ok") else (r.get("check") or {}).get("message", "")),
        })
    return rows


def write_report(rows, out_dir: Path, meta: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["task_id"])
        w.writeheader()
        w.writerows(rows)
    by_task = {}
    for r in rows:
        by_task.setdefault((r["task_id"], r["task_name"]), []).append(r)
    lines = [f"# 第4周基础任务测试结果（{meta.get('time')}）", "",
             f"- 决策模式: {meta.get('act_mode')}，规划: {meta.get('planner')}，模型: {meta.get('model')}，"
             f"OCR: {meta.get('ocr')}，每个任务 {meta.get('repeat')} 次", "",
             "| 任务 | 成功/次数 | 成功率 | 平均用时(秒) | LLM 决策步数 | 失败原因（首个） |", "|---|---|---|---|---|---|"]
    total_ok = 0
    for (tid, name), rs in by_task.items():
        ok = sum(r["success"] for r in rs)
        total_ok += ok
        err = next((r["error"] for r in rs if not r["success"]), "")
        lines.append(f"| {tid} {name} | {ok}/{len(rs)} | {ok / len(rs):.0%} | "
                     f"{sum(r['sec'] for r in rs) / len(rs):.1f} | {sum(r['llm_steps'] for r in rs)} | {err} |")
    if rows:
        lines += ["", f"**总计：{total_ok}/{len(rows)} = {total_ok / len(rows):.0%}**"]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return "\n".join(lines)


def main(argv=None):
    from agent.factory import AgentConfig, build_agent

    ap = argparse.ArgumentParser(description="第4周 5 个基础任务批量测试（会真实操作鼠标键盘）")
    ap.add_argument("--only", default="", help="只跑部分任务，如 T1,T4")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--engine", default="v2", choices=["v1", "v2"], help="第6周起默认 v2；v1 为第4周闭环")
    ap.add_argument("--act", default="hybrid", choices=["hybrid", "llm", "rule"])
    ap.add_argument("--planner", default="auto", choices=["auto", "rule", "llm"])
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--model", default=None)
    ap.add_argument("--ocr", default="easyocr", choices=["easyocr", "paddleocr"])
    ap.add_argument("--countdown", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) if args.out else REPO_ROOT / "report" / "week4_results" / stamp
    chat_log = out_dir / "mock_chat.jsonl"
    cfg = AgentConfig(planner=args.planner, act_mode=args.act, use_llm=not args.no_llm, ocr_engine=args.ocr,
                      chat_log=str(chat_log), run_dir=str(out_dir / "runs" / "init"), engine=args.engine)
    if args.model:
        cfg.model = args.model
    agent, comp = build_agent(cfg)
    ctx = TaskContext(comp["app_manager"], Path(tempfile.gettempdir()) / "gui_agent_week4", chat_log)
    tasks = build_tasks(ctx.work_dir)
    if args.only:
        wanted = {t.strip().upper() for t in args.only.split(",")}
        tasks = [t for t in tasks if t.id in wanted]

    print(f"即将执行 {len(tasks)} 个任务 × {args.repeat} 次，{args.countdown} 秒后开始，期间请不要操作鼠标键盘"
          "（紧急停止：把鼠标甩到屏幕任意一角）")
    time.sleep(max(0, args.countdown))
    results = []
    for rep in range(1, args.repeat + 1):
        for t in tasks:
            agent.trace_dir = out_dir / "runs" / f"{t.id}_run{rep}"
            comp["capture"].save_dir = agent.trace_dir
            print(f"\n======== {t.id} {t.name}（第 {rep} 次）========")
            r = run_task(agent, t, ctx)
            r["run"] = rep
            results.append(r)
            time.sleep(1.0)
    rows = summarize(results)
    meta = {"time": stamp, "act_mode": f"{args.engine} / {agent.act_mode}", "planner": agent.planner.mode,
            "model": cfg.model if cfg.use_llm else "无", "ocr": args.ocr, "repeat": args.repeat}
    print("\n" + write_report(rows, out_dir, meta))
    print(f"\n结果目录: {out_dir}")
    return 0 if all(r["success"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
