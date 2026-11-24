"""
第6周复杂任务（多目标、跨程序）与真实干扰下的鲁棒性任务：定义、客观检查器、批量测试脚本

  C1 记事本写入并保存   打开文件 → 输入一句话 → 保存 → 关闭标签页      检查：文件内容包含该句话，且窗口已关闭
  C2 连发两条消息       给张三发消息，然后给李四发消息                  检查：聊天记录里两条都发对了人
  C3 搜索后通知         浏览器搜索天气预报，再给张三发消息              检查：有标题含搜索词的浏览器窗口 + 消息已发送
  C4 跨三个程序         打开计算器 → 给王五发消息 → 关闭计算器          检查：消息已发送 + 计算器已关闭
  C5 界面加载慢（干扰） 模拟聊天的联系人 4 秒后才出现                  检查：消息已发送
  C6 弹窗打断（干扰）   选中联系人后弹出模态“系统通知”抢走焦点          检查：消息已发送

C5/C6 的干扰由 apps/mock_chat.py 的 --load-delay / --popup-on-select 在真实窗口里产生，用于对比 v1 / v2。
判定“成功”与第4周相同：Agent 自己认为完成 且 检查器通过。

运行（Windows，Anaconda Prompt，运行期间不要动鼠标键盘）：
    python -m agent.complex_tasks                       # 6 个任务各跑 1 次（v2）
    python -m agent.complex_tasks --repeat 3            # 各 3 次
    python -m agent.complex_tasks --engine v1           # 用第4周 v1.0 跑同样的任务（对比）
    python -m agent.complex_tasks --only C5,C6 --repeat 3
结果写入 report/week6_results/<时间>/：results.csv、summary.md、每次运行的 trace.json / events.jsonl / agent.log。
"""
import argparse
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.basic_tasks import (BROWSER_TITLES, CALC_TITLES, BasicTask, TaskContext, _norm,  # noqa: E402
                               read_chat_log, run_task, summarize, write_report)

NOTE_FILE_NAME = "gui_agent_week6_note.txt"
NOTE_TEXT = "GUI Agent 第6周复杂任务测试"
C2_MESSAGES = [("张三", "明天上午10点开会"), ("李四", "收到请回复")]
C3_QUERY, C3_MESSAGE = "天气预报", ("张三", "天气已查好")
C4_MESSAGE = ("王五", "计算器已打开")
C5_MESSAGE = ("张三", "加载慢也能发出")
C6_MESSAGE = ("张三", "弹窗测试通过")


def work_dir_default() -> Path:
    return Path(tempfile.gettempdir()) / "gui_agent_week4"


def note_path(ctx: TaskContext) -> Path:
    return ctx.work_dir / NOTE_FILE_NAME


# ---------------- 检查器 ----------------
def _sent(ctx, to, text) -> bool:
    return any(r.get("event") == "send" and r.get("to") == to and _norm(text) in _norm(r.get("text"))
               for r in read_chat_log(ctx.chat_log))


def make_check_messages(pairs):
    def check(ctx: TaskContext):
        miss = [(to, t) for to, t in pairs if not _sent(ctx, to, t)]
        got = [(r.get("to"), r.get("text")) for r in read_chat_log(ctx.chat_log) if r.get("event") == "send"]
        return not miss, ("消息全部发送正确" if not miss else f"缺少 {miss}，实际发送 {got}")
    return check


def check_note_saved(ctx: TaskContext):
    p = note_path(ctx)
    content = p.read_text(encoding="utf-8-sig", errors="replace") if p.exists() else ""
    saved = _norm(NOTE_TEXT) in _norm(content)
    left = [w.title for w in ctx.apps.find_windows([NOTE_FILE_NAME])]
    ok = saved and not left
    msg = ("文件已保存" if saved else f"文件内容不含目标文字（实际: {content[:40]!r}）") + \
          ("，窗口已关闭" if not left else f"，窗口仍在: {left}")
    return ok, msg


def check_search_and_message(ctx: TaskContext):
    titles = [w.title for w in ctx.apps.find_windows(BROWSER_TITLES)]
    s_ok = any(_norm(C3_QUERY) in _norm(t) for t in titles)
    m_ok = _sent(ctx, *C3_MESSAGE)
    return s_ok and m_ok, f"搜索{'完成' if s_ok else '未完成'}（{titles}），消息{'已发送' if m_ok else '未发送'}"


def check_c4(ctx: TaskContext):
    m_ok = _sent(ctx, *C4_MESSAGE)
    left = [w.title for w in ctx.apps.find_windows(CALC_TITLES)]
    return m_ok and not left, f"消息{'已发送' if m_ok else '未发送'}，" + ("计算器已关闭" if not left else f"计算器仍在: {left}")


# ---------------- setup / teardown ----------------
def setup_note(ctx: TaskContext):
    ctx.work_dir.mkdir(parents=True, exist_ok=True)
    note_path(ctx).write_text("", encoding="utf-8")


def teardown_note(ctx: TaskContext):
    if ctx.apps.find_windows([NOTE_FILE_NAME]):
        ctx.apps.close(NOTE_FILE_NAME)


def setup_chat(ctx: TaskContext, args=()):
    ctx.chat_log.parent.mkdir(parents=True, exist_ok=True)
    ctx.chat_log.write_text("", encoding="utf-8")
    ctx.apps.mock_chat_args = list(args)


def teardown_chat(ctx: TaskContext):
    ctx.apps.mock_chat_args = []
    for t in ("模拟聊天", "系统通知"):
        try:
            if ctx.apps.find_windows(ctx.apps.title_keywords(t)):
                ctx.apps.close(t)
        except Exception:
            pass


def teardown_chat_calc(ctx: TaskContext):
    teardown_chat(ctx)
    if ctx.apps.find_windows(CALC_TITLES):
        ctx.apps.close("计算器")


def build_complex_tasks(work_dir: Optional[Path] = None):
    wd = Path(work_dir) if work_dir else work_dir_default()
    note = wd / NOTE_FILE_NAME
    (to1, m1), (to2, m2) = C2_MESSAGES
    return [
        BasicTask("C1", "记事本写入并保存", f"打开文件 {note}，然后输入“{NOTE_TEXT}”，然后保存，最后按 ctrl+w 关闭标签页",
                  check_note_saved, setup_note, teardown_note),
        BasicTask("C2", "连发两条消息", f"给{to1}发送消息：{m1}，然后给{to2}发送消息：{m2}",
                  make_check_messages(C2_MESSAGES), setup_chat, teardown_chat),
        BasicTask("C3", "搜索后通知", f"在浏览器中搜索{C3_QUERY}，然后给{C3_MESSAGE[0]}发送消息：{C3_MESSAGE[1]}",
                  check_search_and_message, setup_chat, teardown_chat),
        BasicTask("C4", "跨三个程序", f"打开计算器，然后给{C4_MESSAGE[0]}发送消息：{C4_MESSAGE[1]}，最后关闭计算器",
                  check_c4, setup_chat, teardown_chat_calc),
        BasicTask("C5", "界面加载慢", f"给{C5_MESSAGE[0]}发送消息：{C5_MESSAGE[1]}", make_check_messages([C5_MESSAGE]),
                  lambda ctx: setup_chat(ctx, ["--load-delay", "4"]), teardown_chat),
        BasicTask("C6", "弹窗打断", f"给{C6_MESSAGE[0]}发送消息：{C6_MESSAGE[1]}", make_check_messages([C6_MESSAGE]),
                  lambda ctx: setup_chat(ctx, ["--popup-on-select"]), teardown_chat),
    ]


def main(argv=None):
    from agent.factory import AgentConfig, build_agent

    ap = argparse.ArgumentParser(description="第6周复杂任务 / 干扰任务批量测试（会真实操作鼠标键盘）")
    ap.add_argument("--only", default="", help="只跑部分任务，如 C1,C5")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--engine", default="v2", choices=["v1", "v2"])
    ap.add_argument("--act", default="hybrid", choices=["hybrid", "llm", "rule"])
    ap.add_argument("--planner", default="auto", choices=["auto", "rule", "llm"])
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--model", default=None)
    ap.add_argument("--ocr", default="easyocr", choices=["easyocr", "paddleocr"])
    ap.add_argument("--countdown", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) if args.out else REPO_ROOT / "report" / "week6_results" / f"{stamp}_{args.engine}"
    chat_log = out_dir / "mock_chat.jsonl"
    cfg = AgentConfig(planner=args.planner, act_mode=args.act, use_llm=not args.no_llm, ocr_engine=args.ocr,
                      chat_log=str(chat_log), run_dir=str(out_dir / "runs" / "init"), engine=args.engine)
    if args.model:
        cfg.model = args.model
    agent, comp = build_agent(cfg)
    if args.engine == "v1":
        # v1 的规划器不认识 C1“输入/保存/按 ctrl+w”这类上下文写法：用 v2 的拆解结果喂给 v1，只比较执行阶段
        from agent.decomposer import Decomposer
        v1_decomposer = Decomposer(agent.planner.rule, llm_plan=agent.planner.llm_plan, mode=agent.planner.mode)
    ctx = TaskContext(comp["app_manager"], work_dir_default(), chat_log)
    tasks = build_complex_tasks(ctx.work_dir)
    if args.only:
        wanted = {t.strip().upper() for t in args.only.split(",")}
        tasks = [t for t in tasks if t.id in wanted]

    print(f"引擎 {args.engine}：即将执行 {len(tasks)} 个任务 × {args.repeat} 次，{args.countdown} 秒后开始，"
          "期间请不要操作鼠标键盘（紧急停止：把鼠标甩到屏幕任意一角）")
    time.sleep(max(0, args.countdown))
    results = []
    for rep in range(1, args.repeat + 1):
        for t in tasks:
            agent.trace_dir = out_dir / "runs" / f"{t.id}_run{rep}"
            comp["capture"].save_dir = agent.trace_dir
            print(f"\n======== {t.id} {t.name}（第 {rep} 次，{args.engine}）========")
            if args.engine == "v1":
                plan = v1_decomposer.decompose(t.instruction).to_plan()
                orig_run = agent.run
                agent.run = lambda ins, checker=None, _p=plan, _r=orig_run: _r(ins, checker=checker, plan=_p)
                r = run_task(agent, t, ctx)
                agent.run = orig_run
            else:
                r = run_task(agent, t, ctx)
            r["run"] = rep
            results.append(r)
            time.sleep(1.0)
    rows = summarize(results)
    for row, r in zip(rows, results):
        rb = r.get("robustness") or {}
        row.update(engine=args.engine, retries=rb.get("retries", sum(len(s.get("attempts", [])) - 1
                                                                     for s in r.get("steps", []))),
                   recoveries=rb.get("recoveries", 0), goal_replays=rb.get("goal_replays", 0))
    meta = {"time": stamp, "act_mode": f"{args.engine} / {agent.act_mode}", "planner": agent.planner.mode,
            "model": cfg.model if cfg.use_llm else "无", "ocr": args.ocr, "repeat": args.repeat}
    text = write_report(rows, out_dir, meta).replace("第4周基础任务测试结果", f"第6周复杂任务测试结果（{args.engine}）")
    (out_dir / "summary.md").write_text(text + "\n", encoding="utf-8")
    print("\n" + text)
    print(f"\n结果目录: {out_dir}")
    return 0 if all(r["success"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
