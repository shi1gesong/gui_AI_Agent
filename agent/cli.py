"""
GUI Agent v1.0 命令行入口（第4周）

用法（在仓库根目录、Anaconda Prompt 中）：
    python -m agent.cli "在浏览器中搜索天气预报"      执行一条指令
    python -m agent.cli --plan-only "打开记事本"     只看规划结果，不动鼠标
    python -m agent.cli --dry-run "给张三发送消息：你好"  截图+OCR+决策，但不执行动作
    python -m agent.cli --task T4                    执行预置基础任务（T1~T5，见 --list-tasks）
    python -m agent.cli                              交互模式，逐条输入指令，输入 q 退出

常用参数：
    --act hybrid|llm|rule|vlm 动作决策方式（默认 hybrid：规则能推断就用规则，否则问 LLM；
                            vlm：第5周，需要看屏幕的步骤交给 WSL2 里的 Qwen2-VL+LoRA 服务直接输出坐标）
    --vlm-url URL           vlm 服务地址（默认 http://localhost:8765，WSL2 中运行 python -m finetune.serve）
    --planner auto|rule|llm 规划方式（默认 auto：先匹配基础任务模板，匹配不上再问 LLM）
    --no-llm                不连接 Ollama（只能执行模板能识别的指令）
    --model qwen3:8b        Ollama 模型名；--think 打开 qwen3 思考模式（更慢）
    --ocr easyocr|paddleocr OCR 引擎
紧急停止：把鼠标快速移到屏幕任意一个角（pyautogui FAILSAFE），或在命令行按 Ctrl+C。
"""
import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def build_parser():
    ap = argparse.ArgumentParser(prog="python -m agent.cli", description="GUI Agent v1.0：用自然语言操作电脑桌面")
    ap.add_argument("instruction", nargs="*", help="要执行的指令；不填进入交互模式")
    ap.add_argument("--task", default=None, help="执行预置基础任务 T1~T5")
    ap.add_argument("--list-tasks", action="store_true", help="列出预置基础任务")
    ap.add_argument("--plan-only", action="store_true", help="只输出规划结果")
    ap.add_argument("--dry-run", action="store_true", help="感知 + 决策，但不执行动作")
    ap.add_argument("--act", default="hybrid", choices=["hybrid", "llm", "rule", "vlm"])
    ap.add_argument("--vlm-url", default=None, help="--act vlm 时的 VLM 服务地址（默认 http://localhost:8765）")
    ap.add_argument("--vlm-ocr", action="store_true", help="--act vlm 时仍运行 OCR（默认不运行）")
    ap.add_argument("--planner", default="auto", choices=["auto", "rule", "llm"])
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--model", default=None)
    ap.add_argument("--ollama-url", default=None)
    ap.add_argument("--think", action="store_true", help="qwen3 思考模式（默认关闭以加快速度）")
    ap.add_argument("--ocr", default="easyocr", choices=["easyocr", "paddleocr"])
    ap.add_argument("--retries", type=int, default=1, help="子任务执行失败后的重试次数")
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--chat-app", default="模拟聊天", help="“发送消息”使用的聊天程序：模拟聊天（默认）/ 微信")
    ap.add_argument("--countdown", type=int, default=3, help="开始执行前的倒计时秒数")
    ap.add_argument("--run-dir", default=None, help="轨迹与截图保存目录（默认 runs/<时间>）")
    return ap


def make_config(args):
    from agent.factory import AgentConfig
    cfg = AgentConfig(planner=args.planner, act_mode=args.act, use_llm=not args.no_llm, think=bool(args.think),
                      ocr_engine=args.ocr, dry_run=args.dry_run, step_retries=args.retries,
                      max_steps=args.max_steps, chat_app=args.chat_app, run_dir=args.run_dir,
                      vlm_url=getattr(args, "vlm_url", None), vlm_ocr=bool(getattr(args, "vlm_ocr", False)))
    if args.model:
        cfg.model = args.model
    if args.ollama_url:
        cfg.base_url = args.ollama_url
    return cfg


def resolve_task(task_id: str):
    from agent.basic_tasks import build_tasks
    for t in build_tasks():
        if t.id.upper() == str(task_id).upper():
            return t
    raise SystemExit(f"没有预置任务 {task_id}，可用：" + ", ".join(t.id for t in build_tasks()))


def plan_only(args, instruction: str) -> int:
    from agent.planner import HybridPlanner, RulePlanner
    from control.app_manager import AppManager
    llm_plan = None
    if not args.no_llm and args.planner != "rule":
        from agent.factory import AgentConfig  # noqa: F401
        from agent.base_agent import BaseGUIAgent
        kw = {"model_name": args.model or AgentConfig().model, "think": bool(args.think), "temperature": 0.0}
        if args.ollama_url:
            kw["base_url"] = args.ollama_url
        llm_plan = BaseGUIAgent(llm_mode="api", **kw).plan
    planner = HybridPlanner(mode=args.planner if llm_plan else "rule",
                            rule=RulePlanner(chat_app=args.chat_app, app_resolver=AppManager().resolve_app),
                            llm_plan=llm_plan)
    try:
        p = planner.plan(instruction)
    except Exception as e:
        print(f"[规划] 失败: {e}")
        return 2
    print(f"[规划] 来源={p.source}")
    for i, s in enumerate(p.subtasks, 1):
        print(f"  {i}. {s}")
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_tasks:
        from agent.basic_tasks import build_tasks
        for t in build_tasks():
            print(f"{t.id}  {t.name:<8}  指令：{t.instruction}")
        return 0

    task = resolve_task(args.task) if args.task else None
    instruction = task.instruction if task else " ".join(args.instruction).strip()
    if args.plan_only:
        if not instruction:
            print("--plan-only 需要一条指令")
            return 2
        return plan_only(args, instruction)

    from agent.factory import build_agent
    agent, comp = build_agent(make_config(args))
    print(f"GUI Agent v1.0  决策={agent.act_mode}  规划={agent.planner.mode}  "
          f"LLM={'无' if comp['brain'] is None else comp['brain'].llm.inner.model_name}  "
          f"{'（dry-run：不执行动作）' if args.dry_run else ''}")
    print(f"轨迹目录: {comp['run_dir']}")

    def run_one(text, checker=None):
        if args.countdown > 0 and not args.dry_run:
            print(f"{args.countdown} 秒后开始执行，请不要操作鼠标键盘（紧急停止：鼠标甩到屏幕角落）")
            time.sleep(args.countdown)
        return agent.run(text, checker=checker)

    if task:
        from agent.basic_tasks import TaskContext, run_task
        import tempfile
        ctx = TaskContext(comp["app_manager"], Path(tempfile.gettempdir()) / "gui_agent_week4",
                          Path(comp["run_dir"]) / "mock_chat.jsonl")
        comp["app_manager"].mock_chat_log = str(ctx.chat_log)
        if args.countdown > 0 and not args.dry_run:
            print(f"{args.countdown} 秒后开始执行任务 {task.id}，请不要操作鼠标键盘")
            time.sleep(args.countdown)
        res = run_task(agent, task, ctx)
        return 0 if res.get("success") else 1

    if instruction:
        res = run_one(instruction)
        return 0 if res.get("success") else 1

    print("交互模式：输入指令后回车执行，输入 q 退出")
    while True:
        try:
            text = input("\n指令> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text.lower() in ("q", "quit", "exit", "退出"):
            break
        if text:
            run_one(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
