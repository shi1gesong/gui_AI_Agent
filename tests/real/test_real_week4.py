"""第4周 真机 / 真实 LLM 用例（默认跳过）

1) Windows 真机端到端（会真实操作鼠标键盘、打开浏览器/记事本/计算器/模拟聊天）：
       set GUI_AGENT_REAL_DEVICE=1
       pytest tests/real/test_real_week4.py -m real_device -v -s
   运行期间不要动鼠标键盘；浏览器会新开一个必应标签页（不会关闭你已有的浏览器窗口）。
2) 真实 qwen3:8b：
       set OLLAMA_BASE_URL=http://localhost:11434
       pytest tests/real/test_real_week4.py -m llm -v
"""
import os
import sys
import tempfile
import time
from pathlib import Path

import pytest

from semantic_eval import judge, load_week4_cases, screen_context

IS_WIN = sys.platform.startswith("win")
W4_CASES = load_week4_cases()


# ------------------------------------------------------------------ 真机
@pytest.fixture(scope="module")
def real(tmp_path_factory):
    if not IS_WIN:
        pytest.skip("仅 Windows")
    from agent.basic_tasks import TaskContext, build_tasks
    from agent.factory import AgentConfig, build_agent
    out = tmp_path_factory.mktemp("week4_real")
    chat_log = out / "mock_chat.jsonl"
    cfg = AgentConfig(use_llm=False, chat_log=str(chat_log), run_dir=str(out / "runs"))
    agent, comp = build_agent(cfg, log=print)
    ctx = TaskContext(comp["app_manager"], Path(tempfile.gettempdir()) / "gui_agent_week4", chat_log)
    return agent, comp, ctx, {t.id: t for t in build_tasks(ctx.work_dir)}


def _run(real, tid):
    from agent.basic_tasks import run_task
    agent, comp, ctx, tasks = real
    agent.trace_dir = Path(comp["run_dir"]).parent / tid
    r = run_task(agent, tasks[tid], ctx)
    time.sleep(1.0)
    return r


def _why(r):
    return {"check": r.get("check"), "errors": [(s["subtask"], s.get("error")) for s in r.get("steps", [])
                                                 if not s.get("ok")], "error": r.get("error")}


@pytest.mark.real_device
@pytest.mark.functional
def test_real_T1_open_browser(real):
    r = _run(real, "T1")
    assert r["success"], _why(r)


@pytest.mark.real_device
@pytest.mark.functional
def test_real_T2_search(real):
    r = _run(real, "T2")
    assert r["success"], _why(r)


@pytest.mark.real_device
@pytest.mark.functional
def test_real_T3_open_file(real):
    r = _run(real, "T3")
    assert r["success"], _why(r)


@pytest.mark.real_device
@pytest.mark.functional
def test_real_T4_send_message(real):
    r = _run(real, "T4")
    assert r["success"], _why(r)


@pytest.mark.real_device
@pytest.mark.functional
def test_real_T5_close_app(real):
    r = _run(real, "T5")
    assert r["success"], _why(r)


@pytest.mark.real_device
@pytest.mark.functional
def test_real_mock_chat_ocr_recognizes_widgets(real):
    """真实 EasyOCR 能识别模拟聊天程序的关键控件（T4 能否成功的前提）"""
    agent, comp, ctx, _ = real
    apps = comp["app_manager"]
    assert apps.open("模拟聊天")["ok"]
    time.sleep(1.5)
    try:
        texts = comp["perceiver"].perceive().texts()
        joined = "".join(texts)
        for w in ("张三", "李四", "发送", "消息内容"):
            assert w in joined, f"OCR 没识别出「{w}」: {texts[:40]}"
    finally:
        apps.close("模拟聊天")


@pytest.mark.real_device
@pytest.mark.multi_resolution
def test_real_perceiver_scale_matches_dpi(real):
    """感知模块计算出的缩放比 = 截图尺寸 / pyautogui 尺寸，应为常见 DPI 档位"""
    obs = real[1]["perceiver"].perceive()
    sx, sy = obs.scale
    print(f"截图 {obs.image_size}，鼠标坐标系 {obs.logical_size}，scale={sx:.3f}")
    assert abs(sx - sy) < 0.01 and min(abs(sx - d) for d in (1.0, 1.25, 1.5, 1.75, 2.0)) < 0.02


@pytest.mark.real_device
@pytest.mark.exception
def test_real_close_nonexistent_app_fails_safely(real):
    from agent.planner import Plan
    agent = real[0]
    r = agent.run("x", plan=Plan(["关闭一个不存在的程序窗口"], "test"))
    assert r["success"] is False and "没有找到" in (r["steps"][0].get("error") or "")


@pytest.mark.real_device
@pytest.mark.boundary
def test_real_open_missing_file_fails_safely(real):
    agent = real[0]
    r = agent.run("打开文件 C:\\gui_agent_不存在的文件.txt")
    assert r["success"] is False and "文件不存在" in (r["steps"][0].get("error") or "")


# ------------------------------------------------------------------ 真实 LLM
@pytest.fixture(scope="module")
def brain():
    pytest.importorskip("langchain_core")
    from base_agent import BaseGUIAgent
    return BaseGUIAgent(llm_mode="api", model_name=os.environ.get("OLLAMA_MODEL", "qwen3:8b"),
                        base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
                        think=False, temperature=0.0)


@pytest.mark.llm
@pytest.mark.functional
@pytest.mark.parametrize("case", W4_CASES, ids=[c["id"] for c in W4_CASES])
def test_real_llm_week4_action_space(brain, case):
    """qwen3:8b 在扩展动作空间下的 Acting 语义准确率（含校验 + 重试 + 兜底后的最终动作）"""
    brain.action_history.clear()
    action = brain.act(case["subtask"], screen_context(case["screen"]))
    ok, reasons = judge(case, action)
    assert ok, f"{action} -> {reasons}"


@pytest.mark.llm
@pytest.mark.functional
@pytest.mark.parametrize("instruction", ["打开浏览器", "在浏览器中搜索天气预报", "打开文件 C:\\demo.txt",
                                         "给张三发送消息：今晚7点开会", "关闭计算器"])
def test_real_llm_plan_basic_tasks(brain, instruction):
    """纯 LLM 规划（--planner llm）：子任务数量合理、非空"""
    subtasks = brain.plan(instruction)
    assert 1 <= len(subtasks) <= 8, subtasks


@pytest.mark.llm
@pytest.mark.functional
def test_real_llm_drives_sim_t4(brain, tmp_path):
    """真实 LLM 决策每一步（--act llm），在模拟桌面上完成 T4"""
    from agent.basic_tasks import TaskContext, build_tasks, run_task
    from agent.executor import ControllerExecutor
    from agent.gui_agent import GUIAgent
    from agent.planner import HybridPlanner, RulePlanner
    from perception.screen_parser import ScreenPerceiver
    from sim_desktop import SimDesktop
    sim = SimDesktop(tmp_path)
    apps = sim.app_manager()
    agent = GUIAgent(ScreenPerceiver(capture=sim, ocr=sim, logical_size_fn=sim.logical_size),
                     ControllerExecutor(sim, app_manager=apps, sleep=lambda s: None),
                     planner=HybridPlanner("rule", RulePlanner()), brain=brain, act_mode="llm",
                     sleep=lambda s: None, log=print)
    ctx = TaskContext(apps, tmp_path / "w", sim.chat_log)
    t4 = {t.id: t for t in build_tasks(ctx.work_dir)}["T4"]
    r = run_task(agent, t4, ctx)
    assert r["success"], _why(r)
