"""第6周 Windows 真机用例（默认跳过）—— v2.0 复杂任务、干扰任务与感知速度

    set GUI_AGENT_REAL_DEVICE=1
    pytest tests/real/test_real_week6.py -m real_device -v -s

运行期间不要动鼠标键盘。C3 会在默认浏览器新开一个必应搜索页。
"""
import sys
import time
from pathlib import Path

import pytest

IS_WIN = sys.platform.startswith("win")


@pytest.fixture(scope="module")
def real6(tmp_path_factory):
    if not IS_WIN:
        pytest.skip("仅 Windows")
    from agent.basic_tasks import TaskContext
    from agent.complex_tasks import build_complex_tasks, work_dir_default
    from agent.factory import AgentConfig, build_agent
    out = tmp_path_factory.mktemp("week6_real")
    chat_log = out / "mock_chat.jsonl"
    cfg = AgentConfig(use_llm=False, chat_log=str(chat_log), run_dir=str(out / "runs"), engine="v2")
    agent, comp = build_agent(cfg, log=print)
    ctx = TaskContext(comp["app_manager"], work_dir_default(), chat_log)
    return agent, comp, ctx, {t.id: t for t in build_complex_tasks(ctx.work_dir)}, out


def _run(real6, tid):
    from agent.basic_tasks import run_task
    agent, comp, ctx, tasks, out = real6
    agent.trace_dir = out / tid
    comp["capture"].save_dir = agent.trace_dir
    r = run_task(agent, tasks[tid], ctx)
    time.sleep(1.0)
    return r


def _why(r):
    return {"check": r.get("check"), "error": r.get("error"),
            "errors": [(s["subtask"], s.get("error")) for s in r.get("steps", []) if not s.get("ok")]}


@pytest.mark.real_device
@pytest.mark.functional
@pytest.mark.parametrize("tid", ["C1", "C2", "C3", "C4"])
def test_real_v2_complex_task(real6, tid):
    r = _run(real6, tid)
    assert r["success"], _why(r)
    assert len(r.get("goals", [])) >= 2          # 确实拆成了多个目标


@pytest.mark.real_device
@pytest.mark.boundary
def test_real_v2_slow_loading_C5(real6):
    r = _run(real6, "C5")
    assert r["success"], _why(r)


@pytest.mark.real_device
@pytest.mark.exception
def test_real_v2_popup_C6(real6):
    r = _run(real6, "C6")
    assert r["success"], _why(r)
    assert r["robustness"]["recoveries"] >= 1 or r["robustness"]["retries"] >= 1


@pytest.mark.real_device
@pytest.mark.functional
def test_real_monitor_files_written(real6):
    r = _run(real6, "C2")
    d = Path(real6[0].trace_dir)
    run_dirs = [p for p in [d, *d.glob("*")] if (p / "status.json").exists()]
    assert r["success"] and run_dirs, _why(r)
    assert (run_dirs[0] / "events.jsonl").exists() and (run_dirs[0] / "metrics.json").exists()


@pytest.mark.real_device
@pytest.mark.ocr_real
@pytest.mark.functional
def test_real_perception_roi_and_cache_faster(real6):
    """模拟聊天在前台：v2 只识别前台窗口 + 缓存，比 v1 全屏识别快"""
    from perception.screen_parser import ScreenPerceiver
    agent, comp, ctx, tasks, out = real6
    ctx.apps.open("模拟聊天")
    time.sleep(2)
    try:
        v2 = agent.perceiver
        v1 = ScreenPerceiver(capture=comp["capture"], ocr=v2.ocr)
        v1.perceive()
        t = time.perf_counter()
        v1.perceive()
        t1 = time.perf_counter() - t
        v2.reset_cache()
        t = time.perf_counter()
        o2 = v2.perceive()
        t2 = time.perf_counter() - t
        t = time.perf_counter()
        v2.perceive()
        t3 = time.perf_counter() - t
        print(f"\nv1 全屏 {t1:.2f}s  v2 前台窗口 {t2:.2f}s  v2 缓存 {t3:.3f}s")
        assert any(e["text"] == "发送" for e in o2.elements)
        assert t2 < t1 and t3 < t1 * 0.3
    finally:
        ctx.apps.close("模拟聊天")
