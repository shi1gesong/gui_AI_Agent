"""第4周：agent/gui_agent.py 端到端闭环 —— 在模拟桌面（tests/sim_desktop.py）上跑完整流程

覆盖：指令 → 规划（模板/LLM）→ 感知（截图+OCR+DPI 换算）→ 决策（规则/LLM）→ grounding → 执行 → 重试 → 反思 → 检查器 → 轨迹
"""
import json

import pytest

from agent.basic_tasks import TaskContext, build_tasks, run_task
from agent.executor import ControllerExecutor, DryRunExecutor
from agent.gui_agent import GUIAgent
from agent.planner import HybridPlanner, Plan, RulePlanner
from perception.screen_parser import ScreenPerceiver, StaticPerceiver
from sim_desktop import SimDesktop

pytest.importorskip("langchain_core")
from base_agent import BaseGUIAgent, LangChainLLMWrapper  # noqa: E402
from fakes import ScriptedLLM  # noqa: E402


def J(o):
    return json.dumps(o, ensure_ascii=False)


def make(tmp_path, dpi=1.0, logical=(1920, 1080), brain=None, act_mode="hybrid", planner=None, **kw):
    sim = SimDesktop(tmp_path, logical=logical, dpi=dpi)
    apps = sim.app_manager()
    perceiver = ScreenPerceiver(capture=sim, ocr=sim, logical_size_fn=sim.logical_size, region_fn=apps.screen_regions)
    executor = ControllerExecutor(sim, app_manager=apps, sleep=lambda s: None)
    planner = planner or HybridPlanner("auto" if brain else "rule", RulePlanner(app_resolver=apps.resolve_app),
                                       llm_plan=brain.plan if brain else None)
    agent = GUIAgent(perceiver, executor, planner=planner, brain=brain, act_mode=act_mode, sleep=lambda s: None,
                     log=lambda *a: None, trace_dir=str(tmp_path / "trace"), **kw)
    ctx = TaskContext(apps, tmp_path / "work", sim.chat_log, sleep=lambda s: None)
    return agent, sim, ctx


def brain_with(responses):
    b = BaseGUIAgent(llm_mode="api")
    fake = ScriptedLLM([r if isinstance(r, (str, Exception)) else J(r) for r in responses])
    b.llm = LangChainLLMWrapper(inner_llm=fake)
    return b, fake


TASKS = {t.id: t for t in build_tasks()}


def task(tid, ctx):
    t = build_tasks(ctx.work_dir)
    return {x.id: x for x in t}[tid]


@pytest.mark.functional
class TestGUIAgentFunctional:
    @pytest.mark.parametrize("tid", ["T1", "T2", "T3", "T4", "T5"])
    def test_basic_task_success_rule_only(self, tmp_path, tid):
        agent, sim, ctx = make(tmp_path)
        r = run_task(agent, task(tid, ctx), ctx)
        assert r["success"], (r.get("check"), [s.get("error") for s in r["steps"]])

    def test_t1_needs_no_perception(self, tmp_path):
        agent, sim, ctx = make(tmp_path)
        run_task(agent, task("T1", ctx), ctx)
        assert sim.ocr_calls == 0  # 打开/等待都不需要看屏幕

    def test_t2_action_sequence(self, tmp_path):
        agent, sim, ctx = make(tmp_path)
        run_task(agent, task("T2", ctx), ctx)
        assert [c for c in sim.calls if c[0] != "startfile"] == [
            ("hotkey", "ctrl", "l"), ("type_text", "天气预报"), ("press_key", "enter")]
        assert "天气预报" in sim.windows.active.title

    def test_t4_clicks_ocr_grounded_points(self, tmp_path):
        agent, sim, ctx = make(tmp_path)
        run_task(agent, task("T4", ctx), ctx)
        clicks = [c for c in sim.calls if c[0] == "click"]
        assert clicks[0] == ("click", 205, 180)        # 张三 [110,160,300,200] 中心
        assert sim.chat["sent"] == [("张三", "今晚7点开会")]

    def test_t4_closes_chat_in_teardown(self, tmp_path):
        agent, sim, ctx = make(tmp_path)
        run_task(agent, task("T4", ctx), ctx)
        assert "GUI Agent 模拟聊天" not in sim.windows.titles()

    def test_trace_written(self, tmp_path):
        agent, sim, ctx = make(tmp_path)
        r = agent.run("打开浏览器")
        data = json.loads((tmp_path / "trace" / "trace.json").read_text(encoding="utf-8"))
        assert data["success"] is True and data["plan"]["source"] == "template:open_browser"
        assert r["trace_path"].endswith("trace.json")

    def test_steps_record_source_and_detail(self, tmp_path):
        agent, sim, ctx = make(tmp_path)
        r = agent.run("给张三发送消息：你好")
        click = r["steps"][2]
        assert click["source"] == "rule" and click["attempts"][0]["detail"]["grounding"][0]["matched"] == "张三"
        assert click["attempts"][0]["observations"][0]["elements"] >= 4

    def test_llm_planner_for_unknown_instruction(self, tmp_path):
        brain, fake = brain_with([{"subtasks": ["打开记事本", "等待1秒"]},
                                  {"is_complete": True, "reason": "ok"}])
        agent, sim, ctx = make(tmp_path, brain=brain)
        r = agent.run("我想记点笔记")
        assert r["plan"]["source"] == "llm" and r["success"] and "无标题 - 记事本" in sim.windows.titles()

    def test_hybrid_calls_llm_only_when_rule_fails(self, tmp_path):
        """“点击第一个联系人”规则推断不出 → LLM 决策；其余子任务走规则"""
        brain, fake = brain_with([{"action_type": "click", "target_text": "张三", "input_text": ""},
                                  {"is_complete": True, "reason": "ok"}])
        agent, sim, ctx = make(tmp_path, brain=brain)
        plan = Plan(["打开模拟聊天", "点击第一个联系人"], "test")
        r = agent.run("x", plan=plan)
        assert [s["source"] for s in r["steps"]] == ["rule", "llm"] and sim.chat["selected"] == "张三"
        assert len(fake.prompts) == 2  # 1 次 act + 1 次 reflect

    def test_llm_act_mode_full_t4(self, tmp_path):
        """--act llm：每个子任务都问 LLM（ScriptedLLM 模拟正确输出）"""
        acts = [{"action_type": "open_app", "target_text": "模拟聊天", "input_text": ""},
                {"action_type": "wait", "target_text": "", "input_text": "2"},
                {"action_type": "click", "target_text": "张三", "input_text": ""},
                {"action_type": "type", "target_text": "消息内容", "input_text": "今晚7点开会"},
                {"action_type": "click", "target_text": "发送", "input_text": ""},
                {"action_type": "wait", "target_text": "", "input_text": "1"}]
        brain, fake = brain_with(acts + [{"is_complete": True, "reason": "已发送"}])
        agent, sim, ctx = make(tmp_path, brain=brain, act_mode="llm")
        r = run_task(agent, task("T4", ctx), ctx)
        assert r["success"] and all(s["source"] == "llm" for s in r["steps"])

    def test_llm_wrong_action_corrected_by_validator(self, tmp_path):
        """LLM 先点错（李四），校验器拦下并重试，第二次正确"""
        brain, fake = brain_with([{"action_type": "click", "target_text": "李四", "input_text": ""},
                                  {"action_type": "click", "target_text": "张三", "input_text": ""},
                                  {"is_complete": True, "reason": "ok"}])
        agent, sim, ctx = make(tmp_path, brain=brain, act_mode="llm")
        sim.spawn(["python", "mock_chat.py"])
        r = agent.run("x", plan=Plan(["点击联系人“张三”"], "test"))
        assert r["success"] and sim.chat["selected"] == "张三" and r["steps"][0]["attempts"][0]["llm_rejected"] == 1

    def test_retry_after_window_appears(self, tmp_path):
        """第一次感知时联系人还没加载出来，等待后重新感知成功"""
        agent, sim, ctx = make(tmp_path, step_retries=2)
        sim.spawn(["python", "mock_chat.py"])
        sim.hide_contacts = {"张三"}
        orig = sim.detect

        def detect(p):
            out = orig(p)
            sim.hide_contacts = set()
            return out
        sim.detect = detect
        r = agent.run("x", plan=Plan(["点击联系人“张三”"], "test"))
        assert r["success"] and len(r["steps"][0]["attempts"]) == 2

    def test_compound_instruction(self, tmp_path):
        agent, sim, ctx = make(tmp_path)
        r = agent.run("打开计算器，然后关闭计算器")
        assert r["success"] and "计算器" not in sim.windows.titles()

    def test_dry_run_executes_nothing(self, tmp_path):
        sim = SimDesktop(tmp_path)
        agent = GUIAgent(StaticPerceiver([{"text": "发送", "bbox": [0, 0, 10, 10]}]), DryRunExecutor(),
                         log=lambda *a: None)
        r = agent.run("关闭计算器")
        assert r["executed_all"] and sim.calls == []


@pytest.mark.boundary
class TestGUIAgentBoundary:
    def test_max_steps_truncates(self, tmp_path):
        agent, sim, ctx = make(tmp_path, max_steps=2)
        r = agent.run("给张三发送消息：hi")
        assert len(r["steps"]) == 2 and r["executed_all"] is False and r["success"] is False

    def test_empty_plan_fails(self, tmp_path):
        agent, sim, ctx = make(tmp_path)
        r = agent.run("x", plan=Plan([], "test"))
        assert r["success"] is False and r["reflection"]["is_complete"] is False

    def test_continue_after_failure_when_configured(self, tmp_path):
        agent, sim, ctx = make(tmp_path, stop_on_failure=False, step_retries=0)
        r = agent.run("x", plan=Plan(["关闭计算器", "打开记事本"], "test"))
        assert [s["ok"] for s in r["steps"]] == [False, True] and r["success"] is False

    def test_stop_on_failure_default(self, tmp_path):
        agent, sim, ctx = make(tmp_path, step_retries=0)
        r = agent.run("x", plan=Plan(["关闭计算器", "打开记事本"], "test"))
        assert len(r["steps"]) == 1

    def test_retries_zero(self, tmp_path):
        agent, sim, ctx = make(tmp_path, step_retries=0)
        r = agent.run("x", plan=Plan(["关闭计算器"], "test"))
        assert len(r["steps"][0]["attempts"]) == 1

    def test_retry_count(self, tmp_path):
        agent, sim, ctx = make(tmp_path, step_retries=3)
        r = agent.run("x", plan=Plan(["关闭计算器"], "test"))
        assert len(r["steps"][0]["attempts"]) == 4

    def test_invalid_action_not_retried(self, tmp_path):
        agent, sim, ctx = make(tmp_path, step_retries=3)
        r = agent.run("x", plan=Plan(["随便做点什么"], "test"))
        assert len(r["steps"][0]["attempts"]) == 1 and "规则无法推断" in str(r["steps"][0]["action"])

    def test_no_trace_dir(self, tmp_path):
        agent = GUIAgent(StaticPerceiver([]), DryRunExecutor(), log=lambda *a: None)
        assert "trace_path" not in agent.run("打开浏览器")


@pytest.mark.exception
class TestGUIAgentException:
    def test_checker_overrides_agent_claim(self, tmp_path):
        """Agent 认为完成，但检查器发现消息没发出去 → 失败（第3周“自称完成”问题的根治）"""
        agent, sim, ctx = make(tmp_path)
        r = agent.run("打开浏览器", checker=lambda: (False, "前台不是浏览器"))
        assert r["reflection"]["is_complete"] and r["success"] is False and r["check"]["ok"] is False

    def test_checker_exception(self, tmp_path):
        agent, sim, ctx = make(tmp_path)
        r = agent.run("打开浏览器", checker=lambda: 1 / 0)
        assert r["success"] is False and "ZeroDivisionError" in r["check"]["message"]

    def test_llm_reflect_guard(self, tmp_path):
        """子任务失败但 LLM 说完成 → 规则把关判未完成"""
        brain, fake = brain_with([{"is_complete": True, "reason": "都做完了"}])
        agent, sim, ctx = make(tmp_path, brain=brain, step_retries=0)
        r = agent.run("x", plan=Plan(["关闭计算器"], "test"))
        assert r["reflection"]["is_complete"] is False and r["reflection"]["llm_is_complete"] is True

    def test_plan_failure(self, tmp_path):
        brain, fake = brain_with([ConnectionError("Ollama 未启动")])
        agent, sim, ctx = make(tmp_path, brain=brain)
        r = agent.run("帮我写首诗")
        assert r["success"] is False and "规划失败" in r["error"] and r["plan"] is None

    def test_llm_act_failure_recorded(self, tmp_path):
        brain, fake = brain_with([ConnectionError("refused"), ConnectionError("refused"),
                                  {"is_complete": False, "reason": "x"}])
        agent, sim, ctx = make(tmp_path, brain=brain, act_mode="llm", step_retries=1)
        r = agent.run("x", plan=Plan(["打开记事本"], "test"))
        assert r["steps"][0]["ok"] is False and "决策失败" in r["steps"][0]["error"]

    def test_perception_failure_retried(self, tmp_path):
        agent, sim, ctx = make(tmp_path, step_retries=1)
        sim.spawn(["python", "mock_chat.py"])
        n = {"i": 0}
        orig = sim.detect

        def flaky(p):
            n["i"] += 1
            if n["i"] == 1:
                raise RuntimeError("OCR 超时")
            return orig(p)
        sim.detect = flaky
        r = agent.run("x", plan=Plan(["点击联系人“张三”"], "test"))
        assert r["success"] and "OCR 超时" in r["steps"][0]["attempts"][0]["error"]

    def test_llm_reflect_exception_falls_back_to_rule(self, tmp_path):
        brain, fake = brain_with([])
        agent, sim, ctx = make(tmp_path, brain=brain)
        r = agent.run("打开浏览器")
        assert r["reflection"]["mode"] == "rule" and r["reflection"]["is_complete"] is False

    @pytest.mark.parametrize("kw", [{"act_mode": "magic"}, {"act_mode": "llm"}, {"reflect_mode": "x"}])
    def test_invalid_config(self, kw):
        with pytest.raises(ValueError):
            GUIAgent(StaticPerceiver([]), DryRunExecutor(), **kw)

    def test_setup_failure_reported(self, tmp_path):
        agent, sim, ctx = make(tmp_path)
        ctx.apps.spawn = lambda cmd, **kw: None  # 计算器打不开（启动后没有窗口出现）
        r = run_task(agent, task("T5", ctx), ctx)
        assert r["success"] is False and "setup 失败" in r["error"]

    def test_message_to_wrong_contact_detected(self, tmp_path):
        agent, sim, ctx = make(tmp_path)
        t = task("T4", ctx)
        t.setup(ctx)
        r = agent.run("给李四发送消息：今晚7点开会", checker=lambda: t.check(ctx))
        assert r["executed_all"] and r["success"] is False and "没有发给「张三」" in r["check"]["message"]


CONSOLE_LINES = ["[指令] 给张三发送消息：今晚7点开会", "[规划] 来源=template:send_message，6 个子任务",
                 "3.点击联系人“张三”", '4。在"消息内容"框中输入 "今晚7点开会"', "5.点击“发送”"]


@pytest.mark.exception
class TestRealDeviceRegressionT4:
    """2026-11-24 Windows 真机 T4 失败复现：屏幕上同时有 Anaconda Prompt，OCR 读到了命令行里打印的子任务文字，
    输入框占位文字又没识别准，于是“消息内容”匹配到了命令行那一整行，消息被输进了命令行，聊天记录为空。"""

    def test_t4_with_console_and_misread_placeholder(self, tmp_path):
        agent, sim, ctx = make(tmp_path)
        sim.add_console(CONSOLE_LINES)
        sim.placeholder_text = "消息内客"  # OCR 错字
        r = run_task(agent, task("T4", ctx), ctx)
        assert r["success"], (r.get("check"), [s.get("action") for s in r["steps"]])
        assert sim.typed_to_nowhere == []

    def test_t4_with_console_exact_placeholder(self, tmp_path):
        agent, sim, ctx = make(tmp_path)
        sim.add_console(CONSOLE_LINES)
        assert run_task(agent, task("T4", ctx), ctx)["success"]

    def test_console_text_filtered_from_observation(self, tmp_path):
        agent, sim, ctx = make(tmp_path)
        sim.add_console(CONSOLE_LINES)
        sim.spawn(["python", "mock_chat.py"])
        obs = agent.perceiver.perceive()
        assert not any("指令" in t or "规划" in t for t in obs.texts()) and "张三" in obs.texts()
        assert obs.extra["filtered_out"] == len(CONSOLE_LINES)

    def test_long_console_line_not_grounded_without_region_filter(self, tmp_path):
        """即使拿不到窗口位置（不做区域过滤），长句也不能被当成“消息内容”输入框"""
        agent, sim, ctx = make(tmp_path)
        agent.perceiver.region_fn = None
        sim.add_console(CONSOLE_LINES)
        sim.placeholder_text = "消息内客"
        r = run_task(agent, task("T4", ctx), ctx)
        assert r["success"] and sim.typed_to_nowhere == []

    def test_field_not_recognized_step_fails_honestly(self, tmp_path):
        """第二轮真机：EasyOCR 没识别出输入框文字。输入步骤必须报失败，而不是把文字输到别处再报 ✓"""
        agent, sim, ctx = make(tmp_path, step_retries=1)
        sim.add_console(CONSOLE_LINES)
        sim.placeholder_text = None
        r = run_task(agent, task("T4", ctx), ctx)
        step = next(s for s in r["steps"] if "消息内容" in s["subtask"])
        assert r["success"] is False and step["ok"] is False and "找不到输入框" in step["error"]
        assert sim.typed_to_nowhere == [] and not any(c[0] == "type_text" for c in sim.calls)

    def test_t4_without_autofocus_clicks_label(self, tmp_path):
        """即使程序不自动聚焦输入框，点击“消息内容”标签也能把焦点移到输入框"""
        agent, sim, ctx = make(tmp_path)
        sim.autofocus = False
        assert run_task(agent, task("T4", ctx), ctx)["success"]

    def test_console_foreground_not_used_as_include_region(self, tmp_path):
        agent, sim, ctx = make(tmp_path)
        sim.spawn(["python", "mock_chat.py"])
        sim.add_console(CONSOLE_LINES)  # 命令行在前台
        reg = ctx.apps.screen_regions()
        assert reg["include"] is None and len(reg["exclude"]) == 1


@pytest.mark.multi_resolution
class TestGUIAgentMultiResolution:
    @pytest.mark.parametrize("dpi", [1.0, 1.25, 1.5, 1.75, 2.0])
    def test_t4_under_dpi_scaling(self, tmp_path, dpi):
        """Windows 显示缩放 100%~200%：OCR 坐标是物理像素，点击坐标必须换算回逻辑坐标才能点中"""
        agent, sim, ctx = make(tmp_path, dpi=dpi)
        r = run_task(agent, task("T4", ctx), ctx)
        assert r["success"], [s.get("error") for s in r["steps"]]

    def test_unscaled_coordinates_would_miss(self, tmp_path):
        """对照组：如果不做 DPI 换算（误把 scale 当 1），150% 缩放下点击会点偏、消息发不出去"""
        agent, sim, ctx = make(tmp_path, dpi=1.5)
        agent.perceiver.logical_size_fn = lambda: (2880, 1620)   # 错误地使用物理尺寸
        r = run_task(agent, task("T4", ctx), ctx)
        assert r["success"] is False

    @pytest.mark.parametrize("logical", [(1366, 768), (1920, 1080), (2560, 1440), (3840, 2160)])
    def test_t4_on_resolutions(self, tmp_path, logical):
        agent, sim, ctx = make(tmp_path, logical=logical)
        assert run_task(agent, task("T4", ctx), ctx)["success"]

    def test_chat_window_on_second_half_of_screen(self, tmp_path):
        agent, sim, ctx = make(tmp_path, logical=(2560, 1440))
        sim.ox, sim.oy = 1600, 700
        assert run_task(agent, task("T4", ctx), ctx)["success"]
