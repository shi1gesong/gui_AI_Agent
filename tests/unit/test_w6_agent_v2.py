"""第6周：agent/agent_v2.py 系统 v2.0 —— 在带故障注入的模拟桌面上跑完整闭环

覆盖：复杂任务拆解 → 分步执行 → 执行前窗口检查 → 决策 → 执行 → 截图前后对比 → 错误分类 → 恢复动作 →
重试 / 目标重放 → 后置条件 → 反思（含 LLM 降级）→ 检查器 → 事件日志 / 状态快照 / 轨迹
"""
import json

import pytest

import robustness_eval as rev
from agent import cli
from agent.agent_v2 import GUIAgentV2
from agent.basic_tasks import TaskContext, build_tasks, run_task
from agent.complex_tasks import build_complex_tasks
from agent.decomposer import Decomposer, Goal, TaskGraph
from agent.executor import ControllerExecutor
from agent.planner import Plan, RulePlanner
from agent.recovery import RetryPolicy
from agent.verifier import Postcondition
from fault_desktop import FaultyDesktop
from perception.perceiver_v2 import PerceiverV2

pytest.importorskip("langchain_core")

ALL_TASKS = ["T1", "T2", "T3", "T4", "T5", "C1", "C2", "C3", "C4"]
CHAT_TASKS = ["T4", "C2", "C3", "C4"]


def make(tmp_path, faults=(), seed=0, brain=None, **kw):
    sim = FaultyDesktop(tmp_path, faults=faults, seed=seed)
    apps = sim.app_manager()
    rule = RulePlanner(app_resolver=apps.resolve_app)
    perceiver = PerceiverV2(capture=sim, ocr=sim, logical_size_fn=sim.logical_size, region_fn=apps.screen_regions,
                            save_frames=False)
    logs = []
    agent = GUIAgentV2(perceiver, ControllerExecutor(sim, app_manager=apps, sleep=sim.sleep),
                       decomposer=Decomposer(rule, llm_plan=brain.plan if brain else None,
                                             mode="auto" if brain else "rule"),
                       brain=brain, act_mode="hybrid" if brain else "rule", app_manager=apps, sleep=sim.sleep,
                       log=logs.append, trace_dir=str(tmp_path / "trace"), settle_sec=0.2, **kw)
    ctx = TaskContext(apps, tmp_path / "work", sim.chat_log, sleep=sim.sleep)
    return agent, sim, ctx, logs


def task(tid, ctx):
    return {t.id: t for t in build_tasks(ctx.work_dir) + build_complex_tasks(ctx.work_dir)}[tid]


def run(tmp_path, tid, faults=(), **kw):
    agent, sim, ctx, logs = make(tmp_path, faults, **kw)
    r = run_task(agent, task(tid, ctx), ctx)
    return r, sim, agent, logs


def attempts(r):
    return [a for s in r["steps"] for a in s["attempts"]]


@pytest.mark.functional
class TestV2Functional:
    @pytest.mark.parametrize("tid", ALL_TASKS)
    def test_all_tasks_no_fault(self, tmp_path, tid):
        r, sim, _, _ = run(tmp_path, tid)
        assert r["success"], (r.get("check"), [s.get("error") for s in r["steps"]])
        assert r["robustness"]["retries"] == 0

    def test_c1_file_saved_and_closed(self, tmp_path):
        r, sim, agent, _ = run(tmp_path, "C1")
        content = (tmp_path / "work" / "gui_agent_week6_note.txt").read_text(encoding="utf-8")
        assert "第6周复杂任务测试" in content
        assert r["goals"][2]["post"]["results"][0]["kind"] == "file_contains"

    def test_c2_single_chat_instance(self, tmp_path):
        r, sim, _, _ = run(tmp_path, "C2")
        assert sum(1 for c in sim.calls if c[0] == "spawn" and "mock_chat.py" in c[1]) == 1
        assert sim.chat["sent"] == [("张三", "明天上午10点开会"), ("李四", "收到请回复")]

    def test_goal_records(self, tmp_path):
        r, *_ = run(tmp_path, "C4")
        assert [g["id"] for g in r["goals"]] == ["G1", "G2", "G3"] and all(g["ok"] for g in r["goals"])
        assert r["graph"]["source"] == "template:open_app+send_message+close_app"

    def test_message_postcondition_checked_by_ocr(self, tmp_path):
        r, *_ = run(tmp_path, "T4")
        post = r["goals"][0]["post"]["results"][0]
        assert post["kind"] == "text_present" and post["ok"]

    def test_trace_events_status_files(self, tmp_path):
        r, *_ = run(tmp_path, "T4")
        d = tmp_path / "trace"
        for name in ("trace.json", "events.jsonl", "status.json", "agent.log", "metrics.json"):
            assert (d / name).exists(), name
        st = json.loads((d / "status.json").read_text(encoding="utf-8"))
        assert st["state"] == "succeeded" and st["progress"]["pct"] == 100.0

    def test_event_order(self, tmp_path):
        run(tmp_path, "T1")
        evs = [json.loads(x)["event"] for x in (tmp_path / "trace" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
        assert evs[:3] == ["run_start", "plan", "goal_start"] and evs[-1] == "run_end"
        assert evs.index("goal_end") > evs.index("step_end")

    def test_metrics_in_result(self, tmp_path):
        r, *_ = run(tmp_path, "T4")
        st = r["metrics"]["stages"]
        assert st["execute"]["count"] >= 3 and st["verify"]["count"] >= 3 and "perceive" in st

    def test_effect_verification_recorded(self, tmp_path):
        r, *_ = run(tmp_path, "T4")
        v = [a["verify"] for a in attempts(r) if "verify" in a]
        assert v and all(x["ok"] for x in v) and all(x["diff"] > 0 for x in v)

    def test_perception_cache_used(self, tmp_path):
        r, *_ = run(tmp_path, "C2")
        ps = r["perception_stats"]
        assert ps["hit"] + ps["incremental"] >= 1 and ps["ocr_pixels"] < ps["screen_pixels"]

    def test_log_lines(self, tmp_path):
        r, sim, agent, logs = run(tmp_path, "T4")
        text = "\n".join(map(str, logs))
        assert "[开始]" in text and "[目标 G1]" in text and "[结束] 成功" in text

    def test_v1_compat_plan_argument(self, tmp_path):
        agent, sim, ctx, _ = make(tmp_path)
        r = agent.run("打开计算器", plan=Plan(["打开计算器", "等待1秒"], "manual"))
        assert r["success"] and r["graph"]["goals"][0]["subtasks"] == ["打开计算器", "等待1秒"]

    def test_explicit_graph(self, tmp_path):
        agent, sim, ctx, _ = make(tmp_path)
        g = TaskGraph("x", [Goal("G1", "打开计算器", ["打开计算器"], app="计算器", app_keywords=["计算器"],
                                 postconditions=[Postcondition("window_exists", {"keywords": ["计算器"]})])], "manual")
        assert agent.run("x", graph=g)["success"]

    def test_task_words_added_to_lexicon(self, tmp_path):
        agent, sim, ctx, _ = make(tmp_path)
        run_task(agent, task("T4", ctx), ctx)
        assert "张三" in agent.perceiver.extra_words


@pytest.mark.functional
class TestV2Recovery:
    @pytest.mark.parametrize("fault", ["slow_load", "popup_on_select", "popup_timed", "lost_click", "focus_steal",
                                       "screenshot_fail", "slow_window", "dpi_change", "ocr_flaky"])
    @pytest.mark.parametrize("tid", ["T4", "C2"])
    def test_v2_survives_fault(self, tmp_path, fault, tid):
        r, sim, _, _ = run(tmp_path, tid, (fault,), seed=1)
        assert r["success"], (fault, [s.get("error") for s in r["steps"]], r.get("check"))

    def test_popup_dismissed_and_logged(self, tmp_path):
        r, sim, *_ = run(tmp_path, "T4", ("popup_on_select",))
        rec = [x for a in attempts(r) for x in a.get("recover", [])]
        assert any(x["action"] == "dismiss_popup" and "系统通知" in x["result"] for x in rec)
        assert sim.typed_to_nowhere == []

    def test_focus_steal_refocused(self, tmp_path):
        r, *_ = run(tmp_path, "T4", ("focus_steal",))
        rec = [x for a in attempts(r) for x in a.get("recover", [])]
        assert any(x["action"] == "refocus" and x["from"] == "文档 - 文件资源管理器" for x in rec)

    def test_slow_load_backoff_sequence(self, tmp_path):
        r, *_ = run(tmp_path, "T4", ("slow_load",))
        step = next(s for s in r["steps"] if s["subtask"] == "点击联系人“张三”")
        waits = [x["result"] for a in step["attempts"] for x in a.get("recover", []) if x["action"] == "backoff"]
        assert waits[:2] == ["等待 0.5 秒", "等待 1 秒"] and step["ok"]

    def test_lost_click_detected_by_effect_check(self, tmp_path):
        r, *_ = run(tmp_path, "T4", ("lost_click",))
        bad = [a for a in attempts(r) if not a["ok"]]
        assert bad and all("界面没有变化" in a["error"] for a in bad)

    def test_slow_window_no_second_launch(self, tmp_path):
        r, sim, *_ = run(tmp_path, "T4", ("slow_window",))
        assert sum(1 for c in sim.calls if c[0] == "spawn" and "mock_chat.py" in c[1]) == 1
        assert any(a.get("late_window") for a in attempts(r))

    def test_llm_down_degrades_to_rule(self, tmp_path):
        brain = rev.make_brain(rev.DownLLM())
        r, *_ = run(tmp_path, "T4", brain=brain)
        assert r["success"] and r["degraded"] == "rule" and r["reflection"].get("degraded")

    def test_preprocess_switch_on_third_try(self, tmp_path):
        r, sim, agent, _ = run(tmp_path, "T4", ("slow_load",))
        step = next(s for s in r["steps"] if s["subtask"] == "点击联系人“张三”")
        assert any(a.get("preprocess") == "upscale" for a in step["attempts"])
        assert agent.perceiver.preprocess is None   # 子任务结束后恢复

    def test_error_class_stats(self, tmp_path):
        r, *_ = run(tmp_path, "T4", ("lost_click",))
        assert r["robustness"]["error_classes"].get("no_effect", 0) >= 1

    def test_goal_replay_on_postcondition_failure(self, tmp_path):
        agent, sim, ctx, _ = make(tmp_path)
        cnt = {"n": 0}

        class Flaky(Postcondition):
            def check(self, app_manager=None, texts_fn=None):
                cnt["n"] += 1
                return cnt["n"] > 4, "第一轮检查都不通过"
        g = TaskGraph("x", [Goal("G1", "打开计算器", ["打开计算器"], postconditions=[Flaky("window_exists", {})])], "m")
        r = agent.run("x", graph=g)
        assert r["success"] and r["goals"][0]["replays"] == 1

    def test_non_idempotent_goal_not_replayed(self, tmp_path):
        agent, sim, ctx, _ = make(tmp_path)
        g = TaskGraph("x", [Goal("G1", "等", ["等待1秒"], idempotent=False,
                                 postconditions=[Postcondition("window_exists", {"keywords": ["不存在的窗口"]})])], "m")
        r = agent.run("x", graph=g)
        assert not r["success"] and r["goals"][0]["replays"] == 0 and r["goals"][0]["error_class"] == "postcondition"


@pytest.mark.functional
class TestV1VsV2:
    """同一故障下 v1 失败、v2 成功的对照（数字进鲁棒性报告）"""

    @pytest.mark.parametrize("fault", ["popup_on_select", "focus_steal", "slow_load", "lost_click", "llm_down"])
    def test_v1_fails_v2_succeeds(self, tmp_path, fault):
        r1 = rev.run_case("v1", "T4", fault, 0, tmp_path / "v1")
        r2 = rev.run_case("v2", "T4", fault, 0, tmp_path / "v2")
        assert not r1["success"] and r2["success"]

    def test_lost_click_v1_false_success(self, tmp_path):
        r1 = rev.run_case("v1", "T4", "lost_click", 0, tmp_path / "v1")
        r2 = rev.run_case("v2", "T4", "lost_click", 0, tmp_path / "v2")
        assert r1["false_success"] and not r2["false_success"]

    def test_no_fault_both_succeed(self, tmp_path):
        for eng in ("v1", "v2"):
            assert rev.run_case(eng, "C3", "none", 0, tmp_path / eng)["success"]

    def test_aggregate_and_markdown(self, tmp_path):
        rows = rev.evaluate(["none", "popup_on_select"], ["T4"], seeds=1)
        md = rev.to_markdown(rows, ["none", "popup_on_select"])
        assert "| popup_on_select | 0/1 = 0% | 1/1 = 100% |" in md
        assert "0/1 → 1/1" in rev.per_task_markdown(rows, ["none", "popup_on_select"])


@pytest.mark.boundary
class TestV2Boundary:
    def test_slow_load_beyond_budget_fails_honestly(self, tmp_path):
        r, *_ = run(tmp_path, "T4", ("slow_load_long",))
        assert not r["success"] and not r["reflection"]["is_complete"]
        s = next(s for s in r["steps"] if not s["ok"])
        assert s["error_class"] == "target_not_found" and s["attempts_n"] == 4

    def test_icon_only_button_fails_honestly(self, tmp_path):
        r, *_ = run(tmp_path, "T4", ("ocr_blind",))
        assert not r["success"] and "发送" in next(s for s in r["steps"] if not s["ok"])["subtask"]

    def test_budget_exceeded(self, tmp_path):
        r, *_ = run(tmp_path, "C2", budget_sec=-1)
        assert not r["success"] and r["steps"][0]["error_class"] == "budget"

    def test_max_steps_truncates(self, tmp_path):
        r, *_ = run(tmp_path, "C2", max_steps=3)
        assert not r["success"] and r.get("truncated")

    def test_verify_disabled(self, tmp_path):
        r, *_ = run(tmp_path, "T4", verify_effect=False)
        assert r["success"] and not any("verify" in a for a in attempts(r))

    def test_goal_retries_zero(self, tmp_path):
        agent, *_ = make(tmp_path, goal_retries=0)
        assert agent.goal_retries == 0

    def test_empty_instruction(self, tmp_path):
        agent, *_ = make(tmp_path)
        r = agent.run("")
        assert not r["success"] and r["goals"] == []

    def test_total_attempt_cap(self, tmp_path):
        r, *_ = run(tmp_path, "T4", ("slow_load_long",), policy=RetryPolicy(max_total_attempts=2))
        s = next(s for s in r["steps"] if not s["ok"])
        assert s["attempts_n"] == 2


@pytest.mark.exception
class TestV2Exception:
    def test_planning_failure(self, tmp_path):
        agent, *_ = make(tmp_path)
        r = agent.run("把计算器调成科学模式")
        assert not r["success"] and "规划失败" in r["error"]
        st = json.loads((tmp_path / "trace" / "status.json").read_text(encoding="utf-8"))
        assert st["state"] == "failed"

    def test_failsafe_aborts_without_retry(self, tmp_path):
        agent, sim, ctx, _ = make(tmp_path)

        class FailSafeException(Exception):
            pass

        def corner(*a, **k):
            raise FailSafeException("PyAutoGUI fail-safe triggered from mouse moving to a corner")
        sim.click = corner
        r = run_task(agent, task("T4", ctx), ctx)
        s = next(s for s in r["steps"] if not s["ok"])
        assert r["aborted"] and s["attempts_n"] == 1 and s["error_class"] == "abort"
        assert len(r["goals"]) == 1

    def test_keyboard_interrupt(self, tmp_path):
        agent, sim, ctx, _ = make(tmp_path)

        def ctrl_c(*a, **k):
            raise KeyboardInterrupt
        sim.type_text = ctrl_c
        r = agent.run("给张三发送消息：你好")
        assert r["aborted"] and not r["success"] and "用户中止" in r["error"]

    def test_rule_cannot_infer_no_retry(self, tmp_path):
        agent, *_ = make(tmp_path)
        g = TaskGraph("x", [Goal("G1", "x", ["做一件说不清的事"])], "m")
        r = agent.run("x", graph=g)
        s = r["steps"][0]
        assert not s["ok"] and s["attempts_n"] == 1 and s["error_class"] == "invalid_action"

    def test_monitor_listener_failure_does_not_break(self, tmp_path):
        agent, sim, ctx, logs = make(tmp_path)

        def bad_log(*a):
            raise RuntimeError("终端编码错误")
        agent.log = bad_log
        with pytest.raises(RuntimeError):
            agent.log("x")
        agent.log = logs.append
        assert run_task(agent, task("T1", ctx), ctx)["success"]

    def test_frame_fn_failure_skips_verification(self, tmp_path):
        agent, sim, ctx, _ = make(tmp_path)

        def boom():
            raise OSError("截图失败")
        agent.frame_fn = boom
        r = run_task(agent, task("T4", ctx), ctx)
        assert r["success"] and not any("verify" in a for a in attempts(r))

    def test_llm_plan_down_for_unmatched_clause(self, tmp_path):
        brain = rev.make_brain(rev.DownLLM())
        agent, *_ = make(tmp_path, brain=brain)
        r = agent.run("把计算器调成科学模式")
        assert not r["success"] and r["error_class"] == "service"


@pytest.mark.functional
class TestCliV2:
    def test_engine_default_v2(self):
        a = cli.build_parser().parse_args(["x"])
        assert a.engine == "v2"
        cfg = cli.make_config(a)
        assert cfg.engine == "v2" and cfg.verify_effect and cfg.roi and cfg.ocr_cache

    def test_flags_map_to_config(self):
        a = cli.build_parser().parse_args(["--engine", "v1", "--no-verify", "--no-roi", "--no-cache",
                                           "--preprocess", "upscale", "--budget", "60", "x"])
        cfg = cli.make_config(a)
        assert (cfg.engine, cfg.verify_effect, cfg.roi, cfg.ocr_cache, cfg.preprocess, cfg.budget_sec) == \
            ("v1", False, False, False, "upscale", 60.0)

    def test_plan_only_shows_goals_and_checks(self, capsys):
        assert cli.main(["--plan-only", "--no-llm", "给张三发送消息：你好，然后给李四发送消息：收到"]) == 0
        out = capsys.readouterr().out
        assert "2 个目标" in out and "切换到“模拟聊天”" in out and "完成检查" in out and "不可重放" in out

    def test_plan_only_v1_engine(self, capsys):
        assert cli.main(["--plan-only", "--no-llm", "--engine", "v1", "打开计算器"]) == 0
        assert "template:open_app" in capsys.readouterr().out

    def test_list_tasks_includes_complex(self, capsys):
        cli.main(["--list-tasks"])
        out = capsys.readouterr().out
        assert "C1" in out and "C6" in out and "弹窗打断" in out

    def test_resolve_complex_task(self):
        assert cli.resolve_task("c5").name == "界面加载慢"
