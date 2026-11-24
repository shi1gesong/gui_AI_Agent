"""第6周：agent/monitor.py 实时监控与日志（事件流、状态快照、阶段耗时、卡住检测、watch 查看端）"""
import json
import logging

import pytest

from agent.monitor import (STATES, TaskMonitor, atomic_write_json, build_logger, find_latest_run, read_status,
                           render_status, summarize_durations, watch)


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def tick(self, d):
        self.t += d


def plan_ev(mon, n_goals=2, per=3):
    goals = [{"id": f"G{i + 1}", "description": f"目标{i + 1}", "subtasks": [f"s{j}" for j in range(per)]}
             for i in range(n_goals)]
    return mon.event("plan", goals=goals, source="template")


@pytest.mark.functional
class TestMonitorFunctional:
    def test_writes_three_files(self, tmp_path):
        m = TaskMonitor(str(tmp_path))
        m.event("run_start", instruction="打开浏览器", engine="v2")
        m.close()
        assert (tmp_path / "events.jsonl").exists()
        assert (tmp_path / "agent.log").exists()
        assert (tmp_path / "status.json").exists()
        assert (tmp_path / "metrics.json").exists()

    def test_events_jsonl_one_line_per_event(self, tmp_path):
        m = TaskMonitor(str(tmp_path))
        for k in ("run_start", "plan", "goal_start"):
            m.event(k)
        m.close()
        lines = (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
        assert [json.loads(x)["event"] for x in lines] == ["run_start", "plan", "goal_start"]

    def test_event_has_seq_and_relative_time(self):
        clk = Clock()
        m = TaskMonitor(None, clock=clk)
        m.event("run_start")
        clk.tick(2.5)
        ev = m.event("plan", goals=[])
        assert ev["seq"] == 2 and ev["t"] == 2.5

    def test_state_machine_happy_path(self):
        m = TaskMonitor(None)
        assert m.state == "pending"
        m.event("run_start", instruction="x")
        assert m.state == "planning"
        plan_ev(m)
        assert m.state == "running"
        m.event("run_end", success=True, sec=1)
        assert m.state == "succeeded"

    def test_retry_enters_recovering_then_back_to_running(self):
        m = TaskMonitor(None)
        m.event("run_start")
        plan_ev(m)
        m.event("step_start", index=1, subtask="点击“发送”")
        m.event("retry", next_attempt=2, error_class="target_not_found", error="找不到", delay=0.5)
        assert m.state == "recovering"
        assert m.status["current"]["attempt"] == 2
        m.event("step_end", index=1, ok=True, sec=1.0)
        assert m.state == "running"

    def test_progress_and_eta(self):
        clk = Clock()
        m = TaskMonitor(None, clock=clk)
        m.event("run_start")
        plan_ev(m, 2, 2)
        for i in range(2):
            clk.tick(1.0)
            m.event("step_end", index=i + 1, ok=True, sec=1.0)
        p = m.status["progress"]
        assert (p["done"], p["total"], p["pct"]) == (2, 4, 50.0)
        assert m.status["eta"] == pytest.approx(2.0)

    def test_goal_states_tracked(self):
        m = TaskMonitor(None)
        m.event("run_start")
        plan_ev(m)
        m.event("goal_start", goal="G1", description="目标1")
        assert m.status["goals"][0]["state"] == "running"
        m.event("goal_end", goal="G1", ok=True)
        m.event("goal_start", goal="G2")
        m.event("goal_end", goal="G2", ok=False, reason="x")
        assert [g["state"] for g in m.status["goals"]] == ["succeeded", "failed"]
        assert m.counters["goals_ok"] == 1 and m.counters["goals_failed"] == 1

    def test_counters(self):
        m = TaskMonitor(None)
        m.event("retry")
        m.event("retry")
        m.event("recover", action="refocus", result="x")
        m.event("degrade", component="act", to="rule", reason="r")
        m.event("error", error="e")
        c = m.counters
        assert (c["retries"], c["recoveries"], c["degrades"], c["errors"]) == (2, 1, 1, 1)
        assert m.status["degraded"] == "rule"

    def test_stage_timings_from_events(self):
        m = TaskMonitor(None)
        m.event("perceive", sec=0.5)
        m.event("perceive", sec=1.5)
        m.event("execute", sec=0.1)
        st = m.metrics()["stages"]
        assert st["perceive"]["count"] == 2 and st["perceive"]["mean"] == 1.0
        assert "decide" not in st

    def test_timer_context_manager(self):
        clk = Clock()
        m = TaskMonitor(None, clock=clk)
        with m.timer("verify"):
            clk.tick(0.25)
        assert m.events[-1]["event"] == "verify" and m.events[-1]["sec"] == 0.25 and m.events[-1]["ok"] is True

    def test_listeners_called(self):
        seen = []
        m = TaskMonitor(None, listeners=[seen.append])
        m.event("run_start")
        assert seen and seen[0]["event"] == "run_start"

    def test_status_json_snapshot_matches(self, tmp_path):
        m = TaskMonitor(str(tmp_path))
        m.event("run_start", instruction="给张三发消息")
        plan_ev(m)
        st = read_status(tmp_path)
        assert st["instruction"] == "给张三发消息" and st["state"] == "running" and st["progress"]["total"] == 6
        m.close()

    def test_run_end_failed_and_aborted(self):
        m = TaskMonitor(None)
        m.event("run_end", success=False)
        assert m.state == "failed"
        m2 = TaskMonitor(None)
        m2.event("run_end", success=False, aborted=True)
        assert m2.state == "aborted"

    def test_agent_log_contains_human_readable_lines(self, tmp_path):
        m = TaskMonitor(str(tmp_path))
        m.event("run_start", instruction="打开记事本", engine="v2")
        m.event("retry", next_attempt=2, error_class="popup", error="弹窗", delay=0)
        m.close()
        text = (tmp_path / "agent.log").read_text(encoding="utf-8")
        assert "[开始] 打开记事本" in text and "WARNING" in text

    def test_render_status_contains_progress_and_goals(self):
        m = TaskMonitor(None)
        m.event("run_start", instruction="复杂任务")
        plan_ev(m)
        m.event("goal_start", goal="G1", description="目标1")
        m.event("step_start", index=1, subtask="打开模拟聊天")
        text = render_status(m.snapshot())
        assert "复杂任务" in text and "G1" in text and "打开模拟聊天" in text and "0/6" in text

    def test_watch_once_reads_latest_run(self, tmp_path):
        d = tmp_path / "runs" / "20261124_100000"
        m = TaskMonitor(str(d))
        m.event("run_start", instruction="打开浏览器")
        m.event("run_end", success=True, sec=1)
        m.close()
        out = []
        rc = watch(None, once=True, out=out.append, base=str(tmp_path / "runs"))
        assert rc == 0 and "打开浏览器" in out[0] and "succeeded" in out[0]

    def test_watch_loops_until_terminal(self, tmp_path):
        m = TaskMonitor(str(tmp_path))
        m.event("run_start")
        out, sleeps = [], []

        def fake_sleep(s):
            sleeps.append(s)
            m.event("run_end", success=True, sec=1)

        watch(str(tmp_path), interval=0.1, out=out.append, sleep=fake_sleep, max_loops=5)
        assert len(out) == 2 and len(sleeps) == 1

    def test_find_latest_run_picks_newest(self, tmp_path):
        import os
        import time
        for i, name in enumerate(["a", "b"]):
            d = tmp_path / name
            d.mkdir()
            (d / "status.json").write_text("{}", encoding="utf-8")
            os.utime(d / "status.json", (time.time() + i, time.time() + i))
        assert find_latest_run(tmp_path).name == "b"

    def test_metrics_written_on_close(self, tmp_path):
        m = TaskMonitor(str(tmp_path))
        m.event("decide", sec=0.2)
        m.close()
        data = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
        assert data["stages"]["decide"]["count"] == 1

    def test_format_event_known_kinds(self):
        f = TaskMonitor.format_event
        assert "降级" in f({"event": "degrade", "component": "act", "to": "rule", "reason": "x"})
        assert "恢复" in f({"event": "recover", "action": "refocus", "result": "ok"})
        assert f({"event": "unknown_kind"}) == ""


@pytest.mark.boundary
class TestMonitorBoundary:
    def test_no_run_dir_no_files(self, tmp_path):
        m = TaskMonitor(None)
        m.event("run_start")
        m.close()
        assert list(tmp_path.iterdir()) == []

    def test_summarize_empty(self):
        assert summarize_durations([]) == {"count": 0, "total": 0.0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}

    def test_summarize_single(self):
        s = summarize_durations([2.0])
        assert s["p50"] == s["p95"] == s["max"] == 2.0

    def test_summarize_p95(self):
        s = summarize_durations(range(1, 101))
        assert s["p95"] == 95 and s["max"] == 100

    def test_eta_none_before_any_step(self):
        m = TaskMonitor(None)
        m.event("run_start")
        plan_ev(m)
        assert m.status["eta"] is None

    def test_progress_total_grows_when_more_steps_than_planned(self):
        m = TaskMonitor(None)
        m.event("plan", goals=[{"id": "G1", "subtasks": ["a"]}])
        m.event("step_end", ok=True)
        m.event("step_end", ok=True)   # 目标重放：实际步数超过计划
        p = m.status["progress"]
        assert p["done"] == 2 and p["total"] == 2 and p["pct"] == 100.0

    def test_stuck_detection(self):
        clk = Clock()
        m = TaskMonitor(None, clock=clk, stuck_after=10)
        m.event("run_start")
        clk.tick(5)
        assert not m.check_stuck()
        clk.tick(6)
        assert m.check_stuck() and m.status["stuck"]

    def test_not_stuck_after_terminal(self):
        clk = Clock()
        m = TaskMonitor(None, clock=clk, stuck_after=1)
        m.event("run_end", success=True)
        clk.tick(100)
        assert not m.check_stuck()

    def test_render_none_status(self):
        assert "还没有状态文件" in render_status(None)

    def test_render_shows_stuck_warning(self):
        st = {"state": "running", "heartbeat": 0, "stuck_after": 10, "progress": {}, "elapsed": 1}
        assert "可能卡住" in render_status(st, now=100)

    def test_set_state_validates(self):
        m = TaskMonitor(None)
        for s in STATES:
            m.set_state(s)
        with pytest.raises(ValueError):
            m.set_state("flying")

    def test_build_logger_no_duplicate_handlers(self, tmp_path):
        a = build_logger(tmp_path)
        b = build_logger(tmp_path)
        assert a is b and len(b.handlers) == 1
        for h in b.handlers:
            h.close()

    def test_watch_missing_dir_returns_1(self, tmp_path):
        out = []
        assert watch(str(tmp_path / "nope"), once=True, out=out.append) == 1


@pytest.mark.exception
class TestMonitorException:
    def test_listener_exception_does_not_break(self):
        def bad(ev):
            raise RuntimeError("boom")
        m = TaskMonitor(None, listeners=[bad])
        m.event("run_start")
        m.event("plan", goals=[])
        assert m.listener_errors == 2 and m.state == "running"

    def test_unwritable_run_dir_falls_back(self, tmp_path):
        f = tmp_path / "file"
        f.write_text("x")
        m = TaskMonitor(str(f / "sub"))   # 父路径是文件，无法创建目录
        m.event("run_start")
        assert m.write_files is False and m.state == "planning"

    def test_timer_reraises_and_records_failure(self):
        m = TaskMonitor(None)
        with pytest.raises(ValueError):
            with m.timer("perceive"):
                raise ValueError("截图失败")
        ev = m.events[-1]
        assert ev["ok"] is False and "截图失败" in ev["error"]

    def test_read_status_corrupt_json(self, tmp_path):
        (tmp_path / "status.json").write_text("{not json", encoding="utf-8")
        assert read_status(tmp_path) is None

    def test_atomic_write_leaves_no_tmp(self, tmp_path):
        atomic_write_json(tmp_path / "s.json", {"a": 1})
        assert [p.name for p in tmp_path.iterdir()] == ["s.json"]

    def test_bad_plan_payload_recorded_not_raised(self):
        m = TaskMonitor(None)
        ev = m.event("plan", goals=[None])
        assert "monitor_error" in ev

    def test_close_twice_ok(self, tmp_path):
        m = TaskMonitor(str(tmp_path))
        m.close()
        m.close()

    def test_logger_level_warning_for_retry(self, tmp_path, caplog):
        m = TaskMonitor(None)
        m.logger.propagate = True
        with caplog.at_level(logging.WARNING, logger=m.logger.name):
            m.event("retry", next_attempt=2, error_class="x", error="y", delay=0)
        assert any(r.levelno == logging.WARNING for r in caplog.records)
