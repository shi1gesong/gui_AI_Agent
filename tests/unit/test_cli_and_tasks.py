"""第4周：agent/cli.py 命令行、agent/basic_tasks.py 检查器与报告 —— mock 用例（不会动鼠标）"""
import csv
import json

import pytest

from agent import basic_tasks as bt
from agent import cli
from control.app_manager import AppManager
from fakes import FakeWindowProvider


def ctx_with(tmp_path, titles=(), active=None):
    wins = FakeWindowProvider(titles)
    if active is not None:
        wins.active = next(w for w in wins.wins if w.title == active)
    return bt.TaskContext(AppManager(windows=wins, search_paths=[]), tmp_path, tmp_path / "chat.jsonl"), wins


@pytest.mark.functional
class TestCliAndTasksFunctional:
    def test_five_tasks_defined(self):
        tasks = bt.build_tasks()
        assert [t.id for t in tasks] == ["T1", "T2", "T3", "T4", "T5"]
        assert [t.name for t in tasks] == ["打开浏览器", "搜索指定内容", "打开指定文件", "发送消息", "关闭应用"]

    def test_every_task_matches_a_template(self):
        from agent.planner import RulePlanner
        for t in bt.build_tasks():
            assert RulePlanner().match(t.instruction) is not None, t.instruction

    def test_check_browser_foreground(self, tmp_path):
        ctx, _ = ctx_with(tmp_path, ["记事本", "必应 - Microsoft​ Edge"])
        assert bt.check_browser_foreground(ctx)[0]

    def test_check_search(self, tmp_path):
        ctx, _ = ctx_with(tmp_path, ["天气预报 - 搜索 - Microsoft Edge"])
        assert bt.make_check_search("天气预报")(ctx)[0]

    def test_check_file_window(self, tmp_path):
        ctx, _ = ctx_with(tmp_path, [f"{bt.DEMO_FILE_NAME} - 记事本"])
        assert bt.check_file_window(ctx)[0]

    def test_check_message(self, tmp_path):
        ctx, _ = ctx_with(tmp_path)
        ctx.chat_log.write_text(json.dumps({"event": "send", "to": "张三", "text": "今晚7点开会"}, ensure_ascii=False)
                                + "\n", encoding="utf-8")
        assert bt.make_check_message("张三", "今晚7点开会")(ctx)[0]

    def test_check_calc_closed(self, tmp_path):
        ctx, _ = ctx_with(tmp_path, ["记事本"])
        assert bt.check_calc_closed(ctx)[0]

    def test_setup_demo_file(self, tmp_path):
        ctx, _ = ctx_with(tmp_path)
        bt.setup_demo_file(ctx)
        assert ctx.demo_file.read_text(encoding="utf-8-sig").startswith("GUI Agent")

    def test_summarize_and_report(self, tmp_path):
        results = [{"task_id": "T1", "task_name": "打开浏览器", "instruction": "打开浏览器", "success": True,
                    "plan": {"source": "template:open_browser", "subtasks": ["a", "b"]},
                    "steps": [{"ok": True, "source": "rule"}, {"ok": True, "source": "rule"}],
                    "reflection": {"is_complete": True}, "check": {"ok": True}, "sec": 2.5},
                   {"task_id": "T1", "task_name": "打开浏览器", "instruction": "打开浏览器", "success": False,
                    "plan": {"source": "template:open_browser", "subtasks": ["a", "b"]},
                    "steps": [{"ok": False, "source": "rule", "error": "窗口没有出现"}],
                    "reflection": {"is_complete": False}, "check": {"ok": False, "message": "x"}, "sec": 10}]
        rows = bt.summarize(results)
        assert rows[1]["error"] == "窗口没有出现" and rows[0]["steps_ok"] == 2
        md = bt.write_report(rows, tmp_path, {"time": "t", "act_mode": "hybrid", "repeat": 2})
        assert "| T1 打开浏览器 | 1/2 | 50% |" in md and (tmp_path / "summary.md").exists()
        with open(tmp_path / "results.csv", encoding="utf-8-sig") as f:
            assert len(list(csv.DictReader(f))) == 2

    def test_cli_list_tasks(self, capsys):
        assert cli.main(["--list-tasks"]) == 0
        out = capsys.readouterr().out
        assert "T4" in out and "发送消息" in out

    def test_cli_plan_only_rule(self, capsys):
        assert cli.main(["--plan-only", "--no-llm", "在浏览器中搜索天气预报"]) == 0
        out = capsys.readouterr().out
        assert "template:search" in out and "直接输入“天气预报”并按回车" in out

    def test_cli_plan_only_task(self, capsys):
        assert cli.main(["--plan-only", "--no-llm", "--task", "T4"]) == 0
        assert "点击联系人“张三”" in capsys.readouterr().out

    def test_cli_parser_defaults(self):
        a = cli.build_parser().parse_args(["打开", "浏览器"])
        assert a.instruction == ["打开", "浏览器"] and a.act == "hybrid" and a.planner == "auto" and not a.think

    def test_make_config(self):
        a = cli.build_parser().parse_args(["--no-llm", "--model", "qwen3:4b", "--dry-run", "--retries", "2", "x"])
        cfg = cli.make_config(a)
        assert not cfg.use_llm and cfg.model == "qwen3:4b" and cfg.dry_run and cfg.step_retries == 2
        assert cfg.think is False


@pytest.mark.boundary
class TestCliAndTasksBoundary:
    def test_check_search_case_insensitive(self, tmp_path):
        ctx, _ = ctx_with(tmp_path, ["GUI Agent - 搜索 - Microsoft Edge"])
        assert bt.make_check_search("gui agent")(ctx)[0]

    def test_chat_log_missing(self, tmp_path):
        assert bt.read_chat_log(tmp_path / "none.jsonl") == []

    def test_chat_log_bad_lines_skipped(self, tmp_path):
        p = tmp_path / "c.jsonl"
        p.write_text('坏行\n{"event": "send", "to": "张三", "text": "hi"}\n', encoding="utf-8")
        assert len(bt.read_chat_log(p)) == 1

    def test_summarize_empty(self):
        assert bt.summarize([]) == []

    def test_cli_plan_only_needs_instruction(self, capsys):
        assert cli.main(["--plan-only", "--no-llm"]) == 2


@pytest.mark.exception
class TestCliAndTasksException:
    def test_check_browser_not_foreground(self, tmp_path):
        ctx, _ = ctx_with(tmp_path, ["必应 - Microsoft Edge", "记事本"])
        ok, msg = bt.check_browser_foreground(ctx)
        assert not ok and "不是浏览器" in msg

    def test_check_search_wrong_query(self, tmp_path):
        ctx, _ = ctx_with(tmp_path, ["新闻 - 搜索 - Microsoft Edge"])
        assert not bt.make_check_search("天气预报")(ctx)[0]

    def test_check_message_wrong_contact(self, tmp_path):
        ctx, _ = ctx_with(tmp_path)
        ctx.chat_log.write_text('{"event": "send", "to": "李四", "text": "今晚7点开会"}\n', encoding="utf-8")
        ok, msg = bt.make_check_message("张三", "今晚7点开会")(ctx)
        assert not ok and "李四" in msg

    def test_check_calc_still_open(self, tmp_path):
        ctx, _ = ctx_with(tmp_path, ["Calculator"])
        assert not bt.check_calc_closed(ctx)[0]

    def test_cli_unknown_task(self):
        with pytest.raises(SystemExit):
            cli.main(["--plan-only", "--task", "T9"])

    def test_cli_rule_plan_unknown_instruction(self, capsys):
        assert cli.main(["--plan-only", "--no-llm", "今天天气怎么样"]) == 2
        assert "规则模板无法识别" in capsys.readouterr().out


@pytest.mark.multi_resolution
class TestCliAndTasksMultiResolution:
    def test_checkers_use_titles_not_coordinates(self, tmp_path):
        ctx, wins = ctx_with(tmp_path, ["计算器"])
        wins.wins[0].left = 5000  # 窗口在副屏上
        assert not bt.check_calc_closed(ctx)[0]
