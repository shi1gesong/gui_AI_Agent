"""第6周：agent/decomposer.py 复杂任务自动拆解（切分 → 逐段模板匹配 → 上下文 → 后置条件 → LLM 兜底）"""
import pytest

from agent.action_validator import SubtaskIntent, fallback_action
from agent.decomposer import Decomposer, Goal, TaskGraph, mask_quotes, split_clauses
from agent.planner import RulePlanner


def D(**kw):
    return Decomposer(RulePlanner(), **kw)


SPLIT_CASES = [
    ("打开记事本，输入你好，保存", ["打开记事本", "输入你好", "保存"]),
    ("先在浏览器里搜索天气预报，再给张三发送消息：天气已查好", ["在浏览器里搜索天气预报", "给张三发送消息：天气已查好"]),
    ("1. 给张三发消息：明天10点开会 2. 给李四发消息：收到请回复", ["给张三发消息：明天10点开会", "给李四发消息：收到请回复"]),
    ("第一步打开计算器；第二步关闭计算器", ["打开计算器", "关闭计算器"]),
    ("①打开记事本②关闭记事本", ["打开记事本", "关闭记事本"]),
    ("打开浏览器然后搜索天气，接着关闭浏览器", ["打开浏览器", "搜索天气", "关闭浏览器"]),
    ("打开计算器。关闭计算器。", ["打开计算器", "关闭计算器"]),
    ("打开计算器，之后关闭计算器", ["打开计算器", "关闭计算器"]),
    ("打开模拟聊天，最后关闭模拟聊天", ["打开模拟聊天", "关闭模拟聊天"]),
    ("请打开记事本并且输入你好", ["打开记事本", "输入你好"]),
]

NO_SPLIT_CASES = [
    "给张三发送消息：好的，再见",               # “再见”里的“再”不是连接词
    "给张三发消息：最后一次提醒",               # “最后”后面不是动词
    "给张三发送消息：你好，明天见",             # 消息里的逗号
    "打开文件 1.txt",                          # “1.”是文件名，不是编号
    "打开浏览器并搜索天气预报",                 # 一个基础任务
    "在浏览器中搜索天气预报",
]


@pytest.mark.functional
class TestSplit:
    @pytest.mark.parametrize("text,expected", SPLIT_CASES)
    def test_split(self, text, expected):
        assert split_clauses(text) == expected

    @pytest.mark.parametrize("text", NO_SPLIT_CASES)
    def test_no_split(self, text):
        assert len(split_clauses(text)) == 1

    def test_quotes_protected(self):
        assert split_clauses("给张三发送消息：“先吃饭，然后开会”，然后关闭模拟聊天") == [
            "给张三发送消息：“先吃饭，然后开会”", "关闭模拟聊天"]

    def test_mask_restore_roundtrip(self):
        masked, restore = mask_quotes("输入“a，b”和「c」")
        assert "，" not in masked.replace("和", "") or "\x00" in masked
        assert restore(masked) == "输入“a，b”和「c」"

    def test_windows_path_not_split(self):
        p = r"C:\Users\66\AppData\Local\Temp\gui_agent_week4\gui_agent_week6_note.txt"
        assert split_clauses(f"打开文件 {p}，然后保存") == [f"打开文件 {p}", "保存"]


@pytest.mark.functional
class TestGoals:
    def test_c1_style_context(self):
        g = D().decompose("打开文件 D:\\work\\note.txt，然后输入“第6周周报”，然后保存，最后按 ctrl+w 关闭标签页")
        assert [x.params["task_id"] for x in g.goals] == ["open_file", "type", "save", "hotkey"]
        assert g.goals[1].subtasks == ["直接输入“第6周周报”"]
        assert g.goals[1].app_keywords == ["note.txt", "note"]  # 输入目标继承上一个目标的程序
        save = g.goals[2]
        assert save.postconditions[0].kind == "file_contains"
        assert save.postconditions[0].args == {"path": "D:\\work\\note.txt", "text": "第6周周报"}

    def test_two_messages_second_switches(self):
        g = D().decompose("给张三发送消息：你好，然后给李四发送消息：收到")
        assert g.goals[0].subtasks[0] == "打开模拟聊天"
        assert g.goals[1].subtasks[0] == "切换到“模拟聊天”"
        assert "打开模拟聊天" not in g.goals[1].subtasks

    def test_message_after_close_reopens(self):
        g = D().decompose("给张三发送消息：你好，然后关闭模拟聊天，然后给李四发送消息：再见")
        assert g.goals[2].subtasks[0] == "打开模拟聊天"

    def test_message_postcondition_and_not_idempotent(self):
        g = D().decompose("给张三发送消息：今晚7点开会").goals[0]
        assert not g.idempotent
        assert g.postconditions[0].kind == "text_present" and g.postconditions[0].args["text"] == "张三：今晚7点开会"

    def test_search_postcondition(self):
        g = D().decompose("在浏览器中搜索天气预报").goals[0]
        assert g.postconditions[0].args["keywords"] == ["天气预报"] and g.idempotent

    def test_close_postcondition_window_absent(self):
        g = D().decompose("关闭计算器").goals[0]
        assert g.postconditions[0].kind == "window_absent" and "计算器" in g.postconditions[0].args["keywords"]

    def test_open_app_postcondition(self):
        g = D().decompose("打开计算器").goals[0]
        assert g.postconditions[0].kind == "window_exists" and g.app == "计算器"

    def test_switch_goal(self):
        g = D().decompose("切换到模拟聊天").goals[0]
        assert g.subtasks == ["切换到“模拟聊天”"] and g.postconditions[0].kind == "active_window"

    @pytest.mark.parametrize("word,keys", [("全选", "ctrl+a"), ("复制", "ctrl+c"), ("粘贴", "ctrl+v"), ("撤销", "ctrl+z")])
    def test_hotkey_words(self, word, keys):
        g = D().decompose(f"打开记事本，然后{word}").goals[1]
        assert g.subtasks == [f"按 {keys} {word}"] and g.app == "记事本"

    def test_explicit_hotkey(self):
        assert D().decompose("按 Ctrl + W").goals[0].subtasks == ["按 ctrl+w"]

    def test_key_clause(self):
        assert D().decompose("按回车").goals[0].subtasks == ["按回车"]

    def test_url(self):
        g = D().decompose("访问 www.bing.com").goals[0]
        assert g.subtasks[0] == "打开网址“https://www.bing.com”" and g.params["host"] == "bing.com"

    def test_wait_and_click(self):
        g = D().decompose("等待3秒，然后点击“发送”")
        assert g.goals[0].subtasks == ["等待3秒"] and g.goals[1].subtasks == ["点击“发送”"]

    def test_type_into_named_app(self):
        g = D().decompose("在记事本中输入你好").goals[0]
        assert g.subtasks == ["直接输入“你好”"] and "记事本" in g.app_keywords

    def test_type_into_ui_field(self):
        g = D().decompose("在“消息内容”框中输入你好").goals[0]
        assert g.subtasks == ["在“消息内容”框中输入“你好”"]

    def test_ids_sequential(self):
        g = D().decompose("打开计算器，然后关闭计算器")
        assert [x.id for x in g.goals] == ["G1", "G2"]

    def test_source_template_chain(self):
        assert D().decompose("打开计算器，然后关闭计算器").source == "template:open_app+close_app"

    def test_to_plan_compat(self):
        p = D().decompose("打开计算器，然后关闭计算器").to_plan()
        assert p.subtasks == ["打开计算器", "等待2秒", "关闭计算器", "等待1秒"]

    def test_to_dict_serializable(self):
        import json
        json.dumps(D().decompose("给张三发送消息：你好").to_dict(), ensure_ascii=False)

    @pytest.mark.parametrize("tid", ["T1", "T2", "T3", "T4", "T5"])
    def test_week4_basic_tasks_same_subtasks(self, tid):
        """v2 对第4周 5 个基础任务给出的子任务与 v1 完全一致（不破坏已有行为）"""
        from agent.basic_tasks import build_tasks
        t = {x.id: x for x in build_tasks()}[tid]
        assert D().decompose(t.instruction).subtasks == RulePlanner().match(t.instruction).subtasks

    @pytest.mark.parametrize("tid", ["C1", "C2", "C3", "C4", "C5", "C6"])
    def test_complex_tasks_all_rule_parsable(self, tid):
        """C1~C6 拆出的每个子任务都能被 SubtaskIntent/fallback_action 规则解析（不需要 LLM）"""
        from agent.complex_tasks import build_complex_tasks
        t = {x.id: x for x in build_complex_tasks()}[tid]
        g = D().decompose(t.instruction)
        assert g.source.startswith("template")
        for st in g.subtasks:
            a = fallback_action(st, [{"text": w, "bbox": [0, 0, 10, 10]} for w in
                                     ["张三", "李四", "王五", "消息内容", "发送"]])
            assert a is not None, st

    def test_llm_only_for_unmatched_clause(self):
        calls = []

        def llm_plan(c):
            calls.append(c)
            return ["点击“开始”"]
        g = D(llm_plan=llm_plan).decompose("打开计算器，然后把计算器调成科学模式，然后关闭计算器")
        assert calls == ["把计算器调成科学模式"]
        assert [x.source for x in g.goals] == ["template:open_app", "llm", "template:close_app"]
        assert g.source == "mixed"

    def test_llm_mode_whole_instruction(self):
        g = D(llm_plan=lambda c: ["a", "b"], mode="llm").decompose("打开计算器")
        assert g.source == "llm" and g.subtasks == ["a", "b"]


@pytest.mark.functional
class TestSwitchIntent:
    def test_switch_intent_parsed(self):
        i = SubtaskIntent("切换到“模拟聊天”")
        assert i.switch_target == "模拟聊天" and i.open_target is None and i.click_targets == []

    def test_switch_fallback_action(self):
        assert fallback_action("切换到“模拟聊天”", [])["action_type"] == "switch_app"

    def test_switch_validation(self):
        from agent.action_validator import validate_action
        assert validate_action("切换到“模拟聊天”", {"action_type": "switch_app", "target_text": "模拟聊天"}, []) == []
        bad = validate_action("切换到“模拟聊天”", {"action_type": "switch_app", "target_text": "记事本"}, [])
        assert bad and "切换" in bad[0]


@pytest.mark.boundary
class TestDecomposerBoundary:
    def test_empty(self):
        g = D().decompose("   ")
        assert g.goals == [] and g.source == "empty"

    def test_max_goals(self):
        text = "，然后".join(["打开计算器"] * 13)
        with pytest.raises(ValueError, match="超过上限"):
            D().decompose(text)

    def test_exactly_max_goals_ok(self):
        text = "，然后".join(["等待1秒"] * 12)
        assert len(D().decompose(text).goals) == 12

    def test_save_without_file_context_has_no_postcondition(self):
        g = D().decompose("打开记事本，输入你好，保存")
        assert g.goals[2].postconditions == []

    def test_type_without_context(self):
        g = D().decompose("输入你好").goals[0]
        assert g.app is None and g.app_keywords == []

    def test_close_clears_context(self):
        g = D().decompose("打开记事本，关闭记事本，输入你好")
        assert g.goals[2].app is None

    def test_short_file_stem_not_used_as_keyword(self):
        g = D().decompose("打开文件 a.txt").goals[0]
        assert g.app_keywords == ["a.txt"]

    def test_wechat_message_no_text_postcondition(self):
        g = D().decompose("用微信给张三发消息：你好").goals[0]
        assert g.postconditions == []

    def test_goal_to_dict_fields(self):
        d = Goal("G1", "x", ["a"]).to_dict()
        assert set(d) >= {"id", "subtasks", "postconditions", "idempotent", "app_keywords"}

    def test_taskgraph_subtasks_flatten(self):
        tg = TaskGraph("x", [Goal("G1", "a", ["1", "2"]), Goal("G2", "b", ["3"])], "t")
        assert tg.subtasks == ["1", "2", "3"]


@pytest.mark.exception
class TestDecomposerException:
    def test_unmatched_without_llm_names_clause(self):
        with pytest.raises(ValueError, match="把计算器调成科学模式"):
            D().decompose("打开计算器，然后把计算器调成科学模式")

    def test_rule_mode_ignores_llm(self):
        with pytest.raises(ValueError):
            D(llm_plan=lambda c: ["x"], mode="rule").decompose("做一件奇怪的事")

    def test_bad_mode(self):
        with pytest.raises(ValueError):
            D(mode="magic")

    def test_llm_mode_requires_llm(self):
        with pytest.raises(ValueError):
            D(mode="llm")

    def test_llm_failure_propagates(self):
        def boom(c):
            raise ConnectionError("Ollama 连接失败")
        with pytest.raises(ConnectionError):
            D(llm_plan=boom).decompose("做一件奇怪的事")

    def test_llm_returns_none(self):
        g = D(llm_plan=lambda c: None).decompose("做一件奇怪的事")
        assert g.goals[0].subtasks == []
