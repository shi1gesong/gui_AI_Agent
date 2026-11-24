"""Acting 动作校验 / 规则兜底 / 执行器（agent/action_validator.py, agent/executor.py）—— mock 用例"""
import json

import pytest

from action_validator import (SubtaskIntent, fallback_action, find_element, parse_screen_elements,
                              validate_action)
from executor import ControllerExecutor
from fakes import FakePyAutoGUI

ELEMS = [{"text": "搜索", "bbox": [100, 100, 200, 130]}, {"text": "设置", "bbox": [300, 50, 380, 80]},
         {"text": "地址栏", "bbox": [0, 0, 600, 30]}]


def A(t, target="", text=""):
    return {"action_type": t, "target_text": target, "input_text": text}


class RecController:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        return lambda *a, **k: self.calls.append((name, a))


@pytest.mark.functional
class TestValidatorFunctional:
    def test_parse_elements_python_repr(self):
        ctx = "识别到的UI元素: [{'text': '搜索', 'bbox': [1,2,3,4]}]"
        assert parse_screen_elements(ctx) == [{"text": "搜索", "bbox": [1, 2, 3, 4]}]

    def test_parse_elements_json(self):
        ctx = "识别到的UI元素: " + json.dumps(ELEMS, ensure_ascii=False)
        assert [e["text"] for e in parse_screen_elements(ctx)] == ["搜索", "设置", "地址栏"]

    def test_valid_click(self):
        assert validate_action("点击“设置”按钮", A("click", "设置"), ELEMS) == []

    def test_valid_type(self):
        assert validate_action("在地址栏输入天气预报", A("type", "地址栏", "天气预报"), ELEMS) == []

    def test_valid_key(self):
        assert validate_action("按回车键确认", A("key", "", "enter"), ELEMS) == []

    def test_valid_scroll(self):
        assert validate_action("向下滚动页面", A("scroll"), ELEMS) == []

    def test_open_app_click_unrelated_rejected(self):
        r = validate_action("打开默认浏览器", A("click", "设置"), ELEMS)
        assert r and "无关" in r[0]

    def test_type_task_as_click_rejected(self):
        r = validate_action("在浏览器的地址栏输入'天气预报'并按下回车键", A("click", "搜索"), ELEMS)
        assert any("应为 type" in x for x in r)

    def test_type_wrong_content_rejected(self):
        r = validate_action("输入天气预报", A("type", "地址栏", "天气"), ELEMS)
        assert r and "应输入" in r[0]

    def test_quoted_target_mismatch_rejected(self):
        r = validate_action("点击“设置”按钮", A("click", "搜索"), ELEMS)
        assert r and "点名" in r[0]

    def test_intent_extraction(self):
        i = SubtaskIntent("在浏览器的地址栏输入'天气预报'并按下回车键")
        assert i.type_intent and i.expected_text == "天气预报" and i.key == "enter" and i.open_target is None

    def test_intent_open_target(self):
        assert SubtaskIntent("打开默认浏览器").open_target == "浏览器"

    def test_fallback_open_app_on_screen_clicks_it(self):
        elems = ELEMS + [{"text": "Google Chrome", "bbox": [10, 500, 80, 560]}]
        assert fallback_action("打开浏览器", elems)["target_text"] == "Google Chrome"

    def test_fallback_type_with_enter(self):
        fb = fallback_action("在浏览器的地址栏输入'天气预报'并按下回车键", ELEMS)
        assert fb["action_type"] == "type" and fb["input_text"] == "天气预报" and fb["target_text"] == "地址栏"
        assert fb["follow_up"] == [{"action_type": "key", "target_text": "", "input_text": "enter"}]

    def test_real_qwen3_s12_fallback_targets_search_box(self):
        """真机 qwen3:8b 实测：「在微信搜索框输入张三」三次都输出 click「搜索」；兜底应输入到「搜索」框"""
        elems = [{"text": "搜索"}, {"text": "聊天"}, {"text": "通讯录"}]
        fb = fallback_action("在微信搜索框输入张三", elems)
        assert fb["action_type"] == "type" and fb["target_text"] == "搜索" and fb["input_text"] == "张三"

    def test_real_qwen3_s04_next_page_is_not_scroll(self):
        """真机 qwen3:8b 实测：「向下滚动页面查看更多结果」输出 click「下一页」，应被拦下并兜底为向下滚动"""
        elems = [{"text": "搜索结果"}, {"text": "下一页"}]
        r = validate_action("向下滚动页面查看更多结果", A("click", "下一页"), elems)
        assert r and "滚动" in r[0]
        fb = fallback_action("向下滚动页面查看更多结果", elems)
        assert fb["action_type"] == "scroll" and fb["amount"] < 0

    def test_scroll_up_fallback_positive(self):
        assert fallback_action("向上滚动回到页面顶部", [])["amount"] > 0

    def test_field_extraction(self):
        assert SubtaskIntent("在微信搜索框输入张三").field == "微信搜索框"
        assert SubtaskIntent("输入张三").field is None

    def test_executor_click_uses_bbox_center(self):
        ctl = RecController()
        assert ControllerExecutor(ctl)(A("click", "搜索"), ELEMS) == {"ok": True}
        assert ctl.calls == [("click", (150, 115))]

    def test_executor_type_clicks_field_first(self):
        ctl = RecController()
        ControllerExecutor(ctl)(A("type", "地址栏", "天气预报"), ELEMS)
        assert ctl.calls == [("click", (300, 15)), ("type_text", ("天气预报",))]

    def test_executor_key_alias(self):
        ctl = RecController()
        ControllerExecutor(ctl)(A("key", "", "回车"), ELEMS)
        assert ctl.calls == [("press_key", ("enter",))]

    def test_executor_runs_follow_up(self):
        ctl = RecController()
        act = fallback_action("打开默认浏览器", ELEMS)
        ControllerExecutor(ctl)(act, ELEMS)
        assert ctl.calls == [("press_key", ("win",)), ("type_text", ("浏览器",)), ("press_key", ("enter",))]

    def test_week3_task_end_to_end_with_real_controller_and_fake_gui(self, fake_gui, fake_clipboard):
        """第3周 demo：模型反复给出错误动作，修复后的 Agent 经兜底 + 执行器，真正按 Win→输入→回车→输入→回车 执行"""
        pytest.importorskip("langchain_core")
        from base_agent import BaseGUIAgent, LangChainLLMWrapper
        from control.mouse_keyboard import MouseKeyboardController
        from fakes import ScriptedLLM
        J = lambda o: json.dumps(o, ensure_ascii=False)  # noqa: E731
        plan = J({"subtasks": ["打开默认浏览器", "在浏览器的地址栏输入'天气预报'并按下回车键"]})
        act1, act2 = J(A("click", "设置")), J(A("click", "搜索"))
        reflect = J({"is_complete": True, "reason": "ok"})
        agent = BaseGUIAgent(executor=ControllerExecutor(MouseKeyboardController()))
        agent.llm = LangChainLLMWrapper(inner_llm=ScriptedLLM([plan] + [act1] * 3 + [act2] * 3 + [reflect]))
        screen = "识别到的UI元素: [{'text': '搜索', 'bbox': [100,100,200,130]}, {'text': '设置', 'bbox': [300,50,380,80]}]"
        res = agent.run("打开浏览器并搜索天气预报", lambda: screen)
        assert [c[0] for c in fake_gui.calls] == ["press", "hotkey", "press", "hotkey", "press"]
        assert fake_clipboard.history[0] == "浏览器" and "天气预报" in fake_clipboard.history
        assert res["reflection"]["is_complete"] is True


@pytest.mark.boundary
class TestValidatorBoundary:
    def test_parse_elements_empty(self):
        assert parse_screen_elements("") == []

    def test_parse_elements_fallback_regex(self):
        assert [e["text"] for e in parse_screen_elements("{'text': '确定'} 和 {'text': '取消'}")] == ["确定", "取消"]

    def test_find_element_substring(self):
        assert find_element("搜索框", ELEMS)["text"] == "搜索"

    def test_find_element_empty_target(self):
        assert find_element("", ELEMS) is None

    def test_input_box_word_not_type_intent(self):
        """“切换到下一个输入框”里的“输入框”不是输入意图"""
        assert not SubtaskIntent("按Tab键切换到下一个输入框").type_intent

    def test_no_elements_skips_grounding(self):
        assert validate_action("点击确定", A("click", "确定"), []) == []

    def test_fallback_none_when_no_intent(self):
        assert fallback_action("点击第一条结果", ELEMS) is None

    def test_scroll_task_allows_click_back_to_top(self):
        elems = [{"text": "返回顶部"}]
        assert validate_action("向上滚动回到页面顶部", A("click", "返回顶部"), elems) == []


@pytest.mark.exception
class TestValidatorException:
    def test_parse_error_action(self):
        assert validate_action("点击", {"_parse_error": True}, ELEMS) == ["输出无法解析为 JSON 对象"]

    def test_non_dict_action(self):
        assert validate_action("点击", ["click"], ELEMS)

    def test_invalid_action_type(self):
        assert "action_type" in validate_action("点击", A("click/type", "搜索"), ELEMS)[0]

    def test_click_missing_target(self):
        assert "缺少 target_text" in validate_action("点击", A("click"), ELEMS)[0]

    def test_type_empty_input(self):
        assert "input_text 为空" in validate_action("填写表单", A("type", "地址栏", ""), ELEMS)[0]

    def test_unknown_key_name(self):
        assert "无法识别的按键" in validate_action("按键", A("key", "", "hyperkey"), ELEMS)[0]

    def test_executor_missing_target_returns_error(self):
        r = ControllerExecutor(RecController())(A("click", "不存在"), ELEMS)
        assert r["ok"] is False and "LookupError" in r["error"]

    def test_executor_skips_invalid_action(self):
        ctl = RecController()
        r = ControllerExecutor(ctl)({**A("click", "搜索"), "_invalid": True}, ELEMS)
        assert r["ok"] is False and ctl.calls == []

    def test_executor_controller_exception_captured(self):
        class Boom:
            def click(self, *a):
                raise ValueError("坐标超出屏幕范围")
        r = ControllerExecutor(Boom())(A("click", "搜索"), ELEMS)
        assert r["ok"] is False and "坐标超出" in r["error"]

    def test_executor_unknown_type(self):
        assert ControllerExecutor(RecController())({"action_type": "hover"}, ELEMS)["ok"] is False
