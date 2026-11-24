"""第3周：agent/base_agent.py（Planning-Acting-Reflecting 三阶段框架）—— mock 用例

LLM 用 ScriptedLLM 按顺序返回预设文本，覆盖：正常 JSON、夹带解释文字、markdown 代码块、
qwen3 的 <think> 思考段、非法 JSON、LLM 服务异常等情况。
"""
import json
import subprocess
import sys

import pytest

pytest.importorskip("langchain_core")

from fakes import ScriptedLLM  # noqa: E402
from base_agent import BaseGUIAgent, LangChainLLMWrapper  # noqa: E402
from llm_interface import OllamaAPIInterface  # noqa: E402

SCREEN = "识别到的UI元素: [{'text': '搜索', 'bbox': [100,100,200,130]}, {'text': '设置', 'bbox': [300,50,380,80]}]"


def J(obj):
    return json.dumps(obj, ensure_ascii=False)


def make_agent(responses):
    agent = BaseGUIAgent(llm_mode="api")
    fake = ScriptedLLM(responses)
    agent.llm = LangChainLLMWrapper(inner_llm=fake)
    return agent, fake


ACT_CLICK = J({"action_type": "click", "target_text": "搜索", "input_text": ""})
REFLECT_OK = J({"is_complete": True, "reason": "done", "next_action_suggestion": ""})


@pytest.mark.functional
class TestAgentFunctional:
    def test_default_agent_uses_ollama(self):
        agent = BaseGUIAgent()
        assert isinstance(agent.llm.inner, OllamaAPIInterface)

    def test_agent_forwards_llm_kwargs(self):
        agent = BaseGUIAgent(llm_mode="api", model_name="qwen2.5:1.5b")
        assert agent.llm.inner.model_name == "qwen2.5:1.5b"

    def test_wrapper_llm_type(self):
        assert LangChainLLMWrapper(inner_llm=ScriptedLLM([]))._llm_type == "custom_gui_agent_llm"

    def test_wrapper_invoke_delegates(self):
        fake = ScriptedLLM(["回复"])
        assert LangChainLLMWrapper(inner_llm=fake).invoke("问题") == "回复"
        assert fake.prompts == ["问题"]

    def test_plan_parses_subtasks(self):
        agent, _ = make_agent([J({"subtasks": ["打开浏览器", "输入天气预报"]})])
        assert agent.plan("打开浏览器并搜索天气预报") == ["打开浏览器", "输入天气预报"]

    def test_plan_prompt_contains_instruction(self):
        agent, fake = make_agent([J({"subtasks": []})])
        agent.plan("打开记事本")
        assert "用户指令：打开记事本" in fake.prompts[0]

    def test_act_returns_action(self):
        agent, _ = make_agent([ACT_CLICK])
        assert agent.act("点击搜索", SCREEN) == {"action_type": "click", "target_text": "搜索", "input_text": ""}

    def test_act_prompt_contains_subtask_and_screen(self):
        agent, fake = make_agent([ACT_CLICK])
        agent.act("点击搜索", SCREEN)
        assert "点击搜索" in fake.prompts[0] and "[100,100,200,130]" in fake.prompts[0]

    def test_act_records_history(self):
        agent, _ = make_agent([ACT_CLICK])
        agent.act("点击搜索", SCREEN)
        h = agent.action_history
        assert len(h) == 1 and h[0]["subtask"] == "点击搜索" and h[0]["action"] == json.loads(ACT_CLICK)
        assert h[0]["attempts"] == 1 and h[0]["executed"] is False

    def test_type_action_with_chinese_input(self):
        agent, _ = make_agent([J({"action_type": "type", "target_text": "搜索框", "input_text": "天气预报"})])
        assert agent.act("在搜索框输入天气预报", SCREEN)["input_text"] == "天气预报"

    def test_reflect_prompt_lists_history(self):
        agent, fake = make_agent([ACT_CLICK, REFLECT_OK])
        agent.act("点击搜索", SCREEN)
        agent.reflect("搜索")
        assert "- 点击搜索: {" in fake.prompts[1] and "任务目标：搜索" in fake.prompts[1]

    def test_reflect_returns_dict(self):
        agent, _ = make_agent([REFLECT_OK])
        r = agent.reflect("x")
        assert isinstance(r, dict) and r["reason"] == "done"

    def test_reflect_keeps_llm_true_when_all_executed(self):
        agent, _ = make_agent([J({"subtasks": ["点击搜索"]}), ACT_CLICK, REFLECT_OK])
        agent.executor = lambda action, elements: {"ok": True}
        res = agent.run("搜索", lambda: SCREEN)
        assert res["reflection"]["is_complete"] is True and res["action_history"][0]["executed"] is True

    def test_executor_receives_action_and_elements(self):
        agent, _ = make_agent([J({"subtasks": ["点击搜索"]}), ACT_CLICK, REFLECT_OK])
        seen = []
        agent.executor = lambda action, elements: seen.append((action, elements)) or {"ok": True}
        agent.run("搜索", lambda: SCREEN)
        assert seen[0][0]["target_text"] == "搜索" and [e["text"] for e in seen[0][1]] == ["搜索", "设置"]

    def test_retry_with_feedback_then_success(self):
        bad = J({"action_type": "click", "target_text": "设置", "input_text": ""})
        good = J({"action_type": "type", "target_text": "搜索", "input_text": "天气预报"})
        agent, fake = make_agent([bad, good])
        action = agent.act("在搜索框输入天气预报", SCREEN)
        assert action["action_type"] == "type" and agent.action_history[0]["attempts"] == 2
        assert "不合格" in fake.prompts[1] and "不合格" not in fake.prompts[0]
        assert action["_rejected"][0]["output"]["target_text"] == "设置"

    def test_acting_prompt_example_no_longer_settings(self):
        """第3周 Prompt 的示例就是点击“设置”，模型把“打开浏览器”照抄成了点击“设置”"""
        agent, fake = make_agent([ACT_CLICK])
        agent.act("点击搜索", SCREEN)
        assert '"target_text": "设置"' not in fake.prompts[0]

    def test_run_full_loop_call_count(self):
        agent, fake = make_agent([J({"subtasks": ["a", "b"]}), ACT_CLICK, ACT_CLICK, REFLECT_OK])
        agent.run("指令", lambda: SCREEN)
        assert len(fake.prompts) == 4  # plan + 2*act + reflect

    def test_run_calls_screen_fn_each_step(self):
        agent, _ = make_agent([J({"subtasks": ["a", "b", "c"]}), ACT_CLICK, ACT_CLICK, ACT_CLICK, REFLECT_OK])
        n = []
        agent.run("指令", lambda: n.append(1) or SCREEN)
        assert len(n) == 3

    def test_run_passes_fresh_screen_to_each_act(self):
        agent, fake = make_agent([J({"subtasks": ["a", "b"]}), ACT_CLICK, ACT_CLICK, REFLECT_OK])
        screens = iter(["屏幕状态1", "屏幕状态2"])
        agent.run("指令", lambda: next(screens))
        assert "屏幕状态1" in fake.prompts[1] and "屏幕状态2" in fake.prompts[2]

    def test_run_result_structure(self):
        agent, _ = make_agent([J({"subtasks": ["a"]}), ACT_CLICK, REFLECT_OK])
        res = agent.run("指令", lambda: SCREEN)
        assert set(res) == {"subtasks", "action_history", "reflection"}
        assert res["subtasks"] == ["a"] and len(res["action_history"]) == 1

    def test_parse_pure_json(self):
        agent, _ = make_agent([])
        assert agent._safe_json_parse('{"a": 1}') == {"a": 1}

    def test_parse_json_wrapped_in_prose(self):
        agent, _ = make_agent([])
        assert agent._safe_json_parse('好的，结果如下：{"a": 1} 希望有帮助') == {"a": 1}

    def test_parse_markdown_code_fence(self):
        agent, _ = make_agent([])
        assert agent._safe_json_parse('```json\n{"subtasks": ["x"]}\n```') == {"subtasks": ["x"]}

    def test_parse_multiline_pretty_json(self):
        agent, _ = make_agent([])
        assert agent._safe_json_parse('{\n  "a": [1,\n 2]\n}') == {"a": [1, 2]}


@pytest.mark.boundary
class TestAgentBoundary:
    def test_plan_missing_key_returns_empty(self):
        agent, _ = make_agent([J({"steps": ["a"]})])
        assert agent.plan("x") == []

    def test_plan_empty_subtasks(self):
        agent, _ = make_agent([J({"subtasks": []})])
        assert agent.plan("x") == []

    def test_run_zero_subtasks_still_reflects(self):
        agent, fake = make_agent([J({"subtasks": []}), REFLECT_OK])
        res = agent.run("x", lambda: SCREEN)
        assert res["action_history"] == [] and len(fake.prompts) == 2

    def test_run_truncates_to_max_steps(self):
        agent, _ = make_agent([J({"subtasks": list("abcdefg")})] + [ACT_CLICK] * 5 + [REFLECT_OK])
        res = agent.run("x", lambda: SCREEN, max_steps=5)
        assert len(res["action_history"]) == 5 and len(res["subtasks"]) == 7

    def test_run_max_steps_zero(self):
        agent, _ = make_agent([J({"subtasks": ["a"]}), REFLECT_OK])
        assert agent.run("x", lambda: SCREEN, max_steps=0)["action_history"] == []

    def test_run_subtasks_equal_max_steps(self):
        agent, _ = make_agent([J({"subtasks": ["a", "b"]}), ACT_CLICK, ACT_CLICK, REFLECT_OK])
        assert len(agent.run("x", lambda: SCREEN, max_steps=2)["action_history"]) == 2

    def test_reflect_with_empty_history(self):
        agent, fake = make_agent([REFLECT_OK])
        r = agent.reflect("x")
        assert "动作记录（含是否真正执行成功）：\n\n" in fake.prompts[0]
        assert r["is_complete"] is False and r["llm_is_complete"] is True

    def test_parse_empty_string(self):
        agent, _ = make_agent([])
        assert agent._safe_json_parse("")["_parse_error"] is True

    def test_parse_nested_braces_in_values(self):
        agent, _ = make_agent([])
        raw = '结果：{"action_type": "type", "target_text": "框", "input_text": "{x}"}'
        assert agent._safe_json_parse(raw)["input_text"] == "{x}"

    def test_parse_only_closing_brace(self):
        agent, _ = make_agent([])
        assert agent._safe_json_parse("} oops")["_parse_error"] is True

    def test_parse_keeps_raw_on_error(self):
        agent, _ = make_agent([])
        assert agent._safe_json_parse("无法回答")["raw"] == "无法回答"

    def test_screen_context_with_braces_in_prompt(self):
        """screen_context 本身含 { } 时 PromptTemplate.format 不应报错"""
        agent, fake = make_agent([ACT_CLICK])
        agent.act("点击", "{'text': '搜索'} {{双括号}}")
        assert "{{双括号}}" in fake.prompts[0]

    def test_very_long_screen_context(self):
        agent, fake = make_agent([ACT_CLICK])
        agent.act("点击", "元素," * 20000)
        assert len(fake.prompts[0]) > 60000

    def test_parse_qwen3_think_block_with_braces(self):
        agent, _ = make_agent([])
        raw = '<think>用户要我输出 {"subtasks": ...} 格式</think>\n{"subtasks": ["打开浏览器"]}'
        assert agent._safe_json_parse(raw) == {"subtasks": ["打开浏览器"]}

    def test_parse_two_json_objects_takes_first(self):
        agent, _ = make_agent([])
        assert agent._safe_json_parse('{"a": 1}\n{"b": 2}') == {"a": 1}

    def test_run_twice_history_isolated(self):
        agent, _ = make_agent([J({"subtasks": ["a"]}), ACT_CLICK, REFLECT_OK,
                               J({"subtasks": ["b"]}), ACT_CLICK, REFLECT_OK])
        agent.run("任务1", lambda: SCREEN)
        res = agent.run("任务2", lambda: SCREEN)
        assert len(res["action_history"]) == 1

    def test_act_rejects_invalid_action_type(self):
        agent, _ = make_agent([J({"action_type": "click/type", "target_text": "搜索", "input_text": ""})] * 3)
        action = agent.act("点击搜索", SCREEN)
        assert action.get("_invalid") and "action_type" in action["_reasons"][0]

    def test_act_rejects_target_not_on_screen(self):
        agent, _ = make_agent([J({"action_type": "click", "target_text": "Chrome", "input_text": ""})] * 3)
        action = agent.act("打开浏览器", SCREEN)
        assert action["_fallback"] and action["action_type"] == "key" and action["input_text"] == "win"
        assert action["follow_up"][0] == {"action_type": "type", "target_text": "", "input_text": "浏览器"}

    def test_max_retries_zero_no_retry(self):
        agent, fake = make_agent([J({"action_type": "click", "target_text": "不存在", "input_text": ""})])
        agent.max_retries = 0
        assert agent.act("点击按钮", SCREEN)["_invalid"] is True and len(fake.prompts) == 1

    def test_act_list_response_takes_first(self):
        agent, _ = make_agent([J([json.loads(ACT_CLICK), {"action_type": "key"}])])
        assert agent.act("点击搜索", SCREEN)["target_text"] == "搜索"

    def test_parse_only_closing_think_tag(self):
        agent, _ = make_agent([])
        assert agent._safe_json_parse('思考 {x}</think>{"a": 2}') == {"a": 2}

    def test_plan_filters_empty_and_non_string(self):
        agent, _ = make_agent([J({"subtasks": ["打开", "", "  ", None, {"x": 1}, "搜索"]})])
        assert agent.plan("x") == ["打开", "搜索"]


@pytest.mark.exception
class TestAgentException:
    def test_invalid_llm_mode(self):
        with pytest.raises(ValueError):
            BaseGUIAgent(llm_mode="gpt")

    def test_llm_error_in_plan_propagates(self):
        agent, _ = make_agent([ConnectionError("Ollama 未启动")])
        with pytest.raises(ConnectionError):
            agent.plan("x")

    def test_llm_error_mid_run_keeps_partial_history(self):
        agent, _ = make_agent([J({"subtasks": ["a", "b"]}), ACT_CLICK, TimeoutError("timeout")])
        with pytest.raises(TimeoutError):
            agent.run("x", lambda: SCREEN)
        assert len(agent.action_history) == 1

    def test_screen_fn_error_propagates(self):
        agent, _ = make_agent([J({"subtasks": ["a"]})])

        def broken():
            raise RuntimeError("截图失败")
        with pytest.raises(RuntimeError, match="截图失败"):
            agent.run("x", broken)

    def test_garbage_plan_returns_empty(self):
        agent, _ = make_agent(["抱歉，我无法完成"])
        assert agent.plan("x") == []

    def test_garbage_act_marked_parse_error(self):
        agent, fake = make_agent(["我觉得应该点击搜索按钮"] * 3)
        action = agent.act("点击搜索", SCREEN)
        assert action["_parse_error"] is True and action["_invalid"] is True and len(fake.prompts) == 3

    def test_executor_failure_recorded_and_reflect_false(self):
        agent, _ = make_agent([J({"subtasks": ["点击搜索"]}), ACT_CLICK, REFLECT_OK])
        agent.executor = lambda action, elements: {"ok": False, "error": "LookupError"}
        res = agent.run("搜索", lambda: SCREEN)
        assert res["action_history"][0]["error"] == "LookupError"
        assert res["reflection"]["is_complete"] is False

    def test_invalid_action_not_sent_to_executor(self):
        agent, _ = make_agent([J({"subtasks": ["点击按钮"]})] + ["胡言乱语"] * 3 + [REFLECT_OK])
        calls = []
        agent.executor = lambda a, e: calls.append(a) or {"ok": True}
        agent.run("x", lambda: SCREEN)
        assert calls == []

    def test_reflect_garbage_forced_incomplete(self):
        agent, _ = make_agent([J({"subtasks": ["点击搜索"]}), ACT_CLICK, "完成了"])
        agent.executor = lambda a, e: {"ok": True}
        assert agent.run("x", lambda: SCREEN)["reflection"]["is_complete"] is False

    def test_garbage_reflect_marked_parse_error(self):
        agent, _ = make_agent(["任务完成了"])
        assert agent.reflect("x")["_parse_error"] is True

    def test_truncated_json_marked_parse_error(self):
        agent, _ = make_agent(['{"action_type": "click", "target_text": "搜'] * 3)
        assert agent.act("点击", SCREEN)["_parse_error"] is True

    def test_plan_json_array_response(self):
        agent, _ = make_agent([J(["打开浏览器", "搜索"])])
        assert agent.plan("x") == ["打开浏览器", "搜索"]

    def test_plan_subtasks_string_rejected(self):
        agent, _ = make_agent([J({"subtasks": "打开浏览器"})])
        assert agent.plan("x") == ["打开浏览器"]

    def test_import_as_package(self, tmp_path):
        from conftest import REPO_ROOT
        code = "import sys; sys.path.insert(0, %r); import agent.base_agent" % str(REPO_ROOT)
        r = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-300:]
