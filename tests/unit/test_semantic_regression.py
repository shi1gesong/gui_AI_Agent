"""第3周遗留：Acting 阶段语义正确性 —— 回放第3周真实输出 + 黄金用例

第3周报告中的端到端用例（"打开浏览器并搜索天气预报"）真实输出为：
  子任务1「打开默认浏览器」               -> click「设置」      （语义错误）
  子任务2「在浏览器的地址栏输入'天气预报'…」 -> click「搜索」      （应为 type，且没有输入内容）
  Reflecting                             -> is_complete=True   （实际上什么都没执行）
当时“Planning/Reflecting 通过、Acting 部分通过”。这里用回放的方式如实记录为失败（xfail，strict）。
"""
import json

import pytest

pytest.importorskip("langchain_core")

from fakes import ScriptedLLM  # noqa: E402
from base_agent import BaseGUIAgent, LangChainLLMWrapper  # noqa: E402
from semantic_eval import judge, load_cases, screen_context  # noqa: E402

CASES = load_cases()
WEEK3_SCREEN = "识别到的UI元素: [{'text': '搜索', 'bbox': [100,100,200,130]}, {'text': '设置', 'bbox': [300,50,380,80]}]"
WEEK3_PLAN = {"subtasks": ["打开默认浏览器", "在浏览器的地址栏输入'天气预报'并按下回车键"]}
WEEK3_ACT1 = {"action_type": "click", "target_text": "设置", "input_text": ""}
WEEK3_ACT2 = {"action_type": "click", "target_text": "搜索", "input_text": ""}
WEEK3_REFLECT = {"is_complete": True, "reason": "已在浏览器中执行了搜索操作，完成任务目标"}

# 第3周 demo 对应的黄金期望（屏幕上只有『搜索』『设置』，没有浏览器图标）
WEEK3_CASE1 = {"id": "W3-1", "subtask": "打开默认浏览器", "screen": ["搜索", "设置"],
               "expect_type": ["key", "click"], "expect_target": ["浏览器", "Google Chrome", "Edge"],
               "expect_input_any": ["win", "Win", "super"]}
WEEK3_CASE2 = {"id": "W3-2", "subtask": "在浏览器的地址栏输入'天气预报'并按下回车键", "screen": ["搜索", "设置"],
               "expect_type": ["type"], "expect_input": "天气预报"}

IDEAL = {
    "S01": {"action_type": "click", "target_text": "Google Chrome", "input_text": ""},
    "S02": {"action_type": "type", "target_text": "地址栏", "input_text": "天气预报"},
    "S03": {"action_type": "key", "target_text": "", "input_text": "enter"},
    "S04": {"action_type": "scroll", "target_text": "搜索结果", "input_text": ""},
    "S05": {"action_type": "click", "target_text": "设置", "input_text": ""},
    "S06": {"action_type": "click", "target_text": "记事本", "input_text": ""},
    "S07": {"action_type": "type", "target_text": "文本编辑区", "input_text": "Hello GUI Agent"},
    "S08": {"action_type": "click", "target_text": "文件", "input_text": ""},
    "S09": {"action_type": "click", "target_text": "另存为", "input_text": ""},
    "S10": {"action_type": "type", "target_text": "文件名", "input_text": "report.txt"},
    "S11": {"action_type": "click", "target_text": "保存", "input_text": ""},
    "S12": {"action_type": "type", "target_text": "搜索", "input_text": "张三"},
    "S13": {"action_type": "click", "target_text": "张三", "input_text": ""},
    "S14": {"action_type": "type", "target_text": "消息输入框", "input_text": "今晚开会"},
    "S15": {"action_type": "click", "target_text": "发送", "input_text": ""},
    "S16": {"action_type": "key", "target_text": "", "input_text": "tab"},
    "S17": {"action_type": "type", "target_text": "用户名", "input_text": "admin"},
    "S18": {"action_type": "click", "target_text": "登录", "input_text": ""},
    "S19": {"action_type": "scroll", "target_text": "", "input_text": ""},
    "S20": {"action_type": "click", "target_text": "天气预报 - 中国天气网", "input_text": ""},
}


def make_agent(responses):
    agent = BaseGUIAgent(llm_mode="api")
    agent.llm = LangChainLLMWrapper(inner_llm=ScriptedLLM(
        [r if isinstance(r, str) else json.dumps(r, ensure_ascii=False) for r in responses]))
    return agent


@pytest.mark.functional
class TestSemanticFunctional:
    @pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
    def test_golden_action_through_act_pipeline(self, case):
        """正确动作经 act() 的 Prompt 构造 + JSON 解析后，应被评测器判为通过（验证管线不破坏动作内容）"""
        agent = make_agent([IDEAL[case["id"]]])
        action = agent.act(case["subtask"], screen_context(case["screen"]))
        ok, reasons = judge(case, action)
        assert ok, reasons

    def test_evaluator_flags_week3_act1(self):
        ok, reasons = judge(WEEK3_CASE1, WEEK3_ACT1)
        assert not ok and any("设置" in r for r in reasons)

    def test_evaluator_flags_week3_act2(self):
        ok, reasons = judge(WEEK3_CASE2, WEEK3_ACT2)
        assert not ok and any("动作类型错误" in r for r in reasons)

    def test_evaluator_flags_hallucinated_target(self):
        ok, reasons = judge(CASES[0], {"action_type": "click", "target_text": "Firefox", "input_text": ""})
        assert not ok and "不在屏幕元素中" in reasons[0]


@pytest.mark.functional
class TestSemanticWeek3Regression:
    """第3周 demo 用例回放。假设模型每次重试都重复同样的错误（最坏情况），验证修复后的 Agent
    不再把错误动作交出去，并且 Reflect 不再误判完成。修复前这 3 项均失败（见 git 历史中的 strict xfail）。"""

    def _replay(self):
        agent = make_agent([WEEK3_PLAN] + [WEEK3_ACT1] * 3 + [WEEK3_ACT2] * 3 + [WEEK3_REFLECT])
        return agent.run("打开浏览器并搜索天气预报", lambda: WEEK3_SCREEN, max_steps=2)

    def test_week3_replay_runs_without_crash(self):
        res = self._replay()
        assert len(res["action_history"]) == 2

    def test_week3_wrong_outputs_were_rejected(self):
        """模型的原始输出（点击“设置”、点击“搜索”）被校验拦下并留档"""
        h = self._replay()["action_history"]
        assert h[0]["action"]["_rejected"][0]["output"]["target_text"] == "设置"
        assert any("无关" in r for r in h[0]["action"]["_rejected"][0]["reasons"])
        assert any("type" in r for r in h[1]["action"]["_rejected"][0]["reasons"])

    def test_week3_subtask1_semantically_correct(self):
        ok, reasons = judge(WEEK3_CASE1, self._replay()["action_history"][0]["action"])
        assert ok, reasons

    def test_week3_subtask2_semantically_correct(self):
        ok, reasons = judge(WEEK3_CASE2, self._replay()["action_history"][1]["action"])
        assert ok, reasons

    def test_week3_reflection_not_falsely_complete(self):
        """未接执行器，动作从未执行：即使 LLM 说完成，也必须判定未完成"""
        r = self._replay()["reflection"]
        assert r["is_complete"] is False and r["llm_is_complete"] is True
        assert any("未实际执行" in g for g in r["guard_reasons"])
