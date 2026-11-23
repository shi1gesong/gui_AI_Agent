"""真实 LLM 用例（llm 标记，默认跳过）

运行方式（Windows 或 WSL2 均可，需 Ollama 已拉取模型）：
    set OLLAMA_BASE_URL=http://localhost:11434      (WSL2 中用 ip route 得到的网关地址)
    set OLLAMA_MODEL=qwen3:8b
    pytest -m llm -v
"""
import os

import pytest

from semantic_eval import judge, load_cases, screen_context

pytestmark = pytest.mark.llm
CASES = load_cases()


@pytest.fixture(scope="module")
def agent():
    pytest.importorskip("langchain_core")
    from base_agent import BaseGUIAgent
    return BaseGUIAgent(llm_mode="api", model_name=os.environ.get("OLLAMA_MODEL", "qwen3:8b"),
                        base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"))


@pytest.mark.functional
def test_real_ollama_generate_returns_text(agent):
    assert agent.llm.inner.generate("用一句话介绍你自己").strip()


@pytest.mark.functional
def test_real_plan_weather_search(agent):
    subtasks = agent.plan("打开浏览器并搜索天气预报")
    assert 1 <= len(subtasks) <= 6


@pytest.mark.functional
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_real_act_semantic_accuracy(agent, case):
    agent.action_history.clear()
    action = agent.act(case["subtask"], screen_context(case["screen"]))
    ok, reasons = judge(case, action)
    assert ok, f"{action} -> {reasons}"
