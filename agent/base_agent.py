"""
基础GUI Agent框架
设计参考第1周调研结论：借鉴ScreenAgent的"Planning-Acting-Reflecting"三阶段流程，
用LangChain封装Prompt模板与调用链，实现任务拆解与规划能力。

流程：
  用户指令 → [Planning] 拆解为子任务列表 → [Acting] 逐个执行子任务并生成具体动作
           → [Reflecting] 检查执行是否达成目标，判断是否需要调整计划
"""
import json
from typing import Any, List, Optional

from langchain_core.language_models.llms import LLM
from langchain_core.prompts import PromptTemplate

from llm_interface import BaseLLMInterface, build_llm_interface


class LangChainLLMWrapper(LLM):
    """
    把本项目自定义的 BaseLLMInterface（本地模型/Ollama API）包装成LangChain标准LLM接口，
    这样后续可以直接用LangChain生态里的Prompt模板、Chain、Agent工具等组件。
    """
    inner: Any = None

    def __init__(self, inner_llm: BaseLLMInterface, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, "inner", inner_llm)

    @property
    def _llm_type(self) -> str:
        return "custom_gui_agent_llm"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        return self.inner.generate(prompt)


PLANNING_PROMPT = PromptTemplate.from_template(
    """你是一个桌面GUI操作助手。用户给出了一个任务指令，请把它拆解成若干个可以在电脑上
逐步执行的具体子任务，每个子任务应该是一个清晰的、面向界面操作的动作描述。

用户指令：{instruction}

请严格按以下JSON格式输出子任务列表，不要输出多余内容：
{{"subtasks": ["子任务1描述", "子任务2描述", ...]}}
"""
)

ACTING_PROMPT = PromptTemplate.from_template(
    """你正在执行以下子任务：{subtask}

当前屏幕上识别到的UI元素信息如下（文字内容与坐标）：
{screen_context}

请给出下一步应该执行的具体动作。action_type字段必须是下面四个词中的恰好一个（不要写成"click/type/scroll/key"这种并列形式，只能二选一地选出其中一个词）：
- click：点击某个元素
- type：输入文本
- scroll：滚动屏幕
- key：按下功能键（如回车、Tab）

示例输出（点击"设置"按钮）：
{{"action_type": "click", "target_text": "设置", "input_text": ""}}

示例输出（在搜索框输入文字）：
{{"action_type": "type", "target_text": "搜索框", "input_text": "天气预报"}}

现在请按同样的JSON格式，只输出一个JSON对象，不要输出多余文字：
{{"action_type": "从click/type/scroll/key中选一个具体值", "target_text": "要操作的目标文字", "input_text": "如果是type操作，填写要输入的内容，否则留空字符串"}}
"""
)

REFLECTING_PROMPT = PromptTemplate.from_template(
    """任务目标：{instruction}
已执行的子任务与动作记录：
{action_history}

请判断当前任务是否已经完成。严格按以下JSON格式输出：
{{"is_complete": true/false, "reason": "简要说明判断依据", "next_action_suggestion": "若未完成，建议下一步做什么"}}
"""
)


class BaseGUIAgent:
    def __init__(self, llm_mode: str = "api", **llm_kwargs):
        inner_llm = build_llm_interface(mode=llm_mode, **llm_kwargs)
        self.llm = LangChainLLMWrapper(inner_llm=inner_llm)
        self.action_history: List[dict] = []

    def _safe_json_parse(self, raw_text: str) -> dict:
        """大模型输出有时会夹带解释性文字，尝试提取JSON部分"""
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            start = raw_text.find("{")
            end = raw_text.rfind("}")
            if start != -1 and end != -1:
                try:
                    return json.loads(raw_text[start:end + 1])
                except json.JSONDecodeError:
                    pass
            return {"_parse_error": True, "raw": raw_text}

    def plan(self, instruction: str) -> List[str]:
        """Planning阶段：把用户指令拆解成子任务列表"""
        prompt = PLANNING_PROMPT.format(instruction=instruction)
        raw = self.llm.invoke(prompt)
        parsed = self._safe_json_parse(raw)
        subtasks = parsed.get("subtasks", [])
        print(f"[Planning] 拆解出 {len(subtasks)} 个子任务: {subtasks}")
        return subtasks

    def act(self, subtask: str, screen_context: str) -> dict:
        """Acting阶段：针对单个子任务，结合当前屏幕感知结果，生成具体动作"""
        prompt = ACTING_PROMPT.format(subtask=subtask, screen_context=screen_context)
        raw = self.llm.invoke(prompt)
        action = self._safe_json_parse(raw)
        print(f"[Acting] 子任务「{subtask}」-> 动作: {action}")
        self.action_history.append({"subtask": subtask, "action": action})
        return action

    def reflect(self, instruction: str) -> dict:
        """Reflecting阶段：检查任务是否已完成，是否需要调整计划"""
        history_text = "\n".join(
            f"- {h['subtask']}: {h['action']}" for h in self.action_history
        )
        prompt = REFLECTING_PROMPT.format(instruction=instruction, action_history=history_text)
        raw = self.llm.invoke(prompt)
        result = self._safe_json_parse(raw)
        print(f"[Reflecting] 完成判断: {result}")
        return result

    def run(self, instruction: str, screen_context_fn, max_steps: int = 5):
        """
        完整闭环：Plan一次 -> 逐个子任务Act（每步前调用screen_context_fn获取当前屏幕感知结果）
        -> 全部子任务跑完后Reflect一次
        screen_context_fn: 一个无参函数，返回当前屏幕的文字/UI元素描述字符串（对接第2周感知模块）
        """
        subtasks = self.plan(instruction)
        for i, subtask in enumerate(subtasks[:max_steps]):
            screen_context = screen_context_fn()
            self.act(subtask, screen_context)

        reflection = self.reflect(instruction)
        return {
            "subtasks": subtasks,
            "action_history": self.action_history,
            "reflection": reflection,
        }


if __name__ == "__main__":
    # 单元测试：用一个假的screen_context_fn（不依赖真实截图），仅验证Plan/Act/Reflect三阶段的
    # Prompt构造与JSON解析逻辑是否走得通。真实LLM调用需要本机Ollama服务或本地模型，
    # 若服务未启动，此测试会在llm.invoke这一步报错，属预期行为（需先启动LLM服务再测）。

    def fake_screen_context():
        return "识别到的UI元素: [{'text': '搜索', 'bbox': [100,100,200,130]}, {'text': '设置', 'bbox': [300,50,380,80]}]"

    agent = BaseGUIAgent(llm_mode="api", model_name="qwen3:8b")
    result = agent.run("打开浏览器并搜索天气预报", screen_context_fn=fake_screen_context, max_steps=2)
    print("\n===== 运行结果 =====")
    print(json.dumps(result, ensure_ascii=False, indent=2))
