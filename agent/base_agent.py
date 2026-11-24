"""
基础GUI Agent框架
设计参考第1周调研结论：借鉴ScreenAgent的"Planning-Acting-Reflecting"三阶段流程，
用LangChain封装Prompt模板与调用链，实现任务拆解与规划能力。

流程：
  用户指令 → [Planning] 拆解为子任务列表 → [Acting] 逐个执行子任务并生成具体动作
           → [Reflecting] 检查执行是否达成目标，判断是否需要调整计划

2026-11-23 修复（依据 tests/ 测试结果，详见 report/week2-3_测试报告.md）：
- JSON 解析：去掉 qwen3 的 <think> 段；输出多个 JSON 时取第一个；兼容 LLM 直接返回数组
- Planning：subtasks 类型校验（字符串/数组/缺失都能正确处理）
- Acting：动作规则校验（类型、输入内容、屏幕 grounding、目标相关性）+ 带原因重试 + 规则兜底
- Acting Prompt 的示例不再用“设置”（第3周模型把“打开浏览器”照抄成点击“设置”）
- run()：每次开始时清空历史；可传入 executor 真正执行动作并记录执行结果
- Reflecting：动作未执行、未通过校验或输出无法解析时，强制判定未完成（不再只听 LLM 的）
- 支持以包方式导入（agent.base_agent）
"""
import json
import re
from typing import Any, Callable, List, Optional

from langchain_core.language_models.llms import LLM
from langchain_core.prompts import PromptTemplate

try:  # 以包方式导入（agent.base_agent）
    from .action_validator import fallback_action, parse_screen_elements, validate_action
    from .llm_interface import BaseLLMInterface, build_llm_interface
except ImportError:  # 直接运行本文件 / 把 agent/ 加入 sys.path 时
    from action_validator import fallback_action, parse_screen_elements, validate_action
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

请给出下一步应该执行的具体动作。action_type字段必须是下面这些词中的恰好一个：
- click：单击某个元素。target_text 必须从上面的屏幕元素中原样选取，且必须与子任务目标相关
- double_click：双击某个元素（如桌面图标、文件），target_text 同 click
- type：输入文本。凡是子任务要求“输入/填写”内容的都用 type，input_text 填要输入的完整内容，
  target_text 填要先点击的输入框文字（没有就留空字符串，表示输入到当前焦点）
- key：按下单个功能键，input_text 填键名（enter/tab/esc/backspace/win 等）
- hotkey：组合键，input_text 写成 ctrl+l、alt+f4 这样的格式
- scroll：滚动屏幕
- open_app：打开程序或文件，target_text 填程序名（浏览器/记事本/计算器…）或文件路径
- close_app：关闭程序窗口，target_text 填程序名或窗口标题
- wait：等待界面加载，input_text 填秒数
- done：子任务已经完成，无需操作

规则：
- 子任务要打开某个应用而屏幕元素里没有它时，用 open_app，不要点击无关元素
- 不要点击与子任务无关的元素

示例输出（点击屏幕上的“确定”按钮）：
{{"action_type": "click", "target_text": "确定", "input_text": ""}}

示例输出（在搜索框输入文字）：
{{"action_type": "type", "target_text": "搜索框", "input_text": "天气预报"}}
{feedback}
现在请只输出一个JSON对象，不要输出多余文字：
{{"action_type": "上面动作类型中的一个", "target_text": "要操作的目标文字", "input_text": "type 时填写输入内容，key/hotkey 时填写键名，wait 时填秒数，否则留空字符串"}}
"""
)

REFLECTING_PROMPT = PromptTemplate.from_template(
    """任务目标：{instruction}
已执行的子任务与动作记录（含是否真正执行成功）：
{action_history}

请判断当前任务是否已经完成。严格按以下JSON格式输出：
{{"is_complete": true/false, "reason": "简要说明判断依据", "next_action_suggestion": "若未完成，建议下一步做什么"}}
"""
)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class BaseGUIAgent:
    def __init__(self, llm_mode: str = "api", max_retries: int = 2,
                 executor: Optional[Callable] = None, **llm_kwargs):
        """
        max_retries: Acting 输出不合格时带着失败原因重新询问 LLM 的次数
        executor: 可选，callable(action, elements) -> {"ok": bool, "error": str}，如 executor.ControllerExecutor
        """
        inner_llm = build_llm_interface(mode=llm_mode, **llm_kwargs)
        self.llm = LangChainLLMWrapper(inner_llm=inner_llm)
        self.max_retries = max_retries
        self.executor = executor
        self.action_history: List[dict] = []

    def _safe_json_parse(self, raw_text: str):
        """大模型输出常夹带解释文字/思考过程，尽量提取第一个合法的 JSON 对象或数组"""
        text = _THINK_RE.sub("", raw_text or "")
        if "</think>" in text:  # 只有结束标签（开头被截断）的情况
            text = text.split("</think>")[-1]
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        decoder = json.JSONDecoder()
        for i, ch in enumerate(text):
            if ch in "{[":
                try:
                    obj, _ = decoder.raw_decode(text, i)
                    if isinstance(obj, (dict, list)):
                        return obj
                except json.JSONDecodeError:
                    continue
        return {"_parse_error": True, "raw": raw_text}

    def plan(self, instruction: str) -> List[str]:
        """Planning阶段：把用户指令拆解成子任务列表"""
        prompt = PLANNING_PROMPT.format(instruction=instruction)
        raw = self.llm.invoke(prompt)
        parsed = self._safe_json_parse(raw)
        if isinstance(parsed, list):
            subtasks = parsed
        elif isinstance(parsed, dict):
            subtasks = parsed.get("subtasks", [])
        else:
            subtasks = []
        if isinstance(subtasks, str):
            subtasks = [subtasks]
        if not isinstance(subtasks, list):
            subtasks = []
        subtasks = [str(s).strip() for s in subtasks if isinstance(s, (str, int, float)) and str(s).strip()]
        print(f"[Planning] 拆解出 {len(subtasks)} 个子任务: {subtasks}")
        return subtasks

    def act(self, subtask: str, screen_context: str) -> dict:
        """Acting阶段：针对单个子任务，结合当前屏幕感知结果生成动作；不合格时带原因重试，仍不合格则规则兜底"""
        elements = parse_screen_elements(screen_context)
        feedback, rejected = "", []
        action, reasons = None, ["未调用"]
        for attempt in range(self.max_retries + 1):
            prompt = ACTING_PROMPT.format(subtask=subtask, screen_context=screen_context, feedback=feedback)
            raw = self.llm.invoke(prompt)
            action = self._safe_json_parse(raw)
            if isinstance(action, list):  # 返回了动作数组时取第一个
                action = action[0] if action and isinstance(action[0], dict) else {"_parse_error": True, "raw": raw}
            reasons = validate_action(subtask, action, elements)
            if not reasons:
                break
            rejected.append({"output": action, "reasons": reasons})
            feedback = ("\n注意：你上一次的输出 " + json.dumps(action, ensure_ascii=False)
                        + " 不合格，原因：" + "；".join(reasons) + "。请修正后重新输出。\n")

        if reasons:
            fb = fallback_action(subtask, elements)
            if fb is not None:
                action = fb
            else:
                base = action if isinstance(action, dict) else {}
                action = {**base, "_invalid": True, "_reasons": reasons}
        if rejected:
            action = {**action, "_rejected": rejected}
        print(f"[Acting] 子任务「{subtask}」-> 动作: {action}")
        self.action_history.append({"subtask": subtask, "action": action,
                                    "attempts": len(rejected) + (0 if reasons else 1),
                                    "executed": False})
        return action

    def _execute_last(self, elements: List[dict]):
        entry = self.action_history[-1]
        if self.executor is None or entry["action"].get("_invalid"):
            return
        result = self.executor(entry["action"], elements) or {}
        entry["executed"] = bool(result.get("ok"))
        if result.get("error"):
            entry["error"] = result["error"]

    @staticmethod
    def _describe(entry: dict) -> str:
        a = {k: v for k, v in entry["action"].items() if k != "_rejected"}
        status = "已执行" if entry.get("executed") else ("执行失败: " + entry["error"] if entry.get("error") else "未执行")
        return f"- {entry['subtask']}: {a}（{status}）"

    def reflect(self, instruction: str) -> dict:
        """Reflecting阶段：检查任务是否已完成。LLM 判断之外再做规则把关，避免“什么都没做却判定完成”"""
        history_text = "\n".join(self._describe(h) for h in self.action_history)
        prompt = REFLECTING_PROMPT.format(instruction=instruction, action_history=history_text)
        raw = self.llm.invoke(prompt)
        result = self._safe_json_parse(raw)
        if not isinstance(result, dict):
            result = {"_parse_error": True, "raw": raw}

        problems = []
        if result.get("_parse_error"):
            problems.append("Reflect 输出无法解析")
        if not self.action_history:
            problems.append("没有执行任何动作")
        for h in self.action_history:
            if h["action"].get("_invalid"):
                problems.append(f"子任务「{h['subtask']}」的动作未通过校验")
            elif not h.get("executed"):
                problems.append(f"子任务「{h['subtask']}」的动作未实际执行")
        if problems:
            result = {**result, "llm_is_complete": result.get("is_complete"), "is_complete": False,
                      "guard_reasons": problems}
        print(f"[Reflecting] 完成判断: {result}")
        return result

    def run(self, instruction: str, screen_context_fn, max_steps: int = 5):
        """
        完整闭环：Plan一次 -> 逐个子任务Act（每步前调用screen_context_fn获取当前屏幕感知结果，有 executor 时执行动作）
        -> 全部子任务跑完后Reflect一次
        screen_context_fn: 一个无参函数，返回当前屏幕的文字/UI元素描述字符串（对接第2周感知模块）
        """
        self.action_history = []
        subtasks = self.plan(instruction)
        for subtask in subtasks[:max_steps]:
            screen_context = screen_context_fn()
            self.act(subtask, screen_context)
            self._execute_last(parse_screen_elements(screen_context))

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
