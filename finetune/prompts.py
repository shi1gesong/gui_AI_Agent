"""
提示词模板（第5周：提示词工程）

同一张截图 + 同一条指令，用不同的提示词让模型输出动作，对比格式合格率与动作准确率：

  v0_minimal   极简：只给任务，要求输出 JSON 动作（基线，接近第3周的写法）
  v1_schema    结构化：说明坐标约定（0~1000 归一化，左上角为原点）、完整动作清单与字段、输出约束
  v2_fewshot   在 v1 基础上加 3 个文字示例（点击 / 输入+回车 / 组合键），示例不含图片，避免泄漏测试截图

LoRA 训练使用 v1_schema（训练与推理必须用同一提示词）；基座模型三种都测，用来回答
“只改提示词能提升多少、微调又能再提升多少”。
"""
from typing import Dict, Optional

DEFAULT_VARIANT = "v1_schema"

_V0 = """你在操作一台电脑。总体任务：{instruction}
{subtask_line}请根据屏幕截图给出下一步要执行的动作，用 JSON 输出。"""

_SCHEMA = """你是桌面 GUI 智能体，正在看一张电脑屏幕截图，需要输出下一步要执行的鼠标/键盘动作。

总体任务：{instruction}
{subtask_line}
坐标约定：x、y 是 0~1000 的整数，表示相对截图宽、高的千分比，左上角为 (0, 0)，右下角为 (1000, 1000)。

可用动作（每个动作是一个 JSON 对象）：
- {{"action": "click", "x": X, "y": Y}}            左键单击
- {{"action": "double_click", "x": X, "y": Y}}     左键双击
- {{"action": "right_click", "x": X, "y": Y}}      右键单击
- {{"action": "move", "x": X, "y": Y}}             移动鼠标
- {{"action": "drag", "x": X1, "y": Y1, "x2": X2, "y2": Y2}}  从 (X1,Y1) 拖到 (X2,Y2)
- {{"action": "scroll", "amount": N}}              滚动，正数向上、负数向下
- {{"action": "type", "text": "要输入的文字"}}      在当前焦点输入文字
- {{"action": "key", "keys": ["ctrl", "a"]}}        按键，多个键表示同时按下的组合键
- {{"action": "wait", "seconds": 1}}               等待

输出要求：只输出一个 JSON 数组，按执行顺序列出完成当前子任务所需的动作，不要输出任何解释。"""

_FEWSHOT = """

示例（与当前截图无关，只示范格式）：
子任务“点击屏幕中间的确定按钮” → [{"action": "click", "x": 500, "y": 520}]
子任务“在左上角搜索框输入天气并搜索” → [{"action": "click", "x": 120, "y": 45}, {"action": "type", "text": "天气"}, {"action": "key", "keys": ["enter"]}]
子任务“全选文本” → [{"action": "key", "keys": ["ctrl", "a"]}]"""

TEMPLATES: Dict[str, str] = {
    "v0_minimal": _V0,
    "v1_schema": _SCHEMA,
    "v2_fewshot": _SCHEMA + _FEWSHOT.replace("{", "{{").replace("}", "}}"),  # 示例里的花括号要转义
}


def build_prompt(instruction: str, sub_task: Optional[str] = None, variant: str = DEFAULT_VARIANT) -> str:
    if variant not in TEMPLATES:
        raise ValueError(f"未知提示词版本 {variant!r}，可选 {sorted(TEMPLATES)}")
    instruction = (instruction or "").strip()
    sub_task = (sub_task or "").strip()
    if not instruction and not sub_task:
        raise ValueError("instruction 和 sub_task 不能都为空")
    if not instruction:
        instruction = sub_task
    if variant == "v0_minimal":
        line = f"当前子任务：{sub_task}\n" if sub_task and sub_task != instruction else ""
    else:
        line = f"当前子任务：{sub_task}\n" if sub_task and sub_task != instruction else "当前子任务：与总体任务相同\n"
    return TEMPLATES[variant].format(instruction=instruction, subtask_line=line)


def build_messages(instruction: str, sub_task: Optional[str] = None, variant: str = DEFAULT_VARIANT,
                   answer: Optional[str] = None) -> list:
    """Qwen2-VL 聊天格式：图片在前、文字在后；answer 给出时追加助手回答（训练用）"""
    msgs = [{"role": "user", "content": [{"type": "image"},
                                         {"type": "text", "text": build_prompt(instruction, sub_task, variant)}]}]
    if answer is not None:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return msgs
