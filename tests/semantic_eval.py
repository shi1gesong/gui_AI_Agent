"""
Acting 阶段语义评测器（测试辅助代码）

第3周的 Agent 用例只检查了“输出是否是合法 JSON”，没有检查“动作在语义上是否正确”，
导致“打开默认浏览器 -> 点击『设置』”这样的错误动作被记为通过。
本评测器按黄金用例（tests/data/semantic_cases.json）逐项判断：
  1. action_type 是否为期望类型之一
  2. click/type 的 target_text 是否为期望目标，且确实出现在屏幕元素中（grounding）
  3. type 的 input_text 是否包含期望文本；key 的按键是否正确
返回 (是否通过, 失败原因列表)。
"""
import json
from pathlib import Path

CASES_PATH = Path(__file__).resolve().parent / "data" / "semantic_cases.json"
VALID_TYPES = {"click", "type", "scroll", "key", "double_click", "hotkey", "open_app", "close_app", "wait", "done"}


WEEK4_CASES_PATH = Path(__file__).resolve().parent / "data" / "week4_semantic_cases.json"


def load_cases(path=None):
    return json.loads(Path(path or CASES_PATH).read_text(encoding="utf-8"))


def load_week4_cases():
    return load_cases(WEEK4_CASES_PATH)


def screen_context(elements):
    """把元素文本列表转换成与第3周 demo 相同格式的 screen_context 字符串"""
    items = [{"text": t, "bbox": [100 + 120 * i, 100, 200 + 120 * i, 130]} for i, t in enumerate(elements)]
    return "识别到的UI元素: " + json.dumps(items, ensure_ascii=False)


def judge(case, action):
    reasons = []
    if not isinstance(action, dict) or action.get("_parse_error"):
        return False, ["输出无法解析为 JSON"]
    a_type = action.get("action_type")
    target = (action.get("target_text") or "").strip()
    text = action.get("input_text") or ""
    if a_type not in VALID_TYPES:
        reasons.append(f"action_type 非法: {a_type!r}")
    elif a_type not in case["expect_type"]:
        reasons.append(f"动作类型错误: 期望 {case['expect_type']}，实际 {a_type}")
    if a_type in ("click", "double_click"):
        expected = case.get("expect_target") or case.get("expect_target_if_click")
        if target not in case["screen"]:
            reasons.append(f"点击目标「{target}」不在屏幕元素中")
        elif expected and target not in expected:
            reasons.append(f"点击目标错误: 期望 {expected}，实际「{target}」")
    if a_type == "type":
        if case.get("expect_target") and target not in case["expect_target"]:
            reasons.append(f"输入目标错误: 期望 {case['expect_target']}，实际「{target}」")
        if case.get("expect_input") and case["expect_input"] not in text:
            reasons.append(f"输入内容错误: 期望包含「{case['expect_input']}」，实际「{text}」")
    if a_type == "open_app":
        expected = case.get("expect_target") or []
        if expected and not any(e.lower() in target.lower() or target.lower() in e.lower() for e in expected if target):
            reasons.append(f"打开目标错误: 期望 {expected}，实际「{target}」")
    if a_type == "close_app":
        expected = case.get("expect_target") or []
        if expected and not any(e.lower() in target.lower() for e in expected):
            reasons.append(f"关闭目标错误: 期望 {expected}，实际「{target}」")
    if a_type == "hotkey" and case.get("expect_input_any"):
        blob = (text + " " + target).lower().replace(" ", "")
        if not any(k.lower().replace(" ", "") in blob for k in case["expect_input_any"]):
            reasons.append(f"组合键错误: 期望 {case['expect_input_any']}，实际「{blob}」")
    if a_type == "key" and case.get("expect_input_any"):
        blob = text + " " + target
        if not any(k in blob for k in case["expect_input_any"]):
            reasons.append(f"按键错误: 期望 {case['expect_input_any']}，实际「{blob.strip()}」")
    return not reasons, reasons
