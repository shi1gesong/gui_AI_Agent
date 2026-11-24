"""
Acting 阶段的动作校验与兜底（2026-11-23 新增）

第3周的 Acting 只检查输出是不是 JSON，导致：
  「打开默认浏览器」 -> click「设置」         （目标与子任务无关）
  「在地址栏输入'天气预报'并回车」 -> click「搜索」 （输入类任务被当成点击，没有输入内容）
被直接接受。本模块在 LLM 输出之后做规则校验：
  1. action_type 必须是 click / type / scroll / key 之一
  2. 子任务里有“输入X”时必须是 type 且 input_text 包含 X；有“按回车/Tab…”时必须是 key
  3. click 的目标必须真实出现在屏幕元素中（grounding），子任务点名了目标（引号、“打开X”）时目标必须对得上
校验失败时由 Agent 带着失败原因重试；多次重试仍失败时，能用规则推断出正确动作的给出兜底动作，
推断不出的标记为 _invalid，交由 Reflect 判定任务未完成。
"""
import ast
import json
import re
from typing import List, Optional

VALID_ACTION_TYPES = ("click", "type", "scroll", "key")

KEY_ALIASES = {
    "回车": "enter", "回车键": "enter", "enter": "enter", "return": "enter",
    "tab": "tab", "制表": "tab",
    "esc": "esc", "escape": "esc", "退出": "esc",
    "退格": "backspace", "backspace": "backspace",
    "删除": "delete", "delete": "delete",
    "空格": "space", "space": "space",
    "win": "win", "windows": "win", "开始": "win",
    "f5": "f5", "刷新": "f5",
}

APP_ALIASES = {
    "浏览器": ["浏览器", "chrome", "edge", "firefox", "safari", "360", "browser"],
    "记事本": ["记事本", "notepad"],
    "微信": ["微信", "wechat"],
    "文件管理器": ["文件资源管理器", "此电脑", "explorer", "文件管理器"],
    "设置": ["设置", "settings"],
}

_QUOTE_RE = re.compile(r"[“\"'‘「『]([^”\"'’」』]+)[”\"'’」』]")
_TYPE_RE = re.compile(r"(?:输入|键入|填写|填入)(?!框|栏)")
_TYPE_TEXT_RE = re.compile(r"(?:输入|键入|填写|填入)(?!框|栏)[：:\s]*(.+?)(?:并|然后|，|,|。|；|$)")
_KEY_RE = re.compile(r"按(?:下)?\s*(回车|enter|return|tab|esc|escape|退格|backspace|空格|space|win|windows|delete|删除|f5)",
                     re.IGNORECASE)
_OPEN_RE = re.compile(r"^(?:打开|启动|运行)(.+)$")
_FIELD_RE = re.compile(r"在(.+?)(?:中|里|内)?(?:输入|键入|填写|填入)(?!框|栏)")
_SCROLL_CLICK_WORDS = ("顶部", "底部", "返回", "top", "bottom")


def parse_screen_elements(screen_context: str) -> List[dict]:
    """从 screen_context 字符串中提取 [{'text':..., 'bbox':[...]}...]，兼容 JSON 与 Python repr 两种写法"""
    if not screen_context:
        return []
    start, end = screen_context.find("["), screen_context.rfind("]")
    if start != -1 and end > start:
        chunk = screen_context[start:end + 1]
        for loader in (json.loads, ast.literal_eval):
            try:
                data = loader(chunk)
                if isinstance(data, list):
                    return [d for d in data if isinstance(d, dict) and "text" in d]
            except Exception:
                pass
    return [{"text": t} for t in re.findall(r"['\"]text['\"]\s*:\s*['\"]([^'\"]*)['\"]", screen_context)]


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", str(s or "")).lower()


def find_element(target: str, elements: List[dict]) -> Optional[dict]:
    """在屏幕元素中查找目标：先精确匹配，再互相包含匹配"""
    t = _norm(target)
    if not t:
        return None
    for e in elements:
        if _norm(e.get("text")) == t:
            return e
    for e in elements:
        et = _norm(e.get("text"))
        if et and (t in et or et in t):
            return e
    return None


class SubtaskIntent:
    """从子任务描述中用规则抽取意图：要输入什么、要按什么键、要打开什么、点名的目标"""

    def __init__(self, subtask: str):
        self.subtask = subtask or ""
        self.quoted = _QUOTE_RE.findall(self.subtask)
        self.type_intent = bool(_TYPE_RE.search(self.subtask))
        self.expected_text = None
        if self.type_intent:
            if self.quoted:
                self.expected_text = self.quoted[0]
            else:
                m = _TYPE_TEXT_RE.search(self.subtask)
                if m and m.group(1).strip():
                    self.expected_text = m.group(1).strip()
        km = _KEY_RE.search(self.subtask)
        self.key = KEY_ALIASES.get(km.group(1).lower()) if km else None
        self.scroll_intent = any(w in self.subtask for w in ("滚动", "滑动", "翻页"))
        self.scroll_down = not any(w in self.subtask for w in ("向上", "往上", "上滚", "顶部"))
        fm = _FIELD_RE.search(self.subtask) if self.type_intent else None
        self.field = fm.group(1).strip() if fm else None  # “在微信搜索框输入张三” -> “微信搜索框”
        om = _OPEN_RE.match(self.subtask.strip())
        self.open_target = om.group(1).replace("默认", "").strip(" 的“”\"'") if om and not self.type_intent else None
        self.click_targets = self.quoted if (self.quoted and not self.type_intent) else []

    def open_aliases(self) -> List[str]:
        if not self.open_target:
            return []
        for key, aliases in APP_ALIASES.items():
            if key in self.open_target:
                return aliases
        return [self.open_target]


def validate_action(subtask: str, action, elements: List[dict]) -> List[str]:
    """返回不合格原因列表，空列表表示通过"""
    if not isinstance(action, dict) or action.get("_parse_error"):
        return ["输出无法解析为 JSON 对象"]
    reasons = []
    a_type = action.get("action_type")
    target = str(action.get("target_text") or "").strip()
    text = str(action.get("input_text") or "")
    intent = SubtaskIntent(subtask)

    if a_type not in VALID_ACTION_TYPES:
        return [f"action_type 必须是 {'/'.join(VALID_ACTION_TYPES)} 之一，实际是 {a_type!r}"]

    if intent.type_intent and a_type != "type":
        reasons.append(f"子任务要求输入文字，action_type 应为 type，实际是 {a_type}")
    elif intent.key and not intent.type_intent and a_type != "key":
        reasons.append(f"子任务要求按 {intent.key} 键，action_type 应为 key，实际是 {a_type}")
    elif intent.scroll_intent and a_type != "scroll":
        # 只有“返回顶部/底部”这类按钮可以用点击代替滚动；“下一页”是翻页，不是滚动
        if not (a_type == "click" and any(w in target.lower() for w in _SCROLL_CLICK_WORDS)):
            reasons.append(f"子任务要求滚动，action_type 应为 scroll，实际是 {a_type}「{target}」")

    if a_type == "click":
        if not target:
            reasons.append("click 动作缺少 target_text")
        elif elements and find_element(target, elements) is None:
            reasons.append(f"点击目标「{target}」不在当前屏幕元素中")
        if target and intent.open_target and not any(a in _norm(target) for a in map(_norm, intent.open_aliases())):
            reasons.append(f"子任务要打开「{intent.open_target}」，点击「{target}」与之无关")
        if target and intent.click_targets and not any(_norm(q) in _norm(target) or _norm(target) in _norm(q)
                                                       for q in intent.click_targets):
            reasons.append(f"子任务点名的目标是 {intent.click_targets}，实际点击「{target}」")
    elif a_type == "type":
        if not text:
            reasons.append("type 动作的 input_text 为空")
        elif intent.expected_text and _norm(intent.expected_text) not in _norm(text):
            reasons.append(f"应输入「{intent.expected_text}」，实际输入「{text}」")
    elif a_type == "key":
        key = KEY_ALIASES.get(_norm(text)) or KEY_ALIASES.get(_norm(target))
        if key is None:
            reasons.append(f"无法识别的按键「{text or target}」")
        elif intent.key and key != intent.key:
            reasons.append(f"应按 {intent.key}，实际按 {key}")
    return reasons


def fallback_action(subtask: str, elements: List[dict]) -> Optional[dict]:
    """LLM 多次输出都不合格时，按子任务意图用规则生成动作；推断不出返回 None"""
    intent = SubtaskIntent(subtask)
    if intent.open_target:
        for alias in intent.open_aliases():
            hit = find_element(alias, elements)
            if hit:
                return {"action_type": "click", "target_text": hit["text"], "input_text": "", "_fallback": True}
        # 屏幕上没有目标应用：按 Win 键打开开始菜单搜索
        return {"action_type": "key", "target_text": "", "input_text": "win", "_fallback": True,
                "follow_up": [
                    {"action_type": "type", "target_text": "", "input_text": intent.open_target},
                    {"action_type": "key", "target_text": "", "input_text": "enter"},
                ]}
    if intent.type_intent and intent.expected_text:
        hit = find_element(intent.field, elements) if intent.field else None
        field = hit["text"] if hit else next(
            (e["text"] for e in elements
             if any(k in str(e.get("text", "")) for k in ("地址栏", "输入框", "搜索框", "框", "栏"))), "")
        act = {"action_type": "type", "target_text": field, "input_text": intent.expected_text, "_fallback": True}
        if intent.key:
            act["follow_up"] = [{"action_type": "key", "target_text": "", "input_text": intent.key}]
        return act
    if intent.key:
        return {"action_type": "key", "target_text": "", "input_text": intent.key, "_fallback": True}
    if intent.scroll_intent:
        return {"action_type": "scroll", "target_text": "", "input_text": "",
                "amount": -5 if intent.scroll_down else 5, "_fallback": True}
    if intent.click_targets:
        hit = find_element(intent.click_targets[0], elements)
        if hit:
            return {"action_type": "click", "target_text": hit["text"], "input_text": "", "_fallback": True}
    return None
