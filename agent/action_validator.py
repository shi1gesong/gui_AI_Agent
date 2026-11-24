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

2026-11-24 第4周扩展动作空间（端到端闭环 v1.0）：
  新增 open_app / close_app / hotkey / double_click / wait / done 六种动作，对应意图：
  「打开X」-> open_app（屏幕上有X的图标时也接受 click）；「关闭X」-> close_app 或 hotkey alt+f4；
  「按 ctrl+l」-> hotkey；「双击X」-> double_click；「等待2秒」-> wait；「任务完成」-> done。
  「打开X」且屏幕上没有X时，兜底动作从“Win 键 + 输入 + 回车”改为 open_app（由 control/app_manager.py
  直接启动程序，不依赖开始菜单搜索结果），Win 搜索仍作为 AppManager 不认识该程序时的退路。
  输入内容只从“输入”二字之后的引号中提取，避免「在“消息内容”框中输入“你好”」把“消息内容”当成输入内容。
"""
import ast
import json
import re
from typing import List, Optional

VALID_ACTION_TYPES = ("click", "type", "scroll", "key",
                      "double_click", "hotkey", "open_app", "close_app", "wait", "done")
# 不需要屏幕元素就能执行的动作（执行前可以跳过截图 + OCR）
SCREENLESS_ACTION_TYPES = ("key", "hotkey", "open_app", "close_app", "wait", "done", "scroll")
MAX_WAIT_SEC = 30.0

KEY_ALIASES = {
    "回车": "enter", "回车键": "enter", "enter": "enter", "return": "enter",
    "tab": "tab", "制表": "tab",
    "esc": "esc", "escape": "esc", "退出": "esc",
    "退格": "backspace", "backspace": "backspace",
    "删除": "delete", "delete": "delete",
    "空格": "space", "space": "space",
    "win": "win", "windows": "win", "开始": "win",
    "f5": "f5", "刷新": "f5",
    "f4": "f4", "f2": "f2", "f11": "f11",
    "ctrl": "ctrl", "control": "ctrl", "alt": "alt", "shift": "shift",
    "上": "up", "up": "up", "下": "down", "down": "down", "左": "left", "left": "left", "右": "right", "right": "right",
    "home": "home", "end": "end", "pageup": "pageup", "pagedown": "pagedown",
}
_MODIFIERS = ("ctrl", "alt", "shift", "win")

APP_ALIASES = {
    "浏览器": ["浏览器", "chrome", "edge", "firefox", "safari", "360", "browser"],
    "记事本": ["记事本", "notepad"],
    "微信": ["微信", "wechat"],
    "文件管理器": ["文件资源管理器", "此电脑", "explorer", "文件管理器"],
    "设置": ["设置", "settings"],
    "计算器": ["计算器", "calculator", "calc"],
    "画图": ["画图", "paint", "mspaint"],
    "模拟聊天": ["模拟聊天", "聊天应用", "聊天程序", "mock chat", "mock_chat"],
    "命令提示符": ["命令提示符", "cmd"],
}

_QUOTE_RE = re.compile(r"[“\"'‘「『]([^”\"'’」』]+)[”\"'’」』]")
_TYPE_RE = re.compile(r"(?:输入|键入|填写|填入)(?!框|栏)")
_TYPE_TEXT_RE = re.compile(r"(?:输入|键入|填写|填入)(?!框|栏)[：:\s]*(.+?)(?:并|然后|，|,|。|；|$)")
_KEY_RE = re.compile(r"按(?:下)?\s*(回车|enter|return|tab|esc|escape|退格|backspace|空格|space|win|windows|delete|删除|f5)",
                     re.IGNORECASE)
_OPEN_RE = re.compile(r"^(?:打开|启动|运行)(.+)$")
_CLOSE_RE = re.compile(r"^(?:关闭|关掉|退出)(.+)$")
_FIELD_RE = re.compile(r"在(.+?)(?:中|里|内)?(?:输入|键入|填写|填入)(?!框|栏)")
_HOTKEY_RE = re.compile(r"((?:ctrl|control|alt|shift|win)(?:\s*\+\s*[a-z0-9]+)+)", re.IGNORECASE)
_WAIT_RE = re.compile(r"^等待\s*(\d+(?:\.\d+)?)?\s*(秒|s|毫秒|ms)?")
_DOUBLE_RE = re.compile(r"双击")
_DONE_RE = re.compile(r"^(?:任务)?(?:已)?完成|^结束任务")
_FILE_TARGET_RE = re.compile(r"([A-Za-z]:[\\/][^“”\"'‘’「」]+|[^\s“”\"'‘’「」]+\.[A-Za-z0-9]{1,5})$")
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


def parse_hotkey(spec) -> Optional[List[str]]:
    """'ctrl+l' / 'Ctrl + Shift + Esc' / ['ctrl','l'] -> ['ctrl','l']；无法识别返回 None"""
    if isinstance(spec, (list, tuple)):
        parts = [str(p) for p in spec]
    else:
        parts = re.split(r"\s*\+\s*|\s+", str(spec or "").strip())
    keys = []
    for p in parts:
        k = p.strip().lower()
        if not k:
            continue
        k = KEY_ALIASES.get(k, k)
        if not (k in KEY_ALIASES.values() or re.fullmatch(r"[a-z0-9]|f\d{1,2}", k)):
            return None
        keys.append(k)
    if len(keys) < 2 or not any(k in _MODIFIERS for k in keys[:-1]):
        return None
    return keys


def parse_wait_seconds(value, default: float = 1.0) -> Optional[float]:
    """wait 动作的秒数：'2' / '2秒' / '500ms' / 2.5 -> float；负数或无法解析返回 None；上限 MAX_WAIT_SEC"""
    if value is None or value == "":
        return default
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        sec = float(value)
    else:
        m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(秒|s|sec|毫秒|ms)?\s*", str(value))
        if not m:
            return None
        sec = float(m.group(1)) / (1000.0 if m.group(2) in ("毫秒", "ms") else 1.0)
    if sec < 0:
        return None
    return min(sec, MAX_WAIT_SEC)


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
    """从子任务描述中用规则抽取意图：要输入什么、要按什么键、要打开/关闭什么、点名的目标"""

    def __init__(self, subtask: str):
        self.subtask = subtask or ""
        st = self.subtask.strip()
        self.quoted = _QUOTE_RE.findall(self.subtask)
        tm = _TYPE_RE.search(self.subtask)
        self.type_intent = bool(tm)
        self.expected_text = None
        if self.type_intent:
            after = [m.group(1) for m in _QUOTE_RE.finditer(self.subtask) if m.start() >= tm.start()]
            if after:
                self.expected_text = after[0]
            elif self.quoted and not _FIELD_RE.search(self.subtask):
                self.expected_text = self.quoted[0]
            else:
                m = _TYPE_TEXT_RE.search(self.subtask)
                if m and m.group(1).strip():
                    self.expected_text = m.group(1).strip().strip("“”\"'‘’「」『』")
        hm = _HOTKEY_RE.search(self.subtask) if not self.type_intent else None
        self.hotkey = parse_hotkey(hm.group(1)) if hm else None
        km = _KEY_RE.search(self.subtask) if not self.hotkey else None
        self.key = KEY_ALIASES.get(km.group(1).lower()) if km else None
        wm = _WAIT_RE.match(st)
        self.wait_seconds = None
        if wm:
            unit = 1000.0 if wm.group(2) in ("毫秒", "ms") else 1.0
            self.wait_seconds = min(float(wm.group(1)) / unit, MAX_WAIT_SEC) if wm.group(1) else 1.0
        self.done_intent = bool(_DONE_RE.match(st))
        self.double_click = bool(_DOUBLE_RE.search(self.subtask)) and not self.type_intent
        self.scroll_intent = any(w in self.subtask for w in ("滚动", "滑动", "翻页"))
        self.scroll_down = not any(w in self.subtask for w in ("向上", "往上", "上滚", "顶部"))
        fm = _FIELD_RE.search(self.subtask) if self.type_intent else None
        # “在微信搜索框输入张三” -> “微信搜索框”；“在“消息内容”框中输入…” -> “消息内容框”
        self.field = re.sub(r"[“”\"'‘’「」『』]", "", fm.group(1)).strip() if fm else None
        if self.field and self.field.endswith(("框", "栏")) and self.quoted and self.quoted[0] in self.field:
            self.field = self.quoted[0]
        self.direct_type = "直接输入" in self.subtask  # 输入到当前焦点，不点击任何输入框
        om = _OPEN_RE.match(st)
        self.open_target = None
        if om and not self.type_intent and not self.double_click:
            self.open_target = (self.quoted[0] if self.quoted
                                else om.group(1).replace("默认", "").strip(" 的“”\"'"))
        cm = _CLOSE_RE.match(st)
        self.close_target = None
        if cm and not self.type_intent:
            self.close_target = (self.quoted[0] if self.quoted
                                 else cm.group(1).replace("当前", "").strip(" 的“”\"'")) or "当前窗口"
            for suffix in ("窗口", "程序", "应用"):
                if self.close_target.endswith(suffix) and len(self.close_target) > len(suffix):
                    self.close_target = self.close_target[: -len(suffix)]
            if self.close_target in ("当前", "窗口", "程序", "应用"):
                self.close_target = "当前窗口"
        self.click_targets = self.quoted if (self.quoted and not self.type_intent) else []
        if not self.click_targets and self.double_click:
            m = re.search(r"双击(?:打开)?(.+)$", st)
            if m and m.group(1).strip():
                self.click_targets = [m.group(1).strip()]

    @property
    def open_is_file(self) -> bool:
        return bool(self.open_target and (_FILE_TARGET_RE.search(self.open_target) or "文件" in self.subtask))

    def open_aliases(self) -> List[str]:
        return _aliases(self.open_target)

    def close_aliases(self) -> List[str]:
        return _aliases(self.close_target)


def _aliases(name: Optional[str]) -> List[str]:
    if not name:
        return []
    for key, aliases in APP_ALIASES.items():
        if key in name or any(_norm(a) == _norm(name) for a in aliases):
            return aliases
    name = re.sub(r"^文件", "", name).strip() or name
    return [name]


def _matches_any(target: str, aliases: List[str]) -> bool:
    t = _norm(target)
    return bool(t) and any(_norm(a) in t or t in _norm(a) for a in aliases if _norm(a))


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

    # ---- 1. 动作类型与子任务意图是否一致 ----
    if intent.type_intent and a_type != "type":
        reasons.append(f"子任务要求输入文字，action_type 应为 type，实际是 {a_type}")
    elif intent.hotkey and a_type != "hotkey":
        reasons.append(f"子任务要求按组合键 {'+'.join(intent.hotkey)}，action_type 应为 hotkey，实际是 {a_type}")
    elif intent.key and not intent.type_intent and a_type != "key":
        reasons.append(f"子任务要求按 {intent.key} 键，action_type 应为 key，实际是 {a_type}")
    elif intent.wait_seconds is not None and a_type != "wait":
        reasons.append(f"子任务要求等待，action_type 应为 wait，实际是 {a_type}")
    elif intent.close_target and a_type not in ("close_app", "hotkey"):
        reasons.append(f"子任务要关闭「{intent.close_target}」，action_type 应为 close_app，实际是 {a_type}")
    elif intent.double_click and a_type != "double_click":
        reasons.append(f"子任务要求双击，action_type 应为 double_click，实际是 {a_type}")
    elif intent.open_target and a_type not in ("open_app", "click", "double_click", "key"):
        reasons.append(f"子任务要打开「{intent.open_target}」，action_type 应为 open_app，实际是 {a_type}")
    elif intent.scroll_intent and a_type != "scroll":
        # 只有“返回顶部/底部”这类按钮可以用点击代替滚动；“下一页”是翻页，不是滚动
        if not (a_type == "click" and any(w in target.lower() for w in _SCROLL_CLICK_WORDS)):
            reasons.append(f"子任务要求滚动，action_type 应为 scroll，实际是 {a_type}「{target}」")
    elif a_type == "done" and (intent.open_target or intent.click_targets or intent.close_target):
        reasons.append("子任务还有明确的操作目标，不能直接输出 done")

    # ---- 2. 各动作自身参数是否合法 ----
    if a_type in ("click", "double_click"):
        if not target:
            reasons.append(f"{a_type} 动作缺少 target_text")
        elif elements and find_element(target, elements) is None:
            reasons.append(f"点击目标「{target}」不在当前屏幕元素中")
        if target and intent.open_target and not _matches_any(target, intent.open_aliases()):
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
        elif intent.open_target and key != "win":
            reasons.append(f"子任务要打开「{intent.open_target}」，按 {key} 键无法打开")
    elif a_type == "hotkey":
        keys = parse_hotkey(text or target)
        if keys is None:
            reasons.append(f"无法识别的组合键「{text or target}」，应写成 ctrl+l 这样的格式")
        elif intent.hotkey and keys != intent.hotkey:
            reasons.append(f"应按 {'+'.join(intent.hotkey)}，实际按 {'+'.join(keys)}")
        elif intent.close_target and keys != ["alt", "f4"]:
            reasons.append(f"关闭窗口应使用 close_app 或 alt+f4，实际按 {'+'.join(keys)}")
    elif a_type == "open_app":
        if not target:
            reasons.append("open_app 动作缺少 target_text（要打开的程序名或文件路径）")
        elif intent.open_target and not _matches_any(target, intent.open_aliases()):
            reasons.append(f"子任务要打开「{intent.open_target}」，实际打开「{target}」")
    elif a_type == "close_app":
        if not target:
            reasons.append("close_app 动作缺少 target_text（要关闭的程序名/窗口标题）")
        elif intent.close_target and not _matches_any(target, intent.close_aliases()):
            reasons.append(f"子任务要关闭「{intent.close_target}」，实际关闭「{target}」")
    elif a_type == "wait":
        if parse_wait_seconds(text or action.get("seconds")) is None:
            reasons.append(f"wait 的等待时间「{text}」无法解析，应为 0~{int(MAX_WAIT_SEC)} 秒的数字")
    return reasons


def _fb(a_type: str, target: str = "", text: str = "", **extra) -> dict:
    return {"action_type": a_type, "target_text": target, "input_text": text, "_fallback": True, **extra}


def fallback_action(subtask: str, elements: List[dict]) -> Optional[dict]:
    """LLM 多次输出都不合格时（或混合模式下意图明确时），按子任务意图用规则生成动作；推断不出返回 None"""
    intent = SubtaskIntent(subtask)
    if intent.done_intent and not (intent.open_target or intent.click_targets):
        return _fb("done")
    if intent.wait_seconds is not None:
        return _fb("wait", "", f"{intent.wait_seconds:g}")
    if intent.close_target:
        return _fb("close_app", intent.close_target)
    if intent.hotkey and not intent.type_intent:
        return _fb("hotkey", "", "+".join(intent.hotkey))
    if intent.double_click and intent.click_targets:
        hit = find_element(intent.click_targets[0], elements)
        return _fb("double_click", hit["text"]) if hit else None
    if intent.open_target:
        if not intent.open_is_file:
            for alias in intent.open_aliases():
                hit = find_element(alias, elements)
                if hit:
                    return _fb("click", hit["text"])
        # 屏幕上没有目标应用 / 要打开的是文件：交给 AppManager 直接启动
        return _fb("open_app", intent.open_target)
    if intent.type_intent and intent.expected_text:
        field = ""
        if intent.field and not intent.direct_type:
            hit = find_element(intent.field, elements)
            # 没找到时保留输入框名：needs_screen() 据此先截图识别；执行器找不到坐标时直接输入到当前焦点
            field = hit["text"] if hit else intent.field
        elif not intent.direct_type:
            field = next((e["text"] for e in elements
                          if any(k in str(e.get("text", "")) for k in ("地址栏", "输入框", "搜索框", "框", "栏"))), "")
        act = _fb("type", field, intent.expected_text)
        if intent.key:
            act["follow_up"] = [{"action_type": "key", "target_text": "", "input_text": intent.key}]
        return act
    if intent.key:
        return _fb("key", "", intent.key)
    if intent.scroll_intent:
        return _fb("scroll", "", "", amount=-5 if intent.scroll_down else 5)
    if intent.click_targets:
        hit = find_element(intent.click_targets[0], elements)
        if hit:
            return _fb("click", hit["text"])
    return None


def needs_screen(action: Optional[dict]) -> bool:
    """该动作执行前是否需要截图 + OCR（click/double_click 一定需要；type 指定了输入框时需要）"""
    if not action:
        return True
    a_type = action.get("action_type")
    if a_type in ("click", "double_click"):
        return True
    if a_type == "type":
        return bool(action.get("target_text"))
    return a_type not in SCREENLESS_ACTION_TYPES
