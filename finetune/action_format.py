"""
VLM 动作格式（第5周）

训练目标与模型输出都是一个 JSON 数组，每个元素是一个动作，坐标用 0~1000 的归一化整数
（与分辨率无关：1024×768 的 ScreenAgent 截图和 2560×1440 的真机截图用同一套坐标）：

    [{"action": "click", "x": 530, "y": 173}]
    [{"action": "click", "x": 500, "y": 40}, {"action": "type", "text": "天气"}, {"action": "key", "keys": ["enter"]}]

动作集合（ScreenAgent 标注能表达的全部可执行动作）：
    click / double_click / right_click : x, y
    move                                : x, y
    drag                                : x, y（起点）, x2, y2（终点）
    scroll                              : amount（正数向上，负数向下）, 可选 x, y
    type                                : text
    key                                 : keys（列表，多个键表示组合键，如 ["ctrl", "a"]）
    wait                                : seconds

本模块只依赖标准库，Windows 端没有 torch 也能 import。
"""
import json
import math
import re
from typing import List, Optional, Tuple

COORD_MAX = 1000
POINT_ACTIONS = ("click", "double_click", "right_click", "move", "drag")
ACTIONS = POINT_ACTIONS + ("scroll", "type", "key", "wait")

KEY_ALIASES = {
    "control": "ctrl", "ctl": "ctrl", "ctrl_l": "ctrl", "ctrl_r": "ctrl", "lctrl": "ctrl", "rctrl": "ctrl",
    "return": "enter", "回车": "enter", "kp_enter": "enter",
    "escape": "esc", "del": "delete", "bksp": "backspace", "back": "backspace",
    "super": "win", "super_l": "win", "meta": "win", "windows": "win", "cmd": "win", "command": "win", "lwin": "win",
    "alt_l": "alt", "alt_r": "alt", "option": "alt", "shift_l": "shift", "shift_r": "shift",
    "pgup": "pageup", "page_up": "pageup", "prior": "pageup", "pgdn": "pagedown", "page_down": "pagedown",
    "next": "pagedown", "arrowup": "up", "arrowdown": "down", "arrowleft": "left", "arrowright": "right",
    "spacebar": "space", " ": "space", "caps_lock": "capslock",
}

_ACTION_ALIASES = {
    "left_click": "click", "leftclick": "click", "tap": "click", "press": "click",
    "doubleclick": "double_click", "double-click": "double_click", "dblclick": "double_click",
    "rightclick": "right_click", "right-click": "right_click", "context_click": "right_click",
    "mouse_move": "move", "hover": "move", "moveto": "move", "move_to": "move",
    "drag_to": "drag", "dragto": "drag", "drag_and_drop": "drag",
    "input": "type", "type_text": "type", "write": "type", "typing": "type",
    "keypress": "key", "key_press": "key", "press_key": "key", "hotkey": "key", "keyboard": "key",
    "sleep": "wait", "pause": "wait",
    "scroll_up": "scroll", "scroll_down": "scroll",
}


# --------------------------------------------------------------------------- #
# 数据集动作（dataset_loader 统一格式） → VLM 动作
# --------------------------------------------------------------------------- #
def _to_norm(v: float, size: float) -> int:
    return clamp_coord(round(float(v) / float(size) * COORD_MAX))


def clamp_coord(v) -> int:
    return int(max(0, min(COORD_MAX, int(round(float(v))))))


def normalize_key(k) -> str:
    k = str(k).strip().lower()
    if not k:
        return k
    return KEY_ALIASES.get(k, k)


def from_dataset_action(a: dict, screen_size: Tuple[int, int], prev_point: Optional[Tuple[int, int]] = None):
    """把 data/dataset_loader.py 的统一动作转换为 VLM 动作；不支持的动作返回 None。
    drag 在 ScreenAgent 里是“move 到起点 → drag 到终点”两步，起点取上一个动作的坐标（prev_point，已归一化）。"""
    t = a.get("action_type")
    w, h = screen_size

    def point():
        if a.get("x_ratio") is not None and a.get("y_ratio") is not None:
            return clamp_coord(a["x_ratio"] * COORD_MAX), clamp_coord(a["y_ratio"] * COORD_MAX)
        if a.get("x") is not None and a.get("y") is not None and w and h:
            return _to_norm(a["x"], w), _to_norm(a["y"], h)
        return None

    if t in ("click", "double_click"):
        p = point()
        if p is None:
            return None
        kind = "right_click" if t == "click" and a.get("button") == "right" else t
        return {"action": kind, "x": p[0], "y": p[1]}
    if t == "move":
        p = point()
        return None if p is None else {"action": "move", "x": p[0], "y": p[1]}
    if t == "drag":
        p = point()
        if p is None:
            return None
        if prev_point is None:
            return {"action": "drag", "x": p[0], "y": p[1], "x2": p[0], "y2": p[1]}
        return {"action": "drag", "x": prev_point[0], "y": prev_point[1], "x2": p[0], "y2": p[1]}
    if t == "scroll":
        out = {"action": "scroll", "amount": int(a.get("amount", -1) or -1)}
        p = point()
        if p is not None:
            out.update(x=p[0], y=p[1])
        return out
    if t == "type":
        text = a.get("text")
        return None if text is None or text == "" else {"action": "type", "text": str(text)}
    if t == "key":
        keys = [normalize_key(k) for k in (a.get("keys") or []) if str(k).strip()]
        return {"action": "key", "keys": keys} if keys else None
    if t == "wait":
        try:
            sec = float(a.get("seconds", 1))
        except (TypeError, ValueError):
            sec = 1.0
        return {"action": "wait", "seconds": int(sec) if float(sec).is_integer() else sec}
    return None  # mouse_down / mouse_up 等低层动作不进入训练目标


def convert_sequence(actions: List[dict], screen_size: Tuple[int, int]) -> List[dict]:
    """转换整条动作序列：move 紧跟 drag 时合并成一个 drag（起点 = move 的坐标）"""
    out: List[dict] = []
    prev = None
    for a in actions or []:
        v = from_dataset_action(a, screen_size, prev_point=prev)
        if v is None:
            continue
        if v["action"] == "drag" and out and out[-1]["action"] == "move":
            out.pop()
        out.append(v)
        prev = (v["x"], v["y"]) if "x" in v and v["action"] != "drag" else (
            (v["x2"], v["y2"]) if v["action"] == "drag" else prev)
    return out


def to_target_text(actions: List[dict]) -> str:
    """训练目标：紧凑 JSON（键顺序固定，中文不转义）"""
    order = ("action", "x", "y", "x2", "y2", "amount", "text", "keys", "seconds")
    norm = [{k: a[k] for k in order if k in a} for a in actions]
    return json.dumps(norm, ensure_ascii=False, separators=(", ", ": "))


# --------------------------------------------------------------------------- #
# 模型输出 → VLM 动作（尽量宽容：代码块、思考标签、单个对象、Qwen 坐标标记、像素坐标等）
# --------------------------------------------------------------------------- #
_THINK_RE = re.compile(r"<think>.*?</think>", re.S)
_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.S)
_BOX_RE = re.compile(r"\((\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\)(?:\s*,\s*\((\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\))?")


def _extract_json(text: str):
    text = _THINK_RE.sub("", text or "").strip()
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()
    # 先出现的括号决定是数组还是对象（否则 {"keys": ["enter"]} 会被当成数组 ["enter"]）
    pairs = sorted((("[", "]"), ("{", "}")), key=lambda p: (text.find(p[0]) < 0, text.find(p[0])))
    for opener, closer in pairs:
        start = text.find(opener)
        if start < 0:
            continue
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == opener:
                depth += 1
            elif c == closer:
                depth -= 1
                if depth == 0:
                    chunk = text[start:i + 1]
                    try:
                        return json.loads(chunk)
                    except json.JSONDecodeError:
                        try:  # 常见小毛病：单引号、尾逗号
                            fixed = re.sub(r",\s*([\]}])", r"\1", chunk.replace("'", '"'))
                            return json.loads(fixed)
                        except json.JSONDecodeError:
                            break
    return None


def _num(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)) and math.isfinite(v):
        return float(v)
    if isinstance(v, str):
        try:
            f = float(v.strip())
            return f if math.isfinite(f) else None
        except ValueError:
            return None
    return None


def _coord_pair(d: dict, kx: str, ky: str, screen_size):
    """读取一对坐标并统一到 0~1000：≤1 的小数视为比例；>1000 视为像素（需要 screen_size）"""
    x, y = _num(d.get(kx)), _num(d.get(ky))
    if x is None or y is None:
        return None
    if 0 <= x <= 1 and 0 <= y <= 1 and (isinstance(d.get(kx), float) or isinstance(d.get(ky), float)) \
            and not (x == 0 and y == 0) and (x < 1 or y < 1):
        return clamp_coord(x * COORD_MAX), clamp_coord(y * COORD_MAX)
    if (x > COORD_MAX or y > COORD_MAX) and screen_size and screen_size[0] and screen_size[1]:
        return _to_norm(x, screen_size[0]), _to_norm(y, screen_size[1])
    if x < 0 or y < 0:
        return None
    return clamp_coord(x), clamp_coord(y)


def _point_from(d: dict, screen_size):
    for kx, ky in (("x", "y"), ("x1", "y1")):
        p = _coord_pair(d, kx, ky, screen_size)
        if p is not None:
            return p
    for key in ("point", "position", "coordinate", "coordinates", "pos", "start_point", "start"):
        v = d.get(key)
        if isinstance(v, (list, tuple)) and len(v) in (2, 4):
            vals = [_num(t) for t in v]
            if None in vals:
                continue
            if len(vals) == 4:  # bbox → 中心
                vals = [(vals[0] + vals[2]) / 2, (vals[1] + vals[3]) / 2]
            p = _coord_pair({"x": v[0] if len(v) == 2 else vals[0], "y": v[1] if len(v) == 2 else vals[1]},
                            "x", "y", screen_size)
            if p is not None:
                return p
        if isinstance(v, dict):
            p = _coord_pair(v, "x", "y", screen_size) or _coord_pair(v, "width", "height", screen_size)
            if p is not None:
                return p
        if isinstance(v, str):
            m = _BOX_RE.search(v)
            if m:
                return _box_center(m, screen_size)
    return None


def _box_center(m, screen_size):
    x1, y1 = float(m.group(1)), float(m.group(2))
    if m.group(3) is not None:
        x1, y1 = (x1 + float(m.group(3))) / 2, (y1 + float(m.group(4))) / 2
    return _coord_pair({"x": x1, "y": y1}, "x", "y", screen_size)


def normalize_action(d, screen_size=None) -> Optional[dict]:
    """单个动作字典 → 标准 VLM 动作；无法识别返回 None"""
    if not isinstance(d, dict):
        return None
    raw = d.get("action", d.get("action_type", d.get("type_", d.get("name"))))
    if raw is None and isinstance(d.get("type"), str) and d.get("type").lower() in ACTIONS + tuple(_ACTION_ALIASES):
        raw = d.get("type")
    if not isinstance(raw, str):
        return None
    kind = raw.strip().lower().replace(" ", "_")
    kind = _ACTION_ALIASES.get(kind, kind)
    if kind == "click" and str(d.get("button", "")).lower() == "right":
        kind = "right_click"
    if kind == "click" and _num(d.get("clicks")) == 2:
        kind = "double_click"
    if kind not in ACTIONS:
        return None
    if kind in POINT_ACTIONS:
        p = _point_from(d, screen_size)
        if p is None:
            return None
        out = {"action": kind, "x": p[0], "y": p[1]}
        if kind == "drag":
            q = _coord_pair(d, "x2", "y2", screen_size)
            if q is None:
                for key in ("end", "end_point", "to", "target"):
                    v = d.get(key)
                    if isinstance(v, (list, tuple)) and len(v) == 2:
                        q = _coord_pair({"x": v[0], "y": v[1]}, "x", "y", screen_size)
                    elif isinstance(v, dict):
                        q = _coord_pair(v, "x", "y", screen_size)
                    if q is not None:
                        break
            if q is None:
                return None
            out.update(x2=q[0], y2=q[1])
        return out
    if kind == "scroll":
        amt = _num(d.get("amount", d.get("clicks", d.get("delta"))))
        if amt is None:
            direction = str(d.get("direction", raw)).lower()
            amt = 3.0 if "up" in direction else -3.0
        elif "down" in str(d.get("direction", raw)).lower() and amt > 0:
            amt = -amt
        out = {"action": "scroll", "amount": int(amt) if amt else -1}
        p = _point_from(d, screen_size)
        if p is not None:
            out.update(x=p[0], y=p[1])
        return out
    if kind == "type":
        text = d.get("text", d.get("content", d.get("value", d.get("input_text"))))
        if text is None or (isinstance(text, str) and text == ""):
            return None
        return {"action": "type", "text": str(text)}
    if kind == "key":
        keys = d.get("keys", d.get("key", d.get("text", d.get("value"))))
        if isinstance(keys, str):
            keys = [k for k in re.split(r"\s*[+,]\s*|\s+", keys.strip()) if k]
        if not isinstance(keys, (list, tuple)):
            return None
        keys = [normalize_key(k) for k in keys if str(k).strip()]
        return {"action": "key", "keys": keys} if keys else None
    if kind == "wait":
        sec = _num(d.get("seconds", d.get("time", d.get("duration", 1))))
        if sec is None or sec < 0:
            sec = 1.0
        sec = min(sec, 60.0)
        return {"action": "wait", "seconds": int(sec) if float(sec).is_integer() else sec}
    return None


def parse_output(text: str, screen_size=None) -> dict:
    """解析模型输出。返回 {"ok": bool, "actions": [...], "error": str|None, "dropped": int}
    ok=True 表示至少解析出一个合法动作；dropped 是无法识别而丢弃的动作数。"""
    if not isinstance(text, str) or not text.strip():
        return {"ok": False, "actions": [], "error": "空输出", "dropped": 0}
    data = _extract_json(text)
    if data is None:
        return {"ok": False, "actions": [], "error": "找不到合法 JSON", "dropped": 0}
    if isinstance(data, dict):
        for key in ("actions", "action_list", "steps"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
        else:
            data = [data]
    if not isinstance(data, list):
        return {"ok": False, "actions": [], "error": "JSON 不是数组或对象", "dropped": 0}
    actions, dropped = [], 0
    for d in data:
        a = normalize_action(d, screen_size)
        if a is None:
            dropped += 1
        else:
            actions.append(a)
    if not actions:
        return {"ok": False, "actions": [], "error": "没有可识别的动作", "dropped": dropped}
    return {"ok": True, "actions": actions, "error": None, "dropped": dropped}


# --------------------------------------------------------------------------- #
# VLM 动作 → 本项目 Agent 动作（agent/executor.py 可直接执行，坐标为鼠标逻辑坐标）
# --------------------------------------------------------------------------- #
def to_screen_point(x: int, y: int, logical_size: Tuple[int, int], offset: Tuple[int, int] = (0, 0)):
    """0~1000 → 鼠标坐标。按“像素中心”换算：1000 对应最后一个像素，不会越界。"""
    w, h = logical_size
    if not w or not h:
        raise ValueError(f"屏幕尺寸无效: {logical_size}")
    px = min(w - 1, max(0, int(round(clamp_coord(x) / COORD_MAX * (w - 1)))))
    py = min(h - 1, max(0, int(round(clamp_coord(y) / COORD_MAX * (h - 1)))))
    return px + offset[0], py + offset[1]


def _one_agent_action(a: dict, logical_size, offset) -> dict:
    t = a["action"]
    pt = (lambda: list(to_screen_point(a["x"], a["y"], logical_size, offset)))
    if t in ("click", "double_click"):
        return {"action_type": t, "target_text": "", "input_text": "", "point": pt()}
    if t == "right_click":
        return {"action_type": "click", "button": "right", "target_text": "", "input_text": "", "point": pt()}
    if t == "move":
        return {"action_type": "move", "target_text": "", "input_text": "", "point": pt()}
    if t == "drag":
        return {"action_type": "drag", "target_text": "", "input_text": "", "point": pt(),
                "end_point": list(to_screen_point(a["x2"], a["y2"], logical_size, offset))}
    if t == "scroll":
        out = {"action_type": "scroll", "target_text": "", "input_text": "", "amount": int(a.get("amount", -3))}
        if "x" in a:
            out["point"] = pt()
        return out
    if t == "type":
        return {"action_type": "type", "target_text": "", "input_text": a["text"]}
    if t == "key":
        keys = a["keys"]
        if len(keys) == 1:
            return {"action_type": "key", "target_text": "", "input_text": keys[0]}
        return {"action_type": "hotkey", "target_text": "", "input_text": "+".join(keys)}
    if t == "wait":
        return {"action_type": "wait", "target_text": "", "input_text": str(a.get("seconds", 1))}
    raise ValueError(f"不支持的动作 {t!r}")


def to_agent_action(actions: List[dict], logical_size: Tuple[int, int], offset: Tuple[int, int] = (0, 0)) -> dict:
    """VLM 动作序列 → 一个 Agent 动作（第一个动作 + follow_up），与第4周执行器的约定一致"""
    if not actions:
        return {"action_type": None, "_invalid": True, "_reasons": ["VLM 没有给出可执行动作"]}
    converted = [_one_agent_action(a, logical_size, offset) for a in actions]
    head = dict(converted[0])
    if len(converted) > 1:
        head["follow_up"] = converted[1:]
    head["reason"] = "VLM 直接输出坐标"
    return head
