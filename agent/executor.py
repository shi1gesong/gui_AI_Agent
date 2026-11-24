"""
动作执行器：把 Agent 输出的动作交给第2周控制模块真正执行。

2026-11-23 新增：BaseGUIAgent.run() 可以传入 executor，每个动作执行后记录 executed / error。
2026-11-24 第4周扩展：
  - 动作空间：click / double_click / type / key / hotkey / scroll / open_app / close_app / wait / done
  - grounding：target_text → agent/grounding.py 的模糊匹配（容忍 OCR 空格、标点、全角半角差异）→ bbox 中心点。
    元素的 bbox 已由 perception/screen_parser.py 换算成鼠标逻辑坐标（DPI 缩放 ≠ 100% 时也能点准）
  - 每次执行返回 {"ok", "error", "detail"}，detail 记录实际点击坐标、匹配分数、打开的窗口等，写入运行轨迹
"""
import time
from typing import Callable, List, Optional

try:  # 以包方式导入（agent.executor）
    from .action_validator import KEY_ALIASES, parse_hotkey, parse_wait_seconds
    from .grounding import Grounder
except ImportError:  # 直接运行 / 把 agent/ 加入 sys.path 时
    from action_validator import KEY_ALIASES, parse_hotkey, parse_wait_seconds
    from grounding import Grounder


class ControllerExecutor:
    def __init__(self, controller, default_scroll: int = -5, app_manager=None,
                 grounder: Optional[Grounder] = None, sleep: Callable = time.sleep,
                 settle_sec: float = 0.0):
        """
        controller: control.mouse_keyboard.MouseKeyboardController 或接口相同的对象
        app_manager: control.app_manager.AppManager，open_app / close_app 需要；没有时 open_app 退回 Win 搜索
        settle_sec: 每个动作执行后等待界面响应的时间
        """
        self.controller = controller
        self.default_scroll = default_scroll
        self.app_manager = app_manager
        self.grounder = grounder or Grounder()
        self.sleep = sleep
        self.settle_sec = settle_sec

    @staticmethod
    def _center(element: dict):
        bbox = element.get("bbox")
        if not bbox or len(bbox) != 4:
            return None
        x1, y1, x2, y2 = bbox
        return (int(x1) + int(x2)) // 2, (int(y1) + int(y2)) // 2

    def _locate(self, target: str, elements: List[dict]):
        hit = self.grounder.locate(target, elements)
        if hit is None:
            return None
        self._detail.setdefault("grounding", []).append(
            {"target": target, "matched": hit["element"].get("text"), "score": hit["score"], "point": list(hit["point"])})
        return hit["point"]

    def _run_one(self, action: dict, elements: List[dict]):
        a_type = action.get("action_type")
        target = str(action.get("target_text") or "")
        text = str(action.get("input_text") or "")
        if a_type in ("click", "double_click"):
            pt = self._locate(target, elements)
            if pt is None:
                raise LookupError(f"屏幕上找不到点击目标「{target}」的坐标")
            if a_type == "click":
                self.controller.click(*pt)
            else:
                self.controller.double_click(*pt)
        elif a_type == "type":
            if target:
                # 指定了输入框却在屏幕上找不到时如实报错，不再“输入到当前焦点”后报成功
                # （2026-11-24 真机 T4：输入框没被 OCR 识别出来，文字输到了别处，步骤却显示 ✓）
                pt = self._locate(target, elements)
                if pt is None:
                    raise LookupError(f"屏幕上找不到输入框「{target}」，未输入")
                self.controller.click(*pt)
                self.sleep(0.2)
            self.controller.type_text(text)
        elif a_type == "key":
            key = KEY_ALIASES.get(text.strip().lower()) or KEY_ALIASES.get(target.strip().lower()) or text
            self.controller.press_key(key)
        elif a_type == "hotkey":
            keys = parse_hotkey(text or target)
            if keys is None:
                raise ValueError(f"无法识别的组合键「{text or target}」")
            self.controller.hotkey(*keys)
        elif a_type == "scroll":
            amount = action.get("amount", self.default_scroll)
            pt = self._locate(target, elements) if target else None
            self.controller.scroll(amount, *(pt or (None, None)))
        elif a_type == "wait":
            sec = parse_wait_seconds(text or action.get("seconds"))
            if sec is None:
                raise ValueError(f"无法解析的等待时间「{text}」")
            self.sleep(sec)
        elif a_type == "done":
            pass
        elif a_type == "open_app":
            if self.app_manager is None:  # 没有 AppManager：开始菜单搜索
                self.controller.press_key("win")
                self.sleep(0.8)
                self.controller.type_text(target)
                self.sleep(0.8)
                self.controller.press_key("enter")
                self._detail["open"] = {"method": "win_search", "target": target}
            else:
                r = self.app_manager.open(target)
                self._detail["open"] = r
                if not r.get("ok"):
                    raise RuntimeError(r.get("error", f"打开「{target}」失败"))
        elif a_type == "close_app":
            if self.app_manager is None:
                self.controller.hotkey("alt", "f4")
                self._detail["close"] = {"method": "alt+f4"}
            else:
                r = self.app_manager.close(target)
                self._detail["close"] = r
                if not r.get("ok"):
                    raise RuntimeError(r.get("error", f"关闭「{target}」失败"))
        else:
            raise ValueError(f"无法执行的动作类型: {a_type!r}")
        if self.settle_sec and a_type not in ("wait", "done"):
            self.sleep(self.settle_sec)

    def __call__(self, action: dict, elements: Optional[List[dict]] = None) -> dict:
        elements = elements or []
        self._detail = {}
        if not isinstance(action, dict):
            return {"ok": False, "error": "动作不是字典"}
        if action.get("_invalid"):
            return {"ok": False, "error": "动作未通过校验，未执行"}
        try:
            self._run_one(action, elements)
            for step in action.get("follow_up", []):
                self._run_one(step, elements)
            return {"ok": True, "detail": self._detail} if self._detail else {"ok": True}
        except Exception as e:  # 执行失败交给 Reflect / 重试逻辑判断，不中断整个流程
            out = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            if self._detail:
                out["detail"] = self._detail
            return out


class DryRunExecutor:
    """只记录不执行（CLI --dry-run）：用于先看 Agent 会做什么，再决定是否真的执行"""

    def __init__(self, grounder: Optional[Grounder] = None):
        self.grounder = grounder or Grounder()
        self.log = []

    def __call__(self, action: dict, elements: Optional[List[dict]] = None) -> dict:
        if not isinstance(action, dict) or action.get("_invalid"):
            return {"ok": False, "error": "动作未通过校验，未执行"}
        detail = {"dry_run": True}
        if action.get("action_type") in ("click", "double_click"):
            hit = self.grounder.locate(action.get("target_text", ""), elements or [])
            if hit is None:
                return {"ok": False, "error": f"屏幕上找不到点击目标「{action.get('target_text')}」的坐标",
                        "detail": detail}
            detail["point"] = list(hit["point"])
        self.log.append(action)
        return {"ok": True, "detail": detail}
