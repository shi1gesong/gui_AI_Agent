"""
动作执行器（2026-11-23 新增）：把 Agent 输出的动作交给第2周控制模块真正执行。

第3周的 Agent 只生成动作、从不执行，Reflect 却会判定“任务完成”。现在 BaseGUIAgent.run() 可以传入
executor，每个动作执行后记录 executed / error，Reflect 据此判断；未传 executor 时动作一律视为未执行。

grounding：click / type 的 target_text 在屏幕元素中查找，用 bbox 中心点作为点击坐标。
"""
from typing import List, Optional

try:  # 以包方式导入（agent.executor）
    from .action_validator import KEY_ALIASES, find_element
except ImportError:  # 直接运行 / 把 agent/ 加入 sys.path 时
    from action_validator import KEY_ALIASES, find_element


class ControllerExecutor:
    def __init__(self, controller, default_scroll: int = -5):
        """controller: control.mouse_keyboard.MouseKeyboardController 或接口相同的对象"""
        self.controller = controller
        self.default_scroll = default_scroll

    @staticmethod
    def _center(element: dict):
        bbox = element.get("bbox")
        if not bbox or len(bbox) != 4:
            return None
        x1, y1, x2, y2 = bbox
        return (int(x1) + int(x2)) // 2, (int(y1) + int(y2)) // 2

    def _locate(self, target: str, elements: List[dict]):
        hit = find_element(target, elements)
        return self._center(hit) if hit else None

    def _run_one(self, action: dict, elements: List[dict]):
        a_type = action.get("action_type")
        target = action.get("target_text") or ""
        text = action.get("input_text") or ""
        if a_type == "click":
            pt = self._locate(target, elements)
            if pt is None:
                raise LookupError(f"屏幕上找不到点击目标「{target}」的坐标")
            self.controller.click(*pt)
        elif a_type == "type":
            pt = self._locate(target, elements) if target else None
            if pt is not None:
                self.controller.click(*pt)
            self.controller.type_text(text)
        elif a_type == "key":
            key = KEY_ALIASES.get(text.strip().lower()) or KEY_ALIASES.get(target.strip().lower()) or text
            self.controller.press_key(key)
        elif a_type == "scroll":
            amount = action.get("amount", self.default_scroll)
            pt = self._locate(target, elements) if target else None
            self.controller.scroll(amount, *(pt or (None, None)))
        else:
            raise ValueError(f"无法执行的动作类型: {a_type!r}")

    def __call__(self, action: dict, elements: Optional[List[dict]] = None) -> dict:
        elements = elements or []
        if action.get("_invalid"):
            return {"ok": False, "error": "动作未通过校验，未执行"}
        try:
            self._run_one(action, elements)
            for step in action.get("follow_up", []):
                self._run_one(step, elements)
            return {"ok": True}
        except Exception as e:  # 执行失败交给 Reflect 判断，不中断整个流程
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}
