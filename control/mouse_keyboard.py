"""
鼠标键盘控制模块
支持：点击（单击/双击/右键）、输入文本、滚动、拖拽
依赖：pyautogui（主控制逻辑）、pynput（辅助监听，本模块暂只用pyautogui）

⚠️ 安全提示：pyautogui 默认开启 FAILSAFE，鼠标移动到屏幕四角会触发异常并中断程序，
这是防止程序失控误操作的保护机制，测试时不要关闭这个选项。

2026-11-23 修复（依据 tests/ 测试结果）：
- type_text 遇到中文等非 ASCII 字符时自动改走剪贴板，不再静默丢字
- 剪贴板输入后恢复用户原来的剪贴板内容
- 所有坐标先做屏幕范围校验，越界/负坐标直接报 ValueError
- 目标坐标恰好落在屏幕四角像素时向内偏移 1 像素，避免程序自己触发 FAILSAFE
  （用户手动把鼠标甩到角落时 FAILSAFE 仍然有效）
- press_key / hotkey 校验按键名，未知按键名报 ValueError（pyautogui 原本会静默忽略）

2026-11-24 修复（Windows 真机测试）：
- Windows 下 type_text 默认统一走剪贴板粘贴。中文输入法处于中文模式时，typewrite 逐个按下的英文字母
  会被输入法截获（真机上出现过 "Hello GUI Agent" 变成 "HelloGUI Agent"、输入变成 "Y" 等）。
  可用 MouseKeyboardController(prefer_clipboard=False) 恢复逐字敲键盘的方式。
- scroll 的 amount 统一为“滚轮格数”。pyautogui 在 Windows 上把 clicks 原样作为滚轮增量传给系统，
  而 Windows 一格滚轮是 120（WHEEL_DELTA），所以 scroll(-10) 实际只滚了 1/12 格，真机上完全没有滚动；
  现在 Windows 上自动乘以 120。Linux(X11) 上 pyautogui 本来就按格数发送，不需要换算。
"""
import sys
import time
from typing import Optional

import pyautogui

# 每次操作之间的默认间隔，避免操作过快导致目标程序来不及响应
pyautogui.PAUSE = 0.3
pyautogui.FAILSAFE = True

# Windows 滚轮一格对应的增量（WHEEL_DELTA）
WINDOWS_WHEEL_DELTA = 120

# 剪贴板粘贴后等待目标程序读取剪贴板的时间，过早恢复旧内容会导致粘贴进旧内容
CLIPBOARD_SETTLE_SEC = 0.15


class MouseKeyboardController:
    def __init__(self, prefer_clipboard: Optional[bool] = None):
        """prefer_clipboard: 是否所有文本都用剪贴板粘贴输入。None 表示 Windows 上为 True、其他系统为 False"""
        self.screen_width, self.screen_height = pyautogui.size()
        if prefer_clipboard is None:
            prefer_clipboard = sys.platform.startswith("win")
        self.prefer_clipboard = prefer_clipboard
        # 一格滚轮对应传给 pyautogui.scroll 的数值
        self.wheel_delta = WINDOWS_WHEEL_DELTA if sys.platform.startswith("win") else 1

    # ---------- 坐标与按键校验 ----------
    def _safe_point(self, x: int, y: int):
        """校验坐标在屏幕内；若恰好是四角像素，向内偏移 1 像素避免触发 FAILSAFE。返回实际使用的坐标。"""
        w, h = self.screen_width, self.screen_height
        if not (0 <= x < w and 0 <= y < h):
            raise ValueError(f"坐标 ({x}, {y}) 超出屏幕范围 {w}x{h}")
        if x in (0, w - 1) and y in (0, h - 1):
            x = 1 if x == 0 else w - 2
            y = 1 if y == 0 else h - 2
        return x, y

    @staticmethod
    def _check_key(key: str):
        valid = getattr(pyautogui, "KEYBOARD_KEYS", None)
        if valid is not None and key not in valid and key.lower() not in valid:
            raise ValueError(f"未知按键名: {key!r}")

    # ---------- 鼠标操作 ----------
    def click(self, x: int, y: int, button: str = "left", clicks: int = 1):
        """在指定坐标点击，button可选 left/right/middle"""
        x, y = self._safe_point(x, y)
        pyautogui.click(x=x, y=y, button=button, clicks=clicks)

    def double_click(self, x: int, y: int):
        x, y = self._safe_point(x, y)
        pyautogui.doubleClick(x=x, y=y)

    def move_to(self, x: int, y: int, duration: float = 0.2):
        """平滑移动鼠标到指定坐标，duration控制移动耗时（模拟人类操作，避免被识别为脚本）"""
        x, y = self._safe_point(x, y)
        pyautogui.moveTo(x, y, duration=duration)

    def drag_to(self, start_x: int, start_y: int, end_x: int, end_y: int, duration: float = 0.5):
        """从起点拖拽到终点"""
        start_x, start_y = self._safe_point(start_x, start_y)
        end_x, end_y = self._safe_point(end_x, end_y)
        pyautogui.moveTo(start_x, start_y)
        pyautogui.dragTo(end_x, end_y, duration=duration, button="left")

    def scroll(self, amount: int, x: int = None, y: int = None):
        """滚动 amount 格（与鼠标滚轮一格一致），为正向上滚，为负向下滚。可选先移动到指定坐标再滚动"""
        if x is not None and y is not None:
            x, y = self._safe_point(x, y)
            pyautogui.moveTo(x, y)
        pyautogui.scroll(int(amount) * self.wheel_delta)

    # ---------- 键盘操作 ----------
    def type_text(self, text: str, interval: float = 0.02):
        """输入文本，interval控制打字速度（仅逐字输入时有效）。
        含非 ASCII 字符（中文等）或 prefer_clipboard=True 时改用剪贴板粘贴。"""
        if self.prefer_clipboard or not text.isascii():
            self.type_text_unicode(text)
            return
        pyautogui.typewrite(text, interval=interval)

    def type_text_unicode(self, text: str):
        """
        输入包含中文等非ASCII字符的文本。
        pyautogui.typewrite 只支持ASCII，中文输入需要用剪贴板中转；输入完成后恢复原剪贴板内容。
        """
        import pyperclip
        try:
            previous = pyperclip.paste()
        except Exception:
            previous = None
        pyperclip.copy(text)
        pyautogui.hotkey("ctrl", "v")
        time.sleep(CLIPBOARD_SETTLE_SEC)
        if previous is not None:
            try:
                pyperclip.copy(previous)
            except Exception:
                pass

    def press_key(self, key: str):
        """按下单个按键，如 'enter', 'esc', 'tab'"""
        self._check_key(key)
        pyautogui.press(key)

    def hotkey(self, *keys):
        """组合键，如 hotkey('ctrl', 'c')"""
        for k in keys:
            self._check_key(k)
        pyautogui.hotkey(*keys)

    # ---------- 状态查询 ----------
    def get_mouse_position(self):
        return pyautogui.position()


if __name__ == "__main__":
    controller = MouseKeyboardController()
    print(f"屏幕分辨率: {controller.screen_width}x{controller.screen_height}")

    # 单元测试1：获取鼠标当前位置（无副作用，安全测试）
    pos = controller.get_mouse_position()
    print(f"✅ 当前鼠标位置: {pos}")

    # 单元测试2：移动鼠标到屏幕中心（不点击，安全测试）
    center_x, center_y = controller.screen_width // 2, controller.screen_height // 2
    controller.move_to(center_x, center_y, duration=0.3)
    new_pos = controller.get_mouse_position()
    assert abs(new_pos[0] - center_x) < 5 and abs(new_pos[1] - center_y) < 5, "鼠标移动位置偏差过大"
    print(f"✅ 鼠标移动测试通过，移动到: {new_pos}")

    print("\n完整测试请运行: pytest tests/ （说明见 tests/README.md）")
