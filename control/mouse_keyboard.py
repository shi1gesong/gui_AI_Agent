"""
鼠标键盘控制模块
支持：点击（单击/双击/右键）、输入文本、滚动、拖拽
依赖：pyautogui（主控制逻辑）、pynput（辅助监听，本模块暂只用pyautogui）

⚠️ 安全提示：pyautogui 默认开启 FAILSAFE，鼠标移动到屏幕左上角(0,0)会触发异常并中断程序，
这是防止程序失控误操作的保护机制，测试时不要关闭这个选项。
"""
import time

import pyautogui

# 每次操作之间的默认间隔，避免操作过快导致目标程序来不及响应
pyautogui.PAUSE = 0.3
pyautogui.FAILSAFE = True


class MouseKeyboardController:
    def __init__(self):
        self.screen_width, self.screen_height = pyautogui.size()

    # ---------- 鼠标操作 ----------
    def click(self, x: int, y: int, button: str = "left", clicks: int = 1):
        """在指定坐标点击，button可选 left/right/middle"""
        pyautogui.click(x=x, y=y, button=button, clicks=clicks)

    def double_click(self, x: int, y: int):
        pyautogui.doubleClick(x=x, y=y)

    def move_to(self, x: int, y: int, duration: float = 0.2):
        """平滑移动鼠标到指定坐标，duration控制移动耗时（模拟人类操作，避免被识别为脚本）"""
        pyautogui.moveTo(x, y, duration=duration)

    def drag_to(self, start_x: int, start_y: int, end_x: int, end_y: int, duration: float = 0.5):
        """从起点拖拽到终点"""
        pyautogui.moveTo(start_x, start_y)
        pyautogui.dragTo(end_x, end_y, duration=duration, button="left")

    def scroll(self, amount: int, x: int = None, y: int = None):
        """滚动，amount为正向上滚，为负向下滚。可选先移动到指定坐标再滚动"""
        if x is not None and y is not None:
            pyautogui.moveTo(x, y)
        pyautogui.scroll(amount)

    # ---------- 键盘操作 ----------
    def type_text(self, text: str, interval: float = 0.02):
        """输入文本，interval控制打字速度（每个字符间隔秒数）"""
        pyautogui.typewrite(text, interval=interval)

    def type_text_unicode(self, text: str):
        """
        输入包含中文等非ASCII字符的文本。
        pyautogui.typewrite 只支持ASCII，中文输入需要用剪贴板中转。
        """
        import pyperclip
        pyperclip.copy(text)
        pyautogui.hotkey("ctrl", "v")

    def press_key(self, key: str):
        """按下单个按键，如 'enter', 'esc', 'tab'"""
        pyautogui.press(key)

    def hotkey(self, *keys):
        """组合键，如 hotkey('ctrl', 'c')"""
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

    # 以下测试涉及实际点击/输入操作，默认注释掉，需要打开一个文本编辑器/记事本后手动取消注释测试
    # controller.click(center_x, center_y)
    # controller.type_text("Hello GUI Agent")
    # controller.hotkey('ctrl', 'a')
    # controller.scroll(-5)

    print("\n基础安全测试（鼠标定位、移动）已通过。")
    print("点击/输入类操作请打开记事本等测试目标后，取消对应代码注释再运行。")
