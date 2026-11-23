"""Windows 真机用例（real_device 标记，默认跳过）

会真实操作鼠标键盘！运行前请关闭所有记事本窗口，运行期间不要动鼠标：
    set GUI_AGENT_REAL_DEVICE=1
    pytest -m real_device -v
用记事本作为测试目标，通过 Ctrl+A / Ctrl+C 读回剪贴板来验证输入、拖拽选中、滚动等结果。
注意：该文件在 Linux 云端环境无法执行，提交时未经真机验证，首次运行请留意结果。
"""
import subprocess
import sys
import time

import pytest

pytestmark = [pytest.mark.real_device,
              pytest.mark.skipif(not sys.platform.startswith("win"), reason="仅 Windows")]


@pytest.fixture
def notepad():
    import pyautogui
    import pyperclip
    from control.mouse_keyboard import MouseKeyboardController

    proc = subprocess.Popen(["notepad.exe"])
    time.sleep(2.0)
    wins = [w for w in pyautogui.getWindowsWithTitle("记事本") + pyautogui.getWindowsWithTitle("Notepad")]
    assert wins, "未找到记事本窗口"
    win = wins[0]
    win.activate()
    time.sleep(0.5)
    ctl = MouseKeyboardController()
    cx, cy = win.left + win.width // 2, win.top + win.height // 2
    ctl.click(cx, cy)

    def read_text():
        ctl.hotkey("ctrl", "a")
        ctl.hotkey("ctrl", "c")
        time.sleep(0.3)
        return pyperclip.paste().replace("\r\n", "\n")

    yield ctl, win, read_text
    subprocess.run(["taskkill", "/IM", "notepad.exe", "/F"], capture_output=True)
    proc.kill()


@pytest.mark.functional
def test_device_full_screenshot(tmp_path):
    import pyautogui
    from perception.screenshot import ScreenCapture
    img = ScreenCapture(save_dir=str(tmp_path)).capture_full_screen()
    assert img.size == tuple(pyautogui.size()) or img.size[0] >= pyautogui.size()[0]  # DPI 缩放时物理像素 >= 逻辑像素


@pytest.mark.multi_resolution
def test_device_screenshot_vs_pyautogui_scale(tmp_path):
    """记录 DPI 缩放比：截图(物理像素) / pyautogui(逻辑像素)。不为 1 时 OCR 坐标需换算后再点击"""
    import pyautogui
    from perception.screenshot import ScreenCapture
    img = ScreenCapture(save_dir=str(tmp_path)).capture_full_screen(save=False)
    scale = img.size[0] / pyautogui.size()[0]
    print(f"DPI scale = {scale:.2f}")
    assert scale in (1.0, 1.25, 1.5, 1.75, 2.0)


@pytest.mark.functional
def test_device_type_ascii(notepad):
    ctl, _, read = notepad
    ctl.type_text("Hello GUI Agent")
    assert read() == "Hello GUI Agent"


@pytest.mark.functional
def test_device_type_chinese(notepad):
    ctl, _, read = notepad
    ctl.type_text_unicode("打开浏览器搜索天气预报")
    assert read() == "打开浏览器搜索天气预报"


@pytest.mark.functional
def test_device_enter_key(notepad):
    ctl, _, read = notepad
    ctl.type_text("a")
    ctl.press_key("enter")
    ctl.type_text("b")
    assert read() == "a\nb"


@pytest.mark.functional
def test_device_drag_select_text(notepad):
    """拖拽选中第一行文字后输入替换内容，验证拖拽真实生效"""
    ctl, win, read = notepad
    ctl.type_text("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    ctl.press_key("home")
    # 第一行文字所在高度：通过 Home 后的光标无法直接获得，退而用窗口顶部偏移估计
    line_y = win.top + 100
    ctl.drag_to(win.left + 15, line_y, win.left + win.width - 40, line_y, duration=0.5)
    ctl.type_text("Y")
    assert "Y" in read() and len(read()) < 52


@pytest.mark.functional
def test_device_scroll_changes_view(notepad, tmp_path):
    from perception.screenshot import ScreenCapture
    ctl, win, read = notepad
    ctl.type_text_unicode("\n".join(f"第{i}行" for i in range(200)))
    ctl.hotkey("ctrl", "home")
    cap = ScreenCapture(save_dir=str(tmp_path))
    region = (win.left + 10, win.top + 80, 300, 300)
    before = cap.capture_region(*region, save=False)
    ctl.scroll(-10, x=win.left + win.width // 2, y=win.top + win.height // 2)
    time.sleep(0.5)
    after = cap.capture_region(*region, save=False)
    assert list(before.getdata()) != list(after.getdata())


@pytest.mark.functional
def test_device_hotkey_select_all_delete(notepad):
    ctl, _, read = notepad
    ctl.type_text("old text")
    ctl.hotkey("ctrl", "a")
    ctl.press_key("backspace")
    ctl.type_text("new")
    assert read() == "new"


@pytest.mark.boundary
def test_device_long_chinese_text(notepad):
    ctl, _, read = notepad
    text = "长文本输入测试" * 300
    ctl.type_text_unicode(text)
    assert read() == text


@pytest.mark.exception
def test_device_failsafe_corner():
    import pyautogui
    from control.mouse_keyboard import MouseKeyboardController
    ctl = MouseKeyboardController()
    with pytest.raises(pyautogui.FailSafeException):
        ctl.click(0, 0)
    pyautogui.FAILSAFE = False
    pyautogui.moveTo(ctl.screen_width // 2, ctl.screen_height // 2)
    pyautogui.FAILSAFE = True
