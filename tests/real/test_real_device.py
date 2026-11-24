"""Windows 真机用例（real_device 标记，默认跳过）

会真实操作鼠标键盘！运行期间不要动鼠标、不要切换窗口（Anaconda Prompt / cmd）：
    set GUI_AGENT_REAL_DEVICE=1
    pytest -m real_device -v -s

用记事本作为测试目标。每个用例用一个独立的临时 txt 文件打开记事本（窗口标题唯一），
输入后按 Ctrl+S 保存，再直接读文件内容来校验；结束时保存并关闭这个标签页。
这样做的原因（2026-11-24 首次真机运行暴露的问题）：
- Win11 新版记事本会恢复上次未保存的标签页，之前用 taskkill 强杀后，前一个用例的文字会被带到下一个用例
- 用 Ctrl+A/Ctrl+C 读剪贴板校验时，复制偶尔没生效会读到旧剪贴板内容
现在不再强杀记事本，不会影响你自己打开的记事本窗口。
"""
import subprocess
import sys
import time
import uuid

import pytest

pytestmark = [pytest.mark.real_device,
              pytest.mark.skipif(not sys.platform.startswith("win"), reason="仅 Windows")]


@pytest.fixture
def notepad(tmp_path):
    import pyautogui
    from control.mouse_keyboard import MouseKeyboardController

    name = f"gui_agent_test_{uuid.uuid4().hex[:8]}"
    path = tmp_path / f"{name}.txt"
    path.write_text("", encoding="utf-8-sig")  # 带 BOM，保证旧版记事本也按 UTF-8 保存
    subprocess.Popen(["notepad.exe", str(path)])

    win = None
    for _ in range(60):
        wins = pyautogui.getWindowsWithTitle(name)
        if wins:
            win = wins[0]
            break
        time.sleep(0.25)
    assert win is not None, f"15 秒内没有找到标题含 {name} 的记事本窗口"
    try:
        win.activate()
    except Exception:
        pass  # pygetwindow 在部分系统上 activate 会误报错误，下面的点击同样能聚焦
    time.sleep(1.0)
    ctl = MouseKeyboardController()
    ctl.click(win.left + win.width // 2, win.top + win.height // 2)  # 点击编辑区获取焦点
    time.sleep(0.3)

    def read_text():
        ctl.hotkey("ctrl", "s")
        time.sleep(1.0)
        return path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")

    yield ctl, win, read_text

    ctl.hotkey("ctrl", "s")
    time.sleep(0.5)
    ctl.hotkey("ctrl", "w")  # Win11 记事本：关闭当前标签页（文件已保存，不会弹窗）
    time.sleep(0.8)
    if pyautogui.getWindowsWithTitle(name):  # 旧版记事本没有标签页，Ctrl+W 无效
        ctl.hotkey("alt", "f4")
        time.sleep(0.5)


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


def _marker_line(text, marker):
    for i, line in enumerate(text.split("\n")):
        if marker in line:
            return i
    return None


@pytest.mark.functional
def test_device_scroll_changes_view(notepad):
    """滚动是否真的生效：在同一个屏幕位置点击并插入标记，比较滚动前后标记落在第几行。
    （第一版用截图前后对比，在 2560x1440 大窗口下截到的是空白区域，前后都一样，判定不可靠）"""
    ctl, win, read = notepad
    ctl.type_text_unicode("\n".join(f"第{i}行" for i in range(200)))
    assert read().count("\n") == 199  # 先确认 200 行真的写进去了
    x, y = win.left + 80, win.top + win.height // 2

    ctl.hotkey("ctrl", "home")
    time.sleep(0.3)
    ctl.click(x, y)
    ctl.type_text("@")
    before = _marker_line(read(), "@")
    ctl.press_key("backspace")

    ctl.hotkey("ctrl", "home")
    time.sleep(0.3)
    ctl.scroll(-10, x=x, y=y)
    time.sleep(0.5)
    ctl.click(x, y)
    ctl.type_text("#")
    after = _marker_line(read(), "#")
    print(f"同一位置点击：滚动前在第 {before} 行，向下滚 10 格后在第 {after} 行")
    assert before is not None and after is not None
    assert after - before >= 10


@pytest.mark.functional
def test_device_hotkey_select_all_delete(notepad):
    ctl, _, read = notepad
    ctl.type_text("old text")
    ctl.hotkey("ctrl", "a")
    ctl.press_key("backspace")
    ctl.type_text("new")
    assert read() == "new"


@pytest.mark.boundary
def test_device_click_start_button_corner_nudged():
    """Windows 开始按钮在左下角：目标为角点像素时向内偏移，不触发 FAILSAFE"""
    import pyautogui
    from control.mouse_keyboard import MouseKeyboardController
    ctl = MouseKeyboardController()
    ctl.move_to(0, ctl.screen_height - 1, duration=0)
    assert tuple(pyautogui.position()) == (1, ctl.screen_height - 2)
    ctl.move_to(ctl.screen_width // 2, ctl.screen_height // 2, duration=0)


@pytest.mark.functional
def test_device_type_text_chinese_auto(notepad):
    ctl, _, read = notepad
    ctl.type_text("天气预报 weather")
    assert read() == "天气预报 weather"


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
    pyautogui.moveTo(0, 0)  # 模拟用户把鼠标甩到左上角紧急停止
    with pytest.raises(pyautogui.FailSafeException):
        ctl.click(ctl.screen_width // 2, ctl.screen_height // 2)
    pyautogui.FAILSAFE = False
    pyautogui.moveTo(ctl.screen_width // 2, ctl.screen_height // 2)
    pyautogui.FAILSAFE = True
