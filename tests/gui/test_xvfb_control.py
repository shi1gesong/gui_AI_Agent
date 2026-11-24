"""第2周 控制模块 —— Xvfb 虚拟屏幕上的真实鼠标键盘操作（非 mock）

真实 pyautogui 向虚拟 X11 屏幕发送事件，全屏“事件记录窗口”(event_logger_app.py) 记录收到的
按下/抬起/拖动/滚轮/按键/文本内容，用例据此断言。第2周未实测的拖拽、滚动在这里首次真实验证。
坐标按屏幕比例给出，可用 XVFB_RES=2560x1440 等环境变量在不同分辨率下重复运行。
"""
import time

import pytest

from conftest import xvfb_available
from gui.xvfb_utils import EventLogger, tk_available

pytestmark = [pytest.mark.xvfb,
              pytest.mark.skipif(not tk_available(), reason="需要 tkinter 运行事件记录窗口")]


@pytest.fixture(scope="module")
def screen():
    if not xvfb_available():
        pytest.skip("无 Xvfb")
    import pyautogui
    return pyautogui.size()


@pytest.fixture
def env(tmp_path, screen):
    from control.mouse_keyboard import MouseKeyboardController
    w, h = screen
    logger = EventLogger(tmp_path, w, h)
    ctl = MouseKeyboardController()
    ctl_move_safe(w, h)
    yield ctl, logger, w, h
    ctl_move_safe(w, h)
    logger.close()


def pt(w, h, fx, fy):
    return int(w * fx), int(h * fy)


@pytest.mark.functional
class TestRealControlFunctional:
    @pytest.mark.parametrize("fx,fy", [(0.25, 0.25), (0.5, 0.75), (0.9, 0.1)])
    def test_left_click_position(self, env, fx, fy):
        ctl, log, w, h = env
        x, y = pt(w, h, fx, fy)
        ctl.click(x, y)
        evs = log.wait_for(lambda es: any(e["ev"] == "release" for e in es))
        press = [e for e in evs if e["ev"] == "press"][-1]
        assert (press["b"], press["x"], press["y"]) == (1, x, y)

    def test_right_click(self, env):
        ctl, log, w, h = env
        ctl.click(*pt(w, h, 0.3, 0.6), button="right")
        evs = log.wait_for(lambda es: any(e["ev"] == "press" for e in es))
        assert [e["b"] for e in evs if e["ev"] == "press"] == [3]

    def test_middle_click(self, env):
        ctl, log, w, h = env
        ctl.click(*pt(w, h, 0.3, 0.6), button="middle")
        evs = log.wait_for(lambda es: any(e["ev"] == "press" for e in es))
        assert [e["b"] for e in evs if e["ev"] == "press"] == [2]

    def test_double_click(self, env):
        ctl, log, w, h = env
        ctl.double_click(*pt(w, h, 0.4, 0.7))
        evs = log.wait_for(lambda es: any(e["ev"] == "double" for e in es))
        assert len([e for e in evs if e["ev"] == "double"]) == 1

    def test_move_to_exact(self, env):
        ctl, log, w, h = env
        x, y = pt(w, h, 0.33, 0.66)
        ctl.move_to(x, y, duration=0.1)
        assert tuple(ctl.get_mouse_position()) == (x, y)

    def test_drag_press_at_start_release_at_end(self, env):
        ctl, log, w, h = env
        (sx, sy), (ex, ey) = pt(w, h, 0.2, 0.6), pt(w, h, 0.7, 0.9)
        ctl.drag_to(sx, sy, ex, ey, duration=0.3)
        evs = log.wait_for(lambda es: any(e["ev"] == "release" for e in es))
        press = [e for e in evs if e["ev"] == "press"][0]
        release = [e for e in evs if e["ev"] == "release"][-1]
        assert (press["x"], press["y"]) == (sx, sy)
        assert (release["x"], release["y"]) == (ex, ey)

    def test_drag_generates_intermediate_motion(self, env):
        ctl, log, w, h = env
        ctl.drag_to(*pt(w, h, 0.1, 0.6), *pt(w, h, 0.9, 0.6), duration=0.4)
        evs = log.wait_for(lambda es: any(e["ev"] == "release" for e in es))
        motion = [e for e in evs if e["ev"] == "drag_motion"]
        assert len(motion) >= 3  # 不是瞬移，而是带中间轨迹的真实拖拽

    def test_drag_right_to_left(self, env):
        ctl, log, w, h = env
        (sx, sy), (ex, ey) = pt(w, h, 0.8, 0.8), pt(w, h, 0.2, 0.55)
        ctl.drag_to(sx, sy, ex, ey, duration=0.2)
        evs = log.wait_for(lambda es: any(e["ev"] == "release" for e in es))
        rel = [e for e in evs if e["ev"] == "release"][-1]
        assert (rel["x"], rel["y"]) == (ex, ey)

    @pytest.mark.parametrize("amount,direction", [(3, "up"), (-3, "down"), (1, "up"), (-10, "down")])
    def test_scroll_direction_and_count(self, env, amount, direction):
        ctl, log, w, h = env
        ctl.scroll(amount, *pt(w, h, 0.5, 0.75))
        evs = log.wait_for(lambda es: len([e for e in es if e["ev"] == "wheel"]) >= abs(amount))
        wheels = [e for e in evs if e["ev"] == "wheel"]
        assert len(wheels) == abs(amount) and {e["dir"] for e in wheels} == {direction}

    def test_scroll_at_given_position(self, env):
        ctl, log, w, h = env
        x, y = pt(w, h, 0.6, 0.8)
        ctl.scroll(-2, x, y)
        evs = log.wait_for(lambda es: any(e["ev"] == "wheel" for e in es))
        assert all((e["x"], e["y"]) == (x, y) for e in evs if e["ev"] == "wheel")

    def test_type_ascii_text(self, env):
        ctl, log, w, h = env
        ctl.click(*pt(w, h, 0.5, 0.2))
        ctl.type_text("Hello GUI Agent", interval=0)
        assert log.wait_for(lambda es: any(e["ev"] == "text" and e["value"] == "Hello GUI Agent" for e in es))
        assert log.last_text() == "Hello GUI Agent"

    def test_type_chinese_via_clipboard(self, env):
        ctl, log, w, h = env
        ctl.click(*pt(w, h, 0.5, 0.2))
        ctl.type_text_unicode("打开浏览器搜索天气预报")
        log.wait_for(lambda es: any(e["ev"] == "text" and "天气预报" in e["value"] for e in es))
        assert log.last_text() == "打开浏览器搜索天气预报"

    def test_type_mixed_chinese_english_via_clipboard(self, env):
        ctl, log, w, h = env
        ctl.click(*pt(w, h, 0.5, 0.2))
        ctl.type_text_unicode("GUI Agent 第4周：端到端测试 ✅")
        log.wait_for(lambda es: any(e["ev"] == "text" and "✅" in e["value"] for e in es))
        assert log.last_text() == "GUI Agent 第4周：端到端测试 ✅"

    def test_press_enter_creates_newline(self, env):
        ctl, log, w, h = env
        ctl.click(*pt(w, h, 0.5, 0.2))
        ctl.type_text("ab", interval=0)
        ctl.press_key("enter")
        ctl.type_text("cd", interval=0)
        log.wait_for(lambda es: any(e["ev"] == "text" and e["value"] == "ab\ncd" for e in es))
        assert log.last_text() == "ab\ncd"

    def test_hotkey_ctrl_modifier_state(self, env):
        ctl, log, w, h = env
        ctl.click(*pt(w, h, 0.5, 0.2))
        ctl.hotkey("ctrl", "a")
        evs = log.wait_for(lambda es: any(e["ev"] == "key" and e["keysym"] == "a" for e in es))
        key_a = [e for e in evs if e["ev"] == "key" and e["keysym"] == "a"][-1]
        assert key_a["state"] & 0x4  # ControlMask

    def test_click_then_type_then_select_all_delete(self, env):
        """模拟“清空输入框再输入”：输入 -> Ctrl+/(Tk 全选) -> 退格 -> 重新输入"""
        ctl, log, w, h = env
        ctl.click(*pt(w, h, 0.5, 0.2))
        ctl.type_text("old", interval=0)
        ctl.hotkey("ctrl", "/")
        ctl.press_key("backspace")
        ctl.type_text("new", interval=0)
        log.wait_for(lambda es: any(e["ev"] == "text" and e["value"] == "new" for e in es))
        assert log.last_text() == "new"


@pytest.mark.boundary
class TestRealControlBoundary:
    def test_scroll_zero_no_events(self, env):
        ctl, log, w, h = env
        ctl.scroll(0)
        time.sleep(0.4)
        assert log.of("wheel") == []

    def test_drag_zero_distance(self, env):
        ctl, log, w, h = env
        x, y = pt(w, h, 0.5, 0.8)
        ctl.drag_to(x, y, x, y, duration=0.1)
        evs = log.wait_for(lambda es: any(e["ev"] == "release" for e in es))
        rel = [e for e in evs if e["ev"] == "release"][-1]
        assert (rel["x"], rel["y"]) == (x, y)

    def test_click_near_bottom_right(self, env):
        ctl, log, w, h = env
        ctl.click(w - 2, h - 2)
        evs = log.wait_for(lambda es: any(e["ev"] == "press" for e in es))
        assert (evs[-1]["x"], evs[-1]["y"]) == (w - 2, h - 2) or any(
            e["ev"] == "press" and (e["x"], e["y"]) == (w - 2, h - 2) for e in evs)

    def test_type_empty_string_no_change(self, env):
        ctl, log, w, h = env
        ctl.click(*pt(w, h, 0.5, 0.2))
        ctl.type_text("x", interval=0)
        log.wait_for(lambda es: any(e["ev"] == "text" for e in es))
        ctl.type_text("", interval=0)
        time.sleep(0.3)
        assert log.last_text() == "x"

    @pytest.mark.parametrize("corner", ["top_left", "top_right", "bottom_left", "bottom_right"])
    def test_click_exact_corner_pixel_nudged(self, env, corner):
        """目标落在四角像素时向内偏移 1 像素：点击真实生效，不再触发 FAILSAFE（修复前会抛 FailSafeException）"""
        ctl, log, w, h = env
        x = 0 if "left" in corner else w - 1
        y = 0 if "top" in corner else h - 1
        ctl.click(x, y)
        evs = log.wait_for(lambda es: any(e["ev"] == "press" for e in es))
        press = [e for e in evs if e["ev"] == "press"][-1]
        assert (press["x"], press["y"]) == (1 if x == 0 else w - 2, 1 if y == 0 else h - 2)



    def test_type_text_chinese_auto_clipboard(self, env):
        """修复前 type_text 会丢掉中文只剩 abc；现在自动走剪贴板"""
        ctl, log, w, h = env
        ctl.click(*pt(w, h, 0.5, 0.2))
        ctl.type_text("搜索abc", interval=0)
        log.wait_for(lambda es: any(e["ev"] == "text" and "abc" in e["value"] for e in es))
        assert log.last_text() == "搜索abc"

    def test_clipboard_restored_after_chinese_input(self, env):
        import pyperclip
        ctl, log, w, h = env
        pyperclip.copy("用户原来复制的内容")
        ctl.click(*pt(w, h, 0.5, 0.2))
        ctl.type_text_unicode("临时输入")
        log.wait_for(lambda es: any(e["ev"] == "text" and e["value"] == "临时输入" for e in es))
        assert log.last_text() == "临时输入" and pyperclip.paste() == "用户原来复制的内容"


def ctl_move_safe(w, h):
    import pyautogui
    pyautogui.FAILSAFE = False
    pyautogui.moveTo(w // 2, h // 2)
    pyautogui.FAILSAFE = True


@pytest.mark.exception
class TestRealControlException:
    def test_failsafe_top_left_corner(self, env):
        """用户把鼠标甩到左上角紧急停止：FAILSAFE 仍然有效"""
        import pyautogui
        ctl, log, w, h = env
        pyautogui.moveTo(0, 0)
        try:
            with pytest.raises(pyautogui.FailSafeException):
                ctl.click(w // 2, h // 2)
        finally:
            ctl_move_safe(w, h)

    def test_failsafe_blocks_typing(self, env):
        import pyautogui
        ctl, log, w, h = env
        pyautogui.moveTo(w - 1, 0)
        try:
            with pytest.raises(pyautogui.FailSafeException):
                ctl.type_text("should not type", interval=0)
        finally:
            ctl_move_safe(w, h)
        time.sleep(0.3)
        assert log.last_text(timeout=0.5) in (None, "")

    def test_invalid_button_raises(self, env):
        ctl, log, w, h = env
        with pytest.raises(Exception):
            ctl.click(*pt(w, h, 0.5, 0.5), button="side")

    def test_unknown_key_rejected(self, env):
        """pyautogui 对不认识的按键名会静默忽略；控制模块现在先校验并报错"""
        ctl, log, w, h = env
        ctl.click(*pt(w, h, 0.5, 0.2))
        with pytest.raises(ValueError):
            ctl.press_key("not_a_real_key")
        time.sleep(0.3)
        assert log.of("key") == []

    def test_click_outside_screen_rejected(self, env):
        ctl, log, w, h = env
        with pytest.raises(ValueError):
            ctl.click(w + 500, h // 2)
        time.sleep(0.2)
        assert log.of("press") == []
