"""第2周 控制模块：control/mouse_keyboard.py —— mock 用例

所有用例通过 fake_gui fixture 把 pyautogui 换成记录调用的替身，不会真的移动鼠标。
真实鼠标键盘事件验证见 tests/gui/test_xvfb_control.py（Xvfb 虚拟屏幕）。
"""
import pytest

import control.mouse_keyboard as mk
from control.mouse_keyboard import MouseKeyboardController


@pytest.fixture
def ctl(fake_gui):
    # 显式关闭“统一走剪贴板”，使逐字输入相关用例在 Windows 和 Linux 上行为一致
    return MouseKeyboardController(prefer_clipboard=False)


@pytest.mark.functional
class TestControlFunctional:
    def test_init_reads_screen_size(self, fake_gui):
        fake_gui.width, fake_gui.height = 2560, 1440
        c = MouseKeyboardController()
        assert (c.screen_width, c.screen_height) == (2560, 1440)

    def test_click_left_default(self, ctl, fake_gui):
        ctl.click(100, 200)
        assert fake_gui.calls[-1] == ("click", (), {"x": 100, "y": 200, "button": "left", "clicks": 1})

    def test_click_right(self, ctl, fake_gui):
        ctl.click(10, 20, button="right")
        assert fake_gui.calls[-1][2]["button"] == "right"

    def test_click_middle(self, ctl, fake_gui):
        ctl.click(10, 20, button="middle")
        assert fake_gui.calls[-1][2]["button"] == "middle"

    def test_click_multiple_clicks(self, ctl, fake_gui):
        ctl.click(10, 20, clicks=3)
        assert fake_gui.calls[-1][2]["clicks"] == 3

    def test_double_click(self, ctl, fake_gui):
        ctl.double_click(300, 400)
        assert fake_gui.calls[-1] == ("doubleClick", (), {"x": 300, "y": 400})

    def test_move_to_with_duration(self, ctl, fake_gui):
        ctl.move_to(50, 60, duration=0.5)
        assert fake_gui.calls[-1] == ("moveTo", (50, 60), {"duration": 0.5})

    def test_move_to_default_duration(self, ctl, fake_gui):
        ctl.move_to(50, 60)
        assert fake_gui.calls[-1][2]["duration"] == 0.2

    def test_drag_moves_to_start_then_drags(self, ctl, fake_gui):
        ctl.drag_to(100, 100, 400, 300)
        assert fake_gui.names() == ["moveTo", "dragTo"]
        assert fake_gui.calls[0][1] == (100, 100)
        assert fake_gui.calls[1][1] == (400, 300)

    def test_drag_uses_left_button_and_duration(self, ctl, fake_gui):
        ctl.drag_to(0 + 1, 1, 2, 2, duration=1.2)
        assert fake_gui.calls[-1][2] == {"duration": 1.2, "button": "left"}

    def test_drag_ends_at_target_position(self, ctl, fake_gui):
        ctl.drag_to(10, 10, 700, 500)
        assert ctl.get_mouse_position() == (700, 500)

    def test_scroll_up_positive(self, ctl, fake_gui):
        ctl.scroll(5)
        assert fake_gui.calls == [("scroll", (5,), {})]

    def test_scroll_down_negative(self, ctl, fake_gui):
        ctl.scroll(-3)
        assert fake_gui.calls[-1][1] == (-3,)

    def test_scroll_at_position_moves_first(self, ctl, fake_gui):
        ctl.scroll(-2, x=500, y=600)
        assert fake_gui.names() == ["moveTo", "scroll"]
        assert fake_gui.calls[0][1] == (500, 600)

    def test_type_text_ascii(self, ctl, fake_gui):
        ctl.type_text("Hello GUI Agent")
        assert fake_gui.calls[-1] == ("typewrite", ("Hello GUI Agent",), {"interval": 0.02})

    def test_type_text_custom_interval(self, ctl, fake_gui):
        ctl.type_text("abc", interval=0.1)
        assert fake_gui.calls[-1][2]["interval"] == 0.1

    def test_type_unicode_chinese_via_clipboard(self, ctl, fake_gui, fake_clipboard):
        ctl.type_text_unicode("天气预报")
        assert fake_clipboard.history[0] == "天气预报"
        assert fake_gui.calls[-1] == ("hotkey", ("ctrl", "v"), {})

    def test_type_unicode_mixed_text(self, ctl, fake_gui, fake_clipboard):
        ctl.type_text_unicode("北京 weather 2026")
        assert fake_clipboard.history[0] == "北京 weather 2026"

    def test_type_unicode_emoji_and_symbols(self, ctl, fake_gui, fake_clipboard):
        ctl.type_text_unicode("✅ 测试通过！😀")
        assert fake_clipboard.history[0] == "✅ 测试通过！😀"

    def test_type_unicode_japanese_korean(self, ctl, fake_gui, fake_clipboard):
        ctl.type_text_unicode("こんにちは 안녕하세요")
        assert fake_clipboard.history[0] == "こんにちは 안녕하세요"

    def test_type_text_routes_chinese_to_clipboard(self, ctl, fake_gui, fake_clipboard):
        ctl.type_text("搜索天气")
        assert fake_clipboard.history[0] == "搜索天气"
        assert "typewrite" not in fake_gui.names()

    def test_prefer_clipboard_routes_ascii_to_clipboard(self, fake_gui, fake_clipboard):
        """Windows 中文输入法会截获逐字敲入的英文字母，prefer_clipboard=True 时英文也走粘贴"""
        c = MouseKeyboardController(prefer_clipboard=True)
        c.type_text("Hello GUI Agent")
        assert fake_clipboard.history[0] == "Hello GUI Agent" and "typewrite" not in fake_gui.names()

    @pytest.mark.parametrize("platform,expected", [("win32", True), ("linux", False), ("darwin", False)])
    def test_prefer_clipboard_default_by_platform(self, fake_gui, monkeypatch, platform, expected):
        monkeypatch.setattr(mk.sys, "platform", platform)
        assert MouseKeyboardController().prefer_clipboard is expected

    def test_type_text_ascii_does_not_touch_clipboard(self, ctl, fake_gui, fake_clipboard):
        ctl.type_text("hello")
        assert fake_clipboard.history == []

    @pytest.mark.parametrize("key", ["enter", "esc", "tab", "backspace", "f5", "win"])
    def test_press_key(self, ctl, fake_gui, key):
        ctl.press_key(key)
        assert fake_gui.calls[-1] == ("press", (key,), {})

    @pytest.mark.parametrize("keys", [("ctrl", "c"), ("ctrl", "a"), ("alt", "f4"), ("ctrl", "shift", "esc"), ("win", "d")])
    def test_hotkey_combinations(self, ctl, fake_gui, keys):
        ctl.hotkey(*keys)
        assert fake_gui.calls[-1] == ("hotkey", keys, {})

    def test_get_mouse_position(self, ctl, fake_gui):
        fake_gui.pos = (123, 456)
        assert ctl.get_mouse_position() == (123, 456)

    def test_failsafe_enabled_at_module_level(self):
        assert mk.pyautogui.FAILSAFE is True
        assert mk.pyautogui.PAUSE == 0.3

    def test_click_then_type_sequence(self, ctl, fake_gui):
        """记事本场景：先点击输入区再输入文本，调用顺序必须是 click -> typewrite"""
        ctl.click(960, 540)
        ctl.type_text("Hello")
        ctl.press_key("enter")
        assert fake_gui.names() == ["click", "typewrite", "press"]


@pytest.mark.boundary
class TestControlBoundary:
    @pytest.mark.parametrize("pt,expected", [((0, 0), (1, 1)), ((1919, 0), (1918, 1)),
                                             ((0, 1079), (1, 1078)), ((1919, 1079), (1918, 1078))],
                             ids=["top_left", "top_right", "bottom_left", "bottom_right"])
    def test_corner_pixel_nudged_inward(self, ctl, fake_gui, pt, expected):
        """四角像素会触发 pyautogui FAILSAFE，目标坐标落在角点时向内偏移 1 像素"""
        ctl.click(*pt)
        assert (fake_gui.calls[-1][2]["x"], fake_gui.calls[-1][2]["y"]) == expected

    def test_edge_non_corner_not_nudged(self, ctl, fake_gui):
        ctl.click(0, 500)
        assert (fake_gui.calls[-1][2]["x"], fake_gui.calls[-1][2]["y"]) == (0, 500)

    def test_drag_to_corner_nudged(self, ctl, fake_gui):
        ctl.drag_to(100, 100, 1919, 1079)
        assert fake_gui.calls[-1][1] == (1918, 1078)

    def test_scroll_zero(self, ctl, fake_gui):
        ctl.scroll(0)
        assert fake_gui.calls[-1][1] == (0,)

    def test_scroll_large_amount(self, ctl, fake_gui):
        ctl.scroll(-10000)
        assert fake_gui.calls[-1][1] == (-10000,)

    def test_scroll_only_x_given_does_not_move(self, ctl, fake_gui):
        ctl.scroll(1, x=100)
        assert fake_gui.names() == ["scroll"]

    def test_scroll_at_origin_zero_coordinates(self, ctl, fake_gui):
        """x=0,y=0 不是 None，仍应先移动"""
        ctl.scroll(1, x=0, y=0)
        assert fake_gui.names() == ["moveTo", "scroll"]

    def test_drag_zero_distance(self, ctl, fake_gui):
        ctl.drag_to(200, 200, 200, 200)
        assert fake_gui.names() == ["moveTo", "dragTo"]

    def test_drag_zero_duration(self, ctl, fake_gui):
        ctl.drag_to(1, 1, 5, 5, duration=0)
        assert fake_gui.calls[-1][2]["duration"] == 0

    def test_type_empty_string(self, ctl, fake_gui):
        ctl.type_text("")
        assert fake_gui.calls[-1][1] == ("",)

    def test_type_unicode_empty_string(self, ctl, fake_gui, fake_clipboard):
        ctl.type_text_unicode("")
        assert fake_clipboard.history[0] == ""

    def test_type_long_text_10k_chars(self, ctl, fake_gui, fake_clipboard):
        text = "长文本" * 3334
        ctl.type_text_unicode(text)
        assert len(fake_clipboard.history[0]) == 10002

    def test_type_multiline_text(self, ctl, fake_gui, fake_clipboard):
        ctl.type_text_unicode("第一行\n第二行\t制表")
        assert fake_clipboard.history[0] == "第一行\n第二行\t制表"

    def test_type_text_auto_routes_non_ascii_to_clipboard(self, ctl, fake_gui, fake_clipboard):
        ctl.type_text("搜索天气")
        assert fake_clipboard.history[0] == "搜索天气" and fake_clipboard.content == "原剪贴板内容"

    def test_type_unicode_restores_clipboard(self, ctl, fake_gui, fake_clipboard):
        ctl.type_text_unicode("临时内容")
        assert fake_clipboard.content == "原剪贴板内容"

    def test_click_outside_screen_rejected(self, ctl, fake_gui):
        with pytest.raises(ValueError):
            ctl.click(5000, 5000)

    def test_click_negative_coordinate_rejected(self, ctl, fake_gui):
        with pytest.raises(ValueError):
            ctl.click(-1, 100)
        assert fake_gui.calls == []

    def test_coordinate_equal_to_width_rejected(self, ctl, fake_gui):
        with pytest.raises(ValueError):
            ctl.move_to(1920, 10)

    def test_scroll_position_out_of_screen_rejected(self, ctl, fake_gui):
        with pytest.raises(ValueError):
            ctl.scroll(-3, x=100, y=5000)

    def test_drag_end_out_of_screen_rejected_before_moving(self, ctl, fake_gui):
        with pytest.raises(ValueError):
            ctl.drag_to(10, 10, 3000, 10)
        assert fake_gui.calls == []


@pytest.mark.exception
class TestControlException:
    def test_failsafe_exception_propagates(self, ctl, fake_gui):
        def trip(**kw):
            raise fake_gui.FailSafeException("mouse moved to corner")
        fake_gui.click = trip
        with pytest.raises(fake_gui.FailSafeException):
            ctl.click(10, 10)

    def test_backend_error_on_drag_propagates(self, ctl, fake_gui):
        def boom(*a, **k):
            raise OSError("X connection broken")
        fake_gui.dragTo = boom
        with pytest.raises(OSError):
            ctl.drag_to(1, 1, 2, 2)

    def test_pyperclip_missing_raises_import_error(self, ctl, fake_gui, monkeypatch):
        import sys
        monkeypatch.setitem(sys.modules, "pyperclip", None)
        with pytest.raises(ImportError):
            ctl.type_text_unicode("中文")

    def test_clipboard_failure_does_not_paste(self, ctl, fake_gui, fake_clipboard):
        def fail(text):
            raise RuntimeError("clipboard locked")
        fake_clipboard.copy = fail
        with pytest.raises(RuntimeError):
            ctl.type_text_unicode("中文")
        assert "hotkey" not in fake_gui.names()

    def test_invalid_button_error_from_backend(self, ctl, fake_gui):
        def strict_click(x=None, y=None, button="left", clicks=1):
            if button not in ("left", "right", "middle"):
                raise ValueError(f"bad button {button}")
        fake_gui.click = strict_click
        with pytest.raises(ValueError):
            ctl.click(1, 1, button="side")

    def test_press_key_error_propagates(self, ctl, fake_gui):
        def bad(key):
            raise KeyError(key)
        fake_gui.press = bad
        with pytest.raises(KeyError):
            ctl.press_key("not_a_key")

    def test_type_text_non_ascii_not_silently_dropped(self, ctl, fake_gui, fake_clipboard):
        ctl.type_text("中文")
        assert ("hotkey", ("ctrl", "v"), {}) in fake_gui.calls

    def test_unknown_key_rejected(self, ctl, fake_gui):
        fake_gui.KEYBOARD_KEYS = ["enter", "ctrl", "a"]
        with pytest.raises(ValueError, match="未知按键名"):
            ctl.press_key("entr")
        assert fake_gui.calls == []

    def test_unknown_key_in_hotkey_rejected(self, ctl, fake_gui):
        fake_gui.KEYBOARD_KEYS = ["enter", "ctrl", "a"]
        with pytest.raises(ValueError):
            ctl.hotkey("ctrl", "shfit")

    def test_clipboard_read_failure_still_types(self, ctl, fake_gui, fake_clipboard):
        def broken_paste():
            raise RuntimeError("clipboard busy")
        fake_clipboard.paste = broken_paste
        ctl.type_text_unicode("中文")
        assert fake_clipboard.history == ["中文"] and fake_gui.names() == ["hotkey"]


@pytest.mark.multi_resolution
class TestControlMultiResolution:
    RES = [(1280, 720), (1366, 768), (1920, 1080), (2560, 1440), (3840, 2160), (1080, 1920)]

    @pytest.mark.parametrize("w,h", RES, ids=[f"{w}x{h}" for w, h in RES])
    def test_screen_size_detected(self, fake_gui, w, h):
        fake_gui.width, fake_gui.height = w, h
        c = MouseKeyboardController()
        assert (c.screen_width, c.screen_height) == (w, h)

    @pytest.mark.parametrize("w,h", RES, ids=[f"{w}x{h}" for w, h in RES])
    def test_center_click_coordinates(self, fake_gui, w, h):
        fake_gui.width, fake_gui.height = w, h
        c = MouseKeyboardController()
        c.click(c.screen_width // 2, c.screen_height // 2)
        assert (fake_gui.calls[-1][2]["x"], fake_gui.calls[-1][2]["y"]) == (w // 2, h // 2)

    @pytest.mark.parametrize("w,h", RES, ids=[f"{w}x{h}" for w, h in RES])
    def test_drag_diagonal_full_screen(self, fake_gui, w, h):
        fake_gui.width, fake_gui.height = w, h
        c = MouseKeyboardController()
        c.drag_to(10, 10, w - 10, h - 10)
        assert c.get_mouse_position() == (w - 10, h - 10)
