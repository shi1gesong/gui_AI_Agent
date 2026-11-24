"""第4周：扩展动作空间（open_app / close_app / hotkey / double_click / wait / done）的意图识别、校验与兜底"""
import pytest

from action_validator import (MAX_WAIT_SEC, VALID_ACTION_TYPES, SubtaskIntent, fallback_action, needs_screen,
                              parse_hotkey, parse_wait_seconds, validate_action)

ELEMS = [{"text": "张三", "bbox": [20, 80, 200, 120]}, {"text": "消息内容", "bbox": [240, 500, 680, 540]},
         {"text": "发送", "bbox": [700, 500, 780, 540]}, {"text": "报告.docx", "bbox": [50, 300, 150, 330]}]


def A(t, target="", text="", **kw):
    return {"action_type": t, "target_text": target, "input_text": text, **kw}


@pytest.mark.functional
class TestActionSpaceFunctional:
    def test_new_types_registered(self):
        for t in ("double_click", "hotkey", "open_app", "close_app", "wait", "done"):
            assert t in VALID_ACTION_TYPES

    @pytest.mark.parametrize("spec,keys", [("ctrl+l", ["ctrl", "l"]), ("Ctrl + Shift + Esc", ["ctrl", "shift", "esc"]),
                                           ("alt+f4", ["alt", "f4"]), ("win+d", ["win", "d"]),
                                           (["ctrl", "v"], ["ctrl", "v"]), ("ctrl+回车", ["ctrl", "enter"])])
    def test_parse_hotkey(self, spec, keys):
        assert parse_hotkey(spec) == keys

    @pytest.mark.parametrize("v,sec", [("2", 2.0), ("2秒", 2.0), ("0.5", 0.5), ("500ms", 0.5), (3, 3.0), ("", 1.0),
                                       (None, 1.0)])
    def test_parse_wait(self, v, sec):
        assert parse_wait_seconds(v) == sec

    @pytest.mark.parametrize("subtask,expected", [
        ("打开浏览器", ("open_app", "浏览器")), ("打开记事本", ("open_app", "记事本")),
        ("打开模拟聊天", ("open_app", "模拟聊天")), ("打开文件“C:\\a\\b.txt”", ("open_app", "C:\\a\\b.txt")),
        ("关闭计算器", ("close_app", "计算器")), ("关闭模拟聊天窗口", ("close_app", "模拟聊天")),
        ("等待2秒", ("wait", "")), ("按 ctrl+l 聚焦地址栏", ("hotkey", "")), ("按回车", ("key", "")),
        ("点击联系人“张三”", ("click", "张三")), ("双击“报告.docx”", ("double_click", "报告.docx")),
        ("在“消息内容”框中输入“今晚开会”", ("type", "消息内容")), ("直接输入“天气预报”并按回车", ("type", "")),
        ("任务完成", ("done", "")),
    ])
    def test_fallback_for_standard_subtasks(self, subtask, expected):
        fb = fallback_action(subtask, ELEMS)
        assert (fb["action_type"], fb["target_text"]) == expected

    def test_type_text_after_input_keyword(self):
        """引号里的“消息内容”是输入框，不是要输入的内容"""
        i = SubtaskIntent("在“消息内容”框中输入“今晚7点开会”")
        assert i.expected_text == "今晚7点开会" and i.field == "消息内容"

    def test_direct_type_has_enter_follow_up(self):
        fb = fallback_action("直接输入“天气预报”并按回车", ELEMS)
        assert fb["input_text"] == "天气预报" and fb["follow_up"][0]["input_text"] == "enter"

    def test_hotkey_fallback_text(self):
        assert fallback_action("按 ctrl+l 聚焦地址栏", [])["input_text"] == "ctrl+l"

    def test_wait_fallback_seconds(self):
        assert fallback_action("等待3秒", [])["input_text"] == "3"

    @pytest.mark.parametrize("subtask,action", [
        ("打开浏览器", A("open_app", "浏览器")), ("打开浏览器", A("open_app", "Microsoft Edge")),
        ("打开记事本", A("open_app", "notepad")), ("关闭计算器", A("close_app", "计算器")),
        ("关闭计算器", A("hotkey", "", "alt+f4")), ("等待2秒", A("wait", "", "2")),
        ("按 ctrl+l 聚焦地址栏", A("hotkey", "", "ctrl+l")), ("双击“报告.docx”", A("double_click", "报告.docx")),
        ("点击“发送”", A("click", "发送")), ("打开记事本", A("click", "记事本")),
    ])
    def test_valid_new_actions_accepted(self, subtask, action):
        elems = ELEMS + [{"text": "记事本", "bbox": [0, 0, 10, 10]}]
        assert validate_action(subtask, action, elems) == []

    def test_needs_screen(self):
        assert needs_screen(A("click", "发送")) and needs_screen(A("double_click", "x"))
        assert needs_screen(A("type", "消息内容", "hi"))
        assert not any(needs_screen(A(t)) for t in ("open_app", "close_app", "wait", "hotkey", "key", "done"))
        assert not needs_screen(A("type", "", "hi"))

    def test_open_file_intent(self):
        assert SubtaskIntent("打开文件“D:\\资料\\周报.docx”").open_is_file

    def test_open_app_not_file(self):
        assert not SubtaskIntent("打开计算器").open_is_file


@pytest.mark.boundary
class TestActionSpaceBoundary:
    def test_wait_capped(self):
        assert parse_wait_seconds("999") == MAX_WAIT_SEC

    def test_wait_zero_allowed(self):
        assert parse_wait_seconds("0") == 0.0

    def test_wait_intent_default_one_second(self):
        assert SubtaskIntent("等待页面加载").wait_seconds == 1.0

    def test_wait_intent_ms(self):
        assert SubtaskIntent("等待500毫秒").wait_seconds == 0.5

    def test_hotkey_needs_modifier(self):
        assert parse_hotkey("a+b") is None

    def test_single_key_not_hotkey(self):
        assert parse_hotkey("enter") is None

    def test_win_key_alone_is_key_intent(self):
        i = SubtaskIntent("按 win 键打开开始菜单")
        assert i.key == "win" and i.hotkey is None

    def test_win_combo_is_hotkey(self):
        assert SubtaskIntent("按 win+d 显示桌面").hotkey == ["win", "d"]

    def test_close_current_window(self):
        assert SubtaskIntent("关闭当前窗口").close_target == "当前窗口"

    def test_done_not_allowed_with_target(self):
        r = validate_action("点击“发送”", A("done"), ELEMS)
        assert r and "done" in r[0]

    def test_done_allowed_for_generic(self):
        assert validate_action("确认页面已打开", A("done"), ELEMS) == []

    def test_type_into_missing_field_fails_instead_of_typing_elsewhere(self):
        """指定的输入框不在屏幕上：不去点无关的“栏/框”，也不偷偷输入到当前焦点，而是报错（会重新截图重试）"""
        from executor import ControllerExecutor
        from fakes import RecordingController
        elems = [{"text": "侧边栏", "bbox": [0, 0, 10, 10]}]
        fb = fallback_action("在“消息内容”框中输入“你好”", elems)
        assert fb["target_text"] == "消息内容" and fb["input_text"] == "你好" and needs_screen(fb)
        ctl = RecordingController()
        r = ControllerExecutor(ctl, sleep=lambda s: None)(fb, elems)
        assert r["ok"] is False and "找不到输入框" in r["error"] and ctl.calls == []

    def test_double_click_target_not_on_screen_no_fallback(self):
        assert fallback_action("双击“不存在.txt”", ELEMS) is None

    def test_click_contact_not_on_screen_no_fallback(self):
        assert fallback_action("点击联系人“李四”", ELEMS) is None

    def test_open_with_quote(self):
        assert SubtaskIntent("打开“画图”").open_target == "画图"


@pytest.mark.exception
class TestActionSpaceException:
    @pytest.mark.parametrize("v", ["abc", "-1", -2, "2分钟"])
    def test_wait_invalid(self, v):
        assert parse_wait_seconds(v) is None

    @pytest.mark.parametrize("spec", ["", "ctrl+", "hyper+x", "ctrl+不存在"])
    def test_hotkey_invalid(self, spec):
        assert parse_hotkey(spec) is None

    def test_open_app_wrong_target_rejected(self):
        r = validate_action("打开浏览器", A("open_app", "记事本"), ELEMS)
        assert r and "实际打开" in r[0]

    def test_open_app_missing_target(self):
        assert "缺少 target_text" in validate_action("打开程序", A("open_app"), ELEMS)[0]

    def test_close_wrong_app_rejected(self):
        r = validate_action("关闭计算器", A("close_app", "浏览器"), ELEMS)
        assert r and "实际关闭" in r[0]

    def test_close_with_wrong_hotkey_rejected(self):
        r = validate_action("关闭计算器", A("hotkey", "", "ctrl+c"), ELEMS)
        assert r and "alt+f4" in r[0]

    def test_close_as_click_rejected(self):
        r = validate_action("关闭计算器", A("click", "张三"), ELEMS)
        assert r and "close_app" in r[0]

    def test_hotkey_mismatch_rejected(self):
        r = validate_action("按 ctrl+l 聚焦地址栏", A("hotkey", "", "ctrl+t"), ELEMS)
        assert r and "应按 ctrl+l" in r[0]

    def test_hotkey_task_as_key_rejected(self):
        r = validate_action("按 ctrl+l 聚焦地址栏", A("key", "", "enter"), ELEMS)
        assert r and "hotkey" in r[0]

    def test_wait_task_as_click_rejected(self):
        assert "wait" in validate_action("等待2秒", A("click", "发送"), ELEMS)[0]

    def test_wait_bad_seconds_rejected(self):
        assert "无法解析" in validate_action("等待", A("wait", "", "很久"), ELEMS)[0]

    def test_double_click_task_as_click_rejected(self):
        assert "double_click" in validate_action("双击“报告.docx”", A("click", "报告.docx"), ELEMS)[0]

    def test_double_click_missing_target(self):
        assert "缺少 target_text" in validate_action("双击图标", A("double_click"), ELEMS)[0]

    def test_open_as_type_rejected(self):
        assert "open_app" in validate_action("打开记事本", A("type", "", "记事本"), ELEMS)[0]

    def test_open_with_wrong_key_rejected(self):
        r = validate_action("打开记事本", A("key", "", "enter"), ELEMS)
        assert r and "无法打开" in r[0]

    def test_unknown_type_lists_all_types(self):
        msg = validate_action("点击", A("hover", "x"), ELEMS)[0]
        assert "open_app" in msg and "done" in msg


@pytest.mark.multi_resolution
class TestActionSpaceMultiResolution:
    @pytest.mark.parametrize("w,h", [(1366, 768), (1920, 1080), (2560, 1440), (3840, 2160)])
    def test_fallback_independent_of_resolution(self, w, h):
        elems = [{"text": "发送", "bbox": [w - 100, h - 60, w - 20, h - 20]}]
        fb = fallback_action("点击“发送”", elems)
        assert fb["action_type"] == "click" and fb["target_text"] == "发送"


@pytest.mark.functional
class TestAddressBarFallback:
    def test_address_bar_not_visible_uses_ctrl_l(self):
        fb = fallback_action("在浏览器的地址栏输入“天气预报”并按回车", [{"text": "必应", "bbox": [0, 0, 10, 10]}])
        assert fb["action_type"] == "hotkey" and fb["input_text"] == "ctrl+l"
        assert [f["action_type"] for f in fb["follow_up"]] == ["type", "key"]
        assert fb["follow_up"][0]["input_text"] == "天气预报" and not needs_screen(fb)

    def test_address_bar_visible_clicks_it(self):
        fb = fallback_action("在地址栏输入“天气预报”", [{"text": "地址栏", "bbox": [0, 0, 600, 30]}])
        assert fb["action_type"] == "type" and fb["target_text"] == "地址栏"
