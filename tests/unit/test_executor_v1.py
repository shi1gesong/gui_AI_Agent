"""第4周：agent/executor.py 新动作（double_click / hotkey / open_app / close_app / wait / done）与 DryRunExecutor"""
import pytest

from control.app_manager import AppManager
from executor import ControllerExecutor, DryRunExecutor
from fakes import FakeLauncher, FakeWindowProvider, RecordingController

ELEMS = [{"text": "张三", "bbox": [20, 80, 200, 120]}, {"text": "消息内容", "bbox": [240, 500, 680, 540]},
         {"text": "发送", "bbox": [700, 500, 780, 540]}]


def A(t, target="", text="", **kw):
    return {"action_type": t, "target_text": target, "input_text": text, **kw}


@pytest.fixture
def ex():
    ctl = RecordingController()
    slept = []
    wins = FakeWindowProvider(launch_map={"calc.exe": "计算器"})
    launcher = FakeLauncher(wins)
    apps = AppManager(spawn=launcher.spawn, startfile=launcher.startfile, windows=wins, sleep=lambda s: None,
                      search_paths=[], open_timeout=0.5)
    return ControllerExecutor(ctl, app_manager=apps, sleep=slept.append), ctl, slept, wins


@pytest.mark.functional
class TestExecutorV1Functional:
    def test_double_click(self, ex):
        e, ctl, *_ = ex
        assert e(A("double_click", "张三"), ELEMS)["ok"] and ctl.calls == [("double_click", 110, 100)]

    def test_hotkey(self, ex):
        e, ctl, *_ = ex
        assert e(A("hotkey", "", "ctrl+l"), ELEMS)["ok"] and ctl.calls == [("hotkey", "ctrl", "l")]

    def test_hotkey_in_target(self, ex):
        e, ctl, *_ = ex
        e(A("hotkey", "alt+f4"), ELEMS)
        assert ctl.calls == [("hotkey", "alt", "f4")]

    def test_wait(self, ex):
        e, ctl, slept, _ = ex
        assert e(A("wait", "", "2"), ELEMS)["ok"] and slept == [2.0] and ctl.calls == []

    def test_done_noop(self, ex):
        e, ctl, slept, _ = ex
        assert e(A("done"), ELEMS)["ok"] and ctl.calls == [] and slept == []

    def test_open_app(self, ex):
        e, _, _, wins = ex
        r = e(A("open_app", "计算器"), ELEMS)
        assert r["ok"] and r["detail"]["open"]["window"] == "计算器"

    def test_close_app(self, ex):
        e, _, _, wins = ex
        wins.add("计算器")
        assert e(A("close_app", "计算器"), ELEMS)["ok"] and wins.titles() == []

    def test_type_into_field_clicks_center_first(self, ex):
        e, ctl, *_ = ex
        e(A("type", "消息内容", "今晚开会"), ELEMS)
        assert ctl.calls == [("click", 460, 520), ("type_text", "今晚开会")]

    def test_type_without_field_types_to_focus(self, ex):
        e, ctl, *_ = ex
        e(A("type", "", "天气预报", follow_up=[A("key", "", "enter")]), ELEMS)
        assert ctl.calls == [("type_text", "天气预报"), ("press_key", "enter")]

    def test_grounding_detail_recorded(self, ex):
        e, *_ = ex
        d = e(A("click", "发送按钮"), ELEMS)["detail"]["grounding"][0]
        assert d["matched"] == "发送" and d["point"] == [740, 520] and d["score"] < 1

    def test_settle_sleep_after_action(self):
        ctl, slept = RecordingController(), []
        ControllerExecutor(ctl, sleep=slept.append, settle_sec=0.3)(A("click", "发送"), ELEMS)
        assert slept == [0.3]

    def test_dry_run_records_without_executing(self):
        d = DryRunExecutor()
        r = d(A("click", "发送"), ELEMS)
        assert r["ok"] and r["detail"]["point"] == [740, 520] and d.log == [A("click", "发送")]

    def test_dry_run_non_click(self):
        assert DryRunExecutor()(A("open_app", "浏览器"), [])["ok"]

    def test_open_app_without_manager_win_search(self):
        ctl = RecordingController()
        ControllerExecutor(ctl, sleep=lambda s: None)(A("open_app", "记事本"), [])
        assert ctl.calls == [("press_key", "win"), ("type_text", "记事本"), ("press_key", "enter")]

    def test_close_app_without_manager_alt_f4(self):
        ctl = RecordingController()
        ControllerExecutor(ctl)(A("close_app", "记事本"), [])
        assert ctl.calls == [("hotkey", "alt", "f4")]


@pytest.mark.boundary
class TestExecutorV1Boundary:
    def test_wait_capped_30s(self, ex):
        e, _, slept, _ = ex
        e(A("wait", "", "600"), ELEMS)
        assert slept == [30.0]

    def test_wait_default_one_second(self, ex):
        e, _, slept, _ = ex
        e(A("wait"), ELEMS)
        assert slept == [1.0]

    def test_wait_seconds_field(self, ex):
        e, _, slept, _ = ex
        e(A("wait", seconds=0.5), ELEMS)
        assert slept == [0.5]

    def test_elements_none(self, ex):
        e, ctl, *_ = ex
        assert e(A("hotkey", "", "ctrl+c"), None)["ok"]

    def test_dry_run_click_missing_target(self):
        assert DryRunExecutor()(A("click", "不存在"), ELEMS)["ok"] is False


@pytest.mark.exception
class TestExecutorV1Exception:
    def test_bad_hotkey(self, ex):
        r = ex[0](A("hotkey", "", "hyper+x"), ELEMS)
        assert r["ok"] is False and "组合键" in r["error"]

    def test_bad_wait(self, ex):
        assert ex[0](A("wait", "", "很久"), ELEMS)["ok"] is False

    def test_double_click_missing(self, ex):
        r = ex[0](A("double_click", "不存在"), ELEMS)
        assert r["ok"] is False and "LookupError" in r["error"]

    def test_open_app_failure(self, ex):
        r = ex[0](A("open_app", "C:\\不存在.txt"), ELEMS)
        assert r["ok"] is False and "文件不存在" in r["error"] and r["detail"]["open"]["ok"] is False

    def test_close_app_failure(self, ex):
        r = ex[0](A("close_app", "计算器"), ELEMS)
        assert r["ok"] is False and "没有找到" in r["error"]

    def test_non_dict_action(self, ex):
        assert ex[0]("click", ELEMS)["ok"] is False

    def test_follow_up_failure_reported(self, ex):
        r = ex[0](A("type", "", "x", follow_up=[A("hotkey", "", "bad")]), ELEMS)
        assert r["ok"] is False

    def test_controller_error_in_double_click(self):
        ctl = RecordingController(fail_on={"double_click": ValueError("坐标超出屏幕范围")})
        r = ControllerExecutor(ctl)(A("double_click", "张三"), ELEMS)
        assert r["ok"] is False and "坐标超出" in r["error"]

    def test_dry_run_invalid(self):
        assert DryRunExecutor()({"_invalid": True}, ELEMS)["ok"] is False


@pytest.mark.multi_resolution
class TestExecutorV1MultiResolution:
    @pytest.mark.parametrize("w,h", [(1366, 768), (1920, 1080), (2560, 1440), (3840, 2160)])
    def test_click_bottom_right_button(self, w, h):
        ctl = RecordingController()
        elems = [{"text": "发送", "bbox": [w - 100, h - 60, w - 20, h - 20]}]
        ControllerExecutor(ctl)(A("click", "发送"), elems)
        assert ctl.calls == [("click", w - 60, h - 40)]
