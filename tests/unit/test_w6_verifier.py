"""第6周：agent/verifier.py 动作效果校验（截图前后对比）与目标后置条件"""
import pytest
from PIL import Image, ImageDraw

from agent.verifier import (EffectVerifier, Postcondition, cells_to_bbox, changed_cells, check_postconditions,
                            image_diff_ratio, text_matches)
from control.app_manager import AppManager
from fakes import FakeWindowProvider


def blank(size=(640, 360), color="white"):
    return Image.new("RGB", size, color)


def with_box(img, box, color="black"):
    im = img.copy()
    ImageDraw.Draw(im).rectangle(box, fill=color)
    return im


@pytest.mark.functional
class TestEffect:
    def test_identical_frames_zero_diff(self):
        assert image_diff_ratio(blank(), blank()) == 0.0

    def test_small_change_detected(self):
        a = blank()
        b = with_box(a, [100, 100, 130, 120])
        assert image_diff_ratio(a, b) > 0.002

    def test_color_only_change_detected(self):
        a = with_box(blank(), [0, 0, 200, 100], (144, 176, 128))
        b = with_box(blank(), [0, 0, 200, 100], (196, 84, 92))   # 灰度几乎相同、颜色不同
        assert image_diff_ratio(a, b) > 0.05

    @pytest.mark.parametrize("a_type", ["click", "double_click", "type", "drag"])
    def test_required_types_fail_without_change(self, a_type):
        r = EffectVerifier().verify({"action_type": a_type}, blank(), blank())
        assert r["ok"] is False and "没有变化" in r["reason"] and r["checked"]

    @pytest.mark.parametrize("a_type", ["hotkey", "key", "scroll", "wait", "open_app", "close_app", "done"])
    def test_optional_types_never_fail(self, a_type):
        r = EffectVerifier().verify({"action_type": a_type}, blank(), blank())
        assert r["ok"] is True and not r["checked"]

    def test_change_passes_and_reports_bbox(self):
        a = blank()
        b = with_box(a, [400, 200, 480, 260])
        r = EffectVerifier().verify({"action_type": "click"}, a, b)
        x1, y1, x2, y2 = r["changed_bbox"]
        assert r["ok"] and x1 <= 400 and y1 <= 200 and x2 >= 480 and y2 >= 260

    def test_changed_cells_location(self):
        a = blank((160, 90))
        b = with_box(a, [0, 0, 9, 9])
        assert changed_cells(a, b, grid=(16, 9)) == [(0, 0)]

    def test_cells_to_bbox(self):
        assert cells_to_bbox([(1, 1), (2, 3)], (160, 90), (16, 9)) == [10, 10, 30, 40]

    @pytest.mark.parametrize("w,h", [(1366, 768), (1920, 1080), (2560, 1440), (3840, 2160)])
    def test_multi_resolution_frames(self, w, h):
        a = blank((w, h))
        b = with_box(a, [w // 2, h // 2, w // 2 + w // 40, h // 2 + h // 30])
        assert EffectVerifier().verify({"action_type": "click"}, a, b)["ok"]


@pytest.mark.functional
class TestPostconditions:
    def apps(self, *titles, active=None):
        wp = FakeWindowProvider(titles)
        if active is not None:
            wp.active = wp.wins[active]
        return AppManager(windows=wp, sleep=lambda s: None)

    def test_window_exists(self):
        ok, _ = Postcondition("window_exists", {"keywords": ["天气预报"]}).check(self.apps("天气预报 - 搜索"))
        assert ok

    def test_window_absent(self):
        ok, _ = Postcondition("window_absent", {"keywords": ["计算器"]}).check(self.apps("计算器"))
        assert not ok

    def test_active_window(self):
        apps = self.apps("GUI Agent 模拟聊天", "文档", active=0)
        assert Postcondition("active_window", {"keywords": ["模拟聊天"]}).check(apps)[0]

    def test_text_present(self):
        ok, _ = Postcondition("text_present", {"text": "张三：你好"}).check(None, lambda: ["我 → 张三：你好"])
        assert ok

    def test_text_present_tolerates_punct(self):
        assert Postcondition("text_present", {"text": "张三：你好"}).check(None, lambda: ["我→张三:你好"])[0]

    def test_text_absent(self):
        assert Postcondition("text_absent", {"text": "错误"}).check(None, lambda: ["一切正常"])[0]

    def test_file_contains(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("第6周周报", encoding="utf-8")
        assert Postcondition("file_contains", {"path": str(f), "text": "周报"}).check()[0]

    def test_check_all_required(self, tmp_path):
        conds = [Postcondition("window_exists", {"keywords": ["x"]}),
                 Postcondition("window_exists", {"keywords": ["y"]}, required=False)]
        r = check_postconditions(conds, self.apps("x"))
        assert r["ok"] and [x["ok"] for x in r["results"]] == [True, False]

    def test_text_perceived_once(self):
        calls = []

        def texts():
            calls.append(1)
            return ["我 → 张三：好"]
        conds = [Postcondition("text_present", {"text": "张三：好"}), Postcondition("text_absent", {"text": "错误"})]
        assert check_postconditions(conds, None, texts)["ok"] and len(calls) == 1

    def test_describe_and_needs_screen(self):
        c = Postcondition("text_present", {"text": "x"})
        assert c.needs_screen and "x" in c.describe()
        assert not Postcondition("window_exists", {"keywords": []}).needs_screen

    @pytest.mark.parametrize("want,text,ok", [
        ("张三：明天上午10点开会", "我 → 张三：明天上午1O点开会", True),     # OCR 把 0 认成 O
        ("张三：好", "我 → 张王：好", False),                               # 短文本不容错
        ("张三：明天上午10点开会", "当前聊天：张三", False),
    ])
    def test_text_matches_fuzzy(self, want, text, ok):
        assert text_matches(want, text) is ok


@pytest.mark.boundary
class TestVerifierBoundary:
    def test_missing_frames_skip(self):
        r = EffectVerifier().verify({"action_type": "click"}, None, blank())
        assert r["ok"] and not r["checked"]

    def test_diff_with_none_is_full(self):
        assert image_diff_ratio(None, blank()) == 1.0

    def test_different_sizes_compared(self):
        assert image_diff_ratio(blank((640, 360)), blank((1280, 720))) == 0.0

    def test_threshold_ignores_noise(self):
        a = blank()
        b = with_box(a, [0, 0, 639, 359], (250, 250, 250))   # 全屏轻微变化（抗锯齿/压缩噪声）
        assert image_diff_ratio(a, b) == 0.0

    def test_changed_cells_none(self):
        assert changed_cells(None, blank()) == [] and cells_to_bbox([], (10, 10)) is None

    def test_no_window_info_skips(self):
        assert Postcondition("window_exists", {"keywords": ["x"]}).check(None)[0]

    def test_no_texts_fn_skips(self):
        assert Postcondition("text_present", {"text": "x"}).check(None, None)[0]

    def test_empty_conditions_ok(self):
        assert check_postconditions([], None)["ok"]


@pytest.mark.exception
class TestVerifierException:
    def test_unknown_kind(self):
        with pytest.raises(ValueError):
            Postcondition("pixel_perfect", {})

    def test_file_missing(self, tmp_path):
        ok, msg = Postcondition("file_contains", {"path": str(tmp_path / "no.txt"), "text": "x"}).check()
        assert not ok and "不存在" in msg

    def test_texts_fn_raises(self):
        def boom():
            raise OSError("截图失败")
        ok, msg = Postcondition("text_present", {"text": "x"}).check(None, boom)
        assert not ok and "感知失败" in msg

    def test_text_matches_empty(self):
        assert not text_matches("", "abc") and not text_matches("abc", "")
