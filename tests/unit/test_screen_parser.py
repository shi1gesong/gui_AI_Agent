"""第4周：perception/screen_parser.py（截图 → OCR → 过滤/去重/排序 → DPI 坐标换算）—— mock 用例"""
import json

import pytest
from PIL import Image

from perception.screen_parser import Observation, ScreenPerceiver, StaticPerceiver


class FakeCapture:
    def __init__(self, tmp_path, size=(1920, 1080)):
        self.save_dir = tmp_path
        self.size = size
        self.calls = []

    def capture_full_screen(self, monitor_index=1, save=True):
        self.calls.append((monitor_index, save))
        return Image.new("RGB", self.size, "white")


class FakeOCR:
    def __init__(self, results):
        self.results = results
        self.paths = []

    def detect(self, path):
        self.paths.append(path)
        return self.results


def R(text, bbox, conf=0.9):
    return {"text": text, "bbox": bbox, "confidence": conf}


@pytest.fixture
def perceiver(tmp_path):
    def make(results, img=(1920, 1080), logical=(1920, 1080), **kw):
        return ScreenPerceiver(capture=FakeCapture(tmp_path, img), ocr=FakeOCR(results),
                               logical_size_fn=lambda: logical, **kw)
    return make


@pytest.mark.functional
class TestPerceiverFunctional:
    def test_perceive_returns_elements(self, perceiver):
        obs = perceiver([R("发送", [10, 10, 50, 30])]).perceive()
        assert obs.texts() == ["发送"] and obs.elements[0]["bbox"] == [10, 10, 50, 30]

    def test_screenshot_saved_and_passed_to_ocr(self, perceiver, tmp_path):
        p = perceiver([R("a", [0, 0, 5, 5])])
        obs = p.perceive()
        from pathlib import Path
        assert obs.screenshot_path == p.ocr.paths[0] and Path(obs.screenshot_path).exists()
        assert Path(obs.screenshot_path).parent == tmp_path

    def test_capture_not_saved_twice(self, perceiver):
        p = perceiver([])
        p.perceive()
        assert p.capture.calls == [(1, False)]

    def test_context_format_compatible_with_week3(self, perceiver):
        from action_validator import parse_screen_elements
        obs = perceiver([R("发送", [10, 10, 50, 30]), R("张三", [0, 100, 40, 120])]).perceive()
        ctx = obs.to_context()
        assert ctx.startswith("识别到的UI元素: ")
        assert {e["text"] for e in parse_screen_elements(ctx)} == {"发送", "张三"}

    def test_sorted_top_to_bottom_left_to_right(self, perceiver):
        obs = perceiver([R("下", [0, 500, 10, 520]), R("右", [300, 10, 320, 30]), R("左", [0, 12, 20, 32])]).perceive()
        assert obs.texts() == ["左", "右", "下"]

    def test_bbox_px_kept(self, perceiver):
        obs = perceiver([R("a", [100, 100, 200, 200])], img=(3840, 2160), logical=(1920, 1080)).perceive()
        assert obs.elements[0]["bbox_px"] == [100, 100, 200, 200] and obs.elements[0]["bbox"] == [50, 50, 100, 100]

    def test_summary_fields(self, perceiver):
        s = perceiver([R("a", [0, 0, 5, 5])]).perceive().summary()
        assert set(s) >= {"screenshot", "elements", "raw", "image_size", "logical_size", "scale", "sec"}

    def test_static_perceiver(self):
        sp = StaticPerceiver([{"text": "确定", "bbox": [0, 0, 10, 10]}])
        assert sp.perceive().texts() == ["确定"] and sp.calls == 1

    def test_static_perceiver_frames(self):
        sp = StaticPerceiver(frames=[[{"text": "加载中"}], [{"text": "完成"}]])
        assert sp.perceive().texts() == ["加载中"]
        assert sp.perceive().texts() == ["完成"]
        assert sp.perceive().texts() == ["完成"]

    def test_raw_count(self, perceiver):
        obs = perceiver([R("a", [0, 0, 5, 5]), R("", [0, 0, 5, 5])]).perceive()
        assert obs.raw_count == 2 and len(obs.elements) == 1


@pytest.mark.boundary
class TestPerceiverBoundary:
    def test_low_confidence_filtered(self, perceiver):
        obs = perceiver([R("噪声", [0, 0, 5, 5], 0.1), R("发送", [0, 10, 5, 15], 0.9)]).perceive()
        assert obs.texts() == ["发送"]

    def test_confidence_threshold_inclusive(self, perceiver):
        assert perceiver([R("a", [0, 0, 5, 5], 0.3)], min_confidence=0.3).perceive().texts() == ["a"]

    @pytest.mark.parametrize("text", ["", "   ", "。", "·"])
    def test_empty_or_punct_text_filtered(self, perceiver, text):
        assert perceiver([R(text, [0, 0, 5, 5])]).perceive().elements == []

    def test_duplicate_overlapping_removed(self, perceiver):
        obs = perceiver([R("发送", [0, 0, 50, 20], 0.8), R("发送", [1, 1, 51, 21], 0.95)]).perceive()
        assert len(obs.elements) == 1 and obs.elements[0]["confidence"] == 0.95

    def test_same_text_different_place_kept(self, perceiver):
        obs = perceiver([R("确定", [0, 0, 50, 20]), R("确定", [500, 500, 550, 520])]).perceive()
        assert len(obs.elements) == 2

    def test_max_elements(self, perceiver):
        raw = [R(f"e{i}", [0, i * 30, 10, i * 30 + 20]) for i in range(50)]
        assert len(perceiver(raw, max_elements=10).perceive().elements) == 10

    def test_context_truncation_note(self):
        obs = Observation(elements=[{"text": f"e{i}", "bbox": [0, 0, 1, 1]} for i in range(100)])
        ctx = obs.to_context(max_elements=5)
        assert "另有 95 个元素未列出" in ctx
        assert len(json.loads(ctx.split(": ", 1)[1].split("\n")[0])) == 5

    def test_no_ocr_results(self, perceiver):
        obs = perceiver([]).perceive()
        assert obs.elements == [] and obs.to_context() == "识别到的UI元素: []"


@pytest.mark.exception
class TestPerceiverException:
    def test_missing_components(self):
        with pytest.raises(RuntimeError):
            ScreenPerceiver().perceive()

    @pytest.mark.parametrize("bbox", [None, [1, 2], ["x", 0, 1, 1], [10, 10, 5, 5], [0, 0, 0, 0]])
    def test_invalid_bbox_dropped(self, perceiver, bbox):
        assert perceiver([R("a", bbox)]).perceive().elements == []

    def test_non_dict_results_ignored(self, perceiver):
        assert perceiver(["a", None, R("b", [0, 0, 5, 5])]).perceive().texts() == ["b"]

    def test_ocr_exception_propagates(self, tmp_path):
        class Boom:
            def detect(self, p):
                raise RuntimeError("OCR 模型加载失败")
        p = ScreenPerceiver(capture=FakeCapture(tmp_path), ocr=Boom(), logical_size_fn=lambda: (1920, 1080))
        with pytest.raises(RuntimeError, match="OCR"):
            p.perceive()

    @pytest.mark.parametrize("img,logical", [((0, 1080), (1920, 1080)), ((1920, 1080), (0, 0))])
    def test_compute_scale_invalid(self, img, logical):
        with pytest.raises(ValueError):
            ScreenPerceiver.compute_scale(img, logical)

    def test_missing_confidence_defaults_to_one(self, perceiver):
        assert perceiver([{"text": "a", "bbox": [0, 0, 5, 5]}]).perceive().texts() == ["a"]


@pytest.mark.multi_resolution
class TestPerceiverMultiResolution:
    @pytest.mark.parametrize("logical,dpi", [((2560, 1440), 1.0), ((1920, 1080), 1.25), ((1707, 960), 1.5),
                                             ((1463, 823), 1.75), ((1280, 720), 2.0)])
    def test_bbox_converted_to_logical(self, perceiver, logical, dpi):
        img = (round(logical[0] * dpi), round(logical[1] * dpi))
        x, y = 800, 600  # 逻辑坐标中的按钮中心
        raw = [R("发送", [x * dpi - 20 * dpi, y * dpi - 10 * dpi, x * dpi + 20 * dpi, y * dpi + 10 * dpi])]
        obs = perceiver(raw, img=img, logical=logical).perceive()
        b = obs.elements[0]["bbox"]
        assert abs((b[0] + b[2]) / 2 - x) <= 1 and abs((b[1] + b[3]) / 2 - y) <= 1
        assert abs(obs.scale[0] - dpi) < 0.01

    @pytest.mark.parametrize("size", [(1366, 768), (1920, 1080), (2560, 1440), (3840, 2160)])
    def test_dpi_100_identity(self, perceiver, size):
        obs = perceiver([R("a", [10, 20, 30, 40])], img=size, logical=size).perceive()
        assert obs.scale == (1.0, 1.0) and obs.elements[0]["bbox"] == [10, 20, 30, 40]

    def test_second_monitor_offset(self, tmp_path):
        p = ScreenPerceiver(capture=FakeCapture(tmp_path), ocr=FakeOCR([R("a", [10, 10, 20, 20])]),
                            logical_size_fn=lambda: (1920, 1080), offset=(2560, 0), monitor_index=2)
        obs = p.perceive()
        assert obs.elements[0]["bbox"] == [2570, 10, 2580, 20] and p.capture.calls[0][0] == 2

    def test_user_machine_2560x1440_dpi100(self, perceiver):
        """用户真机：2560x1440，DPI 100%（2026-11-24 real_device 用例实测比例 1.00）"""
        obs = perceiver([R("发送", [2400, 1300, 2500, 1340])], img=(2560, 1440), logical=(2560, 1440)).perceive()
        assert obs.elements[0]["bbox"] == [2400, 1300, 2500, 1340]


@pytest.mark.boundary
class TestRegionFilter:
    E = [{"text": "张三", "bbox": [110, 160, 300, 200]}, {"text": "[规划] 6 个子任务", "bbox": [1010, 10, 1890, 40]}]

    def test_include_foreground_window(self):
        from perception.screen_parser import filter_by_region
        assert [e["text"] for e in filter_by_region(self.E, [100, 100, 920, 660])] == ["张三"]

    def test_exclude_console(self):
        from perception.screen_parser import filter_by_region
        assert [e["text"] for e in filter_by_region(self.E, None, [[1000, 0, 1900, 1000]])] == ["张三"]

    def test_invalid_rects_ignored(self):
        from perception.screen_parser import filter_by_region
        assert len(filter_by_region(self.E, [0, 0, 0, 0], [None, [5, 5, 1, 1]])) == 2

    def test_perceiver_applies_region(self, perceiver):
        p = perceiver([R("张三", [110, 160, 300, 200]), R("控制台文字", [1010, 10, 1890, 40])])
        p.region_fn = lambda: {"include": None, "exclude": [[1000, 0, 1900, 1000]]}
        obs = p.perceive()
        assert obs.texts() == ["张三"] and obs.extra["filtered_out"] == 1 and obs.summary()["filtered_out"] == 1

    def test_region_fn_error_keeps_all(self, perceiver):
        p = perceiver([R("张三", [110, 160, 300, 200])])
        p.region_fn = lambda: 1 / 0
        obs = p.perceive()
        assert obs.texts() == ["张三"] and "ZeroDivisionError" in obs.extra["region_error"]
