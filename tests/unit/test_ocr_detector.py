"""第2周 感知模块：perception/ocr_detector.py（EasyOCR / PaddleOCR 统一封装）—— mock 用例

EasyOCR/PaddleOCR 依赖体积大（含模型下载），这里用替身模块模拟两种引擎的真实返回格式：
- EasyOCR.readtext -> [(四点坐标, 文本, 置信度), ...]
- PaddleOCR 3.x predict -> [ {rec_texts, rec_scores, rec_boxes}, ... ]（按页）
真实引擎用例见 tests/real/test_real_ocr.py（ocr_real 标记）。
"""
import os
import sys
import types

import pytest
from PIL import Image

import perception.ocr_detector as ocr_mod
from perception.ocr_detector import OCRDetector, benchmark_engines, find_latest_screenshot


class FakeEasyReader:
    instances = []

    def __init__(self, lang_list, **kw):
        self.lang_list = lang_list
        self.result = []
        self.calls = []
        FakeEasyReader.instances.append(self)

    def readtext(self, image_path):
        self.calls.append(image_path)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class FakePaddle:
    instances = []

    def __init__(self, **kw):
        self.kwargs = kw
        self.result = []
        FakePaddle.instances.append(self)

    def predict(self, image_path):
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


@pytest.fixture
def engines(monkeypatch):
    FakeEasyReader.instances.clear()
    FakePaddle.instances.clear()
    easy = types.ModuleType("easyocr")
    easy.Reader = FakeEasyReader
    paddle = types.ModuleType("paddleocr")
    paddle.PaddleOCR = FakePaddle
    monkeypatch.setitem(sys.modules, "easyocr", easy)
    monkeypatch.setitem(sys.modules, "paddleocr", paddle)
    return types.SimpleNamespace(easy=FakeEasyReader, paddle=FakePaddle)


def quad(x1, y1, x2, y2):
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


@pytest.fixture
def image_file(tmp_path):
    p = tmp_path / "screen.png"
    Image.new("RGB", (400, 300), "white").save(p)
    return str(p)


@pytest.mark.functional
class TestOCRFunctional:
    def test_easyocr_chinese_lang_list(self, engines):
        OCRDetector(engine="easyocr", lang="ch")
        assert engines.easy.instances[-1].lang_list == ["ch_sim", "en"]

    def test_easyocr_english_lang_list(self, engines):
        OCRDetector(engine="easyocr", lang="en")
        assert engines.easy.instances[-1].lang_list == ["en"]

    def test_paddle_chinese_and_disabled_preprocessing(self, engines):
        OCRDetector(engine="paddleocr", lang="ch")
        kw = engines.paddle.instances[-1].kwargs
        assert kw["lang"] == "ch"
        assert kw["use_doc_orientation_classify"] is False
        assert kw["use_doc_unwarping"] is False
        assert kw["enable_mkldnn"] is False

    def test_paddle_english(self, engines):
        OCRDetector(engine="paddleocr", lang="en")
        assert engines.paddle.instances[-1].kwargs["lang"] == "en"

    def test_easyocr_result_unified_format(self, engines, image_file):
        det = OCRDetector(engine="easyocr")
        det.engine.result = [(quad(10, 20, 110, 50), "设置", 0.91)]
        assert det.detect(image_file) == [{"text": "设置", "bbox": [10, 20, 110, 50], "confidence": 0.91}]

    def test_easyocr_multiple_results_keep_order(self, engines, image_file):
        det = OCRDetector(engine="easyocr")
        det.engine.result = [(quad(0, 0, 10, 10), "A", 0.5), (quad(5, 5, 20, 20), "B", 0.6)]
        assert [r["text"] for r in det.detect(image_file)] == ["A", "B"]

    def test_easyocr_passes_image_path(self, engines, image_file):
        det = OCRDetector(engine="easyocr")
        det.detect(image_file)
        assert det.engine.calls == [image_file]

    def test_paddle_result_unified_format(self, engines, image_file):
        det = OCRDetector(engine="paddleocr")
        det.engine.result = [{"rec_texts": ["搜索"], "rec_scores": [0.88], "rec_boxes": [[5, 6, 70, 30]]}]
        assert det.detect(image_file) == [{"text": "搜索", "bbox": [5, 6, 70, 30], "confidence": 0.88}]

    def test_paddle_multi_page_results_merged(self, engines, image_file):
        det = OCRDetector(engine="paddleocr")
        det.engine.result = [
            {"rec_texts": ["a"], "rec_scores": [0.9], "rec_boxes": [[0, 0, 1, 1]]},
            {"rec_texts": ["b", "c"], "rec_scores": [0.8, 0.7], "rec_boxes": [[1, 1, 2, 2], [2, 2, 3, 3]]},
        ]
        assert [r["text"] for r in det.detect(image_file)] == ["a", "b", "c"]

    def test_confidence_is_python_float(self, engines, image_file):
        import numpy as np
        det = OCRDetector(engine="paddleocr")
        det.engine.result = [{"rec_texts": ["x"], "rec_scores": [np.float32(0.5)], "rec_boxes": [np.array([1, 2, 3, 4])]}]
        r = det.detect(image_file)[0]
        assert type(r["confidence"]) is float and all(type(v) is int for v in r["bbox"])

    def test_draw_boxes_returns_image_same_size(self, engines, image_file):
        det = OCRDetector()
        img = det.draw_boxes(image_file, [{"text": "a", "bbox": [10, 10, 50, 50], "confidence": 0.9}])
        assert img.size == (400, 300)

    def test_draw_boxes_draws_red_border(self, engines, image_file):
        det = OCRDetector()
        img = det.draw_boxes(image_file, [{"text": "a", "bbox": [10, 10, 50, 50], "confidence": 0.9}])
        assert img.getpixel((10, 30)) == (255, 0, 0)
        assert img.getpixel((30, 30)) == (255, 255, 255)  # 框内不填充

    def test_draw_boxes_saves_file(self, engines, image_file, tmp_path):
        out = tmp_path / "annotated.png"
        OCRDetector().draw_boxes(image_file, [], save_path=str(out))
        assert out.exists()

    def test_draw_boxes_does_not_modify_source(self, engines, image_file):
        OCRDetector().draw_boxes(image_file, [{"text": "a", "bbox": [0, 0, 100, 100], "confidence": 1.0}])
        assert Image.open(image_file).getpixel((0, 0)) == (255, 255, 255)

    def test_find_latest_screenshot_picks_newest(self, tmp_path):
        a, b = tmp_path / "a.png", tmp_path / "b.png"
        a.write_bytes(b"1")
        b.write_bytes(b"2")
        os.utime(a, (1000, 1000))
        os.utime(b, (2000, 2000))
        assert find_latest_screenshot(str(tmp_path)) == str(b)

    def test_benchmark_reports_both_engines(self, engines, image_file, monkeypatch):
        orig_init = FakeEasyReader.__init__

        def init(self, lang_list, **kw):
            orig_init(self, lang_list)
            self.result = [(quad(0, 0, 5, 5), "a", 0.4), (quad(0, 0, 5, 5), "b", 0.6)]
        monkeypatch.setattr(FakeEasyReader, "__init__", init)
        rep = benchmark_engines(image_file)
        assert set(rep) == {"easyocr", "paddleocr"}
        assert rep["easyocr"]["detected_count"] == 2
        assert rep["easyocr"]["avg_confidence"] == 0.5
        assert {"init_time_sec", "infer_time_sec"} <= set(rep["paddleocr"])


@pytest.mark.boundary
class TestOCRBoundary:
    def test_empty_easyocr_result(self, engines, image_file):
        det = OCRDetector()
        assert det.detect(image_file) == []

    def test_empty_paddle_page(self, engines, image_file):
        det = OCRDetector(engine="paddleocr")
        det.engine.result = [{}]
        assert det.detect(image_file) == []

    def test_rotated_quad_to_axis_aligned_bbox(self, engines, image_file):
        det = OCRDetector()
        det.engine.result = [([[50, 10], [90, 40], [60, 80], [20, 50]], "斜", 0.7)]
        assert det.detect(image_file)[0]["bbox"] == [20, 10, 90, 80]

    def test_float_coordinates_truncated_to_int(self, engines, image_file):
        det = OCRDetector()
        det.engine.result = [(quad(10.7, 20.2, 110.9, 50.5), "t", 0.9)]
        assert det.detect(image_file)[0]["bbox"] == [10, 20, 110, 50]

    def test_paddle_mismatched_lengths_truncated_by_zip(self, engines, image_file):
        det = OCRDetector(engine="paddleocr")
        det.engine.result = [{"rec_texts": ["a", "b", "c"], "rec_scores": [0.9, 0.8], "rec_boxes": [[0, 0, 1, 1]] * 3}]
        assert len(det.detect(image_file)) == 2

    def test_empty_text_kept(self, engines, image_file):
        det = OCRDetector()
        det.engine.result = [(quad(0, 0, 1, 1), "", 0.1)]
        assert det.detect(image_file)[0]["text"] == ""

    def test_draw_threshold_equal_value_kept(self, engines, image_file):
        img = OCRDetector().draw_boxes(
            image_file, [{"text": "a", "bbox": [10, 10, 50, 50], "confidence": 0.3}], conf_threshold=0.3)
        assert img.getpixel((10, 30)) == (255, 0, 0)

    def test_draw_threshold_below_value_skipped(self, engines, image_file):
        img = OCRDetector().draw_boxes(
            image_file, [{"text": "a", "bbox": [10, 10, 50, 50], "confidence": 0.299}], conf_threshold=0.3)
        assert img.getpixel((10, 30)) == (255, 255, 255)

    def test_draw_threshold_zero_draws_everything(self, engines, image_file):
        img = OCRDetector().draw_boxes(
            image_file, [{"text": "a", "bbox": [10, 10, 50, 50], "confidence": 0.0}], conf_threshold=0.0)
        assert img.getpixel((10, 30)) == (255, 0, 0)

    def test_draw_box_partially_outside_image(self, engines, image_file):
        img = OCRDetector().draw_boxes(image_file, [{"text": "a", "bbox": [350, 250, 500, 400], "confidence": 1}])
        assert img.size == (400, 300)

    def test_draw_zero_area_box(self, engines, image_file):
        OCRDetector().draw_boxes(image_file, [{"text": "a", "bbox": [20, 20, 20, 20], "confidence": 1}])

    def test_find_latest_missing_dir(self, tmp_path):
        assert find_latest_screenshot(str(tmp_path / "nope")) is None

    def test_find_latest_empty_dir(self, tmp_path):
        assert find_latest_screenshot(str(tmp_path)) is None

    def test_find_latest_only_annotated_returns_none(self, tmp_path):
        (tmp_path / "annotated_easyocr.png").write_bytes(b"x")
        assert find_latest_screenshot(str(tmp_path)) is None

    def test_find_latest_ignores_non_png(self, tmp_path):
        (tmp_path / "a.jpg").write_bytes(b"x")
        assert find_latest_screenshot(str(tmp_path)) is None

    def test_benchmark_zero_results_avg_conf_zero(self, engines, image_file):
        rep = benchmark_engines(image_file)
        assert rep["easyocr"]["avg_confidence"] == 0 and rep["easyocr"]["detected_count"] == 0

    def test_find_latest_skips_annotated_images(self, tmp_path):
        raw, ann = tmp_path / "screenshot_1.png", tmp_path / "annotated_easyocr.png"
        raw.write_bytes(b"1")
        ann.write_bytes(b"2")
        os.utime(raw, (1000, 1000))
        os.utime(ann, (2000, 2000))
        assert find_latest_screenshot(str(tmp_path)) == str(raw)


@pytest.mark.exception
class TestOCRException:
    def test_unsupported_engine(self, engines):
        with pytest.raises(ValueError, match="不支持的引擎"):
            OCRDetector(engine="tesseract")

    def test_easyocr_not_installed(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "easyocr", None)
        with pytest.raises(ImportError):
            OCRDetector(engine="easyocr")

    def test_paddle_not_installed(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "paddleocr", None)
        with pytest.raises(ImportError):
            OCRDetector(engine="paddleocr")

    def test_engine_runtime_error_propagates(self, engines, image_file):
        det = OCRDetector()
        det.engine.result = RuntimeError("CUDA out of memory")
        with pytest.raises(RuntimeError):
            det.detect(image_file)

    def test_draw_boxes_missing_image(self, engines, tmp_path):
        with pytest.raises(FileNotFoundError):
            OCRDetector().draw_boxes(str(tmp_path / "missing.png"), [])

    def test_draw_boxes_malformed_bbox(self, engines, image_file):
        with pytest.raises(ValueError):
            OCRDetector().draw_boxes(image_file, [{"text": "a", "bbox": [1, 2, 3], "confidence": 1}])

    def test_draw_boxes_missing_confidence_key(self, engines, image_file):
        with pytest.raises(KeyError):
            OCRDetector().draw_boxes(image_file, [{"text": "a", "bbox": [1, 2, 3, 4]}])

    def test_benchmark_captures_engine_error(self, engines, image_file, monkeypatch):
        def broken(self, **kw):
            raise RuntimeError("paddle init failed")
        monkeypatch.setattr(FakePaddle, "__init__", broken)
        rep = benchmark_engines(image_file)
        assert "error" in rep["paddleocr"] and "paddle init failed" in rep["paddleocr"]["error"]
        assert "error" not in rep["easyocr"]

    def test_detect_with_unknown_engine_name_returns_empty(self, engines, image_file):
        det = OCRDetector()
        det.engine_name = "other"
        assert det.detect(image_file) == []


@pytest.mark.multi_resolution
class TestOCRMultiResolution:
    @pytest.mark.parametrize("w,h", [(800, 600), (1920, 1080), (2560, 1440), (3840, 2160)])
    def test_bbox_at_bottom_right_corner(self, engines, tmp_path, w, h):
        p = tmp_path / "s.png"
        Image.new("RGB", (w, h), "white").save(p)
        det = OCRDetector()
        det.engine.result = [(quad(w - 120, h - 40, w - 1, h - 1), "确定", 0.9)]
        r = det.detect(str(p))[0]
        assert r["bbox"] == [w - 120, h - 40, w - 1, h - 1]
        img = det.draw_boxes(str(p), [r])
        assert img.size == (w, h) and img.getpixel((w - 1, h - 20)) == (255, 0, 0)

    @pytest.mark.parametrize("scale", [1.0, 1.25, 1.5, 2.0])
    def test_bbox_from_dpi_scaled_screenshot(self, engines, image_file, scale):
        """Windows 缩放（100%/125%/150%/200%）下截图为物理像素，bbox 应原样保留物理像素坐标。"""
        det = OCRDetector()
        x1, y1, x2, y2 = [int(v * scale) for v in (100, 50, 180, 80)]
        det.engine.result = [(quad(x1, y1, x2, y2), "文件", 0.8)]
        assert det.detect(image_file)[0]["bbox"] == [x1, y1, x2, y2]
