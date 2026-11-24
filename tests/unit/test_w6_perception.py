"""第6周：perception/perceiver_v2.py（ROI / 帧缓存 / 增量识别 / 碎片合并 / 词表纠错）与 perception/ui_detector.py（控件检测 + 标签关联）

截图与“OCR”来自 tests/fault_desktop.py：界面元素真实画在截图上，OCR 从像素反查，
因此裁剪、缓存、增量、DPI 缩放都在真实像素上验证。
"""
import pytest
from PIL import Image, ImageDraw

from fault_desktop import FaultyDesktop
from perception.perceiver_v2 import PerceiverV2, _rect_px, correct_text, merge_fragments
from perception.screen_parser import ScreenPerceiver
from perception.ui_detector import associate_labels, detect_widgets, empty_widgets


def env(tmp_path, dpi=1.0, logical=(1280, 720), open_chat=True, **kw):
    sim = FaultyDesktop(tmp_path, logical=logical, dpi=dpi)
    apps = sim.app_manager()
    if open_chat:
        apps.open("模拟聊天")
    p = PerceiverV2(capture=sim, ocr=sim, logical_size_fn=sim.logical_size, region_fn=apps.screen_regions,
                    save_frames=False, **kw)
    return sim, apps, p


def texts(obs):
    return sorted(e["text"] for e in obs.elements)


@pytest.mark.functional
class TestPerceiverV2:
    def test_roi_crops_to_foreground_window(self, tmp_path):
        sim, apps, p = env(tmp_path)
        o = p.perceive()
        assert o.extra["roi"] == [100, 60, 920, 640]
        assert o.extra["ocr_area_ratio"] < 0.6
        assert "张三" in texts(o) and "此电脑" not in texts(o)   # 后面的资源管理器窗口不识别

    def test_same_elements_as_v1(self, tmp_path):
        """v1 整屏识别后按中心点过滤，会把被前台窗口挡住、只露出边角的后台窗口标题也算进来；v2 先裁剪，只认窗口内的像素"""
        sim, apps, p = env(tmp_path)
        v1 = ScreenPerceiver(capture=sim, ocr=sim, logical_size_fn=sim.logical_size, region_fn=apps.screen_regions)
        a, b = texts(p.perceive()), texts(v1.perceive())
        assert set(b) - set(a) == {"文档 - 文件资源管理器"} and set(a) <= set(b)

    def test_coordinates_match_v1(self, tmp_path):
        sim, apps, p = env(tmp_path)
        v1 = ScreenPerceiver(capture=sim, ocr=sim, logical_size_fn=sim.logical_size, region_fn=apps.screen_regions)
        a = {e["text"]: e["bbox"] for e in p.perceive().elements}
        b = {e["text"]: e["bbox"] for e in v1.perceive().elements}
        assert all(b[k] == v for k, v in a.items())

    def test_cache_hit_on_identical_frame(self, tmp_path):
        sim, apps, p = env(tmp_path)
        p.perceive()
        n = sim.ocr_calls
        o = p.perceive()
        assert o.extra["cache"] == "hit" and sim.ocr_calls == n and o.extra["ocr_area_ratio"] == 0

    def test_incremental_after_small_change(self, tmp_path):
        sim, apps, p = env(tmp_path)
        p.perceive()
        sim.click(206, 140)           # 选中张三：只有联系人高亮/顶部标题变化
        o = p.perceive()
        assert o.extra["cache"] == "incremental" and o.extra["ocr_area_ratio"] < 0.2
        assert "当前聊天：张三" in texts(o) and "请选择联系人" not in texts(o) and "李四" in texts(o)

    def test_incremental_equals_full(self, tmp_path):
        sim, apps, p = env(tmp_path)
        p.perceive()
        sim.click(206, 140)
        sim.click(560, 580)
        sim.type_text("你好")
        inc = {e["text"]: e["bbox"] for e in p.perceive().elements}
        p.reset_cache()
        full = {e["text"]: e["bbox"] for e in p.perceive().elements}
        assert inc == full

    def test_full_refresh_after_window_switch(self, tmp_path):
        sim, apps, p = env(tmp_path)
        p.perceive()
        apps.open("计算器")
        o = p.perceive()
        assert o.extra["cache"] == "full" and "计算器" in texts(o)

    def test_full_every_limits_incremental_chain(self, tmp_path):
        sim, apps, p = env(tmp_path, full_every=2)
        p.perceive()
        statuses = []
        for i, ch in enumerate("abcd"):
            sim.click(560, 580)
            sim.type_text(ch)
            statuses.append(p.perceive().extra["cache"])
        assert statuses == ["incremental", "incremental", "full", "incremental"]

    def test_widgets_and_label_association(self, tmp_path):
        sim, apps, p = env(tmp_path)
        o = p.perceive()
        kinds = [w["kind"] for w in o.extra["widgets"]]
        assert "input" in kinds
        label = next(e for e in o.elements if e["text"] == "消息内容")
        ir = sim.input_rect()
        ib = label["input_bbox"]
        assert abs(ib[0] - ir[0]) <= 3 and abs(ib[2] - ir[2]) <= 3

    def test_executor_types_into_input_box(self, tmp_path):
        from agent.executor import ControllerExecutor
        sim, apps, p = env(tmp_path)
        sim.click(206, 140)
        sim.chat["focus"] = False
        o = p.perceive()
        r = ControllerExecutor(sim, sleep=lambda s: None)(
            {"action_type": "type", "target_text": "消息内容", "input_text": "你好"}, o.elements)
        assert r["ok"] and r["detail"]["grounding"][0].get("input_bbox")
        x, y = r["detail"]["grounding"][0]["point"]
        ir = sim.input_rect()
        assert ir[0] <= x <= ir[2] and sim.chat["text"] == "你好"

    def test_timing_recorded(self, tmp_path):
        sim, apps, p = env(tmp_path)
        t = p.perceive().extra["timing"]
        assert set(t) >= {"capture", "ocr", "post", "widgets", "total"} and t["total"] >= t["ocr"]

    def test_stats_accumulate(self, tmp_path):
        sim, apps, p = env(tmp_path)
        p.perceive()
        p.perceive()
        assert p.stats["calls"] == 2 and p.stats["hit"] == 1 and p.stats["full"] == 1

    def test_summary_contains_cache(self, tmp_path):
        sim, apps, p = env(tmp_path)
        assert p.perceive().summary()["cache"] == "full"

    def test_lexicon_correction(self, tmp_path):
        class TypoOCR:
            def detect(self, path):
                return [{"text": "消息内客", "bbox": [10, 10, 100, 40], "confidence": 0.9}]
        sim = FaultyDesktop(tmp_path)
        p = PerceiverV2(capture=sim, ocr=TypoOCR(), logical_size_fn=sim.logical_size, widgets=False, save_frames=False)
        e = p.perceive().elements[0]
        assert e["text"] == "消息内容" and e["raw_text"] == "消息内客"

    def test_add_words_extends_lexicon(self):
        p = PerceiverV2()
        p.add_words(["文件传输", "文件传输"])
        assert p.extra_words == ["文件传输"]

    def test_grab_returns_image_without_ocr(self, tmp_path):
        sim, apps, p = env(tmp_path)
        n = sim.ocr_calls
        img = p.grab()
        assert img.size == (1280, 720) and sim.ocr_calls == n


@pytest.mark.multi_resolution
class TestPerceiverV2MultiRes:
    @pytest.mark.parametrize("dpi", [1.0, 1.25, 1.5, 2.0])
    def test_dpi_roi_coordinates(self, tmp_path, dpi):
        sim, apps, p = env(tmp_path, dpi=dpi)
        o = p.perceive()
        b = {e["text"]: e["bbox"] for e in o.elements}["张三"]
        assert abs(b[0] - 110) <= 2 and abs(b[1] - 120) <= 2 and abs(b[2] - 300) <= 2

    @pytest.mark.parametrize("logical", [(1280, 720), (1366, 768), (1920, 1080), (2560, 1440)])
    def test_resolutions(self, tmp_path, logical):
        sim, apps, p = env(tmp_path, logical=logical)
        o = p.perceive()
        assert "发送" in texts(o) and o.extra["ocr_area_ratio"] == pytest.approx(820 * 580 / (logical[0] * logical[1]), abs=0.01)

    @pytest.mark.parametrize("dpi", [1.0, 1.5])
    def test_upscale_preprocess_keeps_coordinates(self, tmp_path, dpi):
        sim, apps, p = env(tmp_path, dpi=dpi, preprocess="upscale")
        b = {e["text"]: e["bbox"] for e in p.perceive().elements}
        assert abs(b["发送"][0] - 820) <= 2 and abs(b["发送"][3] - 600) <= 2

    def test_rect_px_scaling(self):
        assert _rect_px([100, 60, 920, 640], (1.5, 1.5), (0, 0), (1920, 1080)) == [150, 90, 1380, 960]


@pytest.mark.boundary
class TestPerceiverV2Boundary:
    def test_no_region_full_screen(self, tmp_path):
        sim = FaultyDesktop(tmp_path)
        p = PerceiverV2(capture=sim, ocr=sim, logical_size_fn=sim.logical_size, save_frames=False)
        o = p.perceive()
        assert o.extra["roi"] == [0, 0, 1280, 720] and o.extra["ocr_area_ratio"] == 1.0

    def test_roi_disabled(self, tmp_path):
        sim, apps, p = env(tmp_path, roi=False)
        assert p.perceive().extra["roi"] == [0, 0, 1280, 720]

    def test_cache_disabled(self, tmp_path):
        sim, apps, p = env(tmp_path, cache=False, incremental=False)
        p.perceive()
        assert p.perceive().extra["cache"] == "full"

    def test_cache_size_bounded(self, tmp_path):
        sim, apps, p = env(tmp_path, cache_size=2, incremental=False)
        for ch in "abcd":
            sim.click(560, 580)
            sim.type_text(ch)
            p.perceive()
        assert len(p._cache) == 2

    def test_tiny_window_rect_ignored(self):
        assert _rect_px([0, 0, 5, 5], (1, 1), (0, 0), (100, 100)) is None
        assert _rect_px(None, (1, 1), (0, 0), (100, 100)) is None

    def test_rect_clamped_to_image(self):
        assert _rect_px([-50, -50, 5000, 5000], (1, 1), (0, 0), (1280, 720)) == [0, 0, 1280, 720]

    def test_widgets_disabled(self, tmp_path):
        sim, apps, p = env(tmp_path, widgets=False)
        assert "widgets" not in p.perceive().extra

    @pytest.mark.parametrize("raw,expected", [
        ([{"text": "发", "bbox": [0, 0, 20, 20]}, {"text": "送", "bbox": [24, 0, 44, 20]}], ["发送"]),
        ([{"text": "发", "bbox": [0, 0, 20, 20]}, {"text": "送", "bbox": [60, 0, 80, 20]}], ["发", "送"]),   # 间距太大
        ([{"text": "发", "bbox": [0, 0, 20, 20]}, {"text": "送", "bbox": [22, 30, 42, 50]}], ["发", "送"]),  # 不同行
        ([{"text": "消息内容", "bbox": [0, 0, 80, 20]}, {"text": "你好世界", "bbox": [84, 0, 164, 20]}],
         ["消息内容", "你好世界"]),                                                                      # 都不是短碎片
        ([{"text": "Se", "bbox": [0, 0, 20, 20]}, {"text": "nd", "bbox": [23, 0, 43, 20]}], ["Se nd"]),
    ])
    def test_merge_fragments(self, raw, expected):
        assert [r["text"] for r in merge_fragments(raw)] == expected

    @pytest.mark.parametrize("text,fixed", [("消息内客", "消息内容"), ("发送", None), ("发逆", None),
                                            ("请选择联系入", "请选择联系人"), ("完全无关的文字", None), ("消息内容", None)])
    def test_correct_text(self, text, fixed):
        from perception.perceiver_v2 import UI_LEXICON
        assert correct_text(text, UI_LEXICON) == fixed


def draw_form(size=(800, 400)):
    im = Image.new("RGB", size, (245, 245, 245))
    d = ImageDraw.Draw(im)
    d.rectangle([200, 300, 600, 340], fill="white", outline=(0, 0, 0))       # 输入框
    d.rectangle([620, 300, 700, 340], fill=(7, 193, 96))                     # 绿色按钮
    d.rectangle([20, 40, 90, 100], fill=(60, 60, 60))                        # 无文字图标按钮
    d.rectangle([200, 120, 600, 160], fill="white", outline=(0, 0, 0))       # 下方的输入框（标签在上面）
    return im


@pytest.mark.functional
class TestUIDetector:
    def test_detects_input_and_buttons(self):
        ws = detect_widgets(draw_form())
        kinds = sorted(w["kind"] for w in ws)
        assert kinds.count("input") == 2 and kinds.count("button") == 2

    def test_label_right_of_input(self):
        ws = detect_widgets(draw_form())
        els = [{"text": "消息内容", "bbox": [100, 305, 190, 335]}]
        associate_labels(els, ws)
        assert abs(els[0]["input_bbox"][0] - 200) <= 3

    def test_label_above_input(self):
        ws = detect_widgets(draw_form())
        els = [{"text": "用户名", "bbox": [205, 90, 280, 115]}]
        associate_labels(els, ws)
        assert abs(els[0]["input_bbox"][1] - 120) <= 3

    def test_text_inside_button(self):
        ws = detect_widgets(draw_form())
        els = [{"text": "发送", "bbox": [640, 308, 680, 332]}]
        associate_labels(els, ws)
        assert els[0]["widget"] == "button" and "input_bbox" not in els[0]

    def test_text_inside_input_is_its_own_input(self):
        ws = detect_widgets(draw_form())
        els = [{"text": "你好", "bbox": [210, 308, 260, 332]}]
        associate_labels(els, ws)
        assert els[0]["widget"] == "input" and els[0]["input_bbox"]

    def test_empty_widgets_are_icons(self):
        ws = detect_widgets(draw_form())
        els = [{"text": "发送", "bbox": [640, 308, 680, 332]}]
        empties = empty_widgets(els, ws)
        assert any(abs(w["bbox"][0] - 20) <= 3 for w in empties)


@pytest.mark.multi_resolution
class TestUIDetectorMultiRes:
    @pytest.mark.parametrize("scale", [1.0, 1.25, 1.5, 2.0])
    def test_scaled_form(self, scale):
        im = draw_form()
        im = im.resize((int(800 * scale), int(400 * scale)), Image.NEAREST)
        ws = detect_widgets(im)
        inputs = [w for w in ws if w["kind"] == "input"]
        assert len(inputs) == 2 and abs(inputs[-1]["bbox"][0] - 200 * scale) <= 4 * scale


@pytest.mark.boundary
class TestUIDetectorBoundary:
    def test_blank_image(self):
        assert detect_widgets(Image.new("RGB", (400, 300), "white")) == []

    def test_tiny_image(self):
        assert detect_widgets(Image.new("RGB", (10, 10), "white")) == []

    def test_too_small_boxes_ignored(self):
        im = Image.new("RGB", (400, 300), "white")
        ImageDraw.Draw(im).rectangle([10, 10, 20, 18], fill="black")
        assert detect_widgets(im) == []

    def test_gap_too_far_no_association(self):
        ws = [{"kind": "input", "bbox": [700, 0, 900, 30]}]
        els = [{"text": "标签", "bbox": [0, 0, 40, 30]}]
        associate_labels(els, ws)
        assert "input_bbox" not in els[0]

    def test_numpy_input(self):
        import numpy as np
        assert isinstance(detect_widgets(np.asarray(draw_form())), list)

    def test_grayscale_input(self):
        assert detect_widgets(draw_form().convert("L"))


@pytest.mark.exception
class TestPerceptionException:
    def test_missing_components(self):
        with pytest.raises(RuntimeError):
            PerceiverV2().perceive()

    def test_region_fn_error_falls_back_to_full(self, tmp_path):
        sim = FaultyDesktop(tmp_path)

        def boom():
            raise OSError("pygetwindow 错误")
        p = PerceiverV2(capture=sim, ocr=sim, logical_size_fn=sim.logical_size, region_fn=boom, save_frames=False)
        o = p.perceive()
        assert "region_error" in o.extra and o.extra["roi"] == [0, 0, 1280, 720]

    def test_widget_detection_error_isolated(self, tmp_path, monkeypatch):
        import perception.ui_detector as ud

        def boom(img):
            raise ImportError("No module named cv2")
        monkeypatch.setattr(ud, "detect_widgets", boom)
        sim, apps, p = env(tmp_path)
        o = p.perceive()
        assert "widgets_error" in o.extra and o.elements

    def test_bad_ocr_boxes_skipped(self, tmp_path):
        class BadOCR:
            def detect(self, path):
                return [{"text": "x"}, {"text": "y", "bbox": ["a", 1, 2, 3]}, {"text": "好的", "bbox": [1, 1, 30, 20]}]
        sim = FaultyDesktop(tmp_path)
        p = PerceiverV2(capture=sim, ocr=BadOCR(), logical_size_fn=sim.logical_size, widgets=False, save_frames=False)
        assert texts(p.perceive()) == ["好的"]

    def test_unknown_preprocess(self, tmp_path):
        sim, apps, p = env(tmp_path, preprocess="sharpen")
        with pytest.raises(ValueError):
            p.perceive()

    def test_capture_failure_propagates(self, tmp_path):
        sim = FaultyDesktop(tmp_path, faults=("screenshot_fail",))
        p = PerceiverV2(capture=sim, ocr=sim, logical_size_fn=sim.logical_size, save_frames=False)
        p.perceive()
        with pytest.raises(OSError, match="截图失败"):
            p.perceive()
