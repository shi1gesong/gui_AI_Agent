"""第4周：agent/grounding.py（文本 → OCR 元素 → 坐标）—— mock 用例"""
import pytest

from agent.grounding import Grounder, bbox_center, match_score, normalize_text, scale_point

ELEMS = [
    {"text": "发送", "bbox": [700, 500, 780, 540], "confidence": 0.98},
    {"text": "消息内容", "bbox": [240, 500, 680, 540], "confidence": 0.95},
    {"text": "张三", "bbox": [20, 80, 200, 120], "confidence": 0.97},
    {"text": "李四", "bbox": [20, 130, 200, 170], "confidence": 0.96},
    {"text": "文件传输助手", "bbox": [20, 230, 200, 270], "confidence": 0.9},
]


@pytest.mark.functional
class TestGroundingFunctional:
    def test_exact_match_center(self):
        hit = Grounder().locate("发送", ELEMS)
        assert hit["point"] == (740, 520) and hit["score"] == 1.0

    def test_locate_contact(self):
        assert Grounder().locate("张三", ELEMS)["element"]["text"] == "张三"

    def test_substring_match(self):
        """LLM 输出「发送按钮」，屏幕上是「发送」"""
        assert Grounder().locate("发送按钮", ELEMS)["element"]["text"] == "发送"

    def test_target_is_substring_of_element(self):
        assert Grounder().locate("传输助手", ELEMS)["element"]["text"] == "文件传输助手"

    @pytest.mark.parametrize("ocr_text", ["发 送", "发送。", "【发送】", "发送 ", "ＦＡ发送", " 发送:"])
    def test_ocr_noise_tolerated(self, ocr_text):
        elems = [{"text": ocr_text, "bbox": [0, 0, 10, 10]}]
        assert Grounder().locate("发送", elems) is not None

    @pytest.mark.parametrize("a,b", [("Ｈｅｌｌｏ", "hello"), ("OK", "ok"), ("确 定", "确定"), ("设置…", "设置")])
    def test_normalize_equivalence(self, a, b):
        assert normalize_text(a) == normalize_text(b)

    def test_prefers_exact_over_substring(self):
        elems = [{"text": "发送文件", "bbox": [0, 0, 100, 20]}, {"text": "发送", "bbox": [0, 50, 40, 70]}]
        assert Grounder().locate("发送", elems)["element"]["text"] == "发送"

    def test_tie_prefers_higher_confidence(self):
        elems = [{"text": "确定", "bbox": [0, 0, 10, 10], "confidence": 0.5},
                 {"text": "确定", "bbox": [0, 100, 10, 110], "confidence": 0.9}]
        assert Grounder().locate("确定", elems)["point"] == (5, 105)

    def test_tie_prefers_smaller_box(self):
        elems = [{"text": "搜索", "bbox": [0, 0, 1000, 100]}, {"text": "搜索", "bbox": [0, 200, 50, 220]}]
        assert Grounder().locate("搜索", elems)["point"] == (25, 210)

    def test_rank_order(self):
        ranked = Grounder(min_score=0.0).rank("发送", ELEMS)
        assert ranked[0][1]["text"] == "发送"

    def test_scale_point_150_percent(self):
        assert scale_point((1500, 900), (1.5, 1.5)) == (1000, 600)

    def test_scale_point_with_offset(self):
        assert scale_point((100, 100), (1.0, 1.0), offset=(2560, 0)) == (2660, 100)

    def test_fuzzy_similar_text(self):
        """OCR 把「通讯录」识别成「通讯彔」也能匹配"""
        elems = [{"text": "通讯彔", "bbox": [0, 0, 60, 20]}]
        assert Grounder().locate("通讯录", elems) is not None


@pytest.mark.boundary
class TestGroundingBoundary:
    def test_empty_target(self):
        assert Grounder().locate("", ELEMS) is None

    def test_empty_elements(self):
        assert Grounder().locate("发送", []) is None

    def test_unrelated_text_not_matched(self):
        assert Grounder().locate("设置", ELEMS) is None

    def test_min_score_threshold(self):
        elems = [{"text": "通讯彔", "bbox": [0, 0, 60, 20]}]
        assert Grounder(min_score=0.99).locate("通讯录", elems) is None

    def test_min_confidence_filters(self):
        elems = [{"text": "发送", "bbox": [0, 0, 10, 10], "confidence": 0.1}]
        assert Grounder(min_confidence=0.3).locate("发送", elems) is None

    def test_zero_size_bbox_center(self):
        assert bbox_center([10, 10, 10, 10]) == (10, 10)

    def test_odd_bbox_rounding(self):
        assert bbox_center([0, 0, 3, 3]) == (2, 2)

    def test_punctuation_only_target(self):
        assert match_score("。。", "发送") == 0.0

    def test_match_score_range(self):
        for e in ELEMS:
            assert 0.0 <= match_score("发送", e["text"]) <= 1.0


@pytest.mark.exception
class TestGroundingException:
    @pytest.mark.parametrize("bbox", [None, [], [1, 2, 3], ["a", 0, 1, 1], [10, 10, 0, 0]])
    def test_invalid_bbox_center_none(self, bbox):
        assert bbox_center(bbox) is None

    def test_element_without_bbox_skipped(self):
        elems = [{"text": "发送"}, {"text": "发送", "bbox": [0, 0, 20, 20]}]
        assert Grounder().locate("发送", elems)["point"] == (10, 10)

    def test_all_without_bbox(self):
        assert Grounder().locate("发送", [{"text": "发送"}]) is None

    def test_non_dict_elements_ignored(self):
        assert Grounder().locate("发送", ["发送", None, 3]) is None

    @pytest.mark.parametrize("scale", [(0, 1), (1, 0), (-1, 1)])
    def test_invalid_scale(self, scale):
        with pytest.raises(ValueError):
            scale_point((1, 1), scale)

    def test_none_text(self):
        assert match_score("发送", None) == 0.0


@pytest.mark.multi_resolution
class TestGroundingMultiResolution:
    @pytest.mark.parametrize("dpi", [1.0, 1.25, 1.5, 1.75, 2.0])
    def test_center_scaled_back_to_logical(self, dpi):
        """截图物理像素坐标按 DPI 缩放换算后，应落回逻辑坐标 (400, 300)"""
        px = (400 * dpi, 300 * dpi)
        assert scale_point(px, (dpi, dpi)) == (400, 300)

    def test_non_uniform_scale(self):
        assert scale_point((1920, 1080), (1.5, 1.25)) == (1280, 864)


@pytest.mark.boundary
class TestGroundingLongTextRegression:
    """2026-11-24 真机：命令行里打印的整句子任务含“消息内容”，被当成了输入框"""
    LINE = '4。在"消息内容"框中输入 "今晚7点开会"'

    def test_long_line_containing_target_rejected(self):
        assert Grounder().locate("消息内容", [{"text": self.LINE, "bbox": [0, 0, 900, 30]}]) is None

    def test_misread_placeholder_beats_long_line(self):
        elems = [{"text": self.LINE, "bbox": [1000, 0, 1900, 30]}, {"text": "消息内客", "bbox": [230, 500, 700, 540]}]
        assert Grounder().locate("消息内容", elems)["element"]["text"] == "消息内客"

    def test_find_element_uses_same_scoring(self):
        from action_validator import find_element
        elems = [{"text": self.LINE, "bbox": [1000, 0, 1900, 30]}, {"text": "消息内客", "bbox": [230, 500, 700, 540]}]
        assert find_element("消息内容", elems)["text"] == "消息内客"

    @pytest.mark.parametrize("target,text", [("张三", "[指令] 给张三发送消息：今晚7点开会"), ("发送", "5.点击“发送”按钮然后等待")])
    def test_other_console_lines_rejected(self, target, text):
        assert Grounder().locate(target, [{"text": text, "bbox": [0, 0, 500, 30]}]) is None

    def test_moderate_containment_still_matches(self):
        """“传输助手”在“文件传输助手”里（长度比 0.67）仍然匹配"""
        assert Grounder().locate("传输助手", [{"text": "文件传输助手", "bbox": [0, 0, 10, 10]}]) is not None
