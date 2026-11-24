"""第5周：finetune/action_format.py（VLM 动作格式：数据集转换、训练目标、输出解析、坐标换算）—— mock 用例"""
import json

import pytest

from finetune.action_format import (ACTIONS, clamp_coord, convert_sequence, from_dataset_action, normalize_action,
                                    normalize_key, parse_output, to_agent_action, to_screen_point, to_target_text)

S = (1024, 768)


# --------------------------------------------------------------------------- #
@pytest.mark.functional
class TestDatasetConversion:
    def test_click_uses_ratio(self):
        a = {"action_type": "click", "x": 543, "y": 133, "x_ratio": 0.5303, "y_ratio": 0.1732, "button": "left"}
        assert from_dataset_action(a, S) == {"action": "click", "x": 530, "y": 173}

    def test_click_from_pixels_when_no_ratio(self):
        assert from_dataset_action({"action_type": "click", "x": 512, "y": 384}, S) == {"action": "click", "x": 500,
                                                                                         "y": 500}

    def test_right_click(self):
        a = {"action_type": "click", "x_ratio": 0.1, "y_ratio": 0.2, "button": "right"}
        assert from_dataset_action(a, S)["action"] == "right_click"

    def test_double_click(self):
        a = {"action_type": "double_click", "x_ratio": 0.1, "y_ratio": 0.2}
        assert from_dataset_action(a, S) == {"action": "double_click", "x": 100, "y": 200}

    def test_type(self):
        assert from_dataset_action({"action_type": "type", "text": "天气"}, S) == {"action": "type", "text": "天气"}

    def test_key_aliases_normalized(self):
        a = {"action_type": "key", "keys": ["Control", "a"]}
        assert from_dataset_action(a, S) == {"action": "key", "keys": ["ctrl", "a"]}

    def test_scroll_with_point(self):
        a = {"action_type": "scroll", "amount": -3, "x_ratio": 0.5, "y_ratio": 0.5}
        assert from_dataset_action(a, S) == {"action": "scroll", "amount": -3, "x": 500, "y": 500}

    def test_wait_integer_seconds(self):
        assert from_dataset_action({"action_type": "wait", "seconds": 2.0}, S) == {"action": "wait", "seconds": 2}

    def test_move_then_drag_merged(self):
        seq = [{"action_type": "move", "x_ratio": 0.1, "y_ratio": 0.1},
               {"action_type": "drag", "x_ratio": 0.5, "y_ratio": 0.6}]
        assert convert_sequence(seq, S) == [{"action": "drag", "x": 100, "y": 100, "x2": 500, "y2": 600}]

    def test_sequence_click_type_key(self):
        seq = [{"action_type": "click", "x_ratio": 0.2, "y_ratio": 0.05},
               {"action_type": "type", "text": "abc"}, {"action_type": "key", "keys": ["Return"]}]
        out = convert_sequence(seq, S)
        assert [a["action"] for a in out] == ["click", "type", "key"] and out[2]["keys"] == ["enter"]

    def test_target_text_is_compact_json(self):
        t = to_target_text([{"y": 2, "action": "click", "x": 1}])
        assert t == '[{"action": "click", "x": 1, "y": 2}]'

    def test_target_text_keeps_chinese(self):
        assert "天气" in to_target_text([{"action": "type", "text": "天气"}])

    def test_target_roundtrip(self):
        acts = [{"action": "click", "x": 10, "y": 20}, {"action": "type", "text": "a b"},
                {"action": "key", "keys": ["ctrl", "s"]}, {"action": "drag", "x": 1, "y": 2, "x2": 3, "y2": 4},
                {"action": "scroll", "amount": -2}, {"action": "wait", "seconds": 1}]
        assert parse_output(to_target_text(acts))["actions"] == acts


@pytest.mark.boundary
class TestDatasetConversionBoundary:
    @pytest.mark.parametrize("t", ["mouse_down", "mouse_up", "unknown"])
    def test_low_level_actions_dropped(self, t):
        assert from_dataset_action({"action_type": t, "x": 1, "y": 1}, S) is None

    def test_empty_type_dropped(self):
        assert from_dataset_action({"action_type": "type", "text": ""}, S) is None

    def test_empty_keys_dropped(self):
        assert from_dataset_action({"action_type": "key", "keys": []}, S) is None

    def test_click_without_coordinates_dropped(self):
        assert from_dataset_action({"action_type": "click"}, S) is None

    def test_ratio_clamped(self):
        a = {"action_type": "click", "x_ratio": 1.3, "y_ratio": -0.1}
        assert from_dataset_action(a, S) == {"action": "click", "x": 1000, "y": 0}

    def test_drag_without_previous_point(self):
        out = convert_sequence([{"action_type": "drag", "x_ratio": 0.5, "y_ratio": 0.5}], S)
        assert out == [{"action": "drag", "x": 500, "y": 500, "x2": 500, "y2": 500}]

    def test_empty_sequence(self):
        assert convert_sequence([], S) == [] and convert_sequence(None, S) == []

    @pytest.mark.parametrize("v,exp", [(-5, 0), (0, 0), (500.4, 500), (999.6, 1000), (1e6, 1000)])
    def test_clamp(self, v, exp):
        assert clamp_coord(v) == exp

    @pytest.mark.parametrize("k,exp", [("Return", "enter"), ("ESCAPE", "esc"), ("super_l", "win"), ("a", "a"),
                                       ("  Ctrl ", "ctrl"), ("pgdn", "pagedown")])
    def test_normalize_key(self, k, exp):
        assert normalize_key(k) == exp


# --------------------------------------------------------------------------- #
@pytest.mark.functional
class TestParseOutput:
    def test_plain_array(self):
        r = parse_output('[{"action": "click", "x": 10, "y": 20}]')
        assert r["ok"] and r["actions"] == [{"action": "click", "x": 10, "y": 20}]

    def test_code_fence(self):
        r = parse_output('好的：\n```json\n[{"action": "double_click", "x": 1, "y": 2}]\n```')
        assert r["actions"][0]["action"] == "double_click"

    def test_think_tags_removed(self):
        r = parse_output('<think>先点击 [1,2]</think>[{"action": "click", "x": 5, "y": 6}]')
        assert r["actions"] == [{"action": "click", "x": 5, "y": 6}]

    def test_single_object(self):
        assert parse_output('{"action": "key", "keys": ["enter"]}')["actions"] == [{"action": "key", "keys": ["enter"]}]

    def test_object_with_actions_field(self):
        r = parse_output('{"actions": [{"action": "type", "text": "x"}]}')
        assert r["actions"] == [{"action": "type", "text": "x"}]

    def test_action_type_alias_key(self):
        r = parse_output('[{"action_type": "left_click", "x": 3, "y": 4}]')
        assert r["actions"] == [{"action": "click", "x": 3, "y": 4}]

    def test_point_list(self):
        r = parse_output('[{"action": "click", "point": [300, 400]}]')
        assert r["actions"] == [{"action": "click", "x": 300, "y": 400}]

    def test_bbox_center(self):
        r = parse_output('[{"action": "click", "coordinate": [100, 100, 300, 200]}]')
        assert r["actions"] == [{"action": "click", "x": 200, "y": 150}]

    def test_qwen_box_string(self):
        r = parse_output('[{"action": "click", "position": "<|box_start|>(100,200),(300,400)<|box_end|>"}]')
        assert r["actions"] == [{"action": "click", "x": 200, "y": 300}]

    def test_ratio_float_coordinates(self):
        r = parse_output('[{"action": "click", "x": 0.25, "y": 0.5}]')
        assert r["actions"] == [{"action": "click", "x": 250, "y": 500}]

    def test_pixel_coordinates_converted(self):
        r = parse_output('[{"action": "click", "x": 2048, "y": 720}]', screen_size=(2560, 1440))
        assert r["actions"] == [{"action": "click", "x": 800, "y": 500}]

    def test_hotkey_string(self):
        r = parse_output('[{"action": "hotkey", "keys": "Ctrl+Shift+Esc"}]')
        assert r["actions"] == [{"action": "key", "keys": ["ctrl", "shift", "esc"]}]

    def test_scroll_direction_word(self):
        r = parse_output('[{"action": "scroll_down"}]')
        assert r["actions"][0]["amount"] < 0

    def test_click_with_clicks_2_is_double(self):
        assert parse_output('[{"action": "click", "x": 1, "y": 1, "clicks": 2}]')["actions"][0]["action"] == "double_click"

    def test_right_button(self):
        assert parse_output('[{"action": "click", "x": 1, "y": 1, "button": "right"}]')["actions"][0]["action"] == \
            "right_click"

    def test_single_quotes_and_trailing_comma(self):
        r = parse_output("[{'action': 'click', 'x': 1, 'y': 2},]")
        assert r["ok"] and r["actions"][0]["x"] == 1

    def test_drag_end_list(self):
        r = parse_output('[{"action": "drag", "start": [1, 2], "end": [3, 4]}]')
        assert r["actions"] == [{"action": "drag", "x": 1, "y": 2, "x2": 3, "y2": 4}]

    def test_partial_invalid_counted(self):
        r = parse_output('[{"action": "click", "x": 1, "y": 2}, {"action": "fly"}]')
        assert r["ok"] and r["dropped"] == 1 and len(r["actions"]) == 1


@pytest.mark.exception
class TestParseOutputInvalid:
    @pytest.mark.parametrize("text", ["", "   ", None, 123])
    def test_empty_or_non_string(self, text):
        assert parse_output(text)["ok"] is False

    @pytest.mark.parametrize("text", ["点击确定按钮", "[not json", "{broken: }", "```json\n```"])
    def test_no_json(self, text):
        r = parse_output(text)
        assert r["ok"] is False and r["actions"] == []

    @pytest.mark.parametrize("obj", [
        [{"action": "fly", "x": 1, "y": 1}],
        [{"action": "click"}],
        [{"action": "click", "x": "a", "y": 2}],
        [{"action": "click", "x": -5, "y": 2}],
        [{"action": "type", "text": ""}],
        [{"action": "key", "keys": []}],
        [{"x": 1, "y": 2}],
        [{"action": "drag", "x": 1, "y": 2}],
        ["click"],
    ])
    def test_unrecognized_actions(self, obj):
        r = parse_output(json.dumps(obj))
        assert r["ok"] is False and r["error"] == "没有可识别的动作"

    def test_json_scalar(self):
        assert parse_output("42")["ok"] is False

    def test_nan_coordinates_rejected(self):
        assert normalize_action({"action": "click", "x": float("nan"), "y": 1}) is None

    def test_bool_coordinates_rejected(self):
        assert normalize_action({"action": "click", "x": True, "y": 1}) is None

    def test_wait_negative_defaults(self):
        assert normalize_action({"action": "wait", "seconds": -3}) == {"action": "wait", "seconds": 1}

    def test_wait_capped(self):
        assert normalize_action({"action": "wait", "seconds": 999})["seconds"] == 60


# --------------------------------------------------------------------------- #
@pytest.mark.multi_resolution
class TestScreenPoint:
    @pytest.mark.parametrize("size", [(1024, 768), (1366, 768), (1920, 1080), (2560, 1440), (3840, 2160)])
    def test_corners_inside_screen(self, size):
        assert to_screen_point(0, 0, size) == (0, 0)
        assert to_screen_point(1000, 1000, size) == (size[0] - 1, size[1] - 1)

    @pytest.mark.parametrize("size", [(1280, 720), (1920, 1080), (2560, 1440)])
    def test_center(self, size):
        x, y = to_screen_point(500, 500, size)
        assert abs(x - size[0] / 2) <= 1 and abs(y - size[1] / 2) <= 1

    def test_offset_added(self):
        assert to_screen_point(0, 0, (100, 100), offset=(1920, 0)) == (1920, 0)

    def test_same_normalized_point_scales(self):
        a = to_screen_point(250, 750, (1280, 720))
        b = to_screen_point(250, 750, (2560, 1440))
        assert abs(b[0] - 2 * a[0]) <= 2 and abs(b[1] - 2 * a[1]) <= 2


@pytest.mark.exception
def test_screen_point_invalid_size():
    with pytest.raises(ValueError):
        to_screen_point(1, 1, (0, 0))


@pytest.mark.functional
class TestToAgentAction:
    def test_click(self):
        a = to_agent_action([{"action": "click", "x": 500, "y": 500}], (2561, 1441))
        assert a["action_type"] == "click" and a["point"] == [1280, 720] and "follow_up" not in a

    def test_click_type_enter_follow_up(self):
        a = to_agent_action([{"action": "click", "x": 0, "y": 0}, {"action": "type", "text": "hi"},
                             {"action": "key", "keys": ["enter"]}], (100, 100))
        assert [f["action_type"] for f in a["follow_up"]] == ["type", "key"]
        assert a["follow_up"][1]["input_text"] == "enter"

    def test_multi_key_becomes_hotkey(self):
        a = to_agent_action([{"action": "key", "keys": ["ctrl", "a"]}], (100, 100))
        assert a["action_type"] == "hotkey" and a["input_text"] == "ctrl+a"

    def test_right_click(self):
        a = to_agent_action([{"action": "right_click", "x": 0, "y": 0}], (100, 100))
        assert a["action_type"] == "click" and a["button"] == "right"

    def test_drag_end_point(self):
        a = to_agent_action([{"action": "drag", "x": 0, "y": 0, "x2": 1000, "y2": 1000}], (101, 101))
        assert a["point"] == [0, 0] and a["end_point"] == [100, 100]

    def test_scroll_without_point(self):
        a = to_agent_action([{"action": "scroll", "amount": -3}], (100, 100))
        assert a["amount"] == -3 and "point" not in a

    def test_wait(self):
        assert to_agent_action([{"action": "wait", "seconds": 2}], (10, 10))["input_text"] == "2"

    def test_empty_is_invalid(self):
        a = to_agent_action([], (100, 100))
        assert a["_invalid"] and a["action_type"] is None

    def test_actions_constant(self):
        assert set(ACTIONS) == {"click", "double_click", "right_click", "move", "drag", "scroll", "type", "key", "wait"}
