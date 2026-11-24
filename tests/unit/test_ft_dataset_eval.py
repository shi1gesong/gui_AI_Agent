"""第5周：提示词模板、微调数据集构建、评测指标与对比报告 —— mock 用例（不需要 torch）"""
import json
from pathlib import Path

import pytest

from finetune import build_dataset as bd
from finetune.evaluate import (ConstPredictor, ReplayPredictor, action_match, compare_table, evaluate,
                               load_summaries, plot_compare, point_distance, score_sample, summarize)
from finetune.prompts import DEFAULT_VARIANT, TEMPLATES, build_messages, build_prompt

REAL_DATA = Path(__file__).resolve().parents[2] / "data_cache" / "sft" / "test.jsonl"


# --------------------------------------------------------------------------- #
# 提示词
# --------------------------------------------------------------------------- #
@pytest.mark.functional
class TestPrompts:
    @pytest.mark.parametrize("variant", sorted(TEMPLATES))
    def test_contains_task_and_subtask(self, variant):
        p = build_prompt("文字高亮", "使用快捷键全选文本", variant)
        assert "文字高亮" in p and "使用快捷键全选文本" in p

    def test_schema_mentions_coordinate_convention(self):
        p = build_prompt("a", "b", "v1_schema")
        assert "0~1000" in p and "JSON 数组" in p

    def test_fewshot_extends_schema(self):
        assert build_prompt("a", "b", "v2_fewshot").startswith(build_prompt("a", "b", "v1_schema"))

    def test_v0_is_shortest(self):
        lens = {v: len(build_prompt("a", "b", v)) for v in TEMPLATES}
        assert lens["v0_minimal"] < lens["v1_schema"] < lens["v2_fewshot"]

    def test_default_variant_is_training_prompt(self):
        assert DEFAULT_VARIANT == "v1_schema"

    def test_messages_image_first(self):
        m = build_messages("a", "b")
        assert m[0]["content"][0] == {"type": "image"} and m[0]["content"][1]["type"] == "text"

    def test_messages_with_answer(self):
        m = build_messages("a", "b", answer="[]")
        assert m[-1]["role"] == "assistant" and m[-1]["content"][0]["text"] == "[]"

    def test_braces_in_instruction_safe(self):
        assert "{x}" in build_prompt("输入 {x}", None)


@pytest.mark.boundary
class TestPromptsBoundary:
    def test_subtask_equal_to_instruction_not_repeated(self):
        p = build_prompt("打开记事本", "打开记事本", "v1_schema")
        assert p.count("打开记事本") == 1

    def test_only_subtask(self):
        assert "全选" in build_prompt("", "全选", "v1_schema")

    def test_none_subtask(self):
        assert "当前子任务" in build_prompt("打开", None, "v1_schema")


@pytest.mark.exception
class TestPromptsInvalid:
    def test_unknown_variant(self):
        with pytest.raises(ValueError):
            build_prompt("a", "b", "v9")

    def test_both_empty(self):
        with pytest.raises(ValueError):
            build_prompt("", "  ")


# --------------------------------------------------------------------------- #
# 数据集构建
# --------------------------------------------------------------------------- #
def _raw(task_id, split="train", img="img.png", actions=None, size=(1024, 768), sub="子任务"):
    return {"task_id": task_id, "instruction": "任务", "sub_task": sub, "screenshot_path": img, "screen_size": list(size),
            "action_sequence": actions if actions is not None else [{"action_type": "click", "x_ratio": 0.5,
                                                                      "y_ratio": 0.5}],
            "split": split, "source": "screenagent"}


@pytest.fixture
def raw_dir(tmp_path):
    (tmp_path / "imgs").mkdir()
    rows = []
    for s in range(20):
        for k in range(3):
            name = f"imgs/s{s}_{k}.png"
            (tmp_path / name).write_bytes(b"x")
            rows.append(_raw(f"sess{s}/{k}", "test" if s >= 17 else "train", name, sub=f"步骤{k}"))
    path = tmp_path / "screenagent_processed.jsonl"
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return tmp_path, path


@pytest.mark.functional
class TestBuildDataset:
    def test_build_writes_splits(self, raw_dir):
        base, path = raw_dir
        st = bd.build(path, base / "sft", val_ratio=0.2, base_dirs=[base])
        for name in ("train", "val", "test"):
            assert (base / "sft" / f"{name}.jsonl").exists()
        assert st["test"]["samples"] == 9 and st["kept"] == 60

    def test_no_session_leakage(self, raw_dir):
        base, path = raw_dir
        st = bd.build(path, base / "sft", val_ratio=0.3, base_dirs=[base])
        assert st["session_overlap"] == {"train_val": 0, "train_test": 0, "val_test": 0}

    def test_target_and_actions_written(self, raw_dir):
        base, path = raw_dir
        bd.build(path, base / "sft", base_dirs=[base])
        row = bd.load_jsonl(base / "sft" / "test.jsonl")[0]
        assert row["actions"] == [{"action": "click", "x": 500, "y": 500}]
        assert json.loads(row["target"]) == row["actions"]
        assert Path(row["image"]).is_absolute()

    def test_deterministic_split(self, raw_dir):
        base, path = raw_dir
        a = bd.build(path, base / "a", seed=7, base_dirs=[base])
        b = bd.build(path, base / "b", seed=7, base_dirs=[base])
        assert a["val"] == b["val"] and (base / "a/train.jsonl").read_text() == (base / "b/train.jsonl").read_text()

    def test_stats_file(self, raw_dir):
        base, path = raw_dir
        bd.build(path, base / "sft", base_dirs=[base])
        st = json.loads((base / "sft" / "stats.json").read_text(encoding="utf-8"))
        assert st["train"]["first_action"] == {"click": st["train"]["samples"]}

    def test_windows_backslash_paths_resolved(self, tmp_path):
        (tmp_path / "d").mkdir()
        (tmp_path / "d" / "a.png").write_bytes(b"x")
        assert bd.resolve_image("d\\a.png", [tmp_path]) == (tmp_path / "d" / "a.png").resolve()

    def test_session_of(self):
        assert bd.session_of({"task_id": "abc/2024-01"}) == "abc"


@pytest.mark.boundary
class TestBuildDatasetBoundary:
    def test_duplicates_removed(self, tmp_path):
        (tmp_path / "a.png").write_bytes(b"x")
        rows = [_raw("s/1", img="a.png"), _raw("s/2", img="a.png")]
        kept, dropped = bd.convert_samples(rows, [tmp_path])
        assert len(kept) == 1 and dropped["重复样本"] == 1

    def test_missing_image_dropped(self, tmp_path):
        kept, dropped = bd.convert_samples([_raw("s/1", img="nope.png")], [tmp_path])
        assert kept == [] and dropped["截图文件不存在"] == 1

    def test_no_actions_dropped(self, tmp_path):
        (tmp_path / "a.png").write_bytes(b"x")
        kept, dropped = bd.convert_samples([_raw("s/1", img="a.png", actions=[{"action_type": "mouse_up"}])],
                                           [tmp_path])
        assert kept == [] and dropped["没有可训练的动作"] == 1

    def test_missing_screen_size_dropped(self, tmp_path):
        kept, dropped = bd.convert_samples([_raw("s/1", size=(0, 0))], [tmp_path], check_images=False)
        assert dropped["屏幕尺寸缺失"] == 1

    def test_val_ratio_zero(self):
        rows = [{"session": f"s{i}", "orig_split": "train", "actions": [{"action": "click"}]} for i in range(5)]
        assert bd.split_samples(rows, 0.0)["val"] == []

    def test_at_least_one_val_session(self):
        rows = [{"session": f"s{i}", "orig_split": "train"} for i in range(3)]
        sp = bd.split_samples(rows, 0.01)
        assert len(sp["val"]) == 1 and len(sp["train"]) == 2

    def test_blank_lines_ignored(self, tmp_path):
        p = tmp_path / "x.jsonl"
        p.write_text('{"a": 1}\n\n{"a": 2}\n', encoding="utf-8")
        assert len(bd.load_jsonl(p)) == 2


@pytest.mark.exception
class TestBuildDatasetInvalid:
    def test_missing_processed_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            bd.build(tmp_path / "nope.jsonl", tmp_path / "out")

    def test_bad_val_ratio(self):
        with pytest.raises(ValueError):
            bd.split_samples([], 1.5)

    def test_corrupt_jsonl_line_reported(self, tmp_path):
        p = tmp_path / "x.jsonl"
        p.write_text('{"a": 1}\n{oops\n', encoding="utf-8")
        with pytest.raises(ValueError, match="第 2 行"):
            bd.load_jsonl(p)


@pytest.mark.functional
@pytest.mark.skipif(not REAL_DATA.exists(), reason="需要先运行 python -m finetune.build_dataset --prepare")
def test_real_screenagent_split_has_282_test_samples():
    rows = bd.load_jsonl(REAL_DATA)
    assert len(rows) == 282 and all(r["actions"] for r in rows)


# --------------------------------------------------------------------------- #
# 评测指标
# --------------------------------------------------------------------------- #
CLICK = {"action": "click", "x": 500, "y": 500}


@pytest.mark.functional
class TestScoring:
    def test_exact_click(self):
        r = score_sample([CLICK], [CLICK], (1024, 768))
        assert r["first_ok"] and r["step_ok"] and r["dist_px"] == 0 and r["hit@25"]

    def test_near_click_within_threshold(self):
        r = score_sample([{"action": "click", "x": 530, "y": 500}], [CLICK], (1024, 768))
        assert round(r["dist_px"], 1) == 30.7 and r["hit@50"] and not r["hit@25"] and r["first_ok"]

    def test_far_click(self):
        r = score_sample([{"action": "click", "x": 900, "y": 900}], [CLICK], (1024, 768))
        assert r["type_ok"] and not r["first_ok"] and not r["hit@100"]

    def test_wrong_type_still_measures_distance(self):
        r = score_sample([{"action": "double_click", "x": 500, "y": 500}], [CLICK], (1024, 768))
        assert not r["type_ok"] and r["hit@25"]

    def test_type_text_case_and_space_insensitive(self):
        gt = [{"action": "type", "text": "Hello  World"}]
        assert score_sample([{"action": "type", "text": " hello world"}], gt)["text_ok"]

    def test_key_order_insensitive(self):
        gt = [{"action": "key", "keys": ["ctrl", "a"]}]
        assert score_sample([{"action": "key", "keys": ["a", "ctrl"]}], gt)["key_ok"]

    def test_step_requires_all_actions(self):
        gt = [CLICK, {"action": "type", "text": "x"}]
        r = score_sample([CLICK], gt)
        assert r["first_ok"] and not r["step_ok"]

    def test_text_found_later_in_sequence(self):
        gt = [CLICK, {"action": "type", "text": "x"}]
        assert score_sample([CLICK, {"action": "type", "text": "x"}], gt)["text_ok"]

    def test_scroll_direction(self):
        assert action_match({"action": "scroll", "amount": -1}, {"action": "scroll", "amount": -5})
        assert not action_match({"action": "scroll", "amount": 2}, {"action": "scroll", "amount": -5})

    def test_drag_needs_both_points(self):
        gt = {"action": "drag", "x": 0, "y": 0, "x2": 500, "y2": 500}
        assert action_match(dict(gt), gt)
        assert not action_match({**gt, "x2": 900}, gt, (1024, 768))

    def test_point_distance_uses_screen_size(self):
        assert point_distance({"x": 0, "y": 0}, {"x": 1000, "y": 0}, (2560, 1440)) == 2560

    def test_summary_rates(self):
        recs = [score_sample([CLICK], [CLICK]), score_sample([], [CLICK], parsed_ok=False)]
        s = summarize(recs)
        assert s["n"] == 2 and s["format_rate"] == 0.5 and s["first_action_acc"] == 0.5 and s["n_point"] == 2

    def test_summary_per_type_and_confusion(self):
        recs = [score_sample([{"action": "key", "keys": ["a"]}], [CLICK]),
                score_sample([{"action": "type", "text": "a"}], [{"action": "type", "text": "a"}])]
        s = summarize(recs)
        assert s["per_type"]["click"]["type_acc"] == 0 and s["per_type"]["type"]["type_acc"] == 1
        assert s["top_confusions"][0] == {"gt": "click", "pred": "key", "n": 1}


@pytest.mark.boundary
class TestScoringBoundary:
    def test_threshold_inclusive(self):
        gt = {"action": "click", "x": 0, "y": 0}
        assert action_match({"action": "click", "x": 50, "y": 0}, gt, (1000, 1000), threshold=50)

    def test_empty_prediction(self):
        r = score_sample([], [CLICK])
        assert not r["format_ok"] and r["pred_type"] is None and r["dist_px"] is None and not r["hit@100"]

    def test_parsed_but_flagged_not_ok(self):
        assert not score_sample([CLICK], [CLICK], parsed_ok=False)["format_ok"]

    def test_summary_empty(self):
        assert summarize([]) == {"n": 0}

    def test_no_point_samples(self):
        s = summarize([score_sample([{"action": "wait", "seconds": 1}], [{"action": "wait", "seconds": 3}])])
        assert s["n_point"] == 0 and s["click_hit@50"] is None and s["median_dist_px"] is None and s["first_action_acc"] == 1


@pytest.mark.exception
class TestScoringInvalid:
    def test_empty_ground_truth(self):
        with pytest.raises(ValueError):
            score_sample([CLICK], [])

    def test_non_dict_prediction(self):
        assert not action_match("click", CLICK)

    def test_missing_coordinates(self):
        assert point_distance({"x": 1}, CLICK) is None


# --------------------------------------------------------------------------- #
# 批量评测与对比报告
# --------------------------------------------------------------------------- #
def _rows(n=6):
    rows = []
    for i in range(n):
        acts = [{"action": "click", "x": 100 * i, "y": 100}] if i % 2 == 0 else [{"action": "type", "text": f"t{i}"}]
        rows.append({"id": f"r{i}", "image": "x.png", "instruction": "任务", "sub_task": "子任务",
                     "screen_size": [1000, 1000], "actions": acts})
    return rows


@pytest.mark.functional
class TestEvaluateRun:
    def test_const_predictor_picks_majority(self):
        p = ConstPredictor(_rows(5))
        assert p.action["action"] == "click" and p.action["x"] == 200

    def test_replay_perfect(self, tmp_path):
        rows = _rows()
        outs = {r["id"]: json.dumps(r["actions"]) for r in rows}
        s = evaluate(ReplayPredictor(outs), rows, tmp_path / "e", log=lambda *a: None, meta={"label": "perfect"})
        assert s["step_success"] == 1.0 and s["label"] == "perfect"
        assert (tmp_path / "e" / "summary.json").exists()

    def test_resume_skips_done(self, tmp_path):
        rows = _rows()
        outs = {r["id"]: json.dumps(r["actions"]) for r in rows}
        evaluate(ReplayPredictor(outs), rows[:3], tmp_path / "e", log=lambda *a: None)
        calls = []

        class Spy(ReplayPredictor):
            def predict(self, *a, **k):
                calls.append(k.get("sample_id"))
                return super().predict(*a, **k)
        s = evaluate(Spy(outs), rows, tmp_path / "e", log=lambda *a: None)
        assert calls == ["r3", "r4", "r5"] and s["n"] == 6

    def test_limit(self):
        s = evaluate(ConstPredictor(_rows()), _rows(), None, limit=2, log=lambda *a: None)
        assert s["n"] == 2

    def test_compare_table_and_plot(self, tmp_path):
        rows = _rows()
        good = {r["id"]: json.dumps(r["actions"]) for r in rows}
        evaluate(ReplayPredictor(good), rows, tmp_path / "eval" / "lora_v1", log=lambda *a: None)
        evaluate(ReplayPredictor({}), rows, tmp_path / "eval" / "base_v0", log=lambda *a: None)
        sums = load_summaries([tmp_path / "eval"])
        table = compare_table(sums)
        assert "lora_v1" in table and "base_v0" in table and "100.0%" in table
        png = plot_compare(sums, tmp_path / "cmp.png")
        assert png is None or png.exists()


@pytest.mark.exception
class TestEvaluateErrors:
    def test_predictor_exception_recorded(self):
        class Boom:
            def predict(self, *a, **k):
                raise RuntimeError("CUDA out of memory")
        s = evaluate(Boom(), _rows(2), None, log=lambda *a: None)
        assert s["n"] == 2 and s["format_rate"] == 0.0

    def test_const_needs_train(self):
        with pytest.raises(ValueError):
            ConstPredictor([])

    def test_compare_empty(self):
        with pytest.raises(ValueError):
            compare_table([])
