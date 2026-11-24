"""第5周：用随机初始化的微型 Qwen2-VL 在 CPU 上跑真实的 批处理 → LoRA → 训练 → 保存/加载 → 生成 → 解析 → 评测

需要 torch + transformers + peft（WSL2 的 gui-agent 环境、云端 CI）；Windows 端 gui-agent-win 没装 torch 时自动跳过。
标记 torch_model 的用例真正执行模型前向/反向；没有该标记的是纯 Python 部分（正则、学习率曲线、配置校验）。
"""
import json
import math
from pathlib import Path

import pytest

from finetune.modeling import LORA_TARGETS, lora_target_regex, model_class_for, resolve_dtype
from finetune.train_lora import TrainConfig, batches, lr_lambda_factory, parse_args

import re


# --------------------------------------------------------------------------- #
# 纯 Python 部分
# --------------------------------------------------------------------------- #
@pytest.mark.functional
class TestLoraTargets:
    @pytest.mark.parametrize("name", [
        "model.layers.0.self_attn.q_proj",                       # transformers 4.45~4.51 的命名
        "model.language_model.layers.27.mlp.down_proj",          # 4.52+ 的命名
        "base_model.model.model.layers.3.self_attn.o_proj",
    ])
    def test_language_model_layers_matched(self, name):
        assert re.fullmatch(lora_target_regex(), name)

    @pytest.mark.parametrize("name", [
        "visual.blocks.0.mlp.gate_proj",                         # Qwen2.5-VL 视觉 MLP 同名，必须排除
        "model.visual.blocks.31.mlp.down_proj",
        "visual.blocks.0.attn.qkv",
        "model.language_model.layers.0.self_attn.rotary_emb",
        "lm_head",
    ])
    def test_vision_and_others_excluded(self, name):
        assert not re.fullmatch(lora_target_regex(), name)

    def test_seven_target_kinds(self):
        assert len(LORA_TARGETS) == 7


@pytest.mark.functional
class TestSchedule:
    def test_warmup_then_cosine(self):
        f = lr_lambda_factory(100, 10)
        assert f(0) == pytest.approx(0.1) and f(9) == pytest.approx(1.0)
        assert f(55) == pytest.approx(0.5, abs=0.01) and f(100) == pytest.approx(0.0, abs=1e-9)

    def test_monotonic_after_warmup(self):
        f = lr_lambda_factory(50, 5)
        vals = [f(i) for i in range(5, 50)]
        assert all(a >= b for a, b in zip(vals, vals[1:]))

    def test_batches_cover_all(self):
        import random
        bl = batches(list(range(7)), 3, random.Random(0))
        assert [len(b) for b in bl] == [3, 3, 1] and sorted(sum(bl, [])) == list(range(7))

    def test_parse_args_defaults(self):
        c = parse_args([])
        assert c.model == "Qwen/Qwen2-VL-2B-Instruct" and c.lora_r == 16 and c.prompt == "v1_schema"
        assert c.gradient_checkpointing is True


@pytest.mark.boundary
class TestScheduleBoundary:
    def test_single_step(self):
        f = lr_lambda_factory(1, 5)
        assert f(0) == pytest.approx(1.0)

    def test_zero_warmup(self):
        assert lr_lambda_factory(10, 0)(0) == pytest.approx(1.0)

    def test_zero_total_treated_as_one(self):
        assert lr_lambda_factory(0, 0)(0) == pytest.approx(1.0)

    def test_empty_batches(self):
        assert batches([], 4) == []


@pytest.mark.exception
class TestConfigErrors:
    @pytest.mark.parametrize("kw", [{"epochs": 0}, {"batch_size": 0}, {"grad_accum": -1}, {"lr": 0}, {"lr": 5},
                                    {"warmup_ratio": 1.0}])
    def test_invalid_config(self, kw):
        with pytest.raises(ValueError):
            TrainConfig(**kw).validate()

    def test_unknown_dtype(self):
        pytest.importorskip("torch")
        with pytest.raises(ValueError):
            resolve_dtype("int3")

    def test_unknown_model_type_falls_back_to_auto(self):
        pytest.importorskip("transformers")
        assert model_class_for("something_new").__name__.startswith("AutoModel")


# --------------------------------------------------------------------------- #
# 微型模型（真实前向 / 反向 / 生成）
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def tiny():
    import tiny_vlm as tv
    proc = tv.build_processor()
    return tv, proc


@pytest.fixture
def rows(tmp_path):
    import tiny_vlm as tv
    return tv.make_screens(tmp_path / "imgs", n=6)


def _collator(proc, **kw):
    from finetune.collator import SFTCollator
    return SFTCollator(proc, max_length=4096, **kw)


@pytest.mark.torch_model
@pytest.mark.functional
class TestCollator:
    def test_labels_mask_prompt(self, tiny, rows):
        tv, proc = tiny
        b = _collator(proc)([rows[0]])
        ids, labels = b["input_ids"][0], b["labels"][0]
        n_ans = int((labels != -100).sum())
        ans = proc.tokenizer(rows[0]["target"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
        assert n_ans == len(ans) and labels[-n_ans:].tolist() == ans and ids[-n_ans:].tolist() == ans

    def test_prompt_prefix_matches_inference(self, tiny, rows):
        """训练时的前缀必须与推理时 add_generation_prompt=True 的输入完全一致"""
        from finetune.collator import prompt_text
        tv, proc = tiny
        img = __import__("PIL.Image", fromlist=["Image"]).open(rows[0]["image"]).convert("RGB")
        infer_ids = proc(text=[prompt_text(proc, rows[0])], images=[img], return_tensors="pt")["input_ids"][0]
        b = _collator(proc)([rows[0]])
        assert b["input_ids"][0][:len(infer_ids)].tolist() == infer_ids.tolist()
        assert (b["labels"][0][:len(infer_ids)] == -100).all()

    def test_image_tokens_match_grid(self, tiny, rows):
        tv, proc = tiny
        b = _collator(proc)([rows[0]])
        pad_id = proc.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        t, h, w = b["image_grid_thw"][0].tolist()
        assert int((b["input_ids"][0] == pad_id).sum()) == t * h * w // 4
        assert b["pixel_values"].shape[0] == t * h * w

    def test_batch_right_padding(self, tiny, rows):
        tv, proc = tiny
        b = _collator(proc)([rows[0], rows[1]])  # rows[1] 是输入+回车，回答更长
        assert b["input_ids"].shape[0] == 2 and b["image_grid_thw"].shape[0] == 2
        short = 0 if b["attention_mask"][0].sum() < b["attention_mask"][1].sum() else 1
        assert b["attention_mask"][short][-1] == 0 and b["labels"][short][-1] == -100

    def test_left_padding(self, tiny, rows):
        tv, proc = tiny
        proc.tokenizer.padding_side = "left"
        try:
            b = _collator(proc)([rows[0], rows[1]])
            assert 0 in (b["attention_mask"][0][0].item(), b["attention_mask"][1][0].item())
        finally:
            proc.tokenizer.padding_side = "right"

    @pytest.mark.parametrize("variant", ["v0_minimal", "v1_schema", "v2_fewshot"])
    def test_each_prompt_variant_encodes(self, tiny, rows, variant):
        tv, proc = tiny
        b = _collator(proc, variant=variant)([rows[0]])
        assert b["input_ids"].shape == b["labels"].shape


@pytest.mark.torch_model
@pytest.mark.exception
class TestCollatorErrors:
    def test_too_long(self, tiny, rows):
        tv, proc = tiny
        with pytest.raises(ValueError, match="max_length"):
            _collator(proc).__class__(proc, max_length=16)([rows[0]])

    def test_missing_target(self, tiny, rows):
        tv, proc = tiny
        with pytest.raises(ValueError):
            _collator(proc)([{**rows[0], "target": ""}])

    def test_empty_batch(self, tiny):
        tv, proc = tiny
        with pytest.raises(ValueError):
            _collator(proc)([])

    def test_missing_image_file(self, tiny, rows):
        tv, proc = tiny
        with pytest.raises(FileNotFoundError):
            _collator(proc)([{**rows[0], "image": "/nope/x.png"}])


@pytest.mark.torch_model
@pytest.mark.functional
class TestLoraOnTinyModel:
    def test_lora_only_on_language_model(self, tiny):
        from finetune.modeling import apply_lora, count_parameters, lora_module_names
        tv, proc = tiny
        m = apply_lora(tv.build_model(proc), r=4, alpha=8)
        names = lora_module_names(m)
        assert len(names) == 2 * 7 and not any("visual" in n for n in names)
        p = count_parameters(m)
        assert 0 < p["trainable"] < p["total"] * 0.2

    def test_base_weights_frozen(self, tiny):
        from finetune.modeling import apply_lora
        tv, proc = tiny
        m = apply_lora(tv.build_model(proc), r=4)
        trainable = [n for n, p in m.named_parameters() if p.requires_grad]
        assert trainable and all("lora_" in n for n in trainable)

    def test_gradient_checkpointing_trains(self, tiny, rows, tmp_path):
        from finetune.modeling import apply_lora
        from finetune.train_lora import run_training
        tv, proc = tiny
        m = apply_lora(tv.build_model(proc), r=4, gradient_checkpointing=True)
        s = run_training(TrainConfig(out=str(tmp_path), epochs=1, grad_accum=2, max_length=4096), m, proc,
                         rows[:2], [], log=lambda *a: None, save=False)
        assert s["optimizer_steps"] == 1 and math.isfinite(s["epochs"][0]["train_loss"])


@pytest.mark.torch_model
@pytest.mark.functional
class TestTrainingLoop:
    def test_lora_training_reduces_loss_and_saves(self, tiny, rows, tmp_path):
        from finetune.modeling import apply_lora
        from finetune.train_lora import run_training
        tv, proc = tiny
        m = apply_lora(tv.build_model(proc), r=16, alpha=32, dropout=0.0)
        cfg = TrainConfig(out=str(tmp_path / "out"), epochs=8, lr=1e-2, grad_accum=2, warmup_ratio=0.0,
                          log_every=1, max_length=4096)
        s = run_training(cfg, m, proc, rows[:4], rows[4:], log=lambda *a: None)
        ep = s["epochs"]
        assert ep[-1]["train_loss"] < ep[0]["train_loss"] and len(ep) == 8
        assert s["best_epoch"] >= 1 and s["best_val_loss"] is not None
        for sub in ("best", "final"):
            assert (tmp_path / "out" / sub / "adapter_config.json").exists()
        log = [json.loads(x) for x in (tmp_path / "out" / "train_log.jsonl").read_text().splitlines()]
        assert len(log) == s["optimizer_steps"] == 16
        summary = json.loads((tmp_path / "out" / "train_summary.json").read_text(encoding="utf-8"))
        assert summary["lora_modules"] == 14 and summary["train_samples"] == 4

    def test_adapter_reload_gives_same_logits(self, tiny, rows, tmp_path):
        import torch
        from peft import PeftModel
        from finetune.modeling import apply_lora
        from finetune.train_lora import run_training
        tv, proc = tiny
        m = apply_lora(tv.build_model(proc, seed=3), r=8, dropout=0.0)
        run_training(TrainConfig(out=str(tmp_path / "o"), epochs=2, lr=1e-2, grad_accum=1, max_length=4096), m, proc,
                     rows[:2], [], log=lambda *a: None)
        batch = _collator(proc)([rows[0]])
        m.eval()
        with torch.no_grad():
            a = m(**batch).logits
            fresh = PeftModel.from_pretrained(tv.build_model(proc, seed=3), str(tmp_path / "o" / "final")).eval()
            b = fresh(**batch).logits
            merged = fresh.merge_and_unload()(**batch).logits
        assert torch.allclose(a, b, atol=1e-5) and torch.allclose(a, merged, atol=1e-4)

    def test_memorize_and_generate_exact_action(self, tiny, rows, tmp_path):
        """训练后能逐字生成目标 JSON，并被解析、评测为完全正确（端到端：训练 → 生成 → 解析 → 打分）"""
        from finetune.evaluate import evaluate
        from finetune.infer import VLMActor
        from finetune.train_lora import run_training
        tv, proc = tiny
        m = tv.build_model(proc)  # 全参数（微型模型 LoRA 容量太小，记不住整句）
        run_training(TrainConfig(out=str(tmp_path), epochs=80, lr=5e-3, grad_accum=1, warmup_ratio=0.0,
                                 max_length=4096), m, proc, rows[:1], [], log=lambda *a: None, save=False)
        actor = VLMActor(m, proc, max_new_tokens=60)
        r = actor.predict(rows[0]["image"], rows[0]["instruction"], rows[0]["sub_task"])
        assert r["ok"] and r["actions"] == rows[0]["actions"], r["raw"]
        s = evaluate(actor, rows[:1], tmp_path / "eval", log=lambda *a: None)
        assert s["step_success"] == 1.0 and s["click_hit@25"] == 1.0


@pytest.mark.torch_model
@pytest.mark.exception
class TestTrainingErrors:
    def test_empty_train(self, tiny, tmp_path):
        from finetune.train_lora import run_training
        tv, proc = tiny
        with pytest.raises(ValueError, match="训练集为空"):
            run_training(TrainConfig(out=str(tmp_path)), tv.build_model(proc), proc, [], [], save=False)

    def test_no_trainable_params(self, tiny, rows, tmp_path):
        from finetune.train_lora import run_training
        tv, proc = tiny
        m = tv.build_model(proc)
        for p in m.parameters():
            p.requires_grad_(False)
        with pytest.raises(ValueError, match="可训练参数"):
            run_training(TrainConfig(out=str(tmp_path)), m, proc, rows[:1], [], save=False)

    def test_nan_loss_detected(self, tiny, rows, tmp_path):
        import torch
        from finetune.train_lora import run_training
        tv, proc = tiny
        m = tv.build_model(proc)
        orig = m.forward

        def bad(*a, **k):
            out = orig(*a, **k)
            out.loss = out.loss * torch.tensor(float("nan"))
            return out
        m.forward = bad
        with pytest.raises(FloatingPointError):
            run_training(TrainConfig(out=str(tmp_path), max_length=4096), m, proc, rows[:1], [], save=False,
                         log=lambda *a: None)

    def test_r_zero(self, tiny):
        from finetune.modeling import apply_lora
        tv, proc = tiny
        with pytest.raises(ValueError):
            apply_lora(tv.build_model(proc), r=0)

    def test_min_pixels_gt_max(self):
        from finetune.modeling import load_processor
        with pytest.raises(ValueError):
            load_processor("whatever", min_pixels=10, max_pixels=5)


@pytest.mark.torch_model
@pytest.mark.functional
class TestFromPretrainedPipeline:
    """从本地目录加载（与真实 Qwen/Qwen2-VL-2B-Instruct 走同一条 from_pretrained 路径）"""

    @pytest.fixture(scope="class")
    def ckpt(self, tmp_path_factory):
        import tiny_vlm as tv
        return tv.save_tiny_checkpoint(tmp_path_factory.mktemp("tiny_ckpt"))

    def test_load_processor_sets_pixels(self, ckpt):
        from finetune.modeling import load_processor
        p = load_processor(str(ckpt), min_pixels=28 * 28, max_pixels=56 * 56)
        assert p.image_processor.max_pixels == 56 * 56

    def test_load_base_model_class(self, ckpt):
        from finetune.modeling import load_base_model
        m = load_base_model(str(ckpt), dtype="fp32")
        assert type(m).__name__ == "Qwen2VLForConditionalGeneration"

    def test_train_cli_end_to_end(self, ckpt, tmp_path):
        import tiny_vlm as tv
        from finetune.build_dataset import write_jsonl
        from finetune.train_lora import main
        rows = tv.make_screens(tmp_path / "imgs", n=4)
        write_jsonl(tmp_path / "sft" / "train.jsonl", rows[:3])
        write_jsonl(tmp_path / "sft" / "val.jsonl", rows[3:])
        s = main(["--model", str(ckpt), "--data-dir", str(tmp_path / "sft"), "--out", str(tmp_path / "out"),
                  "--epochs", "1", "--grad-accum", "1", "--lora-r", "4", "--dtype", "fp32", "--max-length", "4096",
                  "--min-pixels", str(56 * 56), "--max-pixels", str(112 * 112), "--no-gradient-checkpointing"])
        assert s["train_samples"] == 3 and (tmp_path / "out" / "best" / "adapter_model.safetensors").exists()
        TestFromPretrainedPipeline.adapter = tmp_path / "out" / "best"
        TestFromPretrainedPipeline.rows = rows

    def test_actor_base_and_lora(self, ckpt):
        from finetune.infer import VLMActor
        rows = TestFromPretrainedPipeline.rows
        base = VLMActor.from_pretrained(str(ckpt), dtype="fp32", min_pixels=56 * 56, max_pixels=112 * 112,
                                        max_new_tokens=8)
        lora = VLMActor.from_pretrained(str(ckpt), adapter=str(TestFromPretrainedPipeline.adapter), dtype="fp32",
                                        min_pixels=56 * 56, max_pixels=112 * 112, max_new_tokens=8)
        rb = base.predict(rows[0]["image"], "任务", "子任务")
        rl = lora.predict(rows[0]["image"], "任务", "子任务")
        assert base.name == "base" and lora.name == "lora"
        assert isinstance(rb["raw"], str) and isinstance(rl["raw"], str) and rb["sec"] >= 0
        assert not hasattr(lora.model, "peft_config")  # 已合并

    def test_evaluate_cli_with_model(self, ckpt, tmp_path):
        from finetune.build_dataset import write_jsonl
        from finetune.evaluate import main
        rows = TestFromPretrainedPipeline.rows
        write_jsonl(tmp_path / "sft" / "test.jsonl", rows[:2])
        s = main(["--model", str(ckpt), "--data-dir", str(tmp_path / "sft"), "--out", str(tmp_path / "e"),
                  "--prompt", "v2_fewshot", "--max-pixels", str(112 * 112)])
        assert s["n"] == 2 and s["prompt"] == "v2_fewshot" and s["label"] == "base_v2_fewshot"
        assert len((tmp_path / "e" / "predictions.jsonl").read_text().splitlines()) == 2

    def test_serve_with_real_actor(self, ckpt):
        import base64
        import io
        import threading
        import urllib.request
        from PIL import Image
        from finetune.infer import VLMActor
        from finetune.serve import make_server
        actor = VLMActor.from_pretrained(str(ckpt), dtype="fp32", min_pixels=56 * 56, max_pixels=112 * 112,
                                         max_new_tokens=4)
        srv = make_server(actor, "127.0.0.1", 0)
        th = threading.Thread(target=srv.serve_forever, daemon=True)
        th.start()
        try:
            buf = io.BytesIO()
            Image.new("RGB", (200, 150), "green").save(buf, format="PNG")
            body = json.dumps({"image_b64": base64.b64encode(buf.getvalue()).decode(), "instruction": "点击"}).encode()
            req = urllib.request.Request(f"http://127.0.0.1:{srv.server_address[1]}/act", data=body, method="POST")
            with urllib.request.urlopen(req, timeout=30) as r:
                out = json.loads(r.read())
            assert r.status == 200 and "raw" in out and isinstance(out["actions"], list)
        finally:
            srv.shutdown()
            srv.server_close()


@pytest.mark.torch_model
@pytest.mark.multi_resolution
@pytest.mark.parametrize("size", [(1024, 768), (1920, 1080), (2560, 1440), (800, 1280)])
def test_collator_any_screen_size_within_pixel_budget(tiny, tmp_path, size):
    """不同分辨率/宽高比的截图都被缩放到 max_pixels 以内，视觉 token 数有上限"""
    import tiny_vlm as tv
    tv_, proc = tiny
    rows = tv.make_screens(tmp_path, n=1, size=size)
    b = _collator(proc)(rows)
    t, h, w = b["image_grid_thw"][0].tolist()
    assert h * w * 14 * 14 <= proc.image_processor.max_pixels * 1.05
