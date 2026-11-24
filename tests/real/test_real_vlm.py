"""第5周：真实 Qwen2-VL（+ LoRA）在 GPU 上的用例（WSL2，gui-agent 环境）

    export GUI_AGENT_VLM_MODEL=Qwen/Qwen2-VL-2B-Instruct
    export GUI_AGENT_VLM_ADAPTER=outputs/lora_qwen2vl_2b/best     # 不设则只测基座
    pytest tests/real/test_real_vlm.py -m gpu -v -s

前提：已运行 python -m finetune.build_dataset --prepare（需要 data_cache/sft/test.jsonl）。
"""
import json
import os
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.gpu

REPO = Path(__file__).resolve().parents[2]
TEST_JSONL = REPO / "data_cache" / "sft" / "test.jsonl"
N = int(os.environ.get("GUI_AGENT_VLM_N", "30"))


def _rows():
    from finetune.build_dataset import load_jsonl
    if not TEST_JSONL.exists():
        pytest.skip("缺少 data_cache/sft/test.jsonl，先运行 python -m finetune.build_dataset --prepare")
    return load_jsonl(TEST_JSONL)[:N]


@pytest.fixture(scope="module")
def base_actor():
    from finetune.infer import VLMActor
    return VLMActor.from_pretrained(os.environ["GUI_AGENT_VLM_MODEL"])


@pytest.fixture(scope="module")
def lora_actor():
    adapter = os.environ.get("GUI_AGENT_VLM_ADAPTER")
    if not adapter:
        pytest.skip("未设置 GUI_AGENT_VLM_ADAPTER")
    from finetune.infer import VLMActor
    return VLMActor.from_pretrained(os.environ["GUI_AGENT_VLM_MODEL"], adapter=adapter)


@pytest.mark.functional
def test_gpu_available():
    import torch
    assert torch.cuda.is_available(), "torch 没有识别到 GPU"


@pytest.mark.functional
def test_base_model_answers_in_text(base_actor):
    r = _rows()[0]
    out = base_actor.predict(r["image"], r["instruction"], r["sub_task"])
    print("\n基座输出:", out["raw"][:200])
    assert isinstance(out["raw"], str) and out["raw"].strip()


@pytest.mark.functional
@pytest.mark.parametrize("prompt", ["v0_minimal", "v1_schema", "v2_fewshot"])
def test_base_prompt_variants_run(base_actor, prompt):
    from finetune.evaluate import evaluate
    s = evaluate(base_actor, _rows()[:10], None, prompt=prompt, log=lambda *a: None)
    print(f"\n基座 {prompt}: 格式 {s['format_rate']}  类型 {s['type_acc']}  首动作 {s['first_action_acc']}")
    assert s["n"] == 10


@pytest.mark.functional
def test_lora_format_rate(lora_actor):
    from finetune.evaluate import evaluate
    s = evaluate(lora_actor, _rows(), None, log=lambda *a: None)
    print("\nLoRA:", json.dumps({k: v for k, v in s.items() if k not in ("per_type", "top_confusions")},
                                ensure_ascii=False))
    assert s["format_rate"] >= 0.9, "微调后输出格式应几乎全部合格"


@pytest.mark.functional
def test_lora_beats_base_on_first_action(base_actor, lora_actor):
    from finetune.evaluate import evaluate
    rows = _rows()
    b = evaluate(base_actor, rows, None, prompt="v1_schema", log=lambda *a: None)
    lo = evaluate(lora_actor, rows, None, log=lambda *a: None)
    print(f"\n首动作正确率 基座 {b['first_action_acc']} → LoRA {lo['first_action_acc']}")
    assert lo["first_action_acc"] >= b["first_action_acc"]


@pytest.mark.boundary
def test_lora_latency(lora_actor):
    r = _rows()[0]
    lora_actor.predict(r["image"], r["instruction"], r["sub_task"])  # 预热
    t0 = time.time()
    lora_actor.predict(r["image"], r["instruction"], r["sub_task"])
    sec = time.time() - t0
    print(f"\n单步推理 {sec:.2f}s")
    assert sec < 10


@pytest.mark.multi_resolution
@pytest.mark.parametrize("size", [(1920, 1080), (2560, 1440)])
def test_lora_on_upscaled_screens(lora_actor, size):
    """ScreenAgent 截图是 1024×768；放大到真机分辨率后，归一化坐标输出仍应可解析"""
    from PIL import Image
    r = _rows()[0]
    img = Image.open(r["image"]).convert("RGB").resize(size)
    out = lora_actor.predict(img, r["instruction"], r["sub_task"])
    assert out["ok"], out["raw"]
