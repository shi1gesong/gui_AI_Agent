"""
模型加载与 LoRA 配置（第5周）

默认基座：Qwen/Qwen2-VL-2B-Instruct（bf16 约 4.5GB 显存，RTX 5070 Ti 16GB 可直接 LoRA；
         也支持 Qwen/Qwen2.5-VL-3B-Instruct，显存紧张时加 --load-4bit 走 QLoRA）

LoRA 只挂在语言模型的注意力与 MLP 线性层上（q/k/v/o/gate/up/down），视觉编码器冻结：
  - 数据只有 800 条，训练视觉编码器容易过拟合
  - Qwen2.5-VL 的视觉 MLP 也叫 gate_proj/up_proj/down_proj，所以用正则排除 visual 分支
"""
import re
from typing import Optional

DEFAULT_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
# 截图像素上限：1024×768 的 ScreenAgent 截图约 1000 个 patch → 合并后约 250 个视觉 token
DEFAULT_MIN_PIXELS = 256 * 28 * 28
DEFAULT_MAX_PIXELS = 1024 * 28 * 28


def lora_target_regex(targets=LORA_TARGETS) -> str:
    """PEFT 对字符串 target_modules 做 re.fullmatch：匹配名字以这些层结尾、且路径中不含 visual 的模块"""
    names = "|".join(re.escape(t) for t in targets)
    return rf"^(?!.*visual).*\.({names})$"


def model_class_for(model_type: str):
    import transformers
    mt = (model_type or "").lower()
    if mt == "qwen2_5_vl":
        return transformers.Qwen2_5_VLForConditionalGeneration
    if mt == "qwen2_vl":
        return transformers.Qwen2VLForConditionalGeneration
    for name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq"):
        cls = getattr(transformers, name, None)
        if cls is not None:
            return cls
    raise ValueError(f"不支持的模型类型 {model_type!r}")


def resolve_dtype(name: str):
    import torch
    name = (name or "auto").lower()
    if name == "auto":
        if torch.cuda.is_available():
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32
    table = {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16, "fp16": torch.float16, "float16": torch.float16,
             "fp32": torch.float32, "float32": torch.float32}
    if name not in table:
        raise ValueError(f"不支持的 dtype {name!r}，可选 auto/bf16/fp16/fp32")
    return table[name]


def load_processor(model_id: str, min_pixels: int = DEFAULT_MIN_PIXELS, max_pixels: int = DEFAULT_MAX_PIXELS):
    from transformers import AutoProcessor
    if min_pixels > max_pixels:
        raise ValueError(f"min_pixels({min_pixels}) 不能大于 max_pixels({max_pixels})")
    proc = AutoProcessor.from_pretrained(model_id, min_pixels=min_pixels, max_pixels=max_pixels)
    ip = getattr(proc, "image_processor", None)
    if ip is not None:  # 部分版本不认 from_pretrained 的关键字参数，这里再设一次
        ip.min_pixels, ip.max_pixels = min_pixels, max_pixels
        if hasattr(ip, "size") and isinstance(ip.size, dict):
            ip.size = {**ip.size, "shortest_edge": min_pixels, "longest_edge": max_pixels}
    return proc


def load_base_model(model_id: str, dtype: str = "auto", load_4bit: bool = False, device_map: Optional[str] = "auto",
                    attn_implementation: Optional[str] = None):
    import torch
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_id)
    cls = model_class_for(getattr(cfg, "model_type", ""))
    kw = {"torch_dtype": resolve_dtype(dtype)}
    if attn_implementation:
        kw["attn_implementation"] = attn_implementation
    if load_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=kw["torch_dtype"] if kw["torch_dtype"] != torch.float32 else torch.bfloat16)
    if device_map and torch.cuda.is_available():
        kw["device_map"] = device_map
    return cls.from_pretrained(model_id, **kw)


def apply_lora(model, r: int = 16, alpha: int = 32, dropout: float = 0.05, targets=LORA_TARGETS,
               gradient_checkpointing: bool = False, quantized: bool = False):
    from peft import LoraConfig, get_peft_model
    if r <= 0:
        raise ValueError(f"LoRA 秩 r 必须为正整数: {r}")
    if quantized:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)
    elif gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=lora_target_regex(targets),
                     bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    return model


def count_parameters(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "ratio": round(trainable / total, 6) if total else 0.0}


def lora_module_names(model) -> list:
    """实际挂上 LoRA 的模块名（用于检查视觉编码器确实没被改动）"""
    return sorted({n.rsplit(".lora_A", 1)[0] for n, _ in model.named_modules() if ".lora_A" in n})
