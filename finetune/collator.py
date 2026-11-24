"""
多模态 SFT 批处理（第5周）

每条样本 → 提示词（图片 + 文字，带生成提示 "<|im_start|>assistant\\n"）+ 目标 JSON + 结束符。
input_ids = 提示词部分（processor 处理图片，展开成若干 <|image_pad|>）+ 回答部分（tokenizer 直接编码）
labels    = 提示词部分全部置 -100，只对回答和结束符计算损失

这样训练时模型看到的前缀与推理时 apply_chat_template(add_generation_prompt=True) 完全一致。
"""
from pathlib import Path
from typing import Dict, List, Optional

try:
    from .prompts import DEFAULT_VARIANT, build_messages
except ImportError:
    from prompts import DEFAULT_VARIANT, build_messages

IGNORE_INDEX = -100


def load_image(path, max_side: Optional[int] = None):
    from PIL import Image
    img = Image.open(path).convert("RGB")
    if max_side and max(img.size) > max_side:
        r = max_side / max(img.size)
        img = img.resize((max(1, int(img.width * r)), max(1, int(img.height * r))))
    return img


def prompt_text(processor, sample: dict, variant: str = DEFAULT_VARIANT) -> str:
    msgs = build_messages(sample.get("instruction", ""), sample.get("sub_task"), variant)
    return processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def end_text(processor) -> str:
    """回答结束符：Qwen 系列为 <|im_end|>，取 tokenizer 的 eos"""
    tok = getattr(processor, "tokenizer", processor)
    return getattr(tok, "eos_token", None) or "<|im_end|>"


class SFTCollator:
    def __init__(self, processor, variant: str = DEFAULT_VARIANT, max_length: int = 2048,
                 image_loader=load_image, max_side: Optional[int] = None):
        self.processor = processor
        self.tokenizer = getattr(processor, "tokenizer", processor)
        self.variant = variant
        self.max_length = max_length
        self.image_loader = image_loader
        self.max_side = max_side
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = self.tokenizer.eos_token_id
        if pad is None:
            raise ValueError("tokenizer 没有 pad_token_id 也没有 eos_token_id")
        self.pad_id = int(pad)

    def encode_one(self, sample: dict) -> Dict:
        import torch
        if not sample.get("target"):
            raise ValueError(f"样本 {sample.get('id')} 没有 target")
        image = self.image_loader(sample["image"], self.max_side) if self.max_side else self.image_loader(sample["image"])
        ptxt = prompt_text(self.processor, sample, self.variant)
        enc = self.processor(text=[ptxt], images=[image], return_tensors="pt")
        p_ids = enc["input_ids"][0]
        a_ids = self.tokenizer(sample["target"] + end_text(self.processor), add_special_tokens=False,
                               return_tensors="pt")["input_ids"][0]
        ids = torch.cat([p_ids, a_ids])
        labels = torch.cat([torch.full_like(p_ids, IGNORE_INDEX), a_ids.clone()])
        if ids.numel() > self.max_length:
            # 不能截掉图片 token（会与 pixel_values 对不上），所以超长直接报错，让调用方调小 max_pixels
            raise ValueError(f"样本 {sample.get('id')} 长度 {ids.numel()} 超过 max_length={self.max_length}，"
                             f"请减小 --max-pixels 或增大 --max-length")
        out = {"input_ids": ids, "labels": labels, "attention_mask": torch.ones_like(ids),
               "pixel_values": enc["pixel_values"], "image_grid_thw": enc["image_grid_thw"],
               "prompt_len": int(p_ids.numel()), "answer_len": int(a_ids.numel())}
        return out

    def __call__(self, samples: List[dict]) -> Dict:
        import torch
        if not samples:
            raise ValueError("空批次")
        items = [self.encode_one(s) for s in samples]
        L = max(it["input_ids"].numel() for it in items)
        side = getattr(self.tokenizer, "padding_side", "right") or "right"
        ids, labels, mask = [], [], []
        for it in items:
            n = L - it["input_ids"].numel()
            pad_ids = torch.full((n,), self.pad_id, dtype=it["input_ids"].dtype)
            pad_lab = torch.full((n,), IGNORE_INDEX, dtype=it["labels"].dtype)
            pad_msk = torch.zeros((n,), dtype=it["attention_mask"].dtype)
            if side == "left":
                ids.append(torch.cat([pad_ids, it["input_ids"]]))
                labels.append(torch.cat([pad_lab, it["labels"]]))
                mask.append(torch.cat([pad_msk, it["attention_mask"]]))
            else:
                ids.append(torch.cat([it["input_ids"], pad_ids]))
                labels.append(torch.cat([it["labels"], pad_lab]))
                mask.append(torch.cat([it["attention_mask"], pad_msk]))
        return {"input_ids": torch.stack(ids), "labels": torch.stack(labels), "attention_mask": torch.stack(mask),
                "pixel_values": torch.cat([it["pixel_values"] for it in items], dim=0),
                "image_grid_thw": torch.cat([it["image_grid_thw"] for it in items], dim=0)}


def sample_exists(sample: dict) -> bool:
    return bool(sample.get("image")) and Path(sample["image"]).exists()
