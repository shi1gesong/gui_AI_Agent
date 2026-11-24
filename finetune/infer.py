"""
VLMActor：截图 + 指令 → 动作（第5周）

    actor = VLMActor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", adapter="outputs/lora_qwen2vl_2b/best")
    r = actor.predict("screen.png", instruction="打开记事本", sub_task="双击桌面上的记事本图标")
    r["actions"]  # [{"action": "double_click", "x": 103, "y": 220}]

不带 adapter 就是基座模型（零样本），用于微调前后对比；prompt 参数用于提示词工程对比。
"""
import time
from typing import Optional

try:
    from .action_format import parse_output
    from .collator import load_image
    from .modeling import DEFAULT_MAX_PIXELS, DEFAULT_MIN_PIXELS, load_base_model, load_processor
    from .prompts import DEFAULT_VARIANT, build_messages
except ImportError:
    from action_format import parse_output
    from collator import load_image
    from modeling import DEFAULT_MAX_PIXELS, DEFAULT_MIN_PIXELS, load_base_model, load_processor
    from prompts import DEFAULT_VARIANT, build_messages


class VLMActor:
    def __init__(self, model, processor, prompt: str = DEFAULT_VARIANT, max_new_tokens: int = 128, name: str = "vlm"):
        self.model = model
        self.processor = processor
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self.name = name
        self.model.eval()

    @classmethod
    def from_pretrained(cls, model_id: str, adapter: Optional[str] = None, prompt: str = DEFAULT_VARIANT,
                        dtype: str = "auto", load_4bit: bool = False, min_pixels: int = DEFAULT_MIN_PIXELS,
                        max_pixels: int = DEFAULT_MAX_PIXELS, merge: bool = True, max_new_tokens: int = 128):
        processor = load_processor(model_id, min_pixels, max_pixels)
        model = load_base_model(model_id, dtype=dtype, load_4bit=load_4bit)
        name = "base"
        if adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter)
            if merge and not load_4bit:  # 合并权重后推理速度与基座相同
                model = model.merge_and_unload()
            name = "lora"
        return cls(model, processor, prompt=prompt, max_new_tokens=max_new_tokens, name=name)

    def _device(self):
        import torch
        p = next((p for p in self.model.parameters()), None)
        return p.device if p is not None else torch.device("cpu")

    def generate_text(self, image, instruction: str, sub_task: Optional[str] = None,
                      prompt: Optional[str] = None) -> str:
        import torch
        if isinstance(image, (str, bytes)) or hasattr(image, "__fspath__"):
            image = load_image(image)
        msgs = build_messages(instruction, sub_task, prompt or self.prompt)
        text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = self.processor(text=[text], images=[image], return_tensors="pt")
        dev = self._device()
        dtype = next((p.dtype for p in self.model.parameters() if p.is_floating_point()), None)
        enc = {k: (v.to(dev).to(dtype) if k == "pixel_values" and dtype is not None else v.to(dev))
               for k, v in enc.items() if hasattr(v, "to")}
        tok = getattr(self.processor, "tokenizer", self.processor)
        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                                      pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        new = out[0, enc["input_ids"].shape[1]:]
        return tok.decode(new, skip_special_tokens=True)

    def predict(self, image, instruction: str, sub_task: Optional[str] = None, prompt: Optional[str] = None,
                screen_size=None) -> dict:
        """返回 {"raw", "actions", "ok", "error", "sec"}；actions 坐标为 0~1000"""
        t0 = time.time()
        raw = self.generate_text(image, instruction, sub_task, prompt)
        parsed = parse_output(raw, screen_size)
        return {"raw": raw, "actions": parsed["actions"], "ok": parsed["ok"], "error": parsed["error"],
                "dropped": parsed["dropped"], "sec": round(time.time() - t0, 3)}
