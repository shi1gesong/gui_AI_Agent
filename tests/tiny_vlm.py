"""
测试用微型 Qwen2-VL（第5周）

云端没有 GPU、也访问不了 HuggingFace，所以按真实 Qwen2-VL 的结构造一个随机初始化的微型模型：
  - 分词器：字节级 BPE（256 个字节 + Qwen 的 7 个特殊 token），中文按 UTF-8 字节切分
  - 图像处理器：真实的 Qwen2VLImageProcessor（14×14 patch、2×2 合并），像素上限调小
  - 模型：真实的 Qwen2VLForConditionalGeneration 类，2 层、隐藏维 64、视觉编码器 1 层

训练循环、label 掩码、LoRA 挂载位置、保存/加载适配器、generate 推理走的都是真实代码路径，
只有权重是随机的，所以这里验证的是“管线正确”，不是“模型效果”。
"""
import tempfile
from pathlib import Path

SPECIALS = ["<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>"]

CHAT_TEMPLATE = (
    "{% for message in messages %}<|im_start|>{{ message['role'] }}\n"
    "{% if message['content'] is string %}{{ message['content'] }}{% else %}{% for c in message['content'] %}"
    "{% if c['type'] == 'image' %}<|vision_start|><|image_pad|><|vision_end|>"
    "{% elif c['type'] == 'text' %}{{ c['text'] }}{% endif %}{% endfor %}{% endif %}"
    "<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")


def deps_available() -> bool:
    try:
        import peft  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
        from transformers import Qwen2VLForConditionalGeneration  # noqa: F401
        return True
    except Exception:
        return False


def build_tokenizer(tmpdir=None):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from tokenizers.pre_tokenizers import ByteLevel
    from transformers import Qwen2TokenizerFast
    vocab = {c: i for i, c in enumerate(sorted(ByteLevel.alphabet()))}
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    d = Path(tmpdir or tempfile.mkdtemp(prefix="tiny_tok_"))
    tok.save(str(d / "tokenizer.json"))
    t = Qwen2TokenizerFast(tokenizer_file=str(d / "tokenizer.json"), unk_token=None, bos_token=None,
                           eos_token="<|endoftext|>", pad_token="<|endoftext|>")
    t.add_special_tokens({"additional_special_tokens": SPECIALS})
    t.eos_token = "<|im_end|>"
    return t


def build_processor(min_pixels: int = 56 * 56, max_pixels: int = 112 * 112):
    from transformers import Qwen2VLImageProcessor, Qwen2VLProcessor
    tok = build_tokenizer()
    ip = Qwen2VLImageProcessor(min_pixels=min_pixels, max_pixels=max_pixels)
    kw = {}
    try:
        from transformers import Qwen2VLVideoProcessor
        kw["video_processor"] = Qwen2VLVideoProcessor()
    except ImportError:
        pass
    try:
        return Qwen2VLProcessor(image_processor=ip, tokenizer=tok, chat_template=CHAT_TEMPLATE, **kw)
    except TypeError:
        return Qwen2VLProcessor(image_processor=ip, tokenizer=tok, chat_template=CHAT_TEMPLATE)


def build_model(processor, seed: int = 0):
    import torch
    from transformers import Qwen2VLConfig, Qwen2VLForConditionalGeneration
    torch.manual_seed(seed)
    tok = processor.tokenizer
    ids = {s: tok.convert_tokens_to_ids(s) for s in SPECIALS}
    cfg = Qwen2VLConfig(
        vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, rope_scaling={"type": "mrope", "mrope_section": [2, 3, 3]},
        image_token_id=ids["<|image_pad|>"], video_token_id=ids["<|video_pad|>"],
        vision_start_token_id=ids["<|vision_start|>"], vision_end_token_id=ids["<|vision_end|>"],
        vision_config={"depth": 1, "embed_dim": 32, "num_heads": 2, "hidden_size": 64, "mlp_ratio": 2,
                       "patch_size": 14, "spatial_merge_size": 2, "temporal_patch_size": 2, "in_chans": 3},
        max_position_embeddings=4096, tie_word_embeddings=False, pad_token_id=tok.pad_token_id,
        eos_token_id=ids["<|im_end|>"])
    return Qwen2VLForConditionalGeneration(cfg)


def save_tiny_checkpoint(path) -> Path:
    """把微型模型 + processor 存成本地目录，模拟 from_pretrained(本地路径) 的真实加载流程"""
    path = Path(path)
    proc = build_processor()
    model = build_model(proc)
    model.save_pretrained(str(path))
    proc.save_pretrained(str(path))
    return path


def make_screens(tmpdir, n: int = 6, size=(1024, 768)):
    """生成 n 张不同的“截图”（色块位置不同），返回样本列表（与 build_dataset 输出格式一致）"""
    from PIL import Image, ImageDraw
    from finetune.action_format import to_target_text
    tmpdir = Path(tmpdir)
    tmpdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        img = Image.new("RGB", size, (240, 240, 240))
        d = ImageDraw.Draw(img)
        x, y = 80 + i * 120, 100 + i * 80
        d.rectangle([x, y, x + 100, y + 40], fill=(30, 120, 220))
        p = tmpdir / f"screen_{i}.png"
        img.save(p)
        cx, cy = round((x + 50) / size[0] * 1000), round((y + 20) / size[1] * 1000)
        actions = [{"action": "click", "x": cx, "y": cy}]
        if i % 3 == 1:
            actions = [{"action": "type", "text": "hello"}, {"action": "key", "keys": ["enter"]}]
        rows.append({"id": f"s{i // 2}/{i}", "session": f"s{i // 2}", "image": str(p), "instruction": "点击蓝色按钮",
                     "sub_task": f"第{i}步", "screen_size": list(size), "actions": actions,
                     "target": to_target_text(actions), "orig_split": "train"})
    return rows
