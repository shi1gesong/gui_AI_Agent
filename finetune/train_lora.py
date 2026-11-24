"""
LoRA 微调入口（第5周，WSL2 + GPU 环境 gui-agent）

    python -m finetune.train_lora --data-dir data_cache/sft --out outputs/lora_qwen2vl_2b

流程：加载基座（默认 Qwen2-VL-2B-Instruct）→ 挂 LoRA（语言模型 q/k/v/o/gate/up/down，视觉编码器冻结）
     → AdamW + 线性预热 + 余弦退火 → 每个 epoch 在验证集上算损失 → 保存验证损失最低的适配器到 <out>/best

输出：
    <out>/best/           LoRA 适配器（adapter_config.json + adapter_model.safetensors）+ processor 配置
    <out>/final/          最后一个 epoch 的适配器
    <out>/train_log.jsonl 每 log_every 步一行：step、loss、lr、耗时
    <out>/train_summary.json  参数量、各 epoch 训练/验证损失、最佳 epoch、用时、显存峰值

训练循环 run_training() 不依赖具体模型，测试里用随机初始化的微型 Qwen2-VL 在 CPU 上真实跑通。
"""
import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

try:
    from .build_dataset import load_jsonl
    from .collator import SFTCollator, sample_exists
    from .modeling import (DEFAULT_MAX_PIXELS, DEFAULT_MIN_PIXELS, DEFAULT_MODEL, apply_lora, count_parameters,
                           load_base_model, load_processor, lora_module_names)
    from .prompts import DEFAULT_VARIANT
except ImportError:
    from build_dataset import load_jsonl
    from collator import SFTCollator, sample_exists
    from modeling import (DEFAULT_MAX_PIXELS, DEFAULT_MIN_PIXELS, DEFAULT_MODEL, apply_lora, count_parameters,
                          load_base_model, load_processor, lora_module_names)
    from prompts import DEFAULT_VARIANT


@dataclass
class TrainConfig:
    model: str = DEFAULT_MODEL
    data_dir: str = "data_cache/sft"
    out: str = "outputs/lora_qwen2vl_2b"
    epochs: int = 3
    lr: float = 2e-4
    weight_decay: float = 0.0
    batch_size: int = 1
    grad_accum: int = 8
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    min_pixels: int = DEFAULT_MIN_PIXELS
    max_pixels: int = DEFAULT_MAX_PIXELS
    max_length: int = 2048
    dtype: str = "auto"
    load_4bit: bool = False
    gradient_checkpointing: bool = True
    prompt: str = DEFAULT_VARIANT
    seed: int = 42
    limit_train: int = 0
    limit_val: int = 0
    log_every: int = 10
    save_every_epoch: bool = False
    extra: dict = field(default_factory=dict)

    def validate(self):
        if self.epochs <= 0 or self.batch_size <= 0 or self.grad_accum <= 0:
            raise ValueError("epochs / batch_size / grad_accum 必须为正整数")
        if not (0 < self.lr < 1):
            raise ValueError(f"学习率不合理: {self.lr}")
        if not (0 <= self.warmup_ratio < 1):
            raise ValueError(f"warmup_ratio 必须在 [0,1) 内: {self.warmup_ratio}")
        return self


def lr_lambda_factory(total_steps: int, warmup_steps: int) -> Callable[[int], float]:
    """线性预热到 1，再余弦退火到 0"""
    total_steps = max(1, total_steps)
    warmup_steps = max(0, min(warmup_steps, total_steps - 1)) if total_steps > 1 else 0

    def f(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress))))
    return f


def batches(rows: List[dict], batch_size: int, rng: Optional[random.Random] = None) -> List[List[dict]]:
    idx = list(range(len(rows)))
    if rng is not None:
        rng.shuffle(idx)
    return [[rows[i] for i in idx[k:k + batch_size]] for k in range(0, len(idx), batch_size)]


def _to_device(batch: dict, device, dtype=None):
    import torch
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            v = v.to(device)
            if k == "pixel_values" and dtype is not None and v.is_floating_point():
                v = v.to(dtype)
        out[k] = v
    return out


def _model_device_dtype(model):
    import torch
    p = next((p for p in model.parameters() if p.is_floating_point()), None)
    if p is None:
        return torch.device("cpu"), torch.float32
    dtype = p.dtype if p.dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.bfloat16
    return p.device, dtype


def evaluate_loss(model, collator, rows: List[dict], batch_size: int = 1) -> float:
    """验证集平均损失（按回答 token 数加权）"""
    import torch
    if not rows:
        return float("nan")
    device, dtype = _model_device_dtype(model)
    was_training = model.training
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for b in batches(rows, batch_size):
            batch = _to_device(collator(b), device, dtype)
            ntok = int((batch["labels"] != -100).sum())
            loss = model(**batch).loss
            tot += float(loss.detach()) * ntok
            n += ntok
    if was_training:
        model.train()
    return tot / max(1, n)


def run_training(cfg: TrainConfig, model, processor, train_rows: List[dict], val_rows: List[dict],
                 log: Callable = print, save: bool = True) -> dict:
    import torch
    cfg.validate()
    if not train_rows:
        raise ValueError("训练集为空")
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    out = Path(cfg.out)
    if save:
        out.mkdir(parents=True, exist_ok=True)
    collator = SFTCollator(processor, variant=cfg.prompt, max_length=cfg.max_length)
    device, dtype = _model_device_dtype(model)
    for m in (model, getattr(model, "base_model", None)):
        if m is not None and getattr(m, "config", None) is not None:
            m.config.use_cache = False  # 梯度检查点与 KV 缓存不兼容
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("模型没有可训练参数（LoRA 没挂上？）")
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = math.ceil(math.ceil(len(train_rows) / cfg.batch_size) / cfg.grad_accum)
    total = steps_per_epoch * cfg.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda_factory(total, int(total * cfg.warmup_ratio)))
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    summary = {"config": asdict(cfg), "params": count_parameters(model), "train_samples": len(train_rows),
               "val_samples": len(val_rows), "optimizer_steps": total, "epochs": [],
               "lora_modules": len(lora_module_names(model))}
    log(f"[训练] 样本 {len(train_rows)}，验证 {len(val_rows)}，每 epoch {steps_per_epoch} 次参数更新，共 {total} 次；"
        f"可训练参数 {summary['params']['trainable']:,} / {summary['params']['total']:,}")
    log_f = open(out / "train_log.jsonl", "w", encoding="utf-8") if save else None
    best = float("inf")
    step, t0 = 0, time.time()
    model.train()
    try:
        for ep in range(1, cfg.epochs + 1):
            ep_loss, ep_tok, micro = 0.0, 0, 0
            opt.zero_grad(set_to_none=True)
            bl = batches(train_rows, cfg.batch_size, rng)
            for i, b in enumerate(bl, 1):
                batch = _to_device(collator(b), device, dtype)
                ntok = int((batch["labels"] != -100).sum())
                loss = model(**batch).loss
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"第 {ep} 个 epoch 第 {i} 批出现非有限损失 {float(loss)}，请调低学习率")
                (loss / cfg.grad_accum).backward()
                ep_loss += float(loss.detach()) * ntok
                ep_tok += ntok
                micro += 1
                if micro == cfg.grad_accum or i == len(bl):
                    torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
                    opt.step()
                    sched.step()
                    opt.zero_grad(set_to_none=True)
                    micro = 0
                    step += 1
                    if step % cfg.log_every == 0 or step == total:
                        rec = {"step": step, "epoch": ep, "loss": round(float(loss.detach()), 5),
                               "lr": sched.get_last_lr()[0], "sec": round(time.time() - t0, 1)}
                        log(f"  step {step}/{total}  loss={rec['loss']:.4f}  lr={rec['lr']:.2e}  {rec['sec']}s")
                        if log_f:
                            log_f.write(json.dumps(rec) + "\n")
                            log_f.flush()
            train_loss = ep_loss / max(1, ep_tok)
            val_loss = evaluate_loss(model, collator, val_rows, cfg.batch_size) if val_rows else float("nan")
            rec = {"epoch": ep, "train_loss": round(train_loss, 5),
                   "val_loss": None if math.isnan(val_loss) else round(val_loss, 5), "sec": round(time.time() - t0, 1)}
            summary["epochs"].append(rec)
            log(f"[训练] epoch {ep}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
            improved = (not math.isnan(val_loss) and val_loss < best) or (math.isnan(val_loss) and ep == cfg.epochs)
            if improved:
                best = val_loss if not math.isnan(val_loss) else best
                summary["best_epoch"] = ep
                if save:
                    _save_adapter(model, processor, out / "best")
            if save and cfg.save_every_epoch:
                _save_adapter(model, processor, out / f"epoch_{ep}")
    finally:
        if log_f:
            log_f.close()
    if save:
        _save_adapter(model, processor, out / "final")
    summary["best_val_loss"] = None if best == float("inf") else round(best, 5)
    summary["train_sec"] = round(time.time() - t0, 1)
    if torch.cuda.is_available():
        summary["peak_gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)
        summary["gpu"] = torch.cuda.get_device_name(0)
    if save:
        (out / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _save_adapter(model, processor, path: Path):
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(path))
    try:
        processor.save_pretrained(str(path))
    except Exception as e:  # processor 保存失败不影响适配器本身
        (path / "processor_save_error.txt").write_text(str(e), encoding="utf-8")


def load_rows(data_dir: str, split: str, limit: int = 0) -> List[dict]:
    path = Path(data_dir) / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"找不到 {path}，请先运行 python -m finetune.build_dataset --prepare")
    rows = [r for r in load_jsonl(path) if sample_exists(r)]
    return rows[:limit] if limit and limit > 0 else rows


def parse_args(argv=None) -> TrainConfig:
    d = TrainConfig()
    ap = argparse.ArgumentParser(description="多模态模型 LoRA 微调（GUI 动作生成）")
    ap.add_argument("--model", default=d.model, help="HuggingFace 模型名或本地路径")
    ap.add_argument("--data-dir", default=d.data_dir)
    ap.add_argument("--out", default=d.out)
    ap.add_argument("--epochs", type=int, default=d.epochs)
    ap.add_argument("--lr", type=float, default=d.lr)
    ap.add_argument("--batch-size", type=int, default=d.batch_size)
    ap.add_argument("--grad-accum", type=int, default=d.grad_accum)
    ap.add_argument("--warmup-ratio", type=float, default=d.warmup_ratio)
    ap.add_argument("--lora-r", type=int, default=d.lora_r)
    ap.add_argument("--lora-alpha", type=int, default=d.lora_alpha)
    ap.add_argument("--lora-dropout", type=float, default=d.lora_dropout)
    ap.add_argument("--min-pixels", type=int, default=d.min_pixels)
    ap.add_argument("--max-pixels", type=int, default=d.max_pixels)
    ap.add_argument("--max-length", type=int, default=d.max_length)
    ap.add_argument("--dtype", default=d.dtype)
    ap.add_argument("--load-4bit", action="store_true", help="QLoRA：4bit 量化加载基座（需要 bitsandbytes）")
    ap.add_argument("--no-gradient-checkpointing", action="store_true")
    ap.add_argument("--prompt", default=d.prompt)
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--limit-train", type=int, default=0, help="只用前 N 条训练（快速冒烟）")
    ap.add_argument("--limit-val", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=d.log_every)
    ap.add_argument("--save-every-epoch", action="store_true")
    a = ap.parse_args(argv)
    return TrainConfig(model=a.model, data_dir=a.data_dir, out=a.out, epochs=a.epochs, lr=a.lr,
                       batch_size=a.batch_size, grad_accum=a.grad_accum, warmup_ratio=a.warmup_ratio,
                       lora_r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout,
                       min_pixels=a.min_pixels, max_pixels=a.max_pixels, max_length=a.max_length, dtype=a.dtype,
                       load_4bit=a.load_4bit, gradient_checkpointing=not a.no_gradient_checkpointing,
                       prompt=a.prompt, seed=a.seed, limit_train=a.limit_train, limit_val=a.limit_val,
                       log_every=a.log_every, save_every_epoch=a.save_every_epoch).validate()


def main(argv=None):
    cfg = parse_args(argv)
    train_rows = load_rows(cfg.data_dir, "train", cfg.limit_train)
    val_rows = load_rows(cfg.data_dir, "val", cfg.limit_val)
    print(f"[加载] 基座 {cfg.model}（首次运行会下载，国内可先 export HF_ENDPOINT=https://hf-mirror.com）")
    processor = load_processor(cfg.model, cfg.min_pixels, cfg.max_pixels)
    model = load_base_model(cfg.model, dtype=cfg.dtype, load_4bit=cfg.load_4bit)
    model = apply_lora(model, r=cfg.lora_r, alpha=cfg.lora_alpha, dropout=cfg.lora_dropout,
                       gradient_checkpointing=cfg.gradient_checkpointing, quantized=cfg.load_4bit)
    summary = run_training(cfg, model, processor, train_rows, val_rows)
    print(f"[完成] 最佳 epoch {summary.get('best_epoch')}，验证损失 {summary.get('best_val_loss')}，"
          f"用时 {summary['train_sec']} 秒；适配器在 {cfg.out}/best")
    return summary


if __name__ == "__main__":
    main()
