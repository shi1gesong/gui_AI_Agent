#!/usr/bin/env bash
# 第5周一键流程（WSL2，conda 环境 gui-agent，需要 GPU）
#   bash scripts/run_week5_wsl.sh            全流程：数据 → 基线 → 基座 3 种提示词 → LoRA 训练 → 微调后评测 → 对比表 → GPU 用例
#   QUICK=1 bash scripts/run_week5_wsl.sh    冒烟：每项只用 20 条样本、训练 1 个 epoch（约 5 分钟，先确认环境没问题）
# 每一步都会把输出写进 outputs/week5_logs/，中断后重跑会跳过已完成的评测样本。
set -euo pipefail
cd "$(dirname "$0")/.."

# 默认直连 huggingface.co；在国内网络下可先 export HF_ENDPOINT=https://hf-mirror.com 再运行本脚本
MODEL="${MODEL:-Qwen/Qwen2-VL-2B-Instruct}"
OUT=outputs
LOGS=$OUT/week5_logs
mkdir -p "$LOGS"

if [ "${QUICK:-0}" = "1" ]; then
  LIMIT="--limit 20"; TRAIN_EXTRA="--epochs 1 --limit-train 40 --limit-val 10"; EVAL_DIR=$OUT/eval_quick; ADAPTER_DIR=$OUT/lora_quick
else
  LIMIT=""; TRAIN_EXTRA=""; EVAL_DIR=$OUT/eval; ADAPTER_DIR=$OUT/lora_qwen2vl_2b
fi

step() { echo; echo "==================== $* ===================="; }

step "0/7 检查 GPU"
# 新版 peft 导入时会顺带导入 bitsandbytes → triton，WSL 里没装 gcc 时会报 "Failed to find C compiler"。
# 默认训练不用 4bit 量化，不需要 bitsandbytes；只有 --load-4bit 才需要（那时请先 sudo apt install -y build-essential）。
if ! python -c "import peft" >/dev/null 2>&1; then
  if python -c "import bitsandbytes" 2>&1 | grep -q "C compiler"; then
    echo "检测到 bitsandbytes 因缺少 C 编译器无法导入，自动卸载（默认训练用不到它）"
    pip uninstall -y bitsandbytes
  fi
fi
python -c "import torch, transformers, peft; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'); print('transformers', transformers.__version__, 'peft', peft.__version__)" | tee "$LOGS/env.txt"

step "1/7 构建数据集（ScreenAgent → train/val/test）"
if [ ! -f data_cache/sft/test.jsonl ]; then
  python -m finetune.build_dataset --prepare 2>&1 | tee "$LOGS/build_dataset.txt"
else
  echo "已存在 data_cache/sft，跳过"
fi

step "2/7 常数基线（不需要模型）"
python -m finetune.evaluate --baseline const --out "$EVAL_DIR/0_const" --label const_baseline $LIMIT 2>&1 | tail -25 | tee "$LOGS/eval_const.txt"

step "3/7 基座模型零样本：三种提示词"
for P in v0_minimal v1_schema v2_fewshot; do
  python -m finetune.evaluate --model "$MODEL" --prompt $P --out "$EVAL_DIR/1_base_$P" --label base_$P $LIMIT 2>&1 \
    | tail -25 | tee "$LOGS/eval_base_$P.txt"
done

step "4/7 LoRA 微调"
python -m finetune.train_lora --model "$MODEL" --out "$ADAPTER_DIR" $TRAIN_EXTRA 2>&1 | tee "$LOGS/train.txt"

step "5/7 微调后评测"
python -m finetune.evaluate --model "$MODEL" --adapter "$ADAPTER_DIR/best" --prompt v1_schema \
  --out "$EVAL_DIR/2_lora_v1_schema" --label lora_v1_schema $LIMIT 2>&1 | tail -25 | tee "$LOGS/eval_lora.txt"

step "6/7 汇总对比表"
python -m finetune.evaluate --compare "$EVAL_DIR" --report "$EVAL_DIR/compare.md" | tee "$LOGS/compare.txt"

step "7/7 GPU 真实用例"
GUI_AGENT_VLM_MODEL="$MODEL" GUI_AGENT_VLM_ADAPTER="$ADAPTER_DIR/best" \
  python -m pytest tests/real/test_real_vlm.py -m gpu -v -s -p no:cacheprovider 2>&1 | tee "$LOGS/pytest_gpu.txt" || true

echo
echo "全部完成。需要发回的文件："
echo "  $EVAL_DIR/compare.md   $ADAPTER_DIR/train_summary.json   $LOGS/pytest_gpu.txt"
