#!/bin/bash
set -uo pipefail

LF=/root/hanjiaqi/LlamaFactory
YAML="$LF/chatjoy/gemma4_rich_cjk_sft_v69_lora.yaml"
OUT="$LF/chatjoy/saves/gemma4-26b-a4b/rich_cjk_sft_lora_v69"
LOG=/tmp/sft_v69_balanced.log
GPUS=0,1,2,3,5,6,7

export PATH=/home/ubuntu/miniconda3/envs/gemma4-character/bin:$PATH
export DISABLE_VERSION_CHECK=1

cd "$LF" || exit 1
while true; do
  mount -o remount,size=2g /dev/shm 2>/dev/null || true
  if ls -d "$OUT"/checkpoint-* >/dev/null 2>&1; then
    sed -i 's/^resume_from_checkpoint:.*/resume_from_checkpoint: true/' "$YAML"
    echo "[wrapper] 发现 ckpt -> resume"
  fi
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$GPUS \
    llamafactory-cli train "$YAML" 2>&1 | tee -a "$LOG"
  if grep -q "train_runtime" "$LOG"; then
    echo "[wrapper] 训练完成"
    break
  fi
  echo "[wrapper] 中断,30s 后从最近 ckpt 续..."
  sleep 30
done
