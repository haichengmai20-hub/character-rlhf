#!/usr/bin/bash
set -euo pipefail

CHATJOY_DIR="/home/ps/hanjiaqi/LlamaFactory/chatjoy"
PYTHON="/home/ps/miniconda3/envs/tslib/bin/python"
CKPT_HF="${CHATJOY_DIR}/saves/qwen3-30b-moe/chatjoy_rm_v2_filtered"
SCORE_HEAD="${CKPT_HF}/score_head_recovered.pt"
TRAIN_DATA="${CHATJOY_DIR}/data/openrlhf_rm/rm_train.jsonl"
OUTPUT="${CHATJOY_DIR}/data/openrlhf_rm/hard_negatives_long_soft.jsonl"
GAP_THRESHOLD="${1:-0}"
GPU_ID="${2:-4}"

echo "long-only mining policy: rejected_type=length_mismatch_long, bad case if gap <= ${GAP_THRESHOLD}"

cd /home/ps/hanjiaqi/LlamaFactory

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"${PYTHON}" "${CHATJOY_DIR}/mine_hard_negatives.py" \
  --ckpt_hf "${CKPT_HF}" \
  --score_head "${SCORE_HEAD}" \
  --train_data "${TRAIN_DATA}" \
  --filter_rejected_type length_mismatch_long \
  --gap_threshold "${GAP_THRESHOLD}" \
  --max_len 3072 \
  --batch_size 2 \
  --device cuda:0 \
  --output "${OUTPUT}"

echo "long-only soft pool saved to ${OUTPUT}"
