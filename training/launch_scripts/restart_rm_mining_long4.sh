#!/usr/bin/bash
set -euo pipefail

cd /home/ps/hanjiaqi/LlamaFactory
source /home/ps/miniconda3/etc/profile.d/conda.sh
conda activate tslib

OUT_DIR="/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/openrlhf_rm"
SCRIPT="/home/ps/hanjiaqi/LlamaFactory/chatjoy/mine_hard_negatives.py"
CKPT="/home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/qwen3-30b-moe/chatjoy_rm_v2_filtered"
SCORE_HEAD="${CKPT}/score_head_recovered.pt"
TRAIN_DATA="/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/openrlhf_rm/rm_train.jsonl"

mkdir -p "${OUT_DIR}"

for spec in "0 0" "1 1" "2 2" "3 3"; do
  set -- $spec
  gpu=$1
  shard=$2

  CUDA_VISIBLE_DEVICES=$gpu \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/ps/miniconda3/envs/tslib/bin/python \
    "${SCRIPT}" \
    --ckpt_hf "${CKPT}" \
    --score_head "${SCORE_HEAD}" \
    --train_data "${TRAIN_DATA}" \
    --filter_rejected_type length_mismatch_long \
    --output "${OUT_DIR}/hard_negatives_long_soft_shard${shard}.jsonl" \
    --gap_threshold 0 \
    --max_len 3072 \
    --batch_size 2 \
    --num_shards 4 \
    --shard_index "${shard}" \
    --device cuda:0 \
    > "${OUT_DIR}/mining_long_shard${shard}.log" 2>&1 &

  echo "started long-only shard ${shard} on gpu ${gpu}"
  sleep 15
done

wait

/home/ps/miniconda3/envs/tslib/bin/python \
  "${SCRIPT}" \
  merge \
  "${OUT_DIR}/hard_negatives_long_soft_shard0.jsonl" \
  "${OUT_DIR}/hard_negatives_long_soft_shard1.jsonl" \
  "${OUT_DIR}/hard_negatives_long_soft_shard2.jsonl" \
  "${OUT_DIR}/hard_negatives_long_soft_shard3.jsonl" \
  --output "${OUT_DIR}/hard_negatives_long_soft.jsonl" \
  --gap_threshold 0 \
  > "${OUT_DIR}/mining_long_merge.log" 2>&1

echo "done: ${OUT_DIR}/hard_negatives_long_soft.jsonl"
