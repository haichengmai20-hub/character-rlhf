#!/bin/bash
set -euo pipefail

cd /home/ps/hanjiaqi/LlamaFactory
source /home/ps/miniconda3/etc/profile.d/conda.sh
conda activate tslib

mkdir -p chatjoy/data/openrlhf_rm

for spec in "0 0" "1 1" "3 2" "6 3"; do
  set -- $spec
  gpu=$1
  shard=$2

  CUDA_VISIBLE_DEVICES=$gpu \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/ps/miniconda3/envs/tslib/bin/python \
    /home/ps/hanjiaqi/LlamaFactory/chatjoy/mine_hard_negatives.py \
    --ckpt_hf /home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/qwen3-30b-moe/chatjoy_rm_v2_filtered \
    --score_head /home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/qwen3-30b-moe/chatjoy_rm_v2_filtered/score_head_recovered.pt \
    --train_data /home/ps/hanjiaqi/LlamaFactory/chatjoy/data/openrlhf_rm/rm_train_filtered.jsonl \
    --output /home/ps/hanjiaqi/LlamaFactory/chatjoy/data/openrlhf_rm/hard_negatives_v2_shard${shard}.jsonl \
    --gap_threshold -0.3 \
    --max_len 3072 \
    --batch_size 2 \
    --num_shards 4 \
    --shard_index ${shard} \
    --device cuda:0 \
    > /home/ps/hanjiaqi/LlamaFactory/chatjoy/data/openrlhf_rm/mining4_shard${shard}.log 2>&1 &

  echo "started shard ${shard} on gpu ${gpu}"
  sleep 20
done

wait

/home/ps/miniconda3/envs/tslib/bin/python \
  /home/ps/hanjiaqi/LlamaFactory/chatjoy/mine_hard_negatives.py \
  merge \
  /home/ps/hanjiaqi/LlamaFactory/chatjoy/data/openrlhf_rm/hard_negatives_v2_shard0.jsonl \
  /home/ps/hanjiaqi/LlamaFactory/chatjoy/data/openrlhf_rm/hard_negatives_v2_shard1.jsonl \
  /home/ps/hanjiaqi/LlamaFactory/chatjoy/data/openrlhf_rm/hard_negatives_v2_shard2.jsonl \
  /home/ps/hanjiaqi/LlamaFactory/chatjoy/data/openrlhf_rm/hard_negatives_v2_shard3.jsonl \
  --output /home/ps/hanjiaqi/LlamaFactory/chatjoy/data/openrlhf_rm/hard_negatives_v2.jsonl \
  --gap_threshold -0.3 \
  > /home/ps/hanjiaqi/LlamaFactory/chatjoy/data/openrlhf_rm/mining4_merge.log 2>&1
