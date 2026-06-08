#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/hanjiaqi/LlamaFactory/chatjoy"
LOG_DIR="$ROOT/logs/rm_v6_3_roledrop/lever1_full_judge_v63"

mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

exec /home/ubuntu/miniconda3/envs/gemma4-rl/bin/vllm serve /root/Qwen3___6-35B-A3B \
  --host 127.0.0.1 \
  --port 8000 \
  --served-model-name qwen3_6_35b_a3b \
  --tensor-parallel-size 2 \
  --pipeline-parallel-size 3 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code \
  --reasoning-parser qwen3 \
  --disable-custom-all-reduce \
  >> "$LOG_DIR/qwen_vllm_6gpu.log" 2>&1
