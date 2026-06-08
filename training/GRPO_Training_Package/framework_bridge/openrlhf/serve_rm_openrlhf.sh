#!/bin/bash
set -euo pipefail

# Serve a trained ChatJoy RM through OpenRLHF's reward server.

RM_MODEL_PATH="${RM_MODEL_PATH:-/home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/openrlhf/chatjoy_rm}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-5000}"
MAX_LEN="${MAX_LEN:-4096}"
BATCH_SIZE="${BATCH_SIZE:-8}"
PARAM_DTYPE="${PARAM_DTYPE:-bf16}"
VALUE_HEAD_PREFIX="${VALUE_HEAD_PREFIX:-score}"
ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"

python3 -m openrlhf.cli.serve_rm   --reward_pretrain "${RM_MODEL_PATH}"   --host "${HOST}"   --port "${PORT}"   --max_len "${MAX_LEN}"   --batch_size "${BATCH_SIZE}"   --param_dtype "${PARAM_DTYPE}"   --value_head_prefix "${VALUE_HEAD_PREFIX}"   --attn_implementation "${ATTN_IMPL}"   "$@"
