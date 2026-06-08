#!/bin/bash
set -euo pipefail

# OpenRLHF RM training scaffold for ChatJoy pair data.
# Default paths are wired to the current ChatJoy project layout.

PACKAGE_DIR="/home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/framework_bridge"
PAIR_INPUT="${PAIR_INPUT:-/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/runs/rejected_sft_sample20_20260331_155727/rejected_pairs_from_sft_test20.jsonl}"
RM_DATA_DIR="${RM_DATA_DIR:-${PACKAGE_DIR}/outputs/openrlhf_rm_data}"
TRAIN_FILE="${TRAIN_FILE:-${RM_DATA_DIR}/train.jsonl}"
VAL_FILE="${VAL_FILE:-${RM_DATA_DIR}/val.jsonl}"

PRETRAIN_MODEL="${PRETRAIN_MODEL:-/home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/qwen3-30b-moe/clean32_for_sft2_fsdp_qlora_merged}"
SAVE_PATH="${SAVE_PATH:-/home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/openrlhf/chatjoy_rm}"

VAL_RATIO="${VAL_RATIO:-0.02}"
MAX_LEN="${MAX_LEN:-4096}"
MAX_EPOCHS="${MAX_EPOCHS:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
MICRO_TRAIN_BATCH_SIZE="${MICRO_TRAIN_BATCH_SIZE:-1}"
LEARNING_RATE="${LEARNING_RATE:-9e-6}"
ZERO_STAGE="${ZERO_STAGE:-3}"
PARAM_DTYPE="${PARAM_DTYPE:-bf16}"
VALUE_HEAD_PREFIX="${VALUE_HEAD_PREFIX:-score}"
ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"

python3 "${PACKAGE_DIR}/prepare_openrlhf_rm_data.py"   --input "${PAIR_INPUT}"   --train-output "${TRAIN_FILE}"   --val-output "${VAL_FILE}"   --val-ratio "${VAL_RATIO}"

deepspeed --module openrlhf.cli.train_rm   --pretrain "${PRETRAIN_MODEL}"   --dataset "${TRAIN_FILE}"   --eval_dataset "${VAL_FILE}"   --prompt_key prompt   --chosen_key chosen   --rejected_key rejected   --save_path "${SAVE_PATH}"   --save_steps -1   --logging_steps 1   --eval_steps -1   --train_batch_size "${TRAIN_BATCH_SIZE}"   --micro_train_batch_size "${MICRO_TRAIN_BATCH_SIZE}"   --param_dtype "${PARAM_DTYPE}"   --max_epochs "${MAX_EPOCHS}"   --max_len "${MAX_LEN}"   --zero_stage "${ZERO_STAGE}"   --learning_rate "${LEARNING_RATE}"   --value_head_prefix "${VALUE_HEAD_PREFIX}"   --attn_implementation "${ATTN_IMPL}"   --apply_chat_template   --packing_samples   --gradient_checkpointing   "$@"
