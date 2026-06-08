#!/usr/bin/bash
# Gemma4 SFT RM v6.1 fine-grained training launcher.
# v6.1 vs v6: drops delta=2 medium-confidence pairs (38.9% of fine-grained pairs removed).
# Train set: 38,226 rows (v5_anchor=17,712 + fg_high_quality=20,514).

set -euo pipefail

CHATJOY_DIR="/home/ps/hanjiaqi/LlamaFactory/chatjoy"
OPENRLHF_DIR="/home/ps/hanjiaqi/OpenRLHF"

PRETRAIN="${PRETRAIN:-${CHATJOY_DIR}/saves/gemma4-26b-a4b/sft2_ddp_lora_3gpu_567_cut3072_merged}"
TRAIN_FILE="${TRAIN_FILE:-${CHATJOY_DIR}/data/openrlhf_rm/rm_train_v6.1_fg_combined.jsonl}"
EVAL_FILE="${EVAL_FILE:-${CHATJOY_DIR}/data/openrlhf_rm/rm_eval_v6.1_fg_combined.jsonl}"
RUN_ID="${RUN_ID:-chatjoy_rm_gemma4_sft_v6.1_fg_$(date +%Y%m%d_%H%M%S)}"
SAVE_PATH="${SAVE_PATH:-${CHATJOY_DIR}/saves/gemma4-26b-a4b/${RUN_ID}}"
CKPT_PATH="${CKPT_PATH:-${CHATJOY_DIR}/saves/gemma4-26b-a4b/${RUN_ID}_ckpt}"
LOAD_CHECKPOINT="${LOAD_CHECKPOINT:-false}"

GEMMA4_DEEPSPEED="${GEMMA4_DEEPSPEED:-/home/ps/miniconda3/envs/gemma4-rl/bin/deepspeed}"
GEMMA4_PYTHON="${GEMMA4_PYTHON:-/home/ps/miniconda3/envs/gemma4-rl/bin/python}"

if [ ! -d "${PRETRAIN}" ]; then
    echo "ERROR: pretrain model not found: ${PRETRAIN}"
    exit 1
fi
if [ ! -f "${TRAIN_FILE}" ]; then
    echo "ERROR: training file not found: ${TRAIN_FILE}"
    exit 1
fi
if [ ! -f "${EVAL_FILE}" ]; then
    echo "ERROR: eval file not found: ${EVAL_FILE}"
    exit 1
fi
if [ ! -f "${OPENRLHF_DIR}/openrlhf/cli/train_rm.py" ]; then
    echo "ERROR: OpenRLHF source not found: ${OPENRLHF_DIR}"
    exit 1
fi

GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_LIST}"
GPU_COUNT="${#GPU_ARRAY[@]}"
GLOBAL_BATCH="${GLOBAL_BATCH:-64}"
MICRO_BATCH="${MICRO_BATCH:-1}"
ZERO_STAGE="${ZERO_STAGE:-2}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"

if (( GLOBAL_BATCH % GPU_COUNT != 0 )); then
    echo "ERROR: global batch ${GLOBAL_BATCH} is not divisible by GPU count ${GPU_COUNT} (${GPU_LIST})."
    exit 1
fi

mkdir -p "${SAVE_PATH}"

export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
export PYTHONPATH="${OPENRLHF_DIR}:${PYTHONPATH:-}"
export OPENRLHF_FORCE_TEXT_ONLY_GEMMA4="${OPENRLHF_FORCE_TEXT_ONLY_GEMMA4:-1}"
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=0
export NCCL_SOCKET_IFNAME=lo

DEEPSPEED_INCLUDE="${DEEPSPEED_INCLUDE:-localhost:${GPU_LIST}}"
MASTER_PORT="${MASTER_PORT:-29528}"

TRAIN_SAMPLES=$(wc -l < "${TRAIN_FILE}")
EVAL_SAMPLES=$(wc -l < "${EVAL_FILE}")
GRAD_ACCUM=$(( GLOBAL_BATCH / GPU_COUNT / MICRO_BATCH ))

echo "=============================================="
echo "  ChatJoy Gemma4 SFT RM v6.1 Fine-Grained"
echo "  (dual-filter: delta=4 all + delta=2 high-conf)"
echo "  OpenRLHF src  : ${OPENRLHF_DIR}"
echo "  Python        : ${GEMMA4_PYTHON}"
echo "  Pretrain      : ${PRETRAIN}"
echo "  Train file    : ${TRAIN_FILE}"
echo "  Train samples : ${TRAIN_SAMPLES}"
echo "  Eval file     : ${EVAL_FILE}"
echo "  Eval samples  : ${EVAL_SAMPLES}"
echo "  GPUs          : ${CUDA_VISIBLE_DEVICES}"
echo "  Global batch  : ${GLOBAL_BATCH}"
echo "  Micro batch   : ${MICRO_BATCH}"
echo "  Grad accum    : ${GRAD_ACCUM}"
echo "  ZeRO stage    : ${ZERO_STAGE}"
echo "  LR            : ${LEARNING_RATE}"
echo "  Attention impl: ${ATTN_IMPLEMENTATION}"
echo "  Max len       : 3072"
echo "  Output        : ${SAVE_PATH}"
echo "  Ckpt path     : ${CKPT_PATH}"
echo "  Load ckpt     : ${LOAD_CHECKPOINT}"
echo "=============================================="

"${GEMMA4_DEEPSPEED}" --master_port "${MASTER_PORT}" --include "${DEEPSPEED_INCLUDE}" \
    --module openrlhf.cli.train_rm \
    --pretrain "${PRETRAIN}" \
    --dataset "${TRAIN_FILE}" \
    --eval_dataset "${EVAL_FILE}" \
    --prompt_key prompt \
    --chosen_key chosen \
    --rejected_key rejected \
    --save_path "${SAVE_PATH}" \
    --ckpt_path "${CKPT_PATH}" \
    $([ "${LOAD_CHECKPOINT}" = "true" ] && echo "--load_checkpoint") \
    --save_steps 200 \
    --logging_steps 5 \
    --eval_steps 200 \
    --train_batch_size "${GLOBAL_BATCH}" \
    --micro_train_batch_size "${MICRO_BATCH}" \
    --param_dtype bf16 \
    --max_epochs 1 \
    --max_len 3072 \
    --zero_stage "${ZERO_STAGE}" \
    --learning_rate "${LEARNING_RATE}" \
    --lr_scheduler cosine \
    --lr_warmup_ratio 0.03 \
    --value_head_prefix score \
    --attn_implementation "${ATTN_IMPLEMENTATION}" \
    --apply_chat_template \
    --gradient_checkpointing \
    --load_in_4bit \
    --lora_rank 64 \
    --lora_alpha 128 \
    --adam_offload \
    --target_modules q_proj k_proj v_proj o_proj \
    ${EXTRA_ARGS:-} \
    2>&1 | tee "${SAVE_PATH}/train.log"

echo ""
echo "Training finished. Generating metric plots..."
"${GEMMA4_PYTHON}" "${CHATJOY_DIR}/plot_training_metrics.py" --save_path "${SAVE_PATH}" || true
