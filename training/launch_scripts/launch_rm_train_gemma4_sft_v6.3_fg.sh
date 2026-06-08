#!/usr/bin/bash
# ChatJoy Gemma4 RM v6.3 8-GPU launcher.
#
# Inherits v6.2 A+C training recipe (LR 9e-6, LoRA r=128 alpha=256
# target=q,k,v,o,up,down, ZeRO-2 + QLoRA 4bit + adam_offload, max_len 3072,
# bf16, cosine warmup 0.03, eval every 200 steps, dense save every 50 steps).
#
# Only the training/eval data files differ from v6.2 A+C:
#   train: data/openrlhf_rm/rm_train_v6.3_fg_combined.jsonl (40,551 pairs)
#   eval:  data/openrlhf_rm/rm_eval_v6.3_fg_combined.jsonl  (3,423 pairs)
# Composition: v5 anchor (17712 train + 885 eval) + Lever 1 (20574 train +
# 2303 eval) + Lever 2 capped attack pairs (2265 train + 235 eval), attack
# ratio 5.59% in train per handoff.
#
# Expected grad_step: 40551 / 64 = 634 steps for 1 epoch (~12-13h on 8x A100).
# Eval every 100 steps (6 evals: 100/200/300/400/500/600) to catch the loss
# elbow (v6.2 elbow ~step 60-100) and monitor attack-pair learning earlier;
# save at every 50 steps (12 saves, max_ckpt_num=8 keeps the 8 most recent).
#
# Required env: gemma4-rl conda env, OpenRLHF source, SFT2 merged base model.
# Override defaults via env vars; defaults are correct for this host.

set -euo pipefail

CHATJOY_DIR="${CHATJOY_DIR:-/root/hanjiaqi/LlamaFactory/chatjoy}"
OPENRLHF_DIR="${OPENRLHF_DIR:-/home/ps/hanjiaqi/OpenRLHF}"
SAVE_ROOT="${SAVE_ROOT:-${CHATJOY_DIR}/saves/gemma4-26b-a4b}"

RUN_ID="${RUN_ID:-chatjoy_rm_gemma4_sft_v6.3_fg_8gpu_$(date +%Y%m%d_%H%M%S)}"
SAVE_PATH="${SAVE_PATH:-${SAVE_ROOT}/${RUN_ID}}"
CKPT_PATH="${CKPT_PATH:-${SAVE_ROOT}/${RUN_ID}_ckpt}"

PRETRAIN="${PRETRAIN:-${SAVE_ROOT}/sft2_ddp_lora_3gpu_567_cut3072_merged}"
TRAIN_FILE="${TRAIN_FILE:-${CHATJOY_DIR}/data/openrlhf_rm/rm_train_v6.3_fg_combined.jsonl}"
EVAL_FILE="${EVAL_FILE:-${CHATJOY_DIR}/data/openrlhf_rm/rm_eval_v6.3_fg_combined.jsonl}"

GEMMA4_DEEPSPEED="${GEMMA4_DEEPSPEED:-/home/ps/miniconda3/envs/gemma4-rl/bin/deepspeed}"
GEMMA4_PYTHON="${GEMMA4_PYTHON:-/home/ps/miniconda3/envs/gemma4-rl/bin/python}"

GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
GLOBAL_BATCH="${GLOBAL_BATCH:-64}"
MICRO_BATCH="${MICRO_BATCH:-1}"
MASTER_PORT="${MASTER_PORT:-29663}"

SAVE_STEPS="${SAVE_STEPS:-50}"
EVAL_STEPS="${EVAL_STEPS:-100}"
MAX_CKPT_NUM="${MAX_CKPT_NUM:-8}"
MAX_CKPT_MEM="${MAX_CKPT_MEM:-80}"
LOAD_CHECKPOINT="${LOAD_CHECKPOINT:-false}"

for path in "${PRETRAIN}" "${TRAIN_FILE}" "${EVAL_FILE}" "${OPENRLHF_DIR}/openrlhf/cli/train_rm.py" "${GEMMA4_DEEPSPEED}" "${GEMMA4_PYTHON}"; do
    if [ ! -e "${path}" ]; then
        echo "ERROR: required path missing: ${path}"
        exit 1
    fi
done

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_LIST}"
GPU_COUNT="${#GPU_ARRAY[@]}"
if (( GPU_COUNT != 8 )); then
    echo "ERROR: expected 8 GPUs, got ${GPU_COUNT}: ${GPU_LIST}"
    exit 1
fi
if (( GLOBAL_BATCH % GPU_COUNT != 0 )); then
    echo "ERROR: GLOBAL_BATCH=${GLOBAL_BATCH} is not divisible by GPU_COUNT=${GPU_COUNT}"
    exit 1
fi

mkdir -p "${SAVE_PATH}" "${CKPT_PATH}"

export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
export PYTHONPATH="${OPENRLHF_DIR}:${PYTHONPATH:-}"
export OPENRLHF_FORCE_TEXT_ONLY_GEMMA4="${OPENRLHF_FORCE_TEXT_ONLY_GEMMA4:-1}"
export OPENRLHF_DATASET_NUM_PROC="${OPENRLHF_DATASET_NUM_PROC:-1}"
export HF_HOME="${HF_HOME:-${CHATJOY_DIR}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CHATJOY_DIR}/.cache/xdg}"
export TMPDIR="${TMPDIR:-${SAVE_PATH}/tmp}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${XDG_CACHE_HOME}" "${TMPDIR}"

GRAD_ACCUM=$(( GLOBAL_BATCH / GPU_COUNT / MICRO_BATCH ))
TRAIN_SAMPLES=$(wc -l < "${TRAIN_FILE}")
EVAL_SAMPLES=$(wc -l < "${EVAL_FILE}")

{
    echo "RUN_ID=${RUN_ID}"
    echo "SAVE_PATH=${SAVE_PATH}"
    echo "CKPT_PATH=${CKPT_PATH}"
    echo "GPU_LIST=${GPU_LIST}"
    echo "GLOBAL_BATCH=${GLOBAL_BATCH}"
    echo "MICRO_BATCH=${MICRO_BATCH}"
    echo "GRAD_ACCUM=${GRAD_ACCUM}"
    echo "MASTER_PORT=${MASTER_PORT}"
    echo "PRETRAIN=${PRETRAIN}"
    echo "TRAIN_FILE=${TRAIN_FILE}"
    echo "TRAIN_SAMPLES=${TRAIN_SAMPLES}"
    echo "EVAL_FILE=${EVAL_FILE}"
    echo "EVAL_SAMPLES=${EVAL_SAMPLES}"
    echo "SAVE_STEPS=${SAVE_STEPS}"
    echo "EVAL_STEPS=${EVAL_STEPS}"
    echo "MAX_CKPT_NUM=${MAX_CKPT_NUM}"
    echo "MAX_CKPT_MEM=${MAX_CKPT_MEM}"
    echo "LOAD_CHECKPOINT=${LOAD_CHECKPOINT}"
    echo "LEARNING_RATE=9e-6"
    echo "LORA_RANK=128"
    echo "LORA_ALPHA=256"
    echo "TARGET_MODULES=q_proj k_proj v_proj o_proj up_proj down_proj"
    echo "RUBRIC_VERSION=v6.3"
    echo "ATTACK_RATIO_TRAIN=$(${GEMMA4_PYTHON} -c "import json; n=0; t=0
for line in open('${TRAIN_FILE}'):
    t+=1
    r=json.loads(line)
    if (r.get('extra_info') or {}).get('pair_type')=='roledrop_attack': n+=1
print(f'{n}/{t}={100*n/t:.2f}%')
")"
    nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader,nounits || true
} | tee "${SAVE_PATH}/run_env.txt"

"${GEMMA4_DEEPSPEED}" --master_port "${MASTER_PORT}" --include "localhost:${GPU_LIST}" \
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
    --save_steps "${SAVE_STEPS}" \
    --logging_steps 5 \
    --eval_steps "${EVAL_STEPS}" \
    --train_batch_size "${GLOBAL_BATCH}" \
    --micro_train_batch_size "${MICRO_BATCH}" \
    --param_dtype bf16 \
    --max_epochs 1 \
    --max_len 3072 \
    --zero_stage 2 \
    --learning_rate 9e-6 \
    --lr_scheduler cosine \
    --lr_warmup_ratio 0.03 \
    --value_head_prefix score \
    --attn_implementation sdpa \
    --apply_chat_template \
    --gradient_checkpointing \
    --load_in_4bit \
    --lora_rank 128 \
    --lora_alpha 256 \
    --adam_offload \
    --target_modules q_proj k_proj v_proj o_proj up_proj down_proj \
    --save_hf_ckpt \
    --max_ckpt_num "${MAX_CKPT_NUM}" \
    --max_ckpt_mem "${MAX_CKPT_MEM}" \
    2>&1 | tee "${SAVE_PATH}/train.log"

"${GEMMA4_PYTHON}" "${CHATJOY_DIR}/plot_training_metrics.py" --save_path "${SAVE_PATH}" || true
