#!/usr/bin/bash
# Nightly 8-GPU fresh run wrapper for ChatJoy Gemma4 RM v6.2 A+C.
#
# This wrapper intentionally starts from the base model, not from the
# partially saved 20260518 checkpoint. It keeps the original launcher as the
# single source of training arguments, while pinning the runtime environment
# and write locations for tonight's run.

set -euo pipefail

CHATJOY_DIR="${CHATJOY_DIR:-/home/ps/hanjiaqi/LlamaFactory/chatjoy}"
BASE_LAUNCHER="${BASE_LAUNCHER:-${CHATJOY_DIR}/launch_rm_train_gemma4_sft_v6.2_AC_fg.sh}"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ID="${RUN_ID:-chatjoy_rm_gemma4_sft_v6.2_AC_fg_8gpu_${RUN_STAMP}}"
SAVE_ROOT="${SAVE_ROOT:-${CHATJOY_DIR}/saves/gemma4-26b-a4b}"
SAVE_PATH="${SAVE_PATH:-${SAVE_ROOT}/${RUN_ID}}"
CKPT_PATH="${CKPT_PATH:-${SAVE_ROOT}/${RUN_ID}_ckpt}"

GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
GLOBAL_BATCH="${GLOBAL_BATCH:-64}"
MICRO_BATCH="${MICRO_BATCH:-1}"
MASTER_PORT="${MASTER_PORT:-29549}"

PRETRAIN="${PRETRAIN:-${CHATJOY_DIR}/saves/gemma4-26b-a4b/sft2_ddp_lora_3gpu_567_cut3072_merged}"
TRAIN_FILE="${TRAIN_FILE:-${CHATJOY_DIR}/data/openrlhf_rm/rm_train_v6.2_fg_combined.jsonl}"
EVAL_FILE="${EVAL_FILE:-${CHATJOY_DIR}/data/openrlhf_rm/rm_eval_v6.2_fg_combined.jsonl}"
OPENRLHF_DIR="${OPENRLHF_DIR:-/home/ps/hanjiaqi/OpenRLHF}"
GEMMA4_PYTHON="${GEMMA4_PYTHON:-/home/ps/miniconda3/envs/gemma4-rl/bin/python}"
GEMMA4_DEEPSPEED="${GEMMA4_DEEPSPEED:-/home/ps/miniconda3/envs/gemma4-rl/bin/deepspeed}"

if [ "${LOAD_CHECKPOINT:-false}" = "true" ]; then
    echo "ERROR: tonight wrapper is for a fresh run; do not set LOAD_CHECKPOINT=true."
    exit 1
fi

for path in \
    "${BASE_LAUNCHER}" \
    "${GEMMA4_PYTHON}" \
    "${GEMMA4_DEEPSPEED}" \
    "${OPENRLHF_DIR}/openrlhf/cli/train_rm.py" \
    "${PRETRAIN}" \
    "${TRAIN_FILE}" \
    "${EVAL_FILE}"
do
    if [ ! -e "${path}" ]; then
        echo "ERROR: required path missing: ${path}"
        exit 1
    fi
done

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_LIST}"
GPU_COUNT="${#GPU_ARRAY[@]}"
if (( GPU_COUNT != 8 )); then
    echo "ERROR: expected 8 GPUs for this wrapper, got ${GPU_COUNT}: ${GPU_LIST}"
    exit 1
fi
if (( GLOBAL_BATCH % GPU_COUNT != 0 )); then
    echo "ERROR: GLOBAL_BATCH=${GLOBAL_BATCH} is not divisible by GPU_COUNT=${GPU_COUNT}."
    exit 1
fi

FREE_GB=$(df -BG "${SAVE_ROOT}" | awk 'NR==2 {gsub("G","",$4); print $4}')
if (( FREE_GB < 200 )); then
    echo "ERROR: free disk under ${SAVE_ROOT} is ${FREE_GB}G; need at least 200G."
    exit 1
fi

if [ -e "${SAVE_PATH}" ] || [ -e "${CKPT_PATH}" ]; then
    echo "ERROR: output already exists:"
    echo "  SAVE_PATH=${SAVE_PATH}"
    echo "  CKPT_PATH=${CKPT_PATH}"
    echo "Set RUN_ID to a new name if you really want another run."
    exit 1
fi

mkdir -p "${SAVE_PATH}" "${CKPT_PATH}"

export HF_HOME="${HF_HOME:-${CHATJOY_DIR}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CHATJOY_DIR}/.cache/xdg}"
export TMPDIR="${TMPDIR:-${SAVE_PATH}/tmp}"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${XDG_CACHE_HOME}" "${TMPDIR}"

export RUN_ID SAVE_PATH CKPT_PATH
export GPU_LIST GLOBAL_BATCH MICRO_BATCH MASTER_PORT
export PRETRAIN TRAIN_FILE EVAL_FILE
export GEMMA4_PYTHON GEMMA4_DEEPSPEED
export LOAD_CHECKPOINT=false
export DEEPSPEED_INCLUDE="${DEEPSPEED_INCLUDE:-localhost:${GPU_LIST}}"
export OPENRLHF_FORCE_TEXT_ONLY_GEMMA4="${OPENRLHF_FORCE_TEXT_ONLY_GEMMA4:-1}"
export OPENRLHF_DATASET_NUM_PROC="${OPENRLHF_DATASET_NUM_PROC:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"

# Keep DeepSpeed optimizer checkpoints for true resume, and also save HF adapter
# checkpoints as model-artifact fallback for merge/eval if DS resume hits 4-bit
# metadata issues.
EXTRA_ARGS_BASE="--save_hf_ckpt --max_ckpt_num 4 --max_ckpt_mem 120"
if [[ " ${EXTRA_ARGS:-} " != *" --save_hf_ckpt "* ]]; then
    export EXTRA_ARGS="${EXTRA_ARGS:-} ${EXTRA_ARGS_BASE}"
else
    export EXTRA_ARGS="${EXTRA_ARGS:-}"
fi

{
    echo "RUN_ID=${RUN_ID}"
    echo "SAVE_PATH=${SAVE_PATH}"
    echo "CKPT_PATH=${CKPT_PATH}"
    echo "GPU_LIST=${GPU_LIST}"
    echo "GLOBAL_BATCH=${GLOBAL_BATCH}"
    echo "MICRO_BATCH=${MICRO_BATCH}"
    echo "MASTER_PORT=${MASTER_PORT}"
    echo "PRETRAIN=${PRETRAIN}"
    echo "TRAIN_FILE=${TRAIN_FILE}"
    echo "EVAL_FILE=${EVAL_FILE}"
    echo "HF_HOME=${HF_HOME}"
    echo "HF_DATASETS_CACHE=${HF_DATASETS_CACHE}"
    echo "XDG_CACHE_HOME=${XDG_CACHE_HOME}"
    echo "TMPDIR=${TMPDIR}"
    echo "OPENRLHF_DATASET_NUM_PROC=${OPENRLHF_DATASET_NUM_PROC}"
    echo "EXTRA_ARGS=${EXTRA_ARGS}"
    echo "free_disk_gb=${FREE_GB}"
    echo "train_samples=$(wc -l < "${TRAIN_FILE}")"
    echo "eval_samples=$(wc -l < "${EVAL_FILE}")"
    nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader,nounits || true
} | tee "${SAVE_PATH}/run_env.txt"

echo ""
echo "Prepared fresh 8-GPU run:"
echo "  SAVE_PATH=${SAVE_PATH}"
echo "  CKPT_PATH=${CKPT_PATH}"
echo "  log=${SAVE_PATH}/train.log"
echo ""

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "DRY_RUN=1, preflight finished without launching training."
    exit 0
fi

exec bash "${BASE_LAUNCHER}"
