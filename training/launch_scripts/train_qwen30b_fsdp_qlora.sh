#!/bin/bash
set -euo pipefail

ENV_NAME="tslib"
CODE_ROOT="/home/ps/hanjiaqi/LlamaFactory"
ACC_CONFIG="${CODE_ROOT}/examples/accelerate/fsdp_config_offload.yaml"
TRAIN_YAML="${CODE_ROOT}/chatjoy/qwen30b_fsdp_qlora_chatjoy.yaml"

if [[ ! -f "${ACC_CONFIG}" ]]; then
  echo "Missing accelerate config: ${ACC_CONFIG}"
  exit 1
fi

if [[ ! -f "${TRAIN_YAML}" ]]; then
  echo "Missing train yaml: ${TRAIN_YAML}"
  exit 1
fi

IDLE_GPUS="$(python -c "import subprocess; rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True).strip().splitlines(); idle=[r.split(',')[0].strip() for r in rows if int(r.split(',')[1].strip())<=200]; print(','.join(idle))")"

if [[ -z "${IDLE_GPUS}" ]]; then
  echo "No idle GPUs found."
  exit 1
fi

# Keep only first 2 idle GPUs for stable FSDP offload launch.
GPU_A="$(echo "${IDLE_GPUS}" | cut -d',' -f1)"
GPU_B="$(echo "${IDLE_GPUS}" | cut -d',' -f2)"

if [[ -z "${GPU_B}" ]]; then
  echo "Need at least 2 idle GPUs, got: ${IDLE_GPUS}"
  exit 1
fi

USE_GPUS="${GPU_A},${GPU_B}"
echo "Using GPUs: ${USE_GPUS}"

cd "${CODE_ROOT}"

CUDA_VISIBLE_DEVICES="${USE_GPUS}" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
DISABLE_VERSION_CHECK=1 \
conda run -n "${ENV_NAME}" accelerate launch \
  --config_file "${ACC_CONFIG}" \
  src/train.py "${TRAIN_YAML}"
