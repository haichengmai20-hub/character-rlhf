#!/bin/bash
set -euo pipefail

ENV_NAME="gemma4-character"
CODE_ROOT="/home/ps/hanjiaqi/LlamaFactory"
ACC_CONFIG="${CODE_ROOT}/chatjoy/accelerate_ddp_3gpu.yaml"
TRAIN_YAML="${CODE_ROOT}/chatjoy/gemma4_sft2_3gpu_ddp_lora_567.yaml"
CONDA_SH="/home/ps/miniconda3/etc/profile.d/conda.sh"
LOG_DIR="${CODE_ROOT}/chatjoy/logs"
OUTPUT_DIR="${CODE_ROOT}/chatjoy/saves/gemma4-26b-a4b/sft2_ddp_lora_3gpu_567_cut3072"
RUN_TAG="gemma4_ddp_qlora_3gpu567_$(date +%Y%m%d_%H%M%S)"
SESSION_NAME="${RUN_TAG}"
LOG_FILE="${LOG_DIR}/${RUN_TAG}.log"
PID_FILE="${LOG_DIR}/${RUN_TAG}.pid"
PGID_FILE="${LOG_DIR}/${RUN_TAG}.pgid"
SESSION_FILE="${LOG_DIR}/${RUN_TAG}.session"
LATEST_LOG_LINK="${LOG_DIR}/latest_gemma4_3gpu567.log"
LATEST_PID_FILE="${LOG_DIR}/latest_gemma4_3gpu567.pid"
LATEST_PGID_FILE="${LOG_DIR}/latest_gemma4_3gpu567.pgid"
LATEST_SESSION_FILE="${LOG_DIR}/latest_gemma4_3gpu567.session"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-5,6,7}"

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "Missing conda init script: ${CONDA_SH}"
  exit 1
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "Missing tmux in PATH."
  exit 1
fi

mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_DIR}"

cd "${CODE_ROOT}"
: > "${LOG_FILE}"

tmux new-session -d -s "${SESSION_NAME}" \
  "bash -lc 'set -euo pipefail; \
    echo \$\$ > \"${PID_FILE}\"; \
    echo \$(ps -o pgid= \$\$ | xargs) > \"${PGID_FILE}\"; \
    echo \$\$ > \"${LATEST_PID_FILE}\"; \
    echo \$(ps -o pgid= \$\$ | xargs) > \"${LATEST_PGID_FILE}\"; \
    source \"${CONDA_SH}\"; \
    conda activate \"${ENV_NAME}\"; \
    cd \"${CODE_ROOT}\"; \
    export CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}; \
    export DISABLE_VERSION_CHECK=1; \
    exec accelerate launch --config_file \"${ACC_CONFIG}\" src/train.py \"${TRAIN_YAML}\"'"

tmux pipe-pane -o -t "${SESSION_NAME}" "cat >> \"${LOG_FILE}\""

echo "${SESSION_NAME}" > "${SESSION_FILE}"
echo "${SESSION_NAME}" > "${LATEST_SESSION_FILE}"
ln -sfn "${LOG_FILE}" "${LATEST_LOG_LINK}"

echo "Started Gemma4 DDP QLoRA training on GPUs ${CUDA_DEVICES}."
echo "Session file: ${SESSION_FILE}"
echo "PID file: ${PID_FILE}"
echo "Log file: ${LOG_FILE}"
echo "Latest session: ${LATEST_SESSION_FILE}"
echo "Latest PID: ${LATEST_PID_FILE}"
echo "Latest log: ${LATEST_LOG_LINK}"
echo "PGID file: ${PGID_FILE}"
echo "Latest PGID: ${LATEST_PGID_FILE}"
