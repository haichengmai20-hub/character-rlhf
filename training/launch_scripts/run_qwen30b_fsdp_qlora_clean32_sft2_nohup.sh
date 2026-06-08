#!/bin/bash
set -euo pipefail

ENV_NAME="tslib"
CODE_ROOT="/home/ps/hanjiaqi/LlamaFactory"
ACC_CONFIG="${CODE_ROOT}/chatjoy/accelerate_fsdp_8gpu.yaml"
TRAIN_YAML="${CODE_ROOT}/chatjoy/qwen30b_fsdp_qlora_clean32_sft2.yaml"
CONDA_SH="/home/ps/miniconda3/etc/profile.d/conda.sh"
LOG_DIR="${CODE_ROOT}/chatjoy/logs"
OUTPUT_DIR="${CODE_ROOT}/chatjoy/saves/qwen3-30b-moe/clean32_for_sft2_fsdp_qlora"
PROGRESS_FILE="${OUTPUT_DIR}/trainer_log.jsonl"
RUN_TAG="qwen30b_fsdp_qlora_clean32_sft2_$(date +%Y%m%d_%H%M%S)"
SESSION_NAME="${RUN_TAG}"
LOG_FILE="${LOG_DIR}/${RUN_TAG}.log"
PID_FILE="${LOG_DIR}/${RUN_TAG}.pid"
PGID_FILE="${LOG_DIR}/${RUN_TAG}.pgid"
SESSION_FILE="${LOG_DIR}/${RUN_TAG}.session"
LATEST_LOG_LINK="${LOG_DIR}/latest.log"
LATEST_PID_FILE="${LOG_DIR}/latest.pid"
LATEST_PGID_FILE="${LOG_DIR}/latest.pgid"
LATEST_SESSION_FILE="${LOG_DIR}/latest.session"

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
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; \
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
    export DISABLE_VERSION_CHECK=1; \
    exec accelerate launch --config_file \"${ACC_CONFIG}\" src/train.py \"${TRAIN_YAML}\"'"

tmux pipe-pane -o -t "${SESSION_NAME}" "cat >> \"${LOG_FILE}\""

echo "${SESSION_NAME}" > "${SESSION_FILE}"
echo "${SESSION_NAME}" > "${LATEST_SESSION_FILE}"
ln -sfn "${LOG_FILE}" "${LATEST_LOG_LINK}"
echo "Started training."
echo "Session file: ${SESSION_FILE}"
echo "PID file: ${PID_FILE}"
echo "Log file: ${LOG_FILE}"
echo "Latest session: ${LATEST_SESSION_FILE}"
echo "Latest PID: ${LATEST_PID_FILE}"
echo "Latest log: ${LATEST_LOG_LINK}"
echo "PGID file: ${PGID_FILE}"
echo "Latest PGID: ${LATEST_PGID_FILE}"
echo "Progress file: ${PROGRESS_FILE}"
