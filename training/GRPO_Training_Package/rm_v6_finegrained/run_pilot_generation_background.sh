#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/ps/hanjiaqi/LlamaFactory/chatjoy"
PIPELINE_DIR="${ROOT_DIR}/GRPO_Training_Package/rm_v6_finegrained"
LOG_DIR="${ROOT_DIR}/logs/rm_v6_finegrained"
mkdir -p "${LOG_DIR}"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/pilot.sequential_${TS}.log"
PID_FILE="${LOG_DIR}/pilot.sequential.pid"

cd "${PIPELINE_DIR}"

if [[ -s "${PID_FILE}" ]]; then
  old_pid="$(cat "${PID_FILE}")"
  if [[ -n "${old_pid}" ]] && ps -p "${old_pid}" >/dev/null 2>&1; then
    echo "existing generation process is still running: pid=${old_pid}"
    echo "log: ${LOG_FILE}"
    exit 0
  fi
fi

nohup bash -lc '
  source /home/ps/miniconda3/etc/profile.d/conda.sh
  conda activate gemma4-rl
  cd /home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/rm_v6_finegrained
  RUN_SPLIT=pilot GPU_LIST=0,1,2,3 bash generate_pilot_sources_sequential.sh
' > "${LOG_FILE}" 2>&1 &

pid="$!"
echo "${pid}" > "${PID_FILE}"
echo "started rm_v6 pilot generation"
echo "pid: ${pid}"
echo "log: ${LOG_FILE}"
echo "${LOG_FILE}" > "${LOG_DIR}/pilot.sequential.latest_log"
