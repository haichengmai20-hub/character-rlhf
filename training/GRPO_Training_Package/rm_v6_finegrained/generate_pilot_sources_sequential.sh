#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/ps/hanjiaqi/LlamaFactory/chatjoy"
PIPELINE_DIR="${ROOT_DIR}/GRPO_Training_Package/rm_v6_finegrained"
RUN_SPLIT="${RUN_SPLIT:-pilot}"
RUN_DIR="${ROOT_DIR}/data/runs/rm_v6_finegrained/${RUN_SPLIT}"
PROMPTS="${RUN_DIR}/prompts.jsonl"
LOG_DIR="${ROOT_DIR}/logs/rm_v6_finegrained"
PORT="${PORT:-8001}"
HOST="${HOST:-127.0.0.1}"
BASE_URL="http://${HOST}:${PORT}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
TP="${TP:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.82}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-3200}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
MAX_TOKENS="${MAX_TOKENS:-768}"
GEN_CONCURRENCY="${GEN_CONCURRENCY:-8}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-900}"
LIMIT="${LIMIT:-}"
RUN_GROUPS="${RUN_GROUPS:-gemma4_sft,qwen_sft,rl_step550}"

GEMMA4_SFT_MODEL="${ROOT_DIR}/saves/gemma4-26b-a4b/sft2_ddp_lora_3gpu_567_cut3072_merged"
QWEN_SFT_MODEL="${ROOT_DIR}/saves/qwen3-30b-moe/clean32_for_sft2_fsdp_qlora_merged"
RL_STEP550_MODEL="${ROOT_DIR}/saves/gemma4-26b-a4b/chatjoy_grpo_gemma4rm_phase2_step550_hf"
RL_STEP550_VLLM_FIXED="${ROOT_DIR}/saves/gemma4-26b-a4b/chatjoy_grpo_gemma4rm_phase2_step550_hf_vllm_fixed_v2"

mkdir -p "${RUN_DIR}" "${LOG_DIR}"
cd "${PIPELINE_DIR}"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

log() {
  echo "[$(timestamp)] $*" >&2
}

ensure_inputs() {
  if [[ ! -s "${PROMPTS}" ]]; then
    log "prompts not found, sampling ${RUN_SPLIT} prompts"
    python sample_prompts.py --mode "${RUN_SPLIT}" --output "${PROMPTS}"
  fi

  if [[ ! -d "${RL_STEP550_MODEL}" ]]; then
    log "RL step550 HF model not found, exporting first"
    bash export_step550_hf.sh
  fi

  if [[ ! -s "${RL_STEP550_VLLM_FIXED}/model.safetensors.index.json" ]]; then
    log "repairing RL step550 HF model for vLLM: ${RL_STEP550_VLLM_FIXED}"
    python fix_gemma4_layer_scalar_safetensors.py \
      --input "${RL_STEP550_MODEL}" \
      --output "${RL_STEP550_VLLM_FIXED}"
  fi
}

wait_for_server() {
  local log_file="$1"
  python - "${BASE_URL}" "${SERVER_READY_TIMEOUT}" "${log_file}" <<'PY'
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

base_url = sys.argv[1].rstrip("/")
timeout = int(sys.argv[2])
log_file = Path(sys.argv[3])
url = base_url + "/v1/models"
deadline = time.time() + timeout
last_error = ""

while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if payload.get("data") is not None:
            print(json.dumps({"server": base_url, "status": "ready"}, ensure_ascii=False))
            raise SystemExit(0)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        last_error = repr(exc)
        time.sleep(5)

print(json.dumps({"server": base_url, "status": "timeout", "last_error": last_error}, ensure_ascii=False))
if log_file.exists():
    lines = log_file.read_text(errors="ignore").splitlines()[-120:]
    print("\n".join(lines))
raise SystemExit(1)
PY
}

stop_server() {
  local server_pid="${1:-}"
  if [[ -z "${server_pid}" ]]; then
    return
  fi
  log "stopping vLLM process group ${server_pid}"
  kill -TERM "-${server_pid}" 2>/dev/null || true
  sleep 15
  kill -KILL "-${server_pid}" 2>/dev/null || true
}

start_server() {
  local model_path="$1"
  local served_model_name="$2"
  local server_log="$3"
  shift 3

  log "starting vLLM server: ${served_model_name}"
  log "server log: ${server_log}"
  CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
  VLLM_WORKER_MULTIPROC_METHOD=spawn \
  setsid python -m vllm.entrypoints.openai.api_server \
    --model "${model_path}" \
    --served-model-name "${served_model_name}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --tensor-parallel-size "${TP}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --enforce-eager \
    --no-enable-log-requests \
    "$@" \
    > "${server_log}" 2>&1 &
  echo "$!"
}

generate_source() {
  local served_model_name="$1"
  local source_label="$2"
  local temperature="$3"
  local top_p="$4"
  local output="$5"

  log "generating source: ${source_label}"
  local cmd=(
    python generate_candidates_openai.py
    --prompts "${PROMPTS}"
    --output "${output}"
    --base-url "${BASE_URL}"
    --model "${served_model_name}"
    --source-label "${source_label}"
    --temperature "${temperature}"
    --top-p "${top_p}"
    --max-tokens "${MAX_TOKENS}"
    --concurrency "${GEN_CONCURRENCY}"
    --resume
  )
  if [[ -n "${LIMIT}" ]]; then
    cmd+=(--limit "${LIMIT}")
  fi
  "${cmd[@]}"
  log "finished source: ${source_label}"
}

run_model_group() {
  local group_name="$1"
  local model_path="$2"
  local served_model_name="$3"
  shift 3

  local server_log="${LOG_DIR}/${RUN_SPLIT}.${group_name}.vllm.log"
  local server_pid=""
  server_pid="$(start_server "${model_path}" "${served_model_name}" "${server_log}")"
  trap 'stop_server "${server_pid}"' EXIT
  wait_for_server "${server_log}"

  while [[ "$#" -gt 0 ]]; do
    local source_label="$1"
    local temperature="$2"
    local top_p="$3"
    local output="$4"
    shift 4
    generate_source "${served_model_name}" "${source_label}" "${temperature}" "${top_p}" "${output}"
  done

  stop_server "${server_pid}"
  trap - EXIT
}

should_run_group() {
  local group_name="$1"
  [[ ",${RUN_GROUPS}," == *",${group_name},"* ]]
}

run_rl_step550_group() {
  local group_name="rl_step550"
  local server_log="${LOG_DIR}/${RUN_SPLIT}.${group_name}.vllm.log"
  local server_pid=""
  server_pid="$(start_server "${RL_STEP550_VLLM_FIXED}" "gemma4-rl-step550" "${server_log}")"
  trap 'stop_server "${server_pid}"' EXIT
  wait_for_server "${server_log}"

  generate_source \
    "gemma4-rl-step550" \
    "rl_step550_t07_p08" \
    "0.7" \
    "0.8" \
    "${RUN_DIR}/candidates.rl550_t07.jsonl"

  generate_source \
    "gemma4-rl-step550" \
    "rl_step550_t10_p095" \
    "1.0" \
    "0.95" \
    "${RUN_DIR}/candidates.rl550_t10.jsonl"

  stop_server "${server_pid}"
  trap - EXIT
}

main() {
  log "run split: ${RUN_SPLIT}"
  log "run dir: ${RUN_DIR}"
  log "gpu list: ${GPU_LIST}; tp=${TP}; gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
  log "generation concurrency: ${GEN_CONCURRENCY}; max_num_seqs=${MAX_NUM_SEQS}; max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}"
  log "run groups: ${RUN_GROUPS}"
  ensure_inputs

  if should_run_group "gemma4_sft"; then
    run_model_group \
      "gemma4_sft" \
      "${GEMMA4_SFT_MODEL}" \
      "gemma4-sft" \
      "gemma4_sft_t07_p08" "0.7" "0.8" "${RUN_DIR}/candidates.gemma4_sft_t07.jsonl" \
      "gemma4_sft_t10_p095" "1.0" "0.95" "${RUN_DIR}/candidates.gemma4_sft_t10.jsonl"
  fi

  if should_run_group "qwen_sft"; then
    run_model_group \
      "qwen_sft" \
      "${QWEN_SFT_MODEL}" \
      "qwen-sft" \
      "qwen_sft_t07_p08" "0.7" "0.8" "${RUN_DIR}/candidates.qwen_sft_t07.jsonl"
  fi

  if should_run_group "rl_step550"; then
    run_rl_step550_group
  fi

  log "all sources generated"
}

main "$@"
