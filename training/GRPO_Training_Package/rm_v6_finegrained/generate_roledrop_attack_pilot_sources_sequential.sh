#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/root/hanjiaqi/LlamaFactory/chatjoy}"
PIPELINE_DIR="${ROOT_DIR}/GRPO_Training_Package/rm_v6_finegrained"
PYTHON="${PYTHON:-/home/ps/miniconda3/envs/gemma4-character/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="python"
fi

RUN_NAME="${RUN_NAME:-roledrop_attack_single_turn_pilot100_v1}"
RUN_DIR="${RUN_DIR:-${ROOT_DIR}/data/runs/rm_v6_3_roledrop/attack_pilot_single_turn_v1}"
PROMPTS="${PROMPTS:-${ROOT_DIR}/data/runs/rm_v6_3_roledrop/roledrop_attack_single_turn_prompts_pilot100_v1.jsonl}"
BASE_PROMPTS="${BASE_PROMPTS:-${RUN_DIR}/prompts.base_gemma4it_format.jsonl}"

LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/rm_v6_3_roledrop}"
PORT="${PORT:-8001}"
HOST="${HOST:-127.0.0.1}"
BASE_URL="http://${HOST}:${PORT}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
ALLOW_GPU7="${ALLOW_GPU7:-0}"
TP="${TP:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.82}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-1024}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
MAX_TOKENS="${MAX_TOKENS:-768}"
GEN_CONCURRENCY="${GEN_CONCURRENCY:-4}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-900}"
RUN_GROUPS="${RUN_GROUPS:-gemma4_sft,qwen_sft,rl_step550,base_gemma4it}"
DRY_RUN="${DRY_RUN:-0}"

GEMMA4_SFT_MODEL="${GEMMA4_SFT_MODEL:-${ROOT_DIR}/saves/gemma4-26b-a4b/sft2_ddp_lora_3gpu_567_cut3072_merged}"
QWEN_SFT_MODEL="${QWEN_SFT_MODEL:-${ROOT_DIR}/saves/qwen3-30b-moe/clean32_for_sft2_fsdp_qlora_merged}"
RL_STEP550_MODEL="${RL_STEP550_MODEL:-${ROOT_DIR}/saves/gemma4-26b-a4b/chatjoy_grpo_gemma4rm_phase2_step550_hf_vllm_fixed_v2}"
BASE_GEMMA4IT_MODEL="${BASE_GEMMA4IT_MODEL:-${ROOT_DIR}/models/gemma4-26b-a4b-it-textonly}"

FORMAT_SUFFIX='Output must include narrative actions as `*action*`, dialogues in quotes, and inner thoughts in parentheses. Stay in character.'

mkdir -p "${RUN_DIR}" "${LOG_DIR}"
cd "${PIPELINE_DIR}"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

log() {
  echo "[$(timestamp)] $*" >&2
}

gpu7_guard() {
  IFS=',' read -r -a gpu_items <<< "${GPU_LIST}"
  for gpu in "${gpu_items[@]}"; do
    if [[ "${gpu}" == "7" && "${ALLOW_GPU7}" != "1" ]]; then
      log "GPU_LIST includes GPU 7, but ALLOW_GPU7 is not set. Refusing to touch GPU 7."
      exit 2
    fi
  done
}

setup_python_runtime() {
  export PYTHONNOUSERSITE=1
  export NO_PROXY="localhost,127.0.0.1,0.0.0.0"
  export no_proxy="${NO_PROXY}"

  local conda_prefix
  conda_prefix="$(dirname "$(dirname "${PYTHON}")")"
  local py_mm
  py_mm="$("${PYTHON}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  local site_packages="${conda_prefix}/lib/python${py_mm}/site-packages"
  local extra_libs=""
  local d
  for d in "${site_packages}"/nvidia/*/lib; do
    if [[ -d "${d}" ]]; then
      extra_libs="${extra_libs:+${extra_libs}:}${d}"
    fi
  done
  if [[ -n "${extra_libs}" ]]; then
    export LD_LIBRARY_PATH="${extra_libs}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  fi
}

ensure_inputs() {
  if [[ ! -s "${PROMPTS}" ]]; then
    log "missing prompts: ${PROMPTS}"
    exit 1
  fi
}

ensure_model() {
  local model_path="$1"
  if [[ ! -s "${model_path}/model.safetensors.index.json" && ! -s "${model_path}/config.json" ]]; then
    log "missing model path or config: ${model_path}"
    exit 1
  fi
}

build_base_prompts() {
  log "building base Gemma4-it formatted prompts: ${BASE_PROMPTS}"
  "${PYTHON}" - "${PROMPTS}" "${BASE_PROMPTS}" "${FORMAT_SUFFIX}" <<'PY'
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
suffix = sys.argv[3]
dst.parent.mkdir(parents=True, exist_ok=True)

count = 0
with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
    for line in fin:
        if not line.strip():
            continue
        row = json.loads(line)
        messages = [dict(message) for message in row.get("messages", [])]
        for message in messages:
            if message.get("role") == "system":
                content = str(message.get("content") or "").rstrip()
                if suffix not in content:
                    message["content"] = content + "\n\n" + suffix
                break
        else:
            messages.insert(0, {"role": "system", "content": suffix})
        row["messages"] = messages
        fout.write(json.dumps(row, ensure_ascii=False) + "\n")
        count += 1

print(json.dumps({"source": str(src), "output": str(dst), "prompts": count}, ensure_ascii=False))
PY
}

wait_for_server() {
  local log_file="$1"
  "${PYTHON}" - "${BASE_URL}" "${SERVER_READY_TIMEOUT}" "${log_file}" <<'PY'
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
    print("\n".join(log_file.read_text(errors="ignore").splitlines()[-120:]))
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

  if [[ "${DRY_RUN}" == "1" ]]; then
    log "DRY_RUN start server: ${served_model_name} from ${model_path}"
    echo ""
    return
  fi

  log "starting vLLM server: ${served_model_name}; log=${server_log}"
  CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
  VLLM_WORKER_MULTIPROC_METHOD=spawn \
  setsid "${PYTHON}" -m vllm.entrypoints.openai.api_server \
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
    --trust-remote-code \
    --no-enable-log-requests \
    "$@" \
    > "${server_log}" 2>&1 &
  echo "$!"
}

generate_source() {
  local prompts="$1"
  local served_model_name="$2"
  local source_label="$3"
  local temperature="$4"
  local top_p="$5"
  local output="$6"

  log "generating source: ${source_label} -> ${output}"
  local cmd=(
    "${PYTHON}" generate_candidates_openai.py
    --prompts "${prompts}"
    --output "${output}"
    --base-url "${BASE_URL}"
    --model "${served_model_name}"
    --source-label "${source_label}"
    --temperature "${temperature}"
    --top-p "${top_p}"
    --max-tokens "${MAX_TOKENS}"
    --concurrency "${GEN_CONCURRENCY}"
    --resume
    --progress-every 25
  )

  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '%q ' "${cmd[@]}"
    printf '\n'
    return
  fi

  "${cmd[@]}"
  log "finished source: ${source_label}"
}

run_model_group() {
  local group_name="$1"
  local model_path="$2"
  local served_model_name="$3"
  local prompts="$4"
  shift 4

  ensure_model "${model_path}"
  local server_log="${LOG_DIR}/${RUN_NAME}.${group_name}.vllm.log"
  local server_pid=""
  server_pid="$(start_server "${model_path}" "${served_model_name}" "${server_log}")"
  if [[ "${DRY_RUN}" != "1" ]]; then
    trap 'stop_server "${server_pid}"' EXIT
    wait_for_server "${server_log}"
  fi

  while [[ "$#" -gt 0 ]]; do
    local source_label="$1"
    local temperature="$2"
    local top_p="$3"
    local output="$4"
    shift 4
    generate_source "${prompts}" "${served_model_name}" "${source_label}" "${temperature}" "${top_p}" "${output}"
  done

  if [[ "${DRY_RUN}" != "1" ]]; then
    stop_server "${server_pid}"
    trap - EXIT
  fi
}

should_run_group() {
  local group_name="$1"
  [[ ",${RUN_GROUPS}," == *",${group_name},"* ]]
}

main() {
  log "run name: ${RUN_NAME}"
  log "run dir: ${RUN_DIR}"
  log "prompts: ${PROMPTS}"
  log "gpu list: ${GPU_LIST}; tp=${TP}; gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
  log "run groups: ${RUN_GROUPS}; dry_run=${DRY_RUN}"

  gpu7_guard
  setup_python_runtime
  ensure_inputs

  if should_run_group "gemma4_sft"; then
    run_model_group \
      "gemma4_sft" \
      "${GEMMA4_SFT_MODEL}" \
      "gemma4-sft" \
      "${PROMPTS}" \
      "gemma4_sft_t07_p08" "0.7" "0.8" "${RUN_DIR}/candidates.gemma4_sft_t07_p08.jsonl" \
      "gemma4_sft_t10_p095" "1.0" "0.95" "${RUN_DIR}/candidates.gemma4_sft_t10_p095.jsonl"
  fi

  if should_run_group "qwen_sft"; then
    run_model_group \
      "qwen_sft" \
      "${QWEN_SFT_MODEL}" \
      "qwen-sft" \
      "${PROMPTS}" \
      "qwen_sft_t07_p08" "0.7" "0.8" "${RUN_DIR}/candidates.qwen_sft_t07_p08.jsonl"
  fi

  if should_run_group "rl_step550"; then
    run_model_group \
      "rl_step550" \
      "${RL_STEP550_MODEL}" \
      "gemma4-rl-step550" \
      "${PROMPTS}" \
      "rl_step550_t07_p08" "0.7" "0.8" "${RUN_DIR}/candidates.rl_step550_t07_p08.jsonl" \
      "rl_step550_t10_p095" "1.0" "0.95" "${RUN_DIR}/candidates.rl_step550_t10_p095.jsonl"
  fi

  if should_run_group "base_gemma4it"; then
    build_base_prompts
    run_model_group \
      "base_gemma4it" \
      "${BASE_GEMMA4IT_MODEL}" \
      "gemma4-base-it" \
      "${BASE_PROMPTS}" \
      "base_gemma4it_t07_p08" "0.7" "0.8" "${RUN_DIR}/candidates.base_gemma4it_t07_p08.jsonl"
  fi

  log "pilot source generation complete"
}

main "$@"
