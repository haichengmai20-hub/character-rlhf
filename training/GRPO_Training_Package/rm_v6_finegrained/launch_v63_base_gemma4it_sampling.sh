#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/root/hanjiaqi/LlamaFactory/chatjoy}"
PIPELINE_DIR="${ROOT_DIR}/GRPO_Training_Package/rm_v6_finegrained"
PYTHON="${PYTHON:-/home/ps/miniconda3/envs/gemma4-character/bin/python}"

RUN_SPLIT="${RUN_SPLIT:-full}"
SOURCE_RUN_DIR="${SOURCE_RUN_DIR:-${ROOT_DIR}/data/runs/rm_v6_finegrained/${RUN_SPLIT}}"
V63_RUN_DIR="${V63_RUN_DIR:-${ROOT_DIR}/data/runs/rm_v6_3_roledrop/${RUN_SPLIT}}"
PROMPTS_SRC="${PROMPTS_SRC:-${SOURCE_RUN_DIR}/prompts.jsonl}"
PROMPTS_FORMATTED="${PROMPTS_FORMATTED:-${V63_RUN_DIR}/prompts.base_gemma4it_format.jsonl}"
OUTPUT="${OUTPUT:-${V63_RUN_DIR}/candidates.base_gemma4it_t07_p08.jsonl}"

LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/rm_v6_3_roledrop}"
SERVER_LOG="${SERVER_LOG:-${LOG_DIR}/${RUN_SPLIT}.base_gemma4it.vllm.log}"
GEN_LOG="${GEN_LOG:-${LOG_DIR}/${RUN_SPLIT}.base_gemma4it_t07_p08.generate.log}"
SERVER_PID_FILE="${SERVER_PID_FILE:-${LOG_DIR}/${RUN_SPLIT}.base_gemma4it.vllm.pid}"

MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}/models/gemma4-26b-a4b-it-textonly}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-gemma4-base-it}"
SOURCE_LABEL="${SOURCE_LABEL:-base_gemma4it_t07_p08}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8001}"
BASE_URL="http://${HOST}:${PORT}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
TP="${TP:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.88}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-1024}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
MAX_TOKENS="${MAX_TOKENS:-768}"
GEN_CONCURRENCY="${GEN_CONCURRENCY:-4}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-900}"
LIMIT="${LIMIT:-}"

FORMAT_SUFFIX='Output must include narrative actions as `*action*`, dialogues in quotes, and inner thoughts in parentheses. Stay in character.'

mkdir -p "${V63_RUN_DIR}" "${LOG_DIR}"
cd "${PIPELINE_DIR}"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

log() {
  echo "[$(timestamp)] $*" | tee -a "${GEN_LOG}" >&2
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
  if [[ ! -s "${PROMPTS_SRC}" ]]; then
    log "missing prompts: ${PROMPTS_SRC}"
    exit 1
  fi
  if [[ ! -s "${MODEL_PATH}/model.safetensors.index.json" ]]; then
    log "missing base Gemma4-it model: ${MODEL_PATH}"
    exit 1
  fi
}

build_formatted_prompts() {
  log "building formatted prompts: ${PROMPTS_FORMATTED}"
  "${PYTHON}" - "${PROMPTS_SRC}" "${PROMPTS_FORMATTED}" "${FORMAT_SUFFIX}" <<'PY'
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
  "${PYTHON}" - "${BASE_URL}" "${SERVER_READY_TIMEOUT}" "${SERVER_LOG}" <<'PY'
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
  local pid="${1:-}"
  if [[ -z "${pid}" ]]; then
    return
  fi
  log "stopping vLLM process group ${pid}"
  kill -TERM "-${pid}" 2>/dev/null || true
  sleep 15
  kill -KILL "-${pid}" 2>/dev/null || true
}

start_server() {
  log "starting base Gemma4-it vLLM on GPUs ${GPU_LIST}; log=${SERVER_LOG}"
  CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
  VLLM_WORKER_MULTIPROC_METHOD=spawn \
  setsid "${PYTHON}" -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
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
    > "${SERVER_LOG}" 2>&1 &
  echo "$!" | tee "${SERVER_PID_FILE}"
}

generate_candidates() {
  log "generating ${SOURCE_LABEL} -> ${OUTPUT}"
  local cmd=(
    "${PYTHON}" generate_candidates_openai.py
    --prompts "${PROMPTS_FORMATTED}"
    --output "${OUTPUT}"
    --base-url "${BASE_URL}"
    --model "${SERVED_MODEL_NAME}"
    --source-label "${SOURCE_LABEL}"
    --temperature 0.7
    --top-p 0.8
    --max-tokens "${MAX_TOKENS}"
    --concurrency "${GEN_CONCURRENCY}"
    --resume
    --progress-every 25
  )
  if [[ -n "${LIMIT}" ]]; then
    cmd+=(--limit "${LIMIT}")
  fi
  "${cmd[@]}" 2>&1 | tee -a "${GEN_LOG}"
}

main() {
  setup_python_runtime
  ensure_inputs
  build_formatted_prompts

  local server_pid
  server_pid="$(start_server)"
  trap 'stop_server "${server_pid}"' EXIT
  wait_for_server | tee -a "${GEN_LOG}"
  generate_candidates
  stop_server "${server_pid}"
  trap - EXIT
  log "base Gemma4-it sampling complete"
}

main "$@"
