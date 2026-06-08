#!/usr/bin/env bash
# Re-judge pilot + full RM v6 candidates with local Qwen3.6-35B-A3B via vLLM.
# Intended as the second independent judge for the v6.2 cross-judge pipeline.
set -euo pipefail

ROOT="/home/ps/hanjiaqi/LlamaFactory/chatjoy"
PIPELINE="$ROOT/GRPO_Training_Package/rm_v6_finegrained"
RUN_DIR="$ROOT/data/runs/rm_v6_finegrained/qwen36_local"
LOG_DIR="$ROOT/logs/rm_v6_finegrained"

SERVER_PY="${SERVER_PY:-/home/ps/miniconda3/envs/gemma4-rl/bin/python}"
JUDGE_PY="${JUDGE_PY:-/home/ps/miniconda3/envs/tslib/bin/python}"
MODEL_PATH="${MODEL_PATH:-/home/ps/.cache/modelscope/hub/models/Qwen/Qwen3.6-35B-A3B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.6-35b-a3b-local}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-18083}"
BASE_URL="http://${HOST}:${PORT}/v1"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
TP="${TP:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-4}"
JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-2048}"
JUDGE_TIMEOUT="${JUDGE_TIMEOUT:-240}"
NO_RESPONSE_FORMAT="${NO_RESPONSE_FORMAT:-0}"
JUDGE_NO_RATIONALE="${JUDGE_NO_RATIONALE:-1}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-1200}"
LIMIT="${LIMIT:-99999}"

OUTPUT="${OUTPUT:-$RUN_DIR/judged_qwen36_local_all12448.jsonl}"
AUDIT="${AUDIT:-$RUN_DIR/judged_qwen36_local_all12448_audit.jsonl}"
SERVER_LOG="${SERVER_LOG:-$LOG_DIR/qwen36_local_vllm.log}"
JUDGE_LOG="${JUDGE_LOG:-$LOG_DIR/judge_qwen36_local_all12448.log}"

PILOT="$ROOT/data/runs/rm_v6_finegrained/pilot"
FULL="$ROOT/data/runs/rm_v6_finegrained/full"

mkdir -p "$RUN_DIR" "$LOG_DIR"
cd "$PIPELINE"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

log() {
  echo "[$(timestamp)] $*" >&2
}

wait_for_server() {
  "$JUDGE_PY" - "$BASE_URL" "$SERVER_READY_TIMEOUT" "$SERVER_LOG" <<'PY'
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

base_url = sys.argv[1].rstrip("/")
timeout = int(sys.argv[2])
log_file = Path(sys.argv[3])
url = base_url + "/models"
deadline = time.time() + timeout
last_error = ""

while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if payload.get("data") is not None:
            print(json.dumps({"server": base_url, "status": "ready"}, ensure_ascii=False), flush=True)
            raise SystemExit(0)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        last_error = repr(exc)
        time.sleep(5)

print(json.dumps({"server": base_url, "status": "timeout", "last_error": last_error}, ensure_ascii=False), flush=True)
if log_file.exists():
    print("\n".join(log_file.read_text(errors="ignore").splitlines()[-160:]))
raise SystemExit(1)
PY
}

stop_server() {
  local server_pid="${1:-}"
  if [[ -z "$server_pid" ]]; then
    return
  fi
  log "stopping local Qwen vLLM process group $server_pid"
  kill -TERM "-$server_pid" 2>/dev/null || true
  sleep 15
  kill -KILL "-$server_pid" 2>/dev/null || true
}

start_server() {
  log "starting local Qwen judge server"
  log "model: $MODEL_PATH"
  log "server log: $SERVER_LOG"
  export LD_LIBRARY_PATH="/home/ps/miniconda3/envs/cosyvoice/lib/python3.10/site-packages/nvidia/nccl/lib:/home/ps/miniconda3/envs/cosyvoice/lib/python3.10/site-packages/nvidia/nvshmem/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  CUDA_VISIBLE_DEVICES="$GPU_LIST" \
  VLLM_WORKER_MULTIPROC_METHOD=spawn \
  setsid "$SERVER_PY" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --host "$HOST" \
    --port "$PORT" \
    --dtype bfloat16 \
    --quantization fp8 \
    --tensor-parallel-size "$TP" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --kv-cache-dtype fp8 \
    --calculate-kv-scales \
    --generation-config vllm \
    --enable-prefix-caching \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --no-enable-log-requests \
    > "$SERVER_LOG" 2>&1 &
  echo "$!"
}

run_judge() {
  log "running local Qwen judge"
  log "output: $OUTPUT"
  log "audit:  $AUDIT"
  local cmd=(
    "$JUDGE_PY" "$PIPELINE/judge_candidates_llm_openai.py"
    --candidates "$PILOT/candidates.gemma4_sft_t07.jsonl" \
    --candidates "$PILOT/candidates.gemma4_sft_t10.jsonl" \
    --candidates "$PILOT/candidates.qwen_sft_t07.jsonl" \
    --candidates "$PILOT/candidates.rl550_t07.jsonl" \
    --candidates "$PILOT/candidates.rl550_t10.jsonl" \
    --candidates "$FULL/candidates.gemma4_sft_t07.jsonl" \
    --candidates "$FULL/candidates.gemma4_sft_t10.jsonl" \
    --candidates "$FULL/candidates.qwen_sft_t07.jsonl" \
    --candidates "$FULL/candidates.rl550_t07.jsonl" \
    --candidates "$FULL/candidates.rl550_t10.jsonl" \
    --output "$OUTPUT" \
    --audit-output "$AUDIT" \
    --base-url "$BASE_URL" \
    --model "$SERVED_MODEL_NAME" \
    --api-key local \
    --temperature 0.0 \
    --max-tokens "$JUDGE_MAX_TOKENS" \
    --timeout "$JUDGE_TIMEOUT" \
    --concurrency "$JUDGE_CONCURRENCY" \
    --limit "$LIMIT" \
    --resume
  )
  if [[ "$NO_RESPONSE_FORMAT" == "1" ]]; then
    cmd+=(--no-response-format)
  fi
  if [[ "$JUDGE_NO_RATIONALE" == "1" ]]; then
    cmd+=(--no-rationale)
  fi
  "${cmd[@]}" 2>&1 | tee -a "$JUDGE_LOG"
}

main() {
  log "GPU_LIST=$GPU_LIST TP=$TP port=$PORT limit=$LIMIT concurrency=$JUDGE_CONCURRENCY"
  local server_pid=""
  server_pid="$(start_server)"
  trap 'stop_server "$server_pid"' EXIT
  wait_for_server
  run_judge
  stop_server "$server_pid"
  trap - EXIT
  log "done"
}

main "$@"
