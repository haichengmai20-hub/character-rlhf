#!/usr/bin/env bash
set -euo pipefail

DEMO_SESSION="${DEMO_SESSION:-qwen_rpr_demo_1267}"
TUNNEL_SESSION="${TUNNEL_SESSION:-qwen_rpr_demo_tunnel}"
CONDA_SH="${CONDA_SH:-/home/ps/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-gemma4-character}"
CHATJOY_ROOT="${CHATJOY_ROOT:-/home/ps/hanjiaqi/LlamaFactory/chatjoy}"
DEMO_SCRIPT="${DEMO_SCRIPT:-${CHATJOY_ROOT}/demo_character.py}"
MODEL_PATH="${MODEL_PATH:-/home/ps/.cache/huggingface/models--ArliAI--Qwen3-30B-A3B-ArliAI-RpR-v4-Fast/snapshots/97239e7358f8638f73b63cc90ef3aff5f41987fe}"
GPU_SET="${GPU_SET:-1,2,6,7}"
SERVER_PORT="${SERVER_PORT:-7861}"
MAX_LEN="${MAX_LEN:-24576}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.82}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-512}"
RESPONSE_MAX_TOKENS="${RESPONSE_MAX_TOKENS:-384}"
TRANSLATION_MAX_TOKENS="${TRANSLATION_MAX_TOKENS:-512}"
LOG_DIR="${LOG_DIR:-${CHATJOY_ROOT}/logs}"
DEMO_LOG="${DEMO_LOG:-${LOG_DIR}/qwen_rpr_demo_1267.log}"
TUNNEL_LOG="${TUNNEL_LOG:-${LOG_DIR}/qwen_rpr_demo_1267_tunnel.log}"
URL_FILE="${URL_FILE:-${LOG_DIR}/qwen_rpr_demo_1267.urls.txt}"
FRPC_BIN="${FRPC_BIN:-/home/ps/.cache/huggingface/gradio/frpc/frpc_linux_amd64_v0.3}"
CERT_FILE="${CERT_FILE:-.gradio/certificate.pem}"

mkdir -p "${LOG_DIR}"

tmux kill-session -t "${DEMO_SESSION}" 2>/dev/null || true
tmux kill-session -t "${TUNNEL_SESSION}" 2>/dev/null || true

pkill -f "frpc_linux_amd64_v0.3 http .* -l ${SERVER_PORT} " 2>/dev/null || true
pkill -f "${DEMO_SCRIPT}" 2>/dev/null || true

tmux new-session -d -s "${DEMO_SESSION}" "source '${CONDA_SH}' && conda activate '${CONDA_ENV}' && DEMO_SELECTED_GPUS='${GPU_SET}' DEMO_BASE_MODEL_PATH='${MODEL_PATH}' DEMO_VLLM_MAX_MODEL_LEN='${MAX_LEN}' DEMO_VLLM_GPU_MEMORY_UTILIZATION='${GPU_MEMORY_UTIL}' DEMO_VLLM_ENFORCE_EAGER='true' DEMO_VLLM_MAX_NUM_SEQS='${MAX_NUM_SEQS}' DEMO_VLLM_MAX_NUM_BATCHED_TOKENS='${MAX_BATCHED_TOKENS}' DEMO_RESPONSE_MAX_TOKENS='${RESPONSE_MAX_TOKENS}' DEMO_TRANSLATION_MAX_TOKENS='${TRANSLATION_MAX_TOKENS}' DEMO_FAST_RESPONSE_MODE='true' DEMO_VLLM_QUANTIZATION='' DEMO_GRADIO_SHARE='false' GRADIO_SERVER_PORT='${SERVER_PORT}' python '${DEMO_SCRIPT}' > '${DEMO_LOG}' 2>&1"

echo "Waiting for demo server on port ${SERVER_PORT}..."
for _ in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:${SERVER_PORT}" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

if ! curl -sf "http://127.0.0.1:${SERVER_PORT}" >/dev/null 2>&1; then
  echo "Demo server did not become ready. Check log: ${DEMO_LOG}" >&2
  exit 1
fi

PUBLIC_URL="$(sed -n 's/.*公网直连地址（需放通 .* 端口）: //p' "${DEMO_LOG}" | tail -n 1)"
TOKEN="$(python - <<'PY'
import secrets
print(secrets.token_urlsafe(24))
PY
)"

: > "${TUNNEL_LOG}"
tmux new-session -d -s "${TUNNEL_SESSION}" "bash -lc \"'${FRPC_BIN}' http -n '${TOKEN}' -l '${SERVER_PORT}' -i 0.0.0.0 --uc --sd random --ue --server_addr gradio-live.com:7000 --disable_log_color --tls_enable --tls_trusted_ca_file '${CERT_FILE}' > '${TUNNEL_LOG}' 2>&1\""

echo "Waiting for Gradio share URL..."
SHARE_URL=""
for _ in $(seq 1 30); do
  SHARE_URL="$(sed -n 's/.*start proxy success: //p' "${TUNNEL_LOG}" | tail -n 1)"
  if [[ -n "${SHARE_URL}" ]]; then
    break
  fi
  if rg -n "start error|login to server failed" "${TUNNEL_LOG}" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

{
  echo "local_url=http://127.0.0.1:${SERVER_PORT}"
  if [[ -n "${PUBLIC_URL}" ]]; then
    echo "public_url=${PUBLIC_URL}"
  fi
  if [[ -n "${SHARE_URL}" ]]; then
    echo "share_url=${SHARE_URL}"
  fi
  echo "demo_session=${DEMO_SESSION}"
  echo "tunnel_session=${TUNNEL_SESSION}"
  echo "demo_log=${DEMO_LOG}"
  echo "tunnel_log=${TUNNEL_LOG}"
} > "${URL_FILE}"

echo "Demo service started."
echo "Local: http://127.0.0.1:${SERVER_PORT}"
if [[ -n "${PUBLIC_URL}" ]]; then
  echo "Public direct: ${PUBLIC_URL}"
fi
if [[ -n "${SHARE_URL}" ]]; then
  echo "Gradio Share: ${SHARE_URL}"
else
  echo "Gradio Share was not obtained. Check tunnel log: ${TUNNEL_LOG}" >&2
  exit 2
fi
