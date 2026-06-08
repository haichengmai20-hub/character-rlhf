#!/usr/bin/env bash
set -euo pipefail

DEMO_SESSION="${DEMO_SESSION:-qwen_rpr_demo_1267}"
TUNNEL_SESSION="${TUNNEL_SESSION:-qwen_rpr_demo_tunnel}"
CHATJOY_ROOT="${CHATJOY_ROOT:-/home/ps/hanjiaqi/LlamaFactory/chatjoy}"
DEMO_SCRIPT="${DEMO_SCRIPT:-${CHATJOY_ROOT}/demo_character.py}"
SERVER_PORT="${SERVER_PORT:-7861}"

tmux kill-session -t "${DEMO_SESSION}" 2>/dev/null || true
tmux kill-session -t "${TUNNEL_SESSION}" 2>/dev/null || true

pkill -f "${DEMO_SCRIPT}" 2>/dev/null || true
pkill -f "frpc_linux_amd64_v0.3 http .* -l ${SERVER_PORT} " 2>/dev/null || true

echo "Stopped demo session ${DEMO_SESSION} and tunnel session ${TUNNEL_SESSION}."
