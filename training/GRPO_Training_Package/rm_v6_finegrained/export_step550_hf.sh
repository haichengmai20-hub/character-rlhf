#!/usr/bin/env bash
# Export the verl FSDP actor checkpoint at step 550 to a HuggingFace checkpoint for vLLM generation.

set -euo pipefail

VERL_DIR="${VERL_DIR:-/home/ps/hanjiaqi/verl}"
LOCAL_DIR="${LOCAL_DIR:-/home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/verl/chatjoy_grpo_gemma4rm_phase2_600_20260509_161335/global_step_550/actor}"
TARGET_DIR="${TARGET_DIR:-/home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/gemma4-26b-a4b/chatjoy_grpo_gemma4rm_phase2_step550_hf}"
PYTHON_BIN="${PYTHON_BIN:-/home/ps/miniconda3/envs/gemma4-rl/bin/python}"

if [[ ! -d "${LOCAL_DIR}" ]]; then
  echo "ERROR: actor checkpoint not found: ${LOCAL_DIR}" >&2
  exit 1
fi

echo "Exporting verl FSDP actor checkpoint"
echo "  local_dir : ${LOCAL_DIR}"
echo "  target_dir: ${TARGET_DIR}"

cd "${VERL_DIR}"
"${PYTHON_BIN}" scripts/legacy_model_merger.py merge \
  --backend fsdp \
  --local_dir "${LOCAL_DIR}" \
  --target_dir "${TARGET_DIR}"

echo "Export complete: ${TARGET_DIR}"

