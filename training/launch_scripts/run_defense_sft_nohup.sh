#!/bin/bash
# Option-B 防御 LoRA:在 sft2_merged 上训一层新 r=16 LoRA。
# 对齐 SFT2 真实启动方式(accelerate + accelerate_ddp_3gpu.yaml + env gemma4-character)。
# GPU 0,5,6(1-4 被 base server 占用、7 绝不碰)。
set -euo pipefail

ENV_NAME="gemma4-character"
CODE_ROOT="/home/ps/hanjiaqi/LlamaFactory"
ACC_CONFIG="${CODE_ROOT}/chatjoy/accelerate_ddp_6gpu.yaml"
TRAIN_YAML="${CODE_ROOT}/chatjoy/gemma4_defense_sft_3gpu_lora.yaml"
CONDA_SH="/home/ps/miniconda3/etc/profile.d/conda.sh"
LOG="/tmp/defense_sft.log"
SESSION="defense_sft"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,3,4,6,7}"

: > "${LOG}"
tmux new-session -d -s "${SESSION}" \
  "bash -lc 'set -euo pipefail; \
    source \"${CONDA_SH}\"; conda activate \"${ENV_NAME}\"; \
    cd \"${CODE_ROOT}\"; \
    export CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}; \
    export DISABLE_VERSION_CHECK=1; \
    exec accelerate launch --config_file \"${ACC_CONFIG}\" src/train.py \"${TRAIN_YAML}\"'"
tmux pipe-pane -o -t "${SESSION}" "cat >> \"${LOG}\""
echo "launched defense SFT (tmux ${SESSION}) on GPUs ${CUDA_DEVICES}, log -> ${LOG}"
