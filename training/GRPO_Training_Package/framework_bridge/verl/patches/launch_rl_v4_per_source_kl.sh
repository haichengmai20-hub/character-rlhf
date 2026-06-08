#!/usr/bin/env bash
# launch_rl_v4_per_source_kl.sh
# Relaunch ChatJoy v6.3 RL with per-data-source KL ENABLED, resuming from the
# latest checkpoint. Mirrors the verified rl_full_v3 launch exactly, adding only:
#   PER_SOURCE_KL=1 KL_ATTACK_COEF=0.0015 KL_CLEAN_COEF=0.005
# Use ONLY if step-200 val is still plateaued (see README_per_source_kl.md).
#
# Prereqs this script ENFORCES before launching:
#   1. the dp_actor.py patch is applied (apply_per_source_kl.py --check == PATCHED)
#   2. no existing rl_full_* tmux session is running (avoid GPU clash)
# It does NOT kill the old run for you — kill it explicitly first:
#   tmux kill-session -t rl_full_v3
set -euo pipefail

PATCH_DIR="/root/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/framework_bridge/verl/patches"
TRAIN_SH="/home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/framework_bridge/verl/train_grpo_lora_actor_verl.sh"
SAVE_DIR="/root/hanjiaqi/LlamaFactory/chatjoy/saves/verl/v6.3_full"
RM_PATH="/root/hanjiaqi/LlamaFactory/chatjoy/saves/gemma4-26b-a4b/chatjoy_rm_v6.3_fg_8gpu_merged_vllm"
DATA_DIR="/root/hanjiaqi/LlamaFactory/chatjoy/data/runs/verl_rl_data_v6.3_full"
LOG="/tmp/rl_v6.3_psKL_savefreq5.log"
SESSION="rl_full_v5"

# --- guard 1: patch applied to BOTH files (abort if any still original) ---
if python3 "$PATCH_DIR/apply_per_source_kl.py" --check | grep -q "original"; then
  echo "[abort] verl per-source-KL patch NOT fully applied. Run: python3 $PATCH_DIR/apply_per_source_kl.py"
  python3 "$PATCH_DIR/apply_per_source_kl.py" --check
  exit 1
fi
echo "[ok] verl patch present (losses.py + ray_trainer.py)"

# --- guard 2: no rl tmux session already running ---
if tmux ls 2>/dev/null | grep -qE "^rl_full_"; then
  echo "[abort] an rl_full_* tmux session is still running:"
  tmux ls 2>/dev/null | grep -E "^rl_full_"
  echo "        kill it first (e.g. tmux kill-session -t rl_full_v3), then re-run."
  exit 1
fi
echo "[ok] no rl_full_* session running"

# --- guard 3: latest checkpoint exists ---
if [[ ! -f "$SAVE_DIR/latest_checkpointed_iteration.txt" ]]; then
  echo "[abort] no latest_checkpointed_iteration.txt in $SAVE_DIR"
  exit 1
fi
echo "[ok] will resume from global_step_$(cat "$SAVE_DIR/latest_checkpointed_iteration.txt")"

# Auto-restart wrapper: config identical to the known-good psKL run (util/activation
# reverted to baseline -> ZERO change to training quality). Only resilience added:
# SAVE_FREQ=25 (lose <=25 steps on a tail OOM) + auto-resume from latest ckpt on crash.
# Stable EXPERIMENT_NAME so tensorboard/wandb stays one continuous run across restarts.
tmux new -d -s "$SESSION" " \
source /home/ps/miniconda3/bin/activate /home/ps/miniconda3/envs/gemma4-rl && \
export PYTHONPATH=/home/ps/hanjiaqi/verl:\${PYTHONPATH:-} && \
EXP=chatjoy_grpo_lora_v6.3_psKL_oomfix_\$(date +%Y%m%d_%H%M%S) && \
for attempt in \$(seq 1 10); do \
  echo \"==== [auto-restart] attempt \$attempt | resume-from step \$(cat $SAVE_DIR/latest_checkpointed_iteration.txt 2>/dev/null) | \$(date) ====\"; \
  PER_SOURCE_KL=1 KL_ATTACK_COEF=0.0015 KL_CLEAN_COEF=0.005 \
  SKIP_PREPARE_VERL_DATA=1 LORA_RANK=8 LORA_ALPHA=8 ENTROPY_COEFF=0 \
  KL_LOSS_COEF=0.005 ROLLOUT_N=6 \
  LR=1.5e-6 LR_SCHEDULER_TYPE=constant LR_WARMUP_STEPS_RATIO=0 \
  DATA_SHUFFLE=True TEST_FREQ=25 SAVE_FREQ=5 \
  MAX_ACTOR_CKPT_TO_KEEP=2 ACTOR_CKPT_ASYNC_SAVE=False VERL_RM_REWARD_CLIP=5.0 \
  RM_MODEL_PATH=$RM_PATH TRAIN_FILE=$DATA_DIR/train.parquet VAL_FILE=$DATA_DIR/val.parquet \
  EXPERIMENT_NAME=\$EXP SAVE_DIR=$SAVE_DIR \
  bash $TRAIN_SH 'actor_rollout_ref.actor.checkpoint.load_contents=[model,optimizer]'; \
  rc=\$?; \
  echo \"==== [auto-restart] attempt \$attempt exited rc=\$rc | \$(date) ====\"; \
  if [ \$rc -eq 0 ]; then echo '[auto-restart] training finished cleanly'; break; fi; \
  echo '[auto-restart] crash (likely tail OOM); ray stop + GPU cleanup, auto-resume in 30s'; \
  ray stop --force >/dev/null 2>&1; sleep 30; \
done 2>&1 | tee -a $LOG"

echo "[ok] launched tmux session '$SESSION', log -> $LOG"
echo "     verify per-source KL engaged once first step starts (expect BOTH lines):"
echo "       grep 'per-source-KL' $LOG"
echo "       [per-source-KL] inject <n>/<batch> attack rows, mask (..,..)   <- trainer injected tensor"
echo "       [per-source-KL] ACTIVE in ppo_loss: attack@0.0015 clean@0.005  <- actor consumed it"
echo "     if neither line appears, tensor path broke; revert + tell Claude."
