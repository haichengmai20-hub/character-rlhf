#!/usr/bin/env bash
# launch_rl_v6.4_defense.sh
# 全新 GRPO v6.4:actor/参考 = 防御 SFT 合并后的完整多模态模型 sft2_defense_merged_full
# (= sft2_merged 的 vision 塔 + defense-LoRA 合进 LM,与 v6.3 actor 同构,verl 零改动)。
# 相对 v6.3 psKL 仅改:MODEL_PATH(新 base)+ 全新 SAVE_DIR(不 resume,从 step0 起)+ EXP/SESSION/LOG。
# 保留:per-source-KL、reward 链(anti_chain=anti_refusal+anti_repetition+anti_ooc+length_shaping)、
#       RM v6.3(不重训)、SAVE_FREQ=5、自动重启。
# GPU 约定(2026-06-05 起,默认,所有未来启动一致): RM 落物理 GPU0-3 + actor 落物理 GPU4-7,
#   GPU7 无限制(旧"GPU7 绝不碰"已作废)。实现 = 设 CUDA_VISIBLE_DEVICES=4,5,6,7,0,1,2,3:
#   Ray 把 global_pool(actor) 分到前 4 个逻辑卡 = 物理 4,5,6,7;reward_pool(RM) 分到后 4 个 = 物理 0,1,2,3。
set -euo pipefail

PATCH_DIR="/root/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/framework_bridge/verl/patches"
TRAIN_SH="/home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/framework_bridge/verl/train_grpo_lora_actor_verl.sh"
ACTOR_PATH="/root/hanjiaqi/LlamaFactory/chatjoy/saves/gemma4-26b-a4b/rl_v6.4_step160_merged"
SAVE_DIR="/root/hanjiaqi/LlamaFactory/chatjoy/saves/verl/v6.5_full"
RM_PATH="/root/hanjiaqi/LlamaFactory/chatjoy/saves/gemma4-26b-a4b/chatjoy_rm_v6.3_fg_8gpu_merged_vllm"
DATA_DIR="/root/hanjiaqi/LlamaFactory/chatjoy/data/runs/verl_rl_data_v6.3_full"
LOG="/tmp/rl_v6.5.log"
SESSION="rl_full_v65"

# --- guard 1: per-source-KL patch applied ---
if python3 "$PATCH_DIR/apply_per_source_kl.py" --check | grep -q "original"; then
  echo "[abort] verl per-source-KL patch NOT fully applied. Run: python3 $PATCH_DIR/apply_per_source_kl.py"
  exit 1
fi
echo "[ok] verl per-source-KL patch present"

# --- guard 2: no rl tmux session already running ---
if tmux ls 2>/dev/null | grep -qE "^rl_full_"; then
  echo "[abort] an rl_full_* tmux session is still running:"; tmux ls 2>/dev/null | grep -E "^rl_full_"
  exit 1
fi
echo "[ok] no rl_full_* session running"

# --- guard 3: actor model present ---
if [[ ! -f "$ACTOR_PATH/config.json" ]]; then
  echo "[abort] actor model not found: $ACTOR_PATH"; exit 1
fi
echo "[ok] actor = $ACTOR_PATH (fresh start, new SAVE_DIR=$SAVE_DIR)"

# Auto-restart wrapper. attempt 1 = fresh (SAVE_DIR empty -> verl resume_mode=auto -> step0) or
# auto-resume from latest checkpoint (SAVE_FREQ=5). "拉满" OOM resilience: UNLIMITED retries on
# transient crashes (tail OOM), BUT abort after 5 consecutive crashes with NO checkpoint progress
# (= deterministic failure, not transient -> prevents all-night no-progress thrash if a step always OOMs).
tmux new -d -s "$SESSION" " \
source /home/ps/miniconda3/bin/activate /home/ps/miniconda3/envs/gemma4-rl && \
export CUDA_VISIBLE_DEVICES=4,5,6,7,0,1,2,3 && \
export PYTHONPATH=/home/ps/hanjiaqi/verl:\${PYTHONPATH:-} && \
EXP=chatjoy_grpo_lora_v6.5_\$(date +%Y%m%d_%H%M%S) && \
attempt=0; stuck=0; \
while true; do \
  attempt=\$((attempt+1)); \
  cur_ckpt=\$(cat $SAVE_DIR/latest_checkpointed_iteration.txt 2>/dev/null || echo none); \
  echo \"==== [auto-restart] attempt \$attempt | latest=\$cur_ckpt | stuck=\$stuck/5 | \$(date) ====\"; \
  PER_SOURCE_KL=1 KL_ATTACK_COEF=0.0015 KL_CLEAN_COEF=0.003 \
  SKIP_PREPARE_VERL_DATA=1 LORA_RANK=8 LORA_ALPHA=8 ENTROPY_COEFF=0 \
  KL_LOSS_COEF=0.005 ROLLOUT_N=6 \
  LR=1.5e-6 LR_SCHEDULER_TYPE=constant LR_WARMUP_STEPS_RATIO=0 \
  DATA_SHUFFLE=True TEST_FREQ=25 SAVE_FREQ=5 \
  MAX_ACTOR_CKPT_TO_KEEP=2 ACTOR_CKPT_ASYNC_SAVE=False VERL_RM_REWARD_CLIP=5.0 \
  MODEL_PATH=$ACTOR_PATH \
  RM_MODEL_PATH=$RM_PATH TRAIN_FILE=$DATA_DIR/train.parquet VAL_FILE=$DATA_DIR/val.parquet \
  EXPERIMENT_NAME=\$EXP SAVE_DIR=$SAVE_DIR \
  bash $TRAIN_SH 'actor_rollout_ref.actor.checkpoint.load_contents=[model,optimizer]' 'actor_rollout_ref.rollout.temperature=1.2'; \
  rc=\$?; \
  echo \"==== [auto-restart] attempt \$attempt exited rc=\$rc | \$(date) ====\"; \
  if [ \$rc -eq 0 ]; then echo '[auto-restart] training finished cleanly'; break; fi; \
  new_ckpt=\$(cat $SAVE_DIR/latest_checkpointed_iteration.txt 2>/dev/null || echo none); \
  if [ \"\$new_ckpt\" = \"\$cur_ckpt\" ]; then stuck=\$((stuck+1)); else stuck=0; fi; \
  if [ \$stuck -ge 5 ]; then echo '[auto-restart] ABORT: 5 consecutive crashes with NO checkpoint progress -> deterministic failure, giving up'; break; fi; \
  echo \"[auto-restart] crash #\$attempt (likely tail OOM); ray stop + auto-resume in 30s (stuck=\$stuck/5)\"; \
  ray stop --force >/dev/null 2>&1; sleep 30; \
done 2>&1 | tee -a $LOG"

echo "[ok] launched tmux session '$SESSION', log -> $LOG"
echo "     actor=$ACTOR_PATH"
echo "     verify once first step starts:"
echo "       grep -E 'per-source-KL|anti_ooc|global_step|Training Progress' $LOG | head"
