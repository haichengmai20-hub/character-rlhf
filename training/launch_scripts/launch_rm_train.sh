#!/usr/bin/bash
# RM 训练启动脚本 — v2 (OpenRLHF + DeepSpeed ZeRO-2 + 4-bit QLoRA)
#
# Python 环境: tslib (transformers 4.57.1 + deepspeed 0.16.9)
#
# 架构: ZeRO-2 + bitsandbytes 4-bit + LoRA + adam_offload
#   4-bit 量化基础模型: 57GB BF16 → ~15GB NF4/GPU (实测)
#   ZeRO-2: 模型参数每卡各持有副本（不分片），只分片 optimizer states
#   adam_offload: Adam optimizer 状态放 CPU，GPU 只需要 ~15GB 模型 + 激活值
#   gradient_checkpointing: 降低激活值内存 (peak ~20GB/卡 < 32GB ✓)
#   注: ZeRO-2 前向无 allgather，通信量远小于 ZeRO-3，跨 NUMA 影响有限
#
# GPU 配置:
#   GPU 0-3: NUMA0，用于本次训练
#   GPU 5,7: demo 占用，排除
#   内存估算: ~15GB (4-bit 模型) + ~4-5GB (激活) = ~20GB < 32GB ✓
#
# 训练数据: rm_train_filtered.jsonl (14826 samples)
#   相比原始 rm_train.jsonl (16826 samples):
#   移除了 2000 条 reward_gap < 0.2 的 length_mismatch_long 噪声样本
#
# 用法:
#   tmux new-session -d -s rm_train "cd /home/ps/hanjiaqi/LlamaFactory && bash chatjoy/launch_rm_train.sh"
#   tmux attach -t rm_train

set -euo pipefail

CHATJOY_DIR="/home/ps/hanjiaqi/LlamaFactory/chatjoy"
PRETRAIN="${CHATJOY_DIR}/saves/qwen3-30b-moe/clean32_for_sft2_fsdp_qlora_merged"
TRAIN_FILE="${CHATJOY_DIR}/data/openrlhf_rm/rm_train_filtered.jsonl"
EVAL_FILE="${CHATJOY_DIR}/data/openrlhf_rm/rm_eval.jsonl"
SAVE_PATH="${CHATJOY_DIR}/saves/qwen3-30b-moe/chatjoy_rm_v2_filtered"

TSLIB_PYTHON="/home/ps/miniconda3/envs/tslib/bin/python"
TSLIB_DEEPSPEED="/home/ps/miniconda3/envs/tslib/bin/deepspeed"

mkdir -p "${SAVE_PATH}"

# ── GPU 选择：NUMA0 的 4 张卡（GPU0-3），demo 占用 GPU5,7 ────────────────
export CUDA_VISIBLE_DEVICES=0,1,2,3

# ── NCCL 配置 ──────────────────────────────────────────────────────────────
export NCCL_DEBUG=WARNING
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export TORCH_NCCL_BLOCKING_WAIT=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800  # 30min

# ── DeepSpeed ───────────────────────────────────────────────────────────────
export DS_ACCELERATOR=cuda

echo "=============================================="
echo "  ChatJoy RM Training v2 (ZeRO-2 + 4-bit QLoRA)"
echo "  Env      : tslib (transformers 4.57.1, deepspeed 0.16.9)"
echo "  Pretrain : ${PRETRAIN}"
echo "  Train    : $(wc -l < ${TRAIN_FILE}) samples (filtered, noise removed)"
echo "  Eval     : $(wc -l < ${EVAL_FILE}) samples"
echo "  Output   : ${SAVE_PATH}"
echo "=============================================="

"${TSLIB_DEEPSPEED}" --include localhost:0,1,2,3 \
    --module openrlhf.cli.train_rm \
    --pretrain "${PRETRAIN}" \
    --dataset "${TRAIN_FILE}" \
    --eval_dataset "${EVAL_FILE}" \
    --prompt_key prompt \
    --chosen_key chosen \
    --rejected_key rejected \
    --save_path "${SAVE_PATH}" \
    --save_steps 100 \
    --logging_steps 5 \
    --eval_steps 50 \
    --train_batch_size 64 \
    --micro_train_batch_size 1 \
    --param_dtype bf16 \
    --max_epochs 1 \
    --max_len 3072 \
    --zero_stage 2 \
    --learning_rate 9e-6 \
    --lr_scheduler cosine \
    --lr_warmup_ratio 0.03 \
    --value_head_prefix score \
    --attn_implementation flash_attention_2 \
    --apply_chat_template \
    --gradient_checkpointing \
    --load_in_4bit \
    --adam_offload \
    --lora_rank 64 \
    --lora_alpha 128 \
    --target_modules q_proj k_proj v_proj o_proj \
    --save_hf_ckpt \
    --disable_ds_ckpt \
    2>&1 | tee "${SAVE_PATH}/train.log"

# ── 训练完成后自动生成指标图 ───────────────────────────────────────────────
echo ""
echo "Training finished. Generating metric plots..."
"${TSLIB_PYTHON}" \
    "${CHATJOY_DIR}/plot_training_metrics.py" \
    --save_path "${SAVE_PATH}" && \
    echo "Plots saved to ${SAVE_PATH}/plots/" || \
    echo "Plot generation failed (non-fatal)"
