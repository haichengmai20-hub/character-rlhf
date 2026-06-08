#!/usr/bin/env bash
# RM v2 Score Head Recovery + Hard Negative Mining
# =================================================
# 步骤：
#   1. GPU 4: recover_score_head.py  (~1-2h)
#   2. GPU 4 + GPU 6 并行: mine_hard_negatives.py (shard 0 / shard 1)  (~1-2h)
#   3. merge 两个 shard → hard_negatives_v2.jsonl
#
# 用法:
#   tmux new-session -d -s rm_recovery "bash chatjoy/run_v2_recovery_and_mining.sh"
#   tmux attach -t rm_recovery
#
# GPU 分配:
#   GPU 0-3: RM 训练已完成，现在空闲
#   GPU 4,6: 用于本脚本（彼此独立，不跨 NUMA 通信）
#   GPU 5,7: demo 占用

set -euo pipefail

CHATJOY_DIR="/home/ps/hanjiaqi/LlamaFactory/chatjoy"
PYTHON="/home/ps/miniconda3/envs/tslib/bin/python"

V2_CKPT="${CHATJOY_DIR}/saves/qwen3-30b-moe/chatjoy_rm_v2_filtered"
SCORE_HEAD_OUT="${V2_CKPT}/score_head_recovered.pt"
TRAIN_DATA="${CHATJOY_DIR}/data/openrlhf_rm/rm_train_filtered.jsonl"
EVAL_DATA="${CHATJOY_DIR}/data/openrlhf_rm/rm_eval.jsonl"
OUT_DIR="${CHATJOY_DIR}/data/openrlhf_rm"

mkdir -p "${OUT_DIR}"

echo "=============================================="
echo "  Step 1: Score Head Recovery (GPU 4)"
echo "  预计耗时: ~1-2 小时"
echo "=============================================="

CUDA_VISIBLE_DEVICES=4 "${PYTHON}" \
    "${CHATJOY_DIR}/recover_score_head.py" \
    --ckpt_hf      "${V2_CKPT}" \
    --output       "${SCORE_HEAD_OUT}" \
    --train_data   "${TRAIN_DATA}" \
    --validate_on  "${EVAL_DATA}" \
    --max_len      2048 \
    --epochs       5 \
    --device       cuda:0

echo ""
echo "Score head recovered: ${SCORE_HEAD_OUT}"
echo ""
echo "=============================================="
echo "  Step 2: Hard Negative Mining (GPU 4 + GPU 6 并行)"
echo "  预计耗时: ~1-2 小时"
echo "=============================================="

SHARD0_OUT="${OUT_DIR}/hard_negatives_v2_shard0.jsonl"
SHARD1_OUT="${OUT_DIR}/hard_negatives_v2_shard1.jsonl"

# Shard 0 on GPU 4
CUDA_VISIBLE_DEVICES=4 "${PYTHON}" \
    "${CHATJOY_DIR}/mine_hard_negatives.py" \
    --ckpt_hf      "${V2_CKPT}" \
    --score_head   "${SCORE_HEAD_OUT}" \
    --train_data   "${TRAIN_DATA}" \
    --output       "${SHARD0_OUT}" \
    --gap_threshold -0.3 \
    --max_len      3072 \
    --batch_size   2 \
    --num_shards   2 \
    --shard_index  0 \
    --device       cuda:0 \
    > "${OUT_DIR}/mining_shard0.log" 2>&1 &

PID0=$!
echo "  Shard 0 started (PID=$PID0) → log: ${OUT_DIR}/mining_shard0.log"

# Shard 1 on GPU 6 (映射为 cuda:0，因为 CUDA_VISIBLE_DEVICES=6)
CUDA_VISIBLE_DEVICES=6 "${PYTHON}" \
    "${CHATJOY_DIR}/mine_hard_negatives.py" \
    --ckpt_hf      "${V2_CKPT}" \
    --score_head   "${SCORE_HEAD_OUT}" \
    --train_data   "${TRAIN_DATA}" \
    --output       "${SHARD1_OUT}" \
    --gap_threshold -0.3 \
    --max_len      3072 \
    --batch_size   2 \
    --num_shards   2 \
    --shard_index  1 \
    --device       cuda:0 \
    > "${OUT_DIR}/mining_shard1.log" 2>&1 &

PID1=$!
echo "  Shard 1 started (PID=$PID1) → log: ${OUT_DIR}/mining_shard1.log"

echo "  Waiting for both shards ..."
wait $PID0 && echo "  Shard 0 done ✓" || echo "  Shard 0 FAILED (check log)"
wait $PID1 && echo "  Shard 1 done ✓" || echo "  Shard 1 FAILED (check log)"

echo ""
echo "=============================================="
echo "  Step 3: Merge shards"
echo "=============================================="

MERGED_OUT="${OUT_DIR}/hard_negatives_v2.jsonl"
"${PYTHON}" "${CHATJOY_DIR}/mine_hard_negatives.py" merge \
    "${SHARD0_OUT}" "${SHARD1_OUT}" \
    --output       "${MERGED_OUT}" \
    --gap_threshold -0.3

echo ""
echo "=============================================="
echo "  完成！Bad case pool: ${MERGED_OUT}"
echo "=============================================="
