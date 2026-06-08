#!/usr/bin/env bash
# SFT1 全量 Pair Data 重新生成（新规则，GPU 1/2/4/6）
#
# 目标：
#   - 使用 clean32_for_sft1.jsonl 全量重跑
#   - 使用新代码与新规则，失败样本不再 fallback 到 length_mismatch
#   - 避开 GPU0/GPU3/GPU5 的外部占用
#
# 用法：
#   tmux new-session -d -s pairdata_sft1_gpu1246 "bash /home/ps/hanjiaqi/LlamaFactory/chatjoy/launch_pairdata_sft1_rerun_gpu1246.sh"

set -euo pipefail

CHATJOY_DIR="/home/ps/hanjiaqi/LlamaFactory/chatjoy"
MODEL_PATH="${CHATJOY_DIR}/saves/qwen3-30b-moe/clean32_for_sft2_fsdp_qlora_merged"
INPUT_PATH="${CHATJOY_DIR}/data/runs/clean32_for_sft1.jsonl"
OUTPUT_DIR="${CHATJOY_DIR}/data/runs/pair_vllm_outputs_sft1_20260415_newcode_gpu1246"
SCRIPT_PATH="${CHATJOY_DIR}/GRPO_Training_Package/build_rejected_from_sft.py"
VLLM_PYTHON="/home/ps/miniconda3/envs/vllm/bin/python"
PAIR_PYTHON="/home/ps/miniconda3/envs/tslib/bin/python"
SKIP_VLLM_SERVER="${SKIP_VLLM_SERVER:-0}"

VLLM_HOST="127.0.0.1"
VLLM_PORT="8014"
VLLM_BASE_URL="http://${VLLM_HOST}:${VLLM_PORT}"
SERVED_MODEL_NAME="chatjoy-sft-merged"
TP_SIZE=4
GPU_DEVICES="1,2,4,6"
GPU_MEMORY_UTILIZATION=0.90
MAX_MODEL_LEN=4096

NUM_SHARDS=4
MAX_ATTEMPTS=4
MAX_NEW_TOKENS=384
MIN_CHOSEN_SCORE=0.62
MIN_SCORE_GAP=0.18
WRONG_ROLE_WEIGHT=0.60
WRONG_LANGUAGE_WEIGHT=0.35
LENGTH_WEIGHT=0.05
LENGTH_SHORT_PROB=0.25
WRONG_ROLE_ATTEMPTS=6
WRONG_LANGUAGE_ATTEMPTS=5
LENGTH_ATTEMPTS=3
FALLBACK_ON_FAILURE=0
INPUT_ORDERING="language_role_round_robin"
SEED=124

mkdir -p "${OUTPUT_DIR}"

echo "=============================================="
echo "  SFT1 Pair Data Rerun (新规则 + GPU1246)"
echo "  GPU: ${GPU_DEVICES}  TP: ${TP_SIZE}"
echo "  输入: $(wc -l < "${INPUT_PATH}") 条"
echo "  输出: ${OUTPUT_DIR}"
echo "  端口: ${VLLM_PORT}"
echo "=============================================="

if [[ "${SKIP_VLLM_SERVER}" == "1" ]]; then
  echo "[1/2] 复用现有 vLLM 服务: ${VLLM_BASE_URL}"
else
  echo "[1/2] 启动 vLLM 服务..."
  CUDA_VISIBLE_DEVICES="${GPU_DEVICES}" \
    "${VLLM_PYTHON}" -m vllm.entrypoints.openai.api_server \
      --model "${MODEL_PATH}" \
      --served-model-name "${SERVED_MODEL_NAME}" \
      --host "${VLLM_HOST}" \
      --port "${VLLM_PORT}" \
      --tensor-parallel-size "${TP_SIZE}" \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --trust-remote-code \
      --disable-custom-all-reduce \
      --enforce-eager \
      --disable-log-requests \
    > "${OUTPUT_DIR}/vllm_server.log" 2>&1 &

  VLLM_PID=$!
  echo "${VLLM_PID}" > "${OUTPUT_DIR}/vllm_server.pid"
  echo "  vLLM PID: ${VLLM_PID}"
fi

echo "  等待 vLLM 就绪..."
for i in $(seq 1 60); do
  if curl -sf "${VLLM_BASE_URL}/health" > /dev/null 2>&1; then
    echo "  vLLM 服务已就绪（等待 ${i}×5s）"
    break
  fi
  if [[ "${SKIP_VLLM_SERVER}" != "1" ]] && ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "  ERROR: vLLM 进程意外退出，查看 ${OUTPUT_DIR}/vllm_server.log"
    exit 1
  fi
  sleep 5
done

echo ""
echo "[2/2] 启动 ${NUM_SHARDS} 个生成 shard..."

for SHARD_INDEX in $(seq 0 $((NUM_SHARDS - 1))); do
  OUTPUT_PATH="${OUTPUT_DIR}/pairs.shard${SHARD_INDEX}.jsonl"
  LOG_PATH="${OUTPUT_DIR}/pairs.shard${SHARD_INDEX}.log"

  "${PAIR_PYTHON}" "${SCRIPT_PATH}" \
    --backend vllm_openai \
    --model-path "${MODEL_PATH}" \
    --vllm-base-url "${VLLM_BASE_URL}" \
    --vllm-model "${SERVED_MODEL_NAME}" \
    --input "${INPUT_PATH}" \
    --output "${OUTPUT_PATH}" \
    --mode mixed \
    --seed "$((SEED + SHARD_INDEX))" \
    --input-ordering "${INPUT_ORDERING}" \
    --max-attempts "${MAX_ATTEMPTS}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --min-chosen-score "${MIN_CHOSEN_SCORE}" \
    --min-score-gap "${MIN_SCORE_GAP}" \
    --wrong-role-weight "${WRONG_ROLE_WEIGHT}" \
    --wrong-language-weight "${WRONG_LANGUAGE_WEIGHT}" \
    --length-weight "${LENGTH_WEIGHT}" \
    --length-short-prob "${LENGTH_SHORT_PROB}" \
    --wrong-role-attempts "${WRONG_ROLE_ATTEMPTS}" \
    --wrong-language-attempts "${WRONG_LANGUAGE_ATTEMPTS}" \
    --length-attempts "${LENGTH_ATTEMPTS}" \
    --wrong-role-balance-mode source_proportional \
    --wrong-language-balance-mode source_proportional \
    --wrong-language-candidate-set observed_only \
    --length-source-role-balance-mode source_proportional \
    --length-source-language-balance-mode source_proportional \
    --num-shards "${NUM_SHARDS}" \
    --shard-index "${SHARD_INDEX}" \
    --log-every 50 \
    --checkpoint-every 10 \
    --fallback-attempts 2 \
    --fallback-min-score-gap 0.16 \
    --fallback-dimension-gap-scale 0.85 \
    --resume \
    --skip-existing-output \
    > "${LOG_PATH}" 2>&1 &

  echo $! > "${OUTPUT_DIR}/pairs.shard${SHARD_INDEX}.pid"
  echo "  shard ${SHARD_INDEX} PID: $!  → ${OUTPUT_PATH}"
done

echo ""
echo "全部启动完毕。监控命令:"
echo "  tail -f ${OUTPUT_DIR}/vllm_server.log"
echo "  tail -f ${OUTPUT_DIR}/pairs.shard0.log"
echo "  watch -n 10 'wc -l ${OUTPUT_DIR}/pairs.shard*.jsonl 2>/dev/null'"
echo ""
echo "停止所有进程:"
echo "  kill \$(cat ${OUTPUT_DIR}/*.pid)"

wait

echo ""
echo "所有 shard 已完成。"
if [[ "${SKIP_VLLM_SERVER}" != "1" ]]; then
  echo "关闭 vLLM 服务..."
  kill "${VLLM_PID}" 2>/dev/null || true
fi

TOTAL=$(cat "${OUTPUT_DIR}"/pairs.shard*.jsonl 2>/dev/null | wc -l)
echo "本次生成总对数: ${TOTAL}"
