#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-tslib}"
MODEL_PATH="${MODEL_PATH:-/home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/qwen3-30b-moe/clean32_for_sft2_fsdp_qlora_merged}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-chatjoy-sft-merged}"
INPUT_PATH="${INPUT_PATH:-/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/runs/clean32_for_sft1.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/runs/pair_vllm_outputs}"
SCRIPT_PATH="/home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/build_rejected_from_sft.py"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8001}"
VLLM_BASE_URL="http://${VLLM_HOST}:${VLLM_PORT}"
TP_SIZE="${TP_SIZE:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
NUM_SHARDS="${NUM_SHARDS:-4}"
SAMPLE_SIZE="${SAMPLE_SIZE:-}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-384}"
MIN_CHOSEN_SCORE="${MIN_CHOSEN_SCORE:-0.62}"
MIN_SCORE_GAP="${MIN_SCORE_GAP:-0.18}"
INPUT_ORDERING="${INPUT_ORDERING:-language_role_round_robin}"
WRONG_ROLE_WEIGHT="${WRONG_ROLE_WEIGHT:-0.60}"
WRONG_LANGUAGE_WEIGHT="${WRONG_LANGUAGE_WEIGHT:-0.35}"
LENGTH_WEIGHT="${LENGTH_WEIGHT:-0.05}"
LENGTH_SHORT_PROB="${LENGTH_SHORT_PROB:-0.25}"
WRONG_ROLE_ATTEMPTS="${WRONG_ROLE_ATTEMPTS:-6}"
WRONG_LANGUAGE_ATTEMPTS="${WRONG_LANGUAGE_ATTEMPTS:-5}"
LENGTH_ATTEMPTS="${LENGTH_ATTEMPTS:-3}"
FALLBACK_ON_FAILURE="${FALLBACK_ON_FAILURE:-1}"
FALLBACK_ATTEMPTS="${FALLBACK_ATTEMPTS:-2}"
FALLBACK_MIN_SCORE_GAP="${FALLBACK_MIN_SCORE_GAP:-0.16}"
FALLBACK_DIMENSION_GAP_SCALE="${FALLBACK_DIMENSION_GAP_SCALE:-0.85}"
WRONG_ROLE_BALANCE_MODE="${WRONG_ROLE_BALANCE_MODE:-source_proportional}"
WRONG_LANGUAGE_BALANCE_MODE="${WRONG_LANGUAGE_BALANCE_MODE:-source_proportional}"
WRONG_LANGUAGE_CANDIDATE_SET="${WRONG_LANGUAGE_CANDIDATE_SET:-observed_only}"
LENGTH_SOURCE_ROLE_BALANCE_MODE="${LENGTH_SOURCE_ROLE_BALANCE_MODE:-source_proportional}"
LENGTH_SOURCE_LANGUAGE_BALANCE_MODE="${LENGTH_SOURCE_LANGUAGE_BALANCE_MODE:-source_proportional}"
LOG_EVERY="${LOG_EVERY:-50}"
MODE="${MODE:-mixed}"
SEED="${SEED:-42}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10}"
RESUME="${RESUME:-1}"
OVERWRITE_OUTPUT="${OVERWRITE_OUTPUT:-0}"
SKIP_EXISTING_OUTPUT="${SKIP_EXISTING_OUTPUT:-1}"

mkdir -p "${OUTPUT_DIR}"

echo "Starting vLLM server at ${VLLM_BASE_URL}"
echo "Model: ${MODEL_PATH}"

echo "Run this in a separate terminal if the server is not already up:"
echo "conda run -n ${ENV_NAME} python -m vllm.entrypoints.openai.api_server --model ${MODEL_PATH} --served-model-name ${SERVED_MODEL_NAME} --host ${VLLM_HOST} --port ${VLLM_PORT} --tensor-parallel-size ${TP_SIZE} --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} --max-model-len ${MAX_MODEL_LEN}"

echo "Launching ${NUM_SHARDS} pair-generation shards against ${VLLM_BASE_URL}"
for SHARD_INDEX in $(seq 0 $((NUM_SHARDS - 1))); do
  OUTPUT_PATH="${OUTPUT_DIR}/pairs.shard${SHARD_INDEX}.jsonl"
  CMD=(
    conda run -n "${ENV_NAME}" python "${SCRIPT_PATH}"
    --backend vllm_openai
    --model-path "${MODEL_PATH}"
    --vllm-base-url "${VLLM_BASE_URL}"
    --vllm-model "${SERVED_MODEL_NAME}"
    --input "${INPUT_PATH}"
    --output "${OUTPUT_PATH}"
    --mode "${MODE}"
    --seed "$((SEED + SHARD_INDEX))"
    --input-ordering "${INPUT_ORDERING}"
    --max-attempts "${MAX_ATTEMPTS}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --min-chosen-score "${MIN_CHOSEN_SCORE}"
    --min-score-gap "${MIN_SCORE_GAP}"
    --wrong-role-weight "${WRONG_ROLE_WEIGHT}"
    --wrong-language-weight "${WRONG_LANGUAGE_WEIGHT}"
    --length-weight "${LENGTH_WEIGHT}"
    --length-short-prob "${LENGTH_SHORT_PROB}"
    --wrong-role-attempts "${WRONG_ROLE_ATTEMPTS}"
    --wrong-language-attempts "${WRONG_LANGUAGE_ATTEMPTS}"
    --length-attempts "${LENGTH_ATTEMPTS}"
    --wrong-role-balance-mode "${WRONG_ROLE_BALANCE_MODE}"
    --wrong-language-balance-mode "${WRONG_LANGUAGE_BALANCE_MODE}"
    --wrong-language-candidate-set "${WRONG_LANGUAGE_CANDIDATE_SET}"
    --length-source-role-balance-mode "${LENGTH_SOURCE_ROLE_BALANCE_MODE}"
    --length-source-language-balance-mode "${LENGTH_SOURCE_LANGUAGE_BALANCE_MODE}"
    --num-shards "${NUM_SHARDS}"
    --shard-index "${SHARD_INDEX}"
    --log-every "${LOG_EVERY}"
    --checkpoint-every "${CHECKPOINT_EVERY}"
    --fallback-attempts "${FALLBACK_ATTEMPTS}"
    --fallback-min-score-gap "${FALLBACK_MIN_SCORE_GAP}"
    --fallback-dimension-gap-scale "${FALLBACK_DIMENSION_GAP_SCALE}"
  )

  if [[ -n "${SAMPLE_SIZE}" ]]; then
    CMD+=(--sample-size "${SAMPLE_SIZE}")
  fi

  if [[ "${RESUME}" == "1" ]]; then
    CMD+=(--resume)
  fi

  if [[ "${SKIP_EXISTING_OUTPUT}" == "1" ]]; then
    CMD+=(--skip-existing-output)
  fi

  if [[ "${FALLBACK_ON_FAILURE}" == "1" ]]; then
    CMD+=(--fallback-on-failure)
  fi

  if [[ "${OVERWRITE_OUTPUT}" == "1" ]]; then
    CMD+=(--overwrite-output)
  fi

  echo "Starting shard ${SHARD_INDEX}: ${OUTPUT_PATH}"
  nohup "${CMD[@]}" > "${OUTPUT_DIR}/pairs.shard${SHARD_INDEX}.log" 2>&1 &
  echo $! > "${OUTPUT_DIR}/pairs.shard${SHARD_INDEX}.pid"
done

echo "All shards launched. Logs are in ${OUTPUT_DIR}" 
