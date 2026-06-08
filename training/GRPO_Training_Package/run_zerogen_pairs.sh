#!/usr/bin/env bash
# Zero-generation preference pair builder.
#
# Produces wrong_role pairs (cross_character) and wrong_language pairs
# (cross_lingual) without any model inference — just SFT data cross-swap.
# Run this alongside or after run_pair_generation_vllm.sh (which handles
# length_mismatch).  Outputs can be merged with the vLLM-produced pairs
# before RM training.
#
# Usage:
#   bash run_zerogen_pairs.sh
#
# Override any variable before calling, e.g.:
#   INPUT_PATH=/path/to/sft2.jsonl OUTPUT_DIR=/path/to/out bash run_zerogen_pairs.sh
set -euo pipefail

ENV_NAME="${ENV_NAME:-tslib}"
INPUT_PATH="${INPUT_PATH:-/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/runs/clean32_for_sft1.jsonl}"
# ALL_RECORDS_PATH: cross-reference corpus (defaults to INPUT_PATH when empty).
# Set to a larger dataset (e.g. sft2) to enrich the candidate pool.
ALL_RECORDS_PATH="${ALL_RECORDS_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/runs/zerogen_pairs}"
SCRIPT_PATH="/home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/build_rejected_from_sft.py"
ROLES_DIR="/home/ps/hanjiaqi/LlamaFactory/chatjoy/TOP20CHARACTERS_systemprompt"
PROCESS_SCRIPT="/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/process_chatjoy_data.py"

NUM_SHARDS="${NUM_SHARDS:-4}"
SEED="${SEED:-42}"
MIN_CHOSEN_SCORE="${MIN_CHOSEN_SCORE:-0.55}"
MIN_SCORE_GAP="${MIN_SCORE_GAP:-0.12}"
INPUT_ORDERING="${INPUT_ORDERING:-language_role_round_robin}"
LOG_EVERY="${LOG_EVERY:-50}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10}"
RESUME="${RESUME:-1}"
OVERWRITE_OUTPUT="${OVERWRITE_OUTPUT:-0}"
SKIP_EXISTING_OUTPUT="${SKIP_EXISTING_OUTPUT:-1}"
SAMPLE_SIZE="${SAMPLE_SIZE:-}"

mkdir -p "${OUTPUT_DIR}"

echo "========================================"
echo "Zero-gen pair builder"
echo "  INPUT:        ${INPUT_PATH}"
echo "  ALL_RECORDS:  ${ALL_RECORDS_PATH:-<same as input>}"
echo "  OUTPUT_DIR:   ${OUTPUT_DIR}"
echo "  NUM_SHARDS:   ${NUM_SHARDS}"
echo "========================================"

# ---- helper: build the common argument list for one shard ----
build_cmd() {
    local mode="$1"
    local shard="$2"
    local output_path="${OUTPUT_DIR}/pairs_${mode}.shard${shard}.jsonl"

    local cmd=(
        conda run -n "${ENV_NAME}" python "${SCRIPT_PATH}"
        --backend none
        --mode "${mode}"
        --input "${INPUT_PATH}"
        --output "${output_path}"
        --roles-dir "${ROLES_DIR}"
        --process-script "${PROCESS_SCRIPT}"
        --seed "$((SEED + shard))"
        --input-ordering "${INPUT_ORDERING}"
        --min-chosen-score "${MIN_CHOSEN_SCORE}"
        --min-score-gap "${MIN_SCORE_GAP}"
        --num-shards "${NUM_SHARDS}"
        --shard-index "${shard}"
        --log-every "${LOG_EVERY}"
        --checkpoint-every "${CHECKPOINT_EVERY}"
    )

    if [[ -n "${ALL_RECORDS_PATH}" ]]; then
        cmd+=(--all-records-path "${ALL_RECORDS_PATH}")
    fi
    if [[ -n "${SAMPLE_SIZE}" ]]; then
        cmd+=(--sample-size "${SAMPLE_SIZE}")
    fi
    if [[ "${RESUME}" == "1" ]]; then
        cmd+=(--resume)
    fi
    if [[ "${SKIP_EXISTING_OUTPUT}" == "1" ]]; then
        cmd+=(--skip-existing-output)
    fi
    if [[ "${OVERWRITE_OUTPUT}" == "1" ]]; then
        cmd+=(--overwrite-output)
    fi

    echo "${cmd[@]}"
}

# ---- launch cross_character shards ----
echo ""
echo "Launching ${NUM_SHARDS} cross_character shards..."
for SHARD in $(seq 0 $((NUM_SHARDS - 1))); do
    OUT="${OUTPUT_DIR}/pairs_cross_character.shard${SHARD}.jsonl"
    LOG="${OUTPUT_DIR}/pairs_cross_character.shard${SHARD}.log"
    CMD=$(build_cmd cross_character "${SHARD}")
    echo "  shard ${SHARD} → ${OUT}"
    nohup bash -c "${CMD}" > "${LOG}" 2>&1 &
    echo $! > "${OUTPUT_DIR}/pairs_cross_character.shard${SHARD}.pid"
done

# ---- launch cross_lingual shards ----
echo ""
echo "Launching ${NUM_SHARDS} cross_lingual shards..."
for SHARD in $(seq 0 $((NUM_SHARDS - 1))); do
    OUT="${OUTPUT_DIR}/pairs_cross_lingual.shard${SHARD}.jsonl"
    LOG="${OUTPUT_DIR}/pairs_cross_lingual.shard${SHARD}.log"
    CMD=$(build_cmd cross_lingual "${SHARD}")
    echo "  shard ${SHARD} → ${OUT}"
    nohup bash -c "${CMD}" > "${LOG}" 2>&1 &
    echo $! > "${OUTPUT_DIR}/pairs_cross_lingual.shard${SHARD}.pid"
done

echo ""
echo "All shards launched (${NUM_SHARDS} cross_character + ${NUM_SHARDS} cross_lingual)."
echo "Logs and checkpoints are in ${OUTPUT_DIR}"
echo ""
echo "To monitor progress:"
echo "  tail -f ${OUTPUT_DIR}/pairs_cross_character.shard0.log"
echo "  tail -f ${OUTPUT_DIR}/pairs_cross_lingual.shard0.log"
echo ""
echo "To merge outputs after completion:"
echo "  cat ${OUTPUT_DIR}/pairs_cross_character.shard*.jsonl > ${OUTPUT_DIR}/cross_character_all.jsonl"
echo "  cat ${OUTPUT_DIR}/pairs_cross_lingual.shard*.jsonl   > ${OUTPUT_DIR}/cross_lingual_all.jsonl"
