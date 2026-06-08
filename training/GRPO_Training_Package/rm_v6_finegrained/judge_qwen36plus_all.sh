#!/usr/bin/env bash
# Re-judge all 12,448 prompts (pilot + full) with Qwen3.6-plus via Bailian API.
# Output is cross-validated against DeepSeek in merge_judge_votes.py (v6.2 pipeline).
set -euo pipefail

ROOT="/home/ps/hanjiaqi/LlamaFactory/chatjoy"
PIPELINE="$ROOT/GRPO_Training_Package/rm_v6_finegrained"
PY="/home/ps/miniconda3/envs/tslib/bin/python"

OUTPUT="$ROOT/data/runs/rm_v6_finegrained/qwen36plus/judged_qwen36plus_all12448.jsonl"
AUDIT="$ROOT/data/runs/rm_v6_finegrained/qwen36plus/judged_qwen36plus_all12448_audit.jsonl"
LOG="$ROOT/logs/rm_v6_finegrained/judge_qwen36plus_all12448.log"

if [[ -z "${QWEN_API_KEY:-}" ]]; then
  echo "QWEN_API_KEY is required" >&2
  exit 1
fi
QWEN_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_MODEL="qwen3.6-plus"

PILOT="$ROOT/data/runs/rm_v6_finegrained/pilot"
FULL="$ROOT/data/runs/rm_v6_finegrained/full"

echo "[judge_qwen36plus] Starting at $(date)"
echo "[judge_qwen36plus] Output: $OUTPUT"
echo "[judge_qwen36plus] Log:    $LOG"

"$PY" "$PIPELINE/judge_candidates_llm_openai.py" \
    --candidates "$PILOT/candidates.gemma4_sft_t07.jsonl" \
    --candidates "$PILOT/candidates.gemma4_sft_t10.jsonl" \
    --candidates "$PILOT/candidates.qwen_sft_t07.jsonl" \
    --candidates "$PILOT/candidates.rl550_t07.jsonl" \
    --candidates "$PILOT/candidates.rl550_t10.jsonl" \
    --candidates "$FULL/candidates.gemma4_sft_t07.jsonl" \
    --candidates "$FULL/candidates.gemma4_sft_t10.jsonl" \
    --candidates "$FULL/candidates.qwen_sft_t07.jsonl" \
    --candidates "$FULL/candidates.rl550_t07.jsonl" \
    --candidates "$FULL/candidates.rl550_t10.jsonl" \
    --output "$OUTPUT" \
    --audit-output "$AUDIT" \
    --base-url "$QWEN_BASE_URL" \
    --model "$QWEN_MODEL" \
    --api-key "$QWEN_API_KEY" \
    --temperature 0.0 \
    --max-tokens 2048 \
    --timeout 120 \
    --concurrency 16 \
    --limit 99999 \
    --resume \
    2>&1 | tee "$LOG"

echo "[judge_qwen36plus] Done at $(date)"
echo "[judge_qwen36plus] Output rows: $(wc -l < "$OUTPUT")"
