#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/hanjiaqi/LlamaFactory/chatjoy"
PIPELINE="$ROOT/GRPO_Training_Package/rm_v6_finegrained"
OUT_DIR="$ROOT/data/runs/rm_v6_3_roledrop/lever1_full_judge_v63"
LOG_DIR="$ROOT/logs/rm_v6_3_roledrop/lever1_full_judge_v63"

BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
MODEL="${MODEL:-qwen3_6_35b_a3b}"
LIMIT="${LIMIT:-10000}"
CONCURRENCY="${CONCURRENCY:-4}"
TIMEOUT="${TIMEOUT:-300}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
OUTPUT="${OUTPUT:-$OUT_DIR/judged_qwen3_6_35b_a3b_v63_lever1_full.jsonl}"
AUDIT="${AUDIT:-$OUT_DIR/audit_qwen3_6_35b_a3b_v63_lever1_full.jsonl}"

mkdir -p "$OUT_DIR" "$LOG_DIR"
cd "$PIPELINE"

exec python judge_candidates_llm_openai.py \
  --candidates "$ROOT/data/runs/rm_v6_finegrained/full/candidates.gemma4_sft_t07.jsonl" \
  --candidates "$ROOT/data/runs/rm_v6_finegrained/full/candidates.gemma4_sft_t10.jsonl" \
  --candidates "$ROOT/data/runs/rm_v6_finegrained/full/candidates.qwen_sft_t07.jsonl" \
  --candidates "$ROOT/data/runs/rm_v6_finegrained/full/candidates.rl550_t07.jsonl" \
  --candidates "$ROOT/data/runs/rm_v6_finegrained/full/candidates.rl550_t10.jsonl" \
  --candidates "$ROOT/data/runs/rm_v6_3_roledrop/full/candidates.base_gemma4it_t07_p08.jsonl" \
  --output "$OUTPUT" \
  --audit-output "$AUDIT" \
  --base-url "$BASE_URL" \
  --model "$MODEL" \
  --api-key local \
  --rubric-version "v6.3" \
  --min-candidates 6 \
  --max-candidates 6 \
  --limit "$LIMIT" \
  --seed 42 \
  --concurrency "$CONCURRENCY" \
  --timeout "$TIMEOUT" \
  --max-tokens "$MAX_TOKENS" \
  --max-retries 3 \
  --retry-sleep 3 \
  --disable-thinking \
  --no-response-format \
  --no-rationale \
  --resume \
  >> "$LOG_DIR/qwen_cross_judge_v63.log" 2>&1
