#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${PAIR_JUDGE_API_KEY:-}" ]]; then
  echo "PAIR_JUDGE_API_KEY is required" >&2
  exit 2
fi

ROOT="/root/hanjiaqi/LlamaFactory/chatjoy"
OUT_DIR="$ROOT/data/runs/rm_v6_3_roledrop/lever1_full_judge_v63"
LOG_DIR="$ROOT/logs/rm_v6_3_roledrop/lever1_full_judge_v63"

mkdir -p "$OUT_DIR" "$LOG_DIR"
cd "$ROOT/GRPO_Training_Package/rm_v6_finegrained"

exec python judge_candidates_llm_openai.py \
  --candidates "$ROOT/data/runs/rm_v6_finegrained/full/candidates.gemma4_sft_t07.jsonl" \
  --candidates "$ROOT/data/runs/rm_v6_finegrained/full/candidates.gemma4_sft_t10.jsonl" \
  --candidates "$ROOT/data/runs/rm_v6_finegrained/full/candidates.qwen_sft_t07.jsonl" \
  --candidates "$ROOT/data/runs/rm_v6_finegrained/full/candidates.rl550_t07.jsonl" \
  --candidates "$ROOT/data/runs/rm_v6_finegrained/full/candidates.rl550_t10.jsonl" \
  --candidates "$ROOT/data/runs/rm_v6_3_roledrop/full/candidates.base_gemma4it_t07_p08.jsonl" \
  --output "$OUT_DIR/judged_deepseek_v63_lever1_full.jsonl" \
  --audit-output "$OUT_DIR/audit_deepseek_v63_lever1_full.jsonl" \
  --base-url "https://api.deepseek.com/v1" \
  --model "deepseek-chat" \
  --rubric-version "v6.3" \
  --min-candidates 6 \
  --max-candidates 6 \
  --limit 10000 \
  --seed 42 \
  --concurrency 4 \
  --timeout 180 \
  --max-retries 3 \
  --retry-sleep 3 \
  --resume \
  >> "$LOG_DIR/deepseek_full.log" 2>&1
