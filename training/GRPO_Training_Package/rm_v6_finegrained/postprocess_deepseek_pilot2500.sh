#!/usr/bin/env bash
set -euo pipefail

JUDGE_SESSION="${1:-rmv6_judge_pilot2500}"
ROOT="/home/ps/hanjiaqi/LlamaFactory/chatjoy"
PIPELINE="$ROOT/GRPO_Training_Package/rm_v6_finegrained"
OUT="$ROOT/data/runs/rm_v6_finegrained/pilot/judged_llm_deepseek_pilot2500.jsonl"
PAIRS="$ROOT/data/openrlhf_rm/rm_v6_finegrained_pairs_pilot.jsonl"
QA="$ROOT/data/runs/rm_v6_finegrained/pilot/qa_llm_deepseek_pilot2500.json"
LOG="$ROOT/logs/rm_v6_finegrained/postprocess_deepseek_pilot2500.log"
PY="/home/ps/miniconda3/envs/tslib/bin/python"

mkdir -p "$(dirname "$LOG")"

while tmux has-session -t "$JUDGE_SESSION" 2>/dev/null; do
  sleep 60
done

{
  echo "[post] judge session ended at $(date)"
  wc -l "$OUT" || true
} > "$LOG"

rows="$(wc -l < "$OUT" 2>/dev/null || echo 0)"
if [[ "$rows" -lt 2400 ]]; then
  echo "[post] judged rows too low: $rows, skip build_pairs" >> "$LOG"
  exit 2
fi

cd "$PIPELINE"
"$PY" build_pairs.py --judged "$OUT" --output "$PAIRS" >> "$LOG" 2>&1
"$PY" qa_dataset.py --judged "$OUT" --pairs "$PAIRS" > "$QA" 2>> "$LOG"
echo "[post] done at $(date)" >> "$LOG"
