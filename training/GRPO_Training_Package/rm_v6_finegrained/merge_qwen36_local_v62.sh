#!/usr/bin/env bash
# Build RM v6.2 pairs by direction-filtering v6 pairs with the local Qwen3.6 judge.
set -euo pipefail

ROOT="/home/ps/hanjiaqi/LlamaFactory/chatjoy"
PIPELINE="$ROOT/GRPO_Training_Package/rm_v6_finegrained"
PY="${PY:-/home/ps/miniconda3/envs/tslib/bin/python}"

PAIRS="${PAIRS:-$ROOT/data/openrlhf_rm/rm_v6_finegrained_pairs_combined.jsonl}"
JUDGE_B="${JUDGE_B:-$ROOT/data/runs/rm_v6_finegrained/qwen36_local/judged_qwen36_local_all12448.jsonl}"
OUTPUT="${OUTPUT:-$ROOT/data/openrlhf_rm/rm_v6.2_finegrained_pairs_combined.jsonl}"

"$PY" "$PIPELINE/merge_judge_votes.py" \
  --pairs "$PAIRS" \
  --judge-b "$JUDGE_B" \
  --output "$OUTPUT" \
  --cross-judge-name "qwen3.6-35b-a3b-local"
