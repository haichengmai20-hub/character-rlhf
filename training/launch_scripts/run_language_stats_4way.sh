#!/usr/bin/env bash
set -euo pipefail

# =========================
# One-click config (edit here)
# =========================
PROJECT_DIR="/home/ps/hanjiaqi/LlamaFactory/chatjoy"
SCRIPT_NAME="analyze_language_distribution.py"

NUM_SHARDS=4
WORKERS_PER_SHARD=8
ENGINE="process"           # process | thread | auto
PROGRESS_EVERY=100000
MAX_LINES=0                 # 0 means full dataset
GPU_IDS=(4 5 6 7)           # one GPU id per shard
INPUTS="chatjoy_hq_full.json"

# Keep logs/results separated per run timestamp
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$PROJECT_DIR/langstats_runs/$RUN_TAG"
mkdir -p "$OUT_DIR"
export OUT_DIR

cd "$PROJECT_DIR"

echo "[INFO] Run dir: $OUT_DIR"
echo "[INFO] Launching $NUM_SHARDS shards..."

if [[ "${#GPU_IDS[@]}" -ne "$NUM_SHARDS" ]]; then
  echo "[ERROR] GPU_IDS count (${#GPU_IDS[@]}) must equal NUM_SHARDS ($NUM_SHARDS)"
  exit 1
fi

pids=()
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  log_file="$OUT_DIR/shard_${shard}.log"
  gpu_id="${GPU_IDS[$shard]}"

  # Map shard to a specific GPU id
  export CUDA_VISIBLE_DEVICES="$gpu_id"

  LANGSTAT_NUM_SHARDS="$NUM_SHARDS" \
  LANGSTAT_SHARD_INDEX="$shard" \
  LANGSTAT_WORKERS="$WORKERS_PER_SHARD" \
  LANGSTAT_ENGINE="$ENGINE" \
  LANGSTAT_INPUTS="$INPUTS" \
  LANGSTAT_PROGRESS_EVERY="$PROGRESS_EVERY" \
  LANGSTAT_MAX_LINES="$MAX_LINES" \
  LANGSTAT_OUT_CSV_TEMPLATE="$OUT_DIR/language_stats_shard_{shard}.csv" \
  python "$SCRIPT_NAME" > "$log_file" 2>&1 &

  pid=$!
  pids+=("$pid")
  echo "[INFO] shard=$shard gpu=$gpu_id pid=$pid log=$log_file"
done

echo "[INFO] Waiting for all shards to finish..."
failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done

if [[ "$failed" -ne 0 ]]; then
  echo "[ERROR] One or more shard jobs failed. Check logs in $OUT_DIR"
  exit 1
fi

echo "[INFO] All shards completed. Merging CSV..."

python - <<'PY'
import csv
from collections import defaultdict
from pathlib import Path
import os

out_dir = Path(os.environ["OUT_DIR"])
files = sorted(out_dir.glob("language_stats_shard_*.csv"))
if not files:
    raise SystemExit(f"No shard csv files found under: {out_dir}")

agg = defaultdict(lambda: {"language": "", "major_regions": "", "count": 0})

total = 0
for fp in files:
    with fp.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = row["lang_code"]
            count = int(row["count"])
            agg[code]["language"] = row["language"]
            agg[code]["major_regions"] = row.get("major_regions", "")
            agg[code]["count"] += count
            total += count

rows = sorted(agg.items(), key=lambda x: x[1]["count"], reverse=True)
merged_path = out_dir / "language_stats_merged.csv"
with merged_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["lang_code", "language", "major_regions", "count", "ratio_percent"])
    for code, item in rows:
        ratio = (item["count"] * 100 / total) if total else 0.0
        writer.writerow([code, item["language"], item["major_regions"], item["count"], f"{ratio:.6f}"])

print(f"[INFO] Merged CSV: {merged_path}")
PY

echo "[INFO] Done. Outputs: $OUT_DIR"
