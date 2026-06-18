#!/usr/bin/env bash
# Probe pipeline for v6.9b:
# clean-200 + attack-68 + OOC detection + attack RM scoring + clean comparison.
#
# This script is a reproducibility artifact. Review GPU/process ownership before
# running it on a shared machine.
set -euo pipefail

cd /root/rich_cjk_sft_v69

GEMMA4_CHAR=${GEMMA4_CHAR:-/home/ubuntu/miniconda3/envs/gemma4-character/bin/python3}
GEMMA4_RL=${GEMMA4_RL:-/home/ubuntu/miniconda3/envs/gemma4-rl/bin/python3}
V69B=${V69B:-/root/sft_rich_cjk_v69b_merged_full}

wait_server() {
    local port="$1"
    local label="$2"
    for i in $(seq 1 60); do
        sleep 5
        if no_proxy=localhost,127.0.0.1 curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
            echo "[$(date)] ${label} server ready"
            return 0
        fi
        echo "  waiting ${label}... (${i})"
    done
    echo "[ERROR] ${label} server did not start in time"
    exit 1
}

echo "[$(date)] === Phase 1: start v69b vLLM on GPUs 0-3 ==="
pkill -f "vllm.*v69b" 2>/dev/null || true
sleep 3

CUDA_VISIBLE_DEVICES=0,1,2,3 HF_HUB_OFFLINE=1 \
"${GEMMA4_CHAR}" -m vllm.entrypoints.openai.api_server \
    --model "${V69B}" \
    --served-model-name v69b \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 4096 \
    --trust-remote-code \
    --port 8000 \
    >/tmp/vllm_v69b.log 2>&1 &
V69B_PID=$!
echo "v69b vLLM PID: ${V69B_PID}"
wait_server 8000 v69b

echo "[$(date)] === Phase 2: clean probe ==="
"${GEMMA4_CHAR}" gen_clean_probe.py \
    --model v69b --url http://127.0.0.1:8000 --out v69b_clean200.jsonl \
    2>&1 | tee /tmp/gen_v69b_clean.log

echo "[$(date)] === Phase 3: attack probe ==="
"${GEMMA4_CHAR}" gen_attack_probe.py \
    --model v69b --url http://127.0.0.1:8000 --out v69b_attack_probe.jsonl \
    2>&1 | tee /tmp/gen_v69b_attack.log

echo "[$(date)] === Phase 4: stop v69b vLLM ==="
pkill -f "vllm.*v69b" || kill "${V69B_PID}" 2>/dev/null || true
sleep 5

echo "[$(date)] === Phase 5: OOC detection ==="
"${GEMMA4_CHAR}" detect_ooc.py v69b_attack_probe.jsonl v69b 2>&1 | tee /tmp/ooc_v69b.log

echo "[$(date)] === Phase 6: attack RM scoring on GPUs 4-7 ==="
CUDA_VISIBLE_DEVICES=4,5,6,7 no_proxy=localhost,127.0.0.1 \
"${GEMMA4_RL}" score_attack_rm.py \
    --in v69b_attack_probe.jsonl \
    --out v69b_attack_rm.jsonl \
    --label v69b \
    2>&1 | tee /tmp/rm_v69b.log

echo "[$(date)] === Phase 7: clean comparison table ==="
"${GEMMA4_CHAR}" - <<'PYEOF'
import json
from collections import defaultdict
from pathlib import Path

LANGS = ("zh", "zh-tw", "ja", "ko")

def load(path):
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    buckets = defaultdict(list)
    for row in rows:
        buckets[row["lang"]].append(row)
    return buckets

def avg(rows, key):
    vals = [row[key] for row in rows if key in row]
    return sum(vals) / len(vals) if vals else 0.0

def mono_pct(rows):
    return 100 * sum(1 for row in rows if row.get("has_monologue")) / len(rows) if rows else 0.0

base = load("/root/rich_cjk_sft_v69/base_it_clean200.jsonl")
step0 = load("/root/rich_cjk_sft_v69/step0_clean200.jsonl")
v69b_path = Path("/root/rich_cjk_sft_v69/v69b_clean200.jsonl")
v69b = load(str(v69b_path)) if v69b_path.exists() else {}

header = f"{'lang':<8} {'metric':<12} {'base':>6} {'step0':>7} {'v69b':>7} {'v69b/step0':>12} {'v69b/base':>11}"
print(header)
print("-" * len(header))
for lang in LANGS:
    for metric, fn in [
        ("action", lambda rows: avg(rows, "action_blocks")),
        ("dialog", lambda rows: avg(rows, "dialog_blocks")),
        ("mono%", lambda rows: mono_pct(rows)),
    ]:
        b = fn(base.get(lang, []))
        s = fn(step0.get(lang, []))
        v = fn(v69b.get(lang, [])) if v69b else float("nan")
        vs = f"{v / s * 100:.0f}%" if s > 0 else "-"
        vb = f"{v / b * 100:.0f}%" if b > 0 else "-"
        ok = "PASS" if ((metric == "mono%" and v <= s + 10) or (metric != "mono%" and s > 0 and v / s >= 1.0)) else "FAIL"
        print(f"{lang:<8} {metric:<12} {b:>6.2f} {s:>7.2f} {v:>7.2f} {vs:>10} {ok:>5} {vb:>9}")
PYEOF

echo "[$(date)] === ALL DONE ==="
echo "Outputs: v69b_clean200.jsonl, v69b_attack_probe.jsonl, v69b_attack_rm.jsonl"
echo "Quick checks: grep 'break rate' /tmp/ooc_v69b.log ; grep 'attack-split' /tmp/rm_v69b.log"
