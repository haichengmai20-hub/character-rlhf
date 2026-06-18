#!/usr/bin/env bash
# Probe pipeline for v6.9c repair:
# clean-200 + attack-68 + OOC detection + attack RM scoring + 4-way clean table.
#
# This script is a reproducibility artifact. Review GPU/process ownership before
# running it on a shared machine.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd /root/rich_cjk_sft_v69

GEMMA4_CHAR=${GEMMA4_CHAR:-/home/ubuntu/miniconda3/envs/gemma4-character/bin/python3}
GEMMA4_RL=${GEMMA4_RL:-/home/ubuntu/miniconda3/envs/gemma4-rl/bin/python3}
V69C=${V69C:-/root/sft_rich_cjk_v69c_merged_full}

find_helper() {
    local name="$1"
    if [[ -f "$name" ]]; then
        printf '%s\n' "$name"
    elif [[ -f "${SCRIPT_DIR}/${name}" ]]; then
        printf '%s\n' "${SCRIPT_DIR}/${name}"
    elif [[ -f "${SCRIPT_DIR}/probe_analysis/${name}" ]]; then
        printf '%s\n' "${SCRIPT_DIR}/probe_analysis/${name}"
    else
        printf '%s\n' "$name"
    fi
}

GEN_CLEAN="$(find_helper gen_clean_probe.py)"
GEN_ATTACK="$(find_helper gen_attack_probe.py)"
DETECT_OOC="$(find_helper detect_ooc.py)"
SCORE_ATTACK_RM="$(find_helper score_attack_rm.py)"

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
    echo "[ERROR] ${label} server did not start"
    exit 1
}

echo "[$(date)] === Phase 1: start v69c vLLM on GPUs 0-3 ==="
pkill -f "vllm.*v69c" 2>/dev/null || true
sleep 3

CUDA_VISIBLE_DEVICES=0,1,2,3 HF_HUB_OFFLINE=1 \
"${GEMMA4_CHAR}" -m vllm.entrypoints.openai.api_server \
    --model "${V69C}" \
    --served-model-name v69c \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 4096 \
    --trust-remote-code \
    --port 8000 \
    >/tmp/vllm_v69c.log 2>&1 &
wait_server 8000 v69c

echo "[$(date)] === Phase 2: clean probe ==="
"${GEMMA4_CHAR}" "${GEN_CLEAN}" --model v69c --url http://127.0.0.1:8000 \
    --out v69c_clean200.jsonl 2>&1 | tee /tmp/gen_v69c_clean.log

echo "[$(date)] === Phase 3: attack probe ==="
"${GEMMA4_CHAR}" "${GEN_ATTACK}" --model v69c --url http://127.0.0.1:8000 \
    --out v69c_attack_probe.jsonl 2>&1 | tee /tmp/gen_v69c_attack.log

echo "[$(date)] === Phase 4: stop v69c vLLM ==="
pkill -f "vllm.*v69c" || true
sleep 5

echo "[$(date)] === Phase 5: OOC detection ==="
"${GEMMA4_CHAR}" "${DETECT_OOC}" v69c_attack_probe.jsonl v69c 2>&1 | tee /tmp/ooc_v69c.log

echo "[$(date)] === Phase 6: attack RM scoring on GPUs 4-7 ==="
CUDA_VISIBLE_DEVICES=4,5,6,7 no_proxy=localhost,127.0.0.1 \
"${GEMMA4_RL}" "${SCORE_ATTACK_RM}" --in v69c_attack_probe.jsonl \
    --out v69c_attack_rm.jsonl --label v69c 2>&1 | tee /tmp/rm_v69c.log

echo "[$(date)] === Phase 7: clean 4-way comparison ==="
"${GEMMA4_CHAR}" - <<'PYEOF'
import json
from collections import defaultdict
from pathlib import Path

LANGS = ("zh", "zh-tw", "ja", "ko")

def load(path):
    buckets = defaultdict(list)
    if not Path(path).exists():
        return buckets
    for line in open(path, encoding="utf-8"):
        if line.strip():
            row = json.loads(line)
            buckets[row["lang"]].append(row)
    return buckets

def avg(rows, key):
    return sum(row[key] for row in rows if key in row) / len(rows) if rows else 0.0

def mono(rows):
    return 100 * sum(1 for row in rows if row.get("has_monologue")) / len(rows) if rows else 0.0

base = load("base_it_clean200.jsonl")
step0 = load("step0_clean200.jsonl")
v69b = load("v69b_clean200.jsonl")
v69c = load("v69c_clean200.jsonl")

header = f"{'lang':<7} {'metric':<8} {'base':>6} {'step0':>7} {'v69b':>7} {'v69c':>7} {'v69c/step0':>11}"
print(header)
print("-" * len(header))
for lang in LANGS:
    for metric, fn in [
        ("action", lambda rows: avg(rows, "action_blocks")),
        ("dialog", lambda rows: avg(rows, "dialog_blocks")),
        ("mono%", lambda rows: mono(rows)),
    ]:
        b = fn(base.get(lang, []))
        s = fn(step0.get(lang, []))
        b9 = fn(v69b.get(lang, []))
        c = fn(v69c.get(lang, []))
        ok = "PASS" if ((metric == "mono%" and c <= s + 10) or (metric != "mono%" and s > 0 and c / s >= 1.0)) else "FAIL"
        vs = f"{c / s * 100:.0f}%" if s > 0 else "-"
        print(f"{lang:<7} {metric:<8} {b:>6.2f} {s:>7.2f} {b9:>7.2f} {c:>7.2f} {vs:>9} {ok}")
PYEOF

echo "[$(date)] === ALL DONE ==="
echo "Quick checks: grep 'break rate' /tmp/ooc_v69c.log ; grep 'attack-split' /tmp/rm_v69c.log"
