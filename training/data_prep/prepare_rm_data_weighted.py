#!/usr/bin/env python3
"""
RM 训练数据加权处理。

策略:
  - 角色分布: 已与 SFT 一致，不调整
  - 语言分布: 已与 SFT 一致，不调整
  - rejected_type: 按目标比例对 length_mismatch_long 上采样，
                   并给每条样本附加 sample_weight 供训练时加权 loss

目标比例 (参考 wrong_language 信号强度为基准):
  wrong_language       → 25%   (小幅下调，gap 已够强)
  wrong_role           → 25%   (维持)
  length_mismatch_short → 30%  (从 42% 下调，用 sample_weight 补偿)
  length_mismatch_long  → 20%  (从 6.5% 上调，补偿信号弱 + 样本少)

sample_weight 计算:
  在每个 rejected_type 组内，对 reward_gap 做 min-max 归一化，
  再按 gap 越大 → 权重越高 的原则映射到 [0.4, 1.0]。
  再乘以 type_weight（按目标比例的倒数修正），最终 clip 到 [0.3, 1.2]。

输出:
  rm_train_weighted_train.jsonl   (~95%)
  rm_train_weighted_eval.jsonl    (~5%)
"""

import json, random, math
from collections import defaultdict, Counter
from pathlib import Path

random.seed(42)

INPUT   = Path("/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/rm_train_merged.jsonl")
OUT_DIR = Path("/home/ps/hanjiaqi/LlamaFactory/chatjoy/data")

# ── 目标 rejected_type 比例 ─────────────────────────────────────────────────
TARGET_TYPE_RATIO = {
    "wrong_language":        0.25,
    "wrong_role":            0.25,
    "length_mismatch_short": 0.30,
    "length_mismatch_long":  0.20,
}
EVAL_RATIO = 0.05

# ──────────────────────────────────────────────────────────────────────────────
records = [json.loads(l) for l in INPUT.open()]
total_raw = len(records)
print(f"Loaded {total_raw} records")

# 当前分布
cur_counts = Counter(r["rejected_type"] for r in records)
print("\n--- Current rejected_type distribution ---")
for k, v in cur_counts.most_common():
    print(f"  {k:<30} {v:>5}  ({v/total_raw*100:.1f}%)")

# ── Step1: 按 rejected_type 分组 ──────────────────────────────────────────────
by_type: dict[str, list] = defaultdict(list)
for r in records:
    by_type[r["rejected_type"]].append(r)

# ── Step2: 上采样 / 下采样到目标比例 ─────────────────────────────────────────
# 基准：以最大目标数量的组为参照，其他组按比例缩放
# 先算出各组"如果按目标比例，应该有多少条"
# 以 length_mismatch_short 为锚（它本来最多，缩到 30%）
# target_total = raw_short / 0.30

anchor_type = "length_mismatch_short"
anchor_raw  = cur_counts[anchor_type]
target_total = int(anchor_raw / TARGET_TYPE_RATIO[anchor_type])

target_counts = {t: int(target_total * ratio) for t, ratio in TARGET_TYPE_RATIO.items()}
print(f"\n--- Target counts (total ~{target_total}) ---")
for t, tc in target_counts.items():
    raw = cur_counts.get(t, 0)
    print(f"  {t:<30} raw={raw:>5}  target={tc:>5}  ({'upsample' if tc>raw else 'downsample or keep'})")

balanced = []
for t, target_n in target_counts.items():
    items = by_type.get(t, [])
    if not items:
        continue
    # 优先保留 reward_gap 最大的
    items_sorted = sorted(items, key=lambda x: x["reward_gap"], reverse=True)
    if target_n <= len(items_sorted):
        selected = items_sorted[:target_n]
    else:
        # 有放回上采样到目标数量（只补充差额）
        extra = random.choices(items_sorted, k=target_n - len(items_sorted))
        selected = items_sorted + extra
    balanced.extend(selected)

print(f"\nAfter type rebalancing: {len(balanced)}")

# ── Step3: 计算 sample_weight ──────────────────────────────────────────────
# 同 type 内 reward_gap min-max normalize → [0.4, 1.0]
type_gap_bounds: dict[str, tuple] = {}
for t, items in by_type.items():
    gaps = [r["reward_gap"] for r in items]
    type_gap_bounds[t] = (min(gaps), max(gaps))

def compute_weight(r):
    lo, hi = type_gap_bounds[r["rejected_type"]]
    gap_norm = (r["reward_gap"] - lo) / (hi - lo) if hi > lo else 0.5
    w = 0.4 + 0.6 * gap_norm      # [0.4, 1.0]
    return round(max(0.3, min(1.2, w)), 4)

for r in balanced:
    r["sample_weight"] = compute_weight(r)

# ── Step4: shuffle + train/eval split ─────────────────────────────────────────
random.shuffle(balanced)
eval_n    = max(1, int(len(balanced) * EVAL_RATIO))
eval_set  = balanced[:eval_n]
train_set = balanced[eval_n:]

# ── 写出 ──────────────────────────────────────────────────────────────────────
train_path = OUT_DIR / "rm_train_weighted_train.jsonl"
eval_path  = OUT_DIR / "rm_train_weighted_eval.jsonl"

with train_path.open("w") as f:
    for r in train_set:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
with eval_path.open("w") as f:
    for r in eval_set:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ── 最终统计 ──────────────────────────────────────────────────────────────────
import statistics
print(f"\n{'='*60}")
print(f"Train: {len(train_set)}   Eval: {len(eval_set)}")

rt = Counter(r["rejected_type"] for r in train_set)
print("\n--- rejected_type (train) ---")
for k, v in rt.most_common():
    print(f"  {k:<30} {v:>5}  ({v/len(train_set)*100:.1f}%)")

import re
def extract_char_name(system):
    m = re.search(r'\b([A-Z][a-z]{2,}(?:\s[A-Z][a-z]+)?)\s+is\s+a\b', system)
    return m.group(1) if m else "__unknown__"

role_c = Counter(extract_char_name(r['system']) for r in train_set)
print("\n--- character distribution (train vs SFT target) ---")
sft_ref = {"__unknown__": 40.0, "Noa": 15.8, "Alicia Verden": 13.1,
           "Claire": 12.2, "Liora": 5.9, "Nyra": 5.6, "User": 4.3,
           "Elyndra": 2.5, "Marisha": 0.5}
for k, v in role_c.most_common():
    sft_pct = sft_ref.get(k, 0.0)
    rm_pct  = v / len(train_set) * 100
    diff    = rm_pct - sft_pct
    print(f"  {k:<25} {rm_pct:>5.1f}%  (SFT={sft_pct:.1f}%  Δ={diff:+.1f}%)")

lang_c = Counter(r["language"] for r in train_set)
print("\n--- language distribution (train, top 10) ---")
for k, v in lang_c.most_common(10):
    print(f"  {k:<12} {v:>5}  ({v/len(train_set)*100:.1f}%)")

weights = [r["sample_weight"] for r in train_set]
print(f"\n--- sample_weight stats ---")
print(f"  min={min(weights):.3f}  mean={statistics.mean(weights):.3f}  "
      f"median={statistics.median(weights):.3f}  max={max(weights):.3f}")

print(f"\nOutput: {train_path}")
print(f"Output: {eval_path}")
