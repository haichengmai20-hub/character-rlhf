#!/usr/bin/env python3
"""
合并多个 pair_vllm_outputs 目录的 pairs.shard*.jsonl 文件，
去重后输出 RM 训练格式的 JSONL。

去重键: (system, prompt) —— 同一对话上下文只保留一条，
        优先保留 reward_gap（quality_chosen - quality_rejected）最大的。

输出格式 (每行一个 JSON):
  {
    "system":            str,   # 角色卡 / 系统提示
    "prompt":            str,   # 用户消息
    "chosen":            str,   # 高质量回复
    "rejected":          str,   # 低质量回复
    "quality_chosen":    float,
    "quality_rejected":  float,
    "reward_gap":        float,
    "rejected_type":     str,
    "reason":            str,
    "language":          str
  }
"""

import json
import hashlib
import glob
import os
from collections import defaultdict

# ────────────────────────────────────────────────
# 1. 数据源目录
# ────────────────────────────────────────────────
BASE = "/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/runs"
SOURCES = [
    # SFT1 旧 Run（旧代码，fallback=ON，length 占比 53%，低质量对靠去重淘汰）
    "pair_vllm_outputs_sft1_20260407_retry8x4",
    # SFT1 失败重试 Run
    "pair_vllm_outputs_sft1_20260413_failed_retry4x4",
    "pair_vllm_outputs_sft1_20260414_failed_retry4x4",
    # SFT2 新增数据 Run（新代码，fallback=OFF）
    "pair_vllm_outputs_sft2_new_20260414",
    # SFT1 全量重跑（新代码，fallback=OFF，2026-04-15，GPU1246）
    # 与旧 Run1 同一批 SFT1 数据，去重后保留 reward_gap 更大的那条
    "pair_vllm_outputs_sft1_20260415_newcode_gpu1246",
]

OUTPUT_FILE = "/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/rm_train_merged.jsonl"

# ────────────────────────────────────────────────
# 2. 辅助函数
# ────────────────────────────────────────────────
def make_key(system: str, prompt: str) -> str:
    """用 md5 做定长 key，避免字符串长度爆内存"""
    raw = f"{system}\x00{prompt}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def reward_gap(item: dict) -> float:
    qc = item.get("quality_chosen", 0.0) or 0.0
    qr = item.get("quality_rejected", 0.0) or 0.0
    # metadata 里有更精确的 reward_gap
    meta = item.get("metadata", {})
    if "reward_gap" in meta:
        return float(meta["reward_gap"])
    return float(qc) - float(qr)


def to_rm_record(item: dict) -> dict:
    meta = item.get("metadata", {})
    return {
        "system":           item.get("system", ""),
        "prompt":           item.get("prompt", ""),
        "chosen":           item.get("chosen", ""),
        "rejected":         item.get("rejected", ""),
        "quality_chosen":   float(item.get("quality_chosen", 0.0) or 0.0),
        "quality_rejected": float(item.get("quality_rejected", 0.0) or 0.0),
        "reward_gap":       reward_gap(item),
        "rejected_type":    item.get("rejected_type", ""),
        "reason":           item.get("reason", ""),
        "language":         item.get("language", ""),
        "chosen_role":      meta.get("chosen_role", ""),
    }


# ────────────────────────────────────────────────
# 3. 读取 + 去重
# ────────────────────────────────────────────────
# key -> best_item
best: dict[str, dict] = {}

total_read = 0
skipped_invalid = 0

for src in SOURCES:
    pattern = os.path.join(BASE, src, "pairs.shard*.jsonl")
    files = sorted(glob.glob(pattern))
    src_count = 0
    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    skipped_invalid += 1
                    continue

                # 必须有 prompt / chosen / rejected
                if not item.get("prompt") or not item.get("chosen") or not item.get("rejected"):
                    skipped_invalid += 1
                    continue

                total_read += 1
                src_count += 1

                key = make_key(item.get("system", ""), item.get("prompt", ""))
                gap = reward_gap(item)

                if key not in best or gap > reward_gap(best[key]):
                    best[key] = item

    print(f"[{src}]  read {src_count} lines")

print(f"\nTotal read : {total_read}")
print(f"Skipped    : {skipped_invalid}")
print(f"After dedup: {len(best)}")

# ────────────────────────────────────────────────
# 4. 写出
# ────────────────────────────────────────────────
# 按 reward_gap 降序排列（质量更好的在前面）
records = sorted(best.values(), key=reward_gap, reverse=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
    for item in records:
        out.write(json.dumps(to_rm_record(item), ensure_ascii=False) + "\n")

print(f"\nOutput written to: {OUTPUT_FILE}")
print(f"Total records    : {len(records)}")

# ────────────────────────────────────────────────
# 5. 简要统计
# ────────────────────────────────────────────────
from collections import Counter

rejected_types = Counter(r.get("rejected_type", "") for r in records)
languages      = Counter(r.get("language", "")      for r in records)
gaps           = [reward_gap(r) for r in records]

print("\n--- rejected_type distribution ---")
for k, v in rejected_types.most_common():
    print(f"  {k:<30} {v:>6}")

print("\n--- language distribution (top 15) ---")
for k, v in languages.most_common(15):
    print(f"  {k:<10} {v:>6}")

print(f"\n--- reward_gap stats ---")
import statistics
print(f"  min  = {min(gaps):.4f}")
print(f"  mean = {statistics.mean(gaps):.4f}")
print(f"  median = {statistics.median(gaps):.4f}")
print(f"  max  = {max(gaps):.4f}")
