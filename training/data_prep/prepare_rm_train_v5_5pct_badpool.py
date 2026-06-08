#!/usr/bin/env python3
"""
RM v5 data builder:
- keep full rm_train.jsonl (v1 style, no long filtering)
- add bad-case pool to about 5% of total samples

Bad-case pool = hard_negatives_v2.jsonl + hard_negatives_long_soft.jsonl
Sampling policy:
- draw with replacement from the combined bad pool until target count is met
- keep base samples 1x (no reweighting / no deletion)
"""

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path


def load_jsonl(path: Path):
    rows = []
    with path.open() as fin:
        for line in fin:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_rejected_type(row):
    return row.get("extra_info", {}).get("rejected_type", "unknown")


def clone_with_tag(row, source_tag: str, repeat_idx: int):
    new_row = json.loads(json.dumps(row))
    extra = new_row.setdefault("extra_info", {})
    extra["v5_source"] = source_tag
    extra["v5_repeat_idx"] = repeat_idx
    return new_row


def main():
    parser = argparse.ArgumentParser(description="Prepare RM v5 data with ~5% bad-case pool")
    parser.add_argument(
        "--base",
        default="/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/openrlhf_rm/rm_train.jsonl",
    )
    parser.add_argument(
        "--bad-v2",
        default="/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/openrlhf_rm/hard_negatives_v2.jsonl",
    )
    parser.add_argument(
        "--bad-long",
        default="/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/openrlhf_rm/hard_negatives_long_soft.jsonl",
    )
    parser.add_argument(
        "--output",
        default="/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/openrlhf_rm/rm_train_v5_5pct_badpool.jsonl",
    )
    parser.add_argument("--target_bad_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true", default=True)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    base_rows = load_jsonl(Path(args.base))
    bad_v2_rows = load_jsonl(Path(args.bad_v2))
    bad_long_rows = load_jsonl(Path(args.bad_long))

    base_n = len(base_rows)
    bad_pool = [("bad_v2", row) for row in bad_v2_rows] + [("bad_long", row) for row in bad_long_rows]
    bad_pool_n = len(bad_pool)
    if bad_pool_n == 0:
        raise ValueError("Bad-case pool is empty")

    # bad / (base + bad) ~= target
    target_bad_n = int(round(base_n * args.target_bad_ratio / (1 - args.target_bad_ratio)))

    out_rows = []
    source_counter = Counter()
    type_counter = Counter()

    for i, row in enumerate(base_rows):
        tagged = clone_with_tag(row, "base_v1_like", i)
        out_rows.append(tagged)
        source_counter["base_v1_like"] += 1
        type_counter[get_rejected_type(tagged)] += 1

    for i in range(target_bad_n):
        src, row = rng.choice(bad_pool)
        tagged = clone_with_tag(row, f"{src}_5pct_pool", i)
        out_rows.append(tagged)
        source_counter[f"{src}_5pct_pool"] += 1
        type_counter[get_rejected_type(tagged)] += 1

    if args.shuffle:
        rng.shuffle(out_rows)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fout:
        for row in out_rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    total = len(out_rows)
    bad_total = source_counter["bad_v2_5pct_pool"] + source_counter["bad_long_5pct_pool"]
    bad_ratio = bad_total / total if total else 0.0

    print("RM v5 data built")
    print(f"  base rows      : {base_n}")
    print(f"  bad pool rows  : {bad_pool_n} (v2={len(bad_v2_rows)}, long={len(bad_long_rows)})")
    print(f"  target bad rows: {target_bad_n}")
    print(f"  output rows    : {total}")
    print(f"  output bad ratio: {bad_ratio:.4f} ({bad_ratio*100:.2f}%)")
    print(f"  expected optimizer steps (batch_size=64): {math.ceil(total / 64)}")
    print("  source counts:")
    for k, v in source_counter.items():
        print(f"    {k}: {v}")
    print("  rejected_type counts:")
    for k, v in type_counter.most_common():
        print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
