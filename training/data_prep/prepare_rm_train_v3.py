#!/usr/bin/env python3
"""
Build RM v3 training data with:
1. rm_train.jsonl as the base corpus
2. no hard deletion for low-confidence length_mismatch_long
3. optional global hard-negative pool
4. optional long-only soft bad-case pool

Weighting strategy is implemented via explicit repetition because the current
OpenRLHF RewardDataset does not consume sample_weight from jsonl.

Default policy:
- Base rm_train.jsonl:
  - keep every sample at least 1x
  - length_mismatch_long with reward_gap >= 0.2: repeat 2x
  - length_mismatch_long with reward_gap < 0.2: keep 1x only
- Global hard-negative pool:
  - wrong_language: 4x
  - length_mismatch_short: 3x
  - wrong_role: 2x
  - length_mismatch_long: 2x
- Long-only soft pool:
  - gap <= threshold mined from rm_train.jsonl
  - repeat 3x by default
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.open() if line.strip()]


def get_rejected_type(record):
    return record.get("extra_info", {}).get("rejected_type", "unknown")


def get_reward_gap(record):
    return float(record.get("extra_info", {}).get("reward_gap", 0.0))


def clone_record(record, source_tag, repeat_idx):
    cloned = json.loads(json.dumps(record))
    extra = cloned.setdefault("extra_info", {})
    extra["v3_source"] = source_tag
    extra["v3_repeat_idx"] = repeat_idx
    return cloned


def main():
    parser = argparse.ArgumentParser(description="Prepare ChatJoy RM v3 training data")
    parser.add_argument(
        "--base",
        default="/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/openrlhf_rm/rm_train.jsonl",
    )
    parser.add_argument(
        "--global-hard",
        default="/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/openrlhf_rm/hard_negatives_v2.jsonl",
    )
    parser.add_argument(
        "--long-soft",
        default="/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/openrlhf_rm/hard_negatives_long_soft.jsonl",
    )
    parser.add_argument(
        "--output",
        default="/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/openrlhf_rm/rm_train_v3.jsonl",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strong-long-repeat", type=int, default=2)
    parser.add_argument("--low-long-repeat", type=int, default=1)
    parser.add_argument("--global-wrong-language-repeat", type=int, default=4)
    parser.add_argument("--global-wrong-role-repeat", type=int, default=2)
    parser.add_argument("--global-length-short-repeat", type=int, default=3)
    parser.add_argument("--global-length-long-repeat", type=int, default=2)
    parser.add_argument("--long-soft-repeat", type=int, default=3)
    args = parser.parse_args()

    random.seed(args.seed)

    base_path = Path(args.base)
    global_hard_path = Path(args.global_hard)
    long_soft_path = Path(args.long_soft)
    output_path = Path(args.output)

    base_rows = load_jsonl(base_path)
    print(f"Loaded base: {base_path} ({len(base_rows)} rows)")

    global_hard_rows = load_jsonl(global_hard_path) if global_hard_path.exists() else []
    long_soft_rows = load_jsonl(long_soft_path) if long_soft_path.exists() else []
    print(f"Loaded global hard pool: {global_hard_path} ({len(global_hard_rows)} rows)")
    print(f"Loaded long-only soft pool: {long_soft_path} ({len(long_soft_rows)} rows)")

    output_rows = []
    source_counter = Counter()
    type_counter = Counter()

    for row in base_rows:
        rejected_type = get_rejected_type(row)
        reward_gap = get_reward_gap(row)
        if rejected_type == "length_mismatch_long":
            repeat = args.strong_long_repeat if reward_gap >= 0.2 else args.low_long_repeat
            source_tag = "base_length_long_strong" if reward_gap >= 0.2 else "base_length_long_low"
        else:
            repeat = 1
            source_tag = "base"

        for i in range(repeat):
            output_rows.append(clone_record(row, source_tag, i))
            source_counter[source_tag] += 1
            type_counter[rejected_type] += 1

    global_repeat_map = {
        "wrong_language": args.global_wrong_language_repeat,
        "wrong_role": args.global_wrong_role_repeat,
        "length_mismatch_short": args.global_length_short_repeat,
        "length_mismatch_long": args.global_length_long_repeat,
    }
    for row in global_hard_rows:
        rejected_type = get_rejected_type(row)
        repeat = global_repeat_map.get(rejected_type, 1)
        source_tag = f"global_hard_{rejected_type}"
        for i in range(repeat):
            output_rows.append(clone_record(row, source_tag, i))
            source_counter[source_tag] += 1
            type_counter[rejected_type] += 1

    for row in long_soft_rows:
        rejected_type = get_rejected_type(row)
        source_tag = f"long_soft_{rejected_type}"
        for i in range(args.long_soft_repeat):
            output_rows.append(clone_record(row, source_tag, i))
            source_counter[source_tag] += 1
            type_counter[rejected_type] += 1

    random.shuffle(output_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fout:
        for row in output_rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\nOutput summary")
    print(f"  output: {output_path}")
    print(f"  total rows: {len(output_rows)}")
    print("  by source:")
    for key, value in source_counter.most_common():
        print(f"    {key:<32} {value:>6}")
    print("  by rejected_type:")
    for key, value in type_counter.most_common():
        print(f"    {key:<32} {value:>6}  ({value / len(output_rows) * 100:.1f}%)")


if __name__ == "__main__":
    main()
