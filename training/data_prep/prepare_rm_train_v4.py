#!/usr/bin/env python3
"""
Prepare RM v4 training data by filtering RM v3 data:
- remove base_length_long_low
- remove long_soft_length_mismatch_long
"""

import argparse
import json
from collections import Counter
from pathlib import Path


REMOVE_SOURCES = {"base_length_long_low", "long_soft_length_mismatch_long"}


def load_jsonl(path: Path):
    with path.open() as fin:
        for line in fin:
            line = line.strip()
            if line:
                yield json.loads(line)


def get_source(row):
    return row.get("extra_info", {}).get("v3_source", "<none>")


def get_rejected_type(row):
    return row.get("extra_info", {}).get("rejected_type", "unknown")


def main():
    parser = argparse.ArgumentParser(description="Prepare RM v4 data from RM v3")
    parser.add_argument(
        "--input",
        default="/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/openrlhf_rm/rm_train_v3.jsonl",
    )
    parser.add_argument(
        "--output",
        default="/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/openrlhf_rm/rm_train_v4.jsonl",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    kept = 0
    removed = 0
    source_counter = Counter()
    type_counter = Counter()
    removed_counter = Counter()

    with output_path.open("w") as fout:
        for row in load_jsonl(input_path):
            total += 1
            source = get_source(row)
            if source in REMOVE_SOURCES:
                removed += 1
                removed_counter[source] += 1
                continue

            kept += 1
            source_counter[source] += 1
            type_counter[get_rejected_type(row)] += 1
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("RM v4 data prepared")
    print(f"  input : {input_path}")
    print(f"  output: {output_path}")
    print(f"  total : {total}")
    print(f"  kept  : {kept}")
    print(f"  removed: {removed}")
    print("  removed by source:")
    for k in sorted(removed_counter.keys()):
        print(f"    {k}: {removed_counter[k]}")
    print("  kept by source:")
    for k, v in source_counter.most_common():
        print(f"    {k}: {v}")
    print("  kept by rejected_type:")
    for k, v in type_counter.most_common():
        ratio = (v / kept * 100.0) if kept else 0.0
        print(f"    {k}: {v} ({ratio:.1f}%)")


if __name__ == "__main__":
    main()
