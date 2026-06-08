#!/usr/bin/env python3
"""Convert ChatJoy pair data into OpenRLHF RM jsonl files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from common import (
    assistant_message,
    ensure_parent,
    extract_pair,
    load_jsonl,
    to_chat_messages,
    train_val_split,
)


def convert_record(record: Dict) -> Dict:
    prompt_messages = to_chat_messages(record["system"], record["prompt"])
    item = {
        "prompt": prompt_messages,
        "chosen": assistant_message(record["chosen"]),
        "rejected": assistant_message(record["rejected"]),
    }
    if record["reason"]:
        item["reason"] = record["reason"]
    if record["rejected_type"]:
        item["rejected_type"] = record["rejected_type"]
    item["extra_info"] = {
        "language": record["language"],
        "source_index": record.get("source_index"),
        "metadata": record.get("metadata", {}),
    }
    return item


def dump_jsonl(path: Path, rows: List[Dict]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare OpenRLHF RM data from ChatJoy preference pairs.")
    parser.add_argument("--input", type=str, required=True, help="Input pair jsonl.")
    parser.add_argument("--train-output", type=str, required=True, help="Output train jsonl path.")
    parser.add_argument("--val-output", type=str, required=True, help="Output validation jsonl path.")
    parser.add_argument("--val-ratio", type=float, default=0.02, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val split.")
    args = parser.parse_args()

    pairs = [extract_pair(record) for record in load_jsonl(Path(args.input))]
    pairs = [pair for pair in pairs if pair is not None]

    train_pairs, val_pairs = train_val_split(pairs, val_ratio=args.val_ratio, seed=args.seed)
    train_rows = [convert_record(pair) for pair in train_pairs]
    val_rows = [convert_record(pair) for pair in val_pairs]

    dump_jsonl(Path(args.train_output), train_rows)
    dump_jsonl(Path(args.val_output), val_rows)

    print(
        json.dumps(
            {
                "input": args.input,
                "train_output": args.train_output,
                "val_output": args.val_output,
                "train_samples": len(train_rows),
                "val_samples": len(val_rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
