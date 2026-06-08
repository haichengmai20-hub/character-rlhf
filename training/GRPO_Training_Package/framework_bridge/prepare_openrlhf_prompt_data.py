#!/usr/bin/env python3
"""Convert ChatJoy prompts into OpenRLHF online RL prompt jsonl files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from common import (
    dedupe_prompt_records,
    ensure_parent,
    extract_prompt_record,
    load_jsonl,
    to_chat_messages,
    train_val_split,
)


def convert_record(record: Dict) -> Dict:
    return {
        "context_messages": to_chat_messages(record["system"], record["prompt"]),
        "extra_info": {
            "language": record["language"],
            "source_index": record.get("source_index"),
            "record_type": record.get("record_type"),
        },
    }


def dump_jsonl(path: Path, rows: List[Dict]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare OpenRLHF actor prompt data from ChatJoy records.")
    parser.add_argument("--input", type=str, required=True, help="Input jsonl (SFT or pair).")
    parser.add_argument("--train-output", type=str, required=True, help="Output train jsonl path.")
    parser.add_argument("--val-output", type=str, required=True, help="Output validation jsonl path.")
    parser.add_argument("--val-ratio", type=float, default=0.02, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split.")
    args = parser.parse_args()

    prompts = [extract_prompt_record(record) for record in load_jsonl(Path(args.input))]
    prompts = [record for record in prompts if record is not None]
    prompts = dedupe_prompt_records(prompts)

    train_rows, val_rows = train_val_split(prompts, val_ratio=args.val_ratio, seed=args.seed)
    train_rows = [convert_record(record) for record in train_rows]
    val_rows = [convert_record(record) for record in val_rows]

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
