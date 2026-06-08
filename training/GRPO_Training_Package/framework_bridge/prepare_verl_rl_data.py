#!/usr/bin/env python3
"""Convert ChatJoy prompts into verl RL parquet files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from datasets import Dataset

from common import dedupe_prompt_records, extract_prompt_record, load_jsonl, to_chat_messages, train_val_split


def convert_record(record: Dict) -> Dict:
    source_index = record.get("source_index")
    return {
        "data_source": "chatjoy",
        "prompt": to_chat_messages(record["system"], record["prompt"]),
        "ability": "chatjoy_alignment",
        "reward_model": {
            "style": "model",
            "ground_truth": "",
        },
        "extra_info": {
            "index": source_index,
            "source_index": source_index,
            "language": record["language"],
            "record_type": record.get("record_type"),
        },
    }


def dump_parquet(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).to_parquet(str(path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare verl RL parquet data from ChatJoy SFT or pair records.")
    parser.add_argument("--input", type=str, required=True, help="Input jsonl (SFT or pair).")
    parser.add_argument("--train-output", type=str, required=True, help="Output train parquet path.")
    parser.add_argument("--val-output", type=str, required=True, help="Output validation parquet path.")
    parser.add_argument("--val-ratio", type=float, default=0.02, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    prompts = [extract_prompt_record(record) for record in load_jsonl(Path(args.input))]
    prompts = [record for record in prompts if record is not None]
    prompts = dedupe_prompt_records(prompts)
    train_prompts, val_prompts = train_val_split(prompts, val_ratio=args.val_ratio, seed=args.seed)

    train_rows = [convert_record(record) for record in train_prompts]
    val_rows = [convert_record(record) for record in val_prompts]

    dump_parquet(Path(args.train_output), train_rows)
    dump_parquet(Path(args.val_output), val_rows)

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
