#!/usr/bin/env python3
"""Mix full RM v5 anchor data with RM v6 fine-grained pairs."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from common import LABEL_SCHEMA_VERSION, V5_EVAL, V5_TRAIN, iter_jsonl, shuffled, write_jsonl


def mark_v5(row: dict[str, Any], split: str) -> dict[str, Any]:
    row = dict(row)
    extra = dict(row.get("extra_info") or {})
    extra.setdefault("label_schema_version", LABEL_SCHEMA_VERSION)
    extra["label_source"] = f"v5_{split}_anchor"
    extra.setdefault("sample_weight", 1.0)
    row["extra_info"] = extra
    return row


def load_fine_pairs(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in iter_jsonl(path):
        extra = dict(row.get("extra_info") or {})
        extra.setdefault("label_schema_version", LABEL_SCHEMA_VERSION)
        extra.setdefault("label_source", "fine_grained_judge")
        row["extra_info"] = extra
        rows.append(row)
    return rows


def split_fine(rows: list[dict[str, Any]], eval_ratio: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = shuffled(rows, seed)
    eval_n = max(1, int(round(len(rows) * eval_ratio))) if rows else 0
    return rows[eval_n:], rows[:eval_n]


def source_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        source = str((row.get("extra_info") or {}).get("label_source", "unknown"))
        counts[source] = counts.get(source, 0) + 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fine-pairs", type=Path, required=True)
    parser.add_argument("--output-train", type=Path, required=True)
    parser.add_argument("--output-eval", type=Path, required=True)
    parser.add_argument("--v5-train", type=Path, default=V5_TRAIN)
    parser.add_argument("--v5-eval", type=Path, default=V5_EVAL)
    parser.add_argument("--fine-eval", type=Path, default=None)
    parser.add_argument("--fine-eval-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    v5_train = [mark_v5(row, "train") for row in iter_jsonl(args.v5_train)]
    v5_eval = [mark_v5(row, "eval") for row in iter_jsonl(args.v5_eval)]
    fine = load_fine_pairs(args.fine_pairs)

    if args.fine_eval:
        fine_train = fine
        fine_eval = load_fine_pairs(args.fine_eval)
    else:
        fine_train, fine_eval = split_fine(fine, eval_ratio=args.fine_eval_ratio, seed=args.seed)

    train_rows = shuffled(v5_train + fine_train, args.seed)
    eval_rows = shuffled(v5_eval + fine_eval, args.seed + 17)

    train_n = write_jsonl(args.output_train, train_rows)
    eval_n = write_jsonl(args.output_eval, eval_rows)
    print(
        json.dumps(
            {
                "train_output": str(args.output_train),
                "eval_output": str(args.output_eval),
                "train_samples": train_n,
                "eval_samples": eval_n,
                "train_sources": source_counts(train_rows),
                "eval_sources": source_counts(eval_rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

