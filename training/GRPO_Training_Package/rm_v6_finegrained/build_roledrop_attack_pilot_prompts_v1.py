#!/usr/bin/env python3
"""Build a deterministic, balanced pilot subset for roledrop attack prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("/root/hanjiaqi/LlamaFactory/chatjoy")
DEFAULT_PROMPTS = ROOT / "data/runs/rm_v6_3_roledrop/roledrop_attack_single_turn_prompts_v1.jsonl"
DEFAULT_OUTPUT = ROOT / "data/runs/rm_v6_3_roledrop/roledrop_attack_single_turn_prompts_pilot100_v1.jsonl"


def stable_key(seed: int, row: dict[str, Any]) -> str:
    payload = json.dumps(
        {
            "seed": seed,
            "prompt_id": row["prompt_id"],
            "attack_seed_id": row.get("attack_seed_id"),
            "character_id": row.get("character_id"),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_pilot(rows: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    if limit <= 0:
        raise ValueError("--limit must be positive")
    if limit > len(rows):
        raise ValueError(f"--limit {limit} exceeds available rows {len(rows)}")

    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(str(row["attack_category"]), str(row["attack_lang"]))].append(row)

    for bucket_rows in buckets.values():
        bucket_rows.sort(key=lambda row: stable_key(seed, row))

    selected: list[dict[str, Any]] = []
    bucket_keys = sorted(buckets)
    cursor = 0
    while len(selected) < limit:
        made_progress = False
        for bucket_key in bucket_keys:
            bucket_rows = buckets[bucket_key]
            if cursor < len(bucket_rows):
                selected.append(bucket_rows[cursor])
                made_progress = True
                if len(selected) == limit:
                    break
        if not made_progress:
            break
        cursor += 1

    if len(selected) != limit:
        raise RuntimeError(f"selected {len(selected)} rows, expected {limit}")

    selected.sort(key=lambda row: stable_key(seed + 1, row))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=63)
    args = parser.parse_args()

    rows = iter_jsonl(args.prompts)
    pilot_rows = build_pilot(rows, args.limit, args.seed)
    write_jsonl(args.output, pilot_rows)

    print(
        json.dumps(
            {
                "input": str(args.prompts),
                "output": str(args.output),
                "records": len(pilot_rows),
                "seed": args.seed,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
