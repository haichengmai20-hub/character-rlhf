#!/usr/bin/env python3
"""Sample stratified RL prompts for ChatJoy RM v6 fine-grained candidate generation."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from common import RL_TRAIN_PARQUET, RL_VAL_PARQUET, extract_system_user, normalize_messages, prompt_key, role_hash_from_messages, write_jsonl


def load_parquet_rows(path: Path) -> list[dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("pandas/pyarrow are required. Use the gemma4-rl env for this script.") from exc

    df = pd.read_parquet(path)
    return df.to_dict(orient="records")


def load_excluded_keys(paths: list[Path]) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        for row in load_parquet_rows(path):
            messages = normalize_messages(row.get("prompt"))
            if messages:
                excluded.add(prompt_key(messages))
    return excluded


def convert_row(row: dict[str, Any], source_index: int) -> dict[str, Any] | None:
    messages = normalize_messages(row.get("prompt"))
    if not messages:
        return None
    system, user = extract_system_user(messages)
    if not system or not user:
        return None

    extra = row.get("extra_info") or {}
    language = str(extra.get("language") or "unknown").lower()
    key = prompt_key(messages)
    return {
        "prompt_id": key,
        "messages": messages,
        "language": language,
        "role_hash": role_hash_from_messages(messages),
        "source_index": extra.get("source_index", extra.get("index", source_index)),
        "data_source": row.get("data_source", "chatjoy"),
        "ability": row.get("ability", "chatjoy_alignment"),
        "system_chars": len(system),
        "user_chars": len(user),
    }


def stratified_round_robin(rows: list[dict[str, Any]], sample_size: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["language"], row["role_hash"])].append(row)

    buckets: list[tuple[tuple[str, str], list[dict[str, Any]]]] = []
    for key, items in groups.items():
        rng.shuffle(items)
        buckets.append((key, items))
    buckets.sort(key=lambda item: (-len(item[1]), item[0][0], item[0][1]))

    selected: list[dict[str, Any]] = []
    while len(selected) < sample_size:
        made_progress = False
        for _, items in buckets:
            if not items:
                continue
            selected.append(items.pop())
            made_progress = True
            if len(selected) >= sample_size:
                break
        if not made_progress:
            break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-parquet", type=Path, default=RL_TRAIN_PARQUET)
    parser.add_argument("--exclude-parquet", type=Path, action="append", default=[RL_VAL_PARQUET])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sample_size = args.sample_size or (2500 if args.mode == "pilot" else 10000)
    excluded = load_excluded_keys(args.exclude_parquet)

    seen: set[str] = set()
    candidates: list[dict[str, Any]] = []
    for idx, row in enumerate(load_parquet_rows(args.train_parquet)):
        converted = convert_row(row, idx)
        if converted is None:
            continue
        key = converted["prompt_id"]
        if key in seen or key in excluded:
            continue
        seen.add(key)
        candidates.append(converted)

    if len(candidates) < sample_size:
        raise SystemExit(f"Only {len(candidates)} unique non-excluded prompts available, requested {sample_size}.")

    selected = stratified_round_robin(candidates, sample_size=sample_size, seed=args.seed)
    n = write_jsonl(args.output, selected)
    summary = {
        "train_parquet": str(args.train_parquet),
        "output": str(args.output),
        "mode": args.mode,
        "requested": sample_size,
        "written": n,
        "excluded_prompts": len(excluded),
        "available_unique": len(candidates),
        "languages": len({row["language"] for row in selected}),
        "roles": len({row["role_hash"] for row in selected}),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

