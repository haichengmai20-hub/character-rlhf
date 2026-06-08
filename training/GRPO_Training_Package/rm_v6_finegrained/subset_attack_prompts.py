#!/usr/bin/env python3
"""Build a stratified single-turn roledrop attack prompt subset."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common import iter_jsonl, stable_hash, write_jsonl


ROOT = Path("/root/hanjiaqi/LlamaFactory/chatjoy")
DEFAULT_INPUT = ROOT / "data/runs/rm_v6_3_roledrop/roledrop_attack_single_turn_prompts_v1.jsonl"
DEFAULT_OUTPUT = ROOT / "data/runs/rm_v6_3_roledrop/roledrop_attack_single_turn_prompts_subset_1500.jsonl"


def group_key(row: dict[str, Any]) -> tuple[str, str]:
    character = str(row.get("character_id") or row.get("character_index") or "")
    category = str(row.get("attack_category") or "")
    if not character or not category:
        raise ValueError(f"row missing character/category fields: {row.get('prompt_id')}")
    return character, category


def split_name(row: dict[str, Any]) -> str:
    return str(row.get("attack_train_or_eval") or "train")


def distribute(total: int, keys: list[Any], seed: int) -> dict[Any, int]:
    if total < 0:
        raise ValueError("total must be non-negative")
    if not keys and total:
        raise ValueError("cannot distribute a positive total across no keys")
    base = total // len(keys) if keys else 0
    remainder = total % len(keys) if keys else 0
    counts = {key: base for key in keys}
    order = sorted(keys, key=lambda key: stable_hash(f"{seed}:{key}", size=24))
    for key in order[:remainder]:
        counts[key] += 1
    return counts


def allocate_eval_counts(
    rows: list[dict[str, Any]],
    selected_counts: dict[tuple[str, str], int],
    seed: int,
) -> dict[tuple[str, str], int]:
    source_split_counts = Counter(split_name(row) for row in rows)
    target_eval = round(sum(selected_counts.values()) * source_split_counts.get("eval_holdout", 0) / len(rows))
    eval_counts = {key: 0 for key in selected_counts}
    eval_capacity = Counter()
    for row in rows:
        if split_name(row) == "eval_holdout":
            eval_capacity[group_key(row)] += 1

    for key, selected_count in selected_counts.items():
        if eval_capacity[key]:
            eval_counts[key] = min(int(selected_count * source_split_counts.get("eval_holdout", 0) / len(rows)), eval_capacity[key], selected_count)

    remaining = target_eval - sum(eval_counts.values())
    eligible = [
        key
        for key, selected_count in selected_counts.items()
        if eval_counts[key] < selected_count and eval_counts[key] < eval_capacity[key]
    ]
    order = sorted(eligible, key=lambda key: stable_hash(f"eval:{seed}:{key}", size=24))
    idx = 0
    while remaining > 0 and order:
        key = order[idx % len(order)]
        if eval_counts[key] < selected_counts[key] and eval_counts[key] < eval_capacity[key]:
            eval_counts[key] += 1
            remaining -= 1
        idx += 1
        if idx > len(order) * max(1, target_eval + 1):
            break
    return eval_counts


def sample_rows(rows: list[dict[str, Any]], count: int, seed: int, salt: str) -> list[dict[str, Any]]:
    if count > len(rows):
        raise ValueError(f"requested {count} rows from a group with only {len(rows)} rows")
    shuffled = list(rows)
    random.Random(f"{seed}:{salt}").shuffle(shuffled)
    return shuffled[:count]


def select_subset(rows: list[dict[str, Any]], target_size: int, seed: int = 63) -> list[dict[str, Any]]:
    if target_size > len(rows):
        raise ValueError(f"target_size {target_size} is larger than available rows {len(rows)}")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[group_key(row)].append(row)
    keys = sorted(grouped)
    if target_size < len(keys):
        raise ValueError(f"target_size {target_size} cannot cover {len(keys)} character/category groups")

    selected_counts = distribute(target_size, keys, seed)
    for key, count in selected_counts.items():
        if count > len(grouped[key]):
            raise ValueError(f"group {key} needs {count} rows but only has {len(grouped[key])}")

    eval_counts = allocate_eval_counts(rows, selected_counts, seed)
    selected_ids: set[str] = set()
    selected: list[dict[str, Any]] = []

    for key in keys:
        group = grouped[key]
        eval_rows = [row for row in group if split_name(row) == "eval_holdout"]
        train_rows = [row for row in group if split_name(row) != "eval_holdout"]
        eval_needed = eval_counts[key]
        train_needed = selected_counts[key] - eval_needed
        picked = sample_rows(eval_rows, eval_needed, seed, f"{key}:eval")
        picked += sample_rows(train_rows, train_needed, seed, f"{key}:train")
        for row in picked:
            selected_ids.add(str(row["prompt_id"]))
        selected.extend(picked)

    order = {str(row["prompt_id"]): idx for idx, row in enumerate(rows)}
    selected.sort(key=lambda row: order[str(row["prompt_id"])])
    if len(selected_ids) != len(selected):
        raise ValueError("selected subset contains duplicate prompt_id values")
    return selected


def summarize_subset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    per_group = Counter(group_key(row) for row in rows)
    splits = Counter(split_name(row) for row in rows)
    categories = Counter(str(row.get("attack_category") or "") for row in rows)
    characters = {str(row.get("character_id") or row.get("character_index") or "") for row in rows}
    group_counts = list(per_group.values())
    return {
        "prompts": len(rows),
        "characters": len(characters),
        "categories": dict(sorted(categories.items())),
        "splits": dict(sorted(splits.items())),
        "unique_character_category": len(per_group),
        "per_character_category_min": min(group_counts) if group_counts else 0,
        "per_character_category_max": max(group_counts) if group_counts else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-size", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=63)
    args = parser.parse_args()

    rows = list(iter_jsonl(args.input))
    selected = select_subset(rows, target_size=args.target_size, seed=args.seed)
    written = write_jsonl(args.output, selected)
    summary = summarize_subset(selected)
    summary.update({"input": str(args.input), "output": str(args.output), "written": written, "seed": args.seed})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
