#!/usr/bin/env python3
"""Cap Lever 2 attack pairs via curriculum-aware stratified sampling.

Strategy:
- T1 (clean>hard, rank_delta=5, score_delta=1.0): "obvious general attack signal" — 100% kept
- T2a (clean>hard, rank_delta=3, score_delta=0.6): "subtle attack vs subtle hard-fail" — stratified by (attack_category, attack_lang)
- T2b (clean>clean, any delta): "defender quality boundary" — stratified by cell + prefer larger score_delta first

Adds `extra_info.curriculum_tier` to each kept pair so downstream training / analysis can use tier info.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def categorize(pair: dict) -> str:
    ei = pair.get("extra_info") or {}
    chf = ei.get("hard_fails_chosen") or {}
    rhf = ei.get("hard_fails_rejected") or {}
    c_hard = any(chf.values())
    r_hard = any(rhf.values())
    rank_delta = int(ei.get("rank_delta", 0))
    if (not c_hard) and r_hard and rank_delta == 5:
        return "T1"
    if (not c_hard) and r_hard and rank_delta == 3:
        return "T2a"
    if (not c_hard) and (not r_hard):
        return "T2b"
    return "OTHER"


def cell_key(pair: dict, prompts_meta: dict) -> tuple:
    ei = pair.get("extra_info") or {}
    pid = str(ei.get("prompt_id"))
    pinfo = prompts_meta.get(pid, {})
    return (pinfo.get("attack_category"), pinfo.get("attack_lang"))


def stratified_sample(pairs: list, target: int, prompts_meta: dict, rng: random.Random,
                       sort_key=None) -> list:
    """Sample `target` pairs from `pairs` while maximizing (cat, lang) cell coverage.

    If `sort_key` provided, within-cell selection prefers larger sort_key value first;
    otherwise random within cell.
    """
    if target >= len(pairs):
        return list(pairs)

    by_cell = defaultdict(list)
    for p in pairs:
        by_cell[cell_key(p, prompts_meta)].append(p)

    # Within each cell, order pairs by sort_key (or shuffle)
    for cell, items in by_cell.items():
        if sort_key is None:
            rng.shuffle(items)
        else:
            items.sort(key=lambda x: -sort_key(x))

    cells = sorted(by_cell.keys(), key=lambda c: (c[0] or "", c[1] or ""))
    selected = []
    idx_per_cell = {c: 0 for c in cells}

    # Round-robin sampling across cells until we hit target or exhaust all cells
    while len(selected) < target:
        progress = False
        for c in cells:
            if len(selected) >= target:
                break
            items = by_cell[c]
            idx = idx_per_cell[c]
            if idx < len(items):
                selected.append(items[idx])
                idx_per_cell[c] = idx + 1
                progress = True
        if not progress:
            break  # all cells exhausted
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Lever 2 merged pairs jsonl")
    parser.add_argument("--prompts", type=Path, required=True, help="subset prompts.jsonl for stratification metadata")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--target", type=int, default=2500)
    parser.add_argument("--t1-target", type=int, default=None,
                        help="T1 target (default: keep all)")
    parser.add_argument("--t2a-target", type=int, default=950)
    parser.add_argument("--t2b-target", type=int, default=425)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    rows = list(load_jsonl(args.input))
    prompts_meta = {str(p.get("prompt_id")): p for p in load_jsonl(args.prompts)}

    buckets = defaultdict(list)
    for r in rows:
        buckets[categorize(r)].append(r)

    t1_pool = buckets["T1"]
    t2a_pool = buckets["T2a"]
    t2b_pool = buckets["T2b"]
    other_pool = buckets.get("OTHER", [])

    # T1: keep all (or capped)
    t1_target = args.t1_target if args.t1_target is not None else len(t1_pool)
    t1_target = min(t1_target, len(t1_pool))
    if t1_target < len(t1_pool):
        t1_kept = stratified_sample(t1_pool, t1_target, prompts_meta, rng)
    else:
        t1_kept = list(t1_pool)

    # T2a: stratified random
    t2a_target = min(args.t2a_target, len(t2a_pool))
    t2a_kept = stratified_sample(t2a_pool, t2a_target, prompts_meta, rng)

    # T2b: stratified prefer larger score_delta
    t2b_target = min(args.t2b_target, len(t2b_pool))
    t2b_kept = stratified_sample(
        t2b_pool, t2b_target, prompts_meta, rng,
        sort_key=lambda r: float(r["extra_info"].get("score_delta", 0)),
    )

    # Tag tier
    for p in t1_kept:
        p["extra_info"] = dict(p.get("extra_info") or {})
        p["extra_info"]["curriculum_tier"] = "T1"
    for p in t2a_kept:
        p["extra_info"] = dict(p.get("extra_info") or {})
        p["extra_info"]["curriculum_tier"] = "T2a"
    for p in t2b_kept:
        p["extra_info"] = dict(p.get("extra_info") or {})
        p["extra_info"]["curriculum_tier"] = "T2b"

    final = t1_kept + t2a_kept + t2b_kept
    write_jsonl(args.output, final)

    # Coverage stats
    def coverage_stats(pairs):
        cells = defaultdict(int)
        for p in pairs:
            cells[cell_key(p, prompts_meta)] += 1
        return {
            "n_pairs": len(pairs),
            "unique_cells": len(cells),
            "min_per_cell": min(cells.values()) if cells else 0,
            "max_per_cell": max(cells.values()) if cells else 0,
            "mean_per_cell": round(sum(cells.values()) / len(cells), 2) if cells else 0,
        }

    report = {
        "input": str(args.input),
        "output": str(args.output),
        "target_total": args.target,
        "actual_total": len(final),
        "input_total": len(rows),
        "buckets_input": {
            "T1": len(t1_pool),
            "T2a": len(t2a_pool),
            "T2b": len(t2b_pool),
            "OTHER": len(other_pool),
        },
        "buckets_kept": {
            "T1": coverage_stats(t1_kept),
            "T2a": coverage_stats(t2a_kept),
            "T2b": coverage_stats(t2b_kept),
        },
        "final_coverage": coverage_stats(final),
        "dropped": len(rows) - len(final),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
