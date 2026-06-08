#!/usr/bin/env python3
"""QA checks for ChatJoy RM v6 fine-grained data artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from common import iter_jsonl


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def check_candidates(path: Path) -> dict:
    rows = list(iter_jsonl(path))
    by_prompt = defaultdict(list)
    for row in rows:
        by_prompt[row["prompt_id"]].append(row)
    source_counts = Counter(row.get("source_model") for row in rows)
    valid_per_prompt = [sum(1 for row in items if row.get("response")) for items in by_prompt.values()]
    enough = sum(1 for n in valid_per_prompt if n >= 4)
    return {
        "path": str(path),
        "rows": len(rows),
        "prompts": len(by_prompt),
        "prompts_with_4plus": enough,
        "prompts_with_4plus_ratio": enough / max(1, len(by_prompt)),
        "source_counts": dict(source_counts),
    }


def check_judged(path: Path) -> dict:
    rows = list(iter_jsonl(path))
    conf = Counter(row.get("confidence", "unknown") for row in rows)
    low = conf.get("low", 0)
    blocked = 0
    for row in rows:
        for candidate in row.get("ranked_candidates", []):
            hard = candidate.get("hard_fails") or {}
            if any(hard.values()):
                blocked += 1
    return {
        "path": str(path),
        "judged_prompts": len(rows),
        "confidence": dict(conf),
        "low_confidence_ratio": low / max(1, len(rows)),
        "blocked_candidate_flags": blocked,
    }


def check_pairs(path: Path) -> dict:
    rows = list(iter_jsonl(path))
    same_source = 0
    pair_types = Counter()
    label_sources = Counter()
    for row in rows:
        extra = row.get("extra_info") or {}
        pair_types[extra.get("pair_type", "unknown")] += 1
        label_sources[extra.get("label_source", "unknown")] += 1
        if extra.get("source_model_chosen") and extra.get("source_model_chosen") == extra.get("source_model_rejected"):
            same_source += 1
    return {
        "path": str(path),
        "pairs": len(rows),
        "same_source_ratio": same_source / max(1, len(rows)),
        "pair_types": dict(pair_types),
        "label_sources": dict(label_sources),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=None)
    parser.add_argument("--judged", type=Path, default=None)
    parser.add_argument("--pairs", type=Path, default=None)
    parser.add_argument("--mixed-train", type=Path, default=None)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    report = {}
    failures = []
    if args.candidates:
        report["candidates"] = check_candidates(args.candidates)
        if report["candidates"]["prompts_with_4plus_ratio"] < 0.95:
            failures.append("candidate prompts with >=4 valid candidates below 95%")
    if args.judged:
        report["judged"] = check_judged(args.judged)
        if report["judged"]["low_confidence_ratio"] > 0.20:
            failures.append("low-confidence judged prompt ratio above 20%")
    if args.pairs:
        report["pairs"] = check_pairs(args.pairs)
        if report["pairs"]["same_source_ratio"] > 0.35:
            failures.append("fine pair same-source ratio above 35%")
    if args.mixed_train:
        report["mixed_train"] = check_pairs(args.mixed_train)

    report["failures"] = failures
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures and args.strict:
        raise SystemExit("; ".join(failures))


if __name__ == "__main__":
    main()

