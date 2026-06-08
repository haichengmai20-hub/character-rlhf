#!/usr/bin/env python3
"""Filter RM v6 fine-grained pairs to produce v6.1 cleaner dataset.

Filtering rule (dual-filter):
  Keep: rank_delta == 4 (best_vs_worst) — all confidence levels
  Keep: rank_delta == 2 AND label_confidence == "high"
  Drop: rank_delta == 2 AND label_confidence == "medium"  (noisy close pairs)

Rationale: delta=2 medium-confidence pairs are single-judge outputs at temperature=0,
meaning the judge itself was uncertain and there is no way to verify the direction
without a second independent judge.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from common import OPENRLHF_RM_DIR, iter_jsonl, write_jsonl


def should_keep(row: dict) -> bool:
    ei = row.get("extra_info") or {}
    delta = ei.get("rank_delta")
    conf = ei.get("label_confidence", "")
    if delta == 4:
        return True
    if delta == 2 and conf == "high":
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=OPENRLHF_RM_DIR / "rm_v6_finegrained_pairs_combined.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OPENRLHF_RM_DIR / "rm_v6.1_finegrained_pairs_combined.jsonl",
    )
    args = parser.parse_args()

    kept: list[dict] = []
    dropped = Counter()
    total = 0

    for row in iter_jsonl(args.input):
        total += 1
        ei = row.get("extra_info") or {}
        delta = ei.get("rank_delta", "?")
        conf = ei.get("label_confidence", "?")
        if should_keep(row):
            kept.append(row)
        else:
            dropped[(delta, conf)] += 1

    n = write_jsonl(args.output, kept)

    kept_by = Counter()
    for row in kept:
        ei = row.get("extra_info") or {}
        kept_by[(ei.get("rank_delta"), ei.get("label_confidence"))] += 1

    result = {
        "input": str(args.input),
        "output": str(args.output),
        "total_input": total,
        "total_kept": n,
        "total_dropped": total - n,
        "drop_rate": f"{(total - n) / total:.1%}",
        "kept_breakdown": {f"delta={d},conf={c}": v for (d, c), v in sorted(kept_by.items())},
        "dropped_breakdown": {f"delta={d},conf={c}": v for (d, c), v in sorted(dropped.items())},
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
