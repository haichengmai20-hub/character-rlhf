#!/usr/bin/env python3
"""Re-apply the in-script postprocess (add_inferred_hard_fail_flags +
hard-fail demote sort + rank/score reassignment) to an already-judged jsonl.

Useful when ROLEDROP_EXECUTION_RE / REFUSAL_RE or related postprocess code
is updated and you want to refresh existing judge outputs without paying
for a re-judge.

Idempotent: re-running on an already-postprocessed file is a no-op unless
new patterns match additional candidates.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from judge_candidates_llm_openai import (  # type: ignore
    BASE_HARD_FAIL_FLAGS,
    ROLEDROP_HARD_FAIL_FLAG,
    add_inferred_hard_fail_flags,
    blocking_flags,
    has_any_blocking_flag,
    rank_score,
)


def reprocess_candidate(candidate: dict, roledrop_mode: bool) -> None:
    current_flags = list(candidate.get("judge_flags") or [])
    if not current_flags:
        existing_hf = candidate.get("hard_fails") or {}
        current_flags = [k for k, v in existing_hf.items() if v]
        if not current_flags:
            current_flags = ["none"]
    new_flags = add_inferred_hard_fail_flags(current_flags, candidate, roledrop_mode)
    candidate["judge_flags"] = new_flags
    candidate["hard_fails"] = blocking_flags(new_flags, include_roledrop=roledrop_mode)


def reprocess_row(row: dict, roledrop_mode: bool) -> tuple[bool, int]:
    ranked = list(row.get("ranked_candidates") or [])
    if not ranked:
        return False, 0
    before = [(c.get("rank"), dict(c.get("hard_fails") or {})) for c in ranked]
    for c in ranked:
        if "raw_rank" not in c:
            c["raw_rank"] = c.get("rank")
        reprocess_candidate(c, roledrop_mode)
    ranked.sort(key=lambda c: (has_any_blocking_flag(c["hard_fails"]), c.get("raw_rank", c.get("rank"))))
    total = len(ranked)
    for new_rank, c in enumerate(ranked, start=1):
        c["rank"] = new_rank
        c["score"] = rank_score(new_rank, total)
    row["ranked_candidates"] = ranked

    old_by_cid = {c.get("candidate_id"): (old_rank, old_hf) for (old_rank, old_hf), c in zip(before, ranked)}
    changed = False
    new_hard_count = 0
    for c in ranked:
        old_rank, old_hf = old_by_cid.get(c.get("candidate_id"), (None, {}))
        if c.get("rank") != old_rank:
            changed = True
        new_hf = c.get("hard_fails") or {}
        if dict(new_hf) != dict(old_hf or {}):
            changed = True
        for flag, val in new_hf.items():
            if val and not (old_hf or {}).get(flag, False):
                new_hard_count += 1
    return changed, new_hard_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--roledrop-mode", action="store_true", default=False)
    args = parser.parse_args()

    rows = []
    changed_rows = 0
    total_new_flags = 0
    with args.input.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            changed, new_count = reprocess_row(row, args.roledrop_mode)
            if changed:
                changed_rows += 1
            total_new_flags += new_count
            rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "rows": len(rows),
        "rows_changed": changed_rows,
        "new_hard_fail_flags_added": total_new_flags,
        "roledrop_mode": args.roledrop_mode,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
