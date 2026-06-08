#!/usr/bin/env python3
"""Build OpenRLHF preference pairs from ChatJoy RM v6 judged candidate sets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import LABEL_SCHEMA_VERSION, confidence_rank, iter_jsonl, write_jsonl


def assistant_message(text: str) -> list[dict[str, str]]:
    return [{"role": "assistant", "content": text}]


def pair_key(chosen: dict[str, Any], rejected: dict[str, Any]) -> tuple[str, str]:
    return str(chosen["candidate_id"]), str(rejected["candidate_id"])


def has_blocking_fail(candidate: dict[str, Any]) -> bool:
    hard_fails = candidate.get("hard_fails") or {}
    return bool(
        hard_fails.get("wrong_language")
        or hard_fails.get("system_leak")
        or hard_fails.get("user_placeholder")
        or hard_fails.get("empty_or_too_short")
        or hard_fails.get("scoring_error")
    )


def make_pair(judged: dict[str, Any], chosen: dict[str, Any], rejected: dict[str, Any], pair_type: str) -> dict[str, Any]:
    rank_delta = int(rejected["rank"]) - int(chosen["rank"])
    score_delta = float(chosen["score"]) - float(rejected["score"])
    return {
        "prompt": judged["messages"],
        "chosen": assistant_message(str(chosen["response"])),
        "rejected": assistant_message(str(rejected["response"])),
        "extra_info": {
            "label_schema_version": LABEL_SCHEMA_VERSION,
            "label_source": "fine_grained_judge",
            "pair_type": pair_type,
            "prompt_id": judged["prompt_id"],
            "language": judged.get("language"),
            "role_hash": judged.get("role_hash"),
            "judge": judged.get("judge"),
            "label_confidence": judged.get("confidence"),
            "rank_chosen": chosen["rank"],
            "rank_rejected": rejected["rank"],
            "rank_delta": rank_delta,
            "score_chosen": chosen["score"],
            "score_rejected": rejected["score"],
            "score_delta": round(score_delta, 4),
            "source_model_chosen": chosen.get("source_model"),
            "source_model_rejected": rejected.get("source_model"),
            "sampling_chosen": chosen.get("sampling"),
            "sampling_rejected": rejected.get("sampling"),
            "rubric_scores_chosen": chosen.get("rubric_scores"),
            "rubric_scores_rejected": rejected.get("rubric_scores"),
            "hard_fails_chosen": chosen.get("hard_fails"),
            "hard_fails_rejected": rejected.get("hard_fails"),
            "sample_weight": 1.0 if judged.get("confidence") == "high" else 0.8,
        },
    }


def build_pairs_for_prompt(judged: dict[str, Any], min_score_delta: float) -> list[dict[str, Any]]:
    if confidence_rank(judged.get("confidence", "low")) < confidence_rank("medium"):
        return []

    ranked = sorted(judged.get("ranked_candidates") or [], key=lambda row: int(row["rank"]))
    if len(ranked) < 4:
        return []

    pairs: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    best = ranked[0]
    worst = ranked[-1]
    middle = ranked[len(ranked) // 2]

    pairs.append(("best_vs_worst", best, worst))
    pairs.append(("best_vs_mid", best, middle))

    good = ranked[1] if len(ranked) > 3 else best
    bad = ranked[-2] if len(ranked) > 3 else worst
    pairs.append(("good_vs_bad", good, bad))

    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for pair_type, chosen, rejected in pairs:
        key = pair_key(chosen, rejected)
        if key in seen:
            continue
        seen.add(key)
        if has_blocking_fail(chosen):
            continue
        if float(chosen["score"]) - float(rejected["score"]) < min_score_delta:
            continue
        out.append(make_pair(judged, chosen, rejected, pair_type))
    return out[:3]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judged", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-score-delta", type=float, default=0.12)
    args = parser.parse_args()

    pairs: list[dict[str, Any]] = []
    skipped_low = 0
    for judged in iter_jsonl(args.judged):
        new_pairs = build_pairs_for_prompt(judged, min_score_delta=args.min_score_delta)
        if not new_pairs:
            skipped_low += 1
        pairs.extend(new_pairs)

    n = write_jsonl(args.output, pairs)
    print(json.dumps({"output": str(args.output), "pairs": n, "prompts_without_pairs": skipped_low}, indent=2))


if __name__ == "__main__":
    main()

