#!/usr/bin/env python3
"""Cross-validate pairs from two independent judges.

For each (prompt_id, chosen_model, rejected_model) pair derived from judge_a:
  - Look up judge_b's ranking for the same prompt
  - If judge_b also ranks chosen_model higher than rejected_model → KEEP
  - Otherwise → DISCARD

Output: filtered pairs where both judges agree on direction.
This is used to build RM v6.2 training data with cleaner labels.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from common import OPENRLHF_RM_DIR, iter_jsonl, write_jsonl


BLOCKING_HARD_FAIL_KEYS = {
    "wrong_language",
    "refusal",
    "acts_for_user",
    "system_leak",
    "empty_or_garbled",
    "empty_or_too_short",
    "user_placeholder",
    "scoring_error",
    "unsafe_escalation",
    "roledrop_execution",
}


def assistant_message(text: str) -> list[dict[str, str]]:
    return [{"role": "assistant", "content": text}]


def has_blocking_fail(candidate: dict[str, Any]) -> bool:
    hard_fails = candidate.get("hard_fails") or {}
    return any(bool(hard_fails.get(flag)) for flag in BLOCKING_HARD_FAIL_KEYS)


def pair_key(chosen: dict[str, Any], rejected: dict[str, Any]) -> tuple[str, str]:
    return str(chosen["candidate_id"]), str(rejected["candidate_id"])


def make_pair_from_judged(
    judged: dict[str, Any],
    chosen: dict[str, Any],
    rejected: dict[str, Any],
    pair_type: str,
    rubric_version: str,
) -> dict[str, Any]:
    rank_delta = int(rejected["rank"]) - int(chosen["rank"])
    score_delta = float(chosen["score"]) - float(rejected["score"])
    extra_info = {
        "label_schema_version": "chatjoy_rm_v6_finegrained_v1",
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
        "sample_weight": 1.0 if judged.get("confidence") == "high" else 0.8,
        "rubric_version": rubric_version,
        "candidate_count": judged.get("candidate_count", len(judged.get("ranked_candidates") or [])),
    }

    if "rubric_scores" in chosen:
        extra_info["rubric_scores_chosen"] = chosen["rubric_scores"]
    if "rubric_scores" in rejected:
        extra_info["rubric_scores_rejected"] = rejected["rubric_scores"]
    if "density_score" in chosen:
        extra_info["density_score_chosen"] = chosen["density_score"]
    if "density_score" in rejected:
        extra_info["density_score_rejected"] = rejected["density_score"]
    if "hard_fails" in chosen:
        extra_info["hard_fails_chosen"] = chosen["hard_fails"]
    if "hard_fails" in rejected:
        extra_info["hard_fails_rejected"] = rejected["hard_fails"]
    if "roledrop_mode" in judged:
        extra_info["roledrop_mode"] = judged["roledrop_mode"]

    return {
        "prompt": judged["messages"],
        "chosen": assistant_message(str(chosen["response"])),
        "rejected": assistant_message(str(rejected["response"])),
        "extra_info": extra_info,
    }


def build_pairs_for_prompt(
    judged: dict[str, Any],
    min_score_delta: float,
    pair_type_tag: str | None,
    rubric_version: str,
) -> list[dict[str, Any]]:
    ranked = sorted(judged.get("ranked_candidates") or [], key=lambda row: int(row["rank"]))
    if len(ranked) < 4:
        return []

    pair_type_prefix = pair_type_tag or "fine_grained"
    candidates: list[tuple[str, dict[str, Any], dict[str, Any]]] = [
        (f"{pair_type_prefix}:best_vs_worst", ranked[0], ranked[-1]),
        (f"{pair_type_prefix}:best_vs_mid", ranked[0], ranked[len(ranked) // 2]),
        (f"{pair_type_prefix}:good_vs_bad", ranked[1], ranked[-2]),
    ]

    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for pair_type, chosen, rejected in candidates:
        key = pair_key(chosen, rejected)
        if key in seen:
            continue
        seen.add(key)
        if has_blocking_fail(chosen):
            continue
        if float(chosen["score"]) - float(rejected["score"]) < min_score_delta:
            continue
        out.append(make_pair_from_judged(judged, chosen, rejected, pair_type, rubric_version))
    return out[:3]


def build_pairs_from_judge_a(
    path: Path,
    min_score_delta: float,
    pair_type_tag: str | None,
    rubric_version: str,
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for judged in iter_jsonl(path):
        pairs.extend(
            build_pairs_for_prompt(
                judged,
                min_score_delta=min_score_delta,
                pair_type_tag=pair_type_tag,
                rubric_version=rubric_version,
            )
        )
    return pairs


def load_rankings(path: Path) -> dict[str, dict[str, int]]:
    """Returns {prompt_id: {source_model: rank}} from a judged jsonl."""
    rankings: dict[str, dict[str, int]] = {}
    for row in iter_jsonl(path):
        pid = str(row["prompt_id"])
        rc = row.get("ranked_candidates") or []
        rankings[pid] = {
            str(c["source_model"]): int(c["rank"])
            for c in rc
            if c.get("source_model") and c.get("rank")
        }
    return rankings


def judge_b_agrees(
    rankings_b: dict[str, dict[str, int]],
    prompt_id: str,
    chosen_model: str,
    rejected_model: str,
) -> bool:
    """Return True if judge_b also ranks chosen_model above rejected_model."""
    prompt_ranks = rankings_b.get(prompt_id)
    if not prompt_ranks:
        return False
    rank_chosen = prompt_ranks.get(chosen_model)
    rank_rejected = prompt_ranks.get(rejected_model)
    if rank_chosen is None or rank_rejected is None:
        return False
    return rank_chosen < rank_rejected  # lower rank number = better


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs",
        type=Path,
        default=OPENRLHF_RM_DIR / "rm_v6_finegrained_pairs_combined.jsonl",
        help="All v6 fine-grained pairs (from DeepSeek judge)",
    )
    parser.add_argument(
        "--judge-a",
        type=Path,
        default=None,
        help="First judge output jsonl. When set, derive pairs directly from this judge's rankings.",
    )
    parser.add_argument(
        "--judge-b",
        type=Path,
        required=True,
        help="Second judge output jsonl (e.g. judged_qwen36plus_all12448.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OPENRLHF_RM_DIR / "rm_v6.2_finegrained_pairs_combined.jsonl",
    )
    parser.add_argument(
        "--cross-judge-name",
        type=str,
        default="qwen3.6-plus",
        help="Name to write into extra_info.cross_judge for kept pairs.",
    )
    parser.add_argument("--rubric-version", type=str, default="v6.2")
    parser.add_argument("--pair-type-tag", type=str, default=None)
    parser.add_argument("--min-score-delta", type=float, default=0.12)
    args = parser.parse_args()

    print(f"Loading judge_b rankings from {args.judge_b} ...")
    rankings_b = load_rankings(args.judge_b)
    print(f"  Loaded rankings for {len(rankings_b)} prompts")

    kept: list[dict[str, Any]] = []
    stats: dict[str, int] = defaultdict(int)

    source_rows = (
        build_pairs_from_judge_a(
            args.judge_a,
            min_score_delta=args.min_score_delta,
            pair_type_tag=args.pair_type_tag,
            rubric_version=args.rubric_version,
        )
        if args.judge_a
        else iter_jsonl(args.pairs)
    )

    for row in source_rows:
        ei = row.get("extra_info") or {}
        prompt_id = str(ei.get("prompt_id", ""))
        chosen_model = str(ei.get("source_model_chosen", ""))
        rejected_model = str(ei.get("source_model_rejected", ""))
        delta = ei.get("rank_delta", "?")
        conf = ei.get("label_confidence", "?")

        key = f"delta={delta},conf={conf}"
        stats[f"total_{key}"] += 1

        if not prompt_id or not chosen_model or not rejected_model:
            stats["skip_missing_fields"] += 1
            continue

        if judge_b_agrees(rankings_b, prompt_id, chosen_model, rejected_model):
            # Mark as cross-validated
            row = dict(row)
            extra = dict(ei)
            extra["label_confidence"] = "high_crossvalidated"
            extra["cross_judge"] = args.cross_judge_name
            extra["rubric_version"] = args.rubric_version
            if args.pair_type_tag:
                extra["pair_type"] = args.pair_type_tag
            row["extra_info"] = extra
            kept.append(row)
            stats[f"kept_{key}"] += 1
        else:
            stats[f"dropped_{key}"] += 1

    n = write_jsonl(args.output, kept)

    total = sum(v for k, v in stats.items() if k.startswith("total_"))
    result = {
        "pairs_input": total,
        "pairs_kept": n,
        "pairs_dropped": total - n,
        "retention_rate": f"{n / total:.1%}" if total else "n/a",
        "output": str(args.output),
        "judge_b_prompts_loaded": len(rankings_b),
        "breakdown": dict(stats),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
