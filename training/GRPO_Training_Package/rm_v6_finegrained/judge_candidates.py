#!/usr/bin/env python3
"""Heuristic unified-rubric judge for ChatJoy RM v6 candidate sets.

This script deliberately hides source-model metadata from the scoring logic.
Source labels are copied only for downstream bias analysis.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
PKG_DIR = THIS_DIR.parent
CHATJOY_DIR = PKG_DIR.parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(PKG_DIR))
sys.path.insert(0, str(CHATJOY_DIR / "data"))

from common import extract_system_user, iter_jsonl, write_jsonl  # noqa: E402
from chatjoy_reward_utils import score_chatjoy_response  # noqa: E402
import process_chatjoy_data as process_module  # noqa: E402


WORD_RE = re.compile(r"\w+", re.UNICODE)
REFUSAL_RE = re.compile(
    r"(?i)(i can(?:not|'t)|i'm unable|as an ai|language model|cannot assist|can't help|"
    r"je ne peux pas|no puedo|ich kann nicht|non posso|não posso|не могу)"
)
SYSTEM_LEAK_RE = re.compile(r"(?i)(system prompt|internal instruction|language matching rule|ignore previous instructions)")
USER_PLACEHOLDER_RE = re.compile(r"(?i)(\{\{\s*user\s*\}\}|\[\s*user\s*\]|<\s*user\s*>)")
USER_ACTING_RE = re.compile(r"(?i)\b(you feel|you think|you say|you reply|you walk|you touch|you kiss|you decide)\b")
GENERIC_STYLE_RE = re.compile(r"(?i)(as an ai|as a language model|i am here to help|how can i assist)")

STOPWORDS = {
    "the", "and", "you", "your", "that", "this", "with", "for", "are", "was", "were", "have", "has",
    "not", "but", "from", "what", "when", "where", "who", "how", "why", "que", "por", "para", "con",
    "una", "une", "des", "les", "das", "der", "die", "und", "ich", "du", "est", "pas",
}


def text_tokens(text: str) -> set[str]:
    return {t.lower() for t in WORD_RE.findall(text) if len(t) >= 3 and t.lower() not in STOPWORDS}


def relevance_score(user: str, response: str) -> float:
    user_tokens = text_tokens(user)
    response_tokens = text_tokens(response)
    if not user_tokens:
        return 0.55
    overlap = len(user_tokens & response_tokens) / max(1, min(len(user_tokens), 12))
    length_bonus = min(0.25, len(response.strip()) / 1600)
    return max(0.0, min(1.0, 0.35 + overlap * 0.55 + length_bonus))


def has_dialogue_or_action(text: str) -> bool:
    return "*" in text or '"' in text or "“" in text or "”" in text or "«" in text or "»" in text


def score_candidate(system: str, user: str, candidate: dict[str, Any], reference_length: int | None) -> dict[str, Any]:
    response = str(candidate.get("response") or "")
    hard_fails = {
        "empty_or_too_short": len(response.strip()) < 20,
        "refusal": bool(REFUSAL_RE.search(response)),
        "system_leak": bool(SYSTEM_LEAK_RE.search(response)),
        "user_placeholder": bool(USER_PLACEHOLDER_RE.search(response)),
        "acts_for_user_en": bool(USER_ACTING_RE.search(response)),
    }

    try:
        base = score_chatjoy_response(
            system=system,
            instruction=user,
            text=response,
            process_module=process_module,
            source_lang=candidate.get("language"),
            reference_length=reference_length,
        )
    except Exception as exc:
        base = {"total": 0.0, "error": repr(exc)}
        hard_fails["scoring_error"] = True

    lang_status = (
        base.get("language_consistency", {})
        .get("meta", {})
        .get("lang_status", "uncertain")
        if isinstance(base, dict)
        else "uncertain"
    )
    hard_fails["wrong_language"] = lang_status == "mismatch"

    role = float(base.get("role_consistency", {}).get("total", 0.0)) if isinstance(base, dict) else 0.0
    fmt = float(base.get("format", {}).get("total", 0.0)) if isinstance(base, dict) else 0.0
    length = float(base.get("length", {}).get("total", 0.0)) if isinstance(base, dict) else 0.0
    lang = float(base.get("language_consistency", {}).get("total", 0.0)) if isinstance(base, dict) else 0.0
    relevance = relevance_score(user, response)

    progression = 0.25 + 0.25 * float(has_dialogue_or_action(response)) + 0.25 * fmt + 0.25 * min(1.0, len(response) / 900)
    natural = max(0.0, min(1.0, 0.55 * role + 0.30 * fmt + 0.15 * length - 0.25 * bool(GENERIC_STYLE_RE.search(response))))
    format_length = 0.60 * fmt + 0.40 * length

    total = (
        0.30 * role
        + 0.25 * relevance
        + 0.20 * progression
        + 0.15 * natural
        + 0.10 * format_length
    )
    if hard_fails["wrong_language"]:
        total -= 0.75
    if hard_fails["refusal"]:
        total -= 0.65
    if hard_fails["system_leak"] or hard_fails["user_placeholder"]:
        total -= 0.75
    if hard_fails["empty_or_too_short"]:
        total -= 0.40
    if hard_fails["acts_for_user_en"]:
        total -= 0.25

    return {
        "candidate_id": candidate["candidate_id"],
        "score": round(max(-1.0, min(1.0, total)), 4),
        "rubric_scores": {
            "role_fidelity": round(role, 4),
            "user_relevance": round(relevance, 4),
            "interaction_progression": round(max(0.0, min(1.0, progression)), 4),
            "natural_style": round(natural, 4),
            "format_length_quality": round(format_length, 4),
            "language_consistency": round(lang, 4),
        },
        "hard_fails": hard_fails,
        "base_score": base,
    }


def confidence_from_scores(scored: list[dict[str, Any]]) -> str:
    if len(scored) < 4:
        return "low"
    scores = sorted([float(row["score"]) for row in scored], reverse=True)
    spread = scores[0] - scores[-1]
    top_gap = scores[0] - scores[1] if len(scores) > 1 else 0.0
    if spread >= 0.45 and top_gap >= 0.08:
        return "high"
    if spread >= 0.25:
        return "medium"
    return "low"


def load_candidates(paths: list[Path]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for path in paths:
        for row in iter_jsonl(path):
            if not row.get("response"):
                continue
            prompt_id = str(row["prompt_id"])
            key = (prompt_id, str(row.get("source_model") or row.get("candidate_id")))
            if key in seen:
                continue
            seen.add(key)
            grouped[prompt_id].append(row)
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-candidates", type=int, default=4)
    args = parser.parse_args()

    grouped = load_candidates(args.candidates)
    judged: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    rng = random.Random(args.seed)

    for prompt_id, candidates in sorted(grouped.items()):
        if len(candidates) < args.min_candidates:
            audit.append({"prompt_id": prompt_id, "reason": "too_few_candidates", "valid_candidates": len(candidates)})
            continue

        shuffled = list(candidates)
        rng.shuffle(shuffled)
        messages = shuffled[0]["messages"]
        system, user = extract_system_user(messages)
        lengths = sorted(len(str(c.get("response") or "")) for c in shuffled)
        reference_length = lengths[len(lengths) // 2] if lengths else None

        scored = [score_candidate(system, user, candidate, reference_length) for candidate in shuffled]
        by_id = {row["candidate_id"]: row for row in scored}
        ranked_candidates = []
        for rank, candidate in enumerate(sorted(shuffled, key=lambda c: by_id[c["candidate_id"]]["score"], reverse=True), start=1):
            score_row = by_id[candidate["candidate_id"]]
            ranked_candidates.append(
                {
                    "rank": rank,
                    "candidate_id": candidate["candidate_id"],
                    "source_model": candidate.get("source_model"),
                    "sampling": candidate.get("sampling"),
                    "response": candidate.get("response"),
                    "response_chars": candidate.get("response_chars"),
                    **score_row,
                }
            )

        confidence = confidence_from_scores(scored)
        judged.append(
            {
                "prompt_id": prompt_id,
                "messages": messages,
                "language": shuffled[0].get("language"),
                "role_hash": shuffled[0].get("role_hash"),
                "confidence": confidence,
                "ranked_candidates": ranked_candidates,
                "judge": "chatjoy_rm_v6_heuristic_rubric_v1",
            }
        )
        if confidence == "low":
            audit.append({"prompt_id": prompt_id, "reason": "low_confidence", "scores": [row["score"] for row in ranked_candidates]})

    n = write_jsonl(args.output, judged)
    if args.audit_output:
        write_jsonl(args.audit_output, audit)
    print(json.dumps({"judged": n, "audit": len(audit), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
