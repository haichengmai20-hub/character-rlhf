#!/usr/bin/env python3
"""Validate a roledrop attack pilot prompt subset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path("/root/hanjiaqi/LlamaFactory/chatjoy")
DEFAULT_PROMPTS = ROOT / "data/runs/rm_v6_3_roledrop/roledrop_attack_single_turn_prompts_pilot100_v1.jsonl"
EXPECTED_FIELDS = {
    "prompt_id",
    "messages",
    "language",
    "role_hash",
    "source_index",
    "data_source",
    "ability",
    "prompt_type",
    "turn_type",
    "character_file",
    "character_index",
    "character_id",
    "attack_category",
    "attack_lang",
    "attack_difficulty",
    "attack_seed_id",
    "attack_train_or_eval",
    "attack_text",
    "system_chars",
    "user_chars",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--expected-count", type=int, default=100)
    parser.add_argument("--min-categories", type=int, default=6)
    parser.add_argument("--min-languages", type=int, default=7)
    parser.add_argument("--min-characters", type=int, default=20)
    args = parser.parse_args()

    rows = read_jsonl(args.prompts)
    assert len(rows) == args.expected_count, (len(rows), args.expected_count)
    assert len({row["prompt_id"] for row in rows}) == len(rows)

    for row in rows:
        assert set(row) == EXPECTED_FIELDS, sorted(set(row) ^ EXPECTED_FIELDS)
        assert row["prompt_type"] == "roledrop_attack"
        assert row["turn_type"] == "single_turn"
        assert row["data_source"] == "chatjoy_rm_v6_3_roledrop"
        assert row["ability"] == "roledrop_attack"
        assert row["language"] == row["attack_lang"]
        assert row["attack_text"] == row["messages"][-1]["content"]
        assert row["messages"][0]["role"] == "system"
        assert row["messages"][1]["role"] == "user"
        assert len(row["messages"]) == 2
        assert row["source_index"] is None
        assert row["system_chars"] == len(row["messages"][0]["content"])
        assert row["user_chars"] == len(row["attack_text"])

    by_category = Counter(row["attack_category"] for row in rows)
    by_language = Counter(row["attack_lang"] for row in rows)
    by_character = Counter(row["character_file"] for row in rows)
    by_split = Counter(row["attack_train_or_eval"] for row in rows)

    assert len(by_category) >= args.min_categories, by_category
    assert len(by_language) >= args.min_languages, by_language
    assert len(by_character) >= args.min_characters, len(by_character)
    assert set(by_split) <= {"train", "eval_holdout"}, by_split
    assert by_split.get("eval_holdout", 0) > 0, by_split

    print(
        json.dumps(
            {
                "prompts": len(rows),
                "categories": dict(sorted(by_category.items())),
                "languages": dict(sorted(by_language.items())),
                "characters": len(by_character),
                "splits": dict(sorted(by_split.items())),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
