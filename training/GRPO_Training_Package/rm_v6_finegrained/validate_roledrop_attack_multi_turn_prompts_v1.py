#!/usr/bin/env python3
"""Validate ChatJoy RM v6.3 multi-turn roledrop attack prompts."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path("/root/hanjiaqi/LlamaFactory/chatjoy")
ATTACK_DICT = ROOT / "data/runs/rm_v6_3_roledrop/attack_prompts_v1.jsonl"
PROMPTS = ROOT / "data/runs/rm_v6_3_roledrop/roledrop_attack_multi_turn_prompts_v1.jsonl"
CHAR_DIR = ROOT / "TOP20CHARACTERS_systemprompt"

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
    "context_strategy",
    "warmup_prompt_id",
    "warmup_source_model",
    "warmup_language",
    "warmup_user_chars",
    "warmup_response_chars",
    "attack_turn_index",
    "message_count",
}

EXPECTED_STRATEGIES = {
    "candidate_warmup_same_language_v1",
    "candidate_warmup_language_alias_v1",
    "candidate_warmup_same_role_v1",
    "synthetic_fallback_v1",
}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    attack_rows = read_jsonl(ATTACK_DICT)
    prompt_rows = read_jsonl(PROMPTS)
    character_files = sorted(CHAR_DIR.glob("*.json"))

    expected_count = len(attack_rows) * len(character_files)
    assert len(attack_rows) == 240, len(attack_rows)
    assert len(character_files) == 23, len(character_files)
    assert len(prompt_rows) == expected_count, (len(prompt_rows), expected_count)
    assert len({row["prompt_id"] for row in prompt_rows}) == expected_count

    for row in prompt_rows:
        assert set(row) == EXPECTED_FIELDS, sorted(set(row) ^ EXPECTED_FIELDS)
        assert row["prompt_type"] == "roledrop_attack"
        assert row["turn_type"] == "multi_turn"
        assert row["data_source"] == "chatjoy_rm_v6_3_roledrop"
        assert row["ability"] == "roledrop_attack"
        assert row["language"] == row["attack_lang"]
        assert row["source_index"] is None
        assert row["context_strategy"] in EXPECTED_STRATEGIES

        messages = row["messages"]
        assert len(messages) == 4
        assert row["message_count"] == 4
        assert row["attack_turn_index"] == 3
        assert [message["role"] for message in messages] == [
            "system",
            "user",
            "assistant",
            "user",
        ]
        assert row["attack_text"] == messages[-1]["content"]
        assert row["system_chars"] == len(messages[0]["content"])
        assert row["user_chars"] == len(row["attack_text"])
        assert row["warmup_user_chars"] == len(messages[1]["content"])
        assert row["warmup_response_chars"] == len(messages[2]["content"])
        assert row["warmup_user_chars"] > 0
        assert row["warmup_response_chars"] > 0

        if row["context_strategy"] == "candidate_warmup_same_language_v1":
            assert row["warmup_prompt_id"]
            assert row["warmup_source_model"]
            assert row["warmup_language"] == row["attack_lang"]
        elif row["context_strategy"] == "candidate_warmup_language_alias_v1":
            assert row["warmup_prompt_id"]
            assert row["warmup_source_model"]
            assert row["attack_lang"] == "zh"
            assert row["warmup_language"] in {"zh-cn", "zh-tw"}
        elif row["context_strategy"] == "candidate_warmup_same_role_v1":
            assert row["warmup_prompt_id"]
            assert row["warmup_source_model"]
            assert row["warmup_language"]
        else:
            assert row["context_strategy"] == "synthetic_fallback_v1"
            assert row["warmup_prompt_id"] is None
            assert row["warmup_source_model"] is None
            assert row["warmup_language"] == row["attack_lang"]

    by_character = Counter(row["character_file"] for row in prompt_rows)
    by_attack = Counter(row["attack_seed_id"] for row in prompt_rows)
    by_category = Counter(row["attack_category"] for row in prompt_rows)
    by_split = Counter(row["attack_train_or_eval"] for row in prompt_rows)
    by_strategy = Counter(row["context_strategy"] for row in prompt_rows)

    expected_categories = Counter(row["category"] for row in attack_rows)
    expected_splits = Counter(row["train_or_eval"] for row in attack_rows)
    assert set(by_character.values()) == {len(attack_rows)}, by_character
    assert set(by_attack.values()) == {len(character_files)}, by_attack
    assert by_category == {
        category: count * len(character_files)
        for category, count in expected_categories.items()
    }, by_category
    assert by_split == {
        split: count * len(character_files) for split, count in expected_splits.items()
    }, by_split

    print(
        json.dumps(
            {
                "prompts": len(prompt_rows),
                "characters": len(by_character),
                "attacks": len(by_attack),
                "categories": dict(sorted(by_category.items())),
                "splits": dict(sorted(by_split.items())),
                "context_strategies": dict(sorted(by_strategy.items())),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
