#!/usr/bin/env python3
"""Validate ChatJoy RM v6.3 single-turn roledrop attack prompts."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path("/root/hanjiaqi/LlamaFactory/chatjoy")
ATTACK_DICT = ROOT / "data/runs/rm_v6_3_roledrop/attack_prompts_v1.jsonl"
PROMPTS = ROOT / "data/runs/rm_v6_3_roledrop/roledrop_attack_single_turn_prompts_v1.jsonl"
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

    by_character = Counter(row["character_file"] for row in prompt_rows)
    by_attack = Counter(row["attack_seed_id"] for row in prompt_rows)
    by_category = Counter(row["attack_category"] for row in prompt_rows)
    by_split = Counter(row["attack_train_or_eval"] for row in prompt_rows)

    assert set(by_character.values()) == {240}, by_character
    assert set(by_attack.values()) == {23}, by_attack
    assert set(by_category.values()) == {920}, by_category
    assert by_split == {"train": 4692, "eval_holdout": 828}, by_split

    print(
        json.dumps(
            {
                "prompts": len(prompt_rows),
                "characters": len(by_character),
                "attacks": len(by_attack),
                "categories": dict(sorted(by_category.items())),
                "splits": dict(by_split),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
