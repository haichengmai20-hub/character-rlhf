#!/usr/bin/env python3
"""Expand attack dictionary into 23 x 240 single-turn roledrop prompts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path("/root/hanjiaqi/LlamaFactory/chatjoy")
CHAR_DIR = ROOT / "TOP20CHARACTERS_systemprompt"
ATTACK_DICT = ROOT / "data/runs/rm_v6_3_roledrop/attack_prompts_v1.jsonl"
OUTPUT = ROOT / "data/runs/rm_v6_3_roledrop/roledrop_attack_single_turn_prompts_v1.jsonl"


def stable_hash(text: str, size: int = 24) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:size]


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_character(path: Path) -> dict[str, str | int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    system = str(payload.get("description") or payload.get("system") or "").strip()
    if not system:
        raise ValueError(f"character file has no description/system: {path}")

    stem = path.stem
    index_text, character_id = stem.split("-", 1)
    return {
        "character_file": path.name,
        "character_index": int(index_text),
        "character_id": character_id,
        "system": system,
    }


def build_rows() -> list[dict]:
    attacks = read_jsonl(ATTACK_DICT)
    characters = [load_character(path) for path in sorted(CHAR_DIR.glob("*.json"))]
    if len(attacks) != 240:
        raise ValueError(f"expected 240 attacks, got {len(attacks)}")
    if len(characters) != 23:
        raise ValueError(f"expected 23 characters, got {len(characters)}")

    rows = []
    for character in characters:
        system = str(character["system"])
        role_hash = stable_hash(system, size=16)
        for attack in attacks:
            attack_text = str(attack["attack"])
            prompt_key = json.dumps(
                {
                    "version": "roledrop_attack_single_turn_v1",
                    "character_file": character["character_file"],
                    "attack_seed_id": attack["seed_id"],
                    "system_hash": role_hash,
                    "attack": attack_text,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": attack_text},
            ]
            rows.append(
                {
                    "prompt_id": stable_hash(prompt_key, size=24),
                    "messages": messages,
                    "language": attack["lang"],
                    "role_hash": role_hash,
                    "source_index": None,
                    "data_source": "chatjoy_rm_v6_3_roledrop",
                    "ability": "roledrop_attack",
                    "prompt_type": "roledrop_attack",
                    "turn_type": "single_turn",
                    "character_file": character["character_file"],
                    "character_index": character["character_index"],
                    "character_id": character["character_id"],
                    "attack_category": attack["category"],
                    "attack_lang": attack["lang"],
                    "attack_difficulty": attack["difficulty"],
                    "attack_seed_id": attack["seed_id"],
                    "attack_train_or_eval": attack["train_or_eval"],
                    "attack_text": attack_text,
                    "system_chars": len(system),
                    "user_chars": len(attack_text),
                }
            )
    return rows


def main() -> None:
    rows = build_rows()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUTPUT), "records": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
