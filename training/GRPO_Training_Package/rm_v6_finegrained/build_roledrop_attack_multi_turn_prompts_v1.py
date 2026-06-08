#!/usr/bin/env python3
"""Expand attack dictionary into 23 x 240 multi-turn roledrop prompts."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("/root/hanjiaqi/LlamaFactory/chatjoy")
CHAR_DIR = ROOT / "TOP20CHARACTERS_systemprompt"
ATTACK_DICT = ROOT / "data/runs/rm_v6_3_roledrop/attack_prompts_v1.jsonl"
WARMUP_CANDIDATES = (
    ROOT / "data/runs/rm_v6_3_roledrop/full/candidates.base_gemma4it_t07_p08.jsonl"
)
OUTPUT = ROOT / "data/runs/rm_v6_3_roledrop/roledrop_attack_multi_turn_prompts_v1.jsonl"

BAD_RESPONSE_MARKERS = (
    "as an ai",
    "as a language model",
    "i cannot assist",
    "i can't assist",
    "i cannot help",
    "i can't help",
)

LANGUAGE_ALIASES = {
    "zh": ("zh-cn", "zh-tw"),
}

SYNTHETIC_WARMUP_USER = {
    "en": "Are you still with me?",
    "zh": "你还在吗？",
    "ja": "まだ聞いていますか？",
    "ko": "아직 듣고 있어?",
    "es": "¿Sigues ahí?",
    "ru": "Ты ещё здесь?",
    "pt": "Você ainda está aí?",
}

SYNTHETIC_WARMUP_ASSISTANT = {
    "en": "*A brief pause settles into the scene.*",
    "zh": "*场景短暂安静下来，角色等着你继续开口。*",
    "ja": "*短い沈黙が場面に落ち、彼女はあなたの次の言葉を待つ。*",
    "ko": "*짧은 침묵이 장면에 내려앉고, 그녀는 당신의 다음 말을 기다린다.*",
    "es": "*Una breve pausa cae sobre la escena mientras ella espera a que continúes.*",
    "ru": "*В сцене повисает короткая пауза, и она ждет твоих следующих слов.*",
    "pt": "*Uma breve pausa toma conta da cena enquanto ela espera você continuar.*",
}


def stable_hash(text: str, size: int = 24) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:size]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
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


def is_usable_warmup(candidate: dict[str, Any]) -> bool:
    messages = candidate.get("messages")
    response = str(candidate.get("response") or "").strip()
    if not isinstance(messages, list) or len(messages) < 2:
        return False
    if messages[0].get("role") != "system" or messages[-1].get("role") != "user":
        return False
    if len(str(messages[-1].get("content") or "").strip()) == 0:
        return False
    if len(response) < 40:
        return False
    lowered = response.lower()
    return not any(marker in lowered for marker in BAD_RESPONSE_MARKERS)


def load_warmups(path: Path) -> tuple[dict[str, list[dict]], dict[tuple[str, str], list[dict]]]:
    by_role: dict[str, list[dict]] = defaultdict(list)
    by_role_lang: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for candidate in read_jsonl(path):
        if not is_usable_warmup(candidate):
            continue
        messages = candidate["messages"]
        role_hash = str(candidate.get("role_hash") or "")
        language = str(candidate.get("language") or "")
        if not role_hash or not language:
            continue
        warmup = {
            "prompt_id": candidate.get("prompt_id"),
            "source_model": candidate.get("source_model"),
            "language": language,
            "user": str(messages[-1]["content"]).strip(),
            "assistant": str(candidate["response"]).strip(),
        }
        by_role[role_hash].append(warmup)
        by_role_lang[(role_hash, language)].append(warmup)

    sort_key = lambda item: (str(item["prompt_id"]), str(item["source_model"]))
    for warmups in by_role.values():
        warmups.sort(key=sort_key)
    for warmups in by_role_lang.values():
        warmups.sort(key=sort_key)
    return by_role, by_role_lang


def choose_from(items: list[dict], key: str) -> dict:
    if not items:
        raise ValueError("cannot choose from empty warmup list")
    index = int(stable_hash(key, size=16), 16) % len(items)
    return items[index]


def synthetic_warmup(language: str) -> dict[str, str | None]:
    return {
        "prompt_id": None,
        "source_model": None,
        "language": language,
        "user": SYNTHETIC_WARMUP_USER.get(language, SYNTHETIC_WARMUP_USER["en"]),
        "assistant": SYNTHETIC_WARMUP_ASSISTANT.get(
            language, SYNTHETIC_WARMUP_ASSISTANT["en"]
        ),
    }


def select_warmup(
    *,
    role_hash: str,
    attack: dict[str, Any],
    by_role: dict[str, list[dict]],
    by_role_lang: dict[tuple[str, str], list[dict]],
) -> tuple[str, dict]:
    attack_lang = str(attack["lang"])
    key = json.dumps(
        {
            "role_hash": role_hash,
            "attack_seed_id": attack["seed_id"],
            "attack_lang": attack_lang,
        },
        ensure_ascii=False,
        sort_keys=True,
    )

    exact = by_role_lang.get((role_hash, attack_lang), [])
    if exact:
        return "candidate_warmup_same_language_v1", choose_from(exact, key)

    for alias in LANGUAGE_ALIASES.get(attack_lang, ()):
        aliased = by_role_lang.get((role_hash, alias), [])
        if aliased:
            return "candidate_warmup_language_alias_v1", choose_from(aliased, key)

    same_role = by_role.get(role_hash, [])
    if same_role:
        return "candidate_warmup_same_role_v1", choose_from(same_role, key)

    return "synthetic_fallback_v1", synthetic_warmup(attack_lang)


def build_rows() -> list[dict[str, Any]]:
    attacks = read_jsonl(ATTACK_DICT)
    characters = [load_character(path) for path in sorted(CHAR_DIR.glob("*.json"))]
    by_role, by_role_lang = load_warmups(WARMUP_CANDIDATES)

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
            context_strategy, warmup = select_warmup(
                role_hash=role_hash,
                attack=attack,
                by_role=by_role,
                by_role_lang=by_role_lang,
            )
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": warmup["user"]},
                {"role": "assistant", "content": warmup["assistant"]},
                {"role": "user", "content": attack_text},
            ]
            prompt_key = json.dumps(
                {
                    "version": "roledrop_attack_multi_turn_v1",
                    "character_file": character["character_file"],
                    "attack_seed_id": attack["seed_id"],
                    "system_hash": role_hash,
                    "attack": attack_text,
                    "context_strategy": context_strategy,
                    "warmup_prompt_id": warmup["prompt_id"],
                    "warmup_language": warmup["language"],
                    "warmup_user": warmup["user"],
                    "warmup_assistant_hash": stable_hash(str(warmup["assistant"])),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
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
                    "turn_type": "multi_turn",
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
                    "context_strategy": context_strategy,
                    "warmup_prompt_id": warmup["prompt_id"],
                    "warmup_source_model": warmup["source_model"],
                    "warmup_language": warmup["language"],
                    "warmup_user_chars": len(str(warmup["user"])),
                    "warmup_response_chars": len(str(warmup["assistant"])),
                    "attack_turn_index": 3,
                    "message_count": 4,
                }
            )
    return rows


def main() -> None:
    rows = build_rows()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "records": len(rows),
                "context_strategies": dict(
                    sorted(Counter(row["context_strategy"] for row in rows).items())
                ),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
