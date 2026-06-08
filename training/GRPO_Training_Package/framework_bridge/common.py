#!/usr/bin/env python3
"""Shared helpers for ChatJoy RLHF framework bridge scripts."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


def load_jsonl(path: Path) -> Iterator[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            if isinstance(raw, dict) and isinstance(raw.get("item"), dict):
                item = dict(raw["item"])
                item.setdefault("_source_index", raw.get("index"))
                yield item
            elif isinstance(raw, dict):
                yield raw


def _string(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def extract_prompt(record: Dict) -> str:
    return (
        _string(record.get("prompt"))
        or _string(record.get("instruction"))
        or _string(record.get("question"))
        or _string(record.get("query"))
    )


def extract_system(record: Dict) -> str:
    return (
        _string(record.get("system"))
        or _string(record.get("chosen_system"))
        or _string(record.get("prompt_system"))
    )


def extract_language(record: Dict) -> str:
    return _string(record.get("language")) or "unknown"


def extract_output(record: Dict) -> str:
    return (
        _string(record.get("output"))
        or _string(record.get("response"))
        or _string(record.get("chosen"))
    )


def extract_pair(record: Dict) -> Optional[Dict]:
    prompt = extract_prompt(record)
    system = extract_system(record)
    chosen = _string(record.get("chosen")) or extract_output(record)
    rejected = _string(record.get("rejected"))
    if not prompt or not chosen or not rejected:
        return None

    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    return {
        "prompt": prompt,
        "system": system,
        "chosen": chosen,
        "rejected": rejected,
        "language": extract_language(record),
        "rejected_type": _string(record.get("rejected_type")),
        "reason": _string(record.get("reason")),
        "metadata": metadata,
        "source_index": record.get("_source_index", metadata.get("source_index")),
    }


def extract_prompt_record(record: Dict) -> Optional[Dict]:
    prompt = extract_prompt(record)
    system = extract_system(record)
    if not prompt:
        return None

    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    return {
        "prompt": prompt,
        "system": system,
        "language": extract_language(record),
        "source_index": record.get("_source_index", metadata.get("source_index")),
        "record_type": "pair" if "rejected" in record else "sft",
    }


def to_chat_messages(system: str, prompt: str) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return messages


def assistant_message(text: str) -> List[Dict[str, str]]:
    return [{"role": "assistant", "content": text}]


def train_val_split(items: Sequence[Dict], val_ratio: float, seed: int) -> Tuple[List[Dict], List[Dict]]:
    if not items:
        return [], []

    ordered = list(items)
    random.Random(seed).shuffle(ordered)

    if val_ratio <= 0:
        return ordered, []

    val_size = max(1, int(len(ordered) * val_ratio))
    if val_size >= len(ordered):
        val_size = max(0, len(ordered) - 1)

    if val_size == 0:
        return ordered, []

    return ordered[val_size:], ordered[:val_size]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def dedupe_prompt_records(records: Iterable[Dict]) -> List[Dict]:
    deduped: List[Dict] = []
    seen = set()
    for record in records:
        key = (record.get("system", ""), record.get("prompt", ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped
