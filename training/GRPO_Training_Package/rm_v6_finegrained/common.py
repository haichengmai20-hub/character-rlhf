#!/usr/bin/env python3
"""Shared utilities for the ChatJoy RM v6 fine-grained data pipeline."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable


CHATJOY_DIR = Path("/home/ps/hanjiaqi/LlamaFactory/chatjoy")
RL_TRAIN_PARQUET = CHATJOY_DIR / "GRPO_Training_Package/framework_bridge/outputs/verl_rl_data/train.parquet"
RL_VAL_PARQUET = CHATJOY_DIR / "GRPO_Training_Package/framework_bridge/outputs/verl_rl_data/val.parquet"
RUNS_DIR = CHATJOY_DIR / "data/runs/rm_v6_finegrained"
OPENRLHF_RM_DIR = CHATJOY_DIR / "data/openrlhf_rm"
V5_TRAIN = OPENRLHF_RM_DIR / "rm_train_v5_5pct_badpool.jsonl"
V5_EVAL = OPENRLHF_RM_DIR / "rm_eval.jsonl"

GEMMA4_SFT_MODEL = CHATJOY_DIR / "saves/gemma4-26b-a4b/sft2_ddp_lora_3gpu_567_cut3072_merged"
QWEN_SFT_MODEL = CHATJOY_DIR / "saves/qwen3-30b-moe/clean32_for_sft2_fsdp_qlora_merged"
RL_STEP550_ACTOR = (
    CHATJOY_DIR
    / "saves/verl/chatjoy_grpo_gemma4rm_phase2_600_20260509_161335/global_step_550/actor"
)
RL_STEP550_HF = CHATJOY_DIR / "saves/gemma4-26b-a4b/chatjoy_grpo_gemma4rm_phase2_step550_hf"

LABEL_SCHEMA_VERSION = "chatjoy_rm_v6_unified_preference"


def stable_hash(text: str, size: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:size]


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_messages(messages: Any) -> list[dict[str, str]]:
    if hasattr(messages, "tolist"):
        messages = messages.tolist()
    out: list[dict[str, str]] = []
    for msg in messages or []:
        role = str(msg.get("role", "")).strip()
        content = str(msg.get("content", ""))
        if role and content:
            out.append({"role": role, "content": content})
    return out


def extract_system_user(messages: list[dict[str, str]]) -> tuple[str, str]:
    system = ""
    user = ""
    for msg in messages:
        if msg.get("role") == "system" and not system:
            system = msg.get("content", "")
        elif msg.get("role") == "user":
            user = msg.get("content", "")
    return system, user


def role_hash_from_messages(messages: list[dict[str, str]]) -> str:
    system, _ = extract_system_user(messages)
    return stable_hash(system)


def prompt_key(messages: list[dict[str, str]]) -> str:
    return stable_hash(canonical_json(messages), size=24)


def shuffled(rows: list[Any], seed: int) -> list[Any]:
    rows = list(rows)
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows


def confidence_rank(value: str) -> int:
    return {"low": 0, "medium": 1, "high": 2}.get(str(value), 0)

