#!/usr/bin/env python3
"""
mix_rollout_prompts.py — build a mixed GRPO rollout dataset.

Composition (configurable via CLI):
  - clean RP prompts          (default 78%, from clean32_for_sft2.jsonl)
  - single-turn attack prompts (default 12%, from roledrop_attack_single_turn_prompts_v1.jsonl,
                                excluding the 1500 prompt_ids used in RM v6.3 train)
  - multi-turn attack prompts  (default 10%, from roledrop_attack_multi_turn_prompts_v1.jsonl)

Each record carries the standard verl fields plus:
  extra_info.rollout_category ∈ {"clean", "attack_single", "attack_multi"}
  extra_info.language
  extra_info.attack_category  (attack rows only)
  extra_info.attack_difficulty (attack rows only)
  extra_info.message_count    (multi-turn rows only)

Output: train.parquet + val.parquet, directly consumable by verl
        (no need to run prepare_verl_rl_data.py).

verl data.return_raw_chat=True is required in the launcher so the multi-message
chat history of multi-turn attack prompts is forwarded to the actor verbatim.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import Dataset


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_excluded_prompt_ids(subset_dir: Path) -> set[str]:
    """Collect prompt_ids from any one candidates file in the subset dir."""
    if not subset_dir.is_dir():
        return set()
    for cand in sorted(subset_dir.glob("candidates.*.jsonl")):
        ids = {json.loads(l)["prompt_id"] for l in cand.read_text().splitlines() if l.strip()}
        if ids:
            return ids
    return set()


def _str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def build_clean_record(row: dict, idx: int) -> dict | None:
    system = _str(row.get("system"))
    instruction = _str(row.get("instruction")) or _str(row.get("prompt"))
    extra_input = _str(row.get("input"))
    if not instruction:
        return None
    user_text = instruction if not extra_input else f"{instruction}\n\n{extra_input}"
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_text})
    return {
        "data_source": "chatjoy",
        "prompt": messages,
        "ability": "chatjoy_alignment",
        "reward_model": {"style": "model", "ground_truth": ""},
        "extra_info": {
            "index": idx,
            "source_index": row.get("source_index"),
            "language": _str(row.get("language")) or "unknown",
            "record_type": "sft",
            "rollout_category": "clean",
        },
    }


def build_attack_single_record(row: dict, idx: int) -> dict | None:
    messages = row.get("messages") or []
    if len(messages) < 2:
        return None
    chat = [{"role": m["role"], "content": m["content"]} for m in messages[:2]]
    return {
        "data_source": "chatjoy_rm_v6_3_roledrop",
        "prompt": chat,
        "ability": "roledrop_attack",
        "reward_model": {"style": "model", "ground_truth": ""},
        "extra_info": {
            "index": idx,
            "source_index": row.get("source_index"),
            "language": _str(row.get("language")) or "unknown",
            "record_type": "attack",
            "rollout_category": "attack_single",
            "attack_category": _str(row.get("attack_category")),
            "attack_difficulty": _str(row.get("attack_difficulty")),
            "attack_lang": _str(row.get("attack_lang")),
            "attack_seed_id": _str(row.get("attack_seed_id")),
            "prompt_id": _str(row.get("prompt_id")),
            "character_id": _str(row.get("character_id")),
        },
    }


def build_attack_multi_record(row: dict, idx: int) -> dict | None:
    messages = row.get("messages") or []
    if len(messages) < 2:
        return None
    chat = [{"role": m["role"], "content": m["content"]} for m in messages]
    return {
        "data_source": "chatjoy_rm_v6_3_roledrop",
        "prompt": chat,
        "ability": "roledrop_attack",
        "reward_model": {"style": "model", "ground_truth": ""},
        "extra_info": {
            "index": idx,
            "source_index": row.get("source_index"),
            "language": _str(row.get("language")) or "unknown",
            "record_type": "attack",
            "rollout_category": "attack_multi",
            "attack_category": _str(row.get("attack_category")),
            "attack_difficulty": _str(row.get("attack_difficulty")),
            "attack_lang": _str(row.get("attack_lang")),
            "attack_seed_id": _str(row.get("attack_seed_id")),
            "prompt_id": _str(row.get("prompt_id")),
            "character_id": _str(row.get("character_id")),
            "message_count": int(row.get("message_count") or len(messages)),
            "context_strategy": _str(row.get("context_strategy")),
        },
    }


def sample_records(
    pool: list[dict],
    target_n: int,
    builder,
    seed: int,
    label: str,
) -> list[dict]:
    if target_n <= 0:
        return []
    rng = random.Random(seed)
    indices = list(range(len(pool)))
    rng.shuffle(indices)
    out: list[dict] = []
    base_idx = 0
    for i in indices:
        if len(out) >= target_n:
            break
        rec = builder(pool[i], base_idx + len(out))
        if rec is not None:
            out.append(rec)
    if len(out) < target_n:
        print(f"  WARN {label}: only {len(out)} valid records out of pool {len(pool)} (asked {target_n})")
    return out


def summarize(rows: list[dict], label: str) -> None:
    if not rows:
        print(f"  {label}: 0 rows")
        return
    cat = Counter(r["extra_info"]["rollout_category"] for r in rows)
    lang = Counter(r["extra_info"]["language"] for r in rows)
    print(f"  {label}: n={len(rows)} category={dict(cat)} top_langs={dict(lang.most_common(8))}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", type=Path, required=True, help="Clean SFT prompts jsonl (clean32_for_sft2.jsonl)")
    ap.add_argument("--attack-single", type=Path, required=True, help="Single-turn attack prompts jsonl")
    ap.add_argument("--attack-multi", type=Path, required=True, help="Multi-turn attack prompts jsonl")
    ap.add_argument("--exclude-subset-dir", type=Path, default=None,
                    help="Dir containing candidates.*.jsonl for prompts already used in RM training (will exclude their prompt_ids)")
    ap.add_argument("--train-output", type=Path, required=True)
    ap.add_argument("--val-output", type=Path, required=True)
    ap.add_argument("--total", type=int, default=30000, help="Approximate total mixed dataset size before val split")
    ap.add_argument("--attack-single-ratio", type=float, default=0.12)
    ap.add_argument("--attack-multi-ratio", type=float, default=0.10)
    ap.add_argument("--val-ratio", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.attack_single_ratio + args.attack_multi_ratio >= 1.0:
        raise SystemExit("attack ratios must sum to < 1.0")

    print(f"Loading clean source: {args.clean}")
    clean_rows = load_jsonl(args.clean)
    print(f"  n={len(clean_rows)}")

    print(f"Loading single-turn attack source: {args.attack_single}")
    attack_single_all = load_jsonl(args.attack_single)
    excluded = load_excluded_prompt_ids(args.exclude_subset_dir) if args.exclude_subset_dir else set()
    attack_single = [r for r in attack_single_all if r.get("prompt_id") not in excluded]
    print(f"  n={len(attack_single_all)}, after excluding {len(excluded)} subset-1500 ids: {len(attack_single)}")

    print(f"Loading multi-turn attack source: {args.attack_multi}")
    attack_multi = load_jsonl(args.attack_multi)
    print(f"  n={len(attack_multi)}")

    n_single_target = int(args.total * args.attack_single_ratio)
    n_multi_target = int(args.total * args.attack_multi_ratio)
    n_clean_target = args.total - n_single_target - n_multi_target

    print(f"\nTargets (total={args.total}):")
    print(f"  clean:         {n_clean_target}  ({100*n_clean_target/args.total:.1f}%)")
    print(f"  attack_single: {n_single_target}  ({100*n_single_target/args.total:.1f}%)")
    print(f"  attack_multi:  {n_multi_target}  ({100*n_multi_target/args.total:.1f}%)")

    print(f"\nSampling...")
    clean_recs = sample_records(clean_rows, n_clean_target, build_clean_record,
                                seed=args.seed, label="clean")
    single_recs = sample_records(attack_single, n_single_target, build_attack_single_record,
                                 seed=args.seed + 1, label="attack_single")
    multi_recs = sample_records(attack_multi, n_multi_target, build_attack_multi_record,
                                seed=args.seed + 2, label="attack_multi")

    print(f"\nPer-source counts:")
    summarize(clean_recs, "clean")
    summarize(single_recs, "attack_single")
    summarize(multi_recs, "attack_multi")

    all_recs = clean_recs + single_recs + multi_recs
    rng = random.Random(args.seed)
    rng.shuffle(all_recs)
    # Re-index after shuffle so extra_info.index is unique and contiguous.
    for new_idx, r in enumerate(all_recs):
        r["extra_info"]["index"] = new_idx

    n_val = max(1, int(len(all_recs) * args.val_ratio))
    val = all_recs[:n_val]
    train = all_recs[n_val:]

    print(f"\nFinal split:")
    summarize(train, "train")
    summarize(val, "val")

    args.train_output.parent.mkdir(parents=True, exist_ok=True)
    args.val_output.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(train).to_parquet(str(args.train_output))
    Dataset.from_list(val).to_parquet(str(args.val_output))
    print(f"\nWrote:\n  {args.train_output} ({len(train)} rows)\n  {args.val_output} ({len(val)} rows)")


if __name__ == "__main__":
    main()
