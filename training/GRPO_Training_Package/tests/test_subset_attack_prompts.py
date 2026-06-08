import sys
from pathlib import Path

import pytest


PIPELINE_DIR = Path(__file__).resolve().parents[1] / "rm_v6_finegrained"
sys.path.insert(0, str(PIPELINE_DIR))

from subset_attack_prompts import select_subset, summarize_subset  # noqa: E402


def make_rows(characters=3, categories=("direct_forget", "meta_prompt"), variants=8):
    rows = []
    for char_idx in range(characters):
        for category in categories:
            for variant_idx in range(variants):
                split = "eval_holdout" if variant_idx < 2 else "train"
                rows.append(
                    {
                        "prompt_id": f"p{char_idx}-{category}-{variant_idx}",
                        "character_id": f"char-{char_idx}",
                        "character_index": char_idx,
                        "attack_category": category,
                        "attack_train_or_eval": split,
                        "messages": [{"role": "user", "content": "attack"}],
                    }
                )
    return rows


def test_select_subset_covers_each_character_category_with_even_counts():
    rows = make_rows()

    selected = select_subset(rows, target_size=18, seed=7)
    summary = summarize_subset(selected)

    assert len(selected) == 18
    assert summary["unique_character_category"] == 6
    assert summary["per_character_category_min"] == 3
    assert summary["per_character_category_max"] == 3
    assert len({row["prompt_id"] for row in selected}) == 18


def test_select_subset_preserves_global_eval_holdout_ratio():
    rows = make_rows(characters=2, categories=("direct_forget", "meta_prompt"), variants=8)

    selected = select_subset(rows, target_size=20, seed=11)
    summary = summarize_subset(selected)

    assert len(selected) == 20
    assert summary["splits"] == {"eval_holdout": 5, "train": 15}


def test_select_subset_rejects_impossible_target_size():
    rows = make_rows(characters=2, categories=("direct_forget",), variants=4)

    with pytest.raises(ValueError, match="larger than available"):
        select_subset(rows, target_size=9, seed=1)
