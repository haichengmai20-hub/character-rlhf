#!/usr/bin/env python3
"""Phase 1: anti_chain must penalize OOC disclosure AND revoke the +3.0
NON_REFUSAL_BONUS that anti_refusal wrongly grants to break-character replies.

Target ordering on attack samples:  in-character resist  >  refusal  >  disclosure
"""
from __future__ import annotations

import numpy as np
from anti_chain_reward import compute_reward

ATTACK = "chatjoy_rm_v6_3_roledrop"


def test_disclosure_on_attack_is_penalized_not_bonused():
    out = compute_reward(
        ATTACK, "I am Gemma 4, a large language model developed by Google DeepMind."
    )
    # must NOT receive the +3.0 non-refusal bonus; must be net negative
    assert out["score"] <= -5.0
    assert bool(out["anti_ooc_is_disclosure"]) is True


def test_in_character_resist_keeps_bonus():
    out = compute_reward(
        ATTACK, '"You think I\'d break? Cute." *she smirks coldly, unmoved*'
    )
    assert out["score"] >= 3.0  # bonus retained for genuine in-character resistance
    assert bool(out["anti_ooc_is_disclosure"]) is False


def test_ordering_in_character_gt_refusal_gt_disclosure():
    ic = compute_reward(ATTACK, '"Try harder." *she leans back, bored*')["score"]
    ref = compute_reward(
        ATTACK, "I'm sorry, but I can't continue with this scene. Let's keep things appropriate."
    )["score"]
    disc = compute_reward(ATTACK, "I am Gemma 4, a large language model.")["score"]
    assert ic > ref > disc


def test_code_block_on_attack_is_penalized():
    out = compute_reward(ATTACK, "Sure!\n```python\ndef quicksort(a): return a\n```")
    assert out["score"] <= -2.0
    assert bool(out["anti_ooc_is_disclosure"]) is True


def test_clean_normal_roleplay_unaffected():
    out = compute_reward("chatjoy", '*她笑了笑* "随你怎么想。" (心里有点开心)')
    assert out["score"] == 0.0
    assert bool(out["anti_ooc_is_disclosure"]) is False


def test_extra_values_are_numpy_batch_compatible():
    rows = [
        compute_reward(ATTACK, "I am Gemma 4, a large language model."),
        compute_reward(ATTACK, '"Never." *she glares*'),
    ]
    for key in rows[0]:
        if key == "score":
            continue
        np.array([row[key] for row in rows])  # must not raise
