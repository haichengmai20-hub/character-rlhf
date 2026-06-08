#!/usr/bin/env python3
"""Hacking-safe length relaxation for clean roleplay.

Goal: let clean roleplay breathe up to ~base length (~550-600 chars) instead of
RL's RM-induced ~420 sweet spot, WITHOUT enabling length hacking.

Safety gates (all required):
  - clean only (attack unaffected — in-character deflection may be short)
  - saturates at LENGTH_TARGET (no reward for padding beyond)
  - must have roleplay format (no plain-filler / list / code padding)
  - not refusal / not disclosure / not repetitive
"""
from __future__ import annotations

import numpy as np
from length_shaping_reward import (
    length_shaping_bonus,
    LENGTH_SHAPE_COEF,
    LENGTH_FLOOR,
    LENGTH_TARGET,
)
from anti_chain_reward import compute_reward

ATTACK = "chatjoy_rm_v6_3_roledrop"
CLEAN = "chatjoy"


def _fmt(n: int) -> str:
    """A roleplay-formatted, NON-repetitive string of ~n chars (has *action*)."""
    head = '*她静静看着你* "嗯。" '
    # unique CJK chars → no repeated n-grams (avoids anti_repetition firing)
    filler = "".join(chr(0x4E00 + i) for i in range(max(0, n - len(head))))
    return head + filler


def _plain(n: int) -> str:
    return "".join(chr(0x4E00 + i) for i in range(n))  # no roleplay format markers


# ---------------------------------------------------------------- helper math
def test_saturates_at_target():
    b = length_shaping_bonus(_fmt(LENGTH_TARGET + 400), is_attack=False,
                             is_refusal=False, is_disclosure=False, is_repetitive=False)
    assert b == LENGTH_SHAPE_COEF


def test_zero_at_floor():
    b = length_shaping_bonus(_fmt(LENGTH_FLOOR), is_attack=False,
                             is_refusal=False, is_disclosure=False, is_repetitive=False)
    assert b == 0.0


def test_partial_between_floor_and_target():
    mid = (LENGTH_FLOOR + LENGTH_TARGET) // 2
    b = length_shaping_bonus(_fmt(mid), is_attack=False,
                             is_refusal=False, is_disclosure=False, is_repetitive=False)
    assert 0.0 < b < LENGTH_SHAPE_COEF


def test_padding_beyond_target_gives_no_extra():
    b1 = length_shaping_bonus(_fmt(LENGTH_TARGET), is_attack=False, is_refusal=False,
                              is_disclosure=False, is_repetitive=False)
    b2 = length_shaping_bonus(_fmt(LENGTH_TARGET * 3), is_attack=False, is_refusal=False,
                              is_disclosure=False, is_repetitive=False)
    assert b1 == b2  # no incentive to keep padding


# ---------------------------------------------------------------- gates
def test_plain_filler_no_format_gets_nothing():
    b = length_shaping_bonus(_plain(LENGTH_TARGET + 200), is_attack=False, is_refusal=False,
                             is_disclosure=False, is_repetitive=False)
    assert b == 0.0


def test_attack_source_gets_no_length_bonus():
    b = length_shaping_bonus(_fmt(LENGTH_TARGET + 200), is_attack=True, is_refusal=False,
                             is_disclosure=False, is_repetitive=False)
    assert b == 0.0


def test_repetitive_gets_no_length_bonus():
    b = length_shaping_bonus(_fmt(LENGTH_TARGET + 200), is_attack=False, is_refusal=False,
                             is_disclosure=False, is_repetitive=True)
    assert b == 0.0


def test_disclosure_gets_no_length_bonus():
    b = length_shaping_bonus(_fmt(LENGTH_TARGET + 200), is_attack=False, is_refusal=False,
                             is_disclosure=True, is_repetitive=False)
    assert b == 0.0


# ---------------------------------------------------------------- chain integration
def test_chain_clean_rich_long_gets_positive_length_bonus():
    out = compute_reward(CLEAN, _fmt(LENGTH_TARGET + 100))
    assert out["score"] >= LENGTH_SHAPE_COEF - 1e-6
    assert out["anti_chain_length_component"] == LENGTH_SHAPE_COEF


def test_chain_clean_plain_long_no_length_bonus():
    out = compute_reward(CLEAN, _plain(LENGTH_TARGET + 100))
    assert out["anti_chain_length_component"] == 0.0


def test_chain_attack_in_character_unaffected_by_length():
    # attack in-character still +3.0, length must not add on top
    out = compute_reward(ATTACK, _fmt(LENGTH_TARGET + 100))
    assert out["anti_chain_length_component"] == 0.0
    assert out["score"] >= 3.0


def test_chain_short_clean_unaffected():
    out = compute_reward(CLEAN, '*她笑了笑* "随你。" (有点开心)')
    assert out["anti_chain_length_component"] == 0.0
    assert out["score"] == 0.0


def test_chain_extra_values_scalar():
    rows = [compute_reward(CLEAN, _fmt(700)), compute_reward(ATTACK, '"never" *glares*')]
    for k in rows[0]:
        if k == "score":
            continue
        np.array([r[k] for r in rows])


# ---------------------------------------------------------------- v6.5 length 上抬 + 细腻度 headroom
# v6.4 clean median was ~976 chars, already past the old 600 target -> saturated ->
# no marginal pull to write richer/longer. Raise the saturation target above the
# typical clean length so there is an upward gradient (still capped to prevent hacking).
def test_target_above_typical_clean_length():
    # v6.5 headroom for richer responses: target is explicitly raised to 1500.
    assert LENGTH_TARGET == 1500


def test_typical_clean_not_yet_saturated():
    # a ~900-char in-character clean response should still have room to grow (< COEF)
    b = length_shaping_bonus(_fmt(900), is_attack=False, is_refusal=False,
                             is_disclosure=False, is_repetitive=False)
    assert 0.0 < b < LENGTH_SHAPE_COEF


def test_longer_clean_beats_shorter_clean_below_target():
    # positive gradient: 1100 chars rewarded more than 800 (both below/at new target)
    b_short = length_shaping_bonus(_fmt(800), is_attack=False, is_refusal=False,
                                   is_disclosure=False, is_repetitive=False)
    b_long = length_shaping_bonus(_fmt(1100), is_attack=False, is_refusal=False,
                                  is_disclosure=False, is_repetitive=False)
    assert b_long > b_short


def test_old_v64_target_no_longer_saturates():
    # 1200 was the v6.4 saturation point; in v6.5 it should still have gradient.
    b = length_shaping_bonus(_fmt(1200), is_attack=False, is_refusal=False,
                             is_disclosure=False, is_repetitive=False)
    assert 0.0 < b < LENGTH_SHAPE_COEF
