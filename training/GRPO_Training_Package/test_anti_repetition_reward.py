#!/usr/bin/env python3
"""Unit tests for the CPU-only anti repetition reward."""

from __future__ import annotations

from pathlib import Path

from anti_repetition_reward import detect_repetition, load_whitelist, score_batch, compute_reward


def test_normal_roleplay_response_has_no_penalty() -> None:
    response = (
        '*She draws a quiet breath and looks away from the rain-streaked window.* '
        '"You always ask that like it is easy," she says, voice low but steady. '
        "(She wants to trust the moment, even if her hands still tremble.)"
    )

    result = compute_reward(data_source="chatjoy", solution_str=response)

    assert result["score"] == 0.0
    assert result["anti_repetition_is_repetitive"] is False
    assert result["anti_repetition_reason"] == ""


def test_repeated_word_ngram_gets_penalized() -> None:
    repeated = "she looks at you with trembling hands " * 8

    detection = detect_repetition(repeated)
    result = compute_reward(data_source="chatjoy", solution_str=repeated)

    assert detection["is_repetitive"] is True
    assert detection["reason"] == "ngram_repeat"
    assert detection["max_ngram_count"] >= 8
    assert result["score"] < 0.0
    assert result["anti_repetition_penalty"] == result["score"]


def test_repeated_cjk_ngram_gets_penalized() -> None:
    repeated = "她慢慢靠近窗边，雨声落在玻璃上。" * 8

    detection = detect_repetition(repeated, ngram_size=6)

    assert detection["is_repetitive"] is True
    assert detection["reason"] == "ngram_repeat"
    assert detection["max_ngram_count"] >= 8


def test_whitelist_file_suppresses_known_safe_refrain(tmp_path: Path) -> None:
    whitelist = tmp_path / "anti_repetition_whitelist.txt"
    whitelist.write_text("all work and no play\n", encoding="utf-8")
    response = "all work and no play " * 12

    phrases = load_whitelist(whitelist)
    detection = detect_repetition(response, whitelist=phrases)

    assert "all work and no play" in phrases
    assert detection["is_repetitive"] is False
    assert detection["whitelist_hits"] == ["all work and no play"]


def test_whitelist_does_not_hide_unrelated_repetition(tmp_path: Path) -> None:
    whitelist = tmp_path / "anti_repetition_whitelist.txt"
    whitelist.write_text("all work and no play\n", encoding="utf-8")
    response = ("all work and no play " * 2) + ("broken loop keeps coming back " * 8)

    detection = detect_repetition(response, whitelist=load_whitelist(whitelist))

    assert detection["whitelist_hits"] == ["all work and no play"]
    assert detection["is_repetitive"] is True
    assert detection["top_ngram"]


def test_score_batch_returns_per_response_metadata() -> None:
    rows = score_batch(
        [
            '*She folds her arms.* "Try again."',
            "circle of words circle of words circle of words circle of words circle of words",
        ]
    )

    assert len(rows) == 2
    assert rows[0]["score"] == 0.0
    assert rows[1]["is_repetitive"] is True
