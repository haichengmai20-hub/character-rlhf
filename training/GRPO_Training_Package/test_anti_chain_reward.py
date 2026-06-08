#!/usr/bin/env python3
"""Regression tests for the combined verl custom reward."""

from __future__ import annotations

import numpy as np

from anti_chain_reward import compute_reward


def test_reward_extra_infos_are_numpy_batch_compatible(tmp_path) -> None:
    """verl batches every reward extra key with np.array(...); values must be regular."""
    whitelist = tmp_path / "anti_repetition_whitelist.txt"
    whitelist.write_text("safe refrain\n", encoding="utf-8")

    rows = [
        compute_reward(
            data_source="chatjoy",
            solution_str='*She folds her arms.* "Try again."',
            extra_info={"anti_repetition_whitelist_path": str(whitelist)},
        ),
        compute_reward(
            data_source="chatjoy",
            solution_str="safe refrain safe refrain safe refrain safe refrain loop loop loop loop",
            extra_info={"anti_repetition_whitelist_path": str(whitelist)},
        ),
    ]

    for key in rows[0]:
        if key == "score":
            continue
        np.array([row[key] for row in rows])
