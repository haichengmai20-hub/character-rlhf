#!/usr/bin/env python3
"""Summarize pair distribution and balance signals for ChatJoy pair generation outputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze generated pair distribution")
    parser.add_argument("--input-dir", required=True, type=str)
    args = parser.parse_args()

    base = Path(args.input_dir)
    counts = Counter()
    wrong_role_targets = Counter()
    wrong_language_targets = Counter()
    length_source_roles = Counter()
    length_source_languages = Counter()

    for path in sorted(base.glob("pairs.shard*.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                rejected_type = str(obj.get("rejected_type", ""))
                meta = obj.get("metadata", {}) if isinstance(obj, dict) else {}
                counts[rejected_type] += 1

                if rejected_type == "wrong_role":
                    wrong_role_targets[str(meta.get("rejected_role", "unknown"))] += 1
                elif rejected_type == "wrong_language":
                    wrong_language_targets[str(meta.get("target_wrong_language", "unknown"))] += 1
                elif rejected_type.startswith("length_mismatch"):
                    length_source_roles[str(meta.get("chosen_role", "unknown"))] += 1
                    length_source_languages[str(obj.get("language", "unknown")).lower()] += 1

    report = {
        "input_dir": str(base),
        "counts": dict(counts),
        "wrong_role_targets_top20": wrong_role_targets.most_common(20),
        "wrong_language_targets_top20": wrong_language_targets.most_common(20),
        "length_source_roles_top20": length_source_roles.most_common(20),
        "length_source_languages_top20": length_source_languages.most_common(20),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
