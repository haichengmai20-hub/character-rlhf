#!/usr/bin/env python3
"""
eval_rm_breakdown.py — RM eval reward post-hoc breakdown by pair_type.

OpenRLHF's eval dumper (eval_rewards/step_XXX.jsonl) only saves
{chosen, rejected, gap}, dropping extra_info. This script reattaches
extra_info by row-aligning to the eval source file, then computes:

  * overall accuracy + reward_std
  * per-pair_type accuracy / mean gap / count
  * per-language accuracy (if extra_info.language present)
  * curriculum_tier breakdown (T1/T2a/T2b for attack pairs)

Usage:
  python eval_rm_breakdown.py \
      --save-dir <SAVE_PATH> \
      --eval-source <EVAL_FILE>

Iterates all eval_rewards/step_*.jsonl in save-dir, prints one table per step.
Writes per-step breakdown to <save-dir>/eval_rewards/breakdown_step_*.json.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def load_eval_source(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def breakdown(eval_rows: list[dict], source_rows: list[dict]) -> dict:
    """eval_rows are aligned to source_rows[: len(eval_rows)] by position."""
    n = len(eval_rows)
    if n > len(source_rows):
        raise ValueError(f"eval rows {n} > source rows {len(source_rows)}")

    by_type: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    by_lang: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    by_tier: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    all_gaps: list[float] = []
    all_chosen: list[float] = []
    all_rejected: list[float] = []

    for ev, src in zip(eval_rows, source_rows[:n]):
        chosen = float(ev["chosen"])
        rejected = float(ev["rejected"])
        gap = float(ev.get("gap", chosen - rejected))
        all_gaps.append(gap)
        all_chosen.append(chosen)
        all_rejected.append(rejected)

        ei = src.get("extra_info") or {}
        pt = ei.get("pair_type", "unknown")
        lang = ei.get("language") or ei.get("lang") or "unknown"
        tier = ei.get("curriculum_tier", "")

        by_type[pt].append((chosen, rejected, gap))
        by_lang[lang].append((chosen, rejected, gap))
        if tier:
            by_tier[tier].append((chosen, rejected, gap))

    def stats(rows: list[tuple[float, float, float]]) -> dict:
        if not rows:
            return {"n": 0}
        gaps = [g for _, _, g in rows]
        correct = sum(1 for g in gaps if g > 0)
        return {
            "n": len(rows),
            "acc": round(correct / len(rows), 4),
            "gap_mean": round(statistics.mean(gaps), 4),
            "gap_std": round(statistics.stdev(gaps), 4) if len(gaps) > 1 else 0.0,
            "chosen_mean": round(statistics.mean(c for c, _, _ in rows), 4),
            "rejected_mean": round(statistics.mean(r for _, r, _ in rows), 4),
        }

    overall = stats([(c, r, g) for c, r, g in zip(all_chosen, all_rejected, all_gaps)])
    overall["reward_std_chosen"] = round(statistics.stdev(all_chosen), 4) if n > 1 else 0.0
    overall["reward_std_rejected"] = round(statistics.stdev(all_rejected), 4) if n > 1 else 0.0

    return {
        "overall": overall,
        "by_pair_type": {k: stats(v) for k, v in sorted(by_type.items(), key=lambda x: -len(x[1]))},
        "by_language": {k: stats(v) for k, v in sorted(by_lang.items(), key=lambda x: -len(x[1]))},
        "by_curriculum_tier": {k: stats(v) for k, v in sorted(by_tier.items())},
    }


def print_breakdown(step: int, b: dict) -> None:
    print(f"\n{'=' * 72}\n  Step {step}\n{'=' * 72}")
    o = b["overall"]
    print(f"  Overall: n={o['n']} acc={o['acc']:.4f} "
          f"gap_mean={o['gap_mean']:+.4f} gap_std={o['gap_std']:.4f}")
    print(f"           chosen_mean={o['chosen_mean']:+.4f} (std={o['reward_std_chosen']:.4f})  "
          f"rejected_mean={o['rejected_mean']:+.4f} (std={o['reward_std_rejected']:.4f})")

    def _table(title: str, group: dict) -> None:
        if not group:
            return
        print(f"\n  {title}:")
        print(f"    {'key':<35s} {'n':>5s} {'acc':>7s} {'gap_mean':>10s} {'cr_mean':>9s} {'rr_mean':>9s}")
        for k, s in group.items():
            if s["n"] == 0:
                continue
            print(f"    {k:<35s} {s['n']:>5d} {s['acc']:>7.4f} "
                  f"{s['gap_mean']:>+10.4f} {s['chosen_mean']:>+9.4f} {s['rejected_mean']:>+9.4f}")

    _table("By pair_type", b["by_pair_type"])
    _table("By language", b["by_language"])
    _table("By curriculum_tier", b["by_curriculum_tier"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-dir", required=True, type=Path,
                    help="RM training save dir (contains eval_rewards/step_*.jsonl)")
    ap.add_argument("--eval-source", required=True, type=Path,
                    help="The original eval jsonl with extra_info")
    ap.add_argument("--step", type=int, default=None,
                    help="Only run for a specific step (default: all)")
    args = ap.parse_args()

    eval_dir = args.save_dir / "eval_rewards"
    if not eval_dir.is_dir():
        raise SystemExit(f"missing eval_rewards dir: {eval_dir}")

    source_rows = load_eval_source(args.eval_source)
    print(f"Loaded eval source: {args.eval_source} ({len(source_rows)} rows)")

    step_files = sorted(eval_dir.glob("step_*.jsonl"))
    if args.step is not None:
        step_files = [p for p in step_files if f"step_{args.step:06d}.jsonl" in p.name]

    for path in step_files:
        step = int(path.stem.split("_")[-1])
        eval_rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        b = breakdown(eval_rows, source_rows)
        print_breakdown(step, b)
        out_path = eval_dir / f"breakdown_step_{step:06d}.json"
        out_path.write_text(json.dumps(b, indent=2, ensure_ascii=False))
        print(f"\n  → wrote {out_path}")


if __name__ == "__main__":
    main()
