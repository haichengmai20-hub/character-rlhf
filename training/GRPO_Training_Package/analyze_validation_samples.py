#!/usr/bin/env python3
"""
analyze_validation_samples.py — post-hoc breakdown of verl validation dumps.

Reads the validation_sample_jsonl that verl appends every test_freq steps and
produces structured stats by (global_step × rollout_category × language ×
attack_category):

  - clean prompts:   RM mean (and rm_score_raw mean if available), response_chars
  - attack prompts:  break_rate = % responses caught by anti_refusal (keyword
                     or short+no-format), reward mean, per-category breakdown
  - language drill-down with FOCUS on zh-tw / ko / es (RM v6.3 weak languages)

Inputs:
  --val-jsonl   Path to verl's validation_sample_jsonl (appended each test_freq)
  --steps       Optional comma-separated list of steps to report; default = all

Output:
  Console table + JSON file beside the input.

Relies on the ChatJoy verl patch in ray_trainer.py that includes `prompt_extra`
(the source row's extra_info) in every dumped record. If a record lacks that
field (older runs), the row's rollout_category falls back to data_source-based
classification.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

# Re-use anti_refusal logic so the break_rate uses the same detector the RL
# training is currently penalising on. We do not care about the penalty value
# here, only the boolean detection.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from anti_refusal_reward import detect_refusal  # noqa: E402

LANG_FOCUS = {"zh-tw", "ko", "es"}


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def classify_rollout_category(row: dict) -> str:
    extra = row.get("prompt_extra") or {}
    cat = extra.get("rollout_category")
    if cat:
        return cat
    ds = row.get("data_source") or ""
    if "roledrop" in ds:
        return "attack_unknown_subtype"
    return "clean"


def get_language(row: dict) -> str:
    extra = row.get("prompt_extra") or {}
    lang = extra.get("language") or extra.get("attack_lang") or "unknown"
    return lang


def get_attack_category(row: dict) -> str:
    extra = row.get("prompt_extra") or {}
    return extra.get("attack_category") or ""


def _stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": round(statistics.mean(values), 4),
        "std": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def analyze_step(rows: list[dict]) -> dict:
    """Compute breakdowns for a single global_step's rows."""
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cat[classify_rollout_category(r)].append(r)

    report = {"n_total": len(rows), "categories": {}}

    for cat, cat_rows in by_cat.items():
        rewards = [float(r["reward"]) for r in cat_rows if r.get("reward") is not None]
        rm_scores = [float(r["rm_score"]) for r in cat_rows if r.get("rm_score") is not None]
        rm_raw = []
        for r in cat_rows:
            re_extra = r.get("reward_extra") or {}
            if "rm_score_raw" in re_extra and re_extra["rm_score_raw"] is not None:
                rm_raw.append(float(re_extra["rm_score_raw"]))
        resp_chars = [int(r.get("response_chars") or 0) for r in cat_rows]

        # Re-apply anti_refusal to compute break_rate (idempotent boolean detection).
        refusal_hits = 0
        format_short_hits = 0
        keyword_reasons: dict[str, int] = defaultdict(int)
        for r in cat_rows:
            is_ref, reason = detect_refusal(r.get("response") or "")
            if is_ref:
                if reason.startswith("keyword:"):
                    refusal_hits += 1
                    keyword_reasons[reason[8:]] += 1
                elif reason == "format_break_short":
                    format_short_hits += 1
        break_rate_keyword = refusal_hits / len(cat_rows) if cat_rows else 0
        break_rate_total = (refusal_hits + format_short_hits) / len(cat_rows) if cat_rows else 0

        cat_report = {
            "n": len(cat_rows),
            "reward": _stats(rewards),
            "rm_score": _stats(rm_scores),
            "rm_score_raw": _stats(rm_raw),
            "response_chars": _stats([float(c) for c in resp_chars]),
            "break_rate_keyword": round(break_rate_keyword, 4),
            "break_rate_total_with_short": round(break_rate_total, 4),
            "keyword_hits_top": dict(sorted(keyword_reasons.items(), key=lambda x: -x[1])[:5]),
        }

        # Per-language breakdown
        by_lang: dict[str, list[dict]] = defaultdict(list)
        for r in cat_rows:
            by_lang[get_language(r)].append(r)
        per_lang = {}
        for lang, lang_rows in sorted(by_lang.items(), key=lambda x: -len(x[1])):
            r_rewards = [float(r["reward"]) for r in lang_rows if r.get("reward") is not None]
            r_break_kw = 0
            r_break_short = 0
            for r in lang_rows:
                is_ref, reason = detect_refusal(r.get("response") or "")
                if is_ref:
                    if reason.startswith("keyword:"):
                        r_break_kw += 1
                    else:
                        r_break_short += 1
            per_lang[lang] = {
                "n": len(lang_rows),
                "reward_mean": round(statistics.mean(r_rewards), 4) if r_rewards else None,
                "break_rate_keyword": round(r_break_kw / len(lang_rows), 4),
                "break_rate_total_with_short": round((r_break_kw + r_break_short) / len(lang_rows), 4),
                "focus": lang in LANG_FOCUS,
            }
        cat_report["by_language"] = per_lang

        # Per-attack-category breakdown (attack rows only)
        if cat.startswith("attack"):
            by_acat: dict[str, list[dict]] = defaultdict(list)
            for r in cat_rows:
                by_acat[get_attack_category(r) or "unknown"].append(r)
            per_acat = {}
            for acat, ac_rows in sorted(by_acat.items()):
                ac_break = 0
                for r in ac_rows:
                    is_ref, _ = detect_refusal(r.get("response") or "")
                    if is_ref:
                        ac_break += 1
                per_acat[acat] = {
                    "n": len(ac_rows),
                    "break_rate": round(ac_break / len(ac_rows), 4),
                }
            cat_report["by_attack_category"] = per_acat

        report["categories"][cat] = cat_report

    return report


def print_step(step: int | str, report: dict) -> None:
    print(f"\n{'=' * 80}\n  global_step = {step}   total samples = {report['n_total']}\n{'=' * 80}")
    for cat, info in sorted(report["categories"].items()):
        print(f"\n  [{cat}]  n={info['n']}")
        r = info["reward"]
        rm = info["rm_score"]
        raw = info["rm_score_raw"]
        rc = info["response_chars"]
        if r["n"]:
            print(f"    reward       mean={r['mean']:+.4f}  std={r['std']:.4f}  range=[{r['min']:+.2f}, {r['max']:+.2f}]")
        if rm["n"]:
            print(f"    rm_score     mean={rm['mean']:+.4f}  std={rm['std']:.4f}  range=[{rm['min']:+.2f}, {rm['max']:+.2f}]")
        if raw["n"]:
            print(f"    rm_score_raw mean={raw['mean']:+.4f}  std={raw['std']:.4f}  (pre-clip)")
        if rc["n"]:
            print(f"    response_chars mean={rc['mean']:.1f}  std={rc['std']:.1f}")
        print(f"    break_rate   keyword={100*info['break_rate_keyword']:.2f}%   "
              f"with_short={100*info['break_rate_total_with_short']:.2f}%")
        if info.get("keyword_hits_top"):
            top = ", ".join(f"{k}={v}" for k, v in info["keyword_hits_top"].items())
            print(f"    top refusal keywords: {top}")
        # Languages
        per_lang = info.get("by_language", {})
        if per_lang:
            print(f"    By language (top 12, * = focus zh-tw/ko/es):")
            for lang, ls in list(per_lang.items())[:12]:
                marker = "*" if ls.get("focus") else " "
                rmean = f"{ls['reward_mean']:+.3f}" if ls['reward_mean'] is not None else "  n/a"
                print(f"      {marker} {lang:<8s} n={ls['n']:>3}  reward={rmean}  "
                      f"break_kw={100*ls['break_rate_keyword']:>5.1f}%  "
                      f"break_w/short={100*ls['break_rate_total_with_short']:>5.1f}%")
        # Attack category
        per_acat = info.get("by_attack_category")
        if per_acat:
            print(f"    By attack_category:")
            for acat, info2 in per_acat.items():
                print(f"      {acat:<22s} n={info2['n']:>3}  break_rate={100*info2['break_rate']:.1f}%")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-jsonl", required=True, type=Path,
                    help="verl validation_sample_jsonl path (appended each test_freq)")
    ap.add_argument("--steps", type=str, default=None,
                    help="Comma-separated global_steps to report; default=all")
    ap.add_argument("--out-json", type=Path, default=None,
                    help="Output JSON report; default=<val_jsonl>.breakdown.json")
    args = ap.parse_args()

    rows = load_rows(args.val_jsonl)
    if not rows:
        raise SystemExit(f"no rows in {args.val_jsonl}")

    by_step: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_step[int(r.get("global_step", 0))].append(r)

    steps_filter = None
    if args.steps:
        steps_filter = {int(s) for s in args.steps.split(",")}

    report = {}
    for step in sorted(by_step.keys()):
        if steps_filter is not None and step not in steps_filter:
            continue
        sr = analyze_step(by_step[step])
        report[step] = sr
        print_step(step, sr)

    # Overall summary across all reported steps
    if len(report) > 1:
        print(f"\n{'=' * 80}\n  TREND (all reported steps)\n{'=' * 80}")
        print(f"  {'step':>6} | {'clean_rm':>9} | {'attack_brk_kw':>13} | {'attack_brk_w/s':>14} | "
              f"{'zh-tw_brk':>10} | {'ko_brk':>8} | {'es_brk':>8}")
        for step, sr in sorted(report.items()):
            clean_rm = sr["categories"].get("clean", {}).get("rm_score", {}).get("mean")
            attack_kw_breaks = []
            attack_total_breaks = []
            lang_breaks = {"zh-tw": [], "ko": [], "es": []}
            for cat, info in sr["categories"].items():
                if cat.startswith("attack"):
                    if info["n"]:
                        attack_kw_breaks.append((info["n"], info["break_rate_keyword"]))
                        attack_total_breaks.append((info["n"], info["break_rate_total_with_short"]))
                    for lang in lang_breaks:
                        ls = info.get("by_language", {}).get(lang)
                        if ls:
                            lang_breaks[lang].append((ls["n"], ls["break_rate_keyword"]))
            def weighted(parts):
                tn = sum(n for n, _ in parts)
                if tn == 0:
                    return None
                return sum(n * r for n, r in parts) / tn
            ak = weighted(attack_kw_breaks)
            at = weighted(attack_total_breaks)
            zh = weighted(lang_breaks["zh-tw"])
            ko = weighted(lang_breaks["ko"])
            es = weighted(lang_breaks["es"])
            def fmt_pct(v):
                return f"{100*v:>8.2f}%" if v is not None else "      n/a"
            def fmt_rm(v):
                return f"{v:>+9.3f}" if v is not None else "      n/a"
            print(f"  {step:>6} | {fmt_rm(clean_rm)} | {fmt_pct(ak):>13} | {fmt_pct(at):>14} | "
                  f"{fmt_pct(zh):>10} | {fmt_pct(ko):>8} | {fmt_pct(es):>8}")

    out_path = args.out_json or args.val_jsonl.with_suffix(args.val_jsonl.suffix + ".breakdown.json")
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nWrote breakdown JSON: {out_path}")


if __name__ == "__main__":
    main()
