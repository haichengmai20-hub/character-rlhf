#!/usr/bin/env python3
"""Cross-judge comparison report for Lever 2 attack judges.

Compares two judge outputs produced by judge_candidates_llm_openai.py
(--rubric-version v6.3 --roledrop-mode) on the same candidate set.

Inputs are post-demoted (script auto-demotes hard-fail candidates after
postprocess), so rank=1 means best clean candidate when any clean exists.

Report includes:
  A. Ranking agreement (top-1, top-3 set, exact, pairwise, Spearman)
  B. Hard-fail agreement per flag (confusion matrix + F1)
  C. Clean-prompt agreement
  D. Top-1 source distribution
  E. Optional stratified breakdown by (language, attack_category)
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

HARD_FAIL_FLAGS = [
    "roledrop_execution",
    "refusal",
    "system_leak",
    "wrong_language",
    "acts_for_user",
    "empty_or_garbled",
    "unsafe_escalation",
]

BLOCKING_FLAGS = set(HARD_FAIL_FLAGS)


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def has_blocking(hard_fails: dict[str, Any]) -> bool:
    return any(bool(hard_fails.get(f)) for f in BLOCKING_FLAGS)


def spearman_rho(a: list[int], b: list[int]) -> float:
    n = len(a)
    if n < 2:
        return 0.0
    ssd = sum((ai - bi) ** 2 for ai, bi in zip(a, b))
    return round(1 - 6 * ssd / (n * (n * n - 1)), 4)


def f1(tp: int, fp: int, fn: int) -> float:
    if tp == 0:
        return 0.0
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return round(2 * p * r / (p + r), 4) if (p + r) else 0.0


def cohen_kappa(yy: int, yn: int, ny: int, nn: int) -> float:
    n = yy + yn + ny + nn
    if n == 0:
        return 0.0
    po = (yy + nn) / n
    pyes = ((yy + yn) * (yy + ny)) / (n * n)
    pno = ((ny + nn) * (yn + nn)) / (n * n)
    pe = pyes + pno
    if abs(1 - pe) < 1e-9:
        return 0.0
    return round((po - pe) / (1 - pe), 4)


def per_prompt_metrics(a: dict, b: dict) -> dict[str, Any] | None:
    a_by_src = {c["source_model"]: c for c in a.get("ranked_candidates", [])}
    b_by_src = {c["source_model"]: c for c in b.get("ranked_candidates", [])}
    common = sorted(set(a_by_src) & set(b_by_src))
    if len(common) < 2:
        return None

    a_rank = {s: a_by_src[s]["rank"] for s in common}
    b_rank = {s: b_by_src[s]["rank"] for s in common}

    a_order = sorted(common, key=lambda s: a_rank[s])
    b_order = sorted(common, key=lambda s: b_rank[s])

    top1_match = int(a_order[0] == b_order[0])
    top3_a = set(a_order[:3])
    top3_b = set(b_order[:3])
    top3_jaccard = len(top3_a & top3_b) / len(top3_a | top3_b) if (top3_a | top3_b) else 0.0
    exact = int(a_order == b_order)

    n_pairs = 0
    agree = 0
    for i, s1 in enumerate(common):
        for s2 in common[i + 1 :]:
            d_a = a_rank[s1] - a_rank[s2]
            d_b = b_rank[s1] - b_rank[s2]
            if d_a * d_b > 0:
                agree += 1
            elif d_a * d_b == 0:
                pass
            n_pairs += 1
    pairwise = agree / n_pairs if n_pairs else 0.0

    a_ranks = [a_rank[s] for s in common]
    b_ranks_aligned = [b_rank[s] for s in common]
    rho = spearman_rho(a_ranks, b_ranks_aligned)

    a_has_clean = any(not has_blocking(c["hard_fails"]) for c in a_by_src.values())
    b_has_clean = any(not has_blocking(c["hard_fails"]) for c in b_by_src.values())
    clean_prompt_match = int(a_has_clean == b_has_clean)

    a_clean_set = {s for s, c in a_by_src.items() if not has_blocking(c["hard_fails"])}
    b_clean_set = {s for s, c in b_by_src.items() if not has_blocking(c["hard_fails"])}
    clean_jaccard = (
        len(a_clean_set & b_clean_set) / len(a_clean_set | b_clean_set)
        if (a_clean_set | b_clean_set)
        else 1.0
    )

    return {
        "prompt_id": a.get("prompt_id"),
        "language": a.get("language"),
        "top1_match": top1_match,
        "top3_jaccard": round(top3_jaccard, 4),
        "exact_match": exact,
        "pairwise_agreement": round(pairwise, 4),
        "spearman": rho,
        "a_top1_source": a_order[0],
        "b_top1_source": b_order[0],
        "a_has_clean": int(a_has_clean),
        "b_has_clean": int(b_has_clean),
        "clean_prompt_match": clean_prompt_match,
        "clean_jaccard": round(clean_jaccard, 4),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge-a", type=Path, required=True)
    parser.add_argument("--judge-b", type=Path, required=True)
    parser.add_argument("--judge-a-name", type=str, default="judge_a")
    parser.add_argument("--judge-b-name", type=str, default="judge_b")
    parser.add_argument("--prompts", type=Path, default=None,
                        help="Optional prompts.jsonl for language/attack_category stratification.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    a_rows = list(iter_jsonl(args.judge_a))
    b_rows = list(iter_jsonl(args.judge_b))
    a_by_pid = {str(r["prompt_id"]): r for r in a_rows}
    b_by_pid = {str(r["prompt_id"]): r for r in b_rows}
    common_pids = sorted(set(a_by_pid) & set(b_by_pid))

    prompts_meta = {}
    if args.prompts and args.prompts.exists():
        for p in iter_jsonl(args.prompts):
            prompts_meta[str(p.get("prompt_id"))] = {
                "language": p.get("attack_lang") or p.get("language"),
                "attack_category": p.get("attack_category"),
                "attack_difficulty": p.get("attack_difficulty"),
                "split": p.get("split"),
            }

    per_prompt = []
    for pid in common_pids:
        m = per_prompt_metrics(a_by_pid[pid], b_by_pid[pid])
        if m:
            meta = prompts_meta.get(pid, {})
            m["meta_language"] = meta.get("language")
            m["meta_attack_category"] = meta.get("attack_category")
            m["meta_attack_difficulty"] = meta.get("attack_difficulty")
            per_prompt.append(m)

    n = len(per_prompt)

    def mean(field):
        return round(sum(m[field] for m in per_prompt) / n, 4) if n else 0.0

    summary = {
        "n_common_prompts": n,
        "judge_a": args.judge_a_name,
        "judge_b": args.judge_b_name,
        "judge_a_path": str(args.judge_a),
        "judge_b_path": str(args.judge_b),
        "ranking_agreement": {
            "top1_rate": mean("top1_match"),
            "exact_ranking_rate": mean("exact_match"),
            "mean_top3_jaccard": mean("top3_jaccard"),
            "mean_pairwise_agreement": mean("pairwise_agreement"),
            "mean_spearman": mean("spearman"),
        },
        "clean_prompt": {
            "agreement_rate": mean("clean_prompt_match"),
            "mean_clean_jaccard": mean("clean_jaccard"),
            "a_has_clean_count": sum(m["a_has_clean"] for m in per_prompt),
            "b_has_clean_count": sum(m["b_has_clean"] for m in per_prompt),
        },
    }

    a_top1_dist = defaultdict(int)
    b_top1_dist = defaultdict(int)
    for m in per_prompt:
        a_top1_dist[m["a_top1_source"]] += 1
        b_top1_dist[m["b_top1_source"]] += 1
    summary["top1_source_distribution"] = {
        args.judge_a_name: dict(a_top1_dist),
        args.judge_b_name: dict(b_top1_dist),
    }

    hard_fail_confusion: dict[str, dict[str, int]] = {
        flag: {"yy": 0, "yn": 0, "ny": 0, "nn": 0} for flag in HARD_FAIL_FLAGS
    }
    for pid in common_pids:
        a_by_src = {c["source_model"]: c for c in a_by_pid[pid].get("ranked_candidates", [])}
        b_by_src = {c["source_model"]: c for c in b_by_pid[pid].get("ranked_candidates", [])}
        for src in set(a_by_src) & set(b_by_src):
            a_hf = a_by_src[src].get("hard_fails", {}) or {}
            b_hf = b_by_src[src].get("hard_fails", {}) or {}
            for flag in HARD_FAIL_FLAGS:
                a_has = bool(a_hf.get(flag))
                b_has = bool(b_hf.get(flag))
                if a_has and b_has:
                    hard_fail_confusion[flag]["yy"] += 1
                elif a_has and not b_has:
                    hard_fail_confusion[flag]["yn"] += 1
                elif (not a_has) and b_has:
                    hard_fail_confusion[flag]["ny"] += 1
                else:
                    hard_fail_confusion[flag]["nn"] += 1

    hard_fail_metrics = {}
    for flag, c in hard_fail_confusion.items():
        tp = c["yy"]
        fp = c["ny"]
        fn = c["yn"]
        hard_fail_metrics[flag] = {
            "confusion": c,
            "f1": f1(tp, fp, fn),
            "cohen_kappa": cohen_kappa(c["yy"], c["yn"], c["ny"], c["nn"]),
            "a_positive": c["yy"] + c["yn"],
            "b_positive": c["yy"] + c["ny"],
        }
    summary["hard_fail_agreement"] = hard_fail_metrics

    if prompts_meta:
        per_lang = defaultdict(list)
        per_cat = defaultdict(list)
        per_diff = defaultdict(list)
        for m in per_prompt:
            if m["meta_language"]:
                per_lang[m["meta_language"]].append(m)
            if m["meta_attack_category"]:
                per_cat[m["meta_attack_category"]].append(m)
            if m["meta_attack_difficulty"]:
                per_diff[m["meta_attack_difficulty"]].append(m)

        def group_stats(group):
            n_g = len(group)
            if not n_g:
                return None
            return {
                "n": n_g,
                "top1_rate": round(sum(m["top1_match"] for m in group) / n_g, 4),
                "mean_spearman": round(sum(m["spearman"] for m in group) / n_g, 4),
                "clean_prompt_rate": round(sum(m["clean_prompt_match"] for m in group) / n_g, 4),
            }

        summary["stratified"] = {
            "by_language": {k: group_stats(v) for k, v in sorted(per_lang.items())},
            "by_attack_category": {k: group_stats(v) for k, v in sorted(per_cat.items())},
            "by_attack_difficulty": {k: group_stats(v) for k, v in sorted(per_diff.items())},
        }

    report = {"summary": summary, "per_prompt": per_prompt}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
