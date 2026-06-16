#!/usr/bin/env python3
"""v6.9 Richness Selector: best-of-N with balanced composite.

v6.9 composite = action_score + dialogue_score + interaction_progression
                 + sensory_richness + emotional_complexity
(5 axes, equal weight 0.2 each)

Excluded by design:
  - natural_style / language_quality (judge flat, can't differentiate)
  - narrative_layer / monologue (not in target = anti over-shoot by construction)

Fixes vs v68:
  - Density normalization: min(count/8, 1.0) not min(count/len*1000, 1.0) (saturation bug)
  - Monologue gate: presence rate (not share rate) to match base 35.5% metric
"""
from __future__ import annotations
import argparse, json, re, sys, os, time, random
from pathlib import Path
from collections import defaultdict
from typing import Any
import numpy as np


# === Metric functions ===

def count_action_blocks(text: str) -> int:
    """Count *...* action blocks."""
    return len(re.findall(r'\*[^*\n]{2,}\*', text))

def count_dialogue_blocks(text: str) -> int:
    """Count "..." or 「...」 dialogue blocks."""
    return len(re.findall(r'["\u300c\u201c][^"\u300c\u300d\u201d\n]{1,}["\u300d\u201d]', text))

def has_monologue(text: str) -> bool:
    """Presence: does this response contain >=1 inner monologue marker?"""
    patterns = [r'（[^）]{2,}）', r'\([^)]{2,}\)', r'＜[^＞]{2,}＞']
    return any(re.search(p, text) for p in patterns)

def action_score(count: int) -> float:
    """0-1 normalization: typical range 2-8 actions, cap at 8."""
    return min(count / 8.0, 1.0)

def dialogue_score(count: int) -> float:
    """0-1 normalization: typical range 1-5 dialogues, cap at 5."""
    return min(count / 5.0, 1.0)


# === v6.9 composite ===

def v69_composite(rubric: dict, action_count: int, dialogue_count: int) -> float:
    """v6.9 composite from rubric dimensions + regex counts.
    
    Structure half: action_score + dialogue_score (regex, free)
    Semantic half: interaction_progression + sensory_richness + emotional_complexity (from judge)
    Equal weight 0.2 each.
    """
    a = action_score(action_count)
    d = dialogue_score(dialogue_count)
    interaction = rubric.get("interaction_progression", 0.0)
    sensory = rubric.get("sensory_richness", 0.0)
    emotional = rubric.get("emotional_complexity", 0.0)
    return round(0.2 * (a + d + interaction + sensory + emotional), 4)


# === Diversity select ===

def diversity_select(candidates: list[dict], top_k: int, ngram: int = 8) -> list[dict]:
    """Sort by v69_composite desc, pick top-K with opening n-gram dedup."""
    scored = sorted(candidates, key=lambda x: x.get("v69_composite", 0.0), reverse=True)
    selected = []
    seen_ngrams = set()
    for cand in scored:
        resp = cand.get("response", "")
        if not resp:
            continue
        first = re.sub(r'[^\w]+', ' ', resp[:ngram * 3]).strip().lower()[:ngram]
        if first not in seen_ngrams:
            seen_ngrams.add(first)
            selected.append(cand)
            if len(selected) >= top_k:
                break
    return selected


# === Profile check ===

def profile_check(records: list[dict], label: str, base_profile: dict | None = None) -> dict:
    """Check profile against base. Returns pass/fail + details."""
    if base_profile is None:
        base_profile = {"action": 4.1, "dialogue": 1.7, "mono_pct": 35.5}
    
    if not records:
        return {"pass": False, "reason": "no records", "N": 0, "action": 0, "dialogue": 0, "mono_pct": 0, "gates": {}}
    
    actions = [count_action_blocks(r.get("response", "")) for r in records]
    dialogues = [count_dialogue_blocks(r.get("response", "")) for r in records]
    mono_presence = [1 if has_monologue(r.get("response", "")) else 0 for r in records]
    
    mean_action = np.mean(actions)
    mean_dialogue = np.mean(dialogues)
    mono_pct = np.mean(mono_presence) * 100  # presence rate, comparable to base 35.5%
    
    gates = {
        "action_ge_4.5": mean_action >= 4.5,
        "dialogue_ge_base": mean_dialogue >= base_profile.get("dialogue", 1.7),
        "mono_le_35": mono_pct <= 35.0,
    }
    
    return {
        "label": label,
        "N": len(records),
        "action": round(mean_action, 2),
        "dialogue": round(mean_dialogue, 2),
        "mono_pct": round(mono_pct, 1),
        "gates": gates,
        "pass": all(gates.values()),
    }


# === Load functions ===

def load_candidates(paths: list[Path]) -> dict[str, list[dict]]:
    """Load candidates grouped by prompt_id."""
    grouped = defaultdict(list)
    for path in paths:
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if not r.get("response"):
                    continue
                lang = r.get("language", "")
                if lang not in ("zh", "zh-tw", "ja", "ko"):
                    continue
                pid = str(r.get("prompt_id", ""))
                grouped[pid].append(r)
    return grouped


def load_judge(path: Path) -> dict[str, list[dict]]:
    """Load judge results grouped by prompt_id."""
    grouped = defaultdict(list)
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            pid = str(r.get("prompt_id", ""))
            grouped[pid].append(r)
    return grouped


# === Main ===

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates", nargs="+", type=Path, required=True,
                    help="Candidate JSONL files")
    ap.add_argument("--judge", type=Path, required=True,
                    help="Pre-judged JSONL from judge_richness_v68")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output: selected rich training data JSONL")
    ap.add_argument("--top-k", type=int, default=2,
                    help="Number of selections per scene")
    ap.add_argument("--diversity-ngram", type=int, default=8,
                    help="N-gram dedup window")
    ap.add_argument("--seed", type=int, default=69)
    a = ap.parse_args()

    # 1. Load candidates
    candidates = load_candidates(a.candidates)
    total_cands = sum(len(v) for v in candidates.values())
    print(f"[load] {len(candidates)} prompt_ids, {total_cands} CJK candidates")

    # 2. Load judge scores
    judge = load_judge(a.judge)
    print(f"[load] {len(judge)} prompt_ids with judge scores")

    # 3. Merge: attach rubric_scores to candidates
    judge_by_label = {}  # (prompt_id, label) -> rubric
    judge_hf = {}        # (prompt_id, label) -> is_hard_fail
    for pid, results in judge.items():
        for r in results:
            label = r.get("candidate_label", "")
            rubric = r.get("rubric_scores", {})
            is_hf = r.get("is_hard_fail", False)
            judge_by_label[(pid, label)] = rubric
            judge_hf[(pid, label)] = is_hf

    # 4. Score each candidate with v69 composite
    scored_with_judge = 0
    scored_regex_only = 0
    for pid, cands in candidates.items():
        for idx, c in enumerate(cands):
            action = count_action_blocks(c.get("response", ""))
            dialogue = count_dialogue_blocks(c.get("response", ""))
            
            label = chr(65 + idx) if idx < 26 else ""
            rubric = judge_by_label.get((pid, label), {})
            is_hf = judge_hf.get((pid, label), False)
            
            if rubric and not is_hf:
                c["rubric_scores"] = rubric
                c["v69_composite"] = v69_composite(rubric, action, dialogue)
                scored_with_judge += 1
            else:
                c["rubric_scores"] = {}
                c["v69_composite"] = v69_composite({}, action, dialogue)
                scored_regex_only += 1
            
            c["action_count"] = action
            c["dialogue_count"] = dialogue

    print(f"[score] {scored_with_judge} with judge, {scored_regex_only} regex-only")

    # 5. Select top-K per scene with diversity
    selected = []
    for pid, cands in candidates.items():
        picks = diversity_select(cands, a.top_k, a.diversity_ngram)
        selected.extend(picks)

    print(f"[select] {len(selected)} selected ({a.top_k} per scene)")

    # 6. Profile check (Step B hard gate)
    base_profile = {"action": 4.1, "dialogue": 1.7, "mono_pct": 35.5}
    
    print(f"\n{'='*60}")
    print(f"STEP B: Profile Hard Gate")
    print(f"{'='*60}")
    
    p = profile_check(selected, "CJK aggregate", base_profile)
    print(f"  N = {p['N']}")
    print(f"  Action:   {p['action']}  (gate >=4.5: {'PASS' if p['gates']['action_ge_4.5'] else 'FAIL'})")
    print(f"  Dialogue: {p['dialogue']}  (gate >=1.7: {'PASS' if p['gates']['dialogue_ge_base'] else 'FAIL'})")
    print(f"  Mono%:    {p['mono_pct']}%  (gate <=35%: {'PASS' if p['gates']['mono_le_35'] else 'FAIL'})")
    print(f"  Overall:  {'PASS' if p['pass'] else 'FAIL'}")
    
    print(f"\nPer-language:")
    for lang in ["zh", "zh-tw", "ja", "ko"]:
        lang_recs = [r for r in selected if r.get("language") == lang]
        if lang_recs:
            p = profile_check(lang_recs, lang, base_profile)
            print(f"  {lang}: N={p['N']} action={p['action']} dialogue={p['dialogue']} mono={p['mono_pct']}% {'PASS' if p['pass'] else 'FAIL'}")

    # 7. Write output
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for rec in selected:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    
    print(f"\n[wrote] {len(selected)} -> {a.out}")
    
    if not p["pass"]:
        print(f"\n⚠️  PROFILE GATE FAILED — do not proceed to training.")
        print(f"    Adjust composite weights or check selector logic.")
        sys.exit(1)
    else:
        print(f"\n✓ Profile gate passed — safe to proceed to Step C (mix building).")


if __name__ == "__main__":
    main()
