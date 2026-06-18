#!/usr/bin/env python3
"""
在 balanced mix v69 基础上追加 defense supplement 数据,生成 v69b。
用法: python3 build_mix_v69b.py
"""
import json
from pathlib import Path
from collections import Counter

BASE_MIX   = Path("/root/rich_cjk_sft_v69/rich_cjk_sft_mix_v69_balanced.json")
SUPPLEMENT = Path("/root/rich_cjk_sft_v69/defense_supplement.jsonl")
OUT_MIX    = Path("/root/rich_cjk_sft_v69/rich_cjk_sft_mix_v69b.json")
OUT_DIR    = Path("/root/rich_cjk_sft_v69")

def main():
    base = json.load(BASE_MIX.open())
    print(f"[base] {len(base)} records")

    supp = [json.loads(l) for l in SUPPLEMENT.open() if l.strip()]
    print(f"[supplement] {len(supp)} defense records")

    OVERSAMPLE = 4
    # normalize supplement records to match base format, oversample 4×
    norm = []
    for r in supp:
        rec = {
            "system":         r.get("system", ""),
            "instruction":    r.get("instruction", ""),
            "input":          r.get("input", ""),
            "output":         r.get("output", ""),
            "history":        r.get("history", []),
            "meta":           {**r.get("meta", {}), "defense_oversample": OVERSAMPLE},
            "richness_score": r.get("richness_score", 0.0),
        }
        norm.extend([rec] * OVERSAMPLE)

    combined = base + norm
    print(f"[combined] {len(combined)} records total")

    # stats
    sources = Counter(r.get("meta", {}).get("source", "?") for r in combined)
    print("top sources:", sources.most_common(5))
    defense_all = [r for r in combined if r.get("meta", {}).get("defense_oversample")]
    print(f"defense records: {len(defense_all)}")

    OUT_MIX.write_text(json.dumps(combined, ensure_ascii=False, indent=None), encoding="utf-8")
    print(f"[out] {OUT_MIX}")

    # breakdown of supplement by case_id
    supp_cases = Counter(r.get("meta", {}).get("case_id", "?") for r in norm)
    print("\nsupplement breakdown:")
    for c, n in sorted(supp_cases.items()):
        print(f"  {c}: {n}")


if __name__ == "__main__":
    main()
