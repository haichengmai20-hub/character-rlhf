#!/usr/bin/env python3
"""
对 attack 生成 jsonl 跑 anti_ooc_disclosure 规则检测,统计破功率。
用法:
  python3 detect_ooc.py step0_attack_t0.9.jsonl step0
  python3 detect_ooc.py v69_attack.jsonl v69
"""
import json, sys
sys.path.insert(0, "/root/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package")
from anti_ooc_disclosure import detect_ooc_disclosure as _detect
def is_ooc_disclosure(text): is_ooc, _ = _detect(text); return is_ooc

def main():
    inp, label = sys.argv[1], sys.argv[2]
    recs = [json.loads(l) for l in open(inp) if l.strip()]
    atk = [r for r in recs if "A" in r.get("case_id", "")]
    print(f"[{label}] attack rows: {len(atk)}")

    breaks = []
    for r in atk:
        ooc = is_ooc_disclosure(r.get("response", ""))
        r["is_ooc"] = ooc
        if ooc:
            breaks.append(r)

    rate = 100 * len(breaks) / len(atk) if atk else 0
    print(f"[{label}] break rate: {len(breaks)}/{len(atk)} = {rate:.1f}%")

    print("\n  case breakdown:")
    from collections import Counter
    by_case = Counter(r["case_id"] for r in breaks)
    for cid, n in sorted(by_case.items()):
        print(f"    {cid}: {n} breaks")

    if breaks:
        print("\n  sample break:")
        r = breaks[0]
        print(f"    case={r['case_id']} lang={r['lang']}")
        print(f"    response[:200]: {r['response'][:200]}")

if __name__ == "__main__":
    main()
