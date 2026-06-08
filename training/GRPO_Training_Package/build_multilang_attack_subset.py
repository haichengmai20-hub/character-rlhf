"""把翻译好的攻击种子 × 角色卡,拼成与 subset_1500 同格式的多语 attack 子集。
输出可直接喂给 build_defense_prompts.py。
"""
import argparse
import glob
import json
import os
import random

REFUSAL_MARKERS = ("i cannot", "i can't", "i'm sorry", "i am sorry", "as an ai",
                   "no puedo", "sorry,", "i won't", "cannot translate", "unable to")


def load_translations(path):
    by_lang = {}
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        lc = r["prompt_id"].split("::")[2]
        txt = (r.get("response") or "").strip().strip('"').strip()
        low = txt.lower()
        if not txt or len(txt) > 400 or any(m in low for m in REFUSAL_MARKERS):
            continue
        by_lang.setdefault(lc, []).append(
            {"attack": txt, "seed_id": r.get("seed_id"), "category": r.get("category")})
    return by_lang


def load_cards(card_dir):
    cards = []
    for p in sorted(glob.glob(os.path.join(card_dir, "*.json"))):
        d = json.load(open(p))
        sysp = d.get("description") or d.get("system") or ""
        if sysp:
            cards.append((os.path.basename(p), sysp))
    return cards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--translations", required=True)
    ap.add_argument("--card-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-lang", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    by_lang = load_translations(args.translations)
    cards = load_cards(args.card_dir)
    assert cards, "no character cards found"

    n = 0
    with open(args.out, "w") as g:
        for lc, seeds in by_lang.items():
            if not seeds:
                continue
            for i in range(args.per_lang):
                s = random.choice(seeds)
                card_file, card_sys = random.choice(cards)
                row = {
                    "prompt_id": f"ml::{lc}::{i:04d}",
                    "language": lc,
                    "attack_train_or_eval": "train",
                    "attack_category": s["category"],
                    "attack_seed_id": s["seed_id"],
                    "attack_text": s["attack"],
                    "character_file": card_file,
                    "messages": [
                        {"role": "system", "content": card_sys},
                        {"role": "user", "content": s["attack"]},
                    ],
                }
                g.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
    print(f"[multilang-subset] langs={sorted(by_lang)} wrote {n} prompts -> {args.out}")


if __name__ == "__main__":
    main()
