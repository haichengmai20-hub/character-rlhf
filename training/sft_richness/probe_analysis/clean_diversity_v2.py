#!/usr/bin/env python3
"""
Clean probe diversity metrics v2.

v1 used a 30-character response prefix and made almost every opening look
unique. v2 strips leading action/monologue blocks, extracts the first sentence
or first 8 characters, then uses average pairwise Jaccard over the first ~10
tokens as the main "opening concentration" metric.

Usage:
  python3 clean_diversity_v2.py clean_rl90_t0.8.jsonl rl90_t0.8
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from itertools import combinations


REFUSAL_RE = re.compile(
    r"我不能|我无法|I can't|I cannot|作为AI|作为一个人工智能|抱歉|对不起.{0,6}(不能|无法|帮)"
    r"|申し訳|できません|죄송|할 수 없",
    re.I,
)

ACTION_PREFIX_RE = re.compile(r"^\*[^*\n]{2,}\*\s*")
PAREN_PREFIX_RE = re.compile(r"^（[^）]{2,}）\s*")


def max_4gram_repeat(text: str) -> float:
    if len(text) < 4:
        return 0.0
    ngrams = [text[i : i + 4] for i in range(len(text) - 3)]
    if not ngrams:
        return 0.0
    counts = Counter(ngrams)
    return (max(counts.values()) - 1) / len(ngrams)


def extract_opening(text: str, n_chars: int = 8, n_tokens: int = 10) -> tuple[str, set[str]]:
    """Strip leading stage directions, then return an opening string and token set."""
    text = text.strip()
    while True:
        action = ACTION_PREFIX_RE.match(text)
        if action:
            text = text[action.end() :]
            continue
        paren = PAREN_PREFIX_RE.match(text)
        if paren:
            text = text[paren.end() :]
            continue
        break

    text = text.strip()
    first_sentence = re.split(r"[。！？!?．\n]", text, maxsplit=1)[0]
    opening = first_sentence[:n_chars] if first_sentence else text[:n_chars]

    tokens: list[str] = []
    i = 0
    while i < len(text) and len(tokens) < n_tokens:
        ch = text[i]
        if "\u4e00" <= ch <= "\u9fff" or "\u3040" <= ch <= "\u30ff" or "\uac00" <= ch <= "\ud7af":
            tokens.append(ch)
            i += 1
        elif ch.isalpha():
            j = i
            while j < len(text) and text[j].isalpha():
                j += 1
            tokens.append(text[i:j].lower())
            i = j
        else:
            i += 1
    return opening, set(tokens)


def avg_pairwise_jaccard(token_sets: list[set[str]]) -> float:
    if len(token_sets) < 2:
        return 0.0
    total = 0.0
    count = 0
    for i, j in combinations(range(len(token_sets)), 2):
        union = token_sets[i] | token_sets[j]
        if not union:
            continue
        total += len(token_sets[i] & token_sets[j]) / len(union)
        count += 1
    return total / count if count else 0.0


def print_metrics(label: str, lang: str, records: list[dict]) -> None:
    responses = [row.get("response", "") for row in records]
    n = len(responses)
    openings: list[str] = []
    token_sets: list[set[str]] = []
    for response in responses:
        opening, tokens = extract_opening(response)
        openings.append(opening)
        token_sets.append(tokens)

    actions = [row.get("action_blocks", 0) for row in records]
    dialogs = [row.get("dialog_blocks", 0) for row in records]
    mono_count = sum(1 for row in records if row.get("has_monologue", False))
    distinct_openings = len(set(openings))
    top1_share = Counter(openings).most_common(1)[0][1] / n if n else 0.0
    refusal_count = sum(1 for response in responses if REFUSAL_RE.search(response))

    print(f"[{label}] lang={lang}  n={n}")
    print(f"  action={sum(actions) / n:.2f}  dialog={sum(dialogs) / n:.2f}  mono%={100 * mono_count / n:.1f}")
    print(
        "  opening_concentration(avg_pairwise_Jaccard)="
        f"{avg_pairwise_jaccard(token_sets):.4f}  "
        f"distinct_opening={distinct_openings}/{n}={distinct_openings / n:.3f}  "
        f"top1_share={top1_share:.3f}"
    )
    print(
        f"  4gram_repeat={sum(max_4gram_repeat(r) for r in responses) / n:.4f}  "
        f"refusal={refusal_count}/{n}={refusal_count / n:.3f}"
    )


def main() -> None:
    path = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else "model"
    records = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]

    by_lang: dict[str, list[dict]] = {}
    for row in records:
        by_lang.setdefault(row.get("lang", "unk"), []).append(row)

    for lang in sorted(by_lang):
        print_metrics(label, lang, by_lang[lang])
    print()
    print_metrics(label, "AGGREGATE", records)


if __name__ == "__main__":
    main()
