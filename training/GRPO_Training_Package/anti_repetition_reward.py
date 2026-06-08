"""
anti_repetition_reward.py — ChatJoy rollout 重复检测与轻惩罚
================================================================
背景：
  v6.3 训练保留 RM 为主信号，但需要一层 CPU-only 的保护，避免 actor 在
  GRPO 中学到机械复读、循环句式或无限铺陈同一短语。

本模块当前只提供纯 Python 逻辑：
  1. n-gram 重复检测，覆盖空格分词语言与 CJK/日/韩/泰等无空格片段。
  2. 白名单文件读取，用于放过歌词式 refrain、角色口头禅等业务允许重复。
  3. verl custom_reward_function 兼容入口 compute_reward。

RL 注册之后再接；本文件不改训练脚本。
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DEFAULT_NGRAM_SIZE = 4
DEFAULT_MIN_OCCURRENCES = 4
DEFAULT_REPEAT_RATIO_THRESHOLD = 0.18
MIN_REPETITION_PENALTY = -0.20
MAX_REPETITION_PENALTY = -0.80

ENV_WHITELIST_PATH = "ANTI_REPETITION_WHITELIST_PATH"

_TOKEN_RE = re.compile(
    r"[A-Za-z0-9À-ÖØ-öø-ÿ]+(?:['’][A-Za-z0-9À-ÖØ-öø-ÿ]+)?"
    r"|[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af\u0e00-\u0e7f]"
)


def _round4(value: float) -> float:
    return round(float(value), 4)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def normalize_phrase(text: str) -> str:
    return normalize_text(text).lower()


def default_whitelist_path() -> Path:
    return Path(__file__).with_name("anti_repetition_whitelist.txt")


def load_whitelist(path: str | Path | None = None) -> set[str]:
    """
    Load safe repeated phrases from a text or JSON file.

    Text format: one phrase per line, `#` comments allowed.
    JSON format: either a list of phrases or {"phrases": [...]}.
    Missing paths return an empty set, so rollout startup is not blocked.
    """
    selected = path
    if selected is None:
        selected = os.getenv(ENV_WHITELIST_PATH)
    if selected is None:
        candidate = default_whitelist_path()
        selected = candidate if candidate.exists() else None
    if selected is None:
        return set()

    file_path = Path(selected)
    if not file_path.exists():
        return set()

    if file_path.suffix.lower() == ".json":
        data = json.loads(file_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("phrases", [])
        phrases = [str(item) for item in data] if isinstance(data, list) else []
    else:
        phrases = []
        for line in file_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            phrases.append(stripped)

    return {phrase for phrase in (normalize_phrase(item) for item in phrases) if phrase}


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(str(text or ""))]


def _remove_whitelisted_phrases(text: str, whitelist: Iterable[str]) -> tuple[str, list[str]]:
    cleaned = normalize_text(text)
    hits: list[str] = []
    for phrase in sorted({normalize_phrase(item) for item in whitelist if normalize_phrase(item)}, key=len, reverse=True):
        pattern = re.compile(re.escape(phrase), flags=re.IGNORECASE)
        if pattern.search(cleaned):
            hits.append(phrase)
            cleaned = pattern.sub(" ", cleaned)
    return normalize_text(cleaned), hits


def _ngram_stats(tokens: list[str], ngram_size: int) -> dict[str, Any]:
    if ngram_size <= 0 or len(tokens) < max(ngram_size * 2, 8):
        return {
            "repeat_ratio": 0.0,
            "max_ngram_count": 0,
            "top_ngram": "",
            "repeated_instances": 0,
            "total_ngrams": 0,
        }

    ngrams = [tuple(tokens[index : index + ngram_size]) for index in range(len(tokens) - ngram_size + 1)]
    counts = Counter(ngrams)
    top_ngram, max_count = counts.most_common(1)[0]
    repeated_instances = sum(count - 1 for count in counts.values() if count > 1)
    return {
        "repeat_ratio": repeated_instances / max(1, len(ngrams)),
        "max_ngram_count": max_count,
        "top_ngram": " ".join(top_ngram),
        "repeated_instances": repeated_instances,
        "total_ngrams": len(ngrams),
    }


def detect_repetition(
    response: str,
    *,
    ngram_size: int = DEFAULT_NGRAM_SIZE,
    min_occurrences: int = DEFAULT_MIN_OCCURRENCES,
    repeat_ratio_threshold: float = DEFAULT_REPEAT_RATIO_THRESHOLD,
    whitelist: Iterable[str] | None = None,
) -> dict[str, Any]:
    """
    Return structured repetition metadata for one response.

    A response is repetitive only when the most common n-gram repeats enough
    times and the repeated n-gram mass is large enough. This avoids penalizing
    ordinary roleplay replies that naturally reuse names or pronouns.
    """
    raw = str(response or "")
    whitelist_phrases = set(whitelist) if whitelist is not None else load_whitelist()
    cleaned, whitelist_hits = _remove_whitelisted_phrases(raw, whitelist_phrases)
    tokens = tokenize(cleaned)
    stats = _ngram_stats(tokens, ngram_size)

    is_repetitive = (
        stats["max_ngram_count"] >= min_occurrences
        and stats["repeat_ratio"] >= repeat_ratio_threshold
    )
    reason = "ngram_repeat" if is_repetitive else ""

    return {
        "is_repetitive": is_repetitive,
        "reason": reason,
        "ngram_size": ngram_size,
        "token_count": len(tokens),
        "repeat_ratio": _round4(stats["repeat_ratio"]),
        "max_ngram_count": int(stats["max_ngram_count"]),
        "top_ngram": stats["top_ngram"],
        "repeated_instances": int(stats["repeated_instances"]),
        "total_ngrams": int(stats["total_ngrams"]),
        "whitelist_hits": whitelist_hits,
    }


def _penalty_for_detection(detection: dict[str, Any]) -> float:
    if not detection.get("is_repetitive"):
        return 0.0
    ratio = float(detection.get("repeat_ratio") or 0.0)
    penalty = MIN_REPETITION_PENALTY - ratio * 1.2
    return _round4(max(MAX_REPETITION_PENALTY, penalty))


def compute_reward(
    data_source: str,
    solution_str: str,
    ground_truth: dict | None = None,
    extra_info: dict | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    verl custom_reward_function compatible entry.

    Normal replies return 0.0 so RM remains the main signal. Repetitive replies
    receive a bounded negative score, with metadata for rollout diagnostics.
    """
    extra_info = extra_info or {}
    whitelist_path = (
        kwargs.get("whitelist_path")
        or extra_info.get("anti_repetition_whitelist_path")
        or extra_info.get("whitelist_path")
    )
    whitelist = load_whitelist(whitelist_path) if whitelist_path else load_whitelist()
    detection = detect_repetition(solution_str or "", whitelist=whitelist)
    penalty = _penalty_for_detection(detection)

    return {
        "score": penalty,
        "anti_repetition_penalty": penalty,
        "anti_repetition_is_repetitive": bool(detection["is_repetitive"]),
        "anti_repetition_reason": detection["reason"],
        "anti_repetition_ngram_size": detection["ngram_size"],
        "anti_repetition_repeat_ratio": detection["repeat_ratio"],
        "anti_repetition_max_ngram_count": detection["max_ngram_count"],
        "anti_repetition_top_ngram": detection["top_ngram"],
        "anti_repetition_token_count": detection["token_count"],
        "anti_repetition_whitelist_hits": detection["whitelist_hits"],
    }


def score_batch(responses: list[str], *, whitelist_path: str | Path | None = None) -> list[dict[str, Any]]:
    whitelist = load_whitelist(whitelist_path) if whitelist_path else load_whitelist()
    rows = []
    for response in responses:
        detection = detect_repetition(response, whitelist=whitelist)
        penalty = _penalty_for_detection(detection)
        rows.append(
            {
                "response_preview": str(response or "")[:120],
                "score": penalty,
                "is_repetitive": bool(detection["is_repetitive"]),
                "reason": detection["reason"],
                "repeat_ratio": detection["repeat_ratio"],
                "max_ngram_count": detection["max_ngram_count"],
                "top_ngram": detection["top_ngram"],
                "whitelist_hits": detection["whitelist_hits"],
            }
        )
    return rows


if __name__ == "__main__":
    samples = [
        '*She folds her arms.* "Try again."',
        "she looks at you with trembling hands " * 8,
        "她慢慢靠近窗边，雨声落在玻璃上。" * 8,
    ]
    for row in score_batch(samples):
        print(json.dumps(row, ensure_ascii=False))
