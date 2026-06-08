import csv
import json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from collections import Counter
import os
from pathlib import Path
from typing import Iterable, Optional

from langdetect import DetectorFactory, LangDetectException, detect_langs

# Make langdetect deterministic
DetectorFactory.seed = 0

try:
    import orjson
except Exception:
    orjson = None

# =========================
# User Config (edit here)
# =========================
USER_CONFIG = {
    # Input files (.jsonl line-delimited OR .json top-level array)
    "inputs": [
        "chatjoy_hq_full.json",
    ],
    # JSON field(s) containing text.
    # Supports single key "text" or multiple keys "instruction,output".
    "text_key": "instruction,output",
    # Skip too-short text (counted as unknown)
    "min_chars": 2,
    # 0 means process all lines
    "max_lines": 0,
    # Print progress every N processed lines (0 disables)
    "progress_every": 100000,
    # Parallel settings
    "workers": max(1, (os.cpu_count() or 8) // 2),
    "engine": "auto",  # auto | process | thread
    "chunksize": 256,
    # Shard settings: split full dataset into N shards and run one shard per process
    "num_shards": 1,
    "shard_index": 0,
    # Output CSV template. Supports {shard} placeholder.
    "out_csv_template": "language_stats_shard_{shard}.csv",
}


def apply_env_overrides(cfg: dict) -> dict:
    """Allow one-off overrides from environment variables."""
    out = dict(cfg)
    if os.getenv("LANGSTAT_INPUTS"):
        out["inputs"] = [x.strip() for x in os.getenv("LANGSTAT_INPUTS", "").split(",") if x.strip()]
    if os.getenv("LANGSTAT_TEXT_KEY"):
        out["text_key"] = os.getenv("LANGSTAT_TEXT_KEY")
    if os.getenv("LANGSTAT_ENGINE"):
        out["engine"] = os.getenv("LANGSTAT_ENGINE")
    if os.getenv("LANGSTAT_OUT_CSV_TEMPLATE"):
        out["out_csv_template"] = os.getenv("LANGSTAT_OUT_CSV_TEMPLATE")

    int_keys = {
        "LANGSTAT_MIN_CHARS": "min_chars",
        "LANGSTAT_MAX_LINES": "max_lines",
        "LANGSTAT_PROGRESS_EVERY": "progress_every",
        "LANGSTAT_WORKERS": "workers",
        "LANGSTAT_CHUNKSIZE": "chunksize",
        "LANGSTAT_NUM_SHARDS": "num_shards",
        "LANGSTAT_SHARD_INDEX": "shard_index",
    }
    for env_key, cfg_key in int_keys.items():
        val = os.getenv(env_key)
        if val is not None and val != "":
            out[cfg_key] = int(val)
    return out


def get_text_from_item(item: dict, text_key_cfg: str) -> str:
    """
    Build text from one or multiple fields.
    Examples:
      - "text"
      - "instruction,output"
    """
    keys = [k.strip() for k in str(text_key_cfg).split(",") if k.strip()]
    if not keys:
        return ""
    if len(keys) == 1:
        return str(item.get(keys[0], "")).strip()
    parts = []
    for k in keys:
        v = str(item.get(k, "")).strip()
        if v:
            parts.append(v)
    return "\n".join(parts).strip()


# langdetect commonly returns these ISO-639-1 style codes.
LANGUAGE_NAME_MAP = {
    "af": "Afrikaans",
    "ar": "Arabic",
    "bg": "Bulgarian",
    "bn": "Bengali",
    "ca": "Catalan",
    "cs": "Czech",
    "cy": "Welsh",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "et": "Estonian",
    "fa": "Persian",
    "fi": "Finnish",
    "fr": "French",
    "gu": "Gujarati",
    "he": "Hebrew",
    "hi": "Hindi",
    "hr": "Croatian",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "kn": "Kannada",
    "ko": "Korean",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mk": "Macedonian",
    "ml": "Malayalam",
    "mr": "Marathi",
    "ne": "Nepali",
    "nl": "Dutch",
    "no": "Norwegian",
    "pa": "Punjabi",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "so": "Somali",
    "sq": "Albanian",
    "sv": "Swedish",
    "sw": "Swahili",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "tl": "Tagalog",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "zh-cn": "Chinese (Simplified)",
    "zh-tw": "Chinese (Traditional)",
}

# Representative countries/regions for readability (not exhaustive).
LANGUAGE_REGION_MAP = {
    "ar": "Saudi Arabia, Egypt, UAE, etc.",
    "en": "US, UK, Canada, etc.",
    "es": "Spain, Mexico, Argentina, etc.",
    "pt": "Portugal, Brazil",
    "fr": "France, Belgium, Canada, etc.",
    "de": "Germany, Austria, Switzerland",
    "tr": "Turkey",
    "fa": "Iran, Afghanistan",
    "id": "Indonesia",
    "ru": "Russia",
    "uk": "Ukraine",
    "it": "Italy",
    "ja": "Japan",
    "ko": "South Korea",
    "nl": "Netherlands, Belgium",
    "pl": "Poland",
    "hi": "India",
    "ur": "Pakistan, India",
    "zh-cn": "China (Mainland)",
    "zh-tw": "Taiwan, Hong Kong",
}


def detect_language(text: str) -> Optional[str]:
    """Return the top language code from langdetect, or None if detection fails."""
    try:
        predictions = detect_langs(text)
    except LangDetectException:
        return None

    if not predictions:
        return None
    return predictions[0].lang


def detect_one_text(payload: tuple[str, int]) -> Optional[str]:
    text, min_chars = payload
    if len(text) < min_chars:
        return None
    return detect_language(text)


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                if orjson is not None:
                    yield orjson.loads(line)
                else:
                    yield json.loads(line)
            except Exception:
                continue


def iter_json_array(path: Path) -> Iterable[dict]:
    """
    Stream parse a top-level JSON array without loading the whole file into memory.
    """
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as f:
        buf = ""

        # Find first '['
        while True:
            chunk = f.read(65536)
            if not chunk:
                return
            buf += chunk
            stripped = buf.lstrip()
            if not stripped:
                buf = ""
                continue
            if stripped[0] != "[":
                raise ValueError(f"JSON 文件不是数组格式: {path}")
            # Keep content after '[' for downstream object parsing
            left_trim = len(buf) - len(stripped)
            buf = stripped[1:]
            break

        while True:
            # Skip commas and whitespace between elements
            while True:
                stripped = buf.lstrip()
                if not stripped:
                    chunk = f.read(65536)
                    if not chunk:
                        return
                    buf += chunk
                    continue
                if stripped[0] == ",":
                    buf = stripped[1:]
                    continue
                if stripped[0] == "]":
                    return
                buf = stripped
                break

            # Parse one JSON value (we only keep dict items)
            while True:
                try:
                    obj, idx = decoder.raw_decode(buf)
                    buf = buf[idx:]
                    if isinstance(obj, dict):
                        yield obj
                    break
                except json.JSONDecodeError:
                    chunk = f.read(65536)
                    if not chunk:
                        return
                    buf += chunk


def iter_records(path: Path) -> Iterable[dict]:
    if path.suffix.lower() == ".jsonl":
        yield from iter_jsonl(path)
        return
    if path.suffix.lower() == ".json":
        yield from iter_json_array(path)
        return
    raise ValueError(f"不支持的输入文件格式: {path}")


def main() -> None:
    cfg = apply_env_overrides(USER_CONFIG)
    if cfg["num_shards"] < 1:
        raise ValueError("--num-shards 必须 >= 1")
    if cfg["shard_index"] < 0 or cfg["shard_index"] >= cfg["num_shards"]:
        raise ValueError("--shard-index 必须在 [0, num_shards-1] 范围内")

    input_paths = [Path(p) for p in cfg["inputs"]]
    for p in input_paths:
        if not p.exists():
            raise FileNotFoundError(f"文件不存在: {p}")

    total = 0
    processed = 0
    unknown = 0
    counts = Counter()
    next_progress = cfg["progress_every"] if cfg["progress_every"] > 0 else 0
    global_line_no = 0
    actual_engine = "single"

    def iter_texts() -> Iterable[tuple[str, int]]:
        nonlocal total, global_line_no
        for path in input_paths:
            print(f"[INFO] Processing: {path}", flush=True)
            for item in iter_records(path):
                if global_line_no % cfg["num_shards"] != cfg["shard_index"]:
                    global_line_no += 1
                    continue
                global_line_no += 1
                if cfg["max_lines"] > 0 and total >= cfg["max_lines"]:
                    return
                total += 1
                text = get_text_from_item(item, cfg["text_key"])
                yield (text, cfg["min_chars"])

    text_stream = iter_texts()
    workers = max(1, cfg["workers"])
    if workers == 1:
        actual_engine = "single"
        for lang in map(detect_one_text, text_stream):
            processed += 1
            if lang is None:
                unknown += 1
            else:
                counts[lang] += 1

            if next_progress > 0 and processed >= next_progress:
                print(f"[INFO] Processed {processed:,} lines...", flush=True)
                next_progress += cfg["progress_every"]
    else:
        engine = cfg["engine"]
        if engine == "auto":
            engine = "process"

        def consume(executor) -> None:
            nonlocal processed, unknown, next_progress
            for lang in executor.map(detect_one_text, text_stream, chunksize=cfg["chunksize"]):
                processed += 1
                if lang is None:
                    unknown += 1
                else:
                    counts[lang] += 1
                if next_progress > 0 and processed >= next_progress:
                    print(f"[INFO] Processed {processed:,} lines...", flush=True)
                    next_progress += cfg["progress_every"]

        if engine == "process":
            try:
                with ProcessPoolExecutor(max_workers=workers) as executor:
                    actual_engine = "process"
                    consume(executor)
            except Exception as e:
                print(f"[WARN] ProcessPool 失败，自动降级到 ThreadPool: {e}", flush=True)
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    actual_engine = "thread(fallback)"
                    consume(executor)
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                actual_engine = "thread"
                consume(executor)

    detected_total = sum(counts.values())
    denominator = total if total > 0 else 1

    sorted_rows = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    out_csv_str = cfg["out_csv_template"].format(shard=cfg["shard_index"])

    print("\n=== Language Distribution ===")
    print(f"Shard: {cfg['shard_index'] + 1}/{cfg['num_shards']}")
    print(f"Workers: {workers} (engine={actual_engine})")
    print(f"Total lines: {total:,}")
    print(f"Detected lines: {detected_total:,}")
    print(f"Unknown lines: {unknown:,} ({unknown / denominator * 100:.4f}%)")
    print()
    print(f"{'lang':<8} {'language':<24} {'count':>12} {'ratio(%)':>10} {'major_regions'}")
    print("-" * 95)

    for code, cnt in sorted_rows:
        name = LANGUAGE_NAME_MAP.get(code, code)
        ratio = cnt / denominator * 100
        regions = LANGUAGE_REGION_MAP.get(code, "")
        print(f"{code:<8} {name:<24} {cnt:>12,} {ratio:>10.4f} {regions}")

    out_csv = Path(out_csv_str)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["lang_code", "language", "major_regions", "count", "ratio_percent"])
        for code, cnt in sorted_rows:
            writer.writerow([
                code,
                LANGUAGE_NAME_MAP.get(code, code),
                LANGUAGE_REGION_MAP.get(code, ""),
                cnt,
                f"{cnt / denominator * 100:.6f}",
            ])
        writer.writerow([
            "unknown",
            "Unknown/Undetected",
            "",
            unknown,
            f"{unknown / denominator * 100:.6f}",
        ])

    print(f"\n[INFO] Saved CSV: {out_csv.resolve()}")


if __name__ == "__main__":
    main()
