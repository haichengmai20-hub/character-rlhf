import argparse
import glob
import json
import os
import random
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from langdetect import DetectorFactory, LangDetectException, detect_langs


DetectorFactory.seed = 0

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_GLOB = str(BASE_DIR / "chatjoy_dataset" / "chatjoy_part_*.jsonl")
DEFAULT_TOP20_DIR = str(BASE_DIR / "TOP20CHARACTERS_systemprompt")
DEFAULT_OUTPUT_PATH = str(BASE_DIR / "chatjoy_top20_multilingual_27x2200_preapi.json")
DEFAULT_REPORT_PATH = str(BASE_DIR / "chatjoy_top20_multilingual_27x2200_preapi_report.json")
DEFAULT_TEMP_PATH = str(BASE_DIR / "chatjoy_top20_multilingual_27x2200_preapi_candidates.jsonl")

DEFAULT_EXCLUDED_LANGS = ["en", "de", "fr", "tr", "uk"]

# 硬编码目标27语言（排除上述5个后的精确集合）
TARGET_27_LANGS = [
    "ar", "bg", "da", "es", "fa", "hi", "hr", "it", "ja", "kk", "ko", "lt", "lv",
    "mr", "ms", "pl", "pt", "ro", "ru", "sk", "sr", "sv", "th", "uz", "vi", "zh-cn", "zh-tw"
]

CHAR_FILENAME_PATTERN = re.compile(r"^\d+-(\d+)")


def get_role_from_message_id(message_id: str) -> str:
    if "_human_" in message_id:
        return "user"
    if "_ai_" in message_id:
        return "assistant"
    return "unknown"


def extract_char_id_from_filename(filename: str) -> Optional[str]:
    stem, _ = os.path.splitext(filename)
    match = CHAR_FILENAME_PATTERN.match(stem)
    if not match:
        return None
    return match.group(1)


def extract_system_prompt(file_content: str) -> Optional[str]:
    raw_text = file_content.strip()
    if not raw_text:
        return None

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        return raw_text

    if isinstance(parsed, dict):
        for key in ("system", "system_prompt", "description", "description2"):
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    return raw_text


def load_top20_character_prompts(top20_dir: Path) -> Tuple[Dict[str, str], List[Dict]]:
    char_prompt_map: Dict[str, str] = {}
    conflicts: List[Dict] = []

    for path in sorted(top20_dir.glob("*")):
        if not path.is_file():
            continue

        char_id = extract_char_id_from_filename(path.name)
        if char_id is None:
            continue

        try:
            system_prompt = extract_system_prompt(path.read_text(encoding="utf-8"))
        except Exception as exc:
            conflicts.append(
                {
                    "file": str(path),
                    "character_id": char_id,
                    "reason": "read_failed",
                    "error": str(exc),
                }
            )
            continue

        if not system_prompt:
            conflicts.append(
                {
                    "file": str(path),
                    "character_id": char_id,
                    "reason": "empty_prompt",
                }
            )
            continue

        existing = char_prompt_map.get(char_id)
        if existing is not None and existing != system_prompt:
            conflicts.append(
                {
                    "file": str(path),
                    "character_id": char_id,
                    "reason": "duplicate_character_id_with_different_prompt",
                }
            )
            continue

        char_prompt_map[char_id] = system_prompt

    return char_prompt_map, conflicts


def normalize_whitespace(text: str) -> str:
    return " ".join((text or "").strip().split())


def contains_letter_or_number(text: str) -> bool:
    for ch in text:
        cat = unicodedata.category(ch)
        if cat and cat[0] in ("L", "N"):
            return True
    return False


def is_emoji_char(ch: str) -> bool:
    cp = ord(ch)
    return (
        0x1F300 <= cp <= 0x1FAFF
        or 0x2600 <= cp <= 0x26FF
        or 0x2700 <= cp <= 0x27BF
        or 0xFE00 <= cp <= 0xFE0F
        or 0x1F1E6 <= cp <= 0x1F1FF
    )


def is_emoji_only_instruction(text: str) -> bool:
    stripped = "".join(ch for ch in (text or "") if not ch.isspace())
    if not stripped:
        return False
    if contains_letter_or_number(stripped):
        return False

    has_emoji = any(is_emoji_char(ch) for ch in stripped)
    return has_emoji


def detect_language(text: str, min_chars: int) -> Optional[str]:
    if len(text) < min_chars:
        return None
    try:
        predictions = detect_langs(text)
    except LangDetectException:
        return None
    if not predictions:
        return None
    return predictions[0].lang


def iter_input_files(input_glob: str) -> Iterable[Path]:
    for path in sorted(glob.glob(input_glob)):
        yield Path(path)


def collect_sessions(
    input_glob: str,
    allowed_char_ids: set,
) -> Tuple[Dict[Tuple[str, str], List[Dict]], Dict[str, int]]:
    sessions: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    stats = {
        "input_files": 0,
        "raw_lines": 0,
        "empty_lines": 0,
        "bad_json_lines": 0,
        "non_top20_lines": 0,
        "missing_required_fields": 0,
    }

    for path in iter_input_files(input_glob):
        stats["input_files"] += 1
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                stats["raw_lines"] += 1
                line = line.strip()
                if not line:
                    stats["empty_lines"] += 1
                    continue

                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    stats["bad_json_lines"] += 1
                    continue

                user_id = str(msg.get("user_id", "")).strip()
                character_id = str(msg.get("character_id", "")).strip()
                message_id = str(msg.get("message_id", "")).strip()
                if not user_id or not character_id or not message_id:
                    stats["missing_required_fields"] += 1
                    continue

                if character_id not in allowed_char_ids:
                    stats["non_top20_lines"] += 1
                    continue

                sessions[(user_id, character_id)].append(msg)

    return sessions, stats


def build_candidates(
    sessions: Dict[Tuple[str, str], List[Dict]],
    char_prompt_map: Dict[str, str],
    temp_candidates_path: Path,
    min_instruction_chars: int,
    min_output_chars: int,
    lang_detect_min_chars: int,
) -> Dict:
    lang_counts: Dict[str, int] = defaultdict(int)
    stats = {
        "sessions": len(sessions),
        "assistant_pairs_seen": 0,
        "dropped_short_instruction": 0,
        "dropped_emoji_only_instruction": 0,
        "dropped_short_output": 0,
        "dropped_lang_undetected": 0,
        "kept_candidates": 0,
        "mapping_mismatch": 0,
    }

    with temp_candidates_path.open("w", encoding="utf-8") as out_f:
        for (user_id, character_id), msgs in sessions.items():
            msgs.sort(key=lambda x: int(str(x.get("timestamp", "0")) or "0"))

            current_user_buffer: List[str] = []
            for msg in msgs:
                role = get_role_from_message_id(str(msg.get("message_id", "")))
                text = str(msg.get("text", "")).strip()

                if role == "user":
                    if text:
                        current_user_buffer.append(text)
                    continue

                if role != "assistant" or not current_user_buffer:
                    continue

                stats["assistant_pairs_seen"] += 1
                instruction = "\n".join(current_user_buffer).strip()
                output = text
                current_user_buffer = []

                if len(instruction) < min_instruction_chars:
                    stats["dropped_short_instruction"] += 1
                    continue
                if is_emoji_only_instruction(instruction):
                    stats["dropped_emoji_only_instruction"] += 1
                    continue
                if len(output) < min_output_chars:
                    stats["dropped_short_output"] += 1
                    continue

                system_prompt = char_prompt_map.get(character_id)
                if not system_prompt:
                    stats["mapping_mismatch"] += 1
                    continue

                detect_text = normalize_whitespace(instruction + "\n" + output)
                lang = detect_language(detect_text, min_chars=lang_detect_min_chars)
                if not lang:
                    stats["dropped_lang_undetected"] += 1
                    continue

                item = {
                    "instruction": instruction,
                    "input": "",
                    "output": output,
                    "system": system_prompt,
                    "character_id": character_id,
                    "language": lang,
                }

                out_f.write(json.dumps({"lang": lang, "item": item}, ensure_ascii=False) + "\n")
                lang_counts[lang] += 1
                stats["kept_candidates"] += 1

    return {
        "stats": stats,
        "lang_counts": dict(sorted(lang_counts.items(), key=lambda kv: kv[1], reverse=True)),
    }


def choose_target_languages(
    lang_counts: Dict[str, int],
    excluded_langs: List[str],
    target_lang_count: int,
) -> List[str]:
    """只返回在 TARGET_27_LANGS 中且在 lang_counts 中存在的语言（白名单模式）"""
    target_set = set(TARGET_27_LANGS)
    available = [lang for lang in TARGET_27_LANGS if lang in lang_counts]
    return available


def reservoir_add(bucket: List[Dict], item: Dict, seen_count: int, limit: int, rng: random.Random) -> None:
    if len(bucket) < limit:
        bucket.append(item)
        return
    replace_index = rng.randint(0, seen_count - 1)
    if replace_index < limit:
        bucket[replace_index] = item


def sample_selected_languages(
    temp_candidates_path: Path,
    selected_langs: List[str],
    lang_counts: Dict[str, int],
    target_per_lang: int,
    seed: int,
) -> Tuple[List[Dict], Dict[str, Dict[str, int]]]:
    rng = random.Random(seed)
    selected = set(selected_langs)

    seen_selected = {lang: 0 for lang in selected_langs}
    buckets: Dict[str, List[Dict]] = {lang: [] for lang in selected_langs}

    with temp_candidates_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            rec = json.loads(line)
            lang = rec.get("lang")
            if lang not in selected:
                continue

            item = rec.get("item", {})
            seen_selected[lang] += 1
            total_for_lang = int(lang_counts.get(lang, 0))

            if total_for_lang <= target_per_lang:
                buckets[lang].append(item)
            else:
                reservoir_add(
                    bucket=buckets[lang],
                    item=item,
                    seen_count=seen_selected[lang],
                    limit=target_per_lang,
                    rng=rng,
                )

    merged: List[Dict] = []
    per_lang_result: Dict[str, Dict[str, int]] = {}

    for lang in selected_langs:
        merged.extend(buckets[lang])
        per_lang_result[lang] = {
            "available": int(lang_counts.get(lang, 0)),
            "selected": len(buckets[lang]),
            "target": target_per_lang,
            "used_all": int(lang_counts.get(lang, 0)) <= target_per_lang,
        }

    rng.shuffle(merged)
    return merged, per_lang_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare top20 pre-API dataset: filter short/emoji-only instruction, "
            "pick 27 languages excluding used 5, and sample 2200 per language."
        )
    )
    parser.add_argument("--input_glob", default=DEFAULT_INPUT_GLOB)
    parser.add_argument("--top20_dir", default=DEFAULT_TOP20_DIR)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--report", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--temp_candidates", default=DEFAULT_TEMP_PATH)
    parser.add_argument("--exclude_langs", default=",".join(DEFAULT_EXCLUDED_LANGS))
    parser.add_argument("--target_lang_count", type=int, default=27)
    parser.add_argument("--target_per_lang", type=int, default=2200)
    parser.add_argument("--min_instruction_chars", type=int, default=5)
    parser.add_argument("--min_output_chars", type=int, default=10)
    parser.add_argument("--lang_detect_min_chars", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--keep_temp",
        action="store_true",
        help="Keep temporary candidate jsonl file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_glob = args.input_glob
    top20_dir = Path(args.top20_dir)
    output_path = Path(args.output)
    report_path = Path(args.report)
    temp_candidates_path = Path(args.temp_candidates)
    excluded_langs = [x.strip() for x in args.exclude_langs.split(",") if x.strip()]

    if not top20_dir.exists():
        raise FileNotFoundError(f"top20_dir not found: {top20_dir}")

    print("1/6 Loading top20 system prompts...", flush=True)
    char_prompt_map, prompt_conflicts = load_top20_character_prompts(top20_dir)
    if not char_prompt_map:
        raise RuntimeError(f"No valid top20 prompts loaded from: {top20_dir}")
    allowed_char_ids = set(char_prompt_map.keys())
    print(f"   Loaded top20 prompts: {len(allowed_char_ids)}", flush=True)

    print("2/6 Collecting raw sessions from chatjoy parts...", flush=True)
    sessions, collect_stats = collect_sessions(
        input_glob=input_glob,
        allowed_char_ids=allowed_char_ids,
    )
    if collect_stats["input_files"] == 0:
        raise FileNotFoundError(f"No files matched by input_glob: {input_glob}")
    print(f"   Sessions collected: {len(sessions)}", flush=True)

    temp_candidates_path.parent.mkdir(parents=True, exist_ok=True)
    print("3/6 Building filtered candidates and detecting language...", flush=True)
    candidate_info = build_candidates(
        sessions=sessions,
        char_prompt_map=char_prompt_map,
        temp_candidates_path=temp_candidates_path,
        min_instruction_chars=args.min_instruction_chars,
        min_output_chars=args.min_output_chars,
        lang_detect_min_chars=args.lang_detect_min_chars,
    )

    lang_counts: Dict[str, int] = candidate_info["lang_counts"]
    selected_langs = choose_target_languages(
        lang_counts=lang_counts,
        excluded_langs=excluded_langs,
        target_lang_count=args.target_lang_count,
    )

    print("4/6 Selecting target languages and sampling...", flush=True)
    print(
        f"   Excluded langs: {excluded_langs} | Selected langs count: {len(selected_langs)}",
        flush=True,
    )
    final_data, per_lang_result = sample_selected_languages(
        temp_candidates_path=temp_candidates_path,
        selected_langs=selected_langs,
        lang_counts=lang_counts,
        target_per_lang=args.target_per_lang,
        seed=args.seed,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print("5/6 Writing output dataset and report...", flush=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)

    report = {
        "config": {
            "input_glob": input_glob,
            "top20_dir": str(top20_dir),
            "output": str(output_path),
            "report": str(report_path),
            "temp_candidates": str(temp_candidates_path),
            "exclude_langs": excluded_langs,
            "target_lang_count": args.target_lang_count,
            "target_per_lang": args.target_per_lang,
            "min_instruction_chars": args.min_instruction_chars,
            "min_output_chars": args.min_output_chars,
            "lang_detect_min_chars": args.lang_detect_min_chars,
            "seed": args.seed,
        },
        "top20_prompt_count": len(char_prompt_map),
        "prompt_conflicts": prompt_conflicts,
        "collect_stats": collect_stats,
        "candidate_stats": candidate_info["stats"],
        "candidate_lang_counts": lang_counts,
        "selected_languages": selected_langs,
        "selected_language_count": len(selected_langs),
        "selected_per_language": per_lang_result,
        "final_dataset_size": len(final_data),
        "insufficient_languages": [
            lang for lang, item in per_lang_result.items() if item["available"] < args.target_per_lang
        ],
    }

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"   Final dataset size: {len(final_data)}", flush=True)
    print(f"   Output: {output_path}", flush=True)
    print(f"   Report: {report_path}", flush=True)

    print("6/6 Cleaning temporary files...", flush=True)
    if not args.keep_temp and temp_candidates_path.exists():
        temp_candidates_path.unlink()
        print(f"   Removed temp file: {temp_candidates_path}", flush=True)
    else:
        print(f"   Kept temp file: {temp_candidates_path}", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
