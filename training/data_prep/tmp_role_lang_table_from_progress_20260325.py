import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PROGRESS_PATH = BASE_DIR / "chatjoy_top20_multilingual_27x2200_preapi_gemini.progress.jsonl"
CLEANED_PATH = BASE_DIR / "data/runs/20260325_103758_chatjoy_top20_multilingual_27x2200_preapi_gemini.progress.first22331/cleaned.jsonl"
TOP20_DIR = BASE_DIR / "TOP20CHARACTERS_systemprompt"
OUTPUT_CSV_PROGRESS = BASE_DIR / "22331_temp_gemini_v2.csv"
OUTPUT_CSV_CLEANED = BASE_DIR / "data/runs/20260325_103758_chatjoy_top20_multilingual_27x2200_preapi_gemini.progress.first22331/12430_v2.csv"
CHAR_FILENAME_PATTERN = re.compile(r"^\d+-(\d+)")


def load_role_name_map(top20_dir: Path) -> dict[str, str]:
    role_map: dict[str, str] = {}
    for fp in sorted(top20_dir.glob("*.json")):
        stem = fp.stem
        match = CHAR_FILENAME_PATTERN.match(stem)
        if not match:
            continue
        role_map[match.group(1)] = stem
    return role_map


def iter_items(progress_path: Path, max_lines=None):
    count = 0
    with progress_path.open("r", encoding="utf-8") as f:
        for line in f:
            if max_lines is not None and count >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            item = rec.get("item", rec)
            if isinstance(item, dict):
                yield item
                count += 1


def main() -> None:
    if not PROGRESS_PATH.exists():
        raise FileNotFoundError(f"progress file not found: {PROGRESS_PATH}")
    if not CLEANED_PATH.exists():
        raise FileNotFoundError(f"cleaned file not found: {CLEANED_PATH}")
    if not TOP20_DIR.exists():
        raise FileNotFoundError(f"top20 dir not found: {TOP20_DIR}")

    role_name_map = load_role_name_map(TOP20_DIR)

    # Process progress file (first 22331 lines)
    print("[1/2] Processing progress file (first 22331 lines)...")
    role_counts_p = Counter()
    lang_counts_p = Counter()
    matrix_p = defaultdict(Counter)
    total_p = 0
    unknown_role_p = 0
    unknown_lang_p = 0

    for item in iter_items(PROGRESS_PATH, max_lines=22331):
        total_p += 1
        char_id = str(item.get("character_id", "")).strip()
        lang = str(item.get("language", "")).strip() or "unknown"

        role = role_name_map.get(char_id, f"UNKNOWN_{char_id or 'EMPTY'}")
        if role.startswith("UNKNOWN_"):
            unknown_role_p += 1
        if lang == "unknown":
            unknown_lang_p += 1

        role_counts_p[role] += 1
        lang_counts_p[lang] += 1
        matrix_p[role][lang] += 1

    langs_p = [k for k, _ in lang_counts_p.most_common()]
    roles_p = [k for k, _ in role_counts_p.most_common()]

    with OUTPUT_CSV_PROGRESS.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["role", "total_replies", *langs_p])
        for role in roles_p:
            w.writerow([role, role_counts_p[role], *[matrix_p[role].get(l, 0) for l in langs_p]])

    print(f"✓ progress={PROGRESS_PATH}")
    print(f"  output={OUTPUT_CSV_PROGRESS}")
    print(f"  rows={len(roles_p)}, languages={len(langs_p)}, total_replies={total_p}")
    print(f"  unknown_role={unknown_role_p}, unknown_lang={unknown_lang_p}")
    print()

    # Process cleaned file
    print("[2/2] Processing cleaned file...")
    role_counts_c = Counter()
    lang_counts_c = Counter()
    matrix_c = defaultdict(Counter)
    total_c = 0
    unknown_role_c = 0
    unknown_lang_c = 0

    with CLEANED_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            item = rec.get("item", rec)
            if not isinstance(item, dict):
                continue
            
            total_c += 1
            char_id = str(item.get("character_id", "")).strip()
            lang = str(item.get("language", "")).strip() or "unknown"

            role = role_name_map.get(char_id, f"UNKNOWN_{char_id or 'EMPTY'}")
            if role.startswith("UNKNOWN_"):
                unknown_role_c += 1
            if lang == "unknown":
                unknown_lang_c += 1

            role_counts_c[role] += 1
            lang_counts_c[lang] += 1
            matrix_c[role][lang] += 1

    langs_c = [k for k, _ in lang_counts_c.most_common()]
    roles_c = [k for k, _ in role_counts_c.most_common()]

    with OUTPUT_CSV_CLEANED.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["role", "total_replies", *langs_c])
        for role in roles_c:
            w.writerow([role, role_counts_c[role], *[matrix_c[role].get(l, 0) for l in langs_c]])

    print(f"✓ cleaned={CLEANED_PATH}")
    print(f"  output={OUTPUT_CSV_CLEANED}")
    print(f"  rows={len(roles_c)}, languages={len(langs_c)}, total_replies={total_c}")
    print(f"  unknown_role={unknown_role_c}, unknown_lang={unknown_lang_c}")
    print()
    
    # Summary
    print("[SUMMARY]")
    print(f"Progress (原始)    : {total_p:6d} 条, {len(roles_p):2d} 角色, {len(langs_p):2d} 语言")
    print(f"Cleaned (清理后)  : {total_c:6d} 条, {len(roles_c):2d} 角色, {len(langs_c):2d} 语言")
    print(f"过滤率 (dropped+uncertain): {(total_p - total_c) / total_p * 100:.2f}%")


if __name__ == "__main__":
    main()
