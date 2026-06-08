import json
import csv
from collections import Counter, defaultdict
from pathlib import Path

progress_file = Path("/home/ps/hanjiaqi/LlamaFactory/chatjoy/chatjoy_top20_multilingual_27x2200_preapi_gemini.progress.jsonl")
top20 = Path("/home/ps/hanjiaqi/LlamaFactory/chatjoy/TOP20CHARACTERS_systemprompt")
out_csv = Path("/home/ps/hanjiaqi/LlamaFactory/chatjoy/progress_input_lang_role_dist.csv")
out_summary = Path("/home/ps/hanjiaqi/LlamaFactory/chatjoy/progress_input_lang_summary.csv")

# 加载角色映射
role_map = {}
for fp in sorted(top20.glob("*.json")):
    stem = fp.stem
    parts = stem.split("-")
    if len(parts) >= 2 and parts[1].isdigit():
        role_map[parts[1]] = stem

role_counter = Counter()
lang_counter = Counter()
matrix = defaultdict(Counter)
total_records = 0

# 流式读取 progress.jsonl
with progress_file.open("r", encoding="utf-8") as f:
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
        
        # 以 instruction 对应的输入语言为准
        char_id = str(item.get("character_id", "")).strip()
        lang = str(item.get("language", "")).strip().lower()
        
        if lang == "zh":
            lang = "zh-cn"
        if not lang:
            lang = "unknown"
        
        role = role_map.get(char_id, f"UNKNOWN_{char_id or 'EMPTY'}")
        
        role_counter[role] += 1
        lang_counter[lang] += 1
        matrix[role][lang] += 1
        total_records += 1

# 导出角色-语言分布表
langs = [k for k, _ in lang_counter.most_common()]
roles = [k for k, _ in role_counter.most_common()]

with out_csv.open("w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["role", "total_instructions", *langs])
    for role in roles:
        w.writerow([role, role_counter[role], *[matrix[role].get(l, 0) for l in langs]])

# 导出语言汇总
total_lang = sum(lang_counter.values())
with out_summary.open("w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["language", "count", "percentage"])
    for lang, cnt in lang_counter.most_common():
        pct = cnt / total_lang * 100 if total_lang > 0 else 0
        w.writerow([lang, cnt, f"{pct:.2f}%"])

print("✓ Analysis complete")
print(f"  Total records: {total_records}")
print(f"  Total roles: {len(roles)}")
print(f"  Total languages: {len(lang_counter)}")
print(f"  Distribution CSV: {out_csv}")
print(f"  Summary CSV: {out_summary}")
print()
print("Top 10 languages by count:")
for lang, cnt in lang_counter.most_common(10):
    pct = cnt / total_lang * 100
    print(f"  {lang}: {cnt:6d} ({pct:6.2f}%)")
