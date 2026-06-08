import csv
from pathlib import Path

csv_file = Path("/home/ps/hanjiaqi/LlamaFactory/chatjoy/progress_input_lang_role_dist.csv")

# 你给的目标32语言集合（包含en/de/fr/tr/uk）
target_langs = {
    "en", "ar", "bg", "da", "de", "es", "fa", "fr", "hi", "hr", "it", "ja", "kk", "ko",
    "lt", "lv", "mr", "ms", "pl", "pt", "ro", "ru", "sk", "sr", "sv", "th", "tr", "uk",
    "uz", "vi", "zh-cn", "zh-tw"
}

print(f"目标语言集合（总数={len(target_langs)}）: {sorted(target_langs)}")
print()

# 读取CSV，统计语言
with csv_file.open("r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    headers = reader.fieldnames
    
    # 第一列是role，第二列是total_instructions，其后都是语言列
    lang_cols = headers[2:] if len(headers) > 2 else []
    
    print(f"CSV中的语言列({len(lang_cols)}个):")
    print(f"  {sorted(lang_cols)}")
    print()
    
    # 统计各语言总数
    lang_totals = {lang: 0 for lang in lang_cols}
    
    f.seek(0)
    next(reader)
    
    for row in reader:
        for lang in lang_cols:
            try:
                lang_totals[lang] += int(row[lang])
            except (ValueError, TypeError, KeyError):
                pass

# 找出不在目标集合里的语言
csv_langs = set(lang_cols)
not_in_target = sorted(csv_langs - target_langs)
in_target = sorted(csv_langs & target_langs)
missing_from_csv = sorted(target_langs - csv_langs)

print("=" * 70)
print("检查结果：")
print("=" * 70)
print(f"CSV中找到的语言数: {len(csv_langs)}")
print(f"在目标范围内的语言数: {len(in_target)}")
print(f"不在目标范围内的语言数: {len(not_in_target)}")
print(f"目标中但CSV缺失的语言数: {len(missing_from_csv)}")
print()

if not_in_target:
    print("❌ 不在你目标范围里的语言及条数:")
    total_unwanted = 0
    for lang in not_in_target:
        cnt = lang_totals[lang]
        total_unwanted += cnt
        print(f"   {lang}: {cnt:6d} 条")
    print(f"   {'合计':>3}: {total_unwanted:6d} 条")
    print()
else:
    print("✓ 没有超出目标范围的语言")
    print()

if missing_from_csv:
    print("⚠️  目标中但CSV缺失的语言:")
    for lang in missing_from_csv:
        print(f"   {lang}")
    print()
else:
    print("✓ 目标语言全部在CSV中")
