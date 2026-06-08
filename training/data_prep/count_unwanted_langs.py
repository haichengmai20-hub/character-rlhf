import csv
from pathlib import Path

csv_file = Path("/home/ps/hanjiaqi/LlamaFactory/chatjoy/preapi_role_lang_dist.csv")
unwanted_langs = ['bn', 'cs', 'el', 'fi', 'he', 'hu', 'id', 'nl', 'no']

totals = {lang: 0 for lang in unwanted_langs}

with csv_file.open("r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        for lang in unwanted_langs:
            if lang in row:
                try:
                    totals[lang] += int(row[lang])
                except (ValueError, TypeError):
                    pass

print("不要的语言条数统计：")
print("=" * 60)
for lang in sorted(unwanted_langs):
    print(f"{lang}: {totals[lang]:6d} 条")
print("=" * 60)
total = sum(totals.values())
print(f"{'总计':>3}: {total:6d} 条")
