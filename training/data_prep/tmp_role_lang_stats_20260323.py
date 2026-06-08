import csv
import json
from pathlib import Path
from collections import Counter, defaultdict

from langdetect import DetectorFactory, detect_langs, LangDetectException

DetectorFactory.seed = 0

cleaned_path = Path('/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/runs/20260320_143241_chatjoy_top20_multilingual_10k_gemini.progress/cleaned.jsonl')
roles_dir = Path('/home/ps/hanjiaqi/LlamaFactory/chatjoy/TOP20CHARACTERS_systemprompt')
out_path = cleaned_path.parent / 'role_language_table.from_cleaned.csv'

system_to_role = {}
for fp in sorted(roles_dir.glob('*.json')):
    role_name = fp.stem
    data = json.loads(fp.read_text(encoding='utf-8'))
    desc = (data.get('description') or '').strip()
    if desc:
        system_to_role[desc] = role_name

role_lang = defaultdict(Counter)
role_total = Counter()
lang_total = Counter()
unknown_role = 0
unknown_lang = 0
all_rows = 0

with cleaned_path.open('r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        all_rows += 1
        try:
            obj = json.loads(line)
        except Exception:
            continue

        item = obj.get('item') or {}
        system_prompt = (item.get('system') or '').strip()
        role = system_to_role.get(system_prompt, 'UNKNOWN_ROLE')
        if role == 'UNKNOWN_ROLE':
            unknown_role += 1

        text = (item.get('instruction') or '').strip()
        lang = None
        if len(text) >= 2:
            try:
                preds = detect_langs(text)
                lang = preds[0].lang if preds else None
            except LangDetectException:
                lang = None
        if not lang:
            lang = 'unknown'
            unknown_lang += 1

        role_lang[role][lang] += 1
        role_total[role] += 1
        lang_total[lang] += 1

lang_cols = [k for k, _ in sorted(lang_total.items(), key=lambda x: (-x[1], x[0]))]
rows = sorted(role_total.items(), key=lambda x: (-x[1], x[0]))

with out_path.open('w', encoding='utf-8', newline='') as f:
    w = csv.writer(f)
    w.writerow(['role', 'total_replies', *lang_cols])
    for role, total in rows:
        w.writerow([role, total, *[role_lang[role].get(c, 0) for c in lang_cols]])

print(f'INPUT={cleaned_path}')
print(f'OUTPUT={out_path}')
print(f'TOTAL_ROWS={all_rows}')
print(f'ROLES={len(role_total)}')
print(f'LANGS={len(lang_cols)}')
print(f'UNKNOWN_ROLE={unknown_role}')
print(f'UNKNOWN_LANG={unknown_lang}')
print('TOP5_LANG=' + ', '.join([f'{k}:{v}' for k, v in sorted(lang_total.items(), key=lambda x: (-x[1], x[0]))[:5]]))
print('TOP5_ROLE=' + ', '.join([f'{k}:{v}' for k, v in rows[:5]]))
