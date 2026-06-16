#!/usr/bin/env python3
"""v68 SFT GATE validation: check rich subset meets three-axis targets."""
import json, re
from collections import Counter

with open('/root/rich_cjk_sft_v68/rich_cjk_sft_mix_v68_final.json') as f:
    data = json.load(f)

# Rich = has non-empty meta (built from candidates); Replay = empty meta
rich = [r for r in data if r.get('meta') and r['meta'].get('prompt_source')]
replay = [r for r in data if not r.get('meta') or not r['meta'].get('prompt_source')]
print(f'Rich: {len(rich)}, Replay: {len(replay)}')

action_re = re.compile(r'\*[^*\n]{2,}\*')
dialogue_re = re.compile(r'「[^」]{3,}」|\"[^\"]{3,}\"|「[^」\n]{3,}')
english_tail_re = re.compile(r'\n\s*Thoughts?:\s')

actions, dialogues, monologues, english_tails = [], [], [], []
langs = []

for r in rich:
    resp = r.get('output','')
    if not resp:
        continue
    acts = action_re.findall(resp)
    diags = dialogue_re.findall(resp)

    # Monologue detection
    has_cjk_mono = False
    cjk_paren = re.findall(r'[(\（][^\)\）]{5,}[\)\）]', resp)
    if any(re.search(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]', m) for m in cjk_paren):
        has_cjk_mono = True
    if re.search(r'内心|心想|속으로|마음속|心の中|생각하|独白|思った|思う|考える', resp):
        has_cjk_mono = True

    actions.append(len(acts))
    dialogues.append(len(diags))
    monologues.append(has_cjk_mono)
    english_tails.append(bool(english_tail_re.search(resp)))
    langs.append(r['meta'].get('language','?'))

n = len(actions)
print(f'\n=== 验收 GATE (rich 子集, n={n}) ===')
a_mean = sum(actions)/n
d_mean = sum(dialogues)/n
m_pct = sum(monologues)/n
t_pct = sum(english_tails)/n
print(f'动作块 mean:         {a_mean:.2f}  目标≥4.3  {"PASS" if a_mean >= 4.3 else "FAIL"}')
print(f'对话块 mean:         {d_mean:.2f}  目标≥3.0  {"PASS" if d_mean >= 3.0 else "FAIL"}')
print(f'独白率:              {m_pct*100:.1f}%   目标≥50%  {"PASS" if m_pct >= 0.5 else "FAIL"}')
print(f'英文Thoughts尾巴率:  {t_pct*100:.1f}%   目标≤8%   {"PASS" if t_pct <= 0.08 else "FAIL"}')
print(f'\nby_language: {dict(Counter(langs))}')

all_pass = all([a_mean >= 4.3, d_mean >= 3.0, m_pct >= 0.5, t_pct <= 0.08])
print(f'\n总判定: {"PASS - 可进训练" if all_pass else "FAIL - 有GATE未通过，停下报告"}')
