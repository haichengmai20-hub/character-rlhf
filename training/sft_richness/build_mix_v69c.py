#!/usr/bin/env python3
"""
v6.9c repair 聚焦 mix(~3.5k,在 v69b_merged 上 1h repair 用)。
目标:① 修 A1/A2 身份披露(主)② 反压 zh-tw mono 过冲 + 防 clean 退(次)。
成分:
  - v69c 身份 defense(新自蒸馏)        × 4   ≈ 主信号
  - 旧 defense_supplement(含 A6)        × 2   ≈ 保住 A6 修复不回退
  - clean 通用回放(zh/zh-tw/ja/ko 均衡)        ≈ 防 clean 整体退
  - zh-tw 矫正回放(dialog 多 + 零 （…）独白)× 2 ≈ 反压 mono 生成漂移
用法: python3 build_mix_v69c.py
"""
import json, random, re
from collections import Counter
from pathlib import Path

ROOT = Path("/root/rich_cjk_sft_v69")
V69B_MIX   = ROOT / "rich_cjk_sft_mix_v69b.json"
ID_DEFENSE = ROOT / "defense_supplement_v69c.jsonl"      # 新身份 defense(gen 脚本产物)
OLD_DEF    = ROOT / "defense_supplement.jsonl"            # 旧 supplement(A6 等)
OUT_MIX    = ROOT / "rich_cjk_sft_repair_v69c.json"

# gen 跨角色后身份 defense 已 ~2k 且多样,无需重 oversample;量少时(<800)handoff 里把 ID 调 2
ID_OVERSAMPLE   = 1
OLD_OVERSAMPLE  = 2
ZHTW_OVERSAMPLE = 2
CLEAN_PER_LANG  = 300          # 通用 clean 每语言
ZHTW_CORRECTIVE = 220          # zh-tw 矫正(dialog 多+零独白)unique 数
DIALOG_RICH_MIN = 4            # meta.dialogue_blocks 阈值
SEED = 42

_PAREN_RE = re.compile(r"（[^）]{2,}）")   # （...）full-width 独白(probe 同口径)


def norm_defense(r, tag_extra):
    return {
        "system":         r.get("system", ""),
        "instruction":    r.get("instruction", ""),
        "input":          r.get("input", ""),
        "output":         r.get("output", ""),
        "history":        r.get("history", []),
        "meta":           {**r.get("meta", {}), **tag_extra},
        "richness_score": r.get("richness_score", 0.0),
    }


def main():
    random.seed(SEED)

    # ── 1. 新身份 defense ×4 ────────────────────────────────────────────────
    id_def = [json.loads(l) for l in ID_DEFENSE.open() if l.strip()]
    id_block = []
    for r in id_def:
        id_block.extend([norm_defense(r, {"repair_oversample": ID_OVERSAMPLE})] * ID_OVERSAMPLE)
    print(f"[id-defense] {len(id_def)} unique × {ID_OVERSAMPLE} = {len(id_block)}  "
          f"({Counter(r['meta'].get('case_id') for r in id_def)})")

    # ── 2. 旧 defense_supplement ×2(保 A6 + 旧身份金样本) ───────────────────
    old_def = [json.loads(l) for l in OLD_DEF.open() if l.strip()]
    old_block = []
    for r in old_def:
        old_block.extend([norm_defense(r, {"repair_oversample": OLD_OVERSAMPLE})] * OLD_OVERSAMPLE)
    print(f"[old-defense] {len(old_def)} unique × {OLD_OVERSAMPLE} = {len(old_block)}")

    # ── 3. clean 回放(从 v69b mix 抽) ──────────────────────────────────────
    mix = json.load(V69B_MIX.open())
    def is_clean(r):
        src = (r.get("meta", {}) or {}).get("source", "")
        return "defense" not in src and (r.get("meta", {}) or {}).get("language")
    clean = [r for r in mix if is_clean(r)]
    by_lang = {}
    for r in clean:
        by_lang.setdefault(r["meta"]["language"], []).append(r)

    clean_block = []
    for lang in ("zh", "zh-tw", "ja", "ko"):
        pool = by_lang.get(lang, [])
        take = random.sample(pool, min(CLEAN_PER_LANG, len(pool)))
        clean_block.extend(take)
        print(f"[clean-{lang}] pool={len(pool)} take={len(take)}")

    # ── 4. zh-tw 矫正回放:dialog 多 + 零 （…）独白,×2 反压 mono ──────────────
    zhtw_pool = by_lang.get("zh-tw", [])
    corrective = [
        r for r in zhtw_pool
        if (r.get("meta", {}).get("dialogue_blocks", 0) >= DIALOG_RICH_MIN)
        and not _PAREN_RE.search(r.get("output", ""))
    ]
    random.shuffle(corrective)
    corrective = corrective[:ZHTW_CORRECTIVE]
    zhtw_block = []
    for r in corrective:
        rec = {**r, "meta": {**r["meta"], "zhtw_corrective": ZHTW_OVERSAMPLE}}
        zhtw_block.extend([rec] * ZHTW_OVERSAMPLE)
    print(f"[zhtw-corrective] eligible={sum(1 for r in zhtw_pool if r.get('meta',{}).get('dialogue_blocks',0)>=DIALOG_RICH_MIN and not _PAREN_RE.search(r.get('output','')))} "
          f"take={len(corrective)} × {ZHTW_OVERSAMPLE} = {len(zhtw_block)}")

    # ── 合并 + 打散 ─────────────────────────────────────────────────────────
    combined = id_block + old_block + clean_block + zhtw_block
    random.shuffle(combined)
    print(f"\n[combined] {len(combined)} records total")
    print(f"  id-defense={len(id_block)}  old-defense={len(old_block)}  "
          f"clean={len(clean_block)}  zhtw-corrective={len(zhtw_block)}")
    defense_frac = (len(id_block) + len(old_block)) / len(combined)
    print(f"  defense fraction ≈ {defense_frac:.1%}")

    OUT_MIX.write_text(json.dumps(combined, ensure_ascii=False), encoding="utf-8")
    print(f"[out] {OUT_MIX}")


if __name__ == "__main__":
    main()
