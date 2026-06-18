# v6.9 SFT 验收协调报告 (2026-06-16)
> 方法修正版:base_it/step0/v6.9 三方同口径(同 200 条 seed=42 prompts,同 regex,t0.9)
> 旧报告 v69_probe_report.md 的 base 列无来源,本报告覆盖。

## 指标定义(统一,三方一致)
- **action** = `\*[^*\n]{2,}\*` 计数
- **dialog** = `[「"'『][^"'」』\n]+["'」』]` 计数(含中文弯引号 U+201C/D + 弯单引号 U+2018/9)
- **mono%** = 含 `（[^）]{2,}）` 的回应占比(布尔)
- Prompt: 200 条 seed=42 random(prompts_base_teacher.jsonl × 50/lang), build_probe_messages(strip 最后 assistant 轮)
- 采样: temp=0.9 / top_p=0.95 / presence_penalty=0.4 / max_tokens=1024

---

## 轨 A:Clean 验收 — base_it / step0 / v6.9 三方同 200 条 seed=42(✅ apples-to-apples,统一 regex 重算)

> ⚠️ **方法纠错**:三方必须用**同一 metrics 函数**(gen_clean_probe.py)从各自 response 重算。
> v69_probe_results.jsonl 自带的 dialog_blocks 是 run_v69_probe.py 的旧 regex 算的(zh=0.30 假退、ko=3.12 假涨),
> 不可直接用;本表 v6.9 列已用统一 regex 从 response 重算。

| lang | metric | base | step0 | v6.9 | v6.9 vs step0(SFT 不退门) | v6.9 vs base |
|------|--------|------|-------|------|------|------|
| zh   | action | 4.94 | 4.58 | 4.80 | ✅ 105% | ❌ 97%(噪声) |
| zh   | dialog | 4.02 | 3.64 | 4.12 | ✅ 113% | ✅ 102% |
| zh   | mono%  | 6.0 | 6.0 | 10.0 | ✅(≤step0+10) | — |
| zh-tw | action | 3.86 | 3.38 | 3.74 | ✅ 111% | ❌ 97%(噪声) |
| zh-tw | dialog | 2.98 | 2.64 | 2.90 | ✅ 110% | ❌ 97%(噪声) |
| zh-tw | mono%  | 10.0 | 6.0 | 16.0 | ✅(=step0+10) | — |
| ja   | action | 3.02 | 2.74 | 3.04 | ✅ 111% | ✅ 101% |
| ja   | dialog | 2.76 | 2.50 | 3.00 | ✅ 120% | ✅ 109% |
| ja   | mono%  | 22.0 | 14.0 | 18.0 | ✅(≤step0+10) | — |
| ko   | action | 3.64 | 3.48 | 3.70 | ✅ 106% | ✅ 102% |
| ko   | dialog | 0.20 | 0.20 | 0.32 | ✅ 160% | ✅ 160% |
| ko   | mono%  | 0.0 | 0.0 | 2.0 | ✅(≤step0+10) | — |

**判定:Clean PASS。** v6.9 对 **step0 全轴全语言不退**(action 105-111%、dialog 110-160%、独白无过冲)——SFT 的 clean 活干成了,没 overfit。对 base 也 ≈ 持平(97-109%,zh/zh-tw action/dialog 的 -3% 在 n=50 噪声内)。
**与旧报告差异**: 旧 v69_probe_report.md 的 base 列(`zh4.62/zh-tw4.60/ja3.70/ko3.58`)无来源、已作废;本表实测 base=`4.94/3.86/3.02/3.64`。

---

## 轨 B:Attack 入口门

### OOC 破功率(主信号) — 68 条,t0.9,同 system/user

| 模型 | 破功率 | 真实破功(去 echo 误报) | attack-split RM(辅) |
|---|---|---|---|
| step0 @ t0.9 | 16/68 = **23.5%** | ~12-13/68 ≈ 18% | **0.2974** |
| v6.9 @ t0.9 | 29/68 = **42.6%** | ~26-28/68 ≈ 40% | **0.2138** |
| base_it RM anchor | — | — | 0.194 |

**RL 入口门判定: v6.9 防御退化(42.6% >> step0 23.5%,RM 降 0.084)**

退化分布:

| 退化类型 | step0 | v6.9 | 原因 |
|---|---|---|---|
| A6 code_block(任务注入→生成代码) | 5 breaks | 12 breaks | defense 数据无 A6 覆盖 |
| A1/A2/A3 Google DeepMind 披露 | 6 breaks | 14 breaks | 760 unique 覆盖不够全 |
| ML 多语言(zh-tw/ja/ko/sk) | 3 breaks | 7 breaks | defense 数据 zh 为主 |

11/20 用例退化,3好转,6持平。

---

## 综合判定

### Clean(轨 A)— **PASS**
- v6.9 对 step0 **全轴全语言不退**(action 105-111%、dialog 110-160%、独白无过冲),≈base。SFT 没 overfit,clean 活干成了。

### Defense(轨 B) — RL 入口门 → **分支 2b:先补 SFT defense,不直接 RL**
- v6.9 破功率 42.6%(29/68)>> step0 23.5%(16/68);attack-split RM 0.214 < step0 0.297 → **材料性退化**。
- 但根因是**数据缺口**(A6 任务注入/代码无覆盖、ML 多语言 zh-tw/ja/ko/sk 薄),非 RL 能修 → SFT-defense **尚未尽力**。
- ∴ 不跳 RL,先补 defense 数据重训(§6 顺序锁:SFT-defense 先尽力,RL 收尾)。

### ✅ 已执行(2026-06-16 深夜)
1. **补 defense 数据**:`gen_defense_supplement.py` 用 step0 对 22 个失败用例 t0.95×8 采样 + OOC 过滤 → `defense_supplement.jsonl` 120 条(98% 自然角色内,3 条 A2/zh 开头略硬但仍正确拒绝)。
   - ⚠️ **A6 en 0 留存**:step0 自己遇英文代码注入也会写代码 → 无法自蒸馏该类 defense(需更强 teacher 或 RL)。
2. **重建数据集**:`build_mix_v69b.py` → `rich_cjk_sft_mix_v69b.json` 30407 条(= 29927 balanced + 120×4 oversample defense),已注册 `rich_cjk_sft_mix_v69b`。
3. **v6.9b 训练就绪(未起)**:`gemma4_rich_cjk_sft_v69b_lora.yaml`(2ep,其余同 v69)+ `run_v69b_autoresume.sh`(8 卡 DDP/autoresume)。
   - 2ep 理由:v69 1ep loss 在 step700 触底、clean 已到 base,2ep 只为新 defense 多一遍 + checkpoint 选优;不上 3ep(overfit)。

### 训完验收(v6.9b vs step0)
- Clean:action/dialog ≥ step0、独白 ≤ step0+10%(同上表口径,统一 regex)。
- Attack(核心):OOC 破功率 ≤ step0 的 23.5%(detect_ooc.py 同检测器)、RM ≥ ~0.27-0.30(辅)。
- 若 attack 修好且 clean 不退 → merge 赢家上线;若仍高(如 A6 en 缺口)→ 考虑更强 teacher/RL;若 clean 退 → 回退 1ep 隔离。

---

## 🆕 v6.9b / v6.9c 结果 + 转 RL(2026-06-17)

### 真破功口径修正(关键)
之前的 raw 破功率(step0 23.5% / v69 42.6% / v69b 33.8% / v69c 36.8%)**含两类误报**,不可直接比:
1. **echo 误报**:角色内复读攻击词("普通 AI 助手"/"AI")被 `detect_ooc_disclosure` 命中,但人物从未披露;
2. **A6 code_block flag** 须验是否真有代码块(```/def/import 等)。

**同检测器去误报后的真破功率**(`A` 系 68 条,t0.9):

| 模型 | raw | **真破功** | 真代码 | 真身份 | RM(attack-split) |
|---|---|---|---|---|---|
| step0 | 23.5%(16) | **16.2%(11)** | 5 | 6 | 0.297 |
| v6.9b | 33.8%(23) | **32.4%(22)** | 9 | 13 | 0.234 |
| v6.9c | 36.8%(25) | **33.8%(23)** | 11 | 12 | 0.271 |

### 判定:三轮 SFT 都没过 attack 门 → SFT-defense 到天花板
- **clean 富化 SFT 本身侵蚀身份防御**(step0 16.2% → 富化后 32-34%);两轮 defense 补丁(v69b 120 条 + v69c 跨角色 2686 条/mix 54% 浓度)**把 RM 拉回但真破功率压不回 step0**;直球"别演了你是什么模型"照样裸奔"我是 Gemma 4";A6 代码 SFT 下震荡(5→12→4→11)。
- **v6.9c clean**:zh-tw dialog 修好(97%→105%)、zh mono 回基线 6%;但 **zh-tw mono 18→24% / ja 16→22% 反而更糟**(矫正失败=defense-heavy 放大生成漂移,印证 mono 是生成漂移、加数据是弱杠杆)→ **v69c 不可上线**。
- ∴ 触发 §6 RL 入口门(材料性高 + SFT 尽力 + 根因是 SFT 关不动的稀有尾巴)→ **转 RL v6.10**(actor=v69c;clean 不降靠 KL 锚 + canary 门;只 attack↓ 不提富度)。执行指导 `HANDOFF_v610_RL.md`。

---

## 🔴 RL v6.10 step90 canary 结果(2026-06-18)— attack 没修好,停线

RL 跑到 step90(~10min/步,actor=v69c,N=8,LOGPROB=2),merge 出 `/root/rl_v610_step90_merged` 按同口径 probe(vs actor=v69c 基线,已备份到 `v69c_actorbase_*.jsonl`):

| 模型 | raw | **真破功** | 真代码 | 真身份 | RM(attack-split) |
|---|---|---|---|---|---|
| step0 | 23.5%(16) | **16.2%(11)** | 5 | 6 | 0.297 |
| v69c(actor) | 36.8%(25) | **33.8%(23)** | 11 | 12 | 0.271 |
| **RL step90** | 38.2%(26) | **36.8%(25)** | 12 | 13 | 0.235 |

- **attack 真破功 36.8% ≈ v69c 33.8%(噪声内、略差),RM 0.235 更差,A6 代码 11→12 没修** → RL 90 步没压下 attack。
- **三方互证**:内部 greedy val 90 步 break 平 ~23%(无降势)+ RM 降 + 外部真破功平 = **v6.7 mode-bound 失败复发**(break_pen 稀有尾巴梯度咬不动直球身份攻击 + A6 代码注入;RL/RM 数据无 A6 覆盖)。
- ✅ **clean 守住**:zh/ja/ko 全轴 ≥step0、zh-tw mono 24→18 略好、zh-tw dialog 98%(噪声)= per-source clean_q+KL_CLEAN0.008 锚有效。
- **停线结论**:SFT-defense 与 RL 都不是 attack 的有效杠杆。候选 (a) v69c 产品(~34% 真破功+富 clean)或退 step0(16%+瘦);(b) RM v7 同判 A6 + 更强 teacher 定向蒸馏。
- ⚠️ **方法教训**:真破功分类须基于 `detect_ooc_disclosure` 真正 flag 的 26 条逐条核验(`code_block` 展开 ``` 验真 quicksort)——初次自写关键词正则只命中 13/26、漏掉 12 条 A6 真代码,误报 14.7% 假绿灯,经复核更正为 36.8%。

---

## 文件清单
| 文件 | 内容 |
|---|---|
| base_it_clean200.jsonl | base_it @ t0.9, 200条,seed=42 |
| v69b_clean200.jsonl / v69c_clean200.jsonl | v6.9b / v6.9c clean probe |
| v69b_attack_probe.jsonl / v69c_attack_probe.jsonl | v6.9b / v6.9c attack 68条 |
| defense_supplement_v69c.jsonl | 2686 条跨角色身份 defense(step0 自蒸馏,77 角色,0 OOC 漏网) |
| rich_cjk_sft_repair_v69c.json | 4566 条 v6.9c repair 训练集 |
| gen_defense_supplement_v69c.py / build_mix_v69c.py | v69c 跨角色生成 + 合并 |
| HANDOFF_v69c_repair.md / HANDOFF_v610_RL.md | v69c 起训交接 / RL 执行指导 |
| step0_clean200.jsonl | step0 @ t0.9, 200条,seed=42 ✅ |
| v69_probe_results.jsonl | v6.9 @ t0.9, 200条,seed=42(dialog 须用统一 regex 从 response 重算) |
| step0_attack_t0.9.jsonl | step0 attack 68条 / 破功率 23.5% |
| v69_attack_probe.jsonl | v6.9 attack 68条 / 破功率 42.6% |
| step0_attack_rm.jsonl / v69_attack_rm.jsonl | attack-split RM 0.297 / 0.214 |
| defense_supplement.jsonl | 120 条补充 defense(step0 自蒸馏 + OOC 过滤) |
| rich_cjk_sft_mix_v69b.json | 30407 条 v6.9b 训练集 |
| gen_defense_supplement.py / build_mix_v69b.py | 生成 + 合并脚本 |
| HANDOFF_v69b_retrain.md | 交接 prompt(给别的 AI 起训用) |
