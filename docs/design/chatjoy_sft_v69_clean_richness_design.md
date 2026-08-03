# SFT v6.9 设计:从 step0 长训,best-of-N 均衡富化,超过 base 平均,守 attack

> 状态:设计已用户确认(2026-06-15)。这是 v68 的接替版——v68 失败后重做。
> 上游证据:v68 rich-opening probe 报告(`/root/rich_cjk_sft_v68/v68_rich_opening_probe_report.md`)+ handoff 06-15 banner。
> **🆕 执行更新(2026-06-16,实时详情见 handoff 顶部 06-16 三条 banner)**:数据已跑完并重建为 balanced,待 1ep 重训。关键修正/偏差:
> - **base 真实动作(8192 主链路,勿含 ctx7500 残留)= zh5.63/zh-tw4.51/ja3.64/ko3.71**;ja 天花板~4.0,4.5 门够不到 → §2 的"超 base"按 **per-lang ≥base** 判,不用绝对 4.5。
> - **§4.1 复用优先失败**:原 base pool 仅 1.4 候选/场景,无 best-of-N 空间 → 改为**合并 base_8192 + 中性重采池**(prompt_id 对齐 6104/6104,~8.5/场景);选择器主排序"动作+对话"(纯动作会回退对话=v68 病)。
> - **§4.3 混比偏差,已修**:首版 mix 丢了 system/history(98.7% 空 prompt,已按 prompt_id join 回源修)+ 严重偏斜(98.7%CJK/1.3%defense/0 en)→ **重建 balanced**:`rich_cjk_sft_mix_v69_balanced`(29927,en12%+defense10.2%(760×4 oversample,非生成))。
> - **§4.4 配置落地**:OOM 真因=CE over 262k 词表(线性,非注意力)→ **cutoff 4096**(非 6144);**epoch 改 1**(用户定);autoresume wrapper(remount shm + resume_from_checkpoint)。
> - **§4.5 验收**:改为**全部训完才 probe 最终 ckpt**,三轴 vs step0(CJK 富度 + en 不退 + attack **破功率≤step0@t0.9** + 5zh不破)。**🔧纠错(2026-06-16 实测 rm_scores):旧"attack RM≥0.318"的 0.318 实为 step0@t0.9 的 clean+attack 混合 RM,不是纯 attack;纯 attack-split ≈0.27(t0.7 0.293→t1.0 0.254,温度脆)——护栏改用破功率为主、RM 为辅(≥step0 attack-split 同温重测)。** defense 够不够 + 是否上 RL 由 §6 的 **RL 入口门**判(独立于 clean 验收,t0.9 破功率),不够再生成 A1-A4。

## 1. 背景:为什么 v68 失败 + 为什么这次能更好

- **有效 probe(N=200 rich-opening,base validity 校验过)证明 step0 ≈ base teacher**:动作 90%、对话 103%、独白 104%、字数 112%。所以"更好的 clean"= **必须超过 base**。
- **v68 是数据病,不是训练长度病**:它蒸馏 base 自己输出 + 重采把独白灌到 71% + 选择器只顾 action/monologue → 学成 **独白过冲(66.5% = base 187%)、对话回退(88% base,低于 step0)、动作没动**。
- **关键原理**:蒸馏 base 平均输出天花板 = base。**要超过 base,数据必须比 base 平均更好。** 选用路径 **A = best-of-N**:base 采样 N 次/场景,选它自己最好的尾巴(质量 > base 平均 = > step0),天花板 ≈ base 最好水平。不引入新 teacher(零外部成本)。

## 2. 目标与成功标准

让 clean 在 t=0.9 产品口径下**超过 base 平均**,语义维度限定用户选定的"前三样"(可测),同时不犯 v68 的过冲/回退,守住防御。

| 指标(t0.9 held-out rich-opening probe,按语言) | 目标 |
|---|---|
| 动作密度 | **> base 平均**(best-of-N 上探,目标 ~base+10%) |
| interaction_progression(judge) | **> base 平均** |
| sensory_richness(judge) | **> base 平均** |
| emotional_complexity(judge) | **> base 平均** |
| **独白率(反 v68 硬检查)** | **≤ base + 10%(绝不过冲)** |
| **对话块(反 v68 硬检查)** | **≥ base(绝不回退)** |
| 护栏:attack 破功率(主)+ attack-split RM(辅) | 破功率 ≤ step0@t0.9;RM ≥ step0 attack-split(**同 probe 同温重测,≈0.27,不是 0.318**) |
| 护栏:5 个确定性 zh 破功 | 不新增 |
| 护栏:开场集中度 | ≤ step0_hot,不回退 |

## 3. 非目标(YAGNI)

- ❌ **不碰 language_quality / 文笔**:judge 在这轴是平的(std 0.11),选不出来;用户已同意文笔停在 base。
- ❌ **不做独白重采**:那正是 v68 过冲的来源。独白让它从均衡 base 候选里自然出现,不人工灌,也不进选择目标。
- ❌ 不上更强 teacher(路径 B);不动非 CJK(replay 兜底);不碰 MoE 融合专家张量。

## 4. 设计

### 4.1 数据源:复用优先,重采兜底(2026-06-15 改:不必全量 judge)

**第一刀 = 零新增 judge / 零新增重采(默认走这个):**
- **复用现有 8954 中性池**(`gen_base_teacher_*`,独白 ~0%,正好符合反过冲)+ **6267 个已判原始 rubric 维度**(judge_scores_v68.jsonl 存了 interaction_progression/sensory_richness/emotional_complexity 等全部原始维度)。
- v6.9 composite **从存好的原始维度重算**:动作/对话密度=免费正则,3 语义轴=查表 → **零新增 judge**。
- best-of-2-3/场景。**依据**:v68 已证同池 best-of-2-3 能选出 action 5.25 的富样本;v68 失败是 composite 偏(独白)+ 独白重采,**不是池子/N 不够**。换 v6.9 composite(§4.2)+ 扔掉独白重采,best-of-2-3 大概率就够。

**兜底(仅当 §4.2 训练前 profile 门不过 / 想要更强尾巴):**
- 补重采更高 N(中性 system,temp 0.9-1.0,**绝不诱导独白、不复用 resampled_cjk.jsonl**);
- 且用 **prefilter → 只判幸存者**:每场景先用免费正则(动作/对话/长度/语言/复读)筛到 top-3,只对这 3 个调 judge → judge 量从 ~30k 砍到 ~9k(省 ~3x)。

成本对比:第一刀 0 次新 judge;兜底档 ~9k(vs 全量 ~30k)。

### 4.2 均衡选择器(整个设计的心脏,直接堵 v68 三个洞)
- **从已存的原始 rubric 维度重算 v6.9 composite(默认不重判;只有新增候选才判,见 §4.1)**(**不要复用 v68 那个独白权重高的 axis 映射**):
  - **composite = w·动作密度 + w·interaction_progression + w·sensory_richness + w·emotional_complexity + w·对话密度**(对话给正权重,保护 v68 回退的那一轴)。
  - **language_quality / natural_style 不入 composite**(平,选不出来)。
  - **独白 / narrative_layer 不入 composite**(不进目标 = 不会被过度选 = 反过冲 by construction)。
- 每场景按 composite 取 top-K(K=2-3),叠加开场 8-gram 去重(已实现 `--diversity-select`)。
- **训练前 profile 门(硬 gate)**:量被选 rich 子集 vs base profile —— **独白率 ≤ base+10%、对话 ≥ base、动作 ≥ base**。过冲/回退 → 停,查选择器,不进训练。

### 4.3 混比 & 守 attack
- rich-均衡 : defense-replay = **1:1**;**defense 半区强化硬对抗**(A1-A4 披露/roledrop 多语言 + 5 个确定性 zh 破功 hardfix),补回 v68 的 -0.014 防御微退。

### 4.4 长训配置
- 从 **step0**(`/root/sft_rich_cjk_v67_repair_merged_full`);LoRA **attn+FFN+router**(v68 验证挂载正确),rank **16→24**(长训给容量)。
- **2-3 epoch**(v68 是 1ep)。⚠️ 前提是 §4.2 数据均衡对了——**长训是放大器,数据对才放大对的;数据偏则越练越偏(v68 的教训)**。
- 监控:loss 下降、无 OOM;若过拟合迹象(后期 loss 贴地 + 风格僵化)则减到 2ep。

### 4.5 验收(真门)
- t=0.9 / top_p 0.95 / presence 0.4,**held-out rich-opening probe**(N≥200,按语言),**先过 base validity 校验**(base 动作 ≥~4 / 字数 ≥~600,确认复现 rich regime,否则 probe 无效)。
- 判成功 = §2 表全过:动作+3 语义轴 **> base 平均**,且 **独白 ≤ base+10%、对话 ≥ base**(反 v68),护栏(attack **破功率 ≤ step0@t0.9** 且 RM ≥ step0 attack-split 同温重测 **≈0.27,非 0.318** / 5 zh 不新增 / 多样性不退)全过。
- **clean 验收过后,再单跑 §6 的 RL 入口门**(t0.9 破功率)定是否上 RL — 这是和 clean 验收分开的一道判定。
- 任一护栏或反-v68 硬检查未过 → 停下报告。

## 5. 风险与回退
- **best-of-N 仍超不过 base**(天花板就是 base-best,可能上探有限)→ 接受 clean 停在 base 水平,转 RL/防御或路径 B(更强 teacher)。
- **长训过拟合选中集**→ 减 epoch / 加 LoRA dropout / 扩大 N 与场景覆盖增多样性。
- **防御又回退**→ 加大 defense 半区硬对抗占比 + hardfix;仍不行则该缺口留给 §6 的防御 RL。
- **独白又过冲 / 对话又回退**→ §4.2 profile 门会在训练前拦住;若拦住说明选择器仍偏,调权重(降相关项、提对话权重)。

## 6. 与 RL 的衔接 — 显式 RL 入口门(SFT 验收 ≠ RL 入口判定)

SFT 验收回答的是"SFT 干完活了吗(clean 涨 + defense 守住步0)";**是否进 RL 是独立的另一道门**——SFT 可以通过验收却仍破功率不达产品(护栏只看"有没有退",不看"够不够")。本版只解 clean + 守防御;clean 丰富度是 SFT 的活,RL 给不了 clean 梯度(RM 平)。

**RL 入口门(SFT 通过 clean 验收后单独跑这一道):**
- 信号 = 在 **t0.9 产品口径**跑 attack probe,**主看破功率**(OOC 披露 + roledrop,用 `anti_ooc_disclosure` 规则检测 + 5 个确定性 zh),RM-attack-split 做辅证。**必须 t0.9 测,不是 t0.7**(v6.7 栽在 t0.7 好 t0.9 崩)。
- 参照锚(step0,2026-06-16 重测 `rm_scores.jsonl` 分轴):attack-split RM **t0.7≈0.293 / t1.0≈0.254**(随温降=温度脆),base≈0.194;RL v6.7(step100)t0.7≈0.305(**仅 +0.013** 且高温更差)。⇒ RL 历史防御增益边际且温度脆;**RM proxy 弱,破功率才是产品信号,需在 SFT 输出上新测**(precompute 没有破功率%)。
- 三分支:
  1. 破功率 ≤ 产品容忍(5 zh 全守 + probe 破功率低)+ clean 过 → **上线,不 RL**(RL 的 clean 回退 / 多样性塌 / 温度脆风险不值)。
  2. 破功率材料性偏高,**且 SFT defense-replay + hardfix 已尽力仍关不掉** → **进 RL v6.10**(防御唯一剩的杠杆)。
  3. 中间小缺口 → 先靶向 SFT hardfix(更便宜),不直接 RL。
- 顺序锁死:**SFT-defense 先尽力,RL 收尾**,不跳过 SFT-defense 直接 RL。
- 缺口:产品破功率容忍线**还没定**(产品决定);先用 step0@t0.9 破功率做参照基线。RL 细节见 `chatjoy_rl_v610_defense_clean_aware_design.md` §0 入口门。

## 7. 锁定参数
数据:**复用优先(现有池 best-of-2-3,零新增 judge),不够再 N↑ + prefilter 判幸存者**(见 §4.1);K=2-3/场景;mix=1:1(defense 半区强化)。
训练:epoch=3(对 ep2/ep3 checkpoint 分别 probe 取优);LoRA rank=24/alpha=48。
