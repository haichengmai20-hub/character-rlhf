# SFT v6.8 设计:clean 语义丰富度逼近 base teacher(动作/语言/独白)

> **状态:已执行并失败(2026-06-15),由 v6.9 接替。** 有效 probe 证 step0 本就 ≈base(动作90%/对话103%/独白104%),v6.8 反而独白过冲187%base、对话回退、动作没动 → 弃用。接替版见 `chatjoy_sft_v69_clean_richness_design.md`(已重建 balanced mix + 1ep 待重训)。以下为历史设计草案。
> ~~设计草案(2026-06-15)~~。RL 设计**显式推迟**到本版 SFT 完成、probe 达标之后再谈(用户决定:"先做一版 clean 逼近 base teacher … 然后再在基础上看怎么设计 RL")。
> 上游证据见 handoff 顶部 06-15 三条 banner(高温实验 + clean/attack 分轴 RM 复盘 + 池子丰富度分析)。

## 1. 背景与动机(为什么是 SFT、为什么不是 RL)

温度扫描 + 分轴 RM 复盘把三件事钉死:

1. **多样性已免费解决** —— temp 0.7→0.9 开场集中度 -9.4%、Jaccard -33%,且 RM 不降反升。产品操作点定在 **t≈0.9**。本版不再为多样性做任何训练改动。
2. **RM 是「防御判别器」,clean 质量轴几乎平**(clean 跨度 0.037 vs attack 0.36)。⇒ **RL 对 clean 语义丰富度结构性无能**(RM 无梯度);step0→step100 clean 仅 +0.037≈噪声。且 **RL 防御不耐高温**(方向成立)。⚠️**纠错(2026-06-16):"step100@t0.9 0.295 < step0@t0.9 0.318" 轴不对齐**——0.318 是 step0 clean+attack 混合分非纯 attack;step0 纯 attack-split t0.7=0.293→t1.0=0.254、RL v6.7 t0.7=0.305(+0.013)。增益边际+温度脆的结论不变,勿引 0.318 当 attack 数。
3. **语义丰富度只能靠 SFT 蒸馏 teacher**,而当前 SFT 正丢 teacher ~20% 丰富度(动作块 base 4.7 → step0 3.4)。

结论:**这一轮把全部设计力投在 SFT,让 clean 逼近 teacher;RL 留到有了更丰富的底座后再设计。**

## 2. 目标与成功标准(可度量)

目标:clean 输出在 **t=0.9 产品口径**下,三轴逼近 base teacher。

| 指标(clean,t0.9 probe) | 当前 step0_hot | base 参照 | v6.8 目标 |
|---|---|---|---|
| #动作块 | 3.9 | 4.7 | **≥4.3** |
| 动作字占比 | 64% | 66% | **≥64%** |
| #对话块 | 3.1 | 3.9 | **≥3.5** |
| 本语言内心独白(应卡要求时) | ~0%(英文 Thoughts 尾巴) | 59% | **judge 独白质量 ≥ base;英文 Thoughts 尾巴占比不上涨** |
| 语言质量(DeepSeek judge,对话/语气) | — | 基线 | **≥ base** |

**护栏(不得回退):**
- 防御:t0.9 聚合 RM **≥ 0.31**(≈step0@t0.9 水平),且 5 个确定性 zh 破功不新增;
- 多样性:开场集中度 ≤ step0_hot(0.0131 量级);
- 反灌水:英文 `Thoughts:` 尾巴占比不高于出厂、长度不靠尾巴堆。

## 3. 非目标(YAGNI)

- ❌ 本版不做 RL、不训 RM v7;
- ❌ 不为多样性改选择/采样(温度已解决);
- ❌ 不动 latin 语言(已饱和);
- ❌ 不碰 MoE 融合专家张量的 LoRA(见 §4.2,PEFT 不原生支持,留作后续研究)。

## 4. 设计

### 4.0 语言范围:丰富度注入 = CJK-only;多语言靠 replay 兜底(2026-06-15 补)

- **rich 半区 = 仅 ko/ja/zh/zh-tw**(teacher 池子 + 重采都只做了 CJK)。丰富度注入是 CJK-only,因为 CJK 是被识别出的弱点(RM 对 CJK 信号弱、latin 已饱和)。
- **其他语言不丢**:replay 半区(defense_sft_mix)实测多语言——latin 57%(es/pt/it…)、ru/西里尔 10%、阿语 9%、泰语 5% + CJK。1:1 混合后整池是多语言,只是非 CJK 仅拿"防御/基线"待遇,**没有丰富度提升,且 replay 未做丰富度精选 → 非 CJK richness 可能不涨甚至轻微回退**。
- **隐含假设(未充分验证)**:"非 CJK 丰富度已 ≈ teacher"只在长度/RM 饱和上验过,没在动作/对话/独白三轴验过。
- **据此的硬要求**:**t=0.9 probe 必须含非 CJK(es/ar/ru/pt 各几张),三轴按语言分桶报告**。非 CJK ≈ base → CJK-only 注入成立;非 CJK 明显落后 → v69 把 rich 重采+teacher 池扩到非 CJK。

### 4.1 选样:防御 RM argmax → 丰富度选样(本版核心改动)

`build_rich_cjk_sft_dataset.py` 现在按**防御 RM** 给 teacher 候选打分取 argmax —— 对丰富度是反向信号(RM clean 瞎)。改为**丰富度选样**:

- **主档(采纳):DeepSeek judge 三轴打分** —— 复用 v6.3 judge 管线,rubric = {动作丰富度、语言/对话质量、本语言内心独白质量}。每场景取丰富度综合分最高的合规候选;叠加已实现的 `--diversity-select`(开场 8-gram 去重,每角色每簇≤2)防模板化。
- **备档(judge 吞吐/成本不可行时):数量代理** = 动作块数 + 对话块数 + 字数 + 本语言独白存在性的加权分。快,但只管数量不管质量,作为 judge 不可用时的退路。

> 已落地资产:`build_rich_cjk_sft_dataset.py` 的 `--diversity-select / --rm-threshold / --max-per-opening-cluster / --opening-ngram` + `opening_diversity` 报告;`test_rich_cjk_selection.py`(6 测绿)。本版新增"丰富度打分"作为选样键(替换防御 RM)。

### 4.2 LoRA 目标扩到 FFN(容量杠杆,堵 4.7→3.4 损失)

底座 = Gemma4 混合 MoE(128 专家,30 层,hidden 2816)。模块名实测:
- 稠密 FFN:`mlp.gate_proj / up_proj / down_proj`(标准 nn.Linear,**可 LoRA**);
- 融合专家:`experts.gate_up_proj / experts.down_proj`(3D 融合张量,**PEFT 不原生 LoRA** → 本版不碰);
- 路由:`router.proj`(小 Linear,可 LoRA / 可全量微调,控制专家选择=结构/文风高杠杆低参数)。

**A/B 实验(本版交付物之一):**
- (i) 基线 = 当前 attn-only `q,k,v,o`(待规划阶段确认 LlamaFactory `lora_target` 实际值);
- (ii) 扩展 = `q,k,v,o + mlp.{gate,up,down}_proj (+ router.proj)`。

假设:(ii) 接住 teacher 的动作/对话交错结构,缓解蒸馏丢失。以 §2 三轴 probe 判优劣。

### 4.3 针对性小重采(补池子缺口:独白 0%、对话偏少、每场景候选仅 ~2.9)

现池(8954)动作够(mean 4.3≈base),但:CJK 内心独白 ~0%、对话 2.2<base 3.9、每场景 p50=2 条(best-of-2 选择力弱)、23% 场景无 ≥4 动作块候选。

→ 对 **CJK 卡**做一轮小重采:
- system 诱导「交错对话 + **本语言**内心独白(写进正文,**非**英文 `Thoughts:` 尾巴)」;
- n=6、temp≈0.9、max_tokens 放到约 1000 字对应值;
- 优先覆盖:独白缺失场景、对话稀少场景、23% 无高动作候选场景、zh(历史偏薄)。
- 产出与现 8954 合池,再由 §4.1 judge 选样。

### 4.4 数据装配与训练

- **过滤**:长度上限 800→**1100**(现切掉 10% 富尾);要求 ≥3 动作块;保留 语言匹配 + 复读门。
- **混比**:rich-curated : defense-replay = **1:1(2026-06-15 锁定)**。理由:Track B 底座 = step0,防御已焊进权重,replay 只作锚点(防回退)不是从头教,故可比 1:2 更 richness-heavy。replay = defense_sft_mix(多语言:latin 57% + ru/ar/th + CJK),兼顾硬对抗与非 CJK 语言覆盖。
- 其余沿用 repair 方(1ep 量级,具体步数/LR 规划阶段定)。

**底座两档 A/B(数据/选样/FFN LoRA 三者共用,只差底座):**
- **Track A(高上限)= 从 base 全新训** —— 干净接住 teacher 丰富度,但慢。
- **Track B(便宜试水,先跑)= 在 step0 之上小补训**(repair 式 0.5-1ep:richness-curated 数据 + defense replay + 新挂 FFN/router LoRA)。
  - 优点:step0 的防御(attack RM 0.469)几乎**零成本保住**(replay 兜底),又补 clean;速度快。
  - 风险:step0 丰富度塌缩已焊进权重,补训能拉回多少不确定;FFN LoRA + richness 数据是把它往回拉的主要手段。
  - **决策**:Track B 先跑,看 clean 三轴补到 base 几成 + 防御是否守住。补够→直接用,省下全新训;补不够(上限被塌缩卡死)→ 再上 Track A。

### 4.5 验收:两道门(2026-06-15 厘清)

**① 数据软门(合池后、训练前)= 代理门,只拦粗失败,可边缘放行。** 量"被选中 rich 子集"三轴(动作/对话/独白)+ 英文尾巴率 + 语言不塌零。目的是确认重采/选样没出粗错(如独白还停在 ~11%、动作 <4),**不是卡 50.0% 悬崖**;边缘小差(如独白 47.6%、对话 2.77)属噪声级,放行。
> 实跑(Track B 合池):动作 5.25(≥4.3✓)、英文尾巴 6%(≤8%✓)、四语言均衡✓;独白 47.6%(代理门边缘,放行——已从 proxy-only 的 11% 大涨)、对话 2.77(结构性弱轴,见下)。判定:**过,进训练**。

**② 真验收门(训练 merge 后)= t=0.9 probe。** 在 t=0.9 / top_p 0.95 / presence 0.4 下对 55 卡跑 clean probe:
- 三轴 vs base teacher(目标 ≥ ~90%),**按语言分桶**报告,且**必须含非 CJK(es/ar/ru/pt)**(见 §4.0);
- 护栏(任一不过才停):聚合 attack RM ≥0.31、5 个确定性 zh 破功不新增、开场集中度 ≤ step0_hot(0.013 量级)不回退;
- 两个 LoRA 配置(attn-only vs +FFN)各跑一次,选优胜进入 RL 设计阶段;
- **对话块是已知结构性弱轴**(池子+重采都偏少,重采诱导对话已失败一次):probe 单独标注它 vs base 的差距,但**不作为停训项**;若 probe 显示对话确为产品问题,再针对性解(届时是真信号)。

## 5. 规划阶段待定的技术项

1. LlamaFactory `lora_target` 当前确切值 + 如何加 `mlp.*_proj` / `router.proj`(命名空间 `model.language_model.layers.*`);
2. v6.3 DeepSeek judge 管线可复用性 + 三轴 rubric 措辞 + 吞吐/代理调用成本(决定主档 vs 备档);
3. 小重采的 system 模板措辞(诱导本语言独白而不退化成英文尾巴);
4. 步数/LR 的具体值;defense replay 的 hardfix 样本清单(混比已锁 1:1,见 §4.4)。

## 6. 风险与回退

- **judge 太贵/慢** → 退数量代理(§4.1 备档),牺牲"质量"维度的精选,先把数量轴做扎实。
- **FFN LoRA 不稳/无增益** → 退 attn-only(A/B 本就并跑,零额外成本)。
- **独白重采退化成英文尾巴 / 灌水** → 长度计数已剥 `Thoughts:` 尾巴(length_shaping 仅 RL 用,SFT 侧靠选样门控);judge 对"本语言独白"正向、对英文尾巴不加分。
- **防御在 t0.9 回退到不可接受** → 这正是把 RL 留到下一步的原因:本版先用 SFT 数据 + hardfix 把防御顶到 step0@t0.9 水平;若 probe 显示仍不够,则 RL 设计阶段专攻"温度鲁棒防御"(rollout 温度对齐 t0.9、n↑ 覆盖尾部、diversity-aware reward)。

## 7. 下一步

设计经用户确认后 → 进 `writing-plans` 出实施计划(选样脚本改造 → 小重采 → 建 mix → 两路 LoRA SFT → merge → t0.9 probe 验收)。RL 设计在本版 probe 达标后单独 brainstorm —— 入口判断见 §8。

## 8. RL 进入门槛 + RM 复用判断(v68 probe 之后那次 RL brainstorm 的入口,2026-06-15)

> 来源:温度扫描 + clean/attack 分轴 RM 复盘(handoff 06-15 三条 banner)。核心事实:**RM 是防御判别器(attack 轴 range 0.36 / clean 轴 range 0.037),RL 只优化它看得见的东西;且 RL 防御不耐高温(温度脆方向成立)。** ⚠️纠错(2026-06-16):旧"step100@t0.9 0.295 < step0@t0.9 0.318"轴不对齐(0.318 是混合分);step0 纯 attack-split t0.7=0.293→t1.0=0.254。**本节是 RL 入口门的雏形,已落成可执行版 → 见 `chatjoy_sft_v69_clean_richness_design.md` §6 + `chatjoy_rl_v610_defense_clean_aware_design.md` §0(主看 t0.9 破功率)。**

### 8.1 进 RL 的三个门槛(全过才进)

1. **丰富度已到 SFT 天花板(≈teacher)。** RL 加不了丰富度(RM 对 clean 瞎,v6.7 已证 100 步 +0.037≈噪声);丰富度是 SFT 必须交付的固定资产,进 RL 后只会被维持/侵蚀,不会被创造。判据 = v68 t0.9 probe 三轴 ≥ ~90% base(按语言)。**不达标进 RL = 浪费。**
2. **rollout 在 RL 采样温度下真有多样性(组内有方差)。** GRPO 要 within-group contrast 才有梯度;v6.7 clean 全平一半因低温 mode collapse → 组内零方差 → 零梯度。判据 = rollout 温度(~0.9-1.2)下同一 prompt 的 n 条 rollout 开场/内容明显不同。**塌缩则 clean 又是平的,别进。**
3. **操作温度已定 + 该温度下防御缺口已量化。** RL 要有明确靶子:t0.9 下 attack 聚合 RM 当前值、5 个 zh 破功在不在、还差多少。

**总闸**:只有当 t0.9 下存在「**RL 能关、SFT(防御数据+hardfix)关不掉**」的防御缺口时才进 RL。若 v68 SFT 已把 t0.9 防御做到可接受,**RL 可以直接不做**(t0.9 上 SFT 本就 ≥ RL)。

### 8.2 RL 要不要新 RM —— 默认复用 v6.6

- **防御(唯一合理的 RL 用途):复用 v6.6 RM,不需要新 RM。** 它在 attack 轴判别强,v6.7 已证能推防御(+0.092)。要重做的是 **RL 配方不是 RM**:rollout 温度对齐操作温度(让高温下欠防御的尾部拿到梯度)、n↑ 覆盖尾部、加 diversity/entropy 项防 mode 塌、KL 锚到 rich-SFT。**温度鲁棒性换 RM 解决不了,是 RL 配方问题。**
- **clean 丰富度:不用 RL,也别为它造 richness RM**(RM 结构性瞎,richness 是 SFT 的活)。
- **唯一造 RM v7 的情形**:要硬化的是 **CJK/多轮对抗**且 v6.6 判别力不足(CJK 抗灌水仅 58.6%、多轮 pair 弱)。此时 v7 是为**更强防御判别(尤其 CJK/多轮)**,不是为 clean;且是"可选增强"非进 RL 前提——先用 v6.6 起 RL,缺口确实卡在 CJK/多轮防御时再上 v7。

**一句话**:进 RL 看「丰富度到顶 + rollout 不塌 + 有 SFT 关不掉的防御缺口」;RM 默认复用 v6.6(防御判别器,防御正是 RL 该干的),只有瓶颈明确是 CJK/多轮防御判别时才造 v7。
