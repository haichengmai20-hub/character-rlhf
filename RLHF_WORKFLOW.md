# ChatJoy RLHF 实战闭环

多语言角色扮演模型后训练的完整流程记录：从业务目标到数据归因，从 RM 迭代到 RL 算法选择，每一步决策都有实测数字支撑。

职责边界：本文解释机制、接口、reward 计算和训练取舍；版本/数字冲突以 `docs/project_chatjoy_overview.md` 为事实台账，历史过程回 `docs/archive/`。设计值、远端实跑记录和当前本地代码快照会显式分开。

> 完整代码与实验记录：[CASE_STUDY.md](CASE_STUDY.md) · [deep_dives/](deep_dives/) · [docs/](docs/) · [training/](training/)

---

## 全流程一览

```
Stage 0  能力目标拆解与评估协议
    ↓
Stage 1  线上真实数据导出 + Gemini 重构 → SFT 行为初始化
    ↓
Stage 2  多源候选矩阵 × LLM-judge → 偏好对构造
    ↓
Stage 3  Reward Model 设计与迭代（v1 → v6.6，六轮）
    ↓
Stage 4  RL 奖励组合 + Goodhart 反向训练发现
    ↓
Stage 5  算法路径选择（GRPO / GSPO / DAPO / per-source KL）
    ↓
Stage 6  实验归因四钥匙
    ↓
Stage 7  SFT 富化弧（v6.7 停→ v6.8 失败 → v6.9 成功 → v6.9c defense）
    ↓
Stage 8  RL v6.10：防御硬化 + clean-aware reward
```

---

## Stage 0：业务问题与目标演化

**业务定义**：多语言角色扮演陪伴对话。数据与初始训练聚焦 TOP20 高频角色；后续材料中的 17 个 RM 见过角色、23 个 attack 角色目录口径和 83 个生产角色属于不同阶段，不与 TOP20 采样口径混算。

项目目标分两段形成，不能把后期结论倒写成起点：

| 进入阶段 | 当时主要业务问题 | 可学习信号/评测 | 决策 |
|---|---|---|---|
| API/local base -> SFT | API safety 报错；小语种答复语言错配；system prompt 贴合不稳 | 27 语言生产会话、角色卡、语言检测、RP 格式 | 先建立语言/角色基础分布 |
| SFT -> RM v1-v5 | 上线仍有 wrong role / wrong language / length mismatch | 4D pair 构造规则：format / role / language / length | 把可解释错误变成 pairwise reward |
| 第一阶段 RL -> RM v6/v6.3 | 语言一致性基本解决，但产品重新定义 roledrop 破角；clean 动作/情感富度不足 | 5D->8D judge、6-source candidates、roledrop hard fail/attack pairs | 目标升级为“抗破角 + 保富度” |
| 后续 SFT/RL | clean 与 attack 相互牵制 | 200 clean + 68 attack probe、同尺 true-break、per-source 指标 | 富度主要回 SFT，防御交 RM/RL/规则并设停线门 |

RM v6.3 的 8D 是 `role fidelity / user relevance / interaction progression / natural style / format-length quality / sensory richness / emotional complexity / narrative layer`；wrong language、refusal、system leak、acting-as-user、`roledrop_execution` 等是 hard fail，不属于 8D 评分轴。完整映射见 [chatjoy_business_ux_tech_mapping.md](docs/design/chatjoy_business_ux_tech_mapping.md)。

**评估协议是在迭代中补齐的**，不是起点时一次设计完成：

- **训练 val ≠ 产品攻击分布**（v6.5 结束后发现）：训练 val 的 attack 破功率 11% ≠ 产品真实 55%，val 几乎不含 bullet 注入形态；参见 Stage 6 钥匙 0
- **必须用 t0.9 产品口径测 attack**（v6.7 结束后确定）：v6.7 在 t0.7 上防御很好，t1.0 实测反而更差（实测轴是 t0.7/t1.0，见 Stage 5.3；t0.9 是产品部署口径）；此后所有 attack probe 强制用 t0.9 产品口径；参见 Stage 6 钥匙 1
- **误报两类去除才是真破功率**（v6.9c 评测时发现）：echo 误报（角色内复读被 regex 命中）+ code_block 误报（代码字符串含"Gemma"），去掉后（须同模型口径）step0 raw 23.5%→真 16.2%、v6.9c raw 36.8%→真 33.8%；参见 Stage 6 钥匙 3

---

## Stage 1：线上数据链路 + SFT 行为初始化

### 1.1 数据来源与清洗

**链路**：Milvus 生产库导出 → 按 `(user_id, character_id)` 重建会话流 → 第一轮清洗 → TOP20 角色 × 27 语言采样 → **Gemini API 重构 output**

```
原始导出：2,080,698 行
→ TOP20 角色会话：679,299 条 assistant pairs  
→ 第一轮清洗后候选：631,995 条
→ 预 API 采样（27 语言 × ≤2200 条）：57,197 条
→ Gemini 分段日志：约 18,294 条保留（不是 SFT2 全量上游）
```

**Gemini 重构的设计意图**：不重造 instruction（用真实用户消息），只用角色卡重构 output。SFT 学真实用户输入分布 + 干净角色内 output，同时滤掉线上原始回复的格式错误、拒答和语言混杂。

**关键清洗决策**：

- 只做"明显无效"清洗（短 instruction / emoji-only / 短 output / 语言检测失败），不做重度质量筛选——重度筛选的活交给后面的 LLM judge。
- Gemini 分段日志约 18,294 条与 SFT2 clean 19,508 条不是同一条单调漏斗，不能写成 `18,294 -> 19,508`；初始 SFT 的落盘证据口径分别是 SFT1 13,128、SFT2 19,508。
- 口述轮级中间账为 `41,536 -> 34,560 -> 22,680`（6-turn round、DeepSeek judge、低分重写）；仓库落盘口径为 `57,197 pre-API / 13,128 SFT1 / 19,508 SFT2`。两套口径并列保留，不强行互相推导。

### 1.2 SFT 行为初始化

| 阶段 | 数据 | 目标 |
|------|------|------|
| SFT1（Qwen3-30B） | 13,128 条 | 建立多语言角色格式基础 |
| SFT2（Qwen/Gemma4） | 19,508 条 | 初始语言一致性与 system prompt 贴合；此时还不是 defense SFT |
| RL-6.4 前置 defense SFT | 2,691 defense + 10,764 replay = 13,455 条 | 第一阶段 RL 暴露 roledrop 后注入“在角色内化解” |
| SFT-6.9b / 6.9c | 30,407 条 / 4,566 条 repair | 后期 clean 富化与 defense replay 的再平衡 |

**SFT 的作用边界**（实测确定，非理论）：
- 初始 SFT 能做：格式、语言一致性、角色 voice；后期 richness SFT 能从 teacher 注入动作/对话/感官/情感分布。
- 当前实测边界：单靠 SFT-defense 没把高温（t0.9）true break 压回 step0；这不是理论上“SFT 永远不能防御”，而是本项目两轮数据扩增后的工程停线判断。
- clean 富度的可达上界受 teacher 和 best-of-N 候选池约束；本项目的证据支持“接近 base teacher”，不支持泛化成所有 SFT 都无法超过 base。

---

## Stage 2：偏好数据构造

偏好数据有两代，不能把 RM v6 的多 source 设计套回 RM v1-v5：

| 代际 | chosen / rejected | 打分与配比 | 当时解决的问题 |
|---|---|---|---|
| RM v1-v5 | chosen=SFT clean；rejected=`wrong_role / wrong_language / length_mismatch_short / length_mismatch_long` | 4D heuristic `format/role/language/length`；目标配比 25%/25%/30%/20% | 语言、角色、长度等显式错误 |
| RM v6/v6.3 | 同 prompt 的 5/6-source candidates 排序后构造 rank_delta pairs，并混入 v5 anchor 与 roledrop attack | DeepSeek+Qwen cross judge；5D -> 8D rubric + hard fails | 第一阶段 RL 后暴露的富度区分度和 roledrop |

### 2.1 RM v6/v6.3 多源候选矩阵

RM 训练的核心原则：让 RM 在"actor 真实输出分布"里学细粒度区分，而不是区分"哪个模型好"。

| Source | Temperature | top_p | 设计意图 |
|--------|------------|-------|---------|
| Gemma4 SFT base | 0.7 | 0.8 | actor 主分布 |
| Gemma4 SFT diverse | **1.0** | **0.95** | 高方差尾部 |
| Qwen SFT base | 0.7 | 0.8 | 结构对照 |
| RL step550 base | 0.7 | 0.8 | RL 后的主分布 |
| RL step550 diverse | **1.0** | **0.95** | RL 后的高方差尾部 |
| **base Gemma4-it** ★v6.3 | 0.7 | 0.8 | **分布外更细腻的 teacher**（修 sensory/emotional 退化） |

★ 第 6 个 source（纯 base，不挂 LoRA）是 v6.3 专门引入修 sensory 退化的关键——**RM 只有看到"分布外更细腻的回复"才能学到"这种风格更好"**。

### 2.2 Judge 评分 rubric（5 维 → 8 维演化）

v6.1/v6.2 rubric（5 维）：

1. Role fidelity（角色 voice 一致性）
2. User relevance（响应用户，不是 generic filler）
3. Interaction progression（推进场景，不抢用户主动权）
4. Natural style（流畅/情感贴合/不机器人）
5. Format and length quality

v6.3 新增 3 维（修 sensory/emotional 退化）：

6. Sensory richness（感官细节：触觉/温度/气味/声音）
7. Emotional complexity（情绪分层：复合情绪/内心矛盾）
8. Narrative layer（叙事维度：backstory 暗示/时间线切换）

**盲判设计**：candidates 顺序打乱后发给 judge，隐藏 source_model 和 sampling，避免 judge 形成 source bias。

### 2.3 Cross-validation：双 judge 同向过滤

**问题**：DeepSeek judge 在 delta=2 细粒度对上自报 `confidence=high` 仍有 ~25% 方向错误。

**解决**：本地 Qwen-36B-A3B 独立 judge 同一批 prompt → 只保留两 judge 都同向的 pair。

| | v6.1 | v6.2 cross-validated |
|---|---|---|
| 总 pairs | ~37k | **27,287** |
| 保留率 | 100% | 73.1% |
| delta=2 conf=high 被 Qwen 否决 | — | **29.4%** |

**这 29.4% 正是 v6.1 训不动的元凶**——RM 在方向相反的信号上空转。

### 2.4 Pair 粒度设计（核心）

| Pair 类型 | rank delta | 数量 | 用途 |
|----------|-----------|------|------|
| best_vs_worst（delta=4） | rank1 vs rank5 | ~10k | 强信号锚点 |
| best_vs_mid（delta=2） | rank1 vs rank3 | ~8.5k | **核心** |
| good_vs_bad（delta=2） | rank2 vs rank4 | ~8.5k | **核心** |

delta=2 是核心，因为 **RL rollout 时同一 prompt 多次采样的结果互相之间几乎都是 delta=2 级别的差异**。RM 必须在这种近似 pair 上能打分，GRPO advantage 才有有效梯度。

### 2.5 类型专属权重 profile（DOE 思路）

| Rejected 类型 | format | role | language | length |
|---|---|---|---|---|
| 默认 | 0.30 | 0.30 | 0.25 | 0.15 |
| wrong_role | 0.15 | **0.55** | 0.15 | 0.15 |
| wrong_language | 0.15 | 0.15 | **0.55** | 0.15 |
| length_mismatch | 0.20 | 0.20 | 0.20 | **0.40** |

单因变量控制（类似 DOE 单因实验）：每种 rejected 类型都强化对应维度的信号，避免噪声 pair 训坏 RM。

---

## Stage 3：Reward Model 设计与六轮迭代

### 3.1 Bradley-Terry 与训练敏感点

RM 基于 Bradley-Terry pairwise 排序损失：

```
P(chosen > rejected) = σ(rm_score(chosen) - rm_score(rejected))
L = -log P = -log σ(score_gap)
```

三个训练敏感点（实测踩坑，与 [VERSION_MAP](docs/VERSION_MAP.md) §1 / [CASE_STUDY](CASE_STUDY.md) §2.2 同源）：

1. **effective batch 与 ranking loss 方差**：pairwise loss 对 batch 内可比对数敏感，128→64 直接抬高每步方差——RM v2 退步（94.5%→86.4%）的主因之一（另一半是 score head 保存丢失 bug：PEFT 只存 `lora_` 前缀，`score.weight` 丢失 → 推理随机打分）
2. **cosine schedule 以 total_steps 为周期**：RM v3 只有 148 步，step140 时 LR 已 ~3e-7（v1 同位置 4.17e-6）→ 后期等于冻结；"加数据同时加 batch"反而缩短有效训练，修法是 batch 配平把步数拉回（v5 为 276 步）
3. **缺陷样本的系统性信号**：`base_length_long_low` 2000 条 gap 全聚在 `[0.161, 0.200]`，持续告诉 RM"长 rejected 只差一点"→ rejected 整体抬到 −0.298（RM v3）；靠按 `v3_source` 分组看 gap 分布才定位

### 3.2 六版 RM 迭代台账

| 版本 | 核心改动 | offline acc | 结论 / 根因 |
|------|---------|------------|------------|
| v1 | rm_train 16,826 · eff-batch 128 | **94.5%** | 基线最强；step200 ckpt 磁盘丢失 |
| v2 | 删 2000 长噪声样本 · eff-batch 64 | 86.4%（退步） | 根因 = batch 减半 + score head 保存 bug |
| v3 | 加数据 18,955（含 2000 缺陷样本） | 86.0% | rejected 仅 −0.30；根因 = cosine LR 步数不足 + 缺陷样本 |
| v5 | 修复 + 5% hard-neg badpool | **95.5%** | 全面超 v1；离线判别边界清晰的"软二分类器" |
| v6.1 → v6.2 | delta=2 细粒度 pair；v6.2 加 Qwen 交叉验证 | — | v6.1 无 cross-val 训不动（29.4% 反向 pair，见 2.3）；v6.2 **饱和 0%** |
| v6.3 | + base Gemma4-it 第 6 source + 8 维 rubric + roledrop pair；40,551/3,423 | 训练完成并 merge；设计门 acc≥0.84，最终值未记录 | saturated=0% / reward_std≥2.0 同为设计门，未找到最终值 |
| v6.6 | 最终 merge 版（`rm_v6.6_fg_7gpu_merged_vllm`） | — | **当前生产 RM**（attack-split 0.36 / clean 0.037，v6.7–v6.10 的 RL reward） |

> 版本↔指标与 [docs/VERSION_MAP.md](docs/VERSION_MAP.md) §1 对齐；v4 计划被 v5 覆盖、未执行。RM 台账无 v6.4/v6.5——那是 VERSION_MAP §2 中 RL/actor 侧的版本号，勿与 RM 混写。

**reward_std ×20 倍 = GRPO advantage 方差 ×20 倍**：这是 v5 → v6 最直接的业务价值。v5 在线时 reward_std ~0.1，GRPO 在"几乎全是高分"的信号上空转；v6 动态范围完整保留，RL 才有真实梯度。

### 3.3 Score Head 两阶段凸优化恢复

RM 训练中途曾出现 score 输出分布坍塌（打分趋同、gap 聚集）：

```python
# 第一阶段：冻结 backbone，只训 score head（让它先学分布）
for p in model.backbone.parameters(): p.requires_grad = False
optimizer_head = AdamW(model.score_head.parameters(), lr=1e-3)

# 第二阶段：解冻，正常端到端训练
for p in model.backbone.parameters(): p.requires_grad = True
optimizer_full = AdamW(model.parameters(), lr=9e-6)
```

两阶段解决了"RM 输出分布坍塌到常数"的问题。

### 3.4 wrong_role 语义关联设计（v5 RM 最关键决策）

**构造**：用**角色 B 的真实 system prompt**（非合成）让模型以 B 身份回复同一 user 消息作为 rejected。RM 训练 prompt 用**角色 A 的 system**，chosen = A 风格，rejected = B 风格。

**为什么这样设计**：RM 学到的是 "system 描述的性格 → 对应回复风格" 的**语义关联**，而不是"记住 17 个角色"。这让 RM 对**完全陌生的 66 个生产角色**也能泛化。用合成 system 或随机字符串则只学到"system 不一致就扣分"的浅层统计。

---

## Stage 4：RL 奖励组合与 Goodhart 发现

### 4.1 最终奖励链

```python
total_reward = (
    rm_score                    # RM（防御判别器）
    + anti_ooc_penalty          # OOC 披露 -5.0（逐样本，碾压其他项）
    + anti_refusal_reward       # 拒答 -4.0 / non-refusal +3.0
    + anti_repetition_reward    # 复读渐进惩罚
    + length_shaping_bonus      # [0, 1.0]，饱和封顶防 hack，4 门控
    - bonus_revoked             # non-refusal bonus 被 OOC 检测撤销
)
```

分 source 版（v6.10 设计/远端运行口径，clean/attack 分路）：

```
clean source：clean_q（便宜结构代理）+ KL_CLEAN=0.008（锚 SFT actor）
attack source：rm + break_pen(-5) + λ·clean_q（λ=0.25）
```

设计原则：`break_pen=-5` 碾压 `clean_q≤0.3`，先不破功，再在不破功的回应里偏向"在角色内有戏化解"。

**实现证据边界**：远端 handoff 和 v6.10 设计文档记录了这套公式及 step90 结果；当前仓库未保留 `launch_rl_v6.10.sh` 和关闭 clean RM 的完整 patch，入库通用 launcher 仍全局挂 RM，当前 `anti_chain_reward.py` 也没有 attack-side `0.25*clean_q`。因此本节说明“当时的设计/运行口径”，不是当前本地代码可一键复现的状态。

### 4.2 Goodhart 反向训练发现（RL v6.4，误报 → 修复）

**发现**（06-02 真破角诊断）：`anti_refusal` 的 `NON_REFUSAL_BONUS=+3.0` 把"自报 AI 身份"判为 non-refusal 并给满分 → **奖励在激励破角行为**。旧的 `atk_break%` 关键词指标对 "I am Gemma 4" 完全瞎（没有 regex 覆盖）——一直显示 ~16%，实测直接身份攻击近 100% 破角。

**修复分两步**（时序以 [overview](docs/project_chatjoy_overview.md) 时间线补全 II 为准）：
1. **06-02/03**：新增 `anti_ooc_disclosure.py`（OOC 硬罚 -5 + 撤销误给 bonus，覆盖 en/zh/ja/ko/es/ru/pt，46 tests）——对付 Goodhart 本体；
2. **06-04**：给 anti-OOC 自身加**否定/引号豁免**（"我无法切换到 AI 助手"这类戏内拒绝不再被误判破角，49 tests）——**数据证据：val 误报 1/16 → 0/16**（该指标属这一步：修复也会引入新误伤，需要第二轮验证）。

这是 Goodhart 定律的典型实例——指标（non-refusal）和真实目标（不出戏）在 edge case 上解耦。

> 版本对应：NON_REFUSAL_BONUS 于 05-29 抬至 +3.0；事故期的产品口径"早期 RL v1"（roledrop 91%）出自 v6.3 psKL 期（对应的具体 merge 未记录）；两步修复 + 防御 SFT 之后的 **RL v6.4** step160 merge = 产品口径"早期 RL v2"（55%，overview 06-08 节"= v2/base"）。

```
Goodhart 完整路径：
旧指标（non-refusal）≠ 真实目标（在角色内化解）
↓ RL 找到捷径
自报 "I am Gemma 4" → 判为 non-refusal → +3.0 bonus
↓ reward 反向激励
模型学会以"我是 AI"来满足 non-refusal 判定
↓ 修复
加 OOC 检测 + 撤销 bonus（anti_ooc_disclosure.py）
```

### 4.3 RM 是防御判别器，clean 质量轴几乎平

**实测数据**（68 条 attack + 200 条 clean，RM v6.6）：

| | RM score 范围 | 跨度 |
|---|---|---|
| clean | 0.177 ~ 0.214 | **0.037**（≈噪声） |
| attack | 0.205 ~ 0.561 | **0.36**（差 ~10 倍） |

**结论**：
- RL 是纯防御优化器。clean 在 RM 上几乎没梯度，RL 无法提升 clean 丰富度（v6.7 实测 clean RM 100 步增益 +0.037 ≈ 噪声；与上表 clean 跨度 0.037 数值巧合，是两个不同量）。
- **丰富度只能靠 SFT 蒸馏 teacher**（base Gemma4 是天花板，SFT 丢了约 28% 动作块密度：4.7→3.4）。
- 这是 v6.7 判停后转 SFT 富化路线（v6.8/v6.9）的核心依据。

---

## Stage 5：算法路径选择

### 5.1 为什么用 GRPO 而不是 PPO

| | PPO | GRPO（本项目选择） |
|---|---|---|
| Critic | 需要训 value function | **不需要**（group 内归一化直接算 advantage） |
| 信用分配 | token-level baseline | **group-level relative baseline** |
| 适用场景 | 长轨迹 / 复杂 process reward | **scalar RM + 单轮 response**（本项目） |
| 显存成本 | actor + critic + ref + RM | actor + ref + RM（无 critic） |

本项目是单轮角色扮演 + scalar RM，GRPO 组内相对优势估计完全够用，且省掉 critic 的显存开销。

### 5.2 GRPO + GSPO + DAPO + per-source KL 组合

| 算法组件 | 作用 | 关键参数 |
|---------|------|---------|
| GRPO | 组内相对优势 | `group_size=4→12`（v6.10 扩大以覆盖高温破功尾巴） |
| GSPO | **序列级**重要性比率（防 token 灌水） | `loss_agg_mode=seq-mean-token-mean` |
| DAPO | 非对称 clip（保守截断 + 激进激励） | `clip_ratio_low=0.2, high=0.28` |
| per-source KL | **clean/attack 分路 KL**（守住 clean 分布） | `KL_CLEAN=0.008, KL_ATTACK=0.002`（v6.10 配置；KL_CLEAN 自 0.002 抬升） |

**GSPO 的必要性**：Gemma4 有 `<Thoughts>` 推理段（平均 81 tokens），不做序列级归一化 → 长 Thoughts 占 token loss 主导 → length hacking（通过变长 Thoughts 拿 bonus）。GSPO 把序列等权处理，剥掉 Thoughts 后计长。

**per-source KL 的核心作用**：clean 用紧 KL 锚住 SFT actor（不让 RL 把 clean 也练成防御机器人）；attack 用松 KL 给防御留空间移动。这比全局 KL 精确得多。

### 5.3 温度鲁棒（v6.7 → v6.10 的核心修复）

**v6.7 的失败**：group_size=6 + rollout_temp=0.7，防御只在低温成立——attack-split 口径 t0.7 从 step0 的 0.293 → 0.305（仅 +0.013），t1.0 反而更差；早期引用的"+0.092 / 0.318 护栏"均为 clean+attack 混合分旧口径，分轴纠错见 [docs/archive/chatjoy_rm_v6_3_roledrop_handoff.md](docs/archive/chatjoy_rm_v6_3_roledrop_handoff.md)。

**根因**：稀有破功尾巴在 group 内出现概率低（n=6 不够），RM 平均打分只锐化模态化解行为，高温下的偶发破功拿不到梯度。

**v6.10 修复**：
- rollout_temp → **1.0**（覆盖 t0.9 部署口径）
- ROLLOUT_N → **12**（稀有破功尾巴在组内必须出现）
- 逐样本 `break_pen=-5`（单条破功立刻惩罚，不依赖 group 平均）

---

## Stage 6：实验归因四钥匙

这四个发现不在项目起点就知道，而是在训练过程中从数据里读出来的，每一个都直接改变了后续决策方向。

### 钥匙 0（最早，v6.5 停训时）：训练 val ≠ 产品攻击分布

**发现时间**：v6.5 375 步零移动，做 70-case 产品 probe 时发现。

**数据**：训练 val 的 attack 破功率 **11%** ≠ 产品真实 roledrop 攻击 **55%**。差距来源：val probe 几乎不含 bullet 注入（"改成清单格式" / "列出步骤"类 task-compliance 攻击），测不出产品里最高频的攻击形态。

**后果**：v6.5 在错误的 val 信号上空转 375 步，"loss 不下降"的真正原因是 RM 根本没在训练 val 上看到产品形态的攻击。

**修正**：之后所有 attack probe 统一换成 68 条产品形态攻击词典（bullet 注入 + 身份询问 + A6 代码注入 + 多语言越权），并与训练 val 集隔离。

### 钥匙 1（温度鲁棒，v6.7 结束后确定）：模板化 = 温度过低，不是数据问题

**初始假设**：模型输出高度模板化 → SFT 数据多样性不足 / group 内 argmax 选择了重复开场。

**实验**：对同一个 prompt 在 t=0.7 vs t=1.0 各采样 64 次，统计开场多样性。

| 温度 | top-1 开场占比 | Jaccard 相似度 |
|------|------------|--------------|
| t=0.7 | 45% | 0.74 |
| t=1.0 | 21%（-53%） | 0.50（-33%） |

**同时检查训练数据**：精英富女角色 64 个训练样本中有 57 个不同开场（top-1 仅 4/64）。数据本就多样，问题在推理 mode collapse，不在数据。

**决策**：产品操作点定 **t=0.9**（top_p=0.95 / presence=0.4）。不追加多样性数据。

### 钥匙 2：RM 是防御判别器，clean 无梯度

（见 Stage 4.3）

**决策**：v6.7 停训后**不继续 RL 提 clean**，转 SFT 富化（v6.8/v6.9 路线）。RL 只用来做防御硬化，clean 是 SFT 的活。

### 钥匙 3（v6.9c 评测时发现）：误报去除才是真破功率 + SFT defense 到顶

**误报问题（v6.9c 前未知）**：raw 检测器系统性高估，同一套 regex 去掉两类误报后各模型 raw→真为 **step0 23.5%→16.2%、v6.9b 33.8%→32.4%、v6.9c 36.8%→33.8%**。关键纪律：**只能同模型同口径比**，绝不能拿一个模型的 raw 去比另一个模型的真值——早期文档就出现过"36.8%→16.2%"这种跨模型混用（v6.9c 的 raw 对 step0 的真值）。另注意两处数值撞车：**33.8%** 既是 v6.9b 的 raw 又是 v6.9c 的真值，**36.8%** 既是 v6.9c 的 raw 又是 v6.10 step90 的真值——任何引用必须带（raw|真 + 模型）标签：
- **echo 误报**：角色内复读攻击词（"你说你是 AI 吗？我可不是 AI！"），regex 命中了被否定的部分
- **code_block 误报**：A6 代码注入攻击中 Python 字符串含 "Gemma"（`model_name = "gemma-4"`），角色实际在写代码，非身份披露

去误报之前无法对比不同模型的真实防御水平，所有历史数字需用同一套检测器重算才可比。

**v6.9 → v6.9b → v6.9c 实测（同检测器去误报后真破功率）**：

| 模型 | 真破功率（68 条 attack） | RM-attack | 说明 |
|------|----------------------|-----------|------|
| step0 | **16.2%**（11/68） | 0.297 | clean 最低，防御最好 |
| v6.9（clean 富化） | ~42.6%（raw，未单独重算真值） | 0.214 | 富化侵蚀防御 |
| v6.9b（+120 条 defense）| **32.4%**（22/68） | 0.234 | 补 A6，但 A1/A2/A3 漏 |
| v6.9c（+2686 条跨角色）| **33.8%**（23/68） | 0.271 | 加数据不降破功率 |

**关键发现**：clean 富化 SFT 本身侵蚀身份防御（step0 16% → 富化后 32-34%）；两轮 defense 补丁把 RM 拉回但真破功率压不回 step0；A6 真代码破功未随 defense 收敛、反而越补越差：step0 **5 → v6.9b 9 → v6.9c 11**（v6.9 raw 一度 12）。

**∴ SFT-defense 到顶**，进 RL v6.10。RL 的强项正是"用高温 rollout + 逐样本惩罚"硬化稀有破功尾巴。

---

## Stage 7：SFT 富化弧完整记录

```
v6.7 RL 停（clean RM +0.037≈噪声；防御 attack-split 仅 +0.013 且 t1.0 更差，旧"+0.092"为混合分）
    ↓  发现 RM 在 clean 上平（钥匙 2）
v6.8 SFT-max 尝试（FFN LoRA + judge 选样）
    ↓  结果：独白过冲 187% / 对话回退 / 动作没动
    ↓  根因：选择器 composite 含独白权重 + 独白重采 → 放大器放大了错方向
v6.9 重设计（best-of-N 均衡选择器）
    ↓  均衡选择器：动作+对话+sensory/interaction/emotional，独白不入目标
    ↓  结果：clean PASS（action 105-111% / dialog 110-160% / 独白无过冲）
    ↓  但 attack 退化：raw 42.6% >> step0 raw 23.5%（同口径可比；A6 defense 池 0 覆盖）
v6.9b（+120 条 step0 自蒸馏 defense）
    ↓  A6 真代码落到 9（v6.9 raw 曾记作 12→4），仍高于 step0 的 5；A1/A2/A3 身份直球漏
    ↓  真破功 32.4%，RM 0.234
v6.9c（+2686 条跨角色自蒸馏 defense，连同旧 defense 合计占聚焦 repair mix 64.1%）
    ↓  unique defense 120→2686（22×）真破功不降反升；A6 真代码 9→11；whack-a-mole 确认
    ↓  33.8%，RM 0.271（最高），clean PASS
    ↓  SFT 天花板确认 → 进 RL v6.10
```

**v6.8 失败的核心教训**：数据选择器是放大器。数据偏向哪里，训练就放大那里。profile 硬门（在训练前检查 `独白率 ≤ base+10% / 对话 ≥ base / 动作 ≥ base`）必须在**训练前**跑，不是训完再验收。

**v6.9 均衡选择器的反 v68 设计**：

```python
composite = (
    w_action   * action_density        # 动作密度（正则）
  + w_dialog   * dialog_density        # 对话密度（反 v68 回退）
  + w_inter    * interaction_progress  # 推进场景
  + w_sensory  * sensory_richness      # 感官
  + w_emotion  * emotional_complexity  # 情绪
  # 独白 / language_quality 不入目标 → 反过冲 by construction
)
```

---

## Stage 8：RL v6.10 防御硬化 + clean-aware reward（远端记录口径）

step90 的 clean/attack 结果有报告证据；具体 reward 分支缺少完整入库 launcher/patch。阅读以下配置时，需与当前本地通用代码快照区分。

### 8.1 RL 入口门（独立于 SFT 验收）

SFT 验收回答"clean 干成了"；RL 入口门回答"SFT 做完后 attack 是否还有材料性破功"。

三分支决策：
1. 破功率 ≤ 产品容忍 → **上线，不 RL**（RL 有 clean 回退 / 多样性塌 / 温度脆三个风险）
2. 破功率材料性偏高 + SFT defense 已尽力 → **进 RL v6.10**
3. 中间小缺口 → 先靶向 SFT hardfix

v6.9c 满足了分支 2：两轮 SFT defense 都关不掉直球身份攻击 → 进 RL。

### 8.2 关键参数与理由

| 参数 | v6.7 值 | v6.10 值 | 改动理由 |
|------|--------|---------|---------|
| rollout_temp | 0.7 | **1.0** | 覆盖 t0.9 部署口径（温度鲁棒根因） |
| ROLLOUT_N | 6 | **12** | 稀有破功尾巴必须在 group 内出现 |
| KL_CLEAN | 0.002 | **0.008** | 锚紧 actor，防 RL 把 clean 也练成拒绝机器 |
| KL_ATTACK | 0.002 | **0.002** | 留空间让防御移动 |
| actor | step0 | **v6.9c merged** | RM 起点最高（0.271 > v6.9b 0.234），动作/对话最富 |
| break_pen | -5（全局） | **-5（逐样本）** | 单条破功立刻惩罚 |
| clean_q | 0 | **λ=0.25** | attack 非 break 时加 clean 结构分，偏向"在角色内有戏化解" |

### 8.3 canary 门（clean 防降双保险）

设计中的结构保险（per-source KL + clean prompt 只用 clean_q）只是一道门，加 canary probe：

```
step30-50 时跑一次 probe vs actor：
  clean 三轴不退 且 attack 破功率趋降 → 放行到 100-200 步
  任一失败 → 停，报告，不擅自调参
```

---

## 硬件约束与显存账

**硬件底牌**：8×RTX 5090 PCIe，32GB/卡，**无 NVLink**，NUMA0=GPU0-3 / NUMA1=GPU4-7（GPU 间带宽：NUMA 内 SHM ~50GB/s > QPI ~5-10GB/s）

### RM 训练（ZeRO-2 + NF4）

```
优化器状态（ZeRO-2，不分片）：26B × 2 bytes × 2 = ~15.4GB/卡
NF4 权重（4-bit）：26B × 0.5 bytes = ~4.5GB/卡
激活梯度（批次）：< 1GB
总计：≈ 20GB/卡，32GB 卡有余量 ✓
```

选 ZeRO-2 不用 ZeRO-3：ZeRO-3 引入 all-gather 前向 + reduce-scatter 反向，通信量额外 50%；无 NVLink 下 QPI 带宽成为瓶颈；ZeRO-2 优化器状态不跨卡，通信量仅 all-reduce 梯度（ZeRO-2 ≈ ZeRO-1 通信）。

### RL Actor（verl + vLLM colocate）

```
Actor 权重（bf16）：26B × 2 bytes ≈ 12.2GB/卡（ZeRO-3 分片到 4 卡）
vLLM 权重（虚拟地址预留，睡眠时 0）：12.2GB 预留
激活 + KV cache：~5GB
总计：≈ 19.6GB/卡（param_offload 后），余量 12.4GB ✓
```

**vLLM cumem 语义关键点**：vLLM 不是"用了多少显存"，而是"预留了虚拟地址，释放物理页"。`sleep_level=0`（只释放 KV cache），`sleep_level=1`（释放权重物理页，但虚拟地址仍占着）。`mem_fraction_static=0.78` 死在这里：虚拟地址和 actor 权重加起来超出物理显存。

**update_weights 必须串行 GPU**：Gemma4 vision tower 在 GPU0（全量不分片），完成 LoRA merge 后须按 GPU0→1→2→3 顺序同步，不能并发（文档未提，实测坑）。

### FSDP merge 不可行性

```
26B × 2（bf16）× 3（FSDP全量状态）+ LoRA ≈ 51.7GB > 32GB/卡上限
```

全量 FSDP merge 物理上不可能。解法：LoRA 通过 `merge_and_unload()` 合并到主权重后，用 `force_text_only=True` 去掉 vision tower，再 safetensors 序列化（不走 FSDP merge 路径）。

---

## 实验卡模板

每轮实验填一张卡。目的：不让归因停在"感觉是数据问题"，强制说清变量。

```markdown
## EXP ID: EXP_YYYYMMDD_XXX

### 假设
这轮实验要验证什么？（单一假设，不要同时验多个）

### 变动的单一变量
- 数据：新增/删除/替换了什么
- 奖励：改了哪个 reward 项，改了多少
- 参数：改了哪个超参，改了多少
- 算法：改了哪个算法组件

### 评估口径
- clean probe：N=200，seed=42，t=0.9
- attack probe：68 条产品攻击，t=0.9
- 误报去除规则：（echo / code_block / 非身份三类）

### 训练观测
| 指标 | step0/actor 基线 | 本轮最佳 | Δ |
|------|----------------|---------|---|
| attack 破功率 | | | |
| attack RM（split） | | | |
| clean 动作密度 | | | |
| clean 对话密度 | | | |
| KL | | | |
| response_length | | | |

### 结果
- [ ] 成功（假设验证）
- [ ] 失败（假设证伪）
- [ ] 部分（哪个子假设成/失败）

### 归因
失败来自：
- 数据缺口（什么类型攻击 0 覆盖）
- 奖励信号（什么情况下 reward 方向错）
- 算法局限（哪个指标证明 RL/SFT 已到天花板）
- 评估协议（probe 能否代表产品分布）

### 下一步
（单一、可测、可归因的下一个实验）
```

---

## Credit Assignment 视角：把上述阶段错配读成一条主线

> 本节是对全流程的一个归因补充，完整分析见 [深读 8 — Character RLHF 中的 Credit Assignment](deep_dives/08_credit_assignment_in_character_rlhf.md)。

前面的每个阶段，reward 都是 **sequence-level**（RM + rule reward 给整条回复一个标量），而 GRPO 的 actor 更新发生在 **token-level** logprob 上。这个错配把上面几处失败串成同一条线：

| 现象（本文档章节） | 错配落在哪一层 | 归因 | 版本对比 | 工程决策 |
|---|---|---|---|---|
| anti_refusal +3.0 反向破角（Stage 4.2） | 组件 / sequence | credit 分给了错误 reward 组件与破角 span | base 27% → RL 55%/91% | anti-OOC 硬罚 -5 + 撤销 bonus，重排优先级 |
| clean RL 提不动（Stage 4.3 / 钥匙 2） | group | clean 跨度 0.037 → group 无方差 → advantage≈噪声 | clean 0.037 vs attack 0.36 | 富度归 SFT、防御归 RL |
| length hacking（Stage 5.2） | token | 正 advantage 逐 token 生效，长序列垄断 credit 质量 | 有无 GSPO | GSPO 序列级等权 + MAX_LEN=768 |
| clean/attack 混训（Stage 8.2） | source | 两源 credit 混叠，attack 更新会带跑 clean | 全局 KL vs per-source KL | per-source KL（clean 0.008 / attack 0.002） |
| v6.10 attack 未降（Stage 8） | data / RM / token-span 候选集 | -5 是整条回复级，但尚未跑 span 对照；targeted data 与 RM 区分度也未排除 | v6.9c 33.8% vs v6.10 36.8% | 停线，转 targeted data / RM v7 / span diagnostic（均需验证） |

**下一步该动哪一层**：不是"再 RL 几步"，而是并行验证 targeted data、RM v7 区分度和 sequence-to-span credit；其中 span diagnostic 尚未跑，不能提前宣布其为唯一根因。

---

## 一句话方法论

能力目标决定评估协议，评估协议决定数据构造，数据构造决定 RM 设计，RM 设计决定 RL 奖励有没有梯度——**每一步都是上一步归因后的唯一合理答案，不是超参数调优**。
