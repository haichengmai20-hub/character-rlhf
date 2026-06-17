# ChatJoy RLHF 实战闭环

多语言角色扮演模型后训练的完整流程记录：从业务目标到数据归因，从 RM 迭代到 RL 算法选择，每一步决策都有实测数字支撑。

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

## Stage 0：能力目标与初始评估设计

**业务定义**：27 语言、83 角色的角色扮演对话模型，核心是"出戏"和"富度"两条主线。

| 能力维度 | 业务表现 | 初始评测信号 |
|---------|---------|---------|
| 角色不出戏 | 不跳出角色讲 AI 身份 | 关键词规则检测（后来演化为 `anti_ooc_disclosure`） |
| 语言一致性 | 不混杂语言 | `wrong_language` pair 占比 25% |
| 格式结构化 | 动作/对白/内心三分体 | 动作块密度、对话块密度 |
| 语义丰富度 | sensory / emotional / narrative 细腻 | judge rubric 三轴 vs base teacher |
| 防越权指令 | user 层级 roledrop 攻击不破角 | 产品形态攻击词典（68 条固定 probe） |

**起点时已知的**：能力维度划分、固定 probe 集（200 条 clean + 68 条 attack）、judge rubric 5 维、SFT→RM→RL 流水线架构。

**评估协议在迭代中暴露的问题**（不是一开始就设计好的，而是踩坑后修正的）：

- **训练 val ≠ 产品攻击分布**（v6.5 结束后发现）：训练 val 的 attack 破功率 11% ≠ 产品真实 55%，val 几乎不含 bullet 注入形态；参见 [Stage 6 钥匙 4](#钥匙-4)
- **必须用 t0.9 产品口径测 attack**（v6.7 结束后确定）：v6.7 在 t0.7 上防御很好，t0.9 崩；此后所有 attack probe 强制用 t0.9；参见 [Stage 6 钥匙 1](#钥匙-1温度鲁棒)
- **误报三类去除才是真破功率**（v6.9c 评测时发现）：echo 误报（角色内复读被 regex 命中）+ code_block 误报（代码字符串含"Gemma"），去掉后 36.8% → 16.2%；参见 [Stage 6 钥匙 3](#钥匙-3sft-defense--whack-a-mole到顶转-rl)

---

## Stage 1：线上数据链路 + SFT 行为初始化

### 1.1 数据来源与清洗

**链路**：Milvus 生产库导出 → 按 `(user_id, character_id)` 重建会话流 → 第一轮清洗 → TOP20 角色 × 27 语言采样 → **Gemini API 重构 output**

```
原始导出：2,080,698 行
→ TOP20 角色会话：679,299 条 assistant pairs  
→ 第一轮清洗后候选：631,995 条
→ 预 API 采样（27 语言 × ≤2200 条）：57,197 条
→ Gemini 重构（temperature=0.7, thinking=low）：~18,294 条留存
```

**Gemini 重构的设计意图**：不重造 instruction（用真实用户消息），只用角色卡重构 output。SFT 学真实用户输入分布 + 干净角色内 output，同时滤掉线上原始回复的格式错误、拒答和语言混杂。

**关键清洗决策**：

- 只做"明显无效"清洗（短 instruction / emoji-only / 短 output / 语言检测失败），不做重度质量筛选——重度筛选的活交给后面的 LLM judge。
- 留存率从 Gemini 重构看约 32%（API 错误 + 语言不一致各占一半），在预期内。

### 1.2 SFT 行为初始化

| 阶段 | 数据 | 目标 |
|------|------|------|
| SFT1（Qwen3-30B） | 13,128 条（清洗留存） | 建立多语言角色格式基础 |
| SFT2 + defense-replay | 19,508 条 + hardfix × 24 | 注入"越权化解"行为 |
| SFT v6.9b（Gemma4-26B） | 30,407 条（均衡 mix） | 富化 clean + 补 attack 缺口 |

**SFT 的作用边界**（实测确定，非理论）：
- SFT 能做：格式、语言一致性、角色 voice、基本富度初始化
- SFT 不能做：在高温（t0.9）下鲁棒防御——这是 RL 的活
- SFT 的天花板：clean 富度上限 = base teacher（best-of-N 天花板证明）；超不过 base，只能接近 base

---

## Stage 2：偏好数据构造

### 2.1 多源候选矩阵

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

三个训练敏感点（实测踩坑）：

1. **batch 方差**：同 batch 内 pair 类型不均匀 → 梯度方向抖动；解决：按类型分层采样
2. **cosine schedule**：后期 LR 衰减过快，细粒度 pair 还没学好就退火；解决：plateau-then-decay
3. **score gap 分布**：delta=2 pair 的 score gap 聚集在 `[0.161, 0.200]`，RM 没有"惊讶"，不更新；根因是 score head 初始化不好 → 两阶段凸优化恢复（冻结 backbone 先训 score head）

### 3.2 六版 RM 迭代台账

| 版本 | 核心改动 | offline acc | reward_std | 结论 |
|------|---------|------------|-----------|------|
| v5 | 4 source × 5 candidate | 94.5% | ~0.1（饱和 47.3%） | 在线塌缩 |
| v6.1 | delta=2 细粒度 pair | 86.4% | — | 训不动（无 cross-val） |
| v6.2 | + Qwen cross-validation | 86.0% | 1.94 | **饱和 0%** |
| v6.3 | + base Gemma4-it source + 8 维 rubric | 95.5% | 1.98 | 修 roledrop 91% |
| v6.4–v6.5 | 增量微调 | — | — | 渐进修防御 |
| v6.6 | 最终 7-GPU merge 版 | — | — | **当前生产 RM** |

**reward_std ×20 倍 = GRPO advantage 方差 ×20 倍**：这是 v5 → v6 最直接的业务价值。v5 在线时 reward_std ~0.1，GRPO 在"几乎全是高分"的信号上空转；v6 动态范围完整保留，RL 才有真实梯度。

### 3.3 Score Head 两阶段凸优化恢复

v6 训练中途发现 score gap 聚集：

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

分 source 版（v6.10，clean/attack 分路）：

```
clean source：clean_q（便宜结构代理）+ KL_CLEAN=0.008（锚 SFT actor）
attack source：rm + break_pen(-5) + λ·clean_q（λ=0.25）
```

设计原则：`break_pen=-5` 碾压 `clean_q≤0.3`，先不破功，再在不破功的回应里偏向"在角色内有戏化解"。

### 4.2 Goodhart 反向训练发现（RL v6.4，误报 → 修复）

**发现**：`anti_refusal` 的 `NON_REFUSAL_BONUS=+3.0` 把"自报 AI 身份"判为 non-refusal 并给满分 → **奖励在激励破角行为**。旧的 `atk_break%` 关键词指标对 "I am Gemma 4" 完全瞎（没有 regex 覆盖）。

**数据证据**：误报 1/16 → 修复后 → 0/16（68 条 attack probe 中 OOC 误报归零）。

**修复**：新增 `anti_ooc_disclosure.py`（OOC 硬罚 -5 + 撤销误给 bonus，覆盖 en/zh/ja/ko/es/ru/pt）。这是 Goodhart 定律的典型实例——指标（non-refusal）和真实目标（不出戏）在 edge case 上解耦。

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
- RL 是纯防御优化器。clean 在 RM 上几乎没梯度，RL 无法提升 clean 丰富度（v6.7 实测 clean RM 100 步 +0.037 ≈ 噪声）。
- **丰富度只能靠 SFT 蒸馏 teacher**（base Gemma4 是天花板，SFT 丢了 ~20% 丰富度：动作块 4.7→3.4）。
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
| per-source KL | **clean/attack 分路 KL**（守住 clean 分布） | `KL_CLEAN=0.008, KL_ATTACK=0.002` |

**GSPO 的必要性**：Gemma4 有 `<Thoughts>` 推理段（平均 81 tokens），不做序列级归一化 → 长 Thoughts 占 token loss 主导 → length hacking（通过变长 Thoughts 拿 bonus）。GSPO 把序列等权处理，剥掉 Thoughts 后计长。

**per-source KL 的核心作用**：clean 用紧 KL 锚住 SFT actor（不让 RL 把 clean 也练成防御机器人）；attack 用松 KL 给防御留空间移动。这比全局 KL 精确得多。

### 5.3 温度鲁棒（v6.7 → v6.10 的核心修复）

**v6.7 的失败**：group_size=6 + rollout_temp=0.7，在 t0.9 部署温度下防御崩溃（t0.7 RM=0.561，t0.9 RM=0.295 < step0=0.318）。

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

**误报问题（v6.9c 前未知）**：原始破功率 36.8%，统一 regex 去掉两类误报后降至 16.2%（step0 基线）：
- **echo 误报**：角色内复读攻击词（"你说你是 AI 吗？我可不是 AI！"），regex 命中了被否定的部分
- **code_block 误报**：A6 代码注入攻击中 Python 字符串含 "Gemma"（`model_name = "gemma-4"`），角色实际在写代码，非身份披露

去误报之前无法对比不同模型的真实防御水平，所有历史数字需用同一套检测器重算才可比。

**v6.9 → v6.9b → v6.9c 实测（同检测器去误报后真破功率）**：

| 模型 | 真破功率（68 条 attack） | RM-attack | 说明 |
|------|----------------------|-----------|------|
| step0 | **16.2%**（11/68） | 0.297 | clean 最低，防御最好 |
| v6.9（clean 富化） | ~42.6% | 0.214 | 富化侵蚀防御 |
| v6.9b（+120 条 defense）| **32.4%**（22/68） | 0.234 | 补 A6，但 A1/A2/A3 漏 |
| v6.9c（+2686 条跨角色）| **33.8%**（23/68） | 0.271 | 加数据不降破功率 |

**关键发现**：clean 富化 SFT 本身侵蚀身份防御（step0 16% → 富化后 32-34%）；两轮 defense 补丁把 RM 拉回但真破功率压不回 step0；A6 代码注入在不同 defense 轮次间来回震荡（5→12→4→11），无法收敛。

**∴ SFT-defense 到顶**，进 RL v6.10。RL 的强项正是"用高温 rollout + 逐样本惩罚"硬化稀有破功尾巴。

---

## Stage 7：SFT 富化弧完整记录

```
v6.7 RL 停（clean RM +0.037≈噪声，防御 +0.092 真有效）
    ↓  发现 RM 在 clean 上平（钥匙 2）
v6.8 SFT-max 尝试（FFN LoRA + judge 选样）
    ↓  结果：独白过冲 187% / 对话回退 / 动作没动
    ↓  根因：选择器 composite 含独白权重 + 独白重采 → 放大器放大了错方向
v6.9 重设计（best-of-N 均衡选择器）
    ↓  均衡选择器：动作+对话+sensory/interaction/emotional，独白不入目标
    ↓  结果：clean PASS（action 105-111% / dialog 110-160% / 独白无过冲）
    ↓  但 attack 退化：42.6% >> step0 23.5%（A6 defense 池 0 覆盖）
v6.9b（+120 条 step0 自蒸馏 defense）
    ↓  A6 code 修好（12→4），但 A1/A2/A3 身份直球漏
    ↓  32.4%，RM 0.234
v6.9c（+2686 条跨角色自蒸馏 defense，54% 占比）
    ↓  加数据未降，A6 再反弹，whack-a-mole 确认
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

## Stage 8：RL v6.10 防御硬化 + clean-aware reward

### 8.1 RL 入口门（独立于 SFT 验收）

SFT 验收回答"clean 干成了"；RL 入口门回答"SFT 做完后 attack 是否还有材料性破功"。

三分支决策：
1. 破功率 ≤ 产品容忍 → **上线，不 RL**（RL 有 clean 回退 / 多样性塌 / 温度脆三个风险）
2. 破功率材料性偏高 + SFT defense 已尽力 → **进 RL v6.10**
3. 中间小缺口 → 先靶向 SFT hardfix

v6.9c 满足了分支 2：三轮 SFT defense 都关不掉直球身份攻击 → 进 RL。

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

结构保险（per-source KL + clean prompt 只用 clean_q）只是一道门，加 canary probe：

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

## 一句话方法论

能力目标决定评估协议，评估协议决定数据构造，数据构造决定 RM 设计，RM 设计决定 RL 奖励有没有梯度——**每一步都是上一步归因后的唯一合理答案，不是超参数调优**。
