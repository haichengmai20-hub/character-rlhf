---
name: chatjoy-business-ux-tech-mapping
description: ChatJoy 项目 Gemma4 Base → SFT → RM → RL 全链路对 8 个业务体验维度的提升路径、技术手段、关键非显设计点与已知局限
metadata: 
  node_type: memory
  type: reference
  originSessionId: f1d698de-5fac-4322-8516-43346d7dabd5
---

# ChatJoy 业务体验提升 × 技术手段映射

记录每个业务体验维度从 Gemma4 Base 出发，经过 SFT / RM（v5 → v6.2 cross-validated）/ RL 三阶段后的提升幅度与对应技术设计。**关注非显的设计决策**，可推导的代码事实（数据路径、超参数）参见 [[project-chatjoy-overview]] 和 [[training-runbooks]]；RM v6 fine-grained 迭代细节见 [[chatjoy-rm-v6-finegrained-handoff]]。

## 8 维业务体验 × 各阶段贡献

RM 列同时列出 v5（已上线）和 v6 fine-grained（v6.2 cross-validated 训练中）各自贡献。其中维度 1-6 由 v5 RM 已覆盖；7、8 维由 v6 引入。

| 维度 | Base 表现 | SFT 贡献 | RM 贡献（v5 / v6 新增） | RL 贡献 | 综合 |
|---|---|---|---|---|---|
| 1. **格式结构化**（动作/对白/内心） | 依赖 system，格式抖动 | 19,508 条结构化样本 × 2 epoch | v5: format 0.30 含 5 子分（含 alternation 交织）/ v6: 沿用 + 多温度 pair 强化 | format+role 合计 0.60 → GRPO 主梯度 | **明确提升** |
| 2. **多语言一致性**（27 语言）| 跨语漂移偶有 | 27 语言 × TOP20 = 57,197 预 API → 19,508 SFT | v5: language 0.25 + `wrong_language` 占 25% pair / v6: 沿用 | language reward 进 online RM | **明确提升** |
| 3. **角色身份/性格匹配** | 能读 system 但无强对齐训练 | 17 个真实生产角色 in-distribution | v5: `wrong_role` 占 25%（B 真实 system gen + A system 训）学**统计关联** / v6: rubric "role fidelity voice 一致性" + 多温度 pair 学**同角色细分** | format+role 合计 0.60 | **明确提升（见过）+ 可迁移（未见性格）**；v6 进一步升级 |
| 4. **重复文案抑制** | 预训练多样性较好 | 数据已 anti_repetition 过滤 | length 0.15 含 0.20×anti_repetition | MAX_RESPONSE_LENGTH=768 + GSPO seq-mean | **明确提升** |
| 5. **情感贴合度**（细腻情绪表达）| 套路化（笑/哭/叹气循环）| ➖ | **v6 only**: rubric "natural style" — fluent/emotionally fitting/不机器人/不政策模板 | 经 v6 RM 信号传入 GRPO | **v6 引入**；冷笑 vs 苦笑 vs 嘲讽能被区分 |
| 6. **剧情推动力** | 抢戏 or 拖戏 | ➖ | **v6 only**: rubric "interaction progression" — 自然推进场景且不抢用户主动权 | 经 v6 RM 信号传入 GRPO | **v6 引入**；不再替用户决定剧情下一步 |
| 7. **拒绝率**（涉黄角色配合）| Google safety 高拒绝 | ➖ 数据已 safety 过滤 | ➖ 未训 | anti_refusal_reward: keyword `-1.0` / format-short `-0.25` | **部分（仅 RL 兜底）** |
| 8. **多轮记忆/上下文** | 混合 sliding-window attention 较好 | ➖ 单轮为主 | ➖ 无专项 | ➖ 无专项 | **持平/轻微下降**（基线侥幸） |

## 关键非显设计点（**不读这份 md 难推出**）

### 1. wrong_role 的语义关联设计（v5 RM 最关键的一步，v6 沿用）

**构造**：用**角色 B 的真实 system prompt**（不是合成）让模型自然以 B 身份回复同一条 user 消息；得到 rejected。
**训练**：RM 的 prompt 用**角色 A 的 system prompt**，chosen=A 的回复，rejected=B 风格的回复。

**为什么这样设计**：RM 学到的不是"记住 17 个角色长啥样"，而是"system prompt 描述的性格 → 对应回复风格"的**语义关联**。让 RM 对**完全陌生**的生产角色（83-17=66 个未在 SFT 见过）也能泛化。

**替代方案为什么不行**：合成 system gen 或随机字符串当 wrong system，RM 学到的只是"system 不一致就扣分"的统计关联，无法捕捉语义。

### 2. score function 中 role 维度**有意不做语义比对**

`chatjoy_reward_utils.py` 的 role 计算：`0.75 × format_total + 0.25 × name_presence_score`——只看格式规范性 + 角色名是否出现，**不做 system 描述与回复内容的语义比对**。

**为什么有意**：
- 启发式打分速度快，pair 生成时可同步评估每次尝试
- 人设深度判断**由 RM 模型学习**（不是规则）
- SFT 已处理大部分人设对齐；pair 构建目标是"表面特征可区分"的负样本

**代价（v1 eval idx=55 体现）**：高品质错误角色 system 下生成的内容，v5 RM 反而会偏好——因为 v5 没学过"system 性格 vs 实际回复语气是否匹配"的深层语义。

**v6 如何补这块**：通过 fine-grained pair（同 system、同 user、不同温度/source）让 RM 学到**同一角色的不同响应中哪个更贴 voice**——v5 看不到"两个都是角色 A 风格"的细分，v6 能区分。

### 3. RM v6 的 5 source × 多温度候选矩阵

| Source | Temperature | top_p | 设计目的 |
|--------|------------|-------|---------|
| Gemma4 SFT base | 0.7 | 0.8 | 当前 actor 主分布 |
| Gemma4 SFT diverse | **1.0** | **0.95** | 当前 actor **高方差尾部** |
| Qwen SFT base | 0.7 | 0.8 | 替换 base 的对照 |
| RL step550 base | 0.7 | 0.8 | RL 已微调的主分布 |
| RL step550 diverse | **1.0** | **0.95** | RL 微调的高方差尾部 |

**Why**：同 prompt 下 5 个候选**横跨"模型差异 + 采样温度差异"两个轴**。RM 学到的不是"哪个模型好"，而是"在 actor 的真实输出分布里，**rl_step550_t10 vs rl_step550_t07** 哪个更细腻"——直接对应 RL rollout 的真实场景（同模型多次采样）。

### 4. Judge 评分 rubric 5 维（统一 ChatJoy rubric，blind judge）

`judge_candidates_llm_openai.py` 给 judge 的 rubric：

1. **Role fidelity**：保持角色 voice
2. **User relevance**：响应用户输入而非 generic filler
3. **Interaction progression**：自然推进场景，不抢用户主动权
4. **Natural style**：fluent / emotionally fitting / 不机器人 / 不政策模板
5. **Format and length quality**

Hard fails（错语言/拒答/空/乱码/leak system/替用户行动/不安全升级）排到末位。

**Blind judge**：把 candidates 顺序打乱后再发给 judge，隐藏 source_model 和 sampling，避免 judge 形成 source bias。

**已踩过的坑**：早期发现 Qwen 候选常被排差——经检查 Gemma/RL 内部 t07 vs t10 的细粒度 pair 仍然存在，**说明这是 judge 真的判 Qwen 风格不贴 ChatJoy roleplay，不是 source bias**。

### 5. Pair 构造的 delta=4 vs delta=2 双轨（v6 核心）

| Pair 类型 | rank | score_delta | 数量（v6.2） | 用途 |
|----------|------|------------|------|------|
| `best_vs_worst` (delta=4) | rank1 vs rank5 | ~2.0 | ~10k | 强信号，必学 |
| `best_vs_mid` (delta=2) | rank1 vs rank3 | ~0.5 | ~8.5k | 细粒度，**核心目标** |
| `good_vs_bad` (delta=2) | rank2 vs rank4 | ~0.5 | ~8.5k | 细粒度，**核心目标** |

**Why delta=2 是核心**：RL rollout 时，actor 同一 prompt 多次采样的 4 个候选**互相之间几乎都是 delta=2 级别的差异**。RM 必须在这种近似 pair 上能打分，GRPO advantage 才有有效信号。delta=4 是 ranking 锚点，delta=2 才是真正的"细粒度"能力。

### 6. Cross-validation（v6.2）—— 双 judge 同向过滤

**问题**：DeepSeek 在 delta=2 上自报 `confidence=high` 的 pair 仍有 ~25% 方向错误（self-reported confidence 不可信）。

**解决**：本地 Qwen-36B-A3B vLLM 独立 judge 同一批 prompt → 只保留两 judge 都同向的 pair。

**结果**：v6.2 cross-validated **27,287 pair**（保留率 73.1%）。delta=2,conf=high 中 29.4% 被 Qwen 否决——**这部分正是 v6.1 训不动的元凶**。

### 7. Score 饱和问题修复（v5 → v6 最直接、最可量化的业务价值）

v5 Gemma4 RM `|score|>5` 饱和比例 **47.3%**——一半样本的分数被截断到上下限，GRPO 在线打分时这些样本 advantage 信号塌缩。

v6 全程 `saturated=0%`，reward_std=**1.94→1.98**（v5 训练末期 reward_std ~0.1）。

**reward_std × ~20 倍** = GRPO advantage 方差 × 20 倍 = **在线 RL 信号强度的根本提升**。这是 v6 最直接的业务价值——让 GRPO 不再在"全是高分"的塌缩信号上空转。

### 8. Length reward hacking 多层防护（5 层）

| 层 | 机制 | 作用强度 |
|----|------|---------|
| 数据 | clean32_sft2 已 anti_repetition 过滤 | 间接 |
| RM | length 0.15 含 0.20×anti_repetition + v5 重写 length_mismatch_long 样本 | 弱 |
| Actor 硬截断 | `MAX_RESPONSE_LENGTH=768 tokens` | **最强** |
| 梯度聚合 | GSPO `loss_agg_mode=seq-mean-token-mean`（序列等权）| 强 |
| 策略 clip | DAPO 非对称 `clip_ratio_low=0.2, high=0.28` | 中 |

### 9. RM offline / online 对齐修复（已踩过的坑）

OpenRLHF 训练 RM 时对 `(prompt + chosen).rstrip("\n")` 后**在 EOS 对齐 token 上**打分。vLLM `/classify` 必须复现这个尾部处理，否则末 token pooling 位置不一致 → online RM score 整体异常偏负。

**修复**：`reward_loop.py` 对 RM prompt `rstrip("\n")` + 补 EOS。**未来任何换 vLLM 版本或 RM 重训都要复检**。

### 10. wrong_role / wrong_language / length_mismatch 类型专属权重 profile

pair 构造时不只是默认 format=0.30/role=0.30/language=0.25/length=0.15，而是按 rejected 类型切换：

| 类型 | format | role | language | length |
|---|---|---|---|---|
| wrong_role | 0.15 | **0.55** | 0.15 | 0.15 |
| wrong_language | 0.15 | 0.15 | **0.55** | 0.15 |
| length_mismatch_* | 0.20 | 0.20 | 0.20 | **0.40** |

**Why**：强化对应维度的信号，避免噪声 pair 训坏 RM。pair 构造同时要求 chosen 分数、总 gap、目标维度 gap 同时达阈值。

## 用户感知层面的 v5 → v6 差异预期

| 用户感知 | v5 阶段 | v6（v6.2 cross-validated）阶段 | 原因 |
|---------|---------|--------------------------------|------|
| 角色情绪表达 | 套路化（笑/哭/叹气循环）| 细腻（冷笑 vs 苦笑 vs 嘲讽）| rubric "natural style" + 多温度 pair |
| 剧情节奏 | 时而拖戏、时而抢戏 | 接住并推进，不替用户决定 | rubric "interaction progression" 显式约束 |
| 长会话粘性 | 容易出现 generic filler | 持续贴当前 context | rubric "user relevance" |
| 同角色重复使用一致性 | 偶尔风格漂 | 多次 rollout 都贴 voice | delta=2 pair 训练 RM 学到同 source 不同温度的细分 |
| GRPO advantage 信号强度 | reward_std ~0.1（塌缩）| reward_std ~1.98（**×20**）| 饱和 47.3% → 0%，dynamic range 完整保留 |

## 当前未覆盖维度（业务风险预警）

1. **多轮记忆**：SFT/RM/RL 全单轮，靠 Gemma4 mixed attention 侥幸维持。长会话用户可能体感"前后断裂"。需要单独多轮 SFT 数据集。
2. **涉黄宽松**：当前仅 RL 兜底 anti_refusal。若线上拒绝率 > 40% 需要先做 +500-2000 条涉黄 SFT 推动初始分布，否则 RL 信号不足以扭转。

## 当前迭代状态

v6.2 A+C（LR=9e-6, LoRA r=128, +mlp）8 卡训练中（2026-05-19 启动，~9h 完成）。目标 final eval acc 0.84+，**主要提升的就是第 5、6 维（情感贴合 + 剧情推动）的稳定信号**——这是 v5 RM 完全无法覆盖、必须依赖 v6 fine-grained 才能学到的能力。具体训练曲线与对照数据见 [[chatjoy-rm-v6-finegrained-handoff]]。
