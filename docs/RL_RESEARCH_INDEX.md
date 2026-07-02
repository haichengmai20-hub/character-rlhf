# RL Research Index

> 用途：作为 ChatJoy Character RLHF 项目的后训练研究索引。
> 这个文件不追求展开所有细节，而是把 SFT / RM / GRPO / PPO / DPO / RLVR / Agentic RL / Eval / Deploy 的关系串成可导航结构。
> 当前仓库定位：面试复盘与研究能力展示，不要求继续跑训练或补单测。

---

## 0. Master Chain

后训练项目的主链路：

```text
业务失败模式定义
→ 数据构造
→ SFT 行为初始化
→ rollout / sampling
→ logprob 计算
→ reward 计算
→ advantage / return 估计
→ loss 计算
→ 参数更新
→ 自动评测
→ 部署
→ 线上监控 / bad case 回流
```

在 Character RLHF 中，对应到：

```text
roledrop / language mismatch / low richness / refusal / format drift
→ 真实聊天分布重建
→ SFT 注入角色语言与格式
→ actor 生成多候选 response
→ old/new/ref logprob 计算
→ RM + rule reward 打分
→ GRPO group-relative advantage
→ token-level actor loss
→ actor update
→ clean / attack / multilingual probe
→ vLLM serving
→ bad case 归因与下一轮迭代
```

---

## 1. Core Concepts Map

| Concept | Statistical / Optimization Meaning | ChatJoy Mapping | Key Diagnostic |
|---|---|---|---|
| SFT | Conditional MLE | 学语言、格式、角色 voice、基本富度 | token NLL、格式通过率、语言一致性 |
| Reward Model | Pairwise preference model | chosen/rejected 角色回复偏好建模 | acc、gap、reward_std、axis variance |
| PPO | Clipped importance-sampling policy optimization | 可作为 RLHF 标准基线 | ratio、clip fraction、KL、value loss |
| GRPO | Group-relative advantage estimation | 同 prompt 多 response 相对优化 | group_reward_std、advantage_std |
| DPO | KL-regularized preference optimization without explicit RM | offline 偏好优化候选方案 | chosen/rejected margin、KL drift |
| RLVR | Verifier / rule reward RL | anti-OOC、格式规则、语言规则 | rule precision / recall、误伤率 |
| Agentic RL | Trajectory-level policy optimization | 可迁移到 tool-use agent harness | trajectory reward、action credit |
| Auto Eval | Finite-sample statistical inference | clean / attack / multilingual probe | CI、source breakdown、detector bias |
| vLLM Deploy | Serving and decoding control | 多版本合并部署、采样参数 sweep | latency、OOM、temperature sensitivity |

---

## 2. SFT Research Notes

### 2.1 Definition

SFT 数据：

```text
D_sft = {(x_i, y_i)}
```

目标函数：

```text
L_SFT(θ) = - E_(x,y)~D [ Σ_t log π_θ(y_t | x, y_<t) ]
```

含义：给定 prompt / history / persona 后，对目标 response 做最大似然估计。

---

### 2.2 Role in Character RLHF

SFT 负责注入 target behavior support：

- 多语言回应习惯；
- 角色 voice；
- RP 格式；
- 情绪与陪伴感；
- 基础安全边界；
- 常见场景下的稳定行为。

经验判断：

```text
如果模型完全采不到某类行为，优先补 SFT / data。
如果模型能采到但概率不够高，再考虑 RL 重加权。
```

---

### 2.3 SFT Failure Modes

| Failure | Cause | Better Fix |
|---|---|---|
| 回复短、干、模板化 | 高质量长回复 support 不足 | SFT teacher distillation / data augmentation |
| 多语言混乱 | language-specific 数据不足 | 分语言采样与重加权 |
| 角色 voice 弱 | persona-conditioned 数据弱 | 角色卡绑定与同角色多轮样本 |
| 防御差 | attack 数据 support 不足 | attack SFT + RM/RL |
| 过度拒绝 | 拒绝样本比例或安全模板污染 | 调数据分布与偏好对 |

---

## 3. Reward Model Research Notes

### 3.1 Preference Objective

偏好对：

```text
(x, y^+, y^-)
```

Reward Model：

```text
r_φ(x, y) ∈ R
```

Bradley-Terry 概率：

```text
P(y^+ ≻ y^- | x)
= sigmoid(r_φ(x, y^+) - r_φ(x, y^-))
```

RM loss：

```text
L_RM(φ)
= - log sigmoid(r_φ(x, y^+) - r_φ(x, y^-))
```

---

### 3.2 RM Is a Reward Estimator, Not a Truth Oracle

RM 的价值不是“分类准确率高”，而是给 RL 提供可优化、方向正确、方差足够的 reward signal。

关键诊断：

| Metric | Why It Matters |
|---|---|
| accuracy | 基础偏好分类能力 |
| chosen/rejected gap | 分数间隔是否足够大 |
| reward_std | rollout 分布上是否有有效方差 |
| axis-wise variance | 目标轴是否可优化 |
| saturation | 是否所有样本都接近同一极端分数 |
| source breakdown | clean / attack / multilingual 是否分布一致 |
| judge agreement | LLM judge / human rubric 是否一致 |

---

### 3.3 RM Failure Modes

| Failure | Explanation | Consequence |
|---|---|---|
| RM acc 高但 RL 不动 | actor rollout 上 reward_std 太低 | GRPO advantage 近似噪声 |
| RM 偏爱长回复 | reward 与 length 高相关 | length hacking |
| RM 忽略破角 | identity disclosure 没被强识别 | RL 放大 roledrop |
| RM 多目标混叠 | 富度、配合度、防御互相冲突 | clean 上升 attack 下降或相反 |
| RM 语言校准不一致 | 不同语言分数尺度不同 | 低资源语言被压制 |

---

## 4. PPO / GRPO Research Notes

### 4.1 PPO Minimal Flow

```text
rollout actor 采样 response
→ reward / value 计算
→ return / advantage 估计
→ old_logprob 固定
→ current actor 计算 new_logprob
→ ratio = exp(new - old)
→ clipped surrogate loss
→ critic loss
→ KL / entropy regularization
→ actor + critic update
```

PPO 核心：

```text
ρ_t(θ) = π_θ(a_t|s_t) / π_old(a_t|s_t)
```

clip 目标：

```text
min(ρ_t A_t, clip(ρ_t, 1-ε, 1+ε) A_t)
```

---

### 4.2 GRPO Minimal Flow

```text
同一个 prompt 采样 G 个 responses
→ 每个 response 得到 sequence reward
→ 组内 mean/std 标准化 reward
→ 得到 group-relative advantage
→ 用 token-level ratio 更新 actor
→ reference KL 约束策略漂移
```

GRPO advantage：

```text
A_i = (R_i - mean(R_group)) / (std(R_group) + ε)
```

与 PPO 的关键差异：

| Item | PPO | GRPO |
|---|---|---|
| critic | 有 | 通常无 |
| baseline | learned value | group reward mean |
| advantage | GAE / return - value | group-normalized reward |
| memory | actor + ref + critic + RM | actor + ref + RM |
| sensitivity | critic quality | group reward variance |

---

### 4.3 GRPO Entry Conditions

GRPO 不应该无条件启动。

建议入口门：

```text
Gate A: evaluation 能否看见目标 failure？
Gate B: RM / rule reward 在目标轴上是否有方差？
Gate C: actor 是否能采到好坏有差异的 responses？
Gate D: 目标行为是否不是单纯 SFT 数据缺口？
Gate E: credit 是否能定位到真实错误行为？
Gate F: RL 是否不会明显伤害其他核心轴？
```

---

## 5. Logprob Flow Index

详细见：`docs/GRPO_LOGPROB_FLOW.md`。

核心变量：

| Variable | Computed By | Timing | Usage |
|---|---|---|---|
| old_logprob | rollout actor / old policy | rollout 阶段或 rollout 后冻结计算 | importance ratio denominator |
| new_logprob | current actor | actor update 时重新 forward | importance ratio numerator |
| ref_logprob | frozen reference model | reward 后 / update 前 | KL regularization |
| reward | RM + rule reward | response 生成后 | sequence-level score |
| advantage | group normalization | reward 后 | policy gradient weight |

核心公式：

```text
ratio = exp(new_logprob - old_logprob)
```

---

## 6. Credit Assignment Index

详细见：`docs/CREDIT_ASSIGNMENT_TAXONOMY.md`。

核心问题：

```text
Reward 是 sequence-level，
Advantage 是 group / sequence-level，
Loss 是 token-level。
```

因此需要分析：

| Level | Question |
|---|---|
| token-level | 哪些 token 被上调 / 下调？ |
| span-level | 哪个错误片段触发 reward / penalty？ |
| sequence-level | 整条 response 为什么得分高 / 低？ |
| group-level | 同 prompt 下 response 如何相对排序？ |
| source-level | clean / attack / multilingual 哪类样本在驱动更新？ |
| batch-level | batch 平均是否掩盖分源冲突？ |

---

## 7. RLVR / Rule Reward Notes

RLVR 可以理解为 verifier / rule reward 驱动的 RL。

Character RLHF 中的 rule reward 包括：

- anti-OOC / identity disclosure detector；
- anti-refusal reward；
- format reward；
- language match reward；
- length shaping；
- repetition penalty；
- chain-of-thought / meta disclosure penalty。

关键原则：

```text
硬约束优先于软偏好。
identity consistency > non-refusal helpfulness > richness
```

如果顺序错了，reward component 会把 credit 分给错误行为。

---

## 8. Agentic RL Transfer Index

Character RLHF 可以迁移到 Agentic RL：

| Character RLHF | Agentic RL |
|---|---|
| prompt / persona | environment state / instruction hierarchy |
| response token | action / tool call / text action |
| roledrop | instruction hierarchy break / prompt injection |
| anti-OOC detector | verifier / safety rule |
| sequence reward | trajectory reward |
| length hacking | useless tool-call hacking |
| clean / attack probe | task suite / adversarial tasks |
| bad case 回流 | replay / data flywheel |

Agentic RL 链路：

```text
环境定义
→ 任务采样
→ Agent 执行
→ 工具调用
→ 轨迹记录
→ reward / verifier 计算
→ credit assignment
→ 数据回流
→ 训练更新
→ 自动评测
→ 部署迭代
```

---

## 9. Evaluation Research Index

评测不是一个最终分数，而是有限样本统计推断。

建议分层：

| Dimension | Examples |
|---|---|
| source | clean / attack / multilingual |
| language | en / zh / ja / ko / etc. |
| character | top characters / long-tail characters |
| failure type | roledrop / refusal / format / language / repetition |
| decoding | temperature / top_p / min_p / presence penalty |
| detector | regex / LLM judge / manual true-break filter |

评测输出应该至少包括：

```text
metric mean
source breakdown
failure type breakdown
sample size
confidence interval or bootstrap range
representative cases
bad case attribution
```

---

## 10. Interview Storyline

面试讲述顺序：

```text
1. 业务问题：多语言角色扮演模型容易破角、混语言、回复干、过度拒绝。
2. 数据方案：从线上会话重建 user-character-history 条件分布。
3. SFT：用 MLE 注入角色行为 support。
4. RM：用偏好对训练 reward estimator，并诊断 acc/gap/std/axis variance。
5. GRPO：用 group-relative advantage 做策略优化，省 critic，但要求 reward 有方差。
6. 失败归因：发现 reward hacking、anti-refusal 误奖、clean/attack 分布不一致。
7. 评测：用 clean/attack/multilingual fixed probe 做同尺比较和 true-break 过滤。
8. 部署：vLLM serving + decoding sweep，但不把采样参数当作训练信号修复。
9. 反思：新增 credit assignment 视角，解释 sequence reward 如何影响 token-level update。
```

一句话总结能力：

```text
我能把线上 bad case 转化为数据、reward、advantage、loss 和评测指标，并判断一次后训练失败到底是数据分布、RM、reward、RL 入口门、credit assignment 还是评测口径的问题。
```
