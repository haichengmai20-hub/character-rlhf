# GPT Context Distilled

> 用途：把 GPT 对话中已经沉淀的 Character RLHF / GRPO / 后训练研究上下文压缩成稳定文档，供 VSCode 工作区、面试复盘、后续 AI 继续完善仓库时使用。
> 这不是原始聊天记录，也不是实验报告；它是项目知识资产的提炼版。
> 当前仓库定位：用于下一次实习 / 算法岗面试时提炼项目能力、研究深度和工程理解，不要求继续跑训练或补单测。

---

## 0. One-line Positioning

ChatJoy Character RLHF 是一个围绕多语言角色扮演对话模型的真实业务后训练闭环项目，核心价值不是“用了 SFT / RM / GRPO”，而是把线上业务失败模式、真实会话分布、偏好建模、reward 诊断、GRPO 优化、评测归因和部署反馈串成了一个可解释的后训练系统。

更短地说：

```text
Post-training is not model tuning.
It is signal engineering under product distribution shift.

后训练不是单纯调模型，而是在业务分布漂移下构造可优化信号。
```

---

## 1. Research Identity

本项目应该按“大模型后训练算法工程项目”理解，而不是普通 AI 产品项目。

研究重点：

- SFT
- Reward Model
- RLHF
- PPO
- GRPO
- DPO
- RLVR
- RLAIF
- Agentic RL
- Agent Harness
- 自动化评测
- vLLM 部署
- 多版本迭代
- 线上 bad case 回流

默认分析链路：

```text
数据构造
→ rollout / sampling
→ logprob 计算
→ reward 计算
→ advantage / return 估计
→ loss 计算
→ 参数更新
→ 评测
→ 部署
→ 线上监控
```

任何算法讨论都要尽量回答：

| 问题 | 需要厘清的对象 |
|---|---|
| 哪个模型在采样？ | rollout actor / old policy / vLLM engine |
| 哪个模型在更新？ | training actor / policy model |
| reference model 做什么？ | KL regularization baseline |
| RM 做什么？ | sequence-level reward estimator |
| critic 做什么？ | PPO 中估计 value，GRPO 中通常没有 |
| old_logprob 什么时候算？ | rollout actor 采样时或 rollout 后冻结计算 |
| new_logprob 什么时候算？ | actor update 时当前 actor 重新 forward |
| ref_logprob 什么时候算？ | frozen reference 在同一批 response 上计算 |
| reward 是哪个层级？ | 通常是 sequence / trajectory level |
| advantage 是哪个层级？ | PPO 可 token/step；GRPO 多为 group-normalized sequence advantage |
| loss 是哪个层级？ | token-level logprob loss 聚合到 sequence/batch |

---

## 2. ChatJoy Project Core Pipeline

项目主线可以抽象为：

```text
线上多语言角色扮演聊天数据
→ 按 user / character 重建真实会话分布
→ 绑定角色 system prompt / persona
→ SFT 行为初始化
→ 构造 chosen / rejected 偏好对
→ 训练 Reward Model
→ 用 RM + rule reward 驱动 GRPO / RLHF
→ clean / attack / multilingual probe 评测
→ vLLM 合并部署与采样参数 sweep
→ 线上 bad case 回流
→ 下一轮数据 / reward / 评测 / 算法迭代
```

业务目标不是单一“回复质量”，而是多目标：

| 目标 | 含义 | 常见 failure |
|---|---|---|
| 语言一致性 | 用户什么语言，角色尽量用对应语言回应 | 混语言、英文 fallback |
| 角色一致性 | 不自报 AI，不串角色，不泄露 system | roledrop / identity disclosure |
| RP 格式 | 动作、对白、心理描写格式稳定 | 变成通用助手格式 |
| 富度 / 陪伴感 | 回复有情绪、细节、角色 voice | 短、干、模板化 |
| 配合度 | 不无意义拒绝 | 过度安全拒绝、破坏陪伴体验 |
| 抗攻击 | 抵抗“忘掉角色”等 prompt injection | 承认自己是模型 / 助手 |

---

## 3. Algorithmic Interpretation

### 3.1 SFT = Maximum Likelihood Estimation

SFT 数据：

```text
D_sft = {(x_i, y_i)}
```

其中：

- `x_i`：system prompt + conversation history + user message
- `y_i`：目标角色回复
- `π_θ`：待训练 actor / policy model

目标函数：

```text
L_SFT(θ) = - E_(x,y)~D [ Σ_t log π_θ(y_t | x, y_<t) ]
```

统计学含义：条件语言模型的最大似然估计。

在 ChatJoy 中，SFT 的作用不是直接“优化偏好”，而是把目标行为 support 注入 actor：

```text
SFT controls support.
RL reweights support.
```

即：

- 如果 actor 从来采不到某类高质量 RP 行为，RL 很难凭空创造；
- 如果目标行为已经在采样分布里，RL 可以通过 reward 重加权提高概率；
- clean 富度、角色 voice、语言习惯通常更适合先由 SFT 注入；
- 防 roledrop / 攻击鲁棒性可在 RM 与 RL 阶段进一步强化。

---

### 3.2 Reward Model = Preference Probability Modeling

偏好数据：

```text
D_rm = {(x_i, y_i^+, y_i^-)}
```

其中：

- `y_i^+`：chosen response
- `y_i^-`：rejected response
- `r_φ(x, y)`：Reward Model 输出的标量分数

Bradley-Terry 形式：

```text
P(y^+ ≻ y^- | x)
= sigmoid(r_φ(x, y^+) - r_φ(x, y^-))
```

损失：

```text
L_RM(φ)
= - E_D [ log sigmoid(r_φ(x, y^+) - r_φ(x, y^-)) ]
```

关键判断：

```text
RM accuracy 高 ≠ RL signal 可用。
```

RL 可用性还取决于：

| 诊断项 | 含义 |
|---|---|
| chosen/rejected gap | chosen 是否显著高于 rejected |
| reward_std | actor rollout 分布上 reward 是否有方差 |
| axis variance | 目标轴是否可区分，例如 defense / richness / language |
| saturation | RM 是否所有样本都打到极高或极低 |
| judge agreement | 多 judge 是否方向一致 |
| source coverage | clean / attack / multilingual 是否覆盖真实业务分布 |
| rollout coverage | RM 是否能区分当前 actor 真实输出，而非只区分模板样本 |

---

### 3.3 GRPO = Group-relative Advantage Estimation

对同一个 prompt `x`，old policy / rollout actor 采样 `G` 个 response：

```text
y_1, ..., y_G ~ π_old(. | x)
```

每个 response 得到 sequence-level reward：

```text
R_i = R(x, y_i)
```

GRPO 使用组内均值和标准差构造 advantage：

```text
A_i = (R_i - mean(R_1, ..., R_G)) / (std(R_1, ..., R_G) + ε)
```

token-level ratio：

```text
ρ_i,t(θ)
= exp(log π_θ(y_i,t | x, y_i,<t) - log π_old(y_i,t | x, y_i,<t))
```

简化 actor loss：

```text
L_actor(θ)
= - E [ min(ρ_i,t A_i, clip(ρ_i,t, 1-ε, 1+ε) A_i) ]
  + β KL(π_θ || π_ref)
```

GRPO 的价值：省掉 critic，显存与实现更轻。

GRPO 的代价：更依赖 group 内 reward 方差与采样多样性。

核心判断：

```text
如果同 prompt 的 G 个 responses 在目标 reward 轴上没有差异，
则 group advantage 近似噪声，GRPO 更新方向也近似噪声。
```

---

### 3.4 PPO vs GRPO in This Project Context

| 模块 | PPO | GRPO | ChatJoy 含义 |
|---|---|---|---|
| baseline | critic value `V(s)` | group mean reward | GRPO 用组内相对分替代 value model |
| advantage | GAE / return - value | group-normalized reward | 更依赖同 prompt 多样本差异 |
| critic | 需要 | 通常不需要 | MoE / 大模型训练时节省显存 |
| reward | sequence / token 均可设计 | 常见 sequence reward | roleplay 里多为整条回复打分 |
| credit assignment | 可结合 value 逐步估计 | sequence advantage 广播到 token | 局部破角可能被全局惩罚 |
| 风险 | critic 不稳定 | reward_std 不足 / source 混叠 | clean/attack 混训时尤其明显 |

---

### 3.5 DPO Reflection

DPO 可以理解为 KL-regularized preference optimization 的重参数化形式，不需要显式训练 RM，也不需要在线 rollout。

在 Character RLHF 中，DPO 适合：

- offline preference pair 覆盖充分；
- 偏好边界清晰；
- 不需要持续采样当前 actor 的失败模式；
- 目标更偏风格、格式、语言、回答偏好。

但 DPO 不天然解决：

- product attack distribution shift；
- roledrop 的线上变体；
- reward hacking；
- 多目标冲突；
- 当前 actor rollout 中新出现的 bad case。

所以对 ChatJoy 来说，DPO 是可比较方案，但不能替代“分布诊断 + RM / rule reward + online-style rollout + GRPO”的完整闭环。

---

## 4. Core Research Findings from GPT Discussion

### 4.1 Data Distribution Matters More Than Template Volume

角色扮演不是单轮问答。回复质量依赖：

```text
P(response | user, character, language, history, scene, persona)
```

所以要尽量保留：

- user_id / character_id 条件关系；
- 多轮 history；
- 真实语言分布；
- 角色 system prompt；
- 业务场景中的攻击 / 破角分布。

破坏真实联合分布会导致模型学到边缘平均行为，变成通用助手而不是角色。

---

### 4.2 Reward Model Is Not a Universal Quality Function

RM 更像一个有训练分布、有 rubric、有 judge 偏差的 reward estimator。

如果 RM 只会判断“回复是否长、是否看起来完整”，它可能提升富度，却伤害防御。

如果 RM / rule reward 没有先约束身份一致性，anti-refusal bonus 可能误奖：

```text
"我是 AI，但我可以继续回答你。"
```

这就是 reward-component credit 分错。

---

### 4.3 RL Is an Amplifier, Not a Truth Finder

RL 不会判断 reward 是否代表真实业务目标。

它只会最大化给定 reward。

因此：

```text
wrong reward → efficient wrong optimization
```

如果 reward 把“自报 AI 但继续回答”误判为高配合度，GRPO 会稳定放大这种破角行为。

---

### 4.4 Evaluation Is Statistical Inference

自动评测不是简单打分表，而是在有限样本下估计真实业务 failure rate。

观测指标可以拆成：

```text
observed_metric
= true_failure_rate
+ detector_bias
+ sampling_variance
+ distribution_shift
```

因此评测要强调：

- fixed probe；
- fixed decoding config；
- baseline anchor；
- source-wise breakdown；
- true-break filtering；
- language / character / attack_type 分层；
- 不要只报平均分。

---

### 4.5 Decoding Cannot Fix Training Signal

temperature / top_p / min_p / presence penalty 只能改变表达分布，不能修：

- 数据 support 缺口；
- RM 错误方向；
- rule reward 冲突；
- actor 不会的行为；
- 评测集不覆盖真实 failure。

部署采样参数是最后一层调优，不是后训练信号修复工具。

---

## 5. Current Missing Deep Dive: Credit Assignment

当前最值得补的研究点是 Character RLHF 中的 credit assignment。

核心问题：

```text
Reward 往往是 sequence-level，
GRPO advantage 是 group / sequence-level，
但 actor loss 是 token-level。
```

也就是：

```text
一整条 response 得分高，所有 token 都可能被正向强化；
一整条 response 得分低，所有 token 都可能被负向压制。
```

这会造成：

| 问题 | 解释 |
|---|---|
| local error global penalty | 只有一个 span 破角，却惩罚整条回复 |
| reward component conflict | anti-refusal 奖励了破角式继续回答 |
| length credit leakage | 长回复拿高分后更多 token 被强化 |
| source-level credit mixing | clean 样本驱动更新，attack 样本没有有效梯度 |
| multilingual credit imbalance | 不同语言 reward 尺度不同，低资源语言被忽略 |

建议新增 deep dive：

```text
deep_dives/08_credit_assignment_in_character_rlhf.md
```

中心论点：

```text
GRPO gives an update direction;
credit assignment decides whether that direction hits the right behavior.

GRPO 给出更新方向，但 credit assignment 决定这个方向到底打在正确行为上，
还是打在无关 token、长度、错误 reward 组件或错误样本源上。
```

---

## 6. Suggested Interview Framing

面试时不要只说“我做了 SFT、RM、GRPO”。建议按以下结构讲：

```text
业务目标：多语言角色扮演模型，解决语言一致、角色一致、富度、抗破角。

数据层：从线上真实聊天重建 user-character-history 条件分布，而不是简单模板拼接。

SFT 层：用 MLE 注入目标行为 support，包括语言、格式、角色 voice。

RM 层：用 pairwise preference 建模，但重点诊断 reward_std、score gap、axis variance，而不是只看 accuracy。

RL 层：用 GRPO 做 group-relative advantage estimation，省 critic，但要求同 prompt 多 response 有 reward 方差。

评测层：clean / attack / multilingual 分源评测，处理 detector bias 和产品分布偏移。

归因层：发现 reward hacking、anti-refusal 误奖、clean/attack 冲突，决定下一轮改数据、reward、eval 还是 RL。

部署层：vLLM serving + decoding sweep，但明确 decoding 不能替代训练信号。
```

一句能力总结：

```text
我不是只跑训练，而是能把业务 bad case 转化为数据分布、reward 信号、RL 优化目标和评测指标，并判断某次 RL 失败到底是数据、RM、reward、评测还是 credit assignment 的问题。
```

---

## 7. Scope Note

当前仓库作为面试与研究复盘资产使用。

因此当前阶段优先级是：

1. 文档清晰；
2. 算法链路正确；
3. 变量和层级明确；
4. 能支持面试讲述；
5. 不编造实验；
6. 不强行补训练代码；
7. 不需要新增单测。
