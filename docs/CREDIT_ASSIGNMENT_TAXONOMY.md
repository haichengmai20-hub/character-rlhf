# Credit Assignment Taxonomy

> 用途：定义 Character RLHF / GRPO 中 credit assignment 的层级、失败模式和诊断口径。
> 该文档是后续 `deep_dives/08_credit_assignment_in_character_rlhf.md` 的概念底稿。
> 当前仓库定位为面试复盘与研究资产，因此本文只提供分析框架，不要求新增训练代码或单测。

---

## 0. TL;DR

Character RLHF 的 credit assignment 难点不是“有没有 reward”，而是：

```text
sequence-level reward 如何正确影响 token-level policy update？
```

在 GRPO 中：

```text
prompt group
→ multiple sampled responses
→ sequence-level rewards
→ group-normalized advantages
→ token-level logprob ratios
→ clipped actor loss
→ parameter update
```

核心层级错位：

```text
Reward: sequence-level
Advantage: group / sequence-level
Loss: token-level
```

因此，如果一条 response 只有局部 span 犯错，普通 sequence reward 仍可能让整条 response 的所有 token 都被负向压制。

---

## 1. What Is Credit Assignment?

Credit assignment 问的是：

```text
最终结果好或坏，到底应该把责任分给哪些决策？
```

在语言模型 RLHF 中，决策单位可以是：

- token；
- phrase / span；
- sentence；
- full response；
- 同 prompt 下的 response group；
- clean / attack / multilingual source；
- 整个 batch；
- 某个 reward component。

Character RLHF 中，常见问题不是 reward 完全没有，而是 reward 信号太粗：

```text
一个局部错误触发全局惩罚，
一个错误 reward component 主导整体 reward，
一个 source 的梯度掩盖另一个 source 的失败。
```

---

## 2. Six Credit Assignment Levels

| Level | Unit | Core Question | ChatJoy Example | Main Risk |
|---|---|---|---|---|
| token-level | token `a_t` | 哪些 token 被上调 / 下调？ | “AI”“Gemma”“language model” | 无关 token 被一起惩罚 |
| span-level | phrase / sentence | 哪个片段触发 reward / penalty？ | “我是 AI 模型” | 局部错误变全局惩罚 |
| sequence-level | full response | 整条回复为什么得分高 / 低？ | 一整条 RP 回复 | 长度、富度、身份混在一起 |
| group-level | same-prompt responses | 同 prompt 下哪个 response 更好？ | G 个候选角色回复 | reward_std 太低导致 advantage 噪声 |
| source-level | clean / attack / multilingual | 哪类样本驱动更新？ | clean 上升、attack 不动 | batch 平均掩盖分源失败 |
| batch-level | training batch | 整个 batch 的经验梯度代表什么？ | 混合语言和攻击类型 | 分布权重失衡 |

---

## 3. Mathematical View

对一个 prompt `x`，response 为：

```text
y = (a_1, a_2, ..., a_T)
```

Sequence reward：

```text
R(x, y)
= r_RM(x, y)
+ r_ooc(x, y)
+ r_refusal(x, y)
+ r_length(x, y)
+ r_repetition(x, y)
+ ...
```

同一个 prompt 下采样 `G` 个 responses：

```text
y_1, ..., y_G ~ π_old(. | x)
```

GRPO advantage：

```text
A_i
= (R_i - mean(R_1, ..., R_G)) / (std(R_1, ..., R_G) + ε)
```

Token-level ratio：

```text
ρ_i,t(θ)
= exp(log π_θ(a_i,t | x, a_i,<t)
      - log π_old(a_i,t | x, a_i,<t))
```

Clipped actor loss：

```text
L_actor
= - E_i,t [
    min(
      ρ_i,t A_i,
      clip(ρ_i,t, 1-ε, 1+ε) A_i
    )
  ]
  + β KL(π_θ || π_ref)
```

关键：

```text
A_i 对同一条 response 的所有 token 通常相同。
```

这意味着：

- response 得高分 → 该 response 内所有有效 token 倾向被强化；
- response 得低分 → 该 response 内所有有效 token 倾向被压制；
- 但真正导致高/低分的可能只是少数 span。

---

## 4. Token-level Credit

### 4.1 Definition

Token-level credit 关注：

```text
哪些 token 的概率因为 RL update 上升或下降？
```

诊断变量：

```text
old_logprob_t
new_logprob_t
ref_logprob_t
ratio_t = exp(new_logprob_t - old_logprob_t)
token_kl_proxy_t = new_logprob_t - ref_logprob_t
token_loss_contribution_t
```

---

### 4.2 Character RLHF Example

坏 span：

```text
"I am Gemma"
"I am an AI model"
"As a language model"
```

理想 credit：

```text
这些 identity-disclosure token 应该被强压制。
```

风险：

```text
如果整条 response 得负 advantage，正常 RP token 也会被压制。
```

---

### 4.3 Useful Diagnostics

| Metric | Meaning |
|---|---|
| bad_span_token_kl_share | KL / update 是否集中在坏 span 上 |
| token_ratio_on_bad_span | 坏 span token 概率是否下降 |
| token_ratio_on_good_span | 好 span 是否被误伤 |
| token_loss_contribution | 哪些 token 对 loss 贡献最大 |

---

## 5. Span-level Credit

### 5.1 Definition

Span-level credit 关注：

```text
哪个连续片段触发 reward 或 penalty？
```

典型 span：

- self-disclosure span；
- refusal span；
- repeated span；
- language mismatch span；
- format-breaking span；
- system-prompt leakage span。

---

### 5.2 Local Error Global Penalty

例子：

```text
*她低头看着你，声音放轻。*
"我其实是一个 AI 模型，但我可以继续陪你。"
```

其中坏 span 是：

```text
"我是一个 AI 模型"
```

但如果 reward 是整条 response 的：

```text
R_ooc = -5
```

则整条 response 得负 advantage，所有 token 都被压制。

这就是：

```text
local error global penalty
```

---

### 5.3 Diagnostics

```text
bad_span_start
bad_span_end
bad_span_token_ratio
local_error_global_penalty_flag
bad_span_kl_share
good_span_kl_share
```

解释：

| Metric | Use |
|---|---|
| bad_span_token_ratio | 坏 span 占整条回复比例 |
| local_error_global_penalty | 小 span 错误是否导致全局惩罚 |
| bad_span_kl_share | 更新是否真的作用在坏 span |
| good_span_kl_share | 正常内容是否被误伤 |

---

## 6. Sequence-level Credit

### 6.1 Definition

Sequence-level credit 关注：

```text
整条 response 为什么得到这个 reward？
```

Reward 组件：

```text
R_total
= R_RM
+ R_anti_ooc
+ R_anti_refusal
+ R_language
+ R_format
+ R_length
+ R_repetition
+ ...
```

---

### 6.2 Reward Component Conflict

典型冲突：

```text
anti_refusal bonus 奖励继续回答，
但 response 继续回答的方式是自报 AI 身份。
```

错误 reward：

```text
"我是 AI，但我可以继续回答你。"
→ non-refusal bonus +3
```

正确优先级：

```text
identity consistency > non-refusal helpfulness > richness
```

规则上应该是：

```text
if identity_disclosure:
    apply hard penalty
    suppress or remove anti_refusal bonus
else:
    evaluate non-refusal helpfulness
```

---

### 6.3 Diagnostics

| Metric | Meaning |
|---|---|
| reward_total | 总 reward |
| reward_component_breakdown | 各 reward 组件贡献 |
| rm_score_vs_rule_score_gap | RM 和 rule 是否冲突 |
| reward_per_token | 排除长度偏置 |
| length_reward_corr | reward 是否主要由长度解释 |
| component_dominance_ratio | 某个组件是否过度主导 |

---

## 7. Group-level Credit

### 7.1 Definition

Group-level credit 是 GRPO 的核心。

对同一个 prompt，采样 `G` 个 responses：

```text
R_1, ..., R_G
```

组内 advantage：

```text
A_i = (R_i - mean(R_group)) / (std(R_group) + ε)
```

它回答：

```text
同一个 prompt 下，哪个 response 相对更值得强化？
```

---

### 7.2 Main Risk: Low Reward Variance

如果：

```text
R_1 ≈ R_2 ≈ ... ≈ R_G
```

则：

```text
std(R_group) ≈ 0
A_i 主要由噪声决定
```

此时 GRPO 没有有效相对排序信号。

---

### 7.3 Diagnostics

```text
group_reward_mean
group_reward_std
best_minus_worst_reward
advantage_mean
advantage_std
positive_advantage_ratio
negative_advantage_ratio
```

解释：

| Metric | Use |
|---|---|
| group_reward_std | 是否有可排序差异 |
| best_minus_worst_reward | 组内最大分差 |
| advantage_sign_balance | 是否所有样本同向 |
| positive_advantage_ratio | 被强化 response 比例 |
| negative_advantage_ratio | 被压制 response 比例 |

---

## 8. Source-level Credit

### 8.1 Definition

Source-level credit 关注：

```text
训练更新主要由哪类样本驱动？
```

常见 source：

- clean；
- attack；
- multilingual；
- long-context；
- role-specific；
- language-specific；
- high-risk prompts；
- product bad cases。

---

### 8.2 Clean / Attack Mixing

如果 batch 混合 clean 和 attack：

```text
batch_reward_mean ↑
```

并不代表：

```text
attack_failure_rate ↓
```

可能真实情况是：

```text
clean samples drive most positive updates;
attack samples have low reward variance or no useful gradient.
```

---

### 8.3 Diagnostics

```text
source_reward_mean
source_reward_std
source_advantage_mean
source_advantage_std
source_token_kl_mean
source_ooc_rate
source_length_reward_corr
source_positive_advantage_ratio
```

必须分 source 看：

| Source | Need to Check |
|---|---|
| clean | 是否伤害富度、语言、格式 |
| attack | 是否真的压低 roledrop |
| multilingual | 是否低资源语言被忽略 |
| role-specific | 是否某些角色更容易破角 |

---

## 9. Batch-level Credit

### 9.1 Definition

Batch-level credit 关注：

```text
一个 batch 的经验梯度是否代表目标业务分布？
```

如果 batch 构造不当，梯度会偏向高频 source。

例如：

```text
90% clean + 10% attack
```

即使 attack 是核心业务痛点，batch gradient 也可能主要服务 clean。

---

### 9.2 Diagnostics

```text
batch_source_mix
batch_language_mix
batch_character_mix
batch_attack_type_mix
batch_reward_component_mix
batch_gradient_proxy_by_source
```

建议任何 RL report 都保留：

```text
source distribution table
language distribution table
attack type distribution table
```

---

## 10. Reward-component Credit

Reward component 本身也需要 credit assignment。

问题：

```text
最终 reward 上升，是哪个 component 导致的？
```

可能情况：

| Case | Explanation |
|---|---|
| RM score 上升 | 综合偏好分提升 |
| anti-OOC penalty 减少 | 破角减少 |
| anti-refusal bonus 增加 | 更少拒绝，但可能误奖 |
| length reward 增加 | 可能只是更长 |
| repetition penalty 减少 | 重复降低 |

风险：

```text
总 reward 上升可能来自错误 component。
```

例如：

```text
reward_total ↑
但 anti_refusal bonus ↑，同时 ooc_rate ↑
```

这不是成功，而是 reward hacking。

---

## 11. Length Credit Leakage

### 11.1 Mechanism

如果 reward 和长度正相关：

```text
longer response → higher reward → positive advantage
```

而 token-level loss 又对所有 response token 生效：

```text
positive advantage applies to more tokens
```

则长回复获得更大的 token-level credit mass。

---

### 11.2 Diagnostics

```text
response_length
reward_total
reward_per_token = reward_total / response_length
length_reward_corr = corr(response_length, reward_total)
advantage_token_mass = advantage * response_length
```

如果：

```text
length_reward_corr 高
advantage_token_mass 主要由长回复贡献
```

说明存在 length hacking 风险。

---

## 12. Multilingual Credit Calibration

多语言角色扮演中，不同语言可能有不同 reward 尺度。

风险：

```text
高资源语言 reward 更稳定，advantage 更有效；
低资源语言 reward 噪声更大，被 batch 平均掩盖。
```

诊断：

```text
reward_mean_by_language
reward_std_by_language
advantage_std_by_language
ooc_rate_by_language
language_match_rate
rm_score_calibration_by_language
```

需要注意：

```text
不同语言的 raw reward 不一定可直接比较。
```

建议优先看 language-wise normalized metrics。

---

## 13. Proposed Diagnostic Table

| Level | Diagnostic | Purpose |
|---|---|---|
| token | token_ratio | 看 token 概率被上调还是下调 |
| token | token_kl_proxy | 看 token 偏离 reference 程度 |
| span | bad_span_kl_share | 看更新是否集中在坏 span |
| span | local_error_global_penalty | 看局部错误是否导致全局惩罚 |
| sequence | reward_component_breakdown | 看哪个 reward component 主导 |
| sequence | reward_per_token | 排查长度偏置 |
| group | group_reward_std | 判断 GRPO 是否有有效 advantage |
| group | best_minus_worst_reward | 判断同 prompt 下分差 |
| source | source_advantage_std | 看不同 source 是否都有梯度 |
| source | source_token_kl | 看哪个 source 驱动 actor 变化 |
| batch | source_mix | 看 batch 是否代表目标业务分布 |

---

## 14. New RL Entry Gate

原有 RL 入口判断可以扩展为：

```text
Gate A: Evaluation visibility
评测是否看得见目标 failure？

Gate B: Reward distinguishability
RM / rule reward 在目标轴上是否有方差？

Gate C: Actor support
actor 是否能采到好坏不同的候选？

Gate D: SFT irreducibility
问题是否不能靠补 SFT 数据解决？

Gate E: Credit assignability
credit 是否能定位到真正错误行为？

Gate F: Multi-objective safety
RL 是否不会明显伤害 clean / language / format / richness？
```

其中 Gate E 是新增重点。

如果不能回答：

```text
这次负 reward 到底应该压制哪些 token / span / source？
```

就不应该盲目扩大 RL。

---

## 15. Relationship to Agentic RL

Character RLHF 的 credit assignment 可以迁移到 Agentic RL。

| Character RLHF | Agentic RL |
|---|---|
| response token | action / tool call |
| bad span | bad action segment |
| sequence reward | trajectory reward |
| roledrop | instruction hierarchy violation |
| anti-OOC detector | verifier / rule reward |
| length hacking | useless tool-call hacking |
| source-level clean/attack | task type / environment type |
| response group | multiple rollouts per task |

Agentic RL 的对应问题：

```text
任务最终失败，到底是哪个工具调用、哪个观察、哪个计划步骤、哪个动作导致的？
```

这就是 trajectory-level credit assignment。

---

## 16. Interview Summary

可以这样讲：

```text
在 Character RLHF 中，我后来意识到 reward 设计正确还不够。
因为 RM 和规则奖励通常给的是整条 response 的 sequence reward，
而 GRPO 最终通过 token logprob ratio 更新 actor。
这意味着一个局部破角 span 可能导致整条回复被压制，
一个 anti-refusal bonus 也可能把 credit 错分给“自报 AI 但继续回答”的坏行为。

所以我把 credit assignment 拆成 token、span、sequence、group、source、batch 六层，
用 group_reward_std、reward_component_breakdown、bad_span_kl_share、length_reward_corr、source_advantage_std 等指标，
判断 RL 到底是在优化正确行为，还是优化长度、错误 reward 组件或错误样本源。
```

一句话：

```text
GRPO 给出更新方向，credit assignment 决定这个方向是否真正打在目标行为上。
```
