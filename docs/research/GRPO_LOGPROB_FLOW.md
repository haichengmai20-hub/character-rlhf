# GRPO Logprob Flow

> 用途：固定 ChatJoy Character RLHF / GRPO 训练中的 old_logprob、new_logprob、ref_logprob、reward、advantage、KL、actor update 的计算顺序。
> 这个文档面向面试复盘与源码阅读，不依赖某一个框架实现。verl / OpenRLHF / TRL 等框架变量名可能不同，但逻辑链路一致。

---

## 0. TL;DR

GRPO 中最容易混乱的是：

```text
old_logprob、new_logprob、ref_logprob 不是同一个东西，
它们由不同模型、在不同时间、为不同目的计算。
```

一句话：

```text
old_logprob 来自采样时的 old policy，
new_logprob 来自正在更新的 current actor，
ref_logprob 来自冻结的 reference model。
```

核心链路：

```text
prompt batch
→ rollout actor / vLLM 采样 responses
→ 保存或计算 old_logprobs
→ RM + rule reward 计算 sequence rewards
→ reference model 计算 ref_logprobs
→ group-normalized rewards 得到 advantages
→ current actor 重新 forward 得到 new_logprobs
→ ratio = exp(new_logprob - old_logprob)
→ clipped actor loss + KL(new || ref)
→ actor update
→ sync actor weights to rollout engine
```

---

## 1. Model Roles

| Role | Also Called | Trainable? | Main Job |
|---|---|---:|---|
| actor | policy model / training actor | yes | 被 RL 更新的模型 |
| old policy | rollout policy / frozen actor snapshot | no during batch | 生成当前 batch 的 responses，提供 importance denominator |
| rollout engine | vLLM / sampling engine | synced periodically | 高吞吐采样 responses |
| reference model | ref model / frozen SFT model | no | 提供 KL regularization baseline |
| reward model | RM | no during RL usually | 给完整 response 打 reward score |
| rule reward | verifier / detector reward | no | anti-OOC、anti-refusal、length、format 等规则分 |
| critic | value model | no in vanilla GRPO | PPO 有，GRPO 通常不用 |

---

## 2. Variables

### 2.1 Prompt and Response

```text
x_b: 第 b 个 prompt

y_b,g = (a_b,g,1, ..., a_b,g,T):
同一个 prompt 下第 g 个 sampled response
```

其中：

- `b`：batch 中的 prompt index；
- `g`：同 prompt 下第几个 response，`g = 1..G`；
- `t`：response token index；
- `T`：response 长度。

---

### 2.2 Logprobs

| Symbol | Meaning |
|---|---|
| `log π_old(a_t | x, a_<t)` | old policy 对已采样 token 的 logprob |
| `log π_θ(a_t | x, a_<t)` | current actor 对同一 token 的 logprob |
| `log π_ref(a_t | x, a_<t)` | reference model 对同一 token 的 logprob |

注意：三者必须在同一条 response token 上计算，不能各自重新采样。

---

### 2.3 Reward and Advantage

Sequence reward：

```text
R_b,g = R(x_b, y_b,g)
```

一般由 RM + rule reward 组成：

```text
R(x, y)
= r_RM(x, y)
+ r_anti_ooc(x, y)
+ r_anti_refusal(x, y)
+ r_length(x, y)
+ r_repetition(x, y)
+ ...
```

GRPO group advantage：

```text
A_b,g
= (R_b,g - mean_g(R_b,g)) / (std_g(R_b,g) + ε)
```

其中 mean/std 只在同一个 prompt 的 `G` 个 responses 内计算。

---

## 3. Full Execution Order

## Step 1: Prepare Prompt Batch

输入：

```text
batch = {x_1, ..., x_B}
```

每个 `x_b` 通常包含：

- character system prompt；
- persona / role card；
- conversation history；
- current user message；
- language / source / attack_type metadata。

输出：

```text
prompt batch with metadata
```

---

## Step 2: Rollout Sampling

使用 rollout actor / old policy 采样：

```text
y_b,1, ..., y_b,G ~ π_old(. | x_b)
```

输出：

```text
responses
response token ids
attention masks
response masks
sampling metadata
```

关键点：

- `π_old` 是采样那一刻的 actor snapshot；
- 一个 rollout batch 内，`π_old` 应视为固定；
- vLLM 可能是单独 engine，需要和训练 actor 周期同步权重。

---

## Step 3: Compute or Store old_logprobs

对采样出的每个 token 计算：

```text
old_logprob_b,g,t
= log π_old(a_b,g,t | x_b, a_b,g,<t)
```

两种实现：

| Mode | Explanation |
|---|---|
| rollout 时直接返回 | rollout engine 同时输出 sampled token logprob |
| rollout 后 actor_old 重新 forward | 用冻结 old actor 对固定 response 重新算 logprob |

用途：

```text
ratio denominator
```

即：

```text
ρ_t = exp(new_logprob_t - old_logprob_t)
```

---

## Step 4: Compute Reward

对每个完整 response 打分：

```text
R_b,g = R(x_b, y_b,g)
```

常见 reward 组件：

| Component | Level | Purpose |
|---|---|---|
| RM score | sequence | 综合偏好分 |
| anti-OOC penalty | sequence / span-triggered | 惩罚自报 AI、破角色 |
| anti-refusal bonus | sequence | 奖励非无意义拒绝 |
| language match | sequence | 奖励语言一致 |
| format reward | sequence | 奖励 RP 格式 |
| length shaping | sequence | 控制过短或过长 |
| repetition penalty | sequence / token proxy | 惩罚重复 |

重要：

```text
reward 多数是 sequence-level，
不是 token-level。
```

这会直接引出 credit assignment 问题。

---

## Step 5: Compute ref_logprobs

用冻结 reference model 对同一批 response 计算：

```text
ref_logprob_b,g,t
= log π_ref(a_b,g,t | x_b, a_b,g,<t)
```

用途：

```text
KL regularization
```

常见 KL proxy：

```text
kl_t ≈ log π_θ(a_t | context) - log π_ref(a_t | context)
```

注意：

这只是 sampled-token 上的 KL proxy，不等于完整 vocabulary KL。

---

## Step 6: Compute Group Advantage

对每个 prompt 的 `G` 个 responses，组内归一化 reward：

```text
mean_b = mean_g(R_b,g)
std_b = std_g(R_b,g)
A_b,g = (R_b,g - mean_b) / (std_b + ε)
```

解释：

- 同一个 prompt 下，reward 高于组均值的 response 得正 advantage；
- reward 低于组均值的 response 得负 advantage；
- 不同 prompt 之间不直接比较 raw reward；
- 这相当于用 group mean 替代 critic baseline。

风险：

```text
如果 std_b 太小，advantage 主要是噪声。
```

诊断指标：

```text
group_reward_std
best_minus_worst_reward
advantage_sign_balance
```

---

## Step 7: Current Actor Forward for new_logprobs

进入 actor update 时，使用当前 actor `π_θ` 对固定 responses 重新 forward：

```text
new_logprob_b,g,t
= log π_θ(a_b,g,t | x_b, a_b,g,<t)
```

注意：

- response 不重新采样；
- 只是在同一批 token 上重新计算概率；
- 每次 actor 参数更新后，`new_logprob` 会变；
- `old_logprob` 在该 rollout batch 内保持不变。

---

## Step 8: Compute Importance Ratio

```text
ρ_b,g,t(θ)
= exp(new_logprob_b,g,t - old_logprob_b,g,t)
```

解释：

- `ρ > 1`：current actor 比 old policy 更倾向该 token；
- `ρ < 1`：current actor 比 old policy 更不倾向该 token；
- importance ratio 用于修正“数据来自 old policy，但参数更新 current actor”的 off-policy gap。

---

## Step 9: Actor Loss

GRPO / PPO-style clipped surrogate：

```text
L_policy
= - E_b,g,t [
    min(
      ρ_b,g,t A_b,g,
      clip(ρ_b,g,t, 1-ε, 1+ε) A_b,g
    )
  ]
```

KL penalty：

```text
L_KL
= β E_b,g,t [ KL_proxy(π_θ || π_ref) ]
```

总 loss：

```text
L_actor = L_policy + L_KL
```

关键层级：

```text
A_b,g 是 sequence / group-level，
ρ_b,g,t 是 token-level，
loss 在 token-level 计算后 batch 聚合。
```

---

## Step 10: Actor Update

根据 `L_actor` 反向传播，更新 actor 参数：

```text
θ ← θ - η ∇_θ L_actor
```

在一个 rollout batch 上，可能做一个或多个 minibatch epoch。

如果 epoch 过多，会导致：

```text
π_θ 距离 π_old 太远
→ ratio drift
→ off-policy bias 增大
→ clip fraction / KL 异常
```

---

## Step 11: Sync Actor to Rollout Engine

训练 actor 更新后，需要同步到 rollout engine / vLLM：

```text
training actor θ_new
→ export / broadcast / load weights
→ rollout engine π_old becomes new snapshot
```

同步后，下一轮 rollout 的 old policy 就是新的 actor snapshot。

注意：

```text
old policy 不是永久模型，
它是每轮 rollout 时被冻结的 actor 快照。
```

---

## 4. Timeline View

```text
Time k:
  actor θ_k
  rollout engine loads θ_k
  sample responses
  compute old_logprobs under θ_k

Training updates:
  current actor starts at θ_k
  compute new_logprobs under θ
  optimize loss
  θ_k → θ_{k+1}

Next rollout:
  sync θ_{k+1} to rollout engine
  old policy for next batch becomes θ_{k+1}
```

---

## 5. Token / Sequence / Group / Batch Levels

| Level | Object | Example | Used For |
|---|---|---|---|
| token | `a_t` | 某个字 / subword | logprob、ratio、KL、loss |
| sequence | `y = a_1...a_T` | 一整条角色回复 | RM score、rule reward |
| group | `{y_1...y_G}` for same prompt | 同 prompt 多候选 | GRPO advantage |
| batch | 多个 prompt groups | 一个训练 batch | empirical average / optimizer step |

最重要的层级错位：

```text
reward: sequence-level
advantage: group / sequence-level
loss: token-level
```

这就是 credit assignment 的根源。

---

## 6. Common Pitfalls

### 6.1 Confusing old policy and reference model

错误理解：

```text
old policy = reference model
```

正确理解：

```text
old policy 是 rollout 时的 actor snapshot；
reference model 是冻结的 KL anchor，通常来自 SFT / init actor。
```

---

### 6.2 Recomputing old_logprob with updated actor

错误：actor 已经更新后，再用新 actor 计算 old_logprob。

后果：

```text
ratio ≈ 1
importance sampling 失效
训练日志看似稳定但实际错误
```

正确：

```text
old_logprob 必须对应采样 response 的 old policy。
```

---

### 6.3 Treating ref_logprob as denominator of ratio

错误：

```text
ratio = exp(new_logprob - ref_logprob)
```

正确：

```text
ratio = exp(new_logprob - old_logprob)
```

`ref_logprob` 用于 KL regularization，不是 importance ratio denominator。

---

### 6.4 Ignoring group reward std

如果：

```text
group_reward_std ≈ 0
```

则：

```text
A_i ≈ noise
```

此时即使 logprob flow 完全正确，GRPO 也很难产生有效优化。

---

### 6.5 Mixing clean and attack without source diagnostics

batch 平均 reward 上升不代表 attack 改善。

必须分源看：

```text
clean_reward_mean
clean_advantage_std
clean_token_kl
attack_reward_mean
attack_advantage_std
attack_token_kl
```

否则可能出现：

```text
clean 样本驱动更新，attack failure 原地不动。
```

---

### 6.6 Believing decoding sweep can replace RL diagnostics

temperature / top_p 只能影响采样表达。

如果 reward 方向错、RM 无方差、actor 无 support、credit 分错，decoding sweep 不能根治。

---

## 7. Minimal Pseudocode

```python
for step in range(num_rl_steps):
    # 1. sample prompts
    prompts = sample_prompt_batch(dataset)

    # 2. rollout using old actor snapshot
    responses, old_logprobs = rollout_engine.generate(
        prompts,
        n=num_generations_per_prompt,
        return_logprobs=True,
    )

    # 3. compute sequence rewards
    rm_scores = reward_model.score(prompts, responses)
    rule_scores = rule_reward_fn(prompts, responses)
    rewards = rm_scores + rule_scores

    # 4. compute reference logprobs
    ref_logprobs = reference_model.logprobs(prompts, responses)

    # 5. compute group-relative advantages
    advantages = group_normalize_rewards(rewards, group_by="prompt_id")

    # 6. update actor
    for minibatch in make_minibatches(prompts, responses):
        new_logprobs = actor.logprobs(minibatch.prompts, minibatch.responses)
        ratio = exp(new_logprobs - minibatch.old_logprobs)
        policy_loss = clipped_loss(ratio, minibatch.advantages)
        kl_loss = kl_proxy(new_logprobs, minibatch.ref_logprobs)
        loss = policy_loss + beta * kl_loss
        optimizer.step(loss)

    # 7. sync updated actor to rollout engine
    rollout_engine.sync_weights(actor)
```

---

## 8. Interview Explanation Template

可以这样讲：

```text
在 GRPO 中，采样和更新不是同一时刻的模型。

首先，rollout actor，也就是 old policy，生成 responses，并保存 old_logprobs。
然后 RM 和 rule rewards 给每条 response 一个 sequence-level reward。
同一个 prompt 下多个 responses 的 reward 做组内标准化，得到 group-relative advantage。
之后训练中的 current actor 对同一批 token 重新 forward，得到 new_logprobs。
用 exp(new - old) 得到 importance ratio，再乘 advantage 做 clipped policy loss。
reference model 是冻结的 SFT anchor，只负责算 ref_logprobs 来约束 KL，不参与采样，也不是 ratio 的分母。
最后 actor 更新后，再同步权重到 vLLM rollout engine，进入下一轮采样。
```

---

## 9. Link to Credit Assignment

GRPO logprob flow 解释的是“梯度怎么算”。

Credit assignment 进一步问：

```text
这个梯度到底打在了哪些 token / span / source / reward component 上？
```

所以本文件应和：

```text
docs/research/CREDIT_ASSIGNMENT_TAXONOMY.md
```

配合阅读。
