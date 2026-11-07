# Character-RLHF SFT 数据工程：从真实 Session 到 Multi-turn Character SFT Actor

---

# 0. 核心结论

Character-RLHF 的 SFT 阶段，本质不是简单地收集更多聊天记录，而是把真实用户交互中的：

```math
\boxed{\text{角色状态分布}}
```

转换为：

```math
\boxed{\text{稳定、高质量、可学习的监督信号}}
```

完整链路：

```text
真实 Production Conversation

        ↓

Session Reconstruction

        ↓

Trajectory Sampling

        ↓

Pre-Gemini Cheap Rule

        ↓

Gemini 2.5 Flash Teacher Generation

        ↓

Post-Gemini Output Validation

        ↓

Dual Judge Quality Evaluation

        ↓

Repair

        ↓

Head / Continuation Distribution Balance

        ↓

Assistant-only SFT

        ↓

Character SFT Actor
```

最终训练目标：

```math
p_\theta(a_t|r,h_t,u_t)
```

其中：

| 变量含义  |                                  |
| ----- | -------------------------------- |
| `r`   | Role Card / Character Definition |
| `h_t` | 历史 conversation state            |
| `u_t` | 当前 user message                  |
| `a_t` | assistant target response        |

区别于普通 instruction tuning：

```math
p(a|instruction)
```

Character SFT 学习的是：

> 在某个角色定义和已有对话状态下，下一步应该如何回应。

---

# 1. 从第一性原理定义 SFT 数据

## 1.1 SFT 本质：条件最大似然估计

SFT 不优化 reward，也不存在 advantage。

它优化：

```math
\max_\theta \mathbb E_{(x,y)\sim D} \log \pi_\theta(y|x)
```

对于 Character-RLHF：

输入：

```math
x=(r,h_t,u_t)
```

输出：

```math
y=a_t
```

因此：

```math
\mathcal L_{SFT} = - \frac{ \sum_i M_i \log \pi_\theta(y_i|y_{<i}) }{ \sum_i M_i }
```

其中：

```math
M_i
```

是 token-level mask。

| Token是否计算 Loss            |   |
| ------------------------- | - |
| system prompt / role card | 0 |
| user message              | 0 |
| assistant target          | 1 |

---

## 1.2 为什么训练单位必须是 trajectory？

普通 instruction SFT：

```text
Instruction

↓

Answer
```

学习：

```math
p(a|instruction)
```

但是角色聊天：

```text
Role Card

↓

u1 → a1

↓

u2 → a2

↓

u3 → a3
```

第二轮回答依赖：

```math
h_2=(r,u_1,a_1)
```

因此真实分布：

```math
p(a_t|r,h_t,u_t)
```

如果只训练单轮：

模型容易：

- 第一轮像角色；
- 多轮状态丢失；
- 关系变化不稳定；
- persona drift。

所以 SFT 数据单位从：

```math
(question,answer)
```

升级为：

```math
trajectory
```

---

# 2. Session Reconstruction：恢复真实 Conversation State

原始生产数据通常不是训练格式。

例如：

```json
{
"user_id":"xxx",
"character_id":"alice",
"timestamp":123456,
"role":"user",
"content":"..."
}
```

首先根据：

```math
(user\_id,character\_id,timestamp)
```

恢复 session。

---

流程：

```text
Raw Messages


user_id
character_id
timestamp
role
content


        ↓


sort(timestamp)


        ↓


Conversation


u1
a1

u2
a2

u3
a3

...
```

---

## 为什么 Session Reconstruction 是第一步？

因为所有后续判断依赖：

```math
h_t
```

如果顺序错误：

```text
u1

u3

u2
```

那么 Gemini 生成：

```math
a_2
```

实际上是在错误状态下生成。

因此：

> session 错误比单条回答质量差更危险。

---

# 3. Trajectory Sampling：Head 与 Continuation

最终目标不是所有数据都一样。

需要覆盖两个真实产品状态：

1. 用户第一次进入角色；
2. 用户已经和角色聊了一段时间。

---

# 3.1 Head Trajectory

## 目标

学习角色冷启动。

输入：

```text
Role Card


u1 → a1*

u2 → a2*

u3 → a3*

u4 → a4*

u5 → a5*

u6 → a6*
```

六轮 assistant：

全部作为监督。

重点学习：

```math
p(a_1|r,u_1)
```

解决：

> 第一次见用户时如何进入角色。

---

# 3.2 Continuation Trajectory

## 目标

学习已有状态下继续交流。

正确形式：

```text
Role Card


History Context


u1 → old_a1

u2 → old_a2

u3 → old_a3

u4 → old_a4

u5 → old_a5

u6 → old_a6


(mask)


Supervision Window


u7 → new_a7*

u8 → new_a8*

u9 → new_a9*

u10 → new_a10*

u11 → new_a11*

u12 → new_a12*
```

---

关键：

历史：

```math
old_a
```

只作为 context。

不参与：

```math
loss
```

因为：

生产 assistant：

可能：

- 角色偏移；
- 语言问题；
- 风格不一致。

真正监督：

```math
new_a
```

由 Gemini teacher 生成。

---

# 3.3 Sampling 规则

每个 session：

最多：

- 一个 Head；
- 一个 Continuation。

避免：

```text
1-6

2-7

3-8

4-9
```

这种 sliding window。

原因：

同一个 session 被重复监督：

```math
high\ correlation
```

降低数据有效多样性。

---

# 4. 数据漏斗：完整数字版本

最终采用：

| 阶段操作GroupsAssistant Turns             |                      |       |        |
| ------------------------------------- | -------------------- | ----- | ------ |
| Session Reconstruction + Sampling     | 构造候选 trajectory      | 6,922 | 41,532 |
| Pre-Gemini Rule Filter                | 过滤输入损坏               | 5,760 | 34,560 |
| Gemini Flash Generation + Post Filter | 生成并过滤 target         | 5,184 | 31,104 |
| Direct Accept                         | 直接通过 Judge           | 2,700 | 16,200 |
| Repair Pool                           | 边界样本修复               | 2,100 | 12,600 |
| Reject                                | 低质量删除                | 384   | 2,304  |
| Repair Success                        | 修复后通过                | 1,238 | 7,428  |
| High Quality Pool                     | Accept + Repair      | 3,938 | 23,628 |
| Final Balance                         | Head/Continuation 配平 | 3,780 | 22,680 |

---

完整流程：

```text
Candidate Trajectory

6922 groups

41532 assistant targets


        |

        v


========================

Pre-Gemini Rule

========================


Role Card

User Turns

Session

Token Feasibility


        |

        v


5760 groups

34560 targets


        |

        v


========================

Gemini 2.5 Flash

Teacher Generation


========================


        |

        v


========================

Post Filter


========================


Language

Disclosure

Repeat

Format

Length

Safety


        |

        v


5184 groups

31104 targets


        |

        v


========================

Dual Judge


========================


       /        |        \


      /         |         \


 Accept     Repair     Reject


 2700       2100        384


                |

                v


          Flash Repair


                |

                v


          Judge Again


                |

                v


             1238


                |

                v


High Quality Pool

3938 groups


                |

                v


Final Balance


1890 Head

1890 Continuation


                |

                v


3780 groups

22680 turns
```

---

# 5. Pre-Gemini Rule Filter：判断是否值得花 API 成本

## 核心原则

Pre-Gemini 不判断：

> 这个回答好不好。

因为：

旧 assistant：

```math
a^{old}
```

不会作为训练 target。

Pre-Gemini 只判断：

```math
\boxed{ 这个 input 是否值得进入 teacher generation }
```

因此：

只保留：

- cheap；
- deterministic；
- high precision。

---

# 5.1 Role Card Validation

输入：

```math
r
```

检查：

## 存在

```python
role_card is None
```

直接删除。

---

## 格式

检查：

- JSON parse；
- character mapping；
- required field。

---

## Placeholder

regex：

```text
{{name}}

${user}

<PERSONALITY>

[INSERT]
```

如果存在：

```math
match>0
```

reject。

---

## 为什么不用 LLM？

因为：

这是确定错误。

没必要：

Gemini生成

↓

Judge发现角色卡为空。

浪费成本。

---

# 5.2 User Turn Validation

原则：

只删除明显损坏。

不删除真实表达。

例如：

保留：

```text
哈哈

？

😭
```

---

## Empty

规则：

```python
normalize(text).strip()==""
```

reject。

---

## Unicode Corruption

检测：

UTF-8 decode。

例如：

```text
\ufffd\ufffd\ufffd
```

计算：

```math
r= \frac{ replacement\ characters }{ total\ characters }
```

如果：

```math
r>0.05
```

删除。

---

## URL / Emoji / Symbol Domination

不是禁止。

判断：

六轮是否几乎没有语言信息。

计算：

```math
R_{url}
```

```math
R_{emoji}
```

```math
R_{symbol}
```

如果：

6轮中：

```math
\ge4
```

轮主要由：

URL / emoji / symbol

组成：

reject。

---

## Exact Duplicate

流程：

```text
lowercase

↓

unicode normalize

↓

remove whitespace

↓

hash
```

得到：

```math
hash(u_t)
```

计算：

```math
duplicate\ ratio = \frac{ max(count(hash)) }{6}
```

如果：

```math
duplicate\ ratio\ge0.5
```

reject。

---

# 5.3 Session Integrity

要求：

## User

```math
user_1=...=user_6
```

## Character

```math
character_1=...=character_6
```

## Timestamp

```math
t_1<t_2<...<t_6
```

---

Session gap：

不固定写：

30 分钟。

正确：

统计：

```math
\Delta t=t_i-t_{i-1}
```

由生产 session boundary 定义。

Rule 只执行。

---

# 5.4 Token Feasibility

目标：

避免明显超长。

注意：

此时没有：

```math
a_t
```

所以不能计算完整长度。

只计算：

```math
role+user\ history
```

使用：

真实训练 tokenizer + chat template。

仓库真实：

Gemma4 SFT：

```math
\boxed{cutoff\_len=3072}
```

Qwen：

```math
1536
```

因此：

不是：

模型 context 多大。

而是：

训练配置允许多长。

---

# 6. Gemini 2.5 Flash Teacher Generation

Teacher 只负责：

```math
\boxed{ 生成监督答案 }
```

不是评价。

---

## Causal Generation

严格：

```math
a_t \sim \pi_T (a_t|r,h_t,u_t)
```

不能看到未来：

```math
u_{t+1}
```

---

流程：

```text
Wave 1

role + u1

↓

a1*


Wave 2

role + u1,a1*,u2

↓

a2*


...


Wave 6

↓

a6*
```

---

## 并发

不是：

5760 个 trajectory 串行。

采用：

```text
组间并行

+

组内因果
```

即：

6 wave generation。

---

## API Retry

区分：

generation failure

和：

bad sample。

Transient：

- timeout；
- 429；
- 5xx。

采用：

exponential backoff。

Safety block：

不属于 transient。

最终 blocked：

drop。

---

## Context Cache

大量 character：

共享：

- role card；
- rewrite instruction。

可以按：

character\_id

聚类请求。

目的：

提高 cache hit。

---

# 7. Post-Gemini Cheap Filter

目标：

判断：

```math
Gemini\ output
```

是否能作为监督。

---

流程：

```text
Gemini Output

      ↓

API Status

      ↓

Empty / Truncate

      ↓

Disclosure

      ↓

Language

      ↓

Repeat

      ↓

Length

      ↓

Format

      ↓

Judge
```

---

# 7.1 API Status

检查：

- HTTP status；
- safety；
- timeout。

---

# 7.2 Empty / Truncated

Empty：

```python
len(output.strip())==0
```

Truncated：

依据：

```math
finish\_reason
```

例如：

MAX\_TOKENS。

---

# 7.3 Disclosure Detector

Hard Fail：

直接删除：

- I am AI；
- system prompt leak；
- role card leak；
- placeholder；
- wrong identity。

Soft Flag：

交给 Judge：

- AI；
- model；
- language model。

---

# 7.4 Language Detection

采用：

```math
langdetect
```

流程：

```text
trajectory language

        |

Gemini output

        |

langdetect

        |

match?
```

Rule 判断：

语言 ID。

Judge 判断：

语言自然度。

---

# 7.5 Repetition

采用：

4-gram。

条件：

```math
count(ngram)\ge4
```

且：

```math
repeat\ ratio\ge0.18
```

---

# 7.6 Length

不设置：

assistant token 最小值。

因为：

短回复可能合理。

最终检查：

完整 trajectory：

```math
L(\tau)
```

要求：

```math
L(\tau)\le3072
```

超出：

repair compression。

仍超：

reject。

---

# 8. Dual Judge：判断是否值得学习

Rule：

判断：

```math
合法性
```

Judge：

判断：

```math
学习价值
```

---

输入：

完整六轮 trajectory。

两个 evaluator：

| 模型作用           |         |
| -------------- | ------- |
| DeepSeek       | Judge A |
| Gemini 2.5 Pro | Judge B |

---

输出：

```math
[S_f,S_r,S_l,S_{len}]
```

---

# 8.1 Format

评价：

- 对话结构；
- 是否替 user 发言；
- 是否泄露 meta；
- turn boundary。

---

# 8.2 Role

评价：

- persona；
- identity；
- relationship；
- multi-turn consistency。

---

# 8.3 Linguistic

评价：

- 语言自然；
- 风格一致；
- code switch。

---

# 8.4 Length

评价：

回复是否：

- 过短；
- 过度展开；
- 信息密度合理。

---

总分：

```math
S = 0.30S_f + 0.30S_r + 0.25S_l + 0.15S_{len}
```

---

# 8.5 Agreement

两个 Judge：

```math
A_g
```

衡量：

一致性。

不是模型自报 confidence。

---

# 9. Accept / Repair / Reject

## Accept

```math
S\ge4.2
```

且：

```math
A_g\ge0.85
```

Hard Fail=0。

---

## Repair

```math
3.5\le S<4.2
```

或者：

Judge disagreement。

---

## Reject

```math
S<3.5
```

或者：

Hard Fail。

---

# 10. Repair Pipeline

Repair 只做一次。

流程：

```text
Bad Trajectory

↓

Turn Diagnosis

↓

Gemini Flash Rewrite

↓

Cheap Filter

↓

Judge Again

↓

Accept / Reject
```

---

如果问题局限：

```math
a_t
```

只修改当前。

如果影响后续状态：

suffix repair：

```math
a_t',a_{t+1}',...,a_6'
```

---

最大：

```math
max\_repair\_round=1
```

原因：

避免：

- 无限 API 成本；
- 过度编辑；
- 偏离真实分布。

---

# 11. Final Balance

高质量池：

```math
3938
```

分：

```math
Q_H,Q_C
```

最终：

```math
N=2\times min(Q_H,Q_C)
```

目标：

```math
1890H+1890C
```

---

原则：

```math
\boxed{ Quality > Balance > Dataset Size }
```

---

# 12. Train / Validation / Test

按 session split。

避免：

同一个 conversation 出现在不同 split。

最终：

| SplitGroupsAssistant Turns |      |       |
| -------------------------- | ---- | ----- |
| Train                      | 3402 | 20412 |
| Validation                 | 189  | 1134  |
| Test                       | 189  | 1134  |
| Total                      | 3780 | 22680 |

---

# 13. QLoRA SFT Training

## Hardware

训练：

```math
8\times RTX5090
```

单卡：

```math
32GB
```

总：

```math
256GB
```

---

## Model

30B 左右模型：

采用：

```math
NF4\ QLoRA
```

配置：

| 参数配置          |                     |
| ------------- | ------------------- |
| Precision     | 4-bit NF4           |
| LoRA rank     | 32                  |
| Alpha         | 64                  |
| Target module | q/k/v/o             |
| Checkpoint    | Gradient checkpoint |
| Optimizer     | Paged AdamW         |

---

# Distributed Training

推荐：

```math
QLoRA+DeepSpeed\ ZeRO-2
```

或者：

FSDP。

流程：

```text
Dataset

↓

Tokenizer

↓

Distributed DataLoader

↓

Forward

↓

Assistant Token Loss

↓

Backward

↓

Optimizer.step()

↓

Checkpoint
```

---

# Continuation Loss Mask

History：

```text
old_a1
...
old_a6
```

mask：

```math
M=0
```

Teacher target：

```text
new_a7*

...

new_a12*
```

mask：

```math
M=1
```

---

# 最终闭环

```text
真实用户行为


↓

Session State


↓

6-turn Trajectory


↓

Gemini Teacher Target


↓

Rule + Judge Quality Control


↓

Balanced Dataset


↓

QLoRA SFT


↓

Character SFT Actor
```

---

# 最终一句话总结

Character-RLHF 的 SFT 数据工程，本质是：

> 从真实用户 session 中恢复模型需要学习的状态分布，通过 Pre-Gemini cheap rule 去除确定性坏输入，用 Gemini 2.5 Flash 生成因果一致的 teacher target，再通过 Post-Gemini validation 和 DeepSeek + Gemini Pro 双 Judge 判断数据学习价值，对边界样本进行一次 repair，最终平衡 Head/Continuation 分布，通过 assistant-only MLE 训练多轮角色模型。

整个系统中：

```math
\boxed{ Sampling决定学习什么 }
```

```math
\boxed{ Rule保证数据合法 }
```

```math
\boxed{ Teacher提供监督信号 }
```

```math
\boxed{ Judge决定学习价值 }
```

```math
\boxed{ Repair提升数据利用率 }
```

```math
\boxed{ Balance控制最终训练先验 }
```

这就是 Character-RLHF SFT 阶段完整的数据工程闭环。