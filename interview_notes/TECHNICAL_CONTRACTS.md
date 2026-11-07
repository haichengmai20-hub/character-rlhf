# 技术合同参考（从原 AGENTS.md 提取）

来源：2026-09-06 修订前的本目录 AGENTS.md。以下技术章节原文保留，用于查阅既有设计、统计单位和工程边界，不作为自动加载的通用行为指令。

仅在任务涉及相应技术合同时读取相关章节。数字与方案是该快照的既有口径，须与当前 Canonical、实际实现和最新用户授权核对；不能单凭本文认定已实跑或当前仍最优。文本中的“必须/不得”限定于描述既有方案，不禁止提出替代设计。文件权限、事实判断、批准与证据冲突处理统一由当前 AGENTS.md 规定。

## 6. SFT、RM、RL 的共同闭环

统一主线：

```text
真实 Session / 产品问题
→ 数据重建与清洗
→ 六轮 Head / Continuation SFT 注入基础能力
→ SFT 后建立 SR-Clean
→ SFT 在 SR-Clean 上到顶或进入平台
→ RM-A.x 用显式错误学习 SR-Clean 的粗代理
→ RL-A.1 更新 Actor，并继续用 SR-Clean 判断平台
→ 上线或产品抽样发现 Attack Roledrop
→ SR-Bench 扩为 SR-Clean + SR-Attack
→ 隔离并冻结 SR-Attack、Internal/B6 与训练/C-k Prompt Family
→ 检查 RL-A.1 Actor 是否能生成合格 Defender；缺失则先做 Targeted SFT + Clean Replay
→ RM-B.x 同时学习 Clean 细粒度质量与 Attack 防御
→ A/Internal 与旧 RM-Bench 升级为 Internal + B6
→ RM-B.x 通过 Internal、B6、Artifact / Reward 接口、N=6 Go/No-Go 与 Safety Smoke
→ 需要复用固定 Actor 候选比较多个 RM/Reward 配置时，可选做 Frozen Rollout Bank 审计
→ RL-B.1 更新 Actor
→ SR-Clean + SR-Attack 与产品 Bad Case 判断继续、回退或停线
→ 定向 SFT / RM Hardening / Reward 修复 / RL 再训练
```

职责边界：

```text
SFT：解决“目标行为能不能生成”
RM：解决“同 Prompt 好坏候选能不能排对”
Reward Contract：解决“真正送进 RL 的标量方向是否正确”
RL：解决“正确行为的生成概率能否提高”
外部评测：解决“业务行为是否真的改善”
```

通用诊断树：

```text
没有合格候选
→ 回 SFT / 数据 Support

有合格候选但 RM 排错
→ 回 Preference Data / Judge / RM

RM 排对但 Composite Reward 排错
→ 查 Rule、Scale、路由和 Reward 合成

Reward 排对但 Actor 不改善
→ 查 KL、Advantage、Ratio/Clip、Staleness、Mask 和工程链路
```

## 7. SFT 专属合同

SFT 权威锚点是整个 `SFT/` 文件夹。完整主文负责定义流程和机制，快速复习版只从完整主文压缩；历史仓库不得反向夺取优先级。

SFT 正文优先覆盖：

```text
Production Conversation
→ Session Reconstruction
→ Trajectory Sampling
→ Pre-Teacher Rule
→ Teacher Generation
→ Post-Teacher Validation
→ Cross Judge
→ Repair / Balance
→ Assistant-only SFT
```

必须区分统计单位：

```text
Session
Conversation Group
六轮 Trajectory Group
Head Trajectory
Continuation Trajectory
Assistant Turn / Assistant Target
Token
```

不能把 Conversation Group 数、六轮 Trajectory 数、Assistant Target 数和 Token 数混成同一数据量。SFT 主叙事必须使用六轮 Head / Continuation Trajectory，不再退回“单个输入对应一个回复”的单轮样本抽象。

Assistant-only SFT 中：

- Head 六轮保留 `Role Card + u1→a1 ... u6→a6`，其中六个 Assistant 回复 `a1...a6` 全部进入监督 Loss；
- Continuation 六轮保留历史前缀和当前连续六轮。历史前缀的 System/User/Assistant Token 只参与 Attention，前缀 Assistant Token 不进入 Loss；当前窗口六个 Assistant 回复全部进入监督 Loss；
- System/User Token 不进入监督 Loss；
- History Mask、Attention Mask 和 Loss Mask 必须分开解释；
- 前缀不进 Loss 不等于不占显存：它仍参与前向 Attention、Activation 保存/重算和反向依赖；不得把 Loss Mask 写成显存优化；
- Head/Cold-start 与 Continuation 采样必须写清上下文、六轮监督区间、前缀 Mask 和信息泄露边界；
- Teacher Generation 的当前口径以 SFT 完整主文的六轮组间并行、组内因果生成方式为准。仓库中旧的单轮重写脚本只作为历史方法补充，不得覆盖当前 SFT Teacher 设计。

SFT 数据质量应覆盖：

- Session/Role Card 一致性；
- API 状态、Safety、空输出和截断；
- 语言、Placeholder、重复、长度和格式；
- Judge 维度与规则维度的职责；
- Repair 成功/失败分母；
- 角色、语言、长度、场景与来源分布平衡；
- 去重和 Train/Eval 泄露检查。

SFT 训练完成后立即建立并冻结 SR-Clean 的首版协议，用它比较 SFT Checkpoint、判断继续训练、平台或停止。训练 Loss 下降不能替代角色质量、语言、推进、Richness 和格式等 Actor 行为评测；Attack 是后续上线暴露后再加入 SR-Bench 的新 Slice，不倒写为初始 SFT 已知目标。

精确 Funnel、Teacher 版本、数据量、Head/Continuation 配比和训练方案以 `SFT/` 两篇 Canonical 为准；仓库原始配置只用于补充历史实现和工程追问，不得反向修改面试口径。

## 8. RM 专属合同

RM 始终是一个 Scalar Score Head。Pair 构造 Heuristic、LLM Judge 维度、Hard Fail、在线 Rule Reward 和 KL 是不同对象，禁止混成“多 Reward Head”。

RM 面试版本使用独立编号轴，不与 RL 共用连续编号：

```text
RM-A.1：建立四类显式错误偏好（历史 RM v1 映射）
RM-A.2：过度过滤导致边界信号丢失（历史 RM v2 映射）
RM-A.3：固定 Benchmark 定位短板并用 Hard Negative 修复（历史 RM v3；RM v4 为修复设计）
RM-A.4：完整 Base + 少量 Hard，形成 Anchor RM（历史 RM v5 映射）

RM-B.1：Anchor 40% + Clean 50% + Attack 10% 的 Unified RM（历史 RM v6/v6.2/v6.3 只作形成过程映射）
RM-B.2：C-k 定向 Hardening 与 on-policy 校准（历史 RM v6.6 映射）
```

RM-A.x、RM-B.x 是面试机制节点，不是仓库 Checkpoint 名。历史版本号保留在括号、证据段和版本映射中。

稳定统计单位：

```text
1 Pair = 1 条 chosen + 1 条 rejected
Pairwise Accuracy 的分母 = Pair 数
展开 Response 数 = 2 × Pair 数
```

Pair 构造必须：

- chosen/rejected 来自同一 Prompt；
- 写清 chosen 为什么更好、rejected 的目标错误是否真实发生；
- 不自动 Swap rejected-better 或 Tie；
- 将 Clear Roledrop、Subtle Roledrop、Defender Quality 与具体 Attack Type 分开；
- 对 Generator/Source/Temperature 做隐藏或审计，防止 Source Fingerprint；
- 报告 Unique Prompt 数，不能把一个 Prompt 派生的多个 Pair 当成独立样本扩大证据。

RM 的单轮/多轮只作为 Prompt 上下文包装维度，不改变既有 Pair 类型、总量、配比、六 Source、Judge 或 Pairwise Loss：

```text
single_turn：Role Card + 当前 User → 比较当前 chosen/rejected
head6_final：前五轮固定历史 + 第六轮 User → 只比较第六轮 chosen/rejected
continuation6_final：历史前缀 + 当前六轮中前五轮固定历史 + 最后一轮 User
                    → 只比较最后一轮 chosen/rejected
```

所有历史 Token 都参与 Attention，但不单独产生 token-wise RM Loss；RM 在最终 pooling/EOS 位置得到整段 `Prompt + 最后一轮 Response` 的 Scalar，Pairwise Loss 只比较同一完整 Prompt 下最后一轮 chosen/rejected。历史仍会通过 Attention 影响最终 Scalar 和反向梯度。两者必须共享完全相同的 Role Card、History、User、Chat Template 和截断结果。

`single_turn / head6_final / continuation6_final` 是每个既有数据桶内部的正交分层，不是把全量数据复制三份。构数前必须冻结各层配额或采样规则；构数后同时报告每个 Anchor/Clean/Attack 桶内的三层数量、Pair 数、Unique Prompt 数和 non-pad Token 数，总 Pair 与既有业务配比保持不变。

多轮包装必须通过 GPU/Token 可行性门：记录 `Role Card + History + User + Response` 的 non-pad Token 分布、截断率和最长样本；遵守 RM Canonical 的 `max_len`、micro batch 和 gradient accumulation。RM 的 `global batch=64` 指 64 Pair，即每步涉及 128 条 chosen/rejected 序列；相同 Pair 数不等于相同 Token Exposure。超限时只能先裁 Continuation 当前六轮窗口之前的更早前缀；不得裁 Role Card、Head/Continuation 当前六轮、最后一轮 User 或 Candidate Response。仍超限则做已批准的压缩或丢弃，不能保留 `head6_final/continuation6_final` 标签却实际缺轮。未经同配置 Smoke 越过 Forward、Backward、保存和 Reload，不得把完整多轮 RM 写成已可运行事实。

4D / Judge 边界：

```text
早期 4D：format / role / language / length 的构对与过滤 Heuristic
历史 5D/8D：外部 LLM Judge Rubric
当前四轴：Fine Role / Progression / Richness / Density
Hard Fail：Language、Format、System Leak、Acts-for-user、Roledrop 等不可补偿门
Learned RM：最终仍输出一个 Scalar
```

RM 评测职责：

```text
A/Internal Eval：训练开发与 Checkpoint 选择
B6-Bench：固定、跨候选 RM 的业务验收
C-k：与 A/B 隔离的动态 Hard-Mining 训练池，不是评测集
Artifact Gate：保存、重载、Score Head、EOS/Pooling、在线 Scorer 和 Resume 的工程验收
```

B6 的 Core、Clean、Attack 是同一套 Benchmark 的不同 Slice，不合成一个掩盖短板的总 Accuracy。

Frozen Rollout Bank 不是正式业务评测，也不是每次 RM→RL 的强制门。它是可选的跑前审计载体：当需要在同一批当前 Actor 候选上比较多个 RM、Score Scale、Rule/Guard 或 Serving 实现时，冻结 Response 与 Judge 标签以降低重复生成成本。强制要求是完成一次与正式 RL 协议一致的 N=6 Reward 接口 Go/No-Go；该审计可用 Frozen Bank，也可用一次性独立 Audit Batch。Frozen 结果不能替代 B6、SR-Bench、Artifact Gate 或 Safety Smoke。

RM Pairwise Loss、Gap、Q25、B6 配额、C-k 上限、训练参数和 Checkpoint 策略，以 RM 两篇 Canonical 正文为准；原始脚本只作历史实现补充，不得替代或覆盖 Canonical 精确配置。

## 9. RL 专属合同

RL 使用与 RM 分离的独立编号轴：

```text
RL-A.1：消费 RM-A.4 的第一阶段 RL；SR-Clean 平台后上线并暴露 Attack
RL-B.1：消费一个明确写出的最终获批 RM-B.k；用 SR-Clean + SR-Attack 验收完整配方
```

只有真正更新并产出需要评测的 Actor，才增加 RL 编号。RM 迭代、Reward 接口审计、Frozen Rollout Bank、Safety Smoke 和 Checkpoint 选择都不自动产生新的 RL 版本。正文必须显式写 `RL-A.1 consumes RM-A.4`，并在启动 RL-B.1 时把实际入选的 `RM-B.k`、Checkpoint/Hash 和 Reward Contract 写死；不得保留含糊的“RM-B.1 或 RM-B.2”，也不能靠连续数字暗示关系。

RL 数据和计算必须写清：

```text
Parquet Row / data_source / prompt / extra_info
→ Runtime uid
→ 同 Prompt 重复 G 次 Rollout
→ Response + Response Mask
→ Learned RM + Guard
→ Group-relative Advantage
→ Old / Reference / Current Logprob
→ GSPO Policy Loss + Actor-side KL
→ Backward / Optimizer Step
→ Checkpoint / Weight Sync
→ SR-Bench
```

策略角色必须区分：

```text
π_rollout：生成 Response 的推理副本
π_old：当前 Rollout Batch 的固定行为锚点
π_ref：KL Reference
π_θ / π_current：当前 Minibatch 正在更新和重算 Logprob 的 Actor
π_new：本轮更新完成后的 Actor
```

Mask 必须区分：

```text
Attention Mask：哪些 Token 是模型前向中的有效上下文
Causal Mask：阻止看到未来 Token
Response Mask：哪些 Actor 生成 Token 承担 Policy/KL Loss
Span Mask：可选，且必须是 Response Mask 的子集
```

RL 的单轮/多轮与 RM 使用同一上下文包装语义：`single_turn / head6_final / continuation6_final`。历史 Role Card、User/Assistant Turn 全部只参与 Attention；每个 Prompt 仍只生成最后一轮的 `G` 个候选，Response Mask、Policy Loss 和 KL 只覆盖本次最后一轮 Response。历史 Token 不进 Response Mask，但仍占 Prefill、KV Cache、Current/Old/Reference Logprob 前向与 Actor Activation；Mask 不能被描述成节省这部分显存。不得把它写成六次 Actor 动作、User Simulator 交互或 Agentic 多步 RL；RM、RL 始终留在 RLHF 范围内。

多轮 RL 不改变既有 Clean/Attack 配比、Prompt 总量、`G`、GRPO、GSPO、KL 或 Reward 构成，只增加上下文包装与分层报告。长 History 会同时增加 Actor Activation、vLLM KV Cache、`B×G` 生成吞吐和 RM Scoring 成本；必须分别记录 Prompt Token、Response Token、Padding、KV 预算、各阶段同时驻留模型和峰值申请。只有同硬件 Smoke 完成 Rollout、Reward、Backward、Optimizer、Checkpoint 和 Weight Sync 后，才可写为“可运行”；否则标 `[当前设计｜待显存 Smoke]`。必要时按顺序降低最长 History、长样本占比、Token Packing Cap 或并发/Microbatch，不首先改变业务配比和 Reward 目标。

Reward 与 KL 必须分开：

- Learned RM 提供正向质量；
- Rule/Guard 只处理已批准的不可补偿红线和机械退化；
- KL 在 Actor Loss 中时，不得写进 Scalar Reward；
- 配置同时打开 RM 与 Custom Reward 不证明二者已经相加，必须检查真实 Reward Loop；
- Per-source KL 必须验证 Source Mask 确实进入 `ppo_loss`，不能只看到全局 KL 日志就宣称生效。

GRPO、GSPO 和 Span 必须按真实粒度解释：

- GRPO 在同 `uid` 的候选组内构造 Advantage；
- GSPO 使用 Sequence-level Ratio/Clip，不等于解决所有 MoE 训推不一致；
- Outcome-level Reward 会把 Sequence Advantage 广播到 Response Token；
- 真正 Span Credit 需要 Character Span → Token Offset → Span Mask；只把 Sequence Reward 减一个常数不叫 Span-aware；
- Proposed Span、Conditioned KL 或模拟 A/B 不得写成历史已跑成果。

RL 正式 Actor 效果统一使用 SR-Bench。N=6 Reward 接口审计、Reward 对账和 Mask 单测只作 Pre-RL Go/No-Go，不升级为新的正式效果 Benchmark，也不产生新 RL 版本。

训练数据、G、Temperature、KL 系数、Clip、Token Cap、Span 开关和停线门，以 RL Canonical 正文为准；当前 Launcher 和真实日志只补充历史实现与工程证据。设计值、Canary 值与历史 Run 值不得混写。

## 10. 评测与统计单位分工

```text
SFT Actor：训练后立即建立 SR-Clean；以 Scene、Prompt、Response、Seed 为单位
RM-A.x：A/Internal + 旧 RM-Bench，服务粗代理开发与 Anchor 验收
RL-A.1：继续使用 SR-Clean；到平台后上线，产品 Bad Case 暴露 Attack
RM-B.x：Internal + B6-Core/B6-Clean/B6-Attack；同时改善 Clean 与 Attack
RL-B.1：使用 SR-Clean + SR-Attack 判断最终 Actor
Pre-RL 接口：Prompt Group；检查 Actor Support、Top-1 Safety、Ranking Direction
Artifact：固定输入方向一致率、Hash、Reload/Resume 结果；不算业务效果
```

SR-Attack 是上线发现 Attack 后新增的 Slice；历史 Checkpoint 可以在新协议上回溯评测，但不能倒写成初始 SFT/RL 当时已经用 SR-Attack 做选择。SR-Clean 与 SR-Attack 都应按 `single_turn / head6_final / continuation6_final` 分层报告；是否汇总必须预先冻结权重与分母。

线上 Bad Case 进入数据飞轮前必须按 Prompt Family 隔离：一部分进入 RM-B Train/C-k，一部分进入 Internal/B6，一部分进入 SR-Attack；同一 Prompt、同源改写和语义近重复不得跨集合。Internal/B6 必须先冻结再用于训练开发和候选验收，不能在看到待测 RM 结果后改题目、难度或配额。

跨阶段数据遵守“共同业务分布与可追溯血缘，独立训练单位与独立分母”：

- SFT、RM、RL 不强制使用相同 Prompt 总数，不强制数据量相等，更不统一 Pair 数；SFT 统计六轮 Trajectory Group/Assistant Target，RM 统计 Pair/Unique Prompt/Candidate Response，RL 统计 Prompt Group/`B×G` Response/Token；
- RM chosen/rejected、同 Prompt 六 Source 候选必须使用完全相同的 Prompt；RL 同 `uid` 的 `G` 个 Rollout 必须使用完全相同的 Prompt；同一受控 A/B、Checkpoint 对比或同尺 Actor 评测必须冻结相同 Prompt、Seed、生成协议和 Judge；
- SFT、RM、RL 全量训练集只要求角色、语言、场景、Clean/Attack、单轮/多轮等业务分布可以对齐和追溯，不要求复用相同 Prompt ID。需要串联时记录来源关系，不以复制同一条 Prompt 冒充覆盖增加；
- Train/C-k 与 Internal/B6/SR 必须按 Prompt Family 隔离。正式评测和固定 Benchmark 不为追求“跨阶段同 Prompt”而回灌训练；
- 跨阶段公平性主要比较覆盖分布、Unique Prompt、non-pad Token Exposure 和固定评测结果，不通过强行对齐 SFT Group 数、RM Pair 数或 RL Prompt 数实现。

禁止：

- 将 SR-Bench、A/Internal、B6 拼成一个总分；
- 用 RM Pair Accuracy 证明 Actor 已改善；
- 用 Frozen/接口审计 Top-1 率外推线上总体率；
- 把一个 Group 派生 Pair 当成独立 Prompt；
- 把参与早停和选 Checkpoint 的 Dev 集写成独立发布证明；
- 把 Missing 当成 0、把 Tie 当成正确、把 Raw 检测值当 True Break；
- 在没有相同 Prompt、Seed、生成协议和 Judge 时做版本差值归因。

## 11. 版本、数字和因果边界

始终区分：

```text
SFT 版本
RM-A.x / RM-B.x 面试机制节点
RM 历史版本
RL-A.x / RL-B.x 面试训练节点
RL 历史训练版本
Run/Restart 编号
Product 交付版本
```

RM 与 RL 的编号轴严格独立。RM 可以在两次 RL 之间迭代多个节点；任何 RL 节点必须显式记录它消费的 RM 节点、SFT Actor、Reward Contract 和评测协议，不能用连续编号推断血缘。

强制边界：

- 产品 11-card Roledrop 与后期 fixed-68 True Break 不是同一协议，不画成一条趋势；
- 历史 `0.037/0.36` 是跨样本/来源 Score Span，不是同 Prompt G-Rollout Std；
- 4D Heuristic、5D/8D Judge、当前四轴、Hard Fail 和 Scalar RM 不互相替代；
- 当前六 Source、44,280 Pair、40/50/10 和数据 Hardening 方案属于当前设计，除非找到对应运行产物，不写成历史实跑；
- 历史 RM v6.x 渐进过程与压缩后的 Unified 机制节点分开；
- Qwen 与 Gemma、不同 Score Head 的 Raw Reward/Gap 不直接比较；
- 相同 Pair 数或 Step 数不自动等于相同 Token Exposure；
- 相关性、单 Seed、同轮多变量变化和最终结果不能被包装成单变量因果；
- 设计稿、待跑实验、代码已写、Smoke 已过、正式训练完成和产品上线是六种不同状态。

## 12. 工程与代码证据

工程叙事必须写成：

```text
发生阶段
→ 原始错误/指标
→ 根因证据
→ 单变量修复
→ 越过原失败点的验收
→ 对质量与复现的影响
```

重要边界：

- RM 训练可使用 OpenRLHF + DeepSpeed ZeRO-2 + 4-bit QLoRA；这是特定 5090/PCIe/NUMA 环境的工程选择，不是通用最优；
- RL Actor 的 verl/FSDP/LoRA/vLLM 路径与 RM DeepSpeed 路径分开；
- OOM 按模型加载、Activation、Full-vocab Logits、KV/CuMem、NCCL SHM、Checkpoint/Disk 等根因拆分；
- 训练内指标、Checkpoint Reload、Offline/Online Scorer 和 Serving Parity 分别验收；
- 原始大日志不入正文，优先使用 `character-rlhf/reports/framework_logs/LOG_EVIDENCE_INDEX.md` 和脱敏摘录；
- 不在正文复述 Token、API Key、临时外链、用户标识或完整敏感日志。
