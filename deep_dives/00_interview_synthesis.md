# 深读 0 — 面试综合路线：把 9 篇 deep dive 串成一条项目能力链

这不是第十篇孤立 deep dive，而是 `deep_dives/` 的阅读入口。它回答三个问题：

1. 面试时这 9 篇到底服务哪条主线？
2. 哪些版本号是真改进，哪些只是 run/restart/失败实验？
3. 如果继续扩展 deep dive，应该沿哪条逻辑加，才能和现有内容 1+1 大于 2？

## 0. 一句话总线

ChatJoy 的核心故事不是“我们训练了很多版本”，而是：

> 一个多语言虚拟伴侣业务，从 API/system prompt 的 safety 和语言一致性问题出发，先用 SFT 建立语言/角色基础分布，再用 RM v1-v5 把 wrong role / wrong language / length mismatch 转成可学习信号；第一阶段 RL 后才暴露破角和富度不足，于是 RM v6/v6.3 与后续 SFT/RL 转向“抗破角 + 保富度”。多次实验不动时，团队把失败逐层归因到评测、数据、RM、reward rule、credit assignment 和系统约束，据此决定哪些问题回 SFT、哪些问题交给 RL、哪些问题必须先重做 benchmark/RM。

这条总线里，真正有面试价值的不是“版本很多”，而是“每次版本变化背后的判断力”：为什么训、训什么、训完怎么知道错在哪里、下一轮为什么不是拍脑袋。

## 1. 深读分组

| 组 | 文件 | 面试能力点 | 关系 |
|---|---|---|---|
| 系统可行性 | [01_memory_accounting_and_gpu_communication.md](01_memory_accounting_and_gpu_communication.md) | 能解释为什么不是 ZeRO-3/FSDP2 一把梭，为什么 4bit LoRA + ZeRO-2/FSDP1 更稳 | 给所有训练选择提供硬件边界 |
| 奖励信号 | [02_rm_data_reward_and_attribution.md](02_rm_data_reward_and_attribution.md)、[03_rl_dynamics_params_and_strategy.md](03_rl_dynamics_params_and_strategy.md) | 能讲 pairdata、RM、rule reward、GRPO/GSPO、per-source KL 和训练曲线归因 | 把“偏好”变成“可优化信号”，再看信号能不能推动 actor |
| 目标冲突 | [04_richness_defense_tension_and_sft_ceiling.md](04_richness_defense_tension_and_sft_ceiling.md)、[05_when_to_move_from_sft_to_rl.md](05_when_to_move_from_sft_to_rl.md)、[06_coordinating_conflicting_objectives.md](06_coordinating_conflicting_objectives.md) | 能讲为什么 clean 富度回 SFT、防御留给 RL，为什么目标冲突要分层协调 | 把“为什么失败”升级成“如何选择训练阶段和协调机制” |
| 归因闭环 | [07_business_benchmark_and_evidence_preservation.md](07_business_benchmark_and_evidence_preservation.md)、[08_credit_assignment_in_character_rlhf.md](08_credit_assignment_in_character_rlhf.md)、[09_full_chain_attribution.md](09_full_chain_attribution.md) | 能讲 benchmark、同尺评测、credit assignment、全链路差分诊断 | 把“训练经验”升级成“可迁移的后训练方法论” |

## 2. 面试推荐讲法

### 3 分钟版

1. 业务起点：虚拟伴侣依赖语言一致、角色一致和情绪/动作细节，API/system prompt 与 base model 在 safety 和小语种上不稳定。
2. SFT 先解决基础分布，RM v1-v5 解决 wrong role / wrong language / length mismatch，RM v6/v6.3 解决 richness 和 roledrop 可见性。
3. RL 不是“必然更好”：早期 `+3 non-refusal` 把破角当成好行为奖励，导致产品 roledrop 变差。
4. 后续核心能力是归因：把问题拆到评测、数据、RM、rule reward、credit、系统约束，决定回 SFT、改 RM、调 reward 或停线。

### 10 分钟版

按四个转折讲：

| 转折 | 问题 | 证据 | 决策 |
|---|---|---|---|
| API/base -> SFT | 小语种语言错配、API safety | TOP20 × 27 语言、Gemini 重写、clean32 SFT | 先拉基础分布，不直接 RL |
| SFT -> RM | SFT 上线仍有 wrong role/language/length | RM v1-v5 pairdata，v5 17,712 train，acc 95.5%，gap 3.497 | 用 RM 把错误模式显式化 |
| RM/RL -> RM v6 | RL 后破角和富度不足 | non-refusal Goodhart、clean RM 平、attack 高 | 多 source judge + attack pair，RM v6.3 40,551 train |
| RL 迭代 -> 归因闭环 | 调参数不解决 true break | v6.5 零移动、v6.8 monologue 过冲、v6.10 true break 未降 | 富度回 SFT，防御/RM/benchmark 重做，不盲目堆步数 |

### 30 分钟深挖版

1. 先讲业务与数据：用 `README.md` / `project_chatjoy_overview.md` 的 SFT 台账把听众带到同一个上下文。
2. 讲 RM v1-v5：单因负样本、4D pair-build heuristic、25/25/30/20 重采样，以及 `sample_weight` 只写入元数据但未被 OpenRLHF loss 消费；再讲 v1-v5 为什么不是线性变好。
3. 讲 RL Goodhart：`rm + anti_refusal` 为什么奖励了破角，anti-OOC 和 revoke 怎么修。
4. 讲 RM v6/v6.3：6-source、多 judge、8D rubric、v5 anchor + clean/fine + attack 数据混合。
5. 讲 deep dive 4/5/6：为什么 clean 富度和 attack 防御不能放在同一层硬拧。
6. 讲 deep dive 8/9：为什么“方向对了但没降”可能是 credit 分配问题，而不是 RL 参数问题。

## 3. 版本号去魅

很多 v6.x 看起来像“逐版升级”，实际上有三类：真实改进、失败实验、训练启动/重启号。面试时不要把所有编号都讲成“我们迭代得越来越好”。

| 名称 | 应如何理解 | 面试表达 |
|---|---|---|
| RM v1-v3 | 真实 RM 实验，但 v2/v3 不是改进，反而暴露 batch、score head、LR/steps 和坏样本问题 | “RM 不是线性升级，v2/v3 是排错样本” |
| RM v4 | 计划/脚本版，未正式训练 | “被 v5 覆盖，不作为正式版本讲” |
| RM v5 | 错误模式 RM 的可用强基线 | “离线最强；第一阶段 RL 后才暴露富度/roledrop 覆盖不足” |
| `rl_full_v3/v4/v5` | RL-6.3 内部重启/run 计数，不是产品版本 | “run 名不是模型能力版本” |
| RL-6.3 / 产品 v1 | 早期 RL Goodhart 期，语言一致性改善但 roledrop 很差 | “这是暴露问题的版本，不是成功版本” |
| RL-6.4 / 产品 v2 | anti-OOC + 防御 SFT 后的真实改善，但仍差于 base | “从 91% 降到 55%，但没有达到业务标准” |
| RL-6.5 | 硬化 GRPO 失败，375 步零移动 | “证明 val/probe 与产品分布不一致，不是加步数能解决” |
| RL-6.6b / RL-6.7 | 诊断性实验，clean 或 defense 增益接近噪声/温度脆 | “帮助确定富度主要靠 SFT” |
| SFT-6.8 | 富化失败，monologue 过冲 | “选择器偏差被 SFT 放大，是失败证据” |
| SFT-6.9 | clean 富度真实恢复 | “富度回到 SFT 解决” |
| SFT-6.9b/c | defense SFT 补强尝试，但 true break 没降 | “证明 SFT-defense 到天花板” |
| RL-6.10 | clean-aware RL，clean held 但 attack true break 未降 | “停线证据，下一步要 RM v7/targeted data/benchmark” |

## 4. 1+1 大于 2 的读法

单看每篇 deep dive，像 9 个专题；串起来后，它们是一个后训练决策系统：

1. [01](01_memory_accounting_and_gpu_communication.md) 告诉你“系统上什么训练方案可行”。
2. [02](02_rm_data_reward_and_attribution.md) 告诉你“reward 信号怎么造，以及怎么造错”。
3. [03](03_rl_dynamics_params_and_strategy.md) 告诉你“训练曲线怎么读，不要把所有平台都怪算法”。
4. [04](04_richness_defense_tension_and_sft_ceiling.md) 告诉你“富度和防御为什么互相伤”。
5. [05](05_when_to_move_from_sft_to_rl.md) 告诉你“什么时候才值得起 RL”。
6. [06](06_coordinating_conflicting_objectives.md) 告诉你“目标冲突应该在阶段、reward 还是 source 层解决”。
7. [07](07_business_benchmark_and_evidence_preservation.md) 告诉你“没有同尺 benchmark，产品结论会反复打脸训练 val”。
8. [08](08_credit_assignment_in_character_rlhf.md) 告诉你“sequence reward 为什么落不到真正坏 span”。
9. [09](09_full_chain_attribution.md) 告诉你“上线不行时如何从评测到部署逐层排锅”。

所以综合能力不是 9 篇知识点相加，而是一套判断顺序：

```text
先问系统能不能训 -> 再问信号有没有 -> 再问评测看不看得见 -> 再问目标在哪层解决 -> 最后才问 RL 参数怎么调
```

## 5. 后续可新增的 deep dive 方向

以下不是随手加题，而是沿同一条面试逻辑补强“下一步能力”。建议未来只在确实有材料时新增。

| 建议题目 | 放在链路中的位置 | 为什么值得讲 |
|---|---|---|
| Deep Dive 10：从业务口述到数据台账，如何做 SFT/RM/RL 的数据 provenance | 接在 SFT 与 RM 之间 | 解决当前口述数字与仓库数字不完全对齐的问题，展示你能把业务抽样、清洗、judge、重写和训练产物做成可审计账本 |
| Deep Dive 11：RM v7 设计，质量感知 RM 如何同时看 clean richness 与 attack hard fail | 接在 RM v6.3 之后 | 把 v6.10 停线后的下一步讲完整：不是继续调 RL，而是补 RM 目标轴和 attack span credit |
| Deep Dive 12：Attack taxonomy，roledrop / A6 code / identity disclosure 的产品级失败分类 | 接在 benchmark 与 credit 之间 | 面试里可以展示“不是泛泛安全”，而是把产品破角拆成可采样、可打分、可训练的失败族 |
| Deep Dive 13：多语言一致性从 SFT 到 RLHF 的闭环评测 | 回到业务起点 | 用户最早的核心痛点是小语种语言错配，这篇能把语言一致性从 SFT 目标升级为全链路 regression metric |
| Deep Dive 14：MoE actor 的训推一致性：verl / vLLM / LoRA hot-swap / GSPO 为什么会影响行为 | 接在系统可行性与 RL 动力学之间 | 解释“训练配置不是工程细枝末节”，MoE、LoRA merge、vLLM sampling 会改变线上行为，是高级工程判断点 |

如果只能新增一篇，优先写 Deep Dive 11。理由：它承接最后的业务结论“v6.10 没把 attack true break 压下去”，同时连接 RM 数据、judge、credit assignment、benchmark，是最能体现下一步研究判断的一篇。
