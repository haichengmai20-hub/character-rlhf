# ChatJoy RLHF Pipeline Timeline

本文把标准 RLHF 生产流程映射到本项目的真实演化：API/local base 问题 -> SFT -> RM v1-v5 -> RL 第一阶段 -> RM v6/v6.3 -> 多轮 SFT/RL 修复。

职责边界：本文只做阶段索引和面试时间线；精确事实台账以 `docs/project_chatjoy_overview.md` 为准，算法机制与接口细节去 `RLHF_WORKFLOW.md`，不在三处建立相互竞争的 SSOT。

读法：先看“阶段总表”，再看每个阶段的数据、配置、reward 和结论；要把 deep dives 串成面试逻辑，读 [deep_dives/00_interview_synthesis.md](deep_dives/00_interview_synthesis.md)。版本号不要混用：RM v1-v6 是 reward model 线，SFT v6.x 是 actor 数据/富度/防御线，RL-6.x 是策略训练线，产品 v1/v2 是交付口径。

## 阶段总表

| 阶段 | 业务问题 | 数据/模型 | 训练与 reward | 结果 |
|---|---|---|---|---|
| 0. API/local base | API safety 报错，小语种语言错配 | API + system prompt；Qwen/Gemma base 试上线 | 无训练或只靠 prompt | 需要业务后训练 |
| 1. SFT | system prompt 贴合、语言一致性 | TOP20 角色、27 语言、57,197 pre-API，clean32 SFT1 13,128 / SFT2 19,508 | Qwen/Gemma4 QLoRA/LoRA SFT | 基础语言/角色一致性改善 |
| 2. RM v1-v5 | SFT 后仍有 wrong role/language/length | chosen=SFT clean；rejected=wrong_role/wrong_language/length_mismatch | OpenRLHF/Qwen RM，4D pair-build heuristic，ZeRO-2 QLoRA | RM v5 离线最强，17,712 train，acc 95.5%，gap 3.497 |
| 3. RL 第一阶段 | 原语言/角色目标基本改善后，产品新暴露破角与富度不足 | actor=SFT2；RM v5/Gemma4 RM | `rm + anti_refusal`，后续加 rule reward；GRPO/GSPO | 目标从基础一致性升级为“抗破角 + 保富度”；同时出现非拒绝 bonus Goodhart |
| 4. RM v6/v6.3 | RM 需要看见富度和 roledrop | 5/6-source 多候选，DeepSeek+Qwen cross judge，attack data | 8D rubric，v5 anchor + clean/fine + attack | RM v6.3 train 40,551，attack 5.59% |
| 5. RL2/防御 SFT/RL3 | attack 降不动，clean 不变短 | defense SFT、v69 rich/defense mix、RM v6.6 | anti-OOC、revoke bonus、length shaping、per-source KL | 富度靠 SFT 恢复；attack true break 仍难压 |
| 6. RL4 / v6.10 | clean 与 attack 目标冲突 | actor=v6.9c，RM v6.6 | 远端记录的 clean-aware reward，`KL_CLEAN=0.008`、`KL_ATTACK=0.002`；完整 launcher/patch 未入库 | step90 clean held，但 true break 36.8%，未优于 v6.9c |

## 0. 从 API/system prompt 到本地训练

业务是多语言虚拟伴侣。原始方案依赖 system prompt + API 生成，能够快速支持角色，但有两个上线风险：

- API safety 在剧情深入后容易拦截，特别是露骨剧情附近。
- Qwen/Gemma base 或 API 在小语种里可能语言不一致，破坏伴侣体验。

因此 pipeline 的第一个决策不是“直接 RL”，而是先做 SFT，把语言、角色、格式这些基础分布拉回业务可控范围。

## 1. SFT：建立基础行为分布

### 数据

| 步骤 | 数量 | 说明 |
|---|---:|---|
| 原始 Milvus/后端导出 | 2,080,698 行 | 生产会话数据 |
| TOP20 assistant pairs | 679,299 | 聚焦高频角色 |
| 初清候选 | 631,995 | 初步去噪 |
| TOP20 x 27 language pre-API | 57,197 | 每语言最多 2,200，部分语言不足额 |
| SFT1 clean | 13,128 | clean32 第一波 |
| SFT2 clean | 19,508 | 后续主 SFT 数据 |

口述里的“每语言 2,160、6-turn round、41,536 -> 34,560 -> 22,680”保留为面试叙事中的轮级抽样/清洗思路；当前仓库事实表使用 57,197 / 13,128 / 19,508。

### 训练

| 模型 | 配置 | 结果 |
|---|---|---|
| Qwen3-30B-A3B SFT2 | LlamaFactory + Accelerate FSDP FULL_SHARD，8 GPU，4bit QLoRA q/k/v/o r8/alpha16，LR 1e-4 cosine/warmup0.1，cutoff 1536，batch 1，grad_acc 2，2 epoch | 2,440 steps，train loss 1.2335 |
| Gemma4 SFT2 | 3GPU DDP，4bit LoRA q/k/v/o r32/alpha64，LR 1e-4 cosine/warmup0.03，cutoff 3072，batch 1，grad_acc 8，2 epoch | step 1,626，train loss 1.1168 |

### 结论

SFT 解决的是基础分布：语言一致、角色 prompt 贴合、RP 格式稳定。上线后仍有 wrong role / wrong language 的边界案例和长度异常；破角与富度是在第一阶段 RL/产品抽查后才被提升为主要目标，不应倒写成初始 SFT 的验收项。

## 2. RM v1-v5：把错误模式变成 preference

### Pairdata

| 类型 | 构造方式 | 目标 |
|---|---|---|
| `wrong_role` | user prompt/目标角色不变，替换或错配 system prompt/角色回复 | 让 RM 学会角色一致性 |
| `wrong_language` | 角色不变，回复语言偏离用户语言 | 让 RM 学会语言一致性 |
| `length_mismatch_short` | 回复过短或缺 RP 格式 | 惩罚信息量不足 |
| `length_mismatch_long` | 回复过长、重复或失控 | 惩罚长度/重复异常 |

4D pair-build heuristic：`format / role / language / length`，默认权重 `0.30 / 0.30 / 0.25 / 0.15`；目标配比 wrong_language 25%、wrong_role 25%、short 30%、long 20%。它只用于构对过滤，不是 RM v6 之后的 LLM judge。

### 版本时间线

| 版本 | 数据 | 训练 | 结果 | 下一步原因 |
|---|---:|---|---|---|
| RM v1 | 16,826 train / 885 eval | batch128，263 steps，LR 9e-6，ZeRO-2，4bit LoRA r64 q/k/v/o | acc 94.5%，gap 1.601 | 首个可用，但需修数据分布 |
| RM v2 | 14,826 | batch64，231 steps | acc 86.4%，gap 1.122 | batch 变小和 score head 保存问题，退化 |
| RM v3 | 18,955 | batch128，148 steps | acc 86.0%，gap 1.256 | steps 短、bad long 样本污染 |
| RM v4 | 计划版 | 未正式训练 | 无结果 | 被 v5 覆盖 |
| RM v5 | 17,712 | batch64，276 steps，LR 9e-6，LoRA r64 alpha128 q/k/v/o | acc 95.5%，gap 3.497 | 离线强；后续第一阶段 RL 才暴露其富度/roledrop 覆盖不足 |
| Gemma4 RM v5 | 同 v5 | Gemma4 SFT2 base | acc 0.975，gap 6.462，wrong 2.7% | saturation 47.3%，需 normalize/clamp |

## 3. RL 第一阶段：从语言一致性到破角/富度问题

初版 reward：

```text
total_reward = online_rm_score + anti_refusal_custom_reward
```

随后扩展为：

```text
total_reward = rm_score + anti_refusal + anti_repetition + anti_ooc + bonus_revoked + length_shaping
```

关键演化：

| 时间/版本 | 改动 | 结果 |
|---|---|---|
| 早期 RL | RM v5/Gemma4 RM + anti_refusal，KL 在 actor loss | 语言一致性改善，但破角和富度不足 |
| 加 `NON_REFUSAL_BONUS=+3` | attack 非拒绝加分 | Goodhart：非拒绝但自曝模型/系统身份也得分 |
| 加 `anti_ooc=-5` 和 revoke | 破角/OOC 后撤销 `+3` | 修复 reward hacking，但需要控制 false positive |
| 加 length shaping | clean-only，过 RP/非拒绝/非破角/非重复 gate 后给长度 bonus | 只调长度无法创造动作块和语义富度 |

训练技术：verl + vLLM，GRPO/GSPO；KL 不进 reward，而是在 actor loss 里约束。硬件侧避免 ZeRO-3：无 NVLink/P2P、MoE 结构和 31GB 级显存使 ZeRO-3/FSDP2 通信和 full-state 风险较高，实际采用 ZeRO-2/FSDP1 + 4bit LoRA。

## 4. RM v6/v6.3：补富度和 roledrop 信号

### v6/v6.2

- 从启发式错例转向同 prompt 多候选 fine-grained pair。
- 初始 5-source：Gemma4 SFT 两采样、Qwen SFT 一采样、RL step550 两采样。
- DeepSeek + Qwen cross judge，减少单 judge 方向错误。
- v6.2 cross-validated fine pairs 27,287，保留率 73.1%。
- v6.2 训练：42,270 train / 3,614 eval，LR 5e-6，acc 0.802，wrong 18.0%，判断仍偏 underfit。

### v6.3

6-source：Gemma4 SFT 两采样、Qwen SFT 一采样、RL step550 两采样、pure Gemma4-it base 一采样。

8D rubric：5 个原维度 `role fidelity / user relevance / interaction progression / natural style / format-length quality`，再加 `sensory richness / emotional complexity / narrative layer`。wrong language、拒答、system leak 和 `roledrop_execution` 是 hard fail，不占 8D 名额。

数据混合：

| 来源 | train | eval |
|---|---:|---:|
| RM v5 anchor | 17,712 | 885 |
| Lever1 clean/fine roleplay | 20,574 | 2,303 |
| Lever2 roledrop attack | 2,265 | 235 |
| 总计 | 40,551 | 3,423 |

训练：Gemma4 SFT2 base，ZeRO-2 + QLoRA 4bit，batch 64，max_len 3072，LR 9e-6，LoRA r128/alpha256，target q/k/v/o/up/down，约 634 steps。

## 5. RL2 / 防御 SFT / RL3 / RL4

| 口述阶段 | 仓库实验线 | 数据/参数 | 现象 | 结论 |
|---|---|---|---|---|
| RL2 | RL-6.3 / 产品 v1 | RM v6.3，psKL，`NON_REFUSAL_BONUS=+3` | 语言一致性解决；attack 破角高；clean 基本不动 | reward hacking，非拒绝 bonus 奖励破角 |
| per-source KL | RL-6.3 后续 | clean tight，attack loose；v6.10 为 clean 0.008 / attack 0.002 | 改善 clean/attack 拉扯 | 需要强 attack reward 配合 |
| 防御 SFT | RL-6.4 前置 | 仓库实跑：SFT2 merged base，2,691 defense + 10,764 replay，q/k/v/o r16/alpha32，LR 5e-5，2 epoch，562 steps（口述：RL2/step 基座）；FFN/router 属后续富度 SFT | roledrop 降低但 clean 风格受影响 | attack 需要数据注入 |
| RL3 | RL-6.4/6.5/6.6b | anti-OOC、length shaping、rollout n=6、temp 1.2、RM v6.6 | roledrop 比产品 v1 降，但仍不如 base；clean 长度约 205 | 富度不能靠 RL 硬拉 |
| 富度 SFT | SFT-6.8/6.9/b/c | action/dialog/sensory/interaction/emotion selector，defense mix | v6.8 monologue 过冲；v6.9 clean PASS 但 defense erosion | SFT 能补富度，也会暴露防御退化 |
| RL4 | RL-6.10 | 远端口径：actor v6.9c，RM v6.6，GRPO+GSPO，temp 1.0，clean-aware reward；当前仓库缺完整 launcher/patch | step90 clean held，true break 36.8%，比 v6.9c 33.8% 略差 | 下一步应做 RM v7 / targeted attack data / benchmark；实现复现边界需主动说明 |

## 6. Reward 设计速查

| 阶段 | 公式/参数 | 说明 |
|---|---|---|
| 初版 | `rm_score + anti_refusal` | 拒答 -1，短无 RP -0.25 |
| Goodhart 期 | attack 非拒绝 `+3` | 压拒答，但奖励破角式非拒绝 |
| 修复期 | `anti_ooc=-5`，触发 OOC/task compliance 后 revoke `+3` | 把破角从“非拒绝”里分出来 |
| repetition | 4-gram 阈值约 0.18，罚分约 -0.20 到 -0.80 | 防重复刷长度 |
| length shaping | clean-only，floor 300，target 600/1200/1500，coef 0.4/0.5/1.0，后续 0.3 | 只奖励通过 gate 的 clean 回复 |
| v6.10（远端设计/运行口径） | clean：`clean_q[0,0.3] + tight KL`；attack：`rm + break_pen(-5) + 0.25*clean_q` | 分离 clean/attack；当前入库代码未保留 clean 关闭 RM 与 attack-side clean_q 的完整实现 |

## 7. 技术决策速查

| 决策 | 原因 |
|---|---|
| SFT 先于 RL | 语言、角色、格式是基础分布问题，RL 不适合从零修 |
| RM v1-v5 用明确错例 | wrong role/language/length 是线上可解释错误，适合 pairdata |
| RM v6 改多候选 judge | 富度和破角不是简单启发式错例能覆盖 |
| per-source KL | clean 和 attack 需要不同偏离空间；全局 KL 会把 attack 拉回 naive SFT |
| GSPO | sequence-level ratio 更适合 scalar RM 和 MoE actor |
| ZeRO-2/FSDP1 优先 | 无 P2P/NVLink + MoE + 31GB 显存下，ZeRO-3/FSDP2 风险高 |
| 富度回 SFT | 动作块/感官/情绪/叙事层次需要数据注入，不能只靠 length reward |
| attack 回数据/RM/benchmark | v6.10 证明只调 RL 参数不能压 true break |

## 8. 当前终点

当前项目最有价值的结论是负反馈链路已经闭环：

1. SFT 把基本分布拉对。
2. RM 把可解释错误变成 preference。
3. RL 暴露 reward hacking 和 credit assignment 问题。
4. RM v6/v6.3 补上富度和 roledrop judge。
5. 后续 SFT/RL 证明 clean 富度和 attack 防御存在目标冲突。
6. 下一步应优先做更强 attack RM、targeted attack data 和质量感知 benchmark，而不是继续盲调 rollout。
