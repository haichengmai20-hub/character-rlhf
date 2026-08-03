# ChatJoy Character RLHF 项目总览

整理时间：2026-07-08

本文是当前仓库的事实台账和面试追问证据底稿，不替代口述主线。口述主线看 `README.md` / `INTERVIEW_SCRIPT.md`；这里按“业务问题 -> SFT -> RM -> RL -> RM v6/v6.3 -> RL 反复迭代”整理数据、配置、reward 和可追溯证据。

## 口径原则

1. 叙事主线以用户口述为骨架：虚拟伴侣业务、API safety/local base 多语言问题、SFT、RM、RLHF、RM v6、后续 RL/SFT 迭代。
2. 面试逻辑优先采用口述主线；数字和配置用仓库可追溯记录补强。若两者口径不同，本文保留口述叙事价值，同时给出证据口径，方便面试追问时切换。
3. `README.md` 是口述入口，`docs/VERSION_MAP.md` 是版本事实索引，本文是追问台账，大型 handoff 是原始证据归档。
4. “产品 v1/v2”“RM v1-v6”“SFT v6.x”“RL-6.x”是不同命名轴，不能混写。

## 0. 业务背景

ChatJoy 是多语言 AI 虚拟伴侣/角色扮演业务。早期实现依赖 system prompt + API 生成，能够快速上线，但遇到两个核心问题：

- 剧情深入后，API 安全策略容易拦截或报错，尤其在露骨剧情附近更明显。
- 本地 base model 或 API 在小语种场景下会出现语言不一致：用户用 A 语言提问，模型用 B 语言回答。

因此项目进入后训练。第一阶段不是追求通用能力，而是让模型在业务约束下稳定：跟随角色 system prompt、用用户语言回复、保持角色扮演格式和合理回复长度。

这里的阶段顺序是本项目叙事的硬约束：抗破角和保富度不是初始 SFT/RM v1-v5 的并列目标，而是第一阶段 RL 后由产品抽查和行为退化共同暴露、在 RM v6/v6.3 之后才成为双目标。

## 1. SFT 阶段

### 目标

- 贴合角色 system prompt / role card。
- 保证 27 语言场景的语言一致性。
- 用业务真实会话分布约束模型，而不是只靠通用 instruction tuning。

### 数据来源和采样

| 项 | 仓库可追溯口径 | 说明 |
|---|---|---|
| 原始来源 | 生产 Milvus / 后端会话数据 | 按 `(user_id, character_id)` 时间序恢复会话，并绑定角色 prompt |
| 原始规模 | 2,080,698 行 | `RLHF_WORKFLOW.md` / case study 有记录 |
| TOP20 角色 assistant pairs | 679,299（汇总口径，原始报表未找到同名强证据） | 历史报表另记 44,241 个 TOP20 会话；两者统计单位不同，不互相替代 |
| 初清候选 | 631,995 | 初步去噪后候选 |
| 语言与采样 | 27 语言，每语言最多 2,200 条 | 理论 59,400 |
| pre-API 样本 | 57,197 | 部分语言不足额，平均约 2,118/语言 |

口述中提到“每语言 2,160 条、同一用户同一角色连续 6 条交互为一轮”。这适合作为面试解释业务抽样思路：按用户-角色会话流采样，而不是打散单轮 prompt。仓库可追溯产物以 `TOP20 x 27 x <=2200 -> 57,197 pre-API` 为证据口径；面试讲法仍可强调“按用户-角色会话流做轮级质量控制”，最终落盘训练文件再转成单条 instruction/output 格式。

### Gemini 重写与清洗

流程：

1. 从业务数据只保留 user prompt、system prompt、角色和语言 metadata，摘除原始 API assistant 输出。
2. 用更强 Gemini 系列 API 重写 assistant output。仓库脚本记录为只替换 output，不改 instruction/system/language。
3. API 重写参数：`temperature=0.7`、`max_output_tokens=2048`、`thinking=low`，并受 sexually explicit safety threshold 影响。
4. 清洗 API error/safety、placeholder leak、语言不一致等。
5. 输出 `clean32_for_sft1` 和 `clean32_for_sft2`。

仓库可追溯清洗结果：

| 产物/阶段 | 数量 | 说明 |
|---|---:|---|
| pre-API input | 57,197 | 27 语言 TOP20 角色采样 |
| Gemini 分段后保留（非 SFT2 全量上游） | 约 18,294 | 分段日志统计；不与 SFT2 clean 19,508 做单调递减比较 |
| SFT1 clean | 13,128 | 第一波 SFT 数据 |
| SFT2 clean | 19,508 | 后续主要 SFT 数据 |

口述中的 `41,536 -> 34,560 -> 22,680`、DeepSeek judge 聚合归一化 `>4` 和 `<4` 重写，适合作为“轮级清洗/质量控制/低分修复”的面试口径。建议这样处理：

- 主动讲“业务上做过轮级质量打分和低分重写，防止一轮对话里单条坏回复污染整轮”。
- 精确数字追问时，切到本表的可追溯产物口径，并说明轮级中间账与落盘训练文件口径不同。

### SFT 训练配置

| 模型 | 数据 | 框架/并行 | LoRA/QLoRA | 长度/batch | 结果 |
|---|---:|---|---|---|---|
| Qwen3-30B-A3B SFT2 | 19,508 | LlamaFactory + Accelerate FSDP FULL_SHARD，8 GPU | 4bit QLoRA，q/k/v/o，r8/alpha16/dropout0.05；LR 1e-4，cosine，warmup 0.1 | cutoff 1536，batch 1，grad_acc 2，effective batch 16，2 epoch | 2,440 steps，train loss 1.2335，runtime 17:40:47 |
| Gemma4 SFT2 | 19,508 | 3GPU DDP | 4bit LoRA，q/k/v/o，r32/alpha64；LR 1e-4，cosine，warmup 0.03 | cutoff 3072，batch 1，grad_acc 8，2 epoch | global step 1,626，train loss 1.1168，runtime 48,245s |

后续富度/防御 SFT 是另一条数据线，不要与第一波 SFT 混写：

| 版本 | 数据 | 作用 |
|---|---:|---|
| v69 balanced | 29,927 | rich CJK、clean replay、en/non-CJK、defense 平衡混合 |
| v69 final | 24,518 | rich 12,103 + replay 12,103 + defense 312 的早期口径 |
| v69b | 30,407 | 加入 step0 self-distilled defense |
| v69c repair | 4,566 | 2,686 defense + 1,200 clean replay + 440 zh-tw + 240 old defense |

后期 SFT 配置也按“实跑 / 未完整记录”区分：

| 版本 | 训练配置 | 证据状态 |
|---|---|---|
| RL-6.4 前置 defense SFT | base=SFT2 merged；4bit LoRA q/k/v/o r16/alpha32；cutoff 3072；batch1/grad_acc8；LR 5e-5 cosine/warmup0.03；2 epoch，562 steps | 配置与 handoff 均可追溯 |
| SFT-6.8 | 9,950（rich/replay 1:1）；4bit LoRA q/k/v/o/gate/up/down/router.proj r16/alpha32；cutoff 4096；batch1/grad_acc8；LR 5e-5 cosine/warmup0.03；1 epoch | task card/报告可追溯；结果为 monologue 过冲 |
| SFT-6.9 balanced | 29,927；base=step0；LoRA attn+FFN+router r24/alpha48；LR 5e-5；cutoff 4096；最终执行口径 1 epoch | 设计文档顶部执行纠错口径；不要引用底部旧“3 epoch”草案 |
| SFT-6.9b | 30,407；在 v6.9 配置上为新增 defense 运行 2 epoch | report 记录“训练就绪/后续结果”，完整 YAML 未入库 |
| SFT-6.9c repair | 4,566；base=v6.9b merged | 数据与结果可追溯；完整训练超参未入库，面试回答“未完整记录” |

## 2. RM v1-v5 阶段

### 目标

SFT 产品测试通过后，线上仍有语言不一致、错角色、长度异常等问题。RM 的目标是把这些业务错误显式转成 pairwise preference 信号，为后续 RLHF 提供 reward。

### Pairdata 设计

| 字段 | 设计 |
|---|---|
| chosen | SFT clean/Gemini 重写后的正确回复 |
| rejected 来源 | SFT2 merged model 或 cross-sample 构造 |
| rejected 类型 | `wrong_role`、`wrong_language`、`length_mismatch_short`、`length_mismatch_long` |
| wrong_role | user prompt 和目标角色语境不变，但 system prompt / role reply 错配 |
| wrong_language | 角色不变，回复语言偏离用户语言 |
| length_mismatch | 过短或过长/重复；正式类型拆成 short/long |
| 4D pair-build heuristic | `format / role / language / length`；只用于构对过滤，不是 LLM judge |
| 默认权重 | `0.30 / 0.30 / 0.25 / 0.15` |
| rebalancing 目标 | wrong_language 25%、wrong_role 25%、length_short 30%、length_long 20% |
| sample weight 元数据 | `clip(0.4 + 0.6 * gap_norm, 0.3, 1.2)`；当前 OpenRLHF 数据集未消费，训练实际靠重采样/显式重复 |

口述中的 `wrong_length` 可以作为泛称，但文档正式写法采用仓库类型 `length_mismatch_short/long`。

### RM v1-v5 台账

| 版本 | 数据 | 训练配置 | 结果 | 结论 |
|---|---:|---|---|---|
| RM v1 | train 16,826 / eval 885 | Qwen RM，ZeRO-2，4bit LoRA r64 q/k/v/o，LR 9e-6，batch 128，263 steps | acc 94.5%，gap 1.601，chosen +0.948，rejected -0.653 | 首个可用 RM；step200 ckpt 因写入/磁盘问题丢失 |
| RM v2 | 14,826，删除 2,000 个低 gap `length_mismatch_long` | batch 64，231 steps | acc 86.4%，gap 1.122 | batch 变小导致 ranking loss 方差上升，另有 PEFT score head 保存问题 |
| RM v3 | 18,955，加入约 2,000 bad/defective 样本 | batch 128，148 steps | acc 86.0%，gap 1.256，chosen +0.958，rejected -0.298 | total_steps 太短，LR 过早衰减；low-gap long-bad 抬高 rejected |
| RM v4 | 构造脚本/计划版 | 未正式训练 | 无正式结果 | 被 v5 覆盖 |
| RM v5 | 17,712 = 完整 v1 base + 约 5% bad-case pool | batch 64，276 steps，ZeRO-2/QLoRA，LR 9e-6，LoRA r64 alpha128 q/k/v/o | acc 95.5%，gap 3.497，chosen +2.27，rejected -1.23 | 离线最强 Qwen RM，但在线 reward 分布偏软二分类 |
| Gemma4 RM v5 | 同 v5 数据/超参 | Gemma4 SFT2 merged，sdpa | acc_mean 0.975，gap 6.462，wrong 2.7% | 离线强，但 `abs(score) > 5` saturation 约 47.3%，需要 normalize/clamp |

### RM v1-v5 的经验

- 离线 acc/gap 只是必要条件，不保证在线 RL 有足够 reward_std。
- batch、steps、score head 保存细节都会直接影响 RM 可用性。
- v5 给了后续 RL 第一阶段可用 reward；直到第一阶段 RL/产品抽查后，才确认它对富度和 user-level roledrop 的定义不够细。

## 3. RL 第一阶段

### 目标

用 RM v5 或同族 Gemma4 RM 引导 actor，在语言一致性基础上继续提升角色表现。

### Reward 设计演化

初版：

```text
total_reward = online_rm_score + anti_refusal_custom_reward
```

现代链路：

```text
total_reward = rm_score
             + anti_refusal
             + anti_repetition
             + anti_ooc
             + bonus_revoked
             + length_shaping
```

组件说明：

| 组件 | 设计 | 风险/修复 |
|---|---|---|
| `rm_score` | online RM `/classify` 标量分 | 需保证离线/在线 tokenizer、EOS、pooling 一致 |
| `anti_refusal` | 初版拒答 -1；后续拒答 -4、短无 RP -0.25、attack 非拒绝 +3 | `+3` 会奖励破角式非拒绝 |
| `anti_ooc` | 系统/模型/代码/任务执行类 OOC hard penalty -5 | 修复非拒绝 bonus Goodhart |
| `bonus_revoked` | attack 中触发 OOC/task compliance 时撤销 `+3` | 让“非拒绝但破角”不再得利 |
| `anti_repetition` | 4-gram 重复阈值约 0.18，罚分约 -0.20 到 -0.80 | 防止长度 reward 被重复刷分 |
| `length_shaping` | clean-only，通过 RP、非拒绝、非破角、非重复 gate 后按长度给 bonus | 早期 target 600/1200/1500，coef 0.4/0.5/1.0；v6.10 降到 0.3 |

KL 不作为 reward 组件，而是在 actor loss 中控制偏离。早期全局 KL 难以兼顾 clean 和 attack，后续引入 per-source KL。

### 训练技术

| 技术 | 使用位置 | 取舍 |
|---|---|---|
| OpenRLHF RM | RM 训练和在线 scorer | pairwise RM 工程成熟，但要处理 score head、EOS、pooling 对齐 |
| verl | actor RL | 支持 GRPO/GSPO、vLLM rollout、per-source KL patch |
| GRPO | actor policy optimization | 依赖同 prompt 多 rollout 的 reward 方差；RM flat 时梯度弱 |
| GSPO | sequence-level ratio | 更适合 scalar sequence RM，降低 MoE token-level ratio variance |
| vLLM | rollout/RM classify/serving | 提供吞吐，但 LoRA hot-swap、multimodal config、显存参数需谨慎 |
| ZeRO-2/FSDP1 + 4bit LoRA | 主要可行训练组合 | 8x RTX5090、无 NVLink/P2P、单卡约 31GB，ZeRO-3/FSDP2 通信和 full-state 风险高 |

### 第一阶段现象

- 语言一致性得到解决或明显改善。
- 产品定义的破角/roledrop 成为主要问题。
- clean 富度不足，动作块和语义细腻度不够。
- 单纯调温度、长度 target、length bonus 无法稳定补富度。

## 4. RM v6 / v6.2 / v6.3 阶段

### 目标

RM v5 能识别错角色、错语言、长度问题，但不够理解富度和破角。RM v6 目标是加入更细的语义质量、叙事层次和 roledrop 防御信号。

### Fine-grained 多候选设计

| 阶段 | source 组成 | judge | pair 逻辑 |
|---|---|---|---|
| v6/v6.2 | Gemma4 SFT 两采样、Qwen SFT 一采样、RL step550 两采样 | DeepSeek + Qwen cross judge | 同 prompt 多候选排序，按 rank delta 构造 pair |
| v6.3 | 上述 5 source + pure Gemma4-it base 一采样 | DeepSeek + Qwen blind source/order | 6-source ranked candidates，按 rank_delta 构造强弱 pair |

注意：口述里“Gemma4 base 两参数”容易与旧文档的 base sampling setting 混淆。仓库 v6.3 的准确写法是：Gemma4 SFT 两采样、Qwen SFT 一采样、RL step550 两采样、pure Gemma4-it base 一采样。

### Rubric

v6.3 使用更细的 judge 维度：

- v6/v6.2 的 5D：role fidelity、user relevance、interaction progression、natural style、format and length quality。
- v6.3 新增 3D：sensory richness、emotional complexity、narrative layer，合计 8D。
- hard fail：wrong language、拒答、空/乱码、system leak、acting as user 等；这些是门控，不占 8D 名额。
- 辅助：`density_score`。
- attack hard fail：`roledrop_execution`。

不要把这套 8D LLM-judge rubric 与 RM v1-v5 构造 pair 时的 4D heuristic（format/role/language/length）混为一谈。8 个体验维度到技术手段和证据状态的显式映射见 `docs/design/chatjoy_business_ux_tech_mapping.md`。

`best > good > mid > bad > worse` 适合面试口述排序直觉；文档和代码层面更准确地写 ranked candidates / rank_delta。

### RM v6 系台账

| 版本 | 数据 | 配置 | 结果/结论 |
|---|---:|---|---|
| RM v6 | v5 anchor + fine pairs | Gemma4 RM 路线 | acc 0.744，saturation 0%，wrong 25.3%；方向对但 underfit |
| RM v6.2 | 42,270 train / 3,614 eval，cross-validated fine pairs 27,287，保留率 73.1% | LR 5e-6 run；后续建议 A+C 用 LR 9e-6、r128、target 扩 MLP | acc 0.802，wrong 18.0%；仍偏保守 |
| RM v6.3 | 40,551 train / 3,423 eval | Gemma4 SFT2 base，ZeRO-2 + QLoRA 4bit，batch 64，max_len 3072，LR 9e-6，LoRA r128/alpha256，target q/k/v/o/up/down，约 634 steps | 加入 6-source 和 roledrop attack，训练完成并 merge vLLM |

### RM v6.3 数据配比

| 组成 | train | eval | 占比/说明 |
|---|---:|---:|---|
| v5 train/eval anchor | 17,712 | 885 | 保留 RM v1-v5 的基础错误模式 |
| Lever1 clean/fine roleplay | 20,574 | 2,303 | 6-source roleplay/fine-grained pair |
| Lever2 roledrop attack | 2,265 | 235 | attack curriculum，T1/T2a/T2b = 1125/950/425 |
| 总计 | 40,551 | 3,423 | train attack 5.59%，eval attack 6.87% |

其他关键数字：

- Lever1 cross-validated pairs：22,877，保留率 76.7%。
- Lever2 attack prompts：先有 5,520 单轮 attack prompts；实际 v6.3 用 stratified subset 1,500，生成 3,068 attack pairs，再 cap 到 2,500 curriculum pairs。

## 5. RL 反复迭代阶段

### 口述阶段与仓库版本映射

| 口述阶段 | 仓库版本 | 数据/配置 | 现象 | 结论 |
|---|---|---|---|---|
| RL2 | RL-6.3 / 早期产品 v1 | actor=SFT2 merged，RM v6.3，psKL；05-29 加 `NON_REFUSAL_BONUS=+3` | clean_RM 约 1.30-1.35，attack break 15-16% 平台；产品测语言一致性解决但 attack 高 | `+3` 非拒绝 bonus 产生 Goodhart，破角式非拒绝会得分 |
| per-source KL 调整 | RL-6.3 后半 / v6.4 脚本 | clean tight、attack loose；常见 clean 0.003-0.005，attack 0.0015 | attack source 稀疏，全局 KL 会把模型拉回 attack-naive SFT reference | per-source KL 是基础设施，不是单独解法 |
| 防御 SFT | RL-6.4 前置 | 口述称 RL2/step 基座；仓库实跑 base=SFT2 merged，2,691 defense + 10,764 replay，QLoRA q/k/v/o r16/alpha32，LR 5e-5，2 epoch，562 steps。FFN/router 属后续 richness SFT | roledrop 降低但 clean 风格受影响 | attack 需要显式数据补足；两套基座口径并列保留 |
| RL3 | RL-6.4 / 6.5 / 6.6b | RM v6.3/v6.6，anti-OOC，length shaping，rollout n=6，temp 1.2 | 产品 v2 roledrop 约 55%，仍差于 base 约 27%；clean 长度约 205，富度不动 | 只调 RL 参数补不了富度 |
| 富度 SFT | SFT-6.8 / 6.9 / 6.9b / 6.9c | FFN/router LoRA，balanced selector，action/dialog/sensory/interaction/emotion | v6.8 monologue 187% 过冲；v6.9 clean PASS 但 attack raw 约 42.6%；v6.9b/c true break 32.4/33.8 | 富度靠 SFT 可恢复，但 defense erosion 成新问题 |
| RL4 | RL-6.10 | 远端记录：actor=v6.9c，RM v6.6，GRPO+GSPO，temp 1.0，`KL_CLEAN=0.008`、`KL_ATTACK=0.002`，break_pen=-5，clean_q λ=0.25；当前仓库缺 v6.10 launcher/完整 reward patch | step90 clean held；true break 36.8%，比 v6.9c 33.8% 略差 | 结果有报告证据；reward 分支不能仅靠当前本地代码完整复现 |

### 关键 reward 版本

| 阶段 | Reward 设计 | 目的 | 暴露的问题 |
|---|---|---|---|
| 初版 RL | `rm_score + anti_refusal`，拒答 -1，短无 RP -0.25 | 先让模型别拒答、保持 RP 格式 | 无法区分非拒绝但破角 |
| RL-6.3 Goodhart 期 | attack 非拒绝 `+3` | 压拒答 | 奖励“我是 Gemma/我是 AI”这类破角回答 |
| anti-OOC 修复 | OOC hard penalty -5，触发后 revoke `+3` | 让非拒绝但破角不得分 | 需要处理 false positive |
| length shaping | clean-only，过 gate 后按长度给 bonus | 提升动作块/富度 | 只调 length target 不会创造语义富度，可能被重复刷分 |
| v6.10 clean-aware（远端设计/运行口径） | clean 不加 RM，只用 `clean_q[0,0.3] + tight KL`；attack 用 `rm + break_pen + λ*clean_q` | 分离 clean 保持和 attack 防御 | true break 未降；当前入库通用 launcher 仍全局挂 RM，且未保留 attack-side clean_q patch，故实现细节待补档 |

证据边界：`docs/design/chatjoy_rl_v610_defense_clean_aware_design.md` 和远端 handoff 记录上述设计与 step90 结果；本地 `train_grpo_lora_actor_verl.sh`/`anti_chain_reward.py` 是较通用或较早快照。面试时把“结果已记录”和“当前代码可复现”分开回答。

### RL 技术取舍

- GRPO 要求同 prompt 多 rollout 有足够 reward_std；如果 RM flat 或同组全同质，数值变化不会形成有效梯度。
- GSPO 用 sequence-level importance ratio，配 `seq-mean-token-mean`，更适合 sequence RM 和 MoE actor。
- per-source KL 需要在 verl trainer/loss 里把 `data_source` 变成 token-level mask；不能只依赖 tensordict 的非 tensor 字段。
- RL-6.3/6.4/6.5 入库 launcher 的长跑实参为 attention-only LoRA r8/alpha8、entropy 0、LR 1.5e-6 constant、warmup 0、rollout N=6。旧设计中的 rank32/FFN/entropy0.01 没有作为这些版本的长跑配置执行。
- Actor/RM 常分 GPU pool：actor GPU4-7、RM GPU0-3，跨池只传样本/奖励，避免 ZeRO-3 式全局通信。
- Gemma4 MoE + vLLM 的 LoRA merge mode 在 31GB 卡上不可行，实际使用 hot-swap；纯文本 merge 需处理 multimodal config/tensor。

## 6. 版本号审计：哪些是真改进，哪些只是实验编号

版本号本身不是结论。当前仓库最容易误读的是：同一个 `v6.x` 可能属于 RM、SFT、RL 或产品交付口径；同一条 RL 线里的 `rl_full_v3/v4/v5` 又只是训练启动/重启计数。

| 编号 | 它是什么 | 是否代表真实改进 | 应如何讲 |
|---|---|---|---|
| `rl_full_v3/v4/v5` | RL-6.3 内部 run/restart | 否 | 只作运行记录，不进版本主线 |
| RM v2/v3 | RM 数据/训练修正实验 | 否，指标退化 | 讲成排错样本：batch、score head、LR/steps、bad sample |
| RM v4 | 计划/脚本版 | 否，未正式训练 | 被 v5 覆盖 |
| RM v5 | 错误模式 RM 强基线 | 是，但能力范围有限 | 离线强，富度/破角不足 |
| RL-6.3 / 产品 v1 | 早期 RL Goodhart 期 | 业务上失败 | 语言一致性改善，但 roledrop 很差 |
| RL-6.4 / 产品 v2 | anti-OOC + 防御 SFT 后再 RL | 局部改善 | 91% 降到 55%，仍差于 base 27% |
| RL-6.5 | 硬化 GRPO | 否 | 375 步零移动，证明 val/probe 分布不对 |
| SFT-6.8 | 富度 SFT-max | 否 | monologue 过冲 187%，失败样本 |
| SFT-6.9 | balanced richness SFT | 是 | clean 富度恢复 |
| SFT-6.9b/c | defense SFT 扩充 | 否/有限 | defense 数据放大后 true break 不降，证明天花板 |
| RL-6.10 | clean-aware RL | 否/有限 | clean held，但 attack true break 未降，停线转 RM/数据/benchmark |

面试时建议用“版本号审计”主动化解复杂命名：不是版本越多越厉害，而是每个编号对应一个可证伪判断。

## 7. 面试可讲的完整链路

可以按下面 7 句话展开：

1. 业务是多语言虚拟伴侣，prompt/API 方案能快速上线，但 API safety 和小语种语言错配让本地训练成为必要。
2. SFT 用生产 Milvus 会话，按用户-角色会话流抽 TOP20 角色和 27 语言，用 Gemini 重写 assistant output，再过滤 API error、占位符和语言不一致，训练 Qwen/Gemma4 SFT。
3. SFT 上线后产品抽查仍有语言和角色问题，于是 RM v1-v5 用 chosen clean、rejected wrong_role/wrong_language/length_mismatch 显式训练 preference model。
4. RM v5 离线指标很强，但第一阶段 RL 暴露 reward hacking、破角和富度不足，说明 RM 还没有看见产品真正关心的细粒度质量。
5. RM v6/v6.3 改成 6-source 多候选、DeepSeek+Qwen cross judge，并加入 sensory/emotion/narrative 和 roledrop attack 数据。
6. 后续 RL 通过 anti-OOC、revoke bonus、length shaping、per-source KL、GSPO/GRPO 修 reward，但 clean 富度主要靠 SFT 数据补，RL 不能凭空补出动作块和细腻语义。
7. 最终经验是：SFT 负责分布和表达，RM 负责把业务偏好变成可学信号，RL 负责在信号可靠时放大；当 attack true break 不降时，要回到数据、judge 和 benchmark，而不是继续盲调 RL 参数。

## 8. 证据索引

| 主题 | 主要文件 |
|---|---|
| 入口和叙事 | `README.md`、`INTERVIEW_SCRIPT.md`、`CASE_STUDY.md` |
| 版本事实 | `docs/VERSION_MAP.md`、`PIPELINE_TIMELINE.md` |
| SFT 数据/训练 | `RLHF_WORKFLOW.md`、`DATA.md`、`training/data_prep/`、`training/configs/`、`docs/runbooks/training_runbooks.md` |
| RM v1-v5 | `CASE_STUDY.md`、`RLHF_WORKFLOW.md`、`training/data_prep/prepare_rm_data_weighted.py`、`training/GRPO_Training_Package/build_rejected_from_sft.py` |
| RM v6/v6.3 | `docs/archive/chatjoy_rm_v6_finegrained_handoff.md`、`docs/archive/chatjoy_rm_v6_3_roledrop_handoff.md`、`docs/design/chatjoy_rm_v6_3_roledrop_design.md`、`deep_dives/02_rm_data_reward_and_attribution.md` |
| RL reward/训练 | `training/GRPO_Training_Package/anti_chain_reward.py`、`training/GRPO_Training_Package/length_shaping_reward.py`、`training/GRPO_Training_Package/framework_bridge/verl/`、`deep_dives/03_rl_dynamics_params_and_strategy.md` |
| Deep Dive 综合路线 | `deep_dives/00_interview_synthesis.md`：把 01-09 串成面试逻辑链，并标注哪些版本是真改进/失败实验/run 编号 |
| 富度/防御冲突 | `deep_dives/04_richness_defense_tension_and_sft_ceiling.md`、`deep_dives/05_when_to_move_from_sft_to_rl.md`、`deep_dives/08_credit_assignment_in_character_rlhf.md`、`deep_dives/09_full_chain_attribution.md` |

## 9. 待对账口径

以下内容来自口述主线，适合保留在面试故事中，但当前仓库未找到直接同名报表或强证据，后续如找到原始产物可再更新：

| 口述点 | 当前处理 |
|---|---|
| 每语言 2,160 条 | 面试口径写“每语言约 2.1k”；证据口径是每语言最多 2,200，实际 57,197 pre-API |
| 6 条交互为一轮 | 作为业务抽样解释保留；训练落地文件按单条 instruction/output 组织 |
| SFT 中间账 `41,536 -> 34,560 -> 22,680` | 作为轮级清洗/质量控制口径保留；精确落盘数字以 57,197 / 13,128 / 19,508 证据口径回答 |
| SFT 阶段 DeepSeek judge `>4` 和 `<4` Gemini 重写 | 保留为“做过质量打分/重写机制”的叙事，但不绑定到 clean32 数字 |
| 原业务 API = Gemini 3 Flash | 作为口述前史保留；仓库未找到可核对的线上 API 配置，不把它写成代码事实 |
| Gemini 3 Pro | 仓库脚本只证实 Gemini 系列 API 重写；文档写“更强 Gemini 系列 API” |
| Qwen3-35B-A3B | 口述保留原称；仓库配置和训练记录为 Qwen3-30B-A3B，精确配置追问按 30B-A3B 回答 |
| Qwen judge 名称 | v6 handoff/产物曾写 Qwen3.6-35B-A3B、Qwen-36B 等历史简称；主线只写“Qwen judge”，避免与 SFT 的 Qwen3-30B-A3B 混淆 |
| TOP20 / 17 / 23 / 83 角色 | TOP20 是初始业务采样名；17 是部分 RM 分布核查口径，23 是 attack 目录快照，83 是生产角色总量；按阶段分别回答，不混算 |
| 27 语言集合 | 27 是业务采样数量；历史 report 与当前脚本的具体语言列表不完全一致，当前仓库未找到统一冻结清单 |
| 679,299 vs 44,241 | 679,299 被汇总为 assistant pairs，44,241 是历史 report 的 TOP20 会话数；因统计单位和原始证据层级不同，均保留并标单位 |
