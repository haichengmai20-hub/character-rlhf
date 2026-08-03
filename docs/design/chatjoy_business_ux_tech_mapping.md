# ChatJoy 业务体验到技术手段映射

> 本文是当前整理版的显式映射表。它回答三个问题：业务目标是什么时候进入项目的、8D judge 每一维由什么数据/技术承接、当前证据能支持到什么程度。历史设计过程回看 `chatjoy_rm_v6_3_roledrop_design.md`，版本结果以 `../project_chatjoy_overview.md` 和 `../VERSION_MAP.md` 为准。

## 1. 先钉死目标演化

项目最终形成“抗破角 + 保富度”双目标，但这不是立项时的目标。面试应按下面的因果顺序讲：

| 阶段 | 当时主要问题 | 数据/技术手段 | 目标如何变化 |
|---|---|---|---|
| API / local base | API 在剧情深入后受 safety 拦截；小语种答复语言错配 | system prompt + API；Qwen/Gemma base 试上线 | 决定转本地后训练 |
| 初始 SFT | 语言一致性、system prompt/role card 贴合、RP 格式 | 生产 Milvus 会话、TOP20、27 语言、Gemini 重写；Qwen/Gemma SFT | 先把基础分布拉对，不以 roledrop/富度为主目标 |
| RM v1-v5 | SFT 上线仍有 wrong role / wrong language / length mismatch | chosen=SFT clean；4 类 rejected；4D pair-build heuristic；OpenRLHF RM | 把可解释错误变成偏好信号 |
| 第一阶段 RL | 语言一致性基本解决，但产品抽查出现破角；clean 动作/情感细腻度不足 | RM + anti-refusal + GRPO/GSPO；产品评测和人工抽查 | 从这里才把主目标升级为“抗破角 + 保富度” |
| RM v6/v6.3 | 旧 RM 看不清细粒度质量和 user-level roledrop | 5/6-source candidates、DeepSeek/Qwen cross judge、5D->8D、attack hard fail | 给双目标补可学习信号 |
| 后续 SFT/RL | 富度与防御互相牵制 | richness/defense SFT、anti-OOC、revoke、length shaping、per-source KL、clean/attack probe | 富度主要由 SFT 建，RL/RM/规则负责防御和保持；失败时回数据/RM/benchmark |

## 2. 三套信号不要混写

| 信号 | 使用阶段 | 内容 | 用途 |
|---|---|---|---|
| 4D pair-build heuristic | RM v1-v5 构对期 | `format / role / language / length`，默认 `0.30/0.30/0.25/0.15`，按 rejected type 调 profile | 快速过滤和重采样，不是 LLM judge，不直接进入在线 RL |
| 8D LLM-judge rubric | RM v6.3 clean/fine pair | `role_fidelity / user_relevance / interaction_progression / natural_style / format_length_quality / sensory_richness / emotional_complexity / narrative_layer` | 对同 prompt 多候选排序，构造 rank_delta pair |
| Hard fails | RM v6/v6.3 judge 与后续规则 reward | wrong language、refusal、empty/garbled、system leak、acting as user；attack 另有 `roledrop_execution` | 作为不可被低维质量抵消的门控，不占 8D 名额 |

## 3. 8D 业务体验到技术手段

| 8D 维度 | 用户能感知什么 | 何时进入主目标 | 数据 / judge 设计 | SFT / RM / RL 承接 | 当前证据边界 |
|---|---|---|---|---|---|
| 1. Role fidelity | 说话方式、身份和角色卡一致 | 初始 SFT 已关注；v6 细化 | 真实 role card；wrong_role pair；v6 同 prompt 多候选 blind rank | SFT 建 role prior；RM v1-v5 判显式错角；v6 judge 比较 voice；anti-OOC 防身份泄漏 | 基础角色一致性有训练闭环；陌生角色泛化和语义性格没有独立全量指标 |
| 2. User relevance | 回应当前用户，而不是 generic filler | RM v6 | 5D judge 原始维度之一；同 prompt 多 source 候选 | RM v6/v6.3 提供偏好信号；后续 selector 避免只追表面长度 | 有 judge/pair 设计证据，未找到单独的线上 uplift 指标 |
| 3. Interaction progression | 接住并推进剧情，不替用户做决定 | RM v6，后续 richness SFT | `interaction_progression`；best-of-N selector 也纳入 interaction | RM 区分拖戏/抢戏；SFT-6.9 通过 balanced selector 注入 | SFT-6.9 clean profile 通过；不能把改善单独归因给 RM 或 RL |
| 4. Natural style | 情绪贴合、语言自然、不像政策模板 | RM v6 | `natural_style`；DeepSeek/Qwen cross judge | RM 学软偏好；SFT 用高质量候选落到 actor 分布 | 维度过宽，v6.3 后又拆 sensory/emotional/narrative；没有单轴线上指标 |
| 5. Format and length quality | 动作/对白/内心结构稳定，长度不过短也不重复灌水 | 初始 SFT/RM 即存在 | SFT 结构化输出；早期 4D format/length；v6 `format_length_quality` | SFT 建格式；RM 判极端；RL 用 repetition/length gate、cap 和 GSPO | 格式/长度规则可追溯；length shaping 实测不能凭空创造语义富度 |
| 6. Sensory richness | 有触觉、温度、声音、环境和身体反应 | 第一阶段 RL 暴露富度不足后，在 v6.3 新增 | pure Gemma4-it 第 6 source；`sensory_richness`；density 辅助分 | RM v6.3 试图提供区分度；后续 SFT-6.9 selector 直接注入 | v6.3 有数据/训练证据；后续 RM clean 轴近乎平，实际富度恢复主要来自 SFT-6.9 |
| 7. Emotional complexity | 不是“笑/哭/叹气”模板，而是复合、矛盾和层次化情绪 | v6.3 新增 | `emotional_complexity` + base candidate + few-shot anchor | RM 负责排序；richness SFT 把高质量候选写进 actor | clean 富度恢复是综合结果，未找到 emotional 单轴的独立因果提升 |
| 8. Narrative layer | 有背景暗示、后果想象和叙事层次 | v6.3 新增 | `narrative_layer` + base candidate | RM v6.3 负责 judge/pair；SFT-6.9 主 selector 没把 narrative 单列为主优化项 | 有 rubric 和 pair 证据；不能声称 narrative 已被 RL 单独提升 |

## 4. 8D 之外的硬业务门

| Hard gate | 为什么不放进 8D | 技术手段 | 实测/状态 |
|---|---|---|---|
| 语言一致性 | 属于必须满足的基础约束，不应被文笔高分抵消 | 27 语言 SFT；`wrong_language` pair；judge hard fail；回归评测 | 初始主目标，第一阶段 RL/产品口径下明显改善；精确跨语言长期回归仍需补齐 |
| Roledrop / 身份破角 | 是后期产品定义的硬失败，不是“角色分低一点” | v6.3 attack pair、`roledrop_execution`、anti-OOC -5、task-compliance 检测、revoke +3、per-source KL | 产品 v1/v2 为 91%/55%（v1 具体 merge 未记录，v2=RL-6.4 step160）；v6.10 step90 true break 36.8%，未优于 v6.9c 33.8% |
| 拒答/安全拦截 | 业务希望在角色内继续互动，但不能用“非拒绝”替代“不破角” | anti-refusal；破角时撤销 non-refusal bonus；hard fail | `+3` 曾产生 Goodhart，后用 anti-OOC/revoke 修正方向 |
| System leak / acting as user / empty/garbled | 任何一个都足以使候选不可用 | judge hard fail、规则检测和数据清洗 | 有规则/代码证据；不作为软质量维度平均 |

## 5. 最终双目标如何落地

| 最终目标 | 数据层 | RM/judge 层 | RL/rule 层 | 评测与结论 |
|---|---|---|---|---|
| 抗破角 | roledrop attack curriculum；defense SFT 2,691 + replay 10,764；后续 120/2,686 defense 补丁 | v6.3 attack pairs + hard fail；后续 RM v6.6 主要在 attack 轴有区分度 | anti-OOC -5、task compliance、revoke +3、attack 较松 KL | 真实 break 必须同 detector 去误报；v6.10 未压下 true break，下一步回 targeted data/RM/benchmark |
| 保富度 | base teacher candidate；SFT-6.8/6.9/b/c 的 action/dialog/sensory/interaction/emotion selector 与 replay | 8D rubric 尝试补 sensory/emotional/narrative，但后续 RM clean range 仅 0.037 | length shaping 只能守格式/长度；per-source tight KL 用于守 SFT 分布 | SFT-6.8 monologue 过冲；SFT-6.9 clean PASS；说明富度主要靠数据注入，不靠 RL 凭空创造 |

双目标不是一个加权和就能解决：破角是 hard constraint，富度是 soft preference；clean 与 attack 又需要不同 KL。项目因此同时用了阶段分工、hard-fail 门控和 per-source 解耦。

## 6. RM v6.3 数据和配置

| 组成 | Train | Eval | 说明 |
|---|---:|---:|---|
| RM v5 anchor | 17,712 | 885 | 保留基础 wrong role/language/length 能力 |
| Lever1 clean/fine | 20,574 | 2,303 | 6-source、8D judge、rank_delta pairs |
| Lever2 roledrop attack | 2,265 | 235 | train attack 5.59%，eval attack 6.87% |
| 总计 | 40,551 | 3,423 | 训练完成并 merge |

训练配置：Gemma4 SFT2 base，ZeRO-2 + 4bit QLoRA，batch 64，max_len 3072，LR 9e-6，LoRA r128/alpha256，target q/k/v/o/up/down，约 634 steps。

设计上限 `23 roles x 240 attacks x 6 sources = 33,120 candidates` 没有全跑。实际使用 1,500 prompt subset，生成 3,068 pairs，cap 到 2,500 curriculum pairs，最终落盘 attack 为 train 2,265 / eval 235。`acc>=0.84 / saturated=0% / reward_std>=2.0` 是设计验收门，仓库未找到 v6.3 最终数值，不能写成已通过结果。

## 7. Reward 口径与可复现边界

早期 RL 可由当前仓库代码追到：

```text
total_reward = rm_score
             + anti_refusal
             + anti_repetition
             + anti_ooc
             + bonus_revoked
             + length_shaping
```

KL 在 actor loss，不直接加进 reward。RL-6.3/6.4/6.5 实跑 launcher 是 attention-only LoRA r8/alpha8、entropy 0、LR 1.5e-6 constant、rollout N=6；`rank32 + FFN + entropy0.01` 是早期设计候选，FFN/router 后来进入 SFT 富化线。

v6.10 的设计/远端记录口径是：

```text
clean:  clean_q[0, 0.3] + KL_CLEAN=0.008
attack: rm + break_pen(-5) + 0.25*clean_q + KL_ATTACK=0.002
```

step90 结果有报告证据，但当前仓库没有保留完整 `launch_rl_v6.10.sh` 和关闭 clean RM 的 patch；入库通用 launcher 仍全局挂 RM，当前 `anti_chain_reward.py` 也没有 attack-side `0.25*clean_q`。因此这套公式只能标作“远端设计/运行口径”，不能声称本地代码可完整复现。

## 8. 仍需保留的冲突与非目标

- 口述中的 Qwen3-35B-A3B 与仓库 SFT 配置 Qwen3-30B-A3B 不同；精确训练配置以 30B-A3B 回答。judge 产物中的 Qwen 命名又是另一用途，不混成 SFT 基座。
- `sample_weight` 虽写入 JSONL，但当前 OpenRLHF `RewardDataset` 未消费；实际生效的是类型重采样和显式重复。
- 多轮记忆没有专项训练，不能列成“已提升的第 8 维”。
- span-level credit 是高优先级假设，相关 diagnostic 尚未执行；不能把 v6.10 未改善单因归结为 span credit。

## 9. 证据入口

- 阶段/数字台账：[`../project_chatjoy_overview.md`](../project_chatjoy_overview.md)
- 版本结果：[`../VERSION_MAP.md`](../VERSION_MAP.md)
- RM v6.3 历史设计：[`chatjoy_rm_v6_3_roledrop_design.md`](chatjoy_rm_v6_3_roledrop_design.md)
- RM/RL 原始交接：[`../archive/chatjoy_rm_v6_3_roledrop_handoff.md`](../archive/chatjoy_rm_v6_3_roledrop_handoff.md)
- v6.10 设计与停线结果：[`chatjoy_rl_v610_defense_clean_aware_design.md`](chatjoy_rl_v610_defense_clean_aware_design.md)
- RM v6 judge 代码：[`../../training/GRPO_Training_Package/rm_v6_finegrained/judge_candidates_llm_openai.py`](../../training/GRPO_Training_Package/rm_v6_finegrained/judge_candidates_llm_openai.py)
- 早期 reward 代码：[`../../training/GRPO_Training_Package/anti_chain_reward.py`](../../training/GRPO_Training_Package/anti_chain_reward.py)
