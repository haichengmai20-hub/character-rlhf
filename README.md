# ChatJoy Character RLHF 文档入口

这个仓库记录的是一次多语言 AI 虚拟伴侣从 API/system prompt 方案迁移到本地模型、再通过 SFT/RM/RLHF 逐步扩展优化目标的工程复盘：最初先解决语言一致性和 system prompt 贴合；第一阶段 RL 暴露破角与富度退化后，项目目标才升级为“抗破角 + 保富度”。

当前整理版的目标不是把所有历史日志平铺出来，而是服务面试复盘：先讲清业务为什么要训练，再讲每个阶段的数据、训练配置、reward 设计、失败现象和下一轮改动。

文档治理原则：面试口述主线优先，工作区真实记录用来补强证据、补齐数字和准备追问；如果原始记录的流水账顺序不如口述逻辑清楚，就按更适合面试的因果链重排。数字分成“面试叙事口径”和“仓库可追溯口径”，主动讲主线，被追问时再切到证据表。

## 怎么读

面试准备先读三份：

1. `README.md`：建立业务主线、阶段台账和关键结论。
2. `INTERVIEW_SCRIPT.md`：把项目讲成 30 秒 / 3 分钟 / 追问应对三层口述。
3. `CASE_STUDY.md`：把“训练了很多版本”讲成“发现问题、归因、改信号、停线判断”的能力故事。

追证据时再看：`docs/project_chatjoy_overview.md` 查数据/配置/reward 台账，`docs/VERSION_MAP.md` 查版本和指标，`PIPELINE_TIMELINE.md` 查标准 RLHF pipeline 与项目阶段的映射，`RLHF_WORKFLOW.md` / `DATA.md` / `deep_dives/00_interview_synthesis.md` 查技术细节和深读路线。

大型 handoff、plan、debug log 保留证据价值，但不作为第一阅读入口。需要核对原始过程时再进入 `docs/archive/chatjoy_rm_v6_finegrained_handoff.md`、`docs/archive/chatjoy_rm_v6_3_roledrop_handoff.md` 等归档文件。

## 一句话主线

业务起点：API/system prompt 方案在剧情深入后受 safety 拦截，本地 Qwen/Gemma base 或 API 又会在小语种场景发生答复语言错配，因此先做业务后训练。

目标演化：SFT 先解决语言一致性和角色 system prompt 贴合，RM v1-v5 再把 wrong role / wrong language / length mismatch 做成偏好信号；第一阶段 RL 上线后，产品才把破角和 clean 富度不足明确为主要矛盾，随后 RM v6/v6.3 与多轮 SFT/RL 转向“被攻击仍在角色内化解，同时动作块和情感表达不退化”。

面试亮点：项目最有价值的不是“跑了很多版本”，而是每次线上/评测不动时能把问题归因到数据、RM、reward rule、credit assignment 或 benchmark，并据此决定回 SFT、改 RM、调 reward，还是停线。

## 面试讲法

### 1. 业务为什么从 prompt/API 走到本地训练

- 产品形态是多语言虚拟伴侣和角色扮演，用户期望模型始终用用户语言回应，并贴合角色设定。
- 早期方案是 system prompt + API。问题有两个：API 在剧情深入后容易因为安全策略报错；base/local model 上线后，小语种会出现用户语言和模型回复语言不一致。
- 业务因此进入后训练：先 SFT 解决语言一致性和 system prompt 贴合，再用 RM/RLHF 把可见问题转成可优化的 reward。
- 这里不要倒叙成“项目一开始就做抗破角和富度”：这两个目标是在第一阶段 RL 暴露行为退化、产品重新定义破角口径后才进入主线。

### 2. SFT 阶段：先把语言和角色对齐

目标：贴合 system prompt、稳定使用用户语言、保留角色扮演格式。

面试叙事和证据口径：

- 来源：生产 Milvus 会话，按 `(user_id, character_id)` 时间序重建 TOP20 高频角色真实会话，并绑定真实 role card/system prompt。
- 采样：27 种语言、每语言最多 2,200 条，理论 59,400，实际 pre-API 57,197。这个口径和口述中的“每语言约 2,160、6-turn round”是同一业务思路下的不同复盘口径；仓库强证据以 57,197 pre-API 为准。
- 重写：用 Gemini 系列 API 只重写 assistant output，不改 user prompt / system prompt / language metadata。脚本记录参数为 `temperature=0.7`、`max_output_tokens=2048`、`thinking=low`，并受到 API safety filter 影响。
- 清洗：过滤 API error/safety、占位符泄漏、语言不一致等问题；第一波 SFT 产物是 `clean32_for_sft1` 13,128 条、`clean32_for_sft2` 19,508 条。
- 训练：分别得到 Qwen 和 Gemma4 的 SFT 版本。

关键训练配置：

| 模型 | 框架/并行 | 数据 | LoRA/QLoRA | 长度与 batch | 结果 |
|---|---|---:|---|---|---|
| Qwen3-30B-A3B SFT2 | LlamaFactory + Accelerate FSDP FULL_SHARD，8 GPU | 19,508 | 4bit QLoRA，q/k/v/o，r8/alpha16；LR 1e-4，cosine，warmup 0.1 | cutoff 1536，batch 1，grad_acc 2，effective batch 16，2 epoch | 2,440 steps，train loss 1.2335 |
| Gemma4 SFT2 | 3GPU DDP | 19,508 | 4bit LoRA，q/k/v/o，r32/alpha64；LR 1e-4，cosine，warmup 0.03 | cutoff 3072，batch 1，grad_acc 8，2 epoch | global step 1,626，train loss 1.1168 |

口述中提到的 `41,536 -> 34,560 -> 22,680`、`<4 分重写` 保留为“轮级质量控制/低分重写”的面试叙事口径；当前证据表同时列出仓库可追溯的 57,197 / 13,128 / 19,508 产物口径。面试时先讲业务闭环：抽样、重写、过滤、judge、低分修复；只有被追问精确落盘文件时，再说明两套数字口径尚需对齐。

### 3. RM v1-v5：把错误模式显式做成 pairdata

SFT 产品测试口径通过后，线上仍能观察到语言不一致和角色不稳。于是训练 RM，让 reward 能区分错误类型。

pairdata 设计：

- `chosen`：来自 SFT clean 数据。
- `rejected`：用 SFT2 merged model 或 cross-sample 构造错误回复。
- 类型：`wrong_role`、`wrong_language`、`length_mismatch_short`、`length_mismatch_long`。
- wrong role：保留 user prompt 和目标角色语境，但替换/错配 system prompt 或角色回复。
- wrong language：保留角色，让回复语言偏离用户语言。
- length mismatch：把过短或过长/重复的回复作为 rejected。口述里“chosen 0.8-1.2 以外要惩罚”可以作为解释口径；仓库正式类型拆成 short/long。
- 4D pair-build heuristic：`format / role / language / length`，默认权重 `0.30 / 0.30 / 0.25 / 0.15`，并按 rejected 类型调整；它是构对过滤规则，不是 DeepSeek/Qwen LLM judge。
- 数据配比目标：wrong_language 25%、wrong_role 25%、length_short 30%、length_long 20%。
- `sample_weight` 元数据：`clip(0.4 + 0.6 * gap_norm, 0.3, 1.2)`；当前 OpenRLHF `RewardDataset` 未消费该字段，实际生效的是 25/25/30/20 重采样和显式重复。

RM v1-v5 台账：

| 版本 | 数据 | 配置 | 结果 | 结论 |
|---|---:|---|---|---|
| RM v1 | train 16,826 / eval 885 | Qwen RM，ZeRO-2，4bit LoRA r64 q/k/v/o，LR 9e-6，batch 128，263 steps | acc 94.5%，gap 1.601 | 首个可用 RM |
| RM v2 | 14,826，删除 2,000 个低 gap long 样本 | batch 64，231 steps | acc 86.4%，gap 1.122 | batch 变小和 score head 保存问题导致退化 |
| RM v3 | 18,955，加入 bad/defective 样本 | batch 128，148 steps | acc 86.0%，gap 1.256 | steps 太短、低 gap long-bad 污染 |
| RM v4 | 计划/脚本版 | 未正式训练 | 无正式结果 | 被 v5 覆盖 |
| RM v5 | 17,712，完整 v1 base + 约 5% bad-case pool | batch 64，276 steps，LR 9e-6，LoRA r64 alpha128 q/k/v/o | acc 95.5%，gap 3.497 | 离线最强 Qwen RM，但在线 reward 方差偏弱/接近软二分类 |
| Gemma4 RM v5 复现 | 同 v5 数据 | Gemma4 SFT2 base，sdpa | acc 0.975，gap 6.462，wrong 2.7% | 离线强，但 score saturation 高，需要 normalize/clamp |

### 4. RL 第一阶段：RM reward 够强还不等于行为变好

早期 RL 使用 RM v5 或 Gemma4 RM 作为在线 scorer，并叠加 rule reward。初版公式大致是：

```text
total_reward = online_rm_score + anti_refusal_custom_reward
```

后来扩展为：

```text
total_reward = rm_score
             + anti_refusal
             + anti_repetition
             + anti_ooc
             + bonus_revoked
             + length_shaping
```

关键点：

- KL 不直接加到 reward，而是在 actor loss 中作为约束。
- 初版 `anti_refusal` 只惩罚拒答；后来 attack 非拒绝给 `+3`，但这产生 Goodhart：模型可以通过自称模型、泄露系统身份来拿非拒绝 bonus。
- 因此加入 `anti_ooc=-5` 和 `bonus_revoked`：attack 中一旦触发 OOC / task compliance，就撤销非拒绝 bonus。
- `length_shaping` 是 clean-only，需要通过角色扮演格式、非拒绝、非破角、非重复等 gate；早期 target 600/1200，后续到 1500，coef 从 0.4/0.5 到 1.0，再在 v6.10 降到 0.3。

训练栈：SFT 用 LlamaFactory/Accelerate，RM 用 OpenRLHF，actor RL 用 verl + vLLM，算法主线是 GRPO/GSPO。硬件约束是 RTX 5090 级别多卡、无 NVLink/P2P，所以大模型训练主要采用 4bit base + LoRA + ZeRO-2/FSDP1，避免 ZeRO-3 的高通信和卡死风险。

第一阶段现象：

- 语言一致性明显改善。
- 产品重新定义的破角/roledrop 变成主要问题。
- clean 回复富度不足，动作块和语义细腻度不够。
- 单靠调 temperature、length target、length bonus，无法稳定补富度；富度更多要靠 SFT 数据注入，RL 只能在 RM 信号足够强时放大。

### 5. RM v6/v6.3：让 RM 看见富度和破角

目标：在 RM 中加入富度、叙事层次和 roledrop 防御信号。

v6/v6.2 从“启发式错例”转向同 prompt 多候选 fine-grained pair：

- 早期 5-source：Gemma4 SFT 两个采样、Qwen SFT 一个采样、RL step550 两个采样。
- v6.3 变为 6-source：Gemma4 SFT 两采样、Qwen SFT 一采样、RL step550 两采样、纯 Gemma4-it base 一采样。
- DeepSeek API 与 Qwen local/API 交叉 judge，blind source/order。
- 口述中的 `best > good > mid > bad > worse` 可以作为 rank 解释，但 v6.3 实际更准确写作 ranked candidates / rank_delta pair。
- 8D judge rubric：`role fidelity / user relevance / interaction progression / natural style / format-length quality / sensory richness / emotional complexity / narrative layer`。语言错配、拒答、system 泄漏和 `roledrop_execution` 属于 hard fail，不是把语言/安全硬凑进 8 个评分维度。

RM v6.3 数据：

| 组成 | train | eval | 说明 |
|---|---:|---:|---|
| v5 anchor | 17,712 | 885 | 保留原 RM v5 锚点，避免忘掉基础错误模式 |
| Lever1 clean/fine roleplay | 20,574 | 2,303 | 6-source clean/fine-grained judge pairs |
| Lever2 roledrop attack | 2,265 | 235 | attack curriculum，T1/T2a/T2b = 1125/950/425 |
| 总计 | 40,551 | 3,423 | train attack 占 5.59%，eval attack 占 6.87% |

RM v6.3 训练配置：Gemma4 SFT2 base，ZeRO-2 + QLoRA 4bit，batch 64，max_len 3072，LR 9e-6，LoRA r128 / alpha256，target `q_proj,k_proj,v_proj,o_proj,up_proj,down_proj`，约 634 steps，训练完成并 merge vLLM。

8 个体验维度何时进入目标、分别落到哪类数据/judge/reward，以及哪些只是设计目标而非实测收益，统一见 `docs/design/chatjoy_business_ux_tech_mapping.md`。

### 6. RL 反复迭代：clean、attack、富度三者互相牵制

口述 RL2/RL3/RL4 可以对应到仓库里的 v6.3-v6.10 实验线：

| 口述阶段 | 仓库对应 | 现象 | 改动 | 结论 |
|---|---|---|---|---|
| RL2 | RL-6.3 / 早期产品 v1 | 语言一致性解决，但 attack 破角仍高；clean 轴基本不动 | 引入 per-source KL，attack 更松、clean 更紧；修 SAVE_FREQ 和 reward | reward 数值变好不等于真实行为变好 |
| 防御 SFT | RL-6.4 前置 defense SFT | attack 需要显式数据补足 | 口述称 RL2/step 模型作基座；仓库实跑是 SFT2 merged + 2,691 defense / 10,764 replay，QLoRA q/k/v/o r16/alpha32，LR 5e-5，2 epoch，562 steps。FFN/router 是后续 richness SFT，不与本轮混写 | SFT 能注入防御，但会牵动 clean 风格 |
| RL3 | RL-6.4/6.5/6.6b | attack 降低但仍不如 base；clean 动作块变少、回答偏短 | anti-OOC、revoke bonus、length shaping、换 RM v6.6 | clean 富度不能主要靠 RL 修，需要高质量 SFT 数据 |
| 富度 SFT | SFT-6.8/6.9/6.9b/6.9c | v6.8 monologue 过冲；v6.9 clean 富度恢复但 defense erosion | balanced selector、action/dialog/sensory/interaction/emotion，高质量语义由 DeepSeek/proxy 选 | clean 富度恢复，attack 防御变成新瓶颈 |
| RL4 | RL-6.10 | clean held，但 true break 未降，step90 比 v6.9c 略差 | 远端设计/运行口径：clean 只用 clean_q + tight KL，attack 用 rm + break_pen + clean_q；当前仓库未保留 v6.10 launcher/关闭 clean RM 的完整 patch | 结果可追溯，但 reward 分支暂不能仅靠本地代码快照复现；瓶颈仍指向 attack RM/数据/评测粒度 |

关键 RL 配置和设计：

- actor RL：verl + vLLM rollout / RM classify，GRPO + GSPO。
- GSPO：sequence-level importance ratio，配 `seq-mean-token-mean`，更适合 scalar sequence RM 和 MoE 场景。
- per-source KL：clean tight KL 保质量，attack loose KL 允许偏离 attack-naive reference。v6.10 典型配置 `KL_CLEAN=0.008`、`KL_ATTACK=0.002`。
- rollout：多处实验使用 `ROLLOUT_N=6`，后续也试过更大 N；温度从 1.15/1.2 到 v6.10 的 1.0。
- RL-6.3/6.4/6.5 实跑 launcher：attention-only LoRA r8/alpha8、entropy 0、LR 1.5e-6 constant、warmup 0、rollout N=6；rank32/FFN/entropy0.01 属早期候选，不是这三轮长跑实参。
- v6.10 reward 分三层证据：设计文档/远端记录写作 clean=`clean_q[0,0.3] + tight KL`、attack=`rm + break_pen(-5) + 0.25*clean_q`；step90 报告确认 clean held、attack 未降；但当前入库通用 launcher 仍全局挂 RM，`anti_chain_reward.py` 也未包含 attack-side `0.25*clean_q`，所以本地代码快照不能证明该分支的完整实跑实现。

## 版本号怎么看

这个项目里很多编号不是“产品逐版变强”的意思，面试时要主动拆开：

| 名称 | 正确理解 |
|---|---|
| `rl_full_v3/v4/v5` | RL-6.3 内部重启/run 计数，不是模型能力版本。 |
| RM v2/v3 | 真实训练实验，但结果退化；价值是暴露 batch、score head、LR/steps 和 bad sample 问题。 |
| RM v4 | 计划/脚本版，未正式训练，被 RM v5 覆盖。 |
| RL-6.3 / 产品 v1 | 暴露 Goodhart 的早期 RL，不是成功版本。 |
| RL-6.4 / 产品 v2 | 有真实改善，但仍差于 base，不能讲成最终成功。 |
| RL-6.5、RL-6.7、RL-6.10 | 诊断价值大于产品价值：分别证明 val 分布、clean RM 梯度、attack credit 仍有瓶颈。 |
| SFT-6.8 | 失败实验，monologue 过冲；价值是证明 selector 会放大偏差。 |
| SFT-6.9 | clean 富度真实恢复，是后续防御/RL 的新起点。 |
| SFT-6.9b/c | defense SFT 尝试，未把 true break 压回 step0；价值是证明 SFT-defense 天花板。 |

更完整的版本去魅见 `deep_dives/00_interview_synthesis.md`。

## 关键经验

1. 业务问题要先拆成可被数据覆盖的错误模式：语言错、角色错、长度错、破角、富度不足，分别对应不同数据和 reward 信号。
2. SFT 负责把基础分布拉对，尤其是多语言、角色格式、动作块和语义富度；RL 不擅长凭空创造富度。
3. RM 离线 acc 高不代表在线 RL 有梯度；reward_std、score saturation、在线 classify 池化方式都可能让 RL 信号失真。
4. Rule reward 是必要的，但也最容易被 hacking；`+3` 非拒绝 bonus 必须配 anti-OOC 和 revoke 机制。
5. attack 数据占比低时，全局 KL 会把模型拉回不懂 attack 的 SFT reference；per-source KL 是必要基础设施，但不是单独解法。
6. ZeRO-3/FSDP2 在无 P2P/NVLink、MoE 结构和 31GB 级显存条件下成本高；本项目更稳定的组合是 4bit base + LoRA/QLoRA + ZeRO-2/FSDP1。
7. 文档阅读必须区分“产品版本”“RM 版本”“SFT 版本”“RL 实验版本”。例如早期产品 v1/v2 不是 RM v1/v2。

## 当前结论

截至当前整理口径，项目已经形成完整闭环：SFT 解决基础语言/角色一致性，RM v1-v5 建立错误模式 pairdata，RM v6/v6.3 引入富度与 roledrop judge，RL 多轮验证了 reward hacking、per-source KL、defense SFT 和 clean-aware reward 的边界。

最后一轮结论偏工程现实：clean 富度可通过 balanced SFT 恢复，但直接身份披露/A6 code 等 attack true break 没有被 v6.10 RL 压下去。下一步应优先做质量感知 RM v7、targeted attack data 和更细粒度评测，而不是继续只调 rollout 参数。

## 仓库结构

```text
character-rlhf/
  README.md                         # 入口和面试主线
  INTERVIEW_SCRIPT.md               # 口述脚本
  CASE_STUDY.md                     # 对外 case study
  PIPELINE_TIMELINE.md              # 标准 RLHF pipeline × 本项目时间线
  RLHF_WORKFLOW.md                  # 技术主线和阶段流程
  DATA.md                           # 数据/模型/产物边界
  docs/
    README.md                       # 支撑材料索引
    VERSION_MAP.md                  # 版本事实校准
    project_chatjoy_overview.md     # 整理版全量台账
    GLOSSARY.md                     # 术语表
    design/                         # 设计文档
    runbooks/                       # 训练/部署 runbook
    research/                       # 方法论与研究上下文
    archive/                        # handoff / plan / debug log 归档
  deep_dives/                       # 专题深挖；00_interview_synthesis 给面试路线
  training/                         # 数据处理、SFT/RM/RL 训练脚本和配置
  reports/                          # 评估与对账报告
```
