# 面试讲述脚本：ChatJoy Character RLHF 全仓库导读

> **用途**：面试 / 技术评审时的口述稿，从 30 秒电梯到全仓库精读再到被追问的分层应对。
> **约定**：L0 = 30 秒电梯；L1 = 2-3 分钟主线；L2 = 全仓库按目录块走；L3 = 被追问时的应对思路。
> **节奏提示**：L2 每块约 40-60 秒，整个 L2 控制在 5-7 分钟；L3 是被问时查表，不是连续背诵。

---

## L0 — 30 秒电梯

```
这个项目是为多语言角色扮演陪伴对话模型搭完整 RLHF 链路：
线上真实用户聊天 → SFT → 偏好对 → RM → GRPO 强化学习，
目标是让模型在沉浸式角色扮演里既保持富度，又能抗破角攻击。

项目最值钱的不是"跑通了流程"，而是在四次"指标没动"里，
把根因分别判到了评测层、RM 层、数据层、credit 层——
没有一次是靠调 RL 超参解决的，命中率 0/4。
这套差分诊断能力，是我在这个项目里最想体现的能力。
```

---

## L1 — 2-3 分钟主线叙事

```
项目起点：业务有真实用户，角色扮演对话需要同时满足"富度"和"防御破角"两个目标。
我从线上 Milvus 导出真实会话，经 SFT（LlamaFactory/QLoRA，Qwen3-30B MoE）
建起 baseline，再用 LLM-judge 交叉投票生成偏好对，训练 RM（OpenRLHF），
最后接 GRPO（verl + vLLM rollout，actor 是 Gemma4-26B MoE）。

第一轮 RL 暴露了一个关键发现：旧的 anti_refusal +3.0 奖励把
"我是 Gemma，但我可以陪你"这种破角回复判为"无拒绝词=合规"，
给了满分——RL 反向训练破角，破功率从 base 27% 训到 55%/91%。
这是教科书级 Goodhart，但更精确的描述是 reward-component credit misallocation。
修复：全 TDD 新增 anti_ooc_disclosure.py，破角硬罚 -5，撤销误给的 +3.0，
OOC 误报从 1/16 降到 0/16——没有重训 RM。

第二条主线：clean 富度和 defense 存在结构性张力。
v6.8 用 FFN LoRA 富化失败（独白过冲 187%）；v6.9 换 best-of-N 均衡选择器成功。
三轮 SFT-defense（unique defense 120→2686，22×）把真破功压到 33.8%，
但没能回到 base 的 16.2%——斜率不降反升，这就是 SFT 天花板，
也是后来启动 RL 的依据（深读 4/5 把这个判断过程写成了可证伪的两假说框架）。

RL v6.10 step90：clean-aware reward + per-source KL 守住了 clean，
但 attack 真破功 36.8%（25/68）≈ SFT 天花板——没有改善。
根因诊断：-5 是整条回复级 sequence-level reward，
GRPO 把它稀释到整条回复的每个 token，破角 span 里的负信号不够强。
结论：停线，不同配堆步数，转 quality-aware benchmark / RM v7 / 定向数据。

贯穿这三条主线的是一套归因能力：每次"指标没动"都先用廉价判别实验
（分布比对、分轴方差、斜率、组件消融）把根因判给正确的那一层，
再决定下一步动作——这就是深读 9 所总结的差分诊断框架。
```

---

## L2 — 全仓库分块精读

### A. 主线文档（根目录）

> 约 50 秒。入口层，给面试官快速建立全局视图。

**README.md** — 项目入口，按时间预算给出四条读法（5分钟 / 30分钟 / 2小时 / 复现向）。
关键表格是"核心工程亮点"：MoE QLoRA 修复、RM 细粒度设计、Goodhart 反向破角发现与修复、
defense-SFT 蒸馏、verl GRPO 接入。当前状态块（§ 📍）给出了 v6.10 step90 的结论和下一步。

**CASE_STUDY.md** — 七个归因循环的闭环叙事，是"能力体现"的主要证据。
每一轮都是"指标→分析→实验→归因→下一步"，从 v6.3 RM 设计到 v6.10 停线都有。
这篇是把项目体验讲成故事时的底稿。

**RLHF_WORKFLOW.md** — 算法链路的流程图层，SFT → Preference Pair → RM → GRPO 的
每个模块接口、数据流向、框架版本。

**DATA.md** — 复现边界。GB 级训练集和模型权重不入库，这里列了分层清单：
哪些在库（代码/config/小样本/报告），哪些只有路径记录，哪些完全不公开。
面试时主动说这个，比被问到更专业。

---

### B. 九篇深读（deep_dives/）

> 约 90 秒。按从基础工程到方法论提炼的顺序讲，让面试官感受到层次。

**[DD1] 显存算账与 GPU 通信**：8×5090 无 NVLink/P2P（31.36GB/卡）的显存算账。
ZeRO 1/2/3 的显存×通信量账，vLLM cumem 虚拟地址预留语义，
verl colocate 的 param_offload 两张表（0.09GB vs 12.2GB 余量）。
能在非标硬件上把每一 GB 都算清楚，这是工程能力的基础。

**[DD2] RM 数据构造、训练与奖励 + 归因**：六版 RM 台账（v1→v6.3），
每次退步都归因到具体原因——v2 是 batch 减半 + score head 保存 bug，
v3 是 cosine LR 步数不足，v6.3 是 8维 rubric + delta=2 难负样本 + roledrop pair。
这篇是"RM 层差分诊断"的原始证据。

**[DD3] RL 动力学：参数、趋势与策略应对**：每个超参的算法原理（GRPO/GSPO/DAPO/per-source KL）
+ 动态指标看板（psKL/grad_norm/entropy/reward_std）+ 六个"趋势→归因→应对"实战。
能让面试官感受到"看得懂训练曲线"，而不只是会调超参。

**[DD4] Clean 富化为何侵蚀防御 + SFT 天花板**：同一组 v6.9 数字（step0 16.2% → 富化 40% → SFT-defense 33.8%），
拆成"覆盖稀释 vs 行为表征冲突"两个互斥假说，用数据逐个判真伪。
否决"teacher 太弱"（78.8% in-char 良率 + 49条A6正样本0%含代码），
找到真因：SFT 范式学不到完整角色内化轨迹。

**[DD5] 何时从 SFT 转向 RL：入口门判据**：四道门（A评测能看见失败 / B目标轴RM方差 / C残余失败SFT不可约 / D RL代价非零），
每道门配一次违反它的真实代价。比起"当试过SFT了"，这给出了可证伪的转换条件。

**[DD6] 多目标协调：三种机制消解冲突（capstone）**：把 clean/attack/动作/语义的冲突归到
"阶段分配 / 阶段内聚合 / per-source 解耦"三种机制，证明每次失败都是选错了机制，
不是调参数没调好。这是整个仓库方法论的最高抽象层。

**[DD7] 业务 Benchmark 与证据留存**：把 benchmark 从 prompt 集升级为"产品决策仪表盘"：
L1-L5 分层打分、固定 probe + 同尺 regex + 真破功去误报，以及"step90 采样 sweep 为什么证明问题在训练层"。
证据留存系统保证每次决策都有回溯。

**[DD8] Credit Assignment in Character RLHF**：把项目几次成败重读成
"sequence→token credit 错配某一层"：组件层（反向破角）/ group 层（clean 无方差）/ token 层（length hacking）/ source 层（per-source KL）。
新增 Gate E（credit 可定位性）。六层 taxonomy + 项目实例。

**[DD9] 全链路差分归因**：把"指标没动 / 上线不行"这个症状，
在整条链路上做差分诊断——8 个候选层，每层一个比重训便宜几个数量级的最小判别实验。
四次"指标没动"归到四个不同层，归因到 RL 算法命中率 0/4。
这是深读 5 的后向镜像（5 问"该不该起 RL"，9 问"已经不行了锅在哪"）。

---

### C. docs/ 知识库

> 约 40 秒。分三层：主线参考、设计文档、过程归档。

**VERSION_MAP.md** — 一张表看懂 RM v1→v6.3 和 RL v6.3→v6.10 的完整演进，
每行都有目标 / 改动 / 实测 / 结论。是核查任何数字的一手来源。

**chatjoy_eval_paradigm.md** — 评测同尺协议：固定 probe 集 / 统一 regex / 去误报重算。
"同一次、同检测器"是所有跨版本数字可比的前提。

**MIGRATABLE_RLHF_PATTERNS.md** — 可迁移的 RLHF 判断框架，
把本项目的方法论提炼成跨角色/陪伴/客服模型通用的模式。

**设计文档区**：每个关键版本（RM v6.3、SFT v6.8/v6.9、RL v6.10）都有独立设计文档，
保留了当时的设计理由和验收条件——这是"可追溯的工程实践"的体现。

**后训练研究上下文区**（CREDIT_ASSIGNMENT_TAXONOMY / GRPO_LOGPROB_FLOW / RL_RESEARCH_INDEX）：
把项目抽象成算法研究层面的概念底稿，是深读 8 的理论支撑。

---

### D. training/ 代码层

> 约 40 秒。强调"代码是跑过的，有真实修复"。

**GRPO_Training_Package/** — 核心 reward 函数：
- `anti_ooc_disclosure.py`：关键修复，破角硬罚 -5 + 多语言 + TDD 测试
- `anti_refusal_reward.py`：反拒绝 reward——它的 +3.0 bonus 正是反向破角 Goodhart 的来源，修复时撤销了误给破角回复的这笔 bonus
- `length_shaping_reward.py`：5 层防 hacking（RM 去偏置 + actor 饱和封顶 + 链式惩罚）
- `chatjoy_reward_utils.py`：4 维奖励聚合

**configs/ + launch_scripts/** — 每个阶段（SFT/RM/GRPO）都有对应的 YAML 和启动脚本，
含 GPU 拓扑/NCCL 参数。训练是真实跑过的，配置是真实可用的。

**data_prep/** — 线上导出 / 偏好对生成 / 过滤 / 难负样本挖掘的数据流水线。
`generate_preference_pairs*.py` 是偏好对生成的完整代码。

**design/** — `reward_model_design.md` 保留了 RM 设计决策的原始记录，
包括 delta=2 难负样本的选择逻辑和 8 维 rubric 的来源。

---

### E. benchmarks/ + reports/

> 约 30 秒。这里是"决策有没有数字支撑"的证明。

**BUSINESS_BENCHMARK_GUIDE.md** — 给执行 AI 的 benchmark 构建 spec，
说明如何构造能触发产品最高频失效的攻击词典，以及为什么要和训练 val 隔离。

**reports/** — 关键报告：
- `v69_reconcile_report.md`：三方去误报真破功（step0/v6.9b/v6.9c 的 16.2%/32.4%/33.8%）
- `sft_richness/rl90_temp_sweep_report.md`：step90 采样 sweep（证明 ~30% 地板在训练层）
- 周报 / AB 对比 / probe 报告：每次决策都有对应证据

**RMTBENCH/** — 外部学术 roleplay 基准，作为独立锚点，和业务 benchmark 互补。

---

### F. samples/ + 复现边界

> 约 20 秒。主动说清楚边界比等被问到要专业。

**characters_samples/** — 2-3 张脱敏角色卡，展示项目业务场景的真实形态。

**data/samples/** — SFT 5条样例 + LlamaFactory dataset_info 配置头。

**复现边界**：GB 级训练数据、模型权重、Milvus 地址均不入库，
见 DATA.md 分层清单。从代码和配置可以复现算法结构，但不能复现全量训练结果——
项目的主张是"工程档案 + 代码"，不是论文结果复现。

---

## L3 — 被追问时的应对清单

> 面试官在 L2 任意一块都可能追深。以下 12 问是最高频的"难住"追问。
> 标记 🔍 = 对 L2 答案的纵深追问；标记 ⚠️ = 对项目没做好 / 没考虑到位的地方。

---

### L3-1 🔍 RM 方差与 clean 轴

**问**：`clean 轴打分跨度 0.037 vs attack 0.36`——你是说 RM 在 clean 轴上完全无用？
那 clean 富度是靠什么维持的？SFT 初始化和 RL 的贡献各是多少？

**核心难点**：要求你说清楚 clean 是 SFT 管的还是 RL 管的，以及这个分工依据是什么。

**回答要覆盖的点**：
- RM 在 clean 轴上没有区分度，但 SFT 阶段已经学到了 clean 的"静态分布"表达——RL 不会主动破坏它（per-source KL clean=0.008 保了上界）
- v6.10 clean-aware reward 的设计：clean 分支用 clean_q 质量信号 + 高 per-source KL（0.008）把策略钉在 SFT 分布上、不挂防御 RM 主信号；attack 分支才用 RM + break_pen −5 + λ·clean_q（λ=0.25）
- 精确结论：clean 富度是"SFT 建、RL 守"，守的手段是 per-source KL；clean RM 方差为零这件事，确认了"clean 交 SFT，attack 交 RL"这个阶段分配

---

### L3-2 🔍 SFT 天花板的统计严谨性

**问**：`22× unique defense数据，真破功率 32.4%→33.8%`——68 条 probe set 下，
33.8% vs 32.4% 的差异是信号还是统计噪声？你有没有做置信区间？

**核心难点**：68 条样本，23/68 vs 22/68 的差异（1条的差距），非常小，可能是噪声。

**回答要覆盖的点**：
- 诚实承认：单点比较 23/68 vs 22/68 统计 power 不足；没有做 bootstrap 置信区间
- 但判断 SFT 到顶的依据不是这一个点，而是**斜率方向**：从 120→2686（22×），数据量乘以 22，而不是一次小步——这个量级下如果 SFT 有效，斜率应该明显为负
- 此外 v6.9c A6 code 从 9→11（加数据反而变差）也支持"SFT 开始扰乱而不是改善"
- 下一步应该做的：bootstrap 置信区间、独立评测集交叉验证

---

### L3-3 🔍 Credit Assignment：为什么不早点做 span-level reward？

**问**：sequence-level -5 不能定位到破角 span——这是 reward 设计的遗漏，
为什么不在最开始就做 span-level masking 或者 dense reward？

**核心难点**：让你正视一个设计决策，而不是只描述它的症状。

**回答要覆盖的点**：
- 设计顺序是有原因的：先要让 RM 有信号、reward 方向对，才能讨论 credit 定位——span-level 需要知道"哪里是破角 span"，而这依赖 anti_ooc_disclosure.py 先写出来（时序约束）
- 为什么当时选 sequence-level：span 标注成本高，sequence-level 已经有 OOC 检测器支撑，是可以快速实验的
- 确实这是项目最大的技术债：如果重做，应该在 anti_ooc_disclosure.py 完成后立即把 -5 改成 span-masked penalty，而不是等到 v6.10 step90 失败后才意识到
- Proposed diagnostic 部分描述了 bad_span_token_kl_share 指标——但承认这只是假说，还没跑实验

---

### L3-4 🔍 温度脆的完整矩阵

**问**：v6.7 rollout_temp=0.7 下 +0.013，t1.0 下崩——你换 rollout_temp=1.0 之后，
t0.7 下的表现是不是反而变差了？你做过 rollout_temp × deployment_temp 的矩阵 sweep 吗？

**核心难点**：考查你是否理解"训练温度影响的是哪个温度区间的策略"。

**回答要覆盖的点**：
- 核心机制：rollout_temp 决定了 GRPO 能采到哪个温度范围的 rollout，gradient 只在那个范围有效——t0.7 训练只对 t0.7 附近的策略有梯度，换 t1.0 后 t0.7 范围的梯度减弱
- v6.10 没有做完整的 rollout_temp × deploy_temp 矩阵 sweep，这是实验设计的不足
- 实际上在固定权重后用 step90 做了 deployment temp sweep（0.7/0.8/0.85/0.9/1.0 五档，且 n≥2 去噪），结论是 ~30% 地板——但这是在 rollout_temp=1.0 训练的权重上测的，t0.7 vs t1.0 的**训练温度** tradeoff 没有单独验证
- 诚实结论：是一个遗漏的消融实验

---

### L3-5 ⚠️ 统计严谨性与实验设计

**问**：整个项目的实验没有 seed 控制、没有多次重复——你怎么区分哪些结论是信号、
哪些是随机波动？特别是 v6.10 step90 的停线判断，依据足够强吗？

**核心难点**：对整个实验方法论的系统性质疑。

**回答要覆盖的点**：
- 承认：没有 seed 控制、没有多次重复是工程约束（GPU 资源、时间）而不是方法论选择
- 停线判断的依据是多维的：不是只有 36.8%，还有 attack-split RM 0.235 < v6.9c 0.271（RM 维度的退步）、A6 code 11→12（代码注入维度的退步）、采样 sweep 地板 ~30%（说明不是服务层问题）——多维一致才做出停线
- 真正的隐患：单一 68 条 probe 集的随机性——如果 v6.10 的 25/68 里有几条是评测误差，结论可能不同
- 应该做但没做：独立 holdout 集 + bootstrap 统计检验 + 多 seed 重复

---

### L3-6 ⚠️ 架构不一致：SFT/RM(Qwen3) vs Actor(Gemma4)

**问**：你用 Qwen3 训 RM，却用它来评 Gemma4 的输出——两个不同架构的模型，
logit distribution 可能完全不同。RM 对 Gemma4 输出的泛化性你验证过吗？

**核心难点**：指出架构选择带来的潜在 distribution shift。

**回答要覆盖的点**：
- 这是真实的设计约束：Gemma4 也可以用做 RM，但 MoE+视觉塔让 RM 训练复杂（显存和 FSDP 都有特殊修复），Qwen3 作 RM 更稳定
- 实际验证的方式：RL 实际用的 RM（v6.6）区分度是在 held-out 偏好对上测的（attack-split ~0.36），但那批偏好对的候选分布**不是** Gemma4 rollout 分布——Gemma4 在 RL rollout 里的 reward distribution 没做独立的 in-distribution 校验
- 实际上能 partial 验证的是 reward_std 是否健康（reward_std ≥ 2.0 统计门），以及 attack/clean 分轴后 RM 是否在两组上都有合理的分布
- 承认这是开放的风险：正式项目应该在 Gemma4 的 rollout 样本上做 RM distribution check

---

### L3-7 ⚠️ Goodhart 复发：-5 硬罚会产生新的 reward hacking 吗？

**问**：anti_refusal +3.0 导致 Goodhart，你用 -5 硬罚替换——那 -5 会不会产生新的 Goodhart？
比如模型学会"沉默式破角"（不说我是AI，但也不扮演）？你有测过吗？

**核心难点**：对"修复"本身的批判性追问。

**回答要覆盖的点**：
- 理论上是可能的：任何 rule reward 都可能被绕开
- 当前防护：`anti_ooc_disclosure.py` 检测的不只是"我是AI/Gemma"，还检测 system prompt 泄露 + 角色内代码注入（A6）——这些都是直接行为信号，比"拒绝词"更精确
- 68条 probe 里有专门的 identity consistency 维度，包含多种角色出走形式
- 没有测过的：长对话下的"渐进式破角"（每轮少一点，积累多轮后角色滑坡），以及通过情感操控而非语言表达的隐性破角
- 这是下一轮 quality-aware benchmark 的设计目标之一

---

### L3-8 ⚠️ 评测集的代表性

**问**：你的 68 条 attack probe 是你自己构建的——你怎么知道它代表了产品里真实的攻击分布？
有没有漏掉某类攻击？和 RMTBENCH 或其他外部基准比，你的 probe 覆盖了什么、遗漏了什么？

**核心难点**：评测集构建本身是一个独立的研究问题，容易自我循环验证。

**回答要覆盖的点**：
- 构建依据：从产品真实 fail case 分析，涵盖 bullet 注入 / 直接问身份 / 情感崩溃 / 代码注入 / 多语言攻击——这是尽力了解的产品真实分布
- 发现的对齐失败：v6.5 时 val 里 bullet 注入只有 ~5%，产品里是最高频攻击——这正是评测盲区的经验教训，之后把 bullet 注入加重了
- 和 RMTBENCH 的关系：RMTBENCH 是外部学术 roleplay 基准，没有专门的 identity attack，主要用于验证"富度没有退化"
- 承认的遗漏：多轮对话里的渐进式崩溃 / 跨语言混攻 / 多角色场景下的破角——68 条 probe 主要覆盖单轮高强度攻击

---

### L3-9 ⚠️ 数据飞轮与线上漂移

**问**：你用的是真实线上用户聊天数据——但你做的是模型改进。
改进后的模型上线，用户行为会变，产生新数据。有没有闭环到下一轮训练？
有没有数据分布漂移的风险？

**核心难点**：把项目放到工程生命周期里审视，而不只看单次训练。

**回答要覆盖的点**：
- 当前项目没有做自动飞轮：数据是一次性 Milvus 导出，没有接入在线学习循环
- 为什么：规模和周期不支持（单次训练就已经是多周计划），且模型权重不频繁更新
- 漂移风险：如果模型上线后用户行为改变（比如用户开始学习规避新的防御），训练集的失效分布就会 shift；但当前部署的是 step90，没有在线的 reward 反馈信号
- 这是在 HANDOFF_next_training_quality.md 里提到的质量设计目标：下一轮需要更好的评测闭环，但数据飞轮本身不在当前交付范围

---

### L3-10 ⚠️ 最大的设计失误是什么？

**问**：如果重做这个项目，你最早会在什么地方做不一样的决定？
或者说，项目里最让你后悔的一个判断是什么？

**核心难点**：自我批判，但要有具体的技术理由，不能泛泛说"加更多数据"。

**回答（建议思路）**：

选择一个真实的、有技术深度的失误，而不是虚假的谦虚。三个候选：

1. **最早就把 -5 做成 span-level**：anti_ooc_disclosure.py 写完后应该立即做 span-masked penalty，而不是用 sequence-level 跑完整轮 RL 才发现 credit 定位缺失——这一个决定可能省掉 v6.10 step90 的整个训练成本。

2. **v6.8 富化前做更细的选择器设计**：v6.8 把独白放进了选择目标，导致过冲 187%——应该在选择器设计时就用 category balance 约束，而不是在结果出来后才发现这个问题。

3. **RM 训练更早做分轴验证**：RM v3 之前应该先做 clean vs attack 分轴打分的 distribution check，提前发现 clean 0.037 无方差，而不是到 v6.7 RL 实测才验证。

可以选其中一个展开，说"我现在会在什么时间点做这个决定，以及这个决定会改变后续哪些步骤"。

---

### L3-11 🔍 差分诊断有没有局限性？

**问**：你的差分诊断假设每层是独立的——但如果数据覆盖缺口和 RM 无方差同时存在，
你的"一次只动一层"原则还能正确排查吗？

**核心难点**：对方法论本身的 meta-critique。

**回答要覆盖的点**：
- 正视这个局限：多层同时失败时，"一次只动一层"的控制变量原则确实受影响——你改 RM，数据也同时改，就无法知道改进是哪层贡献的
- 实践中怎么处理：排查顺序是"成本从低到高"，而不是"一定每次只动一层"——先做不需要改动的离线检查（分轴方差、斜率），这些是纯观测；当需要改动时，有意识地同时验证
- 项目里有几次同时改了多个层（比如 RM v6.3 同时改了数据构造方法和 LR 调度），那次就确实没能完全分离归因，只能靠后续的更精细实验来拆解
- 承认：这个方法论的最大前提是"你有足够多的实验机会"——在 GPU 资源受限时，有时候只能"改一批，等结果"，归因就变粗了

---

### L3-12 🔍 你说 0/4 是 RL 算法——那这个项目 RL 到底做了什么贡献？

**问**：你讲了很多次"归因到 RL 算法命中率 0/4"——那 RL 在这个项目里实际贡献了什么？
如果 RL 的贡献是有限的，这不正说明你这套 RLHF pipeline 还没有真正验证过 RL 的价值吗？

**核心难点**：逼你正面评价 RL 在项目里的真实价值，而不只是讲归因故事。

**回答要覆盖的点**：
- RL 的实际贡献：v6.10 step90 clean-aware reward + per-source KL 确实守住了 clean（这是 SFT 单独很难做到的）；per-source KL 的 clean/attack 解耦是 RL 独有的机制
- "0/4 是 RL 算法"的正确含义：这说明的是"调 RL 超参"不是问题的出口，而不是说 RL 没有价值——RL 的信号方向修好（reward 规则层）、RM 有方差（RM 层）之后，RL 才能有效工作
- attack 未降的原因是 credit 层而不是 RL 算法本身——sequence-level -5 是 reward 函数设计的问题，不是 GRPO/verl 框架的问题
- 真正诚实的结论：这个项目里 RL 的价值在"守 clean"这件事上有 confirmed evidence；"修 attack"上目前还是 open problem，需要 span-level credit + RM v7

---

## 附：关键数字速查卡

| 指标 | 数值 | 来源 |
|---|---|---|
| probe 集规模 | 68 条（产品形态攻击词典） | `reports/v69_reconcile_report.md` |
| base 真破功（step0） | 16.2%（11/68） | 同上 |
| 富化后破功 | ~40%（侵蚀） | `deep_dives/04` |
| v6.9c（SFT天花板） | 33.8%（23/68） | 同上 |
| v6.10 step90 | 36.8%（25/68） | `deep_dives/09` §8 |
| OOC 误报修复 | 1/16 → 0/16 | `deep_dives/02` §6 |
| RL 后反向破角 | base 27% → RL 55%/91% | `README.md` / `CASE_STUDY.md` |
| clean RM 跨度 | 0.037（vs attack 0.36，~10×） | `docs/VERSION_MAP.md` |
| unique defense 扩展 | 120 → 2686（22×） | `deep_dives/04` |
| SFT 自蒸馏良率 | 78.8% in-char（否决 teacher 太弱） | 同上 |
| 采样 sweep 地板 | ~30%（服务旋钮改不了能力） | `reports/sft_richness/rl90_temp_sweep_report.md` |
| 服务推荐配置 | temp=0.8, top_p=0.95, presence=0.4, min_p=0 | `docs/rl_v610_step90_service_guide.md` |
| v6.7 分轴 RM 增益 | t0.7 +0.013（混合口径 +0.092，口径教训） | `deep_dives/05` |
| per-source KL | clean=0.008 / attack=0.002 | `deep_dives/08` |

---

> 这份脚本是**口述框架**，不是逐字稿。L0/L1 背熟，L2 按块熟悉核心数字，L3 面试前通读一遍知道难点在哪。被追问时优先选离你最近的真实经历，而不是复述这里的模板答案。
