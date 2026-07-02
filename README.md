# ChatJoy — 多语言角色扮演对话模型训练（SFT → RM → RLHF/GRPO）

> 把一个通用 MoE 基座，训练成**多语言、强角色扮演、抗破角**的陪伴对话模型。
> 完整链路：线上真实聊天数据 → SFT → 偏好对（Preference Pair）→ 奖励模型（RM）→ GRPO 强化学习。

**基座**：Qwen3-30B-A3B-Instruct（MoE，SFT/RM）· **Actor**：Gemma-4-26B-A4B-it（MoE+视觉塔，GRPO）
**框架**：LlamaFactory（SFT）+ OpenRLHF（RM）+ verl（Actor GRPO）+ vLLM（rollout/serving）

> ⚠️ 本仓库是**工程档案 + 代码 + 小样本**。GB 级训练数据与模型权重**不入库**，完整清单与路径见 [DATA.md](DATA.md)。生产地址/库名已脱敏。

---

## 🧭 怎么读这个仓库（按时间预算选路径）

| 时间预算 | 读什么 | 适合谁 |
|---|---|---|
| **5 分钟** | 本 README → [VERSION_MAP](docs/VERSION_MAP.md)（一张表看懂版本演进） | 快速判断项目价值与进度 |
| **30 分钟** | README → [CASE_STUDY](CASE_STUDY.md) → [深读 5：何时从 SFT 转 RL](deep_dives/05_when_to_move_from_sft_to_rl.md) | 面试官 / 技术评审 |
| **2 小时** | ＋ [RLHF_WORKFLOW](RLHF_WORKFLOW.md) ＋ [深读 1–6](deep_dives/)（6 是 capstone） | 关心 RLHF 方法论的人 |
| **复现向** | [DATA.md](DATA.md) → [training/README](training/README.md) → [docs/training_runbooks](docs/training_runbooks.md) | 工程复现（先看复现边界）|
| **面试准备** | [INTERVIEW_SCRIPT](INTERVIEW_SCRIPT.md)（L0 电梯 → L1 主线 → L2 全仓库导读 → L3 难住追问） | 本人复习热身 / 面试官提问参考 |

> 想直接拿走可迁移的方法（其他角色/陪伴/客服模型通用）：[docs/MIGRATABLE_RLHF_PATTERNS.md](docs/MIGRATABLE_RLHF_PATTERNS.md)。

---

## 🧭 方法论主线：数据归因驱动的闭环

这个项目想体现的不是"跑过 SFT/RM/RL"，而是一条闭环能力：

```
业务目标 → 数据(真实分布重建) → SFT 初训 → 偏好对+RM → GRPO/RL
   ▲                                                      │
   └──── 调 数据/算法/算力 ◀── 数据归因 ◀── 指标趋势观察 ◀──┘
        (多视角看数据：指标 vs 真实问答 vs 业务)
```

- **数据 / 算法 / 算力 三者互相约束**：算力（5090 无 P2P、MoE 显存）决定能用什么算法（ZeRO-2 + 4-bit、GSPO）；数据归因（RM 退步、指标盲区）决定下一步重训还是改信号。
- **统计视角的数据敏感度**：方差/区分度当一等公民（`reward_std≥2.0`）、分组读根因、**两次识破"指标在骗自己"**（Goodhart）→ 据此把"再 RL"改成"修数据/换信号"。

👉 **完整复盘（七个归因循环）见 [CASE_STUDY.md](CASE_STUDY.md)。** 下方"核心工程亮点"是索引式摘要。

---

## 📚 能力深读（deep_dives/，本仓库主菜）

前三篇是工程/算法三件套（显存、RM、RL 动力学）；中间两篇是从同一组 v6.9 实验里读出的**研究判断**（为什么富化侵蚀防御 / 何时该转 RL）；第六篇是 capstone——把全部串成一张"多个冲突目标如何协调"的框架。

| # | 深读 | 一句话 | 代表性内容 |
|---|---|---|---|
| 1 | [显存算账与 GPU 通信](deep_dives/01_memory_accounting_and_gpu_communication.md) | 在 8×5090（无 NVLink/P2P、31.36GB/卡）上为 26B/30B MoE **把每一 GB 显存和每条通信路径算清楚** | ZeRO 1/2/3 显存×通信量账（all-gather/reduce-scatter 原语级）· NUMA/QPI 拓扑约束 · context length → KV cache → PagedAttention 的服务显存账 · verl colocate 的 param_offload 两张账表（0.09GB vs 12.2GB 余量）· vLLM cumem **虚拟地址预留**语义 + 三档 mem-util 复测（三个值三种死法，每个都能用加法解释）· FSDP merge 51.7GB 不可行性证明 · update_weights 串行换防重排 |
| 2 | [RM 数据构造、训练与奖励 + 归因](deep_dives/02_rm_data_reward_and_attribution.md) | 偏好数据怎么造、RM 怎么训、最终 reward 怎么设计，**每次进退都归因到具体根因** | 三类单因负样本（DOE 思路）+ 门控 + sample_weight · Bradley-Terry 与三个训练敏感点 · 六版台账（94.5→86.4→86.0→95.5）：batch 方差/cosine 周期/缺陷样本逐个归因 · score head 两阶段凸优化恢复 · v6.3 三杠杆（第 6 candidate/roledrop pair/8 维 rubric）+ 统计验收门 · **奖励反向训练破角的发现与修复**（Goodhart 实例，误报 1/16→0/16）|
| 3 | [RL 动力学：参数、趋势与策略应对](deep_dives/03_rl_dynamics_params_and_strategy.md) | 每个 RL 参数的**算法原理依据** + 动态指标看板 + 六个"趋势→归因→应对"实战 | GRPO 组内优势/GSPO 序列级比率（为何配 MoE+标量 RM）/DAPO 非对称 clip/per-source KL/熵正则原理 · bonus 按 σ 标定、SAVE_FREQ 按崩溃间隔定、LR constant vs cosine 的选择逻辑 · psKL 平台期 val 序列与多杠杆响应 · grad_norm 尖峰解读 · v6.5 375 步零移动双重归因 · **base 27% < v2 55% < v1 91%** → 停 RL 转 defense-SFT 的决策链 |
| 4 | [Clean 富化为何侵蚀防御 + SFT 天花板](deep_dives/04_richness_defense_tension_and_sft_ceiling.md) | 一组反直觉数字（step0 16.2% → 富化 40% → 补 defense 仍 33.8%）**拆成两个互斥假说，用数据判真伪** | 数据覆盖稀释 vs 行为表征冲突两假说 · unique defense 120→2686（22×）真破功不降反升=Pareto 碰壁 · 真代码 5→9→11 / 真身份 6→13→12 分类归因 · **否决"teacher 产不出 defense"**（49 条 A6 正样本 0% 含代码、78.8% in-char 良率）· 三处与上游文档不一致的数据自纠 |
| 5 | [何时从 SFT 转向 RL：入口门判据](deep_dives/05_when_to_move_from_sft_to_rl.md) | 把"该不该起 RL"拆成四道可证伪的门，**每道门配一次违反它的真实代价** | 门 A 评测能否看见失败（v6.5 375 步空转）· 门 B RM 目标轴有无梯度（分轴方差 0.037 vs 0.36）· 门 C 残余失败是否 SFT 不可约（加数据看斜率）· 门 D RL 的代价非零（base 27%→RL 55% 把防御训坏 / 温度脆 / canary 门）· 决策树 + RL 有用的充要条件 |
| 6 | [多目标协调：三种机制消解冲突](deep_dives/06_coordinating_conflicting_objectives.md) **(capstone)** | 把 clean/attack/动作/语义等冲突目标归到 RLHF 仅有的**三种协调机制**，证明失败=选错机制 | 机制①阶段分配（富度→SFT/防御→RL）· 机制②阶段内聚合**加权和 vs 字典序门控**（v6.8 选择器放大独白 187% / anti_refusal 把"约束"塞进加权和→反向奖励破功）· 机制③ per-source 解耦（clean KL 0.008 / attack 0.002，守富度⇄改防御）· 冲突越深叠的机制越多（富度⇄防御=三层叠加）· 决策表 + 一页速查 |
| 7 | [业务 Benchmark 与证据留存](deep_dives/07_business_benchmark_and_evidence_preservation.md) | 把 benchmark 从"prompt 集"提升为**产品决策仪表盘 + 同尺协议 + 证据留存系统** | RMTBENCH 外部锚 vs 业务 benchmark 上线门 · L1/L2/L3/L4/L5 分层打分边界 · 为什么 step90 采样 sweep 证明问题在训练层 · 工作区 v69/v6.10 证据如何全文/sample/索引化入库 |
| 8 | [Character RLHF 中的 Credit Assignment](deep_dives/08_credit_assignment_in_character_rlhf.md) | 把项目几次成败**重新读成对 sequence→token credit 错配某一层的回应**：GRPO 给方向，credit 决定方向落在破角 span 还是整条回复/错误组件/错误 source | 反向破角=组件层 credit 分错（base 27%→RL 55%）· clean 提不动=group 层无方差（跨度 0.037）· length hacking=token 层长度泄漏 · per-source KL=项目已建的 source 层 credit 干预 · v6.10 attack 压不动=token/span 层仍缺位 · 新增 Gate E · 迁移到 Agentic trajectory credit（新诊断均标 proposed）|
| 9 | [全链路差分归因](deep_dives/09_full_chain_attribution.md) | 指标没提升/上线不行时，用**差分诊断**把根因判给正确的一层：同一症状、8 个候选病因、每个配一个比重训便宜几个数量级的**最小判别实验** | 四次"指标没动"归到四个不同层（v6.5 评测盲区 / v6.7 RM 无方差 / v6.8 数据放大 / v6.10 credit 缺位，归因到 RL 算法命中率 0/4）· 易混病因对判别关键（RM 无方差 vs credit 无定位 / 数据可补 vs SFT 到顶 / RL 参数 vs credit）· 排查流程图（深读 5 的后向镜像）· 新诊断均标 proposed |

---

## 这是什么

ChatJoy 的目标不是通用问答，而是**多语言角色扮演陪伴对话**。整条链路围绕三件事设计：

1. **保持真实用户分布** — instruction 来自线上真实聊天记录（Milvus 导出），不是人工模板
2. **保持角色卡约束** — 每条样本绑定角色 system prompt，避免只学泛化闲聊
3. **强化多语言一致性 + 角色扮演格式 + 抗破角** — SFT 学基础表达，RM 学会区分角色/语言/长度/格式优劣，RL 用 RM + 反破角信号继续优化在线 actor

覆盖 **TOP20 角色 × 27 语言**。

---

## 全链路

```
线上真实聊天 (Milvus, 按 user/character 会话流重建)
        │  export_data.py → 清洗/过滤/平衡
        ▼
[1] SFT (LlamaFactory, QLoRA)
        │  Qwen3-30B-A3B MoE — 修复 expert 裸 nn.Parameter 4-bit 量化 + FSDP 链
        ▼
[2] Preference Pair 生成
        │  多候选采样 + LLM-judge (Gemini/Qwen 交叉) + 拒绝采样 + 重平衡/sample_weight
        ▼
[3] Reward Model (OpenRLHF)
        │  v1→v6.3 迭代：细粒度 rubric 8 维 + delta=2 难负样本 + roledrop_attack pair
        │  + base 模型作第 6 candidate + reward_std×20 + length-hacking 5 层防护
        ▼
[4] GRPO 强化学习 (verl + vLLM rollout)
        │  Actor=Gemma4-26B-A4B-it；自定义 reward = RM + anti-OOC + anti-refusal + length-shaping
        ▼
   抗破角 / 多语言一致 / 角色扮演陪伴模型
```

---

## 🔧 核心工程亮点（面试关注点）

| 主题 | 难点 | 做法 / 结论 |
|---|---|---|
| **MoE QLoRA 训练** | Qwen3/Gemma4 的 expert 是裸 `nn.Parameter`（非 `nn.Linear`），常规 4-bit 量化/PEFT 注入会崩 | 自定义 experts 4-bit 量化实现 + 4-GPU FSDP 修复链；Gemma4 merge 必须 `force_text_only` 否则视觉塔 q/k/v/o 崩 |
| **RM 细粒度设计 (v6.3)** | 粗粒度 RM 区分不出"角色对但平淡"vs"角色对且细腻" | rubric 8 维打分 + `delta=2` 难负样本 + roledrop_attack 偏好对 + base 模型当第 6 candidate + `reward_std×20` 放大区分度 |
| **Length Reward Hacking** | RL 会发现"写更长 = 更高分"而灌水 | 5 层防护：RM 训练去长度偏置 + actor 层 length-shaping（放宽 + 饱和封顶）+ 重复惩罚 + 链式惩罚 |
| **Reward Hacking 反向破角（关键发现）** | 模型在攻击下"我是 Gemma 4"破角，旧 `atk_break%` 关键词指标却显示只有 ~16% | 根因：`anti_refusal` 的 +3.0 NON_REFUSAL_BONUS 把"自报 AI 身份"判为合规给满分 → **反向训练破角**。修复（全 TDD，不重训 RM）：新增 `anti_ooc_disclosure.py`（破角硬罚 -5、撤销误给 bonus、多语言）+ `length_shaping_reward.py` |
| **Defense-SFT 蒸馏** | RL 信号稀疏修不动破角 | 用 idle 的 base 模型当 teacher，蒸馏"戏内化解"攻击的回复，三门过滤 + 拒绝采样，挂回 SFT |
| **verl GRPO 接入** | Gemma4（MoE+视觉）接 verl/vLLM rollout 多个必死崩溃 | 修 LoRA merge / score head / vLLM patch；纯文本 merge 后须补 `vision_config` + `--limit-mm-per-prompt '{"image":0,"video":0}'` |
| **评测体系** | "破角率"等指标对自报身份完全瞎 | 升级检测器 + 5 维评测 + RM breakdown + pairwise；发现 **val break 11% ≠ 产品 roledrop 55%**（验证集几乎不触发产品式 bullet 注入）|

---

## 📍 当前状态 (2026-06-18) — v6.10 step90 已验证：clean 守住，attack 未修复

完整版本表见 [docs/VERSION_MAP.md](docs/VERSION_MAP.md)；以下是主线：

- **v6.5 停训（战略拐点，2026-06-08）**：硬化 GRPO 375 步零移动；发现 **val 11% ≠ 产品 55%**、且 RL 把 base 原生防御训坏（roledrop 破功 base 27% → RL 55%）→ 不再纯 RL 调参，**富度交 SFT、防御留 RL**。
- **SFT 富化弧（v6.7 → v6.8 失败 → v6.9 成功）**：v6.9 best-of-N 均衡选择器把 clean 动作/对话拉到 ≥step0、逼近 base（clean **PASS**）；但富化**侵蚀了身份防御**（真破功 step0 16.2% → 富化后 ~40%）。
- **三轮 SFT-defense 到天花板（v6.9b/v6.9c）**：unique defense 120 → 2686（22×），真破功只到 32.4%/33.8%，**压不回 step0 的 16.2%** → SFT-defense 是 whack-a-mole（机制见 [深读 4](deep_dives/04_richness_defense_tension_and_sft_ceiling.md)）。
- **RL v6.10 step90（已执行）**：actor=v6.9c，clean-aware reward + per-source KL 确实**守住了 clean**；但外部同尺 attack probe 未改善：真破功 **36.8%（25/68）≈ v6.9c 33.8%**，attack-split RM **0.235 < v6.9c 0.271**，A6 code 11→12。结论：当前 RM/RL/data 对直球身份披露与代码注入仍缺有效梯度，**不建议同配继续堆步数**。
- **部署采样定档（step90 服务口径）**：温度/top_p/presence/min_p sweep 证明采样旋钮只能微调表达，不能把 attack 降下去；服务推荐先用 `temperature=0.8, top_p=0.95, presence_penalty=0.4, min_p=0`。完整服务配置见 [docs/rl_v610_step90_service_guide.md](docs/rl_v610_step90_service_guide.md)，采样 sweep 见 [reports/sft_richness/rl90_temp_sweep_report.md](reports/sft_richness/rl90_temp_sweep_report.md)。
- **下一步**：若产品不能接受约 30% 的真破功地板，下一轮应转 **quality-aware benchmark / RM v7 / 定向 A6+身份披露数据**，而不是靠部署采样或继续同配 RL。下一步训练质量设计见 [docs/HANDOFF_next_training_quality.md](docs/HANDOFF_next_training_quality.md)。

> 早期（至 v6.5）完整时间线见 [docs/project_chatjoy_overview.md](docs/project_chatjoy_overview.md)；v6.6b 之后见 [VERSION_MAP](docs/VERSION_MAP.md) 与深读 4/5。

---

## 仓库结构

```
chatjoy-character-rlhf/
├── README.md                    本文件（项目入口 + 阅读路径 + 当前状态）
├── LICENSE                      版权声明（All Rights Reserved，作品集参考用途）
├── CASE_STUDY.md                ⭐ 闭环叙事：7 个归因循环（业务→数据→训练→归因→再迭代，收在 v6.5 战略转向）
├── DATA.md                      ⭐ 完整数据/模型分层清单（哪些入库、哪些只记录）
├── INTERVIEW_SCRIPT.md          面试口述脚本（L0/L1/L2 全仓库导读 + L3 难住追问 × 12）
│
├── deep_dives/                  ⭐⭐ 能力深读（本仓库主菜）
│   ├── 01_memory_accounting_and_gpu_communication.md   显存算账 × ZeRO × 通信原语 × vLLM/cumem
│   ├── 02_rm_data_reward_and_attribution.md            RM 数据构造/训练/奖励设计 × 数据归因
│   ├── 03_rl_dynamics_params_and_strategy.md           RL 参数原理 × 动态趋势观察 × 策略应对
│   ├── 04_richness_defense_tension_and_sft_ceiling.md  富化为何侵蚀防御 × 双假说判真伪 × SFT 天花板
│   ├── 05_when_to_move_from_sft_to_rl.md                何时转 RL × 四道入口门 × 过早起 RL 的代价
│   ├── 06_coordinating_conflicting_objectives.md       多目标协调 × 三种机制 × 选错机制=失败根源
│   ├── 07_business_benchmark_and_evidence_preservation.md  业务 benchmark × 同尺协议 × 证据留存
│   ├── 08_credit_assignment_in_character_rlhf.md        sequence→token credit 错配 × 六层归因 × 对项目成败的实际影响
│   └── 09_full_chain_attribution.md                     全链路差分归因 × 8 候选病因 × 最小判别实验 × 排查流程图
│
├── docs/                        工程知识库（项目全程原始记录）
│   ├── README.md                          📑 docs 分层索引（主线 / 设计 / 交接归档 / 运行手册）
│   ├── VERSION_MAP.md                     ⭐ 版本地图：v1→v6.10 一张表
│   ├── MIGRATABLE_RLHF_PATTERNS.md        ⭐ 可迁移的 RLHF 判断框架（通用方法论）
│   ├── FRAMEWORK_RUNTIME_LESSONS.md       vLLM / verl / LlamaFactory 运行知识（从日志提炼）
│   ├── chatjoy_eval_paradigm.md           评测范式：固定 probe + 统一 regex + 真破功去误报
│   ├── project_chatjoy_overview.md       全链路总纲 + RM/GRPO 全记录 + 当前状态
│   ├── training_runbooks.md              各阶段可直接运行的启动命令/超参/路径
│   ├── chatjoy_business_ux_tech_mapping.md  8 维业务体验 → 技术手段映射
│   ├── RUN_HANDOFF_INDEX_20260618.md     v69b/v69c/v6.10/step90 调参执行交接索引
│   ├── chatjoy_rm_v6_3_roledrop_design.md   RM v6.3 设计（rubric 8维 / delta=2 / roledrop pair）
│   ├── chatjoy_rm_v6_3_roledrop_plan.md     v6.3 实施计划 + 70-case 验收 grid
│   ├── chatjoy_rm_v6_3_roledrop_handoff.md  v6.3→v6.4→v6.5 交接（最新）
│   ├── chatjoy_rm_v6_finegrained_handoff.md RM 细粒度交接
│   └── chatjoy_rl_takeover_error_fix_log.md RL 接手错误修复全记录
│
├── training/
│   ├── README.md               训练代码导览 + 如何只跑 reward/detector 单测
│   ├── GRPO_Training_Package/   ⭐ 强化学习全套代码
│   │   ├── anti_ooc_disclosure.py        反破角 reward（关键修复）+ 测试
│   │   ├── anti_refusal_reward.py        反拒绝 reward
│   │   ├── length_shaping_reward.py      长度塑形（防 hacking）+ 测试
│   │   ├── anti_repetition_reward.py / anti_chain_reward.py  重复/链式惩罚
│   │   ├── chatjoy_reward_utils.py       4 维奖励聚合
│   │   ├── train_rm_model.py / train_actor_model.py  RM / Actor 训练入口
│   │   ├── generate_preference_pairs*.py 偏好对生成
│   │   ├── build_defense_sft_dataset.py  defense-SFT 数据构建 + 测试
│   │   ├── framework_bridge/             verl / OpenRLHF 数据桥接
│   │   └── rm_v6_finegrained/            v6 细粒度：候选生成 / LLM-judge / 投票合并 / 过滤
│   ├── configs/                YAML（accelerate / QLoRA / defense-SFT / merge）
│   ├── launch_scripts/         RM/SFT/GRPO 启动脚本（含 GPU 拓扑/NCCL）
│   ├── data_prep/              线上导出 / 数据集生成 / 过滤 / 难负样本挖掘
│   └── design/                 reward_model_design.md / README_GRPO / Gemma4 SFT 复盘
│
├── benchmarks/                  评测基准
│   ├── BUSINESS_BENCHMARK_GUIDE.md  ⭐ 业务 benchmark 构造指导（给执行 AI 的 spec）
│   └── RMTBENCH/                外部学术 roleplay 基准（参考锚，en/zh 多轮）
├── reports/                    评测报告（v69 三方真破功 / AB 对比 / 周报 / probe 报告）
│   └── framework_logs/          vLLM/verl/LlamaFactory 日志证据索引 + 脱敏摘录
├── characters_samples/         角色卡示例（产品 IP，仅放 2-3 张做样例）
└── data/samples/               小样本（SFT 5 条 / test head / LlamaFactory dataset_info）
```

---

## 技术栈

- **训练**：LlamaFactory (SFT/QLoRA) · OpenRLHF (RM) · verl 0.8 (GRPO) · DeepSpeed ZeRO-3 · FSDP · PEFT/LoRA
- **推理/服务**：vLLM 0.19（rollout + RM 分类 + demo serving）
- **数据/评测**：Milvus（线上导出）· Gemini / Qwen3 作 LLM-judge（交叉投票）· pairwise + RM breakdown
- **模型**：Qwen3-30B-A3B-Instruct（MoE）· Gemma-4-26B-A4B-it（MoE + 视觉塔）

---

## 注意

- 本仓库**不含**真实用户数据、GB 级训练集、模型权重 —— 见 [DATA.md](DATA.md) 的分层说明与复现路径。
- 角色卡、产品攻击样本等业务 IP 仅放少量脱敏样例。
- 生产 Milvus 地址 / DB 名 / 内网 IP 已统一替换为 `<...>` 占位符。
