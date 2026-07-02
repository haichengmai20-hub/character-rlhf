# docs/ 分层索引

> `docs/` 同时存放**主线文档**、**设计文档**和**过程记录（交接/计划/调试日志）**。本索引按"读者该不该先读"分层，避免在大量内部记录里迷路。
> 外部读者建议从 ⭐ 主线开始；交接/计划/日志是历史归档，需要追溯细节时再看。

---

## ⭐ 主线（先读这些）

| 文件 | 内容 |
|---|---|
| [VERSION_MAP.md](VERSION_MAP.md) | 版本地图：RM v1→v5、RL v6.3→v6.10 一张表（目标/改动/实测/结论）|
| [MIGRATABLE_RLHF_PATTERNS.md](MIGRATABLE_RLHF_PATTERNS.md) | 可迁移的 RLHF 判断框架（通用方法论，跨项目）|
| [FRAMEWORK_RUNTIME_LESSONS.md](FRAMEWORK_RUNTIME_LESSONS.md) | vLLM / verl / LlamaFactory 运行知识：KV cache、PagedAttention、`/dev/shm`、SFT cutoff/merge |
| [chatjoy_eval_paradigm.md](chatjoy_eval_paradigm.md) | 评测范式：固定 probe 集 + 统一 regex 重算 + 真破功去误报（深读 4/5 的口径来源）|
| [project_chatjoy_overview.md](project_chatjoy_overview.md) | 全链路总纲 + RM/GRPO 全记录（早期至 v6.5 战略转向）|
| [training_runbooks.md](training_runbooks.md) | 各阶段可直接运行的启动命令 / 超参 / 路径（复现入口）|
| [rl_v610_step90_service_guide.md](rl_v610_step90_service_guide.md) | RL v6.10 step90 merged 模型服务指导：环境、硬件、vLLM/Gradio 参数、公网隧道 |
| [RUN_HANDOFF_INDEX_20260618.md](RUN_HANDOFF_INDEX_20260618.md) | 近期 v69b/v69c/v6.10/step90 部署调参执行交接索引 |

> 仓库根目录的 [README](../README.md) / [CASE_STUDY](../CASE_STUDY.md) / [RLHF_WORKFLOW](../RLHF_WORKFLOW.md) / [deep_dives](../deep_dives/) 是更高层的主线，本目录是其支撑材料。

## 🎯 设计文档（决策依据，按版本）

| 文件 | 版本 / 主题 |
|---|---|
| [chatjoy_rm_v6_3_roledrop_design.md](chatjoy_rm_v6_3_roledrop_design.md) | RM v6.3：8 维 rubric / delta=2 / roledrop pair / base 第 6 candidate |
| [chatjoy_sft_v68_clean_richness_design.md](chatjoy_sft_v68_clean_richness_design.md) | SFT v6.8 clean 富化（FFN LoRA + judge 选样，**失败案例**）|
| [chatjoy_sft_v69_clean_richness_design.md](chatjoy_sft_v69_clean_richness_design.md) | SFT v6.9 clean 富化（best-of-N 均衡选择器，成功）|
| [chatjoy_rl_v610_defense_clean_aware_design.md](chatjoy_rl_v610_defense_clean_aware_design.md) | RL v6.10 防御 + clean-aware reward（已跑 step90：clean 守住，attack 未降；保留设计与停线结论）|
| [chatjoy_business_ux_tech_mapping.md](chatjoy_business_ux_tech_mapping.md) | 8 维业务体验 → 技术手段映射（参考）|

## 📦 过程记录归档（交接 / 计划 / 调试日志 —— 需要追溯细节时再看）

| 文件 | 性质 |
|---|---|
| [chatjoy_rm_v6_3_roledrop_plan.md](chatjoy_rm_v6_3_roledrop_plan.md) | v6.3 实施计划 + 70-case 验收 grid |
| [chatjoy_rm_v6_3_roledrop_handoff.md](chatjoy_rm_v6_3_roledrop_handoff.md) | v6.3→v6.4→v6.5 交接 |
| [chatjoy_rm_v6_finegrained_handoff.md](chatjoy_rm_v6_finegrained_handoff.md) | RM 细粒度交接 |
| [chatjoy_rl_handoff_vs_current_status.md](chatjoy_rl_handoff_vs_current_status.md) | RL 交接 vs 当前状态对照 |
| [chatjoy_rl_takeover_error_fix_log.md](chatjoy_rl_takeover_error_fix_log.md) | RL 接手错误修复全记录（调试日志）|
| [MEMORY_INDEX.md](MEMORY_INDEX.md) | 工作区记忆索引镜像（含历史 banner，数字以仓库主线/深读为准）|
| [HANDOFF_next_training_quality.md](HANDOFF_next_training_quality.md) | step90 质量审查后，下一轮 quality-aware 训练路线 |

## 🔬 后训练研究上下文（面试复盘 / 研究资产）

> 这几篇是把项目抽象成"后训练算法研究"的上下文层，也是 [深读 8](../deep_dives/08_credit_assignment_in_character_rlhf.md) 的概念底稿。它们不追加实验，只固定算法链路、变量层级与 credit assignment 框架。
> [深读 9 — 全链路差分归因](../deep_dives/09_full_chain_attribution.md) 是深读 8 的**后向配套**：把"指标没提升 / 上线不行"逐层归因到评测 / 数据 / SFT / RM / reward 规则 / RL 参数 / credit / 采样，每层配一个最小判别实验。

| 文件 | 内容 |
|---|---|
| [CREDIT_ASSIGNMENT_TAXONOMY.md](CREDIT_ASSIGNMENT_TAXONOMY.md) | Credit assignment 六层（token/span/sequence/group/source/batch）的定义、失败模式、诊断口径 —— [深读 8](../deep_dives/08_credit_assignment_in_character_rlhf.md) 的概念底稿 |
| [GRPO_LOGPROB_FLOW.md](GRPO_LOGPROB_FLOW.md) | old/new/ref logprob、reward、advantage、KL、actor update 的计算顺序（不依赖具体框架）|
| [RL_RESEARCH_INDEX.md](RL_RESEARCH_INDEX.md) | SFT/RM/PPO/GRPO/DPO/RLVR/Agentic RL/Eval/Deploy 的导航索引 |
| [GPT_CONTEXT_DISTILLED.md](GPT_CONTEXT_DISTILLED.md) | 项目后训练研究上下文的提炼版（定位、算法解释、核心发现）|

## 🔎 元 / 审阅

| 文件 | 内容 |
|---|---|
| [project_readability_and_rlhf_workflow_review.md](project_readability_and_rlhf_workflow_review.md) | 仓库治理与可读性审阅建议（本目录的整理依据）|
| [WORKSPACE_ARTIFACT_AUDIT_20260618.md](WORKSPACE_ARTIFACT_AUDIT_20260618.md) | 2026-06-18 工作区产物补全审计：哪些全文入库、哪些 sample、哪些仅记录 |
| [../reports/framework_logs/LOG_EVIDENCE_INDEX.md](../reports/framework_logs/LOG_EVIDENCE_INDEX.md) | vLLM / verl / LlamaFactory 原始日志索引与脱敏摘录入口 |

---

> 说明：本项目选择**用索引分层**而非把过程记录物理搬进 `archive/` 子目录——这样不破坏既有相对链接和 git 历史，同时达到"主线清晰、归档可追溯"的目的。
> ⚠️ 过程记录里的部分早期数字可能是 raw 或规划期估值；当与主线文档（VERSION_MAP / deep_dives / RLHF_WORKFLOW）冲突时，**以主线的去误报实测值为准**（同尺原则，见深读 4/5 附录）。
