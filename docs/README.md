# docs 支撑材料索引

普通读者先回根目录 `README.md`。本目录只做事实校准、runbook、专题资料和原始交接归档，不再承担第二套主线。

推荐路径：

1. 5 分钟：`../README.md` -> `VERSION_MAP.md`
2. 15 分钟面试：`../INTERVIEW_SCRIPT.md` -> `project_chatjoy_overview.md` 的阶段台账
3. 45 分钟项目复盘：`../CASE_STUDY.md` -> `../PIPELINE_TIMELINE.md` -> `../RLHF_WORKFLOW.md`
4. 技术深挖：`../deep_dives/00_interview_synthesis.md` -> `../deep_dives/05_when_to_move_from_sft_to_rl.md` -> `../deep_dives/09_full_chain_attribution.md` -> `../deep_dives/08_credit_assignment_in_character_rlhf.md` -> `../deep_dives/04_richness_defense_tension_and_sft_ceiling.md`
5. 复现和核对：`../DATA.md` -> `runbooks/training_runbooks.md` -> handoff/debug logs

## 事实校准

| 文件 | 用途 |
|---|---|
| [VERSION_MAP.md](VERSION_MAP.md) | 版本、指标、结论快速校准。读历史文档前先看这里。 |
| [project_chatjoy_overview.md](project_chatjoy_overview.md) | 当前整理版全量台账：SFT/RM/RL/RM v6/RL 迭代的数据、配置、reward 设计。 |
| [GLOSSARY.md](GLOSSARY.md) | 项目术语表。 |
| [chatjoy_eval_paradigm.md](design/chatjoy_eval_paradigm.md) | 评测范式、probe、去误报口径。 |

## 训练与部署 runbook

| 文件 | 用途 |
|---|---|
| [training_runbooks.md](runbooks/training_runbooks.md) | SFT/RM/RL 启动命令、超参、路径。 |
| [rl_v610_step90_service_guide.md](runbooks/rl_v610_step90_service_guide.md) | RL v6.10 step90 merged 模型服务指导。 |
| [RUN_HANDOFF_INDEX_20260618.md](runbooks/RUN_HANDOFF_INDEX_20260618.md) | 近期 v69b/v69c/v6.10/step90 交接索引。 |
| [WORKSPACE_ARTIFACT_AUDIT_20260618.md](runbooks/WORKSPACE_ARTIFACT_AUDIT_20260618.md) | 工作区产物审计；稳定资产边界优先看 `../DATA.md`。 |

## 设计文档

这些文件记录当时的设计决策。读的时候注意：设计文档不一定等于最终实测结论，最终结论以 `VERSION_MAP.md`、`project_chatjoy_overview.md` 和 deep dives 为准。

| 文件 | 主题 |
|---|---|
| [chatjoy_rm_v6_3_roledrop_design.md](design/chatjoy_rm_v6_3_roledrop_design.md) | RM v6.3：8D rubric、6-source、roledrop pair、训练配置。 |
| [chatjoy_sft_v68_clean_richness_design.md](design/chatjoy_sft_v68_clean_richness_design.md) | SFT v6.8 clean 富化设计；结果是 monologue 过冲。 |
| [chatjoy_sft_v69_clean_richness_design.md](design/chatjoy_sft_v69_clean_richness_design.md) | SFT v6.9 clean 富化和平衡选择器。 |
| [chatjoy_rl_v610_defense_clean_aware_design.md](design/chatjoy_rl_v610_defense_clean_aware_design.md) | RL v6.10 clean-aware reward；step90 clean held，attack 未降。 |
| [chatjoy_business_ux_tech_mapping.md](design/chatjoy_business_ux_tech_mapping.md) | 当前目标演化、8D rubric、hard gate 与技术手段的权威映射。 |

## 原始交接与调试归档

这些文件有证据价值，但不是主线入口。它们可能包含历史估值、过程状态或旧口径；若与整理版冲突，以 `VERSION_MAP.md`、`project_chatjoy_overview.md` 和最新 deep dive 为准。

| 文件 | 性质 |
|---|---|
| [chatjoy_rm_v6_finegrained_handoff.md](archive/chatjoy_rm_v6_finegrained_handoff.md) | RM v6 fine-grained 原始交接；顶部已加整理摘要。 |
| [chatjoy_rm_v6_3_roledrop_handoff.md](archive/chatjoy_rm_v6_3_roledrop_handoff.md) | RM v6.3 roledrop 原始交接；顶部已加整理摘要。 |
| [chatjoy_rm_v6_3_roledrop_plan.md](archive/chatjoy_rm_v6_3_roledrop_plan.md) | v6.3 实施计划与验收 grid。 |
| [chatjoy_rl_handoff_vs_current_status.md](archive/chatjoy_rl_handoff_vs_current_status.md) | RL 交接与当前状态对照。 |
| [chatjoy_rl_takeover_error_fix_log.md](archive/chatjoy_rl_takeover_error_fix_log.md) | RL 接手错误修复日志。 |
| [HANDOFF_next_training_quality.md](archive/HANDOFF_next_training_quality.md) | step90 后下一轮 quality-aware 训练路线。 |

## Deep Dive

| 文件 | 用途 |
|---|---|
| [../deep_dives/00_interview_synthesis.md](../deep_dives/00_interview_synthesis.md) | deep dives 阅读入口：把 01-09 串成面试逻辑链，并给出后续 deep dive 方向。 |

## 研究与方法论

| 文件 | 用途 |
|---|---|
| [CREDIT_ASSIGNMENT_TAXONOMY.md](research/CREDIT_ASSIGNMENT_TAXONOMY.md) | credit assignment 六层定义和诊断口径。 |
| [GRPO_LOGPROB_FLOW.md](research/GRPO_LOGPROB_FLOW.md) | old/new/ref logprob、reward、advantage、KL、actor update 计算顺序。 |
| [RL_RESEARCH_INDEX.md](research/RL_RESEARCH_INDEX.md) | SFT/RM/PPO/GRPO/DPO/RLVR/Agentic RL/Eval/Deploy 导航。 |
| [MIGRATABLE_RLHF_PATTERNS.md](research/MIGRATABLE_RLHF_PATTERNS.md) | 可迁移的 RLHF 判断框架。 |
| [FRAMEWORK_RUNTIME_LESSONS.md](research/FRAMEWORK_RUNTIME_LESSONS.md) | vLLM / verl / LlamaFactory / 训练运行时经验。 |

## 内部 agent 辅助资料

以下文件不作为项目事实证据，也不进入面试阅读路径：

| 文件 | 用途 |
|---|---|
| [MEMORY_INDEX.md](archive/MEMORY_INDEX.md) | 工作区记忆索引镜像，可能含历史状态。 |
| [GPT_CONTEXT_DISTILLED.md](archive/GPT_CONTEXT_DISTILLED.md) | 旧 agent 浓缩上下文，可能滞后。 |

## 元文档

| 文件 | 用途 |
|---|---|
| [project_readability_and_rlhf_workflow_review.md](archive/project_readability_and_rlhf_workflow_review.md) | 旧版可读性审阅；当前整理已吸收主要建议。 |
| [../reports/framework_logs/LOG_EVIDENCE_INDEX.md](../reports/framework_logs/LOG_EVIDENCE_INDEX.md) | vLLM / verl / LlamaFactory 原始日志索引与脱敏摘录入口。 |

## 已清理

- 删除 `project_chatjoy_overview.md.bak_20260707_prereorg`：旧备份，无事实增量。
- 删除 `docs/superpowers/` 下的 full-chain attribution plan/spec：最终成果已进入 `deep_dives/09_full_chain_attribution.md`。
