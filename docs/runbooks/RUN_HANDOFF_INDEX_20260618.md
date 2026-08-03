# Run Handoff Index（2026-06-18）

> 目的：把 `/root/rich_cjk_sft_v69/` 里散落的执行交接、probe task、部署调参 prompt 收敛成一个可读索引。原始 prompt 本身偏执行现场，本文件记录**为什么做、怎么做、结果去哪看**，避免后续回顾项目时只看到结果、找不到运行上下文。

---

## 1. v6.9b retrain：补 defense 后重训

| 项 | 内容 |
|---|---|
| 本地源 | `/root/rich_cjk_sft_v69/HANDOFF_v69b_retrain.md` |
| 背景 | v6.9 clean 已 PASS，但 attack 破功显著高于 step0；缺口集中在 A6 任务注入/代码、多语言 A1/A6 覆盖薄 |
| 数据 | `rich_cjk_sft_mix_v69b.json` = v69 balanced + `defense_supplement.jsonl` 120 条 ×4 oversample |
| 训练配置 | base=`/root/sft_rich_cjk_v67_repair_merged_full`，LoRA r24/alpha48，target=q/k/v/o+FFN/router，cutoff 4096，2 epoch |
| 工程要点 | `DISABLE_VERSION_CHECK=1`，不 pip 改 transformers；`overwrite_output_dir:false` + checkpoint 支持 auto-resume |
| 结果入口 | [reports/v69_reconcile_report.md](../../reports/v69_reconcile_report.md) |

结论：v69b 修了一部分 attack，但破功仍高于 step0，进入 v69c repair。

## 2. v6.9c repair：身份披露 + zh-tw clean 反压

| 项 | 内容 |
|---|---|
| 本地源 | `/root/rich_cjk_sft_v69/HANDOFF_v69c_repair.md` |
| 背景 | v69b 剩余破功 13/14 是第一人称身份披露；A1/A2 覆盖不足；附带 zh-tw mono/dialog 退化 |
| 数据生成 | 用 step0 teacher 跨角色生成 identity-defense supplement；`build_mix_v69c.py` 构建聚焦 repair mix |
| 训练配置 | base=v69b merged，LoRA r24/alpha48，LR 4e-5，1.5ep，save_steps 50 |
| 验收 | clean action/dialog 不退；zh-tw mono 回门内；attack 破功 ≤ step0 |
| 结果入口 | [reports/v69_reconcile_report.md](../../reports/v69_reconcile_report.md)，样本见 [data/samples/](../../data/samples/) |

结论：v69c clean 仍富，但 attack 没压回 step0；SFT-defense 进入天花板，触发 RL v6.10。

## 3. RL v6.10：在富 clean actor 上恢复 attack 防御

| 项 | 内容 |
|---|---|
| 本地源 | `/root/rich_cjk_sft_v69/HANDOFF_v610_RL.md` |
| Actor | `/root/sft_rich_cjk_v69c_repair_merged_full` |
| 设计 | attack prompt 用 RM + break_pen + 0.25 clean_q；clean prompt 用 clean_q + 紧 KL；actor 固定物理 GPU4-7，RM 物理 GPU0-3 |
| canary | step30-50 外部 t0.9 probe，看 clean 是否守住、attack 是否下降 |
| 实际结果 | step90 canary：clean 守住，但 attack 真破功约 36.8%，≈v69c actor；RL 未修好 attack |
| 服务入口 | [docs/runbooks/rl_v610_step90_service_guide.md](rl_v610_step90_service_guide.md) |

结论：v6.10 证明 clean-aware + tight-KL 能守 clean，但当前 RM/RL 信号仍压不动直球身份披露和 A6 代码注入。

## 4. RL step90 deployment tuning：采样参数定档

| 项 | 内容 |
|---|---|
| 本地源 | `/root/rich_cjk_sft_v69/HANDOFF_temp_sweep_rl90.md`、`PROMPT_next_ai_deploy_tune.md` |
| 模型 | `/root/rl_v610_step90_merged` |
| 主扫 | temperature ∈ {0.7,0.8,0.85,0.9,1.0} |
| 次扫 | top_p / presence / min_p |
| 修正 | 旧 `distinct_opening` 用 30 字前缀是盲指标；改 `clean_diversity_v2.py` 的 opening Jaccard |
| 最终参数 | `temperature=0.8, top_p=0.95, presence_penalty=0.4, min_p=0, max_tokens=768-1024` |
| 结果入口 | [reports/sft_richness/rl90_temp_sweep_report.md](../../reports/sft_richness/rl90_temp_sweep_report.md) |

结论：部署采样参数调不动真破功，约 30% 是 step90 的模型地板；若产品不接受，必须回训练/RM/数据层。

## 5. Probe task 文档如何对应到仓库

| 本地源 | 仓库对应 |
|---|---|
| `PROBE_TASK.md` | v69 rich-opening probe 任务；结论已进入 [reports/v69_reconcile_report.md](../../reports/v69_reconcile_report.md) |
| `THREE_MODEL_PROBE_TASK.md` | base / step0 / v69 同尺 clean+attack probe；脚本与产物见 `training/sft_richness/` 和 [reports/v69_reconcile_report.md](../../reports/v69_reconcile_report.md) |
| `POST_PROBE_FLOW.md` | probe 后进入 attack gate / RL v6.10 的流程；已被 [docs/design/chatjoy_rl_v610_defense_clean_aware_design.md](../design/chatjoy_rl_v610_defense_clean_aware_design.md) 与本索引覆盖 |
| `HANDOFF_next_training_quality.md` | 已全文入库：[docs/archive/HANDOFF_next_training_quality.md](../archive/HANDOFF_next_training_quality.md) |

## 6. 仍不入库的东西

- `rich_cjk_sft_mix_v69*.json` 等 300MB 级训练集：只记录路径、规模、样本；
- `clean_rl90_*.jsonl` / `attack_rl90_*.jsonl` 全量 sweep：只入库 sample 与最终报告；
- merged 模型和 tokenizer：只记录服务指南，不入库权重。

这些决策与 [DATA.md](../../DATA.md) 和 [WORKSPACE_ARTIFACT_AUDIT_20260618.md](WORKSPACE_ARTIFACT_AUDIT_20260618.md) 保持一致。
