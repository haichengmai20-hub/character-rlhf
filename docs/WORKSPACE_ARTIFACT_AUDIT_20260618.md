# 工作区产物补全审计（2026-06-18）

目的：检查 `/root` 工作区相对 `haichengmai20-hub/chatjoy-character-rlhf` 仓库还有哪些 ChatJoy 关键文件未纳入。原则是：

- 代码、设计、报告、服务指导：尽量全文入库。
- 评测结果：小文件可全文；较大的 jsonl/json 以 sample + 路径索引入库。
- 训练数据、候选池、模型权重：不入库，只记录用途、规模、路径和 sample。

## 1. 仓库状态

审计时仓库：

```text
repo: /root/github-export/chatjoy-character-rlhf
remote: https://github.com/haichengmai20-hub/chatjoy-character-rlhf.git
branch: main
status before audit: clean, origin/main 同步
```

## 2. 本次已补入仓库的关键文件

| 来源 | 入库位置 | 处理方式 | 说明 |
|---|---|---|---|
| `/root/rl_v610_step90_merged/rl_v610_step90_service_guide.md` | `docs/rl_v610_step90_service_guide.md` | 全文 | step90 Gradio/vLLM 服务复现：环境、硬件、参数、tmux、公网隧道 |
| `/root/rich_cjk_sft_v69/rl90_temp_sweep_report.md` | `reports/sft_richness/rl90_temp_sweep_report.md` | 全文 | step90 部署参数 sweep，结论是采样旋钮调不动真破功 |
| `/root/rich_cjk_sft_v69/HANDOFF_next_training_quality.md` | `docs/HANDOFF_next_training_quality.md` | 全文 | base > RL(step90) 后的质量恢复路线 |
| `/root/rich_cjk_sft_v69/v69c_clean200.jsonl` | `data/samples/v69c_clean200_sample5.jsonl` | sample 5 条 | v69c clean probe 样例 |
| `/root/rich_cjk_sft_v69/v69c_attack_probe.jsonl` | `data/samples/v69c_attack_probe_sample5.jsonl` | sample 5 条 | v69c attack probe 样例 |
| `/root/rich_cjk_sft_v69/v69c_attack_rm.jsonl` | `data/samples/v69c_attack_rm_sample5.jsonl` | sample 5 条 | v69c attack RM 打分样例 |
| `/root/rich_cjk_sft_v69/clean_rl90_t08_tp095_pr02_mp005.jsonl` | `data/samples/rl90_clean_temp_sweep_sample5.jsonl` | sample 5 条 | step90 温度/采样 sweep clean 样例 |
| `/root/rich_cjk_sft_v69/attack_rl90_t08_tp095_pr02_mp005.jsonl` | `data/samples/rl90_attack_temp_sweep_sample5.jsonl` | sample 5 条 | step90 温度/采样 sweep attack 样例 |
| `/root/rich_cjk_sft_v69/defense_supplement_v69c.jsonl` | `data/samples/defense_supplement_v69c_sample5.jsonl` | sample 5 条 | v69c defense supplement 样例 |
| benchmark 深读 | `deep_dives/07_business_benchmark_and_evidence_preservation.md` | 新增全文 | benchmark 方法论 + 证据留存策略 |

## 3. 已在仓库中覆盖的本地关键脚本

以下工作区脚本已存在于仓库 `training/sft_richness/` 或其子目录：

- `bon_resample_v69.py`
- `build_mix_v69b.py`
- `build_mix_v69c.py`
- `gen_clean_probe.py`
- `gen_attack_probe.py`
- `gen_defense_supplement.py`
- `gen_defense_supplement_v69c.py`
- `detect_ooc.py`
- `select_v69_balanced.py`
- `run_three_model_probe.py`
- `run_v69_balanced_autoresume.sh`
- `probe_analysis/*` 中的 sweep、diversity、RM scoring 脚本和小型 RM score 产物

因此这些脚本不需要重复上传。

## 4. 仍只记录、不全文上传的大文件

| 路径 | 约大小 | 处理建议 | 理由 |
|---|---:|---|---|
| `/root/rich_cjk_sft_v69/rich_cjk_sft_mix_v69b.json` | 313M | 仅记录 + sample | 训练 mix，大文件 |
| `/root/rich_cjk_sft_v69/rich_cjk_sft_mix_v69_balanced.json` | 313M | 仅记录 + 已有 sample | v69 balanced mix |
| `/root/rich_cjk_sft_v69/rich_cjk_sft_mix_v69_final.json` | 271M | 仅记录 | v69 final mix |
| `/root/rich_cjk_sft_v69/rich_cjk_sft_mix_v69_fixed.json` | 271M | 仅记录 | fixed 中间产物 |
| `/root/rich_cjk_sft_v69/candidates_judged_only.jsonl` | 100M | 仅记录 | judge 候选池 |
| `/root/rich_cjk_sft_v69/rich_v69_judged_only.jsonl` | 92M | 仅记录 | judge 后候选 |
| `/root/rich_cjk_sft_v69/rich_v69_selected.jsonl` | 63M | 仅记录 | 选择器输出 |
| `/root/rich_cjk_sft_v69/rich_cjk_sft_repair_v69c.json` | 41M | 仅记录 + sample | v69c repair 数据 |
| `/root/rich_cjk_sft_v69/defense_supplement_v69c.jsonl` | 21M | sample 入库 | defense supplement |
| `/root/rl_v610_step90_merged` | 49G | 不入库 | merged 模型权重 |
| `/root/sft_rich_cjk_v69c_repair_merged_full` | 49G | 不入库 | merged 模型权重 |

这些文件足以通过 `DATA.md` + sample + 报告复盘项目，不适合直接进 git。

## 5. Benchmark 侧仍建议补的工程项

目前仓库已有：

- `benchmarks/BUSINESS_BENCHMARK_GUIDE.md`
- `benchmarks/RMTBENCH/`
- `training/sft_richness/*` probe/sweep 脚本
- v69/v68/step90 相关报告和 sample

但还没有真正冻结的：

```text
benchmarks/business/
├── README.md
├── data/
├── harness/
└── scorecards/
```

建议下一步把现有 `rich_cjk_sft_v69` + `outputs/v67_step0_vs_step100` 中的 held-out clean/attack/multiturn 材料收敛成 `bizbench-v1`，并给 base_it / step0 / 当前候选生成同尺 scorecard。

## 6. 结论

本次审计后，仓库已经能回顾：

- v69/v69c 为什么从 SFT 富化转向质量恢复；
- v6.10 step90 为什么采样参数无法修复真破功；
- step90 服务如何复现；
- benchmark 应如何从散落 probe 变成冻结业务基准。

剩余不足不是"缺少某个本地文件"，而是还没有把 business benchmark v1 真正实现为一套可运行目录。

