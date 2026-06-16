# DATA.md — 数据与模型分层清单

本仓库是工程档案，**不上传** GB 级训练数据与模型权重。本文件按层级记录**完整数据资产**：每项的用途、规模、原始路径，以及在本仓库中以什么形式存在（完整 / 样本 / 仅记录）。

> 原始根目录：`/home/ps/hanjiaqi/LlamaFactory/chatjoy/`（以下路径相对该目录）
> 生产地址 / 库名 / 内网 IP 已脱敏为 `<...>` 占位符。

---

## 分层总览

| Tier | 内容 | 是否入库 | 形式 |
|---|---|---|---|
| **T0 代码** | 训练/奖励/数据处理脚本、配置、设计文档 | ✅ 全量 | `training/`、`docs/` |
| **T1 小样本** | 几条 SFT / test / 角色卡 / dataset 注册表 | ✅ 样本 | `data/samples/`、`characters_samples/` |
| **T2 训练数据集** | SFT / 多语言 / 偏好对（MB–GB 级） | ❌ 仅记录 | 本文件 §1–§3 |
| **T3 模型权重** | 基座 / SFT / RM / 合并 actor（数十 GB） | ❌ 仅记录 | 本文件 §4 |
| **T4 评测产物** | evaluation CSV / 训练日志 | ❌ 仅记录 | 本文件 §5 |

---

## §1. 线上真实数据（数据源头）

| 项 | 说明 | 规模 |
|---|---|---|
| 数据源 | 正式服 Milvus `HOST=<prod-milvus-host>` / `DB=<prod-db>` / `Collection=vectors` | 线上全量 |
| 导出脚本 | `training/data_prep/export_data.py`（✅ 入库） | — |
| 原始导出 | `chatjoy_dataset/chatjoy_part_*.jsonl` | 最多 10,000,000 条，按 1,000,000 条分片 |

`message_id` 的 `_human_` / `_ai_` 用来恢复 user/assistant 角色；所有下游样本按 `(user_id, character_id)` 会话流**按时间重建**，非随机拼接。

## §2. SFT 数据集（`data/`，T2 — 仅记录）

| 文件 | 大小 | 说明 |
|---|---|---|
| `chatjoy_charactor_data.json` | 3.0 GB | 角色对话全量 |
| `chatjoy_hq_full.json` | 2.9 GB | 高质量全量 |
| `chatjoy_top20_hq_full.json` | 691 MB | TOP20 角色高质量全量 |
| `chatjoy_hq_sample_50k.json` | 208 MB | 高质量 50k 抽样 |
| `chatjoy_top20_multilingual_10k_gemini.json` | 61 MB | TOP20 × 多语言 10k（Gemini 重写） |
| `chatjoy_finetune_final.jsonl` | 58 MB | 最终微调集 |
| `chatjoy_chinese_clean.jsonl` | 58 MB | 中文清洗 |
| `chatjoy_hq_5k_balanced.json` | 48 MB | 5k 平衡集 |
| `chatjoy_alpaca_data.json` | 34 MB | alpaca 格式 |
| `chatjoy_cleaned_for_sft.json` | 29 MB | SFT 清洗后 |
| `chatjoy_filtered.json` | 2.9 MB | 过滤后 |
| **`chatjoy_hq_sample_test5.json`** | 41 KB | **✅ 入库** → `data/samples/sft_sample_5records.json` |

处理脚本（✅ 入库 `training/data_prep/`）：`filter_data.py`、`generate_dataset.py`、`generate_dataset_top20.py`、`prepare_preapi_top20_27langs.py`、`export_data.py`。

## §3. 多语言 / 偏好对数据（T2 — 仅记录）

| 文件 | 大小 | 说明 |
|---|---|---|
| `chatjoy_top20_multilingual_27x2200_preapi.json` | 427 MB | TOP20 × 27 语言 × 2200，预 API 采样 |
| `..._preapi_gemini.progress.jsonl` | 440 MB | Gemini 重写进度（断点续传） |
| `..._preapi_gemini_resumed.json` | 440 MB | Gemini 重写完成 |
| `test_dataset.jsonl` / `test_dataset_chatml.jsonl` | 4.1 MB | 测试集（✅ head5 入库 `data/samples/`） |
| 偏好对 / RM 训练集 | — | 由 `GRPO_Training_Package/generate_preference_pairs*.py` + `framework_bridge/` 生成（脚本 ✅ 入库） |

## §4. 模型权重（T3 — 仅记录，数十 GB/个）

| 模型 | 大小 | 说明 |
|---|---|---|
| Qwen3-30B-A3B-Instruct | — | SFT/RM 基座（MoE） |
| Gemma-4-26B-A4B-it | 48.6 GB | Actor 基座（MoE + 视觉塔，原始 bf16） |
| SFT LoRA adapters | — | `saves/` 下各版本 QLoRA adapter |
| RM checkpoints (v1–v6.3) | — | OpenRLHF 训练产物 |
| `sft2_defense_merged` | 48 GB | defense-SFT 合并模型（teacher 蒸馏） |
| **`rl_v6.4_step160_merged`** | 49 GB | 可服务（= v2/base） |
| **`rl_v6.5_step425_merged`** | 49 GB | 可服务（2026-06-08 merge，供产品 A/B） |

> ⚠️ Gemma4 纯文本 merge 后被 vLLM/verl 加载的两个坑：(a) merge 必须 `force_text_only`；(b) 加载须补回 `vision_config` + 启动带 `--limit-mm-per-prompt '{"image":0,"video":0}'`。详见 `docs/project_chatjoy_overview.md`。

## §5. 评测产物（T4 — 仅记录）

`evaluation_report_jsonl*.csv`（baseline / dpo / v1_gemini / v6_gemini_cot 等，最大 ~5 MB）、`logs/`、`outputs/`、`plots/`、`confusion_matrix.png`、`consensus_moe_report.csv`。评测脚本 ✅ 入库（`training/data_prep/` 的 `analyze_*.py`、`evaluate*.py`）。

## §6. v6.6b–v6.10 富化/防御轮（T2/T3 — 数据仅记录，脚本与报告 ✅ 入库）

> 这一轮的主线是「先用 SFT 把 clean 的文笔丰富度逼近 base teacher，再单独开一道门决定是否进 RL 防御」。脚本、probe 报告、RM 打分小产物均已入库；只有大数据集与合并模型仅记录。

**数据集（T2 — 仅记录，源目录 `/root/rich_cjk_sft_v6{7,8,9}/`）**

| 文件 | 大小 | 说明 |
|---|---|---|
| `rich_cjk_sft_mix_v69_balanced.json` | 29,927 条 | v6.9 均衡 mix（en 12% + defense 10.2% 760×4 oversample + 格式 100%）；✅ 3 条样本入库 `data/samples/sft_v69_balanced_mix_sample3.json`（含 `richness_score` 字段） |
| `rich_cjk_sft_mix_v69_final.json` | 24,518 条 | v6.9 best-of-N 选择器产物（动作+interaction/sensory/emotional 排序，独白不入目标防过冲） |
| `rich_v69_judged_only.jsonl` / `candidates_judged_only.jsonl` | — | judge 已打分的候选池（复用优先，零新增 judge） |
| `bon_candidates_{zh_zhtw,ja_ko}.jsonl` | — | base 重采 N=8–10 的 best-of-N 候选 |

**脚本与报告（✅ 已入库）**

- `training/sft_richness/` — `select_v69_balanced.py`（均衡选择器）、`bon_resample_v69.py`（best-of-N 重采）、`validate_gate_v68.py`（训练前 profile 硬门）、`run_v69_balanced_autoresume.sh`（remount shm + resume_from_checkpoint 自恢复 wrapper）
- `training/sft_richness/probe_analysis/` — step0/step100/base/v68 三轴 held-out probe + 温度甜点扫（t0.8/0.85/0.9/1.0）+ 多样性分析 + RM 分语言打分脚本；**关键产物 `rm_scores.jsonl` / `rm_scores_hot.jsonl`**：clean/attack 分轴打分证「RM 是防御判别器，clean 质量轴几乎是平的」（clean 跨度 0.037 vs attack 跨度 0.36，差 ~10 倍）
- `reports/sft_richness/` — v6.8 三份 probe 报告（heldout / rich_opening / 综合）+ CODEX 任务卡，记录 v6.8 失败归因（独白过冲 187%/对话回退/动作没动 → 弃用，重做 v6.9）

**合并模型（T3 — 仅记录，~49–52 GB/个）**

| 模型 | 大小 | 说明 |
|---|---|---|
| `sft_rich_cjk_v67_repair_merged_full` | 49 GB | v6.7 repair 后 actor（CJK 出厂均长 488 字、语言 55/55、破功 5/34） |
| `sft_rich_cjk_v68_ffn_merged_full` | 52 GB | v6.8 FFN/router LoRA 合并（已弃用，留作 A/B 对照） |
| `rl_v67_step100_merged` | 49 GB | v6.7 RL step100 合并（防御 +0.092 有效，clean 全平 = RM 无梯度） |

> 验证策略（零 merge 筛选）：训完后 vLLM `--enable-lora` 直挂 adapter 筛 ckpt，只对赢家做 merge（merge ~49 GB 仅做一次）。

**评测/周报（✅ 入库 `reports/`）**：`周报-2026-06-12.md`（含 Hermes 迁移 + RL v6.7 判停过程）、`AB对比三方_v2_20260605.md`、`attack_dict_proposal_for_product.md`（防御攻击词典提案）、`chatjoy_rl_vs_base_probe.md`、`response_test.md`。基准对照 `benchmarks/RMTBENCH/`（Claude 4.5/Claude judge 中英文打分）。

---

## 如何复现（高层）

1. **导出 + 清洗**：`export_data.py` 从 Milvus 导出 → `filter_data.py` / `generate_dataset_top20.py` 清洗重建会话流
2. **SFT**：`configs/qwen30b_fsdp_qlora_*.yaml` + `launch_scripts/` 跑 LlamaFactory QLoRA → merge
3. **偏好对 + RM**：`GRPO_Training_Package/generate_preference_pairs_advanced.py` + `rm_v6_finegrained/`（候选→LLM-judge→投票→过滤）→ `train_rm_model.py`
4. **GRPO**：`framework_bridge/prepare_verl_rl_data.py` → `train_actor_model.py` / `run_complete_rlhf.sh`（reward = RM + anti-OOC + anti-refusal + length-shaping）
5. **评测**：`analyze_*.py` + pairwise + RM breakdown

详细超参/路径/启动命令见 [docs/training_runbooks.md](docs/training_runbooks.md)。
