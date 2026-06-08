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

---

## 如何复现（高层）

1. **导出 + 清洗**：`export_data.py` 从 Milvus 导出 → `filter_data.py` / `generate_dataset_top20.py` 清洗重建会话流
2. **SFT**：`configs/qwen30b_fsdp_qlora_*.yaml` + `launch_scripts/` 跑 LlamaFactory QLoRA → merge
3. **偏好对 + RM**：`GRPO_Training_Package/generate_preference_pairs_advanced.py` + `rm_v6_finegrained/`（候选→LLM-judge→投票→过滤）→ `train_rm_model.py`
4. **GRPO**：`framework_bridge/prepare_verl_rl_data.py` → `train_actor_model.py` / `run_complete_rlhf.sh`（reward = RM + anti-OOC + anti-refusal + length-shaping）
5. **评测**：`analyze_*.py` + pairwise + RM breakdown

详细超参/路径/启动命令见 [docs/training_runbooks.md](docs/training_runbooks.md)。
