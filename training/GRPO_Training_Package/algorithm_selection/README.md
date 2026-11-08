# 算法选型对照实验（DPO / GRPO / GSPO，PPO/OPD 条件性加入）

> **状态：[预注册，未执行]**。本目录只包含实验的设计、数据转换与诊断工具；
> 没有任何一轮训练、评分或对照已经运行。判定规则在运行前预注册，
> 见 [`interview_notes/RL/03_算法选型对照实验_DPO_GRPO_GSPO.md`](../../../interview_notes/RL/03_算法选型对照实验_DPO_GRPO_GSPO.md) §2–§5（结果台账留空待填）。
> 任何本目录产出的数字在填入台账前不得写成实跑结论。

## 实验矩阵

| 臂 | 实现方式 | 关键参数 |
| --- | --- | --- |
| GRPO（token 级裁剪） | 复制 `framework_bridge/verl/train_grpo_lora_actor_verl.sh`，把 L219 的 `policy_loss.loss_mode=gspo` 改为 `loss_mode=vanilla`（本地 verl `core_algos.py` L1278 注册名） | LR 1e-6 / 3e-6；clip 沿用默认 0.2/0.28（token 级口径） |
| GSPO（序列级裁剪） | 现有 launcher 原样（`loss_mode=gspo` @L1538；launcher L219） | LR 1e-6 / 3e-6；clip 两档：默认 `CLIP_RATIO_LOW/HIGH=0.2/0.28`（launcher L187–188）与论文式 `3e-4/4e-4`（env 覆盖） |
| DPO | 本目录 `convert_pairs_to_llamafactory_dpo.py` + `configs/dpo_gemma4_lora_beta005.yaml` / `..._beta01.yaml`（LLaMA-Factory `stage: dpo`） | LR 1e-6 固定；β 0.05 / 0.1 两档 |
| 继续 SFT 对照 | LLaMA-Factory `stage: sft`，用合格 chosen（复用既有 SFT 配置改数据） | 与 DPO 同预算 |
| 训推差异诊断（先行） | 本目录 `diagnose_rollout_train_logprob.py` | 32 Prompt × G=6 = 192 Responses；冻结权重；temperature=1、关 top-p/k |

统一边界（所有臂共享）：同一 Targeted-SFT 起点、同一冻结 Reference、同一 RM-B + 固定结构规则、同 tokenizer/模板/输入视图、同 LoRA target/rank、同 response mask 与 EOS；Family 隔离 Train/Dev/Test。

## 运行方式

```bash
# 0) 训推差异诊断：先 dump 两份 per-token logprob JSONL
#    （rollout 侧开启 verl 的 rollout logprob 保存；训练侧对同一批 token 重算导出），
#    schema 与聚合口径见 diagnose 脚本 docstring。
python diagnose_rollout_train_logprob.py \
  --rollout rollout_logprobs.jsonl --train train_logprobs.jsonl \
  --out diag_report.json

# 1) DPO 数据转换（RM pairdata -> LLaMA-Factory ranking 格式）
python convert_pairs_to_llamafactory_dpo.py \
  --input  ../rm_v6_finegrained/output_pairs.jsonl \
  --output data/character_rlhf_dpo.jsonl \
  --emit-dataset-info data/dataset_info.json \
  --shuffle --seed 11 --limit 0        # limit 0 = 开放完整合格偏好池

# 2) DPO 训练（LLaMA-Factory；先替换 yaml 内的占位路径并 dry-run 校验）
llamafactory-cli train configs/dpo_gemma4_lora_beta005.yaml

# 3) GRPO / GSPO 臂：见上表 launcher 说明；预算与筛选顺序按笔记 §2D–§2E
```

## 单测

```bash
cd algorithm_selection
python -m pytest . -q
```

`diagnose` 与 `convert` 均为纯 stdlib 实现，可在无 GPU 环境跑通单测；真实数据上的运行结论按 §5 台账登记。

## 已知边界

- LLaMA-Factory 的 `pref_beta`/`pref_loss`/ranking 数据字段随版本演进，yaml 以本机安装版本 dry-run 为准；`--emit-dataset-info` 生成的是起点模板。
- GRPO 与 GSPO **不得**使用同一 clip 数值制造「公平」：token 级与序列级裁剪的数量级不同（论文口径 0.2/0.27 vs 3e-4/4e-4），且两者 clip fraction 口径不可横比。
- δ 诊断只能确定差异是否存在及其分层分布，不能单独证明任何算法更优。
