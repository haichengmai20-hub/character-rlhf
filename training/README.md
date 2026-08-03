# training/ 导览

> 训练全套代码：数据准备 → SFT → 偏好对 → RM → GRPO/RL 的脚本、配置、reward 实现与单测。
> ⚠️ **完整训练需要不入库的数据与权重**（见 [../DATA.md](../DATA.md)）。本目录可独立运行的是 **reward/detector 单元测试** 与数据构造逻辑——它们是纯 Python、无 GPU、无 `/root` 依赖。

---

## 目录结构

| 子目录 | 负责 |
|---|---|
| `GRPO_Training_Package/` | ⭐ 强化学习全套：reward 实现（anti-OOC / anti-refusal / length-shaping / anti-repetition / anti-chain）、RM/Actor 训练入口、偏好对生成、defense-SFT 数据构建、`framework_bridge/`（verl/OpenRLHF 桥接）、`rm_v6_finegrained/`（候选生成 / LLM-judge / 投票 / 过滤）|
| `configs/` | YAML：accelerate / QLoRA / defense-SFT / merge |
| `launch_scripts/` | RM / SFT / GRPO 启动脚本（含 GPU 拓扑 / NCCL 设置）|
| `data_prep/` | 线上导出 / 数据集生成 / 过滤 / 难负样本挖掘 |
| `design/` | reward_model_design.md / README_GRPO / Gemma4 SFT 复盘 |
| `sft_richness/` | v6.9 富化弧的选择器 / best-of-N 重采 / profile 门脚本 |

## 推荐入口 vs 历史归档

- **推荐入口**：`GRPO_Training_Package/anti_chain_reward.py`（在线 reward 主组装）、`train_rm_model.py` / `train_actor_model.py`、`framework_bridge/verl/patches/`（最新 RL 启动）。
- **历史 / 一次性脚本**：`GRPO_Training_Package/outputs/`（旧 hydra 配置快照）、早期 `generate_preference_pairs*.py` 变体——保留供追溯，不是当前推荐路径。

## 只跑 reward / detector 单元测试（无需 GPU/数据）

reward 与破角检测逻辑是纯 Python，自带单测，可直接验证：

```bash
cd training/GRPO_Training_Package
python -m pytest test_*.py -q        # 反破角 / 长度塑形 / 链式 / 任务注入 / 重复 等
```

依赖很轻：`pytest` + `numpy`（`pip install pytest numpy`）。这些测试覆盖了项目里最关键的"指标有效性"修复——例如 `test_anti_ooc_disclosure.py` 锁定了"自报身份破角必须 −5、戏内拒绝/引号复述不误伤"的边界（对应 [深读 2](../deep_dives/02_rm_data_reward_and_attribution.md) 的 Goodhart 修复）。

> 这是本仓库**可复现的核心**之一：不依赖模型权重，就能验证 reward gate 的行为是否正确。完整 SFT/RM/RL 的运行命令见 [../docs/runbooks/training_runbooks.md](../docs/runbooks/training_runbooks.md)，数据/权重边界见 [../DATA.md](../DATA.md)。
