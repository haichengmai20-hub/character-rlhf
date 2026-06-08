# ChatJoy RL 交接状态与当前状态

更新时间: 2026-04-23 14:25:14 CST

## 1) Claude 交接时状态（来自项目总览）

- 文档来源: `/home/ps/.claude/projects/-home-ps/memory/project_chatjoy_overview.md`
- 交接阶段: `2026-04-23 GRPO pilot 启动前调试`
- 既有前提（文档声明）:
  - RM v5 merged 已完成: `saves/qwen3-30b-moe/chatjoy_rm_v5_merged_vllm/`
  - Gemma4 SFT LoRA merged 已完成: `saves/gemma4-26b-a4b/sft2_ddp_lora_3gpu_567_cut3072_merged/`
  - RL parquet 数据已生成: `framework_bridge/outputs/verl_rl_data/{train,val}.parquet`
  - 训练脚本: `GRPO_Training_Package/framework_bridge/verl/train_grpo_lora_actor_verl.sh`
- 交接目标:
  - 先跑 Phase 0 smoke（50 step）验证链路可启动、显存不爆、可进入训练。

## 2) 当前状态（接手后实测快照）

- 时间: `2026-04-23 14:25:14 CST`
- GPU 状态:
  - 0-7 卡空闲（仅 GPU4 有 3MiB 驱动保留）
  - 当前无计算进程占用
- 相关 RL 进程:
  - `main_ppo/train_grpo/vllm api server/TaskRunner/raylet` 无残留业务进程

## 3) 接手后已验证结论（尚未改代码）

### 3.1 第一次冒烟（原脚本）

- 命令形式: `trainer.total_training_steps=2`（缩短版）
- 结果: 未进入训练 step，失败于 RM vLLM 初始化
- 关键报错:
  - `max seq len 262144` 导致 KV cache 不足（可用约 4.74GiB，需求约 6.0GiB）

### 3.2 第二次冒烟（仅加 max_model_len=8192 覆盖）

- 命令形式: 对 actor rollout + reward rollout 显式传 `max_model_len=8192`
- 结果: RM 侧通过，但 actor rollout 失败
- 关键报错:
  - `Gemma4ForConditionalGeneration does not support LoRA yet.`

## 4) 当前阻塞点

1. `max_model_len` 未在脚本中按 `max_prompt_length + max_response_length` 固化，默认回落到模型上限导致 RM 初始化风险。
2. actor rollout 在 vLLM 侧启用了 LoRA，但加载到的模型类是 `Gemma4ForConditionalGeneration`（该类未声明 SupportsLoRA），导致引擎启动失败。

