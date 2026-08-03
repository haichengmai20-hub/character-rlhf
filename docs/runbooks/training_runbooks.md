---
name: ChatJoy 训练流程 Runbook
description: ChatJoy 项目各训练阶段（SFT LoRA、LoRA merge、RM、GRPO）的环境/数据/权重/启动命令完整记录
type: project
originSessionId: faabf23f-85fa-49c2-a93b-34bb65883354
---
# ChatJoy 训练流程 Runbook

> 最后更新: 2026-04-16  
> 硬件: 8×RTX 5090 PCIe, 32GB/卡. GPU0-3: NUMA0, GPU4-7: NUMA1. 无 NVLink.

---

## 目录

1. [Stage 1 — SFT LoRA 微调 (LlamaFactory + FSDP)](#stage-1--sft-lora-微调)
2. [Stage 2 — LoRA Merge 合并](#stage-2--lora-merge-合并)
3. [Stage 3 — RM 训练 (OpenRLHF + ZeRO-2 + 4-bit)](#stage-3--rm-训练)
4. [Stage 4 — GRPO Actor 训练 (verl + Gemma4)](#stage-4--grpo-actor-训练)
5. [关键路径速查](#关键路径速查)
6. [常见错误与修复记录](#常见错误与修复记录)

---

## Stage 1 — SFT LoRA 微调

### 环境

| 项 | 值 |
|---|---|
| conda env | `llamafactory` (transformers 4.57.1) |
| 框架 | LlamaFactory + Accelerate FSDP |
| Python | `/home/ps/miniconda3/envs/llamafactory/bin/python` |
| GPU | 全部 8 卡 (0-7), FSDP FULL_SHARD |

### 关键路径

| 类型 | 路径 |
|---|---|
| 基础模型 | `/home/ps/.cache/modelscope/hub/models/Qwen/Qwen3-30B-A3B-Instruct-2507` |
| 训练数据 (SFT2) | `/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/runs/clean32_for_sft2.jsonl` (19,508 条) |
| 训练数据 (SFT1) | `/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/runs/clean32_for_sft1.jsonl` (13,128 条) |
| 训练配置 YAML | `/home/ps/hanjiaqi/LlamaFactory/chatjoy/qwen30b_fsdp_qlora_clean32_sft2.yaml` |
| Accelerate 配置 | `/home/ps/hanjiaqi/LlamaFactory/chatjoy/accelerate_fsdp_8gpu.yaml` |
| 输出适配器 | `/home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/qwen3-30b-moe/clean32_for_sft2_fsdp_qlora/` |
| 启动脚本 | `/home/ps/hanjiaqi/LlamaFactory/chatjoy/run_qwen30b_fsdp_qlora_clean32_sft2_nohup.sh` |

### 训练超参数 (SFT2)

```yaml
model_name_or_path: /home/ps/.cache/modelscope/hub/models/Qwen/Qwen3-30B-A3B-Instruct-2507
quantization_bit: 4           # 4-bit QLoRA 训练
finetuning_type: lora
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.05
lora_target: q_proj,v_proj,k_proj,o_proj
stage: sft
dataset: clean32_for_sft2_llamafactory
template: qwen
cutoff_len: 1536
per_device_train_batch_size: 1
gradient_accumulation_steps: 2   # effective batch = 1×2×8GPU = 16
learning_rate: 1.0e-4
num_train_epochs: 2.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true / pure_bf16: true
flash_attn: fa2
gradient_checkpointing: true
```

Accelerate FSDP 配置 (`accelerate_fsdp_8gpu.yaml`):
- `fsdp_sharding_strategy: FULL_SHARD` (ZeRO-3 等价)
- `num_processes: 8`
- `mixed_precision: bf16`

### 启动命令

```bash
# 推荐：使用封装好的启动脚本（自动创建 tmux session + 日志）
cd /home/ps/hanjiaqi/LlamaFactory
bash chatjoy/run_qwen30b_fsdp_qlora_clean32_sft2_nohup.sh

# 查看训练进度
tmux attach -t $(cat chatjoy/logs/latest.session)
tail -f chatjoy/logs/latest.log
```

**或直接手动启动（指定 GPU）：**
```bash
tmux new-session -d -s sft_train "bash -lc '\
    source /home/ps/miniconda3/etc/profile.d/conda.sh && \
    conda activate llamafactory && \
    cd /home/ps/hanjiaqi/LlamaFactory && \
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 && \
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
    export DISABLE_VERSION_CHECK=1 && \
    accelerate launch \
        --config_file chatjoy/accelerate_fsdp_8gpu.yaml \
        src/train.py \
        chatjoy/qwen30b_fsdp_qlora_clean32_sft2.yaml \
        2>&1 | tee chatjoy/saves/qwen3-30b-moe/clean32_for_sft2_fsdp_qlora/train_manual.log'"
tmux attach -t sft_train
```

### 预期结果

- 输出目录: `.../clean32_for_sft2_fsdp_qlora/`
- 包含: `adapter_config.json`, `adapter_model.safetensors`, `checkpoint-*/`
- SFT2 总步数: ~2,440 steps (19,508条 × 2 epoch ÷ 16 effective batch)
- 单步耗时: ~10-15s (8卡 FSDP), 总时长约 8-10小时

---

## Stage 2 — LoRA Merge 合并

### 环境

| 项 | 值 |
|---|---|
| conda env | `llamafactory` |
| 工具 | LlamaFactory `llamafactory-cli export` |
| GPU | 仅需 1 卡 (auto), 约 60GB+ CPU RAM |

### 关键路径

| 类型 | 路径 |
|---|---|
| 基础模型 | `/home/ps/.cache/modelscope/hub/models/Qwen/Qwen3-30B-A3B-Instruct-2507` |
| LoRA 适配器 (SFT2) | `/home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/qwen3-30b-moe/clean32_for_sft2_fsdp_qlora/` |
| 合并后输出 (SFT2) | `/home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/qwen3-30b-moe/clean32_for_sft2_fsdp_qlora_merged/` (~57GB BF16) |
| 合并配置 YAML | 见下方（SFT1 合并配置路径 `/home/ps/hanjiaqi/LlamaFactory/chatjoy/qwen30b_merge_clean32.yaml`） |

**SFT2 合并配置（在 SFT1 配置基础上修改适配器路径）：**
```yaml
# /home/ps/hanjiaqi/LlamaFactory/chatjoy/qwen30b_merge_clean32_sft2.yaml
# 注意: 合并时 NOT 使用 quantization_bit
model_name_or_path: /home/ps/.cache/modelscope/hub/models/Qwen/Qwen3-30B-A3B-Instruct-2507
adapter_name_or_path: /home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/qwen3-30b-moe/clean32_for_sft2_fsdp_qlora
template: qwen
trust_remote_code: true
export_dir: /home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/qwen3-30b-moe/clean32_for_sft2_fsdp_qlora_merged
export_size: 10        # 每个分片最大 10GB
export_device: auto
export_legacy_format: false
```

### 启动命令

```bash
# 方式1: 直接使用已有 yaml (SFT1)
conda activate llamafactory
cd /home/ps/hanjiaqi/LlamaFactory
llamafactory-cli export chatjoy/qwen30b_merge_clean32.yaml

# 方式2: 为 SFT2 创建 merge yaml 并执行
cat > /tmp/merge_sft2.yaml << 'EOF'
model_name_or_path: /home/ps/.cache/modelscope/hub/models/Qwen/Qwen3-30B-A3B-Instruct-2507
adapter_name_or_path: /home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/qwen3-30b-moe/clean32_for_sft2_fsdp_qlora
template: qwen
trust_remote_code: true
export_dir: /home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/qwen3-30b-moe/clean32_for_sft2_fsdp_qlora_merged
export_size: 10
export_device: auto
export_legacy_format: false
EOF

conda activate llamafactory
cd /home/ps/hanjiaqi/LlamaFactory
llamafactory-cli export /tmp/merge_sft2.yaml 2>&1 | tee /tmp/merge_sft2.log

# tmux 版（推荐，防止 SSH 断开）
tmux new-session -d -s merge "bash -lc '\
    source /home/ps/miniconda3/etc/profile.d/conda.sh && \
    conda activate llamafactory && \
    cd /home/ps/hanjiaqi/LlamaFactory && \
    llamafactory-cli export /tmp/merge_sft2.yaml \
    2>&1 | tee /tmp/merge_sft2.log'"
tmux attach -t merge
```

### 预期结果

- 输出 `.../clean32_for_sft2_fsdp_qlora_merged/`: 6个 `.safetensors` 分片文件, 总大小 ~57GB
- 合并耗时约 15-30 分钟（主要受 CPU RAM 限制）

---

## Stage 3 — RM 训练

> **当前状态**: RM v2 训练正在运行 (tmux: `rm_train`, 2026-04-16 启动)

### 环境

| 项 | 值 |
|---|---|
| conda env | `tslib` (transformers 4.57.1, deepspeed 0.16.9) |
| 框架 | OpenRLHF + DeepSpeed ZeRO-2 |
| DeepSpeed | `/home/ps/miniconda3/envs/tslib/bin/deepspeed` |
| Python | `/home/ps/miniconda3/envs/tslib/bin/python` |
| GPU | GPU 0,1,2,3 (NUMA0 节点, 4卡) |
| 架构 | ZeRO-2 + bitsandbytes 4-bit NF4 + LoRA + adam_offload |

**为什么必须用 tslib 而非 gemma4-character:**
- gemma4-character 的 bitsandbytes 与 OpenRLHF 动态 class 不兼容 → 4-bit 回退 BF16 → 57GB/卡 → OOM
- tslib 中 bitsandbytes 正常，模型实际加载 ~15GB/卡 NF4

**为什么用 ZeRO-2 而非 ZeRO-3:**
- ZeRO-3 每层 forward 需 allgather 完整参数 → 230GB/step 跨 NUMA 通信 → 30s/step → 不可行
- ZeRO-2 只通信 LoRA 梯度 (~800MB/step) → 即使跨 NUMA 也可接受
- ZeRO-3 + gradient_checkpointing 会产生 CheckpointError (tensor metadata 不匹配)

### 关键路径

| 类型 | 路径 |
|---|---|
| 初始化模型 (SFT-merged) | `/home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/qwen3-30b-moe/clean32_for_sft2_fsdp_qlora_merged/` |
| 训练数据 (filtered) | `/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/openrlhf_rm/rm_train_filtered.jsonl` (14,826 条) |
| 训练数据 (原始) | `/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/openrlhf_rm/rm_train.jsonl` (16,826 条) |
| 评估数据 | `/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/openrlhf_rm/rm_eval.jsonl` (885 条) |
| 输出路径 (v2) | `/home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/qwen3-30b-moe/chatjoy_rm_v2_filtered/` |
| 输出路径 (v1-参考) | `/home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/qwen3-30b-moe/chatjoy_rm_zero2_8gpu_20260415/` |
| 启动脚本 | `/home/ps/hanjiaqi/LlamaFactory/chatjoy/launch_rm_train.sh` |

**过滤逻辑 (rm_train_filtered.jsonl):**
```python
# 删除 2000 条 length_mismatch_long 且 reward_gap < 0.2 的噪声样本
filtered = [d for d in train_data if not (
    d['extra_info']['rejected_type'] == 'length_mismatch_long'
    and d['extra_info']['reward_gap'] < 0.2
)]
# 16826 → 14826 条
```

### 训练超参数

```
train_batch_size: 64          # global batch
micro_train_batch_size: 1     # per GPU
gradient_accumulation: 16     # = 64 / (4 GPU × 1)
max_epochs: 1
max_len: 3072
learning_rate: 9e-6
lr_scheduler: cosine
lr_warmup_ratio: 0.03
lora_rank: 64
lora_alpha: 128
target_modules: q_proj k_proj v_proj o_proj
zero_stage: 2
load_in_4bit: true
adam_offload: true            # Adam optimizer 状态放 CPU
gradient_checkpointing: true
param_dtype: bf16
attn_implementation: flash_attention_2
value_head_prefix: score
apply_chat_template: true
save_steps: 100               # 保存 LoRA adapter (~500MB)
eval_steps: 50                # 评估 → eval_rewards/ JSONL
save_hf_ckpt: true            # 只存 HF 格式 LoRA
disable_ds_ckpt: true         # 不存 DeepSpeed 格式 (~57GB)
```

### 启动命令

```bash
# 标准启动（推荐）
tmux new-session -d -s rm_train \
    "cd /home/ps/hanjiaqi/LlamaFactory && bash chatjoy/launch_rm_train.sh"
tmux attach -t rm_train

# 查看训练进度
tail -f /home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/qwen3-30b-moe/chatjoy_rm_v2_filtered/train.log

# 查看评估结果
ls /home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/qwen3-30b-moe/chatjoy_rm_v2_filtered/eval_rewards/
cat /home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/qwen3-30b-moe/chatjoy_rm_v2_filtered/eval_rewards/step_*.jsonl \
    | python3 -c "
import sys, json
data = [json.loads(l) for l in sys.stdin if l.strip()]
acc = sum(1 for d in data if d['chosen_reward'] > d['rejected_reward']) / len(data)
gap = sum(d['chosen_reward'] - d['rejected_reward'] for d in data) / len(data)
print(f'acc={acc:.3f}, gap_mean={gap:.3f}, n={len(data)}')
"
```

### 关键代码修复 (已应用)

**1. `min_lr` TypeError** — 修改 `/home/ps/hanjiaqi/OpenRLHF/openrlhf/cli/train_rm.py:104-110`:
```python
# 删除了 scheduler_specific_kwargs={"min_lr": args.learning_rate * 0.1}
scheduler = get_scheduler(
    args.lr_scheduler,
    optim,
    num_warmup_steps=math.ceil(max_steps * args.lr_warmup_ratio),
    num_training_steps=max_steps,
)
```

**2. 磁盘满导致保存崩溃** — 必须加 `--save_hf_ckpt --disable_ds_ckpt` 参数 (LoRA ~500MB vs DS ~57GB)

### 预期结果

- 总步数: 14,826 ÷ 64 = 232 batches × (实际 64/64=1 step per batch with accum) ≈ **232 gradient steps** per epoch  
  *(实际约 3,706 步，因为 micro_batch=1，64 GPUbatches per gradient step × 4GPU = 256 steps)*
- 实际: 3,706 steps (14826条 ÷ 4GPU ÷ 1 micro_batch = 3706 total fwd pass, 每64次更新1次)
- 单步耗时: ~14-15s, 总时长约 **14-15 小时**
- 每 50 步评估一次 → `eval_rewards/step_N.jsonl`
- 每 100 步保存一次 LoRA → `global_step_N/`

### 训练监控指标

```
# 正常状态:
acc > 0.9, gap_mean > 1.0 (step 200 时)
loss 在 0.5-0.7 之间震荡
chosen_reward > rejected_reward

# Bad case 5 个持续失败样本 (来自 v1 分析):
idx=0  (fa, length_mismatch_long, gap≈-0.04)
idx=55 (sv, wrong_role, gap≈-0.53, 在恶化)
idx=65 (ar, length_mismatch_short)
idx=102 (de, length_mismatch_short)
idx=107 (sv, length_mismatch_long)
```

---

## Stage 4 — GRPO Actor 训练

> **状态**: 脚本就绪，待 RM v2 训练完成后执行

### 环境

| 项 | 值 |
|---|---|
| conda env | `gemma4-rl` 或 `gemma4-character` |
| 框架 | verl (GSPO/GRPO) |
| GPU | 全部 8 卡 |

### 关键路径

| 类型 | 路径 |
|---|---|
| Actor 模型 (Gemma4) | `/home/ps/.cache/huggingface/hub/models--gemma-4-26B-A4B-it-bf16/snapshots/main` |
| RM 模型 | `/home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/qwen3-30b-moe/chatjoy_rm_v2_filtered/` (训练完成后) |
| 提示数据源 | `/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/runs/clean32_for_sft2.jsonl` |
| 数据格式转换脚本 | `/home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/framework_bridge/prepare_verl_rl_data.py` |
| 启动脚本 | `/home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/framework_bridge/verl/train_grpo_lora_actor_verl.sh` |
| 输出路径 | `/home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/verl/chatjoy_grpo_lora/` |

### 关键超参数

```bash
TRAIN_BATCH_SIZE=64
PPO_MINI_BATCH_SIZE=16
ROLLOUT_N=4              # 每个 prompt 生成 4 个回复
MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=768
LORA_RANK=32
LORA_ALPHA=32
LR=3e-6
ROLLOUT_GPU_MEMORY_UTILIZATION=0.60
RM_GPU_MEMORY_UTILIZATION=0.45
# GSPO 非对称 clip
CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.28
# MoE 负载均衡
ENTROPY_COEFF=0.01
MOE_AUX_LOSS_COEF=0.001
```

### 启动命令

```bash
# Step 1: 确认 RM 训练完成
ls /home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/qwen3-30b-moe/chatjoy_rm_v2_filtered/

# Step 2: 数据格式转换
conda activate gemma4-rl  # 或 gemma4-character
cd /home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/framework_bridge
python3 prepare_verl_rl_data.py \
    --input /home/ps/hanjiaqi/LlamaFactory/chatjoy/data/runs/clean32_for_sft2.jsonl \
    --train-output outputs/verl_rl_data/train.parquet \
    --val-output outputs/verl_rl_data/val.parquet

# Step 3: 启动 GRPO 训练
RM_MODEL_PATH=/home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/qwen3-30b-moe/chatjoy_rm_v2_filtered \
tmux new-session -d -s grpo_train \
    "bash /home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/framework_bridge/verl/train_grpo_lora_actor_verl.sh \
     2>&1 | tee /home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/verl/chatjoy_grpo_lora/train.log"
tmux attach -t grpo_train
```

---

## 关键路径速查

```
基础模型:
  Qwen3-30B-MoE: /home/ps/.cache/modelscope/hub/models/Qwen/Qwen3-30B-A3B-Instruct-2507
  Gemma4-26B:    /home/ps/.cache/huggingface/hub/models--gemma-4-26B-A4B-it-bf16/snapshots/main

SFT 产物:
  SFT1 adapter:  .../saves/qwen3-30b-moe/clean32_for_sft1_fsdp_qlora/
  SFT1 merged:   .../saves/qwen3-30b-moe/clean32_for_sft1_fsdp_qlora_merged/
  SFT2 adapter:  .../saves/qwen3-30b-moe/clean32_for_sft2_fsdp_qlora/
  SFT2 merged:   .../saves/qwen3-30b-moe/clean32_for_sft2_fsdp_qlora_merged/  ← RM 初始化用

RM 产物:
  RM v1 (参考): .../saves/qwen3-30b-moe/chatjoy_rm_zero2_8gpu_20260415/
  RM v2 (当前): .../saves/qwen3-30b-moe/chatjoy_rm_v2_filtered/              ← 训练中

数据:
  SFT2 data: .../chatjoy/data/runs/clean32_for_sft2.jsonl       (19,508)
  RM train:  .../data/openrlhf_rm/rm_train_filtered.jsonl       (14,826, filtered)
  RM eval:   .../data/openrlhf_rm/rm_eval.jsonl                 (885)

根目录: /home/ps/hanjiaqi/LlamaFactory/chatjoy/
```

---

## 常见错误与修复记录

| 阶段 | 错误 | 原因 | 修复 |
|------|------|------|------|
| RM 模型加载 | OOM (29.8GB 分配时崩溃) | gemma4-character env bitsandbytes 不兼容，4-bit 回退 BF16 | 换 tslib env |
| RM Step 0 | CheckpointError: tensor metadata 不匹配 | ZeRO-3 + gradient_checkpointing 冲突: ZeRO-3 在 fwd 后释放 allgather 参数，recompute 时 metadata 变化 | 改用 ZeRO-2 |
| RM Step 0 | ZeRO-3 被 NCCL 杀掉 | 跨 NUMA (5-10GB/s) 通信量 230GB/step → 超时 | 改用 ZeRO-2 |
| RM 配置 | AssertionError: 64 != 1×9×7 | 7 卡时 train_batch_size=64 不整除 | 使用 4 或 8 卡 |
| RM 训练 | TypeError: get_cosine_schedule_with_warmup() got unexpected argument 'min_lr' | OpenRLHF train_rm.py 调用了不支持的参数 | 删除 scheduler_specific_kwargs，见 train_rm.py:104 |
| RM 保存 | RuntimeError: file write failed (disk full) | DeepSpeed 默认保存完整模型状态 (~57GB) | 加 --save_hf_ckpt --disable_ds_ckpt |
| DeepSpeed 启动 | 忽略 CUDA_VISIBLE_DEVICES | `--num_gpus` 参数覆盖了 env var | 改用 `--include localhost:0,1,2,3` |
