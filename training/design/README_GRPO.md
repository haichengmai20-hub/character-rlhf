# ChatJoy GRPO强化学习训练完整指南

**目标**: 使用GRPO (Group Relative Policy Optimization)对ChatJoy虚拟角色模型进行强化学习微调，提高角色一致性、对话流畅度和安全性。

---

## 快速开始 (3分钟)

### 测试模式 (快速验证)

```bash
cd /home/ps/hanjiaqi/LlamaFactory/chatjoy

# 运行完整流程 (使用前500条数据, 不到2小时)
bash run_grpo_training.sh --sample_size 500 --epochs 1
```

### 生产模式 (完整训练)

```bash
# 使用所有数据, 3个epochs (40-50小时)
bash run_grpo_training.sh --epochs 3
```

---

## 详细说明

### 1. 系统需求

```
GPU: 4×A100 (或等效GPU)
内存: 120GB系统内存
磁盘: 150GB (模型+数据)
环境: Python 3.10+, PyTorch 2.0+
```

### 2. 项目结构

```
chatjoy/
├── grpo_trainer.py                    # GRPO训练引擎
├── generate_reward_data.py            # 奖励数据生成
├── run_grpo_training.sh               # 自动启动脚本
├── reward_model_design.md             # 完整的奖励设计文档
├── grpo_training_implementation.md    # GRPO实现细节
│
├── data/
│   └── clean32_for_sft2.jsonl        # 训练数据
│
├── saves/qwen3-30b-moe/
│   └── clean32_for_sft2_fsdp_qlora_merged/  # SFT模型
│
├── TOP20CHARACTERS_systemprompt/
│   ├── 小红.json
│   ├── 李明.json
│   └── ... (20个角色)
│
└── grpo_output_YYYYMMDD_HHMMSS/
    ├── chatjoy_with_rewards.jsonl           # 带奖励的数据
    ├── chatjoy_with_rewards_stats.json      # 统计信息
    ├── grpo_checkpoint/
    │   ├── checkpoint-50/
    │   ├── checkpoint-100/
    │   └── checkpoint-final/
    └── training_report.md
```

---

## Step-by-Step流程说明

### Step 1: 奖励数据生成

```bash
python generate_reward_data.py \
    --input_file data/clean32_for_sft2.jsonl \
    --output_file grpo_output/chatjoy_with_rewards.jsonl \
    --system_prompt_dir TOP20CHARACTERS_systemprompt
```

**功能**: 
- 读取SFT训练数据
- 为每条数据计算奖励分数 [-1.0, 1.0]
- 输出JSONL格式的数据+奖励

**输出示例**:
```json
{
  "prompt": "小红，今天天气很好，你在干什么？",
  "response": "天气好的日子，我最喜欢在户外玩了！😊 阳光洒在身上暖洋洋的，很舒服呢。",
  "reward": 0.72,
  "reward_breakdown": {
    "role_adherence": 0.8,
    "dialogue_coherence": 0.75,
    "language_consistency": 1.0,
    "safety": 1.0,
    "response_quality": 0.5
  }
}
```

**奖励统计分布**:
```
Excellent  [0.8, 1.0):    10% (杰出回复)
Good       [0.4, 0.8):    35% (优质回复)
Acceptable [0.0, 0.4):    35% (可接受)
Poor       [-0.5, 0.0):   15% (不好)
Severe     [-1.0, -0.5):  5%  (严重问题)
```

### Step 2: GRPO训练

```bash
python grpo_trainer.py \
    --model_path saves/qwen3-30b-moe/clean32_for_sft2_fsdp_qlora_merged \
    --data_file grpo_output/chatjoy_with_rewards.jsonl \
    --output_dir grpo_output/grpo_checkpoint \
    --num_epochs 3 \
    --batch_size 4 \
    --learning_rate 5e-5 \
    --group_size 4 \
    --warmup_steps 500
```

**GRPO核心算法**:

```
对于每个batch:
  group_size = 4 (每个prompt的4条轨迹)
  
  for each trajectory in group:
    relative_reward = reward - mean(group_rewards)  # 组内排名
    
    if relative_reward > 0:
      增加 log_prob(生成该response的概率)
      → 模型学会生成高质量回复
    else:
      减少 log_prob
      → 模型避免生成低质量回复
```

**训练过程监控**:
```
Epoch 1/3
├─ Loss: 0.2341 (↓)
├─ Mean Reward: 0.58
├─ Learning Rate: 5.00e-05
└─ Checkpoint saved: checkpoint-50

Epoch 2/3
├─ Loss: 0.1923 (↓)  # 收敛中
├─ Mean Reward: 0.65
└─ ...

Epoch 3/3
├─ Loss: 0.1654 (↓)  # 继续改进
├─ Mean Reward: 0.71
└─ Checkpoint saved: checkpoint-final
```

### Step 3: 模型合并与测试

```bash
# 对比SFT vs GRPO
python compare_models.py \
    --sft_model saves/qwen3-30b-moe/clean32_for_sft2_fsdp_qlora_merged \
    --grpo_model grpo_output/grpo_checkpoint/checkpoint-final \
    --test_prompts [
      "小红，你今天开心吗？",
      "如何处理人生中的失败？"
    ]
```

---

## 关键参数解释

### Batch Size & Gradient Accumulation

```
per_device_train_batch_size = 4
gradient_accumulation_steps = 2

→ 实际batch_size = 4 × 2 = 8
→ 内存效率: 更新间隔每8条数据
```

### Group Size (GRPO特定)

```
group_size = 4  ✓ 推荐值

群组大小的影响:
  group_size=2: 信号强度弱, 容易过拟合
  group_size=4: ✓ 平衡 (推荐)
  group_size=8: 信号混淆, 收敛慢

实现细节:
  对于同一个prompt,选择4条不同的response
  比较它们的相对奖励值
  进行Policy更新
```

### Learning Rate Schedule

```
warmup_steps = 500       # 前500步线性升温
scheduler: cosine        # 之后余弦衰减

LR变化曲线:
epoch 1: 0 → 5e-5 → 3e-5
epoch 2: 3e-5 → 2e-5
epoch 3: 2e-5 → 1e-5 → 0

效果: 
  - 避免初期剧烈变化
  - 后期缓慢收敛
```

---

## 故障排除

### 1. 内存溢出 (OOM)

```
症状: 
  RuntimeError: CUDA out of memory

解决方案:
  # 减小batch_size
  --batch_size 2  
  
  # 增加gradient_accumulation_steps  
  --gradient_accumulation_steps 4
  
  # 或减小group_size
  --group_size 2
```

### 2. 训练不收敛

```
症状:
  Loss不下降或振荡

检查:
  1. 学习率太高? → 改为 1e-5
  2. 奖励信号设计差? → 检查reward_model
  3. 数据质量差? → 验证training data
  
重启命令:
  python grpo_trainer.py ... --learning_rate 1e-5
```

### 3. 奖励分布异常

```
症状:
  所有奖励都接近0或都是正数

原因:
  1. 奖励模型计算有问题
  2. 数据质量太好或太差
  
检查:
  # 查看奖励统计
  cat grpo_output/chatjoy_with_rewards_stats.json
  
  # 应该有多种分布:
  {
    "excellent": 10%,
    "good": 35%,
    "acceptable": 35%,
    "poor": 15%,
    "severe": 5%
  }
```

---

## 集成到生产环境

### 1. 模型转换

如果使用了LoRA:
```bash
# 合并LoRA权重到基础模型
python merge_lora.py \
    --base_model qwen3-30b-moe \
    --lora_model grpo_output/grpo_checkpoint/checkpoint-final \
    --output_model grpo_output/final_merged_model
```

### 2. 模型部署

```bash
# 使用vLLM启动推理服务器
python -m vllm.entrypoints.openai.api_server \
    --model grpo_output/final_merged_model \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --port 8000

# 测试API
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "final_merged_model",
    "messages": [{"role": "user", "content": "小红，你好!"}]
  }'
```

### 3. A/B测试

```bash
# 运行A/B对比
python ab_test.py \
    --model_a saves/qwen3-30b-moe/clean32_for_sft2_fsdp_qlora_merged \
    --model_b grpo_output/final_merged_model \
    --test_samples 1000 \
    --evaluate_metrics [role_consistency, coherence, safety]
```

---

## 性能指标

### RL前后对比 (预期改进)

```
指标                    SFT     GRPO    改进
─────────────────────────────────────────
角色一致性              0.72    0.81    +12.5%
对话连贯性              0.68    0.78    +14.7%
语言一致性              0.85    0.92    +8.2%
安全性 (低违规率)       0.95    0.98    +3.2%
平均奖励                0.45    0.67    +48.9%
```

### 训练时间

```
PoC (500样本×1epoch):
  - 数据准备: 2分钟
  - GRPO训练: 8分钟
  - 总计: 10分钟

实验 (2000样本×2epoch):
  - 数据准备: 5分钟
  - GRPO训练: 45分钟
  - 总计: 50分钟

生产 (10000样本×3epoch):
  - 数据准备: 15分钟
  - GRPO训练: 12小时
  - 总计: 12.5小时
```

---

## 高级配置

### 多GPU分布式训练

```bash
# 使用4个GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NPROC_PER_NODE=4

# 使用torchrun启动 (自动分布)
torchrun --nproc_per_node=4 grpo_trainer.py \
    --model_path ... \
    --data_file ...
```

### 混合精度训练

```python
# 在grpo_trainer.py中 (已默认启用)
with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
    outputs = self.model(...)
    loss = F.cross_entropy(...)
```

### 自定义奖励函数

编辑 `reward_model_design.md` Part 1中的权重:

```python
# 调整维度权重
WEIGHTS = {
    'role_adherence': 0.40,         # ↑ 强调角色一致
    'dialogue_coherence': 0.25,     # ←
    'language_consistency': 0.15,   # ↓
    'safety': 0.15,                 # ←
    'response_quality': 0.05        # ←
}

# 计算最终奖励
total_reward = sum(WEIGHTS[dim] * score[dim] for dim in WEIGHTS)
```

---

## 监控与调试

### 实时日志查看

```bash
# Terminal 1: 启动训练
python grpo_trainer.py ...

# Terminal 2: 查看GPU占用
watch -n 0.5 'nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader'

# Terminal 3: 查看checkpoint变化
watch -n 5 'ls -lh grpo_output/grpo_checkpoint/checkpoint-*/'
```

### TensorBoard可视化

```bash
# 查看训练曲线 (如果启用)
tensorboard --logdir grpo_output/logs --port 6006
# 访问 http://localhost:6006
```

### 检查点管理

```bash
# 查看所有检查点
ls -lh grpo_output/grpo_checkpoint/

# 恢复特定检查点 (从中断处继续)
python grpo_trainer.py \
    --resume_from_checkpoint grpo_output/grpo_checkpoint/checkpoint-500 \
    ...

# 删除旧检查点 (节省空间)
rm -rf grpo_output/grpo_checkpoint/checkpoint-{1..49}
```

---

## 下一步行动

### 立即开始

```bash
# 1. 测试完整流程 (预计20分钟)
cd /home/ps/hanjiaqi/LlamaFactory/chatjoy
bash run_grpo_training.sh --sample_size 100 --epochs 1

# 2. 检查输出
ls -lh grpo_output_*/

# 3. 查看奖励统计
cat grpo_output_*/chatjoy_with_rewards_stats.json
```

### 根据结果调整

```bash
# 如果奖励分布不理想 → 调整reward_model
# 如果loss不收敛 → 调整learning_rate
# 如果内存不足 → 减小batch_size
# 按需重新运行完整训练
```

### 投入生产

```bash
# 1. 运行完整训练 (3 epochs)
bash run_grpo_training.sh --epochs 3

# 2. A/B测试对比
python ab_test.py --sft_model ... --grpo_model ...

# 3. 部署新模型
python deploy_model.py --checkpoint grpo_output/.../final
```

---

## 常用命令速查

```bash
# 生成奖励数据 (500条用于测试)
python generate_reward_data.py \
    --input_file data/clean32_for_sft2.jsonl \
    --output_file grpo_data_test.jsonl \
    --sample_size 500

# 启动训练
python grpo_trainer.py \
    --model_path saves/qwen3-30b-moe/clean32_for_sft2_fsdp_qlora_merged \
    --data_file grpo_data_test.jsonl \
    --output_dir grpo_checkpoint

# 查看最新检查点
ls -dt grpo_checkpoint/checkpoint-*/  | head -1

# 对比模型
python compare_models.py \
    --sft_model saves/qwen3-30b-moe/... \
    --grpo_model grpo_checkpoint/checkpoint-final
```

---

## 文档索引

- `reward_model_design.md` - 完整的奖励设计 (5000+行)
- `grpo_training_implementation.md` - GRPO实现细节
- `grpo_trainer.py` - 主训练脚本
- `generate_reward_data.py` - 数据生成脚本
- `run_grpo_training.sh` - 自动启动脚本

---

## 支持联系

如遇问题, 检查以下内容:

1. **GPU配置**: `nvidia-smi` 查看GPU状态
2. **环境变量**: `echo $CUDA_VISIBLE_DEVICES`
3. **文件路径**: `ls -la grpo_output_*/`
4. **日志文件**: 查看`grpo_output_*/` 中的日志

最后修改: 2024年
