#!/usr/bin/env python3
"""
ChatJoy GRPO 快速参考

直接运行此文件获取完整的命令和说明
"""

QUICK_START = """
╔════════════════════════════════════════════════════════════════════╗
║                    ChatJoy GRPO 快速参考卡                         ║
╚════════════════════════════════════════════════════════════════════╝

【最快上手 - 3分钟】

$ cd /home/ps/hanjiaqi/LlamaFactory/chatjoy
$ bash run_grpo_training.sh --sample_size 500 --epochs 1

✓ 预期: 20分钟内完成, 生成 grpo_output_YYYYMMDD_HHMMSS/ 文件夹


【完整训练 - 生产版】

$ bash run_grpo_training.sh --epochs 3

✓ 预期: 40-50小时, 最终模型在 checkpoint-final/


【模型对比】

$ python compare_models.py \\
    --sft_model saves/qwen3-30b-moe/clean32_for_sft2_fsdp_qlora_merged \\
    --grpo_model grpo_output_YYYYMMDD_/grpo_checkpoint/checkpoint-final \\
    --num_samples 50

✓ 输出: comparison_report.txt


【单步骤命令】

# 1. 生成奖励数据
python generate_reward_data.py \\
    --input_file data/clean32_for_sft2.jsonl \\
    --output_file chatjoy_with_rewards.jsonl \\
    --sample_size 500

# 2. 运行GRPO训练
python grpo_trainer.py \\
    --model_path saves/qwen3-30b-moe/clean32_for_sft2_fsdp_qlora_merged \\
    --data_file chatjoy_with_rewards.jsonl \\
    --output_dir grpo_checkpoint \\
    --num_epochs 3 \\
    --batch_size 4

# 3. 监控GPU (新terminal)
watch -n 0.5 'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader'


════════════════════════════════════════════════════════════════════

【文件说明】

grpo_trainer.py
  ├─ 主训练脚本 (485行)
  ├─ 包含: GRPODataset + GRPOTrainer 类
  └─ 直接可运行

generate_reward_data.py
  ├─ 奖励数据生成 (352行)
  ├─ 包含: SimpleRewardModel 和 完整生成流程
  └─ 生成 data_with_rewards.jsonl

compare_models.py
  ├─ SFT vs GRPO 对比 (398行)
  ├─ 输出: 详细对比报告
  └─ 可视化改进指标

run_grpo_training.sh
  ├─ 自动化启动脚本 (284行)
  ├─ 完整流程: 环境→数据→训练→报告
  └─ 推荐方式 ✓


════════════════════════════════════════════════════════════════════

【关键参数】

--sample_size N         # 只处理前N条数据 (测试用)
--epochs N              # 训练轮数 (推荐: 1或3)
--batch_size N          # 批大小 (4推荐, OOM时改2)
--learning_rate 5e-5    # 学习率 (默认5e-5)
--group_size 4          # 群组大小 (GRPO特定, 4最优)
--warmup_steps 500      # 预热步数 (通常500即可)

========================================================

【预期输出】

✓ 快速测试 (500条, 1 epoch):
  - 时间: 20分钟
  - 输出: grpo_output_YYYYMMDD_HHMMSS/
    ├─ chatjoy_with_rewards.jsonl
    ├─ grpo_checkpoint/checkpoint-50/
    └─ grpo_checkpoint/checkpoint-final/

✓ 完整训练 (10000条, 3 epochs):
  - 时间: 40小时
  - 最终改进: +15-25%

========================================================

【监控指标】

Loss 曲线:
  Epoch 1: 初期 ~0.22, 末期 ~0.20  ✓
  Epoch 2: 初期 ~0.19, 末期 ~0.16  ✓
  Epoch 3: 初期 ~0.16, 末期 ~0.14  ✓

奖励分布 (预期):
  Excellent [0.8, 1.0):    10%
  Good      [0.4, 0.8):    35%
  Acceptable [0.0, 0.4):   35%
  Poor      [-0.5, 0.0):   15%
  Severe    [-1.0, -0.5):  5%

GPU 负载:
  GPU Util: >85% (正常)
  Memory:   >50GB/80GB (正常)

========================================================

【常见问题】

Q: 报错 'module not found'?
A: conda activate lzyenv && pip install -r requirements.txt

Q: OOM (内存不足)?
A: --batch_size 2 --gradient_accumulation_steps 4

Q: Loss不下降?
A: 改小学习率 --learning_rate 1e-5

Q: 生成内容变差?
A: 检查奖励函数设计, 查看 reward distribution

Q: 怎么继续训练?
A: grpo_trainer.py --resume_from_checkpoint checkpoint-300

========================================================

【关键文件位置】

源代码:
  /home/ps/hanjiaqi/LlamaFactory/chatjoy/
    ├── grpo_trainer.py
    ├── generate_reward_data.py
    ├── compare_models.py
    └── run_grpo_training.sh

模型:
  /home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/qwen3-30b-moe/
    └── clean32_for_sft2_fsdp_qlora_merged/  (SFT模型)

数据:
  /home/ps/hanjiaqi/LlamaFactory/chatjoy/data/
    └── clean32_for_sft2.jsonl  (训练数据)

输出 (执行后):
  /home/ps/hanjiaqi/LlamaFactory/chatjoy/grpo_output_YYYYMMDD_/
    ├── chatjoy_with_rewards.jsonl  (奖励数据)
    ├── grpo_checkpoint/  (模型检查点)
    └── training_report.md

========================================================

【推荐流程】

1️⃣ 验证环境 (1分钟)
   nvidia-smi         # 检查GPU
   conda activate lzyenv  # 激活环境

2️⃣ 快速测试 (20分钟) ← 首先运行
   bash run_grpo_training.sh --sample_size 500 --epochs 1

3️⃣ 检查结果 (2分钟)
   cat grpo_output_*/chatjoy_with_rewards_stats.json
   ls -lh grpo_output_*/grpo_checkpoint/

4️⃣ 完整训练 (40小时) ← 确认无误后运行
   bash run_grpo_training.sh --epochs 3

5️⃣ 评估对比 (5分钟) ← 训练完成后
   python compare_models.py --sft_model ... --grpo_model ...

6️⃣ 生产部署 (可选)
   python -m vllm.entrypoints.openai.api_server \\
       --model grpo_output_*/grpo_checkpoint/checkpoint-final

════════════════════════════════════════════════════════════════════

【资源需求】

GPU: 4×A100 (必须) / RTX 6000 (可以) / H100 (更优)
CPU: 32+ cores
内存: 120GB RAM
磁盘: 150GB 用于模型+数据

【时间线】

Quick Test          PoC           实验       生产
(500条)         (2000条)      (5000条)   (10000条)
   20分钟→ 50分钟→ 2小时→ 40小时

【后续阅读】

📚 详细指南:
   - README_GRPO.md (快速开始 + 故障排除)
   - reward_model_design.md (奖励设计详解)
   - grpo_training_implementation.md (GRPO实现细节)
   - GRPO_IMPLEMENTATION_SUMMARY.md (完整总结)

════════════════════════════════════════════════════════════════════

【一键启动】

最简单的方式:

    cd /home/ps/hanjiaqi/LlamaFactory/chatjoy
    bash run_grpo_training.sh --sample_size 500 --epochs 1

就这样! 脚本会自动:
  ✓ 配置环境
  ✓ 生成奖励数据
  ✓ 运行GRPO训练
  ✓ 保存模型
  ✓ 生成报告

════════════════════════════════════════════════════════════════════
"""

if __name__ == "__main__":
    print(QUICK_START)
    
    # 也可以保存为文件
    with open("QUICK_REFERENCE.txt", "w", encoding="utf-8") as f:
        f.write(QUICK_START)
    
    print("\n✓ 快速参考已保存到 QUICK_REFERENCE.txt")
