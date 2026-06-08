#!/bin/bash

################################################################################
# ChatJoy GRPO强化学习训练完整流程
# 
# 本脚本自动执行以下步骤:
# 1. 环境配置
# 2. 生成奖励数据集
# 3. 运行GRPO训练
# 4. 对比SFT和GRPO模型性能
#
# Usage:
#   bash run_grpo_training.sh [--sample_size N] [--epochs E] [--batch_size B]
#
# Examples:
#   # 快速测试 (500样本, 1 epoch)
#   bash run_grpo_training.sh --sample_size 500 --epochs 1
#   
#   # 完整训练 (所有数据, 3 epochs)
#   bash run_grpo_training.sh
#
################################################################################

set -e  # 遇到错误立即退出

# ======== 配置项 ========

# GPU配置
export CUDA_VISIBLE_DEVICES=1,2,5,6
export TOKENIZERS_PARALLELISM=false

# 路径
CHATJOY_DIR="/home/ps/hanjiaqi/LlamaFactory/chatjoy"
DATA_DIR="${CHATJOY_DIR}/data"
MODEL_BASE_DIR="${CHATJOY_DIR}/saves/qwen3-30b-moe"

# 模型路径 (需要是SFT后的模型)
SFT_MODEL="${MODEL_BASE_DIR}/clean32_for_sft2_fsdp_qlora_merged"

# 数据路径
CLEAN_DATA="${DATA_DIR}/clean32_for_sft2.jsonl"

# 输出配置
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE="${CHATJOY_DIR}/grpo_output_${TIMESTAMP}"
REWARD_DATA="${OUTPUT_BASE}/chatjoy_with_rewards.jsonl"
GRPO_OUTPUT="${OUTPUT_BASE}/grpo_checkpoint"

# 训练参数
SAMPLE_SIZE=0          # 0表示用所有数据
NUM_EPOCHS=3
BATCH_SIZE=4
LEARNING_RATE=5e-5
GROUP_SIZE=4
WARMUP_STEPS=500

# ======== 解析命令行参数 ========

while [[ $# -gt 0 ]]; do
    case $1 in
        --sample_size)
            SAMPLE_SIZE="$2"
            shift 2
            ;;
        --epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            shift
            ;;
    esac
done

# ======== 日志函数 ========

log_header() {
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  $1"
    echo "════════════════════════════════════════════════════════════"
    echo ""
}

log_info() {
    echo "[INFO] $1"
}

log_error() {
    echo "[ERROR] $1" >&2
}

log_success() {
    echo "[✓] $1"
}

# ======== 验证环境 ========

log_header "STEP 0: 环境验证"

if [ ! -d "$CHATJOY_DIR" ]; then
    log_error "ChatJoy目录不存在: $CHATJOY_DIR"
    exit 1
fi

if [ ! -f "$SFT_MODEL/config.json" ]; then
    log_error "SFT模型不存在: $SFT_MODEL"
    exit 1
fi

if [ ! -f "$CLEAN_DATA" ]; then
    log_error "训练数据不存在: $CLEAN_DATA"
    exit 1
fi

log_success "ChatJoy目录存在"
log_success "SFT模型存在"
log_success "训练数据存在"

# ======== 环境激活 ========

log_header "STEP 1: 激活环境"

source /home/ps/miniconda3/etc/profile.d/conda.sh

if ! conda env list | grep -q "lzyenv"; then
    log_error "环境 'lzyenv' 不存在"
    exit 1
fi

conda activate lzyenv
log_success "环境 'lzyenv' 已激活"

# ======== 创建输出目录 ========

log_header "STEP 2: 创建输出目录"

mkdir -p "$OUTPUT_BASE"
mkdir -p "$GRPO_OUTPUT"

log_info "输出目录: $OUTPUT_BASE"
log_success "目录已创建"

# ======== 生成奖励数据 ========

log_header "STEP 3: 生成奖励数据集"

log_info "Input: $CLEAN_DATA"
log_info "Output: $REWARD_DATA"

if [ "$SAMPLE_SIZE" -gt 0 ]; then
    log_info "Mode: Testing (first $SAMPLE_SIZE samples)"
    SAMPLE_ARG="--sample_size $SAMPLE_SIZE"
else
    log_info "Mode: Full training (all samples)"
    SAMPLE_ARG=""
fi

cd "$CHATJOY_DIR"

python generate_reward_data.py \
    --input_file "$CLEAN_DATA" \
    --output_file "$REWARD_DATA" \
    --system_prompt_dir "TOP20CHARACTERS_systemprompt" \
    $SAMPLE_ARG \
    || {
        log_error "奖励数据生成失败"
        exit 1
    }

log_success "奖励数据生成完成: $REWARD_DATA"

# 显示统计信息
STATS_FILE="${REWARD_DATA%.jsonl}_stats.json"
if [ -f "$STATS_FILE" ]; then
    log_info "奖励统计信息:"
    cat "$STATS_FILE" | sed 's/^/  /'
fi

# ======== 运行GRPO训练 ========

log_header "STEP 4: 运行GRPO训练"

log_info "Configuration:"
log_info "  Model: $SFT_MODEL"
log_info "  Data: $REWARD_DATA"
log_info "  Output: $GRPO_OUTPUT"
log_info "  Epochs: $NUM_EPOCHS"
log_info "  Batch Size: $BATCH_SIZE"
log_info "  Learning Rate: $LEARNING_RATE"
log_info "  Group Size: $GROUP_SIZE"
log_info "  Warmup Steps: $WARMUP_STEPS"
echo ""

python grpo_trainer.py \
    --model_path "$SFT_MODEL" \
    --data_file "$REWARD_DATA" \
    --output_dir "$GRPO_OUTPUT" \
    --num_epochs "$NUM_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --group_size "$GROUP_SIZE" \
    --warmup_steps "$WARMUP_STEPS" \
    --save_steps 50 \
    --logging_steps 10 \
    || {
        log_error "GRPO训练失败"
        exit 1
    }

log_success "GRPO训练完成"

# ======== 找到最佳检查点 ========

log_header "STEP 5: 处理模型检查点"

# 查找latest checkpoint
LATEST_CHECKPOINT=$(ls -td "$GRPO_OUTPUT"/checkpoint-* 2>/dev/null | head -n1)

if [ -z "$LATEST_CHECKPOINT" ]; then
    LATEST_CHECKPOINT=$(ls -td "$GRPO_OUTPUT"/checkpoint-* 2>/dev/null | tail -n1)
fi

if [ -n "$LATEST_CHECKPOINT" ]; then
    log_info "Latest checkpoint: $LATEST_CHECKPOINT"
    
    # 创建final-model符号链接
    FINAL_MODEL="${OUTPUT_BASE}/final_grpo_model"
    ln -sf "$LATEST_CHECKPOINT" "$FINAL_MODEL"
    log_success "Final model link: $FINAL_MODEL"
else
    log_error "No checkpoints found"
fi

# ======== 生成总结报告 ========

log_header "STEP 6: 生成报告"

REPORT_FILE="${OUTPUT_BASE}/training_report.md"

cat > "$REPORT_FILE" << EOF
# ChatJoy GRPO Training Report

## Training Summary
- **Timestamp**: $TIMESTAMP
- **SFT Model**: $SFT_MODEL
- **Training Data**: $CLEAN_DATA
- **Output Directory**: $OUTPUT_BASE

## Configuration
- **Epochs**: $NUM_EPOCHS
- **Batch Size**: $BATCH_SIZE
- **Learning Rate**: $LEARNING_RATE
- **Group Size**: $GROUP_SIZE
- **Warmup Steps**: $WARMUP_STEPS

## Results
- **Reward Data**: \`$(basename $REWARD_DATA)\`
- **Final Model**: \`$(basename $FINAL_MODEL)\`

EOF

if [ -f "$STATS_FILE" ]; then
    cat >> "$REPORT_FILE" << 'EOF'

## Reward Distribution

```
EOF
    cat "$STATS_FILE" >> "$REPORT_FILE"
    cat >> "$REPORT_FILE" << 'EOF'
```

EOF
fi

log_success "Report generated: $REPORT_FILE"

# ======== 最终总结 ========

log_header "TRAINING COMPLETE ✓"

cat << EOF

Summary:
  - SFT Model: $SFT_MODEL
  - GRPO Output: $GRPO_OUTPUT
  
Next Steps:
  1. 合并GRPO模型:
     python merge_grpo_model.py \\
       --sft_model $SFT_MODEL \\
       --grpo_model $(basename $FINAL_MODEL) \\
       --output_dir ${OUTPUT_BASE}/merged_model

  2. 测试微调后的模型:
     python demo_character.py \\
       --model_path $(basename $FINAL_MODEL)/final

  3. 比较SFT vs GRPO性能:
     python benchmark_models.py \\
       --sft_model $SFT_MODEL \\
       --grpo_model $(basename $FINAL_MODEL) \\
       --test_file data/test.jsonl

Logs:
  - Training stats: ${REWARD_DATA%.jsonl}_stats.json
  - Report: $REPORT_FILE
  - Checkpoints: $GRPO_OUTPUT

EOF

log_success "All done!"
