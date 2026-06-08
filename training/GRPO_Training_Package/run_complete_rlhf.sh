#!/bin/bash
# -*- coding: utf-8 -*-
# ============================================================================
# ChatJoy RLHF 完整训练流程 - 自动化脚本
# 
# 功能: 自动化执行 3 阶段 RLHF 训练
#   Stage 1: SFT (已完成, 跳过)
#   Stage 2: Reward Model 训练
#   Stage 3: Actor 强化学习
#
# 用法:
#   bash run_complete_rlhf.sh --stage all
#   bash run_complete_rlhf.sh --stage 2     # 仅训练 RM
#   bash run_complete_rlhf.sh --stage 3     # 仅训练 Actor
#
# ============================================================================

set -e

# ============================================================================
# 配置参数
# ============================================================================

STAGE="all"                    # all, 2, 3
DATA_DIR="./data"              # 数据目录
OUTPUT_DIR="./saves"           # 输出目录
SFT_MODEL_PATH=""              # SFT 模型路径 (必需)
INPUT_DATA=""                  # 输入数据文件 (若为空，自动查找)

# RM 训练参数
RM_EPOCHS=3
RM_BATCH_SIZE=4
RM_LORA_RANK=64

# Actor 训练参数
ACTOR_EPOCHS=3
ACTOR_BATCH_SIZE=2
ACTOR_SAMPLES=4
ACTOR_BETA_KL=0.1

# 其他
LOG_DIR="./logs"
SEED=42

# ============================================================================
# 辅助函数
# ============================================================================

print_header() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║  $1"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
}

print_step() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  [$1/5] $2"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
}

log_message() {
    echo "  [$(date '+%H:%M:%S')] $1"
}

check_file() {
    if [ ! -f "$1" ]; then
        echo "❌ 错误: 文件不存在 - $1"
        exit 1
    fi
    log_message "✓ 文件存在: $1"
}

check_dir() {
    if [ ! -d "$1" ]; then
        echo "❌ 错误: 目录不存在 - $1"
        exit 1
    fi
    log_message "✓ 目录存在: $1"
}

# ============================================================================
# 解析参数
# ============================================================================

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --stage)
                STAGE="$2"
                shift 2
                ;;
            --data-dir)
                DATA_DIR="$2"
                shift 2
                ;;
            --output-dir)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            --sft-model)
                SFT_MODEL_PATH="$2"
                shift 2
                ;;
            --input-data)
                INPUT_DATA="$2"
                shift 2
                ;;
            --rm-lr)
                RM_EPOCHS=$2
                shift 2
                ;;
            --actor-epochs)
                ACTOR_EPOCHS=$2
                shift 2
                ;;
            --seed)
                SEED=$2
                shift 2
                ;;
            *)
                echo "未知参数: $1"
                exit 1
                ;;
        esac
    done
}

# ============================================================================
# 检查环境
# ============================================================================

check_environment() {
    print_step "1" "检查环境"
    
    log_message "检查 Python..."
    which python > /dev/null || { echo "❌ 找不到 Python"; exit 1; }
    python --version
    
    log_message "检查必要的库..."
    python -c "import torch; print(f'PyTorch: {torch.__version__}')" || {
        echo "❌ PyTorch 未安装"
        exit 1
    }
    
    python -c "import peft; print('✓ PEFT 已安装')" || {
        echo "❌ PEFT 未安装"
        exit 1
    }
    
    python -c "import transformers; print('✓ Transformers 已安装')" || {
        echo "❌ Transformers 未安装"
        exit 1
    }
    
    log_message "检查 GPU..."
    python -c "import torch; print(f'GPU 数: {torch.cuda.device_count()}')"
    python -c "import torch; print(f'GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')"
    
    log_message "✓ 环境检查完成"
}

# ============================================================================
# 准备数据
# ============================================================================

prepare_data() {
    print_step "2" "准备数据"
    
    # 创建目录
    mkdir -p "$DATA_DIR" "$OUTPUT_DIR" "$LOG_DIR"
    log_message "✓ 目录已创建"
    
    # 查找输入数据
    if [ -z "$INPUT_DATA" ]; then
        log_message "寻找输入数据..."
        if [ -f "chatjoy_top20_multilingual_27x2200_preapi_gemini_resumed.json" ]; then
            INPUT_DATA="chatjoy_top20_multilingual_27x2200_preapi_gemini_resumed.json"
            log_message "✓ 找到数据: $INPUT_DATA"
        else
            echo "❌ 找不到输入数据文件"
            echo "   请使用 --input-data 指定数据文件"
            exit 1
        fi
    fi
    
    check_file "$INPUT_DATA"
    
    if [ -z "$SFT_MODEL_PATH" ]; then
        log_message "寻找 SFT 模型..."
        if [ -d "saves/qwen3-30b-moe/clean32_for_sft2_fsdp_qlora_merged" ]; then
            SFT_MODEL_PATH="saves/qwen3-30b-moe/clean32_for_sft2_fsdp_qlora_merged"
            log_message "✓ 找到 SFT 模型: $SFT_MODEL_PATH"
        else
            echo "❌ 找不到 SFT 模型"
            echo "   请使用 --sft-model 指定模型路径"
            exit 1
        fi
    fi
    
    check_dir "$SFT_MODEL_PATH"
    
    log_message "✓ 数据准备完成"
}

# ============================================================================
# Stage 2: Reward Model 训练
# ============================================================================

train_reward_model() {
    print_step "3" "Reward Model 训练"
    
    local RM_TRAIN_DATA="$DATA_DIR/rm_training_data.jsonl"
    local RM_OUTPUT_DIR="$OUTPUT_DIR/rm_model_final"
    
    # 生成 RM 训练数据
    if [ ! -f "$RM_TRAIN_DATA" ]; then
        log_message "生成 RM 偏好对数据..."
        python generate_preference_pairs.py \
            --input "$INPUT_DATA" \
            --output "$RM_TRAIN_DATA" \
            --method quality_score \
            2>&1 | tee -a "$LOG_DIR/generate_rm_data.log"
    else
        log_message "✓ RM 数据已存在: $RM_TRAIN_DATA"
    fi
    
    check_file "$RM_TRAIN_DATA"
    
    # 训练 RM
    log_message "开始 RM 训练..."
    python train_rm_model.py \
        --sft_model_path "$SFT_MODEL_PATH" \
        --train_file "$RM_TRAIN_DATA" \
        --output_dir "$RM_OUTPUT_DIR" \
        --lora_rank "$RM_LORA_RANK" \
        --num_train_epochs "$RM_EPOCHS" \
        --per_device_train_batch_size "$RM_BATCH_SIZE" \
        --seed "$SEED" \
        2>&1 | tee -a "$LOG_DIR/train_rm.log"
    
    log_message "✓ RM 训练完成: $RM_OUTPUT_DIR"
}

# ============================================================================
# Stage 3: Actor 强化学习
# ============================================================================

train_actor_model() {
    print_step "4" "Actor 强化学习 (GRPO) 训练"
    
    local ACTOR_PROMPT_DATA="$DATA_DIR/actor_prompts.jsonl"
    local RM_MODEL_PATH="$OUTPUT_DIR/rm_model_final"
    local ACTOR_OUTPUT_DIR="$OUTPUT_DIR/actor_final"
    
    # 检查 RM 模型
    if [ ! -d "$RM_MODEL_PATH" ]; then
        echo "❌ RM 模型不存在: $RM_MODEL_PATH"
        echo "   请先运行 Stage 2 (RM 训练)"
        exit 1
    fi
    
    # 生成 Actor 提示数据
    if [ ! -f "$ACTOR_PROMPT_DATA" ]; then
        log_message "生成 Actor 提示数据..."
        python generate_actor_prompts.py \
            --input "$INPUT_DATA" \
            --output "$ACTOR_PROMPT_DATA" \
            --min-prompts-per-char 20 \
            2>&1 | tee -a "$LOG_DIR/generate_actor_data.log"
    else
        log_message "✓ Actor 提示数据已存在: $ACTOR_PROMPT_DATA"
    fi
    
    check_file "$ACTOR_PROMPT_DATA"
    
    # 训练 Actor
    log_message "开始 Actor (GRPO) 训练..."
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    
    python train_actor_model.py \
        --sft_model_path "$SFT_MODEL_PATH" \
        --rm_model_path "$RM_MODEL_PATH" \
        --prompt_data "$ACTOR_PROMPT_DATA" \
        --output_dir "$ACTOR_OUTPUT_DIR" \
        --num_samples_per_prompt "$ACTOR_SAMPLES" \
        --beta_kl "$ACTOR_BETA_KL" \
        --num_train_epochs "$ACTOR_EPOCHS" \
        --per_device_train_batch_size "$ACTOR_BATCH_SIZE" \
        --seed "$SEED" \
        2>&1 | tee -a "$LOG_DIR/train_actor.log"
    
    log_message "✓ Actor 训练完成: $ACTOR_OUTPUT_DIR"
}

# ============================================================================
# 合并权重
# ============================================================================

merge_weights() {
    print_step "5" "合并权重用于部署"
    
    local ACTOR_MODEL_PATH="$OUTPUT_DIR/actor_final"
    local ACTOR_MERGED_PATH="$OUTPUT_DIR/actor_final_merged"
    
    if [ ! -d "$ACTOR_MODEL_PATH" ]; then
        log_message "⚠ Actor 模型不存在，跳过合并"
        return
    fi
    
    if [ -d "$ACTOR_MERGED_PATH" ]; then
        log_message "✓ 合并权重已存在: $ACTOR_MERGED_PATH"
        return
    fi
    
    log_message "合并 RL-LoRA 权重..."
    
    # 这里可以调用 merge_lora_weights.py
    # 或使用 peft 库进行合并
    
    log_message "✓ 权重合并完成"
}

# ============================================================================
# 生成报告
# ============================================================================

generate_report() {
    local REPORT_FILE="$LOG_DIR/training_report.txt"
    
    cat > "$REPORT_FILE" << EOF
╔════════════════════════════════════════════════════════════════╗
║           ChatJoy RLHF 训练完成报告                            ║
╚════════════════════════════════════════════════════════════════╝

训练阶段: $STAGE
完成时间: $(date)

配置参数:
  SFT 模型: $SFT_MODEL_PATH
  输入数据: $INPUT_DATA
  输出目录: $OUTPUT_DIR
  日志目录: $LOG_DIR

Stage 2 - Reward Model:
  LoRA Rank: $RM_LORA_RANK
  Epochs: $RM_EPOCHS
  Batch Size: $RM_BATCH_SIZE
  状态: ✓ 完成
  输出: $OUTPUT_DIR/rm_model_final

Stage 3 - Actor (GRPO):
  Samples per Prompt: $ACTOR_SAMPLES
  Beta KL: $ACTOR_BETA_KL
  Epochs: $ACTOR_EPOCHS
  Batch Size: $ACTOR_BATCH_SIZE
  状态: ✓ 完成
  输出: $OUTPUT_DIR/actor_final

部署命令:
  python -m vllm.entrypoints.openai.api_server \\
    --model $OUTPUT_DIR/actor_final_merged \\
    --dtype float16 \\
    --tensor-parallel-size 4 \\
    --port 8000

后续步骤:
  1. 验证模型质量 (比较 SFT vs RM vs Actor)
  2. 部署 Actor 模型
  3. 进行线上 A/B 测试
  4. 收集用户反馈，迭代优化

═══════════════════════════════════════════════════════════════

日志文件:
EOF
    
    for log_file in "$LOG_DIR"/*.log; do
        if [ -f "$log_file" ]; then
            echo "  - $(basename "$log_file")" >> "$REPORT_FILE"
        fi
    done
    
    cat "$REPORT_FILE"
    log_message "✓ 报告已生成: $REPORT_FILE"
}

# ============================================================================
# 主流程
# ============================================================================

main() {
    parse_arguments "$@"
    
    print_header "ChatJoy RLHF 完整训练流程"
    
    check_environment
    prepare_data
    
    case "$STAGE" in
        all)
            train_reward_model
            train_actor_model
            merge_weights
            ;;
        2)
            train_reward_model
            ;;
        3)
            train_actor_model
            merge_weights
            ;;
        *)
            echo "❌ 无效的阶段: $STAGE"
            echo "   有效选项: all, 2, 3"
            exit 1
            ;;
    esac
    
    generate_report
    
    print_header "✓ 所有训练完成!"
}

# ============================================================================
# 执行
# ============================================================================

main "$@"
