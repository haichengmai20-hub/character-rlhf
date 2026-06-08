#!/bin/bash

# --- 1. 显卡配置 ---
export CUDA_VISIBLE_DEVICES=4,5,6,7

# --- 2. 路径配置 ---
MODEL_PATH="/home/ps/.cache/modelscope/hub/models/Qwen/Qwen/Qwen3-30B-A3B-Instruct-2507"
DATASET_NAME="chatjoy_roleplay"
OUTPUT_DIR="saves/qwen3-30b-moe/chatjoy_v1"

# --- 3. 启动训练 ---
llamafactory-cli train \
    --stage sft \
    --do_train \
    --model_name_or_path "$MODEL_PATH" \
    --dataset "$DATASET_NAME" \
    --dataset_dir ../data \
    --template qwen \
    --finetuning_type lora \
    --quantization_bit 4 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --optim paged_adamw_32bit \
    --cutoff_len 2048 \
    --lora_target q_proj,v_proj,k_proj,o_proj \
    --lora_rank 32 \
    --lora_alpha 64 \
    --lora_dropout 0.1 \
    --output_dir "$OUTPUT_DIR" \
    --overwrite_output_dir \
    --plot_loss \
    --logging_steps 10 \
    --save_steps 500 \
    --save_total_limit 3 \
    --learning_rate 2e-4 \
    --num_train_epochs 5.0 \
    --warmup_ratio 0.1 \
    --fp16 \
    --ddp_timeout 18000000 \
    --gradient_checkpointing \
    --upcast_layernorm \
    --flash_attn fa2