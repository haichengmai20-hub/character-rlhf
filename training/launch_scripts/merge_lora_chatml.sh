#!/bin/bash
set -euo pipefail

# Merge LoRA adapter into base model for vLLM inference.
BASE_MODEL_PATH="/home/ps/.cache/modelscope/hub/models/Qwen/Qwen/Qwen3-30B-A3B-Instruct-2507"
LORA_PATH="/home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/qwen3-30b-moe/chatjoy_hq_full_chatml_lora"
EXPORT_DIR="/home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/qwen3-30b-moe/chatjoy_hq_full_chatml_merged"

llamafactory-cli export \
    --model_name_or_path "$BASE_MODEL_PATH" \
    --adapter_name_or_path "$LORA_PATH" \
    --template chatml \
    --finetuning_type lora \
    --export_dir "$EXPORT_DIR" \
    --export_size 5 \
    --export_device cpu \
    --export_legacy_format False

echo "Merge finished: $EXPORT_DIR"
