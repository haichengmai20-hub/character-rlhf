#!/bin/bash
set -euo pipefail

# Reuse chatjoy LoRA workflow with safer defaults for Qwen3-30B on 32G GPUs.

ENV_NAME="tslib"
CODE_ROOT="/home/ps/hanjiaqi/LlamaFactory"
MODEL_PATH="/home/ps/.cache/modelscope/hub/models/Qwen/Qwen3-30B-A3B-Instruct-2507"
DATASET_NAME="chatjoy_trde_demo_0323"
DATASET_DIR="${CODE_ROOT}/data"
OUTPUT_DIR="/home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/qwen3-30b-moe/chatjoy_trde_demo_0323_qlora_z3"
DS_CONFIG="${CODE_ROOT}/examples/deepspeed/ds_z3_config.json"

if [[ ! -f "${DS_CONFIG}" ]]; then
  echo "DeepSpeed config not found: ${DS_CONFIG}"
  exit 1
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Model path not found: ${MODEL_PATH}"
  exit 1
fi

# Pick idle cards conservatively by memory footprint.
# For this machine, busy cards usually show >200 MiB memory usage.
IDLE_GPUS="$(python -c "import subprocess; rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True).strip().splitlines(); idle=[r.split(',')[0].strip() for r in rows if int(r.split(',')[1].strip())<=200]; print(','.join(idle))")"

if [[ -z "${IDLE_GPUS}" ]]; then
  echo "No idle GPUs found. Please free some GPUs and retry."
  exit 1
fi

NPROC="$(python -c "s='${IDLE_GPUS}'; print(len([x for x in s.split(',') if x]))")"

if [[ "${NPROC}" -lt 2 ]]; then
  echo "Only ${NPROC} idle GPU found (${IDLE_GPUS}). For 30B, at least 2 idle GPUs are recommended with ZeRO-3."
  exit 1
fi

echo "Using GPUs: ${IDLE_GPUS}"
echo "nproc_per_node: ${NPROC}"
echo "Output dir: ${OUTPUT_DIR}"

cd "${CODE_ROOT}"

CUDA_VISIBLE_DEVICES="${IDLE_GPUS}" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
NCCL_P2P_DISABLE=0 \
conda run -n "${ENV_NAME}" llamafactory-cli train \
  --stage sft \
  --do_train \
  --model_name_or_path "${MODEL_PATH}" \
  --dataset "${DATASET_NAME}" \
  --dataset_dir "${DATASET_DIR}" \
  --template qwen \
  --finetuning_type lora \
  --quantization_bit 4 \
  --deepspeed "${DS_CONFIG}" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --optim paged_adamw_32bit \
  --cutoff_len 2048 \
  --lora_target q_proj,v_proj,k_proj,o_proj \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --output_dir "${OUTPUT_DIR}" \
  --overwrite_output_dir \
  --plot_loss \
  --logging_steps 10 \
  --save_steps 200 \
  --save_total_limit 3 \
  --learning_rate 1e-4 \
  --num_train_epochs 2.0 \
  --warmup_ratio 0.1 \
  --fp16 \
  --ddp_timeout 18000000 \
  --gradient_checkpointing \
  --upcast_layernorm \
  --flash_attn fa2

echo "Training completed: ${OUTPUT_DIR}"
