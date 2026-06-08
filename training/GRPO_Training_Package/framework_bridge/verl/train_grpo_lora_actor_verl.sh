#!/bin/bash
set -euo pipefail

# verl actor training — ChatJoy Gemma4 GRPO
#
# 算法: GSPO（序列级重要性比，默认）
#   优势: 回复长度不均时序列等权；RM scalar reward 与序列级策略梯度更一致
#   adv_estimator 仍为 grpo（只改 policy loss，不改 advantage 估计方式）
#
# Actor 模型: Gemma4-26B-A4B SFT merged（bf16，含视觉编码器共 53.8GB）
#   路径: saves/gemma4-26b-a4b/sft2_ddp_lora_3gpu_567_cut3072_merged
#   大小: ~53.8GB（model-00001=52.1GB 文本解码器 + model-00002=1.7GB 视觉编码器）
#   架构: Gemma4ForConditionalGeneration (multimodal HF class)
#         vLLM 通过 gemma4_mm 模块处理；verl patch 的 Gemma4ForCausalLM 匹配 language_model 子模块
#   TP配置: 必须 ROLLOUT_TP_SIZE=4（53.8GB 无法放入单张 31.84GB GPU）
#
# RM: v5 merged（acc=95.5%, gap=3.497，当前最优）
#   路径: saves/qwen3-30b-moe/chatjoy_rm_v5_merged_vllm
#   架构: Qwen3MoeForSequenceClassification + num_labels=1 (vLLM 已 patch)
#
# 运行环境: conda activate gemma4-rl（verl 0.8.0.dev + vLLM 0.19.0）
# 前置检查: 确认 GPU 0-7 空闲（Demo 需先停；5090 实际显存 31.84GB，配置非常紧）
#
# Preferred flow:
# 1. Convert ChatJoy SFT/pair prompts to verl parquet with prepare_verl_rl_data.py
# 2. Run: bash train_grpo_lora_actor_verl.sh

PACKAGE_DIR="/home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/framework_bridge"
ANTI_REFUSAL_REWARD="${PACKAGE_DIR}/../anti_refusal_reward.py"
# v6.3 chain reward = anti_refusal + anti_repetition; verl 只能挂单一 custom_reward_function.path
ANTI_CHAIN_REWARD="${PACKAGE_DIR}/../anti_chain_reward.py"
# Whitelist for anti_repetition_reward; set to override per-character catchphrase list.
ANTI_REPETITION_WHITELIST_PATH="${ANTI_REPETITION_WHITELIST_PATH:-${PACKAGE_DIR}/../anti_repetition_whitelist.txt}"
export ANTI_REPETITION_WHITELIST_PATH
# Symmetric clip on raw RM score before it is added to custom_reward, enforced
# by the local patch in verl/experimental/reward_loop/reward_loop.py. v6.3 RM
# eval saw 22% chosen + 19% rejected with |reward|>5; without a clip, those
# outliers dominate GRPO group std and distort advantage normalization.
# Set to empty to disable. Both rm_score_raw and clipped rm_score are logged.
VERL_RM_REWARD_CLIP="${VERL_RM_REWARD_CLIP:-5.0}"
export VERL_RM_REWARD_CLIP
PROMPT_INPUT="${PROMPT_INPUT:-/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/runs/clean32_for_sft2.jsonl}"
RL_DATA_DIR="${RL_DATA_DIR:-${PACKAGE_DIR}/outputs/verl_rl_data}"
TRAIN_FILE="${TRAIN_FILE:-${RL_DATA_DIR}/train.parquet}"
VAL_FILE="${VAL_FILE:-${RL_DATA_DIR}/val.parquet}"

# Gemma4 SFT merged 模型（53.8GB bf16，含视觉编码器）
# 2026-04-22: 从原始 base 切换为 SFT merged — RL 应从 SFT 后的策略起步
# 生成方式: python3 GRPO_Training_Package/merge_gemma4_sft_lora.py
# 原始 adapter: saves/gemma4-26b-a4b/sft2_ddp_lora_3gpu_567_cut3072
MODEL_PATH="${MODEL_PATH:-/home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/gemma4-26b-a4b/sft2_ddp_lora_3gpu_567_cut3072_merged}"
SFT_LORA_ADAPTER="${SFT_LORA_ADAPTER:-}"
# RM merged 模型路径 — v6.3 RL 必须显式传 env（design §8.2）
# 历史默认值 saves/qwen3-30b-moe/chatjoy_rm_v5_merged_vllm/ 在本机不存在，会让 vLLM serve RM 直接报错
# v6.3 推荐: RM_MODEL_PATH=/root/hanjiaqi/LlamaFactory/chatjoy/saves/gemma4-26b-a4b/chatjoy_rm_gemma4_sft_v6.3_<run_id>_merged_vllm bash $0
# v6.2 baseline 回退: 先用 GRPO_Training_Package/merge_gemma4_rm_lora_v6_fg_for_vllm.py 现 merge
#   adapter: saves/gemma4-26b-a4b/chatjoy_rm_gemma4_sft_v6.2_fg_20260515_195216/
RM_MODEL_PATH="${RM_MODEL_PATH:-}"
if [[ -z "${RM_MODEL_PATH}" || ! -d "${RM_MODEL_PATH}" ]]; then
  echo "ERROR: RM_MODEL_PATH must be set to an existing merged-RM directory." >&2
  echo "       got: '${RM_MODEL_PATH}'" >&2
  echo "       example: RM_MODEL_PATH=/root/hanjiaqi/LlamaFactory/chatjoy/saves/gemma4-26b-a4b/chatjoy_rm_gemma4_sft_v6.3_<runid>_merged_vllm bash ${0}" >&2
  exit 2
fi
SAVE_DIR="${SAVE_DIR:-/home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/verl/chatjoy_grpo_lora}"
LOG_ROOT="${LOG_ROOT:-/home/ps/hanjiaqi/LlamaFactory/chatjoy/logs}"
PROJECT_NAME="${PROJECT_NAME:-chatjoy_verl}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-chatjoy_grpo_lora}"
LOGGER_BACKENDS="${LOGGER_BACKENDS:-[\"console\",\"tensorboard\",\"file\"]}"
VAL_GENERATIONS_TO_LOG="${VAL_GENERATIONS_TO_LOG:-16}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${LOG_ROOT}/tensorboard/${PROJECT_NAME}/${EXPERIMENT_NAME}}"
VERL_FILE_LOGGER_ROOT="${VERL_FILE_LOGGER_ROOT:-${LOG_ROOT}/verl_metrics}"
VALIDATION_SAMPLE_JSONL="${VALIDATION_SAMPLE_JSONL:-${LOG_ROOT}/validation_samples/${EXPERIMENT_NAME}.jsonl}"
export TENSORBOARD_DIR
export VERL_FILE_LOGGER_ROOT
export VERL_VALIDATION_SAMPLE_JSONL="${VALIDATION_SAMPLE_JSONL}"
# Module lazy loading only affects first-use latency and CUDA module residency;
# it does not change samples, rewards, losses, or model math. Do not set
# PYTORCH*_ALLOC_CONF=expandable_segments globally here: vLLM's CuMem memory
# pool rejects expandable_segments and fails during server startup.
for _alloc_conf_var in PYTORCH_ALLOC_CONF PYTORCH_CUDA_ALLOC_CONF; do
  _alloc_conf_value="${!_alloc_conf_var:-}"
  if [[ "${_alloc_conf_value}" == *expandable_segments:True* ]]; then
    echo "WARNING: unsetting ${_alloc_conf_var}=${_alloc_conf_value}; vLLM CuMem memory pool is incompatible with expandable_segments:True"
    unset "${_alloc_conf_var}"
  fi
done
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"

# N_GPUS=4: global_pool（actor+rollout）= 4 卡 + reward_pool = 4 卡，共 8 张（不超额）。
# 配合 reward.reward_model.enable_resource_pool=True + n_gpus_per_node=4。
# 物理 GPU 落点由启动时 CUDA_VISIBLE_DEVICES 的顺序决定：Ray 按序把 global_pool(actor)
# 分到前 4 个逻辑卡、reward_pool(RM) 分到后 4 个。当前项目约定(2026-06-05 起,见
# launch_rl_v6.4_defense.sh): CUDA_VISIBLE_DEVICES=4,5,6,7,0,1,2,3 → actor 落物理 GPU4-7、
# RM 落物理 GPU0-3,GPU7 无限制。不设该变量时回退默认 0-7 顺序(actor 0-3 / RM 4-7)。
N_GPUS="${N_GPUS:-4}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
# Validation batch size only changes validation chunking, not the validation set
# or training signal. Keep it lower to avoid end-of-run validation memory spikes.
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-32}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
LOGPROB_MICRO_BATCH_SIZE_PER_GPU="${LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
# Gemma4 text global attention uses global_head_dim=512. HF flash-attn
# fails on that path ("head dimension at most 256"), so the actor/ref FSDP
# side must use SDPA/eager for logprob and training. vLLM rollout keeps its
# own backend selection and is not affected by this setting.
HF_ATTN_IMPLEMENTATION="${HF_ATTN_IMPLEMENTATION:-sdpa}"
# vLLM rollout enforce_eager=True 禁用 CUDA graph，省 1-2GB/卡但 rollout 慢 30-50%。
# 5090 显存紧绷时保持 True；显存充裕时设 ROLLOUT_ENFORCE_EAGER=False 启用 CUDA graph
# 加速。验证内存稳定后再切换；CUDA graph 第一次构建可能撞 wake_up 致 OOM，需观察
# 第 1 次 update_weights 后的显存余量。
ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-True}"
# 长跑默认同步 checkpoint：verl 在 async_save=True 时会跳过
# latest_checkpointed_iteration.txt，导致自动 resume 可能落到旧 checkpoint。
# 需要极致吞吐的短 pilot 可显式 ACTOR_CKPT_ASYNC_SAVE=True。
ACTOR_CKPT_ASYNC_SAVE="${ACTOR_CKPT_ASYNC_SAVE:-False}"
# remove_padding=True feeds packed tokens without a full attention mask, which
# is only valid for the flash-attn varlen path. Disable it by default with SDPA.
if [[ -z "${USE_REMOVE_PADDING:-}" ]]; then
  if [[ "${HF_ATTN_IMPLEMENTATION}" == flash_attention* ]]; then
    USE_REMOVE_PADDING=True
  else
    USE_REMOVE_PADDING=False
  fi
fi
# q/k/v/o LoRA rank8 在 32GB 卡上 actor backward 峰值会贴近上限；
# activation offload 不改变训练样本或 loss，只用额外 CPU 传输换反向传播显存余量。
ACTOR_ENABLE_ACTIVATION_OFFLOAD="${ACTOR_ENABLE_ACTIVATION_OFFLOAD:-True}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-768}"
# 动态 token-level batching：按 token 上限打包，padding 浪费从 30-50% 降到 <10%，峰值显存更稳。
# 默认 = MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH = 2816；不再 +256 buffer。
# 历史记录（2026-04-30 §23）：3072 时 step 7 actor backward OOM，因为 vLLM sleep 残留
# 4.77GB + FSDP backward 23.96GB 已经 28.7GB，2816 vs 3072 把 SDPA attention activation
# 峰值降 ~16%（O(N²)），实测最长样本 2779 tokens，2816 仍能装下，安全。
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"
# Keep rollout context just large enough for training (prompt + response),
# which avoids oversized KV-cache reservation from model max_position_embeddings.
# +256 buffer gives the actor rollout path room for chat-template tokens while
# keeping KV cache reservation tight on 32GB cards.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 256))}"
# RM uses a different tokenizer/template from the Gemma actor. Keep its context
# budget separate so rare prompt+response samples do not exceed /classify by a
# few tokens under the Qwen RM tokenizer.
RM_MAX_MODEL_LEN="${RM_MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 512))}"
# Cap vLLM scheduler to the actual training context budget instead of the
# permissive verl defaults (8192 tokens / 1024 seqs), which over-profiles
# activation memory and can leave zero KV cache blocks on 32GB cards.
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-${MAX_MODEL_LEN}}"
RM_MAX_NUM_BATCHED_TOKENS="${RM_MAX_NUM_BATCHED_TOKENS:-${RM_MAX_MODEL_LEN}}"
# Concurrency cap only affects throughput; quality is unchanged because
# requests beyond this limit are queued instead of dropped.
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-32}"
RM_MAX_NUM_SEQS="${RM_MAX_NUM_SEQS:-64}"
ROLLOUT_N="${ROLLOUT_N:-4}"
# ROLLOUT_TP_SIZE=4 是必须的：Gemma4 SFT merged 53.8GB ÷ 4 = 13.45GB/卡，单卡 31.84GB 放不下
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-4}"
RM_TP_SIZE="${RM_TP_SIZE:-4}"
# RTX 5090 实际显存: 32607 MiB = 31.84 GB（非 32.6GB）
# ROLLOUT: Gemma4 MM + LoRA 在 vLLM profile 阶段的非 KV 占用明显高于纯文本模型；
# 0.64 在 5090 32GB 上仍会出现 "No available memory for the cache blocks"。
# 将默认值提升到 0.78，确保至少能分到可用 KV block；仍可通过环境变量下调。
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.78}"
# RM: 0.62 × 31.84 = 19.74 GB; Qwen3-30B 57GB÷4 = 14.25 GB; KV cache 5.49 GB ✓
RM_GPU_MEMORY_UTILIZATION="${RM_GPU_MEMORY_UTILIZATION:-0.62}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-16}"
# Gemma4 + vLLM 0.19 adapter hot-swap currently has stability issues on some
# non-attention projections during TP update. Keep a safe default for RL smoke
# tests; users can still override this env var for broader coverage.
TARGET_MODULES="${TARGET_MODULES:-[q_proj,k_proj,v_proj,o_proj]}"
# LORA_MERGE_MODE：合并 LoRA 推全量权重（绕过 add_lora 崩溃）
# 2026-04-23 结论：merge 模式在 5090(31.36GB) × 26B MoE (128 experts) 不可行
#   FSDP1: summon_full_params 需要 51.7GB/rank → OOM
#   FSDP2: fsdp2_load_full_state_dict 要求 rank0 持有全量 52GB 到 GPU → OOM/hang
#   + Gemma4 有 128 experts，FSDP2 wrap 遍历 ~35 万 submodules 极慢
# → 只能回到 hot-swap 模式（LORA_MERGE_MODE=False），并强化 set_lora 同步保护
LORA_MERGE_MODE="${LORA_MERGE_MODE:-False}"
LR="${LR:-3e-6}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
LR_WARMUP_STEPS_RATIO="${LR_WARMUP_STEPS_RATIO:-0.03}"
# ── MoE 负载均衡 ──────────────────────────────────────────────────────────────
ENTROPY_COEFF="${ENTROPY_COEFF:-0.001}"      # 温和探索系数，防止 MoE router 坍塌；entropy 计算只增 log_prob 开销，不增 weight 显存
MOE_AUX_LOSS_COEF="${MOE_AUX_LOSS_COEF:-0.001}"
# ── GSPO 专项参数 ─────────────────────────────────────────────────────────────
CLIP_RATIO_LOW="${CLIP_RATIO_LOW:-0.2}"      # 下界收紧，防止过度抑制好的更新
CLIP_RATIO_HIGH="${CLIP_RATIO_HIGH:-0.28}"   # 上界略宽（非对称 clip，DAPO 思路）
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
TEST_FREQ="${TEST_FREQ:-50}"
IS_SMOKE_RUN=0
case "${EXPERIMENT_NAME} ${SAVE_DIR}" in
  *smoke*|*Smoke*|*SMOKE*) IS_SMOKE_RUN=1 ;;
esac
if [[ -z "${SAVE_FREQ+x}" ]]; then
  if [[ "${IS_SMOKE_RUN}" == 1 ]]; then
    SAVE_FREQ=0
  else
    SAVE_FREQ=50
  fi
fi
if [[ -z "${MAX_ACTOR_CKPT_TO_KEEP+x}" ]]; then
  if [[ "${IS_SMOKE_RUN}" == 1 ]]; then
    MAX_ACTOR_CKPT_TO_KEEP=0
  else
    MAX_ACTOR_CKPT_TO_KEEP=1
  fi
fi

if [[ "${SKIP_PREPARE_VERL_DATA:-0}" == "1" ]]; then
  echo "SKIP_PREPARE_VERL_DATA=1: using existing TRAIN_FILE/VAL_FILE without regenerating parquet"
else
  python3 "${PACKAGE_DIR}/prepare_verl_rl_data.py"   --input "${PROMPT_INPUT}"   --train-output "${TRAIN_FILE}"   --val-output "${VAL_FILE}"
fi

ARGS=(
  # ── 算法：GSPO（序列级重要性比）+ GRPO advantage 估计 ──────────────────────
  algorithm.adv_estimator=grpo
  "actor_rollout_ref.actor.policy_loss.loss_mode=gspo"
  "actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean"
  "actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW}"
  "actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH}"
  # ─────────────────────────────────────────────────────────────────────────────
  trainer.val_before_train=False
  data.train_files="${TRAIN_FILE}"
  data.val_files="${VAL_FILE}"
  data.prompt_key=prompt
  data.return_raw_chat=True
  data.train_batch_size="${TRAIN_BATCH_SIZE}"
  data.val_batch_size="${VAL_BATCH_SIZE}"
  data.max_prompt_length="${MAX_PROMPT_LENGTH}"
  data.max_response_length="${MAX_RESPONSE_LENGTH}"
  data.filter_overlong_prompts=True
  data.truncation='error'
  data.shuffle="${DATA_SHUFFLE:-False}"
  actor_rollout_ref.model.path="${MODEL_PATH}"
  "+actor_rollout_ref.model.override_config.attn_implementation=${HF_ATTN_IMPLEMENTATION}"
  actor_rollout_ref.actor.optim.lr="${LR}"
  actor_rollout_ref.actor.optim.lr_scheduler_type="${LR_SCHEDULER_TYPE}"
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio="${LR_WARMUP_STEPS_RATIO}"
  actor_rollout_ref.model.use_remove_padding="${USE_REMOVE_PADDING}"
  actor_rollout_ref.model.enable_gradient_checkpointing=True
  actor_rollout_ref.model.enable_activation_offload="${ACTOR_ENABLE_ACTIVATION_OFFLOAD}"
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}"
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}"
  # 动态序列打包：质量无损，峰值显存更稳定
  actor_rollout_ref.actor.use_dynamic_bsz=True
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}"
  actor_rollout_ref.actor.use_kl_loss=True
  actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF:-0.001}"
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  # entropy_coeff > 0 防止 MoE router 坍塌为单一专家（原值 0 有坍塌风险）
  actor_rollout_ref.actor.entropy_coeff="${ENTROPY_COEFF}"
  # param_offload=True：rollout 期间 actor FSDP 卸载到 CPU，避免与 vLLM 共存撑爆 31.84GB
  # 代价：每个 rollout/trainer 切换多一次 CPU↔GPU 拷贝（~2-3s/step）
  # 若关闭则 rollout 峰值 = 12.15 (FSDP) + 19.1 (vLLM) = 31.25 GB，余量仅 0.6 GB → 必 OOM
  actor_rollout_ref.actor.fsdp_config.param_offload=True
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
  # model_dtype 默认 fp32：会把 bf16 权重翻倍成 103GB 总量，FSDP 每卡 shard ~26GB → 必 OOM
  # 用 bfloat16 加载：每卡 shard ~6.5GB，留出 vLLM(19GB) + 激活空间
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16
  # 保留 FSDP1（默认）：FSDP2 对 128-experts MoE 遍历 ~35 万 submodules 极慢，
  #   且 fsdp2_load_full_state_dict 要求 rank0 GPU 持有全量 52GB（5090 不够）
  # hot-swap 模式（lora.merge=False）FSDP1 够用：shard 13GB + vLLM 13GB 能共存
  # Gemma4ForConditionalGeneration._no_split_modules 包含 Gemma4AudioLayer（textonly 模型不存在）
  # 显式覆盖 wrap_policy，只包含实际存在的语言模型层，跳过 AudioLayer 导致的 FSDP 报错
  "+actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[Gemma4TextDecoderLayer]"
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOGPROB_MICRO_BATCH_SIZE_PER_GPU}"
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP_SIZE}"
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}"
  actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}"
  actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS}"
  actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}"
  actor_rollout_ref.rollout.n="${ROLLOUT_N}"
  actor_rollout_ref.rollout.load_format=safetensors
  actor_rollout_ref.rollout.layered_summon=True
  # 显存优化（零质量影响）：
  #   enforce_eager=True → 禁用 CUDA graph，省 1-2GB/卡，rollout 慢 30-50%；显存
  #     充裕时设 ROLLOUT_ENFORCE_EAGER=False 启用 graph 加速
  #   free_cache_engine=True → rollout 结束后释放 vLLM KV cache，让 FSDP 能重新 on-load
  actor_rollout_ref.rollout.enforce_eager="${ROLLOUT_ENFORCE_EAGER}"
  actor_rollout_ref.rollout.free_cache_engine=True
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOGPROB_MICRO_BATCH_SIZE_PER_GPU}"
  actor_rollout_ref.ref.fsdp_config.param_offload=True
  actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16
  "+actor_rollout_ref.ref.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[Gemma4TextDecoderLayer]"
  reward.num_workers=1
  # ── RM 状态：已启用（vLLM 0.19 已 patch Qwen3MoeForSequenceClassification）──────
  # v3 adapter 已通过 merge_rm_lora_for_vllm.py 合并到 chatjoy_rm_v3_merged_vllm/
  # 若 merged 模型尚未生成（OUTPUT 目录不存在），先运行:
  #   conda run -n gemma4-rl python3 GRPO_Training_Package/merge_rm_lora_for_vllm.py
  # 如需临时禁用 RM（调试 actor only），将下行改为 False
  reward.reward_model.enable=True
  reward.reward_model.model_path="${RM_MODEL_PATH}"
  # v6.3 链式 reward = anti_refusal + anti_repetition；verl 只支持单一 custom_reward_function.path
  # 所以套一层 anti_chain_reward.py 同时调用两者，score 直接相加，保留两组 metadata 字段。
  # 1) anti_refusal：关键词拒绝 -1.0，短回复无 roleplay 格式 -0.25。
  # 2) anti_repetition：n-gram 重复检测，命中 -0.2 ~ -0.8（依 repeat_ratio），白名单文件 ANTI_REPETITION_WHITELIST_PATH 排除合法口头禅。
  # total_reward = RM score + anti_refusal_penalty + anti_repetition_penalty；KL 只作为 actor loss 项。
  reward.custom_reward_function.path="${ANTI_CHAIN_REWARD}"
  reward.custom_reward_function.name=compute_reward
  reward.reward_model.rollout.name=vllm
  reward.reward_model.rollout.tensor_model_parallel_size="${RM_TP_SIZE}"
  reward.reward_model.rollout.gpu_memory_utilization="${RM_GPU_MEMORY_UTILIZATION}"
  reward.reward_model.rollout.max_model_len="${RM_MAX_MODEL_LEN}"
  reward.reward_model.rollout.max_num_batched_tokens="${RM_MAX_NUM_BATCHED_TOKENS}"
  reward.reward_model.rollout.max_num_seqs="${RM_MAX_NUM_SEQS}"
  reward.reward_model.rollout.prompt_length="${MAX_PROMPT_LENGTH}"
  reward.reward_model.rollout.response_length="${MAX_RESPONSE_LENGTH}"
  reward.reward_model.rollout.enforce_eager=True
  # ── RM 必须保持唤醒 ─────────────────────────────────────────────────────────────
  # RewardModelManager.__init__ 在结尾若 free_cache_engine=True 会立刻 self.sleep()
  # （reward_model.py:47-48）。但当前 ChatJoy 用 async agent reward loop（启用
  # enable_resource_pool=True，走 RewardLoopWorker.compute_score → /classify 路径），
  # 这条路径不会先 wake_up RM。结果：RM 启动后永远 sleep，任何 /classify 请求都
  # 永久挂起（aiohttp total=None 不超时），整个训练 Training Progress: 0/N 死锁。
  # RM 在 GPU 4-7 与 actor (GPU 0-3) 不抢显存，让 RM 一直 awake (~19GB) 没问题。
  reward.reward_model.rollout.free_cache_engine=False
  # ── resource pool：global_pool(actor/rollout) GPU 0-3，reward_pool GPU 4-7 ─────
  # 启用 RM 时才有效；当 enable=False 时这些参数被忽略
  reward.reward_model.enable_resource_pool=True
  reward.reward_model.n_gpus_per_node=4
  reward.reward_model.nnodes=1
  algorithm.use_kl_in_reward=False
  trainer.critic_warmup=0
  trainer.logger="${LOGGER_BACKENDS}"
  trainer.project_name="${PROJECT_NAME}"
  trainer.experiment_name="${EXPERIMENT_NAME}"
  trainer.log_val_generations="${VAL_GENERATIONS_TO_LOG}"
  "+trainer.validation_sample_jsonl=${VALIDATION_SAMPLE_JSONL}"
  trainer.n_gpus_per_node="${N_GPUS}"
  trainer.nnodes=1
  trainer.save_freq="${SAVE_FREQ}"
  trainer.test_freq="${TEST_FREQ}"
  trainer.max_actor_ckpt_to_keep="${MAX_ACTOR_CKPT_TO_KEEP}"
  trainer.total_epochs="${TOTAL_EPOCHS}"
  trainer.default_local_dir="${SAVE_DIR}"
  # 异步 checkpoint：FSDP shard 写盘 ~50s 推后台，主循环不阻塞
  actor_rollout_ref.actor.checkpoint.async_save="${ACTOR_CKPT_ASYNC_SAVE}"
)

if [[ -n "${SFT_LORA_ADAPTER}" ]]; then
  ARGS+=(actor_rollout_ref.model.lora_adapter_path="${SFT_LORA_ADAPTER}")
else
  # target_modules:
  # 默认使用 attention-only（q/k/v/o）以规避 Gemma4 + vLLM adapter 热更新在 TP 下的
  # 非 attention 投影崩溃；可用环境变量 TARGET_MODULES 覆盖为更宽列表。
  ARGS+=(
    actor_rollout_ref.model.lora_rank="${LORA_RANK}"
    actor_rollout_ref.model.lora_alpha="${LORA_ALPHA}"
    "actor_rollout_ref.model.target_modules=${TARGET_MODULES}"
    # Gemma4ForConditionalGeneration 含视觉编码器（SigLIP），其 q_proj/k_proj 是 Gemma4ClippableLinear
    # PEFT 不支持该类型，用 exclude_modules 排除所有 vision_tower 下的 key（regex fullmatch）
    "actor_rollout_ref.model.exclude_modules=.*vision_tower.*"
    # merge 模式（+lora.merge）：关闭 vLLM add_lora/set_lora 热更新路径
    "+actor_rollout_ref.model.lora.merge=${LORA_MERGE_MODE}"
  )
fi

python3 -m verl.trainer.main_ppo "${ARGS[@]}" "$@"
