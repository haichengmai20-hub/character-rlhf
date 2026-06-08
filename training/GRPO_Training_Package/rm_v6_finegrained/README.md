# ChatJoy RM v6 细粒度数据流水线

这个目录实现 RM v6 fine-grained 数据流水线，目标是：

- 完整保留 `rm_train_v5_5pct_badpool.jsonl` 作为稳定 anchor；
- 新增“同 prompt 多候选”的细粒度排序数据；
- 训练新的 Gemma4 SFT-base RM；
- 不覆盖、不修改现有 v5 RM、Gemma4 RM v5、RL 稳定训练脚本和已有 checkpoint。

默认先跑 `pilot`，验证数据质量、RM 离线效果和 RL smoke 后，再决定是否扩大到 `full`。

## 1. 抽样 Prompts

从 RL 训练 parquet 中抽样 prompt，并排除 validation/eval 中出现过的 prompt，避免评估泄漏。

```bash
cd /home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/rm_v6_finegrained
source /home/ps/miniconda3/etc/profile.d/conda.sh
conda activate gemma4-rl

python sample_prompts.py \
  --mode pilot \
  --output /home/ps/hanjiaqi/LlamaFactory/chatjoy/data/runs/rm_v6_finegrained/pilot/prompts.jsonl
```

`pilot` 默认抽 2,500 个 unique prompt；`full` 默认抽 10,000 个 unique prompt：

```bash
python sample_prompts.py \
  --mode full \
  --output /home/ps/hanjiaqi/LlamaFactory/chatjoy/data/runs/rm_v6_finegrained/full/prompts.jsonl
```

输出字段包括 `prompt_id`、`messages`、`language`、`role_hash`、`source_index`。

## 2. 导出 RL Step 550

RL step550 需要先从 verl FSDP checkpoint 导出成 HF 模型，之后才能作为 vLLM candidate source 使用。

只需要执行一次：

```bash
cd /home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/rm_v6_finegrained
bash export_step550_hf.sh
```

导出目录：

```text
/home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/gemma4-26b-a4b/chatjoy_grpo_gemma4rm_phase2_step550_hf/
```

## 3. 逐 Source 生成候选

一次只启动一个 vLLM server，一次只生成一个 source，生成完成后关闭当前 server，再启动下一个 source。这样可以避免多个大模型同时占用显存，影响当前 RL/RM 链路稳定性。

候选统一写到：

```text
/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/runs/rm_v6_finegrained/{pilot|full}/
```

每条候选会记录 `candidate_id`、`prompt_id`、`source_model`、`sampling`、`response`、`tokens/chars`、`finish_reason` 等字段。

如果要按既定顺序自动跑完整个 pilot，可以使用：

```bash
cd /home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/rm_v6_finegrained
source /home/ps/miniconda3/etc/profile.d/conda.sh
conda activate gemma4-rl

RUN_SPLIT=pilot GPU_LIST=0,1,2,3 bash generate_pilot_sources_sequential.sh
```

该脚本会按顺序执行：

1. Gemma4 SFT：`gemma4_sft_t07_p08`
2. Gemma4 SFT：`gemma4_sft_t10_p095`
3. Qwen SFT：`qwen_sft_t07_p08`
4. RL step550：`rl_step550_t07_p08`
5. RL step550：`rl_step550_t10_p095`

server 日志写到：

```text
/home/ps/hanjiaqi/LlamaFactory/chatjoy/logs/rm_v6_finegrained/
```

如果只想单独 resume 某一组 source，可以设置 `RUN_GROUPS`：

```bash
RUN_SPLIT=pilot RUN_GROUPS=rl_step550 GPU_LIST=0,1,2,3 bash generate_pilot_sources_sequential.sh
```

下面是手动分 source 执行方式，便于单独排错或 resume。

### 3.1 Gemma4 SFT base，temperature=0.7，top_p=0.8

启动 vLLM：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
  --model /home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/gemma4-26b-a4b/sft2_ddp_lora_3gpu_567_cut3072_merged \
  --served-model-name gemma4-sft \
  --host 127.0.0.1 \
  --port 8001 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.88 \
  --max-model-len 3200
```

生成候选：

```bash
python generate_candidates_openai.py \
  --prompts /home/ps/hanjiaqi/LlamaFactory/chatjoy/data/runs/rm_v6_finegrained/pilot/prompts.jsonl \
  --output /home/ps/hanjiaqi/LlamaFactory/chatjoy/data/runs/rm_v6_finegrained/pilot/candidates.gemma4_sft_t07.jsonl \
  --base-url http://127.0.0.1:8001 \
  --model gemma4-sft \
  --source-label gemma4_sft_t07_p08 \
  --temperature 0.7 \
  --top-p 0.8 \
  --resume
```

### 3.2 Gemma4 SFT diverse，temperature=1.0，top_p=0.95

继续使用同一个 Gemma4 SFT vLLM server，生成第二个 source：

```bash
python generate_candidates_openai.py \
  --prompts /home/ps/hanjiaqi/LlamaFactory/chatjoy/data/runs/rm_v6_finegrained/pilot/prompts.jsonl \
  --output /home/ps/hanjiaqi/LlamaFactory/chatjoy/data/runs/rm_v6_finegrained/pilot/candidates.gemma4_sft_t10.jsonl \
  --base-url http://127.0.0.1:8001 \
  --model gemma4-sft \
  --source-label gemma4_sft_t10_p095 \
  --temperature 1.0 \
  --top-p 0.95 \
  --resume
```

### 3.3 Qwen SFT base，temperature=0.7，top_p=0.8

关闭 Gemma4 SFT server 后，启动 Qwen SFT server：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
  --model /home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/qwen3-30b-moe/clean32_for_sft2_fsdp_qlora_merged \
  --served-model-name qwen-sft \
  --host 127.0.0.1 \
  --port 8001 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.88 \
  --max-model-len 3200
```

生成候选：

```bash
python generate_candidates_openai.py \
  --prompts /home/ps/hanjiaqi/LlamaFactory/chatjoy/data/runs/rm_v6_finegrained/pilot/prompts.jsonl \
  --output /home/ps/hanjiaqi/LlamaFactory/chatjoy/data/runs/rm_v6_finegrained/pilot/candidates.qwen_sft_t07.jsonl \
  --base-url http://127.0.0.1:8001 \
  --model qwen-sft \
  --source-label qwen_sft_t07_p08 \
  --temperature 0.7 \
  --top-p 0.8 \
  --resume
```

### 3.4 RL step550 base，temperature=0.7，top_p=0.8

关闭 Qwen SFT server 后，启动 RL step550 server。

注意：`legacy_model_merger.py` 导出的 full HF Gemma4 step550 可能会把 `layer_scalar` 从 `[1]` 错导成 `[4]`，导致 vLLM 直接加载时报 shape mismatch。当前脚本会先用 `fix_gemma4_layer_scalar_safetensors.py` 生成 vLLM 修复版：

```text
/home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/gemma4-26b-a4b/chatjoy_grpo_gemma4rm_phase2_step550_hf_vllm_fixed/
```

这个修复只改 `*.layer_scalar` 的导出形状，不改原始 FSDP checkpoint，也不影响 RL 稳定训练脚本。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
  --model /home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/gemma4-26b-a4b/chatjoy_grpo_gemma4rm_phase2_step550_hf_vllm_fixed \
  --served-model-name gemma4-rl-step550 \
  --host 127.0.0.1 \
  --port 8001 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.88 \
  --max-model-len 3200
```

生成候选：

```bash
python generate_candidates_openai.py \
  --prompts /home/ps/hanjiaqi/LlamaFactory/chatjoy/data/runs/rm_v6_finegrained/pilot/prompts.jsonl \
  --output /home/ps/hanjiaqi/LlamaFactory/chatjoy/data/runs/rm_v6_finegrained/pilot/candidates.rl550_t07.jsonl \
  --base-url http://127.0.0.1:8001 \
  --model gemma4-rl-step550 \
  --source-label rl_step550_t07_p08 \
  --temperature 0.7 \
  --top-p 0.8 \
  --resume
```

### 3.5 RL step550 diverse，temperature=1.0，top_p=0.95

继续使用同一个 RL step550 vLLM server，生成第二个 RL source：

```bash
python generate_candidates_openai.py \
  --prompts /home/ps/hanjiaqi/LlamaFactory/chatjoy/data/runs/rm_v6_finegrained/pilot/prompts.jsonl \
  --output /home/ps/hanjiaqi/LlamaFactory/chatjoy/data/runs/rm_v6_finegrained/pilot/candidates.rl550_t10.jsonl \
  --base-url http://127.0.0.1:8001 \
  --model gemma4-rl-step550 \
  --source-label rl_step550_t10_p095 \
  --temperature 1.0 \
  --top-p 0.95 \
  --resume
```

## 4. Judge 并构造 Fine Pairs

judge 时不会使用 source 信息做偏置，只根据统一 ChatJoy rubric 对同 prompt 候选排序。低置信度、解析失败、blocked 样本不会进入训练，只写入审计文件。

```bash
python judge_candidates.py \
  --candidates /home/ps/hanjiaqi/LlamaFactory/chatjoy/data/runs/rm_v6_finegrained/pilot/candidates.gemma4_sft_t07.jsonl \
  --candidates /home/ps/hanjiaqi/LlamaFactory/chatjoy/data/runs/rm_v6_finegrained/pilot/candidates.gemma4_sft_t10.jsonl \
  --candidates /home/ps/hanjiaqi/LlamaFactory/chatjoy/data/runs/rm_v6_finegrained/pilot/candidates.qwen_sft_t07.jsonl \
  --candidates /home/ps/hanjiaqi/LlamaFactory/chatjoy/data/runs/rm_v6_finegrained/pilot/candidates.rl550_t07.jsonl \
  --candidates /home/ps/hanjiaqi/LlamaFactory/chatjoy/data/runs/rm_v6_finegrained/pilot/candidates.rl550_t10.jsonl \
  --output /home/ps/hanjiaqi/LlamaFactory/chatjoy/data/runs/rm_v6_finegrained/pilot/judged.jsonl \
  --audit-output /home/ps/hanjiaqi/LlamaFactory/chatjoy/data/runs/rm_v6_finegrained/pilot/judge_audit.jsonl

python build_pairs.py \
  --judged /home/ps/hanjiaqi/LlamaFactory/chatjoy/data/runs/rm_v6_finegrained/pilot/judged.jsonl \
  --output /home/ps/hanjiaqi/LlamaFactory/chatjoy/data/openrlhf_rm/rm_v6_finegrained_pairs_pilot.jsonl
```

## 5. 混合 v5 Anchor

v5 数据全量并入，不删除、不降权、不重排。v5 样本只加 `extra_info.label_source=v5_anchor` 等标记，训练仍使用统一 pairwise ranking loss。

```bash
python mix_with_v5_anchor.py \
  --fine-pairs /home/ps/hanjiaqi/LlamaFactory/chatjoy/data/openrlhf_rm/rm_v6_finegrained_pairs_pilot.jsonl \
  --output-train /home/ps/hanjiaqi/LlamaFactory/chatjoy/data/openrlhf_rm/rm_train_v6_fg_pilot.jsonl \
  --output-eval /home/ps/hanjiaqi/LlamaFactory/chatjoy/data/openrlhf_rm/rm_eval_v6_fg.jsonl
```

QA：

```bash
python qa_dataset.py \
  --judged /home/ps/hanjiaqi/LlamaFactory/chatjoy/data/runs/rm_v6_finegrained/pilot/judged.jsonl \
  --pairs /home/ps/hanjiaqi/LlamaFactory/chatjoy/data/openrlhf_rm/rm_v6_finegrained_pairs_pilot.jsonl \
  --mixed-train /home/ps/hanjiaqi/LlamaFactory/chatjoy/data/openrlhf_rm/rm_train_v6_fg_pilot.jsonl
```

核心检查目标：

- 每个 prompt 至少 4 个有效候选；
- 候选 source 分布接近均匀；
- low-confidence/blocked 占比不能过高；
- fine pair 中同 source pair 占比不能过高；
- v5 anchor 全量保留。

## 6. 训练 RM v6 Pilot

```bash
cd /home/ps/hanjiaqi/LlamaFactory
SPLIT=pilot bash chatjoy/launch_rm_train_gemma4_sft_v6_fg.sh
```

训练输出目录格式：

```text
/home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/gemma4-26b-a4b/chatjoy_rm_gemma4_sft_v6_fg_pilot_<timestamp>/
```

merge LoRA adapter 到 vLLM 可加载目录：

```bash
python /home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/merge_gemma4_rm_lora_v6_fg_for_vllm.py \
  --adapter /home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/gemma4-26b-a4b/chatjoy_rm_gemma4_sft_v6_fg_pilot_<timestamp> \
  --output /home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/gemma4-26b-a4b/chatjoy_rm_gemma4_sft_v6_fg_pilot_merged_vllm
```

## 7. 后续验证

pilot RM merge 后，先做 classify smoke，确认：

- vLLM 能加载；
- score 是 finite；
- EOS/rstrip 逻辑一致；
- 不出现 classify 长度越界。

然后再跑 30-step RL smoke，只替换 RM path，不改 actor/RL 其他稳定参数。smoke 通过后，再决定是否训练 full RM 和正式 RL。
