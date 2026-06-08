# ChatJoy Framework Bridge

这套桥接脚本只放在 [GRPO_Training_Package](/home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package) 里面，目标是把 ChatJoy 现有 demo 数据接到：

- OpenRLHF 的 pair / RM 训练链路
- verl 的 online actor LoRA / GRPO 链路

## 目录

```text
framework_bridge/
├── common.py
├── prepare_openrlhf_rm_data.py
├── prepare_openrlhf_prompt_data.py
├── prepare_verl_rm_data.py
├── prepare_verl_rl_data.py
├── openrlhf/
│   ├── train_rm_openrlhf.sh
│   └── serve_rm_openrlhf.sh
└── verl/
    └── train_grpo_lora_actor_verl.sh
```

## 当前输入兼容范围

这些脚本都兼容以下几类输入：

1. ChatJoy SFT 原始 jsonl
   - 例如 [clean32_for_sft2.jsonl](/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/runs/clean32_for_sft2.jsonl)
   - 字段: `instruction / output / language / system`

2. ChatJoy 嵌套 `item` 格式
   - 例如 [cleaned.issue_removed.lang19.merged_recoverable_787_v1.jsonl](/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/runs/20260331_114330_gemini_41069_57197_backup/cleaned.issue_removed.lang19.merged_recoverable_787_v1.jsonl)
   - 外层是 `index + item`

3. 已经生成好的 pair jsonl
   - 例如 [rejected_pairs_from_sft_test20.jsonl](/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/runs/rejected_sft_sample20_20260331_155727/rejected_pairs_from_sft_test20.jsonl)
   - 字段: `prompt / system / chosen / rejected / language / rejected_type`

## 脚本职责

### 1. OpenRLHF RM 数据

用 `prepare_openrlhf_rm_data.py` 把 ChatJoy pair 数据转成 OpenRLHF 偏好的消息格式：

```bash
python prepare_openrlhf_rm_data.py   --input /path/to/pairs.jsonl   --train-output /path/to/train.jsonl   --val-output /path/to/val.jsonl
```

输出中的核心字段是：

```json
{
  "prompt": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "chosen": [{"role": "assistant", "content": "..."}],
  "rejected": [{"role": "assistant", "content": "..."}]
}
```

然后直接用 [openrlhf/train_rm_openrlhf.sh](/home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/framework_bridge/openrlhf/train_rm_openrlhf.sh) 开 RM。这个脚本已经补上 `--prompt_key prompt` 和 `--value_head_prefix score`，更适合后续给 verl 侧挂 RM。

### 2. OpenRLHF Prompt 数据

用 `prepare_openrlhf_prompt_data.py` 把 SFT 或 pair 数据转成在线 RL prompt：

```bash
python prepare_openrlhf_prompt_data.py   --input /path/to/clean32_for_sft2.jsonl   --train-output /path/to/train.jsonl   --val-output /path/to/val.jsonl
```

输出字段是 `context_messages`。

### 3. verl RM 数据

用 `prepare_verl_rm_data.py` 生成 parquet 版 pair 数据，方便以后如果你要在 verl 侧做自定义 RM trainer：

```bash
python prepare_verl_rm_data.py   --input /path/to/pairs.jsonl   --train-output /path/to/train.parquet   --val-output /path/to/val.parquet
```

### 4. verl RL 数据

用 `prepare_verl_rl_data.py` 把 SFT 或 pair prompt 转成 verl `main_ppo` 常见的 parquet 结构：

```bash
python prepare_verl_rl_data.py   --input /path/to/clean32_for_sft2.jsonl   --train-output /path/to/train.parquet   --val-output /path/to/val.parquet
```

输出里最重要的是：

```json
{
  "data_source": "chatjoy",
  "prompt": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "ability": "chatjoy_alignment",
  "reward_model": {"style": "model", "ground_truth": ""},
  "extra_info": {"index": 123}
}
```

## 推荐流程

### A. 先训 RM

1. 用 [build_rejected_from_sft.py](/home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/build_rejected_from_sft.py) 产大规模 pair
2. 用 `prepare_openrlhf_rm_data.py` 转 OpenRLHF 数据
3. 用 [openrlhf/train_rm_openrlhf.sh](/home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/framework_bridge/openrlhf/train_rm_openrlhf.sh) 开训
4. 用 [openrlhf/serve_rm_openrlhf.sh](/home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/framework_bridge/openrlhf/serve_rm_openrlhf.sh) 挂 RM

### B. 再训 actor

1. 用 `prepare_verl_rl_data.py` 生成 actor parquet
2. 把 `RM_MODEL_PATH` 指向 OpenRLHF 训练好的 RM
3. 用 [verl/train_grpo_lora_actor_verl.sh](/home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/framework_bridge/verl/train_grpo_lora_actor_verl.sh) 开 actor

## LoRA 模式说明

[verl/train_grpo_lora_actor_verl.sh](/home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/framework_bridge/verl/train_grpo_lora_actor_verl.sh) 支持两种模式：

1. `MODEL_PATH = merged SFT model`，再新挂一个 RL LoRA
2. `MODEL_PATH = base/merged`，同时设置 `SFT_LORA_ADAPTER=/path/to/existing/adapter`

第二种更接近“从现有 SFT adapter 继续做 RL”，但具体是直接续训 adapter，还是用 merged model 再挂新 adapter，还是要按你最终显存和部署策略决定。

## 现阶段的默认路径

脚本里默认指向的是你当前仓库里已经存在的 demo 路径，方便 smoke test：

- pair demo:
  - [rejected_pairs_from_sft_test20.jsonl](/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/runs/rejected_sft_sample20_20260331_155727/rejected_pairs_from_sft_test20.jsonl)
- SFT 原始数据:
  - [clean32_for_sft2.jsonl](/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/runs/clean32_for_sft2.jsonl)
- SFT merged 模型:
  - [clean32_for_sft2_fsdp_qlora_merged](/home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/qwen3-30b-moe/clean32_for_sft2_fsdp_qlora_merged)

正式训练前，建议把 `PAIR_INPUT` 和 `PROMPT_INPUT` 替换成你真正的大规模产物。
