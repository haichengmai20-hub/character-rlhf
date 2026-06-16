# ChatJoy v68 SFT — LoRA B 训练 + Merge + Probe 验收

## 背景

v68 SFT 的核心目标：让 clean 输出在 t=0.9 产品口径下，三轴（动作丰富/语言质量/独白质量）逼近 base teacher。已完成选样改造、judge 打分、CJK 重采、合池 build。现在只跑一路 LoRA B (attn+FFN+router) 训练、merge、然后 probe 验收。

## 第 1 步：训练 LoRA B

```bash
cd /root/hanjiaqi/LlamaFactory
CUDA_VISIBLE_DEVICES=4,5,6,7 llamafactory-cli train chatjoy/gemma4_rich_cjk_sft_v68_lora_ffn.yaml
```

训练启动后确认：
- loss 正常下降（初期 ~2.x，逐步降到 ~1.x 以下）
- 无 OOM
- 输出目录 `/root/hanjiaqi/LlamaFactory/chatjoy/saves/gemma4-26b-a4b/rich_cjk_sft_lora_v68_ffn` 正确生成 checkpoint

如果训练过程中 loss 震荡异常（可能 router.proj LoRA 不稳定），记录但不停，跑完看结果。

## 第 2 步：Merge LoRA B → 全量模型

```bash
cd /root/hanjiaqi/LlamaFactory

# Step 2a: Export merge (text-only)
CUDA_VISIBLE_DEVICES=4 llamafactory-cli export \
    --model_name_or_path /root/sft_rich_cjk_v67_repair_merged_full \
    --adapter_name_or_path chatjoy/saves/gemma4-26b-a4b/rich_cjk_sft_lora_v68_ffn \
    --template gemma4 \
    --finetuning_type lora \
    --trust_remote_code true \
    --export_dir /root/sft_rich_cjk_v68_ffn_merged \
    --export_size 2 \
    --export_device cpu \
    --force_text_only true

# Step 2b: Re-attach vision encoder (CRITICAL — without this merged model is text-only only, not Gemma4ForConditionalGeneration)
/home/ubuntu/miniconda3/envs/gemma4-rl/bin/python3 -c "
from chatjoy.GRPO_Training_Package.framework_bridge.verl.patches.auto_sft_merge_and_launch_rl_v67 import reattach_vision
reattach_vision('/root/sft_rich_cjk_v68_ffn_merged', '/root/sft_rich_cjk_v67_repair_merged_full')
"
```

## 第 3 步：Merge 后验证

```bash
python3 -c "
import json
path = '/root/sft_rich_cjk_v68_ffn_merged'
cfg = json.load(open(f'{path}/config.json'))
print(f'model_type: {cfg.get(\"model_type\")}')
print(f'architectures: {cfg.get(\"architectures\")}')
print(f'has_vision_config: {\"vision_config\" in cfg}')
# must be Gemma4ForConditionalGeneration and have vision_config
assert cfg.get('architectures') == ['Gemma4ForConditionalGeneration'], 'NOT Gemma4ForConditionalGeneration!'
assert 'vision_config' in cfg, 'NO vision_config!'
print('✓ Merge verified: Gemma4ForConditionalGeneration with vision')
"
```

## 第 4 步：t=0.9 Probe 验收（最重要的验收门）

用 55 卡 clean prompt 对 merged 模型跑推理，参数: temp=0.9 / top_p=0.95 / presence_penalty=0.4。

### 三轴目标（vs base teacher 参照）

| 指标 | base teacher 参照 | 目标（≥base ~90%） | 已知弱轴 |
|------|-------------------|--------------------|----------|
| 动作块 mean | 4.7 | ≥4.23 | 否 |
| 对话块 mean | 3.9 | ≥3.51 | **是（结构性弱轴，达不到也只标注不停训）** |
| 独白率 | 59% | ≥53% | 否 |
| 英文 Thoughts: 尾巴率 | — | ≤8% 不涨 | 否 |

### 护栏（任一未过才停）

- attack 聚合 RM ≥ 0.31（step0@t0.9 水平），用 chatjoy_rm_v6.6 merged 给 attack 用例打分
- 5 个确定性 zh 破功不新增
- 开场集中度 ≤ step0_hot（0.013 量级）不回退

### 按语言分桶报告

**不仅看 CJK，还要看非 CJK（es/ar/ru/pt 各几张）**。原因：rich 半区只有 CJK (ko/ja/zh/zh-tw)，非 CJK 只在 replay 半区（多语言防御数据），没有丰富度提升。probe 必须确认非 CJK ≈ base（没回退）。

### 报告格式

输出一张对比表：

```
| 指标              | step0(base) | v68_ffn_merged | vs base | 达标? |
| 动作块 mean       | X.XX        | X.XX           | +X.XX   | ✓/✗   |
| 对话块 mean       | X.XX        | X.XX           | +X.XX   | ✓/标注 |
| 独白率            | XX%         | XX%            | +XXpp   | ✓/✗   |
| Thoughts尾巴率    | XX%         | XX%            | +XXpp   | ✓/✗   |
| attack聚合RM      | 0.XXX       | 0.XXX          | +X.XXX  | ✓/✗   |
| 开场集中度        | 0.XXX       | 0.XXX          | -/+/0   | ✓/✗   |
| zh破功数(5基准)   | X           | X              | +0/+N   | ✓/✗   |
```

分桶版本（按语言）也要出。

## 关键配置信息

| 项 | 值 |
|---|---|
| Base model (step0) | `/root/sft_rich_cjk_v67_repair_merged_full` — 确认是 Gemma4ForConditionalGeneration, 有 vision_config |
| 训练数据 | `rich_cjk_sft_mix_v68` → `/root/rich_cjk_sft_v68/rich_cjk_sft_mix_v68_final.json` |
| 数据规模 | 9950 条 (4975 rich + 4975 replay, 1:1) |
| LoRA B targets | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, router.proj |
| LoRA rank/alpha | 16 / 32 |
| B checkpoint 输出 | `/root/hanjiaqi/LlamaFactory/chatjoy/saves/gemma4-26b-a4b/rich_cjk_sft_lora_v68_ffn` |
| B merged 输出 | `/root/sft_rich_cjk_v68_ffn_merged` |
| RM model (probe用) | chatjoy_rm_v6.6 merged |
| Python env | `/home/ubuntu/miniconda3/envs/gemma4-rl/bin/python3` |
| LlamaFactory | `/root/hanjiaqi/LlamaFactory` |
| Template | gemma4 |

## 训练超参（已在 yaml 中）

- per_device_train_batch_size: 1
- gradient_accumulation_steps: 8
- learning_rate: 5e-5
- num_train_epochs: 1.0
- lr_scheduler_type: cosine
- warmup_ratio: 0.03
- bf16: true
- quantization_bit: 4
- cutoff_len: 4096

## 硬约束

1. **GPU 0-3 严禁动**（RL 训练在跑）。训练和 merge 只用 GPU 4-7。
2. **不删任何 checkpoint/merged**（step0 底座、RL ckpt、RM merged 都要留）。
3. **probe 任一护栏未过才停**。对话块是已知结构性弱轴，达不到不停训，只标注。
4. Merge 后必须 reattach_vision，否则模型不是 Gemma4ForConditionalGeneration。

## 已知风险

- 对话块是已知结构性弱轴（build GATE 2.77 vs base 3.9），probe 时单独标注差距，不作为停训项。
- 非 CJK 语言（es/ar/ru/pt 等）只出现在 replay 半区，没有丰富度提升。probe 必须按语言分桶确认非 CJK ≈ base（没回退）。
- router.proj 是 2816→128 的小 Linear，LoRA 可能不稳定，注意 loss 曲线有无异常震荡。如果异常，记录但继续跑完。
