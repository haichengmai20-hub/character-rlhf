# LlamaFactory 摘录：SFT 数据加载、tokenization 与训练运行

> 以下为脱敏摘录，用于解释 v6.8/v6.9 SFT 的运行链路和配置判断。

## 数据加载与 tokenization

```text
Loading dataset /root/rich_cjk_sft_v68/rich_cjk_sft_mix_v68_final.json...
Converting format of dataset (num_proc=16): 9950/9950
Running tokenizer on dataset (num_proc=16): 9950/9950
```

解读：

- LlamaFactory 会先把自定义 JSON 转成内部 instruction/chat 格式，再跑 tokenizer；
- 多轮 history 修复后，`cutoff_len` 会直接影响 output 是否被截断；
- v69 曾评估 6144 覆盖率更好，但 26B + 262k vocab 的 CE 显存把最终训练折中拉回 4096。

## update steps 与 loss

```text
Num update steps per epoch = 311
{'loss': '0.9573', 'learning_rate': '0', 'epoch': '0.003215'}
{'loss': '0.8009', 'learning_rate': '5e-05', 'epoch': '0.03537'}
```

解读：

- `Num update steps per epoch` 是判断训练时长和 checkpoint 频率的第一参照；
- v69 balanced 选择 1 epoch，是因为 clean 已接近 base，继续多训容易过拟合/防御侵蚀；
- v69b/v69c 的 2 epoch 是为了让新增 defense supplement 多过一遍，但 3 epoch 被明确禁止，避免 v68 式独白过冲。

## SFT 运行建议

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

说明：

- 这个设置缓解 allocator 碎片，不降低理论峰值；
- `save_only_model: false` + `checkpoint-*` 才能精确续训；
- Gemma4 merge 必须走 text-only export 再拼回 vision tower 的两步法。

