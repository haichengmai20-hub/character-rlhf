# vLLM 摘录：KV cache、PagedAttention 与长上下文服务

> 以下为脱敏摘录，保留关键配置和容量行。完整原始日志见 `LOG_EVIDENCE_INDEX.md`，不入库。

## RL v6.10 step90 Gradio 服务（8192 context）

```text
vLLM配置: prefix_caching=True, enforce_eager=True,
max_num_seqs=8, max_num_batched_tokens=1024

INFO ... Using max model len 8192
INFO ... Chunked prefill is enabled with max_num_batched_tokens=1024.
INFO ... All limits ... text-only mode.
INFO ... Available KV cache memory: 13.96 GiB
INFO ... GPU KV cache size: 243,904 tokens
INFO ... Maximum concurrency for 8,192 tokens per request: 79.06x
```

解读：

- `max_model_len=8192` 是服务长角色扮演的上下文上限；
- chunked prefill 把长 prompt 的 prefill 分块，降低一次性 prefill 峰值；
- KV cache token 容量和最大并发是 vLLM 最值得看的显存账；
- PagedAttention 让 KV 用 page/block 管理，但不消灭 KV 成本。

## OpenAI API vLLM 对照服务（4096 context）

```text
non-default args:
  model=/root/rl_v610_step90_merged
  max_model_len=4096
  tensor_parallel_size=4
  gpu_memory_utilization=0.85

INFO ... GPU KV cache size: 209,568 tokens
INFO ... Maximum concurrency for 4,096 tokens per request: 64.37x
INFO ... GPU KV cache usage: 0.5%, Prefix cache hit rate: 77.8%
```

解读：

- `max_model_len` 从 8192 降到 4096 时，单请求最坏 KV 压力减半；
- prefix cache hit rate 对角色卡场景特别重要，因为 system prompt 大量复用；
- `gpu_memory_utilization` 提高会扩大 vLLM 预算，但在 colocate 训练里可能挤压 FSDP。

