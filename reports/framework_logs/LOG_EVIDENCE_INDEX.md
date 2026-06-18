# Framework Log Evidence Index

> 目标：保留 vLLM / verl / LlamaFactory 运行知识的证据链，但不把大日志直接塞进 Git。原始日志多为 MB 级到 GB 级，且可能含临时端口、机器路径、外链；本仓库只入库脱敏摘录和解释。

---

## 入库摘录

| 摘录 | 覆盖内容 |
|---|---|
| [excerpts/vllm_kv_cache_and_paged_attention.md](excerpts/vllm_kv_cache_and_paged_attention.md) | vLLM serving 参数、`max_model_len`、chunked prefill、KV cache token 容量、prefix cache |
| [excerpts/verl_shm_and_rollout_config.md](excerpts/verl_shm_and_rollout_config.md) | verl / Ray `/dev/shm` 警告、NCCL shm 报错、v6.10 rollout 配置 |
| [excerpts/llamafactory_sft_runtime.md](excerpts/llamafactory_sft_runtime.md) | LlamaFactory 数据加载、tokenization、update steps、loss 打印、SFT 运行要点 |

## 原始日志索引（不入库）

| 原始路径 | 大小级别 | 用途 | 入库替代物 |
|---|---:|---|---|
| `/tmp/gradio_rl_v610_step90.log` | ~15KB | Gradio/vLLM 服务：`max_model_len=8192`、chunked prefill、KV cache 243,904 tokens、并发估算 | vLLM 摘录 + `docs/rl_v610_step90_service_guide.md` |
| `/tmp/vllm_rl90_8000.log` / `/tmp/vllm_rl90_8001.log` | ~50KB each | OpenAI API vLLM：`max_model_len=4096`、KV cache 209,568 tokens、prefix cache hit rate | vLLM 摘录 |
| `/tmp/rl_v6.10.log` | MB 级 | verl GRPO：Ray `/dev/shm`、NCCL shm、rollout 配置、per-source KL、step 耗时 | verl 摘录 + 深读 1/3 |
| `/tmp/sft_v68_ffn_train.log` | ~190KB | LlamaFactory SFT：数据加载、tokenizer、update steps、loss | LlamaFactory 摘录 |
| `/tmp/sft_v69_balanced.log` | ~465KB | v69 balanced SFT：autoresume、checkpoint、训练完成 | LlamaFactory 摘录 + DATA.md |
| `/tmp/sft_v69_long_train.log` / `/tmp/sft_v69_long_train2.log` | MB 级 | cutoff / CE OOM / 4096 vs 6144 选择依据 | `docs/FRAMEWORK_RUNTIME_LESSONS.md` |
| `/tmp/gradio_v69c_repair_4gpu.log` | KB 级 | v69c repair serving 对照 | 服务指南与工作区审计 |

## 为什么不直接上传原始日志

1. **可读性差**：大部分是进度条、重复 warning、worker 启停噪声；
2. **安全性差**：日志可能夹带临时公网链接、环境变量、机器内网路径；
3. **仓库体积不合适**：模型训练日志增长很快，Git 不适合承载滚动日志；
4. **复盘价值更高的是解释层**：真正要传承的是"看到这类日志该怎么判断"，不是完整 stdout。

如果需要长期保存原始日志，建议放对象存储或实验追踪系统，并在本文件追加 checksum / artifact id。

