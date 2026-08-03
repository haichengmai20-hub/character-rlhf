# 框架运行知识：从 vLLM / verl / LlamaFactory 日志里沉淀出的经验

> 这篇不是框架教程，而是 ChatJoy 训练链路里反复踩坑后留下的**运行时心智模型**。原始日志不直接入库；可复盘的关键片段见 [reports/framework_logs/LOG_EVIDENCE_INDEX.md](../../reports/framework_logs/LOG_EVIDENCE_INDEX.md)。

---

## 1. vLLM：长上下文服务的核心账是 KV cache，不只是权重

ChatJoy 的 Gemma4 actor 约 49GB，TP=4 后每卡权重约 12-13GB。服务能不能起来，先看权重能否分片装下；服务能不能稳定长聊，要看 **KV cache pool**。

vLLM 的 PagedAttention 把 KV cache 切成 block/page，而不是为每条请求按最大长度开连续大块。它带来三个收益：

1. 短请求不为 8192 context 付满额显存；
2. 长短请求混跑时碎片更少，scheduler 更容易拼 batch；
3. prefix cache / chunked prefill 可以在长 system prompt 场景下复用前缀和分块预填充。

但 PagedAttention 不消灭 KV 成本。实际容量仍由：

```text
KV capacity ≈ 可分配给 KV 的显存 / 每 token 每层 KV 占用
```

日志里最该看的三行是：

```text
Available KV cache memory: ...
GPU KV cache size: ... tokens
Maximum concurrency for ... tokens per request: ...x
```

如果 `max_model_len=8192`，每条请求的最坏上下文是 8192 token；如果 `max_num_seqs=32`，理论上 scheduler 要能容纳 32 条同时增长的 KV。实际运行时还要再加 CUDA context、NCCL buffer、fragmentation、LoRA/adapter buffer。

**服务侧经验**：

- `max_model_len=8192` 适合长角色扮演 demo，但要配 `max_num_batched_tokens=1024` 这类 chunked prefill 限流；
- `gpu_memory_utilization` 是 vLLM 的显存预算上限，不是"越大越好"。colocate 训练时它会和 FSDP actor 抢同一张卡；
- `enforce_eager=True` 慢一些，但在这台 8×5090 上减少 CUDA graph / wake-up OOM 风险，适合 demo 和调试稳定优先的场景；
- `max_num_seqs` 降低通常不改变单条生成质量，但会降低吞吐和并发；训练 rollout 里还会影响采样吞吐。

## 2. verl：GRPO 的慢点通常在三处

verl 这条链的关键结构是：

```text
actor FSDP 训练态 + vLLM rollout 引擎 + RM vLLM scorer
```

在 ChatJoy 的部署里，actor 走 4 张卡，RM 走另外 4 张卡；actor 侧还要让 FSDP 和 vLLM 同卡交替工作。常见慢点：

1. **rollout 生成**：`ROLLOUT_N` 直接放大每个 prompt 的样本数，N=12 比 N=6 多一倍 response；
2. **old logprob / ref logprob**：生成完以后还要对长 response 做 forward-only logprob，MoE 模型这里会很慢；
3. **RM scoring / validation**：内部 val 频率过高时，每个 val 点都会吃掉十几分钟。

v6.10 的一次优化把 `ROLLOUT_N` 从 12 降到 8、`TEST_FREQ` 从 5 拉到 10、并尝试把 `LOGPROB_MICRO_BATCH_SIZE_PER_GPU` 从 1 提到 2。这里的判断标准不是"哪个数字最大"，而是：

- N 太低：GRPO 组内优势估计不稳，温度尾覆盖不足；
- N 太高：rollout 和 logprob 都线性变慢；
- logprob micro batch 太小：吞吐差；
- logprob micro batch 太大：MoE activation 峰值可能 OOM。

另一个重要经验是 `/dev/shm`。Ray 和 NCCL 都依赖共享内存；容器默认 64MB 时，NCCL 初始化可能报：

```text
No space left on device
Error while creating shared memory segment /dev/shm/nccl-...
```

这不是 CUDA OOM，而是系统资源不足。运行时修复是：

```bash
mount -o remount,size=2G /dev/shm
```

持久修复应在容器启动参数里设置 `--shm-size=2g`。

## 3. LlamaFactory：SFT 的坑多在数据长度、CE 显存和 Gemma4 merge

SFT 看起来比 RL 简单，但 Gemma4 + 26B MoE + 262k 词表让交叉熵显存很敏感：

- `cutoff_len=6144` 能覆盖更多多轮上下文，但 CE over vocab 会线性放大显存；
- `cutoff_len=4096` 是 v69 balanced 能稳定跑起来的折中；
- 如果 4096 step0 仍 OOM，降到 3072 比换 flash/sdpa 更有效，因为根因在 CE logits，不是 attention kernel。

运行时建议：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

它不能减少理论显存，但能缓解长训中的 CUDA allocator 碎片。

checkpoint / resume 方面：

- `save_only_model: false` 才能保留 optimizer/scheduler/rng，支持精确续训；
- wrapper 里检测到 `checkpoint-*` 后再把 `resume_from_checkpoint: true` 写入 YAML，可避免"无 checkpoint 却要求 resume"的 HF 报错；
- `save_steps=200` 在 8 卡 26B SFT 上是合理折中：清场最多丢约 200 step，但磁盘不会被过密 checkpoint 打爆。

Gemma4 merge 是单独一条经验：

1. 先用 text-only export 合并 LoRA；
2. 再用脚本把 vision tower / `vision_config` 拼回完整多模态目录；
3. 验证 `config.json` 里 `architectures = Gemma4ForConditionalGeneration` 且包含 `vision_config`。

直接用普通 export 生成"完整模型"容易缺 vision 配置，vLLM/verl 会秒崩。

## 4. 原始日志如何入库：摘录、索引、解释，不入库大文件

这类项目的日志有三种价值：

| 类型 | 入库方式 | 原因 |
|---|---|---|
| 关键配置行 | 摘录入库 | 可复盘启动参数和 runtime 约束 |
| 报错栈 / OOM / NCCL / shm | 摘录入库 | 可复盘根因和修法 |
| 数 MB 到数百 MB 原始训练日志 | 只做索引 | 噪声高、可能含路径/临时链接/环境变量，且 git 不适合存大日志 |

本仓库采用：

- [reports/framework_logs/LOG_EVIDENCE_INDEX.md](../../reports/framework_logs/LOG_EVIDENCE_INDEX.md)：记录原始日志路径、大小、用途、对应结论；
- [reports/framework_logs/excerpts/](../../reports/framework_logs/excerpts/)：保留可公开的小片段；
- 深读文档：把片段转成可迁移的框架知识。

这样未来回顾时可以先读结论，需要查证时再去原机器或归档系统找完整日志。

