# verl / Ray 摘录：`/dev/shm`、rollout 配置与 v6.10 慢点

> 以下为脱敏摘录。重点是把"为什么崩/为什么慢"转成可复盘的配置证据。

## Ray object store 警告

```text
WARNING services.py -- The object store is using /tmp instead of /dev/shm
because /dev/shm has only 67108864 bytes available.
This will harm performance.
```

解读：

- 容器默认 `/dev/shm=64MB` 时，Ray 会退到 `/tmp`，性能变差；
- 对 NCCL/FSDP 更严重的是共享内存 segment 不够，可能导致初始化直接崩。

## NCCL shm 失败

```text
NCCL WARN Error: failed to extend /dev/shm/nccl-... No space left on device (28)
Error while creating shared memory segment /dev/shm/nccl-...
```

解读：

- 这不是 CUDA OOM；
- 运行时修复：`mount -o remount,size=2G /dev/shm`；
- 持久修复：容器启动时加 `--shm-size=2g` 或 compose `shm_size: 2g`。

## v6.10 rollout 关键配置

```text
rollout_n: 12
gpu_memory_utilization: 0.72
max_num_seqs: 32
log_prob_micro_batch_size_per_gpu: 1
test_freq: 25
[validate_config] All configuration checks passed successfully!
```

后续优化方向：

- `ROLLOUT_N 12 -> 8`：减少生成和 logprob 工作量，同时保留比 N=6 更稳的组内优势；
- `LOGPROB_MICRO_BATCH_SIZE_PER_GPU 1 -> 2`：forward-only 阶段吞吐翻倍的候选优化，但需观察 MoE activation OOM；
- `TEST_FREQ` 拉大：内部 val 很贵，频率过高会掩盖真实 step 吞吐。

