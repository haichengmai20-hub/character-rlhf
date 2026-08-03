# 深读 1 — 显存算账与 GPU 通信：在 8×RTX 5090 上训 26B/30B MoE

> 阅读路线：先看 [深读 0 — 面试综合路线](00_interview_synthesis.md)，再按本篇专题深挖。

> **版本命名**：本篇保留原始实验号叙述；哪些是真改进、哪些只是 run/restart/失败实验，以 [VERSION_MAP](../docs/VERSION_MAP.md) 和 [project_chatjoy_overview](../docs/project_chatjoy_overview.md) 的阶段映射为准。

> 本篇展示的能力：**在消费级显卡（无 NVLink、无 P2P、32GB/卡）上，为 30B MoE 的 RM 训练和 26B MoE 的 verl GRPO，把每一 GB 显存和每一条通信路径算清楚** —— ZeRO 1/2/3 的取舍、通信原语的代价、vLLM 显存模型（含 cumem 虚拟地址语义）、以及一系列用"算账"而不是"瞎试"解决的 OOM。
>
> 所有数字来自实测（配置/日志见 [docs/](../docs/)），不是教科书估算。

> ## Attribution Lens
> 本篇把观察到的行为主要归因到 **deployment setting / 算力约束**（显存账、通信原语、vLLM 服务显存、GPU 拓扑），而非 data / reward / GRPO / evaluation。它是"算力反推算法"的一层：先算清每 GB 与每条通信路径，再决定 ZeRO 档位与 merge/部署方案。

---

## 0. 硬件底牌：一切算账的起点

| 项 | 实测值 | 含义 |
|---|---|---|
| GPU | 8× RTX 5090 | 消费卡，**PCIe 互联，无 NVLink** |
| 单卡可用显存 | **31.36 GB**（nvidia-smi 标称 32607MiB ≈ 31.84GiB，实际可分配预算更低；曾按"32GB"做账吃过 profile 误差的亏）| 所有预算按 31.36 算 |
| GPU P2P | **全部 CNS（不支持）** | 卡间数据必须绕 host：GPU→CPU 内存→GPU |
| NUMA 拓扑 | GPU 0-3 ∈ NUMA0，GPU 4-7 ∈ NUMA1 | 同 NUMA 内走 SHM **~50 GB/s**；跨 NUMA 走 QPI loopback **~5-10 GB/s（慢 10×）** |

**直接推论**（后面所有决策的物理基础）：
1. 任何 TP / ZeRO 通信组**必须圈在同一 NUMA 内**（0-3 或 4-7），跨 NUMA 通信带宽掉 10×；
2. 没有 P2P + 没有 NVLink → **通信量大的并行策略（ZeRO-3 / 大 TP）在这台机器上代价远高于论文设定**，算账时通信带宽要按 50GB/s（同 NUMA SHM）而非 NVLink 的 600GB/s 来折算。

---

## 1. ZeRO 1/2/3：显存与通信量的教科书账 + 本机实测

### 1.1 单卡显存的构成（混合精度 Adam，参数量 Φ）

```
bf16 参数:           2Φ
bf16 梯度:           2Φ
Adam 状态(fp32):    12Φ   = master参数 4Φ + momentum 4Φ + variance 4Φ
─────────────────────────
合计:               16Φ   （30B 模型 ≈ 480GB，单卡无解 → 必须切）
```

### 1.2 ZeRO 三档切什么、通信变多少（N 卡）

| 档 | 切分对象 | 单卡显存 | 每 step 通信量 | 用到的原语 |
|---|---|---|---|---|
| DDP 基线 | 不切 | 16Φ | 2Φ（梯度 all-reduce）| all-reduce |
| **ZeRO-1** | Adam 状态 | 4Φ + 12Φ/N | 2Φ（不变）| reduce-scatter(梯度) + all-gather(更新后参数) |
| **ZeRO-2** | + 梯度 | 2Φ + 14Φ/N | 2Φ（不变）| 同上，梯度即时 reduce-scatter 释放 |
| **ZeRO-3** | + 参数 | 16Φ/N | **3Φ（+50%）** | 前向 all-gather(Φ) + 反向 all-gather(Φ) + 梯度 reduce-scatter(Φ) |

**原语关系**（算通信账的基本功）：`all-reduce ≡ reduce-scatter + all-gather`；ring 实现下每卡收发 ~2(N−1)/N×M 数据。ZeRO-1/2 之所以"零额外通信"，就是把 all-reduce 拆成两半、各做一次；ZeRO-3 额外付的是**前向那次参数 all-gather**。

### 1.3 本项目的选择：RM 训练用 ZeRO-2 + 4-bit，而 ZeRO-3 实测全军覆没

**ZeRO-2 + NF4 的账**（Qwen3-30B-A3B MoE，RM 训练）：

```
57GB bf16 基座 → NF4 4-bit 量化 ≈ 15GB     (冻结，每卡完整持有一份 — ZeRO-2 不切参数)
LoRA r=64 (q/k/v/o) ≈ 53.5M 可训参数        (梯度/优化器都只针对它，可忽略)
Adam 状态 → adam_offload 放 CPU              (GPU 端 0)
激活值（gradient_checkpointing 后）≈ 4-5GB
──────────────────────────────────────────
合计 ≈ 20GB << 31.36GB ✓   （前向无 all-gather：参数全在本地）
```

**ZeRO-3 为什么死**（所有尝试 step 0 卡死/被 NCCL watchdog 杀）：
- 每层前向要 all-gather 完整参数：**48 个 MoE block × 1.2GB/block**，在同 NUMA SHM ~50GB/s 下理论尚可，但叠加 **Flash-Attention JIT 首次编译**的长停顿 → 集合通信超时 → watchdog kill；
- 教训：**通信账不能只算稳态带宽，要算上首 step 的编译/初始化长尾**——这正是论文不会告诉你的部分。
- 附带纠错：早期注释写"4-bit 与 OpenRLHF 不兼容"是错的——那是 ZeRO-3+4bit 的失败记录，ZeRO-2+4bit 完全可行。

**两个 DeepSpeed 工程坑**（都属于"账对了、工具骗你"）：
1. `--num_gpus` 会**忽略** `CUDA_VISIBLE_DEVICES` → 必须用 `--include localhost:0,1,2,3` 指卡（顺带把通信组锁进同一 NUMA）；
2. 默认 `save_steps` 存 DeepSpeed ZeRO 全量状态（14-57GB）→ 磁盘写爆训练崩（v1 的 step200 ckpt 因此**永久丢失**）；改 `--save_hf_ckpt --disable_ds_ckpt` 只存 ~500MB LoRA adapter。**磁盘也是要算账的资源。**

### 1.4 通信量的实际代价（无 NVLink 机器上 ZeRO-3 为什么慢到不实用）

理论上 ZeRO-3 只比 ZeRO-2 多付一次前向 all-gather（Φ 数据量），但在本机：

```
26B bf16 参数 = 52GB 数据
ZeRO-3 前向需逐层 all-gather 全部参数：52GB ÷ 同 NUMA SHM ~50GB/s ≈ 1s 量级——只是一次前向的通信
反向再来一次 all-gather：≈ 2s+/step 仅通信（还未计梯度 reduce-scatter）
（注：上文"48 block × 1.2GB"是 30B/Qwen3 的分块账；26B/Gemma4 此处按总量口径估算）
```

对比 ZeRO-2（参数不切，本地完整持有）：每 step 只有梯度 reduce-scatter 一次，约 52GB ÷ 2 × (8-1)/8 ≈ 22GB / 50GB/s ≈ **440ms**。ZeRO-3 的每 step 通信代价在这台机器上是 ZeRO-2 的 ≈5 倍以上——而 ZeRO-2 + NF4 已经把显存压到 ~20GB（有余量），ZeRO-3 的显存收益根本用不着，这才是"ZeRO-3 在本项目是死路"的真正定量根因（不是"实现难"）。

> 结论：**通信档位选择要看"额外收益是否被消耗"，不是看"通信量绝对小"**。ZeRO-3 在有 NVLink（~600GB/s）的机器上才有价值。

---

## 2. verl GRPO 的 colocate 显存账：actor FSDP + vLLM 同卡

RL 阶段一张卡上同时住着 actor 的 FSDP 训练态和 vLLM 推理引擎（colocate，靠 sleep/wake 切换），这是整个项目显存最紧的地方。先把 verl 的三个关键机制查源码确认，再列账：

1. **ref policy 零显存**：LoRA 模式下 `compute_ref_log_prob` 就是 actor 临时 `disable_adapter()`（verl `fsdp_workers.py`）→ 参考策略**不另占一份 26B**；
2. **vLLM sleep_level 语义**：LoRA 热更新模式 → `sleep_level=1`（只释放 KV 物理页，**base 权重 12.15GB 常驻**）；
3. **param_offload**：决定 FSDP actor 在 rollout 阶段是否搬去 CPU。

### 2.1 GPU 0-3（actor+rollout+ref 三合一，TP=4，Gemma4 48.6GB÷4=12.15GB/卡）

**param_offload=False（默认）— 算出来必死**：

| 阶段 | FSDP actor | vLLM | 激活/其他 | 合计 | 余量 |
|---|---|---|---|---|---|
| **Rollout 峰值** | 12.15 | 19.1（base 12.15 + KV 6.95，util=0.60）| ~0.5 | **31.75** | **0.09GB ⚠️ 必 OOM** |
| log_prob / update | 12.15 | 12.15（sleep=1）| 2-3 | ~27 | 4.8 ✓ |

**param_offload=True（采用）**：

| 阶段 | FSDP actor | vLLM | 激活/其他 | 合计 | 余量 |
|---|---|---|---|---|---|
| Rollout | **0（卸到 CPU）** | 19.1 | 0.5 | 19.6 | **12.2 ✓** |
| log_prob / update | 12.15 | 12.15 | 2-3 | ~27 | 4.8 ✓ |

代价：每 step 多两次 CPU↔GPU 全量拷贝，PCIe 5.0 下 ~2-4s——**用 4 秒延迟买 12GB 余量**，是这台机器上唯一可行解。

> 口径注：本节"余量"按 nvidia-smi 标称 31.84GiB 计（31.84 − 31.75 = 0.09；31.84 − 19.6 ≈ 12.2）；§0 的 31.36GB 是预算规划用的保守基线。§2.2 的两行历史记录各自保留了事发当时算式的基数（0.62 行用 32.6 十进制、0.45 行用 31.36 保守值），结论不受基数选择影响。

### 2.1.1 `/dev/shm` 大小是 NCCL FSDP 的隐性前提（差点永远找不到的 bug）

RL v6.10 启动时遇到一个诊断极难的死法：NCCL 集合通信在 FSDP 初始化阶段必崩，不是 OOM、不是 watchdog 超时，而是 **NCCL 申请共享内存 segment 失败（errno = ENOSPC）**。

根因：容器的 `/dev/shm` 默认只分配了 **64MB**；NCCL 单个 rendezvous buffer segment 需要 ~33MB，8 卡多进程同时申请 → 放不下第二个 → init 阶段段错误崩溃。

```bash
# 日志特征（非 OOM，是 shm 分配失败）
NCCL WARN Call to shm_open failed: /nccl-xxx: No space left on device
NCCL WARN Call to mmap failed: Cannot allocate memory
```

修复（运行时，非持久）：

```bash
mount -o remount,size=2G /dev/shm
```

**为什么难找**：NCCL shm 大小不在 PyTorch / verl / NCCL 的任何文档里明确提到；报错出现在 `ncclAllReduce` 初始化路径，不是训练主循环；容器内 `/dev/shm` 大小取决于 `docker run --shm-size`（或 compose `shm_size`），不写就默认 64MB。**持久修复要管理员重建容器加 `--shm-size=2g`**，运行时 remount 重启容器后失效。

教训：**系统级资源（shm / open files / pid limits）是第四类账**，和显存/带宽/磁盘并列，任何分布式训练启动前都应检查一遍。

### 2.2 GPU 4-7（RM 独立 resource pool，Qwen3-30B TP=4 = 14.25GB/卡）

| 项 | 值 |
|---|---|
| vLLM 预算 | util 0.62 × 32.6 ≈ 20.2GB |
| KV cache | 20.2 − 14.25 = **5.95GB**（RM 只做 scoring 前向、非生成，够用）|
| 历史 bug | util 初设 0.45 → 0.45×31.36=14.1GB **< 权重 14.25GB** → 连模型都装不下，OOM。**一行除法就能预判的事故** |

---

## 3. vLLM 显存模型与 cumem 踩坑实录（本篇最深的一节）

### 3.1 先把上下文账算清：context length、KV cache 与 PagedAttention

服务侧最容易被低估的账不是模型权重，而是 **KV cache**。对一个自回归 Transformer 来说，解码第 `t` 个 token 时，每一层都要复用前 `1..t-1` 个 token 的 key/value；因此只要请求还在生成，KV 就必须常驻显存。

粗略心智模型：

```
KV 显存 ∝ layers × kv_heads × head_dim × (prompt_tokens + generated_tokens)
          × concurrent_sequences × dtype_bytes
```

也就是说，`max_model_len` 不是一个"无害的最大长度配置"。它定义的是每条请求可能占用的上下文上限：

```
context length = prompt tokens + generated tokens
```

在 ChatJoy 的服务和 rollout 里，`max_model_len=4096/8192`、`max_tokens=768/1024/1536`、`max_num_seqs`、`ROLLOUT_N` 四个量会一起决定 KV 压力：

- `max_model_len` 决定单条请求可到多长；
- `max_num_seqs` 决定同一时刻最多多少条请求占 KV；
- `max_tokens` 决定实际生成会把上下文推多远；
- `ROLLOUT_N` 在 RL 里把每个 prompt 扩成多条 response，间接增加调度与 KV 峰值。

**PagedAttention 的作用**：vLLM 不再为每条请求按最大长度预留一整块连续 KV，而是把 KV cache 切成 page/block 管理。短请求只占少量页，长请求随着生成追加页；不同请求的页可以在统一 KV block pool 里调度、复用、释放。这解决了两个传统 serving 痛点：

1. 避免"每条请求都按最大长度开一整块连续显存"造成的浪费；
2. 降低长短请求混跑时的碎片化，让 batch scheduler 可以更稳定地塞满 GPU。

但 PagedAttention **不会让 KV 免费**。vLLM 仍然会从 `gpu_memory_utilization` 预算里切出一池 KV blocks。日志里这三行就是最直接的 KV 账：

```text
Available KV cache memory: 13.96 GiB
GPU KV cache size: 243,904 tokens
Maximum concurrency for 8,192 tokens per request: 79.06x
```

这意味着：在该服务配置下，vLLM 能用于 KV 的池子约 13.96GiB，换算成 243,904 个 token 的 KV 容量；如果每条请求都跑满 8192 context，理论并发约 79 条。真实并发还会被 `max_num_seqs`、prefill chunk、调度开销和 CUDA context 限制。

**为什么这对本项目重要**：

- demo/产品服务为了长角色扮演把 `max_model_len` 提到 8192，同时用 `max_num_batched_tokens=1024` 开 chunked prefill，把长 prompt 的 prefill 拆块，避免 prefill 阶段一次性把显存和 scheduler 打爆；
- RL rollout 更危险，因为 actor FSDP 和 vLLM 同卡。PagedAttention 降低了 KV 浪费，但 `gpu_memory_utilization × 31.36GB` 的 vLLM 预算仍会和 FSDP shard、NCCL buffer、CUDA context 争同一张卡；
- 所以 `max_num_seqs 32→24/8`、`max_model_len 8192→4096/2816` 这类调参，本质是在收缩 KV block pool 的需求，不是"随便降一点并发"。

一句话：**PagedAttention 是 vLLM 能高效服务长上下文的原因，但长上下文本身的 KV 账仍必须按 tokens × concurrency 算。**

### 3.2 核心心智模型：`gpu_memory_utilization` 是**虚拟地址预留**

实测得出的最重要认识（vLLM V1 cumem allocator）：

> `sleep(level=1)` 只 unmap **物理页**，`gpu_memory_utilization × 31.36GB` 的**虚拟地址段始终被 vLLM 进程占住**。
> 所以 FSDP 与 vLLM 共享物理卡时，**峰值 = vLLM 虚拟预算 + FSDP shard + 多进程 overhead** —— 不能想当然认为"vLLM 睡了显存就还回来了"。

### 3.3 三档 `ROLLOUT_GPU_MEMORY_UTILIZATION` 复测：三个值、三种死法（每个都能用账解释）

| 设置 | 结果 | 死在哪 | 账 |
|---|---|---|---|
| **0.78** | 崩 | `cumem_allocator.cpp:139` at `wake_up(weights)` | 虚拟预算 0.78×31.36=24.5GB；剩 6.9GB 给 overhead 不够：同卡 **2 个进程**（4 个 FSDP WorkerDict rank + vLLMHttpServer）各自 CUDA context 0.5-1GB + NCCL 工作 buffer + flashinfer JIT 缓存 ≈ **4-6GB 碎片**；且 cumem `create_and_map` 要**物理连续页**，碎片化后找不到 |
| **0.60** | 崩 | FSDP 加载回 GPU 时 CUDA OOM | vLLM 实占 18.60 + FSDP 12.46 = 31.06GB，**只剩 229MB**，FSDP 再要 396MB 爆 —— 证实"虚拟预算常驻"模型 |
| **0.50** | 崩 | vLLM init：`No available memory for the cache blocks` | 预算 15.68 − 权重 13 = 2.68GB，连最小 KV block 都分配不出 |

→ 三档全崩说明**不是调参问题，是结构问题**：peak 时 FSDP 和 vLLM 同时在卡上。

### 3.4 结构性修复：把"同时在卡"改成"串行换防"

重排 verl `engine_workers.update_weights()` 的顺序（配合日志埋点核账）：

```
旧序: resume(weights) → FSDP 上 GPU 取参数 → IPC 推送      # 13GB + 13GB 同卡 → OOM
新序: FSDP 上 GPU 收集 LoRA → 取到 CPU → FSDP 卸回 CPU
      → aggressive_empty_cache → 此时 GPU 空
      → resume(weights)（wake_up 安全）→ IPC 推送           # 峰值只剩一方 ✓
```

配套手段（按账排出的降显存阶梯，**一次只动一项，否则无法归因**）：
`max_num_seqs 32→8`（KV 最小需求 = seqs × max_model_len）→ 压 `MAX_PROMPT/RESPONSE_LENGTH`（max_model_len = 两者之和，KV 按它分块）→ `ROLLOUT_N 4→2` → `micro_batch 2→1` → `PPO_MAX_TOKEN_LEN 8192→4096` → FP8 KV → 改 colocate 架构。

### 3.5 两个"深水区"根因（值得单独记）

**A. LoRA 热更新 `illegal memory access`**：vLLM punica 的 `lora_a_stacked/lora_b_stacked` buffer 被分配进 cumem 的 `weights` tag → `sleep(level=1, offload_tags=("weights",))` 把它们一起 unmap → 下次 `set_lora` 命中已 unmap 区间 → `CUDA error: invalid argument / illegal memory access`。**错的是指针不是形状**（这解释了为什么改 rank/模块都没用）。修复：任意 sleep 模式都 `resume(tags=["weights"])`。

**B. `lora.merge=True` 路径为什么在本机永不可行**（排除清单，防止后人重试）：
- FSDP1：`summon_full_params` 要把 25.84B bf16 = **51.7GB 拉到单 rank** → 31.36GB 装不下；
- FSDP2：Gemma4 是 **128-expert MoE**，wrap targets 遍历 ~35 万个子模块 init 卡死 5min+，且 `load_full_state_dict` 仍要 rank0 装 52GB。
→ 只能走 hot-swap（merge=False），所以 3.5A 的 cumem 问题**必须**解决，绕不开。

### 3.6 资源隔离的一个轻量技巧：`CUDA_VISIBLE_DEVICES` 重映射

物理 GPU0 长期被别的进程占 1.9GB → 不改 verl 资源池代码，启动时 `CUDA_VISIBLE_DEVICES=4,5,6,7,0,1,2,3`：verl 眼中的 actor pool（逻辑 0-3）落到干净的物理 4-7，RM pool（容忍 squatter）落到物理 0-3。**用环境变量做了一次零代码的资源调度。**

---

## 4. TP 切分与部署侧的账

| 场景 | 账 |
|---|---|
| actor rollout TP=4 | 48.6GB ÷ 4 = 12.15GB/卡（SFT merged 53.8÷4=13.45）——单卡装不下，**TP=4 不是选择是必然**；TP 每层一次 all-reduce，必须圈在同 NUMA |
| RM-as-classifier | vLLM 原生无 `Qwen3MoeForSequenceClassification` → 用 `as_seq_cls_model(Qwen3MoeForCausalLM)` 包装 + registry 注册；OpenRLHF LoRA + score head（`[1, 2048]`）手动 delta-merge 进最后一个 shard |
| 纯文本 merge 部署 | Gemma4 含视觉塔：merge 必须 `force_text_only`（否则 PEFT 注进 SigLIP 的 q/k/v/o 直接崩）；vLLM/verl 加载要补回 `vision_config` + `--limit-mm-per-prompt '{"image":0,"video":0}'`；serving `max-model-len 8192`（修线上 4096 长卡溢出）|
| GPU 拓扑约定 | RM 落物理 0-3、actor 落物理 4-7，写进 launch 脚本固化 |

---

## 5. 方法论小结

1. **先算后跑**：每次 OOM 都先把"权重/KV/激活/context/碎片"逐项列出来对 31.36GB，三档复测表里每个崩溃都能被加法解释——能解释才算懂；
2. **通信账 = 原语 × 拓扑**：ZeRO 档位选择不是背表，是把 all-gather/reduce-scatter 的量放到"无 P2P + 跨 NUMA 慢 10×"的真实拓扑上重算；
3. **区分虚拟与物理**：cumem 的虚拟地址预留是这台机器上最反直觉的一课——显存账要按"谁占着地址空间"算，不是按 nvidia-smi 的瞬时占用算；
4. **一次只动一个变量**：降显存阶梯按优先级排好，崩了才能归因到具体项。

> 相关代码/脚本：[training/configs/](../training/configs/)、[training/launch_scripts/](../training/launch_scripts/)、[training/GRPO_Training_Package/framework_bridge/](../training/GRPO_Training_Package/framework_bridge/)；完整事故记录：[docs/archive/chatjoy_rl_takeover_error_fix_log.md](../docs/archive/chatjoy_rl_takeover_error_fix_log.md)。
