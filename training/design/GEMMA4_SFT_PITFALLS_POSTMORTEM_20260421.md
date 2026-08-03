时间: 2026-04-21  
范围: 仅覆盖 Gemma4 SFT 链路（加载、量化、分布式训练）  
证据原则: 只引用当前工作区真实代码与日志，不做外推

## 0. 记录策略（放在哪个 MD）

结论: **新起一份文档**，不继续堆到 takeover 主文档。

原因:

- `GEMMA4_TAKEOVER_*` 主要是“接手过程/实验过程记录”
- 本文是“按问题维度汇总复盘 + 当前可执行 TODO”
- 分开后更利于后续汇报和追踪

## 1. 数据来源（可核查）

- 接手状态: [GEMMA4_TAKEOVER_STATUS_20260420.md](GEMMA4_TAKEOVER_STATUS_20260420.md)
- 接手结果: [GEMMA4_TAKEOVER_RESULT_20260420.md](GEMMA4_TAKEOVER_RESULT_20260420.md)
- 关键训练日志目录: logs
- 历史异常训练记录: [trainer_log.jsonl](#)
- 当前稳定 run 记录: [trainer_log.jsonl](#)

## 2. 从头到尾的坑点与修复（按时间线）

### 2.1 早期多轮 OOM（加载阶段，尚未进入有效训练）

现象:

- 1GPU/FSDP/ZeRO-3 多轮都在加载阶段 OOM，常见是再分配 `484 MiB` 或 `968 MiB` 失败。
- 证据示例: [gemma4_1gpu_extreme_20260417_201523.log:837](#)
- 证据示例: [gemma4_fsdp_qlora_4gpu_20260417_191110.log:868](#)

处理:

- 引入 BNB CPU-first materialization workaround，避免先在 GPU materialize bf16 再量化:
  [patcher.py:55](#)

### 2.2 ZeRO-3 + 4bit 下 Gemma4 权重大量滞留 CPU，随后“补搬运”触发 OOM

现象:

- 诊断显示 `gpu=0.83GB cpu=48.63GB`，说明大量参数未成功留在可用量化状态:
  [gemma4_z3_qlora_4gpu_20260420_154852.log:871](#)
- 随后在 `loader.py` 把 CPU 参数强搬到 CUDA 直接 OOM:
  [gemma4_z3_qlora_4gpu_20260420_154852.log:975](#)

处理:

- 在加载后对 Gemma4 experts 走自定义 4-bit 准备路径:
  [loader.py:190](#)
- 新增 Gemma4 experts 自定义 4-bit 实现:
  [moe.py:40](#)
- 明确不再隐式把 CPU residual 强迁移到 GPU，改为日志告警/显式失败:
  [loader.py:234](#)

### 2.3 Transformers 接口变化导致 trainer 签名不兼容

现象:

- 训练启动后直接报:
  `CustomSeq2SeqTrainer.create_optimizer() takes 1 positional argument but 2 were given`
- 证据: [gemma4_fsdp_qlora_4gpu_20260420_175808.log:992](#)

处理:

- `create_optimizer` 签名改为 `def create_optimizer(self, model=None)` 并透传 `super().create_optimizer(model)`:
  sft/trainer.py:101

### 2.4 Gemma4 训练期输入缺 `mm_token_type_ids`

现象:

- 训练 forward 阶段报:
  ``ValueError: `mm_token_type_ids` is required as a model input when training``
- 证据: [gemma4_fsdp_qlora_4gpu_20260420_181209.log:1069](#)

处理:

- 在 collator 中为 Gemma4 显式补齐 `mm_token_type_ids`（优先从 processor 取，否则 fallback 全 0）:
  [collator.py:180](#)

### 2.5 FSDP + 当前补丁链路触发参数一致性断言

现象:

- 首轮 backward 即报:
  `FSDP does not support changing the parameters between forward and backward for HandleShardingStrategy.SHARD_GRAD_OP`
- 证据: [gemma4_fsdp_qlora_4gpu_20260420_184834.log:1026](#)
- 2026-04-21 在 3GPU FSDP 复现同类问题:
  [gemma4_fsdp_qlora_3gpu567_20260421_130014.log:994](#)

处理:

- 主线先切 DDP 跑通（FSDP 暂不作为主线）。
- FSDP 问题保留为单独技术债，后续单测化处理。

### 2.6 占卡场景下的“伪代码问题” OOM（其实是资源问题）

现象:

- `caching_allocator_warmup()` 在加载时申请 `12.21 GiB`，但 GPU0 仅剩 `1.76 GiB`:
  [gemma4_fsdp_qlora_4gpu_20260420_193151.log:853](#)
- 3GPU DDP 也出现过 barrier 阶段 `CUDA error: out of memory`:
  [gemma4_ddp_qlora_3gpu567_20260421_130612.log:405](#)

处理:

- 启动前先清理冲突进程，再重启训练。
- 这类问题归类为“资源隔离问题”，不是模型逻辑 bug。

### 2.7 历史异常: loss 长期为 0 且速度偏快（可疑 run）

现象:

- 历史 run 中从 step10 到 step100 多点 `loss=0.0`:
  sft2_lora_4gpu/trainer_log.jsonl:1

处理:

- 当前代码增加 Gemma4 batch 有效 label 校验：若全 `IGNORE_INDEX` 直接报错，防止静默退化训练:
  sft/trainer.py:123
- 当前 DDP run 已验证为非零 loss 且持续下降，不再出现该模式。

### 2.8 Gemma4 模板/turn token 对齐

现象:

- Gemma4 tokenizer 使用 `<|turn> / <turn|>`，与旧模板 `<start_of_turn>/<end_of_turn>` 不一致，容易引入模板-词表偏差。

处理:

- 模板已切换为 Gemma4 原生 turn token:
  [template.py:1073](#)

## 3. 当前状态（2026-04-21）

当前主线策略:

- 框架: DDP（非 FSDP）
- 卡: GPU5/6/7
- 配置: `cutoff_len=3072`, `QLoRA 4-bit`, `LoRA`, `grad_accum=8`
  - [gemma4_sft2_3gpu_ddp_lora_567.yaml:20](#)
  - [gemma4_sft2_3gpu_ddp_lora_567.yaml:26](#)

有效训练证据:

- 已进入训练主循环:
  [gemma4_ddp_qlora_3gpu567_20260421_132411.log:940](#)
- loss 非 0 且可持续下降（step1 到 step31）:
  sft2_ddp_lora_3gpu_567_cut3072/trainer_log.jsonl:1
  sft2_ddp_lora_3gpu_567_cut3072/trainer_log.jsonl:31

## 4. 仍存在的已知限制

- FSDP 路径仍有 `SHARD_GRAD_OP` 断言，暂不稳定。
- Gemma4 custom experts 4-bit 路径当前有“忽略 double quantization”的显式告警:
  [moe.py:197](#)
- 4-bit 准备后仍有少量 CPU 参数告警（当前为可接受状态，未再强迁移）:
  [gemma4_ddp_qlora_3gpu567_20260421_132411.log:931](#)

## 5. TODO（下一步）

- [ ] 跑到 step 100/200，输出 loss 曲线统计（MA10/MA50）与吞吐统计
- [ ] 增加固定 eval 集（每 100 step）验证是否出现过拟合/退化
- [ ] 把 FSDP 问题单开最小复现（隔离 patch2/custom experts），避免影响主线 DDP
- [ ] 评估是否把 `cutoff_len=3072` 固化到 4GPU 版本配置
- [ ] 对 `Gemma4 custom 4-bit experts` 增加单元测试（量化后反量化数值误差、训练前后一致性）
- [ ] 评估是否实现 experts 的 double quantization 支持
