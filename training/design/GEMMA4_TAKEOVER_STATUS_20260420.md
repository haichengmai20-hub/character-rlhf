# Gemma4 Loading/Training Takeover Status

时间: 2026-04-20 17:09:57 CST

目的: 记录我正式接手 Gemma4 加载/训练链路前的真实状态，固定当前代码、日志、已知问题和后续处理边界。后续修复完成后，将补第二份变更状态文档。

## 1. 当前目标

当前要解决的问题不是单纯的 ZeRO-3 通信问题，而是:

- Gemma4 在 `transformers 5.5.x + bitsandbytes 4-bit + LlamaFactory` 路径下没有正确完成 4-bit 加载
- 大量参数仍然以普通 `torch.bfloat16 Parameter` 形式停留在 CPU
- 后处理逻辑又尝试把这些残留参数整体搬到 GPU，最终在加载阶段直接 OOM

结论:

- 目前 Gemma4 SFT 训练链路未打通
- 现阶段 blocker 是“量化加载未真正生效”
- ZeRO-2/ZeRO-3/FSDP 只是外层训练封装，根因仍在模型加载阶段

## 2. 当前工作区里已经存在的相关改动

以下文件已经有未提交修改，并且都属于同一条 Gemma4 加载链路:

- `src/llamafactory/model/patcher.py`
- `src/llamafactory/model/loader.py`
- `src/llamafactory/model/model_utils/quantization.py`
- `src/llamafactory/hparams/model_args.py`

这意味着当前代码已经处于“实验中间态”，不是干净基线。

## 3. 当前补丁内容摘要

### 3.1 `patcher.py`

文件: src/llamafactory/model/patcher.py

已经加入 `_patch_bnb_cpu_loading()`:

- patch 1: 重写 `transformers.core_model_loading._materialize_copy`
  - 把原本要 materialize 到 CUDA 的权重先改成落到 CPU
- patch 2: monkey-patch `Bnb4bitQuantize.convert`
  - 试图把 CPU tensor 量化后直接放到当前 CUDA 设备

同时新增 `force_text_only` 针对 Gemma4:

- 在 [patcher.py](#) 附近
- 把 `vision_config` 和 `audio_config` 置空
- 目标是只保留 text tower，避免多模态部分参与初始化

### 3.2 `quantization.py`

文件: src/llamafactory/model/model_utils/quantization.py

当前 4-bit BNB 配置的关键改动:

- `bnb_4bit_quant_storage`:
  - FSDP 下使用 `model_args.compute_dtype`
  - 非 FSDP 下使用 `torch.uint8`

当前注释假设是:

- `transformers 5.x` 对 quantized model 不走真正的 zero.Init 分片
- 如果继续用 bf16 storage，会在 ZeRO-3 路径下爆到每卡 50GB 级别
- 所以尝试用 `uint8` 降低量化存储占用

### 3.3 `loader.py`

文件: src/llamafactory/model/loader.py

当前增加了一段“CPU 参数搬运”后处理:

- 统计 GPU / CPU / meta 上参数量
- 打印前 10 个停留在 CPU 上的参数
- 对所有 CPU 参数执行:
  - `_p.data.to(_cur)`
  - 然后重新包成 `torch.nn.Parameter`

这段逻辑是当前直接 OOM 的触发点。

## 4. 当前最新失败日志

主日志:

- [latest_gemma4.log](#)
- 实际指向: [gemma4_z3_qlora_4gpu_20260420_154852.log](#)

关键证据:

### 4.1 patch2 确实被调用

日志位置:

- [latest_gemma4.log:882](#)

出现:

- `[patch2] first call: model.language_model.layers.0.mlp.down_proj.weight val_dev=cpu new_dev=cuda:*`

说明:

- Claude 打的 BNB convert patch 不是完全没走到
- 至少部分权重的 convert hook 已经触发

### 4.2 但大量参数仍停留在 CPU

日志位置:

- [latest_gemma4.log:870](#)
- [latest_gemma4.log:871](#)

核心诊断:

- `alloc=0.85GB reserved=13.14GB on cuda:0`
- `params: gpu=0.83GB cpu=48.63GB meta_count=0`

前 10 个 CPU 参数样例:

- `model.language_model.embed_tokens.weight`
- `layers.0.self_attn.q_norm.weight`
- `layers.0.self_attn.k_norm.weight`
- `layers.0.input_layernorm.weight`
- `layers.0.router.scale`
- `layers.0.experts.gate_up_proj`

这些参数类型都是普通:

- `dtype=torch.bfloat16`
- `type=Parameter`

这说明:

- 当前并不是“模型已经正常 4-bit，只剩少量 norm/embedding 在 CPU”
- 而是大批权重仍没有进入稳定的 4-bit 参数状态

### 4.3 当前直接炸在 `loader.py` 的 CPU->GPU 强搬运

日志位置:

- [latest_gemma4.log:972](#)
- [loader.py:222](#)

报错:

- `torch.OutOfMemoryError: CUDA out of memory`

rank0 典型错误:

- 尝试再分配 `968 MiB`
- GPU0 实际已使用约 `28.56 GiB`
- PyTorch 已分配约 `27.70 GiB`

rank1/2/3 也同步 OOM。

结论:

- 当前最后一击不是训练 step OOM
- 而是加载完成后的“补搬运”阶段 OOM

## 5. 已有其他 Gemma4 尝试的现状

仓库里目前存在的 Gemma4 路径:

- `run_gemma4_z3_4gpu_nohup.sh`
- `run_gemma4_fsdp_qlora_4gpu_nohup.sh`
- `run_gemma4_ddp_4gpu_nohup.sh`

对应 YAML:

- `gemma4_sft2_z3_4gpu_lora.yaml`
- `gemma4_sft2_4gpu_lora.yaml`
- `gemma4_sft2_ddp_4gpu_lora.yaml`

但从现有日志看:

- ZeRO-3 多次失败
- FSDP 历史上也失败过
- 失败现象仍然是加载期或极早期 OOM

因此当前不能把问题简单归结成:

- “ZeRO-3 不行，换 ZeRO-2 就好了”

更准确的说法是:

- 量化加载链路本身还没被修通

## 6. 当前接手判断

我现在正式接手这条链路，但处理方式会是:

1. 不继续盲目往现有 patch 上叠 patch
2. 先把当前加载失败路径收敛成一个最小可验证问题
3. 核查哪些参数本来就应保持 bf16，哪些本应变成 4-bit 但没有成功转换
4. 再决定 ZeRO-2 / ZeRO-3 / FSDP 哪条训练路径继续推进

## 7. 接手时的边界

本次接手默认由我处理以下部分:

- Gemma4 4-bit 加载逻辑
- Gemma4 text-only 加载分支
- Gemma4 在 LlamaFactory 中的分布式训练入口可行性

本文件只记录接手前状态。

完成修复或形成稳定结论后，将新增第二份文档:

- `GEMMA4_TAKEOVER_RESULT_20260420.md`

用于记录:

- 实际修改内容
- 删除或保留了哪些现有 patch
- 最终跑通/未跑通的训练方案
- 后续仍存在的限制
