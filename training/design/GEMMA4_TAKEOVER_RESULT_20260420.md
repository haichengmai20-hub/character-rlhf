# Gemma4 Loading/Training Takeover Result

时间: 2026-04-20 17:09:57 CST 起接手  
结果记录时间: 2026-04-20 17:10+ CST

对应接手前状态文档:

- [GEMMA4_TAKEOVER_STATUS_20260420.md](GEMMA4_TAKEOVER_STATUS_20260420.md)

## 0. 当前 deep dive TODO

### 2026-04-21 代码侧修复 TODO

- [x] 区分“历史训练 loss 长期为 0”和“最新 `latest_gemma4.log` 报错”是否是同一个问题
- [x] 复查 Claude 后续叠加到 `loader.py` 的训练期干预逻辑
- [x] 移除不是 Gemma4 MoE 量化必需、但会改变训练行为的高风险逻辑
- [x] 保留已经验证有效的 Gemma4 专属修复：
  - 自定义 MoE experts 4-bit 量化
  - `mm_token_type_ids`
  - Gemma4 turn token 模板修复
  - BNB CPU-first loading patch
  - trainer `create_optimizer(model=None)` 兼容
- [ ] 等 GPU 空闲后做 1-2 step smoke test，确认 loss 非 0、step time 合理、无新 runtime 报错
- [ ] 再决定是否恢复更激进的训练期优化或额外诊断

- [x] 确认 Gemma4TextExperts 的真实参数表示和 forward 形态
- [x] 确认标准 BNB 4-bit 路径为什么不会覆盖 Gemma4 experts
- [x] 验证 `Params4bit` 对 3D 张量、索引、`chunk/split` 的行为边界
- [x] 验证 `QuantState` 是否可以按 expert 手工切片并单独反量化
- [x] 评估当前最小可行实现路线和明显不可行路线
- [x] 用 GPU 做 Gemma4 experts 4-bit 最小数值验证
- [x] 先做 Gemma4 experts 自定义 4-bit forward 原型
- [x] 把原型接到 LlamaFactory 的 model loading 路径
- [x] 做独立模块级 GPU 验证
- [x] 用真实 Gemma4 checkpoint 做整模加载验证
- [ ] 再决定是否继续打通完整 SFT/ZeRO-2/ZeRO-3 训练链路

## 1. 本轮接手结论

本轮没有把 Gemma4 训练直接跑通，但已经把问题从“加载期 CUDA OOM”收敛成了更具体、可验证的根因：

- **Gemma4 的 MoE expert 权重不是标准 `nn.Linear.weight`**
- 而是 `Gemma4TextExperts` 里的 3D 裸 `nn.Parameter`
- Hugging Face + bitsandbytes 的标准 4-bit on-the-fly 路径只会处理线性层量化
- 因此这批 expert 权重不会自动变成 `Params4bit`
- 它们会以 `bf16 Parameter` 形式残留在 CPU
- 之前代码再尝试把这些残留参数整体搬到 GPU，最后表现为“看起来像 patch 失败”的 OOM

换句话说：

- 当前主问题不再是单纯的 ZeRO-3
- 而是 **Gemma4 的 MoE 参数表示方式与 BNB 4-bit 标准加载路径不兼容**

### 2026-04-21 补充结论

今天重新核对后，Gemma4 当前需要拆成两个独立问题看：

1. 历史 `trainer_log.jsonl` 里确实存在一段明显异常的训练记录
   - 多个 step 的 `loss=0.0`
   - step time 偏短
   - 这更像训练行为被额外 hook/monkey-patch 干扰后的伪成功，而不是正常收敛

2. 当前最新 [latest_gemma4.log](#) 里的报错不是这个问题
   - 最新日志是在模型加载阶段失败
   - 直接原因是 `transformers` 的 `caching_allocator_warmup()` 在 GPU0 上申请 `12.21 GiB`
   - 当时 GPU0 只剩 `1.76 GiB` 可用，因为别的进程已占用约 `29 GiB`
   - 所以这次最新报错本质上是**占卡导致的加载期 OOM**，不是新的 Gemma4 代码根因

因此，2026-04-21 的代码侧修复目标是：

- 先回退会改变训练行为的高风险逻辑
- 保留已经证明是必需的 Gemma4 兼容与量化路径
- 等卡空闲后再做最小训练验证

## 2. 新确认的根因证据

### 2.1 最新日志里的 CPU 残留参数

日志:

- [latest_gemma4.log](#)

关键诊断:

- `params: gpu=0.83GB cpu=48.63GB meta_count=0`

前几个残留参数包括:

- `model.language_model.embed_tokens.weight`
- `model.language_model.layers.0.router.scale`
- `model.language_model.layers.0.router.per_expert_scale`
- `model.language_model.layers.0.experts.gate_up_proj`

其中最关键的是:

- `experts.gate_up_proj`
- `experts.down_proj`

它们不是 BNB 会自动替换的线性层权重，而是 Gemma4 自己定义的裸参数。

### 2.2 HF 源码确认 Gemma4 experts 是裸参数

在 `transformers` 里，Gemma4 experts 的实现是:

- 文件: `/home/ps/miniconda3/envs/gemma4-character/lib/python3.12/site-packages/transformers/models/gemma4/modeling_gemma4.py`

关键定义:

- `self.gate_up_proj = nn.Parameter(...)`
- `self.down_proj = nn.Parameter(...)`

而不是:

- `nn.Linear(...)`

这意味着标准 `bitsandbytes` 4-bit 线性层替换逻辑不会覆盖它们。

### 2.3 参数量级验证

Gemma4 text 配置当前是:

- `hidden_size = 2816`
- `num_hidden_layers = 30`
- `num_experts = 128`
- `moe_intermediate_size = 704`
- `enable_moe_block = True`

按这套配置估算，单层 experts 的 bf16 参数量约:

- `~1.42 GB / layer`

30 层总计约:

- `~42.54 GB bf16`

这个量级与日志里观察到的 `48.63GB CPU residual` 高度吻合。  
因此可以确认：

- **残留在 CPU 上的大头，正是 Gemma4 的 MoE expert 裸参数**

## 3. 本轮实际代码修改

### 修改文件

- src/llamafactory/model/patcher.py
- src/llamafactory/model/model_utils/moe.py
- src/llamafactory/model/loader.py

### 修改内容

本轮已经不只是诊断，还落了第一版 Gemma4 experts 自定义 4-bit 原型。

#### 3.1 `moe.py`

新增了这几部分:

- `gemma4_bnb4_eager` 自定义 experts implementation
- Gemma4 experts 的 4-bit packed 表示构造逻辑
- active expert 的子 `QuantState` 重建逻辑
- 自定义 4-bit experts forward
- `prepare_gemma4_bnb4_experts(model, model_args)` 后处理入口

当前原型的设计点:

- 只针对 `Gemma4TextExperts`
- 只处理:
  - `gate_up_proj`
  - `down_proj`
- 加载后把原始 raw `nn.Parameter` 移除
- 改成:
  - `*_qweight`
  - `*_absmax`
  - `*_quant_map`
  这些 buffer + 若干元信息属性
- 然后把 `config._experts_implementation` 切到:
  - `gemma4_bnb4_eager`

当前限制:

- experts 这部分暂时**忽略 double quantization**
- 即使全局配置 `double_quantization=True`
- 当前 experts 原型仍然使用单层 `nf4/fp4 + absmax`

这是一个有意的简化，用来先验证结构路线，而不是最终优化版本。

#### 3.2 `loader.py`

在 `load_class.from_pretrained(...)` 之后、CPU residual 诊断之前，新增了:

- 对 Gemma4 + 4-bit 的 `prepare_gemma4_bnb4_experts(...)` 调用

这样如果原型工作正常:

- Gemma4 experts 会先被转成自定义 4-bit packed 表示
- 再进入后续的 CPU residual 检查

### 3.5 2026-04-21 对 `loader.py` 的进一步收敛

今天额外移除了三段 Claude 后续叠加的高风险训练期逻辑：

1. 标准 `bnb Linear4bit` 的自定义 `forward` 重写
2. 加载后把残余 CPU 参数隐式迁移到 GPU 的逻辑
3. 首个训练 forward 的 one-shot monkey-patch 诊断

这样做的理由：

- 这三段逻辑都不是 Gemma4 MoE 自定义量化必需条件
- 但它们都会直接改变训练期行为
- 在历史 `loss=0.0` 的可疑现象下，这三段逻辑属于优先应当移除的干扰项

当前 `loader.py` 的策略已经收敛成：

- 只负责 Gemma4 custom MoE 4-bit 准备
- 只做只读诊断
- 如果 4-bit 准备后仍有 CPU residual，则显式失败
- **不再隐式修改训练 forward 路径，也不再偷偷迁移参数**

#### 3.3 `patcher.py`

我还把原来的 CPU-first materialization patch 触发范围扩大了。

之前:

- 这个 patch 只在 `ZeRO-3 / FSDP + 4-bit` 场景触发

现在:

- 对 **Gemma4 + 4-bit + trainable** 也会触发

原因很直接:

- 如果不先把 `core_model_loading._materialize_copy` 改成 CPU-first
- 单卡 Gemma4 4-bit 在 `from_pretrained(...)` 阶段就会先因为 bf16 materialization OOM
- 根本轮不到后面的 custom experts hook 执行

#### 3.4 之前已有的 loader 诊断保护仍保留

我没有移除之前加的“禁止把几十 GB residual bf16 experts 直接搬上 GPU”的保护。

原逻辑:

- 发现 CPU 残留参数后，尝试把所有残留参数 `.to(cuda)` 再重新包成 `nn.Parameter`

问题:

- 对 Gemma4 这种残留了几十 GB bf16 expert 权重的模型，这一步只会制造一个泛化的 CUDA OOM
- 会掩盖真正问题: 这些参数本来就没有被 4-bit 量化

现在改成:

- 保留详细诊断日志
- 如果检测到大规模 CPU residual，或者残留参数名中包含:
  - `.experts.gate_up_proj`
  - `.experts.down_proj`
- 则直接抛出清晰的 `RuntimeError`
- 明确指出: Gemma4 的 MoE expert 是 raw `nn.Parameter`，不走标准 BNB 4-bit conversion 路径

这样后续再跑时，不会再得到一个误导性的“再分配 484MB / 968MB OOM”，而会直接暴露真实 blocker。

## 4. 本轮没有改动的东西

本轮没有继续修改以下逻辑:

- `patcher.py` 中 Claude 已加的 CPU-first materialize patch
- `Bnb4bitQuantize.convert` monkey patch
- `quantization.py` 里的 `bnb_4bit_quant_storage` 选择逻辑

原因:

- 当前已经确认更底层的问题不是“convert hook 完全没跑”
- 而是 Gemma4 有大量不属于线性层的 expert 权重，标准 BNB 4-bit 路径本来就不覆盖
- 在这个前提下继续微调 convert patch，不会真正解决主要残留权重

## 5. 当前判断

当前可以明确下结论:

1. **Gemma4 不是一个“简单换 ZeRO-2 就能跑”的问题**
   - 训练方式只是外层包装
   - 主 blocker 仍然是 4-bit 加载本身

2. **当前 HF/BnB 标准 QLoRA 路径不直接支持 Gemma4 这类 MoE raw-Parameter experts**
   - 至少在你当前这套 `transformers 5.5.4 + bitsandbytes 0.49.2 + LlamaFactory` 组合里是这样

3. **如果要真正让 Gemma4 走 4-bit 训练，后续需要的是“自定义 expert quantization 路径”**
   - 而不是继续在当前 loader 尾部做参数搬运

## 5.1 本轮 deep dive 新结论

在继续深挖 `Gemma4TextExperts` 和 `bitsandbytes` 之后，现在可以把“自定义 expert quantization 路径”收敛得更具体一些。

### A. `Params4bit` 本身不排斥 3D 张量

使用 `bitsandbytes.functional.quantize_4bit()` 和 `bnb.nn.Params4bit` 的最小实验表明:

- 3D 张量可以被量化
- `QuantState.shape` 也会记录原始 3D 形状

因此，Gemma4 experts 的 3D 权重 **不是因为维度是 3D 就完全不能量化**。

### B. 真正的问题是 Gemma4 当前 forward 依赖“按 expert 取 2D 切片”

Gemma4 当前 eager forward 是:

- `self.gate_up_proj[expert_idx]`
- `self.down_proj[expert_idx]`

再把得到的 2D 权重送进 `nn.functional.linear(...)`。

而 `Params4bit` 的实验结果是:

- 普通索引 `p[expert_idx]` 会直接返回普通 `Tensor`
- 这个切片 **不会保留 `quant_state`**
- `torch.chunk` / `torch.split` 虽然会保留 `Params4bit` 包装，但挂的仍然是原始整块的 `quant_state`

这意味着:

- **不能直接把 `gate_up_proj / down_proj` 换成一个 3D `Params4bit`，然后保持 Gemma4 当前 eager forward 不变**

### C. `QuantState` 可以按 expert 手工切片

我做的最小实验还确认了另一件关键事实:

- 对一个 3D 量化张量
- 其 packed weight 和 `QuantState.absmax` 都可以按 expert 的连续块手工切片
- 用切出来的子 `QuantState` 调 `dequantize_4bit(...)`
- 可以正确恢复出单个 expert 的 2D 权重

这说明:

- **Gemma4 experts 做自定义 4-bit 前向在结构上是可行的**
- 但需要我们自己维护“每个 expert 对应哪一段 packed weight / absmax”

### D. `grouped_mm` 路线比“直接复用当前 eager 索引”更有希望

HF 的 `@use_experts_implementation` 装饰器和 `transformers.integrations.moe.ALL_EXPERTS_FUNCTIONS` 提供了一个明确扩展点:

- 当前内置实现只有:
  - `eager`
  - `batched_mm`
  - `grouped_mm`

其中:

- `eager` 路线按 expert 直接索引 2D 权重，和 3D `Params4bit` 天然不兼容
- `grouped_mm` 路线保留整块 3D 权重，更接近可扩展形态

但现在的 `grouped_mm` / `batched_mm` 实现仍然假设权重是普通 dense tensor，没有任何 4-bit kernel 路径，所以:

- **可扩展点存在**
- **但现成实现不能直接拿来用**

### E. 这个扩展点可以从项目侧接入，不必 fork 整个 Gemma4 模型类

继续核对 `transformers` 配置和接口后，现在可以更明确地说:

- `config._experts_implementation` / `experts_implementation=...` 是框架原生支持的配置项
- `ExpertsInterface.register(key, fn)` 可以全局注册新的 experts forward 实现

这意味着如果后面要做 Gemma4 experts 自定义 4-bit forward，较合理的项目接入方式是:

1. 在项目侧注册一个新的 experts implementation
2. 在加载后把 Gemma4 experts 变成自定义 packed 4-bit 表示
3. 再把 `config._experts_implementation` 切到新的实现

因此后续实现不一定要:

- 修改 Hugging Face 的 Gemma4 主类定义
- 或者长期维护一份完整 fork 的 `modeling_gemma4.py`

这点很重要，因为它显著降低了后续改造的侵入性。

### F. GPU 最小验证结果: 这条路线数值上可接受

我补做了一轮 GPU 侧最小实验，使用:

- `GPU 1`
- 真实 Gemma4 experts 尺度
  - `hidden = 2816`
  - `intermediate = 704`
  - `gate_up = (num_experts, 1408, 2816)`
  - `down = (num_experts, 2816, 704)`
- `nf4`
- `blocksize = 64`
- 权重初始化尺度取接近真实训练初始值的小方差 (`std = 0.02`)

比较了两条路径:

1. **Path A**
   - 每个 expert 的 2D 权重独立量化
   - forward 时直接反量化该 expert

2. **Path B**
   - 整块 3D experts 权重一次量化
   - forward 时按 expert 手工切 packed weight 和 `QuantState.absmax`
   - 再构造子 `QuantState` 反量化该 expert

结果:

- `Path A` 和 `Path B` 的单 expert 反量化权重 **完全一致**
  - `A_vs_B_gate_max = 0.0`
  - `A_vs_B_down_max = 0.0`

也就是说，在当前 BNB blockwise layout 下:

- **“整块 3D 量化后按 expert 切片”不会引入额外误差**
- 至少对当前这组 Gemma4 尺度张量，它与 per-expert 独立量化是数值等价的

单 expert forward 相对 bf16 baseline 的输出相似度:

- `ref_abs_mean = 0.3155`
- `out_abs_mean = 0.3138`
- `diff_abs_mean = 0.0508`
- `diff_abs_max = 0.2656`
- `rmse = 0.0638`
- `cosine = 0.9870`
- `corrcoef = 0.9870`

这里最重要的不是逐元素相对误差，因为当输出接近 0 时相对误差会被放大；更有参考价值的是:

- 输出量级与 bf16 baseline 非常接近
- cosine / corr 都接近 `0.99`

所以目前可以更明确地说:

- **Gemma4 experts 自定义 4-bit forward 路线在 GPU 上是数值可行的**
- 当前剩下的是工程落地问题，不是“这条路线会不会天然数值崩坏”的问题

### G. 新原型已经通过独立模块级 GPU 验证

在这轮代码修改后，我又做了一次命中新 helper 的独立模块验证，验证对象不是手写仿真模块，而是:

- 实际 `transformers.models.gemma4.modeling_gemma4.Gemma4TextExperts`

验证方式:

1. 先用 `Gemma4TextExperts` 原始 dense 参数跑一次 forward
2. 调用 `prepare_gemma4_bnb4_experts(...)`
3. 确认模块被转换成 custom 4-bit experts
4. 再用相同输入走 custom forward

验证结果:

- `converted = 1`
- `impl = gemma4_bnb4_eager`
- `quantized_flag = True`
- 原始 `gate_up_proj` 参数已经移除
- `gate_up_proj_qweight` / `down_proj_qweight` 已存在
- 这次独立模块测试里，转换前后输出:
  - `diff_mean = 0.0`
  - `diff_max = 0.0`

这个验证说明:

- 新的项目侧 hook 已经真正命中
- Gemma4 experts 的 raw parameter → custom 4-bit packed 表示的转换链路可以跑通
- 自定义 experts implementation 也已经被实际调用

### H. 真实 Gemma4 checkpoint 单卡加载 smoke test 已成功

我随后又跑了一次更接近真实链路的验证，不再是独立模块，而是:

- 实际 Gemma4 checkpoint
- 实际 LlamaFactory `load_tokenizer(...) + load_model(...)`
- 实际 YAML:
  - [gemma4_sft2_1gpu_lora.yaml](#)
- `GPU 1`

结果:

- `from_pretrained(...)` 已成功完成
- CPU-first materialization patch 已命中
- custom experts hook 已命中
- `Prepared Gemma4 custom 4-bit experts for 30 layer(s)`
- 最终 `MODEL_LOAD_OK = PeftModelForCausalLM`
- `experts_impl = gemma4_bnb4_eager`
- `converted_layers = 30`
- `has_any_qweight = True`

这说明当前已经可以明确说:

- **真实 Gemma4 checkpoint 的单卡 4-bit 加载链路，已经被这套 patch 打通**

同时，这次 smoke test 还给出了新的 residual 诊断:

- custom experts 转换完成后:
  - `gpu = 0.83GB`
  - `cpu = 2.95GB`

这部分 CPU residual 已经不再是 48GB 级别的 experts 大头，而是一些小参数，例如:

- `embed_tokens.weight`
- `q_norm.weight`
- `k_norm.weight`
- `layernorm.weight`
- `router.scale`
- `router.per_expert_scale`

也就是说:

- 当前最主要的 Gemma4 MoE raw-parameter blocker 已经被拿掉
- 剩下的是“还有少量 bf16 参数留在 CPU”，但它们规模已经降到可接受范围，不再阻止整模加载

## 5.2 当前最小可行实现路线

如果继续往前做，我目前认为最低风险的路线是:

1. 保留 Gemma4 现有模块结构，不去改 HF 大模型主干
2. 仅针对 `Gemma4TextExperts` 做自定义后处理
3. 在加载后把:
   - `gate_up_proj`
   - `down_proj`
   转成自定义 4-bit packed 表示
4. 不再依赖标准 `Params4bit[expert_idx]`
5. 注册一个新的 experts implementation，例如:
   - `gemma4_bnb4_eager`
   - 或 `gemma4_bnb4_grouped`
6. 在 forward 里:
   - 按 active expert 取 packed slice
   - 手工构造该 expert 的子 `QuantState`
   - 对单个 expert 做反量化
   - 再执行当前的 expert matmul

这条路线的额外好处是:

- 可以把真正的结构改动限制在 `Gemma4TextExperts` 这一个局部
- 不需要修改 Gemma4 attention / router / decoder 层主干
- GPU 验证也已经说明这条路线没有明显的额外数值劣化风险

这个路线的前提是:

- experts 保持冻结
- 我们只需要梯度回传到输入 hidden states 和 LoRA adapter
- 不需要对 quantized expert 权重本身求梯度

对当前 SFT/QLoRA 目标来说，这个前提是成立的。

## 5.3 当前不建议走的路线

下面这些路线现在看都不合适:

1. **继续补 `Bnb4bitQuantize.convert` 小 patch**
   - 这只对标准线性层有效
   - 解决不了 raw expert parameter

2. **把 experts 改写成海量 `Linear4bit` 子模块**
   - 30 层 × 128 experts × 2 projection，子模块数太大
   - 加载、映射、保存、兼容性成本都过高

3. **继续押 ZeRO-3 / FSDP 作为首个目标**
   - 即使 experts 自定义量化做通了
   - ZeRO-3/FSDP 仍然会叠加新的 state/partition 问题
   - 第一优先级应该是先在 DDP 或 ZeRO-2 上验证 Gemma4 experts 自定义 4-bit 路线本身

## 6. 下一步建议

后续如果继续修 Gemma4，有两条路：

### 路线 A：继续攻 Gemma4 当前栈

方向:

- 针对 `Gemma4TextExperts.gate_up_proj / down_proj` 设计自定义量化与 forward 路径
- 优先目标不是 ZeRO-3，而是先做:
  - [x] experts packed 4-bit 表示
  - [x] active-expert 子 `QuantState` 切片
  - [x] 自定义 experts forward
  - [x] 项目侧 loading hook 接入
  - [x] 真实 Gemma4 checkpoint 可加载验证
  - [ ] DDP / ZeRO-2 可加载验证

代价:

- 这是结构级支持，不是小 patch
- 改动面会显著大于当前 LlamaFactory patch

### 路线 B：承认当前栈不支持 Gemma4 QLoRA

方向:

- 停止继续在当前 HF/BnB 路径上做局部 patch
- 如果短期要推进 RL 或 SFT 生产链路，使用已跑通的 Qwen3 方案

代价:

- 与线上 Gemma4 不直接对齐

## 7. 当前状态一句话版

本轮接手已经把问题从“Gemma4 ZeRO-3 / FSDP / DDP 都 OOM”推进到:

> Gemma4 的核心 MoE expert 权重虽然不属于 bitsandbytes 标准 4-bit 线性层转换覆盖范围，但项目侧已经实现了自定义 experts 4-bit packed 表示、自定义 experts forward 和加载后转换 hook；真实 Gemma4 checkpoint 的单卡 4-bit 加载已经打通，当前剩下的是继续验证 DDP / ZeRO-2 / ZeRO-3 训练链路，而不再是最初的 raw expert loading blocker。

## 8. 4-GPU FSDP Bring-Up 准备

在启动 4 卡训练前，先做了两项对齐：

1. `gemma4_sft2_4gpu_lora.yaml` 改为使用已经 smoke test 通过的 checkpoint：
   - `/home/ps/hanjiaqi/LlamaFactory/chatjoy/models/gemma4-26b-a4b-it-textonly`
2. 把 4 卡 YAML 的加载相关设置对齐单卡成功配置：
   - `low_cpu_mem_usage: false`
   - `flash_attn: disabled`

这样做的原因是：

- 旧 4 卡 YAML 仍指向较早的 HF cache snapshot，不是这轮 custom experts 4-bit 验证所使用的 text-only checkpoint
- bring-up 阶段优先减少变量，先验证：
  - FSDP 多卡加载是否能稳定走完
  - custom experts conversion 是否在多 rank 下都命中
  - 是否还会出现旧的 48GB CPU residual / `.to(cuda)` OOM 故障模式

下一步执行：

- 在 `GPU0,1,2,3` 上直接启动 `run_gemma4_fsdp_qlora_4gpu_nohup.sh`
- 检查 early log，重点看：
  - FSDP 初始化是否正常
  - `Prepared Gemma4 custom 4-bit experts for 30 layer(s)` 是否出现
  - 是否进入第一个 train step，而不是停在加载/OOM

## 9. 4-GPU FSDP + 4-bit 启动状态（2026-04-20 17:54 CST）

已执行：

- `CUDA_VISIBLE_DEVICES=0,1,2,3 bash chatjoy/run_gemma4_fsdp_qlora_4gpu_nohup.sh`

启动产物：

- session: `/home/ps/hanjiaqi/LlamaFactory/chatjoy/logs/gemma4_fsdp_qlora_4gpu_20260420_175447.session`
- log: `/home/ps/hanjiaqi/LlamaFactory/chatjoy/logs/gemma4_fsdp_qlora_4gpu_20260420_175447.log`
- latest log: `/home/ps/hanjiaqi/LlamaFactory/chatjoy/logs/latest_gemma4.log`

当前观察到的状态：

- 4 rank 训练进程已经拉起，`accelerate launch` 正常存在
- 训练没有在启动后立刻触发旧的 `.to(cuda)` OOM
- 当前日志停留在数据预处理阶段：
  - `Loading dataset ... clean32_for_sft2_llamafactory.json`
  - `Running tokenizer on dataset (num_proc=16)`
- 当前显存占用仍较低，说明还未进入真正的模型加载/训练阶段

因此这一步的结论是：

- **4 卡 FSDP 任务已成功挂起**
- **当前还不能宣布“4 卡训练链路完全跑通”**
- 还需要继续观察是否在后续模型加载阶段出现 FSDP 特有问题，尤其是：
  - custom experts 4-bit conversion 是否在多 rank 下命中
  - 是否重新出现大规模 CPU residual
  - 是否在第一个 train step 前报错

## 10. 本次 4-GPU FSDP 首次失败的真实根因

4 卡任务随后失败，但失败点已经从“量化/加载”推进到了 **PEFT + FSDP auto-wrap**：

- custom experts conversion 已成功命中：
  - `Prepared Gemma4 custom 4-bit experts for 30 layer(s) with experts_implementation=gemma4_bnb4_eager`
- residual 诊断也保持正常：
  - `gpu=0.83GB`
  - `cpu=2.95GB`
- 首个真实异常不是 OOM，而是：
  - `Exception: Could not find the transformer layer class to wrap in the model.`

原因分析：

- `Gemma4PreTrainedModel._no_split_modules` 包含：
  - `Gemma4TextDecoderLayer`
  - `Gemma4VisionEncoderLayer`
  - `Gemma4AudioLayer`
- PEFT 的 `fsdp_auto_wrap_policy()` 会把这些名字全部拿来解析
- 当前运行配置是 `force_text_only`
- 因此 text-only 模型里没有：
  - `Gemma4VisionEncoderLayer`
  - `Gemma4AudioLayer`
- PEFT 遇到任意一个找不到的类名就直接抛异常，而不是跳过

因此当前可执行修复是：

- 在 4 卡 FSDP 启动脚本里显式设置：
  - `FSDP_TRANSFORMER_CLS_TO_WRAP=Gemma4TextDecoderLayer`

这样会把 FSDP auto-wrap 范围收窄到 text-only Gemma4 实际存在的 decoder layer，避免再次因为 vision/audio 类缺失而报错。

## 11. 加入 wrap-class 修复后的二次启动结果

已在启动脚本中加入：

- `export FSDP_TRANSFORMER_CLS_TO_WRAP=Gemma4TextDecoderLayer`

随后重新启动 4 卡任务，新的 session 为：

- `/home/ps/hanjiaqi/LlamaFactory/chatjoy/logs/gemma4_fsdp_qlora_4gpu_20260420_175808.session`

当前结果：

- 旧错误
  - `Could not find the transformer layer class to wrap in the model.`
  - **没有再次出现**
- `accelerate launch` 仍在运行
- 4 个 `src/train.py` rank 进程都仍然存活
- 日志当前已重新进入：
  - `Loading dataset /home/ps/hanjiaqi/LlamaFactory/chatjoy/data/runs/clean32_for_sft2_llamafactory.json...`

这说明：

- 本次修复已经拿掉了上一次 4 卡 FSDP 的首个真实 blocker
- 当前任务已经成功越过 PEFT/FSDP auto-wrap 失败点
- 下一步继续观察的重点重新回到：
  - 模型加载阶段是否稳定
  - custom experts 4-bit conversion 是否在多 rank 下继续命中
  - 是否能真正进入第一个 train step

## 12. Trainer API 兼容性修复

在继续往下跑时，4 卡任务又暴露出一个独立问题：

- `TypeError: CustomSeq2SeqTrainer.create_optimizer() takes 1 positional argument but 2 were given`

原因：

- 当前 `gemma4-character` 环境中的 `transformers` 基类签名已经是：
  - `create_optimizer(self, model=None)`
- 而项目内多个自定义 trainer override 仍使用旧签名：
  - `create_optimizer(self)`
- 当上游把 `model` 位置参数传入时，就会在本地 override 这里直接报 `TypeError`

已修复文件：

- `src/llamafactory/train/sft/trainer.py`
- `src/llamafactory/train/rm/trainer.py`
- `src/llamafactory/train/dpo/trainer.py`
- `src/llamafactory/train/kto/trainer.py`
- `src/llamafactory/train/pt/trainer.py`

修复方式：

- 统一改为兼容新签名：
  - `create_optimizer(self, model=None)`
- 优先使用传入的 `model` 创建自定义优化器
- 再调用：
  - `super().create_optimizer(model)`

这个修复不是只为 Gemma4/FSDP 服务，而是一次性补齐当前项目对新版 Transformers Trainer API 的兼容性。

## 13. 加入 Trainer API 兼容修复后的三次启动结果

在完成 `create_optimizer(self, model=None)` 兼容修复后，再次重启 4 卡任务：

- session:
  - `/home/ps/hanjiaqi/LlamaFactory/chatjoy/logs/gemma4_fsdp_qlora_4gpu_20260420_180557.session`

当前观察到的结果：

- 旧错误
  - `TypeError: CustomSeq2SeqTrainer.create_optimizer() takes 1 positional argument but 2 were given`
  - **没有再次出现**
- `accelerate launch` 仍在运行
- 4 个 `src/train.py` rank 进程都仍然存活
- 任务已重新进入数据/配置加载阶段，尚未再次在旧的 Trainer API 位置报错

这说明：

- `create_optimizer` 的接口不兼容问题已经按项目级别一次性修掉
- 这不是只修 SFT 一处，而是把当前项目内同类 trainer override 全部对齐到了新版 Transformers 的调用方式

## 14. 首个 train step 设备断言的真实根因与修复

在继续往下跑后，4 卡任务首次进入真实 train step，但新的失败点是：

- `torch.AcceleratorError: CUDA error: device-side assert triggered`
- 底层断言来自：
  - `vectorized_gather_kernel index out of bounds`
- 出错位置落在：
  - `Gemma4TextScaledWordEmbedding.forward`
  - 即 `embed_tokens(input_ids)` 阶段

这说明不是 attention / MoE / FSDP shard 本身出错，而是：

- 有 token id 超出了 embedding 行数范围

结合日志与模板实现核对后，根因是：

- 项目里的 `gemma4` 模板沿用了旧的：
  - `<start_of_turn>`
  - `<end_of_turn>`
- 但当前 Gemma4 tokenizer 原生 turn 标记实际是：
  - `<|turn>` (id = 105)
  - `<turn|>` (id = 106)
- 旧模板在训练启动时会把不匹配的 turn 字符串当作新 special token 处理
- 对量化模型来说，embedding 不能安全 resize
- 最终表现就是首个 train step 中某些 token id 超出 `embed_tokens` 的索引范围

已执行修复：

- 仅修改 `gemma4` 模板，不影响其他模板：
  - `src/llamafactory/data/template.py`
- 把 `gemma4` 的 turn 标记替换为 tokenizer 原生 token：
  - `<start_of_turn>` -> `<|turn>`
  - `<end_of_turn>` -> `<turn|>`
- `stop_words` 也同步改成：
  - `["<turn|>"]`

修复后验证结果：

- tokenizer 长度保持：
  - `len = 262144`
  - `vocab_size = 262144`
- 不再出现新增的 `<end_of_turn>` token
- 原生 turn token 仍是：
  - `turn_open = 105`
  - `turn_close = 106`

这次修复是 Gemma4 模板专属修复，不会改动其他模型模板或其他已正常运行的训练路径。

## 15. `mm_token_type_ids` 缺失的真实根因与修复

在继续往下跑并越过 embedding 越界后，4 卡任务进入首个真实训练前向，又暴露出新的 Gemma4 专属接口问题：

- `ValueError: mm_token_type_ids is required as a model input when training`

根因不是数据本身缺字段，而是：

- 当前 Transformers `Gemma4Processor` / `Gemma4Model` 接口使用的是：
  - `mm_token_type_ids`
- 但 LlamaFactory 的现有多模态整理逻辑仍沿用旧字段名：
  - `token_type_ids`
- 结果就是：
  - `mm_plugin` 已经算出了 token-type 信息
  - 但 `collator` 在进入 batch 时仍把它塞成 `token_type_ids`
  - `Gemma4` 训练态 forward 看不到 `mm_token_type_ids`，于是直接报错

已执行修复：

- `src/llamafactory/data/collator.py`
  - 当 `model.config.model_type == "gemma4"` 时：
    - 把 `mm_inputs["token_type_ids"]` 桥接到 `feature["mm_token_type_ids"]`
  - 对其他模型继续保留原行为：
    - `feature["token_type_ids"]`
- `src/llamafactory/train/kto/trainer.py`
  - 前向构造 `model_inputs` 时优先读取：
    - `mm_token_type_ids`
  - 若不存在，再回退到：
    - `token_type_ids`

这样做的影响范围是受控的：

- Gemma4 训练路径获得它实际要求的字段名
- 非 Gemma4 模型保持原有 `token_type_ids` 路径，不会被这次修复改坏
- KTO 额外补了兼容，避免后续 Gemma4 偏好训练再踩同样问题

## 16. 进入首个真实前向后的新 blocker：FSDP 配置实际仍是 ZeRO-3

在补齐 `mm_token_type_ids` 后，4 卡任务已经能进入首个真实前向，新的失败点变成：

- `bitsandbytes/autograd/_functions.py`
- `assert quant_state is not None`

触发位置在标准 attention 线性层：

- `self.q_proj(hidden_states)`
- `peft.tuners.lora.bnb`
- `bnb.matmul_4bit(...)`

这说明：

- `mm_token_type_ids` 这条链已经打通
- 当前 batch 已经走进第一层 attention
- 新的 blocker 不是数据字段，而是 **FSDP 包裹后的标准 `Linear4bit` 权重丢了 `quant_state`**

进一步核对后确认，之前 Gemma4 启动脚本实际仍在用：

- `chatjoy/accelerate_fsdp_4gpu.yaml`
- 其中：
  - `fsdp_sharding_strategy: FULL_SHARD`

也就是 **ZeRO-3 语义**，不是本轮要验证的 “FSDP(ZeRO-2)”。

这与当前报错是一致的：

- `FULL_SHARD` 会对父层参数做更激进的 sharding / gather
- 标准 `bnb.nn.Linear4bit` 在这种路径下更容易丢失运行时依赖的 `quant_state`
- 对 Gemma4 当前这条 4-bit 训练链来说，这个问题发生在标准 attention/projection 层，还不是自定义 MoE experts

因此这一步的结论是：

- **之前的 FSDP 配置选错了 stage**
- 当前不是 “FSDP(ZeRO-2) 也不行”，而是 “我们实际上还在跑 FSDP ZeRO-3”

已执行修复：

- 新增 Gemma4 专用配置：
  - `chatjoy/accelerate_fsdp_zero2_4gpu.yaml`
- 核心改动：
  - `fsdp_sharding_strategy: SHARD_GRAD_OP`
  - `fsdp_offload_params: false`
- 启动脚本：
  - `chatjoy/run_gemma4_fsdp_qlora_4gpu_nohup.sh`
  - 已切到新的 `accelerate_fsdp_zero2_4gpu.yaml`

这样做的作用域是受控的：

- 只影响 Gemma4 当前这条 4 卡 FSDP 启动链
- 不改动原有 `accelerate_fsdp_4gpu.yaml`
- 其他已经正常的 FSDP 任务配置不会被这次修复连带改坏

## 17. 当前 ToDo

- [x] 修复 Gemma4 turn token 模板不匹配
- [x] 修复 Trainer `create_optimizer(model=None)` 兼容性
- [x] 补齐 Gemma4 训练态 `mm_token_type_ids`
- [x] 确认首个真实前向已越过 `mm_token_type_ids` 报错
- [x] 识别当前 FSDP 配置实际上仍是 `FULL_SHARD`
- [x] 单独切出 Gemma4 专用 `FSDP SHARD_GRAD_OP` 配置
- [ ] 重启 4 卡任务验证 `Linear4bit quant_state` 断言是否消失
- [ ] 确认能稳定进入第一个 train step
- [ ] 确认 loss 开始更新，不再只停在前向初始化

## 18. `Linear4bit quant_state` 断言的更直接根因

在把 FSDP 配置切到 `SHARD_GRAD_OP` 后再次验证，`assert quant_state is not None` 依然存在。

这说明：

- 仅仅把 stage 从 `FULL_SHARD` 改为 `SHARD_GRAD_OP` 还不够
- 更直接的问题在于 **标准 `bnb.nn.Linear4bit` 层的量化态初始化本身不完整**

进一步检查 bitsandbytes 源码后确认：

- `Params4bit(...).to(cuda)` 在量化完成后，会把量化态回填到：
  - `self.quant_state`
  - 以及 `self.module.quant_state`（前提是 `module` 被正确传入）
- 我们当前的 BNB CPU-first patch (`_convert_force_cuda`) 在构造标准 4-bit 权重时：
  - 创建了 `Params4bit(value, ...)`
  - 但**没有把所属 `Linear4bit` 模块对象传给 `module=`**

这会导致：

- 权重虽然被量化了
- 但 `Linear4bit.quant_state` 不会被同步设置
- 后续一旦 `weight.quant_state` 因参数转换 / FSDP 包装丢失
- bitsandbytes 的恢复逻辑就找不到 `module.quant_state`
- 最终在首个 matmul 处触发：
  - `assert quant_state is not None`

已执行修复：

- `src/llamafactory/model/patcher.py`
- 在 `_convert_force_cuda()` 中：
  - 改为 `Params4bit(..., module=module, ...)`
  - 并在量化后显式兜底：
    - 若 `module.quant_state is None` 且 `new_value.quant_state` 已存在
    - 则同步 `module.quant_state = new_value.quant_state`

这个修复的目标很明确：

- 不是继续给 FSDP 打补丁
- 而是确保标准 `Linear4bit` 层在加载阶段就拥有完整量化态
- 这样即使后续权重对象被重新包装，bnb 也能从 `module.quant_state` 恢复

## 19. `quant_state` 断言消失后，新的 backward blocker：FSDP 检测到前向期间参数被修改

在补齐 `module.quant_state` 后再次验证，新的结果是：

- 首个真实前向已经成功完成
- `assert quant_state is not None` **不再出现**
- 训练已经进入：
  - `trainer.training_step(...)`
  - `self.accelerator.backward(loss)`

新的失败点变成：

- `AssertionError: FSDP does not support changing the parameters between forward and backward for HandleShardingStrategy.SHARD_GRAD_OP`

这说明当前状态已经进一步推进：

- 数据字段问题已过
- 标准 `Linear4bit` 量化态缺失问题已过
- 现在卡在 **bnb 4-bit 标准线性层仍然在 first forward 内做惰性状态写入**

结合 bitsandbytes `Linear4bit.forward()` 源码，可以看到它在首个前向里仍可能做几类就地初始化：

- 修复 `weight.quant_state`
- 设置 `compute_dtype`
- 标记 `compute_type_is_set`
- 在部分情况下调整 bias dtype

对普通训练这通常没问题，但对 `FSDP SHARD_GRAD_OP` 而言，这会被判定为：

- forward 和 backward 之间参数/权重对象被改动

因此这一步的结论是：

- `SHARD_GRAD_OP` 本身是对的
- 但 **必须把 bnb `Linear4bit` 的首前向惰性初始化提前到加载阶段完成**
- 否则 FSDP 会在 backward 前的 writeback 阶段直接拒绝

已执行修复：

- `src/llamafactory/model/loader.py`
- 新增加载后预初始化逻辑：
  - 仅对 `Gemma4 + 4bit + trainable` 生效
  - 遍历标准 `bnb Linear4bit / LinearFP4 / LinearNF4`
  - 在训练开始前提前完成：
    - `module.quant_state` / `weight.quant_state` 对齐
    - `weight.module = module`
    - `compute_dtype` 设为训练 dtype
    - `compute_type_is_set = True`
    - bias dtype 预对齐

目的很明确：

- 把原本发生在 `Linear4bit.forward()` 第一次调用里的状态写入
- 前移到模型加载阶段一次性完成
- 让 FSDP 在 forward/backward 之间不再观察到参数相关对象被更改

## 20. 继续收敛：把标准 `Linear4bit.forward()` 改成无副作用版本

在做完预初始化后再次验证，`SHARD_GRAD_OP` 的 backward 断言仍然存在。

这说明：

- 仅仅预先设置 `quant_state / compute_dtype / bias dtype`
- 还不足以完全消除标准 bnb `Linear4bit.forward()` 的内部状态写入路径

因此又做了一层更窄的收敛：

- `src/llamafactory/model/loader.py`
- 仅对 `Gemma4 + 4bit + trainable` 的模型实例
- 在加载阶段为所有标准 `bnb Linear4bit / LinearFP4 / LinearNF4` 安装实例级别的无副作用 `forward`

这个自定义 `forward` 保留原始数值路径：

- 读取已准备好的 `quant_state`
- 使用既定 `compute_dtype`
- 调用 `bnb.matmul_4bit(...)`

但明确去掉了原实现中那些会在首个前向里发生的就地状态更新：

- 不在 forward 里修复/替换权重对象
- 不在 forward 里设置 `compute_type_is_set`
- 不在 forward 里原地改写 bias dtype

目的：

- 让标准 4-bit 线性层的第一次前向也保持“只读”
- 如果这一步有效，`FSDP SHARD_GRAD_OP` 应该不再在 backward 前报“forward/backward 间参数被修改”

## 21. 当前结果：Gemma4 4卡 FSDP(ZeRO-2语义) + 4-bit SFT 已进入稳定训练

在安装标准 `Linear4bit` 的无副作用 `forward` 后，重新启动 4 卡任务，当前结果是：

- session:
  - `/home/ps/hanjiaqi/LlamaFactory/chatjoy/logs/gemma4_fsdp_qlora_4gpu_20260420_185255.session`
- latest log:
  - `/home/ps/hanjiaqi/LlamaFactory/chatjoy/logs/latest_gemma4.log`

关键信号：

- `mm_token_type_ids` 报错未再出现
- `quant_state is not None` 断言未再出现
- `FSDP does not support changing the parameters between forward and backward` 未再出现
- 日志已连续进入多个真实训练 step

最新观测到的训练进度：

- `7/1220`
- 随后继续推进到：
  - `8/1220`
  - `9/1220`

同时 GPU 占用已经稳定在训练态，而不是初始化后立即退出：

- GPU0: ~27.5 GB
- GPU1: ~27.6 GB
- GPU2: ~27.6 GB
- GPU3: ~27.5 GB

当前可下的结论：

- **Gemma4 当前这条 4 卡 FSDP(ZeRO-2 语义) + 4-bit + LoRA SFT 训练链已经跑通**
- 当前不是停留在“能加载模型”或“能过第一个 forward”
- 而是已经进入持续 step 推进的真实训练阶段

## 22. ToDo 更新

- [x] 修复 Gemma4 turn token 模板不匹配
- [x] 修复 Trainer `create_optimizer(model=None)` 兼容性
- [x] 补齐 Gemma4 训练态 `mm_token_type_ids`
- [x] 识别并改正 Gemma4 启动链误用 `FULL_SHARD`
- [x] 修复标准 `Linear4bit` 的 `quant_state` 初始化缺口
- [x] 把标准 `Linear4bit.forward()` 改成无副作用版本
- [x] 让 4 卡 Gemma4 FSDP(ZeRO-2语义) + 4-bit 训练进入真实 step
- [ ] 继续观察是否能稳定跑到 checkpoint/save_steps
- [ ] 后续再决定是否恢复 `flash_attn` 或进一步提速

## 23. 2026-04-21 二次收敛（无空闲 GPU 下的代码侧修复）

### 23.1 本轮目标

在“当前没有空闲显卡”的前提下，先把会导致假训练信号（loss 异常）的代码路径收敛，避免下次占到卡后再浪费训练时长。

### 23.2 已完成修复

1. 收敛 `loader.py` 的高风险训练期干预逻辑
   - 移除标准 `Linear4bit` 的训练期自定义 forward 注入
   - 移除加载后隐式 CPU->GPU 参数迁移
   - 移除首个 forward 的 one-shot monkey-patch 诊断
   - 保留 Gemma4 自定义 MoE 4-bit 量化和 CPU residual 显式失败

2. 增加 Gemma4 训练“防伪成功”保护
   - 文件：`src/llamafactory/train/sft/trainer.py`
   - 仅对 `gemma4/gemma4_text` 生效
   - 在 `compute_loss` 中新增两条硬校验：
     - 若当前 batch `labels` 全部为 `IGNORE_INDEX`，立即报错退出
     - 若 `loss == 0.0`，立即报错退出并输出有效标签数

这样可以保证后续一旦再次出现“loss 长期为 0”的异常，不会静默跑很多 step 才发现。

### 23.3 待 GPU 验证 TODO

- [ ] 占到 `GPU0,1,2,3` 后重启 Gemma4 4卡 FSDP(ZeRO-2语义) + 4-bit SFT
- [ ] 验证首个 train step：
  - 不报 `mm_token_type_ids` 缺失
  - 不报 `quant_state is None`
  - 不报 FSDP forward/backward 参数变更
- [ ] 验证训练日志：
  - loss 非 0
  - step time 落在历史正常区间
- [ ] 连续跑到至少一个 checkpoint/save_steps

## 24. 2026-04-21 GPU2 smoke 验证结果（单卡）

验证方式：

- 设备：`CUDA_VISIBLE_DEVICES=2`
- 配置：
  - `chatjoy/gemma4_sft2_1gpu_lora_smoke_gpu2.yaml`
  - `max_steps: 40`（smoke 用）
- 命令：
  - `llamafactory-cli train chatjoy/gemma4_sft2_1gpu_lora_smoke_gpu2.yaml`

关键结果：

- 训练已进入真实 step（非只加载/预处理）
- 前 3 个 step 的 loss 均为非 0：
  - step1: `loss=4.516`
  - step2: `loss=5.384`
  - step3: `loss=4.19`
- step 用时约 `28s/step`，没有出现历史“异常短步时 + loss=0”模式

附加观察：

- Gemma4 custom MoE 4-bit 量化路径正常触发：
  - `Prepared Gemma4 custom 4-bit experts for 30 layer(s)`
- 仍有约 `2.95GB` 的非 experts 参数在 CPU（embed/layernorm/router scale），当前策略为告警放行，不做隐式迁移
- smoke 在 step3 后由人工 `Ctrl+C` 中断，目的是完成快速验证并释放 GPU

TODO 状态更新：

- [x] 占到 `GPU0,1,2,3` 后重启 Gemma4 4卡 FSDP(ZeRO-2语义) + 4-bit SFT（已在 GPU2 做单卡前置验证）
- [x] 验证首个 train step（单卡 smoke 已通过）
- [x] 验证训练日志：loss 非 0、step time 非异常短（单卡 smoke 已通过）
- [ ] 连续跑到至少一个 checkpoint/save_steps（待正式长跑）

## 25. 2026-04-21 回看 Claude 修复链路 + RL 资源评估补充记录

### 25.1 回看范围（含 Claude 阶段日志）

本轮额外回看并交叉核对了以下文件（聚焦 Gemma4 SFT 与后续 RL 可行性）：

- `chatjoy/logs/gemma4_fsdp_qlora_4gpu_20260420_185255.log`
- `chatjoy/logs/gemma4_fsdp_qlora_4gpu_20260420_191853.log`
- `chatjoy/logs/latest_gemma4.log`
- `chatjoy/saves/gemma4-26b-a4b/sft2_lora_4gpu/trainer_log.jsonl`
- `chatjoy/GRPO_Training_Package/framework_bridge/verl/train_grpo_lora_actor_verl.sh`
- `verl/trainer/main_ppo.py`

### 25.2 对 Claude 修复链路的再结论（SFT 侧）

1. 2026-04-20 的 4 卡 Gemma4 训练确实进入过真实 step（不是只停在加载）  
   但在 `185255/191853` 这两次 run 里，训练日志仍出现：
   - `loss='0'`
   - `grad_norm='nan'`
   - step time 约 `~8.1-8.3s/step`

2. `185255` 与 `191853` 两次 run 的 `all params` 统计不一致：  
   - `2,418,135,040`（185255）
   - `3,156,332,544`（191853）  
   这进一步说明当时模型加载/参数路径仍处在不稳定切换期，不适合直接作为最终可训练证据。

3. `latest_gemma4.log` 的一次失败是“占卡导致的加载 OOM”，不是新代码根因：  
   - `Tried to allocate 12.21 GiB`
   - `GPU0 free 1.76 GiB`
   - 同时已有进程占用约 `29.09 GiB`

4. 单卡 smoke（GPU2）非 0 loss 的结论仍成立，说明“修复后的主链路可进入真实优化”，但 4 卡正式长跑仍需要继续验证到 checkpoint。

### 25.3 RL 资源与通信评估（基于当前代码，不拍脑袋）

#### A. 角色拆分限制（当前 verl 语义）

- 当前配置 `use_legacy_worker_impl=disable`。  
- `ref` 在新实现里已融合到 `ActorRolloutRefWorker`，不是独立可自由切 2 卡的 worker 组。  
- 因此“ref 单独 2 卡、actor+rollout 4 卡”在当前实现下不成立（除非改回 legacy worker 流程，不建议在当前阶段切换执行语义）。

#### B. 模型体量与默认参数冲突点

- Gemma4 text-only base 体量（本地 snapshot）：约 `49G`
- Qwen RM base（`clean32_for_sft2_fsdp_qlora_merged`）：约 `57G`
- 你当前 GRPO 脚本默认：
  - `ROLLOUT_TP_SIZE=1`
  - `RM_TP_SIZE=4`
  - `ROLLOUT_GPU_MEMORY_UTILIZATION=0.60`
  - `RM_GPU_MEMORY_UTILIZATION=0.45`

对“RM 只放 2 卡”的直接结论：

- `57G / 2 = 28.5G/卡`（仅权重静态量级）
- 但 `gpu_memory_utilization=0.45` 在 32G 卡上只给 vLLM 约 `14.4G`
- 数学上不成立，必然高风险（接近必炸），所以 RM 两卡方案不推荐。

#### C. 通信侧结论

- 机器拓扑仍是双 NUMA，跨 NUMA 明显慢于同 NUMA。  
- RL 训练若按 `0-3` 与 `4-7` 分池，重通信留在各自池内，跨池主要是样本与 reward 往返，不是 ZeRO-3 式全局 allgather，卡死风险可控。  
- 若把多角色混放在跨 NUMA 且高并发设置下，抖动与超时风险明显上升。

### 25.4 当前建议的可执行分配（与当前代码匹配）

1. GPU 分池（推荐）  
   - `0,1,2,3`：Actor/Ref/Rollout
   - `4,5,6,7`：RM（standalone resource pool）

2. 必开项  
   - `reward.reward_model.enable_resource_pool=True`
   - `reward.reward_model.n_gpus_per_node=4`
   - `reward.reward_model.nnodes=1`

3. 首轮稳态参数（先求跑通）  
   - `ROLLOUT_TP_SIZE=4`
   - `RM_TP_SIZE=4`
   - `ROLLOUT_N=2`
   - `PPO_MICRO_BATCH_SIZE_PER_GPU=1`
   - `LOGPROB_MICRO_BATCH_SIZE_PER_GPU=2`

4. 额外注意  
   - `train_grpo_lora_actor_verl.sh` 默认 `MODEL_PATH` 指向的 `...models--gemma-4-26B-A4B-it-bf16...` 目录当前本机不存在，启动前必须显式覆盖成已验证可加载的 Gemma4 路径。

### 25.5 补充 TODO（本节新增）

- [ ] 用上述 4+4 + 保守参数启动一轮短程 GRPO（先 20~50 step）
- [ ] 记录 rollout/RM 两侧峰值显存与 step time
- [ ] 若稳定后再逐步上调 `ROLLOUT_N` 与 micro batch
- [ ] 完成后回填本文件“RL 首轮实测结果（是否出现通信卡死/OOM）”
