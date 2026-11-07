# 02｜训练参数选择与 LoRA 技术分层

> 技术分层：P0 讲清选参逻辑；P1 会解释参数机制与数量级；P2 能设计对照、定位瓶颈并回答工程追问。
>
> 本文以 ChatJoy 的 SFT、RM、RL 为场景。配置值按 2026-09-14 本地文件核对；“配置存在”“历史记录”“当前设计”“选择理由的推断”“教学算例”分别说明。未找到受控证据的地方，不把某个 rank、学习率或并行方案写成最优结论。

## 1. P0：面试先讲“怎么选择”，再报数字

“我的顺序是先确定训练目标和数据，再确认模型、显存与部署链路，最后选择可训练范围、优化参数和评测门。SFT 学目标回复，RM 学同 Prompt 偏好，RL 根据采样到的 reward 更新 Actor，所以三个阶段不能机械共用一组参数。

在显存受限的大模型 SFT/RM 中，量化冻结底座并训练 LoRA，可以显著减少底座存储和优化器状态。LoRA 的 rank 决定每个目标矩阵的更新容量，alpha 控制缩放，target modules 决定改哪些部件。项目里 SFT、RM、RL 使用过不同组合；这反映任务和工程约束不同，并不证明 RM 天生需要 r128、RL 必须 r16。

我会把已有配置当作起点，用固定数据和评测判断它是否欠拟合、过拟合或受工程限制。要证明某个 rank 更好，就固定底座、模块、训练预算和评测，比较收益与成本；如果同时改 rank 和 FFN target，就不能把变化只归因于 rank。”

## 2. P0：先认清项目中到底是哪套参数

下表把配置和设计分开。文件名、注释中的版本不是唯一证据；可被环境变量覆盖的 launcher 默认值也不等于某次实跑值。

| 场景与证据身份 | LoRA：rank / alpha / target | 优化与长度 | 关键边界 |
| --- | --- | --- | --- |
| Qwen SFT2：存档 YAML 配置 | 8 / 16 / q,k,v,o；dropout 0.05 | LR 1e-4；2 epoch；cutoff 1536；micro=1，GA=2；warmup 0.1 | 项目台账记载 8 GPU；配置本身不能证明本次执行规模 |
| Gemma SFT2：存档 YAML 配置 | 32 / 64 / q,k,v,o | LR 1e-4；2 epoch；cutoff 3072；micro=1，GA=8；warmup 0.03 | 3 GPU DDP 是存档运行方案；dropout 未显式写在该 YAML |
| Gemma Defense SFT：存档 YAML 配置 | 16 / 32 / q,k,v,o | LR 5e-5；2 epoch；cutoff 3072；micro=1，GA=8；warmup 0.03 | 新 adapter 挂在 SFT2 merged；不是把同一 adapter 的 rank 原地减半 |
| RM-A 后期：既有正文的历史配置 | 64 / 128 / q,k,v,o | LR 9e-6；global batch 64；max_len 3072 | Qwen 与 Gemma RM 结果不可混成一个模型 |
| 历史 RM v6.3：存档 launcher | 128 / 256 / q,k,v,o,up,down | LR 9e-6；batch 64 Pair；micro=1；8 GPU；1 epoch；3072；warmup 0.03 | 训练集 40,551 Pair 的脚本口径，不是当前设计 44,280 |
| 当前 RM-B.1/B.2：统一设计 | 128 / 256 / q,k,v,o,up,down | LR 9e-6；batch 64 Pair；8 GPU；GA=8；1 epoch；3072 | 两者从同一 SFT merged 独立初始化，用 44,280 Pair 做数据对照 |
| 当前存档 RL launcher 默认 | 16 / 16 / q,k,v,o | LR 3e-6；prompt batch 32；rollout.n=4；warmup 0.03；prompt/response=2048/768 | 允许环境变量覆盖；与当前面试 N=6 协议不同，不能直接称统一设计实跑值 |

配置来源：[Qwen SFT2](../../character-rlhf/training/configs/qwen30b_fsdp_qlora_clean32_sft2.yaml)、[Gemma SFT2](../../character-rlhf/training/configs/gemma4_sft2_3gpu_ddp_lora_567.yaml)、[Defense SFT](../../character-rlhf/training/configs/gemma4_defense_sft_3gpu_lora.yaml)、[RM v6.3 launcher](../../character-rlhf/training/launch_scripts/launch_rm_train_gemma4_sft_v6.3_fg.sh)、[RL launcher](../../character-rlhf/training/GRPO_Training_Package/framework_bridge/verl/train_grpo_lora_actor_verl.sh)。设计来源：[RM-B 正文 §8](../RM/RMv6-v6.6_从Unified_Clean_Attack到定向Hardening完整面试复习.md)、[全链路总览](../00_ChatJoy后训练全链路统一面试复习.md)。

SFT 主文的 NF4、r32/alpha64、Paged AdamW、8×5090 和 ZeRO-2/FSDP 描述属于一套设计说明；不能把它与 Gemma 的 3GPU DDP YAML 拼成一次执行记录。本文涉及“为何选择”时，优先解释哪种瓶颈会使这套选择合理，并说明如何验证。

## 3. P1：LoRA 的三个核心参数到底改了什么

### 3.1 rank：限制的是每个矩阵的增量秩，不是整个模型的秩

对线性层 $W_0\in\mathbb{R}^{d_{\mathrm{out}}\times d_{\mathrm{in}}}$，标准 LoRA 写为：

$$
h=W_0x+sBAx,\qquad s=\frac{\alpha}{r}
$$

$$
A\in\mathbb{R}^{r\times d_{\mathrm{in}}},
\qquad
B\in\mathbb{R}^{d_{\mathrm{out}}\times r}
$$

底座 $W_0$ 冻结，学习增量 $\Delta W=sBA$，其秩不超过 $r$。LoRA 参数数为：

$$
P_{\mathrm{LoRA}}=r(d_{\mathrm{in}}+d_{\mathrm{out}})
$$

同一个 r 作用于不同形状的矩阵，参数量不同；多层、多目标模块的预算是逐矩阵求和。低秩是对“任务所需权重变化”的建模约束，并不表示原模型或所有任务都只需低秩。[LoRA 原论文 §4](https://arxiv.org/html/2106.09685v2)

**教学算例：仅考虑一个 4096×4096 的矩阵，不代表项目具体模型形状。**

| rank | LoRA 参数数 | 相对该全量矩阵的参数比例 |
| --- | ---: | ---: |
| 8 | 65,536 | 0.390625% |
| 16 | 131,072 | 0.78125% |
| 32 | 262,144 | 1.5625% |
| 64 | 524,288 | 3.125% |
| 128 | 1,048,576 | 6.25% |

r32→r128 对同一组矩阵使 adapter 参数数变为 4 倍；若同时增加 up/down，增长还要加上新的矩阵。不能用这一张单层表推算整个 MoE 的真实 trainable parameter 数。

rank 增加可能增加可表达的更新容量，也增加参数、梯度、优化器状态、通信及 checkpoint 成本；总训练显存不一定按 rank 成倍增长，因为底座、activation、logits 和 KV 等还占大量空间。

### 3.2 alpha：缩放项，与 rank 和学习率相互作用

标准 LoRA 的显式缩放为 $\alpha/r$。项目中的 r8/alpha16、r32/alpha64、r64/alpha128、r128/alpha256 都是缩放 2；RL launcher 的 r16/alpha16 是缩放 1。

选择 alpha=2r 可以在换 rank 时保持**显式前向缩放系数**一致，使配置更易比较；它不保证 $\Delta W$ 范数、梯度分布或训练轨迹一致。A/B 的维度、初始化、优化器归一化和数据都会影响实际更新。

不能说“alpha 翻倍等于学习率翻倍”。alpha 位于前向与梯度链中，学习率作用于优化器更新；两者相关但不等价。训练时应结合梯度范数、更新范数、loss 和外部指标观察。

PEFT 提供 rsLoRA，缩放改为 $\alpha/\sqrt r$；是否启用必须看 adapter 配置的 use_rslora，不能只看到 rank/alpha 就默认所有实现相同。[PEFT LoRA 配置说明](https://huggingface.co/docs/peft/v0.20.0/package_reference/lora)

### 3.3 为什么初始化通常一边随机、一边零

标准常见初始化是 A 随机、B 为零，初始增量为零，起步时保持底座行为。第一步 B 能收到梯度；A 的梯度因 B=0 而暂为零，之后 B 更新，A 再开始学习。若 A/B 都初始化为零，两边可能都收不到有效梯度而无法启动。

不同初始化方法会改变这一性质；面试时应限定“标准常见初始化”，不要把它说成所有 LoRA 变体的唯一做法。当前 PEFT 的默认行为见[初始化说明](https://huggingface.co/docs/peft/v0.20.0/package_reference/lora)。

### 3.4 target modules：决定在哪些矩阵上提供更新自由度

| 目标 | 人话解释 | 对 ChatJoy 的合理作用假设 | 代价与验证 |
| --- | --- | --- | --- |
| q_proj、k_proj | 改变 Attention 的匹配方式 | 更好地使用角色卡、历史和当前用户信息 | 看长上下文与角色行为，不把效果直接归给某个头 |
| v_proj、o_proj | 改变被聚合和输出的信息表示 | 调整上下文信息传递与输出表征 | 同 rank 对比更窄/更宽 Attention target |
| up_proj、down_proj | 改变 FFN 的特征变换 | 为细粒度偏好、表达差异提供更多可训练空间 | 参数量、activation 和运行时兼容性增加 |
| gate_proj（gated FFN） | 对 FFN 中间特征做门控，通常与 up/down 配套 | 增加 FFN 内特征变换的适配自由度 | 不是专家选择器；不在已核验 RM target 列表中 |
| MoE router | 产生专家选择及混合权重 | 可能改变专家使用分布 | 需核对真实模块；不能凭 gate 字样认定它已训练 |

q/k/v/o→q/k/v/o/up/down 是**模块范围变化**，r64→r128 是**容量变化**；历史同时变化，不能归因成“rank 翻倍解决了富度”。

“Attention 只学格式、FFN 只学知识”过度简化。角色、语言和表达由网络共同实现，这些目标仅提供不同位置的可训练自由度。

MoE 可能把多个专家存成三维 Parameter，而不是普通 Linear。字符串中写了 up_proj，不等于每个专家都成功注入。需列出实际命中模块/参数、trainable parameter 数和梯度，并检查 merge、vLLM hot-swap 与张量并行支持。最新 PEFT 对目标参数的支持不代表历史自定义模型已经支持。

### 3.5 dropout、bias、score head：小字段也能改变实验

LoRA dropout 是训练时对 adapter 路径输入的随机正则。Qwen SFT2 YAML 明确为 0.05；Gemma 两份 SFT YAML 未显式指定，不能凭经验填成 0 或 0.05，应查当时解析后的配置和框架版本。

dropout 增加可能缓解特定数据上的过拟合，也可能让短训练的拟合更慢。是否需要应看 train/dev 差距、重复率和外部指标；它不等于 RL rollout 的 temperature，也不等于语言模型输出采样。

bias 是否训练改变可训练参数集合；目标未配置时不要自行补历史值。RM 还多一个 **scalar score head**，它需要训练并与 adapter 一起保存。只保存 LoRA A/B 而漏掉 score.weight，会让重载后的 scorer 失去原来的评分尺度甚至排序。

当前 RM-B 设计要求 PEFT 包装前初始化 score head、将其纳入 modules_to_save，并对保存前后固定 Pair 做 reload parity。**文件夹里有 adapter 不是产物可用的充分证据。**

## 4. P1：为什么用 QLoRA，它究竟节省哪部分显存

LoRA 决定可训练增量；QLoRA 进一步把冻结底座量化存储，在计算时使用相应精度的运算，并让梯度穿过底座传到 adapter。4-bit 不意味着 optimizer、adapter、activation 全部为 4-bit。

QLoRA 原方法包含 NF4、double quantization 与 paged optimizer 等技术，它们分别针对权重存储、量化元数据和内存峰值。项目配置仅写 quantization_bit=4 时，不能自动证明完整采用全部组件；SFT 主文的 NF4/Paged AdamW 设计也不能替代实际解析配置。[QLoRA 原论文](https://arxiv.org/html/2305.14314v1)、[bitsandbytes 量化说明](https://huggingface.co/docs/transformers/quantization/bitsandbytes)

可用账本：

$$
M_{\mathrm{peak}}\approx
M_{\mathrm{base}}+
M_{\mathrm{adapter}}+
M_{\mathrm{grad}}+
M_{\mathrm{optimizer}}+
M_{\mathrm{activation}}+
M_{\mathrm{workspace}}+
M_{\mathrm{runtime}}
$$

RL 还要按执行阶段加上 rollout/RM/reference、KV cache 和权重同步的同时驻留。不能把每项独立峰值简单相加后叫实测，也不能忽略必然同时存在的组件。

**教学下界**：30B 参数若全部只按 4 bit 存储，裸权重为 15GB，约 13.97GiB；若全部按 BF16 存储，裸权重为 60GB，约 55.88GiB。这 30B 若包含 embedding/norm，它们已经计入参数总量。4-bit 下界尚未计量化元数据、部分模块保持高精度带来的额外存储；两种裸权重账均未计 adapter、训练状态、缓存和 activation，也不表示真实 30B MoE 必能装入某张卡。

MoE 的 A3B/A4B 等激活参数量影响每 token 计算量，不代表只需存储 3B/4B 参数。单卡能否驻留要看总权重、专家存储和分片策略。

QLoRA 相对纯 BF16 LoRA 主要减少冻结底座存储；adapter 优化器状态通常仍按其实际精度维护。速度也不保证提高：反量化、kernel、offload、batch 和通信共同决定吞吐。

## 5. P1：为什么各阶段用不同的 LoRA

### 5.1 SFT：r32/alpha64 是可用起点，不是能力定理

Gemma SFT2 的配置是 4-bit、r32/alpha64、q/k/v/o。它让语言、角色和回复习惯的适配落在有限参数预算内，适配存档 3GPU DDP 路径。

合理的工程解释是先利用底座已有语言与角色知识，用适度容量学习项目数据；但“r32 足够”必须由 SR-Clean、语言/角色切片和训练拟合情况支持。当前没有找到同底座、同数据的 r16/r32/r64 受控表，因此不能把 r32 说成搜索最优。

Qwen SFT2 使用 r8，并不证明 Qwen 比 Gemma 只需要四分之一适配能力：两者架构、长度、并行与数据处理路径不同。

### 5.2 Defense SFT：较小 rank 和 LR 是保守适配假设

Defense SFT 配置从 SFT2 merged 新挂 r16/alpha32，LR 从 1e-4 降到 5e-5。其业务目标更集中，适度限制更新与 clean replay 有助于设计“补防御、观察 clean 回退”的流程。

这里是**合理解释，非已完成归因实验**。即便 rank 小，依然可能明显改变风格或造成遗忘；有效控制还包括数据混合、步数、早停和 SR 双切片。低 rank 不能保证底座行为不变。

### 5.3 RM：为何扩大到 r128 并加入 up/down

RM-B 学习细粒度好坏、自然防御和多种错误边界；候选可能都通过粗规则，所以需要表征层提供更细的可分性。扩大 rank 和目标模块，为这种需求提供更多自由度，是一个可检验的容量假设。

历史 RM v6.3 launcher 能证明 r128/alpha256、q/k/v/o/up/down、LR 9e-6 等参数确实写入训练命令。它没有证明“r64 已经饱和、r128 提升多少”或“只增加 rank 修复了 Subtle Roledrop”。

先排除标签噪声、source 风格捷径、目标轴不足和数据覆盖，再讨论容量。若 train 已很好、Internal/B6 差，继续增 rank 可能让记忆训练集更容易；若 train/dev 都差，还需检查优化和注入是否生效，不能直接断言欠容量。

当前 RM-B.1/B.2 保持相同 LoRA，是为了让数据定向替换更可比。它回答“数据改动是否有帮助”，不能同时回答“哪个 rank 更好”。

### 5.4 RL：r16/alpha16 还受在线推理链路约束

存档 RL launcher 默认 q/k/v/o、r16/alpha16，并在代码注释中记录非 Attention 模块在当时 hot-swap/TP 路径上的稳定性问题。由此能解释为什么该运行路径先限制 target 范围；它不是“RL 算法只能训练 Attention”。

RL 每轮要完成 rollout、reward、logprob、反向更新和权重同步。可训练参数增加，不只是 optimizer 更大，还可能影响 adapter 传输、部署支持和下轮生成一致性。

当前设计固定 G=6；存档 launcher 默认 G=4。正文使用哪个值必须声明，正式复现需锁定环境覆盖和解析后的运行配置。该 Actor 路径也不能因使用 LoRA 就自动称 4-bit QLoRA：需核对训练侧实际 dtype/量化加载，不能照搬 RM 方案。

## 6. P1：学习率、Batch、步数为什么这样选

### 6.1 学习率不是由参数总量单独决定

| 场景 | 本地可核对值 | 可解释的考虑 | 需要观察什么 |
| --- | --- | --- | --- |
| 基础 SFT | 1e-4 | 对已有表征做目标回复适配，希望有限预算内形成信号 | supervised-token loss、SR、语言切片、角色保持 |
| Defense SFT | 5e-5 | 集中补行为时采取较保守更新 | Attack 改善及 Clean 回退 |
| RM | 9e-6 | 偏好标签较少且可能有噪声，兼顾底座与新 score head 的优化 | pairwise loss、Internal 分层、score 饱和、B6 |
| RL launcher | 3e-6 | 数据随策略变化、reward 有噪声，希望控制策略漂移 | KL、clip fraction、entropy、reward 对账、SR |

这些数字是来源中的配置，理由是机制解释；缺少 LR sweep 时，不能说 9e-6 优于 1e-5 已经被证明。不同 loss 的尺度、归一化、adapter 缩放和优化器都会影响合适学习率，不能把大小直接读成“某阶段学习更慢”。

RM 的 pairwise logistic loss：

$$
L_{\mathrm{RM}}=-\log\sigma(d),
\qquad
\frac{\partial L_{\mathrm{RM}}}{\partial d}=\sigma(d)-1
$$

容易 Pair 已有很大正 gap 时，梯度趋近 0。看到 Accuracy 高、Gap 大，不等于细粒度任务已解决；可能需要可靠的困难数据，而不是一味提高 LR。噪声标签的大负 gap 又会产生持续错误梯度，因此先保标签质量。

### 6.2 micro batch、梯度累积和 global batch

在纯数据并行、各 rank 样本数一致的常见配置下：

$$
B_{\mathrm{global}}=
W_{\mathrm{DP}}\times B_{\mathrm{micro}}\times G_{\mathrm{accum}}
$$

$W_{\mathrm{DP}}$ 是数据并行规模，不能在还存在 TP/PP 的系统里直接替换为总 GPU 数。

当前 RM-B：8×1×8=64 **Pair**。每个 Pair 有 chosen/rejected 两支，所以每优化步涉及 128 个序列槽位；它不是 128 个独立 Prompt。

在完整 microbatch 条件下，3GPU Gemma SFT 的 3×1×8=24 条训练序列；台账中的 8GPU Qwen SFT 对应 8×1×2=16。多轮一条序列包含多个 assistant target，样本数与被监督 token 数又不同。

增加 gradient accumulation 可以在不增单次 microbatch 峰值的情况下积累更多样本，但会增加单步耗时、改变同 epoch 的更新步数。变长序列下，“各 microbatch loss 等权平均”不一定等于“所有有效 token 等权平均”；检查实际 loss reduction 和分母，避免长短样本权重被意外改变。

### 6.3 epochs 和 steps：相同 epoch 不等于相同优化预算

当前 RM-B 设计 44,280 Pair、global batch 64、1 epoch：

$$
44{,}280/64=691.875
$$

若最后不足一批仍更新，预期 692 step；前 691 步覆盖 44,224 Pair，尾部 56 unique Pair。sampler 若补齐、drop_last 或跨 epoch 合并，实际情况可能不同，要查 global_step、samples_seen 和重复样本。

历史 v6.3 的 40,551 Pair 在同样理想规则下预期 634 step。这是两套数据规模，不能把 692 当作旧 v6.3 实跑步数。

为什么 RM 当前设计先用 1 epoch？有限标签反复暴露有记忆和噪声放大风险，先以固定预算观察 Internal 与业务切片；这只是保守起点。欠拟合时可延长，过拟合时提早停，不存在“RM 一定只能一轮”。

选择 checkpoint 使用固定 Internal 协议；B6 用于阶段验收。不要每隔几步看 B6 后随意改训练，又把最终 B6 分数当完全独立结果。

### 6.4 warmup、cosine、optimizer 和梯度裁剪

warmup 让 LR 从较小值进入目标值，降低初始 adapter/score head 和优化器状态未稳定时的突变。warmup_ratio=0.03 在 692 step 上是约 20.76 步，最终整数由 scheduler 实现决定；不能直接声称一定 20 或 21。

cosine 逐步降低后期 LR，适合预先确定总训练预算的方案；训练步数变化而继续使用旧总步数，会改变实际 LR 曲线。它并非普遍最优，也不能靠一条平滑的 loss 曲线证明业务效果。

Adam/AdamW 的 beta、epsilon、weight_decay 和 max_grad_norm 需要从实际框架的解析配置取值。现有文件未显式锁定的项目，不在本文伪造“项目最终参数”。Paged AdamW、CPU Adam offload 和普通 AdamW 是不同实现/放置策略，不可混写。

梯度裁剪限制单步梯度范数，不会修复错误标签或错误 reward。若频繁触发，结合原始梯度、LR、异常批次和 loss normalization 定位，而不是只把阈值调大。

## 7. P1：长度、Mask、精度和并行属于训练参数

### 7.1 为什么选 3072，何时不能直接截断

3072 是多份 Gemma SFT/RM 配置和当前设计的长度预算；不是模型理论最大上下文。选择它应结合角色卡+历史+当前 User+Response 的 tokenizer 后分布、截断率和显存 Smoke。

项目多轮合同要求保护角色卡、当前六轮及最后候选。超长优先处理当前窗口之前的更早历史；仍超限就按已定义方案压缩或丢弃。不能删掉当前六轮的一部分，却仍把样本标为完整 head6_final/continuation6_final。

Qwen 存档 cutoff=1536 与当前六轮 3072 设计不同；不能说旧配置已验证当前多轮合同。

不同 tokenizer/chat template 的实际 token 数不同。RL 的 Prompt limit、Response limit、Actor max_model_len 和 RM scoring limit 要分别核验，不能只把两个数字相加就保证所有 serving 路径通过。

### 7.2 Mask 比调 LR 更先决定“在学什么”

当前设计：

- SFT Head：六个 assistant 回答都监督；Continuation：历史前缀只参与 Attention，当前六个回答监督。
- RM：完整历史作为条件，chosen/rejected 最后有效 token/EOS 的 scalar score 做偏好损失；不是给历史 token 全部做 CE。
- RL：只对本轮生成 Response 计算策略损失和 KL；历史 assistant 不是本轮动作。

“train_on_prompt=false”只能证明意图是不监督 Prompt；复杂六轮包装是否正确，还需看实际 labels/mask。训练 loss 漂亮也不能替代抽样检查。

### 7.3 bf16、gradient checkpointing、offload 分别换什么

bf16 是计算/张量精度选择；QLoRA 的底座存储可为 4 bit，二者并不冲突。实际 dtype 与算子支持需看框架和硬件。

Gradient checkpointing 通过反向时重算部分前向换 activation 显存，不是保存磁盘 checkpoint。CPU offload 通过传输和 CPU 资源换 GPU 驻留空间；代价可能是吞吐下降。两者都应测关键阶段峰值和每步耗时，不能只说“更省显存所以更好”。

### 7.4 为什么 RM 用 ZeRO-2，SFT/RL 又有 DDP/FSDP

ZeRO-2 分片优化器状态与梯度；ZeRO-3 还分片参数。若量化底座已能单卡驻留、可训练 adapter 较小，而机器跨 NUMA/互联受限，减少底座参数的频繁 gather 可能让 ZeRO-2 更合适。

这是特定拓扑和实现的工程解释。LoRA 优化器状态本来就较小，ZeRO-2 相对 DDP 的收益未必像全参训练那么大；当单卡仍放不下底座时，又可能必须分片。当前 RM-B 的 ZeRO-2+Adam offload 是设计选择，不是所有模型的结论。

RL 同时涉及训练与 vLLM 推理副本，FSDP/TP、sleep/offload 和同步次序一起决定可运行性。8×32GB 并不等于某个单卡算子能拿到连续的 256GB。单卡物理限制要逐阶段核算。

## 8. P1：RL 的参数为什么还要看 reward 与采样

### 8.1 G、temperature、top_p 是数据分布的一部分

每个 Prompt 生成 G 条 Response，GRPO 才能比较同题差异。增 G 增加候选覆盖机会，也增加 rollout、scoring 和 logprob 成本。若所有候选都相同或都坏，仅增 G 不保证得到有用信号。

当前面试设计 G=6；存档 launcher 默认 G=4，均需标清。教学计算：prompt batch=32 时，G=4 有 128 Response 槽位；G=6 有 192。训练预算应同时报告 Prompt、Response 和 token，不只报“batch 32”。

temperature/top_p 调整采样多样性，过低可能缺相对差异，过高可能产生大量无效答案。它们不能替代缺失的能力和可靠的 reward。API Judge 的低温是降低判分波动，Actor 的采样温度是在探索，两者目的不同。

未在项目当前设计中明确冻结的 temperature/top_p，不应直接用框架默认值补写成正式口径。

### 8.2 GRPO、GSPO、KL：各自控制不同问题

GRPO 当前组内优势可写为：

$$
A_i=\frac{R_i-\mu_g}{\sigma_g+\epsilon}
$$

$\mu_g,\sigma_g$ 只来自同一 Prompt/uid 的候选。reward 标准差接近零时，需看 epsilon 是否主导、差异是否只是噪声及方向是否可靠；不能仅凭非零就宣布有训练信号。

GSPO 使用 Response 有效 token 平均 log-ratio 的指数作为 sequence ratio：

$$
\rho_i=
\exp\left(
\frac{1}{T_i}\sum_t m_{i,t}
\left[\log\pi_\theta(y_{i,t}\mid c_{i,t})
-\log\pi_{\mathrm{old}}(y_{i,t}\mid c_{i,t})\right]
\right)
$$

这里 $c_{i,t}$ 是同一历史和生成前缀，$T_i=\sum_t m_{i,t}>0$。存档 launcher 的 clip 下/上参数为 0.2/0.28，即 ratio 区间 0.8—1.28；但目标函数 clipping 不是对最终策略变化的硬保证。

KL 约束相对 reference 的漂移，LR 控制优化步长，两者作用不同。存档 launcher 的默认 KL loss coefficient=0.001、use_kl_in_reward=False；当前 Clean/Attack 不同 KL 的设计需要检查分支实际接入，不能仅因配置有字段就说生效。

reference 也要核对 lineage：LoRA 场景禁用当前 adapter 后，得到的是当前 merged 底座，不一定是项目最早的 Base。改变 merged 底座就改变了 reference 对象。

### 8.3 PPO mini batch 不要与 Prompt Batch 和 micro batch 混为一谈

Prompt Batch 决定一轮收集多少 Prompt；rollout.n 决定每题多少 Response；PPO mini batch 决定一次优化更新使用的分组；micro batch 是因显存拆成的小批。

存档配置中的 mini batch=16 不能脱离所用 verl 版本解释：部分路径会按 rollout.n 扩展或在训练前重写大小。需查看解析配置和实际批次维度后，才计算“每轮几次 optimizer step”。

同一轮固定 rollout、old logprob、reference logprob 和优势；每次更新后的 current logprob 需要对当前参数重算。跨 mini batch 复用旧的 current logprob，会破坏 ratio。

### 8.4 熵、Reward Scale 和隐蔽的优化变化

Token entropy 鼓励输出分布保持多样性，不等同于 MoE router 的负载均衡。存档 launcher 同时出现 entropy coefficient 与 MoE auxiliary coefficient，应检查两者分别计算什么，不能照抄注释说“token entropy 保证 router 不坍塌”。

RM raw score 的 affine scale、clip、Guard 和 KL 位置共同决定 RL 优化行为。即使单独 RM 排序不变，和其他 reward 相加后相对权重可能变；标准化也受 epsilon 和裁剪影响。先做同一 Response 的 Reward Reconciliation，再调 LR/G/LoRA。

这些机制的项目上下文见 [RL 正文](../RL/01_RL完整流程与关键机制演进.md) 与 [全链路总览 §9—10](../00_ChatJoy后训练全链路统一面试复习.md)。本文不把“写出公式”当成当前仓库已完整接通的证明。

## 9. P2：怎样把“合理解释”变成可验证的选参理由

### 9.1 先判瓶颈，再决定是否加实验

| 现象 | 竞争性解释 | 最小区分办法 | 对选参的影响 |
| --- | --- | --- | --- |
| train/dev 都差 | 欠容量、LR 不合适、标签冲突、mask 错、LoRA 未注入 | 检查实际 trainable 参数/梯度，核验少量可靠样本拟合 | 排除实现/数据问题后才试 rank |
| train 很好、dev 差 | 过拟合、数据泄露、source 捷径、分布变化 | 按语言/来源/长度/难度分层，检查去重和 gap | 不优先增 rank 或训练轮数 |
| RM 排得好、Actor 不变 | 无合格候选、Composite Reward 错、KL 太紧、同步问题 | 固定候选做支持度、Reward 对账、策略更新检查 | 回数据/reward/链路，不盲调 RM rank |
| OOM | 底座、activation、KV、同步双驻留或 checkpoint 峰值 | 定位具体阶段和申请量 | 对准瓶颈降并发/token 或调 offload |
| 高 rank 更慢但指标接近 | 容量并非当前瓶颈、数据限制、收益小于噪声 | 同预算配对评测与多 seed 复核 | 选成本更低且达到业务门的方案 |

### 9.2 LoRA 最小受控对照：一次回答一个问题

以下是**待执行实验设计**，不是本项目已有跑分表。

1. 固定同一 merged 底座/hash、tokenizer、训练 manifest、Judge 标签、长度策略、batch、优化预算及随机种子。先对配置做短 Smoke，验证 forward/backward、保存、重载及 serving。
2. 选一个当前基线与一个相邻 rank 比较。例如 SFT 的 16 vs 32，或 RM 的 64 vs 128；先固定 target modules，不同时加入 FFN。
3. 标准 LoRA 下先保持 alpha/r 不变，以控制显式 scaling；同时承认这不是所有优化因素都相同。若目标是找到各 rank 的最佳可用配方，还需为各 rank 给同等预算的 LR 调整，并将其称为“配方比较”。
4. rank 对比结束，再固定 rank 比较 q/k/v/o 与 q/k/v/o/up/down。这回答模块范围问题，不能与前一步合并归因为单因素。
5. 开发选择用 Internal/SR-dev；B6 做固定阶段验收。业务关键项不能回退，收益接近判分/seed 噪声时不宣称更优。

两种预算回答不同问题：相同 token/optimizer step 比较给定训练暴露下的效果；相同墙钟/显存预算比较工程性价比。高 rank 每步更慢时，不能同时声称这两种条件都相同。

通过初筛的方案再做多个 seed 检查稳定性；无需一开始把 r×alpha×target×LR 全网格铺开。新增实验应由当前瓶颈驱动。

### 9.3 每组实验至少记录什么

~~~text
目的：rank 容量 / target 范围 / LR / 数据替换中的哪一个
底座、adapter init、数据 manifest、tokenizer/template 与代码版本 hash
实际命中模块和 trainable 参数数；rank/alpha/scaling/dropout/bias
实际量化方式、compute dtype、optimizer、scheduler 与所有解析默认值
DP/TP/PP、micro batch、GA、global batch 及其统计单位
有效监督/Response token、pair exposure、optimizer steps、末批处理
seed、样本顺序、截断/丢弃率、每阶段峰值显存与吞吐
Internal / B6 / SR 分层结果、paired fix/break、Judge 版本与 coverage
保存重载、score head、pooling、reference lineage 和 weight sync 结果
~~~

当前 RM-B.1→B.2 的数据-only 设计还要求同槽位替换、相同比例、同 total Pair，并监控 non-pad token exposure；数量相同不等于监督暴露相同。其“token 差控制在 1% 内”是现有对照目标，不是证明所有混杂均被消除。

### 9.4 什么时候停止调参

达到业务门、关键切片稳定，而新增容量只增加成本且收益处于噪声内，就保留较简单方案。若主要失分来自标注歧义或模型不能生成目标行为，继续扫 LR/rank 的机会成本较高，应回到标签、SFT 数据或 reward 设计。

“这个参数为什么选”最可信的回答由四件事组成：实际约束、选择机制、观察到的证据、会让自己改选的条件。若有对应 Smoke 与验收记录、但没有选参对照，可以说“这是通过当前可运行性和业务验收的基线”；若只有配置或设计，就按该身份描述，不补造“已经验收”或“已经搜索最优”。

## 10. P0/P2：高频追问速答

**LoRA 会不会完全不遗忘？**  
不会。虽然底座权重冻结，部署时输出取决于底座加 adapter，行为仍会改变；需要 replay、评测和更新预算控制。

**rank 越大越好吗？**  
表达容量通常变大，但数据质量、泛化、算力和部署约束可能成为瓶颈。选择满足业务要求的容量，不能只用训练 loss 决定。

**alpha=2r 是经验公式还是理论最优？**  
在标准 LoRA 下它让显式 scaling=2，是便于管理的配置关系；没有理论保证它适合所有模型和任务。

**LoRA 为什么能节省这么多优化器显存？**  
优化器只为实际训练的 adapter 和额外 head 等参数维护状态，冻结底座不需要同等训练状态；底座本身仍要存储并参加前反向计算。

**为什么还会 OOM？**  
activation、长序列、logits、KV、分片 gather、推理副本和同步峰值不会因底座冻结自动消失。先定位峰值阶段。

**为什么 LoRA 推理不是永远零开销？**  
能合并到普通线性权重时，可消除额外 adapter 路径；不合并的多 adapter/hot-swap 仍有计算管理成本。量化和 MoE 合并还需检查支持与数值一致性。

**RM score head 需要 LoRA 吗？**  
它本身很小，通常作为额外可训练模块保存；关键是初始化、训练和持久化都正确。adapter 保存策略不能漏掉它。

**为什么不用更大的 batch 和更多 epoch？**  
batch 改变梯度估计与同 epoch 更新次数，epoch 增加暴露与过拟合风险。先明确预算单位与验证目标，再比较。

**r128/up/down 的提升能算单变量实验吗？**  
不能，两种因素同时变化；若还换了数据、底座或 Judge，混杂更多。应分别验证，或诚实称为整体配方升级。

## 11. 证据索引与尚不能回答成“已证明”的问题

| 证据 | 重点读取位置 | 用途 |
| --- | --- | --- |
| [SFT 完整正文](../SFT/Character-RLHF_SFT_Canonical_Full.md) §13 | QLoRA、Mask、硬件方案 | 当前设计 |
| [RM-A 完整正文](../RM/RMv1-v5_从SFT平台到Anchor_RM完整面试复习.md) §4—5 | r64、训练预算、产物保存 | 既有历史配置与复现口径 |
| [RM-B 完整正文](../RM/RMv6-v6.6_从Unified_Clean_Attack到定向Hardening完整面试复习.md) §8—9 | r128、44,280 Pair、数据-only | 当前统一设计 |
| [Qwen SFT 配置](../../character-rlhf/training/configs/qwen30b_fsdp_qlora_clean32_sft2.yaml) | method/train | r8、dropout、LR 与长度 |
| [Gemma SFT 配置](../../character-rlhf/training/configs/gemma4_sft2_3gpu_ddp_lora_567.yaml) | method/train | r32、LR、GA |
| [Defense SFT 配置](../../character-rlhf/training/configs/gemma4_defense_sft_3gpu_lora.yaml) | model/method/train | r16 与新 adapter 底座 |
| [RM v6.3 脚本](../../character-rlhf/training/launch_scripts/launch_rm_train_gemma4_sft_v6.3_fg.sh) | 实际命令及可覆盖默认值 | 历史参数实现；注释不代替运行日志 |
| [RL 脚本](../../character-rlhf/training/GRPO_Training_Package/framework_bridge/verl/train_grpo_lora_actor_verl.sh) | LoRA/Batch/LR/Clip/KL/上下文设置 | 当前存档默认和 serving 限制 |
| [Benchmark 专题](01_Benchmark制作与指标计算_Judge一致性技术分层.md) | 指标与一致性 | 选参结果如何测量 |

本次没有运行训练或付费 API，也没有找到足以证明“r32/r128 最优”“9e-6 优于相邻 LR”或“扩 FFN 单独带来某个百分比收益”的受控记录。下一次若要把选择理由提升为实验结论，需要补同条件对照与完整运行产物；当前材料已经足以解释这些参数分别控制什么、为什么会作为项目基线以及怎样验证是否该调整。
