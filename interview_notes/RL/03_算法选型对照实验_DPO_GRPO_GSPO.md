# 03｜算法选型对照实验：DPO / GRPO / GSPO（PPO、OPD 条件性加入）

> 口径声明（2026-11-09）：本文分四类内容——**[机制事实]**（公式与论文口径，可直接引用）、**[当前设计｜预注册，未执行]**（实验协议与判定规则，结果出来前不改规则）、**[本地配置｜已核实]**（仓库脚本行号级事实）、**[面试模拟｜假设推演]**（方向性预期，不是结果）。**所有实验均未运行；结果台账在 §5 留空待填，本文任何数字不得写成实跑结论。**配套实现在仓库 `training/GRPO_Training_Package/algorithm_selection/`。
>
> 阅读顺序：[RL 完整流程](01_RL完整流程与关键机制演进.md)（GRPO/GSPO/KL 的计算合同）→ 本文（为什么选它们、怎么对照）→ [交付与部署](02_模型交付与上线部署.md)。

## 0. 结论先行

算法选型分**两层对照**，回答两个不同的问题：

| 对照 | 回答的问题 | 能归因到什么 |
| --- | --- | --- |
| GRPO vs GSPO | 同一奖励、数据与采样下，优化方式差在哪 | 比率构造、裁剪粒度及其优化行为 |
| DPO vs RM＋在线 RL | 用现有资产（偏好对 vs 学好的 RM），预算内哪条路线更值 | 完整路线的质量/成本/工程取舍 |
| PPO / OPD 条件性加入 | critic / teacher 的额外成本是否值得 | 对应完整方案，不凭机制先宣布失败 |

选择规则（§4）先于结果预注册。面试口径：**设计 + 实现 + 预注册 + 结果待填**——不预先认定 GSPO 最好，也不把未跑的实验说成已跑。

## 1. 机制澄清（六条，引用前先对齐）

### 1.1 [机制事实] GRPO 没有把 token ratio 连乘

每个 token 的比率 $r_t(\theta)=\pi_\theta(y_t\mid x,y_{<t})/\pi_{\mathrm{old}}(y_t\mid x,y_{<t})$。outcome reward 下 GRPO 让整条回答共享同一组内优势 $A$，但在**每个 token 上**计算裁剪代理目标后求平均：

$$
L_{\mathrm{GRPO}}
=
-\frac1T\sum_t
\min\left(r_t A,\ \operatorname{clip}(r_t)A\right)
$$

各 token 项相加/平均，不是连乘。「GRPO 因 token ratio 连乘误差累积」是错误解释；log 概率对应求和，同样不是连乘。

### 1.2 [机制事实] GSPO = token ratio 的几何平均 + 序列级裁剪

$$
s(\theta)=\exp\left(\frac1T\sum_t\log r_t(\theta)\right)=\left(\prod_t r_t(\theta)\right)^{1/T},
\qquad
L_{\mathrm{GSPO}}=-\min\left(sA,\ \operatorname{clip}(s)A\right)
$$

即**长度归一化的序列比率 + 序列粒度裁剪代理**，与 sequence reward / advantage 粒度一致。`min` 与优势符号共同决定裁剪是否生效，不是把越界比率一律钳住。

纯数学例：100 个 token 的 log-ratio 均为 0.02——

| 对象 | 数值 |
| --- | ---: |
| 每个 token ratio | $e^{0.02}\approx1.0202$ |
| 未归一化完整序列概率比 | $e^{100\times0.02}\approx7.389$ |
| GSPO 长度归一化比率 | $\approx1.0202$ |
| GRPO 使用什么 | 各位置分别用 ≈1.0202，不用 7.389 |

此例解释「完整序列比率为何要控长度效应」，不能反推 GRPO 自己连乘。两条边界必须保留：token 级策略梯度配 sequence reward 本身不矛盾（PPO/GRPO 都可这么做）；GSPO 的归一化比率是优化代理量，**不等于完整序列概率比的精确重要性权重，更不意味着更新无偏**。

### 1.3 [机制事实] 「训推不一致」必须拆成三种变化

| 变化 | 例子 | 定性 |
| --- | --- | --- |
| 解码分布变化 | temperature / top-p / top-k 不同 | 可能是有意设置，须统一概率口径 |
| 同权重同输入跨后端差异 | vLLM vs 训练后端的精度/kernel/并行/MoE 执行 | 需诊断的实现差异 |
| 参数更新后的策略变化 | $\pi_{old}\to\pi_\theta$，路由随之改变 | 首先是策略更新，不是实现误差 |

三策略分解（$\mu$=rollout 分布，$\pi_{old}$=训练侧固定锚点，$\pi_\theta$=更新中策略）：

$$
\log\frac{\pi_\theta}{\mu}
=
\underbrace{\log\frac{\pi_\theta}{\pi_{old}}}_{\text{更新前后的策略变化}}
+
\underbrace{\log\frac{\pi_{old}}{\mu}}_{\text{采样侧与训练侧的差异}}
$$

两项不是一回事：old logprob 在训练侧重算时，更新前 $\pi_\theta=\pi_{old}$，优化比率可以是 1，但 $\mu$ 仍可能与两者不一致——**优化比率正常 ≠ 训推一致**（verl rollout correction 文档同口径）。

### 1.4 [机制事实] MoE 路由变化是「更新后」现象，不是必然训推差异

GSPO 论文 §5.3 讨论的关键现象是**梯度更新后**新旧策略对同一回答激活的专家发生变化；同权重跨后端的路由差异也可能存在，但要单独测量，不能从论文现象直接推定本地实现也有。

合并表述：后端差异造成采样与训练计算不一致；MoE 路由变化可能放大概率变化。它们影响梯度估计与稳定性，但不是同一种误差，也不通过 token ratio 连乘体现。**GSPO 对近似零均值、弱相关的扰动可能因平均而减小波动；同方向系统偏差平均后仍在；权重同步错误、模板错误、严重数值差不会被 GSPO「平均正确」。**

### 1.5 [机制事实] PPO 成本是优先级判断，不是算法定义

常见 PPO-RLHF 要学 value/critic：增加前向/反向/优化器成本、显存压力和 GAE/value loss 调试；但 **critic 不必是独立同尺寸模型**（可共享参数），不能写成「PPO 必然多一个完整大模型」。本项目论证：已有回答级奖励 + 同 Prompt 多候选计划 → 组相对优势免 critic → **优先**比较 GRPO/GSPO；PPO 是否值得加 critic 由收益与实测开销决定。同时 **G=6 采样也要花钱**——算 PPO 的 critic 成本时必须一并算 GRPO/GSPO 的额外候选生成成本。

### 1.6 [机制事实] OPD 的排除理由要换；DPO 的偏好对是支持不是排除

**OPD（On-Policy Distillation）**：学生自己生成，教师在学生实际前缀上给下一 token 分布/logprob。它缓解 exposure bias 但不消除所有偏移；token 前缀的连续变化 $(x,y_{<t})\to y_t$ 在单轮回复内部也存在，所以「不是 Agentic RL」不构成排除理由。正确的暂不采用理由：**当前有可验证的业务偏好评分器，但尚不具备同样可靠、可对任意学生序列做 teacher-forced logprob、成本合适的教师**（教师五项检查：业务能力/任意序列评分接口/tokenizer 兼容/错误前缀上的指导可靠性/成本与副作用）。「outcome reward 已经足够」是待验证假设，不能由「单轮任务」推出。

**DPO**：与 RM 共用同一偏好对资产；RM 学评分，DPO 直接更新 Actor，RM+在线 RL 对新候选持续评分。**现有偏好对是支持 DPO 的理由**。复用前五项检查：两侧同前缀；raw→summary 后方向是否仍成立；负例是简单显式错误还是覆盖当前 Actor 细粒度差异；chosen 是否夹带 Teacher 风格/超长/不良拒绝；与 Dev/Test 隔离。选型允许得出「DPO 同预算下达到同等或更好质量 → 选 DPO」；online/iterative DPO 存在，「DPO 只能学旧数据」不是定义。

## 2. [当前设计｜预注册，未执行] 实验设计

### 2A. 固定阶段，避免初始化混杂

```text
Actor 起点：同一个 Targeted-SFT checkpoint
Reference：冻结的同一个 checkpoint
在线奖励：通过诊断的固定 RM-B + 固定结构规则
```

统一：tokenizer/模板/输入视图/截断；LoRA target、rank、可训练参数；response mask、EOS；GRPO/GSPO 的优势计算与 loss 聚合；KL/entropy 等共同项；同步方式与每批数据复用次数。Family 隔离 Train/Dev/Test 先于摘要或攻击改写。若要解释 SFT-final 后的**第一次**算法选择，须用那一阶段的资产另做，不把第二阶段结果倒写。

### 2B. 数据账

| 数据 | 设置 |
| --- | --- |
| 在线 RL Prompt 池 | 800 个不同训练 Prompt：Clean 560 + Attack 240（五类各 48） |
| DPO 数据 | 开放与 RM 同源的**完整合格偏好池**，按预算平衡抽样（不得只给 800 Pair 就淘汰 DPO） |
| 继续 SFT 对照 | 合格 chosen，判断收益是否只需继续监督学习 |
| Dev | Clean 200 Scene + Attack 100 Scene，固定生产候选解码协议 |
| Final-Test | 独立家族；配置锁定后比较入围方案与固定基线 |

RM 原数据是 raw history 而实验用 summary 时，先完成输入转换与标签复核；来不及则在共同 raw 协议下诊断并限定结论。

### 2C. 训推差异诊断（先行，192 个回答）

32 个固定 Prompt × 6 = 192 Responses：冻结同一权重版本不更新；固定 token IDs/前缀/mask/精度/logprob 定义；诊断用 `temperature=1`、关 top-p/top-k；保存 rollout logprob 并在训练后端重算；能记路由则比较同位置同层专家集合。

$$
\delta_t=\log p_{\mathrm{train}}(y_t\mid x,y_{<t})-\log p_{\mathrm{rollout}}(y_t\mid x,y_{<t})
$$

报告：δ 有符号均值（只看均值会漏正负抵消）、|δ| 的 p95/p99、按回答长度与 Clean/Attack 分层、可观测时的路由一致率。更新前后分别检查策略变化，不把更新后的概率变化记作后端误差。**此诊断只确定差异是否存在，不单独证明 GSPO 更好。** 实现见 `algorithm_selection/diagnose_rollout_train_logprob.py`。

### 2D. 批量与预算

| 项目 | 设置 |
| --- | --- |
| 每 iteration | 32 Prompt |
| 每 Prompt 候选 | G=6 → 192 Responses/iter |
| Minibatch | verl 口径 16 Prompt 展开 96 Responses |
| 每批优化遍数 | 1 → 每 iteration 2 次参数更新 |
| 完整短试 | 25 iterations = 800 Prompt visits / 4,800 Responses / 50 updates |
| 筛选预算 | ≈10 iterations ≈ 1,920 Responses/run |

只有不重复抽取时 800 visits 才是 800 个不同 Prompt。**至少多次参数更新**：起点 current=old、归一化与正则全同时，GRPO/GSPO 初始梯度可能相同，只比第一次更新看不出差别。GRPO vs GSPO：同 Prompt 顺序、生成预算、更新安排，各自从当前策略生成；DPO vs 在线 RL：比**相同新增资源预算**（用短 profile 估 25 个 GSPO iteration 的 GPU 小时为上限；DPO 可提前达标不必花满）。GPU 小时计入 Actor/Ref/RM/Critic/Teacher 实际占用；另列从零构建成本，不隐藏 RM 的历史投入。

### 2E. 超参与快速筛选

| 算法 | 起步探索 |
| --- | --- |
| GRPO | LR 1e-6 / 3e-6；token 级裁剪 |
| GSPO | 同两档 LR；序列级裁剪，范围单独校准 |
| DPO | LR 1e-6 固定；β 0.05 / 0.1 两档 |

[本地配置｜已核实] 当前 launcher `train_grpo_lora_actor_verl.sh` L187–188 默认 `CLIP_RATIO_LOW=0.2 / HIGH=0.28`（非对称，注释标 DAPO 思路）；GSPO 原论文比较用的是 3e-4/4e-4，GRPO 用 0.2/0.27——**数量级不同，不能给两者强塞同一 clip 值制造「公平」，也不能照搬默认值当合理基线**；实验前先查各自的有效裁剪情况（clip fraction 口径也不同，不能横比）。launcher L40 另有 `VERL_RM_REWARD_CLIP=5.0`（raw RM 分对称裁剪，注释：22% chosen+19% rejected |score|>5）——属已存在的尺度合同。DPO 的 β 与 RL 的 KL 系数不数值强行对齐；用共同评测口径测得的**策略偏移**检查是否只是「走得远」。

执行顺序：①三算法各两配置筛选（≈10 iter 预算）→ ②按 Dev/稳定性/成本留两个 → ③入围方案各 2 个训练 seed 完整短试 → ④接近则加独立场景或第三个 seed，仍不明报告「暂不能区分」。两个 seed 只是快速重复性检查。

### 2F. PPO / OPD 条件加入

- **PPO**：先测 critic 显存/吞吐/链路成本；资源可接受则给合理 value 初始化再短试，不让随机 critic 学几步就判死。资源不足记「本轮预算不采用」，不是「被证明更差」。
- **OPD**：过 §1.6 五项教师资格检查后纳入同预算短试，教师成本全计；未过则记录具体前提缺失。

## 3. 判定指标（三层）

| 层次 | 指标 | 用途 |
| --- | --- | --- |
| 核心业务 | Attack 真破角率、合格角色内化解质量 | 是否解决目标问题 |
| 旧能力保留 | Clean 盲评偏好胜率、角色贴合/承接/推进/丰富度/密度 | 是否牺牲正常体验换防御 |
| 行为护栏 | 语言/格式错误、通用拒绝、重复、截断、生成失败 | 防指标投机 |
| 更新诊断 | 共同口径 KL、熵、梯度异常、ratio 尾部、有效梯度比例 | 训练是否健康 |
| 奖励诊断 | 组内 reward std、零优势组比例、RM 排错样本 | 区分「算法不行」与「无有效信号」 |
| 成本 | GPU 小时、峰值显存、生成/评分量、达标成本 | 收益是否值得 |

不可省略：训练 reward 上升 ≠ 业务改善（正式行为评测用独立 Judge，不用训练 RM 自验收）；三类 loss 不横比大小；GSPO/GRPO 的 clip fraction 单位含义不同；按 Scene/Family 算配对差异与置信区间（同 Scene 两次采样不是两道题）；100 Attack Scene 只筛明显差异，两三个百分点不定胜负；「没改善」先排除零优势/RM 无区分/更新被阻断。

## 4. 预注册选择规则（看结果前冻结）

| 结果 | 决策 |
| --- | --- |
| DPO 同业务质量且成本更低 | 选 DPO |
| GSPO 同采样预算下业务更好，或质量相当但更稳/更省 | 支持选 GSPO |
| GSPO reward 升但独立评测不改善 | 不选 GSPO，查奖励利用 |
| GSPO 优势只在训推差异存在时出现，修复后消失 | 结论限于该运行条件，不泛化 |
| OPD 过教师资格且短试更优 | 可改选 OPD |
| 各法差异落在噪声内 | 按成本与工程负担暂定，保留不确定性 |

若要支撑「GSPO 对训推差异更稳健」，补 2×2 对照（GRPO/GSPO × 当前实现/经验证的一致性改进如路由 replay 或 rollout correction），一次只改一种措施；**不能只给 GSPO 开校正不给 GRPO 开，再把收益归因 GSPO**。

## 5. 结果台账（待填，本文核心纪律）

| 实验 | 对象 | 主指标 | 诊断指标 | GPU 小时 | 决策 |
| --- | --- | --- | --- | --- | --- |
| 训推差异诊断（192 resp） | 待填 | δ 均值 / \|δ\| p95 | 路由一致率 | 待填 | 待填 |
| 筛选（3 算法×2 配置） | 待填 | Dev 双 SR | KL/熵/ratio 尾部 | 待填 | 待填 |
| 短试（2 方案×2 seed） | 待填 | Dev 双 SR | 同上 + 零优势组率 | 待填 | 待填 |
| 入围 vs 基线 Final-Test | 待填 | Roledrop / Clean 胜率 | 护栏全集 | 待填 | 待填 |

## 6. [面试模拟｜假设推演] 方向性预期（不是结果）

- δ 诊断预期存在非零 |δ| 尾部（跨后端 bf16/kernel 差异量级），但有符号均值接近 0；若出现同方向系统偏移，优先查模板/EOS/padding 而非算法。
- GRPO vs GSPO 的差异预期在**长回复与更新中后期**显现（ratio 尾部与裁剪粒度不同）；首轮更新两者可能几乎相同。
- DPO 预期在显式错误型负例上有快速收益、在当前 Actor 细粒度差异上受旧数据覆盖限制——这正是两层对照要量化的东西。
- 以上均为待验证假设；每个假设被哪张表证伪/证实，跑完在 §5 登记。

## 7. 面试口述（诚实版，可直接背）

> 算法选择我们分两层做。机制层比较 GRPO 和 GSPO：GRPO 是逐 token 裁剪代理的 平均，没有连乘；GSPO 用 token ratio 的几何平均构成长度归一化的序列比率再做序列级裁剪，和回答级 reward/advantage 粒度一致。因为奖励针对完整回答、模型是 MoE、回复偏长，我们认为序列粒度值得检验，但不预设它赢——两边用相同奖励、数据和采样预算对照，且 clip 范围各自校准，不塞同一个数制造公平。路线层比较 DPO 和 RM+在线 RL：偏好对资产两边都能用，DPO 更便宜，所以给 DPO 开放完整偏好池、按相同新增资源预算比；只有在线采样对当前策略新错误的增益抵得过成本才选在线路线。PPO 是优先级判断不是定义——组相对优势免掉了 critic，但 G=6 的采样成本也计入总账；OPD 暂缓是因为还没有可对任意学生序列评分且成本合适的教师。所有判定规则在看结果前预注册：独立双 SR 评业务、共同口径 KL 查策略偏移、按 Scene/Family 算置信区间。这套对照的代码和配置在仓库 algorithm_selection/ 下，结果跑出来按预注册规则填。

## 8. 本地实现对应与来源

| 内容 | 位置 |
| --- | --- |
| δ 诊断脚本 + 单测 | `../../character-rlhf/training/GRPO_Training_Package/algorithm_selection/diagnose_rollout_train_logprob.py`（+ `test_` 同目录） |
| 偏好对 → LLaMA-Factory DPO 格式转换 | `algorithm_selection/convert_pairs_to_llamafactory_dpo.py` |
| DPO 配置（β=0.05 / 0.1 两档） | `algorithm_selection/configs/dpo_gemma4_lora_beta005.yaml`、`..._beta01.yaml` |
| GRPO/GSPO 对照运行方式（含 clip 覆盖） | `algorithm_selection/README.md`；launcher L187–188 环境变量 |
| 实验矩阵与预注册规则 | 本文 §2–§5 |

来源：[GSPO 论文](https://arxiv.org/abs/2507.18071)（§2–4 机制、§5.1 clip 设置、§5.3 路由变化）、[DPO 论文](https://arxiv.org/abs/2305.18290)、[GKD/OPD](https://arxiv.org/abs/2306.13649)、[PPO 论文](https://arxiv.org/abs/1707.06347)、[Thinking Machines OPD 博客](https://thinkingmachines.ai/blog/on-policy-distillation/)、[verl rollout correction](https://verl.readthedocs.io/en/latest/algo/rollout_corr_math.html)；本地 launcher 行号为 2026-11-09 核实快照。
