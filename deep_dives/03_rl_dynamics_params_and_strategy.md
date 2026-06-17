# 深读 3 — RL 训练动力学：参数设置、动态指标趋势观察与策略应对（含算法原理）

> 本篇展示的能力：**GRPO/GSPO 的每个参数为什么这么设（算法原理）→ 训练中盯哪些动态指标 → 每一次趋势异常如何归因并选择应对**（调参 / 修奖励 / 修运维 / 还是停训转向）。
>
> 所有 val 数字来自真实训练日志；完整时间线见 [docs/chatjoy_rm_v6_3_roledrop_handoff.md](../docs/chatjoy_rm_v6_3_roledrop_handoff.md)。

---

## 1. 算法选型与原理

### 1.1 GRPO：用组内相对优势替代 Critic

PPO 需要一个 value model 估计 baseline；GRPO 对每个 prompt 采样一组（本项目 `ROLLOUT_N=4→6`）回复，用**组内标准化**构造优势：

```
A_i = (r_i − mean(r_1..r_n)) / std(r_1..r_n)
```

- **为什么适合本项目**：① 省掉 critic 的 26B 显存（在 31.36GB/卡 的机器上是硬约束，见深读 1）；② reward 是"RM 标量 + 规则惩罚"的混合信号，学 value function 很难，组内对比天然鲁棒；
- **隐含的统计要求**：组内 reward 必须有方差——`std≈0` 时优势全为 0、没有梯度。这就是 RM 验收门 `reward_std ≥ 2.0、saturated=0%` 的来源，也是后面"攻击信号稀疏=没梯度"问题的理论根源（§5）。

### 1.2 GSPO：序列级重要性比（本项目的关键选择）

GRPO 默认 token 级重要性比 `π_new(t)/π_old(t)`；GSPO 把它换成**序列级**（长度归一的几何平均），裁剪也在序列级做。选它的三个理由：

1. **奖励粒度对齐**：RM 给整条回复一个标量，token 级比率会把序列级信号错误地摊到每个 token 上，方差大且有偏；
2. **长度公平**：`seq-mean-token-mean` 聚合下不同长度序列等权 → 长序列不因 token 多而主导梯度（length-hacking 防护的算法层，配合 `MAX_RESPONSE_LENGTH=768` 物理层）；
3. **MoE 稳定性**：token 级比率的高方差更新会放大 expert 路由抖动；GSPO 序列级平滑是 **MoE router 不坍塌**的已知做法。

### 1.3 其余组件的原理一览

| 组件 | 设置 | 原理 |
|---|---|---|
| **DAPO 非对称裁剪** | `clip_ratio_low=0.2, high=0.28` | 上行方向放宽（鼓励探索好样本）、下行收紧；限制单步策略位移，防 reward-hacking 暴跳 |
| **KL 惩罚** | `kl_loss_coef 0.001→0.005`，对参考策略 | 把策略锚在 ref 附近防分布崩坏；LoRA 模式 ref=actor 关 adapter（零显存，深读 1）|
| **per-source KL** | clean / attack 数据源各自 KL 系数 | 两类数据的最优偏离度不同：clean 想贴近 ref（保质量），attack 想离开 ref（改行为）——单一 KL 系数无法两全 |
| **熵正则** | `entropy_coeff=0.01` | 防 mode collapse；MoE 下同时维持 router 输入多样性 |
| **LoRA 范围** | r=32，`[q,k,v,o,gate_proj,up_proj,down_proj]`，**不含 router `gate`** | RL 梯度噪声大，动 128-expert 的路由器极易坍塌；`gate_proj` 是专家 FFN 的 SiLU 门，与路由无关 |
| **LR** | `1.5e-6 constant`（**不用 cosine**）| RL 总步数事先未知（随时可能停/续/resume），cosine 的"周期=总步数"假设不成立——这正是 RM v3 在 SFT 侧踩过的坑（深读 2 §3.3）的反向应用 |
| **采样温度** | 1.15→1.2 | 抬 rollout 多样性=抬组内 reward 方差=抬梯度信号（§1.1）；上调时盯 reward_std 与 KL 防失控 |

---

## 2. 完整参数表（verl，含算力侧根据）

```bash
N_GPUS=4  ROLLOUT_TP_SIZE=4            # 48.6GB÷4=12.15GB/卡，TP=4 是必然（深读 1）
ROLLOUT_GPU_MEMORY_UTILIZATION=0.60    # 三档复测得出（0.78/0.50 各有死法，深读 1 §3.2）
RM: enable_resource_pool=True, n_gpus_per_node=4, TP=4, util=0.62   # RM 独占 GPU 4-7
MAX_PROMPT_LENGTH=2048  MAX_RESPONSE_LENGTH=768   # max_model_len=2816, KV 按此分块
TRAIN_BATCH_SIZE=64  ROLLOUT_N=4→6
loss_mode=gspo  loss_agg_mode=seq-mean-token-mean
clip_ratio_low=0.2  clip_ratio_high=0.28
kl_loss_coef=0.001→0.005 (per-source)  entropy_coeff=0.01
lr=1.5e-6 constant   lora_rank=32  alpha=32
temperature=1.15→1.2
SAVE_FREQ=25→5                          # ⭐ 运维参数也是训练参数（§4.1）
```

---

## 3. 动态指标看板：每个指标在回答一个问题

| 指标 | 回答的问题 | 健康参考（本项目实测）|
|---|---|---|
| `clean_RM`（val）| 角色扮演质量在涨吗 | 1.28–1.39 区间；**±0.05 内波动视为噪声** |
| `atk_RM` / `break%`（val）| 抗攻击行为在改善吗 | break 用**新检测器**（泄露∪任务顺从）口径 |
| 训练 reward 均值 | 策略在爬升还是平台 | 平台 + val 平 = 收敛 |
| `response_length` | 有没有 length hacking | 稳定 ~298 tok = 无灌水（若单调上涨且 reward 同涨 → 红灯）|
| `kl_loss` | 偏离 ref 多远、是否均衡 | 稳 ~1.2-6.5 的均衡点；持续上行 = 偏移失控 |
| `grad_norm` | 更新是否平稳 | ~2 健康；偶发 190/530 尖峰（§4.3）|
| `reward_std`（组内）| GRPO 还有没有梯度 | 接近 0 = 优势归零，训练失效 |
| `lenBonus` / 各 reward 分量 | 每条信号是否按设计生效 | 如 attack 上 `ooc≈−0.58`、`bonus_revoked≈−0.24` 确认惩罚在生效 |
| 误报抽查（人工读 val 样本）| **指标本身可信吗** | 每次指标突变先查口径（§4.4）|

> 原则：**分量化监控**。total reward 是多路信号之和，只看总分无法归因——`anti_chain_*_component` 全部单独落盘（[anti_chain_reward.py](../training/GRPO_Training_Package/anti_chain_reward.py)）。

---

## 4. 趋势 → 归因 → 应对：六个实战案例

### 4.1 「净进度 = 0」死循环：SAVE_FREQ 是被忽视的训练参数

- **现象**：auto-restart 跑了一天，ckpt 永远停在 step 200。
- **归因**：OOM 复现点在 step ~202–214，而 `SAVE_FREQ=25` 的下一个存档点是 225 → **崩溃间隔 < 存档间隔** → 每次 restart 从 200 重来、跑十几步又崩，resilience 配置形同虚设。还出现过"val 进了 tensorboard 但 ckpt 没存下"的幽灵最优点（step225 观测到却不可 resume）。
- **应对**：`SAVE_FREQ 25→5`（其余零改动）→ 当日从 200 一路爬到 350，中途崩 1 次自动从 235 续上。
- **原理化**：把训练当随机过程看，**有效推进条件是 `P(存活到下个存档点) 足够大`** —— save 频率必须按实测崩溃间隔设，不是按磁盘心情设。

### 4.2 psKL 平台期：完整的 val 序列 + 多杠杆响应

| ckpt | clean_RM | atk_RM | break% | 读数 |
|---|---|---|---|---|
| step100 | 1.345 | −1.847 | 15.14% | 旧最优 |
| step150 | 1.288 | −1.972 | 15.14% | attack RM 变差、break 不动 |
| step175(bonus+1) | 1.256 | −1.934 | 16.38% | **bonus +1.0 无行为改善**——只是给非拒绝样本算术加分 |
| psKL step200 | 1.287 | −1.864 | 15.88% | atk_RM 改善、break 反弹 |
| psKL step225 | 1.352 | −1.894 | 14.64% | clean 恢复、attack 微改善 |

应对组合（每项都有显式依据）：
- `NON_REFUSAL_BONUS +1.0→+3.0`：**按 reward 分布的 σ 标定**（−4.0 的拒绝罚 ≈1.55σ），+1.0 在组内标准化后不足以改变优势排序，+3.0 才能把"戏内化解"推到组内前列；
- `ROLLOUT_N 4→6`：增大组规模 → 优势估计的方差降、且更可能采到稀有的好行为；
- `KL 0.001→0.005` + per-source：clean 锚紧、attack 允许更大偏离；
- 225→350 再观察 6 个 val：clean_RM 钉死 1.30–1.33、break 钉死 ~15.9% → **判定收敛，停训**。选 step325（clean 1.318 / 拒答 0.07% / atk_RM −1.878 最均衡）。
- **结论先于情绪**："加步数无效"是从 6 个连续 val 的零趋势读出的，不是从耐心耗尽读出的。

### 4.3 grad_norm 尖峰：不是事故，是收敛诊断

平台期偶发 grad_norm 190/530（pg_loss 尖峰）但 reward/KL/长度全平 → 解读为**收敛点附近的横跳**（策略在平坦区随机游走，偶尔踩到大比率样本被 clip 兜住），佐证"已收敛"而非"训坏了"。无需动作——**不是每个异常值都需要应对，要看它与其他指标的联动**。

### 4.4 v6.4 "破角率 6%"：先审指标，再动模型

- **现象**：上新检测器后 val 显示破角 6%。
- **归因**：逐条人工读 16 个判破样本 → 全是**误报**：模型在戏内拒绝变 AI（"我无法切换到 AI 助手的身份，我将继续以 Noa 的身份…"）、或讥讽复述攻击词（"什么'AI 助手'？"）。但 reward 已把这些**好行为罚到 −5.42** —— 检测器在惩罚我们想要的行为。
- **应对**：否定语境豁免（同子句、短语**之前**有否定标记；"作为一个大型语言模型，我无法退出角色"中否定在短语后、被逗号隔开 → 仍判破角）+ 引号复述豁免。误报 1/16→**0/16**，TDD 49→101 tests。
- **原理化**：reward 检测器的**查准率比查全率重要**——RL 会精确放大你给的任何信号，包括错的。

### 4.5 v6.5 硬化版：375 步零移动的双重归因 → 停训

配置拉满（fresh-from v6.4_160、temp 1.2、LENGTH_TARGET 1500、rollout n=6、OOM 硬化后 0 OOM 连跑 424 步）。结果：

- **clean 死平**：clean_RM step25=1.339 → step400=1.339（全程 1.28–1.39 噪声）；length 平 ~205 tok、lenBonus 平 ~0.50 —— **temp 和 length 两个杠杆都拉满了，actor 纹丝不动**。
  → 归因：**冻结 RM 的 ~1.33 是天花板**。RM ±5 的信号量级碾压 length +1.0 的拉力；actor 已坐在冻结 RM 的最优点上，组内优势≈0。"不动 RM、榨剩余梯度"被实测证伪——**这是花 60h GPU 买到的负结果，它把决策空间从'调参'收缩到'换信号'**。
- **attack 死平**：break ~11–12%、task-comp ~5–6%；分量监控确认惩罚在生效（`ooc≈−0.58`）但行为不变。
  → 归因（**分布层面，最致命**）：**val break 11% ≠ 产品 roledrop 55%**。val 攻击分布几乎不含产品式 bullet 注入（val task-comp 5% vs 产品 55%）→ ① 该失效在训练数据里**稀疏到没有梯度**（组内全是同样行为 → 优势=0，§1.1）；② val **测不出**产品关心的改善。训练分布、验证分布、业务分布三者错位。

### 4.6 战略终局：一张对照表改变路线

产品侧 11 卡 roledrop eval（统一口径：新检测器∪拒绝关键词）：

```
破功率:  base 27%  <  v2(RL后) 55%  <  v1(RL后) 91%
```

**原始 base 防御最强 —— RL 把它训坏了**（anti_refusal bonus 反向训练，深读 2 §4.2），v2 的 reward 修复只收回一半。结合 4.5 的双重归因，决策：
- **停止纯 RL 调参**（两轴都到顶：clean 被冻结 RM 卡死，attack 信号稀疏且 plateau）；
- ATTACK → **defense-SFT v2**：蒸馏 base 的"戏内化解"（它本来就会），定向 6 张失败角色卡 × 产品式攻击 —— SFT 直接教行为，比在稀疏信号上做 RL 高效得多；
- CLEAN → 要再进只能 **LLM-judge 在环 / 刷新 RM**（换掉天花板本身）。

### 4.7 v6.7 温度脆：RL 防御的模态绑定问题

v6.7 在 t0.7 rollout 温度下跑满 100 步，attack RM 从 step0 的 0.293 提升到 0.305（+0.013）。转 SFT 富化弧（v6.8→v6.9）之后，回头在 t0.9 产品温度重测 v6.7，结果崩了：

```
RM-attack-split（t0.7）：step0=0.293 → v6.7 step100=0.305（+0.013，轻微有效）
RM-attack-split（t0.9）：step0≈0.274 → v6.7 step100 < step0（有害无益）
RM-attack-split（t1.0）：step0=0.254 → v6.7 step100 更差
```

**温度越高，v6.7 防御越差，甚至不如 step0**。这被称为"温度脆"（temperature brittleness）。

**归因**：ROLLOUT_N=6 + rollout_temp=0.7 的组合，使得"t0.9 下的偶发破功"在每组 6 个样本里出现概率太低，进不了训练批次，RM 对它的梯度接近零。RL 只强化了"t0.7 分布下的化解行为"，对高温稀有破功尾巴毫无约束。

**这是 v6.7 评估协议错误（t0.7 测、t0.9 部署）和算法设计错误（n 太小）的双重暴露**。此后 attack probe 强制用 t0.9，成为不可撼动的规则。

**v6.10 的对应设计**：
- rollout_temp → **1.0**（比部署温度 0.9 更高，覆盖尾部）  
- ROLLOUT_N → **12**（让稀有破功在组内必然出现，拿到梯度）  
- 逐样本 `break_pen=-5`（单条破功立刻惩罚，不依赖 group 平均）

这三项改动不是调优，是**对温度脆根因的直接对症处置**。

### 4.8 val 噪声带 ±0.1：两次险些误判的操作教训

val 指标观测到了两次险些触发错误决策的异常：

```
B0（基线同 ckpt 两次测）：-0.136 / -0.250   差值 0.114，全是噪声
v6.7 clean val：step0→step100 变化 +0.037 ≈ clean 轴全量程（深读 2 §4.4）
```

**±0.1 是本项目 val 协议的噪声带**（n=4 rollout，固定 seed，200 条 probe）。在这个范围内的变化不是信号，是采样波动。

两次险情：
1. B0 测出 -0.136 时差点判"基线异常低、RL 没效果"，第二次测才发现 -0.250 才是正常范围；
2. clean val 100 步 +0.037 差点被误读成"RL 提升了 clean"，分轴归因才知道这是 clean 轴的全部量程（= 噪声）。

**操作规范**：
- val 突变 ≤ ±0.1 → 先做第二次测，确认噪声前不做决策；
- 用"连续 N 个 val 点同向"（本项目是 6 个点）而非单点来判断趋势；
- clean 和 attack 分轴打分，不用混合 RM 均值（混合分把两个独立信号的噪声叠加在一起，更难判断）。

---

## 5. 方法论小结

1. **参数要有可追溯的依据**：bonus 按 σ 标定、SAVE_FREQ 按崩溃间隔定、LR 形态按"总步数是否已知"选、util 按三档复测定——没有一个参数是"别人都这么设"；
2. **指标分量化 + 口径先行**：total reward 必拆分量落盘；指标突变先人工抽查原始样本验证口径，再谈模型（两次大发现都源于此）；
3. **区分噪声与信号**：±0.1 的 clean_RM 单点波动、偶发 grad_norm 尖峰是噪声；gap 持续恶化、6 个 val 零趋势是信号；温度脆是信号，不是"高温时 RM 精度低"；
4. **三个分布必须对齐**：训练分布、验证分布、业务分布——v6.5 的根本教训是三者错位时，RL 既学不到（没梯度）也测不出（val 失明）；
5. **rollout 温度必须 ≥ 部署温度**：这是 v6.7 温度脆留下的最具体教训，不是经验之谈，是实测代价；
6. **知道什么时候停**：RL 的杠杆（步数/温度/长度/KL/bonus）全部拉过且零移动 → 瓶颈在信号与数据，继续烧 GPU 是沉没成本谬误。

> 相关：奖励链与归因细节 → [深读 2](02_rm_data_reward_and_attribution.md)；显存/通信账 → [深读 1](01_memory_accounting_and_gpu_communication.md)；全程叙事 → [CASE_STUDY.md](../CASE_STUDY.md)。
