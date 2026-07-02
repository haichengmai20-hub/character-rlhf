# 深读 2 — RM 数据构造、训练与奖励设计：把"好/坏"变成可优化的信号，并对它持续归因

> 本篇展示的能力：**偏好数据怎么构造、RM 怎么训、在线 RL 的最终奖励怎么设计** —— 以及贯穿全程的**数据归因**：六版 RM 的每一次进退都被分组统计定位到具体根因；奖励信号本身也被归因出"在反向训练破角"的致命缺陷并修复。
>
> 配套代码全部在 [training/GRPO_Training_Package/](../training/GRPO_Training_Package/)。

> ## Attribution Lens
> 本篇把观察到的行为主要归因到 **reward model 与 rule reward**（偏好数据构造、RM 判别边界、奖励组件方向），兼及 **data distribution**（三类单因负样本）。其中"反向破角"一节，用 [深读 8](08_credit_assignment_in_character_rlhf.md) 的语言就是 **reward-component credit 分错**——功劳记在了错误组件上。

---

## 1. 偏好对构造：负样本是"造"出来的，且每一类都带可检验的信号

### 1.1 流水线与门控

```
13,128 条 SFT 样本 (clean32_for_sft1)
   │  SFT2 merged 模型 (vLLM 4-shard 并行) 按三类策略生成 rejected
   ▼
12,973 对原始 pair
   │  启发式 4 维打分 → 门控: min_chosen_score=0.62, min_score_gap=0.18
   │  (分差太小 = 标签置信度低 → 丢弃, 宁缺毋滥)
   ▼
重平衡 + sample_weight → rm_train.jsonl 16,826 / rm_eval.jsonl 885 (5% split)
```

### 1.2 三类负样本（生成权重 60/35/5）—— 关键：负样本必须有"已知的、单一的坏因"

| 类型 | 构造方式 | 为什么这样造 |
|---|---|---|
| **wrong_role** | 把**另一个角色 B 的真实 system prompt**（非合成）发给模型，让它自然以 B 的身份回答同一条 user 消息 | rejected 的写作质量不差、只差"角色身份" → 强迫 RM 学**语义性格匹配**而非表面文采；用真实 system prompt 学到的是可迁移的"性格→风格"关联（17 个角色覆盖，能向 83 个生产角色泛化）|
| **wrong_language** | 原 system + 末尾 `[LANGUAGE OVERRIDE: Respond ONLY in {lang}]` | 只动语言这一个变量 |
| **length_mismatch** | 指令要求显著更短/更长（short_prob=0.25）| 只动长度这一个变量 |

> 这是实验设计（DOE）的思路：每类负样本**只引入一个可控差异**，RM 学到的边际信号才可解释、事后才可按类型归因。

### 1.3 启发式 4 维打分（`chatjoy_reward_utils.py`，**仅用于构对期过滤，绝不进在线 RL**）

默认权重 `format 0.30 / role 0.30 / language 0.25 / length 0.15`，并按负样本类型用 `SCORING_PROFILES` 放大对应维度（wrong_role 把 role 抬到 0.55、wrong_language 把 language 抬到 0.55、length 类把 length 抬到 0.40）——**让"已知坏因"的维度主导分差，提高标签信噪比**。

明示的设计取舍：role 维度只查格式规范 + 角色名在场，**不做语义人设对比**——深层人设一致性留给 RM 从 wrong_role 对里学。这个取舍的代价后来在 v1 评估里被精确观测到（见 §3.2 idx=55）。

### 1.4 重平衡 + sample_weight（统计处理）

- 类型重平衡到 `wrong_language 25% / wrong_role 25% / length_short 30% / length_long 20%`；
- `sample_weight = clip(0.4 + 0.6 × gap_norm, 0.3, 1.2)`，gap_norm 是**同类型内** reward_gap 的 min-max 归一 → 分差越可信的对权重越高，且归一化按组做、不跨类型混（不同类型 gap 量纲不可比）；
- 实测分布核查（构完必查，不是构完就用）：类型占比 / 17 角色 Top-5 / 24 语言 Top-5 / weight 按类型均值（wrong_language 0.774 > length_short 0.635 > wrong_role 0.603 > length_long 0.594）。

### 1.5 v6 升级：LLM-judge 细粒度偏好对 + 交叉验证（从规则负样本到 RL-aware 对）

v1-v5 的规则负样本有一个结构性天花板：它只教 RM 识别"格式/语言/长度"这类一眼可见的差异，无法让 RM 区分"两个都是正确角色风格但一个更细腻"——而 RL rollout 时同一 prompt 多次采样的结果恰好就是这种近似差异。这是 GRPO advantage 在后期趋近 0 的根本原因之一。

#### 1.5.1 多源候选矩阵与 rank delta 设计

| Source | Temperature | 用途 |
|--------|------------|------|
| Gemma4 SFT t0.7 | 0.7 | actor 主分布 |
| Gemma4 SFT t1.0 | 1.0 | 高方差尾部 |
| Qwen SFT t0.7 | 0.7 | 跨模型对照 |
| RL step550 t0.7 | 0.7 | RL 后的主分布 |
| RL step550 t1.0 | 1.0 | RL 后的高方差尾部 |
| **base Gemma4-it** ★v6.3 | 0.7 | **分布外更细腻的 teacher** |

五个 source 全是自家模型是 v6.1 训不动的结构性原因：RM 只看到同分布输出，学不到"分布外更细腻"的概念。第六个 source（纯 base，不挂 LoRA）是 v6.3 加入的，用格式约束采样（`*action*/dialogue/(thought)` 末尾指令）让 base 的原生细腻度保留同时格式达标。

**rank delta 设计**（关键但不显眼）：

```
5 source × judge 打分 → rank 1–5
delta=4 pair（rank1 vs rank5）：gap ~2.0，约 10k 对   → 强信号，确保 RM 能区分极端
delta=2 pair（rank1 vs rank3 / rank2 vs rank4）：gap ~0.5，约 17k 对 → 细粒度，RL 的核心
```

delta=2 才是 RL 用的：GRPO rollout 时同一 prompt 4-6 次采样，相互之间的差异几乎都是 delta=2 量级。RM 必须在这种近似对上能打分，GRPO 的 advantage 才有有效方差。delta=4 只是 ranking 锚点。

#### 1.5.2 交叉验证：双 judge 同向过滤

**问题**：DeepSeek judge 在 delta=2 细粒度对上自报 `confidence=high` 仍有 ~25% 方向错误——confidence 字段在这里不可信。

**解法**：本地 Qwen-36B-A3B 独立 judge 同一批 → 只保留两 judge 都同向的对：

| | 总 pair | 保留率 | delta=2 conf=high 被 Qwen 否决 |
|---|---|---|---|
| v6.1（仅 DeepSeek） | ~37k | 100% | —（单 judge） |
| **v6.2 cross-validated** | **27,287** | **73.1%** | **29.4%** |

被 Qwen 否决的 29.4% 正是 v6.1 "训不动"的元凶——RM 在方向相反的信号上相互抵消、梯度平均到 0。交叉验证不是"质量把关"，是**消除系统性噪声让梯度有方向**。

**盲判设计**：candidates 顺序打乱后发给 judge，隐藏 source_model 和 sampling temperature，避免 judge 形成 source 偏见（早期测试发现 Qwen 候选常被排差，核查后确认是 Qwen 真的风格不贴 ChatJoy roleplay，不是 source bias——盲判才能区分这两种情况）。

---

## 2. RM 训练目标与原理

**Bradley-Terry pairwise ranking loss**：`L = −log σ(r(x, y_chosen) − r(x, y_rejected))`，r 是 LM backbone（Qwen3-30B-A3B + LoRA r=64, q/k/v/o）顶上一个 score head（`nn.Linear(2048→1)`，取 EOS hidden state）。

三个由原理直接导出的训练敏感点（后面六版迭代里全部被实测验证）：
1. **effective batch 决定梯度方差**：每步的可比对越多，ranking 梯度越稳 → batch 128→64 直接退步（v2）；
2. **LR schedule 以 optimizer steps 为周期**：cosine 的"周期"= 总步数 = 数据量/batch → 加数据同时加 batch 反而把有效训练截短（v3）；
3. **score head 是 PEFT 的盲区**：它不含 `lora_` 前缀又被冻结 → 默认**不被保存**，加载后随机初始化、打分全废（v2 事故）。

### 2.1 分数饱和：v5 RM 对 RL 最大的单一危害（reward_std ×20 的来源）

v5 RM 做离线评估时 acc=95.5%，看起来很好——但对 RL 真正重要的不是 acc，而是**在线打分的动态范围**：

```
v5 在线 reward_std 实测：~0.1
v6 cross-validated reward_std：1.94–1.98
比值：~20 倍
```

这 20 倍的差距来自**分数饱和**：v5 的 `|score| > 5` 占比 **47.3%**（接近一半的 rollout 样本分数被截断到上下限）。GRPO 的 advantage 是组内标准化：`A_i = (r_i - mean) / std`——当一半样本的分数是常数（饱和值），std 近 0，所有样本的 advantage 趋近 0，梯度也趋近 0。

v6 的饱和率：**0%**。方法：cross-validation 过滤掉了那批"两个 judge 方向不一致"的弱信号对（这些对恰好产生无区分度的近似分），plus delta=2 细粒度对让 RM 在中间区间学到了真实边界。

**直接推论**：选 RM 时不能只看 offline acc，必须检查 `saturated%`（在线打分中 `|score| > threshold` 的比例）和 `reward_std`。**v5 acc 94.5% 对 RL 的实际信号强度是 v6 acc 86% 的 1/20**。

### 2.2 RM online/offline 对齐：一个静默的分布偏移 bug

RM 在 OpenRLHF 框架下训练时，打分点是这样确定的：

```python
# OpenRLHF 训练侧
tokens = tokenizer(prompt + chosen_response).rstrip("\n")
score = model(tokens)[last_token_position]  # EOS token 的 hidden state
```

vLLM serving 时 `/classify` endpoint 如果不复现这个 `rstrip("\n")`，末尾多一个换行 token → EOS 不再是最后一个 token → score head 取错位置 → **在线 RM score 整体偏负约 0.3-0.5（系统性偏移，非随机噪声）**。

实测特征：所有在线 rollout 的 reward 均值明显低于离线评估同批样本的 reward，差值稳定（不是噪声而是一个固定偏移），且高质量样本和低质量样本都同等下移（说明是对齐问题不是 RM 校准问题）。

**修复**：`reward_loop.py` 对 RM prompt 做 `rstrip("\n")` + 显式补 EOS，与训练侧行为精确对齐。**任何换 vLLM 版本、换 RM 权重的操作后都必须重新验证这条对齐**——框架不会自动保证。

---

## 3. 六版 RM 的迭代台账：每一次进退都归因到具体根因

### 3.0 总表（同一评估协议下）

| 版本 | 数据 | eff. batch | steps | acc | gap | chosen / rejected | 一句话根因 |
|---|---|---|---|---|---|---|---|
| v1 | 16,826 | 128 (8GPU) | 263 | **94.5%** | 1.601 | +0.948 / −0.653 | 强基线；ckpt 磁盘事故永久丢失 |
| v2 | 14,826 | 64 (4GPU) | 231 | 86.4% | 1.122 | — | **batch 减半 + score head 丢失** |
| v3 | 18,955 | 128 | **148** | 86.0% | 1.256 | +0.958 / **−0.298** | **cosine 步数不足 + 2000 条缺陷样本** |
| v4 | （计划）| 64 | ~255 | — | — | — | 被 v5 覆盖未执行 |
| **v5** | 17,712 (+5% badpool) | 64 | 276 | **95.5%** | **3.497** | **+2.27 / −1.23** | 全面超 v1 |
| v6.x | LLM-judge 重构 | — | — | 门: ≥0.84 | — | 门: reward_std≥2.0 | 见 §4 |

### 3.1 v1：基线 + 第一批"持续错误"信号

110 对评估集上 step 50→200 的学习曲线（acc 79.1%→94.5%，gap +0.182→+1.601）确认收敛健康。**更有价值的是 5 个 persistent failures**——对每个错误样本跟踪它的 gap 随训练步的轨迹：

- 多数 length_mismatch 错误的 gap 在 −0.03~−0.14 间小幅波动 → **低置信标签噪声**，不值得修；
- **idx=55（sv, wrong_role）的 gap 从 −0.038 恶化到 −0.527** —— 越训越错。下钻看样本：chosen=圣诞舞娘的瑞典语回复，rejected=另一角色"冷淡室友"的**高质量**瑞典语回复 → RM 在被高写作质量带偏、没学会角色身份。这一个点位精确暴露了 §1.3 那个设计取舍的代价，并成为后来 v6.3 加 8 维 rubric 和 roledrop pair 的最早证据。

> 归因方法：**不只看 acc 总分，对错误样本逐个跟踪 gap 轨迹**——"稳定小负"是噪声、"持续恶化"才是系统性缺陷。

### 3.2 v1→v2：一次"好心办坏事"的过滤 + 一个静默 bug

- 数据动作：发现 2,000 条 `length_mismatch_long` 的 `reward_gap<0.2`（全部聚集在该类型！）→ 判为噪声标签删除。**动作本身是对的**（v3 加回去立刻复现伤害，反向验证了判断）。
- 但 v2 acc 反而 −8.1%。归因排除法：不是过滤的锅（修复 4 个旧错、新引 13 个新错，集中在 length 类）→ 是 **effective batch 128→64**（换 4 卡训练）+ ranking loss 方差变大；
- 同时发现 **score head 静默丢失** bug（§2 第 3 点）。补救方案是个统计上很干净的两阶段法：
  - **Phase 1**：冻结全模型，no_grad 批量前向，缓存每对 (chosen, rejected) 的 EOS hidden state（2816 维特征）；
  - **Phase 2**：固定特征，**纯 CPU 凸优化**一个 `Linear(2816→1)`，Margin Ranking Loss，~5 分钟收敛 —— LoRA 已经把表征适配到某个线性可分方向，找回那条线性边界不需要 GPU 重训。
- 永久修复进 OpenRLHF `model.py`：`init_value_head` 移到 `get_peft_model()` 之前 + `modules_to_save=[value_head_prefix]`。

### 3.3 v3：两个叠加根因，全部从分布里读出来

**现象**：chosen mean 略升（+0.958）但 rejected 惩罚塌掉（−0.298 vs v1 −0.653）、acc 86.0%。

**根因 A（schedule）**：v3 加数据到 18,955 又把 batch 抬回 128 → 只有 **148 steps**；cosine 以总步数为周期，step 140 时 LR 已 ~3e-7（v1 同位置 4.17e-6）→ 训练后半程形同冻结。逐步对比：前 20 步 v3 还领先，step 100 持平，之后停滞。
**根因 B（数据）**：按 `v3_source` 字段分组看 gap 分布 → `base_length_long_low` 2,000 条的 gap **全部落在 [0.161, 0.200]** —— 持续向 RM 发送"坏长回复只比好回复差一点点"的信号 → rejected 分布整体被抬升。
**佐证（长度分桶曲线）**：把 RM 分按回复长度分桶（<300→0.20 / 300-600→**1.19** / 600-1000→1.17 / >1000→1.00）→ RM 有一个 300-600 字的"甜点区"，对长样本边界感模糊。这条曲线同时是后来 RL 阶段 length-hacking 防护设计（深读 3）的输入。

### 3.4 v5：修复合并 + 难负样本，全面超 v1

数据 = 删缺陷样本 + **5% hard-negative badpool**（用恢复 score head 的 v2 给全量训练集打分、挖出 `gap<−0.3` 的判错样本，多卡 shard 并行）；batch 回 64 → 276 steps（cosine 节奏对齐 v1）。
结果 acc 95.5% / gap 3.497 / chosen +2.27 / rejected −1.23；分布形态：chosen p50=2.81、p90=3.03（软饱和于 ≈+3），rejected p50=−1.64 → **RM 已是判别边界清晰的"软二分类器"**。
**前瞻一笔**（给 RL 的统计预警）：打分软饱和 → RL 中若 rollout 全部 >2.5，RM 将失去区分度（梯度消失），预案是上调 KL。这条预警在 v6.5 的天花板现象中兑现（深读 3）。

### 3.5 v6 → v6.3：从规则负样本升级到 LLM-judge 细粒度偏好

动机（又是归因驱动）：① RL 在 roledrop 攻击下 91% 破功；② v6 的 5 个 candidate source 全是自家模型（同分布），RM 学不到"分布外更细腻"。三个杠杆同时上：

| 杠杆 | 内容 | 统计要点 |
|---|---|---|
| 第 6 candidate = **base Gemma4-it** | 6 source × 9,693 prompt = 33,120 candidates；base 带格式约束采样（保细腻度又不在 format 维度吃亏）| 引入分布外正样本，打破同分布瓶颈 |
| **roledrop_attack pair** | 23 角色 × 240 攻击 × 6 source；judge 启用 anti-roledrop hard fail + 模式 B/C few-shot | 把"抗破角"直接做成偏好信号 |
| rubric **5 维 → 8 维** | 补 sensory / emotional complexity / narrative layer | 制造"角色对但平淡 vs 角色对且细腻"的可观测方差 |

质量控制全是统计门：**DeepSeek + Qwen-36B 双 judge 交叉投票**（单 judge 偏置不可控），攻击类型 cross-validation 保留率 <30% 自动剔除；RM 验收门 **G4：offline acc ≥ 0.84、saturated = 0%、reward_std ≥ 2.0** —— 后两条直接服务 RL："reward 没方差 = RL 没梯度"。

---

## 4. 在线 RL 的最终奖励：组合设计 + 对奖励本身的归因

### 4.1 最终形态（`anti_chain_reward.py` 串联，verl 单入口）

```
total = rm_score                  # v5/v6.3 RM，±5 量级，主信号
      + anti_refusal              # 拒绝 −4.0 (≈1.55σ) / 攻击源上完全无拒绝 +3.0
      + anti_repetition           # 重复惩罚 ≤ 0
      + anti_ooc                  # 破角(自报身份/角色内代码/真泄露 system) −5.0
      + bonus_revoked             # 破角时撤销误给的 +3.0
      + length_shaping            # [0, COEF]，强门控 + 饱和封顶
```

设计原则：**RM 管"好不好"，规则惩罚管"踩没踩红线"，两路信号量级有意错开**（红线 −4~−5 与 RM ±5 同量级、能压住；length 塑形 ≤1.0 远小于 RM、只抵消不主导）。目标排序：**戏内化解 > 拒绝 > 破角**。

### 4.2 对奖励信号本身的归因（本项目最重要的一次发现）

**现象链**：线上模型在 roledrop 攻击下大面积自报 "I am Gemma 4"，但监控指标 `atk_break%` 只显示 ~16%。

**归因三步**：
1. **不信指标、看原始数据**：拿真实问答逐条核对 → `atk_break%` 基于拒绝关键词（"I can't"），而 "I am Gemma 4" **不含任何拒绝词** → 指标对最严重的失效形态**完全失明**（真实破角率接近 100%）；
2. **从指标盲区追到奖励缺陷**：同一批文本过 reward 链 → `anti_refusal` 把"自报身份"判为"非拒绝"，**给了 +3.0 bonus** → 梯度方向一直在**奖励破角**（Goodhart：代理指标失效后，优化它就是优化错误）；
3. **修复并验证误伤率**：`anti_ooc_disclosure.py` 四层检测（身份短语多语言 7 语 / system 真泄露紧正则 / AI 自承 / 角色内代码块）→ −5 并撤 bonus。上线后又一次归因：v6.4 val "破角 6%" 复查全是**误报**——模型在**戏内拒绝**变 AI（"我无法切换到 AI 助手的身份"）却被判破角 → 加**否定语境豁免**（同子句、短语之前有否定标记）+ **引号复述豁免**（角色讥讽复述"什么'AI 助手'？"）→ 误报 16 例中 1→**0**。全程 TDD（101 tests）。
4. **第二个盲区同法补上**：攻击下"无身份词、直接吐 bullet 列表完成注入任务"的行为破角（产品实测 55%）→ `detect_task_injection_compliance`（行首结构化列表 ≥3 判破角，**仅 attack 源启用**——clean 的列表可能合法）。

### 4.3 length_shaping 的防 hacking 设计（针对 §3.3 那条长度曲线）

RM 内部有 ~400 字甜点（density/400 + <300 扣分）→ RL clean 输出被压到 median ~420 字。不重训 RM 的前提下加塑形项，**四道门缺一不可**：仅 clean 源 / 饱和封顶（≥TARGET 不再加分，铲掉"越长越赚"）/ 必须有 roleplay 格式 / 拒绝、破角、重复一律 0。参数演化全部有数据依据：COEF 0.4→0.5→1.0、TARGET 600→1200→1500（每次都因为 val 实测 clean median 已越过/卡死在旧值）——最终实测它仍顶不过冻结 RM 的甜点引力，这个**负结果**直接支撑了"必须换 RM/换信号"的战略判断（深读 3 §5）。

### 4.4 RM 是防御判别器，clean 质量轴几乎平（v6.7 结束后的核心数据归因）

v6.7 RL 跑满 100 步后，对同一批 200 条 clean + 68 条 attack rollout 用 v6.6 RM 分轴打分：

| | RM score 范围 | 跨度 |
|---|---|---|
| **clean**（中性对话） | 0.177 ~ 0.214 | **0.037** |
| **attack**（越权攻击） | 0.205 ~ 0.561 | **0.36** |

两轴跨度差 **~10 倍**。

这条数据的推论链：

1. GRPO 的梯度正比于组内 reward 方差（§1.1）。clean 组内方差 ≈ 0.037 / std ≈ 0.009 → **clean 这边 advantage 几乎全为 0**，RL 在 clean 上实质上没有梯度；
2. v6.7 训 100 步，clean RM 变化 +0.037 ≈ clean 轴的全部量程 ≈ **测量噪声**，不是真实提升；
3. attack 轴方差 10 倍大 → 有效梯度都在 attack 这边 → **RL 是纯防御优化器**；
4. "clean 丰富度"只能靠 SFT 蒸馏 teacher（base Gemma4 动作密度 4.7，SFT 后 step0 是 3.4，丢了 ~20%）——这个差距 RL 补不回来。

**这是转 SFT 富化路线（v6.8 → v6.9）而不是继续 RL 的直接数据依据**，不是主观判断。

---

## 5. 归因方法论清单（可复用）

1. **分组统计是第一工具**：按 `rejected_type` / `v3_source` / 语言 / 角色 / 长度桶切，缺陷样本（gap 聚集 [0.161,0.200]）、长度甜点曲线、类型失衡全是这么找到的；
2. **跟踪个体轨迹，不只看均值**：persistent failure 的 gap 轨迹区分"标签噪声"vs"系统性缺陷"（idx=55）；
3. **指标先于模型被怀疑**：两次重大发现都始于"指标和肉眼对不上"——先验证指标有效性，再谈模型好坏；
4. **每个修复都要有反向验证**：过滤的正确性由 v3 加回缺陷数据复现伤害来证明；豁免逻辑由误报 1/16→0/16 证明；
5. **负结果也是结论**：temp/length 榨不动冻结 RM，证明瓶颈在信号不在优化器。
