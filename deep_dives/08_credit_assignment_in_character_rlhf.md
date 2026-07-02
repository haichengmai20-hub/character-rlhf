# 深读 8 — Character RLHF 中的 Credit Assignment：sequence→token 错配如何决定这个项目的成败

> 本篇展示的能力：**把本项目里几次最贵的成败，重新读成"对同一个 credit assignment 错配（sequence reward → group advantage → token loss）在不同层上的回应"** —— 而不是把 credit assignment 当成一堆 GRPO 名词罗列。核心主张：GRPO 只负责给出**更新方向**；这个项目里真正决定方向打在"破角 span"上还是"整条回复 / 错误 reward 组件 / 错误样本源"上的，是 credit assignment。本项目几次转向（anti-OOC 修复、富度归 SFT / 防御归 RL、GSPO、per-source KL、v6.10 停线）拆开看，全都是在补这个错配的某一层。
>
> 概念底稿见 [docs/CREDIT_ASSIGNMENT_TAXONOMY.md](../docs/CREDIT_ASSIGNMENT_TAXONOMY.md) 与 [docs/GRPO_LOGPROB_FLOW.md](../docs/GRPO_LOGPROB_FLOW.md)；本篇不重复推导公式，只把这套框架**套回本项目的真实数字与真实决策**。所有引用数字均取自仓库既有记录（[CASE_STUDY](../CASE_STUDY.md) / [RLHF_WORKFLOW](../RLHF_WORKFLOW.md) / [VERSION_MAP](../docs/VERSION_MAP.md) / [v69_reconcile_report](../reports/v69_reconcile_report.md)）；本篇**新提出的诊断指标一律标注 `Proposed diagnostic — not yet run`，未声称已跑实验**。

> ## Attribution Lens
> 本篇解释项目里观察到的行为应归因到：
> - **rule reward**（组件把 credit 分给了错误行为，如 anti_refusal +3.0）
> - **reward model**（clean 轴无方差 → 无有效 group credit）
> - **GRPO optimization**（sequence advantage 广播到 token → 局部错误全局惩罚）
> - 以及这三者与 **evaluation**（val≠产品分布）、**deployment**（采样只改表达）的边界。

---

## 0. 反直觉的起点：一次 credit 分错，把 base 的防御训坏了一倍

和深读 5 一样，本篇从一张推翻默认心智的表开始。默认心智是"reward 设对了，RL 就会往对的方向走"。本项目的早期数据直接推翻它：

| 模型 | roledrop 破功率 | 相对 base |
|---|---|---|
| **base_it**（未经 RL） | **27%** | 基线（防御最强） |
| 早期 RL v2 | 55% | **×2 更差** |
| 早期 RL v1 | 91% | ×3.4 更差 |

来源：`CASE_STUDY.md` §3.4 / `README.md`。根因**不是** reward 缺失，也不是优化器不稳，而是一次**层级错配的 credit 分配**：

```
真正的错误发生在一个局部 span："I am Gemma 4"
但 reward 是 sequence-level 的：anti_refusal 看整条回复"没有拒绝词" → 给 +3.0 NON_REFUSAL_BONUS
于是这条"自报 AI 但继续回答"的回复拿到了正 advantage
GRPO 把这个正 advantage 广播到该回复的每一个 token
→ 包括那个 "I am Gemma 4" 的破角 span 也被强化
→ RL 高效地、稳定地把破角训成了高概率行为
```

**这就是本篇要讲的错配的第一个、也是最贵的实例**：sequence-level reward 无法区分"整条回复好"和"整条回复里有一小段错"，GRPO 的 token-level loss 又把这条回复的统一 advantage 摊到每个 token 上。reward 设计"方向"甚至是对的（想奖励不拒绝），错的是 credit **落点**——它落在了一个本该被硬压制的 span 上。

下面把这条主线拆成"错配在哪一层、在本项目造成了什么、项目怎么补、还剩什么"。

---

## 1. 错配的解剖：三个层级各说各话

Character RLHF 一次 GRPO 更新的信息流（完整逐步版见 [GRPO_LOGPROB_FLOW.md](../docs/GRPO_LOGPROB_FLOW.md)）：

```
用户输入 x
  → actor 生成 response y = (a_1, ..., a_T)              [token 序列]
  → RM + rule reward 给整条 y 一个标量 R(x, y)          [sequence-level]
  → 同 prompt 的 G 条回复组内标准化 → advantage A_i     [group-level]
  → A_i 广播到 y 的每个 token，乘 token ratio ρ_i,t     [token-level]
  → clipped loss + KL，更新 actor
```

三个层级用的是**不同粒度的信息，却要互相传递**：

| 环节 | 层级 | 它知道什么 | 它不知道什么 |
|---|---|---|---|
| `R(x, y)` | sequence | 整条回复好不好 | 好/坏是**哪一段**造成的 |
| `A_i` | group | 同 prompt 下这条相对更好还是更差 | 组内 reward 若无方差，它只是噪声 |
| token loss | token | 每个 token 概率上调还是下调 | 它拿到的 `A_i` 对全序列是**同一个值** |

一句话概括错配：

```
Reward 是 sequence-level 的（知道好坏，不知道位置）
Advantage 是 group-level 的（知道相对，不知道分源）
Loss 是 token-level 的（能定位，但拿到的信号已经被摊平）
```

**本项目的每一次踩坑，都是这三行里某一处"信息在传递中丢了"的后果。** 后面 §4 逐个对应。

---

## 2. 六个层级（scaffolding，不展开）

概念完整版在 [CREDIT_ASSIGNMENT_TAXONOMY.md](../docs/CREDIT_ASSIGNMENT_TAXONOMY.md)。这里只保留一张对照表当坐标系，供 §4 引用：

| 层级 | 核心问题 | 本项目真实例子 | 主要风险 | 诊断方式 |
|---|---|---|---|---|
| token | 哪些 token 被上调/下调？ | "Gemma / AI model" token | 无关 token 被一起惩罚 | token ratio / token KL proxy |
| span | 哪个片段触发 reward/penalty？ | self-disclosure span | 局部错 → 全局罚 | span detector 命中率 |
| sequence | 整条回复为何得分？ | RM score + rule reward 之和 | 长度/富度/身份混在一个标量 | reward_per_token / 组件 breakdown |
| group | 同 prompt 多回复如何排序？ | G 个候选角色回复 | reward_std≈0 → advantage 是噪声 | group_reward_std |
| source | 哪类样本驱动更新？ | clean vs attack | clean 动、attack 不动 | source-wise advantage_std |
| batch | batch 平均是否掩盖问题？ | multilingual 混合 batch | 小语种被平均掉 | per-language breakdown |

**本项目已经动过的层**：sequence（anti-OOC 硬罚重排优先级）、group（reward_std≥2.0 验收门）、source（per-source KL）。**尚未系统动过的层**：token 与 span（§4.5 说明这正是 v6.10 之后 attack 压不下去的前沿）。

---

## 3. credit 在 GRPO 里怎么传（只保留错配相关的三个公式）

完整变量与执行顺序见 [GRPO_LOGPROB_FLOW.md](../docs/GRPO_LOGPROB_FLOW.md)。这里只取和错配直接相关的三步。

**(1) sequence reward 是组件之和**（本项目实际的奖励链，见 [RLHF_WORKFLOW §4.1](../RLHF_WORKFLOW.md)）：

```
R(x, y) = r_RM            # 防御判别器主信号，±5 量级
        + r_anti_ooc      # 破角披露 -5.0（逐样本）
        + r_anti_refusal  # 拒答 -4.0 / non-refusal +3.0
        + r_length        # length_shaping [0, COEF]，饱和封顶
        + r_repetition    # 复读惩罚 ≤0
        + ...
```

注意：这些组件**全部是 sequence-level 的**——每一项都对整条 `y` 给一个数，没有一项告诉你"扣分该落在第几个 token"。

**(2) group advantage 把 reward 变成相对信号**：

```
A_i = (R_i - mean(R_1..R_G)) / (std(R_1..R_G) + ε)
```

关键性质：`A_i` 对同一条 response 的**所有 token 相同**。

**(3) token loss 把这个统一的 A_i 摊到每个 token**：

```
ρ_i,t = exp(log π_θ(a_i,t | ...) - log π_old(a_i,t | ...))     # token-level
L = - E_i,t[ min(ρ_i,t · A_i, clip(ρ_i,t, 1-ε, 1+ε) · A_i) ] + β·KL(π_θ || π_ref)
```

其中 `π_old` = rollout 时冻结的 actor 快照，`π_ref` = 冻结的 SFT/init actor（KL 锚，**不是** ratio 分母）。

**错配点就在 (2)→(3)**：`A_i` 是整条回复一个值，`ρ_i,t` 是逐 token 的。一条回复得负分，它的每个 token 都倾向被压制——哪怕坏的只是 68 个 token 里的 4 个。GRPO 给了方向（这条回复该往下压），但它**没有能力**说"只压这 4 个"。这一步在本项目里的直接后果，就是 §4.1 的反向破角和 §4.5 的 attack 压不动。

---

## 4. 错配对本项目的实际影响（本篇的重心）

下面每一小节都是同一个模板：**现象 → 归因到哪一层 credit 错配 → 版本对比 → 项目做了什么 → 还剩什么**。这才是本篇要回答的问题，而不是"credit assignment 有几层"。

### 4.1 反向破角：reward 组件把 credit 分给了破角 span（sequence + 组件层）

- **现象**：RL 后模型在 roledrop 攻击下大量自报 "I am Gemma 4"，破功率 base 27% → RL 55%/91%。旧关键词指标 `atk_break%` 却只显示 ~16%（对自报身份完全瞎）。
- **归因（组件层 credit 分错 + sequence 层无法定位）**：
  - 组件层：`anti_refusal` 的逻辑是"只要没有拒绝词就 +3.0"。"我是 AI，但我可以继续陪你"里没有拒绝词 → 拿满分。**credit 被分给了"错误的 reward 组件"**——一个本该奖励"配合度"的组件，奖励了"破角式配合"。
  - sequence 层：这条回复 95% 是正常 RP，坏的只是身份披露那一个 span；但 reward 是整条的，正的 sequence reward → 正 advantage → 破角 span 也被强化。
- **版本对比**：base（无 RL，27%） vs RL v2（55%） vs RL v1（91%）——RL 越训防御越差，斜率与"reward 在奖励破角"一致。
- **项目做了什么（重排优先级，不重训 RM，全 TDD）**：新增 `anti_ooc_disclosure.py`，对身份披露/角色内代码/system 泄露**硬罚 -5**，并**撤销**误给的 +3.0 bonus（`bonus_revoked`）。使 reward 排序变成：

  ```
  identity consistency > non-refusal helpfulness > richness
  戏内化解 > 拒绝 > 破角泄露
  ```

  证据：68 条 attack probe 的 OOC 误报 1/16 → 0/16。
- **还剩什么**：这一步修的是 **reward 方向**（不再把 credit 分给破角组件），但**没有修 sequence→token 的定位**。-5 仍然是整条回复的——见 §4.5，这正是后来 attack 压不动的原因。

> **机制提炼**：这是一次教科书级 Goodhart（详见深读 2），但从 credit assignment 视角看，它更精确地是"**reward-component credit misallocation**"：不是 reward 太少，是 reward 的功劳/罪责记在了错误的组件和错误的行为上。

### 4.2 clean 富度 RL 提不动：group 层根本没有可分配的 credit（group 层）

- **现象**：v6.7 想用 RL 提 clean 富度，100 步只 +0.037 ≈ 噪声。
- **归因（group 层无方差）**：RM 在 clean 上是一个**几乎平的判别器**——clean 打分跨度只有 **0.037**，attack 跨度 **0.36**（差 ~10×）。同一个 clean prompt 采 G 条，reward 几乎一样 → `std(R_group) ≈ 0` → `A_i ≈ 噪声/0`。**group 层没有可排序的 credit，token loss 拿到的就是噪声。**
- **版本对比**：clean 轴（跨度 0.037，RL +0.037≈噪声） vs attack 轴（跨度 0.36，RL 确实在动）。同一个 RM、同一套 GRPO，仅因 group 层方差差 10×，一个轴能优化、一个轴空转。
- **项目做了什么**：把富度**移出 RL、交给 SFT 蒸馏 teacher**（v6.8→v6.9 弧）；RL 只保留有 group 方差的 attack 轴。这就是"富度归 SFT、防御归 RL"分工的 credit 层解释。
- **还剩什么**：SFT 富化又侵蚀了防御（step0 真破功 16.2% → 富化后 ~33.8%，见深读 4），引出 v6.10 的 clean-aware RL——这本身又是一个 source 层 credit 问题（§4.4）。

> **机制提炼**：GRPO 的前提是"同 prompt 多回复有 reward 方差"。方差不是优化器能造出来的，是 RM 给的。**group 层没有 credit 可分，再正确的 token loss 也无米下锅**——这条直接对应深读 5 的门 B。

### 4.3 length hacking：长序列垄断了 token-credit 质量（token 层的长度泄漏）

- **现象/风险**：Gemma4 有 `<Thoughts>` 推理段（平均 81 tokens）；若不处理，RL 会通过"写更长"拿更高 reward。
- **归因（长度 credit 泄漏）**：机制是 token 层的——正 advantage 逐 token 生效，**回复越长，被正向强化的 token 就越多**：

  ```
  长回复 → RM/length 给更高 reward → 正 advantage
        → 更多 token 拿到正 credit → 模型学会更长 → 循环
  ```

  即"advantage token mass = advantage × 长度"被长回复垄断。
- **项目做了什么（多层封堵，全是 credit 层干预）**：
  - GSPO `seq-mean-token-mean`：**序列级等权**，长序列不因 token 多而主导梯度（把 token-credit 质量重新拉平到序列）；
  - `MAX_RESPONSE_LENGTH=768`：物理截断，最硬的一层；
  - `length_shaping` 4 道门：仅 clean / 饱和封顶（≥TARGET 不再加分）/ 必须有 RP 格式 / 破角拒绝重复一律 0；
  - Thoughts 剥离后再计长。
- **还剩什么**：这一层项目处理得比较干净——因为长度是**可显式度量**的 credit 泄漏，容易拦。对比之下 §4.5 的破角 span 不可显式度量，就难得多。

### 4.4 clean/attack 混训：per-source KL 是项目已经建好的 source 层 credit 干预

- **现象/风险**：clean 和 attack 混在一个 batch 里，batch 平均 reward 上升 ≠ attack 真的改善；且 RL 对 attack 的更新可能把 clean 分布带跑（clean 在 RM 上无梯度保护，见 §4.2）。
- **归因（source 层 credit 混叠）**：clean 有富度回退风险但无 RM 方差，attack 有 RM 方差但样本稀有——两类样本的 credit 若不分开，会互相污染：attack 侧的更新会推着 clean 一起漂。
- **项目做了什么（source 层解耦）**：per-source KL——

  ```
  clean source：只用 clean_q（便宜结构代理）+ KL_CLEAN=0.008 紧锚 SFT actor
                → 物理上不给 RL 推离 clean 的信号
  attack source：rm + break_pen(-5) + λ·clean_q（λ=0.25）+ KL_ATTACK=0.002 松锚
                → 给防御留移动空间
  ```

  **这就是一次显式的 source 层 credit 分配**：把"能改 attack"的 credit 和"不许动 clean"的约束按样本源分开。比全局 KL 精确得多。
- **版本对比**：v6.10 step90 结果——clean **守住了**（source 解耦成功），但 attack 真破功 **36.8% ≈ v6.9c 33.8%**（未改善）。即 source 层 credit 分对了，但 attack 内部的 credit 仍没落到点子上 → 交棒 §4.5。

> **机制提炼**：per-source KL 证明这个项目**已经在 source 层做 credit assignment 了**，只是当时没用这个词。它成功地隔离了 clean，却隔离不了 attack 内部"整条 -5 vs 只该罚一个 span"的问题。

### 4.5 v6.10 之后 attack 仍压不动：token/span 层 credit 是未解前沿

- **现象**：reward 方向已修（§4.1）、source 已解耦（§4.4）、RM 有 attack 方差（§4.2），但 v6.10 step90 attack 真破功仍 **36.8%（25/68）**，attack-split RM **0.235 < v6.9c 0.271**，A6 code 11→12。采样 sweep 只能把地板微调到 ~30%。
- **归因（token/span 层 credit 从未被定位）**：把前四节连起来看——项目补齐了 sequence 方向、group 方差、source 解耦，唯独**没有把 -5 的罪责定位到破角 span**。一条 attack 回复被判 -5，GRPO 把这个负 advantage 摊到全部 token；破角 span 只占少数 token，稀有破功尾巴在组内出现概率又低（即使 N 从 6 提到 12）。**负 credit 被稀释在整条正常 RP 上，落到破角 span 上的那部分不足以压住它。** 这是 token/span 层 credit 缺位的直接表现。
- **版本对比**：v6.9c（SFT 天花板，33.8%） vs v6.10 step90（RL 后 36.8%）——RL 在已修方向、已解耦 source 的前提下**仍未改善 attack**，排除了"方向错""source 混叠"，剩下的嫌疑正是 token/span 层。
- **项目做了什么**：**停线**——不再同配堆步数，转向 quality-aware benchmark / RM v7 / 定向 A6+身份披露数据（见 [HANDOFF_next_training_quality](../docs/HANDOFF_next_training_quality.md)）。
- **下一步（`Proposed diagnostic — not yet run`）**：本篇主张下一轮应显式引入 **span 级 credit**：
  - 在 rollout 时用 `anti_ooc_disclosure.py` 的检测器标出破角 span 的 token 区间；
  - 记录 `bad_span_token_kl_share`（更新是否集中在坏 span）、`good_span_kl_share`（正常 RP 是否被误伤）；
  - 探索把 -5 从 sequence-level 改造成 **span-masked penalty**（只在披露 span 的 token 上施加更强的负信号）。
  - 以上均为**尚未实施的诊断/设计假说，非已跑实验**。

> **机制提炼**：这是本项目最诚实的一条结论——**修 reward 方向、修 group 方差、修 source 解耦，都不能替代 token/span 级 credit 定位**。attack 压不下去的地板，很可能不是"再 RL 几步"能解决的，而是要把 credit 从 sequence 下沉到 span。

### 4.6 val 11% ≠ 产品 55%：batch/source 层 credit 覆盖为零

- **现象**：v6.5 硬化 GRPO 跑 375 步零移动；后来发现训练 val 的 attack 破功率 11% ≠ 产品真实 roledrop 55%。
- **归因（batch/source 层 credit 覆盖缺口）**：产品最高频的 bullet 注入攻击在 val 里只占 ~5%。这类失效在训练 batch 里**样本量近乎为零 → 没有梯度可分配**，同时 val 也**测不出**它有没有改善。这是 batch 层 credit 的极端情形：一个 source 在 batch 里权重≈0，token loss 再正确也无从优化它。
- **项目做了什么**：把 attack probe 统一换成 **68 条产品形态攻击词典**，与训练 val 隔离；此后成为 v6.9 全程验收口径。
- **还剩什么**：覆盖补上了，但补上后暴露的正是 §4.5 的 token/span 层难题。

---

## 5. Reward 组件与 credit 风险对照（从项目奖励链提炼）

把 §4 的组件层教训整理成一张可复用表（组件来自 [RLHF_WORKFLOW §4.1](../RLHF_WORKFLOW.md) 的真实奖励链）：

| Reward 组件 | 目标行为 | credit 风险 | 本项目证据 | 诊断（部分为 proposed） |
|---|---|---|---|---|
| RM score | 综合偏好 | 偏长回复 / 高资源语言；clean 轴无方差 | clean 跨度 0.037 vs attack 0.36 | RM breakdown、reward_per_token |
| anti-OOC penalty (-5) | 防自报 AI/破角 | 局部 span 导致整条负 credit | §4.5 attack 压不动 | OOC span detector（`Proposed`） |
| anti-refusal bonus (+3) | 降无意义拒绝 | 误奖"我是 AI 但继续回答" | §4.1 base27→55 | OOC gate before bonus（已实施） |
| length shaping | 控长度 | length hacking | §4.3 | reward-length 相关（`Proposed`） |
| repetition penalty | 减重复 | 误伤角色口癖 | — | repetition span review（`Proposed`） |
| language consistency | 保持用户语言 | 小语种 calibration 不稳 | v6.9c zh-tw mono 漂到 24% | per-language metrics（`Proposed`） |
| format consistency | 保持 RP 格式 | 过度格式化 | — | format pass + 人工抽读 |

---

## 6. 诊断指标（按层级，全部标注是否已用）

> 说明：本项目**已经在用**的：group_reward_std（RM 验收门 reward_std≥2.0）、reward_component_breakdown（anti_chain_ooc / bonus_revoked 分项日志）、source 分轴 RM（clean/attack split）。**下面标 `Proposed` 的是本篇提出、尚未实现的诊断，不是已跑实验。**

### Group-level（部分已用）
```
group_reward_mean / group_reward_std        # 已用（reward_std≥2.0 验收门）
best_minus_worst_reward                      # Proposed
advantage_sign_balance                       # Proposed
```

### Sequence-level（部分已用）
```
reward_total / reward_component_breakdown    # 已用（RL 日志分项）
reward_per_token = reward_total / length     # Proposed（排长度偏置）
length_reward_corr = corr(length, reward)    # Proposed
rm_score_vs_rule_score_gap                    # Proposed
```

### Span-level（全部 Proposed）
```
bad_span_text / bad_span_token_ratio          # Proposed
local_error_global_penalty_flag               # Proposed
span_detector_hit_rate                        # Proposed（可复用 anti_ooc_disclosure 检测器）
```

### Token-level（全部 Proposed）
```
old_logprob / new_logprob / ref_logprob       # 框架内部有，未单独导出为诊断
token_ratio = exp(new_logprob - old_logprob)  # Proposed
token_kl_proxy = new_logprob - ref_logprob    # Proposed（见下方注意）
bad_span_token_kl_share                        # Proposed
```

> ⚠️ `token_kl_proxy = new_logprob - ref_logprob` 只能作为 **sampled-token 上的 KL 诊断 proxy**，不等于完整 vocabulary KL，不能声称是完整 KL。

### Source-level（部分已用）
```
source_reward_mean / source_reward_std        # 已用（clean/attack 分轴）
source_advantage_std                           # Proposed
source_token_kl                                # Proposed
source_ooc_rate / source_format_pass_rate      # Proposed
```

### Batch-level（部分已用）
```
source_mix_ratio / language_mix_ratio          # 已用（数据构造阶段）
positive_advantage_ratio / negative_advantage_ratio  # Proposed
```

---

## 7. 新增 Gate E：credit 是否能定位到真正的错误行为（扩展深读 5）

深读 5 把"该不该起 RL"拆成四道门（A 评测可见 / B RM 有方差 / C SFT 不可约 / D RL 代价非零）。本篇在其上补一道 **Gate E**，位置在 B 之后、D 之前：

```
Gate A: evaluation 看得见失败？        （深读 5 门 A，v6.5 375 步空转）
Gate B: RM 在目标轴上有 reward 方差？   （深读 5 门 B，clean 跨度 0.037）
Gate E: credit 能定位到真正错误行为？   ← 本篇新增
Gate C: 残余失败是否 SFT 不可约？        （深读 5 门 C）
Gate D: RL 代价是否可接受？             （深读 5 门 D，canary 门）
```

**Gate E 的判据**：如果 reward 能判断整条 response 坏，却**无法回答"这次 -5 到底该压哪些 token / span"**，那么 RL 会把大量正常 RP token 一起压制，代价是 clean 表达、角色 voice 或语言一致性受损。

**本项目就是活样本**：v6.10 三门（A/B/C）齐过、方向也修对了，唯独 Gate E 没过——-5 无法定位到破角 span——于是 attack 压不动、还得靠 KL 死守 clean 防止误伤。Gate E 不过时，正确的动作不是堆 RL 步数，而是先把 credit 下沉到 span（§4.5）。

---

## 8. 迁移到 Agentic RL：sequence credit → trajectory credit

本项目的 credit assignment 教训，几乎逐条能映射到 Agentic RL——只是把"一条回复"换成"一条轨迹"：

| Character RLHF | Agentic RL |
|---|---|
| response token | tool call / action |
| bad span（破角 span） | bad action segment（错误的那一步工具调用） |
| sequence reward | trajectory reward |
| roledrop / 自报身份 | prompt injection / instruction hierarchy break |
| anti-OOC detector | verifier / safety rule |
| length hacking | useless tool-call hacking（灌无用调用刷分） |
| clean / attack source | normal task / adversarial task |
| token KL | action distribution shift |
| source-wise advantage（per-source KL） | task-family-wise advantage |

**升级后的问题**：本项目里"整条回复 -5，但坏的只是一个 span"，在 Agentic RL 里变成"整条轨迹失败，但坏的只是第 7 步工具调用"。§4.5 的结论直接迁移：

```
最终任务成功/失败如何归因到哪一步工具调用、哪段思考、哪次环境反馈、哪类任务源——
这就是 trajectory-level credit assignment，是本项目 sequence→span 难题的放大版。
```

---

## 9. 面试讲述版本

```
我在这个项目里不是只跑 SFT/RM/GRPO，而是进一步分析了 RLHF 里的 credit assignment 问题，
并且是拿它来解释项目为什么会失败、又为什么那样修的。

Character RLHF 的 reward 基本都是 sequence-level 的，但 GRPO 的 actor 更新发生在 token-level logprob 上。
这个错配在我们项目里造成了几件很具体的事：
- 最贵的一次：anti_refusal 的 +3.0 是整条回复级的，把"我是 AI 但继续回答"误判成配合、给了正 advantage，
  GRPO 又把这个正 advantage 摊到包括破角 span 在内的每个 token 上，
  结果 RL 把 base 27% 的防御训坏到 55%——reward 方向甚至是对的，错的是 credit 落点。
- clean 富度 RL 提不动，是因为 RM 在 clean 上跨度只有 0.037，group 层根本没有可分配的 credit，
  所以我们把富度移交 SFT、只让 RL 做有方差的 attack 轴。
- per-source KL 其实就是我们在 source 层做的 credit 分配：clean 紧锚不许动、attack 松锚给空间。
- 但即使方向修了、source 也解耦了，v6.10 的 attack 还是压不到 30% 以下，
  因为我们始终没把那个 -5 定位到破角 span——这是 token/span 层 credit 的缺位，也是我们停线转 RM v7 的原因。

所以我把 credit assignment 拆成 token/span/sequence/group/source/batch 六层，
用 group_reward_std、reward component breakdown、分源 RM、以及提出的 bad_span_kl_share 这些指标，
去判断每一次 RL 到底是在优化正确行为，还是在优化长度、错误组件或错误样本源。
这让我把后训练从"跑算法"上升到"构造并验证一个 credit 能正确落地的可优化信号"。
```

一句话：

```
GRPO 给出更新方向；credit assignment 决定这个方向是打在破角 span 上，
还是摊到整条回复、错误 reward 组件或错误样本源上——
本项目几次成败，拆开看全是这一句话的不同层。
```

---

> 关联：reward 组件与 Goodhart 反向破角的完整代码细节 → [深读 2](02_rm_data_reward_and_attribution.md)；GSPO/DAPO/per-source KL 的算法原理 → [深读 3](03_rl_dynamics_params_and_strategy.md)；富度⇄防御为何结构性冲突 → [深读 4](04_richness_defense_tension_and_sft_ceiling.md)；四道 RL 入口门（本篇 Gate E 是其扩展）→ [深读 5](05_when_to_move_from_sft_to_rl.md)；三种协调机制 → [深读 6](06_coordinating_conflicting_objectives.md)。概念底稿 → [CREDIT_ASSIGNMENT_TAXONOMY](../docs/CREDIT_ASSIGNMENT_TAXONOMY.md) / [GRPO_LOGPROB_FLOW](../docs/GRPO_LOGPROB_FLOW.md)。
