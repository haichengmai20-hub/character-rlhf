# RM v1–v5 面试复习最终版：从 SFT 平台到 RM v5 Anchor

统计单位先固定：

```text
1 Pair =
  1条chosen
  + 1条rejected

250 Pair =
  250行pairwise数据
  = 500条展开后的response

RM Accuracy的分母是250 Pair
不是500条response
```

证据口径：

```text
[历史实跑]
仓库已有数据、配置、日志或结果

[当前设计]
统一后的Pair、Judge、Benchmark和Hard Negative方案

[面试模拟｜待真实重跑]
分子、分母和计算公式可复算
但不是历史实跑结果
```

当前设计主线（附件母版基线）：

```text
SFT-final 平台（Support 高 / Mixed 高）
→ 每个 SFT-final Group 随机选四个 turn 位置、四类错误各一
   （候选槽位上限 3,780 × 4 = 15,120）
→ chosen 来自 Teacher（同 raw 前缀；不匹配则按当前 x 重生）
→ rejected 由较弱 Gemma policy 受控生成（错误只改目标维度）
→ 最少确定性检查（同 x、非空、完整结束、可解析、无来源/标签泄漏）
→ 双 Judge 匿名 A/B（winner / difference_valid / confidence / target_type）
→ 单 Gap 门：双 Judge 平均分差 g >= 0.5（五分制）
   Medium 仅用于 Long 高置信不足时补数
→ RM-A = 10,000 Pair（wrong-role 3,000 / wrong-language 3,000 / short 2,000 / long 2,000）
→ OpenRLHF + ZeRO-2 + QLoRA；SFT-final backbone + 新 scalar head
```

注意：当前设计**不使用 4D cheap filter**（4D 属历史线机制）。本文以下 §2–§4 的 4D/Q25/A/B/C-k 流程与 §4 的四阶段演进是**历史实跑证据链**（RM v1→v5，16,826→17,712），回答「为什么这样设计」；数值引用当前设计时一律以 10,000 主账为准。

历史线主线：

```text
RL前SFT进入平台
→ 构造Preference Pair
→ A上做4D training-free校准
→ 冻结4D阈值
→ 正式训练数据走4D + Judge Pipeline
→ 四阶段RM迭代
→ B-RMBench贯穿同尺评测
→ RM v5进入RL smoke
→ 每50 step跑SFT/RL共用Benchmark
```

版本映射：

```text
阶段1 = 历史RM v1
阶段2 = 历史RM v2
阶段3 = 历史RM v3 + RM v4修复设计
阶段4 = 历史RM v5
```

## 1. 为什么不继续 SFT，为什么先做 RM

### SFT、RM、RL 的职责

```text
SFT：
  L = -log P(chosen | prompt)

  学习一个合格回复长什么样
  适合语言、格式、角色表达和基础能力
```

```text
RM：
  L = -log σ[
        RM(chosen) - RM(rejected)
      ]

  学习同一个prompt下哪个回复更好
  适合错角色、错语言和长度边界
```

```text
RL：
  在当前actor的on-policy rollout上
  提高高reward行为出现的概率

  只能放大RM已经看见的信号
  不能凭空创造RM无法区分的能力
```

因此不是：

```text
SFT loss降不动
→ 直接开始RL
```

而是：

```text
SFT统一Benchmark明显改善
→ 继续训练进入平台
→ 同一prompt下同时出现好坏候选
→ 训练RM学习相对偏好
→ RM通过A/B评测
→ RL smoke
```

### SFT/RL 共用 SR-Bench

SR-Bench 同时用于：

- 比较 RL 前 SFT checkpoint；
- 把 SFT actor 固定为 RL step0；
- RL 每训练 50 step 做一次同尺评测；
- 判断继续训练、停止或回到数据/RM。

SR-Bench 不使用 RM 的 `wrong-role` 整体角色替换。正常 SFT/RL 的主要问题是对当前 role card 贴合得不够，而不是完全变成另一个角色。

评测轴：

```text
1. Role-card Fidelity
   人物身份、性格、关系、说话方式和行为边界

2. Language Consistency
   用户语言匹配、混语和语言漂移

3. Plot Continuity & Progression
   承接历史、推动剧情、不抢用户行动权

4. Richness
   动作、对白、情绪、感官和叙事的信息增量

5. Semantic Verbosity
   是否重复、空转、用长度伪装丰富

6. Character-specific Format
   角色要求的动作、对白、独白格式是否正确使用
```

评分分两层：

```text
Rule/Rubric：
  Language Consistency
  Character-specific Format
  重复率、截断等确定性护栏
```

```text
LLM Judge：
  Role-card Fidelity
  Plot Continuity & Progression
  Richness
  Semantic Verbosity
```

格式不能替代 role fidelity：

> 动作块、对白和独白只能证明模型使用了某种角色扮演格式，不能证明人物行为符合 role card。否则模型可以机械堆格式，形成 Goodhart 代理攻击。

SR-Bench 同时保留：

```text
单轮场景：
  role、language、richness、verbosity、format

多轮场景：
  role、前文状态一致性、剧情连续和推动
```

所有 checkpoint 使用相同：

```text
prompt
conversation history
system/role card
生成参数
seed
Judge模型和版本
Judge prompt/rubric
metric代码
blind模型身份
```

本版只锁定这个框架；封闭规则指标与开放 LLM Judge 如何进一步统一校准，后续单独展开。

口径注：本节评测轴与 SR 数字属旧协议口径（N=4、Semantic Verbosity）。当前统一协议为 200 Scene × 3 组解码配置 × N=2，Verbosity 升级为 Density/Brevity 并把重复下沉为 Guard（见总览 §4.6）。旧 N=4 数字（Support 88.5% / Mixed 32.5% / All-Fail 11.5% 等）作为历史发现证据保留，不与新 N=2 面板拼成同一条趋势。

### RL 前 SFT 平台证据

只比较 RL 前模型：

```text
SFT1：
  13,128条第一波SFT产物

SFT2-e1：
  19,508条SFT2数据
  第一轮checkpoint

SFT2-final：
  同一SFT2数据继续训练后的checkpoint
```

模拟 SR-Bench：

```text
[面试模拟｜待真实重跑]

                        SFT1      SFT2-e1   SFT2-final

Language Pass          164/200    176/200    175/200
                         82.0%      88.0%      87.5%

Format Pass            152/200    166/200    165/200
                         76.0%      83.0%      82.5%

Role Fidelity           3.10       3.65       3.68
Plot Progression        2.75       3.25       3.24
Richness                2.60       3.15       3.10
Verbosity Quality       3.00       3.45       3.40
```

开放质量盲评：

```text
SFT2-e1 vs SFT1：

e1胜       122
SFT1胜      70
Tie          8

tie-adjusted win =
(122 + 0.5×8) / 200
= 63.0%
```

```text
SFT2-final vs SFT2-e1：

final胜     98
e1胜        98
Tie          4

tie-adjusted win =
(98 + 0.5×4) / 200
= 50.0%
```

结论：

- SFT1→SFT2-e1 明显改善；
- SFT2-e1→final 基本持平；
- language、format、plot 和 richness 出现轻微回弹；
- 继续 CE 已进入当前数据与训练方案的平台。

### Rollout 如何发现 SFT 失败模式

对固定 200 个 SR-Bench 场景，每个场景生成 4 个候选：

```text
四个候选都通过基础质量门：
112/200 = 56.0%

同一prompt同时出现好/坏候选：
65/200 = 32.5%

四个候选都低于综合质量门：
23/200 = 11.5%

至少存在1个合格候选：
177/200 = 88.5%
```

“低于综合质量门”不等于错语言、错角色等 hard-fail，因此不与后续 RM 的 `Top-1 Hard-Fail` 冲突。

65 个 mixed-quality 场景：

```text
role fidelity漂移      23
language drift         10
剧情停滞/断裂          14
richness/verbosity问题 18
总计                   65
```

识别方法：

- role fidelity：Judge 读取完整 role card 和历史，判断行为、关系和语气；
- language：detector 加混语复核；
- plot：Judge 检查是否承接前文、是否真正推进；
- richness：Judge 检查动作、情绪和感官是否带来信息增量；
- verbosity：重复规则加 Judge 语义判断；
- format：rule 检查角色要求的结构是否出现且正确。

这说明：

```text
88.5%的场景至少能生成一个合格候选
但32.5%的场景同时生成好坏回复
```

模型不是完全不会，而是不能稳定选择。因此首先支持：

```text
SFT → RM
```

尚不能直接支持：

```text
SFT → RL
```

## 2. Pair 构造、4D Heuristic 与 Judge Pipeline

### Pair 构造

`chosen` 来自 clean SFT output。

`rejected` 只使用同 prompt 的 model-generated 回复：

```text
wrong_role
wrong_language
length_short
length_long
```

不使用：

```text
zero-gen
cross-character
cross-lingual
```

因为跨记录回复不能保证回答的是同一 user prompt，容易把“无关回复”污染成“错角色或错语言”。

当前训练配比目标：

```text
wrong_role      30%
wrong_language  30%
length_short    20%
length_long     20%
```

历史 train 总量 16,826 时，对应理论采样目标约为：

```text
wrong_role       5,048
wrong_language   5,048
length_short     3,365
length_long      3,365
```

这只是当前采样目标，不冒充历史落盘精确分布。

### Wrong-role 的 Role Card 替换

错误 role card 保留：

```text
chosen角色名字
全局语言规则
全局格式规则
产品通用约束
```

替换为其他角色的：

```text
personality
background
relationship
scenario
speaking style
habit
behavior boundary
```

因此 rejected 仍使用 chosen 的角色名字，但行为和人物逻辑来自另一个 role card。

它避免 RM 只靠角色名字判断，并形成泛化难度：

```text
role card差异大：
  easy wrong-role

role card部分相似：
  borderline wrong-role

role card高度相似：
  hard wrong-role
```

B-RMBench 中 75 对 wrong-role 按相似度和 Judge 难度固定分层：

```text
easy        45
borderline  20
hard        10
总计        75
```

### `target_error_confirmed`

定义：

> rejected 预设要制造的目标错误确实发生了，而且该错误是 chosen 优于 rejected 的主要原因。

按类型：

```text
wrong-role：
  rejected行为符合注入的其他role card，
  并与正确role card冲突；
  不能只是风格略有差异。

wrong-language：
  rejected确实错语、严重混语
  或违反system语言要求。

length-short：
  rejected缺少必要信息、情绪承接或剧情回应；
  不能只是更简洁。

length-long：
  新增长度主要来自重复、空转或无效展开；
  不能只是更丰富。
```

如果 `target_error_confirmed=false`，该 Pair 不进入训练。

### 4D Heuristic

四维均为 `[0,1]` 的确定性分数，不是 LLM Judge。

```text
h_format =
    0.30×动作结构
  + 0.25×对白
  + 0.20×心理标记
  + 0.15×特殊结构
  + 0.10×动作/对白交替
```

```text
h_role =
    0.75×角色扮演形态与反泄漏
  + 0.25×角色名/身份信号
```

```text
h_language =
    0.50×instruction-output语言一致
  + 0.30×source language一致
  + 0.20×system语言规则遵守
```

```text
h_length =
    0.80×相对长度适配
  + 0.20×反重复
```

Chosen 全局门：

```text
H_default(chosen) =
    0.30×h_format
  + 0.30×h_role
  + 0.25×h_language
  + 0.15×h_length

H_default(chosen) >= 0.62
```

目标维度门：

```text
wrong_role：
  heuristic_role_gap >= 0.12
  + 身份/角色检查

wrong_language：
  heuristic_language_gap >= 0.30
  + language detector

length：
  heuristic_length_gap >= 0.20
  + 长度比例/重复检查

short：
  rejected_len < max(40, 0.7×chosen_len)

long：
  rejected_len > 1.2×chosen_len
```

当前压缩方案不再使用 type-specific total，也不再给 rejected 加 fixed target penalty。

### Gap 定义

```text
heuristic_target_gap：
  4D规则产生
  用于Judge前筛选
```

```text
primary_judge_gap：
  J_primary(chosen) - J_primary(rejected)
```

```text
dual_judge_gap：
  两个Judge归一化gap的平均
```

```text
rm_gap：
  RM(chosen) - RM(rejected)
```

“训练数据的 reward gap”统一指 `primary_judge_gap`，不能与 `rm_gap` 混写。

### Primary Q25 阈值

Primary 对每个 type 分别计算：

```text
τ_type =
Q25(
  normalized_primary_gap
  | chosen_better
  | target_error_confirmed=true
  | rejected_type=type
)
```

不先按 confidence 过滤，避免用 high-confidence 数据计算阈值后再筛 medium/low 的循环偏差。

统计目的：

- 每类 gap 较高的约 75% 数据可低成本直接保留；
- gap 最低约 25% 集中送 Secondary；
- 不是说最低 25% 一定错误，而是它们最接近边界；
- 每类单独计算，避免 length-long 的天然低 gap 污染其他类型；
- 高 gap 区间仍随机抽审 5%～10%，估计漏错率。

### Judge 分流

直接保留：

```text
chosen_better
target_error_confirmed=true
confidence=high
primary_judge_gap >= τ_type
```

直接丢弃：

```text
rejected_better
或
target_error_confirmed=false
```

进入 Secondary：

```text
wrong-role全部
Primary tie
confidence=medium/low
primary_judge_gap < τ_type
各类型随机抽审5%～10%
全部A-eval
全部B-RMBench
```

Primary 和 Secondary 都使用相同 0～5 rubric，并分别归一到 `[0,1]`：

```text
dual_mean_gap =
(
  normalized_primary_gap
  + normalized_secondary_gap
) / 2
```

两次 Judge 后保留：

```text
两个Judge都chosen_better
target_error_confirmed=true
dual_mean_gap >= τ_type
```

否则隔离或丢弃，不自动 swap。

RM v1 的 length-long 例外：

```text
aggregate confidence=medium
但两个Judge方向一致
target_error_confirmed=true
dual_mean_gap >= τ_length_long
  → 可以进入RM v1
```

后续阶段再验证删除这些 medium long 是否破坏边界覆盖。

## 3. A-eval、B-RMBench 与 C-k

### A-eval：4D 校准与训练 Eval

A 使用当前 SFT merged 构造。

先生成大于 250 对的候选，经过 Dual Judge 后，冻结 250 个有效 Pair：

```text
wrong_role      75
wrong_language  75
length_short    50
length_long     50
总计           250 Pair
```

展开后为 500 条 response，但 A-eval 的统计分母仍是 250 Pair。

覆盖要求：

```text
每种语言不少于5对、不超过20对
每个角色不少于5对、不超过20对
```

数据流：

```text
当前SFT merged生成Pair候选
→ Dual Judge确权并得到250个有效Pair
→ 在250对上扫描4D阈值
→ 冻结4D阈值
→ 原250对直接成为A-eval/dev
→ 正式训练候选才开始走冻结后的4D + Judge Pipeline
```

A 不从 train 中拆，也不进入训练和 Hard Mining。

A 用于两个不同指标。

#### A-4D Retention

```text
通过最终4D全部门槛的Pair数
/ 250
```

#### A-4D Accuracy

```text
在4D保留的Pair中，
4D目标方向与Dual Judge方向一致的Pair数
/ 4D保留Pair数
```

模拟阈值实验：

```text
[面试模拟｜待真实重跑]

宽松阈值：
  Retention = 230/250 = 92.0%
  Accuracy  = 198/230 = 86.1%

中心阈值：
  Retention = 211/250 = 84.4%
  Accuracy  = 193/211 = 91.5%

严格阈值：
  Retention = 172/250 = 68.8%
  Accuracy  = 164/172 = 95.3%
```

中心点在 Accuracy 和 Coverage 之间更平衡，并满足语言/角色 5～20 对覆盖，因此冻结：

```text
chosen     0.62
role gap   0.12
lang gap   0.30
length gap 0.20
```

A 同时用于 RM 训练过程中的 dev 评测。因为 A 已参与阈值选择和 checkpoint 判断，它是开发评测，不是独立最终测试。

模拟 A-RM Accuracy：

```text
Stage1   223/250 = 89.2%
Stage2   215/250 = 86.0%
Stage3   221/250 = 88.4%
Stage4   240/250 = 96.0%
```

### B-RMBench：固定强 Pair Benchmark

B 使用 Gemini Flash 生成 chosen/rejected，其他构造逻辑与正式 Pair 一致。

目的：

- chosen/rejected 质量更稳定；
- 减少弱生成器造成的伪差异；
- 生成更多高质量 hard/borderline Pair；
- 更准确地测试 RM 在语义边界上的表现。

Gemini Flash 只负责生成，不作为同源唯一 Judge。Primary、Secondary 必须隐藏 generator/source，且至少一个 Judge 与 Gemini Flash 异质。

B 经过完整 Dual Judge Pipeline 后冻结 250 对：

```text
wrong_role      75
wrong_language  75
length_short    50
length_long     50
总计           250 Pair
```

A、B、C 在 prompt、conversation、role-card 组合和近重复语义上完全隔离。

B 不进入训练；B 暴露的短板只能改变 C 的采样预算，不能复制具体 Pair。

### B 的难度定义

Benchmark 难度不能由待测 RM 的 `rm_gap` 决定，否则不同 RM 面对的难度定义不同。

先定义：

```text
Borderline：
  两个Judge都chosen_better
  target error成立
  τ_type <= dual_mean_gap <= Q50_type
```

剩余样本中：

```text
Hard：
  构造上具有高业务混淆
  但Dual Judge标签清楚，
  dual_mean_gap > Q50_type
```

```text
Easy：
  目标错误明显
  表面线索清楚
  dual_mean_gap > Q50_type
```

判定优先级：

```text
先判Borderline
再在剩余样本中区分Hard/Easy
```

B 固定难度配比：

```text
easy        175 = 70%
borderline   50 = 20%
hard         25 = 10%
总计        250
```

类型 × 难度配额：

```text
                  easy   borderline   hard   total

wrong_role         45        20        10      75
wrong_language     60        10         5      75
length_short       40         7         3      50
length_long        30        13         7      50

总计              175        50        25     250
```

典型 Hard：

- 高相似 role-card wrong-role；
- 轻度混语或相邻语言；
- 看似简洁但缺少关键承接；
- 看似丰富但主要内容重复。

### B 指标

总准确率：

```text
B-Accuracy =
  count(rm_gap > 0) / 250
```

因为 B 本身就是 `75/75/50/50`，等价于 `30/30/20/20` 加权准确率。

```text
Misrank Rate =
  count(rm_gap <= 0) / 250
```

跨 RM 版本的 low-margin 不能使用 raw score。

对当前 RM 在固定 B response bank 上做标准化：

```text
z_v(response) =
[
  RM_v(response) - μ_v(B)
] / σ_v(B)

Δz_v =
  z_v(chosen) - z_v(rejected)
```

冻结 `τ_z` 后：

```text
Low-Margin Rate =
  count(0 < Δz_v < τ_z) / 250
```

定义：

```text
Hard Accuracy =
  RM在Hard Slice中方向正确数 / 25

Borderline Accuracy =
  RM在Borderline Slice中方向正确数 / 50
```

B-hard 与 C-Hard Negative 不同：

```text
B-hard：
  预先冻结的Benchmark难度
  由构造和Dual Judge定义
  永不训练
```

```text
C-Hard Negative：
  标签已被Dual Judge确权
  但当前RM misrank或low margin
  可以进入下一阶段训练
```

### C-k：动态 Hard-Mining Pool

C 每轮根据 B 的类型 breakdown 调整预算，但与 A/B 完全隔离。

```text
独立离线prompt
→ 当前actor rollout
→ Dual Judge确权
→ 当前RM打分
→ misrank或low margin
→ 进入C-k
```

C 重点覆盖：

```text
高质量wrong-role
语言边界
concise vs insufficient
rich vs verbose
```

B 只影响 C 的类型预算，不能降低 Judge 标签门槛。

本版删除 D，不再维护 benchmark extension。B 一经冻结不再变化。

## 4. 四阶段 RM 演进

以下历史结果与模拟 A/B 指标必须分开理解：

```text
历史结果：
  来自仓库旧评测快照

A/B模拟：
  来自当前统一250-pair设计
  待真实重跑
```

### 阶段1：让 RM 学会四类显式错误（历史 RM v1）

历史数据：

```text
raw pair      12,973
train         16,826
eval pool        885
```

历史结果：

```text
accuracy      94.5%
gap           1.601
chosen mean   +0.948
rejected mean -0.653
```

历史难点：

- 高质量 wrong-role；
- length-long；
- 少量 length-short。

RM v1 允许 Dual Judge 确权、gap 达标的 medium-confidence length-long 进入，避免边界数据过少。

统一模拟：

```text
A-Accuracy       223/250 = 89.2%
B-Accuracy       218/250 = 87.2%
Worst-Type       39/50   = 78.0%（length-long）
Hard Accuracy    14/25   = 56.0%
Borderline Acc   39/50   = 78.0%
Misrank Rate     32/250  = 12.8%
Low-Margin Rate  49/250  = 19.6%
```

B 显示：

- wrong-language 和 short 较容易；
- length-long 最弱；
- 高相似 wrong-role 仍容易 misrank。

阶段2验证：

> 删除低 gap 和 medium-confidence Pair，是否会让标签更干净、RM 更稳定。

### 阶段2：删低 Gap/中置信后反而退化（历史 RM v2）

过滤：

```text
primary_judge_gap < τ_type
或
aggregate confidence=medium
```

主要集中在 length-long。

历史映射：

```text
16,826 → 14,826
删除约2,000条
```

历史结果：

```text
accuracy      86.4%
gap           1.122
修复4个旧错
新增13个错
```

统一模拟：

```text
A-Accuracy       215/250 = 86.0%
B-Accuracy       210/250 = 84.0%
Worst-Type       33/50   = 66.0%（length-long）
Hard Accuracy    12/25   = 48.0%
Borderline Acc   36/50   = 72.0%
Misrank Rate     40/250  = 16.0%
Low-Margin Rate  65/250  = 26.0%
```

数据归因：

```text
标签更干净
但rich vs verbose、
concise vs insufficient附近的边界覆盖被删弱
```

工程归因：

- GPU、batch、steps、scheduler 同时变化；
- v1 batch 台账存在冲突；
- v1 大型 DeepSpeed checkpoint 写盘失败；
- v2 score head 在训练内有效，但未随 PEFT adapter 保存。

因此：

```text
训练内accuracy退化
≠
adapter重载后score head随机
```

两者是独立事故。

结论：

> 只追求标签纯度不够，需要恢复标签可靠、但接近真实决策边界的数据。

### 阶段3：B 定位短板，C 补真实难例（历史 RM v3；v4 为修复设计）

阶段3加入两类数据。

第一类，C-k Hard Negative：

```text
Dual Judge：
  chosen_better
  target error成立
  方向一致
  dual gap达标

当前RM：
  misrank
  或 low margin
```

第二类，选择性恢复 medium length-long：

```text
confidence=medium
两个Judge方向一致
target error成立
dual_mean_gap >= τ_length_long
```

历史 v3 数据：

```text
base                         13,459
strong length-long            2,734
low-gap length-long           2,000
long-soft                       600
global hard                     162
总计                         18,955
```

历史真正的 global hard 只有 162 条，不能概括为“Hard Negative 太多”。主要增加的是：

```text
2,000 low-long
+ 600 long-soft
```

历史训练：

```text
global batch   128
optimizer step 148
cosine schedule
```

历史结果：

```text
accuracy      86.0%
gap           1.256
chosen mean   +0.958
rejected mean -0.298
```

统一模拟：

```text
A-Accuracy       221/250 = 88.4%
B-Accuracy       217/250 = 86.8%
Worst-Type       35/50   = 70.0%（length-long）
Hard Accuracy    14/25   = 56.0%
Borderline Acc   39/50   = 78.0%
Misrank Rate     33/250  = 13.2%
Low-Margin Rate  38/250  = 15.2%
```

说明 C 的 rollout-oriented 数据改善了困难分布，但完整基础边界仍未稳定。

失败原因：

- weak/soft long 回灌过重；
- hard、borderline 没有分层；
- batch 变大后只有 148 steps；
- cosine 更早衰减；
- 数据与工程变量同时变化。

RM v4 只作为内部修复设计：

- 限制 weak/soft long；
- 固定 batch、steps、scheduler；
- 恢复完整 base；
- 不列正式新版本。

结论：

> Hard Negative 能修局部边界，但不能替代完整基础分布。

### 阶段4：完整 Base + 少量 Hard，形成 Anchor RM（历史 RM v5）

历史数据：

```text
完整v1 base  16,826
+ 约5% bad pool
= 17,712
```

必须区分：

```text
历史RM v5：
  从离线bad pool有放回抽样约5%

当前设计：
  C-k中Dual Judge确权、
  按类型限额的Hard Negative
```

历史配置：

```text
OpenRLHF
DeepSpeed ZeRO-2
4-bit QLoRA
LoRA r64 / alpha128
q/k/v/o
global batch 64
276 optimizer steps
LR 9e-6
cosine
warmup 0.03
max_len 3072
```

历史结果：

```text
accuracy      95.5%
gap           3.497
chosen mean   +2.27
rejected mean -1.23
```

统一模拟：

```text
A-Accuracy       240/250 = 96.0%
B-Accuracy       237/250 = 94.8%
Worst-Type       45/50   = 90.0%（length-long）
Hard Accuracy    20/25   = 80.0%
Borderline Acc   45/50   = 90.0%
Misrank Rate     13/250  = 5.2%
Low-Margin Rate  12/250  = 4.8%
```

分类型：

```text
wrong-role      71/75 = 94.7%
wrong-language  73/75 = 97.3%
length-short    48/50 = 96.0%
length-long     45/50 = 90.0%
```

Stage4 同时恢复基础类型、Hard 和 Borderline，成为后续 RL 的 Anchor RM。

核心原则：

> 完整 Base 负责全局业务分布，约 5% Hard Pool 只修局部边界。

RM v5 仍主要解决显式 wrong-role、wrong-language 和长度异常，没有充分覆盖：

- hard-pass 回复之间的 richness；
- 剧情连续与推进；
- attack 下的 roledrop 防御。

## 5. 训练工程与产物可靠性

RM 始终只有一个 scalar score head；4D 和 Judge 只负责标签准入，不是多 reward head。

受控复现实验固定：

```text
同一Qwen SFT2 base
同一GPU/NUMA拓扑
OpenRLHF
DeepSpeed ZeRO-2
4-bit QLoRA
r64 / alpha128
q/k/v/o
global batch64
max_len3072
固定276 optimizer steps
LR 9e-6
cosine
warmup0.03
固定seed和token exposure
```

该台账用于当前受控复现实验设计，不改写历史 v1/v2/v3 的真实配置。

初始化对齐：当前设计的 RM-A 从 **SFT-final backbone（Gemma 线）+ 新 scalar head** 初始化；本节「同一 Qwen SFT2 base」是历史受控复现口径。历史 Qwen RM v5（acc 95.5% / gap 3.497）与 Gemma4 RM v5 复现（acc 97.5% / gap 6.462，score saturation 高、需 normalize/clamp）是两个不同模型，不能混成一条增长线。

工程取舍：

- 当前机器无 NVLink/P2P、双 NUMA；
- ZeRO-3 需要频繁 all-gather，早期出现 step0 卡死/watchdog；
- 4-bit QLoRA + ZeRO-2 + adam offload + gradient checkpointing 更适合当时硬件；
- 这不是“ZeRO-2 普遍优于 ZeRO-3”的通用结论。

产物验收：

```text
score head在PEFT前初始化
→ modules_to_save包含score head
→ 保存HF adapter
→ 关闭大型DS checkpoint
→ 保存后重新加载
→ A/B固定Pair方向一致
→ OpenRLHF与vLLM的EOS/pooling一致
```

必须分开：

```text
训练内指标有效
checkpoint可加载
online scorer与离线方向一致
```

## 6. RM v5 进入 RL 后的第一轮观察

RM 通过条件不止总体 Pair Accuracy：还看分类型正确率、近似同质候选上的区分能力、长度捷径、离群分数、不同语言表现与同候选 raw/summary 对照。**高准确率不自动证明能指导 RL**——若考试主要是明显错语言/错角色，在线候选却大多形式正确，必须补测「都还不错的候选如何排序」。评分差小也不能直接推出无梯度：结合组内标准化、epsilon、数值精度与排序正确性判断（见 RM-B 正文 `group_reward_std` 一节）。

RM v5 通过 A/B 后，进入 RL smoke。

```text
step0：
  冻结SFT actor基线

step50：
  运行完整SR-Bench

有超过噪声带的增益
且关键轴不退
  → 继续到下一50 step

无有效增益或关键轴退化
  → 停止并回查RM、reward和数据
```

继续规则：

```text
Language不退
Format不退
Role-card Fidelity不退
Semantic Verbosity不恶化

并且：

Plot Progression
或 Richness
至少一个有有效提升
```

SR-Bench 反复参与早停和 checkpoint 选择，因此属于 RL-dev benchmark，不单独证明最终线上泛化，也不能把其 prompt 回灌训练。

第一次使用 RM v5 做 RL 的阶段观察：

```text
Role-card Fidelity：
  相对稳定

Language：
  相对稳定

Richness：
  只上涨很少，接近噪声带

正式上线后：
  仍出现attack roledrop
```

这说明：

- RM v5 已能学习基础角色、语言和长度错误；
- 对 hard-pass 回复之间的 richness、剧情推进信号仍弱；
- 对 attack roledrop 没有充分覆盖。

本版只记录这个业务发现：

```text
RM v5 + 第一轮RL
→ richness增益小
→ 上线出现attack roledrop
→ 后续需要升级RM和attack数据
```

不展开 RM v6.x。

## 7. 90 秒面试回答

> 我们先用 SFT/RL 共用的 SR-Bench 比较 RL 前的 SFT checkpoint。Benchmark 不用 wrong-role 整体换角，而是评 role-card fidelity、语言、剧情连续与推动、richness、verbosity 和角色特定格式。语言、格式用规则，role、剧情和 richness 用 LLM Judge，而且格式只做护栏，不作为 role fidelity 的代理。SFT1 到 SFT2 明显改善，但继续训练后开放质量盲评接近50%，同时88.5%的场景至少能生成一个合格候选，却有32.5%的场景同时出现好坏回复，所以问题从能力覆盖转成相对选择，先进入 RM。
>
> RM 数据只使用同 prompt 的 model-generated wrong-role、wrong-language、length-short 和 length-long，比例30/30/20/20。4D heuristic 用格式、角色、语言和长度规则做低成本筛选，Primary Judge 全量复核；每个 type 用 Primary gap 的 Q25 作为二审阈值，把最不确定的约25%集中送 Secondary。Wrong-role、tie、中低置信、抽审和全部 A/B 都双 Judge；只有两个 Judge 都判 chosen 更好、target error 成立、归一化平均 gap 达标才保留。
>
> A-eval 是当前 SFT merged 生成、Dual Judge 后冻结的250对，用于4D training-free和训练eval；B-RMBench 是 Gemini Flash 生成、Dual Judge 后冻结的另一套250对，按75/75/50/50覆盖四类，再按70% easy、20% borderline、10% hard测试所有 RM。四阶段分别是建立基础错误判断、过滤低 gap 后退化、恢复可信边界和 Hard Negative、最后完整 Base 加约5%难例。统一模拟中 B-Accuracy 从87.2%降到84.0%、恢复到86.8%，最终到94.8%，Hard Accuracy达到80%，Misrank降到5.2%，因此 RM v5 可以进入 RL smoke。
>
> RL 每50 step跑同一个 SR-Bench。第一轮里 role 和 language 相对稳定，但 richness 增益很小，正式上线后仍出现 attack roledrop。这暴露了 RM v5 的信号边界，也成为后续升级 RM 和 attack 数据的入口。

本轮已按全部确认口径生成完整版本，没有修改项目事实台账。所有新增精确模拟数字必须保留：

```text
[面试模拟｜协议可复现｜待真实重跑]
```
