# 技术合同参考（从原 AGENTS.md 提取）

来源：2026-09-06 修订前的本目录 AGENTS.md。以下技术章节原文保留，用于查阅既有设计、统计单位和工程边界，不作为自动加载的通用行为指令。

仅在任务涉及相应技术合同时读取相关章节。数字与方案是该快照的既有口径，须与当前 Canonical、实际实现和最新用户授权核对；不能单凭本文认定已实跑或当前仍最优。文本中的“必须/不得”限定于描述既有方案，不禁止提出替代设计。文件权限、事实判断、批准与证据冲突处理统一由当前 AGENTS.md 规定。

## 7. 稳定统计单位与配比

```text
1 Pair = 1条chosen + 1条rejected
250 Pair = 250行pairwise数据
         = 500条展开response
```
RM Accuracy 的分母是 Pair 数，不是展开后的 response 数。

当前 A/B 类型配比：

```text
wrong-role      75
wrong-language  75
length-short    50
length-long     50
总计           250 Pair
```
对应训练目标权重：

```text
30% / 30% / 20% / 20%
```
B-RMBench 难度配比：

```text
easy        175 = 70%
borderline   50 = 20%
hard         25 = 10%
```
这些是当前稳定契约。每阶段完整历史/模拟结果表只维护在主文档中，不在本规则中复制。

## 8. SR-Bench：SFT/RL 共用评测

SR-Bench 只用于比较 SFT actor、RL step0 和每 50 step 的 RL checkpoint，不用于 RM 的 wrong-role Pair Accuracy。

评测轴：

```text
Role-card Fidelity
Language Consistency
Plot Continuity & Progression
Richness
Semantic Verbosity
Character-specific Format
```
确定性层：

```text
Language
Format
重复、截断等规则护栏
```
开放 Judge 层：

```text
Role-card Fidelity
Plot Continuity & Progression
Richness
Semantic Verbosity
```
格式不能作为 role fidelity 的代理。动作块、对白、独白数量只能证明格式使用，不能证明人物行为符合 role card。

SR-Bench 不包含整体 wrong-role 替换；角色错误边界由 A/B 的 wrong-role Pair 负责。

RL 每 50 step 使用同一 prompt、history、生成参数、seed、Judge 版本和 metric。继续训练必须满足：

```text
Language不退
Format不退
Role Fidelity不退
Verbosity不恶化
且Progression或Richness至少一个有超过噪声带的提升
```
SR-Bench 会参与早停和 checkpoint 选择，因此属于 RL-dev，不得声称它独立证明最终线上泛化，也不得把其 prompt 回灌训练。

## 9. RM Pair 构造

chosen 来自 clean SFT output。rejected 只使用同 prompt 的 model-generated：

```text
wrong-role
wrong-language
length-short
length-long
```
禁止使用：

```text
zero-gen
cross-character
cross-lingual
```
Wrong-role 构造保留 chosen 角色名字、通用语言/格式/产品规则，替换另一角色的：

```text
personality
background
relationship
scenario
speaking style
habit
behavior boundary
```
目的是让 RM 判断行为是否贴合正确 role card，而不是靠角色名走捷径。

## 10. 4D、Judge 与 Gap 边界

RM v1–v5 的 4D 是 Pair 构造/筛选 heuristic，不是四个 RM head：

```text
format / role / language / length
```
当前冻结门：

```text
H_default(chosen) >= 0.62
role gap          >= 0.12
language gap      >= 0.30
length gap        >= 0.20
short < max(40, 0.7×chosen_len)
long  > 1.2×chosen_len
```
必须区分：

```text
heuristic_target_gap
primary_judge_gap
dual_mean_gap
rm_gap
```
`target_error_confirmed=true` 表示 rejected 预设的目标错误确实发生，而且是 chosen 优于 rejected 的主要原因；不是简单重复“chosen 更好”。

Primary Q25 阈值按 rejected type 分别计算：

```text
τ_type = Q25(
  normalized_primary_gap
  | chosen_better
  | target_error_confirmed=true
  | rejected_type=type
)
```
不先按 confidence 过滤。底部约 25% 只是更不确定，不代表一定错误；它们集中送 Secondary。高 gap 区间仍随机抽审 5%～10%。

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
或target_error_confirmed=false
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
两个 Judge 的 0～5 rubric 分别归一到 `[0,1]`，再计算：

```text
dual_mean_gap =
(normalized_primary_gap + normalized_secondary_gap) / 2
```
只有两个 Judge 都判 chosen_better、target error 成立且 mean gap 达标才保留。不得自动 swap。

Primary tie 只是边界候选，不是 Hard Negative。

## 11. A-eval、B-RMBench 与 C-k

### A-eval

```text
来源：当前SFT merged
规模：Dual Judge后固定250 Pair
用途：4D training-free + RM训练dev
```
A 不从 train 中拆，不进入训练和 Hard Mining。先用 A 冻结 4D 阈值，再让正式训练候选经过冻结后的 4D + Judge Pipeline。

记录两个指标：

```text
A-4D Retention = 通过最终4D门槛的Pair数 / 250
A-4D Accuracy =
  4D保留Pair中与Dual Judge方向一致数
  / 4D保留Pair数
```
A 已参与阈值和 checkpoint 选择，因此是 calibration/dev，不是独立测试。

### B-RMBench

```text
来源：Gemini Flash生成chosen/rejected
规模：异质Dual Judge后固定250 Pair
用途：所有RM阶段的固定推理主指标
```
Gemini Flash 只负责生成，Judge 必须隐藏 generator/source，且至少一个 Judge 与生成器异质。

B 不进入训练；B 暴露的短板只能调整 C 的类型预算，不能复制具体 Pair。

B 的 Easy/Borderline/Hard 由冻结构造规则和 Dual Judge 定义，不由待测 RM 的 gap 定义。

主要指标：

```text
B-Accuracy
Worst-Type Accuracy
Hard Accuracy
Borderline Accuracy
Misrank Rate
Low-Margin Rate
Artifact Reload Parity
```
不同 RM 的 raw score 尺度不直接可比。跨版本 low-margin 必须先对当前 RM 在固定 B 上做 checkpoint-level 标准化，再比较 normalized gap。

### C-k

```text
来源：与A/B完全隔离的离线rollout-dev
用途：每轮动态Hard-Mining，只训练不评测
```
C-Hard Negative 必须满足：

```text
Dual Judge方向一致
target_error_confirmed=true
标签gap达标
且当前RM misrank或low margin
```
B-hard 是固定 Benchmark 难度；C-Hard Negative 是当前模型不会的训练样本。两者不得混写。

本目录当前不使用 D；不得重新引入 `D-sealed`、`Online Shadow` 或旧 A/B 500-Pair 口径，除非用户明确批准。

## 12. 四阶段 RM 主线

固定叙事：

```text
阶段1：让RM学会四类显式错误（历史RM v1）
阶段2：删低Gap/中置信后反而退化（历史RM v2）
阶段3：B定位短板，C补真实难例
       （历史RM v3；RM v4仅修复设计）
阶段4：完整Base + 少量Hard，形成Anchor RM
       （历史RM v5）
```
每个阶段必须写全：

```text
业务问题
→ 数据动作
→ 训练/工程变化
→ 历史结果
→ 当前A/B模拟
→ 数据归因与工程归因
→ 下一阶段决策
```
不要把 RM v4 写成正式训练结果。不要把 v3 简化成“Hard Negative 太多”；历史真正的 global hard 只有 162，主要问题是 weak/soft long 回灌和训练节奏混杂。

历史 RM v5 的约 5% bad pool 来自离线池有放回抽样；当前 C-k rollout-dev 是升级设计，不能冒充历史实现。

RM v5 结束边界：进入第一轮 RL 后 role/language 相对稳定、richness 增益小，正式上线仍有 attack roledrop；这里只引出后续 RM/attack 数据升级，不擅自展开 RM v6.x。

## 13. 训练工程边界

- RM 始终是单 scalar score head；
- 4D/Judge 不是多 reward head；
- KL 不属于 scalar reward；
- 历史 batch 台账冲突不得伪造成确定事实；
- ZeRO-2 是当前无 NVLink/P2P、双 NUMA、4-bit QLoRA 环境下的工程取舍，不是通用优越结论；
- v1 大型 DeepSpeed checkpoint 写盘失败与 v2 score head 未随 PEFT 保存是两个独立事故；
- score head 在 PEFT 前初始化，并通过 `modules_to_save` 保存；
- OpenRLHF 与 vLLM 的 EOS/final-token pooling 契约必须一致。

必须分别验收：

```text
训练内指标有效
checkpoint可重新加载
offline与online scorer方向一致
```

## 14. 版本与指标禁止混用

始终区分：

```text
RM版本
SFT版本
RL训练版本
产品交付版本
```
禁止：

- 将历史不同评测切片画成同尺实跑曲线；
- 将模拟 A/B 面板写成历史结果；
- 直接跨 score head 比 raw gap；
- 将 SR-Bench、A、B 拼成一个总指标；
- 用版本号暗示效果单调提升；
- 将相关性或多变量共变写成单一因果；
- 将 RM v5 离线强解释为已经解决 richness 或 attack roledrop。
