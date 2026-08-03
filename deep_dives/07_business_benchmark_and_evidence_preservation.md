# 深读 7 — 业务 Benchmark：把产品决策、同尺评测和证据留存串成一套可回归系统

> 阅读路线：先看 [深读 0 — 面试综合路线](00_interview_synthesis.md)，再按本篇专题深挖。

> **版本命名**：本篇保留原始实验号叙述；哪些是真改进、哪些只是 run/restart/失败实验，以 [VERSION_MAP](../docs/VERSION_MAP.md) 和 [project_chatjoy_overview](../docs/project_chatjoy_overview.md) 的阶段映射为准。

> 本篇展示的能力：**把角色模型的评测从"看几条回答好不好"升级为一套可复现的业务 benchmark**。它不是替代 RMTBENCH 这类学术基准，而是把真实语言分布、攻击分布、部署温度、去误报检测器、人工抽读和 scorecard 组织成一个能回答"这个 checkpoint 能不能上线"的决策系统。同时，本篇也说明哪些工作区证据必须以全文、样本或索引形式进仓库，避免项目日后只能靠本机路径回忆。

关联文档：[业务 benchmark 构造指导](../benchmarks/BUSINESS_BENCHMARK_GUIDE.md) · [评测范式](../docs/design/chatjoy_eval_paradigm.md) · RMTBENCH · [深读 4](04_richness_defense_tension_and_sft_ceiling.md) · [深读 5](05_when_to_move_from_sft_to_rl.md) · [深读 6](06_coordinating_conflicting_objectives.md)。

> ## Attribution Lens
> 本篇把观察到的行为主要归因到 **evaluation protocol**（同尺协议、去误报检测器、证据留存）：当 benchmark 结论与人工抽读冲突时，先怀疑尺子而非模型。这也是 [深读 8](08_credit_assignment_in_character_rlhf.md) 中 batch/source 层 credit 的前置条件——评测覆盖为零的失效，训练里也拿不到 credit。

---

## 0. 先给结论：benchmark 在本项目里不是"排行榜"，而是"决策仪表盘"

ChatJoy 的 benchmark 要回答的不是"模型在通用 roleplay 榜单排第几"，而是：

> 在真实产品口径下，这个 checkpoint 是否比上一个可交付版本更值得上线？它的收益在哪个业务维度，代价又落在哪个红线维度？

这决定了本项目需要两类 benchmark：

| 类型 | 代表 | 作用 | 边界 |
|---|---|---|---|
| 外部学术锚 | `benchmarks/RMTBENCH/` | 给多轮 roleplay 能力提供外部可比结构，避免只在自家样本上自嗨 | 不含产品角色回复、不覆盖真实 attack 分布，不直接给上线结论 |
| 业务 benchmark | `BUSINESS_BENCHMARK_GUIDE.md` 规划的 `benchmarks/business/` | 回答上线、回退、继续 SFT 还是 RL 的产品决策问题 | 必须按真实语言/攻击/温度/检测器冻结，换口径必须全量重算 |

当前仓库已经具备**构造 business benchmark 的方法论和脚本基础**，但尚未真正落地一个冻结版本的 `benchmarks/business/` 数据 + harness + scorecard。也就是说：RMTBENCH 已入库，业务 benchmark 的 spec 已入库，**业务 benchmark v1 还待实现**。

---

## 1. 为什么普通 benchmark 不够：本项目每次大转向都来自"评测口径错了"

如果只看传统自动指标，本项目至少会做错三次决策：

1. **v6.5 val 低估产品破功**：训练 val break 约 11%，但产品 roledrop 真实可到 55%。原因不是模型突然变坏，而是 val 几乎没覆盖 bullet/task 注入。这个错误直接导致 375 步 RL 空转。
2. **raw OOC 高估真破功**：`detect_ooc.py` raw flag 会把 echo/code-block 误报混进来；必须人工/规则三分类成"真身份披露 / 真代码顺从 / echo 误报"才可用。否则会把 36.8%（v6.9c 的 raw 值；去误报真值 33.8%）读成真实风险。
3. **clean RM 误当质量分**：v6.6/v6.7 后发现 RM 在 clean 轴跨度只有 0.037，而 attack 轴约 0.36。RM 能看防御，却几乎看不见 clean 富度。继续用 RM 判 clean，就会误以为 RL 在提质量。

这些不是小瑕疵，而是 benchmark 的地基问题：**一个 benchmark 如果看不见目标失败，或者用错打分层，它会比没有 benchmark 更危险**，因为它会给错误行动提供确定感。

---

## 2. Business Benchmark 的核心对象：不是样本，而是"同尺协议"

业务 benchmark 的最小可用形态不是一堆 prompt，而是一份冻结协议：

```text
prompt set
+ 采样参数
+ 检测器版本
+ metric 函数版本
+ judge/rubric 版本
+ base_it / step0 锚
+ 去误报规则
+ 人工抽读规范
= 一个可回归的 scorecard 口径
```

这就是 [BUSINESS_BENCHMARK_GUIDE](../benchmarks/BUSINESS_BENCHMARK_GUIDE.md) 的三条铁律：

1. **分布对齐**：失败分布必须像产品，而不是像训练 val。
2. **同尺原则**：所有模型必须同一次、同检测器、同 metric 重算。
3. **产品温度口径**：用部署温度评，不用低温贪心掩盖高温尾部。

在这个定义下，benchmark 本质上是一个**版本化测量仪器**。改了 prompt、检测器、温度、judge，都不是"补一列"，而是"换了尺子"，必须升版本并全模型重算。

---

## 3. 三层数据：学术锚、业务主集、canary 子集

建议把 benchmark 数据分成三层，避免一个集合同时承担所有职责：

| 层 | 数据 | 回答的问题 | 用法 |
|---|---|---|---|
| B0 外部锚 | RMTBENCH en/zh 多轮 | 通用 roleplay 结构能力是否有外部可比性 | 作为报告附录/对外锚，不作为上线主门 |
| B1 业务主集 | clean + attack + multiturn，约 300 条 | 候选是否比 step0/base 更适合产品 | 每个候选 checkpoint 同尺重跑 |
| B2 canary | 30–50 条高风险样本 | 训练中是否已经触停线 | RL step30/50/100 快速筛，不替代主集 |

本项目已经自然长出了这些材料：

- `outputs/v67_step0_vs_step100/`：step0/step100/base/v68 的同卡 probe 和 sweep，是 B1/B2 的原型。
- `rich_cjk_sft_v69/*clean200*` / `*attack*`：v69/v69b/v69c 的 clean/attack 回归材料，是 business benchmark 的候选主集来源。
- `RMTBENCH/`：外部多轮用户中心 roleplay 结构，是 B0。

缺口是：它们还没有被统一冻结成 `benchmarks/business/v1/`。

---

## 4. 指标分层：每个维度只能交给它能看见的打分层

[BUSINESS_BENCHMARK_GUIDE](../benchmarks/BUSINESS_BENCHMARK_GUIDE.md) 已经把指标分成 L1–L5。深读视角下，更关键的是**不要跨层误用**：

| 维度 | 正确打分层 | 错误用法 | 本项目证据 |
|---|---|---|---|
| 动作/对白/独白结构 | L1 regex + per-language 归一 | 绝对阈值，例如 ja 也要求动作 ≥4.5 | v68/v69 中 ja/ko 天花板不同，绝对门会误杀 |
| 多语言一致性 | L1 语言检测 + 人工 spot-check | 只看平均分 | step25 曾出现韩语 clean 回中文 |
| 角色质量/情绪/用户承接 | L3 LLM-judge + 人工抽读 | 用 RM 代替 judge | clean RM 轴几乎平，无法区分 base>RL 的质量差 |
| 防御 | L4 真破功去误报 + L2 RM 辅证 | 直接报 raw detect_ooc | raw 与真值口径多次冲突 |
| 多样性 | 开场集中度/Jaccard + 人工读 | distinct 前缀一项定胜负 | 旧 distinct 指标几乎全 100%，对模板化失明 |

这条规则可以概括为：

> **regex 看结构，RM 看防御，judge 看语义质量，人工校准指标有效性。**

任何单层指标都可能被刷；business benchmark 的可靠性来自多层交叉，而不是某个指标很聪明。

---

## 5. Benchmark 如何反向决定训练策略

本项目的训练决策其实已经被 benchmark 逻辑支配：

1. **发现 val 与产品分布不一致** → 不再相信训练 val，构造 68 条产品 attack probe。
2. **分轴 RM 打分发现 clean 轴平** → clean 富度退出 RL，转 SFT 蒸馏 base。
3. **v68 独白过冲** → benchmark 加入"独白 ≤ base+10%"反向门，v69 选择器移除独白正向目标。
4. **v69b/v69c defense SFT 斜率反转** → 判断 SFT-defense 到天花板，转 v6.10 RL。
5. **step90 部署参数 sweep** → temperature/top_p/presence/min_p 都调不动真破功，说明问题在模型/RL 层，不在部署旋钮。

所以 benchmark 不只是验收，它也是**训练路线选择器**：

```text
benchmark 看不见失败 → 先修 benchmark
RM 在目标轴无方差 → 换 stage/SFT/judge，不要 RL
SFT 加数据仍斜率反转 → 目标可能需要 on-policy RL
部署旋钮扫不动 → 回训练层，不要继续调采样参数
```

---

## 6. 和 RMTBENCH 的关系：借结构，不借结论

RMTBENCH 的价值在三点：

1. **多轮结构**：character_understanding / maintenance / implicit_motivations / user_preference，正好补足单轮 clean probe 看不见的多轮递进。
2. **用户中心**：它强调用户偏好和隐含动机，适合检测"用户承接"和"不抢主动权"。
3. **外部可解释性**：对外汇报时，比完全自定义 probe 更容易被理解。

但 RMTBENCH 不能替代业务 benchmark：

- 它不覆盖 ChatJoy 的 27 语言分布。
- 它不含产品角色卡的真实格式和风格。
- 它不含 roledrop / task injection / bullet 注入等产品攻击。
- 它没有现成角色回复，需要用本模型生成，再用自定义 judge/rubric 评。

正确用法是：**RMTBENCH 做 B0 外部锚，业务主集做上线门。**

---

## 7. 证据留存：哪些东西必须进仓库，哪些只能 sample

本项目的工作区里有大量不适合直接上传的大文件，但如果只记录"训过 v69/v6.10"，未来就无法复盘。建议分三类：

| 类别 | 处理方式 | 例子 |
|---|---|---|
| 决策报告/设计/服务说明 | 全文入库 | `rl90_temp_sweep_report.md`、`HANDOFF_next_training_quality.md`、`rl_v610_step90_service_guide.md` |
| 小型 probe 结果/scorecard | 可全文或 sample 入库 | v69c clean/attack/RM 各 5 条 sample |
| 训练数据/候选池/merged 模型 | 不入库，记录路径+大小+用途+sample | `rich_cjk_sft_mix_v69b.json`、`rich_cjk_sft_repair_v69c.json`、`rl_v610_step90_merged` |

这次仓库补齐了以下证据：

- `docs/runbooks/rl_v610_step90_service_guide.md`：step90 服务启动参数和硬件条件。
- `reports/sft_richness/rl90_temp_sweep_report.md`：step90 部署参数 sweep 的完整结论。
- `docs/archive/HANDOFF_next_training_quality.md`：下一轮质量恢复路线的原始草案，记录 "base > RL(step90)" 后的诊断。
- `data/samples/*v69c*sample5.jsonl` / `*rl90*sample5.jsonl`：代表性 probe/RM 输出样本。

仓库仍不应该上传 300MB 级 mix 文件和 49GB 模型权重，但必须在 `DATA.md` 里记录它们的位置、规模和用途。

---

## 8. 当前 Benchmark 侧缺口清单

| 缺口 | 为什么重要 | 建议落点 |
|---|---|---|
| 没有冻结的 `benchmarks/business/v1/` | 现在有 spec 和散落 probe，但还没有一键可跑 benchmark | `benchmarks/business/README.md` + `data/` + `harness/` + `scorecards/` |
| L3 judge harness 未统一 | 角色质量、用户承接、多轮递进仍靠临时 GPT 判断 | `benchmarks/business/harness/judge.py` |
| L4 真破功去误报未 benchmark 化 | `detect_ooc.py` 与人工三分类还散在训练目录 | `benchmarks/business/harness/detect_true_break.py` |
| RMTBENCH 未接入模型生成脚本 | 数据已入库，但还不是可跑流程 | `benchmarks/RMTBENCH/run_chatjoy_rmtbench.py` |
| scorecard schema 未冻结 | 后续模型容易各报各的指标 | `benchmarks/business/scorecards/schema.json` |

这些缺口并不影响当前仓库作为工程档案使用，但会影响它作为"可复现实验系统"使用。

---

## 9. 一页执行建议

下一步如果要把 benchmark 真正落地，按这个顺序：

1. 从 `rich_cjk_sft_v69` 和 `outputs/v67_step0_vs_step100` 抽出 held-out clean/attack/multiturn，冻结 `bizbench-v1`。
2. 把 `gen_clean_probe.py`、`gen_attack_probe.py`、`detect_ooc.py`、`score_attack_rm.py` 迁入 `benchmarks/business/harness/` 或写 thin wrapper。
3. 建 `scorecard.md + scorecard.json` 双格式，固定 base_it / step0 / 当前候选三列。
4. 每次新增 checkpoint，只追加一列；任何检测器或 prompt 改动，升 benchmark 版本并全量重算。
5. 每次训练路线切换，都要求 benchmark 给出"为什么不是继续调参"的证据。

最重要的判断标准：

> 如果 benchmark 结论和人工抽读冲突，不是人工错了，而是 benchmark 的某个代理指标失效了。先修尺子，再训模型。

