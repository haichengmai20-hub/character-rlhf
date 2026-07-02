# 深读 9（全链路差分归因）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增一篇 deep dive（`deep_dives/09_full_chain_attribution.md`），教读者在"业务指标没提升 / 上线不行"时，用差分诊断法把根因判给 RLHF 链路上正确的一层；配 insert-only 索引更新。

**Architecture:** 纯文档任务。单个 Markdown 交付物，按 house style 组织（能力 blurb + Attribution Lens + §0 反直觉表 + §1 差分总表 + §2–§9 逐层 + §10 决策图 + §11 方法论 + §12 面试版 + footer）。所有数字取自仓库既有文件，写作即核对；新提诊断标 `Proposed diagnostic — not yet run`。

**Tech Stack:** Markdown。无代码 / 无 pytest / 无依赖。"测试"= 数字对照源文件 + house-style 清单。

## Global Constraints

- 不新增单测 / pytest；不新增训练代码；不改训练 objective；不引入依赖。（逐字来自 spec §5）
- 不编造不存在的训练结果 / 指标 / checkpoint；所有数字有仓库出处；新提诊断一律标 `Proposed diagnostic — not yet run`。
- 原始仓库关于整个 Character RLHF 流程 flow 的内容严格完整保留；本篇为新增文档，对既有文件仅 insert-only，不删不改现有行。
- 中文；标题固定：`深读 9 — 全链路差分归因：指标没提升 / 上线不行时，如何把根因判给正确的那一层`。
- house style 必备件：`> 本篇展示的能力` blurb、`> ## Attribution Lens` block、`## 0. 反直觉的起点`、`**机制提炼**` callout、决策图 code-block、footer 关联链接、附录数字来源。
- 与深读 5/8 分工须在正文点明：5=前向门、8=credit 单透镜、9=后向差分；credit 公式只在深读 8，不重推。
- 不 git commit（仓库非 git 仓库，且用户未要求提交）。

## 数字来源速查（写作时逐一核对，不得凭记忆）

| 数字 | 源文件 |
|---|---|
| base 27% / RL 55% / 91%；v6.5 375 步零移动；val 11%≠产品 55% | `CASE_STUDY.md` / `README.md` L52,L107,L115 / 深读 3、5 |
| clean 跨度 0.037 vs attack 0.36；RM v2 batch 减半+score head bug；RM v3 cosine LR 步数不足 | `docs/VERSION_MAP.md` §1（L30-37）/ 深读 5 §2 |
| v6.9→v6.9c unique defense 120→2686（22×）真破功 32.4%→33.8%；78.8% in-char 良率；真代码 5/9/11 | 深读 5 §3 表 + `reports/v69_reconcile_report.md` |
| v6.7 分轴 +0.013 vs 混合分 +0.092；温度脆 t0.7 0.293→0.305 / t1.0 崩 | 深读 5 §4 + 附录 / `docs/chatjoy_rl_v610_defense_clean_aware_design.md` §0 |
| v6.8 独白过冲 187%（选择器=放大器） | `docs/VERSION_MAP.md` §2（L50）/ 深读 4、6 |
| anti_refusal +3.0 反向破角 Goodhart；OOC 误报 1/16→0/16 | 深读 2 / 深读 8 §4.1 / `README.md` L104 |
| v6.10 step90 真破功 36.8%（25/68）；attack-split RM 0.235<0.271；A6 code 11→12；采样 sweep ~30% 地板 | `README.md` L118-119 / `docs/VERSION_MAP.md` §2（L54）/ `reports/sft_richness/rl90_temp_sweep_report.md` |
| credit 层机制（sequence→token/span、per-source KL 0.008/0.002、bad_span_kl_share） | 深读 8 + `docs/CREDIT_ASSIGNMENT_TAXONOMY.md`（§8 深链，不重推） |

---

### Task 1: 文件骨架 + §0 反直觉表 + §1 差分总表

**Files:**
- Create: `deep_dives/09_full_chain_attribution.md`

**Interfaces:**
- Produces: 文件存在，含标题、能力 blurb、Attribution Lens、`---`、§0、§1。后续 Task 2–6 向同一文件追加 §2–§12 与 footer。

- [ ] **Step 1: 写文件头（标题 + 能力 blurb + Attribution Lens）**

写入（逐字）：
- H1：`# 深读 9 — 全链路差分归因：指标没提升 / 上线不行时，如何把根因判给正确的那一层`
- `> 本篇展示的能力：` blurb 一段，主张：把"指标没动"从条件反射调 RL 参数，升级成一套**从症状出发、按成本从低到高逐层排除**的差分诊断；并点明与深读 5（前向门）、深读 8（credit 单透镜）的分工——本篇是**后向**差分，credit 只是其中一列，深链深读 8 不重推。
- `> ## Attribution Lens` block：列出本篇会把"指标没动"这个症状归因到的 8 个候选层（评测/度量、数据、SFT 强度、RM、reward 规则、RL 参数/动力学、credit assignment、部署采样）。
- 一条 `---` 分隔线。

- [ ] **Step 2: 写 §0 反直觉的起点（钩子表）**

`## 0. 反直觉的起点：同一个症状，四个不同病灶`。含下表（逐字）：

| 事件 | 表面症状 | 一线直觉归因 | 真实根因层 |
|---|---|---|---|
| v6.5 375 步零移动 | attack 不降 | "RL 学不动=算法/参数" | 评测层（val 11%≠产品 55%） |
| v6.7 clean +0.037 | clean 提不动 | "RL 步数不够" | RM 层（clean 轴跨度 0.037 无方差） |
| v6.8 独白过冲 187% | 富度坏了 | "SFT 数据不够" | 数据/选择器层（选择器=放大器） |
| v6.10 attack 压不动 | attack 不降 | "RL 没调好" | credit 层（-5 无法定位到破角 span） |

表后一句结论：4 次"指标没动"，一线直觉全指向 RL 算法，真实根因 0 次在算法本身 → 归因错层比问题本身更贵，本篇讲怎么不被症状骗。

- [ ] **Step 3: 写 §1 差分诊断总表**

`## 1. 差分诊断总表（一页速查）`。一段引子说明"列=候选病因，逐行=特征指纹/最小判别实验/相对成本/项目确诊案例"，并说明排查按成本从低到高。含 8 行主表（每层一行，credit 与 RL 参数分两行）：

| 候选层 | 特征指纹（区别于相邻层） | 最小判别实验 | 成本 | 项目确诊案例 |
|---|---|---|---|---|
| 评测/度量 | 指标动/不动都可能是度量假象 | 生产 vs 评测失败分布比对；同尺重算 | 极低 | v6.5 val 11%≠产品 55% |
| 数据 | 加数据斜率仍为负 | 加定向数据看斜率 | 中 | A6 0 覆盖；v6.8 选择器放大 |
| SFT 强度 | 加数据斜率反转 | 22× 数据看是否不降反升 | 中 | v6.9→v6.9c 120→2686 反转 |
| RM | 目标轴 reward 方差≈0 | 分轴方差离线打分 | 低 | clean 0.037 vs attack 0.36 |
| reward 规则 | 组件把 credit 记到错误行为 | 去 bonus/加硬罚消融 + 组件 breakdown | 低 | anti_refusal +3.0 反向破角 |
| RL 参数/动力学 | 改超参有动力学响应 | 扫 temp/N/KL/GSPO 看响应 | 高 | v6.7 温度脆；375 步硬化 |
| credit assignment | 超参调对、方向对，仍压不动 | bad_span_kl_share（`Proposed`） | 高 | v6.10 attack 36.8% |
| 部署采样 | 改采样只改表达不改能力 | 固定权重扫采样参数 | 极低 | sweep ~30% 地板 |

- [ ] **Step 4: 核对与自检**

核对表内每个数字对上"数字来源速查"。确认 §0/§1 与源文件一致；确认 Attribution Lens 的 8 层与 §1 的 8 行一致。

- [ ] **Step 5:（跳过 commit）** 记录进度，非 git 仓库不提交。

---

### Task 2: §2 评测/度量层 + §3 数据层

**Files:**
- Modify: `deep_dives/09_full_chain_attribution.md`（追加 §2、§3）

**Interfaces:**
- Consumes: Task 1 建立的文件与 §1 总表列名。
- Produces: §2、§3 两节，均用统一模板。

- [ ] **Step 1: 写 §2 评测/度量层**

`## 2. 评测 / 度量层：先证明"指标真降了"`。统一模板五段：
- 症状表现：业务指标没动 / 上线差，但可能根本没测对。
- 特征指纹：训练 val 与生产失败分布不重合；业务指标与技术代理错配；raw vs 去误报口径不一致。
- 最小判别实验：拉真实生产输入统计失败类型分布 vs val/probe 分布；同尺重算（同检测器去 echo/code 误报）。成本极低。
- 项目确诊案例：v6.5 val attack 11% ≠ 产品 roledrop 55%（bullet 注入在 val 仅 ~5%）；v6.7 "防御 +0.092"（clean+attack 混合分）经分轴重算实为 +0.013——一次口径不对导致结论高估。
- 常见误判：把"评测盲区"误判成"RL 学不动"（v6.5 正是如此）。
- `**机制提炼**`：RL 只优化评测能打分的东西；失败分布≠生产分布会同时"学不动"+"看不见"。

- [ ] **Step 2: 写 §3 数据层**

`## 3. 数据层：覆盖缺口 vs 分布偏`。统一模板：
- 症状：某类失效怎么训都不改善。
- 特征指纹：该失效在训练 batch 里样本量≈0；或选择器把某模态放大。
- 最小判别实验：补定向数据后看斜率是否转负；检查目标失效在训练集覆盖率。成本中。
- 项目确诊案例：A6 任务注入/代码在 defense 池 0 覆盖；v6.8 选择器=放大器（独白进目标→过冲 187%）。
- 常见误判：与 §4 SFT 强度层易混——区别在"加数据斜率还为负（数据层可补）"vs"斜率反转（SFT 到顶）"。
- `**机制提炼**`：数据层的失败能靠覆盖补；能补的别怪算法。

- [ ] **Step 3: 核对与自检**（数字对源；模板五段齐全；§3 明确点出与 §4 的判别关键）

---

### Task 3: §4 SFT 强度层 + §5 RM 层

**Files:**
- Modify: `deep_dives/09_full_chain_attribution.md`（追加 §4、§5）

**Interfaces:**
- Consumes: Task 2 的模板与 §3 末尾的"§3⇄§4 判别"。
- Produces: §4、§5。

- [ ] **Step 1: 写 §4 SFT 强度层**

`## 4. SFT 强度层：base 分布能不能当 RL 起点`。模板：
- 症状：残余失败像"该上 RL"，但其实 SFT 还没到顶 / 或已到顶。
- 特征指纹：加数据斜率**反转**（不降反升）=SFT 范式到天花板。
- 最小判别实验：把 unique 数据放大一个数量级看斜率；用 self-distill 良率否决"teacher 太弱"假说。成本中。
- 项目确诊案例：v6.9→v6.9c unique defense 120→2686（22×），真破功 32.4%→33.8%（斜率反转）；step0 自蒸馏 78.8% in-char 良率、49 条 A6 正样本 0% 含代码 → 否决"teacher 产不出 defense"，确认是 SFT 范式（静态分布对齐）修不动稀有轨迹尾巴。
- 常见误判：与 §3（数据可补）、与 §7/§8（该不该起 RL）易混。
- `**机制提炼**`：斜率反转是"SFT 到顶"的可观测信号；到顶才轮到 RL。

- [ ] **Step 2: 写 §5 RM 层**

`## 5. RM 层：目标轴上有没有梯度`。模板：
- 症状：RL 在目标轴上空转（如 clean +0.037）。
- 特征指纹：把 RM 打分按目标轴分组，目标轴 reward 方差/跨度≈0。
- 最小判别实验：一次离线打分，按目标轴分组算方差/跨度（比烧 GPU 便宜几个数量级）。成本低。
- 项目确诊案例：clean 跨度 0.037 vs attack 0.36（~10×）→ RM 是防御判别器不是通用质量器；另附 RM 训练本身的归因（v2 acc 退步根因=batch 减半+score head 保存 bug；v3 根因=cosine LR 步数不足）作"RM 内部也要归因"的例子。
- 常见误判：与 §8 credit 层易混——两者都表现为"attack/目标轴不降"，区别：RM 层是**分轴方差为零**（没信号），credit 层是**有方差但落不到 span**（有信号但错位）。这条判别关键在此写清，§8 呼应。
- `**机制提炼**`：RL 只能提升 RM 能排序的轴；方差是 RM 给的，不是优化器能造的。

- [ ] **Step 3: 核对与自检**（§5 必须写清"RM 无方差 vs credit 无定位"判别关键；数字对源）

---

### Task 4: §6 reward 规则层 + §7 RL 参数/动力学层 + §8 credit assignment 层

**Files:**
- Modify: `deep_dives/09_full_chain_attribution.md`（追加 §6、§7、§8）

**Interfaces:**
- Consumes: §5 末尾的"RM 无方差 vs credit 无定位"判别。
- Produces: §6、§7、§8；§7 与 §8 是用户明确要求拆开的"算法原因 vs credit 原因"。

- [ ] **Step 1: 写 §6 reward 规则层**

`## 6. reward 规则层：组件把 credit 记到了错误行为`。模板：
- 症状：方向像对的，行为却越训越坏（reward 上升但真实目标退化）。
- 特征指纹：某 rule 组件对"错误但表面合规"的行为给正分。
- 最小判别实验：组件 breakdown 日志 + 去 bonus/加硬罚的消融（改 reward 组件不重训 RM，看行为变化）。成本低。
- 项目确诊案例：anti_refusal +3.0 NON_REFUSAL_BONUS 把"我是 AI 但继续回答"判为合规 → 反向训练破角，base 27%→RL 55%/91%；修复=anti_ooc_disclosure 硬罚 -5 + 撤 bonus，OOC 误报 1/16→0/16（全 TDD 不重训 RM）。
- 常见误判：与 §8 易混——reward 规则层是**功劳记错组件/行为**（方向错），credit 层是**方向对但落点错**（span 错）。
- `**机制提炼**`：Goodhart 的本质是 credit 记在错误组件上；先修方向（规则层），再谈定位（credit 层）。链深读 2。

- [ ] **Step 2: 写 §7 RL 参数 / 优化动力学层**

`## 7. RL 参数 / 优化动力学层：超参与动力学能不能榨出梯度`。模板：
- 症状：目标轴该有梯度但没榨出来 / 只在训练温度附近成立。
- 特征指纹：**改超参有动力学响应**（temp/N/KL/GSPO/步数一动，指标或熵/grad_norm 就变）。
- 最小判别实验：扫 rollout_temp、group N、KL、GSPO 聚合，看响应曲线；对比训练温度与部署温度。成本高（要跑 RL）。
- 项目确诊案例：v6.7 N=6/temp0.7 只锐化低温模态（温度脆）；v6.10 rollout_temp=1.0 + N=12 让稀有破功尾巴入组；v6.5 硬化 GRPO 375 步榨不出梯度（叠加评测层根因）。
- 常见误判：与 §8 credit 层是本篇最关键的一对区分——写清判别：**改超参有响应=参数层**；**超参调对、方向对、仍压不动=credit 层**（下一节）。
- `**机制提炼**`：参数层能调"放大器强度和方向覆盖面"，但调不出"把罚精确落到 span"的能力。

- [ ] **Step 3: 写 §8 credit assignment 层**

`## 8. credit assignment 层：方向对了，罚落不到破角 span`。模板（**深链深读 8，不重推公式**）：
- 症状：评测对（§2 过）、数据到顶（§4 过）、RM 有方差（§5 过）、方向修对（§6 过）、超参调对（§7 无更多响应），attack 仍压不动。
- 特征指纹：sequence-level 的 -5 被 GRPO 摊到全部 token，破角 span 只占少数 token → 负 credit 稀释。
- 最小判别实验（均标 `Proposed diagnostic — not yet run`）：`bad_span_token_kl_share`（更新是否集中在坏 span）、`good_span_kl_share`（正常 RP 是否被误伤）；探索 span-masked penalty。
- 项目确诊案例：v6.10 step90 attack 真破功 36.8%（25/68）、attack-split RM 0.235<0.271、A6 code 11→12；采样 sweep 只能到 ~30% 地板 → 排除评测/数据/RM/方向/参数后，剩下的嫌疑就是 token/span credit 缺位。
- `**机制提炼**` + 一句"完整机制与六层 taxonomy 见深读 8，本篇只把它定位成差分树的最后一层"。

- [ ] **Step 4: 核对与自检**（§7↔§8 判别关键写清；credit 公式未在本篇重推；Proposed 标注齐全；数字对源）

---

### Task 5: §9 部署采样层 + §10 排查流程图

**Files:**
- Modify: `deep_dives/09_full_chain_attribution.md`（追加 §9、§10）

- [ ] **Step 1: 写 §9 部署采样层**

`## 9. 部署 / 采样层：旋钮只改表达，不改能力`。模板：
- 症状：上线效果差，第一反应调采样。
- 特征指纹：固定模型权重扫采样参数，指标只在小范围抖动、有地板。
- 最小判别实验：固定 ckpt 扫 temperature/top_p/presence/min_p，看是"表达变化"还是"能力变化"。成本极低。
- 项目确诊案例：v6.7 温度脆 t0.7 +0.013 / t1.0 崩；v6.10 采样 sweep 只能把 attack 地板微调到 ~30%，证明问题在训练层不在服务层。
- `**机制提炼**`：采样层是最便宜的排除项，排除它能把注意力逼回训练链路。

- [ ] **Step 2: 写 §10 排查流程图**

`## 10. 把差分串成一张排查流程图`。一个 code-block 决策树，按成本从低到高：评测→部署采样→RM 分轴方差→reward 组件→数据斜率→SFT 斜率→RL 参数响应→credit 定位，每分支标"确诊它会看到什么"。正文点明这是深读 5 决策图的**镜像**（5 问要不要起 RL，9 问已经不行锅在哪）。

- [ ] **Step 3: 核对与自检**（决策树分支顺序与 §1 成本列一致；数字对源）

---

### Task 6: §11 方法论提炼 + §12 面试版 + footer

**Files:**
- Modify: `deep_dives/09_full_chain_attribution.md`（追加 §11、§12、footer）

- [ ] **Step 1: 写 §11 归因方法论提炼（可迁移）**

`## 11. 归因方法论提炼（可迁移）`。含 5 条原则（(a) 先证症状为真再归因 (b) 每层都有便宜几个数量级的离线预检 (c) 控制变量一次只动一层 (d) 廉价判别器：斜率/方差/分轴/分源/同尺 (e) 易混病因对小表）。含"易混病因对 + 判别关键"表（逐字）：

| 易混病因对 | 共同表面症状 | 判别关键 |
|---|---|---|
| RM 无方差(§5) vs credit 无定位(§8) | attack/目标轴不降 | 分轴方差是否≈0（有信号无信号） |
| 数据覆盖缺口(§3) vs SFT 到顶(§4) | 加数据看似该有效 | 加数据斜率是否反转 |
| RL 参数(§7) vs credit(§8) | 目标轴压不动 | 改超参是否有动力学响应 |
| 评测盲区(§2) vs RL 学不动(§7) | 指标零移动 | 生产/评测失败分布是否重合 |

- [ ] **Step 2: 写 §12 面试讲述版本**

`## 12. 面试讲述版本`。一段可口述版：落在"我把'指标没动'训练成一套按成本从低到高的差分诊断，先证症状为真、再逐层用离线预检排除，而不是条件反射去调 RL 参数——这个项目里 4 次指标没动，真实根因分别在评测/RM/数据/credit 四个不同层"。

- [ ] **Step 3: 写 footer 关联链接**

一行 `>` 关联：深读 2（reward 组件/Goodhart）、深读 4（SFT 天花板）、深读 5（前向门，本篇镜像）、深读 8（credit 单透镜，§8 深链）、[VERSION_MAP](../docs/VERSION_MAP.md)、[chatjoy_eval_paradigm](../docs/chatjoy_eval_paradigm.md)。

- [ ] **Step 4: 全文自检**（对 spec §7 验收标准逐条核：8 层各有真实案例、§7/§8 判别、Proposed 标注、与 5/8 分工点明、house style 齐全）

---

### Task 7: insert-only 索引更新（README + docs/README）

**Files:**
- Modify: `README.md`（📚 索引表 L57 后加行；dir-tree L143 后加行）
- Modify: `docs/README.md`（主线区补一条指向深读 9 的链接）

**Interfaces:**
- Consumes: Task 1–6 完成的 `deep_dives/09_full_chain_attribution.md`。

- [ ] **Step 1: README 📚 索引表加行**

在 `README.md` 深读 8 行（L57）之后插入一行（不改其它行）：

```
| 9 | [全链路差分归因](deep_dives/09_full_chain_attribution.md) | 指标没提升/上线不行时，用差分诊断把根因判给正确的一层：同一症状、8 个候选病因、每个配最小判别实验 | v6.5 评测盲区 / v6.7 RM 无方差 / v6.8 数据放大 / v6.10 credit 缺位 四次"指标没动"归到四个不同层 · RM 无方差 vs credit 无定位 / 数据可补 vs SFT 到顶 / RL 参数 vs credit 的判别关键 · 排查流程图（深读 5 的镜像）· 新诊断均标 proposed |
```

- [ ] **Step 2: README 仓库结构 dir-tree 加行**

在 `README.md` 的 `08_credit_assignment...md` 行（L143）之后插入一行：

```
│   └── 09_full_chain_attribution.md              全链路差分归因 × 8 候选病因 × 最小判别实验 × 排查流程图
```

并把上一行 `08_...md` 的树枝符号从 `└──` 改为 `├──`（保持 tree 结构正确——此为结构性字符修正，非内容删改）。

- [ ] **Step 3: docs/README.md 加链接**

在 `docs/README.md` 主线区末尾（L21 "仓库根目录的 README/CASE_STUDY/..." 提示行之前或之中）补一处指向深读 9 的引用，保持 insert-only。

- [ ] **Step 4: 核对**（三处插入均未改动既有行内容；链接路径可点；深读 8 行仅树枝符号 └→├ 的结构修正）

---

## Self-Review（写计划后自检）

- **Spec coverage**：spec §3 的 §0–§12 全部映射到 Task 1–6；spec §6 insert-only 改动映射到 Task 7；spec §7 验收标准落到各 Task 的"核对与自检"步与 Task 6 Step 4。✅
- **Placeholder scan**：表格数字均为实值并附来源速查；无 TBD/TODO（除按约束要求在正文标注的 `Proposed`）。✅
- **一致性**：§0 四事件 ↔ §1 八行 ↔ Attribution Lens 八层 ↔ §11 易混对表，交叉引用一致；§5/§7 均指向 §8 的判别关键，措辞统一（"分轴方差是否≈0"、"改超参是否有响应"）。✅
