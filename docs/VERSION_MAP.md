# 版本地图（VERSION_MAP）

> 一张表看懂 ChatJoy 从早期 RL 到 v6.10 的演进：每个版本**想解决什么、改了什么、实测结果、最终结论**。
> 数字均来自仓库内可核对的记录（CASE_STUDY / deep_dives / reports）；标注"raw"者为未去误报的原始检测值，"真"为同检测器去 echo/code 误报后的真破功率（口径见 [深读 5 附录](../deep_dives/05_when_to_move_from_sft_to_rl.md) 与评测范式）。
> 更新时间：2026-07-08。**版本对应与时序的事实校准 = [project_chatjoy_overview.md](project_chatjoy_overview.md)（整理版全量台账）+ [chatjoy_rm_v6_3_roledrop_handoff.md](archive/chatjoy_rm_v6_3_roledrop_handoff.md)（原始交接证据，顶部有摘要）**；本表是汇总视图，冲突时优先看整理版 overview，再回 handoff 核原始过程。
> **版本命名**：全项目新命名 `RM-N / SFT-N / RL-N`（如 RM-6.3 / RL-6.3 / SFT-6.9c，消除三线共用 `v6.x` 同号不同物）↔ 原始号 ↔ 产品交付号 的对照以本表和 overview 的阶段映射为准。要点：`v6.4/v6.5` 是 **RL** 占号（RM 从 6.3 直接跳 6.6）；`rl_full_v3/4/5` 是 RL-6.3 重启计数非版本号；产品只 v1(91%)/v2(55%=RL-6.4·160) 坐实、其余交付版（RL-6.6b/6.5·425/6.10·step90）未逐一编号。

---

## 0. 一句话主线

```
RM 迭代(RM-1→RM-5 软判别器) → RM-6.x 细粒度 RM(8维 rubric/rank_delta/roledrop pair)
   → RL-6.3 平台 + Goodhart 事故(产品 v1, 91%) → 两步修复+防御 SFT → RL-6.4(=产品 v2, 55%) → RL-6.5 硬化触顶(375 步零移动 + val≠产品分布)
   → 战略转向：clean 富度交给 SFT(SFT-6.7→SFT-6.8 失败→SFT-6.9 成功)，防御留给 RL
   → 两轮 SFT-defense 到天花板(SFT-6.9b/6.9c)
   → RL-6.10 step90：clean 守住但 attack 未降 → 转 quality/RM/数据层
（命名：RM-/SFT-/RL- 为新号，括注或旁注保留原始 v6.x；阶段解释见 overview 和本表）
```

两条贯穿全程的判断（详见 [深读 5](../deep_dives/05_when_to_move_from_sft_to_rl.md)）：
1. **RL 不能凭空创造目标分布**——分布由 SFT 注入，RL 只放大已存在的偏好。
2. **RL 不是免费的**——早期 RL 曾把 base 自带的防御训坏（roledrop 破功 base 27% → RL 55%/91%）。

---

## 1. RM 演进（判别器，先于在线 RL）

| 版本 | 目标 | 关键改动 | 实测指标 | 结论 |
|---|---|---|---|---|
| RM v1 | 偏好判别基线 | rm_train 16,826 / eff-batch 128 | **acc 94.5%**，gap +1.60 | 最强基线；step200 ckpt 磁盘丢失 |
| RM v2 | 删长噪声样本 | filtered 14,826 / eff-batch 64 | acc 86.4%（**退步**） | 根因=batch 减半 + **score head 保存丢失 bug** |
| RM v3 | 加数据 | 18,955（含 2000 缺陷样本）| acc 86.0%，rejected 仅 −0.30 | 根因=**cosine LR 步数不足** + 缺陷样本拉抬 rejected |
| RM v5 | 修复 + hard-neg | 17,712（5% badpool）/ batch 64 | **acc 95.5%**，gap 3.50，chosen +2.27/rej −1.23 | 判别边界清晰的"软二分类器"，可当 RL reward |
| RM v6/v6.2 | 跨验证去单 judge 偏置 | 5 source 同分布 + DeepSeek/Qwen 交叉投票 | saturated 0% | 引入 LLM-judge 流水线 |
| RM v6.3 | 细腻度 + 抗破角做进偏好 | **8 维 rubric + rank_delta 难负样本 + roledrop_attack pair + base 当第 6 candidate**；40,551 train / 3,423 eval | 训练完成并 merge；acc≥0.84 / saturated=0% / reward_std≥2.0 是**设计验收门，最终数值未记录** | 作为后续 RL 主信号；不要把门槛写成通过结果。详见 [深读 2](../deep_dives/02_rm_data_reward_and_attribution.md) |
| RM v6.6 | 后续 RL reward | 承接 v6.3 细粒度方向 | attack-split 区分度 ~0.36（clean 仅 0.037）| v6.7–v6.10 的 RL reward；**clean 上几乎平=RL 无 clean 梯度** |

> RM 的核心发现：它是一个**防御判别器**，在 clean 质量轴上几乎没有区分度（范围 0.037 vs attack 0.36，约 10×）。这决定了"clean 富度交给 SFT、防御交给 RL"的分工。

---

## 2. 在线 RL / SFT 版本（actor 侧）

| 版本 | 阶段 | 目标 | 关键改动 | 核心指标（实测） | 结论 / 是否保留 |
|---|---|---|---|---|---|
| **RL-6.3**（原 v6.3） | RL | roledrop 防御 | actor=SFT2-merged，RM v6.3，psKL（run=rl_full_v5；05-29 NON_REFUSAL_BONUS 抬至 +3.0；06-01 SAVE_FREQ 25→5 解"净进度 0 死循环"） | clean_RM ~1.29–1.35，attack break **~15–16% 纹丝不动**，平台；最优点 step325 手动保全 | 06-02 停训。**产品口径"早期 RL v1"（roledrop 破功 91%）出自本期**（+3.0 Goodhart 事故期；v1 对应的具体 merge 未记录）；06-02 真破角诊断实测 325 在直接身份攻击下近 100% 破角 → 催生 anti-OOC 修复 |
| **RL-6.4**（原 v6.4） | RL | anti-OOC 修复 + 防御 SFT 后再 RL | actor=`sft2_defense_merged_full`（06-03 防御 SFT：2,691 defense + 10,764 replay），RM v6.3，psKL；06-04 于 step110 加载二次修复 reward（否定/引号豁免，误报 1/16→0/16）续训 | 续训至 step160 → merge `rl_v6.4_step160_merged` | **= 产品口径"早期 RL v2"**（roledrop 破功 **55%**：较 v1 91% 大幅改善、但仍差于 base 27%）；overview 06-08 §5"= v2/base"、v6.5 冒烟"仍 ~v2"两处印证 |
| **RL-6.5**（原 v6.5） | RL | 硬化 GRPO 再榨梯度 | fresh from v6.4_step160，temp 1.2，length TARGET 1500，n=6 | **375 步零移动**：clean_RM 卡 ~1.33 天花板、break ~11–12% 不降 | ❌ **停训**；发现 val 11%≠产品 55%；→ 战略转向 |
| **RL-6.6b**（原 v6.6b） | RL | — | RM-6.6 | val −0.601→**−0.315** 后停 | 当时最优点 `rl_v6.6b_best_merged`（保留） |
| **RL-6.7**（原 v6.7；actor=SFT-6.7） | rich-SFT + RL | 提 clean 富度 + 防御 | actor=rich CJK SFT（`v67_repair_merged`）+ RL | clean RM **+0.037≈噪声**；防御 t0.7 +0.013（**t1.0 更差=温度脆**）| ❌ RL 提不动 clean；确立"富度=SFT 的活"；温度脆 → v6.10 修 |
| **SFT-6.8**（原 v6.8） | SFT-max | clean 逼近 base teacher | FFN/router LoRA + judge 选样（独白进了目标）| **独白过冲 187%**、对话回退、动作没动 | ❌ **失败**：选择器=放大器，偏哪放大哪；弃用 |
| **SFT-6.9**（原 v6.9） | rich-SFT | clean 富化（修 SFT-6.8）| best-of-N 均衡选择器（动作+对话+sensory/interaction/emotion，**独白不入目标**）| clean **PASS**（动作 105–111%/对话 110–160%/无独白过冲）；attack raw **42.6%**↑ | clean 干成；但富化侵蚀防御（A6 池 0 覆盖）→ 补 defense |
| **SFT-6.9b**（原 v6.9b） | rich-SFT+defense | 补防御第一轮 | +120 条 step0 自蒸馏 defense | 真破功 **32.4%**（22/68），RM 0.234 | 比 v6.9 改善，但压不回 step0 16.2% |
| **SFT-6.9c**（原 v6.9c） | rich-SFT+defense | 补防御第二轮（跨角色）| +2686 条跨角色自蒸馏 defense（连同旧 defense 240 条合计占 4566 mix **64.1%**）| 真破功 **33.8%**（23/68），RM **0.271**（最高）；clean PASS 但 zh-tw mono 漂到 24% | **SFT-defense 到天花板**（22× 数据不降）；作 RL v6.10 的 actor |
| **RL-6.10**（原 v6.10） | defense RL | 防御硬化 + 守 clean | 远端记录：actor=v6.9c，RM v6.6，rollout_temp=1.0，N=8/12，per-source KL，break_pen −5、clean_q λ=0.25；当前仓库缺完整 v6.10 launcher/reward patch | step90 外部 probe：clean 守住；真破功 **36.8%（25/68）≈ v6.9c 33.8%**；attack RM **0.235 < 0.271**；A6 code 11→12；部署 sweep 真破功约 30% 地板 | ❌ **停线**：结果有报告证据，但本地代码不能完整复现 reward 分支；转 quality-aware benchmark / RM v7 / 定向数据 |

---

## 3. 关键里程碑（按"决策"而非"版本"组织）

| 决策点 | 触发证据 | 结论 | 深读 |
|---|---|---|---|
| 发现 RL 在奖励破角 | `is_refusal` 对"我是 Gemma 4"完全瞎；真实破功≈100% 却显示 16% | anti-OOC 硬罚 −5 + 撤销误给 bonus（不重训 RM，全 TDD）| [深读 2](../deep_dives/02_rm_data_reward_and_attribution.md) |
| v6.5 停训 | 375 步零移动 + val 11%≠产品 55% + RL 把防御训坏 | 停 RL，回数据/信号层 | [深读 5 门 A/B/D](../deep_dives/05_when_to_move_from_sft_to_rl.md) |
| 富度归 SFT、防御归 RL | RM 在 clean 上平（range 0.037），v6.7 RL clean +0.037≈噪声 | 分工锁定 | [深读 4](../deep_dives/04_richness_defense_tension_and_sft_ceiling.md) |
| SFT 选择器是放大器 | v6.8 独白进目标 → 过冲 187% | profile 硬门须在**训练前**跑 | [深读 4](../deep_dives/04_richness_defense_tension_and_sft_ceiling.md) |
| SFT-defense 到天花板 | unique defense 120→2686（22×），真破功 32.4%→33.8%（不降反升）| 转 RL v6.10 | [深读 4/5](../deep_dives/04_richness_defense_tension_and_sft_ceiling.md) |
| v6.10 step90 停线 | clean 守住但 attack 真破功 36.8%，RM 低于 v6.9c；采样 sweep 只能把地板微调到约 30% | 当前问题不在服务采样，而在 RM/数据/训练信号；进入 quality-aware benchmark 与 RM v7 设计 | [深读 7](../deep_dives/07_business_benchmark_and_evidence_preservation.md) / [采样报告](../reports/sft_richness/rl90_temp_sweep_report.md) |

---

## 3.5 Credit Assignment 视角：把版本演进读成"补哪一层 credit"

> 完整分析见 [深读 8 — Character RLHF 中的 Credit Assignment](../deep_dives/08_credit_assignment_in_character_rlhf.md)。

上面的版本表还可以按"每一版在补 sequence→token credit 错配的哪一层"来读——这解释了为什么方向修对了、source 也解耦了，attack 仍压不动：

| 版本 | 补的是哪一层 credit | 观察结果 | 归因 | 决策 |
|---|---|---|---|---|
| 早期 RL v1 → anti-OOC 修复 → v6.4（=产品 v2） | 组件 / sequence（方向） | v1 91% → 修复后 v2 55%（base 27%） | +3.0 bonus 把 credit 分给破角 | 硬罚 -5 + 撤 bonus + 否定/引号豁免 |
| v6.7（clean RL 停） | group（方差） | clean +0.037≈噪声 | RM clean 跨度 0.037 无 group credit | 富度归 SFT |
| v6.10（per-source KL） | source（解耦） | clean 守住、attack 未降 | source credit 分开但 attack 内部未定位 | step90 停线 |
| 下一步（RM v7 / 定向数据） | token / span（定位）**（proposed）** | — | -5 仍无法落到破角 span | 把 credit 下沉到 span |

**主线一句话**：v1→v6.10 先修 reward 方向、再补 group 方差、再做 source 解耦；token/span 定位仍缺位，但它只是 v6.10 后的高优先级假设之一，需与 targeted data 和 RM 区分度一起验证。

---

## 4. 怎么用这张表

- **想快速对齐项目进度**：看 §0 主线 + §2 最后三行（v6.9c / v6.10 / 下一步质量线）。
- **想理解"为什么这么决策"**：从 §3 的决策点点进对应深读。
- **想看完整复盘叙事**（到 v6.5 战略转向为止）：[CASE_STUDY.md](../CASE_STUDY.md)；v6.6b→v6.10 与“哪些版本是真改进/失败实验/run 编号”的判断，先看 [深读 0](../deep_dives/00_interview_synthesis.md)，再回本表和深读 4/5/8/9。

> 注：CASE_STUDY.md 的叙事时间线收在 v6.5 的战略转向（2026-06-08）；其后的 rich-SFT 弧（v6.7→v6.9c）与 RL v6.10 由本表、[RLHF_WORKFLOW.md](../RLHF_WORKFLOW.md) Stage 7–8 和深读 4/5 记录。
