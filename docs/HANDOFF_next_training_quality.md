# 下一步训练设计:把角色扮演质量恢复到 base 天花板(quality-aware 全链路)

> 触发:2026-06-18 GPT LLM-judge 头对头(**只看角色扮演质量**)判 **base > RL(step90)**。
> 因 RL step90 的 clean 被 KL 锚定到 actor=v69c(clean prompt 只拿 length-bonus+KL),∴ **base > v69c 也成立 = 整条 rich-CJK SFT 链从未到 base 质量**。
> 关联:[[project_chatjoy_overview]] §⑦、[[chatjoy-eval-paradigm]]、[[chatjoy_rl_v610_defense_clean_aware_design]] §6。

## GPT 判的 5 维(base 全胜,= 现成 rubric)
① 角色状态递进(高傲→恐惧→被迫谈判→失控,心理链路连续) ② 场景沉浸 ③ **人设稳定**(base "傲慢大小姐被秘密压制"稳;RL 滑向"普通受害者",大小姐感弱) ④ 情绪张力 ⑤ 用户承接(对每步动作接得细腻)。

## 重新诊断:一直在测错的东西(已逐项核验)
1. **块计数是弱代理**:v69 selector 实测 = action+dialog 计数 + interaction/sensory/emotional 三个 judge 维(单条、等权 0.2),v69 因此"块计数≈base"。**但它单条、且缺 GPT 的 ③人设稳定/①跨轮递进/⑤用户承接**——这三个是**多轮**属性,选择器和 RM 从没测过 → 质量没到 base。
2. **RL 没提质量(预期内)**:RM 是 defense 判别器、clean 轴平(range 0.037),clean prompt 只优化了 length+KL锚 → 质量无梯度。
3. **defense 数据疑似直接伤人设**:v69b/c 的 2686 条跨角色 defense 教"角色内化解/deflect",可能把高傲人设压平成"普通受害者"(= GPT 说的 ③ 退化的机制)。→ **再证 defense 不该混进 quality SFT**。

## 缺失的核心 = 角色扮演质量信号(eval/选择/reward 三处都没有)
下一步主线 = 把 GPT 的 5 维质量信号**贯穿进 eval、SFT 选择、RM/reward**。

## 设计(4 角度,按杠杆/成本排序)

### Step 1【必做·最便宜·先做】eval:把 GPT 的 n=1 → 多案例 LLM-judge harness
- 固定一组**多轮**角色扮演案例(zh/zh-tw/ja/ko × 不同人设),base vs 候选成对,LLM-judge 按 5 维打分/胜负。
- 产出 = 统一质量尺子,供 SFT 选择 / RM 蒸馏 / 验收共用。**没有它后面全是盲飞**(块计数已误导过一次)。扩进 eval 范式 md。
- ⚠️ 现 GPT 判是 **n=1 案例** → 先系统化、确认 base>候选 在多案例上稳,再下重注。

### Step 2【数据+SFT·质量恢复·中成本】judge-选择的 SFT 再蒸馏 base
- 复用 best-of-N 管线(`bon_resample_v69.py`+`select_v69_balanced.py`),**把选择信号从单条 composite 换成 5 维 judge**,且**改多轮**(GPT 评的是跨轮递进+人设稳定+用户承接,单轮选不出来)。
- teacher = base(质量天花板;SFT 上限=base,目标把 v69c 拉回 ~base)。
- **关键决策:质量优先、defense 分离**——**别再往 quality SFT 里混 defense**(v69b/c 已证 whack-a-mole + 压人设)。这步接受 defense 回退到 base/step0 之间,defense 留 Step 3/4 单独加。

### Step 3【RM·解锁 RL·重档】RM v7 = 质量(5维)+ defense 双信号
- 现 RM 只判 defense、clean 平 → RL 永远提不了质量。RM v7:从 LLM-judge 5 维蒸馏**角色扮演质量判别**(pairwise:base/高分=chosen vs 低分=rejected)+ 保留 roledrop defense 判别。
- = rl_v610 §6 早标的"重档替代",现在有 GPT 的具体 rubric 了。
- 风险(已知):clean 质量轴难做区分度 + Goodhart(v68 独白过冲是预演)→ 多维 rubric(非单分)、pairwise margin、held-out judge 验证、人工抽查。

### Step 4【RL/DPO·收尾】per-source 加 defense 而不伤质量
- 有 RM v7 后,per-source RL:clean prompt 拿**质量 reward(RM v7 质量头)**+轻 KL;attack prompt 拿 defense reward+break_pen(-5)。
- 这样 RL 加 defense 时质量被 RM v7 **托住**(不像现在 RM 不看质量 → 漂向"普通受害者")。
- 更轻探路:**DPO** 直接用 base-vs-模型偏好对(base=chosen)→ 直接教"像 base",省 RM 训练,可先做一版验证偏好学习能否recover质量。

## 推荐执行顺序 + 终局
- **先 1+2**(eval harness + judge-选择多轮 SFT),测 SFT 能否把质量拉回 base。
- 若质量回来但 defense 退 → **3+4**(RM v7 质量+defense → RL/DPO 加 defense 托住质量)。
- **终局 = base 质量 + step0 defense**:靠"质量 SFT 蒸 base"+"质量-RM 托住的 per-source RL 单独加 defense"组合,**不是单模型一步到位**;别再在 SFT 里混 defense。

## 诚实边界
- base 是质量天花板但 defense 最差(真破功 base>step0 16%);所以不能直接上 base。
- SFT 蒸馏不超过 teacher(base);要超过 base 才需 RL+能区分质量的 RM(难)。
- 质量 RM/judge 易被刷(长度/独白/华丽辞藻)→ 多维+人工抽查+held-out 防 Goodhart。
- 全部建立在 Step 1 的多案例质量尺子上;n=1 不下重注。
