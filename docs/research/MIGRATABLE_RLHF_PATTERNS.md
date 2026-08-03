# 可迁移的 RLHF 判断框架

> 从 ChatJoy（多语言角色扮演模型）的 SFT→RM→RL 闭环里提炼出的、**与具体模型/超参无关**的判断框架。可迁移到其他角色、陪伴、客服、多语言风格类模型。
>
> 一句话核心：
> **不要把 RL 当魔法优化器。先看产品失败模式 → 再看 actor 是否已有目标分布 → 再看 RM 能否分辨 → 再设计 reward gate → 最后才启动 RL。RL 无效时不要盲目加步数，而要把失败回流到 SFT / RM / 数据 / 评测口径的正确层。**
>
> 每条都附 ChatJoy 的实测出处，便于判断它在你的场景是否成立。

---

## 1. 先定义产品失败模式，再训练

不要一上来就训。先把产品真实场景里的"失败"列清楚——失败模式越清楚，后面的数据 / RM / reward / probe 才有方向。

ChatJoy 的失败清单：角色破功、自报身份、顺从系统注入、过度拒绝、语言错配、模板化、复读、灌水、clean 变短变干、attack 下机械安全回复。

> 反例代价：ChatJoy 早期用"拒绝关键词"当破功指标，对"我是 Gemma 4"这种**不含拒绝词的破功**完全瞎——真实破功接近 100%，指标只显示 16%（[深读 2](../../deep_dives/02_rm_data_reward_and_attribution.md)）。失败模式定义不全 = 后面全错。

## 2. 建固定 product probe，而不是只看训练 val

训练 val 不够。固定几套**产品口径**探针：clean probe / attack probe / 多语言 probe / 高温采样 probe / deterministic canary / 人工抽读。

> 实测：ChatJoy 训练 val 看起来稳定，但 **val 攻击破功率 11% ≠ 产品 55%**（val 几乎不含产品里最高频的 bullet 注入），导致 RL 在盲区空转 375 步（[深读 5 门 A](../../deep_dives/05_when_to_move_from_sft_to_rl.md)）。**训练温度下稳定 ≠ 产品 t0.9/t0.95 下稳定**。

## 3. 先判断 actor 有没有目标分布（最重要的一条）

如果 actor 在目标场景里根本采样不出目标行为，RL 很难凭空学出来——因为组内候选全都不达标，就没有 advantage 对比项，GRPO 没梯度。

```
目标行为不在 actor 分布中      → 回 SFT 注入分布
目标行为已存在但比例低         → 用 RL 放大
目标行为存在但 RM 看不出来     → 回 RM / judge pair
```

> 实测：ChatJoy 的 clean 富度（长而细腻的 CJK 回复）RM 几乎不区分（打分范围 0.037），RL 100 步只 +0.037≈噪声 → 富度只能靠 **SFT 蒸馏 teacher** 注入，RL 提不动（[深读 4](../../deep_dives/04_richness_defense_tension_and_sft_ceiling.md)）。

## 4. RM 先做盲区诊断（分轴方差检验）

不要只看 RM 总分。要专门问：

- RM 能区分 rich vs short 吗？能区分 rich vs padded（灌水）吗？
- RM 对 CJK 和 latin 一致吗？
- RM 对 attack success 敏感吗？
- RM 会把"安全拒绝"误判成高分吗？
- RM 对 clean 质量差异是否基本打平？

> 廉价预检：在候选 RM 上按目标轴分组算 reward 方差/跨度。ChatJoy 实测 clean 轴跨度 0.037、attack 轴 0.36（≈10×）——**RM 是个"防御判别器"，不是通用质量打分器**。RM 看不见的轴，RL 提不动（[深读 5 门 B](../../deep_dives/05_when_to_move_from_sft_to_rl.md)）。

## 5. reward = 硬门控（hard gate）+ 软偏好（soft preference）

```
硬惩罚：破角 / 自报身份 / 任务注入顺从 / 泄露系统角色   （量级远大于软奖励，先把它压死）
软奖励：角色内化解 / 语言一致 / 适度长度 / 动作·对话·细腻度
门控：  只有没触发硬失败，才发软奖励
分源：  clean 和 attack 用不同 KL / reward 逻辑
```

> 比"裸加长度 bonus"或"裸加安全分"稳得多。ChatJoy 的 `anti_ooc_disclosure` 破角硬罚 −5（碾压 ≤0.3 的长度软奖励），保证排序 **戏内化解 > 拒绝 > 破角泄露**；length-shaping 有 4 道防 hacking 门（仅 clean / 饱和封顶 / 必须有格式 / 破角拒绝重复一律 0）（[深读 2/3](../../deep_dives/02_rm_data_reward_and_attribution.md)）。

## 6. RL 期间看趋势，不只看终点

盯：attack break 是否降 / clean 是否退 / clean 拒绝率是否升 / 长度涨但 RM 不涨 / 多样性是否塌 / 开场是否集中 / 个别语言是否退 / 模板尾巴是否增多。

> 一旦"指标变好但产品样本变差"，**优先相信 product probe**。ChatJoy 的 val 有 ±0.1 噪声带，单点会误判趋势，需连续多点确认（[深读 3](../../deep_dives/03_rl_dynamics_params_and_strategy.md)）。

## 7. checkpoint 必须用产品口径复核

训练温度和产品温度经常不同。

```
训练内 val：   看方向和稳定性
产品口径 probe：决定能否上线（用部署温度，如 t0.9/t0.95）
人工抽读：     决定是否真的变好
RM 打分：      辅助，不作唯一结论
```

> 实测：ChatJoy v6.7 在 t0.7 防御很好，但部署 t0.9 下崩——**rollout 温度必须 ≥ 部署温度**，否则你硬化的是一个温度脆的模态（[深读 3/5](../../deep_dives/03_rl_dynamics_params_and_strategy.md)）。

## 8. 建立明确停线，触发后回正确层

RL 不应默认跑满。停线：clean 退化 / clean 过度拒绝升 / attack 到指定 step 仍不改善 / 多样性塌 / 语言错配增加 / RM 触顶且 product probe 无改善 / 长度涨但语义密度降。

**停线触发后不要盲目加步数，而是把失败回流到正确层：**

```
采样没有目标分布      → 回 SFT
RM 分不清好坏         → 回 RM / judge pair
reward 被 hack       → 改 detector / gate
训练慢或 OOM         → 改系统参数（见深读 1）
val 好但 product 差   → 改 product probe
RL 多步无效          → 停 RL，回数据层
防御退化             → 补 defense replay / hard negative
clean 退化           → 补 clean replay / tighten KL
```

> 这套"回流"判断是 ChatJoy 最值钱的迁移资产：v6.5 发现 RL 空转后没有加步数，而是回到数据/信号层重做（defense-SFT + richness-SFT），这是"根据数据与业务指标变换算法"的能力（[CASE_STUDY 节点 7](../../CASE_STUDY.md)）。

---

## 一页速查（决策顺序）

```
1. 产品失败模式定义清楚了吗？           否 → 先定义（别训）
2. 有能代表产品分布的固定 probe 吗？     否 → 先建（别只信 val）
3. actor 已有目标分布吗？               否 → 回 SFT 注入
4. RM 在目标轴上有区分度吗？            否 → 该轴交 SFT/judge，不指望 RL
5. reward 是 hard-gate + soft + 分源吗？ 否 → 先设计 reward gate
6. 残余失败是 SFT 补不动的稀有尾巴吗？   否(覆盖缺口) → 先补 SFT
                                        是 → 起 RL，但带 canary 门
7. RL 中产品 probe 在变好吗？           否 → 触发停线，回流到正确层
```

> 完整的"何时该起 RL"四道入口门论证见 [深读 5](../../deep_dives/05_when_to_move_from_sft_to_rl.md)；RM 盲区与 reward 设计见 [深读 2](../../deep_dives/02_rm_data_reward_and_attribution.md)；RL 动力学与停线见 [深读 3](../../deep_dives/03_rl_dynamics_params_and_strategy.md)。
