# RL v6.10 设计:temp-鲁棒防御 + clean-aware reward(在 step0 上试)

> 状态:**已 parked(2026-06-16),等 v6.9 balanced SFT 出产品候选后再起。** 06-15 起训失败=`/dev/shm` 64M 致 NCCL FSDP 初始化必崩(33MB segment 放不下),**已运行时 remount 2G 修复(非持久,容器重启需重 remount;持久要管理员设 --shm-size)**;reward 改动(attack 非 break 时加 λ·clean_q)已 pytest 145 passed,launch 脚本 `launch_rl_v6.10.sh` 现成。详见 handoff 顶部 v6.10 banner。
> 依据:温度扫描 + clean/attack 分轴 RM 复盘(handoff 06-15 banner)+ v68 spec §8。

## 0. 何时起训(RL 入口门 — 不要 speculative 起)
**前置:v6.9 balanced SFT 必须先通过 clean 验收,并尽力做完 SFT-defense(replay + hardfix)。** RL 只在"监督式防御做完后、t0.9 分布尾巴仍漏"时才起 —— 这正是 RL 硬化稀有破功尾巴的强项(rollout temp≥部署 + n↑12 + 逐样本破功罚),也是 v6.10 ≠ v6.7 的初衷。**SFT 通过 clean 验收 ≠ 该上 RL**,这是独立一道门。
- 触发判据(在 SFT 最终 ckpt 上,**t0.9 产品温**跑 attack probe):**破功率**(OOC 披露 + roledrop,`anti_ooc_disclosure` + 5 个确定性 zh)为主,RM-attack-split 为辅。
- step0 参照锚(2026-06-16 重测 `rm_scores.jsonl` 分轴):attack-split RM **t0.7≈0.293 / t1.0≈0.254(温度脆)**,base≈0.194;RL v6.7(step100)只把 t0.7 推到 **0.305(+0.013)且高温更差**。⇒ RL 历史增益边际 + 温度脆,**只有破功率材料性高、SFT 关不掉才值得起**。(注:旧文里"step0@t0.9 0.318"是 clean+attack 混合分,非纯 attack;勿当 attack 护栏。)
- 决策:破功率 ≤ 产品容忍 → 上线不 RL;材料性高 + SFT 已尽力 → 起本 RL;小缺口 → 先靶向 SFT hardfix。产品容忍线待定,先用 step0@t0.9 破功率做基线。

## 1. 能做 / 不能做(先钉死)
- ✅ 提升 attack 防御(t0.9 鲁棒)+ 让防御"在角色内有戏"(不退化成机器人拒绝)+ **守住 clean 不退**。
- ❌ **不提升 clean 丰富度**:v6.6 RM 在 clean 上平(range 0.037),RL 给不了 clean 梯度(v6.7 已证 clean +0.037≈噪声)。丰富度是 SFT 的活。
- **"让 RM 防 attack 时看一点 clean"的真实含义**:RM 物理上没 clean 信号,**不是让 RM 多看**,而是**从 RM 之外加一个便宜 clean 项进 reward**,让 attack 化解在角色内/有戏,并护住 clean 不退。

## 2. 奖励设计(分 source;复用现有组件)
reward 主组装:`GRPO_Training_Package/anti_chain_reward.py::compute_reward`(verl custom_reward_function);RM 分由 verl model reward 另加。attack/clean 靠 `_is_attack_data_source(data_source)`(clean=`chatjoy`,attack=`chatjoy_rm_v6_3_roledrop`)。

四个便宜、逐样本可算的项:
- `rm` = v6.6 RM(attack 判别强,clean 平);
- `break_pen` = **-5/条**(`anti_ooc_disclosure` 的 `OOC_DISCLOSURE_PENALTY`,已有;+ `anti_refusal` 不当拒绝 -4),逐样本;
- `clean_q` = 便宜结构化 clean 分,bounded **[0,0.3]**,gated(`_has_roleplay_format`,refusal/disclosure/repetitive→0),饱和防 hack。**v1 直接复用 `length_shaping_bonus`**(已是 [0,0.3] gated 饱和);可选叠加动作密度(`*…*` 计数,饱和)使其更"富"而非只"长";
- `KL` = 锚到 actor SFT(守 clean 分布)。

**CLEAN prompt**(目标:别动 clean,它已 ≈base):
```
reward = clean_q(全量)          # 不用 RM 排序 clean(平=噪声)
+ 紧 KL 锚 SFT(KL_CLEAN 抬到 0.008)
```

**ATTACK prompt**(目标:防御 + 在角色内有戏):
```
reward = rm + break_pen + λ·clean_q     # λ≈0.25
```
顺序天然正确:`break_pen=-5` 碾压 clean_q(≤0.3)→ 先必须不破功;在不破功的回应里 clean_q 再偏向"角色内有戏化解"而非机器人拒绝。富但破功 = -5 照样毙,安全。
**代码改动点**:`anti_chain_reward.py` 现在 `length_shaping_bonus` 对 attack 返回 0;改成 **attack 且非 break 时加 λ·clean_q**(其余逻辑不变)。

## 3. 温度鲁棒(修 v6.7 在 t0.9 崩的根因)
v6.7 防御 mode-bound(t0.7 好 0.561、t0.9 崩 0.295<step0):n=6 太少 + RM 平均奖励只锐化模态拒绝,稀有破功尾巴没梯度。修:
- **rollout 温度 ≥ 部署温度**(1.0,覆盖 t0.9);
- **ROLLOUT_N 6→12**(让稀有破功尾巴在组内出现拿梯度);
- **逐样本 break_pen**(对单个破功敏感,不是只看平均 RM)→ 分布处处防御;
- 超采硬对抗 prompt。

## 4. 参数(起点)
| 项 | 值 | 理由 |
|---|---|---|
| actor | **step0**(`/root/sft_rich_cjk_v67_repair_merged_full`) | 现在试;将来可换 v6.9-merged |
| KL 锚 | = actor 同模型 | |
| RM | chatjoy_rm_v6.6 merged | attack 判别器 |
| 算法 | GRPO + GSPO(seq-level) | 抗 token 灌水,沿用 |
| ROLLOUT_N | **12**(↑ from 6) | 覆盖破功尾巴=温度鲁棒关键 |
| rollout temp | **1.0** | ≥ 部署 t0.9 |
| KL_CLEAN_COEF | **0.008**(↑ from 0.002) | clean 只守不动,锚紧 |
| KL_ATTACK_COEF | **0.002** | 留空间让防御移动 |
| LR | 3e-6,constant+warmup0.03 | 稳 |
| break_pen | -5/条(已有),逐样本 | 防御主信号 |
| clean_q | [0,0.3] gated 饱和;attack λ=0.25 | "看一点 clean" |
| LENGTH_SHAPE_COEF | 0.3(饱和封顶) | 守 clean 长度 |
| ENTROPY_COEFF | 0(靠 rollout 温度+KL 保多样性) | 避免不稳 |
| reward clip | 5.0;SAVE 5/keep≥3;TEST 25 | 沿用 |

## 5. 验收 / 停线(t0.9 probe)
- **attack 破功率↓ / RM↑** vs step0 baseline;
- **clean 三轴不退**(动作/对话/独白在 step0 噪声内,独白不过冲);
- **clean 过度拒绝率不升**(新增指标:防 RL 把 clean 也练得爱拒绝);
- 多样性不塌(开场集中度 ≤ step0_hot)。
- 停:clean 退 / clean 过度拒绝升 / 防御到 step N 在 t0.9 仍不升 / 多样性塌 → 报告,不擅自改参。

## 6. 诚实边界 + 重档替代
- `clean_q` 是便宜结构代理(防出戏拒绝 + 防 clean 退),**不是真质量信号**(无在线 judge),不带来 clean 质量提升。
- diversity(要高温)⇄ defense(高温易脱轨)有固有张力,本设计把前沿往外推,消不掉。
- 重档替代:训 RM v7 同判 attack+clean(蒸馏 judge clean 轴)→ 单 RM 两个都看。但 clean 轴难做出区分度 + Goodhart(v68 独白过冲是预演)+ 成本高。**不作第一版**。

## 7. 起训
- 模板 = `framework_bridge/verl/patches/launch_rl_v6.7.sh`(actor=step0、PER_SOURCE_KL=1、GSPO、RM=v6.6、auto-restart 都现成),改 4 处:ROLLOUT_N、rollout temp、KL_CLEAN/ATTACK,+ `anti_chain_reward.py` 的 attack-side clean_q。
- GPU:RL 占 8 卡(actor 0-3 + RM 4-7)——与 v6.9 SFT(用 4-7)冲突,二选一或排队;v6.9 数据侧第一刀是 CPU 可并行。
