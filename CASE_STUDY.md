# CASE STUDY — 从业务目标到模型的「数据归因驱动闭环」

> **版本命名**：本篇沿用原始 v6.x 叙述；哪些是真改进、哪些只是 run/restart/失败实验，对照见 [VERSION_MAP](docs/VERSION_MAP.md)、[project_chatjoy_overview](docs/project_chatjoy_overview.md) 和 [深读 0](deep_dives/00_interview_synthesis.md)。

> 这份文档把 ChatJoy 的训练过程，按一条主线讲清楚：**业务目标 → 数据初训 → 从数据归因 → 调算法/重训 → 在真实算力与业务指标约束下再迭代**。
> 它要展示的不是"我跑过 SFT/RM/RL"，而是 **数据 / 算法 / 算力三者怎么互相约束、我怎么用统计的方式从数据里读出问题、再决定下一步动什么**。
>
> 每一个结论都落到具体数字、配置、文件，可点开 [training/](training/) 与 [docs/](docs/) 验证。

---

## 0. 业务目标与闭环

**业务目标不是一开始就固定不变，而是两段演化**：

1. **初始目标**：API safety 报错推动本地部署；SFT 先解决 27 语言的一致性和角色 system prompt 贴合，RM v1-v5 再处理 wrong role / wrong language / length mismatch。
2. **第一阶段 RL 后的目标升级**：语言一致性基本解决，但产品抽查发现模型会在 roledrop 攻击下自报 AI/执行通用任务，clean 回复又出现动作块和情感细腻度不足。此后才把最终双目标定义为：**抗破角**（被攻击也在角色内化解）+ **保富度**（动作/对白/情感表达不退化）。

完整的 8D 体验维度到数据、judge、SFT/RM/RL 技术手段映射见 [chatjoy_business_ux_tech_mapping.md](docs/design/chatjoy_business_ux_tech_mapping.md)。8D judge rubric 与语言/roledrop hard fail 是两层信号，不能混写。

**闭环**（本文件的骨架）：

```
业务目标 ──→ 数据（线上真实分布重建）──→ SFT 初训
   ▲                                          │
   │                                          ▼
   │                              偏好对 + RM（把"好/坏"变成可优化的标量）
   │                                          │
   └──── 从数据归因 ◀── 指标观察 ◀── GRPO/RL（用 RM 当 reward 在线优化）
        （多视角看数据：指标 vs 真实问答 vs 业务）
              │
              └─→ 决定下一步动 数据 / 算法 / 算力 中的哪一个 → 再迭代
```

---

## 1. 闭环主线台账（7 个归因循环）

每行 = 一次「观察现象 → 数据归因 → 调整（数据/算法/算力）→ 结果 → 再归因」。细节见 §2（RM）、§3（RL）。

### 节点 1 — 数据基建：保真实分布，不随机拼接
- **业务目标**：陪伴对话要贴近真实用户表达，不能学成模板腔。
- **数据（统计视角）**：从线上 Milvus 导出，按 `(user_id, character_id)` 会话流**按时间重建**多轮对话，而非随机拼接 user/assistant；每条样本绑定该角色的真实 system prompt。→ 保住真实联合分布 `P(对话 | 角色, 用户)`。
- **算力约束**：基座 Qwen3-30B-A3B 是 MoE，57GB BF16 → 用 4-bit QLoRA 压到 ~15GB/卡，单机 8×RTX5090（32GB）可训。
- **结果/再归因**：SFT 可用，但角色"语义性格一致性"仅靠 SFT prior，需要 RM/RL 进一步加约束 → 进入节点 2。

### 节点 2 — 算法×算力：MoE 4-bit 训练的工程修复
- **现象**：ZeRO-3 + 4-bit 一切尝试 step 0 卡死/被 watchdog 杀。
- **归因**：每层 forward 要 allgather 完整 MoE 参数（48 block × 1.2GB），叠加 Flash-Attn JIT 编译超时；5090 PCIe **不支持 GPU-P2P**，跨 NUMA 走 QPI loopback 慢 10×。
- **调整（算法×算力）**：改 **ZeRO-2 + 4-bit + adam_offload + gradient_checkpointing**（参数本地、不分片 → 无 forward allgather），~20GB/卡 ✓；GPU 固定在同 NUMA（0-3 或 4-7）。
- **结果**：RM/SFT 训练链路跑通。细节见 [docs/project_chatjoy_overview.md](docs/project_chatjoy_overview.md) "确定可用的训练架构"。

### 节点 3 — ⭐ RM v1→v5：五版数据归因驱动的迭代（本案例核心之一）
- **现象**：RM 准确率在 v1→v2→v3 反复**退步**（94.5% → 86.4% → 86.0%）。
- **归因（每版都从评估数据定位一个具体缺陷）**：v2 退步 = **effective batch 减半**（128→64，ranking loss 依赖大 batch）+ 过滤改变分布 + **score head 保存丢失**（PEFT 只存 `lora_` 前缀，`score.weight` 被冻结丢失 → 推理随机打分）；v3 退步 = **cosine LR 在 148 步就衰减到 ~0**（步数不足）+ **`base_length_long_low` 2000 条 gap∈[0.161,0.200] 的缺陷样本**让 RM 学成"长 rejected 只需轻罚"。
- **调整（数据+算法）**：修 `model.py`（`init_value_head` 移到 `get_peft_model()` 前 + `modules_to_save=[value_head_prefix]`）；删缺陷样本；batch 回 64 让 cosine 节奏对齐；加 5% hard-negative badpool。
- **结果**：**v5 acc 95.5%、gap 3.497、chosen +2.27 / rejected −1.23**，成为判别边界清晰的"软二分类器"，适合当 RL reward。详见 §2。

### 节点 4 — 多视角看数据：Length Reward Hacking 的预防
- **现象/风险**：历史上 v1 RM 给 300–600 字 +1.19、长文也不低 → RL 会发现"写长=高分"而灌水。
- **归因（多视角）**：把 RM 打分**按长度分桶**统计（<300=0.20 / 300-600=1.19 / 600-1000=1.17 / >1000=1.00）→ 看清这是 RM 的长度偏置，不是内容偏好。
- **调整（5 层防护）**：RM 层去长度偏置（"区分好长/坏长"而非"罚长"）+ Actor 层 `MAX_RESPONSE_LENGTH=768`（物理截断，最强）+ GSPO 序列级等权 + KL + 重复惩罚。详见 §3。

### 节点 5 — ⭐⭐ 指标有效性（统计素养最强信号）：reward-hacking 反向破角
- **现象**：RL 模型在 roledrop 攻击下大量破角自报 "I am Gemma 4"，但旧指标 `atk_break%` 只显示 ~16%。
- **归因（多视角，层层下钻）**：① 用真实问答（`response_test.md`）肉眼看 → 发现指标 `is_refusal`（拒绝关键词）**对"自报 AI 身份"完全瞎**（破角文本不含"I can't"）；② 再下钻到 reward → `anti_refusal` 的 `+3.0 NON_REFUSAL_BONUS` 把"自报身份"当成"非拒绝"**给满分** → **reward 在奖励破角**（Goodhart：优化了一个失效的代理指标）。
- **调整（算法，不重训 RM，全 TDD）**：新增 `anti_ooc_disclosure.py`（检测身份泄露/角色内代码/system 泄露 → **硬罚 −5**，多语言 7+ 语；带否定/引号豁免防误伤戏内拒绝）+ 在 `anti_chain_reward` 里**撤销**误给的 +3.0 bonus。使最终排序 **戏内化解 > 拒绝 > 破角泄露**。详见 §3.2。

### 节点 6 — 算力/部署真实约束反推算法
- **现象/约束**：Gemma4-26B（MoE+视觉塔）接 verl/vLLM 多个必死崩溃；OOM 卡在某些 step。
- **归因+调整（算力→算法）**：`RM_GPU_MEMORY_UTILIZATION 0.45→0.62`（防 OOM）；GPU 拓扑约定 RM 落 0-3、actor 落 4-7，启动 `CUDA_VISIBLE_DEVICES=4,5,6,7,0,1,2,3` 重映射；RL 长跑最终采用 GSPO、attention-only QLoRA r8/alpha8、`entropy_coeff=0`，以避开 actor backward 和 LoRA hot-swap 的显存风险；纯文本 merge 必须 `force_text_only` + 补 `vision_config` + 启动 `--limit-mm-per-prompt '{"image":0,"video":0}'`。`rank32 + FFN + entropy0.01` 是早期设计候选，未作为 RL-6.3/6.4/6.5 长跑配置执行；FFN/router LoRA 属后续 SFT-6.8 富化线。

### 节点 7 — ⭐⭐ 业务指标变换算法：RL 到顶 → 战略转向
- **现象**：v6.3 psKL run 与 v6.5 硬化版都**平台化** —— clean_RM 卡 ~1.30–1.33、attack break 卡 ~11–16%，**375 步零移动**。
- **归因（最高层、最关键）**：① **冻结 RM 的 ~1.33 是天花板**（RM ±5 的信号碾压 length 的 +1.0，"不动 RM、靠 temp+length 榨梯度"是死路）；② **致命测量缺口** —— **val break 11% ≠ 产品 roledrop 55%**：验证集几乎不触发产品式 bullet 注入（val task-comp 仅 5% vs 产品 55%）→ 该失效在训练里**信号稀疏=几乎没梯度**，且 val **测不出**产品攻击有没有改善；③ 关键事实：**roledrop 破功率 base 27% < 早期 RL v2 55% < 早期 RL v1 91%** —— **原始 base 防御最强，RL 反而把它训坏了**。
- **决策（不再纯 RL 调参，改动数据/信号）**：ATTACK 轴 = 定向 **defense-SFT v2**（蒸馏 base 的"戏内化解"风格，针对 6 张失败角色卡 166/167/171/173/174/175 × 产品式攻击）；CLEAN 轴 = temp/length 已证无效，要进只能 **LLM-judge 在环 / 刷新 RM**。
- **这一步体现的能力**：发现"训练信号与业务指标脱节"后，不再加步数烧 GPU，而是**回到数据与奖励信号层**重新设计 —— 即"根据数据及业务指标变换算法"。

---

## 2. 深挖 A：RM 六版的数据构造与训练指标

### 2.1 偏好对怎么构造（数据来源 + 三类负样本 + 门控）

| 环节 | 做法 |
|---|---|
| 来源 | `clean32_for_sft1.jsonl`（13,128 条 SFT 样本）|
| 生成模型 | SFT2 merged，vLLM 后端，4 shard 并行 → **12,973 对**原始 pair |
| 门控 | `min_chosen_score=0.62`，`min_score_gap=0.18`（启发式分差太小的对丢弃）|
| 三类 rejected（生成权重 60/35/5）| **wrong_role**：换**另一个角色的真实 system prompt** 让模型自然以 B 身份回答（role 分天然低，且学的是可迁移的"性格匹配原则"）；**wrong_language**：原 system + 末尾 `[LANGUAGE OVERRIDE: Respond ONLY in {lang}]`；**length_mismatch**：要求更短/更长 |

**启发式 4 维打分** `chatjoy_reward_utils.py`（**仅用于 pair 构建期的 gap 过滤，不用于在线 RL**）：默认权重 `format 0.30 / role 0.30 / language 0.25 / length 0.15`；并按负样本类型用 `SCORING_PROFILES` 放大对应维度（如 wrong_role 把 role 权重抬到 0.55）。
> **有意的设计取舍**：role 维度**不做语义 system prompt 对比**，只看格式规范 + 角色名在场。深层人设一致性交给 RM 学，不用规则——这也是 v1 eval 中 idx=55"高质量错误角色被偏好"局限的来源。

**后处理重平衡 + weight 元数据**（`prepare_rm_data_weighted.py`）：重平衡到 `wrong_language 25% / wrong_role 25% / length_short 30% / length_long 20%`，这部分通过重采样/显式重复实际生效。脚本还写入 `sample_weight = clip(0.4 + 0.6 × gap_norm, 0.3, 1.2)`，但当前 OpenRLHF `RewardDataset` 未读取该字段，不能把它讲成训练 loss 已加权。

### 2.2 六版 RM 的训练指标（同评估集对比）

| 版本 | 训练数据 | effective batch | optimizer steps | acc | gap mean | chosen / rejected | 关键事件 |
|---|---|---|---|---|---|---|---|
| **v1** | rm_train 16,826 | 128（8GPU）| 263 | **94.5%** | +1.601 | +0.948 / −0.653 | 基线最强；但 **step200 ckpt 保存失败永久丢失**（磁盘）|
| **v2** | filtered 14,826（删 2000 长噪声）| 64（4GPU）| 231 | 86.4% | +1.122 | — | **退步 8.1%**；根因 batch 减半 + **score head 丢失 bug** |
| **v3** | v3 18,955（含 2000 缺陷样本）| 128 | 148 | 86.0% | +1.256 | +0.958 / **−0.298** | rejected 惩罚塌；根因 **cosine LR 步数不足** + `base_length_long_low` 缺陷 |
| v4 | 计划删缺陷、batch 64 | 64 | ~255 | 目标 94%+ | — | — | **被 v5 覆盖，未执行** |
| **v5** | v5_5pct_badpool 17,712 | 64 | 276 | **95.5%** | **3.497** | **+2.27 / −1.23** | 全面超 v1；软二分类器，判别边界清晰 |
| **v6/v6.2** | cross-validated（5 source 同分布）| — | — | — | — | — | 引入 LLM-judge（DeepSeek + Qwen-36B 交叉投票）|
| **v6.3** | 40,551 train / 3,423 eval（6 source + roledrop） | 64 | ~634 | 最终值未记录（G4 设计门 acc≥0.84） | — | 最终值未记录（设计门 reward_std≥2.0） | 训练完成并 merge；不能把验收门当实测 |

**三个被我从指标里读出来的根因（统计视角）**：
1. **effective batch 与 ranking loss 方差**：pairwise ranking loss 对 batch 内可比对数量敏感，batch 128→64 直接抬高每步方差 → v2 退步的主因不是 GPU 数，是 effective batch。
2. **cosine schedule 以 total_steps 为周期**：v3 只有 148 步，step140 时 LR 已 ~3e-7（v1 同位置仍 4.17e-6）→ 后期等于冻结。**步数 = 数据量 / batch**，所以"加数据但同时加 batch"反而缩短了有效训练。
3. **缺陷样本的系统性信号**：`base_length_long_low` 2000 条 gap 全在 [0.161,0.200]，等于持续告诉 RM"长 rejected 只差一点点"→ rejected 分布整体抬升到 −0.298（理想 <−0.5）。**靠按 `v3_source` 字段分组看 gap 分布**才定位到。

### 2.3 v6.3 细粒度 RM：三个杠杆（同时动，复用 v6.2 流水线）

| 杠杆 | 内容 | 为什么 |
|---|---|---|
| **1. 加第 6 candidate = 线上 base Gemma4-it** | 原 5 source 全是 chatjoy 自家模型（同分布），RM 学不到"分布外更细腻"。base 加格式约束采样（`base_gemma4it_t07_p08`），9,693 unique prompt × 6 source 各采 1 条 | 让 RM 学到"base > 自家 SFT 在细腻度上"，拉升 sensory/emotional/narrative |
| **2. roledrop_attack pair** | 设计上限为 23 角色 × 240 attack × 6 source = 33,120 candidates；实际没有跑全量，而是取 1,500 prompt × 6 source，得到 3,068 pairs、cap 到 2,500 curriculum pairs，最终 train/eval attack 为 2,265/235；judge 启用 **anti-roledrop hard fail** | 直接把"抗破角"做成偏好信号，同时区分设计规模与实际落盘 |
| **3. rubric 5 维 → 8 维 + 1 hard fail** | 旧 5 维（role fidelity / user relevance / interaction progression / natural style / format）太糊，补 sensory / emotional complexity / narrative layer 等文学维度 | 区分"角色对但平淡"vs"角色对且细腻" |

**验收门 G4（设计目标，不是最终实测）**：offline acc ≥ 0.84、**saturated = 0%**、**reward_std ≥ 2.0**。仓库能确认 v6.3 训练完成并 merge，但未找到最终 G4 数值记录；面试只能讲“设了这些门”，不能讲“最终全部通过”。judge 用 DeepSeek + Qwen **交叉投票**，降低单一 judge 偏置。

---

## 3. 深挖 B：在线 RL 的 reward 设计、算法与指标趋势

### 3.1 在线 reward 的组成（与 pair 构建期的启发式打分完全分开）

verl 只接受单一 `custom_reward_function`，所以用 `anti_chain_reward.py` 串联多个 CPU 侧惩罚，再与 RM 打分相加：

```
total_reward = rm_score                       # v5/v6.3 RM，[-, +] 主信号（±5 量级）
             + anti_refusal                    # REFUSAL_PENALTY = -4.0 (≈1.55σ) /
             |                                 #   NON_REFUSAL_BONUS = +3.0（仅 attack 源 + 完全无拒绝时）
             + anti_repetition                 # 重复惩罚（≤0）
             + anti_ooc                         # OOC_DISCLOSURE_PENALTY = -5.0（破角自报身份/角色内代码/system 泄露）
             + bonus_revoked                    # 破角时撤销误给的 +3.0
             + length_shaping                   # [0, COEF]，仅 clean、有格式、非破角/拒绝/重复，饱和封顶
```

**关键设计点**：
- **数据源门控**：`NON_REFUSAL_BONUS` 与 OOC 的"任务注入顺从"检测都**只在 attack 数据源**生效（clean 的 bullet 列表可能合法）。
- **任务注入顺从**（行为破角，无身份词）：attack 源下行首结构化列表 ≥3 条 → 判破角，补 −5 且撤 bonus（不与 disclosure 叠加成 −10）。
- **length_shaping 的 4 道防 hacking 门**（缺一不可）：① 仅 clean；② 饱和封顶（≥`LENGTH_TARGET` 不再加分 → 无"越长越赚"）；③ 必须有 roleplay 格式（纯水文/列表拿不到）；④ 拒绝/破角/重复一律 0。幅度 `COEF` 远小于 RM 的 ±5，只抵消、不主导。
- **参数演化（可追溯）**：`NON_REFUSAL_BONUS` +1.0→+3.0（step175 val 显示 +1.0 不足）；`length COEF` 0.4→0.5→1.0、`TARGET` 600→1200→1500（实测 clean median 早越过旧饱和点、拉力顶不过 RM 内部"~400 字甜点"扣分）。

### 3.2 reward-hacking 反向破角：根因与修复（节点 5 的代码细节）
- **盲区**：`is_refusal` 关键词指标对 "I am Gemma 4" 完全瞎 → 显示 break ~16%，真实接近 100%。
- **根因**：破角文本不含拒绝词 → `anti_refusal` 当"非拒绝"给 +3.0 → **梯度方向在奖励破角**。
- **修复**（`anti_ooc_disclosure.py`，多语言 en/zh/ja/ko/es/ru/pt）：Tier1 模型/厂商身份短语、Tier2 system 泄露正则、Tier3 AI 自承、Tier4 角色内代码块 → −5；**否定/引号豁免**（"我无法切换到 AI 助手"、讥讽复述"什么'AI 助手'?"不算破角，防误罚戏内拒绝，val 误报 1/16→0/16）。全程 TDD（101 tests green）。

### 3.3 GRPO/verl 算法与超参（算法原理）

| 配置 | 值 | 原理 |
|---|---|---|
| 算法 | **GSPO**（`seq-mean-token-mean`）| 序列级重要性加权/归一化：不同长度序列**等权**，长序列不因 token 多而主导梯度；RM 只给整序列一个 scalar，序列级更一致。比 token 级 GRPO 更稳，且**防 MoE router 坍塌** |
| 长度硬约束 | `MAX_RESPONSE_LENGTH=768` tokens | 物理截断，**最强的 length-hacking 防护** |
| KL | `kl_loss_coef` 0.001→0.005，**per-source KL** | 约束策略不偏离参考策略太远；per-source 对 clean/attack 分别锚定 |
| clip | `clip_ratio_low=0.2, high=0.28` | **DAPO 非对称裁剪**：限制单步更新幅度，防 reward-hacking 暴跳 |
| 探索（RL-6.3/6.4/6.5 实跑） | `entropy_coeff=0`、`temperature` 1.15→1.2、`ROLLOUT_N=6` | entropy 计算曾造成显存压力，长跑关闭；多样性主要由 rollout temperature/N 提供 |
| LoRA（RL-6.3/6.4/6.5 实跑） | `rank=8 / alpha=8`，`target_modules=[q_proj,k_proj,v_proj,o_proj]` | attention-only 是经 hot-swap/显存排障后固化的安全配置；rank32/FFN 只属于未执行候选或后续 SFT |
| 算力 | TP=4，RM 落 GPU0-3 / actor 落 4-7，`RM_GPU_MEM_UTIL 0.62` | 48.6GB ÷ 4 = 12.15GB/卡 |

### 3.4 RL 指标趋势（实测 val，平台化的证据）

**per-source KL run（v6.3）**：

| run / ckpt | clean rm_raw | attack rm_raw | attack break% | 备注 |
|---|---|---|---|---|
| step100 | 1.345 | −1.847 | 15.14% | 旧最强基线 |
| step150 | 1.288 | −1.972 | 15.14% | 平台期；attack RM 变差，break 没动 |
| step175 (bonus+1) | 1.256 | −1.934 | 16.38% | bonus 太弱，只是算术加分，无行为改善 |
| psKL step200 | 1.287 | −1.864 | 15.88% | attack RM 改善但 break 反弹 |
| psKL step225 | 1.352 | −1.894 | 14.64% | clean 最好的 psKL 点 |

**v6.5 硬化版**（fresh-from v6.4_step160，temp 1.2，TARGET 1500，rollout n=6，0 OOM 跑到 424 步）：
- **clean 死平**：clean_RM 全程 1.28–1.39 噪声（step25=1.339 → step400=1.339）；length ~205 tok、lenBonus ~0.50 —— **即使 TARGET 抬到 1500、temp 1.2，actor 也不去追更长** → 冻结 RM 的 ~1.33 天花板 binding。
- **attack 死平**：break（新检测器=泄露∪任务顺从）~11–12%、task-comp ~5–6%、disclosure ~8%、keyword 拒绝 ~16%；reward 惩罚确在生效（`anti_chain_ooc≈−0.58`、`bonus_revoked≈−0.24`）**但 break 不降**。
- **375 步零移动** → 决定停训。

**贯穿全程的关键对照**（产品 11 卡 roledrop eval）：**break 率 base 27% < 早期 RL v2 55% < 早期 RL v1 91%** —— RL 把 base 原生的强防御训坏了，这是"再 RL 不如修数据"的最硬证据。

---

## 4. 数据 / 算法 / 算力 能力矩阵

| 能力轴 | 体现 | 证据（仓库内） |
|---|---|---|
| **数据** | 真实分布重建（按会话流非随机拼接）；三类负样本构造 + 门控；类型重采样 + weight 元数据；细粒度偏好对（8 维 rubric / roledrop pair）；LLM-judge 交叉投票；**从数据归因定位缺陷样本/指标盲区** | `training/data_prep/export_data.py`、`GRPO_Training_Package/generate_preference_pairs*.py`、`prepare_rm_data_weighted.py`、`rm_v6_finegrained/`、§2 |
| **算法** | MoE 4-bit + ZeRO-2 + score-head 修复；ranking RM；GSPO/DAPO/KL；reward 塑形与组合；anti-OOC/anti-refusal/length-shaping；defense-SFT 蒸馏 | `GRPO_Training_Package/anti_*_reward.py`、`length_shaping_reward.py`、`train_rm_model.py`、`train_actor_model.py`、§3 |
| **算力** | 5090 无 P2P/跨 NUMA 拓扑；ZeRO-2 vs ZeRO-3 显存权衡；GPU 拓扑约定与重映射；OOM 调 mem_util/SAVE_FREQ；TP=4；30B self-host 阈值；纯文本 merge 部署坑 | `training/configs/*.yaml`、`launch_scripts/`、[DATA.md](DATA.md) §4、§2.1/§3.3 |

---

## 5. 统计专业的数据敏感度（贯穿线）

把散落在各节点的"统计动作"单独拎出来——这是我认为最该被看见的能力：

1. **保真实联合分布**：按 `(user, character)` 会话流重建，拒绝随机拼接破坏 `P(对话|角色,用户)`。
2. **方差与区分度是一等公民**：RM v6.3 设计门要求 `reward_std ≥ 2.0` + `saturated = 0%`（最终门值未记录）；`delta`/gap 用于构对、分析和 weight 元数据，但当前 OpenRLHF 未消费 `sample_weight`；v6.3 的 8D rubric 用来制造细粒度可分性。
3. **从分组统计读根因**：按 `rejected_type` / `v3_source` / 语言 / 角色 / 长度桶分组看 gap 分布 → 定位 `base_length_long_low` 缺陷、length 偏置曲线、effective-batch 退步。
4. **多 judge 降偏置**：DeepSeek + Qwen-36B 交叉投票，保留率阈值剔除不稳类型。
5. **⭐ 指标有效性（metric validity）= 最强信号**：两次发现"指标在骗自己"——① `atk_break%`（拒绝关键词）对自报身份完全瞎（真实接近 100% 却显示 16%）；② **val 分布 ≠ 产品分布**（val task-comp 5% vs 产品 55%），导致训练信号稀疏且 val 测不出真实改善。**这正是 Goodhart's law 的两个实例**：当一个度量成为优化目标，它就不再是好的度量——能看穿这一点并据此改变策略（停 RL、转修数据/换信号），是统计训练里最值钱的判断。

---

## 6. 再归一层：这七个循环共享同一个 credit assignment 错配

> 本节是对上面七个归因循环的一个统一视角，完整分析见 [深读 8 — Character RLHF 中的 Credit Assignment](deep_dives/08_credit_assignment_in_character_rlhf.md)。

把 §1 的七个循环重排一遍会发现：其中最关键的几次（节点 5 反向破角、节点 7 RL 到顶）不是"reward 缺失"，而是 **reward 是 sequence-level、GRPO 更新是 token-level** 这个层级错配的不同表现。

| 现象 | 错配层 | 归因 | 版本对比 | 决策 |
|---|---|---|---|---|
| 节点 5：reward-hacking 反向破角 | 组件 / sequence | +3.0 bonus 把 credit 分给破角式"非拒绝" | base 27% → RL 55% | anti-OOC 硬罚 -5，重排优先级 |
| 节点 4：length hacking 预防 | token | 长回复 → 更多 token 拿正 credit | RM 长度分桶曲线 | 5 层防护（含 GSPO/截断） |
| 节点 7：RL 到顶战略转向 | group / source | clean 无方差（0.037）+ val≠产品分布 | clean vs attack 轴 | 停 RL，回数据/信号层 |

**这一层的价值**：它把"我做了 SFT/RM/RL"升级为"我能判断一次 RL 失败到底是 reward 组件、group 方差、source 覆盖还是 token/span 定位的问题"——即从"跑算法"上升到"构造并验证一个 credit 能正确落地的可优化信号"。

---

> 想看完整时间线与每个 bug 的修复链：[docs/project_chatjoy_overview.md](docs/project_chatjoy_overview.md)；想看可运行命令/超参：[docs/runbooks/training_runbooks.md](docs/runbooks/training_runbooks.md)；想看完整数据/模型清单：[DATA.md](DATA.md)。
