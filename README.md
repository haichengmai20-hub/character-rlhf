# ChatJoy — 多语言角色扮演对话模型训练（SFT → RM → RLHF/GRPO）

> 把一个通用 MoE 基座，训练成**多语言、强角色扮演、抗破角**的陪伴对话模型。
> 完整链路：线上真实聊天数据 → SFT → 偏好对（Preference Pair）→ 奖励模型（RM）→ GRPO 强化学习。

**基座**：Qwen3-30B-A3B-Instruct（MoE，SFT/RM）· **Actor**：Gemma-4-26B-A4B-it（MoE+视觉塔，GRPO）
**框架**：LlamaFactory（SFT）+ OpenRLHF（RM）+ verl（Actor GRPO）+ vLLM（rollout/serving）

> ⚠️ 本仓库是**工程档案 + 代码 + 小样本**。GB 级训练数据与模型权重**不入库**，完整清单与路径见 [DATA.md](DATA.md)。生产地址/库名已脱敏。

---

## 🧭 方法论主线：数据归因驱动的闭环

这个项目想体现的不是"跑过 SFT/RM/RL"，而是一条闭环能力：

```
业务目标 → 数据(真实分布重建) → SFT 初训 → 偏好对+RM → GRPO/RL
   ▲                                                      │
   └──── 调 数据/算法/算力 ◀── 数据归因 ◀── 指标趋势观察 ◀──┘
        (多视角看数据：指标 vs 真实问答 vs 业务)
```

- **数据 / 算法 / 算力 三者互相约束**：算力（5090 无 P2P、MoE 显存）决定能用什么算法（ZeRO-2 + 4-bit、GSPO）；数据归因（RM 退步、指标盲区）决定下一步重训还是改信号。
- **统计视角的数据敏感度**：方差/区分度当一等公民（`reward_std≥2.0`）、分组读根因、**两次识破"指标在骗自己"**（Goodhart）→ 据此把"再 RL"改成"修数据/换信号"。

👉 **完整复盘（七个归因循环 + RM 六版数据与指标 + 在线 RL reward 与趋势）见 [CASE_STUDY.md](CASE_STUDY.md)。** 下方"核心工程亮点"是它的索引式摘要。

---

## 这是什么

ChatJoy 的目标不是通用问答，而是**多语言角色扮演陪伴对话**。整条链路围绕三件事设计：

1. **保持真实用户分布** — instruction 来自线上真实聊天记录（Milvus 导出），不是人工模板
2. **保持角色卡约束** — 每条样本绑定角色 system prompt，避免只学泛化闲聊
3. **强化多语言一致性 + 角色扮演格式 + 抗破角** — SFT 学基础表达，RM 学会区分角色/语言/长度/格式优劣，RL 用 RM + 反破角信号继续优化在线 actor

覆盖 **TOP20 角色 × 27 语言**。

---

## 全链路

```
线上真实聊天 (Milvus, 按 user/character 会话流重建)
        │  export_data.py → 清洗/过滤/平衡
        ▼
[1] SFT (LlamaFactory, QLoRA)
        │  Qwen3-30B-A3B MoE — 修复 expert 裸 nn.Parameter 4-bit 量化 + FSDP 链
        ▼
[2] Preference Pair 生成
        │  多候选采样 + LLM-judge (Gemini/Qwen 交叉) + 拒绝采样 + 重平衡/sample_weight
        ▼
[3] Reward Model (OpenRLHF)
        │  v1→v6.3 迭代：细粒度 rubric 8 维 + delta=2 难负样本 + roledrop_attack pair
        │  + base 模型作第 6 candidate + reward_std×20 + length-hacking 5 层防护
        ▼
[4] GRPO 强化学习 (verl + vLLM rollout)
        │  Actor=Gemma4-26B-A4B-it；自定义 reward = RM + anti-OOC + anti-refusal + length-shaping
        ▼
   抗破角 / 多语言一致 / 角色扮演陪伴模型
```

---

## 🔧 核心工程亮点（面试关注点）

| 主题 | 难点 | 做法 / 结论 |
|---|---|---|
| **MoE QLoRA 训练** | Qwen3/Gemma4 的 expert 是裸 `nn.Parameter`（非 `nn.Linear`），常规 4-bit 量化/PEFT 注入会崩 | 自定义 experts 4-bit 量化实现 + 4-GPU FSDP 修复链；Gemma4 merge 必须 `force_text_only` 否则视觉塔 q/k/v/o 崩 |
| **RM 细粒度设计 (v6.3)** | 粗粒度 RM 区分不出"角色对但平淡"vs"角色对且细腻" | rubric 8 维打分 + `delta=2` 难负样本 + roledrop_attack 偏好对 + base 模型当第 6 candidate + `reward_std×20` 放大区分度 |
| **Length Reward Hacking** | RL 会发现"写更长 = 更高分"而灌水 | 5 层防护：RM 训练去长度偏置 + actor 层 length-shaping（放宽 + 饱和封顶）+ 重复惩罚 + 链式惩罚 |
| **Reward Hacking 反向破角（关键发现）** | 模型在攻击下"我是 Gemma 4"破角，旧 `atk_break%` 关键词指标却显示只有 ~16% | 根因：`anti_refusal` 的 +3.0 NON_REFUSAL_BONUS 把"自报 AI 身份"判为合规给满分 → **反向训练破角**。修复（全 TDD，不重训 RM）：新增 `anti_ooc_disclosure.py`（破角硬罚 -5、撤销误给 bonus、多语言）+ `length_shaping_reward.py` |
| **Defense-SFT 蒸馏** | RL 信号稀疏修不动破角 | 用 idle 的 base 模型当 teacher，蒸馏"戏内化解"攻击的回复，三门过滤 + 拒绝采样，挂回 SFT |
| **verl GRPO 接入** | Gemma4（MoE+视觉）接 verl/vLLM rollout 多个必死崩溃 | 修 LoRA merge / score head / vLLM patch；纯文本 merge 后须补 `vision_config` + `--limit-mm-per-prompt '{"image":0,"video":0}'` |
| **评测体系** | "破角率"等指标对自报身份完全瞎 | 升级检测器 + 5 维评测 + RM breakdown + pairwise；发现 **val break 11% ≠ 产品 roledrop 55%**（验证集几乎不触发产品式 bullet 注入）|

---

## 📍 当前状态 (2026-06-08) — RL 已到顶，战略拐点

- **v6.5（硬化版 GRPO）已停训**：两轴 375 步零移动 — `clean_RM` 卡冻结 RM 的 ~1.33 天花板；attack break 平在 ~11-12% 不降。收敛形态同 v6.3。
- **致命问题**：验证集信号 ≠ 产品真实攻击（val 几乎不触发 bullet 注入 → 训练信号稀疏 + val 测不出产品攻击）。
- **下一步不是再纯 RL 调参**：ATTACK 轴 = 定向 **defense-SFT v2**（teacher 蒸馏戏内化解，针对 6 张失败角色卡）；CLEAN 轴 = temp/length 已证无效，要进只能 LLM-judge 在环 / 刷新 RM。

> 完整时间线见 [docs/project_chatjoy_overview.md](docs/project_chatjoy_overview.md) 顶部"当前状态 (2026-06-08)"。

---

## 仓库结构

```
chatjoy-character-rlhf/
├── README.md                    本文件
├── DATA.md                      ⭐ 完整数据/模型分层清单（哪些入库、哪些只记录）
│
├── docs/                        工程知识库（项目全程记录，面试主菜）
│   ├── project_chatjoy_overview.md       全链路总纲 + RM/GRPO 全记录 + 当前状态
│   ├── training_runbooks.md              各阶段可直接运行的启动命令/超参/路径
│   ├── chatjoy_business_ux_tech_mapping.md  8 维业务体验 → 技术手段映射
│   ├── chatjoy_rm_v6_3_roledrop_design.md   RM v6.3 设计（rubric 8维 / delta=2 / roledrop pair）
│   ├── chatjoy_rm_v6_3_roledrop_plan.md     v6.3 实施计划 + 70-case 验收 grid
│   ├── chatjoy_rm_v6_3_roledrop_handoff.md  v6.3→v6.4→v6.5 交接（最新）
│   ├── chatjoy_rm_v6_finegrained_handoff.md RM 细粒度交接
│   └── chatjoy_rl_takeover_error_fix_log.md RL 接手错误修复全记录
│
├── training/
│   ├── GRPO_Training_Package/   ⭐ 强化学习全套代码
│   │   ├── anti_ooc_disclosure.py        反破角 reward（关键修复）+ 测试
│   │   ├── anti_refusal_reward.py        反拒绝 reward
│   │   ├── length_shaping_reward.py      长度塑形（防 hacking）+ 测试
│   │   ├── anti_repetition_reward.py / anti_chain_reward.py  重复/链式惩罚
│   │   ├── chatjoy_reward_utils.py       4 维奖励聚合
│   │   ├── train_rm_model.py / train_actor_model.py  RM / Actor 训练入口
│   │   ├── generate_preference_pairs*.py 偏好对生成
│   │   ├── build_defense_sft_dataset.py  defense-SFT 数据构建 + 测试
│   │   ├── framework_bridge/             verl / OpenRLHF 数据桥接
│   │   └── rm_v6_finegrained/            v6 细粒度：候选生成 / LLM-judge / 投票合并 / 过滤
│   ├── configs/                YAML（accelerate / QLoRA / defense-SFT / merge）
│   ├── launch_scripts/         RM/SFT/GRPO 启动脚本（含 GPU 拓扑/NCCL）
│   ├── data_prep/              线上导出 / 数据集生成 / 过滤 / 难负样本挖掘
│   └── design/                 reward_model_design.md / README_GRPO / Gemma4 SFT 复盘
│
├── characters_samples/         角色卡示例（产品 IP，仅放 2-3 张做样例）
└── data/samples/               小样本（SFT 5 条 / test head / LlamaFactory dataset_info）
```

---

## 技术栈

- **训练**：LlamaFactory (SFT/QLoRA) · OpenRLHF (RM) · verl 0.8 (GRPO) · DeepSpeed ZeRO-3 · FSDP · PEFT/LoRA
- **推理/服务**：vLLM 0.19（rollout + RM 分类 + demo serving）
- **数据/评测**：Milvus（线上导出）· Gemini / Qwen3 作 LLM-judge（交叉投票）· pairwise + RM breakdown
- **模型**：Qwen3-30B-A3B-Instruct（MoE）· Gemma-4-26B-A4B-it（MoE + 视觉塔）

---

## 注意

- 本仓库**不含**真实用户数据、GB 级训练集、模型权重 —— 见 [DATA.md](DATA.md) 的分层说明与复现路径。
- 角色卡、产品攻击样本等业务 IP 仅放少量脱敏样例。
- 生产 Milvus 地址 / DB 名 / 内网 IP 已统一替换为 `<...>` 占位符。
