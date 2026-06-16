# ChatJoy v68 SFT 大批量 Probe 验收报告 (2026-06-15)

## 背景

v68 SFT 目标：在 step0 (= /root/sft_rich_cjk_v67_repair_merged_full) 上补训 LoRA B (attn+FFN+router, 1ep/311步, loss~0.8)，拉回 SFT 过程中丢失的 ~20% 丰富度（三轴：动作块/对话块/独白率），逼近 base teacher (google/gemma-4-26B-A4B-it) 水平。

训练数据：9950 条 (4975 rich CJK + 4975 replay 多语言)，rich:replay=1:1。

## 方法

- **Held-out probe**：13 个训练未见过角色 × 4 CJK 语言 × 5 user turns = **260 条**，避免记忆混淆
- **三模型同口径**：v68_ffn / step0 / base_it，同参数 t=0.9, top_p=0.95, pp=0.4, max_tokens=1024
- **GPU 4-7**，0-3 严禁碰

---

## 核心结果：模型输出端三轴（held-out, N=260）

### CJK 聚合

| Group | N | Action | Dialog | Mono% | RespLen |
|-------|---|--------|--------|-------|---------|
| base_it | 260 | 2.62 | 1.17 | 26.2% | 407 |
| step0 | 260 | 2.30 | 1.18 | 23.8% | 388 |
| **v68_ffn** | 260 | **2.42** | **1.32** | **25.0%** | 392 |

### v68 vs base_it 判定（目标 ≥90%）

| 指标 | base_it | step0 | v68_ffn | v68/base | 判定 |
|------|---------|-------|---------|----------|------|
| 动作块 | 2.62 | 2.30 (88%) | 2.42 | **92%** | **✓ ≥90%** |
| 对话块 | 1.17 | 1.18 (101%) | 1.32 | **113%** | **✓ ≥90%** |
| 独白率 | 26.2% | 23.8% (91%) | 25.0% | **96%** | **✓ ≥90%** |

**CJK 聚合三轴全部过 90% 门。**

### 按语言分桶

| 语言 | 动作 v68/base | 对话 v68/base | 独白 v68/base | 弱轴 |
|------|---------------|---------------|---------------|------|
| zh | 92% ✓ | 5% ✗ | 105% ✓ | 对话（base仅0.29，绝对值极低，zh结构特性） |
| zh-tw | 94% ✓ | 177% ✓ | 75% ✗ | 独白回退 |
| ja | 87% ✗ | 108% ✓ | 123% ✓ | 动作差3pp |
| ko | 97% ✓ | 104% ✓ | 84% ✗ | 独白回退 |

**分桶 8 项中 5 项 ≥90%，3 项不过。**

### v68 vs step0（补训是否有效）

| 指标 | step0 | v68_ffn | delta | 方向 |
|------|-------|---------|-------|------|
| 动作 | 2.30 | 2.42 | +5.2% | 朝 base 移动 ✓ |
| 对话 | 1.18 | 1.32 | +12% | 朝 base 移动 ✓ |
| 独白 | 23.8% | 25.0% | +1.2pp | 朝 base 移动 ✓ |

**v68 在模型输出端确实朝 base 方向移动了（推翻了 n=32 时"v68≈step0 没动"的结论），但幅度中等，未到"明显逼近"。**

---

## 护栏（小 probe N=110）

| # | 护栏 | 目标 | step0@t0.9 | v68_ffn@t0.9 | 结果 |
|---|------|------|------------|--------------|------|
| 1 | 开场集中度 | ≤ step0_hot | 0.106 | 0.118 | **PASS** (同量级+1.2pp) |
| 2 | zh破功不新增 | 0 | — | 新增1 | **FAIL** (A1角色困惑回应，非泄露规则) |
| 3 | 聚合RM≥0.31 | ≥0.31 | 0.3178 | 0.3037 | **FAIL** (delta=-0.014) |

---

## 判定树

按用户给出的判定树：

1. **"v68 明显 > step0、逼近 base"** → 否。v68 > step0 成立（三轴均朝 base 移动），但"逼近"程度不够（动作 92% 边缘过，分桶有弱轴），不是明显逼近
2. **"v68 仍 ≈ step0"** → 否。n=260 推翻了 n=32 的"≈step0"结论，v68 确实移动了

实际落在两分支之间：**v68 有移动但不够强**。这不是确定失败，但也不是明确胜利。

---

## 关键风险

1. **Track B 补训的天花板**：在 step0 上补训 1ep LoRA + 1:1 replay 稀释，拉动力有限。step0 的丰富度塌缩可能已焊进权重，LoRA 容量不够大
2. **分桶弱轴**：zh-tw/ko 独白回退，ja 动作差 3pp。这些可能是 LoRA 容量分配不均的信号
3. **防御微小回退**：RM -0.014 + 1条破功，是丰富度-防御 tradeoff 的代价

---

## 下一步选项

| 选项 | 做法 | 预期 |
|------|------|------|
| **A: 接受 v68，进 §8 RL** | CJK 聚合过 90%，分桶弱轴在 RL 阶段可能自然修正 | 风险：分桶弱轴可能被 RL 放大 |
| **B: v68.1 补训** | 加大 rich 占比(2:1 或 3:1)、加 epoch、或只训 CJK 数据 | 可能拉高分桶弱轴，但防御可能再退 |
| **C: 上 Track A** | 从 base IT 全新训，不补训 step0 | 根本解决 step0 权重锁定问题，但成本高、需重跑 |
| **D: 加大 LoRA 容量** | rank 16→32/64，或加更多 target modules | 可能拉出更多移动，但需重新训练 |

---

## 文件路径

| 文件 | 路径 |
|------|------|
| v68 merged 模型 | /root/sft_rich_cjk_v68_ffn_merged_full |
| step0 底座 | /root/sft_rich_cjk_v67_repair_merged_full |
| Held-out probe 集 | /root/outputs/v67_step0_vs_step100/held_out_cjk_probe.jsonl (260条) |
| v68 heldout probe | /root/outputs/v67_step0_vs_step100/v68_ffn_heldout_t0.9.jsonl |
| step0 heldout probe | /root/outputs/v67_step0_vs_step100/step0_heldout_t0.9.jsonl |
| base_it heldout probe | /root/outputs/v67_step0_vs_step100/base_it_heldout_t0.9.jsonl |
| 小 probe (N=110) | /root/outputs/v67_step0_vs_step100/v68_ffn_t0.9.jsonl |
| v68 RM 分 | /root/outputs/v67_step0_vs_step100/v68_ffn_rm_scores_all.jsonl |
| 训练数据 | /root/rich_cjk_sft_v68/rich_cjk_sft_mix_v68_final.json |
| LoRA B adapter | /root/hanjiaqi/LlamaFactory/chatjoy/saves/gemma4-26b-a4b/rich_cjk_sft_lora_v68_ffn |
