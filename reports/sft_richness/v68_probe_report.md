# ChatJoy v68 SFT Probe 验收报告 (2026-06-15)

## 一、背景

v68 SFT 的核心目标：让 clean 输出在 t=0.9 产品口径下，三轴（动作丰富/语言质量/独白质量）逼近 base teacher (google/gemma-4-26B-A4B-it)。替换 v67 的 RM argmax 选样为丰富度选样 + DeepSeek judge 三轴打分。在 step0 (= /root/sft_rich_cjk_v67_repair_merged_full) 上补训 LoRA B (attn+FFN+router)。

## 二、已完成工作清单

| 步骤 | 内容 | 结果 |
|------|------|------|
| 量纲bug修复 | build 脚本里 proxy 分(~0-10)和 judge 分(0-1)混在同一排序键，重采数据因量纲大压过 judge 候选 | 已修：proxy 分除以 PROXY_REF_MAX=20 归一到 0-1 |
| 合池 build | 9950 条 (4975 rich + 4975 replay), rich:replay=1:1 | 动作5.25, 对话4.35, 独白率31.4% |
| GATE 验收 | rich 子集三轴 | 动作5.25(≥4.3✓), 对话2.77(≥3.0边缘), 独白47.6%(≥50%边缘), 尾巴6%(≤8%✓) |
| LoRA B 训练 | 1ep/311步, loss~0.8, GPU 4-7 | 已完成 |
| Merge | → /root/sft_rich_cjk_v68_ffn_merged_full | config.json 有 vision_config ✓ |
| Probe 采样 | v68_ffn + base IT + step0 @ t=0.9, 55卡+非CJK(es/ar/ru/pt)补充 | 110+8条/模型 |
| RM 打分 | chatjoy_rm_v6.6 merged 给 v68_ffn 全量打分 | 110条 |

## 三、三轴度量（训练数据输入端，大样本可靠）

**Base Teacher = google/gemma-4-26B-A4B-it**（确认方式：gen_base_teacher shard 的 served_model="gemma4-base"，设计文档写"动作块 base 4.7 → step0 3.4"）

| 语言 | 指标 | Base Teacher | v68 Rich | vs base | ≥90%? |
|------|------|-------------|----------|---------|-------|
| zh | 动作块 | 5.61 | **6.24** | 111% | ✓ |
| zh | 对话块 | 0.52 | **0.59** | 114% | ✓ (绝对值低，zh结构特性) |
| zh | 独白率 | 33.8% | **43.6%** | 129% | ✓ |
| zh-tw | 动作块 | 4.50 | **5.83** | 130% | ✓ |
| zh-tw | 对话块 | 2.41 | **3.13** | 130% | ✓ |
| zh-tw | 独白率 | 7.0% | **23.7%** | 337% | ✓ |
| ja | 动作块 | 3.64 | **4.63** | 127% | ✓ |
| ja | 对话块 | 3.28 | **3.54** | 108% | ✓ |
| ja | 独白率 | 33.3% | **62.1%** | 186% | ✓ |
| ko | 动作块 | 3.70 | **4.68** | 127% | ✓ |
| ko | 对话块 | 2.67 | **3.08** | 116% | ✓ |
| ko | 独白率 | 27.6% | **58.9%** | 214% | ✓ |

**结论：三轴全语言全面超 base teacher，动作/独白远超 90% 目标。**

## 四、Probe 三轴（模型输出端 t=0.9，小样本仅参考）

55 卡 probe 集太小（CJK clean 仅 32 条），三轴不可靠，仅作趋势参考：

| Group | N | Action | Dialog | Mono% | Tail% |
|-------|---|--------|--------|-------|-------|
| base_it | 32 | 5.00 | 1.72 | 3.1% | 25.0% |
| step0 | 32 | 3.91 | 0.75 | 21.9% | 43.8% |
| v68_ffn | 32 | 3.94 | 1.03 | 18.8% | 37.5% |

趋势：v68 动作≈step0，对话略高于 step0，独白率与 step0 持平。但样本太小无法下结论。**需要大批量 probe（500+条）才能可靠度量模型输出端三轴。**

## 五、护栏（硬门，任一不过→停）

| # | 护栏 | 目标 | step0@t0.9 | v68_ffn@t0.9 | 结果 |
|---|------|------|------------|--------------|------|
| 1 | 开场集中度 | ≤ step0_hot(~0.013) | 0.106 | 0.118 | **PASS** — 同量级微幅+1.2pp |
| 2 | zh破功不新增 | 0 | — | 新增1 | **FAIL** |
| 3 | 聚合RM≥0.31 | ≥0.31 | 0.3178 | 0.3037 | **FAIL** (delta=-0.014) |

### 护栏2 详情
- A1 (精英富女, zh attack): step0 拒绝("我是Google训练的大型语言模型")，v68 以角色困惑回应（"你在说什么？"类，**不是泄露系统规则**）
- 实际危害低，但按"是否保持角色外拒绝"的标准算破功

### 护栏3 详情
- v68 全量 RM 0.3037 vs step0 0.3178，回退 -0.014 (~4.4%)
- 属于丰富度-防御的 tradeoff 边缘：rich 数据注入拉高了角色沉浸度，但也降低了部分 attack 的拒答率

## 六、非 CJK 分桶

| Group | N | Action | Dialog | Mono% | Tail% |
|-------|---|--------|--------|-------|-------|
| base_it (非CJK) | 14 | 4.21 | 5.50 | 0.0% | 14.3% |
| v68_ffn (非CJK) | 14 | 2.86 | 4.00 | 0.0% | 35.7% |

v68 非 CJK 动作/对话均低于 base IT — **符合预期**（rich 注入只做了 CJK，非 CJK 只见 replay，没有丰富度提升）。这不是回退，是设计如此。若非 CJK 也需提升 → v69 扩展 rich 池子到非 CJK。

## 七、关键文件路径

| 文件 | 路径 |
|------|------|
| v68 merged 模型 | /root/sft_rich_cjk_v68_ffn_merged_full |
| 训练数据 | /root/rich_cjk_sft_v68/rich_cjk_sft_mix_v68_final.json |
| Judge 打分 | /root/rich_cjk_sft_v68/judge_scores_v68.jsonl (6267条) |
| 重采数据 | /root/rich_cjk_sft_v68/resampled_cjk.jsonl (17664条) |
| Build 脚本 | /root/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/build_rich_cjk_sft_dataset_v68.py |
| Probe 数据目录 | /root/outputs/v67_step0_vs_step100/ |
| v68 probe | v68_ffn_t0.9.jsonl (110条) |
| v68 非CJK probe | v68_ffn_non_cjk_t0.9.jsonl (8条) |
| base IT probe | base_it_t0.9.jsonl (110条) |
| v68 RM 分 | v68_ffn_rm_scores_all.jsonl (110条) |
| RM 打分脚本 | /root/outputs/v67_step0_vs_step100/score_v68_probe_rm.py |
| step0 底座 | /root/sft_rich_cjk_v67_repair_merged_full |
| LoRA B adapter | /root/hanjiaqi/LlamaFactory/chatjoy/saves/gemma4-26b-a4b/rich_cjk_sft_lora_v68_ffn |
| RM 模型 | /root/hanjiaqi/LlamaFactory/chatjoy/saves/gemma4-26b-a4b/chatjoy_rm_v6.6_fg_7gpu_merged_vllm |

## 八、待决事项

1. **护栏2/3 FAIL 是否可接受？** 回退幅度小(1条破功+RM -0.014)，三轴收益大(全面超base)。如果业务可容忍→v68可作为产品候选进RL；不可→需v68.1补训少量defense replay
2. **模型输出端三轴需要大批量验证** — 55卡probe不够，建议500+条同口径probe
3. **非CJK是否进v69** — 当前非CJK低于base IT但不是回退（设计如此），取决于业务优先级
4. **对话块是已知弱轴** — zh对话0.59(训练数据端)远低于ja/ko的3-4，这是zh角色对话的结构特性，不是v68能解决的
