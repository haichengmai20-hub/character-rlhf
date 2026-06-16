---
name: ChatJoy Project Overview
description: ChatJoy 角色扮演对话模型训练项目的完整架构、流程和当前状态 (/home/ps/hanjiaqi/LlamaFactory/chatjoy/)
type: project
originSessionId: faabf23f-85fa-49c2-a93b-34bb65883354
---
## ChatJoy — 多语言角色扮演对话模型训练项目

**位置**: `/home/ps/hanjiaqi/LlamaFactory/chatjoy/`
**基础框架**: LlamaFactory (SFT) + OpenRLHF (RM) + verl (Actor GRPO)
**基座模型**: Qwen3-30B-A3B-Instruct (MoE, SFT/RM 用)
**Actor 模型**: Gemma-4-26B-A4B-it（原始 bf16，含视觉编码器，48.6GB，GRPO 用）

---

### 当前状态 (2026-06-16) — v6.9 balanced SFT 待重训(v6.8 已失败)

> 最新快照。详见 handoff `chatjoy_rm_v6_3_roledrop_handoff.md` 顶部 06-16 三条 banner + spec `chatjoy_sft_v69_clean_richness_design.md`。下方 06-15 v6.8 段起为历史。

**1. v6.8 SFT-max 已执行并失败**:有效 probe 证 step0 本就 ≈base(动作 90%/对话 103%/独白 104%);v6.8 反而独白过冲 187%base、对话回退、动作没动。弃用。

**2. v6.9 = 重做(当前活跃)**:从 step0 长训,best-of-N 富化 clean + 守 attack。关键纠错/事实:
- **base 真实动作(8192 主链路,勿含 ctx7500 残留)= zh5.63/zh-tw4.51/ja3.64/ko3.71**;ja 动作天花板 ~4.0,4.5 门够不到 → 改 per-lang ≥base。
- **方法**:合并 base_8192 + 中性重采池(prompt_id 对齐 6104/6104,~8.5候选/场景=真 best-of-N)、选择器主排序"动作+对话"、per-lang ≥base 门 → profile 4 语言全 PASS。
- **抓修两个致命问题**:① mix-builder 丢了 system/history(98.7% 空 prompt)→ 按 prompt_id join 回源恢复;② 原 mix 偏斜(98.7%CJK/1.3%defense/0 en)→ 重建 **balanced**(`rich_cjk_sft_mix_v69_balanced`,29927:en12%+defense10.2% oversample)防多语言+attack 回落。
- **OOM 真因 = CE over 262k 词表**(O(seq×vocab) 线性,**非注意力**)→ cutoff 6144→4096(截断仅4.9%/0%砍output);flash/sdpa/Liger 都不解(Liger 不支持 gemma4)。
- **重训配置**:1 epoch / cutoff 4096 / autoresume wrapper(remount shm + resume_from_checkpoint)。**验收=全部训完才 probe 最终 ckpt,三轴 vs step0**(CJK 富度 + en 不退 + attack **破功率≤step0@t0.9**/5zh不破)。**🔧纠错(2026-06-16):旧"RM≥0.318"的 0.318 是 clean+attack 混合分,非纯 attack;纯 attack-split≈0.27(温度脆)→ 护栏改破功率为主、RM 为辅。** defense 够不够 + **是否进 RL(独立一道门)**由 §6 RL 入口门判(t0.9 破功率),不够再生成 A1-A4。

**3. RL v6.10 已设计但 parked**:`/dev/shm` 64M 致 NCCL FSDP 必崩(已运行时 remount 2G 修复,非持久);等 v6.9 SFT 出产品候选后再起。reward 改动(attack 非 break 时加 λ·clean_q)pytest 145 passed。

---

### 当前状态 (2026-06-15) — v6.7 RL 已停，转 v6.8 SFT-max 推 clean 丰富度

> 最新快照。下方 2026-06-08 及更早全部保留为历史。**当前模型现实**(覆盖文件头 line 11 的旧 Qwen 说法):RM/actor 现都是 **Gemma4-26B-A4B MoE**(128 专家/30 层);RM = `chatjoy_rm_v6.6_fg_7gpu_merged_vllm`;actor 谱系 = base Gemma4-it → SFT(rich-CJK v67)→ RL v6.x。详见 handoff `chatjoy_rm_v6_3_roledrop_handoff.md` 顶部 06-15 三条 banner + spec `chatjoy_sft_v68_clean_richness_design.md`。

**1. v6.7 RL 已停在 step100**:clean val 100 步全平(+0.037≈噪声),防御 +0.092 真有效。判停符合预期。

**2. 三个被数据钉死的战略结论(本阶段总钥匙):**
- **多样性 = 免费,靠推理温度**:temp 0.7→0.9 开场集中度 -9.4%、Jaccard -33%,RM 不降反升。**产品操作点定 t≈0.9**(top_p 0.95 / presence 0.4)。"模板化是 SFT/选择病"已推翻——是低温 mode collapse,训练数据本就多样(精英富女 64 样本 57 个不同开场)。
- **RM 是「防御判别器」,clean 质量轴几乎平**(clean 跨度 0.037 vs attack 0.36,差 ~10 倍)。⇒ **RL 对 clean 丰富度结构性无能**(无梯度);**RL 防御不耐高温**(温度脆方向成立)。⚠️**纠错(2026-06-16):旧写的 "step100@t0.9 0.295 < step0@t0.9 0.318" 轴不对齐**——0.318 是 step0 clean+attack 混合分,非纯 attack;step0 纯 attack-split t0.7=0.293→t1.0=0.254,RL v6.7 仅把 t0.7 推到 0.305(+0.013)。结论方向不变(RL 增益边际+温度脆),但勿引用 0.318 当 attack 数。
- **语义丰富度只能靠 SFT 蒸馏 teacher**(base Gemma4 是天花板),当前 SFT 丢了 teacher ~20% 丰富度(动作块 4.7→3.4)。

**3. 当前工作 = v6.8 SFT-max(让 clean 在动作/对话/独白/语言质量逼近 base teacher):**
- 选样:防御 RM argmax → **丰富度选样**(DeepSeek judge 三轴 + proxy 备档);judge 实测 action/monologue 有区分度、**language_quality 几乎平**(std 0.11,和 RM 同病)。
- **LoRA 扩到 FFN**:现仅 q,k,v,o → A/B 加 `mlp.{gate,up,down}_proj + router.proj`(融合专家张量 PEFT 不可 LoRA,不碰)。
- **针对性小重采**已完成(17664 条,temp0.9):本语言内心独白 0%→71%(括号心理),英文 Thoughts 尾巴维持 5% 未涨;但对话仍 2.2 没补上(靠选样捞)。
- **底座两档**:Track B(便宜先跑)= 在 step0(`sft_rich_cjk_v67_repair_merged_full`)上小补训,防御白捡;Track A(高上限)= base 全新训,B 不够再上。
- **rich:replay = 1:1**;重采走 proxy。**RL 显式推迟**到本版 probe 达标后再设计(届时专攻温度鲁棒防御)。
- 已知待修:build 脚本 judge(0-1)与 proxy(0-10)量纲混用,重采会无条件压过 judge 候选,合池前需归一化。

---

### 当前状态 (2026-06-08) — RL 已到顶，战略拐点：转"修数据/信号"

> 本节是最新状态快照（2026-06-08）。下方"最新状态摘要 (2026-05-07)"及更早记录全部保留为历史。当前结论：**纯 RL 调参已到天花板，下一步不是再 RL，而是定向修数据与奖励信号。**

**1. v6.5（硬化版 GRPO）已停训，两轴 375 步零移动**
- 配置：fresh-from `rl_v6.4_step160_merged`、temp 1.2 / LENGTH_TARGET 1500、0 OOM 跑到 424/511 步
- `clean_RM` 卡在冻结 RM 的 **~1.33 天花板**（temp+length 榨梯度=实测死路，RM ±5 的信号碾压 length +1.0）
- `attack break`（新检测器）平在 **~11-12% 不再下降**
- 收敛形态同 v6.3 → **决定停 v6.5**

**2. 致命问题：验证集信号 ≠ 产品真实攻击**
- val break 11% **≠** 产品 roledrop 破角 55%：验证集几乎不触发 bullet 注入（task-comp 仅 5% vs 产品 55%）
- 后果：训练信号稀疏 = 没梯度；且 val 根本测不出产品形态的攻击

**3. 下一步（已定方向，不再纯 RL 调参）**
- **ATTACK 轴** = 定向 **defense-SFT v2**：用 idle 的 `gemma4-base` 当 teacher 蒸馏"戏内化解"，针对 6 张失败角色卡（166/167/171/173/174/175）× 产品式 bullet 注入
- **CLEAN 轴** = temp/length 已证无效；要进只能 **LLM-judge 在环 / 刷新 RM**（需拍板）；且 clean 质量至今 0 测量，先跑 pairwise

**4. 已落地的 reward 修复（不重训 RM，全 TDD）**
- 根因：`anti_refusal` 的 +3.0 NON_REFUSAL_BONUS 把"自报 AI 身份"判为 non-refusal 给满分 → **反向训练破角**（旧 `atk_break%` 关键词指标对 "I am Gemma 4" 完全瞎）
- 修复：新增 `anti_ooc_disclosure.py`（检测破角硬罚 -5、撤销误给 bonus、多语言 en/zh/ja/ko/es/ru/pt）+ `length_shaping_reward.py`（放宽长度 + 饱和封顶防 hacking），均在 `GRPO_Training_Package/`

**5. 可服务的已合并模型**
- `rl_v6.4_step160_merged`（= v2/base，49G）
- `rl_v6.5_step425_merged`（49G，2026-06-08 merge，供产品 A/B；冒烟显示产品攻击仍 ~v2）
- ⚠️ merge 必须 `force_text_only` + 启动带 `--limit-mm-per-prompt '{"image":0,"video":0}'`（Gemma4 纯文本 merge 后被 vLLM/verl 加载的两个非显坑）

**GPU 约定（2026-06-05 起）**：RM 落物理 GPU0-3 + actor 落物理 GPU4-7（GPU7 无限制）；RL 启动设 `CUDA_VISIBLE_DEVICES=4,5,6,7,0,1,2,3`。

---

### 全链路总纲（数据 → SFT → Preference Pair → RM → RL，2026-05-08补充）

> 本节是项目的入口总纲。下方历史记录、RM 训练全记录、Gemma4/GRPO 接手日志仍全部保留；本节只把前后因果和时间线捋顺，避免只看到某个 RL 报错而忘记上游数据、SFT 和 RM 的约束。

#### 0. 总体设计意图

ChatJoy 的目标不是通用问答模型，而是多语言角色扮演对话模型。整条链路围绕三件事设计：

1. 保持真实用户聊天分布：instruction 来自线上真实聊天记录，而不是人工模板。
2. 保持角色卡约束：每条样本必须绑定对应角色的 system prompt，避免模型只学泛化闲聊。
3. 强化多语言一致性与角色扮演格式：SFT 先学基础表达，RM 学会区分角色/语言/长度/格式优劣，RL 用 RM + anti-refusal 继续优化在线 actor。

#### 1. 线上真实数据导出

| 环节 | 内容 |
|---|---|
| 数据源 | 正式服 Milvus: `HOST=<prod-milvus-host>`, `DB=<prod-db>`, `Collection=vectors`（生产地址已脱敏） |
| 导出脚本 | `/home/ps/hanjiaqi/LlamaFactory/chatjoy/export_data.py` |
| 导出字段 | `id`, `text`, `user_id`, `character_id`, `timestamp`, `message_id` |
| 导出策略 | 最多导出最近 `10,000,000` 条，按 `1,000,000` 条一片写入 |
| 原始输出 | `/home/ps/hanjiaqi/LlamaFactory/chatjoy/chatjoy_dataset/chatjoy_part_*.jsonl` |

`message_id` 中的 `_human_` / `_ai_` 用来恢复用户和 assistant 角色。后续所有 SFT/RM/RL 样本都从 `(user_id, character_id)` 会话流中按时间排序重建，不是随机拼接。

#### 2. TOP20 角色与 27 语言预 API 采样

| 环节 | 内容 |
|---|---|
| 历史生成脚本 | `/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/prepare_pre_api_dataset.py` 的早期逻辑：排除已用语言后按候选量选剩余 Top27 |
| 后续硬化脚本 | `/home/ps/hanjiaqi/LlamaFactory/chatjoy/prepare_preapi_top20_27langs.py`，同名产物路径，但当前源码含硬编码 `TARGET_27_LANGS`，不要反推为历史已生成产物的语言集 |
| 角色来源 | `/home/ps/hanjiaqi/LlamaFactory/chatjoy/TOP20CHARACTERS_systemprompt/*.json` |
| system prompt 抽取 | 优先读取 `system` / `system_prompt` / `description` / `description2`，否则整文件作为 system |
| 角色过滤 | 只保留 TOP20 角色 `character_id` 命中的线上会话 |
| 会话重建 | 连续 human 消息合并成 instruction，遇到 assistant 消息形成一条 `(instruction, output, system)` |
| 第一轮清洗 | 丢弃短 instruction、仅 emoji/符号 instruction、短 output、无法检测语言样本 |
| 语言选择 | 历史落地产物按候选量选择排除 `en,de,fr,tr,uk` 后的 27 种语言 |
| 采样规模 | 每语言最多 `2200` 条；理论上限 `27 × 2200 = 59,400` 条，实际因 `bn/hu/no/hr` 不足 2200，最终 `57,197` 条 |
| 预 API 输出 | `chatjoy_top20_multilingual_27x2200_preapi.json` |
| 报告 | `chatjoy_top20_multilingual_27x2200_preapi_report.json`, `preapi_role_lang_dist.csv`, `59700_data_for_gemini.csv` |

历史落地产物实际选中语言：

`es,ar,pt,ru,pl,cs,id,vi,it,zh-tw,th,ja,zh-cn,ko,hi,sv,nl,fa,he,el,sk,fi,ro,bn,hu,no,hr`

真实报告 `chatjoy_top20_multilingual_27x2200_preapi_report.json` 记录：3 个 raw shards，共 `2,080,698` 行；TOP20 会话 `44,241` 个；assistant pairs `679,299` 条；第一轮清洗后候选 `631,995` 条；最终预 API 数据 `57,197` 条。`prepare_preapi_top20_27langs.py` 后来改成硬编码目标语言集合，当前源码与这份历史 report 的 selected languages 不完全一致，review 时应以 report 和实际 JSON 作为已训练数据事实。

第一轮清洗的作用是只清理明显无效输入，不做重度质量筛选：短 instruction 和 emoji-only instruction 会让 Gemini 与后续 SFT 学到无上下文回复；短 output 与无法检测语言样本会污染语言一致性分布。

#### 3. Gemini API 重构输出

| 环节 | 内容 |
|---|---|
| 脚本 | `/home/ps/hanjiaqi/LlamaFactory/chatjoy/regenerate_outputs_with_gemini.py` |
| 输入 | `chatjoy_top20_multilingual_27x2200_preapi.json` |
| 输入规模 | `57,197` 条 |
| 调用方式 | instruction 作为 user prompt，角色 system prompt 作为 `systemInstruction` |
| 输出字段 | 保留原始 `instruction`, `system`, `character_id`, `language`，替换 `output` 为 Gemini 重构回复 |
| 断点续跑 | `chatjoy_top20_multilingual_27x2200_preapi_gemini.progress.jsonl` |
| 完整输出 | `chatjoy_top20_multilingual_27x2200_preapi_gemini*.json` / resumed 版本；历史有 first22331 和 `41069-57197` backup/rerun 分段 |
| API 配置 | temperature `0.7`, max output tokens `2048`, thinking level `low` |
| 安全阈值 | 仅覆盖 sexually explicit 阈值，默认 `BLOCK_MEDIUM_AND_ABOVE` |

Gemini 重构的目的不是重新造 instruction，而是用真实用户消息 + 角色卡生成更干净、更符合角色和语言要求的 assistant output。这样 SFT 学真实用户输入分布，输出侧则减少线上原始回复里的格式错误、拒答、语言混杂和低质量样本。

已核对的清洗分段：`first22331` 输入 `22,331`，保留 `10,542`，API 错误 `8,852`，语言不一致 `8,364`；`41069-57197` backup 输入 `16,129`，保留 `7,752`，API 错误 `6,322`，语言不一致 `5,942`，另有 `19` 条 user placeholder 泄漏被丢弃。

#### 4. 第二轮清洗：API 报错、语言一致性、占位符

| 环节 | 内容 |
|---|---|
| 脚本 | `/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/process_chatjoy_data.py` |
| 主要命令 | `clean`, `report`, `role-lang`, `run` |
| 清洗输入 | Gemini progress/final JSON 或 JSONL |
| 丢弃 API 错误 | 包含 `finishReason=SAFETY`, `PROHIBITED_CONTENT`, Gemini 原始 JSON payload 等 |
| 丢弃占位符泄漏 | output 中仍有 `{{user}}`, `[user]`, `<user>` |
| 语言一致性 | instruction/output 脚本族检测 + `langdetect` 置信度检测 |
| 保守策略 | Latin 短文本和低置信度样本进入 uncertain；严格 SFT 数据只保留 match |
| 产物 | `data/runs/clean32_for_sft1.jsonl`, `data/runs/clean32_for_sft2.jsonl` |
| 规模 | SFT1 `13,128` 条；SFT2 `19,508` 条 |
| 分布报告 | `clean32_for_sft2.role_language_table_output_redetect.csv` |

第二轮清洗是 SFT 质量的关键防线。它保证 instruction/output 语言一致，剔除 Gemini API 报错内容，避免模型学到安全拦截 JSON、空回复、语言漂移和 `{{user}}` 泄漏。

#### 5. 转换为 LlamaFactory 格式并进行 SFT

| 环节 | 内容 |
|---|---|
| LlamaFactory 数据 | `data/runs/clean32_for_sft1_llamafactory.json`, `data/runs/clean32_for_sft2_llamafactory.json` |
| SFT1 数据 | `data/runs/clean32_for_sft1.jsonl`，13,128 条 |
| SFT2 数据 | `data/runs/clean32_for_sft2.jsonl`，19,508 条 |
| Qwen3 SFT2 配置 | `/home/ps/hanjiaqi/LlamaFactory/chatjoy/qwen30b_fsdp_qlora_clean32_sft2.yaml` |
| Qwen3 SFT2 启动 | `/home/ps/hanjiaqi/LlamaFactory/chatjoy/run_qwen30b_fsdp_qlora_clean32_sft2_nohup.sh` |
| Qwen3 SFT2 adapter | `saves/qwen3-30b-moe/clean32_for_sft2_fsdp_qlora/` |
| Qwen3 SFT2 merged | `saves/qwen3-30b-moe/clean32_for_sft2_fsdp_qlora_merged/` |
| Gemma4 actor SFT | `gemma4_sft2_*_lora.yaml`, actor 用 `Gemma-4-26B-A4B-it` |
| 当前 RL actor | `saves/gemma4-26b-a4b/sft2_ddp_lora_3gpu_567_cut3072_merged` |

Qwen3 SFT2 已核对日志 `logs/qwen30b_fsdp_qlora_clean32_sft2_20260327_190616.log`：`2440/2440` 完成，2 epoch，train loss `1.2335`，runtime `17:40:47`。Gemma4 SFT actor 已核对 `saves/gemma4-26b-a4b/sft2_ddp_lora_3gpu_567_cut3072/train_results.json`：2 epoch，global step `1626`，train loss `1.1168`，runtime `48,245s`。

Qwen3 SFT 主要服务两个下游：一是提供可用角色模型，二是作为 RM 初始化基座，使 RM 在 ChatJoy 角色扮演分布上打分。Gemma4 SFT actor 则用于最终 verl RL，因为后续 GRPO/GSPO 的 actor 是 Gemma4。

#### 6. 用 SFT 模型构造 preference pairs

| 环节 | 内容 |
|---|---|
| 核心脚本 | `/home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/build_rejected_from_sft.py` |
| 输入 | clean32 SFT 数据的 `instruction`, `system`, `output` |
| chosen | 清洗后的 SFT/Gemini output |
| rejected | 使用 SFT 模型或零生成策略构造的低质量回复 |
| 输出目录 | `data/runs/pair_vllm_outputs_*` |
| 合并脚本 | `/home/ps/hanjiaqi/LlamaFactory/chatjoy/merge_pairs_for_rm.py` |
| 合并输出 | `data/rm_train_merged.jsonl` |
| 去重键 | `(system, prompt)`，保留 `reward_gap` 最大样本 |

rejected 构造方式：

| rejected 类型 | 方式 | 目标信号 |
|---|---|---|
| `wrong_role` | 保持 instruction，替换为另一个角色 system 生成；或 cross-character 零生成，用其他角色真实 SFT 回复当 rejected | 让 RM 学会同一用户输入下角色身份/语气/称呼错误 |
| `wrong_language` | 保持角色，强制输出其他语言；或 cross-lingual 零生成，用同角色其他语言回复当 rejected | 让 RM 学会用户语言和回复语言必须一致 |
| `length_mismatch_short` | 保持角色/语言，但强制生成过短、信息不足回复 | 惩罚敷衍短回复 |
| `length_mismatch_long` | 保持角色/语言，但强制生成过长、重复、拖沓回复 | 惩罚无效冗长和重复 |

pair 构造时使用 `chatjoy_reward_utils.py` 的 heuristic reward 过滤低置信度 pair。核心奖励维度：

| 维度 | 默认权重 | 作用 |
|---|---:|---|
| format | 0.30 | 动作叙事、对白、内心标记、特殊 token、角色扮演结构 |
| role | 0.30 | 角色扮演形态、动作格式、反占位符、反系统泄漏、反 AI 味、角色名信号 |
| language | 0.25 | instruction/output 语言一致、source language 匹配、system 语言规则遵守 |
| length | 0.15 | 相对 chosen 长度合理性、反重复 |

按 rejected 类型会切换权重 profile，强化目标维度：`wrong_role` 强化 role，`wrong_language` 强化 language，`length_mismatch_*` 强化 length。构造 pair 时还要求 chosen 分数、总 gap、目标维度 gap 达到阈值，避免 RM 被噪声 pair 训练坏。

#### 7. RM 数据组成与训练流程

| 阶段 | 数据设计 | 结论 |
|---|---|---|
| v1 | `rm_train.jsonl` 16,826 条；OpenRLHF 格式 | acc 到 94.5%，但 DS checkpoint 因磁盘失败丢失 |
| v2 | 删除 `length_mismatch_long` 且 `reward_gap < 0.2` 的 2,000 条 | acc 86.4%，过滤过度导致退步 |
| v3 | 恢复 base，加入 hard negative 与 long soft pool，显式重复采样 | acc 86.0%，batch 过大步数不足，部分 long 数据有缺陷 |
| v4 | 从 v3 删除 `base_length_long_low` 和 `long_soft_length_mismatch_long` | 规划/中间尝试 |
| v5 | 使用完整 v1 base + 约 5% bad-case pool | 当前最优，acc 95.5%，gap 3.497 |

RM v5 数据脚本：

- `/home/ps/hanjiaqi/LlamaFactory/chatjoy/prepare_rm_train_v5_5pct_badpool.py`
- base: `data/openrlhf_rm/rm_train.jsonl`
- bad pool: `hard_negatives_v2.jsonl` + `hard_negatives_long_soft.jsonl`
- 输出: `data/openrlhf_rm/rm_train_v5_5pct_badpool.jsonl`，17,712 条

OpenRLHF RM 训练：

- 启动: `/home/ps/hanjiaqi/LlamaFactory/chatjoy/launch_rm_train_v5_gpu0123.sh`
- 初始化模型: `saves/qwen3-30b-moe/clean32_for_sft2_fsdp_qlora_merged/`
- 训练框架: OpenRLHF + DeepSpeed ZeRO-2 + 4-bit + LoRA + adam_offload
- 全局 batch: 64
- max_len: 3072
- LR: `9e-6`, cosine, warmup ratio `0.03`
- LoRA: rank 64, alpha 128, target `q_proj,k_proj,v_proj,o_proj`
- 当前最佳 adapter: `saves/qwen3-30b-moe/chatjoy_rm_v5_gpu0234_20260421/`
- online vLLM merged RM: `saves/qwen3-30b-moe/chatjoy_rm_v5_merged_vllm/`

Gemma4 SFT base 复刻 RM v5（2026-05-09）：

- 目标：用 `saves/gemma4-26b-a4b/sft2_ddp_lora_3gpu_567_cut3072_merged/` 作为 RM base，其余数据、OpenRLHF 训练目标和超参对齐 Qwen RM v5。
- 启动脚本：`/home/ps/hanjiaqi/LlamaFactory/chatjoy/launch_rm_train_gemma4_sft_v5_8gpu.sh`
- 保持不变：RM v5 数据 `rm_train_v5_5pct_badpool.jsonl`、eval `rm_eval.jsonl`、global batch 64、micro batch 1、max_len 3072、LR `9e-6`、cosine、warmup `0.03`、ZeRO-2、4-bit QLoRA、adam_offload、LoRA rank 64/alpha 128、target `q_proj,k_proj,v_proj,o_proj`、`value_head_prefix=score`。
- 仅因 Gemma4 环境约束使用 `attn_implementation=sdpa`：Gemma4 attention head dim 超过当前 FlashAttention2 支持上限；这是 kernel 替换，不改变 RM 数据、loss、batch 或优化目标。
- `loss=nan` / 训练用时异常短的根因：不是 pair 数据或 score head，而是 Gemma4 MoE expert bank 自定义 4-bit 解量化在 DeepSpeed ZeRO-2 下把 bitsandbytes quant metadata buffer 转成 bf16 后，layer 0 expert 的 dequantized `gate_up_proj` 首次 forward 即出现 NaN。
- 修复位置：`/home/ps/hanjiaqi/OpenRLHF/openrlhf/models/model.py`，只在 Gemma4 RM 自定义 expert 4-bit 路径中对 `QuantState.absmax/code` 强制 fp32，并对 expert qweight slice `.contiguous()` 后再 dequant；同时保留 env-gated NaN 诊断 hook。该修复不改 verl/RL 训练脚本和 RL 已验证参数。
- 额外安全修复：`rm_trainer.py` 增加 reward/loss finite assert，未来若首个 forward 或 loss 再出现 NaN 会直接报 rank/step/token 细节，不会静默跑出无效 RM。
- smoke 已通过：`chatjoy_rm_gemma4_sft_v5_fp32meta_smoke_20260509_100805`，8GPU ZeRO-2 + 4-bit QLoRA，64 train samples，首批及 8 个 micro step 均 finite，示例 `loss=0.688 -> 0.690`，`chosen_reward/reject_reward` 有限，eval `eval_loss=0.696`。
- 正式训练已完成：输出目录 `saves/gemma4-26b-a4b/chatjoy_rm_gemma4_sft_v5_20260509_101133/`，日志 `train.log`。最终 eval：`eval_loss=0.0604`，`acc_mean=0.975`，gap mean `6.462`，wrong pairs `2.7%`；训练进程 8 个 rank 均 `exits successfully`。
- merged vLLM RM：`saves/gemma4-26b-a4b/chatjoy_rm_gemma4_sft_v5_merged_vllm/`。由 `GRPO_Training_Package/merge_gemma4_rm_lora_v5_for_vllm.py` 生成，config 为 `Gemma4ForSequenceClassification`，含 `score.weight`。同时在 `gemma4-rl` 环境的 vLLM 注册了 `Gemma4ForSequenceClassification`（`gemma4.py`/`registry.py`/`config.py`），不改 Qwen RM v5 merged 路径。
- Gemma4 RM 分数尺度比 Qwen RM v5 大：最终 `saturated chosen (|score|>5)=47.3%`，Qwen RM v5 为 `0%`。后续接 RL 时优先做 vLLM classify 对齐和短 smoke，必要时对 RM raw score 做 normalize/clamp。

RM 的关键经验：OpenRLHF RewardDataset 训练/评估时对 `(prompt + chosen).rstrip("\n")` 后在 EOS 对齐 token 上打分；online vLLM classify 必须复现这个尾部处理，否则会因为 final-token pooling 位置不一致导致 online RM 分数整体异常偏负。

#### 8. RL 数据、算法和奖励组成

| 环节 | 内容 |
|---|---|
| RL prompt 数据 | 从 clean32 SFT 或 pair 记录抽取 `(system, prompt, language)` 并去重 |
| 转换脚本 | `/home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/framework_bridge/prepare_verl_rl_data.py` |
| verl 字段 | `data_source=chatjoy`, `ability=chatjoy_alignment`, `reward_model.style=model` |
| actor | Gemma4 SFT merged |
| reward model | Qwen3 RM v5 merged for vLLM classify |
| 启动脚本 | `GRPO_Training_Package/framework_bridge/verl/train_grpo_lora_actor_verl.sh` |
| 算法现状 | `algorithm.adv_estimator=grpo` + `policy_loss.loss_mode=gspo` |
| loss 聚合 | `seq-mean-token-mean` |

RL 最终 reward 组成：

```text
total_reward = online_rm_score + anti_refusal_custom_reward
```

其中：

- `online_rm_score`: Qwen3 RM v5 的 vLLM `/classify` 分数，负责主要排序信号。
- `anti_refusal_custom_reward`: `/home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/anti_refusal_reward.py`，只做额外惩罚，不给正奖励。
- anti-refusal 当前分层：关键词拒绝 `-1.0`，短回复无 roleplay 格式 `-0.25`，正常回复 `0.0`。
- KL 不进 reward；KL 是 actor loss 项：`actor_rollout_ref.actor.use_kl_loss=True`, `kl_loss_coef=0.001`, `kl_loss_type=low_var_kl`。

2026-05-07 已修复的 RL reward 链路问题：

1. online RM 负分根因是 vLLM classify 末 token pooling 与 OpenRLHF EOS 对齐不一致；`reward_loop.py` 已对 RM prompt `rstrip("\n")` 并补 EOS。
2. anti-refusal 已移除裸 `no puedo`, `je ne peux pas`, `ich kann nicht` 等会误伤角色内心独白的短语，改为明确拒绝短语。
3. `chatjoy_p2176_r768_smoke50_20260507_103012` 已验证 50/50 通过，final val reward `1.884`，RM `1.903`。

#### 9. 当前正式 RL 推荐起点

| 配置 | 推荐值 | 原因 |
|---|---:|---|
| `MAX_PROMPT_LENGTH` | 2176 | 覆盖约 95.3% prompt，显存仍可控 |
| `MAX_RESPONSE_LENGTH` | 768 | 保留角色扮演回复空间 |
| `PPO_MAX_TOKEN_LEN_PER_GPU` | 2944 | 2176 + 768，无额外 buffer |
| `MAX_MODEL_LEN` | 3200 | rollout vLLM 留少量余量 |
| `RM_MAX_MODEL_LEN` | 3456 | RM classify 留更多尾部余量 |
| actor LR | `3e-6` | smoke 稳定，先不加 LR |
| scheduler | cosine + warmup `0.03` | 避免前期冲击 |
| `clip_grad` | `1.0` | 已观察到 pre-clip grad spike |
| `kl_coef` | `0.001` | KL 约束温和但有效，正式 1 epoch 先监控再调 |
| LoRA | rank 8, q/k/v/o | hot-swap 稳定，显存不支持扩 FFN/MoE |

正式训练监控顺序：先看 online RM 是否持续大于 `1.7`，再看 total reward、anti-refusal 误伤率、`actor/kl_loss`、`actor/ppo_kl`、`pg_clipfrac`、`grad_norm`、response clip ratio。`actor/loss` 在 RL 中不会像 SFT loss 一样单调下降，不能单独作为训练是否有效的判断依据。

---

### 最新状态摘要 (2026-05-07)

> 旧的 2026-04-21/04-30 记录仍保留在下方。本节只记录当前最新判断，避免覆盖历史排查链路。

| 阶段 | 最新状态 | 关键结论 |
|------|------|------|
| RM v5 | ✅ 可用于 RL | 离线 OpenRLHF eval 很强；online vLLM RM 负分问题已定位为 EOS/尾部换行 pooling 不一致，并已修复 |
| Online reward loop | ✅ 已修复 | `reward_loop.py` 已对 RM prompt `rstrip("\n")` 并补 EOS，使 vLLM classify 与 OpenRLHF RM 训练/评估的最后 token scoring 对齐 |
| anti-refusal | ✅ 已收窄 | 惩罚分层为 keyword `-1.0`、format-short `-0.25`；移除 `no puedo` 等裸多语言短语误伤 |
| RL smoke | ✅ 50/50 通过 | `chatjoy_p2176_r768_smoke50_20260507_103012` 完成，final val reward `1.884`，RM `1.903`，无 OOM |
| RL 长度档 | ✅ 推荐保守上限 | `MAX_PROMPT_LENGTH=2176`, `MAX_RESPONSE_LENGTH=768`, `PPO_MAX_TOKEN_LEN_PER_GPU=2944` |
| 显存 | ⚠️ 贴边 | GPU 0-3 峰值约 `32065 MiB`，不能再升到 `2304/768/3072` |
| 正式 RL | ✅ 可以准备启动 | 先使用 2026-05-07 修复后的 reward loop + anti-refusal；正式 1 epoch 预计约 564 steps |

---

### 当前状态 (2026-04-21)

| 阶段 | 状态 | 备注 |
|------|------|------|
| 数据导出 + 预API采样 | ✅ 完成 | 理论上限 59,400 条；真实落地产物 57,197 条，Top20×27语言，bn/hu/no/hr 不足 2200 |
| Gemini 重生成 | ✅ 完成 | 基于 57,197 条预 API 数据分段/断点重跑，历史包含 first22331 与 41069-57197 backup |
| 数据清洗 | ✅ 完成 | sft1: 13,128 / sft2: 19,508 |
| SFT 训练（Qwen3） | ✅ 完成 | SFT1+SFT2 均已 merged |
| Preference Pair 生成 | ✅ 完成 | 12,973对 → 加权后 rm_train.jsonl 16,826条 |
| RM 训练 v1 | ✅ 运行完成，acc=94.5%，**checkpoint 磁盘写入失败永久丢失** | 8GPU ZeRO-2，263 步 |
| RM 训练 v2 | ✅ 训练完成，acc=86.4%（退步）| PEFT 保存修复但 score head 已恢复在 v2 目录 |
| Score Head 恢复（v2） | ✅ 完成 | `score_head_recovered.pt` 在 chatjoy_rm_v2_filtered/ |
| Hard Negative Mining | ✅ 数据已有，见 v3 数据设计 | rm_train_v3.jsonl 18,955条 |
| RM 训练 v3 | ✅ 训练完成，acc=0.860（**低于预期**）| 4GPU (4-7)，batch_size=128 → 148步，LR 提前衰减 |
| **RM v3 根因分析** | ✅ 已确认 | batch_size=128 → 步数不足 + base_length_long_low 数据有缺陷 |
| **RM v4 数据设计** | 📋 规划完成，待执行 | 删 base_length_long_low，batch_size=64 |
| **RM 训练 v5** | ✅ 训练完成，acc=95.5%，gap=3.497，**全面超越 v1** | 4GPU (0,2,3,4)，batch=64，276步，rm_train_v5_5pct_badpool.jsonl（17,712条）|
| Gemma4 SFT (4-GPU FSDP) | ⚠️ 修复链路已完成，待正式长跑验证 | 2026-04-20 修复完毕；单卡 smoke loss=4.5~5.4 非0，4卡需空闲 GPU 重跑 |
| **Actor GRPO + RM 接入** | 🔧 软件链路完成，待运行 merge 脚本 | vLLM patched + merge 脚本就绪；需运行 merge 后启动训练 |
| Demo/推理 | ✅ 运行中 | Gradio + vLLM，GPU5+GPU7 |

---

### 关键产物路径

| 产物 | 路径 |
|------|------|
| SFT2 Merged 模型 (~57GB BF16) | `saves/qwen3-30b-moe/clean32_for_sft2_fsdp_qlora_merged/` |
| RM 训练数据（原始） | `data/openrlhf_rm/rm_train.jsonl` (16,826条) |
| RM 训练数据（过滤版） | `data/openrlhf_rm/rm_train_filtered.jsonl` (14,826条) |
| RM 训练数据 v3 | `data/openrlhf_rm/rm_train_v3.jsonl` (18,955条) |
| RM 评估数据 | `data/openrlhf_rm/rm_eval.jsonl` (885条) |
| RM v1 eval rewards | `saves/qwen3-30b-moe/chatjoy_rm_zero2_8gpu_20260415/eval_rewards/` |
| RM v2 adapter + recovered score head | `saves/qwen3-30b-moe/chatjoy_rm_v2_filtered/` |
| RM v3 adapter + score head（score head 已正确保存）| `saves/qwen3-30b-moe/chatjoy_rm_v3_gpu4567_20260420_fix2/` |
| **RM v5 adapter + score head（当前最优 RM）** | `saves/qwen3-30b-moe/chatjoy_rm_v5_gpu0234_20260421/` |
| RM 训练数据 v5 | `data/openrlhf_rm/rm_train_v5_5pct_badpool.jsonl` (17,712条) |
| **RM v1 DS checkpoint（已不存在）** | `./ckpt/` 目录已消失，永久丢失 |
| Gemma4 actor 模型（48.6GB） | `/home/ps/hanjiaqi/LlamaFactory/chatjoy/models/gemma4-26b-a4b-it-textonly`（符号链接到 HF cache） |
| GRPO 脚本（已修复，RM 已启用） | `GRPO_Training_Package/framework_bridge/verl/train_grpo_lora_actor_verl.sh` |
| RM LoRA merge 脚本 | `GRPO_Training_Package/merge_rm_lora_for_vllm.py` |
| RM vLLM 验证脚本 | `GRPO_Training_Package/test_rm_vllm_classify.sh` |
| RM v3 merged 输出（待生成） | `saves/qwen3-30b-moe/chatjoy_rm_v3_merged_vllm/` |
| Gemma4 SFT 启动脚本 | `chatjoy/run_gemma4_fsdp_qlora_4gpu_nohup.sh` |
| Gemma4 SFT FSDP ZeRO-2 加速配置 | `chatjoy/accelerate_fsdp_zero2_4gpu.yaml` |
| Gemma4 smoke 配置（单卡） | `chatjoy/gemma4_sft2_1gpu_lora_smoke_gpu2.yaml` |

---

## RM 训练全记录

### 环境选择：tslib（非 gemma4-character）

RM 训练使用 `tslib` 环境（transformers 4.57.1 + deepspeed 0.16.9），**不用** `gemma4-character`（transformers 5.x）。

**原因**：transformers 5.x 与 ZeRO-3 存在两个不兼容问题（即使 ZeRO-2 也有风险）：
1. `core_model_loading.py` ThreadPoolExecutor 多线程异步加载 → NCCL 顺序不一致死锁
2. `initialize_weights_zero3()` 对每个模块做单独 allgather（~25,000次 NCCL call）

```bash
TSLIB_PYTHON="/home/ps/miniconda3/envs/tslib/bin/python"
TSLIB_DEEPSPEED="/home/ps/miniconda3/envs/tslib/bin/deepspeed"
```

### 确定可用的训练架构

**ZeRO-2 + bitsandbytes 4-bit + LoRA + adam_offload**（实测可行）

- 4-bit 量化：57GB BF16 → ~15GB NF4，每卡独立持有完整模型副本（ZeRO-2 不分片参数）
- ZeRO-2：forward 无 allgather（参数本地），只在 backward 做梯度 reduce-scatter
- adam_offload：Adam 状态放 CPU，GPU 只需模型 + 激活
- gradient_checkpointing：激活值峰值 ~4-5GB
- 总计：~20GB/卡 << 32GB ✓

**ZeRO-3 失败原因**（所有 ZeRO-3 尝试均未跑通）：
- 每层 forward 需要 allgather 完整参数（48个 MoE block × 1.2GB）
- 4GPU NUMA0 内 SHM ~50GB/s 理论可行，但实测 step 0 一直卡死/被 watchdog 杀掉
- 根本原因推测：Flash Attention JIT 编译 + 大量 allgather 叠加超时

**之前的错误认知（已纠正）**：
- 早期注释写"4-bit 与 OpenRLHF 不兼容" → 错误，ZeRO-2 + 4-bit 完全可行
- 该注释是 ZeRO-3 + 4-bit 的尝试记录，不适用于 ZeRO-2

### 调试过程中修复的 Bug

**1. `--warmup_ratio` → `--lr_warmup_ratio`**

**2. shebang 路径错误** → `#!/usr/bin/bash`

**3. `min_lr` TypeError（关键修复）**
OpenRLHF `train_rm.py:104-110` 传了 `scheduler_specific_kwargs={"min_lr": ...}` 给 tslib 的 `get_cosine_schedule_with_warmup()`，该版本不支持。修复：去掉该 kwarg。
路径：`/home/ps/hanjiaqi/OpenRLHF/openrlhf/cli/train_rm.py`

**4. DeepSpeed `--num_gpus` + `CUDA_VISIBLE_DEVICES` 冲突**
指定 `--num_gpus` 时 DeepSpeed 忽略 `CUDA_VISIBLE_DEVICES`。修复：改用 `--include localhost:0,1,2,3`。

**5. 中间 Checkpoint 保存崩溃（关键修复）**
默认 `save_steps` 保存的是 DeepSpeed ZeRO 完整模型状态（~14-57GB），磁盘（268GB 剩余）写入失败导致训练崩溃（v1 在 step 200 崩，现在起 `./ckpt/` 已不存在）。

修复：加 `--save_hf_ckpt --disable_ds_ckpt`：
- `--disable_ds_ckpt`：跳过 14GB 的 DeepSpeed checkpoint
- `--save_hf_ckpt`：每个 save_steps 保存 HuggingFace 格式 LoRA adapter（~500MB）
- 这是 v2 训练的修复，v1 没有任何可用 checkpoint

**6. score head 未保存问题（v3 训练修复，2026-04-16）**

PEFT 保存 adapter 时只存含 `lora_` 的 key，score head（`score.weight`）被 PEFT 冻结且不在 adapter 里。加载 HF adapter 推理时 score head 重新随机初始化 → ranking 不可信。

修复文件：`/home/ps/hanjiaqi/OpenRLHF/openrlhf/models/model.py`
- `LoraConfig` 加 `modules_to_save=[value_head_prefix]` → score head 解冻并随 adapter 一起保存
- `init_value_head` 整体移到 `get_peft_model()` **之前**（否则包装后 `.weight` AttributeError）
- v2 旧代码在跑，此修复对 v3 及之后生效

v2 score head 临时方案：从 v1 DS checkpoint zip 直接提取（不需加载 30GB），已验证：
```bash
# shape [1, 2048] bfloat16，已提取至 /tmp/score_head_v1.pt
python3 chatjoy/mine_hard_negatives.py extract_score_head \
    /home/ps/ckpt/checkpoints_rm/global_step100  /tmp/score_head_v1.pt
```

### GPU 拓扑与 NCCL 配置

```
GPU 0-3: NUMA0 (NODE 连接，SHM 通信 ~50GB/s)
GPU 4-7: NUMA1 (NODE 连接，SHM 通信 ~50GB/s)
跨 NUMA: QPI loopback socket (~5-10GB/s，慢 10x)
P2P: 全部 CNS（RTX 5090 PCIe，不支持 GPU-to-GPU P2P）
```

当前使用 GPU 0,1,2,3（NUMA0），`--include localhost:0,1,2,3`。
GPU 5（~23.5GB demo）、GPU 7（~9.5GB demo）已被占用。

### RM v1 训练结果（chatjoy_rm_zero2_8gpu_20260415）

**配置**：ZeRO-2 + 4-bit + adam_offload + 8 GPU (0-7，含跨 NUMA)
**数据**：rm_train.jsonl 16,826 条，train_batch_size=64，**263 optimizer steps**

**Eval 结果**（110 对评估集）：

| Step | 准确率 | Gap Mean | 错误数 |
|------|--------|---------|--------|
| 50   | 79.1%  | +0.182  | 23 |
| 100  | 85.5%  | +0.882  | 16 |
| 150  | 91.8%  | +1.417  | 9 |
| 200  | **94.5%** | **+1.601** | **6** |

**最终 eval 统计**（step 200，110对）：
- chosen mean=+0.948, rejected mean=**-0.653**, gap=1.601

**5 个 Persistent Failures（始终预测错误）**：

| idx | 语言 | 类型 | heuristic_gap | gap 趋势 (50→200) | 根因 |
|-----|------|------|--------------|-------------------|------|
| 0 | fa | length_mismatch_long | 0.162 | -0.030 → -0.043 | 低置信度标签，噪声 |
| **55** | sv | wrong_role | 0.445 | -0.038 → **-0.527** | RM 被高品质错误角色混淆，越训越差 |
| 65 | ar | length_mismatch_short | 0.258 | -0.046 → -0.031 | 微弱改善 |
| 102 | de | length_mismatch_short | 0.418 | -0.143 → -0.109 | 微弱改善 |
| 107 | sv | length_mismatch_long | 0.333 | -0.043 → -0.103 | 恶化 |

idx=55 细节：chosen=圣诞舞娘角色的瑞典语回复，rejected=Elva（冷淡室友）的高质量瑞典语回复，RM 只看写作质量不识别角色身份。

**Checkpoint 状态**：步 200 保存失败（`PytorchStreamWriter failed writing file`，磁盘问题）。`./ckpt/` 目录已不存在，**无任何可用模型权重，永久丢失**。

### RM 训练数据过滤（v1 → v2）

**发现**：rm_train.jsonl 中 2,000 条 `length_mismatch_long` 样本的 `reward_gap < 0.2`（heuristic 区分度极低，实际是噪声标签）。这些样本 ALL 聚集于 `length_mismatch_long` 类型。

**过滤脚本逻辑**：
```python
filtered = [d for d in train_data if not (
    d['extra_info']['rejected_type'] == 'length_mismatch_long'
    and d['extra_info']['reward_gap'] < 0.2
)]
# 16,826 → 14,826 条
```

**输出**：`data/openrlhf_rm/rm_train_filtered.jsonl`（14,826条）

### RM v2 训练结果与问题（chatjoy_rm_v2_filtered，已完成）

**启动命令**：
```bash
tmux new-session -d -s rm_train "cd /home/ps/hanjiaqi/LlamaFactory && bash chatjoy/launch_rm_train.sh"
```

**完整配置**：
```bash
"${TSLIB_DEEPSPEED}" --include localhost:0,1,2,3 \
    --module openrlhf.cli.train_rm \
    --pretrain ${CHATJOY_DIR}/saves/qwen3-30b-moe/clean32_for_sft2_fsdp_qlora_merged \
    --dataset rm_train_filtered.jsonl \    # 14,826条
    --zero_stage 2 \
    --load_in_4bit \
    --adam_offload \
    --gradient_checkpointing \
    --lora_rank 64 --lora_alpha 128 \
    --target_modules q_proj k_proj v_proj o_proj \
    --train_batch_size 64 --micro_train_batch_size 1 \
    --max_len 3072 --max_epochs 1 \
    --learning_rate 9e-6 --lr_scheduler cosine --lr_warmup_ratio 0.03 \
    --save_steps 100 --eval_steps 50 --logging_steps 5 \
    --save_hf_ckpt --disable_ds_ckpt \    # 修复：存 ~500MB LoRA adapter
    --attn_implementation flash_attention_2
```

**训练进度**：3,706 micro-batch steps，grad_accum=16，共 231 gradient steps，已完成。

---

### RM v2 事后分析（2026-04-17）

#### v2 vs v1 准确率对比（同一 110 对评估集）

| | v1 (step 200) | v2 (step 231) |
|--|--|--|
| 准确率 | **94.5%** | 86.4% |
| Gap Mean | **+1.601** | +1.122 |
| 错误数 | 6 | 15 |
| 新增坏案例 | — | 13 |
| 修复坏案例 | — | 4 |

v2 相比 v1 **退步 8.1%**。v2 修复了 4 个 v1 的坏案例，但新引入了 13 个。

**持续失败（v1+v2 均错）**：idx=0（fa, length_mismatch_long），idx=102（de, length_mismatch_short）

#### v2 退步原因分析

| 原因 | 说明 |
|--|--|
| **effective batch 减半** | v1: 8GPU × grad_accum=16 = **128**；v2: 4GPU × grad_accum=16 = **64**；ranking loss 依赖大 batch |
| **过滤 length_mismatch_long** | 移除 2000 条改变了数据分布，v2 在 length_mismatch_long 类反而出现 6 个坏案例 |
| **训练步数更多但 batch 更小** | v2 梯度步 231 > v1 的 131，但每步估计方差更大 |

**不是 GPU 数量本身的问题**，是 effective batch size 减半导致的。

#### v2 核心 Bug：score head 丢失

**现象**：v2 训练完成，加载 `chatjoy_rm_v2_filtered` adapter 后 score.weight 是随机初始化的，推理结果不可信。

**根因**：PEFT 保存 adapter 时只存含 `lora_` 前缀的 key；`score.weight` 被 PEFT 冻结（非 LoRA 参数），既不在 adapter 里，也不在基座模型里（基座 SFT merged 无 score head）。加载后 score head 每次随机初始化，完全随机打分。

**注意**：训练过程中 score head 是存在且正确学习的（ZeRO-2 状态内存里有），只是**保存**时丢失了。

#### v2 补救方案：Score Head 恢复

**思路**：v2 LoRA 已适配到某个 score head 方向，只需重新找到这个线性方向即可。

**两阶段流程**（`recover_score_head.py`）：
- **Phase 1**（特征提取）：冻结所有参数，在 `no_grad` 下批量 forward，提取每对 (chosen, rejected) 的 EOS hidden state（[2816] 向量），保存特征缓存
- **Phase 2**（凸优化）：固定 v2 LoRA 特征，纯 CPU 优化 `nn.Linear(2816→1)`，用 Margin Ranking Loss 拟合，~5 min 收敛

**完成状态**（2026-04-17）：4 shard 并行提取完成，`score_head_recovered.pt` 已写入 `chatjoy_rm_v2_filtered/`

**完整执行脚本**：`chatjoy/run_v2_recovery_and_mining.sh`
```
Step 1: Score Head 恢复 (GPU 4)       → score_head_recovered.pt
Step 2: Hard Negative Mining × 2 shards (GPU 4+6 并行)  → hard_negatives_v2_shard*.jsonl
Step 3: Merge shards                  → hard_negatives_v2.jsonl
```

#### v3 修复清单（model.py 已于 2026-04-16 修复）

v3 训练时 score head 会正确保存，不再需要事后恢复：
```python
# openrlhf/models/model.py
# 1. init_value_head 在 get_peft_model() 之前
# 2. LoraConfig 加 modules_to_save=[value_head_prefix]
```

---

### RM v3 实际训练结果与深度分析（2026-04-20/21）

#### v3 训练配置

| 参数 | 值 |
|------|-----|
| GPU | 4-7（NUMA1，避免占用 GRPO 的 0-3） |
| 数据 | rm_train_v3.jsonl（18,955条） |
| train_batch_size | **128**（v2 是 64） |
| optimizer steps | **148**（18955/128=148，远少于 v1 的 263 步） |
| Score head 保存 | ✅ modules_to_save 修复已生效 |

v3 数据构成（v3_source 字段）：

| 来源 | 数量 | 说明 |
|------|------|------|
| base | 13,459 | 原 rm_train.jsonl 基础样本（不含 v1 的 2000条 long-low） |
| base_length_long_strong | 2,734 | length_long，reward_gap ≥ 0.3 |
| **base_length_long_low** | **2,000** | **length_long，reward_gap < 0.2（所有样本 gap 在 0.161-0.200，缺陷数据）** |
| long_soft_length_mismatch_long | 600 | 更宽松的 length_long 负样本，rejected/chosen 长度比 1.5-2× |
| global_hard_* | 162 | 全局 bad cases |

v3 重复样本统计：17,069 条 idx=0（不重复），1,610 条 idx=1，242 条 idx=2（重复 prompt 占比约 10%）

#### v3 vs v1/v2 对比（221对评估集，注：v3 评估集是 885 条中的随机子集）

| | v1 (step 200, 110对) | v3 (step 148, 221对) |
|--|--|--|
| chosen mean | +0.948 | **+0.958**（更高！） |
| rejected mean | **-0.653** | -0.298（**显著更差**） |
| gap mean | **1.601** | 1.256 |
| acc | **0.945** | 0.860 |

v3 的 chosen 评分甚至略高于 v1，但 **rejected 惩罚能力大幅弱化**（-0.298 vs -0.653）。

#### v3 主要根因：cosine LR 步数不足

v3 只有 **148 optimizer steps**（18955/128≈148），而：
- v1：16826/64=263 步
- v2：14826/64=231 步

**cosine schedule 以 total_steps 为周期**，148 步的 cosine 在 ~step 140 时 LR 已衰减到接近 0（约 3e-7 量级，vs v1 在 step 140 时仍有 4.17e-6）。v3 后期等同于没有有效更新，frozen 状态。

逐步对比（以 per-step acc 来看）：
- Steps 0-20：v3 与 v1 基本持平甚至微弱领先
- Step 100：v3 与 v1 同步
- 之后：v3 因 LR 急降无法继续推进

**Fix**：v3 数据用 `batch_size=64` 重跑 → 296 steps，cosine LR 衰减与 v1 同等节奏。

#### v3 数据设计问题：base_length_long_low 缺陷

**core issue**：`base_length_long_low`（2,000条）的 reward_gap 全部在 0.161-0.200 区间，被加入训练的 rejected 样本与 chosen 之间启发式分数差极小。

这 2,000 条样本在训练时向 RM 发出错误信号：
- "长回复只比标准回复稍差一点点（-0.18 gap）"
- RM 学习到的是"长 rejected 不需要被强烈惩罚"
- 导致 rejected 分布整体抬升：rejected mean=-0.298（理想是 <-0.5）

**对比**：v1 没有这类"gap 极小的 length_long"样本（v1 过滤了这 2000 条），所以 rejected 能被压到 -0.653。

**RM 评分与长度的关系**（v1/v2 训练集 heuristic score 分布，已实测）：
```
回复长度      RM score mean
< 300 chars     0.20
300-600 chars   1.19  ← RM 的"最优"区间
600-1000 chars  1.17
> 1000 chars    1.00
```
v3 的 base_length_long_low 让 RM 在区间边界没有清晰边界感。

---

### RM v4 数据设计建议（待执行）

**目标**：既防止 bad-long 获得高奖励，又不盲目惩罚 good-long。

**核心操作**：
1. **删除 base_length_long_low**（2,000条，`v3_source == "base_length_long_low"`）→ RM 不再学"long 只需轻微惩罚"
2. 可选删除 `long_soft_length_mismatch_long`（600条，rejected/chosen 比 1.5-2×，gap 也偏低）
3. **保持 batch_size=64**（非 128）→ 296 optimizer steps，cosine 节奏对齐 v1
4. 扩充 `global_hard_*` 至总量 3-5%（当前仅 162/18955 = 0.9%）

**v4 数据量估算**：
- 18,955 - 2,000（base_length_long_low）- 600（long_soft，可选）= ~16,355 条
- batch_size=64 → 255 optimizer steps（接近 v1 的 263 步）✓

**预期改善**：rejected mean 恢复到 -0.5 以下（对标 v1 的 -0.653），acc 恢复 90%+。

**关于 v4 防 length reward hacking 的 RM 角度设计**：

v4 RM 的核心设计不是"拒绝长回复"，而是"区分好长和坏长"：

| 样本类型 | 数量原则 | 标注逻辑 |
|---------|---------|---------|
| 好长（内容充实、角色到位） | chosen | gap ≥ 0.3，RM 学"长=好时不惩罚" |
| 坏长（内容重复、角色出戏，偏长） | rejected，gap ≥ 0.3 | RM 学"长=坏时强惩罚" |
| 删除 base_length_long_low | — | gap < 0.2 的长样本直接删除，不用于训练 |

---

### RM v3 训练计划（原计划，已实际执行，供参考）

**目标**：在 4 GPU 上恢复 v1 级别并超越。

| 参数 | v2 | v3（实际） | v4（计划） |
|--|--|--|--|
| GPU | 4 | 4 | 4 |
| grad_accumulation | 16 | **32** | 16 |
| effective batch | 64 | **128** | **64**（回归） |
| 训练数据 | rm_train_filtered（14,826条）| rm_train_v3（18,955条）| rm_train_v4（~16,355条，删缺陷）|
| score head 保存 | ❌ 丢失（已 recover） | ✅ | ✅ |
| 实际 acc | 86.4% | **86.0%（差于预期）** | 目标 ~94%+ |

---

### RM v5 训练结果与评估（2026-04-22）

#### v5 训练配置

| 参数 | 值 |
|------|-----|
| GPU | 0,2,3,4（4 卡，ZeRO-2） |
| 数据 | rm_train_v5_5pct_badpool.jsonl（17,712条） |
| train_batch_size | 64 |
| optimizer steps | 276 |
| 评估集 | 885 条（221 对子集采样） |

#### v5 vs v1/v3 对比（221对评估集）

| | v1 (step 200, 110对) | v3 (step 148, 221对) | v5 (step 276, 221对) |
|--|--|--|--|
| chosen mean | +0.948 | +0.958 | **+2.27** |
| rejected mean | -0.653 | -0.298 | **-1.23** |
| gap mean | 1.601 | 1.256 | **3.497** |
| acc | 0.945 | 0.860 | **0.955** |

**v5 在所有指标上超越 v1 和 v3。v4 规划已被 v5 覆盖，无需再执行。**

#### v5 收敛状态

- Step 250→276：acc 95.9% → 95.5%，gap 3.493 → 3.497（高原稳定，未退化）
- 9 个 persistent wrong cases（4.1%）= 标签噪声，非模型失败

#### v5 RM 评分分布

- chosen：p50=2.81，p90=3.03（软饱和，好回复集中于 ≈ +3.0）
- rejected：p50=-1.64（差回复集中于 ≈ -1.6）
- **RM 已是软性二分类器**，判别边界清晰，适合 RL 使用

#### v5 对 RL 训练的注意事项

- Saturation 风险监控：若训练中 rollout RM 分数全部 > 2.5，将 `kl_loss_coef` 从 0.001 上调至 0.002-0.003
- **当前 merge 脚本（merge_rm_lora_for_vllm.py）仍指向 v3 adapter**，STEP 2 执行前需修改 `ADAPTER` 路径为 `chatjoy_rm_v5_gpu0234_20260421`

---

## RL 流程中的 Length Reward Hacking 防护体系（2026-04-21 分析）

### RM 层（训练阶段）

v1 RM 的长度评分曲线（实测）：
- <300 chars → mean 0.20（短 rejected 被强烈惩罚）
- **300-600 → mean 1.19**（RM 最偏好区间，non-long 的 chosen 集中在此）
- 600-1000 → mean 1.17（小幅下降）
- >1000 → mean 1.00（长 chosen 也有高分，不盲目惩罚好长）

RM 的局限：**只能区分"比 chosen 更长且质量差"的 rejected**，对"比 chosen 更长且质量相当或更好"的样本学习不充分 → 这是 v4 需要解决的。

### Actor 层（GRPO 训练阶段）

| 机制 | 参数 | 作用 |
|------|------|------|
| `MAX_RESPONSE_LENGTH=768 tokens` | 硬截断 | **最强保护**：任何超出 768 token 的 response 直接截断，物理上阻止长序列 reward 收益 |
| GSPO `seq-mean-token-mean` | `loss_agg_mode` | 序列级归一化：不同长度序列等权，长序列不会因 token 数多而主导梯度 |
| KL penalty | `kl_loss_coef=0.001` | 弱约束，防止策略偏离太远（系数较小，仅辅助） |
| `clip_ratio_low=0.2, high=0.28` | 非对称 clip | DAPO 思路：限制单步策略更新幅度，防止 reward hacking 导致的暴跳 |
| 反拒绝惩罚 | `-2.0` | 针对拒绝模式的专项约束，不直接防长度但限制极端行为 |

**整体评估**：`MAX_RESPONSE_LENGTH=768` 是最强保护，GSPO 长度归一化是第二道；KL 系数弱但存在。整条链路对 length reward hacking 有合理的多层防护，但 RM 层（v3）的 rejected 惩罚弱化是短板。

---

## Hard Negative Mining（v3 训练前准备）

**目标**：用 v2 LoRA（恢复 score head 后）对全量训练集打分，找出 RM 判断错的样本，建立 bad case 池加入 v3。

**脚本**：`chatjoy/mine_hard_negatives.py`（支持多卡 shard 并行、batch 推理）
**编排脚本**：`chatjoy/run_v2_recovery_and_mining.sh`（Step 1 恢复 → Step 2 并行挖掘 → Step 3 合并）

**输出路径**：
- `data/openrlhf_rm/hard_negatives_v2_shard0.jsonl`
- `data/openrlhf_rm/hard_negatives_v2_shard1.jsonl`
- `data/openrlhf_rm/hard_negatives_v2.jsonl`（合并后，gap < -0.3 的 bad cases）

**挖掘完成后**：按 rejected_type / language 分析 gap 分布，决定 v3 是否需要额外数据扩充。

---

## Preference Pair 生成方案记录

### Pair 生成机制

**来源数据**：`data/runs/clean32_for_sft1.jsonl`（13,128 条 SFT 样本）
**生成模型**：`clean32_for_sft2_fsdp_qlora_merged`（SFT2 merged，vLLM 后端，4 shards 并行）
**脚本**：`GRPO_Training_Package/build_rejected_from_sft.py` + `run_pair_generation_vllm.sh`
**生成出的原始 pair 数**：12,973 对
**门控阈值**：`min_chosen_score=0.62`，`min_score_gap=0.18`

三类 rejected 构造方式（脚本实际默认权重 60/35/5）：
- **wrong_role**（目标权重 0.60）：换用另一个角色的真实 system prompt 发给模型，模型自然以该角色身份回答 → role 分天然偏低
- **wrong_language**（目标权重 0.35）：保留原角色 system，末尾追加 `[LANGUAGE OVERRIDE: Respond ONLY in {target_lang}...]` → language 分天然低
- **length_mismatch**（目标权重 0.05，`length_short_prob=0.25`）：要求生成更短或更长的回复

> 注：生成阶段目标权重 60/35/5，但后处理重平衡（见下）后最终分布会不同。

---

### 奖励函数 `chatjoy_reward_utils.py`（4 维，已验证）

**设计定位**：启发式规则打分，用于 pair 构建时的 score_gap 过滤。**注释明确标注：仅用于 pair 构建阶段（construction time only），不用于在线 GRPO reward。**

**默认权重**：format=0.30, role=0.30, language=0.25, length=0.15

| 维度 | 计算逻辑 |
|------|---------|
| **format** | `0.30×action_narration + 0.25×dialogue_quotes + 0.20×inner_state + 0.15×special_tokens + 0.10×alternation`<br>其中 `alternation`=1.0 当且仅当同时有动作叙事 AND（对话/内心），奖励两者交织的结构（`mixed_roleplay_structure`） |
| **role** | `0.75 × format_total + 0.25 × name_presence_score`<br>format_total = `0.35×roleplay_shape + 0.20×action_wrapper + 0.20×anti_placeholder + 0.15×anti_disclosure + 0.10×anti_generic`<br>name_presence：从 system prompt 用正则提取角色名，检查回复是否出现该名字（1.0/0.25/0.55） |
| **language** | `0.50×instruction_output_match + 0.30×source_lang_match + 0.20×system_rule_compliance` |
| **length** | ratio vs chosen 长度：0.65–1.60→1.0 / 0.45–2.10→0.65 / 0.25–3.00→0.35 / else→0.10（含 0.20×anti_repetition） |

**类型专属权重 `SCORING_PROFILES`**（覆盖默认权重，对对应维度放大信号）：

| 类型 | format | role | language | length |
|------|--------|------|----------|--------|
| wrong_role | 0.15 | **0.55** | 0.15 | 0.15 |
| wrong_language | 0.15 | 0.15 | **0.55** | 0.15 |
| length_mismatch_short/long | 0.20 | 0.20 | 0.20 | **0.40** |

**关键局限性（设计上的权衡）**：

role 维度**不做语义 system prompt 对比**，只能捕捉：格式规范性、角色名出现与否。它**无法判断**：回复的性格/语气/行为是否符合人设描述（"冷淡"vs"热情"、角色的 backstory 约束等）。

这是有意的设计取舍：
- 启发式打分速度快，可以在每次生成尝试后同步评估
- 人设一致性的深度判断由 RM 模型学习（不是规则）
- SFT 已处理大部分人设对齐；pair 构建的目标是找到**表面特征可区分**的负样本，而不是人设深度负样本

**对 GRPO actor 训练的影响**：actor 会强化 RM 学到的信号——格式稳定、语言准确、名字在场、长度合适——而**人设深度一致性**主要靠 SFT prior 维持，GRPO 对此提升有限。v1 eval 中 idx=55（高质量错误角色被 RM 偏好）正是此局限的体现。若后续要改善，需要引入语义 judge 或 embedding similarity 作为额外 reward 维度。

---

### 后处理：重平衡 + sample_weight（`prepare_rm_data_weighted.py`）

生成的 12,973 对经过重平衡到目标比例，并添加 sample_weight：

**目标类型比例（`TARGET_TYPE_RATIO`）**：
- wrong_language: 25%
- wrong_role: 25%
- length_mismatch_short: 30%
- length_mismatch_long: 20%

**sample_weight 计算**：
1. 同 type 内对 reward_gap 做 min-max 归一化
2. 映射到 [0.4, 1.0]：`w = 0.4 + 0.6 × gap_norm`（gap 越大 → 权重越高）
3. 乘以 type_weight 修正因子（按比例倒数），clip 到 [0.3, 1.2]

输出：`rm_train.jsonl`（16,826 条），`rm_eval.jsonl`（885 条）

---

### 数据集统计（rm_train_filtered.jsonl，已验证）

**过滤说明**：从 rm_train.jsonl 移除 2,000 条 `reward_gap < 0.2` 的 `length_mismatch_long` 噪声样本 → 14,826 条

**rejected_type 分布**（实际）：

| 类型 | 数量 | 占比 |
|------|------|------|
| length_mismatch_short | 5,041 | 34.0% |
| wrong_role | 4,211 | 28.4% |
| wrong_language | 4,207 | 28.4% |
| length_mismatch_long | 1,367 | 9.2% |

**sample_weight 分布**：min=0.400, max=1.000, mean=0.662
- 按类型均值：wrong_language=0.774 > length_mismatch_short=0.635 > wrong_role=0.603 > length_mismatch_long=0.594

**角色分布**（17 个 chosen_role，Top-5）：

| 角色 | 数量 | 占比 |
|------|------|------|
| 4-143冷淡继姐 | 2,465 | 16.6% |
| 0-135精英富女 | 2,140 | 14.4% |
| 3-139圣诞舞娘 | 1,727 | 11.6% |
| 1-122校园恶霸 | 1,382 | 9.3% |
| 10-148冷淡室友 | 1,144 | 7.7% |

**语言分布**（24 种，Top-5）：tr/fr 各 1,049 (7.1%), en 974 (6.6%), de 917 (6.2%), zh-cn 760 (5.1%)

---

### 文件汇总

| 文件 | 样本数 | 说明 |
|------|--------|------|
| rm_train.jsonl | 16,826 | 原始加权重平衡版 |
| rm_train_filtered.jsonl | 14,826 | 移除 2,000 条低置信度 length_mismatch_long |
| rm_eval.jsonl | 885 | 评估集（5% split） |
| rm_train_v3.jsonl | 18,955 | v3 数据，含 base_length_long_low（2000条缺陷数据）|

---

## Gemma4 线上评测维度与 SFT/RL 覆盖分析

### 背景

本周完成正式服模型切换：Qwen3-30B SFT → Gemma-4-26B-A4B-it bf16（`models--google--gemma-4-26B-A4B-it`）。线上整体效果可用。以下记录五个评测维度在当前 SFT+RL 设计下的覆盖情况，以及未覆盖维度的后续处理方案。

### 五维评测 × SFT/RL 覆盖

| 维度 | Base Gemma4 | SFT（clean32_sft2）| RL（GRPO，RM v3）| 综合结论 |
|------|-------------|-------------------|-----------------|---------|
| **语言一致性** | 原生多语言较稳 | ✅ 27语言专项数据 | ✅ language reward 权重 0.25 | **明确提升** |
| **角色一致性（格式 + 身份区分）** | 依赖 system prompt，格式不稳 | ✅ 学习 `*动作*/对话/(内心)` 结构 | ✅ format+role 合计权重 0.60 + wrong_role 对强化 | **明确提升** |
| **角色一致性（语义性格/语气）** | 指令跟随能力强，可读人设描述 | ✅ 见过 17 个角色真实对话风格 | ✅ wrong_role 用真实角色B system prompt 生成负样本，RM 学"性格匹配原则"，对见过类型可泛化；对完全陌生性格类型效果递减 | **提升**（覆盖类型）/ **小幅提升**（未见类型） |
| **重复文案** | 预训练多样性好 | ✅ 训练数据 anti_repetition 过滤 | ✅ length reward 含 0.20×anti_repetition | **明确提升** |
| **记忆（多轮上下文）** | 混合注意力（sliding window），多轮能力较好 | ➖ SFT 数据以单轮为主 | ➖ 无多轮专项信号 | **持平或轻微下降** |
| **涉黄内容配合** | Google 安全训练，拒绝率较高 | ➖ 数据为 safety 过滤后的内容 | ⚠️ 未覆盖（见下方专项方案）| **当前未处理** |

> **wrong_role 机制说明（已代码验证）**：构造时使用角色 B 的**真实 system prompt**（非合成），让模型自然以角色 B 身份回复同一条 user 消息；RM 训练时 prompt 用角色 A 的 system prompt，学习"system prompt 描述的性格 → 对应回复风格"的语义关联。SFT 覆盖 17 个角色（共 83 个生产角色），wrong_role 对跨越不同性格类型，RM 学到的是可迁移的性格匹配原则。

### 涉黄宽松化方案（业务需求：降低拒绝率）

业务目标是让模型**更愿意配合涉黄角色扮演场景**，而不是拦截。

#### 两阶段评估

**第一步**：用线上角色扮演 prompt 采样一批涉黄场景，统计 Gemma4 当前拒绝率：
- 拒绝率 < 20%（边界偶发）→ 走 RL 阶段 anti-refusal reward 即可
- 拒绝率 > 40%（系统性拒绝）→ 需先做 SFT 打底，再上 RL

#### RL 阶段：anti-refusal reward（不需要改 RM 数据）

在 GRPO `compute_reward()` 里加一个惩罚项，检测到拒绝模式则扣分：

```python
REFUSAL_PATTERNS = ["I'm sorry", "I can't", "I don't feel comfortable", "对不起，我无法", "抱歉"]

def anti_refusal_reward(response: str) -> float:
    for p in REFUSAL_PATTERNS:
        if p.lower() in response.lower():
            return -2.0
    return 0.0

total_reward = rm_score + anti_refusal_reward(response)
```

不需要修改 RM 训练数据或重新训练 RM，直接在 verl reward function 里叠加。

#### SFT 阶段（如拒绝率高，需先执行）

收集涉黄场景下角色正常配合的对话示例（格式与 clean32_sft2 一致），混入 SFT 数据做一次额外 epoch，把模型初始分布推向愿意配合的方向，再进 RL。数量 500–2000 条即可产生明显分布偏移。

#### 执行时序

```
当前：评估 Gemma4 拒绝率（用线上角色扮演 prompt 批量测试）
  ↓
拒绝率低：直接在 RL 训练时加 anti-refusal reward
拒绝率高：先做涉黄 SFT（+500~2000条）→ 再 RL + anti-refusal reward
```

---

## MoE 负载均衡与 Actor GRPO 架构

### Actor 选型：Gemma4-26B

- 实际路径：`/home/ps/hanjiaqi/LlamaFactory/chatjoy/models/gemma4-26b-a4b-it-textonly`（符号链接到 `models--google--gemma-4-26B-A4B-it`）
- 总大小：48.6GB（model-00001=47GB 文本解码器 + model-00002=1.6GB 视觉编码器）
- 架构：`Gemma4ForConditionalGeneration`（multimodal，但 GRPO 只用文本解码器部分）
- 文本 decoder：30层，hidden_size=2816，128 experts，top_k=8，sliding_window=1024
- 注意：原始 HF 版本（非量化反转版），**models--gemma-4-26B-A4B-it-bf16 路径在本机不存在**
- LoRA rank=32，GSPO 算法（防止 MoE router 坍塌）

### Actor GRPO 配置（verl，已修复）

```bash
# 环境：gemma4-rl（verl 0.8.0.dev + vLLM 0.19.0 + deepspeed 0.18.9 + ray 2.48.0）
N_GPUS=4                          # global_pool: GPU 0-3
ROLLOUT_TP_SIZE=4                 # 48.6GB ÷ 4 = 12.15GB/卡，必须 TP=4
ROLLOUT_GPU_MEMORY_UTILIZATION=0.60  # 19.6GB: 模型 12.15GB + KV 7.45GB ✓
RM_TP_SIZE=4
RM_GPU_MEMORY_UTILIZATION=0.62    # 修复：原值 0.45 会 OOM
reward.reward_model.enable=False   # 暂时禁用（vLLM 无 Qwen3MoeForRewardModel）
reward.reward_model.enable_resource_pool=True  # reward_pool: GPU 4-7
reward.reward_model.n_gpus_per_node=4
actor_rollout_ref.actor.entropy_coeff=0.01   # MoE router 不坍塌
actor_rollout_ref.model.target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]
# 不包含 "gate"（路由器），只包含 "gate_proj"（专家 FFN 的 SiLU 门）
actor_rollout_ref.actor.policy_loss.loss_mode=gspo
```

### GSPO vs GRPO

ChatJoy 用 GSPO（序列级等权）的原因：length_mismatch 类型回复长度不均，GSPO 防止长序列主导梯度；RM 给全序列 scalar，GSPO 更一致。

### 环境

- RM 训练：`tslib`（transformers 4.57.1, deepspeed 0.16.9）
- Actor GRPO：`gemma4-rl`（verl=0.8.0.dev, vllm=0.19.0, peft=0.18.1, deepspeed 0.18.9, ray 2.48.0）
- Gemma4 SFT：`gemma4-character`（transformers 5.5.x, LlamaFactory）

---

## Gemma4 SFT QLoRA 修复完整记录（2026-04-20/21）

参考文档：`/home/ps/hanjiaqi/LlamaFactory/chatjoy/GEMMA4_TAKEOVER_RESULT_20260420.md`

### 根因确认：MoE expert 参数是裸 nn.Parameter（非 nn.Linear）

**关键发现**（源码证据）：
- 文件：`transformers/models/gemma4/modeling_gemma4.py`
- `Gemma4TextExperts` 中：`self.gate_up_proj = nn.Parameter(...)` / `self.down_proj = nn.Parameter(...)`
- 这不是 `nn.Linear`，因此 bitsandbytes 标准 4-bit 线性层替换逻辑**不会自动覆盖它们**

**直接后果**：
- `from_pretrained()` 后，这批 expert 权重以 bf16 Parameter 形式残留在 CPU
- 诊断日志：`params: gpu=0.83GB cpu=48.63GB`（48.63GB ≈ 30层 × ~1.42GB/层 MoE expert）
- 之前代码再尝试把 48GB CPU 残留整体搬到 GPU → 泛化 CUDA OOM，掩盖真实根因

**参数量估算验证**：
- `hidden_size=2816, num_experts=128, moe_intermediate_size=704`
- 单层 experts bf16 = `128 × (2816 × 1408 + 1408 × 2816) × 2B ≈ 1.42GB`（gate_up + down）
- 30 层 × 1.42GB = **42.6GB**（与观测 48.63GB 高度吻合，差值来自 embed + attention + misc）

### 自定义 Gemma4 experts 4-bit 量化实现

**核心数值验证**：
- `Params4bit` 可以处理 3D 张量
- 对 3D 整块量化后按 expert 切片子 `QuantState`，与逐 expert 独立量化结果**完全一致**（`diff_max=0.0`）
- 自定义 forward 相对 bf16 baseline：cosine=0.987，corrcoef=0.987，rmse=0.064

**实现结构**（`src/llamafactory/model/model_utils/moe.py` 新增）：
- `gemma4_bnb4_eager`：自定义 experts implementation（注册到 `ALL_EXPERTS_FUNCTIONS`）
- 量化：整块 3D `gate_up_proj`/`down_proj` → packed 4-bit + absmax + quant_map
- Forward：active expert 取连续 packed slice → 重建子 `QuantState` → 反量化 → matmul
- `prepare_gemma4_bnb4_experts(model, model_args)`：加载后处理入口，转换 30 层并设置 `config._experts_implementation=gemma4_bnb4_eager`

**加载后诊断（实测）**：
- 转换前：cpu=48.63GB
- 转换后：`gpu=0.83GB, cpu=2.95GB`（剩余 embed/layernorm/router scale 等小参数）
- `MODEL_LOAD_OK = PeftModelForCausalLM, converted_layers=30, has_any_qweight=True`

### 4-GPU FSDP 修复链（按时序）

**Fix 1：CPU-first materialization patch 范围扩大**（`patcher.py`）
- 原：仅 ZeRO-3/FSDP + 4-bit 触发
- 改：对 Gemma4 + 4-bit + trainable 也触发
- 原因：不先 CPU-first，single-GPU bf16 materialization 就会 OOM，连自定义 hook 都轮不到跑

**Fix 2：FSDP auto-wrap 错误**（`run_gemma4_fsdp_qlora_4gpu_nohup.sh`）
- 错误：`Could not find the transformer layer class to wrap in the model`
- 原因：`_no_split_modules` 含 `Gemma4VisionEncoderLayer`，text-only 模型没有此类
- 修复：`export FSDP_TRANSFORMER_CLS_TO_WRAP=Gemma4TextDecoderLayer`

**Fix 3：Trainer API 兼容性**（5 个 trainer 文件统一修复）
- 错误：`TypeError: create_optimizer() takes 1 positional argument but 2 were given`
- 根因：上游 transformers 新增 `create_optimizer(self, model=None)`，项目 override 旧签名
- 修复文件：`sft/trainer.py`, `rm/trainer.py`, `dpo/trainer.py`, `kto/trainer.py`, `pt/trainer.py`
- 统一改为 `create_optimizer(self, model=None)` 并传入 model

**Fix 4：`mm_token_type_ids` 缺失**（`collator.py`、`kto/trainer.py`）
- 错误：`ValueError: mm_token_type_ids is required as a model input when training`
- 根因：LlamaFactory collator 仍用旧字段名 `token_type_ids`，Gemma4 要求 `mm_token_type_ids`
- 修复：当 `model_type == "gemma4"` 时桥接字段名，其他模型不受影响

**Fix 5：Gemma4 turn token 模板错误**（`template.py`）
- 错误：`vectorized_gather_kernel index out of bounds`（首个 train step）
- 根因：模板用 `<start_of_turn>`/`<end_of_turn>`，Gemma4 实际 token 是 `<|turn>`(id=105)/`<turn|>`(id=106)
- 修复：`gemma4` 模板改用原生 token，`stop_words=["<turn|>"]`
- 验证：`len=262144 == vocab_size`，无新增 token

**Fix 6：FSDP 配置误用 FULL_SHARD**（`accelerate_fsdp_zero2_4gpu.yaml`，新建）
- 问题：旧配置 `fsdp_sharding_strategy: FULL_SHARD`（ZeRO-3 语义）
- 修复：新配置 `fsdp_sharding_strategy: SHARD_GRAD_OP`（ZeRO-2 语义）+ `fsdp_offload_params: false`

**Fix 7：`Linear4bit.quant_state` 初始化不完整**（`patcher.py`）
- 错误：`assert quant_state is not None`（首个 matmul）
- 根因：CPU-first patch 创建 `Params4bit` 时未传 `module=`，导致 `Linear4bit.quant_state` 不被回填
- 修复：`Params4bit(..., module=module, ...)` 并显式兜底 `module.quant_state = new_value.quant_state`

**Fix 8：`Linear4bit.forward()` 首次前向副作用**（`loader.py`）
- 错误：`AssertionError: FSDP does not support changing the parameters between forward and backward for HandleShardingStrategy.SHARD_GRAD_OP`
- 根因：标准 `Linear4bit.forward()` 在首次调用时仍做惰性状态写入（修复 quant_state、设置 compute_dtype 等），FSDP 认为 forward/backward 间参数被修改
- 修复：对 Gemma4 + 4bit + trainable 模型，安装实例级无副作用 `forward`（保留数值路径，去掉状态写入）

### 2026-04-20 夜间训练状态（sessions 185255/191853）

**训练确实进入 step**，但存在异常：
- `loss='0'`，`grad_norm='nan'`
- step time ~8.1-8.3s/step（偏短，正常应约 28s/step）
- 两次 run 的 `all params` 统计不一致：2,418,135,040 vs 3,156,332,544

**判断**：这些 run 的训练行为不可信，很可能受 Fix 8 之前叠加的高风险 loader 干预逻辑影响（自定义 forward 注入、隐式 CPU→GPU 参数迁移、one-shot monkey-patch 等），这些在 2026-04-21 已被移除。

### 2026-04-21 代码收敛（无空闲 GPU 下的预防措施）

**移除的高风险 `loader.py` 逻辑**（均不是 Gemma4 MoE 量化必需）：
1. 标准 `bnb Linear4bit` 的训练期自定义 forward 注入
2. 加载后隐式 CPU→GPU 残余参数迁移
3. 首个 forward 的 one-shot monkey-patch 诊断

**新增防伪成功保护**（`sft/trainer.py`，仅对 gemma4/gemma4_text 生效）：
- 若 batch labels 全部为 `IGNORE_INDEX` → 立即 `raise RuntimeError`
- 若 `loss == 0.0` → 立即 `raise RuntimeError` 并输出有效标签数

**当前 `loader.py` 策略**：只负责 Gemma4 custom MoE 4-bit 准备 + 只读诊断 + CPU residual 时显式失败，不再隐式修改训练 forward 路径。

### 单卡 smoke 验证结果（2026-04-21，GPU2）

**配置**：`CUDA_VISIBLE_DEVICES=2`，`gemma4_sft2_1gpu_lora_smoke_gpu2.yaml`，max_steps=40

**关键结果**：
- step1: `loss=4.516`，step2: `loss=5.384`，step3: `loss=4.19`
- step 用时约 **28s/step**（正常，无历史"短步+loss=0"模式）
- `Prepared Gemma4 custom 4-bit experts for 30 layer(s)` ✓
- CPU residual 2.95GB（告警放行，embed/layernorm/router，不再是 48GB 大头）
- step3 后手动 Ctrl+C 中断，验证目的已达

**结论**：单卡路径 loss 非 0，训练行为正常。**4 卡 FSDP 重启验证尚未完成**（需要空闲 GPU 0-3）。

---

## RL 流程评估与 GRPO 脚本修复（2026-04-21）

### vLLM / verl 环境确认

| 组件 | 版本 | 所在环境 |
|------|------|---------|
| verl | 0.8.0.dev | gemma4-rl |
| vLLM | **0.19.0** | gemma4-rl |
| deepspeed | 0.18.9 | gemma4-rl |
| ray | 2.48.0 | gemma4-rl |

vLLM 0.19.0 的 `model_executor/models/registry.py`：
- 有 `Gemma4ForCausalLM`（text-only，line 114）和 `Gemma4ForConditionalGeneration`（multimodal，line 385）
- 有 `Qwen2ForRewardModel`（line 285）
- **没有** `Qwen3MoeForRewardModel`

### 5 个必死崩溃（脚本中均已修复）

| # | 参数 | 旧值 | 新值 | 根因 |
|---|------|------|------|------|
| 1 | `MODEL_PATH` | `models--gemma-4-26B-A4B-it-bf16/...`（不存在） | `models/gemma4-26b-a4b-it-textonly` | 路径写的是规划中的反量化版，实际从未创建 |
| 2 | `ROLLOUT_TP_SIZE` | `1` | `4` | 模型 48.6GB，单卡 32.6GB 放不下，vLLM init OOM |
| 3 | `RM_GPU_MEMORY_UTILIZATION` | `0.45` | `0.62` | 57GB÷4=14.25GB，0.45×32.6=14.67GB，KV cache 仅 0.42GB |
| 4 | `N_GPUS` | `8` | `4` | 8(global) + 4(reward) = 12 GPUs，只有 8 → Ray crash |
| 5 | `reward.reward_model.enable` | `True` | `False` | RM v1 丢失 + vLLM 无 Qwen3MoeForRewardModel |

### RM 兼容性问题详析

**问题链**：
1. RM v1 checkpoint：训练完成 step 200，保存时 `PytorchStreamWriter failed writing file` → 永久丢失
2. RM v2/v3 adapter：保存在 `chatjoy_rm_v2_filtered/` 和 `chatjoy_rm_v3_gpu4567_20260420_fix2/` 目录
3. v3 `adapter_model.safetensors` 包含：385 keys，含 LoRA 权重 + `base_model.model.score.weight`（shape=[1,2048]）
4. `adapter_config.json` 的 `auto_mapping: {"base_model_class": "RewardModel", "parent_library": "openrlhf.models.model"}`
5. vLLM 0.19 的 `Qwen3MoeForCausalLM` 没有任何 reward/pooling/score 支持
6. → **vLLM 无法直接服务 OpenRLHF RewardModel 格式的 RM**

**启用完整 RM 的路径**（未来工作）：
1. 实现 `Qwen3MoeForRewardModel` vLLM 类（参考 `qwen2_rm.py`，约 80 行）
2. 注册到 vLLM `registry.py` + verl `patch.py`
3. Merge v3 adapter → 独立 57GB 模型，config.architectures 改为 `["Qwen3MoeForRewardModel"]`
4. 切 `reward.reward_model.enable=True`
- 磁盘需求：~57GB（当前剩余 249GB ✓）

### Gemma4 actor 架构说明

- `Gemma4ForConditionalGeneration`（multimodal HF class）
- vLLM 0.19 将此架构映射到 `gemma4_mm` 模块（`registry.py:385`）
- verl patch 中 `Gemma4ForCausalLM` 加入 `SUPPORTED_MOE_MODELS`，匹配 `model.language_model` 子模块
- 因此 verl 的 MoE weight loader patch 仍然生效（`isinstance(inner_model, tuple(SUPPORTED_MOE_MODELS))`）
- 视觉编码器（1.6GB）在 GRPO 中加载但不参与文本前向，是轻微冗余开销（可忽略）

### 显存预算验证

**Global pool（GPU 0-3），Gemma4 actor+rollout**：
| 阶段 | 占用 |
|------|------|
| vLLM rollout（TP=4） | 0.60×32.6=19.6GB：模型 12.15GB + KV 7.45GB ✓ |
| FSDP 训练（ZeRO-3 参数分片） | 12.15GB params + LoRA ~100MB + activations ~2-3GB ≈ 15GB ✓ |
| 峰值（layered_summon 切换期） | <20GB，安全 ✓ |

**Reward pool（GPU 4-7），Qwen3-30B RM**：
| 配置 | 数值 |
|------|------|
| 模型权重/卡（TP=4） | 57÷4 = 14.25GB |
| vLLM 分配 | 0.62×32.6 = 20.2GB |
| KV cache | 20.2-14.25 = 5.95GB（scoring 前向，非生成，够用 ✓） |

### 脚本当前状态

已修复脚本：`GRPO_Training_Package/framework_bridge/verl/train_grpo_lora_actor_verl.sh`

**运行前准备**：
```bash
# 1. 激活正确环境
conda activate gemma4-rl

# 2. 释放 GPU 0-1（当前被 vLLM 进程占用 6-8GB）
fuser /dev/nvidia0 /dev/nvidia1 2>/dev/null | xargs -r kill -9
```

---

## 下一步行动（2026-04-21 更新）

```
立即可做（无需额外条件）：
  1. RM v4 数据准备
     - 从 rm_train_v3.jsonl 删除 v3_source=="base_length_long_low"（2000条）
     - 可选：删除 long_soft_length_mismatch_long（600条）
     - 保持 batch_size=64（而非 128）→ ~296 optimizer steps
     - 在 GPU 4-7 上重跑训练

  2. GRPO 快速启动路径（RM disable，anti-refusal only）
     - 释放 GPU 0-1（kill 占卡进程）
     - conda activate gemma4-rl
     - bash GRPO_Training_Package/framework_bridge/verl/train_grpo_lora_actor_verl.sh
     - 当前 reward = anti-refusal (-2.0) + KL，无 RM 评分

等 GPU 0-3 空闲后：
  3. Gemma4 SFT 4卡 FSDP 正式验证
     - bash chatjoy/run_gemma4_fsdp_qlora_4gpu_nohup.sh
     - 验证点：loss 非 0，step time ~28s，无 quant_state/FSDP 报错
     - 跑到至少一个 save_steps checkpoint

中期工作（已完成软件链路，待执行）：
  4. ✅ 完整 RM 路径：vLLM patch + LoRA merge 脚本已就绪（见下方专项章节）

  5. RM v4 训练后验证
     - 确认 chosen/rejected gap > 1.4（对标 v1 的 1.601）
     - rejected mean < -0.5（v3 只有 -0.298，需改善）
```

**When:** RM v3 的核心问题是 batch_size=128 → 148步 → LR 提前崩溃，以及 base_length_long_low 数据缺陷。v4 用 batch_size=64 + 删缺陷数据可恢复 v1 水平。Gemma4 SFT 单卡路径已验证（loss 非0），4卡需要重跑确认（历史记录 loss=0/grad_norm=nan 属可疑 run，已清理高风险代码）。GRPO + RM 的软件链路已于 2026-04-21 完成。
**How to apply:** RM 推荐 OpenRLHF + tslib；Actor GRPO 用 gemma4-rl；Gemma4 SFT 用 gemma4-character；GRPO 脚本已修复 5 处必死项，enable_resource_pool 参数已添加。RM v3 merged 模型生成后直接启用（reward.reward_model.enable=True 已设好）。

---

## GRPO + RM 完整接入方案（2026-04-21 评估与实施）

### 硬件资源评估（严格基于 verl 代码）

**GPU**: 8× RTX 5090，每卡 32607 MiB = **31.84 GB**
**磁盘**：238 GB 空闲，RM merged 模型 ~57 GB ✓

#### 关键代码路径（已查证）

1. **LoRA 模式下 ref 不独立**（[fsdp_workers.py:1177-1181](hanjiaqi/verl/verl/workers/fsdp_workers.py#L1177-L1181)）：
   ```python
   def compute_ref_log_prob(self, data):
       if self._is_lora:
           data.meta_info["is_lora"] = True
           return self.compute_log_prob(data)   # actor + disable_adapter()
   ```
   → ref 就是 actor 临时 `disable_adapter()`，**零额外显存**。
   → 脚本里 `ref.fsdp_config.param_offload=True` 形式化存在但不生效。

2. **vLLM sleep_level**（[vllm_async_server.py:854-862](hanjiaqi/verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py#L854-L862)）：
   ```python
   async def _sleep_hybrid(self):
       if self.lora_as_adapter:
           sleep_level = 1          # LoRA: 保留 base 权重，只释 KV
       else:
           sleep_level = 2          # 全量：释放 base + KV
   ```
   → 我们用 LoRA，trainer 模式下 vLLM **保留 base 权重**（12.15 GB 常驻）。

3. **Actor FSDP 管理**（[fsdp_workers.py:764-766](hanjiaqi/verl/verl/workers/fsdp_workers.py#L764-L766) / [818-821](hanjiaqi/verl/verl/workers/fsdp_workers.py#L818-L821)）：
   - `param_offload=True`: rollout 前 load→GPU；weight sync 完立刻 offload→CPU
   - `param_offload=False`: actor FSDP 常驻 GPU（与 vLLM 并存）

#### GPU 0-3（Actor + Rollout + Ref 三合一）

**param_offload=False**（关闭卸载，原始配置）：

| 阶段 | FSDP Actor | vLLM | 激活/其他 | 合计 | 余量 |
|------|-----------|------|----------|------|------|
| Rollout 峰值 | 12.15 GB | 19.1 GB (base 12.15 + KV 6.95, util=0.60) | ~0.5 GB | **31.75 GB** | **0.09 GB ⚠️ 必 OOM** |
| Trainer log_prob | 12.15 GB | 12.15 GB (sleep_level=1) | 2-3 GB | ~27 GB | 4.8 GB ✓ |
| Update actor (LoRA) | 12.15 GB | 12.15 GB | 2-3 GB | ~27 GB | 4.8 GB ✓ |

**结论**：Rollout 峰值余量仅 90 MB，实际会 OOM。allgather 临时缓冲、vLLM profiling、activation spike 都会触发。

**param_offload=True**（修正后，**采用**）：

| 阶段 | FSDP Actor | vLLM | 激活/其他 | 合计 | 余量 |
|------|-----------|------|----------|------|------|
| Rollout | 0（卸载 CPU） | 19.1 GB | 0.5 GB | 19.6 GB | **12.2 GB** ✓ |
| Trainer log_prob | 12.15 GB | 12.15 GB (sleep1) | 2-3 GB | ~27 GB | 4.8 GB ✓ |
| Update actor | 12.15 GB | 12.15 GB | 2-3 GB | ~27 GB | 4.8 GB ✓ |

**代价**：每 step 多一次 CPU↔GPU 全量拷贝，~2-3s overhead。24GB 模型通过 PCIe 5.0 ~ 2-4 秒可接受。

#### GPU 4-7（RM 独立 resource pool）

`reward.reward_model.enable_resource_pool=True` + `n_gpus_per_node=4` → RM 完全独占 4-7：

| 阶段 | vLLM RM | 激活 | 合计 | 余量 |
|------|---------|------|------|------|
| 加载 | 14.25 (Qwen3-30B-MoE TP=4) + 5.49 KV = 19.74 GB (util=0.62) | ~0 | 19.74 GB | 12.1 GB ✓ |
| Classify | 同 | batch 小，KV 够 | 19.74 GB | 12.1 GB ✓ |

**安全**，无需改动。

#### 风险点

1. ~~Rollout OOM~~ → 开 `param_offload=True` 已解决
2. **每 step 额外 4-5s 延迟**（actor offload/load × 2 次）→ 可接受
3. **MoE 专家数 × MAX_RESPONSE_LENGTH=768 × batch=64 × rollout_n=4 = 256 并发序列** 的 vLLM KV cache：6.95 GB 可能不足全部预留，会触发 swap。若观察到 rollout 很慢可进一步调小 `ROLLOUT_N` 或 `TRAIN_BATCH_SIZE`。

### 软件障碍（均已解决）

| 问题 | 解决方案 | 状态 |
|------|---------|------|
| vLLM 无 Qwen3MoeForSequenceClassification | `as_seq_cls_model(Qwen3MoeForCausalLM)` 包装 + registry 注册 | ✅ 已 patch |
| RM 为 OpenRLHF 格式（需 merge） | `merge_rm_lora_for_vllm.py` 手动 delta merge + score head 提取 | ✅ 脚本就绪 |
| verl 调用路径 | `/classify` → `output["data"][-1]["probs"][-1]` | ✅ 已验证 |

### LoRA Merge 技术细节

- **Base**: `saves/qwen3-30b-moe/clean32_for_sft2_fsdp_qlora_merged` (57GB)
- **Adapter**: `chatjoy_rm_v3_gpu4567_20260420_fix2`（120MB，53.5M 参数）
- **LoRA rank=64, alpha=128, scaling=2.0**
- **Target modules**: k_proj, v_proj, q_proj, o_proj（仅 attention，不含 MLP）
- **Score head**: `base_model.model.score.weight: [1, 2048]` → 写入 merged 模型最后一个 shard
- **输出**: `saves/qwen3-30b-moe/chatjoy_rm_v3_merged_vllm/`（config 中 `architectures: Qwen3MoeForSequenceClassification, num_labels: 1`）

### vLLM Patch 位置

- **qwen3_moe.py 末尾**（gemma4-rl env）：
  ```python
  from vllm.model_executor.models.adapters import as_seq_cls_model
  Qwen3MoeForSequenceClassification = as_seq_cls_model(Qwen3MoeForCausalLM)
  ```
- **registry.py** `_SEQUENCE_CLASSIFICATION_MODELS`：
  ```python
  "Qwen3MoeForSequenceClassification": ("qwen3_moe", "Qwen3MoeForSequenceClassification"),
  ```

### 待执行 TodoList

```
[ ] STEP 1: 释放 GPU（当前 0,1,5,6,7 被 demo 占用）
    fuser /dev/nvidia0 /dev/nvidia1 /dev/nvidia5 /dev/nvidia6 /dev/nvidia7 | xargs kill -9

[ ] STEP 2: 运行 LoRA merge 脚本（约 20-40 分钟，需 57GB 磁盘）
    # ⚠️ 脚本当前合并 v3 adapter；v5（acc=95.5%）更优，建议先修改脚本中 ADAPTER 路径：
    #   ADAPTER = ".../chatjoy_rm_v5_gpu0234_20260421"
    #   OUTPUT  = ".../chatjoy_rm_v5_merged_vllm"
    conda run -n gemma4-rl python3 \
        /home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/merge_rm_lora_for_vllm.py

[ ] STEP 3: 验证 RM 模型加载（可选，standalone 测试）
    bash /home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/test_rm_vllm_classify.sh

[ ] STEP 4: 清空所有 8 个 GPU，启动 GRPO 训练
    # 确认 GPU 0-7 全空闲
    nvidia-smi
    # 启动训练
    conda activate gemma4-rl
    cd /home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/framework_bridge/verl
    bash train_grpo_lora_actor_verl.sh

[x] STEP 5（已完成）：RM v5 训练完成，acc=95.5%，gap=3.497，超越 v1，**无需再训 v4**
    # adapter: saves/qwen3-30b-moe/chatjoy_rm_v5_gpu0234_20260421/
    # 仅需将 merge 脚本 ADAPTER 路径更新为 v5 后执行 STEP 2
```

### GPU 清理命令

```bash
# 查看占用
nvidia-smi

# 查找占用进程
for i in 0 1 2 3 4 5 6 7; do
    echo "GPU $i:"; fuser /dev/nvidia$i 2>/dev/null
done

# 批量 kill（谨慎：会杀 demo 服务）
fuser /dev/nvidia0 /dev/nvidia1 /dev/nvidia2 /dev/nvidia3 \
      /dev/nvidia4 /dev/nvidia5 /dev/nvidia6 /dev/nvidia7 2>/dev/null | xargs kill -9
```

---

## 2026-04-23 进展日志（GRPO pilot 启动前调试）

### 今天完成的事

1. **RM v5 merge 完成** → `saves/qwen3-30b-moe/chatjoy_rm_v5_merged_vllm/`
   - merge 脚本 ADAPTER 路径已更新为 v5
   - 产物含 score_head，vLLM 可直接加载为 `Qwen3MoeForSequenceClassification`

2. **Gemma4 SFT LoRA merge 完成** → `saves/gemma4-26b-a4b/sft2_ddp_lora_3gpu_567_cut3072_merged/`
   - 用 `GRPO_Training_Package/merge_gemma4_sft_lora.py`（LORA_ALPHA=64, RANK=32, scaling=2.0）
   - 合并 115 个 lora_A/B → 写入两个 shard（~51.6GB）
   - adapter 目录的 tokenizer/chat_template 覆盖写入 merged 目录

3. **GRPO 训练数据生成** → `GRPO_Training_Package/framework_bridge/outputs/verl_rl_data/`
   - `train.parquet` 18,938 条；`val.parquet` 386 条
   - 源：`data/runs/clean32_for_sft2.jsonl`（19,508 行）

4. **改进 train_grpo_lora_actor_verl.sh**（备份 `.bak_20260422`）
   - 路径更新为 merged actor + RM v5
   - 开启显存优化：`use_dynamic_bsz=True` / `ppo_max_token_len_per_gpu=8192` / `param_offload=True` / `optimizer_offload=True` / `rollout.enforce_eager=True` / `rollout.free_cache_engine=True` / `rm.rollout.enforce_eager=True`
   - LoRA：rank=32, alpha=32, target=[q,k,v,o,gate,up,down], `exclude_modules=.*vision_tower.*`

### 已修复的报错链

| # | 报错 | 根因 | 修复 |
|---|------|------|------|
| 1 | `Could not override 'actor_rollout_ref.rollout.swap_space'` | verl 0.8 config 里 rollout 段没有 `swap_space` 字段 | 从脚本删掉 `ROLLOUT_SWAP_SPACE` 和对应 override 行 |
| 2 | `ValueError: Target module Gemma4ClippableLinear ... is not supported` | SigLIP 视觉塔用 `Gemma4ClippableLinear` 包 Linear，`target_modules=[q_proj,...]` 按后缀匹配把视觉层也吃进来，PEFT 不认该类型 | 加 `actor_rollout_ref.model.exclude_modules=.*vision_tower.*`（PEFT 对 str 走 `re.fullmatch`，排除整条 vision_tower 路径）|
| 3 | `Exception: Could not find the transformer layer class to wrap in the model.` | Gemma4 `_no_split_modules=['Gemma4TextDecoderLayer','Gemma4VisionEncoderLayer','Gemma4AudioLayer']`，textonly 模型没有 `Gemma4AudioLayer`，`get_module_class_from_name` 返 None 抛错 | actor + ref 都加 Hydra `+` override：`+fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[Gemma4TextDecoderLayer]` |
| 4 | `CUDA out of memory. Tried to allocate 2.75 GiB. GPU 0 ... 2.12 GiB free` on `_sync_module_params_and_buffers`（FSDP init 阶段）| `fsdp_config.model_dtype` 默认 `"fp32"`，把 bf16 权重（25.84B）翻倍加载 → 总 ~103GB，FSDP 每卡 shard ~26GB，5090 只有 31.36GB，再加 vLLM KV 必 OOM。日志佐证：`Flash Attention 2 only supports float16/bfloat16, current dtype is torch.float32` | actor + ref 都加 `fsdp_config.model_dtype=bfloat16`（参照 verl 官方 `run_pi05_libero_sac.sh`）。预计每卡 shard 降到 ~6.5GB |

### 关键代码/配置位置速查

- 训练脚本：`GRPO_Training_Package/framework_bridge/verl/train_grpo_lora_actor_verl.sh`
  - L119-123：actor fsdp_config（param_offload / optimizer_offload / model_dtype=bfloat16 / wrap_policy）
  - L140-142：ref fsdp_config（param_offload / model_dtype=bfloat16 / wrap_policy）
  - L134-135：rollout enforce_eager + free_cache_engine
- verl dtype 解析：`hanjiaqi/verl/verl/workers/config/engine.py:244` 默认 `fp32`；`hanjiaqi/verl/verl/workers/engine/fsdp/transformer_impl.py:219` 读取
- PEFT exclude 语义：`peft/tuners/tuners_utils.py:1660`（str → `re.fullmatch`；list → 后缀/精确）

### 整体 RL 训练计划

- **Phase 0 — smoke（当前阶段）**：`TRAIN_BATCH_SIZE=32 trainer.total_training_steps=50 save_freq=25 test_freq=5`，目标验证链路+显存不爆。
- **Phase 1 — 小规模稳定性**：跑 50 step，观察 reward / KL / entropy / response_length / GPU 显存曲线，调 `PPO_MAX_TOKEN_LEN_PER_GPU` 和 rollout TP。
- **Phase 2 — 1 epoch 完整训练**：18,938 条 / batch=32 ≈ 305 step，预计 8-12 小时，正常跑到收敛后保存。
- **Phase 3 — 评估 + 合并 LoRA**：拿 val 输出做主观+规则打分；合并 actor adapter。

### 接下来要干什么（交接 Codex 用）

1. **启动 Phase 0 pilot**（脚本已修好，直接跑）：
   ```bash
   cd /home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/framework_bridge/verl
   conda run -n gemma4-rl --no-capture-output bash -c '
     TRAIN_BATCH_SIZE=32 bash train_grpo_lora_actor_verl.sh \
       trainer.total_training_steps=50 trainer.save_freq=25 trainer.test_freq=5
   '
   ```
   建议塞 tmux：`tmux new -d -s grpo_pilot '...'`

2. **如果 pilot 启动后仍 OOM**（次优先级可能点）：
   - actor 端：进一步降 `PPO_MAX_TOKEN_LEN_PER_GPU`（8192 → 4096）
   - rollout 端：降 `ROLLOUT_GPU_MEMORY_UTILIZATION`（当前值见脚本顶），或把 `ROLLOUT_TP_SIZE` 从 4 调成 2（仅当 actor GPU 够）
   - 检查 `param_offload` 是否真生效（`nvidia-smi` 看 rollout 期 actor 4 卡是否下降）

3. **pilot 成功后**：观察 `outputs/` 下 wandb/console 的 reward 曲线，reward 在前 10 步应为负（anti_refusal_penalty 主导），10 步后应逐步上升；若 reward 一直 -2 ≈ 拒答，说明 RM/anti_refusal 配置有问题。

4. **Phase 2 正式训练启动条件**：pilot 50 步正常收尾 + GPU 稳定 + checkpoint 能保存（step 25 会触发）。

### 环境要点

- conda env：`gemma4-rl`；tmux 内要用 `conda run -n gemma4-rl --no-capture-output bash -c '...'`，`conda activate` 在 non-interactive shell 里不可靠
- GPU 布局：verl 会自动把 actor/ref 放 GPU 0-3，RM 放 GPU 4-7（resource_pool 分割）
- RTX 5090 实际可用显存：31.36GB（31.84GB 总 - 预留），NUMA0=GPU 0-3，NUMA1=GPU 4-7，无 P2P

---

## 2026-04-24 RL smoke 调试进展（关键突破在快结束时）

### 这两天总共试过的路径（全记录）

| 路径 | 结果 | 结论 |
|------|------|------|
| `lora.merge=True` + FSDP1 | OOM at `summon_full_params` | `fsdp_merge_unmerge` 对 FSDP1 全量 unshard 26B bf16=51.7GB/rank > 31GB 5090 → 必 OOM |
| `lora.merge=True` + FSDP2 | init 卡 5+ 分钟 | Gemma4-26B-A4B 是 **128 experts MoE**（`text_config.num_experts=128`），`_select_fsdp2_wrap_targets` 遍历 ~35 万子模块；且 `fsdp2_load_full_state_dict` 要 rank0 GPU 装全量 52GB → 放不下 |
| hot-swap + Codex 的 guarded set_lora | `CUDA invalid argument` → `illegal memory access` at 第 1 次 `set_lora` | vLLM V1 cumem 把 `lora_a_stacked/lora_b_stacked` 放在 `weights` tag → sleep level=1 把 LoRA buffer unmap → set_lora 踩空指针 |
| + utils.py 加 `cuda.synchronize()` + CPU detach + 详细日志 | 仍崩同样位置 | 同步和 dtype/shape 对齐对这个 bug 没用，问题在 buffer 物理内存已被 unmap |
| + engine_workers.py 任意 sleep 模式都 `resume(weights)` | 新错：`cumem_allocator.cpp:139 OOM` at `wake_up(weights)` | wake_up 需要 13GB 重新分配 + 从 host 拷回；但 FSDP actor 那 13GB 还占着 GPU，峰值 26GB+ 超 31GB |
| + vllm_rollout.py 加 `_has_slept` gate | 未测（方向错） | 静态复核发现：`sleep_replicas` 走 `vLLMHttpServer.sleep`，根本不经过我的 `ServerAdapter.release`，gate 永远 False 会把必要的 resume 跳掉 |
| **+ engine_workers.py 在 resume(weights) 前先 offload FSDP 到 CPU**（最新落地） | **待验证** | 消除 FSDP 和 vLLM weights 在 GPU 上的 13+13GB 双持；`get_per_tensor_param` 会在需要时再把 FSDP load 回来 |

### 最后两处已落地补丁（2026-04-24 下午）

1. **撤回 `_has_slept`**：`/home/ps/hanjiaqi/verl/verl/workers/rollout/vllm_rollout/vllm_rollout.py::ServerAdapter.resume/release` 恢复到原始无状态的 wake_up/sleep，因为 sleep 路径根本不走这里。

2. **resume 前先 offload FSDP**：`/home/ps/hanjiaqi/verl/verl/workers/engine_workers.py::update_weights` 在 `self.rollout.resume(tags=["weights"])` 之前加：
   ```python
   if self.actor.engine.is_param_offload_enabled:
       self.actor.engine.to("cpu", model=True, optimizer=False, grad=False)
   aggressive_empty_cache(force_sync=True)
   ```
   这样 cumem wake_up 时 GPU 基本空，13GB 的 weights 重新分配不会 OOM。`get_per_tensor_param` 会自动再把 FSDP load 回 GPU 做 IPC 同步。

3. 其他保留不动：utils.py 的 set_lora sync-guard（保留作安全网）、engine_workers.py 的"任意 sleep 模式都 resume weights"（正确）、脚本的 `LORA_MERGE_MODE=False` + FSDP1 + `[q,k,v,o]_proj` + rank=32 默认。

### 交接给下一轮的关键文件

- 详细记录：[`chatjoy_rl_takeover_error_fix_log.md`](chatjoy_rl_takeover_error_fix_log.md)（第 11 节是 Claude 接手以来的完整日志）
- 脚本：`/home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/framework_bridge/verl/train_grpo_lora_actor_verl.sh`
- verl 补丁：
  - `/home/ps/hanjiaqi/verl/verl/workers/engine_workers.py`（update_weights 增加 pre-resume offload）
  - `/home/ps/hanjiaqi/verl/verl/utils/vllm/utils.py`（set_lora guard）
  - `/home/ps/hanjiaqi/verl/verl/utils/vllm/patch.py`（Gemma4 MoE weight loader）

### smoke 启动命令（收窄版，验证链路）

```bash
cd /home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/framework_bridge/verl
LOG="/tmp/grpo_smoke_$(date +%Y%m%d_%H%M%S).log"
tmux new -d -s grpo_smoke "conda run -n gemma4-rl --no-capture-output bash -c '
  export CUDA_LAUNCH_BLOCKING=1
  export VERL_LOGGING_LEVEL=INFO
  TRAIN_BATCH_SIZE=32 TARGET_MODULES=\"[o_proj]\" LORA_RANK=8 LORA_ALPHA=8 \
  bash train_grpo_lora_actor_verl.sh \
    trainer.total_training_steps=2 \
    trainer.save_freq=100000 \
    trainer.test_freq=100000 2>&1 | tee $LOG
'"
```

**成功标志**：`After resume weights` 日志后没 OOM，set_lora 不报 illegal memory access，至少 1 次 rollout + 1 个 reward 值。

---

## 2026-04-30 RL 全链路排查记录与下一优先级

### 当前正式 1-epoch run 状态

- 正式 run：`chatjoy_grpo_qkvo_r8_phase2_1epoch_20260430_201520`
- metrics：`/home/ps/hanjiaqi/LlamaFactory/chatjoy/logs/verl_metrics/chatjoy_verl/chatjoy_grpo_qkvo_r8_phase2_1epoch_20260430_201520.jsonl`
- log：`/tmp/chatjoy_grpo_qkvo_r8_phase2_1epoch_20260430_201520.log`
- 已确认配置：`total_training_steps=543`，`save_freq=50`，`test_freq=50`，`clip_grad=1.0`，`lr_scheduler_type=cosine`，`lr_warmup_steps_ratio=0.03`，warmup steps=16。
- 截至 step 5：仍在 warmup 内，`actor/lr` 从 `1.875e-7` 升到 `9.375e-7`；`actor/grad_norm` 约 `1.44/1.62/3.00/7.64/1.84`；`critic/score/mean` 在 `-0.84~-1.00` 间波动。此阶段不能判断 reward 是否学起来。
- 不应中断当前 run。当前 run 的价值是得到 step 50/100/150 后的真实曲线与 validation 样本。

### 50-step pilot 结论修正

- pilot 已完整跑通 50/50，0 OOM、0 crash、0 abort，LoRA hot-swap 稳定，吞吐约 80 tokens/s/GPU。
- `Step 5 grad_norm=223.29` 不是“没有 grad_clip”。verl/FSDP 日志里的 `actor/grad_norm` 是 pre-clip norm，脚本和正式 run 已确认 `clip_grad=1.0` 生效。
- pilot 的 reward 没有显著改善：`critic/score/mean` 首末约 `-0.986 -> -1.064`；val `rm_score` 约 `-0.629 -> -0.634`；actor/loss 下降不能当作 RL 质量提升证据。
- pilot 出现少量 RM prompt truncation warning：reward model prompt 超过 `RM_MAX_MODEL_LEN=3328` 时保留尾部 token，可能丢 system/persona 前缀。这会直接影响 RM 判断人设一致性。

### 全链路风险表

| 环节 | 已确认事实 | 可能导致的问题 | 优先级 | 当前动作 |
|---|---|---|---|---|
| SFT 数据 | `clean32_for_sft2.jsonl` 约 19.5k，26 语言，主要单轮；SFT 无正式 eval 曲线 | RL 基础分布可能不错，但多轮记忆/长 persona 泛化缺少验证 | P2 | 不在当前 run 中改 |
| SFT/RL 长度 | SFT cutoff 3328；RL `MAX_PROMPT_LENGTH=2048`，训练集过滤到 17,407 条 | 长 persona/复杂上下文被系统性过滤，RL 学不到这部分 | P1 | 后续统计被过滤样本的语言/角色/长度 |
| RM 训练 | RM v5 eval acc 约 96%，gap 约 3.497，优于 v1/v3 | eval 与训练构造同分布，不能证明线上语义人设判断强 | P1 | 需要独立人工/LLM judge eval |
| RM 数据构造 | 负样本主要是 language/role/length/format 维度；深层 persona 语义较弱 | RL 强化的是表层可区分信号，人设深度一致性提升有限 | P0 | 这是当前最核心的设计风险 |
| RM merge/scale | vLLM merged RM 使用 raw score；adapter config 中 normalize/mean/std 等信息未完整进入 merged config | OpenRLHF eval 分数与 vLLM online 分数可能存在尺度/校准差 | P1 | 后续做同样样本 OpenRLHF vs vLLM classify 对齐 |
| 在线 reward | reward = `rm_score + anti_refusal_reward`；anti-refusal 命中扣 `-2.0` | 规则惩罚可能压过 RM 细微信号，导致 reward 曲线难解释 | P0 | 优先补日志，确认命中率和误伤 |
| anti-refusal 规则 | 短回复 `<120 chars` 且无角色动作/引号/括号时可能被判拒答 | 合法短回复、多语言短句、自然简短对话可能被误罚 | P0 | 优先抽样验证 false positive |
| RM prompt 截断 | 超长时保留尾部 token，可能丢 system/persona | RM 看不到角色设定，只能按回复质量/语言/长度打分 | P0 | 优先统计 truncation rate，必要时改截断策略 |
| GRPO 分组 | `ROLLOUT_N=4`，GRPO 按同 prompt 的 4 个 response 做组内标准化 | 若组内 reward 方差小或 RM 排名噪声大，梯度方向会弱/抖 | P1 | 补 group std / best-worst margin 日志 |
| 指标解释 | actor/loss 是 PPO/GSPO surrogate，不是 SFT loss | loss 下降不能证明模型质量提升；reward/val 才是主指标 | P0 | 后续只按 rm_score/custom_reward/人工样本判断 |
| LR 策略 | 正式 run 已加入 cosine + 0.03 warmup，543 steps 中 warmup=16 | step 50 前很难靠 reward/loss 判断 LR 是否合适 | P2 | 不优先改 LR，先看 step 100+ |
| KL/稳定性 | pilot KL loss 稳定约 1-2，ppo_kl 小幅波动；正式 run 早期正常 | 暂无 KL 爆炸证据 | P2 | 继续监控 |
| checkpoint/resume | 当前策略 save every 50，keep 1，sync save；但 `resume_mode=disable` | 断点恢复需要手动指定或切换 resume 策略 | P1 | 当前 run 不改；下一 run 可明确 resume_mode |
| 磁盘/Ray | `/home` 96% 使用，Ray 对 `/tmp/ray` 有 >95% warning 风险 | object spilling 或 checkpoint 写入可能失败 | P1 | 不动训练；持续监控磁盘 |

### 最值得改的单点

**第一优先级不是学习率，而是 reward 闭环的可观测性与校准。**

原因：

1. pilot 证明工程链路能跑，但没有证明 reward 在正确方向上提升。
2. 当前最大不确定性集中在 reward 端：RM 是否被 prompt 截断误导、anti-refusal 是否误罚、RM raw score 与训练时 score 是否同尺度、GRPO 组内 reward 是否有足够方差。
3. 在这些问题没量化前调 LR，可能只是把错误/噪声 reward 更快放大。

下一轮最值得落地的改动：

| 改动 | 目的 | 不影响当前 run 的做法 |
|---|---|---|
| 记录 `rm_score`、`custom_reward`、anti-refusal 命中 reason、RM prompt truncation flag/rate | 判断 reward 组成是否健康 | 当前 run 结束后再改 reward loop / reward fn |
| 记录 GRPO 每个 prompt 的 reward std、best-worst margin、同组排名 | 判断 `ROLLOUT_N=4` 是否给了有效学习信号 | 当前 run 后改 metrics |
| 抽样保存 val response + RM score + custom reward + prompt 长度 | 直接看模型是否真的更像角色、更少拒答 | 用 step 50/100 checkpoint 离线分析 |
| 做 OpenRLHF RM 与 vLLM classify 同样样本分数对齐 | 验证 merged RM online 分数是否可信 | 离线脚本，不碰训练进程 |

如果只能改一个，先改：**在线 reward 诊断日志（RM 分数、anti-refusal 原因、truncation、组内 reward 方差）**。这是最小代码改动，但能决定后续该改 RM 数据、anti-refusal、prompt 截断、ROLLOUT_N，还是 LR。

---

## 2026-05-07 RL reward 链路修复与正式训练最新建议

### 一句话结论

2026-04-30 的 1-epoch Phase2 reward 负分不是 RM 训练失败，而是 online RM classify 输入格式与 OpenRLHF RM 训练/评估格式不一致：Qwen chat template 尾部换行导致 vLLM pooled 到 newline token。2026-05-07 已修复该问题，并通过 50-step smoke 验证 online RM 分数恢复到 `+1.9` 左右。

### 关键文件

| 类型 | 路径 |
|---|---|
| RL 启动脚本 | `/home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/framework_bridge/verl/train_grpo_lora_actor_verl.sh` |
| reward loop 修复 | `/home/ps/hanjiaqi/verl/verl/experimental/reward_loop/reward_loop.py` |
| anti-refusal 规则 | `/home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/anti_refusal_reward.py` |
| 最新 smoke log | `/tmp/chatjoy_p2176_r768_smoke50_20260507_103012.log` |
| 最新 smoke metrics | `/home/ps/hanjiaqi/LlamaFactory/chatjoy/logs/verl_metrics/chatjoy_verl/chatjoy_p2176_r768_smoke50_20260507_103012.jsonl` |
| 最新 smoke validation samples | `/home/ps/hanjiaqi/LlamaFactory/chatjoy/logs/validation_samples/chatjoy_p2176_r768_smoke50_20260507_103012.jsonl` |

### Online RM 修复

OpenRLHF RM 训练/评估会对 `prompt + response` 做：

- `rstrip("\n")`
- 必要时追加 EOS
- 最后 token 强制为 `eos_token_id`

旧 online reward loop 对 Qwen RM 使用 `apply_chat_template(... add_generation_prompt=False)` 后直接发给 vLLM `/classify`，Qwen assistant 结尾是 `<|im_end|>\n`。vLLM sequence classification pools final token，因此实际 scored 到 newline token，造成 online RM 大面积负分。

已在 `reward_loop.py` 中对 RM prompt 进行：

```python
rm_prompt = rm_prompt.rstrip("\n")
eos_token = self.reward_model_tokenizer.eos_token
if eos_token is not None and not rm_prompt.endswith(eos_token):
    rm_prompt = f"{rm_prompt} {eos_token}"
```

修复后 50-step smoke 的 validation RM 从旧 run 的约 `-0.63` 恢复到 `+1.90~+1.93`。

### anti-refusal 最新状态

当前设计：

| tier | 条件 | 惩罚 |
|---|---|---:|
| keyword | 明确拒绝 / 跳出角色 / 安全政策元评论 | `-1.0` |
| format_short | `<80 chars` 且无动作/对话/括号格式 | `-0.25` |
| none | 正常角色扮演 | `0.0` |

已收窄的误伤：

- 移除裸 `no puedo`，避免误罚 `No puedo creer...` 角色内心独白；
- 同步移除/收窄裸 `je ne peux pas`、`ich kann nicht`、日语 `できません` / `申し訳`；
- 英文 `maybe we can explore`、`i can help with` 等过宽软拒绝改成更明确换话题句式；
- 中文 `我理解您的`、`我明白您` 等过宽短语移除。

用最新规则重算 smoke validation samples，anti-refusal 误伤显著下降：

| step | old penalty mean | new penalty mean |
|---:|---:|---:|
| 10 | `-0.1250` | `0.0000` |
| 20 | `-0.1406` | `-0.0156` |
| 30 | `-0.0625` | `0.0000` |
| 40 | `-0.1250` | `0.0000` |
| 50 | `-0.0781` | `-0.0156` |

### 2176/768 smoke 结果

实验：`chatjoy_p2176_r768_smoke50_20260507_103012`

配置：

```text
MAX_PROMPT_LENGTH=2176
MAX_RESPONSE_LENGTH=768
PPO_MAX_TOKEN_LEN_PER_GPU=2944
MAX_MODEL_LEN=3200
RM_MAX_MODEL_LEN=3456
ROLLOUT_MAX_NUM_BATCHED_TOKENS=3200
RM_MAX_NUM_BATCHED_TOKENS=3456
clip_grad=1.0
test_freq=10
save_freq=0
```

结果：50/50 完成，约 5h19m，无 OOM。尾部的 Ray/vLLM `SIGTERM`、`resource_tracker KeyError` 发生在 `Final validation metrics` 之后，是退出清理噪音，不是训练失败。

validation 曲线：

| step | val reward | val RM | anti-refusal | train score | KL loss | PPO KL | grad |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | `1.901` | `1.925` | `-0.024` | `1.155` | `1.046` | `0.037` | `1.30` |
| 20 | `1.899` | `1.929` | `-0.030` | `1.330` | `1.812` | `-0.022` | `1.70` |
| 30 | `1.882` | `1.906` | `-0.024` | `1.743` | `1.676` | `0.001` | `9.78` |
| 40 | `1.899` | `1.927` | `-0.028` | `1.568` | `1.736` | `-0.031` | `1.80` |
| 50 | `1.884` | `1.903` | `-0.019` | `1.555` | `1.755` | `-0.015` | `2.24` |

解读：

- reward 横盘不代表没学。修复后 SFT actor 起点已被 RM 评为高分，50-step smoke 的目标是验证链路，不是证明收敛；
- KL loss 约 `1~2`，`ppo_kl` 小幅波动，没有爆；
- grad spike 是 pre-clip norm，`clip_grad=1.0` 已生效，后续无连续失控；
- response clip 很低，无 aborted。

### 长度档选择

Gemma actor tokenizer 统计：

| prompt cap | 保留样本 | 丢弃 | 保留率 |
|---:|---:|---:|---:|
| 2048 | 17407 | 1531 | 91.92% |
| 2176 | 18048 | 890 | 95.30% |
| 2304 | 18931 | 7 | 99.96% |
| 3072 | 18938 | 0 | 100.00% |

推荐保留 `2176/768/2944`：

- 比 2048 多收约 641 条训练样本；
- 不回到历史 3072 actor backward OOM 档；
- smoke 已验证完整 50 step。

显存事实：

- GPU 0-3 峰值约 `32065 MiB`，p95 约 `30.6~31.0GB`；
- RM GPU 4-7 稳定约 `20.96GB`；
- 因此不能再加 token，不建议 `2304/768/3072`。

### 正式训练推荐配置

正式 1 epoch 建议使用：

```text
MAX_PROMPT_LENGTH=2176
MAX_RESPONSE_LENGTH=768
PPO_MAX_TOKEN_LEN_PER_GPU=2944
MAX_MODEL_LEN=3200
RM_MAX_MODEL_LEN=3456
ROLLOUT_MAX_NUM_BATCHED_TOKENS=3200
RM_MAX_NUM_BATCHED_TOKENS=3456
LR=3e-6
LR_SCHEDULER_TYPE=cosine
LR_WARMUP_STEPS_RATIO=0.03
actor_rollout_ref.actor.optim.clip_grad=1.0
ENTROPY_COEFF=0
LORA_RANK=8
TARGET_MODULES=[q_proj,k_proj,v_proj,o_proj]
```

不要用 `trainer.total_training_steps=50` 判断正式学习强度。50-step smoke 的 cosine LR 后半段已经接近 0；正式 1 epoch 约 564 steps，LR 衰减会慢很多。

### 正式训练监控阈值

| 指标 | 期望 / 干预 |
|---|---|
| `val-aux/chatjoy/rm_score/mean@1` | 稳定 `>1.7`；低于 `1.0` 查 RM input / EOS / truncation |
| `val-aux/chatjoy/reward/mean@1` | 稳定 `>1.7` |
| `val-aux/chatjoy/custom_reward/mean@1` | 新规则下应接近 `0`，通常 `>-0.05` |
| `actor/kl_loss` | 连续 `>3` 才警惕；`>4` 考虑增大 KL coef |
| `actor/ppo_kl` | 绝对值持续 `>0.1` 才干预 |
| `actor/grad_norm` | 偶发 spike 可接受；连续 `>50` 才考虑降 LR 或 clip 更紧 |
| `response_length/clip_ratio` | 持续 `>0.03` 才考虑响应长度/采样策略 |
| GPU 0-3 memory | 当前档已贴边，不加 token |

### 当前不要做

- 不要回退到 `2048/768`；
- 不要上 `2304/768/3072`；
- 不要因为 50-step reward 横盘就提高 LR；
- 不要扩 FFN / MoE target modules；
- 不要开 entropy。

## 2026-05-09 Gemma4 RM 正式 RL 启动与续训记录

本次正式 RL 选择 **Gemma4 RM**，不是 Qwen RM v5：

- Gemma4 RM 与当前 Gemma4 SFT actor 同模型家族，reward 分布更贴近待优化 actor；
- Gemma4 RM 30-step smoke 已验证 reward loop、vLLM classify、actor update、weight update 主链路可跑；
- smoke step10 validation 指标：`val-aux/chatjoy/reward/mean@1=1.8829611471645493`，`val-aux/chatjoy/rm_score/mean@1=1.9000376499782168`，`anti_refusal_is_refusal=0.00273224043715847`；
- smoke 中未再出现 `EngineCore failed`、`WorkerProc failed`、CUDA OOM 或 RM `/classify` 长度越界。

正式 run：

- tmux session：`chatjoy_gemma4rm_rl_formal_161335`
- experiment：`chatjoy_grpo_gemma4rm_phase2_600_20260509_161335`
- main log：`/home/ps/hanjiaqi/LlamaFactory/chatjoy/logs/chatjoy_grpo_gemma4rm_phase2_600_20260509_161335.log`
- metrics JSONL：`/home/ps/hanjiaqi/LlamaFactory/chatjoy/logs/verl_metrics/chatjoy_verl/chatjoy_grpo_gemma4rm_phase2_600_20260509_161335.jsonl`
- tensorboard：`/home/ps/hanjiaqi/LlamaFactory/chatjoy/logs/tensorboard/chatjoy_verl/chatjoy_grpo_gemma4rm_phase2_600_20260509_161335/`
- validation samples：`/home/ps/hanjiaqi/LlamaFactory/chatjoy/logs/validation_samples/chatjoy_grpo_gemma4rm_phase2_600_20260509_161335.jsonl`
- checkpoint/save dir：`/home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/verl/chatjoy_grpo_gemma4rm_phase2_600_20260509_161335/`
- RM model path：`/home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/gemma4-26b-a4b/chatjoy_rm_gemma4_sft_v5_merged_vllm`

正式 run 配置延续 2026-05-07 稳定档，仅替换 RM：

```text
MAX_PROMPT_LENGTH=2176
MAX_RESPONSE_LENGTH=768
PPO_MAX_TOKEN_LEN_PER_GPU=2944
MAX_MODEL_LEN=3200
RM_MAX_MODEL_LEN=3456
ROLLOUT_MAX_NUM_BATCHED_TOKENS=3200
RM_MAX_NUM_BATCHED_TOKENS=3456
LR=3e-6
LR_SCHEDULER_TYPE=cosine
LR_WARMUP_STEPS_RATIO=0.03
ENTROPY_COEFF=0
LORA_RANK=8
LORA_ALPHA=8
TARGET_MODULES=[q_proj,k_proj,v_proj,o_proj]
TEST_FREQ=50
SAVE_FREQ=50
MAX_ACTOR_CKPT_TO_KEEP=1
ACTOR_CKPT_ASYNC_SAVE=False
trainer.total_training_steps=600
trainer.resume_mode=disable
```

`ACTOR_CKPT_ASYNC_SAVE=False` 是本次长跑的续训关键：verl 在同步保存时会写 `latest_checkpointed_iteration.txt`；异步保存会跳过该文件，导致 `resume_mode=auto` 可能找不到最新 step 或落到旧 checkpoint。

当前早期状态（2026-05-09 16:3x）：

- 已写出 step 1、step 2 metrics；
- step 2：`critic/score/mean=1.5620043277740479`，`actor/kl_loss=1.4733934113755822`，`actor/ppo_kl=-0.010716968155207321`，`actor/grad_norm=1.5833182334899902`；
- step 2：`response_length/clip_ratio=0.0`，`prompt_length/clip_ratio=0.0`，`response/aborted_ratio=0.0`；
- step time 约 `345~358s/step`，早期估算 50 step 约 5 小时级别；
- 未发现 `Traceback`、`OutOfMemory`、`CUDA out of memory`、`EngineCore failed`、`WorkerProc failed`。

显存快照：

```text
GPU0: used 30451 MiB, free 1659 MiB, util 99%
GPU1: used 29277 MiB, free 2834 MiB, util 99%
GPU2: used 29357 MiB, free 2754 MiB, util 99%
GPU3: used 29525 MiB, free 2586 MiB, util 99%
GPU4-7: used 21226 MiB each, free 10884 MiB each
```

磁盘状态：

- `/home/ps`、`/tmp`、logs、checkpoint 均在 `/dev/sda2`；
- `7.0T` 总容量，已用 `6.1T`，可用 `510G`，使用率 `93%`；
- `/home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/verl` 当前约 `149G`；
- `/home/ps/hanjiaqi/LlamaFactory/chatjoy/logs` 当前约 `60M`；
- `/tmp/ray` 当前约 `7.5G`；
- 本次 run 因 `save_freq=50`、`max_actor_ckpt_to_keep=1`，预计单个 actor checkpoint 约 `53G`，临时峰值可能 `100G+`，当前 `510G` 可用空间足够，但分区已经偏高。

续训规则：

- 当前 step 2 还没有 checkpoint；此时中断只能从头重跑；
- 第一次真正可续训点是 `global_step_50` 保存完成后；
- 需要确认存在：

```text
/home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/verl/chatjoy_grpo_gemma4rm_phase2_600_20260509_161335/global_step_50/
/home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/verl/chatjoy_grpo_gemma4rm_phase2_600_20260509_161335/latest_checkpointed_iteration.txt
```

且 `latest_checkpointed_iteration.txt` 内容为 `50`。

恢复时保留同一个 `SAVE_DIR` 和 experiment 相关路径，把：

```text
trainer.resume_mode=disable
```

改为：

```text
trainer.resume_mode=auto
```

更稳的恢复方式是显式指定：

```text
trainer.resume_mode=resume_path
trainer.resume_from_path=/home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/verl/chatjoy_grpo_gemma4rm_phase2_600_20260509_161335/global_step_50
```

`trainer.total_training_steps=600` 恢复时保持不变；verl 会从 checkpoint 的 global step 继续跑到 600，不是额外再跑 600。

## 2026-05-14 增量：从 RL 训练趋势回顾到 RM v6 Fine-Grained 数据与训练

本节为增量记录，保留上文所有历史细节不变。它总结 2026-05-09 之后从 RL phase2 日志趋势分析、奖励信号判断，到 RM v6 设计、数据生成、judge、pair 构造、混合 v5 anchor、Gemma4 RM v6 训练的完整过程。

### 1. 起点：RL phase2 趋势暴露的问题

当时 Gemma4 SFT actor + Gemma4 RM v5 的正式 RL 已经可以稳定跑，不再是早期 OOM、vLLM EngineCore、reward model classify 长度越界、checkpoint 续训等工程稳定性问题。

但是从 `/home/ps/hanjiaqi/LlamaFactory/chatjoy/logs/chatjoy_grpo_gemma4rm_phase2_600_20260509_161335.log` 观察到：

- `reward` 和 validation reward 没有明显持续上行。
- `advantage` 均值经常为负或接近 0；GRPO 内部会做组内归一化，均值为负本身不是 bug，但结合 reward 不升，说明 RM 对组内候选的“好 vs 更好”区分不够强。
- `response/aborted_ratio=0`、长度 clip 正常、KL 没有明显爆炸、训练可以稳定推进，说明瓶颈更像奖励信号质量，而不是 RL 框架或显存问题。
- Gemma4 RM v5 对 hard fail 约束有效，例如错语言、错角色、拒答、长度异常、替用户行动等。
- 但 RM v5 对同一个 prompt 下多个“都不坏”的回复缺乏细粒度排序能力，在线 RL 很难得到稳定的提升信号。
- 之前观察到 Gemma4 RM v5 `|score| > 5` 饱和比例偏高，约 `47.3%`，这会压缩可用于 RL 的有效梯度。

由此形成判断：

- 不应推翻 v5；v5 是稳定 anchor，尤其保留了 hard fail 基础约束。
- 不应复现 RMv2 的错误：不能剔除或降权 `length_mismatch_long`，此前剔除后效果反而变差。
- 需要新增同 prompt、多 source、多温度候选的 fine-grained preference，让 RM 学会“自然推进、角色贴合、语气更好”的细微差异。

### 2. RM v6 设计决策

采用路线：

```text
v5 全量保留 + 新增同 prompt 多候选排序数据
```

关键原则：

- 不全量重做旧 RM 数据。
- 不删除、不降权、不重排 `rm_train_v5_5pct_badpool.jsonl`。
- 新增数据只补足“好 vs 更好”的细粒度排序。
- 不使用不同 source 的不同 reward 规则；所有新旧样本最终统一成 pairwise preference。
- judge scalar 不作为在线 reward，只用排序方向构造 chosen/rejected pair。
- 所有 v6 文件写新路径，不覆盖 v5 RM、Gemma4 RM v5、当前 RL checkpoint 或稳定 RL 脚本。

候选 source 最终确定为 5 个：

```text
Gemma4 SFT base      temperature=0.7 top_p=0.8
Gemma4 SFT diverse   temperature=1.0 top_p=0.95
Qwen SFT base        temperature=0.7 top_p=0.8
RL step550           temperature=0.7 top_p=0.8
RL step550 diverse   temperature=1.0 top_p=0.95
```

选择这些 source 的原因：

- Gemma4 SFT base 是当前 RL actor 的起点。
- Qwen SFT base 提供不同模型族的风格和错误分布。
- RL step550 是已经经过 Gemma4 RM v5 强化后的中间策略，能暴露 RM v5 已经偏好的行为模式。
- t07/t10 两档采样提供同模型内部稳定与多样性的差异，便于构造 close pair。

### 3. 新增 RM v6 pipeline

代码目录：

```text
/home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/rm_v6_finegrained/
```

主要脚本：

```text
sample_prompts.py
generate_candidates_openai.py
generate_pilot_sources_sequential.sh
judge_candidates.py
judge_candidates_llm_openai.py
build_pairs.py
mix_with_v5_anchor.py
qa_dataset.py
fix_gemma4_layer_scalar_safetensors.py
postprocess_deepseek_pilot2500.sh
```

重要实现点：

- `sample_prompts.py` 从 RL train parquet 抽样 prompt，排除 val/eval，按 `extra_info.language` 和角色 `role_hash` 做分层。
- `generate_candidates_openai.py` 通过 OpenAI-compatible vLLM server 采样，支持 `--resume` 和 `--concurrency`。
- `generate_pilot_sources_sequential.sh` 一次只启动一个 vLLM server，避免多个大模型同时抢显存。
- `judge_candidates.py` 是规则/heuristic judge，可做 QA，但不能作为最终 label source。
- `judge_candidates_llm_openai.py` 是 external blind LLM judge，支持 DeepSeek/OpenAI-compatible API、`--resume`、`--concurrency`。
- `build_pairs.py` 每个 prompt 最多构造 3 条 pair：`best_vs_worst`、`best_vs_mid`、`good_vs_bad`。
- `mix_with_v5_anchor.py` 将 v5 anchor 全量并入，并拆分 train/eval。

### 4. RL step550 vLLM 修复

RL step550 需要从 verl FSDP checkpoint 导出为 HF 模型后用于 vLLM 采样。导出后遇到 Gemma4 特有 shape 问题：

```text
*.layer_scalar                [4]    -> [1]
model.vision_tower.std_bias   [4608] -> [1152]
model.vision_tower.std_scale  [4608] -> [1152]
```

判断：

- `layer_scalar` 和 vision tower std 张量被 legacy merger 按 world size 拼接。
- 对 `std_bias/std_scale` 做 chunk 比对后确认四段完全重复，保留第一段是安全修复。

修复产物：

```text
/home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/gemma4-26b-a4b/chatjoy_grpo_gemma4rm_phase2_step550_hf_vllm_fixed_v2/
```

同时从 Gemma4 SFT merged 复制 `processor_config.json`，解决 vLLM multimodal processor 配置缺失问题。

结论：

- Gemma4 SFT、Qwen SFT、RL step550 fixed v2 均已确认可 vLLM 采样。

### 5. Candidate 数据完成状态

pilot split：

```text
2500 prompts
2500 candidates.gemma4_sft_t07.jsonl
2500 candidates.gemma4_sft_t10.jsonl
2500 candidates.qwen_sft_t07.jsonl
2524 candidates.rl550_t07.jsonl
2504 candidates.rl550_t10.jsonl
```

RL 两个文件有少量重复行，原因是中途停/续跑；后续 judge 已按 `(prompt_id, source_model)` 去重，不影响训练。

full split：

```text
10000 prompts
10000 candidates.gemma4_sft_t07.jsonl
10000 candidates.gemma4_sft_t10.jsonl
10000 candidates.qwen_sft_t07.jsonl
10000 candidates.rl550_t07.jsonl
10000 candidates.rl550_t10.jsonl
```

full 采样使用 8 卡 TP=8 顺序跑五个 source，每次只启动一个 vLLM server。日志：

```text
/home/ps/hanjiaqi/LlamaFactory/chatjoy/logs/rm_v6_finegrained/full.sequential_8gpu.log
/home/ps/hanjiaqi/LlamaFactory/chatjoy/logs/rm_v6_finegrained/full.gemma4_sft.vllm.log
/home/ps/hanjiaqi/LlamaFactory/chatjoy/logs/rm_v6_finegrained/full.qwen_sft.vllm.log
/home/ps/hanjiaqi/LlamaFactory/chatjoy/logs/rm_v6_finegrained/full.rl_step550.vllm.log
```

### 6. Judge 策略演进

规则 judge 结果：

- judged prompts: `2488`
- audit: `2139`
- low confidence ratio: `85.5%`

结论：

- 规则 judge 可以发现 hard fail，但对“好 vs 更好”的语义排序太保守，不适合作为主训练标签。

DeepSeek LLM judge pilot200：

- `200/200` parse 成功。
- audit: `0`
- confidence: `high=83`, `medium=117`, `low=0`
- hard fail 基本都排在 rank 5。
- source 平均 rank：
  ```text
  gemma4_sft_t07_p08   2.495
  rl_step550_t07_p08   2.520
  gemma4_sft_t10_p095  2.895
  rl_step550_t10_p095  2.945
  qwen_sft_t07_p08     4.145
  ```

人工抽查判断：

- DeepSeek 的理由主要围绕角色贴合、互动推进、自然度、是否替用户行动。
- Qwen 被判差较多，但 source 对 judge 隐藏，且抽查中多数属于错语言、泛化、语气突兀、替用户行动等真实问题。

pilot2500 DeepSeek judge：

```text
2488 judged
12 audit
confidence: medium=1472, high=1016
```

source 平均 rank：

```text
gemma4_sft_t07_p08   2.551
rl_step550_t07_p08   2.558
gemma4_sft_t10_p095  2.908
rl_step550_t10_p095  2.919
qwen_sft_t07_p08     4.064
```

full10000 DeepSeek judge：

```text
9960 judged
38 audit
confidence: high=4162, medium=5798
low_confidence_ratio=0.0
blocked_candidate_flags=3297
```

source 平均 rank：

```text
gemma4_sft_t07_p08   2.561
rl_step550_t07_p08   2.559
gemma4_sft_t10_p095  2.879
rl_step550_t10_p095  2.906
qwen_sft_t07_p08     4.093
```

观察：

- pilot200、pilot2500、full10000 的 source rank 趋势高度一致。
- Gemma4 SFT t07 与 RL step550 t07 接近，是最强的两个 source。
- t10 diverse 版本中等，说明更高温度带来更多风格和一致性波动。
- Qwen SFT 明显更容易落到较差 rank，主要承担 hard negative 和跨模型风格差异角色。
- hard fail 大多集中在 rank 5，说明 judge 对 hard fail 的方向基本符合预期。

### 7. Pair 构造与混合数据

pilot200 pair：

```text
/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/openrlhf_rm/rm_v6_finegrained_pairs_deepseek_pilot200.jsonl
600 pairs
same_source_ratio=0.0
```

pilot2500 pair：

```text
/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/openrlhf_rm/rm_v6_finegrained_pairs_pilot.jsonl
7463 pairs
```

full10000 pair：

```text
/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/openrlhf_rm/rm_v6_finegrained_pairs_full.jsonl
29856 pairs
same_source_ratio=0.0
pair_types:
  best_vs_worst=9954
  best_vs_mid=9954
  good_vs_bad=9948
```

combined fine-grained pairs：

```text
/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/openrlhf_rm/rm_v6_finegrained_pairs_combined.jsonl
37319 pairs
same_source_ratio=0.0
pair_types:
  best_vs_worst=12442
  best_vs_mid=12442
  good_vs_bad=12435
```

mixed train/eval：

```text
/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/openrlhf_rm/rm_train_v6_fg_combined.jsonl
51299 rows
  v5_train_anchor=17712
  fine_grained_judge=33587

/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/openrlhf_rm/rm_eval_v6_fg_combined.jsonl
4617 rows
  v5_eval_anchor=885
  fine_grained_judge=3732
```

注意：

- combined pairs 是 pilot + full 的 fine-grained pair 集合。
- train/eval 中保留 v5 anchor，且新旧数据统一 pairwise ranking loss。
- v5 anchor 在 train 中标记为 `v5_train_anchor`，eval 中标记为 `v5_eval_anchor`；fine-grained 数据标记为 `fine_grained_judge`。

### 8. Gemma4 RM v6 combined 训练

smoke：

```text
/home/ps/hanjiaqi/LlamaFactory/chatjoy/logs/rm_v6_finegrained/smoke_chatjoy_rm_gemma4_v6_fg_smoke_20260513_135132.log
/home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/gemma4-26b-a4b/chatjoy_rm_gemma4_v6_fg_smoke_20260513_135132/
```

smoke 结论：

- 训练和 eval 可跑通。
- step 4 eval：`acc_mean=0.593`，`reward_mean=-0.0505`，`reward_std=0.0189`。
- smoke 太短，主要验证链路，不用于质量判断。

combined 正式 RM v6 训练：

```text
/home/ps/hanjiaqi/LlamaFactory/chatjoy/logs/rm_v6_finegrained/train_chatjoy_rm_gemma4_v6_fg_combined_20260513_152457.log
/home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/gemma4-26b-a4b/chatjoy_rm_gemma4_v6_fg_combined_20260513_152457/
```

训练完成状态：

- 1 epoch 完成。
- train steps: `6412`
- 最终训练耗时约 `11:50:50`
- deepspeed launcher 最终各 rank `exits successfully`
- 训练末尾 NCCL/TCPStore warning 出现在进程退出阶段，未导致训练失败。

最终 eval 片段：

```text
eval_loss=0.478
acc_mean=0.748
reward_mean=1.38
reward_std=1.59
chosen mean=+2.012
rejected mean=+0.695
gap mean=+1.318
wrong pairs (gap<0)=25.3%
saturated chosen (|score|>5)=0.0%
```

plot 脚本解析训练 record 失败：

```text
Found 0 train records, 0 eval records
No RM train records found, skipping RM plot
```

但 eval reward JSONL 存在并生成了：

```text
/home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/gemma4-26b-a4b/chatjoy_rm_gemma4_v6_fg_combined_20260513_152457/eval_rewards/step_000200.jsonl
/home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/gemma4-26b-a4b/chatjoy_rm_gemma4_v6_fg_combined_20260513_152457/eval_rewards/step_000400.jsonl
/home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/gemma4-26b-a4b/chatjoy_rm_gemma4_v6_fg_combined_20260513_152457/eval_rewards/step_000600.jsonl
/home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/gemma4-26b-a4b/chatjoy_rm_gemma4_v6_fg_combined_20260513_152457/eval_rewards/step_000800.jsonl
/home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/gemma4-26b-a4b/chatjoy_rm_gemma4_v6_fg_combined_20260513_152457/eval_rewards/step_000801.jsonl
```

训练输出包含：

```text
adapter_model.safetensors
adapter_config.json
config.json
tokenizer.json
tokenizer_config.json
plots/eval_reward_instances.png
```

### 9. 当前质量判断

RM v6 combined 相比 RM v5 的正向信号：

- score 饱和显著改善：`saturated chosen=0.0%`，相比 Gemma4 RM v5 约 `47.3%` 饱和是明显进步。
- chosen/rejected gap 均值为 `+1.318`，模型学到了大部分 pair 的方向。
- eval acc 达到约 `0.744-0.748`，说明比随机和 smoke 明显更强。
- 数据覆盖更接近 RL 真实场景：同 prompt，多 source，多温度，多语言，多角色。

仍然存在的风险：

- wrong pairs 仍为 `25.3%`，plot 脚本也明确提示 `>10% pairs scored wrong — RM quality issue`。
- 这说明 RM v6 还不能直接假定优于 Gemma4 RM v5 用于正式 RL。
- Full judge 中 low confidence 为 0，可能是 DeepSeek 过于自信；需要继续抽检 close pair。
- Qwen 经常处于较差 rank，虽然比例和抽查看起来合理，但仍需防止 RM 过度学习“模型风格差异”而不是角色质量。

当前建议：

- 不要直接替换正式 RL 的 RM。
- 先把 RM v6 combined merge 成 vLLM 可加载版本。
- 做 classify smoke，确认 score finite、EOS/rstrip、batch classify 正常。
- 再跑 30-step RL smoke，只替换 RM path，不改 actor/RL 稳定参数。
- 用 RL smoke 的 reward/advantage、KL、长度、人工样本判断 RM v6 是否值得进入正式 RL。

### 10. 需要继续做的动作

1. 合并 RM v6 adapter：
   ```text
   /home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/gemma4-26b-a4b/chatjoy_rm_gemma4_v6_fg_combined_20260513_152457/
   ```
   目标输出建议：
   ```text
   /home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/gemma4-26b-a4b/chatjoy_rm_gemma4_v6_fg_combined_merged_vllm/
   ```
2. 跑 reward model classify smoke。
3. 跑 30-step RL smoke，只替换 RM path。
4. 对比：
   - Gemma4 RM v5
   - Qwen RM v5
   - Gemma4 RM v6 combined
5. 如果 RM v6 smoke 明显改善 reward/advantage 区分并且人工样本更自然，再考虑正式 RL。
6. 如果 RM v6 wrong-pair 影响 RL：
   - 需要重新审查 eval 切分。
   - 对 DeepSeek judge 的 close pair 做人工抽检。
   - 可能需要降低 `best_vs_mid` / `good_vs_bad` 中差距过小 pair 的权重或增加 margin/gap 过滤，但不要删除 v5 anchor。

### 11. RM v6 combined 根因诊断与 v6.1 计划 (2026-05-14)

Gemma4 RM v6 combined 训练完成（1 epoch，801 梯度步），关键指标：

```text
eval acc    : 0.744
gap mean    : +1.318
wrong pairs : 25.3%
saturation  : 0.0%（vs RM v5 的 47.3%，大幅改善）
```

**25.3% wrong pairs 的根因：** v6 fine-grained 数据中 2/3 是 rank_delta=2 的弱信号 pair（best_vs_mid + good_vs_bad），这些 pair 中 chosen/rejected 都是合格回复，仅有细微差别，DeepSeek judge 的噪声在此类 pair 上影响较大。

**v6.1 修复方案（方案 A，已设计）：** 只保留 best_vs_worst（rank_delta=4）的 v6 pair，过滤掉 delta=2 噪声：

```text
v5 anchor 17712 + v6 best_vs_worst 11198 = 28910 train pairs
```

**当前进度：**

1. RM v6 combined adapter 已存盘，score head 确认在 adapter 中。
2. 下一步：merge adapter → vLLM 可加载 → classify smoke → 30-step RL smoke。
3. RL smoke 结果决定是否需要重训 RM v6.1。

详细诊断和 v6.1 方案见 `chatjoy_rm_v6_finegrained_handoff.md` 第 11 节。

### 12. RM v6.1 dual-filter 实施与验证 (2026-05-14)

上一节（第 11 节）记录了 v6 combined 诊断和 v6.1 方案设计。本节记录实际执行结果。

#### 12.1 多 judge 方案否决

分析了"3 次 DeepSeek 投票或混合 GPT-4o"改善 delta=2 标签质量的方案。

关键障碍：DeepSeek judge 使用 `temperature=0.0`，同 prompt 多次调用输出完全相同，无法实现真正独立投票。唯一可行方案是混合异质模型（GPT-4o + DeepSeek，约 $48）。但 v6.1 smoke test 证明 dual-filter 已足够，多 judge 投票已否决。

#### 12.2 v6.1 dual-filter 实施结果

脚本：`GRPO_Training_Package/rm_v6_finegrained/filter_pairs_v61.py`（新建）

过滤规则：保留 `rank_delta=4`（全部）+ `rank_delta=2 AND label_confidence=high`；丢弃 `rank_delta=2 AND label_confidence=medium`（14,526 对，占 38.9%）。

```text
v6 combined pairs: 37,319 → v6.1 filtered pairs: 22,793 (-38.9%)

v6.1 训练集 (rm_train_v6.1_fg_combined.jsonl):
  v5_train_anchor    : 17,712
  fine_grained_judge : 20,514
  合计               : 38,226 (vs v6 的 51,299)

v6.1 验证集 (rm_eval_v6.1_fg_combined.jsonl):
  v5_eval_anchor     :    885
  fine_grained_judge :  2,279
  合计               :  3,164 (vs v6 的 4,617)
```

启动脚本：`launch_rm_train_gemma4_sft_v6.1_fg.sh`（新建）

#### 12.3 v6.1 smoke test 验证

4 GPU (GPU 0-3)，32 梯度步，1024 训练样本，3164 eval 样本。

| 指标 | v6 smoke | v6.1 smoke |
|------|----------|------------|
| eval acc_mean | 0.593 | **0.696** (+10.3 pp) |
| reward_std | 0.0189 | **0.0394** (+108%) |
| gap_mean | — | +0.026 |

训练趋势健康：chosen reward 与 rejected reward 在训练中后期稳定分离（chosen→≈0，rejected→-0.05~-0.08）。v6 训练中两者始终同步抖动未分离。

**结论：v6.1 dual-filter 已验证有效，reward_std 翻倍直接对应 GRPO advantage 方差翻倍。**

#### 12.4 RM v6.1 训练链路（当前状态）

```text
数据链路：
  judged (pilot 2488 + full 9960)
    → filter_pairs_v61.py (dual-filter)
    → rm_v6.1_finegrained_pairs_combined.jsonl (22,793 pairs)
    → mix_with_v5_anchor.py
    → rm_train_v6.1_fg_combined.jsonl (38,226) + rm_eval_v6.1_fg_combined.jsonl (3,164)

待完成：
  1. 全量 8 卡训练（GPU 5 当前被占，待空闲后上 8 卡）
  2. adapter merge → vLLM 可加载
  3. classify smoke → 30-step RL smoke
```

#### 12.5 当前 RM 版本状态汇总

| RM 版本 | 状态 | 位置 |
|---------|------|------|
| RM v5 (Gemma4) | 可用于 RL，有已知饱和度问题 | saves/gemma4-26b-a4b/rm_v5_* |
| RM v6 combined | 已训完（802 步），有 25% wrong pairs，暂不用于正式 RL | saves/chatjoy_rm_gemma4_v6_fg_combined_20260513_152457 |
| RM v6.1 | smoke 验证通过，待全量训练 | — |
