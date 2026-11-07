# 遗珠底账 —— project_chatjoy_overview.md 深挖全量记录

> 2026-07-07 通读 `character-rlhf/docs/project_chatjoy_overview.md`（2318+ 行）后整理的**未进学习体系**（阅读学习指导 / 深读 / GLOSSARY）的学习点全量底账，共 60 条、五组。
> 列说明：**精选✅** = 已补进 `阅读学习指导_character-rlhf.md` 对应段落的"遗珠"行（约 26 条）；其余留在本底账备查。
> ⚠️ overview 已于 2026-07-07 当天稍晚**整体重排**（时间正序 + 补全 I/II/III 三章 + 文末四快照 + 「版本总账」章节），本表的 L__ 行号是**重排前**快照、已全部失效——**定位一律用节名**（节名未变）。另：重排后新增三条遗珠 A12/C17/D12（见各组表尾）。
> **版本命名**：项目已统一 `RM-N/SFT-N/RL-N`（RM-6.3/RL-6.3/SFT-6.9c）+ 双轨（训练号 vs 产品交付号 v1/v2/…）；本底账引用版本时若用旧 v6.x，对照 overview「版本总账」换算。关键：v6.4/6.5 是 RL 占 RM 号；`rl_full_v3/4/5`=RL-6.3 重启计数非版本号；产品号只 v1/v2 坐实。

---

## A. Reward 可观测性与校准（11 条）

| # | 学习点（含数字） | overview 出处 | 关联引用 | 挂靠段 | 精选 |
|---|---|---|---|---|---|
| A1 | **"只能改一个，先改在线 reward 诊断日志"**（RM 分/anti-refusal 原因/truncation 率/组内 reward 方差）——可观测性优先于调 LR：没量化前调参只是把噪声 reward 更快放大 | 「最值得改的单点」L1581-1600 | 深读 9（归因链的工程前置） | ⑨d | ✅ |
| A2 | 04-30 **全链路风险表**：14 个环节 P0-P2 分级，P0 全部落在 reward 端（RM 数据表层化 / reward 可观测性 / anti-refusal 误伤 / RM prompt 截断 / 指标解释） | 「全链路风险表」L1562-1579 | 深读 9 | ⑪ | ✅ |
| A3 | **online RM 负分根因 = chat template 尾部换行被 vLLM pool 到 newline token**；`rstrip("\n")`+补 EOS 修复后 val RM **−0.63→+1.90** | 「2026-05-07 RL reward 链路修复」L1604-1640 | fix log；深读 2 | ⑧ | ✅ |
| A4 | anti-refusal 误伤收窄方法：删裸 `no puedo` 等多语词表陷阱（"No puedo creer..." 内心独白被误罚），用重算 penalty mean 验证（step10 −0.125→0.000） | L1642-1667 | 深读 2 §6 | ⑧ | |
| A5 | **监控阈值表**：rm_score 稳定>1.7；kl_loss 连续>3 才警惕、>4 才加 KL coef；\|ppo_kl\|持续>0.1 才干预；grad_norm 连续>50 才动 LR；clip_ratio>0.03 才动长度 | 「正式训练监控阈值」L1753-1764 | 深读 3 看板 | ⑨d | ✅ |
| A6 | `actor/grad_norm` 是 **pre-clip norm**（223.29 ≠ 没 clip）；actor loss 是 surrogate，**下降不能当质量证据**——只按 rm_score/custom_reward/人工样本判断 | L1555-1559、风险表"指标解释"行 L1575 | 深读 3 | ⑨d | ✅ |
| A7 | **"reward 横盘不代表没学"**：修复后 SFT 起点已被 RM 评高分；50-step smoke 的目的=验证链路而非收敛 | L1700-1705 | — | ⑩ | ✅ |
| A8 | **RM prompt 截断保尾会丢 system/persona 前缀** → RM 看不到人设、只按回复表面打分（风险表 P0） | L1560、L1573 | 深读 2 | ⑨c | ✅ |
| A9 | truncation rate 是必须统计的量（pilot 已出现少量 warning 即入 P0） | L1560 | 并入 A8 | ⑨c | |
| A10 | OpenRLHF eval 分数与 vLLM online 分数可能有尺度/校准差（merged config 未带 normalize/mean/std）→ 同样本双引擎对齐验证 | 风险表 L1570 | — | ⑧ | |
| A11 | 组内 reward std / best-worst margin 日志：判断 ROLLOUT_N=4 是否给出有效学习信号（GRPO 组内方差小 → 梯度弱/抖） | L1574、L1596 | 深读 8 group 层 | ⑨d | |
| A12 | **Goodhart 修复是两步**：06-02/03 anti-OOC 硬罚 −5 + 撤 bonus（对付事故本体，46 tests）→ 06-04 否定/引号豁免（对付修复自身误罚好行为，49 tests，误报 1/16→0/16）——**修复也会引入新误伤，需要第二轮验证** | overview 时间线补全 II（源：handoff 06-02/06-04 节） | CASE_STUDY §3.1；WORKFLOW 4.2 | ⑧ | ✅（已入 ⑧ 正文 Goodhart 段） |

## B. RM 数据构造方法论（16 条）

| # | 学习点（含数字） | overview 出处 | 关联引用 | 挂靠段 | 精选 |
|---|---|---|---|---|---|
| B1 | 构造期启发式 reward（4 维）注释明标 **construction time only、不进在线 GRPO**——防规则分泄漏成优化目标（Goodhart 预防的边界设计） | 「奖励函数 chatjoy_reward_utils.py」L830-832 | 深读 2 | ⑤ | ✅ |
| B2 | **role 维度不做语义对比是有意取舍**（速度/与 RM 分工/只找表面可区分负样本），代价=人设深度靠 SFT prior；v1 eval **idx=55**（高质量错角色被 RM 偏好）即此局限现形 | L851-860 | 深读 2 | ⑤ | ✅ |
| B3 | **wrong_role 用角色 B 的真实 system prompt**（非合成）：RM 学"system prompt 性格 → 回复风格"的可迁移匹配原则；17/83 角色覆盖靠此泛化 | L822、L941 | 深读 2 | ⑤ | ✅ |
| B4 | sample_weight = 0.4+0.6×gap_norm，×type 修正 clip[0.3,1.2]；生成权重 60/35/5 与重平衡后 25/25/30/20 是两个阶段的数 | L864-877、L821-826 | 深读 2 | ⑤ | |
| B5 | Hard Negative Mining 流程：v2 恢复 score head 后全量打分，gap<−0.3 入 bad case 池 → v5 的 5% badpool 来源 | L795-807 | VERSION_MAP RM v5 行 | ⑧ | |
| B6 | RM 长度评分曲线实测：<300 chars→0.20 / 300-600→1.19 / >1000→1.00；"RM 只能区分比 chosen 更长且更差的 rejected" | 「Length Reward Hacking 防护」L771-779 | 深读 2 | ⑧ | |
| B7 | **47.3% 的定义 = Gemma4 RM v5 \|score\|>5 的饱和比例**——压缩 RL 有效梯度的量化刻画 | L1897、L2299 | VERSION_MAP；深读 2 | ⑧ | ✅ |
| B8 | RM v6 设计五原则：**v5 全量当 anchor 不推翻**／不复现 RMv2 剔除错误（剔 length_mismatch_long 反而变差）／新数据只补"好 vs 更好"／**judge scalar 不作在线 reward、只用排序方向**／新文件新路径不覆盖 | L1899-1920 | 深读 2 | ⑧ | ✅ |
| B9 | 5 个 candidate source 的选择理由：Gemma4 SFT=actor 起点／Qwen SFT=跨模型族错误分布／**RL step550=暴露 RM 已偏好的行为**／t0.7+t1.0 双档构造 close pair | L1922-1937 | 深读 2 | ⑫ | ✅ |
| B10 | **规则 judge low-confidence 85.5% → 只配做 QA、不配当 label source**；外部盲测 LLM judge 接棒 | 「Judge 策略演进」L2034-2044 | 深读 2 | ⑧ | ✅ |
| B11 | **跨规模 rank 一致性**是 judge 信号稳定的判据：pilot200/pilot2500/full10000 三档 source 平均 rank 高度一致（如 gemma4_t07 2.495/2.551/2.561） | L2047-2110 | — | ⑧ | |
| B12 | v6 combined 两面性：饱和 **47.3%→0.0%** 大改善，但 **wrong pairs 25.3%**（>10% 即 RM quality issue）→ **不假定 v6 优于 v5**，先 classify smoke + 30-step RL smoke 再定 | 「当前质量判断」L2245-2267 | rm_v6_finegrained_handoff | ⑧ | ✅（与 B13 合并） |
| B13 | 25.3% wrong pairs 根因：2/3 是 rank_delta=2 弱信号 pair，judge 噪声在 close pair 上放大 → v6.1 方案 A 只留 best_vs_worst（delta=4）：17712+11198=28910 | L2291-2308 | handoff §11 | ⑧ | ✅（并入 B12） |
| B14 | pair 三档构造：best_vs_worst / best_vs_mid / good_vs_bad（每 prompt 最多 3 条）+ same_source_ratio=0.0 | L1969、L2129-2150 | — | ⑧ | |
| B15 | "DeepSeek low confidence=0 可能是过于自信"——对 judge 校准保持怀疑，close pair 需人工抽检 | L2258、L2288 | — | ⑧ | |
| B16 | Qwen source 常居差 rank：承担 hard negative/跨模型风格差异角色，但要防 RM 学成"模型风格判别器"而非"角色质量判别器" | L2109、L2259 | — | ⑫ | |

## C. 显存与工程排障（16 条）

| # | 学习点（含数字） | overview 出处 | 关联引用 | 挂靠段 | 精选 |
|---|---|---|---|---|---|
| C1 | **CE over 262k 词表才是 OOM 真因**（O(seq×vocab) 线性、非注意力）；cutoff 6144→4096 只截 4.9% 样本/0% 砍 output；flash/sdpa/Liger 都不解、Liger 不支持 gemma4 | 06-16 状态节 L36 | 深读 1；FRAMEWORK_RUNTIME_LESSONS | ⑥ | ✅ |
| C2 | **transformers 5.x × ZeRO-3 两个不兼容**：ThreadPoolExecutor 异步加载→NCCL 顺序死锁；initialize_weights_zero3 每模块单独 allgather **~25,000 次** → RM 训练锁 tslib 环境 | 「环境选择：tslib」L392-398 | FRAMEWORK_RUNTIME_LESSONS | ⑥ | ✅ |
| C3 | ZeRO-2+4bit+LoRA+adam_offload 的显存构成账（~20GB/卡<<32GB）；ZeRO-3 实测 step0 卡死（48 block×1.2GB allgather + Flash JIT 叠加超时推测） | L405-418 | 深读 1（已部分覆盖） | ⑥ | |
| C4 | "4-bit 与 OpenRLHF 不兼容"是 ZeRO-3 时代注释的误传——**错误认知要挂时代戳**，防止旧结论污染新决策 | L420-422 | — | ⑥ | |
| C5 | DS 完整 ckpt 14-57GB 写崩磁盘（**RM v1 step200 永久丢失**）→ `--save_hf_ckpt --disable_ds_ckpt`（500MB adapter） | L437-443、L1175 | VERSION_MAP RM v1 行 | ⑥ | ✅（与 C6 合并） |
| C6 | score head 保存事故链：PEFT 只存含 `lora_` 的 key → `modules_to_save=[score]`；`init_value_head` 必须在 `get_peft_model()` **之前**；从 DS zip 直接提取 score head（免加载 30GB） | L445-459 | 深读 2 §3.2；OpenRLHF model.py | ⑥ | ✅（并入 C5） |
| C7 | 5 个必死崩溃表：规划路径不存在／TP=1 装不下 48.6GB／RM util 0.45→0.62（KV 只剩 0.42GB）／N_GPUS 8+4>8 Ray crash／vLLM 无 Qwen3MoeForRewardModel | L1162-1170 | fix log | ⑨b | |
| C8 | vLLM 服务 OpenRLHF RM 的兼容鸿沟 → **自己 patch**：`as_seq_cls_model(Qwen3MoeForCausalLM)` 包装 + registry 注册 + 手动 delta merge + score head 写入末 shard | L1172-1187、L1335-1362 | 深读 2 | ⑨b | ✅ |
| C9 | 04-24 **六条失败路径表**：FSDP1 全量 unshard 51.7GB 必 OOM／FSDP2 遍历 35 万子模块 init 卡死／cumem sleep 把 LoRA buffer unmap→set_lora 踩空指针／resume(weights) 双持 13+13GB OOM／`_has_slept` gate 方向错／**最终解=resume 前先 offload FSDP 到 CPU** | 「2026-04-24 RL smoke」L1489-1513 | fix log 第 11 节；深读 1 | ⑨d | ✅ |
| C10 | `ACTOR_CKPT_ASYNC_SAVE=False` 是续训关键：异步保存跳过 latest_checkpointed_iteration.txt → resume_mode=auto 失灵；更稳=显式 resume_path | L1819、L1873-1880 | — | ⑨d | |
| C11 | **长度档选择法**：tokenizer 统计保留率（2048→91.9%／2176→95.3%／2304→99.96%）+ 显存贴边实测（GPU0-3 峰值 32065MiB）→ 2176/768 甜点档 | 「长度档选择」L1707-1728 | — | ⑨b | ✅ |
| C12 | **一次只启动一个 vLLM server** 顺序采样 5 个 source（8 卡 TP=8），防多个大模型抢显存 | L1966、L2025 | — | ⑨b | ✅ |
| C13 | legacy merger 按 world size 拼接 layer_scalar/vision std 张量（[4]→[1]）→ **chunk 比对确认四段重复后取第一段**（安全修复方法论：先证明冗余再删） | L1972-1993 | — | ⑨b | |
| C14 | Gemma4 纯文本 merge 两个非显坑：`force_text_only` + 启动带 `--limit-mm-per-prompt '{"image":0,"video":0}'` | 06-08 状态节 L89 | 服务指南 | ⑨b | |
| C15 | **gate vs gate_proj 命名陷阱**：target_modules 含 `gate_proj`（专家 FFN 的 SiLU 门）但不含 `gate`（路由器）——一字之差训崩路由 | L1009-1011 | GLOSSARY MoE 条 | ⑨d | |
| C16 | GPU 拓扑：NUMA 内 SHM ~50GB/s vs 跨 NUMA QPI 5-10GB/s（慢 10×）；5090 全 CNS 无 P2P | L461-468 | 深读 1（已覆盖，此处是原始出处） | ⑥ | |
| C17 | **SAVE_FREQ 死循环**：OOM 崩溃点 step 202–214 < 首个存档点 225 → auto-restart 每次从 200 重来、净进度 0；修法 SAVE_FREQ 25→5（崩溃间隔内两次落袋）。原文教训："resilience 配置在 save 频率 > 崩溃间隔时是无效的" | overview 时间线补全 II（源：handoff 06-01 节） | fix log | ⑨d | |

## D. 训练策略与止损（11 条）

| # | 学习点（含数字） | overview 出处 | 关联引用 | 挂靠段 | 精选 |
|---|---|---|---|---|---|
| D1 | 信号量级判断：**RM ±5 碾压 length +1.0**，temp+length 榨梯度=实测死路（v6.5 教训的量化根据） | 06-08 状态节 L70 | 深读 3 | ⑨d | |
| D2 | 50-step smoke 的 cosine LR 后半段已近 0 → **不能用 smoke 判断正式学习强度**（正式 564 步衰减慢得多） | L1751 | — | ⑩ | ✅（并入 A7） |
| D3 | "当前不要做"**负清单**：不回退 2048／不上 2304／不因横盘提 LR／不扩 FFN／不开 entropy——止损决策的显式化 | L1766-1772 | — | ⑨d | |
| D4 | 05-14 **排除法定位瓶颈层**：aborted=0、clip 正常、KL 稳 → 工程侧全排除 → 瓶颈=奖励信号质量（advantage≈0 + reward 不升 + RM 对"都不坏"的候选无细粒度排序力） | 「05-14 增量 §1」L1886-1897 | 深读 9 | ⑪ | ✅ |
| D5 | 正式 RL 选 **Gemma4 RM 而非 Qwen RM v5 的理由：与 actor 同族、reward 分布更贴近待优化 actor**（L3-6 底座代际问题的正面依据） | 「2026-05-09 正式 RL」L1774-1779 | INTERVIEW_SCRIPT L3-6 | ⑧ | |
| D6 | **多样性=免费**：t0.7→0.9 开场集中度 −9.4%、Jaccard −33%、RM 不降反升；"模板化=低温 mode collapse 非数据病"（精英富女 64 样本 57 个不同开场） | 06-15 状态节 L50 | RLHF_WORKFLOW 钥匙 1 | ⑦ | ✅ |
| D7 | **per-lang ≥base 门**：ja 动作天花板 ~4.0，统一 4.5 门永远不过 → 门槛必须按语言分设 | 06-16 状态节 L33 | v69 design | ⑦ | ✅ |
| D8 | 验收纪律：**全部训完才 probe 最终 ckpt**（不半途 probe 造成误判），三轴 vs step0（CJK 富度+en 不退+attack≤step0@t0.9） | L37 | v69 design | ⑦ | |
| D9 | 0.318 口径纠错案例：clean+attack 混合分被当 attack 数引用 → 06-16 分轴纠错 banner（同尺原则的又一现场） | L37、L47、L51 | RLHF_WORKFLOW §5.3 | ② | |
| D10 | mix-builder 丢 system/history 致 **98.7% 空 prompt** → 按 prompt_id join 回源修复——数据管道的静默损坏比训练 bug 更贵 | 06-16 状态节 L35 | v69 design | ④ | ✅ |
| D11 | **balanced mix 防偏斜**：原 mix 98.7%CJK/1.3%defense/0 en → 重建 balanced（29927 条，en12%+defense10.2% oversample）防多语言+attack 回落 | L35 | v69 design | ④ | ✅ |
| D12 | **step0≡step100 判定法**：同 case、同 seed、同产品口径双模型实采，**全破 (case,char) 名单逐条比对**（不是只看均值）→ "RL 100 步采样层零变化"的硬证据；残留破功 2/2 seed 逐字相同 = 确定性深盆，与概率性破功要分开治 | overview 时间线补全 III（源：handoff 06-12 banner） | 深读 5 门 D | ⑩ | |

## E. 产品与评测（6 条）

| # | 学习点（含数字） | overview 出处 | 关联引用 | 挂靠段 | 精选 |
|---|---|---|---|---|---|
| E1 | **五维评测 × SFT/RL 覆盖表**：语言一致/角色格式/角色语义/重复 4 维提升，**记忆（多轮）与涉黄 2 维诚实标注未覆盖**（记忆"持平或轻微下降"） | 「五维评测」L930-939 | business_ux 映射 | ① | ✅ |
| E2 | **涉黄宽松化两阶段方案**：拒绝率<20% 只用 RL anti-refusal（-2.0 惩罚项）；>40% 先 SFT 打底再 RL；**500-2000 条即可产生明显分布偏移** | L943-982 | — | ① | ✅ |
| E3 | **val task-comp 仅 5% vs 产品 55% bullet 注入**——"训练信号稀疏=没梯度，且 val 测不出产品形态攻击"的量化原文 | 06-08 状态节 L74-76 | 深读 5 门 A；VERSION_MAP | ② | ✅ |
| E4 | 环境三分法：tslib（RM）/gemma4-rl（GRPO）/gemma4-character（SFT）各司其职——版本冲突用环境隔离而非强行升级统一 | L1019-1023 | FRAMEWORK_RUNTIME_LESSONS | ⑥ | |
| E5 | **GSPO 的第二理由**：候选长度天然不均（length_mismatch 类）+ RM 给全序列 scalar → 序列级等权与 reward 粒度一致（MoE 之外的独立论据） | 「GSPO vs GRPO」L1015-1017 | 深读 3 | ⑨a | ✅ |
| E6 | 视觉编码器 1.6GB 在 GRPO 中加载但不参与文本前向——**已知且可忽略的冗余**（知道哪些浪费可以不修也是判断） | L1195 | — | ⑨b | |

---

> 使用方式：精选✅ 条目已入指导正文各段"遗珠"行；面试前若某段被问穿，回本底账查同组未入选条目补弹药。发现 overview 新增内容时在对应组追加编号。
