# ChatJoy RL 接手后报错与修改日志

更新时间: 2026-05-08 11:53:02 CST

## 0) ChatJoy 完整链路入口（SFT → RM → RL）

> 这是 RL 接手文件的开头索引。RL 的 reward、KL、actor loss 是否合理，必须放回上游 SFT 数据、preference pair、RM 训练和 online reward loop 的完整链路里判断。更完整的项目总纲见 `/home/ps/.claude/projects/-home-ps/memory/project_chatjoy_overview.md`，本文件后续章节保留所有 RL 报错、补丁和复测细节。

### 0.1 数据源与 SFT 数据构造

1. 正式服数据导出：
   - 脚本：`/home/ps/hanjiaqi/LlamaFactory/chatjoy/export_data.py`
   - 数据库：Milvus `<prod-db>.vectors`
   - 字段：`id`, `text`, `user_id`, `character_id`, `timestamp`, `message_id`
   - 输出：`/home/ps/hanjiaqi/LlamaFactory/chatjoy/chatjoy_dataset/chatjoy_part_*.jsonl`
   - 作用：保留真实线上用户聊天分布，而不是用人工 instruction 模板替代。

2. TOP20 角色和 27 语言预 API 采样：
   - 历史生成脚本：早期 `/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/prepare_pre_api_dataset.py` 逻辑，排除已用语言后按候选量取剩余 Top27。
   - 后续硬化脚本：`prepare_preapi_top20_27langs.py`，当前源码含硬编码 `TARGET_27_LANGS`，与历史已生成 report 的 selected languages 不完全一致。
   - 角色卡：`TOP20CHARACTERS_systemprompt/*.json`
   - 会话恢复：按 `(user_id, character_id)` 聚合，按 `timestamp` 排序；连续 human 消息合并为 instruction，遇到 assistant 形成一条 `(instruction, output, system)`。
   - 第一轮清洗：删除短 instruction、emoji-only/符号-only instruction、短 output、无法检测语言样本。
   - 语言策略：历史落地产物排除已使用的 `en,de,fr,tr,uk`，按候选量选择 27 语言，每语言最多 2200 条；理论上限 `59,400`，实际 `57,197` 条，因为 `bn/hu/no/hr` 不足 2200。
   - 历史实际语言：`es,ar,pt,ru,pl,cs,id,vi,it,zh-tw,th,ja,zh-cn,ko,hi,sv,nl,fa,he,el,sk,fi,ro,bn,hu,no,hr`。
   - 预 API 输出：`chatjoy_top20_multilingual_27x2200_preapi.json`，真实报告 `chatjoy_top20_multilingual_27x2200_preapi_report.json` 记录 raw `2,080,698` 行、TOP20 会话 `44,241` 个、候选 `631,995` 条、最终 `57,197` 条；附带 `preapi_role_lang_dist.csv` 和 `59700_data_for_gemini.csv`。

3. Gemini API 重构：
   - 脚本：`regenerate_outputs_with_gemini.py`
   - 输入：`57,197` 条真实 instruction + 对应角色 system prompt。
   - 输出：保留真实 instruction/system/character/language，替换 output 为 Gemini 重构回复。
   - 断点续跑：`chatjoy_top20_multilingual_27x2200_preapi_gemini.progress.jsonl`
   - 历史分段：first22331 输入 `22,331` 保留 `10,542`；`41069-57197` backup 输入 `16,129` 保留 `7,752`。
   - 目的：让 SFT 学真实用户输入分布，同时让 assistant output 更干净、更符合角色和语言。

4. 第二轮清洗：
   - 脚本：`data/process_chatjoy_data.py`
   - 清理 API 报错：`SAFETY`, `PROHIBITED_CONTENT`, Gemini 原始 JSON payload 等。
   - 清理占位符泄漏：`{{user}}`, `[user]`, `<user>`。
   - 语言一致性：脚本族检测 + `langdetect` 置信度；严格数据只保留 `match`。
   - 产物：`data/runs/clean32_for_sft1.jsonl` 13,128 条；`data/runs/clean32_for_sft2.jsonl` 19,508 条。

5. LlamaFactory SFT：
   - LlamaFactory 数据：`clean32_for_sft1_llamafactory.json`, `clean32_for_sft2_llamafactory.json`
   - Qwen3 SFT2 配置：`qwen30b_fsdp_qlora_clean32_sft2.yaml`
   - Qwen3 SFT2 merged：`saves/qwen3-30b-moe/clean32_for_sft2_fsdp_qlora_merged/`；日志 `2440/2440` 完成，2 epoch，train loss `1.2335`。
   - Gemma4 actor SFT：`gemma4_sft2_*_lora.yaml`
   - 当前 RL actor：`saves/gemma4-26b-a4b/sft2_ddp_lora_3gpu_567_cut3072_merged`；对应 adapter 训练 2 epoch，global step `1626`，train loss `1.1168`。

### 0.2 Preference Pair 与 RM 链路

1. pair 构造：
   - 核心脚本：`GRPO_Training_Package/build_rejected_from_sft.py`
   - chosen：清洗后的 SFT/Gemini output。
   - rejected：由 SFT 模型生成或零生成替换得到。
   - 合并脚本：`merge_pairs_for_rm.py`
   - 合并输出：`data/rm_train_merged.jsonl`
   - 去重规则：同一 `(system, prompt)` 只保留 `reward_gap` 最大的 pair。

2. rejected 类型和目标信号：
   - `wrong_role`：同 prompt 换角色 system 生成，或 cross-character 使用其他角色真实回复；训练 RM 识别角色身份/语气/称呼错误。
   - `wrong_language`：强制输出其他语言，或 cross-lingual 使用同角色其他语言回复；训练 RM 识别用户语言和回复语言不一致。
   - `length_mismatch_short`：过短、敷衍、信息不足。
   - `length_mismatch_long`：过长、重复、拖沓。

3. pair 过滤奖励：
   - 脚本：`GRPO_Training_Package/chatjoy_reward_utils.py`
   - 维度：format、role、language、length。
   - 默认权重：format `0.30`，role `0.30`，language `0.25`，length `0.15`。
   - 按 rejected 类型切换权重 profile：wrong_role 强化 role，wrong_language 强化 language，length_mismatch 强化 length。
   - pair 必须满足 chosen 分数、总 gap 和目标维度 gap 阈值，避免低置信度噪声训练 RM。

4. RM 数据版本：
   - v1：`rm_train.jsonl` 16,826 条，acc 94.5%，但 DS checkpoint 因磁盘失败丢失。
   - v2：删除 2,000 条低 gap `length_mismatch_long`，acc 86.4%，说明过滤过度。
   - v3：加入 hard negative 和 long soft pool，但 batch/步数和 long 数据质量导致 acc 约 86.0%。
   - v4：删除有缺陷 long source 的中间方案。
   - v5：完整 v1 base + 约 5% bad-case pool，当前最优，acc 95.5%，gap 3.497。

5. 当前 RM：
   - 数据脚本：`prepare_rm_train_v5_5pct_badpool.py`
   - 训练脚本：`launch_rm_train_v5_gpu0123.sh`
   - adapter：`saves/qwen3-30b-moe/chatjoy_rm_v5_gpu0234_20260421/`
   - vLLM merged：`saves/qwen3-30b-moe/chatjoy_rm_v5_merged_vllm/`
   - 关键约束：OpenRLHF RM 训练/评估对 `(prompt + chosen).rstrip("\n")` 后 EOS 对齐位置打分；online vLLM classify 必须复现，否则 final-token pooling 会错位。

### 0.3 RL 数据、算法和 reward 组成

1. RL 数据：
   - 转换脚本：`GRPO_Training_Package/framework_bridge/prepare_verl_rl_data.py`
   - 输入：clean32 SFT 或 pair 记录抽取出的 `(system, prompt, language)`。
   - 输出：verl parquet，`data_source=chatjoy`, `ability=chatjoy_alignment`, `reward_model.style=model`。

2. RL actor 和算法：
   - actor：Gemma4 SFT merged。
   - 启动脚本：`GRPO_Training_Package/framework_bridge/verl/train_grpo_lora_actor_verl.sh`
   - 当前算法表述：`algorithm.adv_estimator=grpo` + `actor_rollout_ref.actor.policy_loss.loss_mode=gspo`，即 GRPO advantage + GSPO policy loss。
   - loss 聚合：`seq-mean-token-mean`。

3. RL reward：

```text
total_reward = online_rm_score + anti_refusal_custom_reward
```

   - `online_rm_score`：Qwen3 RM v5 vLLM `/classify` 分数，是主排序信号。
   - `anti_refusal_custom_reward`：`anti_refusal_reward.py`，只惩罚拒绝，不给正奖励。
   - 当前 anti-refusal：关键词拒绝 `-1.0`，短回复无 roleplay 格式 `-0.25`，正常回复 `0.0`。
   - KL 不属于 reward，而是 actor loss 项：`use_kl_loss=True`, `kl_loss_coef=0.001`, `kl_loss_type=low_var_kl`。

4. 2026-05-07 reward 链路最新状态：
   - online RM 负分根因已修：`reward_loop.py` 对 RM prompt `rstrip("\n")` 并补 EOS，使 vLLM classify 与 OpenRLHF RM 对齐。
   - anti-refusal 已收窄：移除裸 `no puedo`, `je ne peux pas`, `ich kann nicht` 等会误伤内心独白的短语。
   - `chatjoy_p2176_r768_smoke50_20260507_103012` 完成 50/50，final val reward `1.884`，RM `1.903`，无 OOM。

5. 正式 RL 推荐起点：
   - `MAX_PROMPT_LENGTH=2176`
   - `MAX_RESPONSE_LENGTH=768`
   - `PPO_MAX_TOKEN_LEN_PER_GPU=2944`
   - `MAX_MODEL_LEN=3200`
   - `RM_MAX_MODEL_LEN=3456`
   - actor LR `3e-6`
   - cosine + warmup ratio `0.03`
   - `clip_grad=1.0`
   - `kl_coef=0.001`
   - LoRA rank 8, target `q_proj,k_proj,v_proj,o_proj`

RL 指标解释原则：优先看 online RM、total reward、anti-refusal 误伤率、`actor/kl_loss`、`actor/ppo_kl`、`pg_clipfrac`、`grad_norm` 和 response clip ratio。`actor/loss` 在 policy-gradient RL 中不应按 SFT loss 的单调下降方式解读。

## 1) 记录范围

- 目标脚本: `/home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/framework_bridge/verl/train_grpo_lora_actor_verl.sh`
- 相关补丁:
  - `/home/ps/hanjiaqi/verl/verl/utils/vllm/utils.py`
  - `/home/ps/hanjiaqi/verl/verl/utils/vllm/patch.py`
- 当前目标:
  - 2-step smoke 已在 `PPO_MAX_TOKEN_LEN_PER_GPU=3072` 基线下复跑通过；
  - smoke 使用真实 Gemma4 SFT actor 与真实 Qwen3 RM；`q/k/v/o rank8` 已可跑通但显存余量极薄，`activation_offload=True` 未明显改善余量且使 actor update 变慢。

## 2) reconnect 说明（持续性保障）

- 现象: 任务期间出现多次 “reconnect / 中断后继续”。
- 原因: 训练进程是长任务，且对话会话会切换模型或被中断；不是代码状态丢失。
- 持续性措施:
  - 所有改动都落盘到固定文件；
  - 每次复测都记录具体 override 和报错栈；
  - 通过此日志保持“当前状态 -> 下一步”连续。

## 3) 报错追踪（最新）

| 序号 | 时间 | 报错/现象 | 根因判断 | 状态 |
|---|---|---|---|---|
| 1 | 14:09 左右 | `PermissionError: [Errno 1] Operation not permitted`（Ray init） | 沙箱网络/套接字限制 | 已绕过 |
| 2 | 14:12-14:58 | `No available memory for the cache blocks`（actor vLLM） | rollout 侧 `max_model_len`/调度上限和显存预算过紧，KV 无法分配 | 已阶段性修复（见下） |
| 3 | 14:18 | `Gemma4ForConditionalGeneration does not support LoRA yet` | vLLM multimodal Gemma4 类缺 LoRA 协议属性 | 已修复 |
| 4 | 14:4x | `To support LoRA for MoE model, get_expert_mapping must be implemented` | Gemma4 MoE 缺 `get_expert_mapping` | 已修复 |
| 5 | 15:5x-16:03 | `column_parallel_linear.set_lora` 触发 `CUDA error: invalid argument` + `illegal memory access`（发生在 `actor_rollout_update_weights`） | vLLM LoRA 热更新阶段（IPC weight sync）仍有 TP/模块映射兼容问题；已不再是 KV cache 问题 | 当前主阻塞 |

## 4) 已落地代码修改

1. `max_model_len` 固定为训练上下文长度:
   - `MAX_MODEL_LEN=${MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH}`
   - 同步写入 actor rollout 与 reward rollout。

2. rollout 调度显式收敛（避免默认 8192/1024）:
   - actor: `max_num_batched_tokens=MAX_MODEL_LEN`, `max_num_seqs=32`
   - reward: `max_num_batched_tokens=MAX_MODEL_LEN`, `max_num_seqs=64`

3. actor 显存预算上调以通过 KV 初始化:
   - `ROLLOUT_GPU_MEMORY_UTILIZATION` 默认调到 `0.78`（可被环境变量覆盖）。

4. Gemma4 vLLM LoRA 兼容补丁:
   - 在 `verl/utils/vllm/utils.py` 注入 Gemma4 multimodal LoRA 协议属性；
   - 注入 Gemma4 MoE `get_expert_mapping`。

5. Gemma4 MoE 权重 loader 兼容补丁:
   - 在 `verl/utils/vllm/patch.py` 处理 Gemma4 `inner_model.model.layers` 路径。

6. target modules 改为可配置并默认 attention-only:
   - `TARGET_MODULES=${TARGET_MODULES:-[q_proj,k_proj,v_proj,o_proj]}`
   - 目的: 先规避已观测到的 LoRA 热更新崩溃路径，便于稳定复测。

## 5) 最新复测结论（关键）

- 已确认通过:
  - RM vLLM 正常启动；
  - actor vLLM 在 `gpu_memory_utilization=0.78` 下可以完成 EngineCore 启动（KV cache 问题已不再是首要阻塞）。
- 当前失败点:
  - 在第一次 `checkpoint_manager.update_weights -> actor_rollout_update_weights` 时，
    `self.add_lora(...)` 进入 vLLM `column_parallel_linear.set_lora` 触发 CUDA `invalid argument`，随后 `illegal memory access`。
- 结论:
  - 主阻塞已从“启动阶段显存不足”切换为“LoRA 热更新阶段的 TP/模块兼容”。

## 6) 下一步执行计划（立即）

1. 在 `verl/workers/rollout/vllm_rollout/utils.py` 增加 LoRA 热更新的调试信息（模块名/shape），定位具体出错层。
2. 基于定位结果做最小兼容修复（优先保证 `add_lora` 不崩溃，再评估是否恢复更宽 target modules）。
3. 继续 1-step smoke，目标是完整跑过首次 `update_weights` 与 rollout 调用链。

## 7) 16:42 新增进展（中断后续跑）

1. 已定位“reconnect”高频原因（不是单一网络问题）:
   - Ray/vLLM actor 在 LoRA 热更新阶段触发 CUDA 异常后进程崩溃，driver 侧会反复重连 worker，看起来像“不断 reconnect”；
   - 同时 Ray session 目录持续打印 `over 95% full`，增加了对象回收/重试压力，放大了中断观感。

2. 已确认启动阶段问题已基本转移:
   - `max_model_len` 与 rollout 显存预算调整后，RM/actor vLLM 初始化总体可通过；
   - 当前主故障稳定集中在首次 `update_weights -> add_lora -> set_lora`。

3. 最新故障位点（已打点）:
   - 触发模块: `language_model.model.layers.0.self_attn.qkv_proj`
   - 观测到 A/B tensor shape 对齐，但 copy 仍报 `CUDA error: invalid argument`，随后出现 `illegal memory access`。
   - 说明主因已不是简单的 shape 不匹配，而是该路径的 LoRA 热更新执行与当前 Gemma4/vLLM 组合存在更深兼容性问题。

4. 连续性保障:
   - 本文件作为“故障 -> 修改 -> 复测”主索引持续更新；
   - 每次复测均记录会话与失败点，避免因对话中断丢失上下文。

## 8) 17:37 新增进展（最新 takeover）

1. 最新两轮 smoke（`trainer.total_training_steps=2`）结果:
   - 会话 `session_2026-04-23_16-50-06_608968_726978`：仍在首次 `add_lora` 崩溃。
   - 会话 `session_2026-04-23_16-55-53_675397_782639`：同样在首次 `add_lora` 崩溃。
   - 共同栈顶: `vllm/lora/layers/column_parallel_linear.py:272`，`copy_(lora_a_i, non_blocking=True)` -> `CUDA error: invalid argument`，随后 `illegal memory access`。

2. 关键新发现:
   - 报错回到 vLLM 原始 `column_parallel_linear.py`，而不是本地 `verl/utils/vllm/utils.py` 的 guarded `set_lora`；
   - 说明 TP worker 实际执行时，`MergedColumnParallelLinearWithLoRA.set_lora` 没被 monkey-patch 成功替换。

3. 根因定位（已验证）:
   - `BaseLinear` patch 与 `Column/Merged` patch 共用同名 guard 标记（`_verl_guarded_set_lora`）；
   - 该标记被继承后导致列并行 patch 提前返回，`MergedColumnParallelLinearWithLoRA` 未被替换；
   - 因此热更新时仍走 vLLM 原始 copy 路径并崩溃。

4. 已落地修复（刚完成）:
   - 在 `verl/utils/vllm/utils.py` 中拆分独立 guard 标记:
     - `Base`: `_verl_base_guarded_set_lora`
     - `Column`: `_verl_column_guarded_set_lora`
     - `Merged`: `_verl_merged_guarded_set_lora`
   - 确保 `MergedColumnParallelLinearWithLoRA.set_lora` 必定被替换，不再被继承标记误判跳过。
   - 保留并继续使用 worker 热更新入口强制 `VLLMHijack.hijack()`（`vllm_rollout/utils.py:update_weights_from_ipc`）。

5. 当前下一步:
   - 立刻重跑 2-step smoke；
   - 验证首次 `add_lora` 时堆栈是否进入本地 patched `set_lora`；
   - 目标: smoke 至少完整通过首次 `update_weights`，不再触发 reconnect 链路。

## 9) 17:42 新增进展（续接复测）

1. 会话状态:
   - 已完成续接复测，会话为 `session_2026-04-23_17-37-54_960530_909123`；
   - 训练仍在首次 `update_weights -> add_lora` 阶段失败退出。

2. 与上一轮相比的关键变化:
   - 日志已出现本地 guard 打点:
     - `Skip LoRA-A copy failure for ... qkv_proj slice=*`
     - `Skip LoRA-B copy failure for ... qkv_proj slice=*`
   - 说明 monkey-patch 已经进入执行路径，不再是“完全未生效”状态。

3. 当前根因判断:
   - 在 `qkv_proj` LoRA 热更新过程中，TP worker 先触发 `CUDA illegal memory access`；
   - guard 虽能跳过单次 copy 异常，但 CUDA 上下文已进入错误态，后续 worker 仍整体失败；
   - 因此仅靠“copy 异常跳过”不足以让 smoke 通过，需要避免触发该高风险模块路径。

4. 后续修复方向（已确定）:
   - 将 rollout 侧 LoRA `target_modules` 在 smoke 阶段收敛到稳定子集（优先保留 `o_proj`，临时避开 `qkv_proj`）；
   - 先完成 smoke 链路打通，再逐步扩回 `q_proj/k_proj/v_proj` 做增量验证。

## 10) 17:45-17:50 新增进展（交接前最终状态）

1. 已执行复测（使用已批准命令，`TARGET_MODULES=[o_proj]`）:
   - 会话: `session_2026-04-23_17-45-06_853677_963725`
   - 配置: `trainer.total_training_steps=2`, `trainer.test_freq=100000`, `trainer.save_freq=100000`
   - 结果: 仍在首次 `update_weights -> add_lora` 失败。

2. 关键结论更新:
   - 崩溃模块已从先前的 `qkv_proj` 扩展到 `o_proj`（当前最小 attention 子集仍会触发）；
   - 首次异常日志:
     - `Skip LoRA-A copy failure for language_model.model.layers.0.self_attn.o_proj ... illegal memory access`
   - 随后在 vLLM `activate_adapter -> reset_lora` 继续报 `illegal memory access`，worker 整体退出。

3. 结论（供接手方直接使用）:
   - 当前不是单一 `qkv_proj` 路径问题，而是 **Gemma4 + vLLM LoRA 热更新（add_lora）整体不稳定**；
   - 仅靠 `set_lora` copy guard 不足以通过 smoke。

4. 未完成但已验证方向:
   - 计划改为绕过热更新路径（`actor_rollout_ref.model.lora.merge=True`，让 rollout 收全量合并权重）；
   - 尝试发起 1-step merge 验证命令时被权限确认拒绝，故该路径尚未完成实跑。

5. Claude 接手建议优先级:
   - P0: 在脚本增加 smoke 专用 merge 同步开关（如 `trainer.total_training_steps<=2` 自动追加 `actor_rollout_ref.model.lora.merge=True`，或显式环境变量开关）并复测；
   - P1: 若 merge 模式 smoke 可过，再决定长期策略（保留 adapter hot-swap 或仅在 smoke 使用 merge）；
   - P2: 继续追查 add_lora 根因时，建议加 `CUDA_LAUNCH_BLOCKING=1` 的最小复现场景。

---

## 11) 2026-04-23 晚~2026-04-24 10:30 Claude 接手进展（关键：根因已找到）

### 11.1 试过的路径和结果

| # | 方案 | 结果 | 原因 |
|---|------|------|------|
| A | `actor_rollout_ref.model.lora.merge=True` + FSDP1 | **OOM** | `fsdp_merge_unmerge` 对 FSDP1 用 `FSDP.summon_full_params(module, writeback=True)`，把 25.84B bf16 模型 = 51.7GB 拉到单 rank，5090 31.36GB 装不下（实测 28GB allocated）|
| B | `lora.merge=True` + `fsdp_config.strategy=fsdp2` | **init 卡死 5+ min** | Gemma4-26B-A4B 是 **128 experts MoE**（`text_config.num_experts=128`），`_select_fsdp2_wrap_targets` 遍历 named_modules 约 35 万个子模块；且 `fsdp2_load_full_state_dict` 要求 rank0 把全量 52GB 挪到 GPU → 根本放不下 5090 |
| C | 回退到 hot-swap（`lora.merge=False`）+ FSDP1，把 `TARGET_MODULES=[o_proj] LORA_RANK=8 LORA_ALPHA=8` + `CUDA_LAUNCH_BLOCKING=1` + utils.py guard 加 `torch.cuda.synchronize + CPU detach + 详细打印` | **仍在首次 set_lora 崩**，但 guard 日志更详细了 | CUDA 上下文一旦被 `copy_` 污染，后续 reset_lora 同样报 illegal memory access |
| **D** | **定位 `lora_a_stacked` 在 cumem 中被标记为 `weights` → sleep(tags=["weights"]) 把 LoRA buffer 也 unmap 了** | **根因确认** | 见下节 |

### 11.2 真正的根因（2026-04-24）

**vLLM V1 的 cumem 内存池 + sleep_level=1 导致 LoRA 权重 buffer 被 unmap。**

- 路径：`vllm/v1/worker/gpu_worker.py::sleep(level)` → `allocator.sleep(offload_tags=("weights",))`（level=1 时） → 把所有进入过 `weights` tag 上下文的张量（包括 `lora_a_stacked` / `lora_b_stacked`）从物理显存 unmap；
- 后果：verl 每次 `update_weights` 前要 resume(tags=["weights"])，如果只 resume `weights` 而没包含 LoRA buffer 所在 tag，之后走 `self.add_lora → activate_adapter → reset_lora/set_lora` 访问 `lora_a_stacked` 就命中已 unmap 区间 → **`CUDA error: invalid argument` 然后 `illegal memory access`**；
- 现象解释（Codex 之前的观察完全吻合）：
  - o_proj / qkv_proj 都会崩，因为它们都属于 weights tag；
  - guard 能跳过首次 copy 异常，但 CUDA context 污染不可逆；
  - shape 都对得上也崩，因为错的是指针不是大小。

### 11.3 已落地的代码修改（Claude 在 Codex 之上的追加）

1. **`/home/ps/hanjiaqi/verl/verl/workers/rollout/vllm_rollout/vllm_rollout.py`**
   - 增加 `self._has_slept = False` 初始状态（`ServerAdapter.__init__`）；
   - `resume(tags=[...])` 前判断：若 `not self._has_slept` 则直接 return（**避免 cumem 对从未 sleep 的 weights 做双重 map，导致 `CUDA Error: out of memory at cumem_allocator.cpp:139`**）；
   - `release()` 调用 sleep 成功后设置 `self._has_slept = True`。

2. **`/home/ps/hanjiaqi/verl/verl/workers/engine_workers.py`**
   - `update_weights()` 把原来 `if sleep_level != 1: resume` 改成 **任意 sleep 模式都 resume(tags=["weights"])**（详细注释在 line 671-680），保证 sleep_level=1 场景下 LoRA buffer 也会被 re-map；
   - 注意：必须配合 (1) 的 `_has_slept` gate，否则首次 update_weights 前 vLLM 还没 sleep 过，resume 等于 cumem 双重分配 → OOM。

3. **`/home/ps/hanjiaqi/verl/verl/utils/vllm/utils.py`**
   - `_patch_column_parallel_set_lora_guard._safe_copy` 与 `_patch_base_linear_set_lora_guard._safe_copy`：
     - 进入前 `torch.cuda.synchronize()`；
     - 源张量先 `.detach().to("cpu", copy=True)` 断开与任何 async 生产者（FSDP reshard / IPC）的关系，再 blocking H2D；
     - 加 `[verl-lora-guard] ... src=... dst=...` 信息日志，便于复测时在 log 里核对 shape/dtype/device。

4. **`/home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/framework_bridge/verl/train_grpo_lora_actor_verl.sh`**
   - 新增 `LORA_MERGE_MODE="${LORA_MERGE_MODE:-False}"`，加在 ARGS 末尾：`"+actor_rollout_ref.model.lora.merge=${LORA_MERGE_MODE}"`；
   - 注释里写明了 merge 模式在本机不可行的原因（FSDP1 OOM、FSDP2 init 卡 + rank0 装不下）；
   - 默认仍是 hot-swap（merge=False）。

### 11.4 最近一次 smoke 结果（2026-04-24 10:23~10:28）

- 配置：`TARGET_MODULES=[o_proj] LORA_RANK=8 LORA_ALPHA=8 CUDA_LAUNCH_BLOCKING=1 VERL_LOGGING_LEVEL=INFO trainer.total_training_steps=2`（会话目录 `session_2026-04-24_10-24-09_060951_1854511`）
- 进度：
  - ✓ 数据过滤、FSDP init、RM vLLM 加载、actor vLLM 加载都过了；
  - ✓ 第一次 `actor_rollout_update_weights` 进入了；
  - ✗ 在 `await self.rollout.resume(tags=["weights"])` 里 vLLM 报 **`CUDA Error: out of memory at cumem_allocator.cpp:139`**（wake_up 双重分配）；
- 这证实了 11.3(1) 的 `_has_slept` gate 是必须的。**该补丁刚写完但尚未复测**（用户说显卡被占用，先等）。

### 11.5 接下来该做什么（交给 Codex 或 Claude 下一轮）

1. **P0 — 复测 smoke**：显卡空下来后，跑同样命令（脚本里已自动带 merge=False）：
   ```bash
   cd /home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/framework_bridge/verl
   LOG=/tmp/grpo_smoke_$(date +%H%M).log
   tmux new -d -s grpo_smoke "conda run -n gemma4-rl --no-capture-output bash -c '
     export CUDA_LAUNCH_BLOCKING=1
     export VERL_LOGGING_LEVEL=INFO
     TRAIN_BATCH_SIZE=32 TARGET_MODULES=\"[o_proj]\" LORA_RANK=8 LORA_ALPHA=8 \
     bash train_grpo_lora_actor_verl.sh \
       trainer.total_training_steps=2 \
       trainer.save_freq=100000 \
       trainer.test_freq=100000 2>&1 | tee \$LOG
   '"
   ```
   成功标志：
   - 第 1 次 update_weights 不再报 `cumem_allocator.cpp:139`；
   - 不再看到 `set_lora` 的 `CUDA error: invalid argument` / `illegal memory access`；
   - `Rollout.generate_sequences` 出来，并至少打印一次 reward。

2. **P1 — 如果 P0 成功**，逐步扩展 `TARGET_MODULES` 到 `[q_proj,k_proj,v_proj,o_proj]`、`LORA_RANK=32 LORA_ALPHA=32`，验证多模块并发 set_lora 仍稳。

3. **P1/P2 — 如果 P0 又在别处挂**：
   - 优先怀疑 activate_adapter → reset_lora 的另一条路径（vLLM V1 内部 punica wrapper 可能也在 weights tag 里）；
   - 备选：把 LoRA buffer 显式分配到 `kv_cache` tag 或非 cumem 池（需动 `vllm/lora/model_manager.py` 或加 monkey patch）。

4. **已排除但值得记录**（别再重复试）：
   - `lora.merge=True` 路径 — FSDP1/FSDP2 在本机（8×5090 31.36GB）都不可行；
   - 收窄 `TARGET_MODULES=[o_proj]` / 降 `LORA_RANK` — 不能解决 cumem unmap 问题，因为触发位置跟 shape 无关。

### 11.6 关键文件速查

- 训练脚本 & 入口：`/home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/framework_bridge/verl/train_grpo_lora_actor_verl.sh`
- LoRA copy guard：`/home/ps/hanjiaqi/verl/verl/utils/vllm/utils.py`（`_patch_column_parallel_set_lora_guard` / `_patch_base_linear_set_lora_guard`）
- Gemma4 MM 兼容补丁：`/home/ps/hanjiaqi/verl/verl/utils/vllm/utils.py::_patch_gemma4_mm_lora_support` + `/home/ps/hanjiaqi/verl/verl/utils/vllm/patch.py`
- sleep/resume gate：`/home/ps/hanjiaqi/verl/verl/workers/rollout/vllm_rollout/vllm_rollout.py::ServerAdapter.resume/release` (加了 `_has_slept`)
- update_weights 主流程：`/home/ps/hanjiaqi/verl/verl/workers/engine_workers.py::EngineWorker.update_weights`（L662-722）
- FSDP merge（已判无效路径但留作参考）：`/home/ps/hanjiaqi/verl/verl/utils/fsdp_utils.py::fsdp_merge_unmerge` + `merged_lora_context`

### 11.7 必须保留的环境约束

- 8×5090，**每卡 31.36GB 可用**（非 32GB，profile 误差曾坑了一次）；
- GPU 0-3：actor FSDP + actor vLLM 共同占用（colocate，靠 sleep 切换）；
- GPU 4-7：RM vLLM（`enable_resource_pool=True n_gpus_per_node=4`）；
- 不要动 `ROLLOUT_TP_SIZE=4`：Gemma4 SFT merged 53.8GB ÷ 4 = 13.45GB/卡，单卡装不下；
- `gpu_memory_utilization=0.78` 是 actor 5090 的 profile 下限，再高会在 KV block 初始化炸；**但 0.78 对 cumem wake_up 留的 overhead 不够**（见 12.x），smoke 阶段先调到 0.60；
- `conda activate gemma4-rl` 在非交互 shell 不可靠，**必须用** `conda run -n gemma4-rl --no-capture-output bash -c '...'` 来跑。

---

## 12) 2026-04-24 晚接续（复测 + 新的理解）

### 12.1 static-review 发现 `_has_slept` 方向错了

静态复核 `sleep_replicas()` 路径：
- init 后 [ray_trainer.py:892](hanjiaqi/verl/verl/trainer/ppo/ray_trainer.py#L892) 调 `sleep_replicas()`；每步 rollout 之后 [ray_trainer.py:1379](hanjiaqi/verl/verl/trainer/ppo/ray_trainer.py#L1379) 也调；
- `sleep_replicas` 走的是 `RolloutReplica.sleep → vLLMHttpServer.sleep → _sleep_hybrid → engine.collective_rpc("sleep", level=1)`，**不经过** `ServerAdapter.release`；
- 所以我之前加的 `_has_slept` gate 根本不会被 flip，永远 False，会把必要的 `resume(weights)` 跳掉；
- **结论**：`_has_slept` 补丁已撤回，`resume/release` 回到原始无状态。

### 12.2 加上"先 offload FSDP actor 再 resume(weights)" 也没解决

在 [engine_workers.py::update_weights](hanjiaqi/verl/verl/workers/engine_workers.py#L668) 的 `resume(tags=["weights"])` 之前加：
```python
if self.actor.engine.is_param_offload_enabled:
    self.actor.engine.to("cpu", model=True, optimizer=False, grad=False)
aggressive_empty_cache(force_sync=True)
```

复测（19:53-19:57，`TARGET_MODULES=[o_proj] LORA_RANK=8 LORA_ALPHA=8 CUDA_LAUNCH_BLOCKING=1`）结果：**仍然 `CUDA Error: out of memory at /workspace/csrc/cumem_allocator.cpp:139` 在 `wake_up(tags=["weights"])`**。日志里 `Memory cleanup attempt 1: Freed 0.00 GB reserved, 0.00 GB allocated` — 说明 FSDP actor init 之后就已经自动 offload 到 CPU 了（`FSDPEngine.__init__` 末尾已经 `self.to(device="cpu", model=self._is_offload_param, ...)`）。所以我的"再 offload 一次"是 no-op，GPU 上本来就没 FSDP。

### 12.3 真因：0.78 gpu_memory_utilization 给 cumem wake_up 留的 overhead 不够

- vLLM process 占住 **虚拟地址** 0.78 × 31.36 = 24.5GB；sleep(level=1) 只释放物理页，虚拟地址还在；
- 同 GPU 上有 4 个 WorkerDict (FSDP TP rank) + 1 个 vLLMHttpServer = **2 个进程共用同一张卡**，各自 CUDA context ~0.5-1GB；加 NCCL 工作 buffer + flashinfer JIT 缓存 + PyTorch cache，碎片开销 **能吃到 4-6GB**；
- cumem `create_and_map` 要的是 **物理连续页**，碎片化后 driver 找不到 → OOM；
- 0.22 × 31.36 = 6.9GB 的 overhead 预算对这个多进程场景不够。

### 12.4 接下来要试（按优先级）

1. **P0 — 把 `ROLLOUT_GPU_MEMORY_UTILIZATION` 从 0.78 降到 0.60**：
   - vLLM 虚拟预算 = 0.60 × 31.36 = 18.8GB（模型 13GB + KV 5.8GB，smoke 足够）；
   - 剩 40% = 12.5GB 给 overhead，应该充足；
   - smoke 命令：
     ```bash
     tmux new -d -s grpo_smoke "conda run -n gemma4-rl --no-capture-output bash -c '
       cd /home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/framework_bridge/verl
       export CUDA_LAUNCH_BLOCKING=1
       export VERL_LOGGING_LEVEL=INFO
       ROLLOUT_GPU_MEMORY_UTILIZATION=0.60 TRAIN_BATCH_SIZE=32 \
       TARGET_MODULES=\"[o_proj]\" LORA_RANK=8 LORA_ALPHA=8 \
       bash train_grpo_lora_actor_verl.sh \
         trainer.total_training_steps=2 \
         trainer.save_freq=100000 \
         trainer.test_freq=100000 2>&1 | tee /tmp/grpo_smoke_\$(date +%H%M).log
     '"
     ```
2. **P1 — 若 0.60 仍 OOM**：降到 0.50，或者再把 actor 和 vLLM 的 CUDA 进程分开（`actor_rollout_ref.rollout.enforce_eager=True` 已经设了，不能再省）；
3. **P1 — 若 wake_up 过了但 set_lora 还挂**：回到 `utils.py` 的 guard 日志看 tensor meta，看是不是新的对齐问题；
4. **P2 — 若 wake_up 过了，rollout 也过了，但 update_actor 阶段 OOM**：reduce `PPO_MAX_TOKEN_LEN_PER_GPU` 从 8192 到 4096。

### 12.5 到目前为止已落地的 verl 补丁（只改这三个文件）

1. `/home/ps/hanjiaqi/verl/verl/workers/rollout/vllm_rollout/vllm_rollout.py`
   - `_has_slept` 相关代码已撤回，回到原始的 `resume/release`；
2. `/home/ps/hanjiaqi/verl/verl/workers/engine_workers.py`
   - `update_weights()` 第一步在 `resume(tags=["weights"])` 之前加了 FSDP offload + aggressive_empty_cache（虽然目前证明是 no-op，但保留作保险）；
   - 注释里写了 cumem sleep 把 LoRA buffer 一起 unmap 的原因；
   - 保留"任意 sleep 模式都 resume weights" 改动（不再 gate on sleep_level != 1）；
3. `/home/ps/hanjiaqi/verl/verl/utils/vllm/utils.py`
   - set_lora guard 已加 `torch.cuda.synchronize()` + CPU detach + 详细日志；
   - 作为安全网保留，不依赖它阻止崩溃。

脚本 `train_grpo_lora_actor_verl.sh` 本身不改，smoke 用环境变量临时覆盖 `ROLLOUT_GPU_MEMORY_UTILIZATION`。

### 12.6 三档 `ROLLOUT_GPU_MEMORY_UTILIZATION` 复测结论

| 设置 | 结果 | 报错位置 | 原因 |
|------|------|----------|------|
| **0.78**（脚本默认） | 崩 | `cumem_allocator.cpp:139` at `wake_up(tags=["weights"])` | vLLM 虚拟预算 24.5GB，物理碎片化后 cumem 找不到连续页；多进程 CUDA context + NCCL + PyTorch cache 叠加塞爆 |
| **0.60** | 崩 | `CUDA out of memory` at FSDP worker 加载回 GPU 阶段（`WorkerDict.actor_rollout_update_weights`） | vLLM 实际占 18.60GB + FSDP 12.46GB = 31.06GB，**只剩 229MB**，FSDP 再要 396MB 爆了。证实同进程共用物理卡 |
| **0.50** | 崩 | vLLM init 阶段 `ValueError: No available memory for the cache blocks` | 预算 15.68GB − 模型 13GB = 2.68GB 给 KV，vLLM 拒绝分配最小 KV block |

### 12.7 关键新认识：cumem 的虚拟地址预留永远存在

- 即使 `sleep(level=1)` unmap 物理页，vLLM 进程的 `gpu_memory_utilization × 31.36` 虚拟地址段**始终占用**；
- 所以 FSDP 和 vLLM **共享物理 GPU 时峰值 = vLLM 虚拟预算 + FSDP shard**；
- 31.36GB ÷ (13 FSDP + 13 vLLM weights + KV + overhead) → **5090 是临界**，再想挤只能靠降 `ROLLOUT_MAX_NUM_SEQS` 去压 KV；
- 进一步不能降就得改 colocate 的架构（例如把 actor FSDP 和 vLLM 排到不同物理卡上）。

### 12.8 下一步（正在试）

- **smoke 新组合**（当前跑中，log `/tmp/grpo_smoke_20260424_201709.log`）：
  ```bash
  ROLLOUT_GPU_MEMORY_UTILIZATION=0.55 ROLLOUT_MAX_NUM_SEQS=8 \
  TARGET_MODULES="[o_proj]" LORA_RANK=8 LORA_ALPHA=8 \
  bash train_grpo_lora_actor_verl.sh \
    trainer.total_training_steps=2 trainer.save_freq=100000 trainer.test_freq=100000
  ```
- 0.55 给 vLLM 17.25GB（13GB 模型 + 4.25GB KV，max_num_seqs=8 够装）；
- 留给 FSDP reload 约 31.36 − 17.25 − 2 (ctx 等) = 12.1GB，可能还差 1GB，再降 `MAX_PROMPT_LENGTH/MAX_RESPONSE_LENGTH` 或 `ROLLOUT_N` 应对。

### 12.9 备选降显存参数（按优先级）

1. **`ROLLOUT_MAX_NUM_SEQS=8`**（当前已用）— 把 KV block 最小要求从 32 × 2816 降到 8 × 2816 tokens
2. **`MAX_PROMPT_LENGTH=1024 MAX_RESPONSE_LENGTH=512`**（默认 2048+768=2816）— 进一步压 KV/max_model_len
3. **`ROLLOUT_N=2`**（默认 4）— GRPO 每 prompt 采样份数减半
4. **`PPO_MICRO_BATCH_SIZE_PER_GPU=1`**（默认 2）— 训练阶段激活显存减半
5. **`PPO_MAX_TOKEN_LEN_PER_GPU=4096`**（默认 8192）— 动态打包上限减半
6. **`ROLLOUT_TP_SIZE=8`**：让 actor vLLM 横跨 8 卡（但 RM 要占 GPU 4-7，冲突；需另规划）
7. **把 actor FSDP 和 actor vLLM 排到不同物理卡**：需改 resource pool，Phase 0 不做
8. **FP8 KV cache**（`kv_cache_dtype=fp8`）— KV 显存减半，但 5090 对 fp8 kv cache 的硬件支持要查

> 如果 12.8 的 0.55 也崩，下一步按 2→3→4 递增降。切忌一次改太多，否则无法定位。

### 12.10 当前 smoke 观察流程

- 让 `Monitor` 盯 `/tmp/grpo_smoke_*.log`，grep 触发项：`update_weights done | time cost | generate_sequences.*done | reward | rm_scores | cumem_allocator | CUDA.*out of memory | Error executing job`；
- 成功判据：看到 **`update_weights done`** + **至少一次 rollout** + **至少一个 reward/rm_scores** 即 smoke 通过。

---

## 13) 2026-04-29 第 3 轮接续：reorder + GPU swap + RM 不睡（重大突破）

### 13.1 实际跑通的关键修复（按落地顺序）

**A. reorder `engine_workers.update_weights` 让 FSDP/vLLM 不同时占 GPU**

文件：`/home/ps/hanjiaqi/verl/verl/workers/engine_workers.py`

之前的顺序：`resume(weights)` → `get_per_tensor_param`（FSDP 加载到 GPU）→ IPC 推送，导致 vLLM 13GB + FSDP 13GB 同卡 OOM。

新顺序：
```python
set_expandable_segments(False)
log_gpu_memory_usage("Before collect weights", logger=logger, level=logging.INFO)
per_tensor_param, peft_config = self.actor.engine.get_per_tensor_param(
    layered_summon=self.layered_summon, base_sync_done=self.base_sync_done
)  # FSDP 加载 → 提取 LoRA 到 CPU → 自动 offload FSDP 回 CPU
aggressive_empty_cache(force_sync=True)
log_gpu_memory_usage("Before resume weights", logger=logger, level=logging.INFO)
if self.config.rollout.free_cache_engine:
    await self.rollout.resume(tags=["weights"])  # 此时 GPU 空，wake_up 安全
log_gpu_memory_usage("After resume weights", logger=logger, level=logging.INFO)
```

**B. CUDA_VISIBLE_DEVICES 互换 GPU pool**

GPU 0 长期被 root 用户的 `python -u main.py` 占了 1.9GB。原 layout：actor 在 0-3（受影响）、RM 在 4-7（干净）。

新 layout：`CUDA_VISIBLE_DEVICES=4,5,6,7,0,1,2,3`：
- verl 看到的"global_pool" GPU 0-3 → 物理 4-7（actor，干净）
- verl 看到的"reward_pool" GPU 4-7 → 物理 0-3（RM 容忍 squatter）

**C. `MAX_MODEL_LEN=3072`（之前 2816）**

第一次 smoke 跑通 update_weights 后，rollout + RM /classify 阶段卡 token-length：
```
VLLMValidationError: This model's maximum context length is 2816 tokens. However, you requested 0 output tokens and your prompt contains at least 2817 input tokens
```
Chat template 在 prompt+response 之外加了 1 个 token（BOS/format header）。给 256 token buffer，RM 不再返回 400。

**D. `reward.reward_model.rollout.free_cache_engine=False`（关键最后一步）**

[reward_model.py:47-48](hanjiaqi/verl/verl/experimental/reward_loop/reward_model.py#L47-L48):
```python
if self.config.rollout.free_cache_engine:
    self.sleep()  # RM init 完立刻睡了
```

但我们用的是 **async agent reward loop**（`enable_resource_pool=True`），通过 HTTP `/classify` 调 RM。这条路径**不会先 wake_up RM**，请求挂在 sleeping RM 上**永远不返回**（aiohttp `total=None` 不超时）。

修复：让 RM 不自动 sleep。RM 在物理 GPU 0-3，actor 在 GPU 4-7，本来就不抢显存，RM 一直醒着 ~19GB 占用没问题。

**E. `trainer.resume_mode=disable`**

避免自动加载之前残留的 `global_step_2` checkpoint（一些早期失败 smoke 已经写入），保证从 base 模型干净起步。

### 13.2 完整 smoke 启动命令（v4，已验证可跑到 rollout）

```bash
cd /home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/framework_bridge/verl
LOG="/tmp/grpo_smoke_$(date +%Y%m%d_%H%M%S).log"
tmux new -d -s grpo_smoke "conda run -n gemma4-rl --no-capture-output bash -c '
  cd /home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/framework_bridge/verl
  export CUDA_VISIBLE_DEVICES=4,5,6,7,0,1,2,3
  export CUDA_LAUNCH_BLOCKING=1
  export VERL_LOGGING_LEVEL=INFO
  TRAIN_BATCH_SIZE=32 TARGET_MODULES=\"[o_proj]\" LORA_RANK=8 LORA_ALPHA=8 \
  MAX_MODEL_LEN=3072 \
  bash train_grpo_lora_actor_verl.sh \
    trainer.total_training_steps=2 \
    trainer.save_freq=100000 \
    trainer.test_freq=100000 \
    trainer.resume_mode=disable \
    reward.reward_model.rollout.free_cache_engine=False 2>&1 | tee \$LOG
'"
```

### 13.3 当前观测：smoke v4 进入 rollout 阶段（17:01-17:09+）

- `Before collect weights` 5.59GB → `Before resume weights` 5.59GB ✓（reorder 生效，FSDP 没占 GPU）
- `vLLM load weights, loaded_params: 60` ✓（60 = 30 layers × o_proj × 2 LoRA-A/B）
- `Training Progress: 0%|...| 0/2` ✓（训练循环开始）
- 接着 4 分钟无 INFO 输出，但 GPU 持续大量占用：
  - GPU 4-7（actor）: **26GB / 31GB, util 60-93%** — 实际在做 rollout generation
  - GPU 0-3（RM）: **21-25GB / 31GB**（不睡），util 0%（等 actor 出 response 才会调）
  - VLLM Worker_TP CPU 各 72%，EngineCore 59% — 实际计算中
- `is_fx_tracing` warning 出现 → FSDP 在 trace `compute_log_prob`

**结论：reorder + GPU swap + RM 不睡 三个修复一起让 smoke 真正跑起来了。** 之前所有 OOM/illegal-memory/cumem 错误全清掉了。

### 13.4 仍需观察 / 可能的下一坑

1. **rollout 巨慢**：`enforce_eager=True` + `CUDA_LAUNCH_BLOCKING=1` 让 generation 慢 100×（每个 CUDA op 同步）。后续可在稳定后**移除** `CUDA_LAUNCH_BLOCKING=1`，速度会提升至少 10×。
2. **`actor_rollout_compute_log_prob` 阶段**：如果失败可能是 actor FSDP 在 GPU 4-7 上做 forward 时显存不够（rollout 完后 vLLM 还占 26GB；actor train 需要把 vLLM 睡掉再上 FSDP）。这个 sleep/wake 切换由 verl 自动管理。
3. **LoRA delta 推送回 vLLM 阶段**：每步 update_weights 的 IPC 现在只推 60 个 o_proj LoRA 参数（很小），不应再 OOM。但 set_lora 的 illegal memory access 是否真的修了，要等第二步 update 验证。
4. **Phase 1 计划**：smoke 通过后，扩到 `TARGET_MODULES=[q,k,v,o]_proj LORA_RANK=32 LORA_ALPHA=32` 跑 50 step pilot。

### 13.5 仍生效的之前修改（不要回滚）

- `vllm_rollout.py`：`_has_slept` 已撤回，恢复原始 `resume/release`；
- `engine_workers.py`：保留"任意 sleep_level 都 resume(weights)"的注释和逻辑；
- `utils.py`：set_lora sync-guard（cuda.synchronize + CPU detach + 详细日志）作为安全网。

### 13.6 给 Codex 接手的关键提示

如果 smoke v4 最终跑通，**记住下面这些设置必须长期生效**（不要回退）：
1. `engine_workers.py` 中 `get_per_tensor_param` 必须在 `resume(weights)` 之前；
2. 同主机有 GPU squatter 时用 `CUDA_VISIBLE_DEVICES` 重排 actor 到干净卡；
3. 用 async reward loop 时 `reward.reward_model.rollout.free_cache_engine=False`（因为 verl 没在 async 路径里 wake RM）；
4. RM 的 chat template 会加 1 token，`MAX_MODEL_LEN ≥ MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 16` 留 buffer。

---

## 14) 🎉 2026-04-29 17:28 — SMOKE TEST 完整跑通 2/2 步

### 14.1 关键指标（log: `/tmp/grpo_smoke_20260429_170131.log`）

**Step 1**（17:05-17:12）：
- `actor/loss: 0.0734`、`pg_loss: 0.0721`、`kl_loss: 1.36`、`grad_norm: 1.05`、`lr: 3e-6`
- `actor/ppo_kl: -0.0074`（GSPO clip 工作正常）、`pg_clipfrac: 0.148`
- `critic/score/mean: -1.00`、`max: -0.075`、`min: -3.12`
- `response_length/mean: 308`、`max: 609`、`min: 13`、**`aborted_ratio: 0.0`**
- timing: step=379s（gen 168s + update_actor 129s + ref 36s + old_log_prob 41s + **update_weights 4.4s**）

**Step 2**（17:12-17:28，包含 validation 和 save_checkpoint）：
- `actor/loss: 0.0047`（**↓15× vs step1**，模型实际在学）、`pg_loss: 0.0031`
- `actor/grad_norm: 1.10`、`pg_clipfrac: 0.117`（↓ 说明更新更接近 GSPO ratio）
- `critic/score/mean: -0.81`（**↑19% vs step1**，reward 在改善）
- `val-aux/chatjoy/reward: -0.85`、`val-aux/chatjoy/rm_score: -0.62`、`custom_reward: -0.23`
- timing: step=409s + **testing 553s + save_checkpoint 51s**，update_weights **5.0s**

**全部成功标志达成**：
1. ✅ 第 1/2 次 update_weights → vLLM `loaded_params: 60`（30 layers × o_proj × 2 LoRA-A/B），无 cumem/illegal-memory；
2. ✅ rollout `Rollout.generate_sequences` 正常完成，`aborted_ratio=0.0`；
3. ✅ RM `/classify` 返回 reward（async loop，RM 一直醒）；
4. ✅ actor update_actor 完成 GRPO+GSPO loss 梯度更新；
5. ✅ 第 2 步 LoRA hot-swap 仍稳定（update_weights 5.0s）；
6. ✅ validation + save_checkpoint 都正常。

### 14.2 这套修复 = 5 个独立问题的串联解决

按发现顺序：
1. **set_lora illegal memory** ← cumem sleep_level=1 把 LoRA buffer 一起 unmap → 加"任意 sleep 都 resume(weights)"
2. **cumem wake_up OOM** ← FSDP 13GB + vLLM 13GB 同时占 GPU → reorder `get_per_tensor_param` 在 `resume(weights)` 之前
3. **GPU 0 squatter** ← root 用户的进程占 1.9GB → `CUDA_VISIBLE_DEVICES=4,5,6,7,0,1,2,3` swap pool
4. **RM /classify 返回 400** ← chat template 加 1 token 超 max_model_len → `MAX_MODEL_LEN=3072`
5. **RM /classify 永远不返回（hang）** ← RewardModelManager init 后立刻 sleep，但 async loop 不 wake_up → `reward.reward_model.rollout.free_cache_engine=False`

任何一个不修，smoke 都过不了。

### 14.3 性能基线（CUDA_LAUNCH_BLOCKING=1 的 debug 模式）

- **每步 ~6-7 分钟**，主要消耗在 generation（44%）和 update_actor（34%）
- `enforce_eager=True` + `CUDA_LAUNCH_BLOCKING=1` 让 CUDA 同步执行，预计去掉后能快 5-10×
- update_weights 4.4-5.0s 已经很快，LoRA delta IPC 是高效的

### 14.4 接下来要做的事（Phase 1 pilot）

1. **去掉 `CUDA_LAUNCH_BLOCKING=1`** — 加速 5-10×
2. **扩展 LoRA 范围**：`TARGET_MODULES=[q_proj,k_proj,v_proj,o_proj] LORA_RANK=32 LORA_ALPHA=32` — 验证多模块 hot-swap 仍稳
3. **跑 50 step pilot**：`trainer.total_training_steps=50 trainer.save_freq=25 trainer.test_freq=10`
4. **观察 reward 曲线**：critic/score/mean 应从 -1.0 → 上升到 0+ 区间
5. **若 50 step 通过 → Phase 2**：1 epoch 完整训练（~305 step，预计 8-12 小时）

### 14.5 已落地补丁清单（不要回滚）

| 文件 | 修改 | 作用 |
|------|------|------|
| `verl/workers/engine_workers.py` | `update_weights` reorder：`get_per_tensor_param` 提前到 `resume(weights)` 之前 | 避免 FSDP+vLLM 同 GPU OOM |
| `verl/workers/engine_workers.py` | "任意 sleep_level 都 resume(weights)"（替换原 `!= 1`） | 修 cumem unmap LoRA buffer |
| `verl/workers/rollout/vllm_rollout/vllm_rollout.py` | 撤回 `_has_slept`，恢复原始 resume/release | 路径无关性，状态追踪走错位置 |
| `verl/utils/vllm/utils.py` | set_lora guard 加 `cuda.synchronize` + CPU detach + 详细日志 | 安全网，捕捉残余异常 |
| `verl/utils/vllm/utils.py` | `_patch_gemma4_mm_lora_support` 注入 LoRA 协议属性 | Gemma4 MM 支持 LoRA |
| `verl/utils/vllm/patch.py` | Gemma4 MoE inner_model.model.layers 路径处理 | 修 weight loader |

### 14.6 Smoke 实测命令（已跑通，留存）

```bash
cd /home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/framework_bridge/verl
LOG="/tmp/grpo_smoke_$(date +%Y%m%d_%H%M%S).log"
tmux new -d -s grpo_smoke "conda run -n gemma4-rl --no-capture-output bash -c '
  cd /home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/framework_bridge/verl
  export CUDA_VISIBLE_DEVICES=4,5,6,7,0,1,2,3
  export CUDA_LAUNCH_BLOCKING=1
  export VERL_LOGGING_LEVEL=INFO
  TRAIN_BATCH_SIZE=32 TARGET_MODULES=\"[o_proj]\" LORA_RANK=8 LORA_ALPHA=8 \
  MAX_MODEL_LEN=3072 \
  bash train_grpo_lora_actor_verl.sh \
    trainer.total_training_steps=2 \
    trainer.save_freq=100000 \
    trainer.test_freq=100000 \
    trainer.resume_mode=disable \
    reward.reward_model.rollout.free_cache_engine=False 2>&1 | tee \$LOG
'"
```

---

## 13) 2026-04-24 20:40-21:19 Codex 接手进展（当前最新）

### 13.1 已确认的新进展

1. 首次 `update_weights` 已经跑通：
   - 最新可用日志：`/tmp/grpo_smoke_codex_20260424_210802.log`
   - actor vLLM 初始化完成后进入 sleep；
   - `Before collect weights -> Before resume weights -> vLLM load weights, loaded_params: 60` 正常出现；
   - 说明 vLLM V1 cumem sleep/resume + LoRA hot-swap 的最早阻塞已经越过，不再卡在 `set_lora invalid argument/illegal memory access`。

2. rollout + RM 链路已经越过：
   - 日志显示 actor rollout 生成完成后，actor vLLM 再次 sleep；
   - RM vLLM 在 GPU 4-7 上启动和 sleep 正常；
   - 之前 `compute_reward()` 不兼容 verl 新参数的问题已修。

3. old logprob 链路已经越过：
   - 之前失败在 `compute_old_log_prob -> entropy_from_logits(logits)`，需要额外分配 3.7-4.9GB，导致 CUDA OOM；
   - 现在把 old-logprob 阶段的 entropy 改为按配置计算，`ENTROPY_COEFF=0` 时不再强制计算 `entropys`；
   - 最新 smoke 已经越过 old-logprob，失败点后移到 actor update。

### 13.2 本轮已落地修改

1. `/home/ps/hanjiaqi/verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py`
   - 在 `vLLMHttpServer` 增加 `_sleeping_tags` 状态；
   - 拦截 `collective_rpc("wake_up")`，避免重复/无意义 wake；
   - sleep level 1 按 `{"weights", "kv_cache"}` 记录，避免只恢复 weights 时遗漏 vLLM 内部被 discard 的 KV/cumem 区域。

2. `/home/ps/hanjiaqi/verl/verl/workers/engine_workers.py`
   - `update_weights()` 顺序调整为：vLLM 仍 sleep 时先从 actor FSDP 收集 LoRA 参数，然后清 CUDA cache，再 `rollout.resume(tags=["weights"])`；
   - 增加 `Before collect weights / Before resume weights / After ...` 显存日志；
   - 该改动是首次 `vLLM load weights, loaded_params: 60` 能稳定出现的关键。

3. `/home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/anti_refusal_reward.py`
   - `compute_reward(...)` 增加 `**kwargs`；
   - 修复 verl 传入 `reward_router_address` 等额外字段导致的 `unexpected keyword argument`。

4. `/home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/framework_bridge/verl/train_grpo_lora_actor_verl.sh`
   - 新增 `HF_ATTN_IMPLEMENTATION=${HF_ATTN_IMPLEMENTATION:-sdpa}`；
   - 通过 `+actor_rollout_ref.model.override_config.attn_implementation=${HF_ATTN_IMPLEMENTATION}` 覆盖 HF actor/ref；
   - 当 HF attention 不是 flash-attn 时自动 `USE_REMOVE_PADDING=False`；
   - 原因：Gemma4 text global head dim = 512，HF flash-attn 报 `head dimension at most 256`。vLLM rollout 仍用自己的 backend，不受这个 HF 配置影响。

5. `/home/ps/hanjiaqi/verl/verl/trainer/ppo/ray_trainer.py`
   - `_compute_old_log_prob()` 不再无条件 `calculate_entropy=True`；
   - 只有 `actor.calculate_entropy=True` 或 `entropy_coeff != 0` 时才计算 `entropys`；
   - 主循环只在 `old_log_prob.batch` 中存在 `entropys` 时记录 `actor/entropy` 并 pop；
   - smoke 用 `ENTROPY_COEFF=0` 先跑通主链路。

### 13.3 本轮遇到并解决/绕过的问题

| 日志 | 结果 | 处理 |
|---|---|---|
| `/tmp/grpo_smoke_codex_20260424_204226.log` | update_weights/rollout 已过，reward 阶段报 `compute_reward() got an unexpected keyword argument 'reward_router_address'` | reward 函数加 `**kwargs` |
| `/tmp/grpo_smoke_codex_20260424_204833.log` | reward 已过，old logprob 报 `FlashAttention forward only supports head dimension at most 256` | HF actor/ref 改 SDPA，关闭 remove padding |
| `/tmp/grpo_smoke_codex_20260424_205835.log` | SDPA 生效，但 old logprob 的 entropy 计算 OOM | old-logprob entropy 改为配置控制，smoke 用 `ENTROPY_COEFF=0` |
| `/tmp/grpo_smoke_codex_20260424_210802.log` | old logprob 已过，actor update 报 `ValueError: mm_token_type_ids is required as a model input when training` | 当前最新阻塞，下一步补 text-only `mm_token_type_ids=zeros_like(input_ids)` |

### 13.4 当前最新阻塞

- 失败位置：`actor_rollout_update_actor -> train_batch -> Gemma4ForConditionalGeneration.forward`
- 根因：Transformers Gemma4 在 `self.training == True` 时要求 `mm_token_type_ids`。当前 ChatJoy RL 是 text-only 输入，但模型仍是 multimodal `Gemma4ForConditionalGeneration`，训练前向不能省这个字段。
- 栈顶真实异常：
  - `transformers/models/gemma4/modeling_gemma4.py:create_causal_mask_mapping`
  - `ValueError: mm_token_type_ids is required as a model input when training`
- 后面的 `offload_fsdp_model_to_cpu AssertionError` 是异常清理阶段的二次错误，不是主因。

### 13.5 下一步

1. 在 FSDP engine 的 `prepare_model_inputs()` 里给 Gemma4 text-only 训练补：
   - `model_inputs["mm_token_type_ids"] = torch.zeros_like(input_ids)`
   - 只在模型需要/字段不存在时补，避免影响其他模型；
2. 重新跑同一条 2-step smoke：
   - `TRAIN_BATCH_SIZE=32`
   - `PPO_MICRO_BATCH_SIZE_PER_GPU=1`
   - `LOGPROB_MICRO_BATCH_SIZE_PER_GPU=1`
   - `PPO_MAX_TOKEN_LEN_PER_GPU=4096`
   - `HF_ATTN_IMPLEMENTATION=sdpa`
   - `USE_REMOVE_PADDING=False`
   - `ENTROPY_COEFF=0`
   - `TARGET_MODULES=[o_proj] LORA_RANK=8 LORA_ALPHA=8`
3. 如果 actor update 之后再 OOM，再按 smoke 优先级降：
   - `MAX_RESPONSE_LENGTH=512/384`
   - `TRAIN_BATCH_SIZE=16`
   - `ROLLOUT_N=2`
   - `PPO_MAX_TOKEN_LEN_PER_GPU=2048`

### 13.6 RM GPU 4-7 显存占用备注

- 截图中 GPU 4-7 的 `VLLM::Worker_TP*` 只占约 1002MiB，是因为 RM vLLM 已经执行了 sleep，权重物理页被 offload/unmap，只剩 CUDA context、worker 运行时和少量常驻内存；
- RM 真正 active 时不是 1GB：前面日志显示 RM 模型加载约 14.16GiB/卡，KV cache 约 5.18GiB/卡，sleep 后才降到约 1.02GiB；
- 2026-04-24 21:19 复查 `nvidia-smi` 时训练失败后进程已清理，GPU 0-7 基本空闲，截图是失败清理前/清理中的瞬时状态。

### 13.7 释放 GPU 4-7 给 actor/ref/rollout 的可能性

结论：**可以作为下一阶段架构调整方向，但不建议在当前 P0 smoke 前先做**。当前最新阻塞是 actor update 缺 `mm_token_type_ids`，不是 RM 显存。

可选方案：

1. 临时禁用 GPU RM：
   - 设置 `reward.reward_model.enable=False` 或增加脚本级 `ENABLE_RM=False` 开关；
   - GPU 4-7 释放后，可把 actor/ref/rollout 规划到 8 卡；
   - 代价是 smoke 只能验证 actor/RL 主链路，不能验证真实 RM 打分链路。

2. RM 改 CPU/外部服务：
   - 理论上能释放 GPU 4-7；
   - 但 Qwen3-30B-MoE RM 在 CPU 上会非常慢，并需要大量 CPU RAM；
   - 更实际的做法是把 RM 放到独立机器或独立 GPU 服务，通过 reward router/RPC 调用。

3. actor/rollout/ref 改 8 卡布局：
   - 需要重配 resource pool，例如 `trainer.n_gpus_per_node=8`，并重新规划 `ROLLOUT_TP_SIZE=8`；
   - 潜在收益：actor vLLM 每卡权重/KV 压力下降，FSDP/ref/rollout 共存余量更大；
   - 风险：现有脚本假设 global_pool 用 GPU 0-3、reward_pool 用 GPU 4-7，直接改 8 卡会和 RM 资源池冲突；还要重新验证 Ray placement、FSDP shard、ref logprob 和 vLLM TP。

建议顺序：

1. 先补 `mm_token_type_ids`，用当前 4+4 布局把 2-step smoke 主链路继续往前推；
2. 如果后续再次遇到 actor update 或 ref/actor logprob 显存瓶颈，再做一个 `ENABLE_RM=False + trainer.n_gpus_per_node=8 + ROLLOUT_TP_SIZE=8` 的 actor-only smoke；
3. actor-only 8 卡 smoke 通过后，再考虑把 RM 改成外部服务或低频异步打分，而不是让 RM 常驻占 GPU 4-7。

---

## 14) 2026-04-27 smoke 接续状态（当前最新）

### 14.1 已补齐的旧阻塞

1. `mm_token_type_ids` 已修复：
   - `/home/ps/hanjiaqi/verl/verl/utils/chat_template.py` 会在裁剪 dummy user prefix 时同步裁剪 `mm_token_type_ids`；
   - `/home/ps/hanjiaqi/verl/verl/workers/engine/fsdp/transformer_impl.py` 会在 Gemma4 训练态缺少该字段时补 `zeros_like(input_ids)`；
   - 旧错误 `ValueError: mm_token_type_ids is required as a model input when training` 已不再是当前阻塞。

2. `PPO_MAX_TOKEN_LEN_PER_GPU=4096` 的 actor backward OOM 已通过降 token micro-batch 绕过：
   - 旧失败日志：`/tmp/grpo_smoke_codex_20260424_212349.log`；
   - 失败点：`actor_rollout_update_actor -> loss.backward()`；
   - 现象：需要额外约 2.78GiB，GPU0 只剩约 2.60GiB，sleep 后 actor vLLM 仍有约 4.74GiB 常驻。

### 14.2 最新通过的 smoke

- 最新通过日志：`/tmp/grpo_smoke_codex_20260427_095422_tok3072.log`
- 关键 override：
  - `TRAIN_BATCH_SIZE=32`
  - `PPO_MICRO_BATCH_SIZE_PER_GPU=1`
  - `LOGPROB_MICRO_BATCH_SIZE_PER_GPU=1`
  - `PPO_MAX_TOKEN_LEN_PER_GPU=3072`
  - `HF_ATTN_IMPLEMENTATION=sdpa`
  - `USE_REMOVE_PADDING=False`
  - `ENTROPY_COEFF=0`
  - `TARGET_MODULES=[o_proj]`
  - `LORA_RANK=8`
  - `LORA_ALPHA=8`
- 结果：
  - `Training Progress: 100%|...| 2/2`；
  - 保存了 `global_step_2`；
  - step1/step2 均打印 actor loss、KL、grad norm、reward、response length 等指标；
  - 没有训练主链路 `OutOfMemory`；
  - 没有再次出现 `mm_token_type_ids` 错误。
- 尾部的 `resource_tracker KeyError` / vLLM worker SIGTERM 出现在训练完成后的进程清理阶段，目前按清理噪声处理；若后续正式长跑中导致退出码异常，再单独处理。

### 14.2.1 2026-04-27 晚间复跑结果

- 当前复跑日志：`/tmp/grpo_smoke_codex_20260427_190131_tok3072_rerun.log`
- 当前保存目录：`/home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/verl/chatjoy_grpo_lora_smoke_20260427_190131_tok3072`
- 启动前状态：
  - GPU 0-7 无无关 compute app；
  - 未发现残留 Ray/vLLM/training 进程占 GPU。
- 已确认配置：
  - `PPO_MAX_TOKEN_LEN_PER_GPU=3072`
  - `TRAIN_BATCH_SIZE=32`
  - `PPO_MICRO_BATCH_SIZE_PER_GPU=1`
  - `LOGPROB_MICRO_BATCH_SIZE_PER_GPU=1`
  - `HF_ATTN_IMPLEMENTATION=sdpa`
  - `USE_REMOVE_PADDING=False`
  - `ENTROPY_COEFF=0`
  - `TARGET_MODULES=[o_proj]`
  - `LORA_RANK=8`
  - `LORA_ALPHA=8`
- 关键进展：
  - RM vLLM 和 actor vLLM 均完成 sleep；
  - actor vLLM `Before collect weights` / `Before resume weights` / `vLLM load weights` marker 均出现；
  - 第 1/2 step 已完成：`Training Progress: 50%|...| 1/2 [06:29<06:29, 389.70s/it]`；
  - 第 2/2 step 已完成：`Training Progress: 100%|...| 2/2 [22:46<00:00, 683.27s/it]`；
  - 已保存 `global_step_2`；
  - 主进程退出码为 0；
  - GPU compute app 已清空。
- step2 关键指标：
  - `training/global_step:2`
  - `actor/loss:0.027697949670255184`
  - `actor/grad_norm:0.417794793844223`
  - `critic/rewards/mean:-0.171875`
  - `response_length/mean:331.5859375`
  - `prompt_length/mean:1267.53125`
  - `timing_s/gen:183.60701438901015`
  - `timing_s/update_actor:128.1593143279897`
  - `timing_s/save_checkpoint:41.99887735408265`
  - `timing_s/update_weights:6.441442986950278`
  - `timing_s/step:434.24400405201595`
  - `timing_s/testing:542.5721716930857`
  - `perf/throughput:58.92050266958941`
- 最终验证：
  - `val-aux/chatjoy/reward/mean@1:-0.26062322946175637`
  - `val-core/chatjoy/acc/mean@1:-0.26062322946175637`
  - `val-aux/num_turns/mean:2.0`
- 仍需注意：
  - 尾部 `resource_tracker KeyError` / Ray worker SIGTERM 仍复现，但出现在 `Training Progress: 100%` 和 `Final validation metrics` 之后，且主命令退出码为 0，当前仍按清理噪声处理；
  - 最终 validation 会在 last step 强制触发，即使 `trainer.test_freq=100000`，本次耗时约 542.57s；
  - checkpoint 写入 4 个 FSDP rank，每个 model shard 约 13.26GiB，总 checkpoint 体积约 53GiB，正式长跑前需要规划 `max_actor_ckpt_to_keep` 或定期清理。

### 14.2.2 q/k/v/o rank8 nosave probe 结果

- probe 日志：`/tmp/grpo_probe_codex_20260427_194342_qkvo_rank8_nosave.log`
- checkpoint 目录参数：`/tmp/grpo_probe_codex_20260427_194342_qkvo_rank8_nosave_ckpt_disabled`
- 启动前状态：
  - GPU 0-7 无无关 compute app；
  - 未发现残留 Ray/vLLM/training 进程占 GPU。
- 已确认配置：
  - `TARGET_MODULES=[q_proj,k_proj,v_proj,o_proj]`
  - `LORA_RANK=8`
  - `LORA_ALPHA=8`
  - `ENTROPY_COEFF=0`
  - `PPO_MAX_TOKEN_LEN_PER_GPU=3072`
  - `TRAIN_BATCH_SIZE=32`
  - `PPO_MICRO_BATCH_SIZE_PER_GPU=1`
  - `LOGPROB_MICRO_BATCH_SIZE_PER_GPU=1`
  - `trainer.total_training_steps=2`
  - `trainer.save_freq=0`
  - `trainer.test_freq=0`
- 结果：
  - `Training Progress: 100%|...| 2/2 [13:00<00:00, 390.44s/it]`；
  - 主命令退出码为 0；
  - `Final validation metrics: None`，确认 final validation 被跳过；
  - 未创建 checkpoint 目录，未写 `global_step_*` checkpoint；
  - GPU compute app 已清空。
- 关键发现：
  - q/k/v/o rank8 的 LoRA hot-swap 成功，`vLLM load weights, loaded_params: 230`；
  - 对照 o_proj rank8 的 `loaded_params: 60`，本轮确实覆盖了更多 attention LoRA 参数；
  - 第 1/2 step actor update 峰值已很紧，但通过；
  - 第 2/2 step actor update 观测到 GPU2 只剩约 `12 MiB`，GPU0/GPU3 也低于 `300 MiB`，说明 q/k/v/o rank8 在当前配置下显存余量极薄；
  - 不建议下一步直接升到 `LORA_RANK=32` 或同时恢复 `ENTROPY_COEFF=0.01`。
- step2 关键指标：
  - `training/global_step:2`
  - `actor/loss:0.003380758687853813`
  - `actor/grad_norm:0.4673628807067871`
  - `critic/rewards/mean:-0.109375`
  - `response_length/mean:330.28125`
  - `prompt_length/mean:1267.53125`
  - `timing_s/gen:174.5918600129662`
  - `timing_s/old_log_prob:39.46527556295041`
  - `timing_s/ref:36.120738254045136`
  - `timing_s/update_actor:122.35395839298144`
  - `timing_s/update_weights:5.355658838059753`
  - `timing_s/step:377.9265150469728`
  - `perf/throughput:67.64542571674947`
- 尾部：
  - `resource_tracker KeyError` / Ray worker SIGTERM 仍出现，但在 `Training Progress: 100%` 和 `step:2` metrics 后，且主命令退出码为 0，继续按清理噪声处理。
- 下一步建议：
  - `q/k/v/o rank8 + actor_rollout_ref.model.enable_activation_offload=True` 已执行，见 14.2.3；
  - `q/k/v/o rank16` 或更高 rank 必须预期 OOM 风险很高；
  - 暂不恢复 `TRAIN_BATCH_SIZE=64`、`PPO_MICRO_BATCH_SIZE_PER_GPU=2`、`LOGPROB_MICRO_BATCH_SIZE_PER_GPU=4`、`ENTROPY_COEFF=0.01`。

### 14.2.3 q/k/v/o rank8 + activation offload nosave probe 结果

- probe 日志：`/tmp/grpo_probe_codex_20260427_201532_qkvo_rank8_actoff_nosave.log`
- checkpoint 目录参数：`/tmp/grpo_probe_codex_20260427_201532_qkvo_rank8_actoff_nosave_ckpt_disabled`
- 已确认配置：
  - `TARGET_MODULES=[q_proj,k_proj,v_proj,o_proj]`
  - `LORA_RANK=8`
  - `LORA_ALPHA=8`
  - `ENTROPY_COEFF=0`
  - `PPO_MAX_TOKEN_LEN_PER_GPU=3072`
  - `TRAIN_BATCH_SIZE=32`
  - `PPO_MICRO_BATCH_SIZE_PER_GPU=1`
  - `LOGPROB_MICRO_BATCH_SIZE_PER_GPU=1`
  - `actor_rollout_ref.model.enable_activation_offload=True`
  - `trainer.total_training_steps=2`
  - `trainer.save_freq=0`
  - `trainer.test_freq=0`
- 结果：
  - `Training Progress: 100%|...| 2/2 [13:20<00:00, 400.30s/it]`；
  - 主命令退出码为 0；
  - `Final validation metrics: None`，确认 final validation 被跳过；
  - checkpoint 目录未创建，未写 `global_step_*` checkpoint；
  - 退出后 GPU compute app 已清空。
- 关键发现：
  - actor 侧 `enable_activation_offload=True` 生效；
  - q/k/v/o rank8 的 LoRA hot-swap 仍成功，`vLLM load weights, loaded_params: 230`；
  - activation offload 没有明显改善当前 q/k/v/o rank8 的 actor update 峰值显存余量：第 1/2 step 采样到 GPU0 仅剩约 `52 MiB`、GPU1 `92 MiB`、GPU2 `112 MiB`；第 2/2 step 最低采样约 `424 MiB`；
  - 对比 14.2.2 no-offload：step1 `update_actor` 从 `136.35s` 变为 `143.31s`，step2 从 `122.35s` 变为 `137.90s`，训练更新阶段变慢；
  - 因此不建议把 activation offload 作为 q/k/v/o rank8 的默认开关；它可作为 OOM 时的兜底尝试，但这次没有提供稳定显存余量。
- step1 关键指标：
  - `training/global_step:1`
  - `actor/loss:0.0037131905555725098`
  - `actor/grad_norm:0.2909170091152191`
  - `critic/rewards/mean:-0.234375`
  - `response_length/mean:299.015625`
  - `prompt_length/mean:1271.09375`
  - `timing_s/gen:156.59492379205767`
  - `timing_s/old_log_prob:45.719349906081334`
  - `timing_s/ref:36.86536124208942`
  - `timing_s/update_actor:143.31350389192812`
  - `timing_s/update_weights:7.118644749047235`
  - `timing_s/step:389.64393517107237`
  - `perf/throughput:64.4736071381949`
- step2 关键指标：
  - `training/global_step:2`
  - `actor/loss:0.005571899004280567`
  - `actor/grad_norm:3.781763434410095`
  - `critic/rewards/mean:-0.125`
  - `response_length/mean:323.0546875`
  - `prompt_length/mean:1267.53125`
  - `timing_s/gen:189.03501736698672`
  - `timing_s/old_log_prob:40.17493632307742`
  - `timing_s/ref:37.69613167701755`
  - `timing_s/update_actor:137.90214827097952`
  - `timing_s/update_weights:5.891693428973667`
  - `timing_s/step:410.7461338340072`
  - `perf/throughput:61.95889115850992`
- 尾部：
  - `resource_tracker KeyError` / Ray worker SIGTERM 仍出现，但在 `Training Progress: 100%`、`step:2` metrics 和 `Final validation metrics: None` 后，且主命令退出码为 0，继续按清理噪声处理。
- 下一步建议：
  - 当前 4 GPU 配置下，`q/k/v/o rank8` 能跑通但显存余量不可依赖；`activation_offload=True` 未解决这个问题；
  - 不建议马上尝试 `q/k/v/o rank16/32` 或恢复 `ENTROPY_COEFF=0.01`；
  - 若要提高质量参数，优先找能释放 actor update 阶段常驻显存的低风险开关，再逐项恢复 rank/entropy；否则先用 `q/k/v/o rank8` 或回到更稳的 `o_proj rank8` 做短 smoke/小步长长跑验证。

### 14.3 关于 `PPO_MAX_TOKEN_LEN_PER_GPU=3072`

结论：当前建议保留 `3072`，不要继续降。

- 该参数只控制 actor/ref 动态 micro-batch 的 token 上限，不改变 prompt/response 长度，不裁剪样本，不改变 rollout 分布；
- 当前 `MAX_PROMPT_LENGTH=2048`，`MAX_RESPONSE_LENGTH=768`，理论最大上下文为 2816，`3072` 有约 256 token 余量；
- 降到 `2048` 可能低于单条样本实际序列长度，触发 `max_token_len must be greater than the sequence length`，或迫使同时降低 prompt/response 长度，从而改变训练分布；
- 如果恢复更真实配置后再次 OOM，不建议继续压 `PPO_MAX_TOKEN_LEN_PER_GPU`；`activation_offload=True` 已证明不能稳定释放当前瓶颈余量，应优先寻找能降低 actor update 阶段常驻显存的配置。

### 14.4 当前 smoke TODO

1. 资源检查：
   - 确认 GPU 0-7 无无关 compute app；
   - 确认没有 Ray/vLLM/`train_grpo_lora_actor_verl.sh` 残留；
   - 若 GPU 被 demo 或其他用户进程占用，先暂停 smoke，不主动误杀。

2. 复跑当前稳定 smoke 基线：
   - 使用 14.2 的 override；
   - 目标是确认当前环境下仍能稳定完成 2 step；
   - 日志写到 `/tmp/grpo_smoke_codex_YYYYMMDD_HHMMSS_tok3072_rerun.log`。

3. 若 smoke 失败，按错误类型处理：
   - `mm_token_type_ids`：回查 `chat_template.py` 和 `transformer_impl.py` 的补零/裁剪路径；
   - actor backward OOM：先检查是否有外部 GPU 占用；`enable_activation_offload=True` 已验证不能稳定增加余量，仅作兜底排查项；
   - old/ref logprob OOM：确认 `ENTROPY_COEFF=0`、`LOGPROB_MICRO_BATCH_SIZE_PER_GPU=1`、SDPA 和 `USE_REMOVE_PADDING=False` 是否生效；
   - vLLM sleep/resume 或 LoRA hot-swap 错误：优先检查 `Before collect weights / Before resume weights / vLLM load weights` marker。

4. 若 smoke 通过：
   - 记录日志路径、完成 step、关键指标；
   - `q/k/v/o rank8` 已确认可跑通，但显存余量极薄；`activation_offload=True` 未提供稳定余量且使 actor update 变慢；
   - 下一阶段不要直接升 `LORA_RANK=16/32` 或恢复 `ENTROPY_COEFF=0.01`，除非先找到能降低 actor update 常驻显存的配置；
   - 每次只恢复一类质量相关参数，避免一次改太多导致定位困难。

### 14.5 validation rollout 可观测性增强

- 2026-04-29 已加轻量 JSONL 采样日志，不改变训练/rollout/reward 主链路：
  - `/home/ps/hanjiaqi/verl/verl/trainer/ppo/ray_trainer.py`
  - `/home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/framework_bridge/verl/train_grpo_lora_actor_verl.sh`
- 同时修正 reward 组合路径：
  - `/home/ps/hanjiaqi/verl/verl/experimental/reward_loop/reward_loop.py`
  - 之前 `reward.custom_reward_function.path` 存在时会直接返回自定义 reward，导致 Qwen RM 虽被启动但不进入最终 reward；
  - 现在当 `reward.reward_model.enable=True` 且自定义 reward 存在时，最终 reward = `rm_score + custom_reward`；
  - `reward_extra` 会写入 `rm_score`、`custom_reward`、`score`、`acc`，validation samples JSONL 可直接看到 RM 分与反拒绝惩罚。
- 默认 logger 从 `["console"]` 扩展为 `["console","tensorboard","file"]`：
  - TensorBoard：`${LOG_ROOT}/tensorboard/${PROJECT_NAME}/${EXPERIMENT_NAME}`
  - verl metrics JSONL：`${LOG_ROOT}/verl_metrics/${PROJECT_NAME}/${EXPERIMENT_NAME}.jsonl`
  - validation samples JSONL：`${LOG_ROOT}/validation_samples/${EXPERIMENT_NAME}.jsonl`
- 默认每次 validation 采样 `VAL_GENERATIONS_TO_LOG=16` 条，字段包括：
  - `global_step`
  - `sample_index`
  - `uid`
  - `data_source`
  - `prompt`
  - `response`
  - `reward`
  - `rm_score`（若 reward_extra 中存在 RM 分数字段，否则为 null）
  - `ground_truth`
  - `prompt_tokens`
  - `response_tokens`
  - `prompt_chars`
  - `response_chars`
  - `aborted`
  - `reward_extra`
- 可用环境变量覆盖：
  - `PROJECT_NAME`
  - `EXPERIMENT_NAME`
  - `LOGGER_BACKENDS`
  - `VAL_GENERATIONS_TO_LOG`
  - `LOG_ROOT`
  - `TENSORBOARD_DIR`
  - `VERL_FILE_LOGGER_ROOT`
  - `VALIDATION_SAMPLE_JSONL`
- 同时将 `train_grpo_lora_actor_verl.sh` 的默认训练参数收敛到已验证的 q/k/v/o rank8 pilot：
  - `TRAIN_BATCH_SIZE=32`
  - `PPO_MICRO_BATCH_SIZE_PER_GPU=1`
  - `LOGPROB_MICRO_BATCH_SIZE_PER_GPU=1`
  - `PPO_MAX_TOKEN_LEN_PER_GPU=3072`
  - `LORA_RANK=8`
  - `LORA_ALPHA=8`
  - `ENTROPY_COEFF=0`
  - `SAVE_FREQ=50`
  - `TEST_FREQ=50`
  - `MAX_ACTOR_CKPT_TO_KEEP=2`
- 校验：
  - `python -m py_compile /home/ps/hanjiaqi/verl/verl/trainer/ppo/ray_trainer.py` 通过；
  - `python -m py_compile /home/ps/hanjiaqi/verl/verl/experimental/reward_loop/reward_loop.py` 通过；
  - `bash -n /home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/framework_bridge/verl/train_grpo_lora_actor_verl.sh` 通过。

### 14.6 2026-04-29 smoke 复跑记录

- GPU 排布策略：
  - 使用 `CUDA_VISIBLE_DEVICES=4,5,6,7,0,1,2,3`；
  - actor/ref/rollout 先占物理 GPU 4-7；
  - Qwen RM resource pool 后占物理 GPU 0-3；
  - 这样避开物理 GPU0 上已有的 `python -u main.py` 约 2GB 常驻进程。
- 第一次复跑进入 reward 阶段后失败：
  - 日志：`/tmp/grpo_smoke_codex_20260429_153147_qkvo_r8_rmjson_gpu4567.log`
  - 报错：RM 请求 `/classify` 返回 404；
  - 根因：`verl/workers/rollout/vllm_rollout/vllm_async_server.py` 调用 vLLM `build_app(args, supported_tasks)` 时没有传入 `model_config`，当前 vLLM 的 pooling API 在 `model_config is None` 时不会注册 `/classify`；
  - 修复：若 `build_app` 支持 `model_config` 参数，则传入 `vllm_config.model_config`；
  - 校验：`python -m py_compile /home/ps/hanjiaqi/verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py` 通过。
- 第二次复跑未再出现 `/classify` 404，但在正式 generate 前卡住：
  - 日志：`/tmp/grpo_smoke_codex_20260429_153942_qkvo_r8_rmjson_gpu4567.log`
  - `AgentLoopManager` 已启动 actor rollout vLLM：`<internal-host>:37735`；
  - `/v1/models` 可正常返回 actor 模型，vLLM server 本身可用；
  - `/metrics` 只看到探测用的 `/v1/models` 请求，没有实际 generation 请求；
  - Ray debug state 显示多个 `AgentLoopWorker.__init__` 仍在初始化，主训练停在 `Training Progress: 0/2`；
  - 因该状态继续等待只会占用 GPU，已手动中断并确认 GPU 释放，仅保留物理 GPU0 原有 2GB 进程。
- 下一次 smoke 建议使用更小的链路验证配置：
  - `ROLLOUT_N=1`
  - `TRAIN_BATCH_SIZE=4`
  - `PPO_MINI_BATCH_SIZE=4`
  - `VAL_BATCH_SIZE=4`
  - `MAX_RESPONSE_LENGTH=128`
  - `ROLLOUT_MAX_NUM_SEQS=4`
  - `RM_MAX_NUM_SEQS=4`
  - `actor_rollout_ref.rollout.agent.num_workers=1`
  - 仍保留 `q/k/v/o rank8`、Qwen RM、custom reward、file/tensorboard/validation JSONL，用于验证完整链路而不是吞吐。
- 备注：
  - 继续启动上述 tiny smoke 时，平台提权审批/额度拦截了命令；
  - 该 run 未实际开始，不代表 tiny smoke 通过或失败；
  - 截止中断后检查，GPU 已释放，除物理 GPU0 原有 `171475 python -u main.py` 约 2GB 外无训练残留。

---

## 15) 编号说明（避免后续混乱）

- 同一时间段，**Claude** 和 **Codex** 各自在文件里写了 §13 和 §14：
  - 第一份 §11-§14（lines 172-583）：**Claude 主线**，最终在 §14 的 2026-04-29 17:28 跑通 2-step smoke；
  - 第二份 §13-§14（lines 586-1029）：**Codex 主线**，2026-04-24 → 2026-04-27 跑通 codex 自己的 2-step smoke，2026-04-29 进一步加 `q/k/v/o rank8` + 新 reward 逻辑后 smoke 卡死；
- 两条主线共同促成了最终能稳定跑的配置。本 §15 之后，新进展统一往后续 §16、§17... 累加，**不再重复用 13/14**。

---

## 16) 2026-04-29 17:28 Claude smoke 通过 → Codex 2026-04-29 卡死的统一解释

### 16.1 Codex §14.6 的 hang 根因 = Claude §14 修过的同一个 bug

Codex 第二次复跑（§14.6 line 1009-1015）的现象：
- `/v1/models` 可返回 actor 模型，vLLM server 本身健康；
- `/metrics` 只看到探测用 `/v1/models` 请求；
- `Training Progress: 0/2` 永久卡住；
- 多个 `AgentLoopWorker.__init__` 还在 init。

Claude §14 跑前发现的同一 hang（v3 run，log `/tmp/grpo_smoke_20260429_162107.log`）：
- 现象完全一致：actor vLLM `/v1/models` 通、`/health` 200，但 `/classify` 永不返回；
- 根因：[reward_model.py:47-48](hanjiaqi/verl/verl/experimental/reward_loop/reward_model.py#L47-L48) 在 RewardModelManager init 末尾若 `rollout.free_cache_engine=True` 立刻 `self.sleep()`，把 RM 物理页 unmap；
- async agent reward loop（`enable_resource_pool=True`）的代码路径是 `RewardLoopWorker.compute_score → compute_score_disrm → POST /classify`，**不调用 wake_up**；
- aiohttp `total=None` 永不超时，请求卡在 sleeping RM 上整个进程死锁。

**Claude 修法**：smoke 命令里加 `reward.reward_model.rollout.free_cache_engine=False` 让 RM 启动后保持唤醒。RM 在物理 GPU 0-3，actor 在 4-7，本来就不抢显存，RM 一直占 ~19GB 没问题。

### 16.2 最终 = Codex 修复链 + Claude 修复链 共同生效

把两条主线的所有补丁汇总（按文件分组）：

| 文件 | 修改 | 来源 | 作用 |
|------|------|------|------|
| `verl/workers/engine_workers.py` | `update_weights` reorder：`get_per_tensor_param` 移到 `resume(weights)` 之前 + `Before/After collect/resume weights` 显存日志 | Codex §13.2#2 + Claude §13 | 避免 FSDP+vLLM 同 GPU OOM |
| `verl/workers/engine_workers.py` | "任意 sleep_level 都 resume(weights)" | Claude §11.3 | 修 cumem unmap LoRA buffer |
| `verl/workers/rollout/vllm_rollout/vllm_async_server.py` | `_sleeping_tags` 状态 + 拦截重复 wake_up；level=1 时记录 `{weights, kv_cache}` | Codex §13.2#1 | 防 wake_up 双重 map |
| `verl/workers/rollout/vllm_rollout/vllm_async_server.py` | `build_app(args, supported_tasks)` 传入 `model_config` 才能注册 `/classify` | Codex §14.6 | 修 RM 404 |
| `verl/workers/rollout/vllm_rollout/vllm_rollout.py` | （Claude 加的 `_has_slept` 已撤回，无修改） | — | — |
| `verl/utils/vllm/utils.py` | `_patch_gemma4_mm_lora_support` 注入 LoRA 协议属性 | Codex 早期 §4 | Gemma4 MM 支持 LoRA |
| `verl/utils/vllm/utils.py` | set_lora guard 加 `cuda.synchronize` + CPU detach + 详细日志 | Claude §11 + Codex §8 | 安全网，捕捉残余异常 |
| `verl/utils/vllm/patch.py` | Gemma4 MoE `inner_model.model.layers` 路径处理 | Codex §4 | 修 weight loader |
| `verl/utils/chat_template.py` | 裁剪 dummy user prefix 时同步裁剪 `mm_token_type_ids` | Codex §14.1 | 修 mm_token_type_ids 字段 |
| `verl/workers/engine/fsdp/transformer_impl.py` | Gemma4 训练态缺 `mm_token_type_ids` 时补 `zeros_like(input_ids)` | Codex §14.1 | 修 actor update Gemma4 错误 |
| `verl/trainer/ppo/ray_trainer.py` | `_compute_old_log_prob()` 改为按配置控制 entropy；validation samples JSONL 采样写入 | Codex §13.2#5 + §14.5 | OOM + 可观测性 |
| `verl/experimental/reward_loop/reward_loop.py` | `rm_score + custom_reward` 组合，`reward_extra` 写入 `rm_score/custom_reward/score/acc` | Codex §14.5 | 修 reward 路径 |
| `LlamaFactory/chatjoy/GRPO_Training_Package/anti_refusal_reward.py` | `compute_reward(...)` 加 `**kwargs` | Codex §13.2#3 | 兼容 `reward_router_address` 等额外字段 |
| `LlamaFactory/chatjoy/GRPO_Training_Package/framework_bridge/verl/train_grpo_lora_actor_verl.sh` | `HF_ATTN_IMPLEMENTATION=sdpa` + `USE_REMOVE_PADDING=False` 自动联动；`logger=["console","tensorboard","file"]`；validation JSONL；q/k/v/o rank8 默认 | Codex §13.2#4 + §14.5 | Gemma4 head=512 不能用 flash-attn；可观测性 |

### 16.3 最终 smoke 启动命令（Claude § 14 验证可跑通）

```bash
cd /home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/framework_bridge/verl
LOG="/tmp/grpo_smoke_$(date +%Y%m%d_%H%M%S).log"
tmux new -d -s grpo_smoke "conda run -n gemma4-rl --no-capture-output bash -c '
  cd /home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/framework_bridge/verl
  export CUDA_VISIBLE_DEVICES=4,5,6,7,0,1,2,3
  export CUDA_LAUNCH_BLOCKING=1
  export VERL_LOGGING_LEVEL=INFO
  TRAIN_BATCH_SIZE=32 \
  TARGET_MODULES=\"[o_proj]\" LORA_RANK=8 LORA_ALPHA=8 \
  MAX_MODEL_LEN=3072 \
  bash train_grpo_lora_actor_verl.sh \
    trainer.total_training_steps=2 \
    trainer.save_freq=100000 \
    trainer.test_freq=100000 \
    trainer.resume_mode=disable \
    reward.reward_model.rollout.free_cache_engine=False 2>&1 | tee \$LOG
'"
```

> 想用 codex §14.5 默认的 q/k/v/o rank8 + Qwen RM + custom reward + tensorboard/validation JSONL：把 `TARGET_MODULES="[q_proj,k_proj,v_proj,o_proj]"` 替换上面的 `[o_proj]`，其它不变。

### 16.4 仍需要在脚本里固化的两件事（不要每次靠 CLI 覆盖）

1. **`reward.reward_model.rollout.free_cache_engine=False` 必须长期生效**：
   - 现在脚本默认 `reward.reward_model.rollout` 没显式设这个值，会随 actor rollout 默认 True；
   - **建议改 `train_grpo_lora_actor_verl.sh`**：在 ARGS 里加 `reward.reward_model.rollout.free_cache_engine=False`；
   - 或在 RewardModelManager 修：仅当 RM 与 actor 共享物理 GPU 时才 sleep（异步路径下不要 sleep）；
2. **`MAX_MODEL_LEN` 默认值** 保持比 `MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH` 多至少 16-256 token，给 RM chat template 留 buffer。

### 16.5 Phase 1 pilot（接下来的事）

1. **smoke 已稳定通过**（Claude §14 + Codex §14.2.1）；
2. 下一步建议（按 codex §14.4-§14.5 收敛的安全配置开始）：
   - `TARGET_MODULES=[q_proj,k_proj,v_proj,o_proj]`、`LORA_RANK=8`、`LORA_ALPHA=8`（codex §14.2.2 验证 hot-swap OK，但显存余量薄）；
   - **不要**直接升 `LORA_RANK=16/32` 或恢复 `ENTROPY_COEFF=0.01`，除非先找到能降 actor update 常驻显存的配置；
   - 跑 50 step pilot：`trainer.total_training_steps=50 trainer.save_freq=25 trainer.test_freq=10`；
   - **最关键**：从 smoke 阶段就**移除** `CUDA_LAUNCH_BLOCKING=1`，速度能提升 5-10×（smoke 一步 379s → 预计 50-80s）；
3. **观察 reward 曲线**：critic/score/mean 应从 -1.0 区间往 0 靠；
4. **若 50 step 通过 → Phase 2**：1 epoch ~305 step，预计 6-10 小时（去掉 CUDA_LAUNCH_BLOCKING 后）。

### 16.6 Codex / Claude 公共结论

- **5 个独立 bug 串联在一起**，任何一个不修 smoke 都过不了：
  1. cumem level=1 unmap LoRA buffer（Claude）
  2. wake_up + FSDP 同时占 GPU OOM（Claude reorder）
  3. GPU 0 squatter 占 1.9GB（Claude `CUDA_VISIBLE_DEVICES`）
  4. RM chat template +1 token 超 max_model_len（Claude `MAX_MODEL_LEN=3072`）
  5. RM 自动 sleep async 路径不 wake（Claude `free_cache_engine=False`）
- **此外 Codex 修了**：Gemma4 head=512 + flash-attn 兼容（SDPA）、`mm_token_type_ids`、entropy OOM、`/classify` 404、reward 组合路径、可观测性。
- **smoke 状态**：Claude §14 通过、Codex §14.2.1 + §14.2.2 + §14.2.3 都通过；最近一次 codex §14.6 在 GPU swap 之后 hang，根因已被 §16.1 的 `free_cache_engine=False` 解决。
- 下一轮接手者只需照 §16.3 启动命令复跑，应该一次过。

---

## 17) 2026-04-30 脚本固化（4 处改动 → smoke/pilot 命令更短，可观测性 + 加速）

### 17.1 已固化到脚本的 4 处改动

`/home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/framework_bridge/verl/train_grpo_lora_actor_verl.sh`

| # | 改动 | 默认值 | 环境变量覆盖 | 解决的问题 |
|---|------|--------|--------------|------------|
| 1 | `MAX_MODEL_LEN` 默认加 256 buffer | `prompt + response + 256 = 3072` | `MAX_MODEL_LEN` | RM chat template 加 1 token 超 max_model_len（§14.6）|
| 2 | `reward.reward_model.rollout.free_cache_engine=False` 写入 ARGS | 固定 False | 无 | RM init 后自动 sleep + async loop 不 wake → /classify 永久 hang（§16.1）|
| 3 | `ROLLOUT_ENFORCE_EAGER` 改为环境变量 | `True`（保守，省 1-2GB/卡） | `ROLLOUT_ENFORCE_EAGER=False` 启用 CUDA graph | 长期想加速可一键开 graph，无需改源码 |
| 4 | `actor.checkpoint.async_save` 写入 ARGS | `True`（默认异步保存） | `ACTOR_CKPT_ASYNC_SAVE=False` | save_checkpoint ~50s 推后台，不卡主循环 |

### 17.2 现在 smoke / pilot 启动命令变短

**smoke**（2 step，验证链路）：

```bash
cd /home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/framework_bridge/verl
LOG="/tmp/grpo_smoke_$(date +%Y%m%d_%H%M%S).log"
tmux new -d -s grpo_smoke "conda run -n gemma4-rl --no-capture-output bash -c '
  cd /home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/framework_bridge/verl
  export CUDA_VISIBLE_DEVICES=4,5,6,7,0,1,2,3
  export VERL_LOGGING_LEVEL=INFO
  TRAIN_BATCH_SIZE=32 TARGET_MODULES=\"[o_proj]\" LORA_RANK=8 LORA_ALPHA=8 \
  bash train_grpo_lora_actor_verl.sh \
    trainer.total_training_steps=2 \
    trainer.save_freq=100000 \
    trainer.test_freq=100000 \
    trainer.resume_mode=disable 2>&1 | tee \$LOG
'"
```

**关键变化**：相比 §16.3，CLI 不再需要 `MAX_MODEL_LEN=3072`、`reward.reward_model.rollout.free_cache_engine=False`、`CUDA_LAUNCH_BLOCKING=1`。脚本默认值就是对的。

**Phase 1 pilot**（50 step + q/k/v/o + 异步保存默认开）：

```bash
cd /home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/framework_bridge/verl
LOG="/tmp/grpo_pilot_$(date +%Y%m%d_%H%M%S).log"
tmux new -d -s grpo_pilot "conda run -n gemma4-rl --no-capture-output bash -c '
  cd /home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/framework_bridge/verl
  export CUDA_VISIBLE_DEVICES=4,5,6,7,0,1,2,3
  export VERL_LOGGING_LEVEL=INFO
  TRAIN_BATCH_SIZE=32 \
  TARGET_MODULES=\"[q_proj,k_proj,v_proj,o_proj]\" \
  LORA_RANK=8 LORA_ALPHA=8 ENTROPY_COEFF=0 \
  bash train_grpo_lora_actor_verl.sh \
    trainer.total_training_steps=50 \
    trainer.save_freq=25 \
    trainer.test_freq=10 \
    trainer.resume_mode=disable 2>&1 | tee \$LOG
'"
```

显存稳定后想再加速一档（启用 vLLM CUDA graph）：

```bash
ROLLOUT_ENFORCE_EAGER=False ... bash train_grpo_lora_actor_verl.sh ...
```

### 17.3 加速预期（去掉 CUDA_LAUNCH_BLOCKING + 异步 save）

Smoke v4 step 时间分解（参考 §14.1）：

| 阶段 | smoke v4（含 CUDA_LAUNCH_BLOCKING） | 移除 CLB 后预期 |
|------|------------------------------------|------------------|
| gen | 168s | ~85-110s |
| update_actor | 129s | ~30-50s |
| old_log_prob + ref | 77s | ~20-40s |
| update_weights | 4-5s | ~4-5s（不变）|
| save_checkpoint | 51s（save 步） | **0s**（异步推后台） |
| **step total** | **379-409s** | **~80-130s/step** |

50 step pilot 预计：**~70-110 分钟**（之前 5+ 小时 if CLB on）。

继续 `ROLLOUT_ENFORCE_EAGER=False`：再省 30-40s/step → **~50-90s/step**。

### 17.4 仍未做的事（按优先级）

1. **wandb 默认开** — `LOGGER_BACKENDS='["console","tensorboard","wandb","file"]'`（如需远程 dashboard）；
2. **reward 直方图** — 在 `_maybe_log_val_generations` 加 `add_histogram("val/reward", scores, step)`，给 TensorBoard 显示分布；
3. **MoE expert routing 统计** — 想看 Gemma4 128 experts 是否坍塌，要 hook vLLM 的 expert dispatcher，记录 expert hit count；
4. **`kv_cache_dtype=fp8`** — 进一步省 KV 显存腾出空间给 CUDA graph；需先验 vLLM 0.19 + Gemma4 兼容性。

3、4 是 Phase 2 才考虑的优化，Phase 1 不动。

### 17.5 校验

- `bash -n train_grpo_lora_actor_verl.sh` 通过
- 4 个新设置的 grep：`MAX_MODEL_LEN ... +256 / free_cache_engine=False / ROLLOUT_ENFORCE_EAGER / async_save` 都在
- 不破坏前面 §16 的 reorder + GPU swap + RM 不睡修复链

## 18. 2026-04-30 50-step pilot：RM `/classify` 3072 token 越界修复

### 18.1 现象

50-step q/k/v/o rank8 pilot 跑到中段后失败：

- 外层错误：`aiohttp.client_exceptions.ClientResponseError: 500 ... /classify`
- vLLM RM 端真实错误：
  `VLLMValidationError: maximum context length is 3072 tokens ... prompt contains at least 3073 input tokens`

这不是 OOM，也不是 RM sleep/free_cache 问题。根因是 Gemma actor 的 prompt/response 长度限制按 actor tokenizer 控制，但 RM 使用 Qwen tokenizer 和 RM chat template；少数样本在 RM tokenizer 下会超过 `reward.reward_model.rollout.max_model_len=3072`。

### 18.2 修复

已做两层保护：

1. 在 `train_grpo_lora_actor_verl.sh` 中将 RM context 与 actor rollout 解耦：
   - actor/rollout 默认仍是 `MAX_MODEL_LEN=MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH+256=3072`；
   - 新增 `RM_MAX_MODEL_LEN=MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH+512=3328`；
   - `reward.reward_model.rollout.max_model_len` 和 `RM_MAX_NUM_BATCHED_TOKENS` 改用 `RM_MAX_MODEL_LEN`。
2. 在 `/home/ps/hanjiaqi/verl/verl/experimental/reward_loop/reward_loop.py` 中，`/classify` 前用 RM tokenizer 做兜底检查；如果单条 RM prompt 仍超过 `max_model_len`，保留尾部并截断到 `max_model_len-16` token，避免 vLLM 500 终止训练。

### 18.3 校验

- `bash -n /home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/framework_bridge/verl/train_grpo_lora_actor_verl.sh` 通过
- `python -m py_compile /home/ps/hanjiaqi/verl/verl/experimental/reward_loop/reward_loop.py` 通过
- 失败 run 的 monitor tmux 已停止；GPU 已释放回普通空闲状态

## 19. 2026-04-30 50-step pilot：q/k/v/o rank8 actor update OOM

### 19.1 现象

`chatjoy_grpo_qkvo_r8_50step_plot_20260430_122332` 在 step 6 之后失败：

- 外层错误：`ray.exceptions.RayTaskError(OutOfMemoryError)`
- 失败位置：`WorkerDict.actor_rollout_update_actor()` → `engine_workers.py:update_actor()` → `fsdp/transformer_impl.py:loss.backward()`
- CUDA OOM 数字：actor 反向传播尝试额外申请约 `3.0-3.1 GiB`，当时卡上只剩约 `1.8-2.1 GiB`
- 同卡还有睡眠态 rollout vLLM worker，约 `4.77 GiB`

这次不是 RM `/classify` 越界，也不是 checkpoint 写盘问题；RM `max_model_len=3328` 修复已经生效，训练能正常跑到多步后才在 actor backward 峰值处 OOM。

### 19.2 判断

当前 32GB 卡上的 q/k/v/o rank8 配置显存结构大致是：

- actor FSDP backward 进程约 `24.4-24.7 GiB`
- 睡眠态 vLLM rollout 残留约 `4.77 GiB`
- 剩余显存不足以覆盖 backward 临时峰值

`PYTORCH_ALLOC_CONF=expandable_segments:True` 只能缓解碎片；日志里 reserved-but-unallocated 只有约 `140-180 MiB`，不是主因。单纯降低 `PPO_MAX_TOKEN_LEN_PER_GPU` 也未必完全治本，因为单条样本最长接近 `2048+768`，动态 batching 不能把一条长样本切开。

### 19.3 修复

在 `train_grpo_lora_actor_verl.sh` 增加 actor activation offload 默认开关：

- 新增：`ACTOR_ENABLE_ACTIVATION_OFFLOAD="${ACTOR_ENABLE_ACTIVATION_OFFLOAD:-True}"`
- 传入：`actor_rollout_ref.model.enable_activation_offload="${ACTOR_ENABLE_ACTIVATION_OFFLOAD}"`

理由：verl FSDP 路径已有 `enable_activation_offload` 实现，专门减少 backward activation 峰值；它不改变 prompt/response、reward、loss 或 GRPO group，只用额外 CPU 传输换显存余量。代价是 `update_actor` 会更慢。

暂不默认降低 `PPO_MAX_TOKEN_LEN_PER_GPU=3072`。如果开启 activation offload 后仍 OOM，再优先试 `PPO_MAX_TOKEN_LEN_PER_GPU=2048` 或缩短 `MAX_RESPONSE_LENGTH=512`，但这两项分别带来更多 microbatch 时间或改变 rollout 长度预算。

### 19.4 校验

- `bash -n /home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/framework_bridge/verl/train_grpo_lora_actor_verl.sh` 通过
- 失败 run 的 monitor tmux 已停止

## 20. 2026-04-30 memsafe smoke 前的零质量影响显存保护

### 20.1 原则

本轮只改不影响训练语义的设置；不改以下会影响训练信号或 rollout 行为的参数：

- 不降 `MAX_RESPONSE_LENGTH=768`
- 不降 `ROLLOUT_N=4`
- 不降 `TRAIN_BATCH_SIZE=32`
- 不改 q/k/v/o rank8 target modules
- 不改 RM、custom reward、KL/loss 配置

### 20.2 修改

在 `train_grpo_lora_actor_verl.sh` 中新增或确认：

- `actor_rollout_ref.model.enable_activation_offload=True`：减少 actor backward activation 峰值；不改 loss/reward/sample。
- `CUDA_MODULE_LOADING=LAZY`：减少启动后一次性 CUDA module 占用；不改模型数学。
- `VAL_BATCH_SIZE=32`：只改变 validation 分批，不改变 validation set 或训练信号；避免正式训练在验证阶段 OOM。

不要在全局环境里设置 `PYTORCH_ALLOC_CONF=expandable_segments:True` 或 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`：vLLM V1 的 CuMem memory pool 会 assert 这两个环境变量中不能包含 `expandable_segments:True`，否则 reward/rollout vLLM server 启动阶段直接失败。

暂不默认降低 `ROLLOUT_GPU_MEMORY_UTILIZATION=0.78`：历史记录中 0.60/0.50 曾分别触发 FSDP reload OOM / vLLM KV block 不足；当前主要 OOM 点是 actor backward，优先用 activation offload 解决。

### 20.3 校验

- `bash -n /home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/framework_bridge/verl/train_grpo_lora_actor_verl.sh` 通过

## 21. 2026-04-30 memsafe smoke 首次启动失败：vLLM CuMem 与 expandable_segments 冲突

### 21.1 现象

`chatjoy_grpo_qkvo_r8_memsafe_smoke_20260430_131403` 在 vLLM RM server 初始化时失败，尚未进入训练 step：

- 失败位置：`vLLMHttpServer.launch_server()` → `AsyncLLM.from_vllm_config()` → vLLM WorkerProc `load_model()`
- 根因报错：
  `AssertionError: Expandable segments are not compatible with memory pool`

### 21.2 根因

脚本全局导出了 `PYTORCH*_ALLOC_CONF=expandable_segments:True`。actor 侧可能受益，但同一个训练进程树会启动 vLLM reward/rollout server；vLLM V1 使用 CuMem memory pool，并显式禁止 `expandable_segments:True`。

这不是 GPU OOM，也不是 RM 模型或 `/classify` 问题。

### 21.3 修复

已从 `train_grpo_lora_actor_verl.sh` 撤回全局 allocator 环境变量，只保留：

- `actor_rollout_ref.model.enable_activation_offload=True`
- `CUDA_MODULE_LOADING=LAZY`
- `VAL_BATCH_SIZE=32`

并加了 guard：如果外层环境带入 `PYTORCH_ALLOC_CONF` 或 `PYTORCH_CUDA_ALLOC_CONF` 且包含 `expandable_segments:True`，脚本会打印 warning 并 unset，防止 vLLM CuMem 启动失败。

### 21.4 校验

- `bash -n /home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/framework_bridge/verl/train_grpo_lora_actor_verl.sh` 通过

## 22. 2026-04-30 memsafe smoke 重新启动状态

### 22.1 当前 run

已用新实验名重新启动，避免和失败日志混在一起：

- experiment：`chatjoy_grpo_qkvo_r8_memsafe_smoke_20260430_132658`
- train tmux：`grpo_memsafe_132658`
- monitor tmux：`grpo_memsafe_mon_132658`
- train log：`/tmp/chatjoy_grpo_qkvo_r8_memsafe_smoke_20260430_132658.log`
- monitor log：`/tmp/chatjoy_grpo_qkvo_r8_memsafe_smoke_20260430_132658_monitor.log`
- total steps：`8`
- save：`trainer.save_freq=0`
- validation：`trainer.test_freq=8`

### 22.2 当前观察

截至 2026-04-30 13:29 左右：

- 新 run 仍在运行；
- 旧的 `expandable_segments` / vLLM CuMem assert 没有再次出现；
- 已通过 config validation、数据过滤和 dataloader 构建；
- 日志已进入 actor WorkerDict / FSDP 初始化和权重加载阶段；
- 还没有进入 `Training Progress` step，因此 actor backward OOM 修复还未被验证。

### 22.3 仍需继续观察

重点看三个节点：

1. RM vLLM server 是否完整启动，不再出现 `WorkerProc failed to start` / `Engine core initialization failed`；
2. 第 1 个 actor update 是否通过，并且 monitor 中 actor 卡保留至少 1-2GB 空余；
3. 第 8 step validation 是否通过，确认 `VAL_BATCH_SIZE=32` 没有 validation 阶段 OOM。

---

## 23. 2026-04-30 memsafe smoke step 7 OOM：降 PPO_MAX_TOKEN_LEN_PER_GPU=2816

### 23.1 现象

`chatjoy_grpo_qkvo_r8_memsafe_smoke_20260430_132658` 跑过 6 步，在 step 7 actor backward 失败：

- 日志：`/tmp/chatjoy_grpo_qkvo_r8_memsafe_smoke_20260430_132658.log`（1630 行）
- 失败位置：`engine_workers.update_actor → engine.train_batch → forward_backward_batch → loss.backward()`
- 错误：`torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 3.04 GiB. GPU 0 has a total capacity of 31.36 GiB of which 2.55 GiB is free`
- 进程显存：FSDP actor `23.96 GiB` + sleeping vLLM rollout `4.77 GiB`（cumem level=1 残留）= **28.73 GiB / 31.36 GiB**

PyTorch reserved-but-unallocated 只有 194 MiB，**不是 fragmentation**，是真的没空间给 backward 的 ~3GB 峰值。

### 23.2 关键观察（前 6 步指标）

| step | prompt_max | response_max | global_seq_max | step_time | update_actor |
|------|-----------|--------------|----------------|-----------|---------------|
| 1 | 2017 | 601 | 57919 | 319s | 114s |
| 3 | **2011** | **768** | **65964** | 348s | 113s |
| 5 | 1690 | 768 | 53759 | 290s | 97s |
| 6 | 1749 | 731 | 52778 | 302s | 106s |
| 7 | (失败) | (失败) | — | 351s 后 OOM | OOM |

实测最长序列 = `prompt_max(2011) + response_max(768) = 2779 tokens`，**严格小于 2816**。`PPO_MAX_TOKEN_LEN_PER_GPU=3072` 给了 ~300 tokens 富余，dynamic batcher 可能在长样本上额外打包 padding，把 SDPA 的 O(N²) 注意力矩阵推到峰值。

### 23.3 修复（已落地）

**`PPO_MAX_TOKEN_LEN_PER_GPU` 默认从 3072 降到 2816**（= `MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH`）：

文件：`/home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/framework_bridge/verl/train_grpo_lora_actor_verl.sh`（L113）

```bash
# 旧
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-3072}"
# 新
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"
```

实际默认值：`2048 + 768 = 2816`。

**为什么这样修**：
- `use_dynamic_bsz=True` 按 token 打包 micro-batch，3072 上限会让单个 micro-batch 装到 3072 token；
- SDPA attention activation O(N²)：3072² → 2816² 减 ~16% 显存；
- 实测最长 2779 < 2816，**所有样本仍能装下，不影响训练分布**；
- 不改 prompt/response 长度，不改 reward 信号，不改 GRPO group。

**预期收益**：actor backward 峰值省 ~1.5-2 GiB（注意力矩阵 + 中间 activation 一起压），把 step 7 这种长样本踢出 OOM 区。

### 23.4 不改的事（按 §19/§20 原则）

- 不降 `MAX_RESPONSE_LENGTH=768`（影响 rollout 分布）
- 不降 `ROLLOUT_N=4`（GRPO group 变小，方差变大）
- 不降 `TRAIN_BATCH_SIZE=32`（影响 PPO mini-batch 数）
- 不改 `target_modules=[q,k,v,o]_proj` 或 `LORA_RANK=8`
- 不改 `enforce_eager=True`（CUDA graph 还会再吃 1-2GB 不安全）
- 不改 `ACTOR_ENABLE_ACTIVATION_OFFLOAD=True`（已开，仍不够，但保留）

### 23.5 失败分支（如果 2816 仍 OOM）

按代价从小到大：

1. **`PPO_MINI_BATCH_SIZE=8`**（默认 16）— 把 PPO 更新次数翻倍但每次反向传播显存减半。代价：步时增加 ~30%，可能影响 GRPO 优化稳定性。
2. **`PPO_MICRO_BATCH_SIZE_PER_GPU=1` + `PPO_MAX_TOKEN_LEN_PER_GPU=2304`** — 进一步缩 token packing 上限，前提是确认所有样本都能 ≤2304。会导致最长样本无法装下报错。
3. **干掉 vLLM sleep 4.77GB 残留**（最理想但最难）— 改 sleep_level=2 完全释放物理页；担心 §13.1 wake_up OOM 复发，需先小范围验证。
4. **`MAX_RESPONSE_LENGTH=640`**（最后兜底）— 改 rollout 长度预算，影响训练分布。

### 23.6 校验

- `bash -n train_grpo_lora_actor_verl.sh` 通过
- 变量顺序修正：`PPO_MAX_TOKEN_LEN_PER_GPU` 引用 `MAX_PROMPT_LENGTH/MAX_RESPONSE_LENGTH`，已挪到这两个变量之后定义
- 默认值实测：`2048 + 768 = 2816`

---

## 24. 🎉 2026-04-30 50-step Pilot 完整跑通

### 24.1 总体结果

实验：`chatjoy_grpo_qkvo_r8_pilot_50step_20260430_145346`
- log：`/tmp/chatjoy_grpo_qkvo_r8_pilot_50step_20260430_145346.log`
- 完成度：**100% (50/50 步)**
- 总耗时：**4h41m32s**
- 平均 step time：**338s/step**（含 step 25 / 50 的 checkpoint + final validation）
- 配置：`PPO_MAX_TOKEN_LEN_PER_GPU=2816`、`q/k/v/o rank8`、`ENTROPY_COEFF=0`、`HF_ATTN=sdpa`、`use_remove_padding=False`、`activation_offload=True`、`async_save=True`
- 失败：**0**（无 OOM、无 illegal memory、无 NaN、无 abort）

### 24.2 关键指标 (3-step buckets)

| step | reward | actor/loss | grad_norm | resp_len | step_s |
|------|--------|-----------|-----------|----------|--------|
| 1 | -0.986 | 0.038 | 2.19 | 307 | 359 |
| 5 | -1.007 | 0.042 | **223.29** | 287 | 334 |
| 10 | -1.399 | 0.029 | 1.80 | 236 | 275 |
| 15 | -0.769 | 0.025 | 1.61 | 341 | 331 |
| 20 | -1.020 | 0.039 | 5.56 | 302 | 315 |
| 25 | -0.974 | 0.031 | 5.53 | 281 | 315 |
| 30 | -0.760 | 0.020 | 1.53 | 344 | 328 |
| 35 | -1.034 | 0.110 | 3.17 | 319 | 310 |
| 40 | -0.985 | 0.023 | 1.44 | 306 | 298 |
| 45 | -1.106 | 0.009 | 2.21 | 260 | 270 |
| 50 | -1.064 | 0.019 | 2.44 | 287 | 348 |

**Validation (step 50)**：`val/reward=-0.912`、`val/rm_score=-0.634`、`val/custom_reward=-0.278`

### 24.3 关键发现

1. **OOM 修复确认**：上次 step 7 OOM 在这次 step 7-50 全过，2816 default 起作用。
2. **Reward 几乎不动**：50 步 reward 平均 -0.97（step1-10）→ -0.97（step41-50）。**50 step 仅占 1 epoch (592 step) 的 8.5%**，看不出学习曲线是正常的。但 actor/loss 从 0.038 → 0.019 已经下降 50%，说明在学，只是 reward signal 的方差掩盖了。
3. **梯度异常**：
   - Step 5: `grad_norm=223.29` ⚠️
   - Step 24: 52.83
   - Step 28: 14.23
   - Step 47: 10.14
   - 其他步 1-5 正常
   - **当前没有 grad_clip！正式训练必须加 `clip_grad=1.0`**
4. **吞吐稳定**：70-87 tokens/s/GPU，平均 ~81 tokens/s。
5. **Response 行为**：平均 287 tokens，无 abort，偶有 `clip_ratio=0.008-0.023` 触到 max_response_length=768。
6. **Step time 稳定**：270-365s（除最后一步 348s 含 testing 630s）。

### 24.4 给 Phase 2 正式训练的指导

#### A. 必加（防异常）

1. **梯度裁剪**：
   ```bash
   actor_rollout_ref.actor.optim.clip_grad=1.0
   ```
   step 5 / 24 / 28 的 grad_norm spike 必须压住，否则一次 spike 可能毁掉训练。

2. **学习率衰减**：
   ```bash
   actor_rollout_ref.actor.optim.lr_scheduler=cosine
   actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03
   ```
   3e-6 → 1e-6 余弦下降。

3. **保留所有 §16/§17/§23 的修复**：reorder、GPU swap、free_cache_engine=False、MAX_MODEL_LEN+256、PPO_MAX_TOKEN_LEN_PER_GPU=2816、activation_offload、async_save 不要回退。

#### B. 训练规模建议

数据 18,938 条，`TRAIN_BATCH_SIZE=32` → 1 epoch = **592 步**，按当前 338s/step 计算：
- 1 epoch = **55 小时（~2.3 天）**
- 2 epoch = ~4.6 天
- 3 epoch = ~7 天

建议 Phase 2 用 **`total_training_steps=600`（≈1 epoch）** 起步，看 reward 趋势再决定是否延长。

#### C. 监控点（每 50 步检查一次）

1. **`critic/score/mean` 应该开始上升**（pilot 50 步还没动）。如果 200 步内仍在 -1.0 ± 0.2 区间，说明：
   - LR 太小 → 提到 5e-6
   - 或 LoRA rank 不够（8 → 16），但要先验证 rank 16 不 OOM
   - 或 anti_refusal_penalty 太重 → 调整权重
2. **`actor/grad_norm` 应该稳定 1-5**。如果频繁 >10，clip_grad 要更紧（1.0 → 0.5）。
3. **`actor/kl_loss` 应该 < 2**。如果飙升说明 actor 漂移太远，`kl_loss_coef` 从 0.001 → 0.005。
4. **`response_length/clip_ratio` 应 < 5%**。如果上升，说明 actor 学会输出长 response（reward hacking 信号），考虑加长度惩罚或限制 max_response_length。
5. **`val/rm_score`** 是最干净的信号（不含 anti_refusal_penalty），主要看它是否 > -0.5。

#### D. 时间预算 + 检查点策略

- `save_freq=50` 每 50 步存一次（每张 ckpt ~50GB，磁盘开销可控）
- `max_actor_ckpt_to_keep=2` 保留最近 2 个
- `test_freq=50` 每 50 步 validate 一次（~10 分钟开销）
- 长跑期间另起 monitor 每 ~30 分钟 grep 一次 `step:[0-9]+` + `cumem|illegal|OOM|Traceback`

#### E. 想加速到 ~3-4 天/epoch（可选）

按 §17.4 加速建议依次启用（每开一项观察 1 epoch 看是否稳）：
1. `ROLLOUT_ENFORCE_EAGER=False`（启用 vLLM CUDA graph，gen 提速 30-50%）
2. 升 `PPO_MICRO_BATCH_SIZE_PER_GPU=2`（看 actor backward 显存余量是否够）
3. `wandb` backend 记录在线曲线

### 24.5 不建议立即改动的事

- **不要改 LoRA rank 8 → 16/32**：pilot 显存余量已紧（actor backward 24GB / 31GB），rank 16 可能新 OOM。
- **不要扩 target_modules 到 FFN**：Gemma4 MoE 有 128 experts，扩 FFN 会让 LoRA 参数翻 30-100×。
- **不要 enable entropy_coeff**：之前 §13 试过 entropy 计算 OOM，得先解决 actor backward 余量才能开。

### 24.6 Phase 2 一键启动（建议命令）

```bash
cd /home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/framework_bridge/verl
TS=$(date +%Y%m%d_%H%M%S)
EXP="chatjoy_grpo_qkvo_r8_phase2_1epoch_${TS}"
LOG="/tmp/${EXP}.log"

tmux new -d -s grpo_phase2 "conda run -n gemma4-rl --no-capture-output bash -c '
  cd /home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/framework_bridge/verl
  export CUDA_VISIBLE_DEVICES=4,5,6,7,0,1,2,3
  export VERL_LOGGING_LEVEL=INFO
  TRAIN_BATCH_SIZE=32 \
  TARGET_MODULES=\"[q_proj,k_proj,v_proj,o_proj]\" \
  LORA_RANK=8 LORA_ALPHA=8 ENTROPY_COEFF=0 \
  EXPERIMENT_NAME=$EXP \
  bash train_grpo_lora_actor_verl.sh \
    trainer.total_training_steps=600 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.resume_mode=disable \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03 2>&1 | tee \$LOG
'"
```

预期 1 epoch 用时：**~55 小时**（2.3 天）。如果想更快，先用 §17.4 加速选项独立跑一组 ~50 step 再决定。

### 24.7 文件 / 目录速查

- 训练 log：`/tmp/chatjoy_grpo_qkvo_r8_pilot_50step_20260430_145346.log`
- TensorBoard：`/home/ps/hanjiaqi/LlamaFactory/chatjoy/logs/tensorboard/chatjoy_verl/chatjoy_grpo_qkvo_r8_pilot_50step_20260430_145346/`
- File metrics JSONL：`/home/ps/hanjiaqi/LlamaFactory/chatjoy/logs/verl_metrics/chatjoy_verl/chatjoy_grpo_qkvo_r8_pilot_50step_20260430_145346.jsonl`
- Validation samples：`/home/ps/hanjiaqi/LlamaFactory/chatjoy/logs/validation_samples/chatjoy_grpo_qkvo_r8_pilot_50step_20260430_145346.jsonl`
- Checkpoint：`/home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/verl/chatjoy_grpo_lora/global_step_50/`

---

## 25) 2026-05-07 RL reward 链路修复、anti-refusal 收窄与 2176/768 smoke 结论

### 25.1 背景：为什么 Phase2 1-epoch reward 一直负

正式 run `chatjoy_grpo_qkvo_r8_phase2_1epoch_20260430_201520` 已完整跑完：

- log：`/tmp/chatjoy_grpo_qkvo_r8_phase2_1epoch_20260430_201520.log`
- metrics：`/home/ps/hanjiaqi/LlamaFactory/chatjoy/logs/verl_metrics/chatjoy_verl/chatjoy_grpo_qkvo_r8_phase2_1epoch_20260430_201520.jsonl`
- validation samples：`/home/ps/hanjiaqi/LlamaFactory/chatjoy/logs/validation_samples/chatjoy_grpo_qkvo_r8_phase2_1epoch_20260430_201520.jsonl`
- 结果：543/543 完成，约 49h，0 OOM / 0 CUDA crash。

但最终 validation 仍很差：

| 指标 | Phase2 final |
|---|---:|
| `val-aux/chatjoy/reward/mean@1` | `-0.8994` |
| `val-aux/chatjoy/rm_score/mean@1` | `-0.6275` |
| `val-aux/chatjoy/custom_reward/mean@1` | `-0.2720` |

离线 OpenRLHF RM v5 表现却很好：

- chosen mean `+2.2704`
- rejected mean `-1.2262`
- gap mean `+3.4966`
- acc 约 `0.96`

因此根因不应优先怀疑 RM 训练本身，而是在线 RM scoring path。

### 25.2 根因确认：vLLM classify pooled 到 Qwen chat template 尾部换行

OpenRLHF `RewardDataset` 训练/评估逻辑：

- `chosen = (prompt + chosen).rstrip("\n")`
- 必要时追加 `eos_token`
- 强制最后一个 input id 为 `eos_token_id`
- RewardModel forward 在最后非 pad/EOS 位置打分。

在线 reward loop 原逻辑：

- `apply_chat_template(..., add_generation_prompt=False)`
- Qwen chat template 会生成 assistant 结尾：`<|im_end|>\n`
- vLLM sequence classification pools final token
- 结果实际 pooled 到尾部 newline token，而不是 `<|im_end|>`。

这解释了为什么 actor 输出肉眼上类似 SFT chosen，但 online RM 分数大面积为负。

### 25.3 已落地修复：RM input 与 OpenRLHF EOS 行为对齐

修改文件：

- `/home/ps/hanjiaqi/verl/verl/experimental/reward_loop/reward_loop.py`

在 `_preprocess_reward_inputs` 中加入：

```python
rm_prompt = rm_prompt.rstrip("\n")
eos_token = self.reward_model_tokenizer.eos_token
if eos_token is not None and not rm_prompt.endswith(eos_token):
    rm_prompt = f"{rm_prompt} {eos_token}"
```

目的：

- 去掉 Qwen chat template 的尾部换行；
- 确保 classify pooled token 与 OpenRLHF 训练/离线 eval 的最后 EOS scoring 行为一致。

### 25.4 anti-refusal 第一轮修复：惩罚分层和可观测性

修改文件：

- `/home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/anti_refusal_reward.py`

已落地：

- `REFUSAL_PENALTY = -1.0`
- `FORMAT_SHORT_PENALTY = -0.25`
- `_SHORT_THRESHOLD = 80`
- 返回 dict 中新增：
  - `anti_refusal_penalty`
  - `anti_refusal_reason`
  - `anti_refusal_tier`
  - `anti_refusal_is_refusal`
  - `anti_refusal_is_format_short`

说明：

- keyword refusal 是明确拒绝，扣 `-1.0`；
- 短回复且无 roleplay 格式只是弱信号，扣 `-0.25`；
- 不再让短多语言正常回复被硬惩罚打到底。

### 25.5 anti-refusal 第二轮修复：`no puedo` 误伤

smoke validation samples 发现：

- 西语角色内心独白 `No puedo creer...` 被裸关键词 `no puedo` 命中；
- 示例上下文：`Thoughts: "No puedo creer que tenga que rebajarme a esto..."`；
- 这不是拒绝，而是角色心理描写。

已修复：

- 移除裸 `no puedo`
- 改为明确拒绝短语：
  - `no puedo ayudar`
  - `no puedo continuar`
  - `no puedo seguir`
  - `no puedo participar`
  - `no puedo generar/crear/escribir/proporcionar`

同步收窄同类误伤：

- 德语裸 `ich kann nicht` -> `ich kann nicht helfen/fortfahren/...`
- 法语裸 `je ne peux pas` -> `je ne peux pas aider/continuer/...`
- 日语裸 `できません` / `申し訳` -> 更明确的拒绝句式
- 英文过宽软拒绝如 `maybe we can explore`、`i can help with` -> 更明确的换话题/非 explicit 句式
- 中文移除过宽 `我理解您的`、`我明白您`，收窄 `建议我们`

回归测试通过：

- `No puedo creer...` 不扣分
- `Je ne peux pas croire...` 不扣分
- `Ich kann nicht glauben...` 不扣分
- `信じることができません` 不扣分
- `No puedo ayudar...` 仍扣 `-1.0`
- `Je ne peux pas continuer...` 仍扣 `-1.0`
- 短无格式句仍扣 `-0.25`

用 50-step smoke validation samples 重算，新规则误伤下降：

| step | old penalty mean | new penalty mean |
|---:|---:|---:|
| 10 | `-0.1250` | `0.0000` |
| 20 | `-0.1406` | `-0.0156` |
| 30 | `-0.0625` | `0.0000` |
| 40 | `-0.1250` | `0.0000` |
| 50 | `-0.0781` | `-0.0156` |

### 25.6 长度档排查：2176/768 是当前可跑的保守上限

用 Gemma actor tokenizer 统计 `clean32_for_sft2` prompt 长度：

| prompt cap | 保留样本 | 丢弃 | 保留率 |
|---:|---:|---:|---:|
| 2048 | 17407 | 1531 | 91.92% |
| 2176 | 18048 | 890 | 95.30% |
| 2304 | 18931 | 7 | 99.96% |
| 3072 | 18938 | 0 | 100.00% |

2048 会过滤约 8% 数据；2176 能多收约 641 条，同时不回到历史 3072 OOM 档。

当前推荐长度：

```text
MAX_PROMPT_LENGTH=2176
MAX_RESPONSE_LENGTH=768
PPO_MAX_TOKEN_LEN_PER_GPU=2944
MAX_MODEL_LEN=3200
RM_MAX_MODEL_LEN=3456
ROLLOUT_MAX_NUM_BATCHED_TOKENS=3200
RM_MAX_NUM_BATCHED_TOKENS=3456
```

不要上 `MAX_PROMPT_LENGTH=2304, MAX_RESPONSE_LENGTH=768`，因为 `PPO_MAX_TOKEN_LEN_PER_GPU=3072` 曾在历史 run 中触发 actor backward OOM。

### 25.7 2026-05-07 50-step smoke 结果

smoke：

- experiment：`chatjoy_p2176_r768_smoke50_20260507_103012`
- log：`/tmp/chatjoy_p2176_r768_smoke50_20260507_103012.log`
- monitor：`/tmp/chatjoy_p2176_r768_smoke50_20260507_103012_monitor.log`
- metrics：`/home/ps/hanjiaqi/LlamaFactory/chatjoy/logs/verl_metrics/chatjoy_verl/chatjoy_p2176_r768_smoke50_20260507_103012.jsonl`
- validation samples：`/home/ps/hanjiaqi/LlamaFactory/chatjoy/logs/validation_samples/chatjoy_p2176_r768_smoke50_20260507_103012.jsonl`

配置：

```text
max_prompt_length = 2176
max_response_length = 768
ppo_max_token_len_per_gpu = 2944
rollout max_model_len = 3200
RM max_model_len = 3456
clip_grad = 1.0
test_freq = 10
save_freq = 0
```

结果：50/50 完成，约 5h19m，0 OOM，0 RuntimeError。

validation 曲线：

| step | val reward | val RM | anti-refusal | train score | KL loss | PPO KL | grad |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | `1.901` | `1.925` | `-0.024` | `1.155` | `1.046` | `0.037` | `1.30` |
| 20 | `1.899` | `1.929` | `-0.030` | `1.330` | `1.812` | `-0.022` | `1.70` |
| 30 | `1.882` | `1.906` | `-0.024` | `1.743` | `1.676` | `0.001` | `9.78` |
| 40 | `1.899` | `1.927` | `-0.028` | `1.568` | `1.736` | `-0.031` | `1.80` |
| 50 | `1.884` | `1.903` | `-0.019` | `1.555` | `1.755` | `-0.015` | `2.24` |

结论：

- online RM 已从 Phase2 的 `-0.63` 恢复到 `+1.90`，EOS/scoring 修复有效；
- reward 横盘是正常的：修复后 SFT actor 起点已经高，不应期待 50 step 内继续大涨；
- KL 可控：`ppo_kl` 在小范围波动，无爆炸；
- `actor/grad_norm` 有早期 spike（step6 `240.6`, step12 `84.8`），但这是 pre-clip norm，`clip_grad=1.0` 已生效；后续无连续失控；
- response clip 很低，无 aborted。

### 25.8 日志尾部报错解释

smoke 尾部出现：

```text
worker died or was killed ...
Exit code: 1. The process receives a SIGTERM
resource_tracker ... KeyError: '/psm_xxx'
```

这些发生在：

- `Training Progress: 100%|50/50`
- `Final validation metrics`

之后，是 Ray/vLLM 退出清理 worker 和 shared memory 的噪音，不代表训练失败。

关键确认：

- `CUDA out of memory: 0`
- `RuntimeError: 0`
- final validation 已写出

### 25.9 正式训练建议（当前最新）

正式训练前必须使用已修过的 `anti_refusal_reward.py` 和 `reward_loop.py`。

推荐正式训练配置：

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
clip_grad=1.0
ENTROPY_COEFF=0
LORA_RANK=8
TARGET_MODULES=[q_proj,k_proj,v_proj,o_proj]
```

不要在正式训练中覆盖 `trainer.total_training_steps=50`。50-step smoke 的后半段 cosine LR 已接近 0，不能代表正式训练学习强度。

按 `2176` prompt cap 估算：

- 可用训练样本约 18,048 条；
- `TRAIN_BATCH_SIZE=32` 时 1 epoch 约 564 steps；
- 比旧 2048 配置多约 641 条训练样本。

### 25.10 监控阈值

正式 run 每 50 step 检查：

| 指标 | 健康范围 / 动作 |
|---|---|
| `val-aux/chatjoy/rm_score/mean@1` | 应稳定 `> 1.7`；低于 `1.0` 需查 RM input / EOS / truncation |
| `val-aux/chatjoy/reward/mean@1` | 应稳定 `> 1.7`；若低于 RM 很多，查 anti-refusal |
| `val-aux/chatjoy/custom_reward/mean@1` | 新规则下应接近 `0`，通常 `>-0.05` |
| `actor/kl_loss` | 连续 `>3` 才警惕；`>4` 考虑 `kl_loss_coef 0.001 -> 0.002/0.005` |
| `actor/ppo_kl` | 绝对值持续 `>0.1` 才干预 |
| `actor/grad_norm` | 偶发 spike 可接受；连续 `>50` 才考虑降 LR 或 clip 更紧 |
| `response_length/clip_ratio` | 持续 `>0.03` 才考虑 response 长度或采样策略 |
| GPU 0-3 memory | 当前档峰值已到 `~32065 MiB`，不要再加 token |

### 25.11 不要做的事

- 不要回到 `2048/768`，会无必要地过滤更多长 prompt 数据；
- 不要上 `2304/768/3072`，显存已贴边；
- 不要因为 50-step reward 横盘就提高 LR；
- 不要扩 FFN / MoE target modules；
- 不要开 entropy，之前显存余量不足。

## 26) 2026-05-09 Gemma4 RM 正式 RL 启动与可续训状态

### 26.1 RM 选择

正式 RL 选择 **Gemma4 RM**，不是 Qwen RM v5。

原因：

- 当前 actor 是 Gemma4 SFT；Gemma4 RM 与 actor 同模型家族，作为 online RM 更贴近被优化分布；
- Gemma4 RM smoke 已跑通 reward loop、vLLM classify、actor update、weight update；
- smoke step10 validation：`val-aux/chatjoy/reward/mean@1=1.8829611471645493`，`val-aux/chatjoy/rm_score/mean@1=1.9000376499782168`，`anti_refusal_is_refusal=0.00273224043715847`；
- 未复现此前 RM `/classify` 长度越界、vLLM `EngineCore failed`、`WorkerProc failed` 或 CUDA OOM。

### 26.2 正式 run 信息

- tmux session：`chatjoy_gemma4rm_rl_formal_161335`
- experiment：`chatjoy_grpo_gemma4rm_phase2_600_20260509_161335`
- main log：`/home/ps/hanjiaqi/LlamaFactory/chatjoy/logs/chatjoy_grpo_gemma4rm_phase2_600_20260509_161335.log`
- metrics JSONL：`/home/ps/hanjiaqi/LlamaFactory/chatjoy/logs/verl_metrics/chatjoy_verl/chatjoy_grpo_gemma4rm_phase2_600_20260509_161335.jsonl`
- tensorboard：`/home/ps/hanjiaqi/LlamaFactory/chatjoy/logs/tensorboard/chatjoy_verl/chatjoy_grpo_gemma4rm_phase2_600_20260509_161335/`
- validation samples：`/home/ps/hanjiaqi/LlamaFactory/chatjoy/logs/validation_samples/chatjoy_grpo_gemma4rm_phase2_600_20260509_161335.jsonl`
- checkpoint/save dir：`/home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/verl/chatjoy_grpo_gemma4rm_phase2_600_20260509_161335/`
- RM model path：`/home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/gemma4-26b-a4b/chatjoy_rm_gemma4_sft_v5_merged_vllm`

启动保持 §25.9 的稳定训练档，仅替换 RM：

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

### 26.3 当前早期稳定性

截至 2026-05-09 16:3x：

- 已完成并写出 step 1、step 2 metrics；
- 未发现 `Traceback`、`OutOfMemory`、`CUDA out of memory`、`RuntimeError`、`EngineCore failed`、`WorkerProc failed`；
- GPU0-3 actor 侧显存仍贴边但保留约 `1.6~2.8GB` 余量；GPU4-7 RM 侧保留约 `10.8GB` 余量；
- 训练速度约 `345~358s/step`，主要耗时仍在 generation + actor update。

step 1：

```text
critic/score/mean=1.6668407917022705
critic/score/max=5.0836591720581055
critic/score/min=-2.2870399951934814
actor/kl_loss=0.979651961941272
actor/ppo_kl=-0.07529948705834816
actor/grad_norm=2.1291407346725464
response_length/clip_ratio=0.0
prompt_length/clip_ratio=0.0
response/aborted_ratio=0.0
timing_s/step=344.85097714001313
```

step 2：

```text
critic/score/mean=1.5620043277740479
critic/score/max=5.162975788116455
critic/score/min=-1.7212215662002563
actor/kl_loss=1.4733934113755822
actor/ppo_kl=-0.010716968155207321
actor/grad_norm=1.5833182334899902
response_length/clip_ratio=0.0
prompt_length/clip_ratio=0.0
response/aborted_ratio=0.0
timing_s/step=357.9563549598679
```

显存快照：

```text
GPU0: used 30451 MiB, free 1659 MiB, util 99%
GPU1: used 29277 MiB, free 2834 MiB, util 99%
GPU2: used 29357 MiB, free 2754 MiB, util 99%
GPU3: used 29525 MiB, free 2586 MiB, util 99%
GPU4: used 21226 MiB, free 10884 MiB, util 0%
GPU5: used 21226 MiB, free 10884 MiB, util 0%
GPU6: used 21226 MiB, free 10884 MiB, util 0%
GPU7: used 21226 MiB, free 10884 MiB, util 0%
```

### 26.4 磁盘与 checkpoint 策略

磁盘：

```text
/dev/sda2  7.0T total  6.1T used  510G avail  93% mounted at /home/ps
```

相关目录：

```text
/home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/verl  149G
/home/ps/hanjiaqi/LlamaFactory/chatjoy/logs        60M
/tmp/ray                                           7.5G
```

本次正式 run：

- `SAVE_FREQ=50`
- `MAX_ACTOR_CKPT_TO_KEEP=1`
- `ACTOR_CKPT_ASYNC_SAVE=False`

按历史 q/k/v/o rank8 actor checkpoint 估计，单个 checkpoint 约 `53G`，保存时临时峰值可能 `100G+`。当前 `510G` 可用空间足够，但分区使用率已经 `93%`，Ray 仍可能打印 `/tmp/ray session over 95% full` 风险提示；继续训练期间要避免额外大文件写入同分区。

### 26.5 续训规则

当前 step 2 尚无 checkpoint；此时中断只能从头重跑，损失约前几步时间，不影响模型质量。

第一次真正可续训点是 step 50 保存完成后。必须确认：

```text
/home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/verl/chatjoy_grpo_gemma4rm_phase2_600_20260509_161335/global_step_50/
/home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/verl/chatjoy_grpo_gemma4rm_phase2_600_20260509_161335/latest_checkpointed_iteration.txt
```

并且：

```text
latest_checkpointed_iteration.txt == 50
```

续训时保持同一个 `SAVE_DIR` 和相同核心训练配置，把：

```text
trainer.resume_mode=disable
```

改为：

```text
trainer.resume_mode=auto
```

更稳的显式恢复：

```text
trainer.resume_mode=resume_path
trainer.resume_from_path=/home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/verl/chatjoy_grpo_gemma4rm_phase2_600_20260509_161335/global_step_50
```

恢复时 `trainer.total_training_steps=600` 保持不变；verl 会从 checkpoint 的 global step 继续到 600，不是再额外跑 600。

不要在日志显示 `save_checkpoint` 正在写入时硬杀进程；若显卡后续要被占用，优先等 step 50 checkpoint 写完再停。
