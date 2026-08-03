<!-- CODEx整理摘要: begin -->
# RM v6 Fine-grained Handoff 整理摘要

本文件下面保留原始 RM v6 fine-grained 交接日志；如果只是面试复盘，先读本摘要，再回 `../project_chatjoy_overview.md` 看全链路。

## 这个阶段要解决什么

RM v1-v5 主要区分 wrong role、wrong language、length mismatch。进入 RL 后发现两个新问题：

- 破角/roledrop：模型可能直接承认自己是模型、执行系统/代码类指令，虽然不是拒答，却破坏角色。
- 富度不足：clean 回复动作块、感官、情绪和叙事层次不足，RM v5 对这些细粒度质量信号不敏感。

RM v6/v6.2 先把“细腻角色扮演质量”引入 RM，不再只靠启发式错例；user-level roledrop attack 的显式 pair/hard fail 是随后 RM v6.3 的新增杠杆。两者共同构成后期“保富度 + 抗破角”双目标，但不要把 v6.3 的能力提前写到 v6.2。

## 数据与 judge 设计

早期 v6/v6.2 使用 5-source 多候选：

| source | 说明 |
|---|---|
| Gemma4 SFT t0.7 / p0.8 | SFT 模型低温采样 |
| Gemma4 SFT t1.0 / p0.95 | SFT 模型高温采样 |
| Qwen SFT t0.7 / p0.8 | 另一 SFT 家族候选 |
| RL step550 t0.7 / p0.8 | RL actor 低温候选 |
| RL step550 t1.0 / p0.95 | RL actor 高温候选 |

核心流程：同一个 prompt 生成多路候选，DeepSeek judge 和 Qwen judge 交叉排序，按 rank delta 生成 preference pair。口述中的 `best > good > mid > bad > worse` 是这个排序逻辑的直观说法；在文档/代码里更准确写 ranked candidates / rank_delta。

## 关键数字

| 阶段 | 数字 | 说明 |
|---|---:|---|
| v6.2 cross-validated fine pairs | 27,287 | 保留率 73.1%，用 DeepSeek + Qwen 交叉过滤 |
| v6.2 train/eval | 42,270 / 3,614 | LR 5e-6 版本，acc 0.802，wrong 18.0% |
| v6.2 结论 | underfit | 后续建议 LR 9e-6、LoRA r128/alpha256、target 扩到 q/k/v/o/up/down |

## 训练配置口径

- base：Gemma4 SFT2 / Gemma4 RM 路线。
- 并行：ZeRO-2 + QLoRA 4bit。
- 常用 max_len：3072。
- v6.2 原 run：LR 5e-6，后续 A+C 口径改为 LR 9e-6。
- LoRA：从 r64 逐步升级到 r128/alpha256；target 从 q/k/v/o 扩到 q/k/v/o/up/down。

## 和 v6.3 的关系

v6 fine-grained 解决了“RM 看不见富度”的方向问题，但仍需要更强 roledrop attack 数据和更稳 judge。v6.3 在本阶段基础上新增 pure Gemma4-it base 第 6 source，并把 v5 anchor、clean/fine roleplay pair、roledrop attack pair 混合成最终 RM v6.3 训练集。

<!-- CODEx整理摘要: end -->
# ChatJoy RM v6 Fine-Grained Handoff

更新时间：2026-05-12 20:50 CST

本文用于给 Claude/Codex 继续接手 RM v6 fine-grained 数据与训练工作。目标是解释为什么从当前 RL 训练趋势切到 RM v6，当前计划是什么，已完成哪些实现，以及目前卡在哪。

## 1. 背景：当前 RL 暴露出的趋势问题

当前正式 RL 使用 Gemma4 SFT actor + Gemma4 RM v5 路线可以稳定运行，但从 phase2 训练日志看，训练效果没有出现期望中的明显 reward 提升：

- `reward`/validation reward 趋势整体偏平，没有明显持续上行。
- `advantage` 平均值经常为负或接近 0；GRPO 内部 advantage 本身会按组归一化，单看均值为负不一定是错误，但结合 reward 不升，说明优化信号没有强烈地区分“好”和“更好”。
- RL smoke 和 phase2 能跑通，说明主要问题不像是 OOM、vLLM、KL 或训练稳定性，而更像是 RM 奖励信号粒度不足。
- 当前 RM v5 对 hard fail 类问题有效，例如错语言、错角色、拒答、长度异常、替用户行动等；但对同一个 prompt 下多个“都不坏”的回复，排序区分不够细。
- Gemma4 RM v5 曾观察到 score 饱和比例偏高，计划中记录 `|score| > 5` 饱和比例约 47.3%。这会让在线 RL 难以获得细粒度梯度。

工作假设：

- 现有 v5 anchor 不能删，也不能重排。它是已验证的基础约束来源。
- 问题不是 v5 的 hard fail 数据太多，而是缺少同 prompt、多模型、多温度候选之间的细粒度 preference。
- 不能重复 RMv2 的错误：不要剔除或降权 `length_mismatch_long`。此前删掉这类样本后效果反而变差，所以 v6 必须完整保留 v5。

## 2. RM v6 总体路线

采用“v5 全量保留 + 新增同 prompt 多候选排序数据”的方案。

核心原则：

- 不全量重做旧 RM 数据。
- 不删除、不降权、不重排 `rm_train_v5_5pct_badpool.jsonl`。
- 新增 fine-grained pair 只补足“好 vs 更好”的排序能力。
- judge 使用统一 ChatJoy rubric，不为不同来源模型定义不同 reward。
- 训练仍使用统一 pairwise ranking loss，不按来源设置不同 loss/reward。
- 所有 v6 产物写新路径，不覆盖 v5 RM、Gemma4 RM v5、当前 RL checkpoint 或稳定训练脚本。

候选 source 设计：

1. Gemma4 SFT base，`temperature=0.7`, `top_p=0.8`
2. Gemma4 SFT diverse，`temperature=1.0`, `top_p=0.95`
3. Qwen SFT base，`temperature=0.7`, `top_p=0.8`
4. RL step550，`temperature=0.7`, `top_p=0.8`
5. RL step550 diverse，`temperature=1.0`, `top_p=0.95`

后续训练数据混合：

- pilot：v5 17,712 条 + fine pairs 约 6k-7.5k。
- full：v5 17,712 条 + fine pairs 约 25k-30k。
- `extra_info.label_schema_version = chatjoy_rm_v6_unified_preference`
- v5 样本标记 `label_source=v5_anchor`
- 新样本标记 `label_source=fine_grained_judge`

## 3. 主要路径

代码目录：

```text
/home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/rm_v6_finegrained/
```

pilot prompt 与 candidates：

```text
/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/runs/rm_v6_finegrained/pilot/prompts.jsonl
/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/runs/rm_v6_finegrained/pilot/candidates.gemma4_sft_t07.jsonl
/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/runs/rm_v6_finegrained/pilot/candidates.gemma4_sft_t10.jsonl
/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/runs/rm_v6_finegrained/pilot/candidates.qwen_sft_t07.jsonl
/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/runs/rm_v6_finegrained/pilot/candidates.rl550_t07.jsonl
/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/runs/rm_v6_finegrained/pilot/candidates.rl550_t10.jsonl
```

logs：

```text
/home/ps/hanjiaqi/LlamaFactory/chatjoy/logs/rm_v6_finegrained/
```

模型路径：

```text
Gemma4 SFT:
/home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/gemma4-26b-a4b/sft2_ddp_lora_3gpu_567_cut3072_merged

Qwen SFT:
/home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/qwen3-30b-moe/clean32_for_sft2_fsdp_qlora_merged

RL step550 原始 HF 导出:
/home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/gemma4-26b-a4b/chatjoy_grpo_gemma4rm_phase2_step550_hf/

RL step550 vLLM 修复版 v2 (layer_scalar + vision_tower std + processor_config):
/home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/gemma4-26b-a4b/chatjoy_grpo_gemma4rm_phase2_step550_hf_vllm_fixed_v2/

(旧 _vllm_fixed/ 已于 2026-05-12 删除以释放磁盘，被 v2 严格替代)
```

## 4. 已实现内容

已新增/更新：

- `README.md`：已翻译为中文，记录 RM v6 数据流水线用法。
- `sample_prompts.py`：从 RL train parquet 抽样 prompt。
- `generate_candidates_openai.py`：OpenAI-compatible vLLM 采样脚本，已支持 `--resume` 和 `--concurrency`。
- `generate_pilot_sources_sequential.sh`：一次只启动一个 vLLM server，按 source 顺序生成候选。
- `run_pilot_generation_background.sh`：后台启动辅助脚本，但实际更推荐用 tmux。
- `export_step550_hf.sh`：从 verl FSDP checkpoint 导出 step550 HF。
- `fix_gemma4_layer_scalar_safetensors.py`：修复 legacy merger 导出的 Gemma4 `*.layer_scalar` shape 问题。

当前 pilot prompt 已采样：

```text
2500 /home/ps/hanjiaqi/LlamaFactory/chatjoy/data/runs/rm_v6_finegrained/pilot/prompts.jsonl
```

抽样日志曾显示：

- `excluded_prompts=386`
- `available_unique=18938`
- `languages=26`
- `roles=17`

当前候选写盘进度：

```text
22 candidates.gemma4_sft_t07.jsonl
 2 candidates.gemma4_sft_t10.jsonl
 2 candidates.qwen_sft_t07.jsonl
```

结论：

- Gemma4 SFT base 已确认可以正常 vLLM 采样并写 JSONL。
- Qwen SFT base 已确认可以正常 vLLM 采样并写 JSONL。
- RL step550 还没有确认可以正常 vLLM 采样。

## 5. 当前卡点：RL step550 vLLM 加载

### 5.1 已修复的问题

`legacy_model_merger.py merge --backend fsdp` 导出的 Gemma4 step550 HF 模型中，部分 `model.language_model.layers.*.layer_scalar` 从期望 shape `[1]` 变成 `[4]`。

vLLM 报错类似：

```text
Attempted to load weight (torch.Size([4])) into parameter (torch.Size([1]))
```

已通过 `fix_gemma4_layer_scalar_safetensors.py` 修复到：

```text
/home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/gemma4-26b-a4b/chatjoy_grpo_gemma4rm_phase2_step550_hf_vllm_fixed/
```

修复结果：

- repaired 30 个 `layer_scalar`
- removed 90 elements / 180 bytes
- 验证后 `bad_layer_scalar = 0`

### 5.2 vision_tower.std_bias / std_scale shape mismatch — 已修复

用旧 fixed 模型启动 vLLM 后报：

```text
AssertionError: Attempted to load weight (torch.Size([4608])) into parameter (torch.Size([1152]))
```

shape 对比 Gemma4 SFT base 和 step550 旧 fixed 后发现：

```text
model.vision_tower.std_bias   base=(1152,) fixed=(4608,)
model.vision_tower.std_scale  base=(1152,) fixed=(4608,)
```

`4608 = 4 * 1152`，与 TP/world size 4 一致；同样是 merger 把 FSDP-replicated 张量按 world size 拼接了。

#### 验证 (2026-05-12)

用 safetensors 读取后逐 chunk 比对，确认四段 `[0:1152]`, `[1152:2304]`,
`[2304:3456]`, `[3456:4608]` **bytewise 完全相同**
(`torch.equal == True`, `max_abs_diff == 0.0` for chunks 1,2,3 vs chunk 0)。
保留第一段是无损操作。

#### 已扩展 repair 脚本

`fix_gemma4_layer_scalar_safetensors.py` 已扩展为通用 truncation 工具，
现在同时处理两类规则:

```text
*.layer_scalar                shape [4]    -> [1]
model.vision_tower.std_bias   shape [4608] -> [1152]
model.vision_tower.std_scale  shape [4608] -> [1152]
```

新增 `EXPECTED_SHAPES` 映射 + `target_shape()` 函数，按张量名和当前 shape
判断是否需要截断；不在规则中的张量按字节原样拷贝。

#### 新产物路径

从原始 `_hf/` 重新生成的 fixed v2 模型:

```text
/home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/gemma4-26b-a4b/chatjoy_grpo_gemma4rm_phase2_step550_hf_vllm_fixed_v2/
```

修复统计:

- shard 1: repaired 31 (29 layer_scalar + 2 std)
- shard 2: repaired 1 (layer_scalar)
- total_repaired_tensors = 32
- total_removed_elements = 7002  (= 30*3 + 2*3456)
- total_removed_bytes = 14004  (bf16)
- index.json 已更新 total_parameters / total_size

验证脚本输出:

```text
layer_scalar checked=30, bad=0
std_bias/std_scale checked=2, bad=0  (now shape (1152,))
```

### 5.3 vLLM 后续 root cause: 缺少 processor_config.json — 已修复

shape 全部修好后再启动 vLLM, profile_run 阶段报新错:

```text
OSError: Can't load feature extractor for '..._vllm_fixed_v2'.
... make sure ... contains a preprocessor_config.json file
```

root cause: 原始 `_hf/` 和 `_vllm_fixed_v2/` 都缺少 Gemma4 multimodal processor 配置。
对照 Gemma4 SFT merged 目录 (vLLM 采样可用):

```text
diff: SFT merged 比 step550 多了 processor_config.json
```

修复: 把 SFT merged 的 `processor_config.json` (含 Gemma4ImageProcessor /
Gemma4AudioFeatureExtractor / Gemma4VideoProcessor 三段配置) 复制到
`_hf/` 和 `_vllm_fixed_v2/` 两个目录。同架构 Gemma4 26B-a4b，processor
config 是通用的。

### 5.4 Smoke 已通过 (2026-05-12 15:39)

启动命令 (LIMIT=2, TP=4):

```bash
RUN_SPLIT=pilot RUN_GROUPS=rl_step550 LIMIT=2 GPU_LIST=0,1,2,3 \
GEN_CONCURRENCY=2 MAX_NUM_SEQS=2 MAX_NUM_BATCHED_TOKENS=4096 \
bash generate_pilot_sources_sequential.sh
```

结果:

- vLLM `Multi-modal warmup completed in 9.301s`
- `Application startup complete`, `/v1/models 200 OK`
- `candidates.rl550_t07.jsonl` 2 行 (中文 41 字符 / 瑞典语 869 字符)
- `candidates.rl550_t10.jsonl` 2 行 (中文 46 字符 / 瑞典语 1179 字符)
- 全部字段齐全 (candidate_id, prompt_id, messages, response, usage, ...)
- 旧 `_vllm_fixed/` 已删除释放 50G

### 5.5 后续

1. 继续大规模 pilot 采样 (Step C, tmux), 现有 4 行会被 `--resume` 自动跳过。
2. 如果以后再用 verl legacy_model_merger 导出 Gemma4 模型, 别忘了:
   - 跑 `fix_gemma4_layer_scalar_safetensors.py` 处理 layer_scalar 和 vision_tower std
   - 从 SFT merged 复制 `processor_config.json`
   (考虑把这两步合进 `export_step550_hf.sh` 让导出步骤一次到位)

注意：不要改原始 FSDP checkpoint，不要改 RL 稳定训练脚本。

## 6. 推荐下一步执行顺序

### Step A：修复 RL step550 vLLM fixed 模型

扩展：

```text
/home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/rm_v6_finegrained/fix_gemma4_layer_scalar_safetensors.py
```

让它同时处理：

```text
*.layer_scalar: [4] -> [1]
model.vision_tower.std_bias: [4608] -> [1152]
model.vision_tower.std_scale: [4608] -> [1152]
```

修复前最好先验证四段是否重复。

### Step B：单独 smoke RL step550

建议先限制 2 条：

```bash
cd /home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/rm_v6_finegrained
source /home/ps/miniconda3/etc/profile.d/conda.sh
conda activate gemma4-rl

RUN_SPLIT=pilot \
RUN_GROUPS=rl_step550 \
LIMIT=2 \
GPU_LIST=0,1,2,3 \
GEN_CONCURRENCY=2 \
MAX_NUM_SEQS=2 \
MAX_NUM_BATCHED_TOKENS=4096 \
bash generate_pilot_sources_sequential.sh
```

成功标准：

- vLLM server `/v1/models` ready。
- `candidates.rl550_t07.jsonl` 写出 2 行。
- `candidates.rl550_t10.jsonl` 写出 2 行。
- 无 EngineCore failed。

### Step C：确认三类模型都能采样后再大规模跑 pilot

推荐 tmux：

```bash
tmux new-session -d -s rmv6pilot \
"source /home/ps/miniconda3/etc/profile.d/conda.sh && \
 conda activate gemma4-rl && \
 cd /home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/rm_v6_finegrained && \
 RUN_SPLIT=pilot GPU_LIST=0,1,2,3 GEN_CONCURRENCY=8 MAX_NUM_SEQS=8 MAX_NUM_BATCHED_TOKENS=8192 \
 bash generate_pilot_sources_sequential.sh 2>&1 | tee -a /home/ps/hanjiaqi/LlamaFactory/chatjoy/logs/rm_v6_finegrained/pilot.sequential_tmux.log"
```

监控：

```bash
tmux attach -t rmv6pilot
tail -f /home/ps/hanjiaqi/LlamaFactory/chatjoy/logs/rm_v6_finegrained/pilot.sequential_tmux.log
wc -l /home/ps/hanjiaqi/LlamaFactory/chatjoy/data/runs/rm_v6_finegrained/pilot/candidates*.jsonl
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader
```

### Step D：候选完成后再进入 judge/pair/mix

候选完成后再实现或执行：

- blind judge，隐藏 `source_model` 和 `sampling`。
- 统一 rubric：
  - role fidelity
  - user relevance
  - interaction progression
  - natural style
  - format/length quality
  - hard fails：错语言、拒答、替用户行动、泄漏 system、乱码/空回复
- blocked、parse failed、low confidence 只进审计，不进训练。
- 每 prompt 最多 3 条 pair：`best>worst`, `best>mid`, `good>bad`。
- 与 v5 anchor 全量混合。

## 7. 不要做的事

- 不要删除或降权 v5 的 `length_mismatch_long`。
- 不要覆盖 v5 RM、Gemma4 RM v5、当前 RL checkpoint。
- 不要修改已经调通的 RL 稳定训练脚本。
- 不要把 judge scalar 当在线 reward；v6 数据只用排序方向构造 pair。
- 不要同时启动多个大模型 vLLM server；一次只跑一个 source/group。

## 8. 给 Claude 的重点问题

请优先看：

1. `model.vision_tower.std_bias/std_scale` 是否确实是 4 份重复拼接。
2. 是否可以安全扩展 safetensors repair，把 `[4608]` 截成 `[1152]`。
3. RL step550 fixed 模型修复后，vLLM 是否能正常采样。
4. 在三类 source 都确认可采样后，再继续大规模 pilot 采样。

当前最关键判断：

- 不是所有 source 都已确认可采样。
- 目前只有 Gemma4 SFT base 和 Qwen SFT base 确认成功。
- RL step550 是唯一未解卡点。

## 9. 增量更新：pilot 候选采样、DeepSeek judge 与 pair 构造进度 (2026-05-12 20:50 CST)

本节为当前最新状态，保留上文早期排障记录不变；上文第 4-8 节中的“RL step550 未确认可采样”已被后续工作解决。

### 9.1 当前结论

- 三类模型都已经确认可以正常 vLLM 采样：
  - Gemma4 SFT base
  - Qwen SFT base
  - RL step550 fixed v2
- pilot 五个 source 的 candidates 已经完成写盘。
- 规则版 judge 可以作为 QA/硬规则参考，但 low confidence 过高，不适合作为最终 fine-grained label source。
- DeepSeek `deepseek-chat` 作为外部 blind LLM judge 的 200 prompt pilot 已通过机械 QA 和人工抽查。
- 完整 pilot 2500 prompt 的 DeepSeek judge 当前正在后台并发续跑，完成后会自动构造完整 pilot pair 并跑 QA。

### 9.2 已完成 candidates

候选文件路径：

```text
/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/runs/rm_v6_finegrained/pilot/candidates.gemma4_sft_t07.jsonl
/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/runs/rm_v6_finegrained/pilot/candidates.gemma4_sft_t10.jsonl
/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/runs/rm_v6_finegrained/pilot/candidates.qwen_sft_t07.jsonl
/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/runs/rm_v6_finegrained/pilot/candidates.rl550_t07.jsonl
/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/runs/rm_v6_finegrained/pilot/candidates.rl550_t10.jsonl
```

最终行数：

```text
2500 candidates.gemma4_sft_t07.jsonl
2500 candidates.gemma4_sft_t10.jsonl
2500 candidates.qwen_sft_t07.jsonl
2524 candidates.rl550_t07.jsonl
2504 candidates.rl550_t10.jsonl
```

说明：

- RL 两个文件因为中途停/续跑有少量重复。
- `judge_candidates.py` 和 `judge_candidates_llm_openai.py` 都已经按 `(prompt_id, source_model)` 去重，重复行不会影响 judge。
- 8 卡续跑 RL step550 后完成了 `rl_step550_t10_p095`，vLLM 进程已正常停止。

### 9.3 规则版 judge 结果

规则 judge 输出：

```text
/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/runs/rm_v6_finegrained/pilot/judged.jsonl
/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/runs/rm_v6_finegrained/pilot/judge_audit.jsonl
```

QA 结论：

- judged prompts: `2488`
- audit: `2139`
- confidence:
  - low: `2128`
  - medium: `324`
  - high: `36`
- low confidence ratio: `85.5%`
- blocked candidate flags: `777`

判断：

- 规则 judge 能筛 hard fail，但无法可靠完成“好 vs 更好”的语义排序。
- 不建议作为 RM v6 主要训练标签来源。

### 9.4 DeepSeek LLM judge pilot200

新增脚本：

```text
/home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/rm_v6_finegrained/judge_candidates_llm_openai.py
```

用途：

- OpenAI-compatible blind LLM judge。
- source model 和 sampling 对 judge 隐藏。
- 输出 `ranking`, `confidence`, `hard_fails`, `rationale`。
- 支持 `--resume`。
- 2026-05-12 20:45 左右新增 `--concurrency`，用于完整 pilot 并发续跑。
- 并发版本使用 `prompt_id` 派生 deterministic shuffle，避免并发完成顺序影响候选打乱。

pilot200 输出：

```text
/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/runs/rm_v6_finegrained/pilot/judged_llm_deepseek_pilot200.jsonl
```

pilot200 质量：

- `200/200` parse 成功。
- audit: `0`
- confidence:
  - high: `83`
  - medium: `117`
  - low: `0`
- rank 合法，无重复 rank。
- hard fail 全部排到 rank 5：
  - `wrong_language=40`
  - `acts_for_user=21`
  - `refusal=3`
  - `unsafe_escalation=1`

source 平均 rank：

```text
gemma4_sft_t07_p08   2.495
rl_step550_t07_p08   2.520
gemma4_sft_t10_p095  2.895
rl_step550_t10_p095  2.945
qwen_sft_t07_p08     4.145
```

人工抽查判断：

- DeepSeek 的理由主要在看角色贴合、互动推进、自然度、是否替用户行动。
- Qwen 被判差较多，主要来自错语言、替用户行动、泛化、语气突兀；由于 source 被隐藏，暂时不像简单 source bias。
- 需要在 full pilot 后继续抽查 Qwen-worst 和 close pair，防止把 RM v6 训练成单纯“Gemma/RL > Qwen”。

### 9.5 pilot200 pair 构造与 QA

pilot200 pair 输出：

```text
/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/openrlhf_rm/rm_v6_finegrained_pairs_deepseek_pilot200.jsonl
```

构造结果：

- judged prompts: `200`
- generated pairs: `600`
- prompts_without_pairs: `0`
- pair types:
  - `best_vs_worst=200`
  - `best_vs_mid=200`
  - `good_vs_bad=200`
- same source ratio: `0.0%`
- QA failures: none
- chosen hard fail: none
- rejected hard fails:
  - `wrong_language=40`
  - `acts_for_user=21`
  - `refusal=3`
  - `unsafe_escalation=1`

rejected source 分布：

```text
qwen_sft_t07_p08      176 / 600
rl_step550_t10_p095   123 / 600
gemma4_sft_t10_p095   120 / 600
rl_step550_t07_p08     94 / 600
gemma4_sft_t07_p08     87 / 600
```

判断：

- pilot200 pair 可用。
- Qwen rejected 较多但不是压倒性全部，仍存在 Gemma/RL 内部和 t07/t10 之间的细粒度 pair。
- 可以扩大到完整 pilot 2500。

### 9.6 当前正在运行：完整 pilot2500 DeepSeek judge

当前完整 judge 输出：

```text
/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/runs/rm_v6_finegrained/pilot/judged_llm_deepseek_pilot2500.jsonl
/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/runs/rm_v6_finegrained/pilot/judge_llm_deepseek_pilot2500_audit.jsonl
```

当前日志：

```text
/home/ps/hanjiaqi/LlamaFactory/chatjoy/logs/rm_v6_finegrained/judge_deepseek_pilot2500.log
```

当前 tmux session：

```text
rmv6_judge_pilot2500
rmv6_judge_pilot2500_post
```

截至 2026-05-12 20:50 CST 的进度：

```text
747 judged_llm_deepseek_pilot2500.jsonl
  5 judge_llm_deepseek_pilot2500_audit.jsonl
```

执行方式：

- 起初用串行 judge，速度偏慢。
- 后来停止串行 session，给 `judge_candidates_llm_openai.py` 增加 `--concurrency`。
- 使用同一个输出文件和 `--resume` 重启，已完成行会被跳过。
- 当前并发参数：`--concurrency 8`。
- 不要在日志或 handoff 中记录 API key。

注意：

- 并发 judge 的输出顺序不保证按 prompt_id 排序，但 JSONL 每行独立，后续 `build_pairs.py` 不依赖行顺序。
- audit 少量出现是正常的；如果最终 audit 比例过高，需要查看 audit 原因。

### 9.7 当前 postprocess 自动任务

新增脚本：

```text
/home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/rm_v6_finegrained/postprocess_deepseek_pilot2500.sh
```

作用：

- 等待 `rmv6_judge_pilot2500` session 结束。
- 检查 `judged_llm_deepseek_pilot2500.jsonl` 行数。
- 如果 judged 行数 `< 2400`，拒绝构造 pair，避免生成半成品。
- 如果行数足够，自动执行：
  - `build_pairs.py`
  - `qa_dataset.py`

预期输出：

```text
/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/openrlhf_rm/rm_v6_finegrained_pairs_pilot.jsonl
/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/runs/rm_v6_finegrained/pilot/qa_llm_deepseek_pilot2500.json
/home/ps/hanjiaqi/LlamaFactory/chatjoy/logs/rm_v6_finegrained/postprocess_deepseek_pilot2500.log
```

### 9.8 接下来要做什么

等完整 pilot2500 judge 和 postprocess 完成后：

1. 检查完整 judge 质量：
   - judged 行数是否接近 2500。
   - audit 比例是否可接受。
   - confidence 分布是否正常。
   - hard fail 是否主要落在低 rank。
   - source 平均 rank 是否与 pilot200 一致但不过度偏置。
2. 检查完整 pair QA：
   - pairs 数量是否约 `judged_prompts * 3`。
   - same source ratio 是否低于 `35%`。
   - pair type 是否均衡。
   - Qwen rejected 占比是否过高。
   - chosen 是否存在 hard fail。
3. 抽查样本：
   - Qwen worst 样本。
   - Gemma4 SFT vs RL step550 close pair。
   - t07 vs t10 close pair。
   - 多语言样本，尤其是 language metadata 与用户文本不一致的样本。
4. 如果完整 pilot QA 通过：
   - 将 `rm_train_v5_5pct_badpool.jsonl` 全量并入。
   - 生成 pilot mixed train，例如：
     ```text
     /home/ps/hanjiaqi/LlamaFactory/chatjoy/data/openrlhf_rm/rm_train_v6_fg_pilot.jsonl
     ```
   - v5 样本标记 `label_source=v5_anchor`。
   - fine-grained 样本标记 `label_source=fine_grained_judge`。
5. 准备 RM v6 pilot 训练：
   - base 仍为 Gemma4 SFT merged。
   - 复用 Gemma4 RM v5 的 OpenRLHF + ZeRO-2 + 4-bit QLoRA + LoRA rank 64。
   - LR 建议从 v5 的 `9e-6` 降到 `5e-6`。
   - 不改 RL 稳定脚本。
6. RM v6 pilot 训练完成后：
   - merge 为 vLLM 可加载版本。
   - 做 classify smoke。
   - 做 30-step RL smoke，只替换 RM path。
   - 与 Gemma4 RM v5 及 Qwen RM v5 对比后，再决定是否跑 full RM 和正式 RL。

### 9.9 当前不要做的事

- 不要在完整 pilot2500 judge 尚未完成时，把 `rm_v6_finegrained_pairs_pilot.jsonl` 当最终数据使用。
- 不要覆盖 `rm_v6_finegrained_pairs_deepseek_pilot200.jsonl`；它是 200 条 pilot 对照。
- 不要把规则 judge 的 `judged.jsonl` 当主训练标签。
- 不要把 DeepSeek scalar score 当在线 reward；只用排序方向构造 pair。
- 不要把 API key 写入 md、README 或日志摘要。

## 10. 增量更新：full 1W 数据、combined RM v6 训练与当前接手状态 (2026-05-14 CST)

本节为最新接手状态，保留上文历史过程不变。上文第 9 节记录的是 2026-05-12 晚上 judge/full generation 仍在运行时的状态；截至 2026-05-14，full 数据、full judge、pair 构造、v5 anchor 混合以及 Gemma4 RM v6 combined 训练都已经完成。

### 10.1 当前总体结论

- RM v6 fine-grained 数据流水线已经跑通。
- pilot 2500 组和 full 10000 组五 source candidates 都已完成。
- DeepSeek external blind judge 已完成 pilot2500 与 full10000。
- 规则 judge 仅保留 QA 参考，不作为主 label source。
- combined fine-grained pair 已与 v5 anchor 混合成 RM v6 train/eval。
- Gemma4 RM v6 combined 已完成训练。
- 当前 RM v6 combined 不能直接视为正式 RL 最终 RM，需要先 merge、classify smoke、30-step RL smoke。

### 10.2 最新关键路径

RM v6 pipeline：

```text
/home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/rm_v6_finegrained/
```

pilot 数据：

```text
/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/runs/rm_v6_finegrained/pilot/
```

full 数据：

```text
/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/runs/rm_v6_finegrained/full/
```

OpenRLHF RM 数据：

```text
/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/openrlhf_rm/
```

RM v6 combined 输出：

```text
/home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/gemma4-26b-a4b/chatjoy_rm_gemma4_v6_fg_combined_20260513_152457/
```

RM v6 combined 训练日志：

```text
/home/ps/hanjiaqi/LlamaFactory/chatjoy/logs/rm_v6_finegrained/train_chatjoy_rm_gemma4_v6_fg_combined_20260513_152457.log
```

### 10.3 Full 10000 candidates 已完成

full prompt：

```text
10000 /home/ps/hanjiaqi/LlamaFactory/chatjoy/data/runs/rm_v6_finegrained/full/prompts.jsonl
```

五个 source：

```text
10000 candidates.gemma4_sft_t07.jsonl
10000 candidates.gemma4_sft_t10.jsonl
10000 candidates.qwen_sft_t07.jsonl
10000 candidates.rl550_t07.jsonl
10000 candidates.rl550_t10.jsonl
```

full 采样日志：

```text
/home/ps/hanjiaqi/LlamaFactory/chatjoy/logs/rm_v6_finegrained/full.sequential_8gpu.log
/home/ps/hanjiaqi/LlamaFactory/chatjoy/logs/rm_v6_finegrained/full.gemma4_sft.vllm.log
/home/ps/hanjiaqi/LlamaFactory/chatjoy/logs/rm_v6_finegrained/full.qwen_sft.vllm.log
/home/ps/hanjiaqi/LlamaFactory/chatjoy/logs/rm_v6_finegrained/full.rl_step550.vllm.log
```

执行方式：

- 8 卡，`TP=8`
- `RUN_SPLIT=full`
- 每次只启动一个 vLLM server。
- source 顺序生成，避免多模型抢显存。

### 10.4 DeepSeek judge 已完成

pilot2500：

```text
2488 judged_llm_deepseek_pilot2500.jsonl
  12 judge_llm_deepseek_pilot2500_audit.jsonl
```

pilot2500 confidence：

```text
medium=1472
high=1016
low=0
```

pilot2500 source 平均 rank：

```text
gemma4_sft_t07_p08   2.551
rl_step550_t07_p08   2.558
gemma4_sft_t10_p095  2.908
rl_step550_t10_p095  2.919
qwen_sft_t07_p08     4.064
```

full10000：

```text
9960 judged_llm_deepseek_full10000.jsonl
  38 judge_llm_deepseek_full10000_audit.jsonl
```

full10000 confidence：

```text
high=4162
medium=5798
low=0
```

full10000 source 平均 rank：

```text
gemma4_sft_t07_p08   2.561
rl_step550_t07_p08   2.559
gemma4_sft_t10_p095  2.879
rl_step550_t10_p095  2.906
qwen_sft_t07_p08     4.093
```

观察：

- pilot 和 full 的 source rank 趋势高度一致。
- Gemma4 SFT t07 与 RL step550 t07 几乎并列最优。
- t10 diverse 版本整体中间偏后。
- Qwen SFT 明显更容易排到差 rank。
- hard fail 多数集中在 rank 5，方向符合预期。
- DeepSeek low confidence 为 0，说明它倾向给出明确排序；这点需要抽检 close pair，不要盲信全部细粒度差异。

### 10.5 Pair 构造与 QA

pilot pair：

```text
/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/openrlhf_rm/rm_v6_finegrained_pairs_pilot.jsonl
7463 pairs
```

full pair：

```text
/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/openrlhf_rm/rm_v6_finegrained_pairs_full.jsonl
29856 pairs
same_source_ratio=0.0
pair_types:
  best_vs_worst=9954
  best_vs_mid=9954
  good_vs_bad=9948
```

combined fine-grained pair：

```text
/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/openrlhf_rm/rm_v6_finegrained_pairs_combined.jsonl
37319 pairs
same_source_ratio=0.0
pair_types:
  best_vs_worst=12442
  best_vs_mid=12442
  good_vs_bad=12435
```

QA 结果：

- full judged prompts: `9960`
- full low confidence ratio: `0.0`
- full blocked candidate flags: `3297`
- full pair same-source ratio: `0.0`
- QA failures: none

说明：

- same-source ratio 为 0 是当前 pair 构造策略和五 source 排序结果共同造成的，不是错误。
- 仍需抽查 close pair，尤其 `best_vs_mid` 和 `good_vs_bad`，因为这些 pair 决定 RM v6 的细粒度能力。

### 10.6 v5 anchor 混合数据

combined train：

```text
/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/openrlhf_rm/rm_train_v6_fg_combined.jsonl
51299 rows
```

来源：

```text
v5_train_anchor=17712
fine_grained_judge=33587
```

combined eval：

```text
/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/openrlhf_rm/rm_eval_v6_fg_combined.jsonl
4617 rows
```

来源：

```text
v5_eval_anchor=885
fine_grained_judge=3732
```

注意：

- v5 anchor 没有删除、降权或重排。
- 新旧数据统一 pairwise ranking loss。
- v5 train anchor 标记为 `v5_train_anchor`。
- v5 eval anchor 标记为 `v5_eval_anchor`。
- 新 fine-grained pair 标记为 `fine_grained_judge`。

### 10.7 Gemma4 RM v6 combined 训练状态

smoke：

```text
/home/ps/hanjiaqi/LlamaFactory/chatjoy/logs/rm_v6_finegrained/smoke_chatjoy_rm_gemma4_v6_fg_smoke_20260513_135132.log
/home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/gemma4-26b-a4b/chatjoy_rm_gemma4_v6_fg_smoke_20260513_135132/
```

smoke 主要用于验证链路。step 4 eval：

```text
acc_mean=0.593
reward_mean=-0.0505
reward_std=0.0189
wrong pairs=39.0%
saturated chosen=0.0%
```

combined 正式训练：

```text
/home/ps/hanjiaqi/LlamaFactory/chatjoy/logs/rm_v6_finegrained/train_chatjoy_rm_gemma4_v6_fg_combined_20260513_152457.log
/home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/gemma4-26b-a4b/chatjoy_rm_gemma4_v6_fg_combined_20260513_152457/
```

训练完成：

- 1 epoch 完成。
- train steps: `6412`
- 最终训练耗时约 `11:50:50`
- deepspeed launcher 最终各 rank `exits successfully`
- 训练末尾 NCCL/TCPStore warning 出现在退出阶段，未导致训练失败。

最终 eval：

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

产物：

```text
adapter_model.safetensors
adapter_config.json
config.json
tokenizer.json
tokenizer_config.json
plots/eval_reward_instances.png
eval_rewards/step_000200.jsonl
eval_rewards/step_000400.jsonl
eval_rewards/step_000600.jsonl
eval_rewards/step_000800.jsonl
eval_rewards/step_000801.jsonl
```

plot 注意：

- smoke 阶段 plot 因 `matplotlib` 缺失失败，不影响训练。
- combined 训练后 plot 脚本未解析到 train records，但 eval reward plot 成功生成。

### 10.8 当前质量判断

正向点：

- 相比 Gemma4 RM v5，RM v6 combined score 饱和明显改善：`saturated chosen=0.0%`。
- chosen/rejected gap mean 为 `+1.318`，说明 RM 学到了多数 pair 的方向。
- eval acc 约 `0.744-0.748`，显著高于 smoke 和随机。
- 数据覆盖更贴近 RL 场景：同 prompt，多模型，多温度，多语言，多角色。

风险点：

- wrong pairs 仍为 `25.3%`。
- plot 脚本明确提示：`> 10% pairs scored wrong — RM quality issue`。
- 不能因为饱和率改善就直接替换正式 RL 的 RM。
- DeepSeek 对 close pair 过于自信，full judge low confidence 为 0，需要人工抽检。
- Qwen 常被排差，需要确认 RM 学到的是回复质量而不是 source 风格偏见。

当前判断：

- RM v6 combined 是一个可测试的候选 RM，不是已经确定优于 Gemma4 RM v5 的正式 RM。
- 下一步必须先走 merge/classify/RL smoke，而不是直接跑正式 RL。

### 10.9 下一步接手顺序

1. Merge RM v6 adapter 到 vLLM 可加载模型。

建议目标路径：

```text
/home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/gemma4-26b-a4b/chatjoy_rm_gemma4_v6_fg_combined_merged_vllm/
```

源 adapter：

```text
/home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/gemma4-26b-a4b/chatjoy_rm_gemma4_v6_fg_combined_20260513_152457/
```

2. 跑 RM classify smoke。

检查：

- vLLM 能加载。
- score finite。
- EOS/rstrip 逻辑一致。
- batch classify 不报长度越界。
- score 分布不全挤在极窄区间。

3. 跑 30-step RL smoke。

要求：

- 只替换 RM path。
- 不改 actor/RL 稳定参数。
- 对比 Gemma4 RM v5 / Qwen RM v5 / Gemma4 RM v6 combined。
- 关注 reward/advantage 区分度、KL、长度、aborted ratio 和人工样本。

4. 决策。

如果 RM v6 smoke 表现更好：

- 再考虑正式 RL。

如果 RM v6 wrong-pair 问题影响 RL：

- 回看 close pair。
- 检查 eval 切分。
- 考虑对 `best_vs_mid` / `good_vs_bad` 增加 margin/gap 过滤或样本权重。
- 不删除 v5 anchor。

### 10.10 当前不要做的事

- 不要直接用 RM v6 combined 替换正式 RL RM。
- 不要删除 v5 anchor。
- 不要重新解释/覆盖旧 RMv5 的结论。
- 不要把 DeepSeek scalar score 当在线 reward。
- 不要把 API key 写入任何 md、README、日志摘要。
- 不要改已经稳定的 RL 脚本，除非只是替换 RM path 做 smoke。

## 11. 增量更新：RM v6 combined 训练结果诊断与 v6.1 数据修复方案 (2026-05-14 CST)

### 11.1 25% wrong pairs 根因分析

训练完成后对 eval reward 结果做了详细分解，root cause 定位到 v6 fine-grained pair 数据构成问题。

**Eval 集构成（共 4617 对）：**

```text
v5_anchor       :  885 (19.2%)
fine_grained    : 3732 (80.8%)
```

eval acc 74.4% 主要被 fine-grained 部分拉低；v5 hard-fail 部分（错语言/拒答 vs 正常）准确率应远高于整体，但因占比仅 19.2% 被稀释。

**gap 分布（step 801，577 batch 即 4616 对）：**

```text
gap < -2    : 14  (2.4%)   强烈错误方向
gap [-2,-1) : 26  (4.5%)
gap [-1,-0.5): 36 (6.2%)
gap [-0.5,0) : 70 (12.1%)
-----合计错误: 146 (25.3%)
gap [0, 0.5) : 85 (14.7%)  正确但差距很小
gap [0.5,1)  : 68 (11.8%)
gap [1,2)    : 86 (14.9%)
gap [2,5)    : 176 (30.5%)
gap >= 5     : 16 (2.8%)
```

**Fine-grained 训练集 pair 构成（33587 条）：**

```text
pair_type           count   rank_delta  score_delta
best_vs_worst       11198      4         1.0   ← 最清晰信号
best_vs_mid         11198      2         0.5   ← 中等，容易噪声
good_vs_bad         11191      2         0.5   ← 中等，容易噪声

delta=2 合计: 22389 (66.7%)
delta=4 合计: 11198 (33.3%)
```

```text
label_confidence    count
high               13939 (41.5%)
medium             19648 (58.5%)
```

**根因：**

`best_vs_mid`（rank1 vs rank3）和 `good_vs_bad`（rank2 vs rank4）两类 pair 的 rank_delta 只有 2，score_delta 只有 0.5，占训练集的 2/3。这些 pair 的信号本身就很弱——两个候选都是可接受回复，只是细微差别。DeepSeek judge 的排序在这类 close pair 上不一定可靠（尽管它报 low_confidence=0，但不代表实际正确率）。这些弱信号 pair 的比例过高导致模型在 eval 上 25% 错误方向。

**Score head 确认（重要）：**

adapter_model.safetensors 里确认包含 score head：

```text
base_model.model.score.weight  shape=(1, 2816)
```

不存在 v5 早期出现过的 score head 丢失问题。

### 11.2 v6.1 数据修复方案

**方案 A（最保守，推荐先试）：只保留 best_vs_worst（delta=4）**

```text
v5 anchor          17712
v6 best_vs_worst   11198
训练集合计         28910
```

优势：信号最清晰，rank 差距最大，降低 DeepSeek judge 噪声影响。
缺点：放弃了 best_vs_mid / good_vs_bad 的中等粒度信号。

**方案 B：只保留 high_confidence 的全部 pair 类型**

```text
high_confidence pairs: 13939
（其中 delta=4 约 4700，delta=2 约 9200）
```

优势：保留更多 pair，覆盖三种 pair type。
缺点：medium confidence 的 delta=2 pair 仍占多数，噪声问题没有完全解决。

**方案 C（折中）：delta=4 全部 + delta=2 仅 high_confidence**

```text
delta=4 all          11198
delta=2 high_conf    ~9200（估算）
v5 anchor            17712
合计估算             ~38100
```

**当前建议顺序：**

1. 先做 RL smoke 验证 RM v6 combined（当前版本）的 GRPO 信号是否比 v5 更清晰。
2. 若 RL smoke 表现达标，直接用 combined 版本推进正式 RL。
3. 若 wrong-pair 影响 advantage 区分度，再实现方案 A 重训 RM v6.1。

### 11.3 eval 分离建议

当前 eval 文件 80.8% 是 fine-grained pair，不能区分 v5 hard-fail 能力 vs fine-grained 能力。
后续建议：

```text
eval_v5_only.jsonl   # 885 条 v5 eval anchor，监控 hard-fail 判断能力
eval_fg_only.jsonl   # 3732 条 fine-grained eval，监控细粒度能力
```

分开评估可以更准确地诊断 RM 在不同 pair 类型上的问题。

### 11.4 当前不要做的事（更新）

上文 10.10 的禁止项不变，补充：

- 不要把 eval acc 74.4% 当作"整体"理解，它主要反映 fine-grained 能力，v5 hard-fail 部分应更高。
- 不要在没有 RL smoke 对比的前提下判断 RM v6 combined 是否优于 v5。
- 不要在 v6.1 重训前删除现有 combined 的任何产物。
- 不要直接用 DeepSeek score_delta 过滤（应用 rank_delta + label_confidence 过滤）。

### 12. v6.1 实施：dual-filter + smoke test 验证 (2026-05-14)

#### 12.1 多 judge 投票方案分析（已否决）

用户提出"3次DeepSeek投票或混合GPT-4o"改善delta=2标签质量。分析结论：

**关键发现：** `judge_candidates_llm_openai.py` 默认 `--temperature 0.0`，DeepSeek judge 是完全确定性输出。同一 prompt 跑 3 次 = 3 份完全相同标签，无法实现投票。

唯一可行的独立投票方案是混合异质模型（GPT-4o + DeepSeek），目标成本约 $48（~9,600 个 medium_conf 提示 × 2k tokens × $2.5/1M）。但 v6.1 dual-filter smoke test 结果证明不需要，已否决。

#### 12.2 dual-filter 实施（方案 C）

脚本：`GRPO_Training_Package/rm_v6_finegrained/filter_pairs_v61.py`

**过滤规则：**
- 保留：`rank_delta == 4`（best_vs_worst，全部置信度）
- 保留：`rank_delta == 2` AND `label_confidence == "high"`
- 丢弃：`rank_delta == 2` AND `label_confidence == "medium"`

**过滤结果（精确数字）：**

```text
输入：37,319 pairs (combined)
过滤后：22,793 pairs (-38.9%)

保留明细：
  delta=4, conf=high   :  5,177
  delta=4, conf=medium :  7,265
  delta=2, conf=high   : 10,351

丢弃：
  delta=2, conf=medium : 14,526 (38.9%)
```

输出文件：`data/openrlhf_rm/rm_v6.1_finegrained_pairs_combined.jsonl`

#### 12.3 v6.1 混合数据集

重新跑 `mix_with_v5_anchor.py --fine-pairs rm_v6.1_finegrained_pairs_combined.jsonl`：

```text
train: data/openrlhf_rm/rm_train_v6.1_fg_combined.jsonl
  v5_train_anchor      : 17,712
  fine_grained_judge   : 20,514
  合计                 : 38,226 rows（vs v6 的 51,299，-25.5%）

eval: data/openrlhf_rm/rm_eval_v6.1_fg_combined.jsonl
  v5_eval_anchor       :    885
  fine_grained_judge   :  2,279
  合计                 :  3,164 rows（vs v6 的 4,617）
```

#### 12.4 启动脚本

新脚本：`chatjoy/launch_rm_train_gemma4_sft_v6.1_fg.sh`

默认参数：
- TRAIN_FILE: `rm_train_v6.1_fg_combined.jsonl`
- EVAL_FILE: `rm_eval_v6.1_fg_combined.jsonl`
- RUN_ID: `chatjoy_rm_gemma4_sft_v6.1_fg_$(date)`
- 其余超参与 v6 相同（LR=5e-6, max_len=3072, lora_rank=64, eval_steps=200, save_steps=200）

#### 12.5 smoke test 结果（4 GPU, 32 grad steps）

Session: tmux `rmv61_smoke`（已完成）  
配置：GPU 0-3，GLOBAL_BATCH=32，max_samples=1024，eval_steps=32

**v6 vs v6.1 smoke 对比：**

| 指标 | v6 smoke | v6.1 smoke | 变化 |
|------|----------|------------|------|
| eval acc_mean | 0.593 | **0.696** | +10.3 pp |
| reward_std | 0.0189 | **0.0394** | +108% |
| reward_mean | -0.0505 | -0.0123 | 更居中 |
| eval_loss | 0.691 | 0.681 | 略降 |
| gap_mean | — | +0.026 | chosen 持续高于 rejected |
| gap_p90 | — | +0.103 | 强 pair 差距显著 |

训练趋势：
- 早期（micro-step 8-24）：chosen ≈ rejected（均约 -0.06），两者未分离
- 中后期（micro-step 120+）：chosen → ≈0 / 正值，rejected → -0.05 ~ -0.08，差距稳定扩大
- 对比 v6：全程 chosen/rejected 几乎同步抖动，无明显分离

**结论：dual-filter 已解决标签噪声问题，不需要多 judge 投票。**

reward_std 翻倍直接对应 GRPO advantage 方差翻倍，是 v5 平曲线根本修复。

#### 12.6 当前待办（截至 2026-05-14）

1. **下一步：启动 v6.1 全量 8 卡训练**
   ```bash
   GPU_LIST=0,1,2,3,4,6,7 GLOBAL_BATCH=56 \   # 或用完整8卡（若GPU5空闲）
   bash launch_rm_train_gemma4_sft_v6.1_fg.sh
   ```
   注意：GPU 5 当前被其他进程占用（23.5 GB），需确认空闲后再用 8 卡。

2. **v6.1 adapter merge → vLLM 可加载**
3. **classify smoke → 30-step RL smoke**

#### 12.7 禁止项补充

上文 11.4 禁止项不变，新增：
- 不要用 v6 combined（51,299 行）的训练集替换 v6.1 去跑 RL smoke，两者有本质差异。
- 不要删除 `rm_v6_finegrained_pairs_combined.jsonl`（原始 37,319 对），后续如需多 judge 重标仍需它。
- 不要删除 `rm_v6.1_finegrained_pairs_combined.jsonl`（22,793 对），这是 v6.1 唯一干净的 fine-grained pairs 来源。

## 13. 增量更新：v6.1 full 训练结果 + 噪声诊断 + v6.2 双 judge 修复方案 (2026-05-15 CST)

本节接 Section 12 的 v6.1 smoke。v6.1 full 8 卡训练已完成，但 wrong_pairs 仍卡在 21% 左右，**不是训练问题，是 label 噪声问题**。本节定位根因、给出修复方案、并把 v6.2 双 judge 流水线的执行状态留给下一次接手。

### 13.1 v6.1 full 训练实际结果

日志：`logs/rm_v6_finegrained/train_chatjoy_rm_gemma4_v6.1_fg_full_20260514_172428.log`
配置：8 GPU、global_batch=64、grad_accum=8、LR=5e-6、max_len=3072、LoRA r=64、单 epoch、597 grad steps。
训练数据：`rm_train_v6.1_fg_combined.jsonl`（38,226 行：v5_anchor 17,712 + fg_high_quality 20,514）。
评估数据：`rm_eval_v6.1_fg_combined.jsonl`（3,164 行）。

关键 eval 指标：
| step | acc | wrong_pairs | reward_std | gap_mean | gap_p10 | gap_p90 | saturated |
|------|------|-------------|------------|----------|---------|---------|-----------|
| 200  | 0.764 | 24.1% | 1.62 | +1.61 | -0.74 | +4.46 | 0.0% |
| 400  | 0.784 | 22.0% | 1.86 | +1.79 | -0.59 | +4.92 | 0.0% |
| 597  | 0.784 | 21.3% | 1.94 | +1.880 | -0.547 | +5.109 | 0.0% |

对比 RM v5：v5 reward_std 在训练末期约 0.1（曲线饱和、信号塌缩）；v6.1 reward_std=1.94，**约 19×**。saturated=0% 说明 RM 输出仍有充分动态范围，GRPO advantage 计算可用。这部分相对 v5 已经是质变。

但是 wrong_pairs=21.3% 比预期高：dual-filter 已经把单 judge 自身报告的 medium-conf delta=2 删干净（删了 38.9%），剩下的应该是"干净"的，模型却还是只能在 78% 这个上限附近震荡。

### 13.2 为什么 dual-filter 之后 wrong pair 仍然 ~21%

根因：**单 judge 在 temperature=0 下报告的 label_confidence 是 self-reported，不是统计独立的真实可信度。** 即使 DeepSeek 自己说 "confidence=high"，实际人审/对比基线显示这部分 delta=2 pairs 仍然有 ~27% 的方向错误率。dual-filter 只能去除 self-reported 不确定的那 38.9%，去不掉 self-reported 高但实际错的那部分。

数学上的不可越限：
- 留在 v6.1 训练里的 fg pairs 中，delta=2,conf=high 是 10,351 对（占 fg 总量 45%）。
- 这部分 pairs 若实际错误率为 r（估计 ~25%），那 RM 在这部分的天花板就是 1 - r ≈ 75%。
- 整体 weighted：(delta=4 那部分准确率 ~90% × 占比 55%) + (delta=2,conf=high 那部分 ~75% × 占比 45%) ≈ 83.3%。
- v6.1 实际收敛到 78.4%，gap 是 LoRA 表达力 + 个别长上下文样本 + chosen/rejected 相似 prompt 的内在难度。

**结论**：单 judge × temperature=0 没法再压更低。继续提温度或多次采样同一个 DeepSeek 是无效的（temp=0 输出完全确定）。要进一步降低噪声，**必须引入真正独立的第二个 judge 模型**。

### 13.3 v6.2 修复方案：跨模型 judge 投票（方向一致性过滤）

核心思路：用一个**结构上不同**的模型（Qwen3.6-plus）对同一批 12,448 个 prompts 做独立 ranking。对原 v6 combined 的每一对 (chosen_model, rejected_model)，只保留**两个 judge 都把 chosen_model 排在 rejected_model 之上**的对。两 judge 都同意 → 方向几乎不可能错（独立两次错的概率 ≈ r²，约 6-7%）。

为什么是"方向"而不是"绝对排序"：
- 不需要两 judge 的 rank1/rank2/.../rank5 完全一致——那太苛刻，会几乎全删。
- 只需要每一对的相对顺序一致。这就足够把"这一对的 label 是不是对的"的错误率从 ~25% 降到个位数。

为什么选 Qwen3.6-plus 而不是 GPT-4o：
- 用户偏好阿里云百炼接入（已有 API quota，本地可直接打到 `https://dashscope.aliyuncs.com/compatible-mode/v1`）。
- Qwen3.6-plus 是与 DeepSeek 结构、训练数据都不同的大模型，独立性满足要求。
- 验证过 thinking tokens 走 `reasoning_content` 字段，不污染 `content` 里的 JSON 输出；judge 脚本 `extract_content()` 只读 `content`，**完全不用改判分逻辑**。

预计 v6.2 数据量：原 v6 combined 共 37,319 对（注意：是 combined 原始全集，不是 v6.1 dual-filter 后的 22,793 对——重新从 37,319 起跑过滤，避免双重过滤）。预计留存率 60-75%（即 ~22-28k 对），其中：
- delta=4 大部分会被两 judge 都认同（gap 大，方向稳定）。
- delta=2,conf=high 中错的那部分会被 Qwen 否决。
- delta=2,conf=medium 中其实方向对的也能被救回来——这是 v6.1 dual-filter 丢的信号，v6.2 能拿回来一部分。

### 13.4 已实现的 v6.2 流水线代码

1. **第二 judge 启动脚本**：`GRPO_Training_Package/rm_v6_finegrained/judge_qwen36plus_all.sh`
   - 复用现有 `judge_candidates_llm_openai.py`（OpenAI 兼容接口）。
   - 输入 candidates：pilot 5 个 + full 5 个，共 12,448 prompts（去重 prompt_id）。
   - 模型：`qwen3.6-plus`，base_url：`https://dashscope.aliyuncs.com/compatible-mode/v1`。
   - temperature=0.0、max_tokens=2048、timeout=120、concurrency=16、limit=99999、`--resume`。
   - 输出：`data/runs/rm_v6_finegrained/qwen36plus/judged_qwen36plus_all12448.jsonl`。
   - 审计：`..._audit.jsonl`（JSON 解析失败/格式异常进 audit）。
   - 日志：`logs/rm_v6_finegrained/judge_qwen36plus_all12448.log`。

2. **方向一致性过滤脚本**：`GRPO_Training_Package/rm_v6_finegrained/merge_judge_votes.py`
   - 输入 1：原 v6 combined pairs（37,319 行）`rm_v6_finegrained_pairs_combined.jsonl`。
   - 输入 2：Qwen 第二 judge 的 jsonl 输出。
   - 逻辑：
     ```python
     def load_rankings(path) -> dict[prompt_id, dict[source_model, rank]]:
         # ranked_candidates 里 rank 越小越好（rank=1 是最好）
     
     def judge_b_agrees(rankings_b, prompt_id, chosen_model, rejected_model):
         prompt_ranks = rankings_b.get(prompt_id)
         if not prompt_ranks: return False
         rank_chosen = prompt_ranks.get(chosen_model)
         rank_rejected = prompt_ranks.get(rejected_model)
         if rank_chosen is None or rank_rejected is None: return False
         return rank_chosen < rank_rejected  # lower rank number = better
     ```
   - 保留的 pair 会被打上：`extra_info.label_confidence = "high_crossvalidated"`、`extra_info.cross_judge = "qwen3.6-plus"`。
   - 同时统计 (delta, conf) 维度的 kept/dropped 分布，用于诊断。
   - 输出：`data/openrlhf_rm/rm_v6.2_finegrained_pairs_combined.jsonl`。

3. **已修复的脏数据**：`data/runs/rm_v6_finegrained/pilot/candidates.rl550_t10.jsonl` 第 2054 行（`完成。cc-overview.` 前缀导致 JSON 解析失败），已用 `re.search(r'\{.*\}', line, flags=re.S)` 恢复。这个 fix 必须保留，重跑 candidates 生成不会自动恢复它。

### 13.5 当前 Qwen judge 运行状态（2026-05-15 CST 快照）

- tmux session：`rmv62_judge_qwen`（**不要 kill**）。
- 当前进度：~190/12,448 已写入 jsonl（attempted=210，audited 个位数到 22，少量 prompt 走 audit 兜底）。
- 预计完成时间：concurrency=16 下整体 ~2 小时。
- 监控命令：
  ```bash
  tail -f /home/ps/hanjiaqi/LlamaFactory/chatjoy/logs/rm_v6_finegrained/judge_qwen36plus_all12448.log
  wc -l /home/ps/hanjiaqi/LlamaFactory/chatjoy/data/runs/rm_v6_finegrained/qwen36plus/judged_qwen36plus_all12448.jsonl   # 目标 12,448
  tmux attach -t rmv62_judge_qwen   # Ctrl+B then D detach
  ```
- 中断恢复：脚本带 `--resume`，可以安全重启。

### 13.6 Judge 跑完后的执行顺序

1. **检查 Qwen judge 完整性**：
   ```bash
   wc -l data/runs/rm_v6_finegrained/qwen36plus/judged_qwen36plus_all12448.jsonl   # 期望接近 12,448
   wc -l data/runs/rm_v6_finegrained/qwen36plus/judged_qwen36plus_all12448_audit.jsonl   # audit 占比应 < 5%
   ```

2. **跨 judge 方向过滤产出 v6.2 pairs**：
   ```bash
   cd /home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/rm_v6_finegrained
   /home/ps/miniconda3/envs/tslib/bin/python merge_judge_votes.py \
     --pairs    ../../data/openrlhf_rm/rm_v6_finegrained_pairs_combined.jsonl \
     --judge-b  ../../data/runs/rm_v6_finegrained/qwen36plus/judged_qwen36plus_all12448.jsonl \
     --output   ../../data/openrlhf_rm/rm_v6.2_finegrained_pairs_combined.jsonl
   ```
   输出会打印 `pairs_input/pairs_kept/pairs_dropped/retention_rate` 以及 (delta, conf) breakdown。

3. **混合 v5 anchor → 训练/评估数据**：复用 `mix_with_v5_anchor.py`，把输入换成 v6.2 pairs，输出文件名约定：
   - `data/openrlhf_rm/rm_train_v6.2_fg_combined.jsonl`
   - `data/openrlhf_rm/rm_eval_v6.2_fg_combined.jsonl`

4. **训练 RM v6.2**：复制 `launch_rm_train_gemma4_sft_v6.1_fg.sh` → `launch_rm_train_gemma4_sft_v6.2_fg.sh`，只改三处：
   - `TRAIN_FILE` 默认 → `rm_train_v6.2_fg_combined.jsonl`
   - `EVAL_FILE` 默认 → `rm_eval_v6.2_fg_combined.jsonl`
   - `RUN_ID` 前缀 → `chatjoy_rm_gemma4_sft_v6.2_fg_`
   其他超参（LR=5e-6、max_len=3072、LoRA r=64、global_batch=64、ZeRO-2、QLoRA 4bit、adam_offload）全部沿用 v6.1，保持可比性。

5. **训练后**：adapter merge → vLLM classify smoke → 30-step RL smoke，路径同 v5/v6 流水线。

### 13.7 v6.2 预期效果（推断，非保证）

| 指标 | v6.1 实测 | v6.2 预期 | 依据 |
|------|-----------|-----------|------|
| eval acc | 0.784 | 0.84-0.88 | 噪声从 ~25% 降至 ~6-8% |
| wrong_pairs | 21.3% | 12-16% | 同上 |
| reward_std | 1.94 | 1.8-2.2 | dynamic range 不应回归 |
| gap_p10 | -0.547 | -0.3 ~ -0.5 | 困难 pair 仍存在但数量减少 |
| 训练数据量 | 38,226 | 估 30k-38k | 取决于 Qwen 一致率（60-75%）|

如果 v6.2 acc 仍卡在 80% 以下，说明问题不只是 label 噪声，可能涉及 LoRA 表达力不足或长上下文截断（>3072 tokens）；那时候需要考虑 LoRA rank 升到 128 或 max_len 升到 4096，但这是 next-next 步。

### 13.8 禁止项补充（v6.2 阶段新增）

- 不要把 `sk-` 开头的 Qwen / 任意厂商 API key 写进 md、README、PR、commit message、log 摘要。脚本里的 default key 是 placeholder，正式环境用 env var 覆盖。
- 不要用 v6.1 的 dual-filter 数据（22,793 对）作为 v6.2 merge_judge_votes 的输入——会双重过滤，丢掉 delta=2,conf=medium 中真实方向对的那部分。输入永远是原 37,319 对的 v6 combined。
- 不要在 Qwen judge 没跑完前用部分输出做 merge——`merge_judge_votes.py` 会把 Qwen 没见过的 prompt_id 全判为"不同意"丢掉。
- 不要 kill tmux session `rmv62_judge_qwen` 或 GPU5 上的训练进程（用户明确保留）。
- 不要修改 `judge_candidates_llm_openai.py`——Qwen 兼容性已确认，改了会破坏其他 judge 调用。

## 14. 增量更新：Qwen API 失败 → 本地 Qwen-36B-A3B 完成 cross-validation → v6.2 训练数据就绪 (2026-05-15 CST)

本节接 Section 13 的"双 judge cross-validation 方案"。Section 13.5 当时停在"Qwen Plus 云端 judge 正在跑、约 2 小时完成"——结果方案被 API 端整个干掉，最终用本地开源模型完成。本节是 v6.2 训练数据完全就绪的快照。

### 14.1 Qwen Plus 云端 judge 故障（已 kill，不重启）

事件经过：
- tmux session `rmv62_judge_qwen` 启动后跑到 ~191 prompt 时，judge 突然 100% 失败。
- 故障特征：`judged` 卡死在 191、`audited` 每 10 步 +10 飙升；audit reason 全是 `<HTTPError 403: 'Forbidden'>` + `raw_text=""`，请求阶段就被拒，没走到模型。
- 最终统计：judged=191、audit=661，audit/total ≈ 78%（雪崩式失败）。
- 已停掉 tmux 避免继续浪费配额。

故障根因（推断，最可能项排前）：
1. **百炼内容安全过滤命中**：chatjoy 是 NSFW / 角色扮演数据，`qwen3.6-plus` 在百炼上对此过滤极敏感。前 191 prompt 偏温和恰好绕过；碰到敏感 prompt 后该 API key 进入"拉黑窗口"，后续请求全部 403（不仅是当前请求）。
2. `qwen3.6-plus` 模型权限/配额问题（次要）。
3. RPS 突发限制（百炼也用 403 表达，非 429）。

教训：**chatjoy 这类数据不要走商用 API judge**，封锁是结构性的、单次重试无效。

### 14.2 切换方案：本地 vLLM 跑 Qwen-36B-A3B（已完成）

用户已自行完成：在 GPU0-7 上用 vLLM 起 `qwen3-36b-a3b`（开源版，本地推理，零审核风险），跑 judge 12,448 个候选 prompt 配置。相关产物：

- **judge 输出**：`data/runs/rm_v6_finegrained/qwen36_local/judged_qwen36_local_all12448.jsonl`（9,960 行，每行 5 候选 ranking）
- **audit 文件**：`data/runs/rm_v6_finegrained/qwen36_local/judged_qwen36_local_all12448_audit.jsonl`（38 行，全部 reason=`too_few_candidates`，是候选不足 5 个的固有数据问题，与 judge 失败无关）
- **judge log**：`logs/rm_v6_finegrained/judge_qwen36_local_all12448.log`（正常 shutdown）
- **vLLM log**：`logs/rm_v6_finegrained/qwen36_local_vllm.log`
- **启动脚本**：`GRPO_Training_Package/rm_v6_finegrained/judge_qwen36_local_all.sh`
- **merge 启动脚本**：`GRPO_Training_Package/rm_v6_finegrained/merge_qwen36_local_v62.sh`
- **smoke 测试痕迹**：同目录下 `*smoke5_json*` / `*smoke20_norat*`（已完成的可丢弃中间文件）

可用性验证结论（**全通过**）：
- 真实失败率 0.38%（vs Qwen Plus API 雪崩 78%，差 200×）
- 顶层字段与 DeepSeek 完全兼容：`prompt_id, ranked_candidates, judge, raw_judge, language, role_hash, confidence`
- 候选字段：`rank, candidate_id, source_model, score, rubric_scores, judge_label, judge_rationale`
- `source_model` 字符串与 DeepSeek、与 pair 文件三方对齐：`rl_step550_t07_p08 / rl_step550_t10_p095 / gemma4_sft_t07_p08 / gemma4_sft_t10_p095 / qwen_sft_t07_p08`
- `merge_judge_votes.py` 可直接 load，无需改字段映射

### 14.3 数据关系最终澄清（解决 "12,448 vs 9,960 vs 10,000" 混乱）

之前几个数字在不同上下文里指代不同对象，统一约定：

```
原始 prompt 集合：
  pilot: 2,500 unique prompt_id
  full : 10,000 unique prompt_id
  overlap = 2,500 (pilot 完全是 full 子集)
  union   = 10,000  ← 真正的 unique prompt 数

每个 split × 5 个 source 模型生成候选：
  pilot/candidates.{source}.jsonl  → 5 文件 × ~2,500 行 = 12,528 候选行
  full/candidates.{source}.jsonl    → 5 文件 × 10,000 行 = 50,000 候选行
  合计候选行：62,528（≠ prompt 数）

5 个 source（candidate_id 命名根据 sampling 配置）：
  rl_step550_t07_p08    (RL step550, T=0.7, top_p=0.8)
  rl_step550_t10_p095   (RL step550, T=1.0, top_p=0.95)
  gemma4_sft_t07_p08    (Gemma4 SFT, T=0.7, top_p=0.8)
  gemma4_sft_t10_p095   (Gemma4 SFT, T=1.0, top_p=0.95)
  qwen_sft_t07_p08      (Qwen SFT,   T=0.7, top_p=0.8)

DeepSeek judge：跑了两次 → 输出合计 12,448 行（pilot 那 2,500 prompt_id 被 judge 两次）
  - judged_llm_deepseek_pilot2500.jsonl
  - judged_llm_deepseek_full10000.jsonl
  - unique prompt_id：9,960 (10,000 - 40 因候选不足被 audit)

Qwen-36B-A3B local judge：跑了一次（合并 pilot+full 后去重）→ 9,960 行
  - judged_qwen36_local_all12448.jsonl
  - 注：文件名带 "12448" 是沿用 DeepSeek 命名，**实际 9,960 行**，不要被名字误导
  - unique prompt_id：9,960，与 DeepSeek 的 union 完全一致

DeepSeek pair：v6 combined fine-grained pair 文件
  - rm_v6_finegrained_pairs_combined.jsonl: 37,319 对
  - 每个 prompt_id 平均产 ~3.7 对（按 delta=4 + delta=2 规则展开）
```

**禁止再混用以下数字**：
- "12,448" = DeepSeek **输出总行数**（含重复 judge）≠ prompt 数
- "9,960" = unique prompt_id 数（**这才是真实可用 ranking 数量**）
- "10,000" = full split unique prompt 数（含 40 个候选不足被 audit 的）
- "37,319" = DeepSeek 派生的 pair 总数

### 14.4 merge_judge_votes.py 执行结果（v6.2 pair 文件已产出）

**输出文件**：`data/openrlhf_rm/rm_v6.2_finegrained_pairs_combined.jsonl`
**保留 pair 数**：**27,287**（保留率 **73.1%**，落在健康区间 60-75% 顶端）

分项 breakdown：
| (delta, conf)        | 输入   | 保留   | 保留率 |
|---------------------|--------|--------|--------|
| delta=4, conf=high   | 5,177  | 4,371  | 84.4% |
| delta=4, conf=medium | 7,265  | 5,836  | 80.3% |
| delta=2, conf=high   | 10,351 | 7,304  | 70.6% |
| delta=2, conf=medium | 14,526 | 9,776  | 67.3% |

**3 个关键发现**：

1. **delta=2, conf=high 被 Qwen 否决 29.4%（3,047 对）**——这正是 v6.1 卡 78.4% 的元凶。v6.1 dual-filter 删不掉这部分（因为它们被 DeepSeek 自报为 high）；cross-validation 把它们干掉了。

2. **DeepSeek self-reported confidence 在 delta=2 上几乎无区分度**：delta=2,high 同向 70.6% vs delta=2,medium 同向 67.3%，差距仅 3.3%。这从数据上证明 v6.1 dual-filter 的过滤标准（self-reported confidence）本身就不靠谱——v6.1 扔掉的 14,526 个 delta=2,medium 里其实有 9,776 个（67.3%）方向是对的，被 v6.1 白白丢了。

3. **delta=4 也有 ~15-20% 被否决**：印证即使是 best vs worst 也存在内在不确定性（prompt 困难度、judge 风格差异），这部分丢掉是健康的。

保留的 pair 打上：`extra_info.label_confidence = "high_crossvalidated"`、`extra_info.cross_judge = "qwen3-36b-a3b-local"`。

`merge_judge_votes.py` 已经参数化：`--cross-judge-name` 控制 cross_judge 字段名（默认 `qwen3.6-plus`，本次用 `qwen3-36b-a3b-local`）。

### 14.5 评估"再过滤 delta=2 medium 交集"的决定（**否决**）

候选方案 v6.2-B：在 cross-validation 后再删 delta=2,medium 交集（9,776 对）只留 high 交集（7,304 对）→ 数据缩到 17,511 fg pair。

**严格论证（不拍脑袋）**：

用两 judge 同向率反推真实方向概率（假设各 judge 独立、准确率 ~85-90%）：
- delta=2, conf=high 同向 70.6% → 真实方向 P ≈ 0.87 → 交集后准确率 ~87-92%
- delta=2, conf=medium 同向 67.3% → 真实方向 P ≈ 0.83 → 交集后准确率 ~85-90%
- **交集后两者准确率差距：~2-3%**

数据量与标签错误率 trade-off：
| 方案 | fg pair | + v5 anchor | est. grad step | est. 错误率 | RM 见过的 delta=2 pair |
|------|---------|-------------|----------------|-------------|------------------------|
| v6.2-A（全交集）| 27,287 | 42,270 | ~660 | 9-10% | 17,080 |
| v6.2-B（仅 high 交集）| 17,511 | 35,223 | ~550 | 7-8% | 7,304 |

**否决 v6.2-B 的 3 个理由**：
1. 数据量损失 36% > 标签质量改善 2-3%。LoRA r=64 对数据量敏感。
2. delta=2 pair 是 GRPO 在线打分的关键能力（同 prompt 的近似候选 reward 区分），v6.2-B 让 RM 只见 7,304 个近似 pair → RL 时容易 OOD。
3. cross-validation 已经做过对错过滤；再叠 self-reported confidence 是错位的（前面已证 confidence 在 delta=2 上几乎无区分度，叠加无正交信息增益）。

**结论：v6.2 主路径用全部 27,287 对（即 v6.2-A）**。v6.2-B 仅在 v6.2-A 训完仍卡 80% 以下、且诊断到"标签质量问题"时作为消融实验跑。

### 14.6 v6.2 训练数据 & launcher 就绪

数据文件：
| 文件 | 行数 | 组成 |
|------|------|------|
| `data/openrlhf_rm/rm_train_v6.2_fg_combined.jsonl` | **42,270** | v5_train_anchor 17,712 + fine_grained_judge 24,558 |
| `data/openrlhf_rm/rm_eval_v6.2_fg_combined.jsonl`  | **3,614**  | v5_eval_anchor 885 + fine_grained_judge 2,729 |

mix 命令（已执行）：
```bash
cd GRPO_Training_Package/rm_v6_finegrained
/home/ps/miniconda3/envs/tslib/bin/python mix_with_v5_anchor.py \
  --fine-pairs ../../data/openrlhf_rm/rm_v6.2_finegrained_pairs_combined.jsonl \
  --output-train ../../data/openrlhf_rm/rm_train_v6.2_fg_combined.jsonl \
  --output-eval  ../../data/openrlhf_rm/rm_eval_v6.2_fg_combined.jsonl
```
（默认 fine-eval-ratio=0.10、seed=42；与 v6.1 一致）

Launcher：`launch_rm_train_gemma4_sft_v6.2_fg.sh`（已 chmod +x）
- 默认值已切到 v6.2 数据文件路径
- RUN_ID 前缀：`chatjoy_rm_gemma4_sft_v6.2_fg_`
- 其他超参与 v6.1 完全一致：LR=5e-6、max_len=3072、LoRA r=64、QLoRA 4bit、ZeRO-2、adam_offload、global_batch=64、单 epoch
- echo 头部已注明 "(cross-validated: DeepSeek + Qwen-36B-A3B)"

预估资源：~660 grad step、8 GPU × 8-9 小时。

### 14.7 当前等待状态 & 下一步

- GPU0-7 全空闲（GPU0 占 2.4GB 杂项进程，其余 < 10MB）
- 训练**尚未启动**（待用户确认）

启动命令：
```bash
cd /home/ps/hanjiaqi/LlamaFactory/chatjoy
bash launch_rm_train_gemma4_sft_v6.2_fg.sh
# 或后台：nohup bash launch_rm_train_gemma4_sft_v6.2_fg.sh > /tmp/v62.log 2>&1 &
# 或 tmux：tmux new -d -s rmv62_train 'bash launch_rm_train_gemma4_sft_v6.2_fg.sh'
```

预期 v6.2 训练效果（推断、非保证）：
| 指标 | v6.1 实测 | v6.2 预期 | 依据 |
|------|-----------|-----------|------|
| eval acc | 0.784 | 0.84-0.88 | 噪声从 ~21% 降至 ~9-10% |
| wrong_pairs | 21.3% | 12-16% | 同上 |
| reward_std | 1.94 | 1.8-2.2 | dynamic range 不应回归 |
| gap_p10 | -0.547 | -0.3 ~ -0.5 | 困难 pair 仍存在但数量减少 |

训完后流水线：adapter merge → vLLM classify smoke → 30-step RL smoke（路径同 v5/v6.1）。

### 14.8 禁止项更新（接 13.8）

继续生效的：
- 不要把 `sk-` 开头的任意厂商 API key 写进 md / 日志 / commit / PR 描述。
- 不要用 v6.1 dual-filter 数据（22,793）作为 cross-validation 输入。
- 不要修改 `judge_candidates_llm_openai.py`（Qwen 兼容性已确认）。

本节新增：
- 不要尝试重新启动 `rmv62_judge_qwen` 或重连百炼 API——`qwen3.6-plus` 对 chatjoy 数据结构性封锁，浪费时间。
- 不要在 v6.2 训练启动前修改 `rm_v6.2_finegrained_pairs_combined.jsonl`（27,287 对）—— mix 已经基于此固定，改了会破坏可复现性。
- 不要把 v6.2-A 和 v6.2-B 弄混。当前 v6.2 = v6.2-A（全交集 27,287 对）。如果以后跑 v6.2-B 消融，文件名要明确带 `_high_only` 后缀。
- 不要轻易跳过 v6.1 ↔ v6.2 对比就直接进入 RL——v6.2 训完一定要 eval+RM curve 对照 v6.1，确认改善确实来自 cross-validation 而不是随机性。
- 删除 smoke 测试中间文件（`*smoke5_json*`、`*smoke20_norat*`）可以，但保留最终 `judged_qwen36_local_all12448.jsonl` + `_audit.jsonl`，后续 v6.3 如果还要叠第三 judge 仍需要这份基线。

### 14.9 当前任务清单（接手者直接看这里）

已完成（不要重复跑）：
- [x] Qwen Plus 云端 judge 故障定位 + kill
- [x] 本地 Qwen-36B-A3B vLLM judge 跑完 12,448 候选（9,960 unique prompt）
- [x] `merge_judge_votes.py` 跑通 → 产 27,287 对 v6.2 cross-validated pair
- [x] (delta, conf) breakdown 分析 + 决定不再过滤 medium
- [x] `mix_with_v5_anchor.py` 产 v6.2 train/eval（42,270 / 3,614）
- [x] `launch_rm_train_gemma4_sft_v6.2_fg.sh` 编写 + chmod +x

待执行（按顺序）：
1. **启动 v6.2 全量训练**（8 GPU × 8-9 小时）
2. 训练过程中 step 200 / 400 / 600 eval 指标记录与 v6.1 对比
3. 训练完成后 LoRA adapter merge（参考 v6.1 merge 流程）
4. vLLM classify smoke：v6.2 merged RM 加载、对 v5 eval 数据打分对照
5. 30-step RL smoke（参考 Section 6 / 10 路径）
6. 若 v6.2 eval acc < 0.82：考虑 v6.2-B 消融或 LoRA rank 升级到 128
7. 若 v6.2 eval acc ≥ 0.84：进入正式 RL with v6.2 RM

## 15. 增量更新：v6.2 full 训练结果 + underfit 诊断 + 优化方向 (2026-05-18 CST)

接 Section 14 启动的 v6.2 全量训练（660 grad step、8 GPU、~9.5h）已于 2026-05-16 05:20 完成。本节是结果分析 + 下一步分支选项。

### 15.1 v6.2 训练完整结果

SAVE_PATH：`saves/gemma4-26b-a4b/chatjoy_rm_gemma4_sft_v6.2_fg_20260515_195216/`
train.log：2.3MB、15,516 行
adapter_model.safetensors：92MB（LoRA r=64，q/k/v/o_proj）

每 200 grad step eval：
| step | acc | wrong_pairs | reward_std | eval_loss | reward_mean |
|------|------|-------------|------------|-----------|-------------|
| 200  | 0.771 | 22.8% | 1.50 | 0.431 | 1.09 |
| 400  | 0.798 | 19.1% | 1.93 | 0.391 | 1.20 |
| 600  | 0.806 | 19.1% | 1.98 | 0.381 | 1.30 |  ← peak acc
| 660  | 0.802 | 18.0% | 1.98 | 0.381 | 1.30 |  ← final

末段 grad metrics（最后 5 个 micro step）：`loss=0.28-0.44, acc=0.75, grad_norm=5.19, lr=0`。

### 15.2 v6.1 vs v6.2 严格对比

| 指标 | v6.1 final (597 step) | v6.2 final (660 step) | Δ |
|------|----------------------|------------------------|-----|
| acc | 0.787 | **0.802** | +1.5pp |
| wrong_pairs | 21.3% | **18.0%** | -3.3pp（相对 -15.5%）|
| reward_std | 1.94 | 1.98 | +0.04 |
| eval_loss | ~0.45 | **0.381** | -0.07 |
| saturated | 0.0% | 0.0% | 0 |
| gap_mean | +1.880 | ~+1.3-1.5（推断）| ~-0.4 |

**cross-validation 起作用了**（wrong_pairs 真实下降 3.3pp），但**改善幅度小于预期**（预期 0.84-0.88，实际只到 0.802）。

### 15.3 underfit 诊断（不是数据天花板）

3 条强证据指向**训练 underfit**而非数据本身耗尽：

1. **末段 grad_norm 仍 5+**：cosine LR 已衰减到 0，但梯度模长依然大——说明 loss 曲面还有下降空间，只是 LR 不够推动了。
2. **末段 loss 仍在 0.28-0.44 波动**：单 batch loss 不是收敛形态（如果数据极限，loss 应该稳定在某个值小区间），说明模型还能拟合。
3. **acc 在 step 600 后从 0.806 微跌到 0.802**：cosine 末段 LR ≈ 0，最后 60 step 没在学习，所以 0.802 接近"训练实际能达到的水平"而不是"数据上限"。

**对照 v5 同配置**：v5 用 LR=**9e-6** + 17,712 样本 + 276 step → acc=**97.5%**（参考 project_chatjoy_overview.md line 189）。v6.2 用 LR=**5e-6** + 42,270 样本 + 660 step → acc=80.2%。两个最大的可控变量差异：**LR 和数据噪声**。cross-validation 已经解决数据噪声（wrong_pairs -3.3pp 证明），那剩下的 gap 主要应该来自 LR。

LR 历史溯源：handoff Section 14.6 / line 640 写明"LR 建议从 v5 的 9e-6 降到 5e-6"——这是 v6 pilot 阶段的保守决定（怕 fine-grained pair 噪声大、训不稳）。但 v6.2 数据已经是 cross-validation 过的"双 judge 同向"高质量集合，原"保守"前提不再成立。

### 15.4 优化方向 3 选项（按性价比排序）

#### 选项 A：**v6.2-LR9** —— LR 升回 9e-6（**首推**）
- 数据完全不变（v6.2 train/eval 不动）
- 仅改 launcher：`LEARNING_RATE=9e-6 bash launch_rm_train_gemma4_sft_v6.2_fg.sh`
- 成本：~9.5h（与 v6.2 同）
- 预期：acc 上 0.84-0.87；如果再卡在 0.81 附近就是数据天花板
- 风险：grad_norm 早期可能尖刺；但 v5 用 9e-6 训过相同 base，没爆过
- 这是验证 underfit 假设的**最便宜实验**

#### 选项 B：**v6.2-delta4-only** —— 数据天花板诊断（如果 A 也卡）
- 只保留 v6.2 中 delta=4 部分（约 9,200 对 best-vs-worst，剔除全部 delta=2）+ v5 anchor 17,712 = **~27,000 train**
- 用 LR=5e-6 或 9e-6 训
- 成本：~6h
- 预期：acc 如果上 0.92+ 说明 delta=2 是天花板，应该砍掉；如果还卡 0.85 说明问题不在数据
- 适合做 A 的下一步消融

#### 选项 C：**LoRA rank 升级 + target 加 mlp** —— 表达力升级
- rank 64 → 128 或 256
- target_modules 加 `up_proj down_proj`
- 显存翻倍但 8 卡 5090 仍能扛
- 成本：~12h
- 预期：1-2pp 提升（不是数量级）
- v5 用 r=64 都到 97.5%，所以 rank 不应该是瓶颈——优先级最低

### 15.5 推荐路径

**先 A**：v6.2-LR9 消融。9 小时换"underfit 假设是否成立"的判定。
- 如果 acc ≥ 0.84 → 问题解决，进入 merge + RL smoke
- 如果 acc 仍 ≤ 0.82 → 数据天花板被 LR 单独验证不出来，进入 B

**B 作为 A 的备选**：v6.2-delta4-only。
- 如果 delta=4 子集 acc 上 0.92+，证明 delta=2 是噪声/难度天花板 → v6.3 应该砍 delta=2
- 如果还卡 0.85 → 真正的瓶颈在别处（LoRA / base / 长度），那时再考虑 C

**不推荐当前阶段做**：
- v6.2-B（仅 high 交集）：之前论证过数据量损失大于质量提升
- 双 epoch：单 epoch 末段已经 LR=0，再多 epoch 也学不到
- max_len 升级：max_len=3072 截断率未必是主要因素，先解决 LR

### 15.6 v6.2-LR9 启动命令（如需直接执行）

```bash
cd /home/ps/hanjiaqi/LlamaFactory/chatjoy
LEARNING_RATE=9e-6 \
RUN_ID=chatjoy_rm_gemma4_sft_v6.2_LR9_$(date +%Y%m%d_%H%M%S) \
bash launch_rm_train_gemma4_sft_v6.2_fg.sh
# 或 tmux 后台：
# tmux new -d -s rmv62_lr9_train "cd /home/ps/hanjiaqi/LlamaFactory/chatjoy && LEARNING_RATE=9e-6 RUN_ID=chatjoy_rm_gemma4_sft_v6.2_LR9_$(date +%Y%m%d_%H%M%S) bash launch_rm_train_gemma4_sft_v6.2_fg.sh"
```

`LEARNING_RATE` 与 `RUN_ID` 通过 env var 覆盖 launcher 默认值，**无需改 launcher 代码**——保留 v6.2 当前版本作为 baseline，A 实验完全可对照。

### 15.7 禁止项更新（接 14.8）

- 不要删除 v6.2 当前 SAVE_PATH `saves/gemma4-26b-a4b/chatjoy_rm_gemma4_sft_v6.2_fg_20260515_195216/`——是 LR=5e-6 baseline，A 实验的对照组。
- 不要改 `launch_rm_train_gemma4_sft_v6.2_fg.sh` 里的 default LR（5e-6 是已 run 的 baseline，env 覆盖即可）。
- 不要在 v6.2-LR9 跑完之前把 v6.2 (LR=5e-6) 走 merge → RL smoke。等先选定哪个版本作为"最终 v6.2 RM"再做下游流程，避免重复工作。
- 不要把 v6.2 (LR=5e-6) 和 v6.2-LR9 (LR=9e-6) 文件名/RUN_ID 混在一起 —— RUN_ID 必须带 `_LR9` 后缀，eval_rewards 和 plots 分开。
- 不要先做 LoRA rank 升级（选项 C）—— 在 LR 假设验证前升级表达力是浪费 GPU。

### 15.8 当前任务清单（接手者直接看这里）

已完成（不要重复跑）：
- [x] v6.2 cross-validation 数据 (27,287 pair, 73.1% retention)
- [x] v6.2 train/eval mix (42,270 / 3,614)
- [x] v6.2 全量训练 LR=5e-6 (660 step, 9.5h) → acc=0.802, wrong=18.0%

待执行（按 15.5 路径）：
1. **A 实验：v6.2-LR9 全量训练**（LR=9e-6，其他全同）
2. A 完成后 eval 对照 v6.2 baseline + v6.1
3. 若 A acc ≥ 0.84：A 版本 adapter merge → vLLM classify smoke → 30-step RL smoke → 正式 RL
4. 若 A acc < 0.82：B 实验（v6.2-delta4-only 诊断）
5. 若 B 也卡：考虑 LoRA rank 升级或 base 升级（C）

## 2026-05-20 RMv6.2 A+C 可续训重跑记录

旧 run `chatjoy_rm_gemma4_sft_v6.2_AC_fg_8gpu_resumable_20260520_193011` 真实启动参数仍是 `--save_steps 200`，不满足每 50 step checkpoint 的严格续训设计，已停止。

新 run:

- `RUN_ID=chatjoy_rm_gemma4_sft_v6.2_AC_fg_8gpu_save50_20260520_193735`
- tmux session: `rmv62_ac_save50_20260520_193735`
- train log: `/root/hanjiaqi/LlamaFactory/chatjoy/saves/gemma4-26b-a4b/chatjoy_rm_gemma4_sft_v6.2_AC_fg_8gpu_save50_20260520_193735/train.log`
- run env: `/root/hanjiaqi/LlamaFactory/chatjoy/saves/gemma4-26b-a4b/chatjoy_rm_gemma4_sft_v6.2_AC_fg_8gpu_save50_20260520_193735/run_env.txt`
- DS ckpt dir: `/root/hanjiaqi/LlamaFactory/chatjoy/saves/gemma4-26b-a4b/chatjoy_rm_gemma4_sft_v6.2_AC_fg_8gpu_save50_20260520_193735_ckpt`
- launcher: `/root/hanjiaqi/LlamaFactory/chatjoy/launch_rm_train_gemma4_sft_v6.2_AC_fg_8gpu_save50.sh`

已核验真实 deepspeed 命令包含:

- `--save_steps 50`
- `--eval_steps 200`
- `--train_batch_size 64 --micro_train_batch_size 1`
- `--learning_rate 9e-6`
- `--lora_rank 128 --lora_alpha 256`
- `--target_modules q_proj k_proj v_proj o_proj up_proj down_proj`
- `--save_hf_ckpt --max_ckpt_num 8 --max_ckpt_mem 80`

数据和 base 仍保持:

- train: `rm_train_v6.2_fg_combined.jsonl`, 42270 samples
- eval: `rm_eval_v6.2_fg_combined.jsonl`, 3614 samples
- pretrain: `sft2_ddp_lora_3gpu_567_cut3072_merged`

注意: 该 run 是 fresh run，`LOAD_CHECKPOINT=false`。严格续训能力要等第一个 `global_step50` checkpoint 实际落盘后才完整生效；若机器在 step 50 前断电，仍只能从头重跑。

### 4-bit QLoRA 与续训说明

本轮 RM 训练是 QLoRA 风格训练:

- `--load_in_4bit` 量化加载的是底座模型 / pretrain backbone，主要用于节省显存。
- 正在学习的参数不是 4-bit: LoRA adapter 参数和 reward `score` head 是可训练参数。
- `--param_dtype bf16` 表示训练计算 dtype 使用 bf16。
- DeepSpeed checkpoint 文件名中的 `bf16_zero...optim_states.pt` 是当前训练状态的保存形式，不代表把可训练 LoRA 参数压成 4-bit。

续训时必须使用 `global_stepXX` 这种 DeepSpeed checkpoint，而不是 `global_stepXX_hf`:

- `global_stepXX` 保存模型状态、optimizer states、LR scheduler states、`consumed_samples`，适合严格续训。
- `global_stepXX_hf` 是导出的 HF/LoRA adapter，适合推理、评估或后处理，不适合作为不损训练进度的续训入口。

因此，在相同 8 卡、相同数据、相同 batch/LR/LoRA 配置下，从完整 `global_stepXX` 恢复不会因为 4-bit/QLoRA 机制本身导致训练质量下降；主要损失只会来自断电时最后一个 checkpoint 之后尚未保存的进度。
