# ChatJoy GRPO 训练落地实现

## 1. 目标

这份文档只回答一个问题:

**在 ChatJoy 当前项目里, 如何从已有的 SFT 成果继续往 Reward Model 和 Actor 强化学习推进。**

这里采用的主线是:

1. 先做高质量 SFT
2. 再基于 SFT 模型构造 preference / rejected 数据
3. 用 SFT 模型初始化 Reward Model
4. 用 SFT 模型初始化 Actor, 再做 PPO/GRPO

核心原则:

- RM 和 Actor 都不要从裸基座直接起步
- 都要建立在已经对齐过角色、语言、格式的 **SFT 模型** 之上
- LoRA 继续是主训练形式, 不动大 Backbone

---

## 2. 三阶段实现总览

```text
阶段 1: SFT（监督微调）
├─ 基础模型 (4-bit 量化)
└─ + SFT-LoRA (可训练) ──→ 完成后可选择【合并】或【保留】

阶段 2: Reward Model 训练
├─ 基于 SFT 模型（必须！不要用基础模型初始化）
├─ + RM-LoRA (可训练，rank≥64 更佳)
└─ 冻结 Backbone

阶段 3: Actor 强化学习（PPO/GRPO）
├─ 策略模型: SFT 模型 + RL-LoRA (可训练)
├─ 参考模型: SFT 模型 (完全冻结，用于 KL 散度约束)
└─ 奖励模型: 阶段 2 产物 (冻结，仅推理)
```

这三阶段的关系不是平行的, 而是串行依赖:

- Stage 1 负责把模型教成“像 ChatJoy”
- Stage 2 负责让模型知道“什么样的 ChatJoy 回复更好”
- Stage 3 负责让策略模型在 RM 反馈下继续优化

---

## 3. 你当前项目中的实际对应物

### 3.1 SFT 数据

当前最关键的 SFT 数据文件是:

- [/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/runs/clean32_for_sft2.jsonl](/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/runs/clean32_for_sft2.jsonl)

这份数据已经包含了:

- `instruction`
- `system`
- `output`
- `language`

它非常适合作为后续 preference 数据构造的起点, 因为:

- `instruction` 给了用户任务
- `system` 给了角色和行为约束
- `output` 是当前 chosen

### 3.2 当前 SFT 模型

当前 SFT merged 模型目录:

- [/home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/qwen3-30b-moe/clean32_for_sft2_fsdp_qlora_merged](/home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/qwen3-30b-moe/clean32_for_sft2_fsdp_qlora_merged)

这就是后续 RM 和 Actor 的共同起点。

### 3.3 当前 rejected 构造脚本

已经为当前项目补好的 rejected 构造脚本:

- [/home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/build_rejected_from_sft.py](/home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/build_rejected_from_sft.py)

它的定位不是通用 demo, 而是专门服务于你当前 `clean32_for_sft2.jsonl + merged SFT model` 的数据管线。

---

## 4. 数据构造: 从 SFT 样本到 Preference Pair

## 4.1 基本结构

后续 RM / PPO / GRPO 不再直接吃 `instruction/output` 单条监督样本, 而是吃偏好对数据。

推荐的统一结构:

```json
{
  "prompt": "...",
  "system": "...",
  "chosen": "...",
  "rejected": "...",
  "language": "de",
  "rejected_type": "wrong_role",
  "quality_chosen": 0.85,
  "quality_rejected": 0.35,
  "reason": "角色错配"
}
```

这里最重要的是:

- `chosen` 来自当前清洗后的高质量 SFT 数据
- `rejected` 来自 SFT 模型的定向生成
- `rejected_type` 用来区分错误来源

## 4.2 为什么 chosen 用原始 output

不建议把 chosen 也交给模型重新生成。

原因:

- 当前 `clean32_for_sft2.jsonl` 已经是清洗过的高质量数据
- 直接把它作为 chosen, 标签更稳定
- RM 的目标是学习“现有好答案优于人为构造的坏答案”

所以标准做法是:

- `chosen = item["output"]`
- `prompt = item["instruction"]`
- `system = item["system"]`

## 4.3 为什么 rejected 要从 SFT 模型生成

直接从数据里硬配 rejected 有两个问题:

- 语义容易不对齐
- 角色和语言可能同时错, 噪声大

让 SFT 模型在同一 `instruction + system` 条件下定向生成坏答案, 好处是:

- 语义仍围绕原任务
- 错误维度可控
- 更接近真实部署中的失败模式

---

## 5. Rejected 数据构造方案

当前建议只做三类 rejected:

1. 错角色
2. 错语言
3. 长度不匹配

推荐混合比例:

- `wrong_role`: 0.45
- `wrong_language`: 0.20
- `length_mismatch`: 0.35

## 5.1 错角色 `wrong_role`

### 定义

同一个 `instruction`, 保持任务不变, 但把 `system` 替换成其他角色的 system prompt, 再让当前 SFT 模型生成回复。

### 目标

让 RM 学会:

- 回复不只是“答对问题”
- 还要“保持正确角色风格”

### 实现方法

输入:

- 原始 `instruction`
- 原始 `chosen system`

构造:

- 从角色池中选一个“错误角色 system”
- 用错误角色 system + 原 instruction 生成回复

输出:

- `chosen = 原 output`
- `rejected = 错角色回复`

### 过滤规则

- 不能和 chosen 高度重复
- 不能包含 `{{user}}`
- 语言最好仍与原 prompt 一致
- 角色差异要足够明显

## 5.2 错语言 `wrong_language`

### 定义

保持原角色 system 不变, 但强制模型用错误语言回答。

### 目标

让 RM 学会:

- 语言一致性是重要质量维度
- 即使角色对、内容顺, 语言错了也应该降分

### 实现方法

输入:

- 原始 `instruction`
- 原始 `system`

构造:

- 添加错误语言约束
- 例如 prompt 是德语, 但强制模型输出英语或法语

输出:

- `chosen = 原 output`
- `rejected = 错语言回复`

### 过滤规则

- 用 `process_chatjoy_data.py` 的 `output_redetect` 方式复检语言
- 确认 rejected 语言确实不等于源语言
- 不要把乱码、极短文本、空输出收进去

## 5.3 长度不匹配 `length_mismatch`

### 定义

保持角色和语言都正确, 只让长度明显不合适。

### 两种子类型

- `length_mismatch_short`
- `length_mismatch_long`

### 目标

让 RM 学会:

- 回复太短会信息不足
- 回复太长会冗余拖沓

### 实现方法

过短:

- 目标长度约为 chosen 的 `35% ~ 60%`
- 优先按句子边界截断或引导模型生成简化版

过长:

- 目标长度约为 chosen 的 `140% ~ 180%`
- 引导模型生成啰嗦版而不是简单重复

### 注意

这类 rejected 容易让 RM 学出“长度 = 质量”的伪规律, 所以不能占比过高。

---

## 6. 阶段 1: SFT（监督微调）

## 6.1 实现目标

先把基础模型调成符合 ChatJoy 风格的 SFT 模型。

这个阶段解决的问题是:

- 角色人设
- 多语言基础对齐
- 输出格式
- 安全占位符清洗
- 基础会话质量

## 6.2 实现方式

```text
基础模型 (4-bit 量化)
+ SFT-LoRA (可训练)
→ 完成后可选择【合并】或【保留】
```

推荐做法:

- Backbone 4-bit 量化加载
- 只训练 SFT-LoRA
- 训练完成后保留两份产物:
  - adapter 版
  - merged 版

### 为什么两份都保留

保留 adapter:

- 方便继续做 LoRA 链路实验
- 方便之后做多 adapter 管理

保留 merged:

- 方便直接做推理、vLLM、Gradio
- 方便作为 RM / Actor 初始化参考

## 6.3 当前项目中的对应

SFT merged 模型:

- [/home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/qwen3-30b-moe/clean32_for_sft2_fsdp_qlora_merged](/home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/qwen3-30b-moe/clean32_for_sft2_fsdp_qlora_merged)

这就是后续阶段的公共起点。

---

## 7. 阶段 2: Reward Model 训练

## 7.1 核心原则

```text
基于 SFT 模型（必须！不要用基础模型初始化）
+ RM-LoRA (可训练，rank≥64 更佳)
冻结 Backbone
```

### 为什么 RM 必须基于 SFT 模型

如果用裸基座来做 RM:

- 它还没学会你的角色风格
- 它不理解多语言约束细节
- 它不理解 system prompt 对回答行为的控制

那 RM 学到的“奖励”就会偏离你真实想要的 ChatJoy 目标。

所以 RM 的初始化必须来自:

- Stage 1 的 SFT 模型

而不是:

- 原始 Qwen 基座

## 7.2 推荐结构

推荐 RM 训练结构:

- Backbone: 冻结
- Head: 标量 reward head
- LoRA: 加在 transformer 主干上

LoRA 建议:

- `rank >= 64`
- 比 SFT LoRA 更大一些

原因:

- RM 不是纯生成任务
- 它要学细粒度偏好判断
- 表征容量不够时很容易学不稳

## 7.3 输入形式

RM 训练时的输入不是“单句评分”, 而是偏好对比较:

- 同一个 `prompt/system`
- 一条 `chosen`
- 一条 `rejected`

训练目标通常可以是:

- pairwise ranking loss
- `score(chosen) > score(rejected)`

## 7.4 RM 数据构造建议

推荐把 rejected 类型混合后再训练 RM:

- 错角色
- 错语言
- 长度不匹配

这样 RM 不会只学一个单维度规则。

## 7.5 当前代码层面的注意事项

当前目录里的 RM 训练脚本:

- [/home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/train_rm_model.py](/home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/train_rm_model.py)

它目前更像原型代码, 正式开训前建议再检查:

- 前向里是否真的使用了带 LoRA 的模型对象
- reward head 是否参与反向传播
- 是否真的冻结了 Backbone 而不是把 LoRA 也一并冻住

也就是说, 当前 RM 思路是对的, 但工程实现应以“复核后再训练”为前提。

---

## 8. 阶段 3: Actor 强化学习（PPO/GRPO）

## 8.1 推荐结构

```text
策略模型: SFT 模型 + RL-LoRA (可训练)
参考模型: SFT 模型 (完全冻结，用于 KL 散度约束)
奖励模型: 阶段 2 产物 (冻结，仅推理)
```

### 三个模型分别干什么

策略模型:

- 当前真正被优化的模型
- 负责生成回复

参考模型:

- 提供 KL 约束
- 防止策略模型在 RL 中偏离 SFT 太远

奖励模型:

- 只负责打分
- 给策略更新提供 reward

## 8.2 为什么策略模型也要从 SFT 模型初始化

因为 Actor 的目标不是“从零学会说话”, 而是:

- 在已对齐的 SFT 基础上继续优化偏好

如果 Actor 从基座直接开始:

- 样本效率差
- 不稳定
- 很容易把角色、多语言和格式能力弄坏

## 8.3 为什么参考模型也应该是 SFT 模型

KL 约束的本质是:

- 不要离“合理初始策略”太远

而对 ChatJoy 来说, 合理初始策略不是裸基座, 而是已经 SFT 好的模型。

所以参考模型应该是:

- SFT 模型完全冻结版

不是:

- 原始基座

## 8.4 PPO 与 GRPO 的区别

PPO:

- 更传统
- 需要 reference / KL
- 常和 value function 一起使用

GRPO:

- 更偏组内相对排序
- 适合同 prompt 多回复场景
- 很适合你这种通过 chosen / rejected 或多候选生成构造出的分组数据

对 ChatJoy 当前项目而言:

- 如果 preference 数据是标准 pairwise 形式, 可以先用 PPO / DPO 风格思路
- 如果每个 prompt 后面能拿到 3~4 个不同质量回复, GRPO 会更自然

---

## 9. 推荐的工程落地顺序

## 9.1 数据阶段

1. 固定 chosen 数据源
2. 用当前 SFT merged 模型生成三类 rejected
3. 抽样人工检查
4. 输出统一 preference pair JSONL

## 9.2 RM 阶段

1. 用 SFT 模型初始化 RM
2. 加 RM-LoRA
3. 冻结 Backbone
4. 用 pairwise ranking loss 训练
5. 检查 chosen/rejected 打分是否正确分离

## 9.3 Actor 阶段

1. 用 SFT 模型初始化策略模型
2. 复制一份完全冻结的 SFT 作为参考模型
3. 加 RL-LoRA 到策略模型
4. 用 RM 作为 reward evaluator
5. 做 PPO/GRPO 更新

---

## 10. 最关键的结论

### 结论 1

**RM 不应该基于基础模型初始化, 必须基于 SFT 模型初始化。**

### 结论 2

**Actor 的策略模型和参考模型都应该从 SFT 模型出发。**

### 结论 3

**当前项目最值得先做扎实的不是直接开跑 RL, 而是先把 rejected / preference 数据构造好。**

### 结论 4

**LoRA 仍然是整个 Stage 2 / Stage 3 的主训练形式, 不建议直接做全量微调。**

---

## 11. 对应文件索引

当前目录中与你这条主线最相关的文件:

- rejected 构造:
  - [/home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/build_rejected_from_sft.py](/home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/build_rejected_from_sft.py)
  - /home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/使用模型生成Rejected完整指南.md

- RM / Actor 原型脚本:
  - [/home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/train_rm_model.py](/home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/train_rm_model.py)
  - [/home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/train_actor_model.py](/home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/train_actor_model.py)
  - [/home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/generate_actor_prompts.py](/home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/generate_actor_prompts.py)
  - [/home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/run_complete_rlhf.sh](/home/ps/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/run_complete_rlhf.sh)

- 当前 SFT 数据 / 模型:
  - [/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/runs/clean32_for_sft2.jsonl](/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/runs/clean32_for_sft2.jsonl)
  - [/home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/qwen3-30b-moe/clean32_for_sft2_fsdp_qlora_merged](/home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/qwen3-30b-moe/clean32_for_sft2_fsdp_qlora_merged)
