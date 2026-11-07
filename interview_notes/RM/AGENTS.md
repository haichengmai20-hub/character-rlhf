# RM 面试资料局部规则

继承 [../AGENTS.md](../AGENTS.md) 的范围、证据、授权与验证原则；不重复增加批准门。Codex 不默认读取 CLAUDE.md。

- 按当前任务读取 v1–v5 或 v6–v6.6 正文的相关段落，不因进入目录就完整读取所有资料。
- 既有配额、Benchmark、Judge、Gap、训练与评测定义查 [TECHNICAL_CONTRACTS.md](TECHNICAL_CONTRACTS.md)；首次使用先读开头的快照适用说明，再读相关章节并与当前主文核对。旧维护优先级不覆盖父级事实判断原则，参考快照不冻结未来可提出的替代方案。
- Pair 构造 heuristic、LLM Judge、Hard Fail、单一 scalar RM、在线 rule reward 和 KL 是不同对象；历史、当前设计和模拟分别标注。
- 核查 chosen/rejected 是否共享完整 Prompt、模板与截断语义；Pair 数与序列数分开，历史上下文参与 attention 不代表拥有 token-wise RM loss。
- 比较 RM 时核对评测集、Prompt Family 隔离、generator/judge 关系、score 尺度、分母和候选选择过程。诊断集、开发评测和独立测试不能混用。
- 用户可以批准方案与文档组织方式，但这不证明实验已执行。发现历史证据与主文冲突，直接判断并解释；修订权限按父级规则处理。
- 已授权修正时完成必要局部修改与复核；不默认为其他目录同步，不把完整学习正文改成短摘要。
