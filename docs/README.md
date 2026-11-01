# 文档入口

后续工作先读[接续状态](continuation.md)与[问题总台账](issue-log.md)。用户要求每次遇到的问题和改进都保存到仓库，包括历史回填、当前修复和未来工作的跟进；[历史问题 SIG-001–019](historical-issues.md)已保留原报告证据。新增记录、修改状态、最终验证与下一步不能只留在聊天。

| 内容 | 入口 |
|---|---|
| 安装、批量embedding、文字/图片检索 | [仓库README](../README.md) |
| SigLIP介绍、损失公式、猫狗例子与梯度 | [学习笔记](../SigLIP学习笔记.md)、[原始对话](../对话记录.md)、[损失代码](../src/siglip_tools/losses.py) |
| 模型选择、身份与输入兼容 | [模型说明](models.md)、[设计](design.md) |
| 最初数据是否足够、真实接口验证 | [数据评估](data-assessment.md)、[验证记录](validation.md) |
| 40图 / 8中英意图原始检索比较 | [设备评估](equipment-benchmark-v1.md)、[数据与复现](../benchmarks/equipment-v1/README.md) |
| 三条中文退步与提供英文的融合 | [双语分析](bilingual-fusion-v1.md)、[固定输入](../benchmarks/bilingual-fusion-v1/README.md) |
| OPUS每请求真翻译及完整耗时 | [自动翻译](auto-translation-v1.md)、[复现](../benchmarks/auto-translation-v1/README.md) |
| NLLB1.3B / OPUS术语、效果与耗时 | [更大译者比较](stronger-translation-v1.md)、[冻结与审计](../benchmarks/stronger-translation-v1/README.md) |
| OPUS领域术语约束与16条新查询 | [术语报告](terminology-constraints-v1.md)、[复现输入](../benchmarks/terminology-constraints-v1/README.md)、[实现计划](superpowers/plans/2026-10-31-opus-term-constraints.md) |
| 约束失败→原始OPUS / 中文直接回退 | [回退报告](constraint-fallback-v1.md)、[固定协议与复现](../benchmarks/constraint-fallback-v1/README.md)、[计划](superpowers/plans/2026-10-31-constraint-fallback.md) |
| 重复检测/真实提前拒绝→OPUS回退 | [早停报告](repetition-early-stop-v1.md)、[开发两阶段及复现](../benchmarks/repetition-early-stop-v1/README.md)、[计划](superpowers/plans/2026-11-01-repetition-early-stop.md) |
| 已完成候选保护 / 长正常与合法重复开发 | [保护报告](completed-protection-v1.md)、[24开发与四方案复现](../benchmarks/completed-protection-v1/README.md)、[验收](../benchmarks/completed-protection-v1/verification.json) |
| 所有问题、修复与后续动作 | [当前台账](issue-log.md)、[历史台账](historical-issues.md)、[改进路线](improvements.md) |
| 后续执行入口与未完成事项 | [接续状态](continuation.md)、[项目工作约束](../AGENTS.md) |
| 备用训练代码 | [训练说明](../training/README.md)，检索命令不会自动训练 |

新增实验使用新目录/协议，记录固定输入、代码、权重、来源和结果哈希。区分现场真值、助手标注、文本审阅与模型分数；纠正旧标签或改变方法时保留旧版本。图像、权重、索引和原始逐次结果保存在本机忽略目录，Git内保存可审查的摘要、失败原因与命令。
