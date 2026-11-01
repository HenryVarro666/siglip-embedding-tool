# 约束失败后的显式回退：Base / So400m 固定查询比较

状态：2026-10-31正在进行真实MPS测量，尚无本轮最终结论。代码、协议、输入已经冻结；不训练，不修改16查询/40图库或上一轮结果。最终测量、审计和验收完成后本页将记录实际数值与边界。

实现提交d005f31、冻结71ead7f；[完整复现](../benchmarks/constraint-fallback-v1/README.md)、[协议](../benchmarks/constraint-fallback-v1/protocol.json)、[冻结记录](../benchmarks/constraint-fallback-v1/freeze.json)、[台账SIG-038起](issue-log.md)。代码由`constraint_fallback.py`完成真实先尝试再回退，9策略重新测同期控制，分别报告主失败/触发/最终失败与完整时间。

上一轮约束6/16无EOS，失败按零计入效果分母；完整旧证据仍见[术语约束报告](terminology-constraints-v1.md)。本轮只添加失败处理，不能把备用成功解释为术语、接线、否定或安装关系修复。
