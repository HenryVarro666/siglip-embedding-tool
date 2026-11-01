# 已完成候选保护：固定开发控制后的四方案实验

本轮继续不训练。原始 OPUS、旧约束失败回退、旧重复早停回退、新完成候选保护回退，均以中文/自动英文各 0.5 融合检索。同 16 条已诊断查询、40 图和助手标签保持字节一致；t09 是已知工程回归，并非新的独立验证。

## 开发及固定规则

先在 Git 70cf127 保存 24 条开发源句：原 i01–i08、8 条长正常 l01–l08、8 条合法重复 r01–r08；源行语义 SHA256 为 `2917c1aea48f166ad2af5c07664b979902fe11ef6ac59e28843fc09744e9919c`。不把“写得长”直接当实际检测覆盖，观测后分组报告原方案接受、实际≥32步检查和新旧拒绝。

沿用上一轮固定重复 profile `bounded-n16-r2-min32-q4-tail4`。没有新的数值网格或正式查询调参：仅当重复触发时，当前保留的真实 EOS 完成候选通过原 token 包含、完整词形和非空检查，才抑制该次拒绝并继续原搜索。不会直接返回、救回或重排候选。新规则在 `completed_protection_v1.json`，SHA256 `a8fb1d276fc53ace8c59b76dabd13a24035c323aa637f184c768b20db2464cdc`。

完成池前缀不含 EOS；证据来自固定 scorer.process 的真实 EOS 分支，不能使用 finalize 加入的活动前缀或补出的 EOS。请求局部函数/类隔离，原模块、全局、源码、权重、词表、旧接口及默认方案不变。记录实际 EOS 提案/得分、当前成员、逐词检查、受保护步骤及实际早拒绝；词汇合法和完成不保证关系语义正确。

开发完整四种 MT 控制，主失败后真正重新调用原始 OPUS。不检索开发输入，不给译者参考英文，不缓存译文。先保存开发结果并冻结，再开始正式两模型；当前源码/协议/正式结果状态以 [接续](../../docs/continuation.md)和 [台账 SIG-051 起](../../docs/issue-log.md)为准。

## 正式执行与完整耗时

每策略每模型 3 次预热、16 次完整 40 图评分、5×16 次热请求；共 128 次评分、640 次热请求。按意图/轮旋转策略，Base→So400m 分开进程。总墙钟含源处理、候选观察与 CPU 复制、检测、失败前段、验证/异常清理、实际备用翻译及检索；序列化在请求外。加载和第一调用另列。所有尾样本保留。

```bash
uv sync --frozen --extra dev --extra translation
# 沿用已有固定模型/解码器、40图和两套索引；来源和准备见上一轮 README。
uv run python -m siglip_tools.protected_development
uv run python -m siglip_tools.protected_benchmark run \
  --model base --index artifacts/equipment-benchmark/base-v1/index.npz \
  --output artifacts/completed-protection/base-v1
uv run python -m siglip_tools.protected_benchmark run \
  --model quality --index artifacts/equipment-benchmark/quality-v1/index.npz \
  --output artifacts/completed-protection/quality-v1
uv run python -m siglip_tools.protected_benchmark compare \
  --base artifacts/completed-protection/base-v1/result.json \
  --quality artifacts/completed-protection/quality-v1/result.json \
  --output artifacts/completed-protection/comparison.json
uv run python examples/summarize_completed_protection.py \
  --base artifacts/completed-protection/base-v1/result.json \
  --quality artifacts/completed-protection/quality-v1/result.json \
  --output artifacts/completed-protection/reexport-v1
# 等所有测试/构建/CLI等日志完成后，再审计/取共享日志摘要。
uv run python examples/audit_completed_protection.py \
  --base artifacts/completed-protection/base-v1/result.json \
  --quality artifacts/completed-protection/quality-v1/result.json \
  --output artifacts/completed-protection/reaudit-v1.json
```

文件存在会拒绝覆盖。复现请使用新输出路径；开发重跑需复制固定 dev 源/协议到新目录并显式传路径。不要覆盖原协议/开发/模型结果。改变规则、源句或标签需新版本。

## 直接调用

```bash
uv run python -m siglip_tools.completed_protection search \
  --index artifacts/equipment-benchmark/base-v1/index.npz \
  --query '变压器的散热片在垂下的黑色粗电缆后方，电缆连接顶部的套管。' \
  --mode auto_fusion --fallback opus --device mps --local-files-only
```

Python 入口是 `CompletedProtectedTranslator(base, profile=profile, decoder_dir=...)`，配合现有 `timed_fallback_search` 可复用常驻模型。模型实例按串行请求使用。输出保持原主/备用轨迹、实际使用文本/路由、TopK 和完整时间；早拒绝片段不会进入 embedding。主约束拒绝、原重复触发被保护、实际早拒绝和最终失败分别统计。观察器契约异常属于基础设施错误，不是静默回退成功。

原图/权重/索引/原始运行结果留在本机忽略目录；Git 内保存输入、开发观测/验证、冻结规则/协议、完整评分与热计时摘要、复算、验收和报告。旧结果与失败尝试保留，来源和许可见 [上一轮 README](../repetition-early-stop-v1/README.md)。
