# 约束失败后回退：固定16查询评估

本轮不训练。逐字节复用[上一轮16查询/40图](../terminology-constraints-v1/README.md)，不新增/修改标签。这批查询的历史失败已知，不是新独立留出集；每请求仍实际运行约束翻译，不能按ID跳过。参考英文不输入译者。完整结论见[报告](../../docs/constraint-fallback-v1.md)，发现与修复见[台账SIG-038起](../../docs/issue-log.md)，后续从[接续状态](../../docs/continuation.md)开始。

## 方法与冻结

- 代码提交d005f31，冻结提交71ead7f；核心实现SHA e377efdeb1bcfe1ff44ac93eca1ee8489d1a5a2032fcba0eabbfdeae958f8675；协议SHA b4e64cd4e9a7314a227386c62507363f22c594a9d8856299301bbca4e175efc1。
- `freeze.json` / `protocol.json`固定全部输入、支持文件、OPUS/解码器文件、两个索引、检索权重字节SHA与解码参数。旧核心44b19b…未修改。查询/支持文件另以源v1常量摘要固定，不能连改新freeze蒙混。
- 9策略：中文直接、原始OPUS英文/融合、约束英文/融合、不成功后回退原始OPUS的英文/融合、不成功后中文直接的英文/融合入口。融合仍各0.5余弦；回退中文只执行中文一次。
- 每模型各策略3预热、16完整评分、5轮×16热请求，策略按意图/轮旋转。720热请求/模型，顺序Base→So400m，一进程一检索模型、共享同一冻结OPUS权重，无译文缓存。
- 只有本次生成输出的EOS/空输出/术语拒绝能触发回退；无效输入、程序/来源/基础设施错误不能偷偷变成成功。原始备用翻译再失败就是最终失败，无隐藏级联。
- 直接测完整墙钟：失败约束尝试→异常处理→备用翻译（若选OPUS）→最终检索。阶段分位数不能相加作为总时间；首调用/加载单列。序列化/保存摘要在请求计时外。
- `primary_rejection`、`fallback_trigger`、`terminal_failure`分母均16（热统计80）；`fallback_success`分母是实际触发数，0触发时为null。失败指标按零保留16意图分母。

## 重现命令

先安装Python3.12环境：

```bash
uv sync --frozen --extra dev --extra translation
uv run python -m siglip_tools.translation prepare --output models/opus-mt-zh-en
uv run python -m siglip_tools.opus_constraints prepare --output models/opus-constrained-decoder
```

图片和两个索引沿用[设备集下载/建库步骤](../equipment-v1/README.md)。必须恢复与协议匹配的完整模型文件和索引，不能绕过身份检查；本机这些已存在，不重复下载/转换。模型权重、原图、索引和原始结果保持忽略，仓库保存来源、固定输入和精选可审查记录。

```bash
uv run python -m siglip_tools.fallback_benchmark run \
  --model base --index artifacts/equipment-benchmark/base-v1/index.npz \
  --output artifacts/constraint-fallback/base-v1
uv run python -m siglip_tools.fallback_benchmark run \
  --model quality --index artifacts/equipment-benchmark/quality-v1/index.npz \
  --output artifacts/constraint-fallback/quality-v1
uv run python -m siglip_tools.fallback_benchmark compare \
  --base artifacts/constraint-fallback/base-v1/result.json \
  --quality artifacts/constraint-fallback/quality-v1/result.json \
  --output artifacts/constraint-fallback/comparison.json
uv run python examples/summarize_fallback_benchmark.py \
  --base artifacts/constraint-fallback/base-v1/result.json \
  --quality artifacts/constraint-fallback/quality-v1/result.json \
  --output artifacts/constraint-fallback/reexport-v1
uv run python examples/audit_fallback_benchmark.py \
  --base artifacts/constraint-fallback/base-v1/result.json \
  --quality artifacts/constraint-fallback/quality-v1/result.json \
  --output artifacts/constraint-fallback/reaudit-v1.json
```

输出存在时拒绝覆盖，重跑必须新目录。`results-summary.json`保存全部指标/率/阶段及分支耗时、身份/加载和源结果SHA；`requests.json`保存完整评分排名/原样轨迹及每次热请求时间/分支/可重建摘要；`audit.json`为不调用生产评分/统计函数的独立复算；`verification.json`保存测试/打包/真实CLI/检出/CI验收口径。本轮两模型已完成、独立算法复算通过。analysis.json追加同6/10意图组的描述分析及尾延迟样本；候选改进不是已验证结论。

## 直接调用

```bash
uv run siglip-tools search-text \
  --index artifacts/equipment-benchmark/base-v1/index.npz \
  --query '变压器顶部的套管连接导线，侧面有散热片。' \
  --auto-translate fusion --translator opus --opus-decoding constrained \
  --constraint-fallback chinese --device mps --local-files-only
```

改为`--constraint-fallback opus`使用新生成的原始beam1英文；默认`error`保留拒绝报错。显式回退返回的JSON记录触发原因、两阶段轨迹、实际查询文本、最终路由与计时；预期最终失败仍输出JSON并退出2。基础设施异常或默认error沿用stderr错误/退出2。Python应用复用模型和索引对象，调用`siglip_tools.constraint_fallback.timed_fallback_search`可避免每次CLI重新加载。

```python
from pathlib import Path
from siglip_tools.index import ImageIndex
from siglip_tools.encoder import SigLIPEmbedder
from siglip_tools.translation import LocalTranslator
from siglip_tools.opus_constraints import OPUSConstraintTranslator
from siglip_tools.constraint_fallback import timed_fallback_search

index = ImageIndex.load("artifacts/equipment-benchmark/base-v1/index.npz")
identity = index.model_identity
encoder = SigLIPEmbedder(identity["model_id"], identity["revision"],
                         device="mps", local_files_only=True)
index.assert_compatible(encoder.identity)
original = LocalTranslator(Path("models/opus-mt-zh-en"), device="mps")
constrained = OPUSConstraintTranslator(original,
    decoder_dir=Path("models/opus-constrained-decoder"))  # 共用原始预训练权重
result = timed_fallback_search(index, encoder, constrained, original,
    "变压器顶部的套管连接导线，侧面有散热片。",
    mode="auto_fusion", fallback="chinese", top_k=5)
print(result["effective_route"], result["timing"], result["matches"])
# 后续请求复用上面的三个对象；只替换查询，不缓存译文。
```

本固定小集中回退提高可返回率，不能证明逆变器/套管/散热片的译法、否定、接线、安装和对象归属得到修复；上一轮匿名审阅与全部负结果保留。
