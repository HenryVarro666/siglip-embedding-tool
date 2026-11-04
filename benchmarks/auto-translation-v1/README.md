# 自动翻译检索评估 v1

使用原40图 / 8中文意图，固定一个本地预训练译者，对比中文直接、自动英文、中文+自动英文等权融合。每次自动请求都重新翻译，端到端时间包含翻译，没有训练、译文修订或译文文本缓存。

- [完整报告与改进点](../../docs/auto-translation-v1.md)
- [protocol.json](protocol.json)：固定中文输入、译者revision、贪心解码、0.5/0.5融合、预热/重复和计时口径。
- [freeze.json](freeze.json)：数据/代码/索引/翻译文件哈希、运行前冻结与生成配置修复记录。
- [translator-provenance.json](translator-provenance.json)：官方checkpoint源SHA、CC-BY4.0来源及本地safetensors转换哈希。
- [translations.json](translations.json)：8条原中文及逐字保留的自动英文；没有用此前编写的英文纠正输出。
- [results-summary.json](results-summary.json)：六组指标、阶段/总时间分位数、首请求/加载、逐意图正例排名和Top5、原始结果哈希。

## 安装与准备

在仓库根目录运行：

```bash
uv sync --frozen --extra dev --extra translation
uv run python -m siglip_tools.translation prepare --output models/opus-mt-zh-en
```

模型来自[Helsinki-NLP/opus-mt-zh-en](https://huggingface.co/Helsinki-NLP/opus-mt-zh-en)，固定revision `cf109095479db38d6df799875e34039d4938aaa6`。官方源只提供PyTorch权重，准备阶段用`weights_only=True`加载，然后保存为本地safetensors；推理只读取经过哈希验证的本地文件。转换不更新参数。模型卡列CC-BY4.0，作者README随本地模型保留。

准备下载/转换不计入热请求耗时，已有输出目录拒绝覆盖。生成策略固定为不采样、beam1、max_new_tokens64，清理max_length与强制EOS的模型默认回填以正确检测截断。源文本上限256token；超长、空输出或未正常结束明确报错。当前自动模式要求输入包含中文，可含英文/数字。

首次克隆还需按[设备集说明](../equipment-v1/README.md)下载图像并生成Base/So400m各自的冻结`index.npz`。原图、模型权重、索引、原始结果保存在本机忽略目录，不随Git分发。

## 直接检索

```bash
# 自动译文英文检索
uv run siglip-tools search-text \
  --index artifacts/equipment-benchmark/quality-v1/index.npz \
  --query "能看到套管或散热片的室外电力变压器。" \
  --auto-translate english --device mps --local-files-only

# 中文与自动译文等权融合
uv run siglip-tools search-text \
  --index artifacts/equipment-benchmark/quality-v1/index.npz \
  --query "能看到套管或散热片的室外电力变压器。" \
  --auto-translate fusion --device mps --local-files-only
```

`--translation-dir`默认`models/opus-mt-zh-en`。`--auto-translate`与`--english-query`互斥。JSON保留自动英文、翻译器指纹、结果及`seconds.translation/retrieval/query`；`query`为实际翻译+检索时间，`load`另列。

Python可复用常驻`LocalTranslator`及原encoder/index，调用`siglip_tools.auto_retrieval.timed_search(index, encoder, translator, chinese, mode='auto_en', top_k=5)`。模式还有`zh`和`auto_fusion`；它不会缓存译文，每次自动调用都会执行生成。

## 完整基准复现

按顺序启动两个独立进程，使用固定协议MPS/float32：

```bash
HF_HUB_OFFLINE=1 uv run python -m siglip_tools.auto_benchmark run \
  --model base --index artifacts/equipment-benchmark/base-v1/index.npz \
  --output artifacts/auto-translation/base-v3

HF_HUB_OFFLINE=1 uv run python -m siglip_tools.auto_benchmark run \
  --model quality --index artifacts/equipment-benchmark/quality-v1/index.npz \
  --output artifacts/auto-translation/quality-v3

uv run python -m siglip_tools.auto_benchmark compare \
  --base artifacts/auto-translation/base-v3/result.json \
  --quality artifacts/auto-translation/quality-v3/result.json \
  --output artifacts/auto-translation/comparison.json

uv run python examples/equipment_review.py \
  --fusion-result artifacts/auto-translation/base-v3/result.json \
  --fusion-result artifacts/auto-translation/quality-v3/result.json \
  --output artifacts/auto-translation/review.html
```

已有结果拒绝覆盖，复跑使用新输出目录。`--metadata`、`--data-root`沿用equipment-v1默认路径，`--translation-dir`指定本地译者，`--protocol`可指定复制后的新实验协议。其他硬件改协议中的device/translator_device，并用新实验版本记录；不得将不同设备结果混为同环境对比。

每模型每策略3次预热、8意图各10次热请求，80样本/策略；固定轮转执行顺序，前后同步MPS。实际逐请求时间保存于原始JSON，不用阶段分位数相加代替总时间。索引/模型加载、输入及权重验证、首次自动英文请求另列。此处“首次”指本进程首个请求，不是清空硬件/OS缓存的测量。

两模型的译者身份与8条译文必须完全相同；重复译文/排名必须一致，分数容差1e-5。数据、协议、代码、图片、各自索引以及译者所有文件/来源元数据在运行后复核。重建索引或转换环境可能使二进制哈希变化，复跑记录自身来源，保留本次摘要不覆盖。

这是已被用于诊断的助手标注小集，未经独立人类审核；图片预训练重叠未知。原型结果用于比较固定方案，不能推导一般生产精度或稳定时延。

## 冻结源码快照（2026-11-04 起）

`frozen-source.json` 记录测量提交 `f0cd1ab` 中被实现指纹绑定的源码文件 SHA-256。冻结记录的 `implementation_sha256` 按该快照历史公式复算（测量时实现指纹不含 nllb 文件）；复算，工作区源码可以维护而不改写测量；执行本协议的新测量仍要求当前源码与快照一致，否则属于新版本。核对：`uv run python -m siglip_tools.frozen_source verify benchmarks/auto-translation-v1`。
