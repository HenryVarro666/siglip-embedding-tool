# 中英文等权查询融合 v1

对原有40图 / 8对中英文意图，比较中文单独、英文单独与0.5/0.5余弦分数融合。两套模型分别运行，不训练；图片、查询、标签、模型权重与原图向量保持原样。每个策略每个意图只生成一份排序。

- [完整中文报告](../../docs/bilingual-fusion-v1.md)：三条中文AP退步分析、全部意图指标、耗时与改进点。
- [protocol.json](protocol.json)：运行前固定策略、权重、设备、预热/重复次数、分数稳定性容差和计时范围。
- [freeze.json](freeze.json)：此次运行的数据/协议/代码/索引/旧结果哈希，记录融合打分之前的状态。
- [regression-diagnosis.json](regression-diagnosis.json)：融合打分之前完成的旧结果诊断；正负例排名、AP贡献、查询语义与视觉边界、解释假设。
- [results-summary.json](results-summary.json)：本轮实际模型身份、六组指标、逐意图正例排名/Top5、分词长度及计时汇总。
- [原数据及许可](../equipment-v1/README.md)：未改变的40图清单、16条原始查询与正例标签。

所有标签为助手目视/来源整理，未经过独立人类审核。现成英文是按图库/意图编写的对照查询；本轮融合不执行自动翻译，耗时不包括翻译或生成英文。这个数据集已经用于诊断，结果应作为探索性观察，下一步需要独立用户查询留出集。

## 直接使用

在仓库根目录，安装 `uv sync --frozen --extra dev`。提供一条中文查询及相应英文即可等权融合：

```bash
uv run siglip-tools search-text \
  --index artifacts/equipment-benchmark/quality-v1/index.npz \
  --query "能看到套管或散热片的室外电力变压器。" \
  --english-query "An outdoor power transformer with visible bushings or cooling fins." \
  --device mps --local-files-only --top-k 5
```

不传 `--english-query` 时，原有单查询行为保持原样。API为 `siglip_tools.fusion.search_texts(index, encoder, texts, weights=None, top_k=5)`；自动检查匹配模型/processor身份，默认等权。返回的分数是平均余弦相似度。

## 复现实验

首次克隆需要按[上一轮评估说明](../equipment-v1/README.md)下载图库、缓存checkpoint并运行两套原始benchmark，生成各自的 `index.npz` 和 `result.json`。原图、权重、向量和原始结果都不在Git中。

同一模型先完成单独进程的测量，再启动另一模型：

```bash
HF_HUB_OFFLINE=1 uv run python -m siglip_tools.fusion_benchmark run \
  --model base --index artifacts/equipment-benchmark/base-v1/index.npz \
  --reference-result artifacts/equipment-benchmark/base-v1/result.json \
  --output artifacts/bilingual-fusion/base-v1

HF_HUB_OFFLINE=1 uv run python -m siglip_tools.fusion_benchmark run \
  --model quality --index artifacts/equipment-benchmark/quality-v1/index.npz \
  --reference-result artifacts/equipment-benchmark/quality-v1/result.json \
  --output artifacts/bilingual-fusion/quality-v1

uv run python -m siglip_tools.fusion_benchmark compare \
  --base artifacts/bilingual-fusion/base-v1/result.json \
  --quality artifacts/bilingual-fusion/quality-v1/result.json \
  --output artifacts/bilingual-fusion/comparison.json

uv run python examples/equipment_review.py \
  --fusion-result artifacts/bilingual-fusion/base-v1/result.json \
  --fusion-result artifacts/bilingual-fusion/quality-v1/result.json \
  --output artifacts/bilingual-fusion/review.html
```

已有结果拒绝覆盖，重复运行请指定新输出目录。`--metadata`默认`benchmarks/equipment-v1`，`--data-root`默认`data/equipment-v1`，`--protocol`默认本目录protocol.json。

本次协议是MPS/float32，查询batch1/1/2、每策略预热3次、8意图各重复10次。每个模型共240次热测量，每策略80次。执行顺序轮转，计时前后同步加速器。任何重复或历史排名漂移、绝对分数误差大于1e-5都会在结果发布前报错。

其他硬件请复制protocol.json为新实验文件并修改device，再给两条run命令传同一`--protocol`。新索引重建会把路径/运行身份写入元数据，因此二进制索引哈希可能变化；freeze.json中的索引哈希对应本次发布，重新运行会记录实际文件哈希。对比要求两套结果的数据、协议、实现、硬件环境和运行库一致。

耗时包括文字分词、编码、CPU搬运、身份核对、平均分数和40图完整排序；不包括模型加载、图库编码、翻译、下载和报告写入。模型/索引加载另行记录。原始计时和完整排名保留在本机`artifacts/bilingual-fusion/`。
