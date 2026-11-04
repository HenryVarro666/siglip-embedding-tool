# OPUS术语约束与新增查询 v1

固定同一40图图库，使用16个新中文意图（32个中英配对字符串），比较原始OPUS greedy/beam1、无约束beam4、领域术语约束beam4；各自测自动英文和中英等权融合，并保留中文控制，共7策略。Base、So400m各一个独立进程，顺序运行。全部是预训练推理，没有训练或生成后改写英文。

本轮代码/输入冻结提交：`b68cb99`。固定数据、配置、代码、OPUS文件、索引和外部解码文件SHA（SigLIP固定revision/processor身份，权重字节SHA由最终审计补录）见[freeze.json](freeze.json)、[protocol.json](protocol.json)。原图库manifest为旧文件同字节副本，图片和两个索引不重建。

## 查询和标签的独立性

- 词表与上下文规则先在`3e0da59`冻结，新查询编写者没有访问历史查询、译文或分数，目视检查全部40张原图。
- 16个新意图有23个正例判断；中英共享同一标签，1280个显式二元单元是32字符串×40图，不能当作1280个独立标注或32个独立意图。
- [第二助手复核](label-review.json)检查原图/标签，没有发现明确qrel错误；其按入口阅读README时见过历史汇总，已披露。两条备注额外闭壳属性不进入翻译审阅，按中文查询实际要求评价。
- 图库仍是旧40图，标签未经领域人员/独立人类审核，公开图像预训练重叠未知。这是未用于选规则的新查询测试，不是现场图像留出集。
- [queries.json](queries.json)、[qrels.json](qrels.json)、[annotation-notes.json](annotation-notes.json)保存全部新文本、正例、完整二元表、对象归属/可见性/接线/安装依据。配对英文不传给译者，也不替换输出。

## 安装与准备

```bash
uv sync --frozen --extra dev --extra translation
uv run python -m siglip_tools.translation prepare --output models/opus-mt-zh-en
uv run python -m siglip_tools.opus_constraints prepare --output models/opus-constrained-decoder
```

已有目录拒绝覆盖。按[设备集说明](../equipment-v1/README.md)下载40张图并准备各模型自己的索引；已有完整本机资源时无需重复准备。

OPUS沿用固定`Helsinki-NLP/opus-mt-zh-en` revision `cf109095479db38d6df799875e34039d4938aaa6`，CC-BY-4.0。检索模型仍是Base224与So400m384各自固定revision与processor，不混用向量。

Transformers5.22.1的约束beam搜索迁到了[独立生成实现](https://huggingface.co/transformers-community/constrained-beam-search)，本轮固定revision `07b2f120b7db38f1d7bac617ad65ea130508f297`。准备阶段只下载三个Python文件及README，校验内置SHA，保存在忽略目录；没有下载该仓库附带的Qwen权重。代码经过本地检查，通过本地callable调用，不在查询时加载远程生成字符串。该仓库未声明仓库级许可，当前不在Git中重新分发其Python源文件；保留文件头、原README和[来源清单](decoder-provenance.json)。OPUS/检索模型仍关闭远程自定义代码。

## 可复用CLI

```bash
# 领域术语约束，再用英文检索
uv run siglip-tools search-text \
  --index artifacts/equipment-benchmark/quality-v1/index.npz \
  --query "微型逆变器固定在横向金属导轨上，机身上方有一根裸露铜线，右侧连着黑色电缆。" \
  --auto-translate english --translator opus --opus-decoding constrained \
  --device mps --local-files-only

# 同样beam4但不加术语约束；不要把beam变化误认为术语收益
uv run siglip-tools search-text \
  --index artifacts/equipment-benchmark/quality-v1/index.npz \
  --query "微型逆变器固定在横向金属导轨上，机身上方有一根裸露铜线，右侧连着黑色电缆。" \
  --auto-translate english --translator opus --opus-decoding beam \
  --device mps --local-files-only
```

`--opus-decoding original`为原始默认；`constrained`/`beam`要求自动翻译且译者为OPUS。`--auto-translate fusion`改为中英各0.5余弦分数融合。可用`--constraint-decoder-dir`指定已准备的外部解码器目录。JSON保存原样英文、逐词/句结束状态、译者指纹、阶段/总时间。约束或EOS不满足时明确报错，不附加词语后当作成功。

```python
from pathlib import Path
from siglip_tools.translation import LocalTranslator
from siglip_tools.opus_constraints import OPUSConstraintTranslator

base = LocalTranslator(Path("models/opus-mt-zh-en"), device="mps")
translator = OPUSConstraintTranslator(base)
english = translator.translate("微型逆变器固定在横向金属导轨上，机身上方有一根裸露铜线，右侧连着黑色电缆。")
print(english)
print(translator.last_trace)
```

包装器共享base冻结的模型/tokenizer，不修改base配置或参数；无目标术语时走普通beam4。它能传给现有`auto_retrieval.timed_search`，复用已加载encoder/index。单次CLI仍包含加载，常驻请求与首次请求成本分开看。

## 固定规则

[glossary.json](glossary.json)与安装包中的词表同字节。最长中文匹配先处理微型逆变器，避免再强制独立inverter；套管只在变压器语境触发，显式电缆/线缆套管不触发bushing。目标为microinverter(s)、inverter(s)、bushing(s)、cooling fins或radiator fins，允许句首大写；目标token经Marian的`text_target`侧编码。

这些是词语包含约束，不保证接线、否定、同一设备归属、壁/柱支撑、板下/背面或成排关系。词语出现在句尾碎片也不自动算整句正确。文本/上下文根据[预先固定的规则](review-rules.json)独立审阅，不使用检索分数挑判定。

上游最终候选可能回退到未满足约束的序列，适配器因此独立核验目标token/完整词和EOS。单复数token重复/前缀关系在送入解码器前归约，原始允许形式与实际序列都保留在轨迹中。未满足、空输出或截断保持失败。

## 基准复现

确认准备完整后，在仓库根目录按顺序运行，避免其他模型同时工作：

```bash
HF_HUB_OFFLINE=1 uv run python -m siglip_tools.term_benchmark run \
  --model base --index artifacts/equipment-benchmark/base-v1/index.npz \
  --output artifacts/terminology-constraints/base-v1

HF_HUB_OFFLINE=1 uv run python -m siglip_tools.term_benchmark run \
  --model quality --index artifacts/equipment-benchmark/quality-v1/index.npz \
  --output artifacts/terminology-constraints/quality-v1

uv run python -m siglip_tools.term_benchmark compare \
  --base artifacts/terminology-constraints/base-v1/result.json \
  --quality artifacts/terminology-constraints/quality-v1/result.json \
  --output artifacts/terminology-constraints/comparison.json
```

现有结果拒绝覆盖，复跑需新目录。每策略3次预热、16意图各10次热请求，160样本/策略、1120/模型、2240总计时。7策略按意图/轮数轮转。各请求前后同步MPS，分别测翻译、检索和真实wall总时间，不相加阶段分位数替代总时间；模型、索引、文件验证及各解码首调用另列，首调用共享进程/权重，不是独立冷启动。

每次自动请求重新翻译，约束匹配、目标token编码、生成、解码和核验都在翻译计时内；诊断序列/文件持久化在请求外。失败不检索，仍记录失败时间，效果在全16意图分母内计0，另列成功条件指标与失败率。不能删掉失败查询制造收益。重复成功/错误状态、原样英文、约束轨迹和完整排名必须一致，分数容差1e-5。

两模型共享同一OPUS和生成设置，模型分配前验证声明代码SHA/输入/索引；结束重核全部源文件、元数据和译者/解码器清单。比较器要求相同译文与轨迹、协议/实现/环境和各自索引绑定。

原始结果/逐次计时/日志留在`artifacts/terminology-constraints/`；Git保存摘要、原样译文、审阅、失败与改进证据。旧8条仅用作工程预检，其一次有术语但无EOS的失败保存在[development-smoke.json](development-smoke.json)，不计入新集效果。

问题衔接见[总台账](../../docs/issue-log.md)、[历史回填](../../docs/historical-issues.md)与[接续状态](../../docs/continuation.md)。新增问题即时登记，修复补验证，不覆盖前次结果。

## 结果、摘要与Top5页

[完整报告](../../docs/terminology-constraints-v1.md)、[数值摘要](results-summary.json)、[原样译文及失败轨迹](translations.json)、[匿名文本审阅](text-review-blind.json)、[解盲统计](text-review.json)、[独立复算](audit.json)均保存到仓库。两模型的约束策略均失败6/16，不建议替换默认。套管+鳍片组合可能产生有目标词却无正常EOS的重复文本，不能将该片段当作可用翻译。

生成可核查摘要与兼容原审查页的六组视图：

```bash
uv run python examples/summarize_term_benchmark.py \
  --base artifacts/terminology-constraints/base-v1/result.json \
  --quality artifacts/terminology-constraints/quality-v1/result.json \
  --output benchmarks/terminology-constraints-v1/results-summary.json \
  --views-output artifacts/terminology-constraints/views

uv run python examples/equipment_review.py \
  --metadata benchmarks/terminology-constraints-v1 \
  --fusion-result artifacts/terminology-constraints/views/base-original.json \
  --fusion-result artifacts/terminology-constraints/views/base-beam.json \
  --fusion-result artifacts/terminology-constraints/views/base-constrained.json \
  --fusion-result artifacts/terminology-constraints/views/quality-original.json \
  --fusion-result artifacts/terminology-constraints/views/quality-beam.json \
  --fusion-result artifacts/terminology-constraints/views/quality-constrained.json \
  --output artifacts/terminology-constraints/review.html
```

摘要和视图默认拒绝覆盖已有文件；本仓库已有冻结摘要，复跑应为`--output`与`--views-output`选新路径。导出器校验本v1冻结数据、原样跨模型译文/轨迹/索引绑定和控制组；它不运行模型。`implementation_reference_commit`指固定实现来源，不冒充导出时的Git HEAD。

文本输入以A/B/C匿名化，仅包含原中文与原样输出/完成状态；原始审阅保存后才用text-review-key.json解盲。词语表面与失败模式可提示身份，因此仍不是严格人类双盲。t16柜外下方/柜内下部歧义只作事后敏感性，冻结查询和主分母均不改。

## 冻结源码快照（2026-11-04 起）

`frozen-source.json` 记录测量提交 `f079027` 中被实现指纹绑定的源码文件 SHA-256。冻结记录的 `implementation_sha256` 按该快照复算，工作区源码可以维护而不改写测量；执行本协议的新测量仍要求当前源码与快照一致，否则属于新版本。核对：`uv run python -m siglip_tools.frozen_source verify benchmarks/terminology-constraints-v1`。
