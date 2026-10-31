# 更强预训练译者对照 v1

固定原 40 图 / 8 中文意图，比较 OPUS-MT 和 NLLB-200 1.3B，分别搭配 SigLIP 2 Base224、So400m384。每种组合测中文直接、自动英文、中文与自动英文各 0.5 的余弦分数融合。每次自动请求重新翻译，端到端时间包含翻译；没有训练、改写译文、修改标签或逐查询调参。

- [完整报告](../../docs/stronger-translation-v1.md)：术语、检索效果、阶段和端到端耗时、限制及改进点。
- [OPUS 协议](protocol-opus.json)、[NLLB 协议](protocol-nllb.json)、[运行前冻结](freeze.json)：模型、数据、代码、索引、生成设置和文件哈希。
- [OPUS 来源](opus-provenance.json)、[NLLB 来源](nllb-provenance.json)：固定源权重及本地 safetensors 转换清单。
- [术语规则](term-review-rules.json)：在看到 NLLB 译文与检索结果之前固定的 6 处评分规则及厂商术语来源。
- [术语审阅](term-review.json)：仅看原中文和原样译文完成；助手审阅，未经领域人员审核。
- [原样译文](translations.json)、[结果摘要](results-summary.json)：两译者的全部输出、12 组指标、逐意图正例位置和 Top5、时间分位数及原始结果 SHA256。

## 准备

在仓库根目录运行：

```bash
uv sync --frozen --extra dev --extra translation
uv run python -m siglip_tools.translation prepare --output models/opus-mt-zh-en
uv run python -m siglip_tools.nllb_translation prepare --output models/nllb-200-1.3B
```

NLLB 使用 [facebook/nllb-200-1.3B](https://huggingface.co/facebook/nllb-200-1.3B)，revision `b0de46b488af0cf31749cd8da5ed3171e11b2309`，CC-BY-NC-4.0。它是研究用途的通用翻译模型，本实验不证明领域可靠性或生产适用性。OPUS 使用 [Helsinki-NLP/opus-mt-zh-en](https://huggingface.co/Helsinki-NLP/opus-mt-zh-en)，revision `cf109095479db38d6df799875e34039d4938aaa6`，CC-BY-4.0。模型权重不随仓库重新分发。

准备阶段下载固定源文件，`weights_only=True` 加载官方权重并转换为本地 safetensors，不更新参数；推理验证全部转换文件与来源清单。NLLB 源权重约 5.48 GB，转换文件也需约 5.48 GB，另需缓存、SigLIP 权重和运行内存。已有模型目录拒绝覆盖。

按[设备集说明](../equipment-v1/README.md)准备原图与两个检索模型各自的 `index.npz`。不要将两个模型的向量混用。当前结果复用了已有索引，没有重新提取图库向量。

## 单次调用

```bash
uv run siglip-tools search-text \
  --index artifacts/equipment-benchmark/quality-v1/index.npz \
  --query "能看到套管或散热片的室外电力变压器。" \
  --auto-translate english --translator nllb --device mps --local-files-only
```

`--auto-translate fusion` 使用中英等权融合；`--translator opus` 为默认译者。`--translation-dir` 可指定已经准备的相应模型目录，默认分别为 `models/opus-mt-zh-en` 和 `models/nllb-200-1.3B`。自动模式与 `--english-query` 互斥。JSON 原样保留英文和译者身份，分别输出加载、翻译、检索与真实请求总时间。

Python 可构造 `NLLBTranslator(Path("models/nllb-200-1.3B"), device="mps")`，配合已有 encoder/index 调用 `siglip_tools.auto_retrieval.timed_search`；接口与 `LocalTranslator` 相同。同一进程中复用模型可避免每次重新加载，但不会缓存生成的英文。

## 完整复现

以下四条命令应依次完成，使用四个独立进程，避免同时运行模型任务。本次就是此顺序。新运行使用新输出目录，已有结果拒绝覆盖。

```bash
HF_HUB_OFFLINE=1 uv run python -m siglip_tools.auto_benchmark run \
  --model base --translator opus \
  --protocol benchmarks/stronger-translation-v1/protocol-opus.json \
  --index artifacts/equipment-benchmark/base-v1/index.npz \
  --output artifacts/stronger-translation/opus-base

HF_HUB_OFFLINE=1 uv run python -m siglip_tools.auto_benchmark run \
  --model base --translator nllb \
  --protocol benchmarks/stronger-translation-v1/protocol-nllb.json \
  --index artifacts/equipment-benchmark/base-v1/index.npz \
  --output artifacts/stronger-translation/nllb-base

HF_HUB_OFFLINE=1 uv run python -m siglip_tools.auto_benchmark run \
  --model quality --translator opus \
  --protocol benchmarks/stronger-translation-v1/protocol-opus.json \
  --index artifacts/equipment-benchmark/quality-v1/index.npz \
  --output artifacts/stronger-translation/opus-quality

HF_HUB_OFFLINE=1 uv run python -m siglip_tools.auto_benchmark run \
  --model quality --translator nllb \
  --protocol benchmarks/stronger-translation-v1/protocol-nllb.json \
  --index artifacts/equipment-benchmark/quality-v1/index.npz \
  --output artifacts/stronger-translation/nllb-quality
```

每个组合每策略先预热 3 次，再按 8 意图各测 10 次，80 个热请求 / 策略，四组合共 960 个计时样本。请求顺序轮转，每次前后同步 MPS；原始 JSON 保存阶段/总时间与全量排序，并在重复请求期间检查译文、排名及分数漂移。模型/索引加载、文件哈希验证与进程首次请求另列。不是 HTTP 服务测量，也未清空 OS/硬件缓存。

生成设置为不采样、beam1、max_new_tokens64、源上限256tokens、float32；长输入、空输出或未正常 EOS 结束直接报错。NLLB 使用 `zho_Hans` 源语言、强制 `eng_Latn` 目标 BOS；其必要语言 token 与 OPUS 不同，全部记录于译者身份。没有给模型注入参考英文、术语表或人工改写。

```bash
uv run python -m siglip_tools.auto_benchmark compare \
  --base artifacts/stronger-translation/opus-base/result.json \
  --quality artifacts/stronger-translation/opus-quality/result.json \
  --output artifacts/stronger-translation/compare-opus.json

uv run python -m siglip_tools.auto_benchmark compare \
  --base artifacts/stronger-translation/nllb-base/result.json \
  --quality artifacts/stronger-translation/nllb-quality/result.json \
  --output artifacts/stronger-translation/compare-nllb.json

uv run python -m siglip_tools.auto_benchmark compare-translators \
  --opus artifacts/stronger-translation/opus-base/result.json \
  --nllb artifacts/stronger-translation/nllb-base/result.json \
  --output artifacts/stronger-translation/compare-base.json

uv run python -m siglip_tools.auto_benchmark compare-translators \
  --opus artifacts/stronger-translation/opus-quality/result.json \
  --nllb artifacts/stronger-translation/nllb-quality/result.json \
  --output artifacts/stronger-translation/compare-quality.json

uv run python examples/equipment_review.py \
  --fusion-result artifacts/stronger-translation/opus-base/result.json \
  --fusion-result artifacts/stronger-translation/nllb-base/result.json \
  --fusion-result artifacts/stronger-translation/opus-quality/result.json \
  --fusion-result artifacts/stronger-translation/nllb-quality/result.json \
  --output artifacts/stronger-translation/review.html
```

跨检索模型比较要求相同译者身份、原样译文、数据、协议、代码和运行环境。跨译者比较要求相同检索 checkpoint、索引、环境、翻译运行库和共同协议字段，且中文控制组的指标、排名、分数一致。两者都会拒绝身份或输入漂移。

本次冻结的计时代码 commit 为 `f023fa2`；旧 OPUS 报告保留原测量口径，不能将旧耗时混入本次对照。完整源权重和索引二进制哈希可能随转换环境或重建而变化，复跑须另存自身清单与结果。换硬件/生成设置也应另建实验协议，保留本次记录。

该小集已经多次用于诊断，只有 8 个意图，标签来自助手，未经独立人类审核，公开图片的预训练重叠未知。时间是本机单次实验的重复请求分位数，不能代替真实现场留出集和并发服务评估。
