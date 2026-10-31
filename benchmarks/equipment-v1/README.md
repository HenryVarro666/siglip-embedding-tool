# 设备检索评估集 v1

40 张来源明确的 Commons 图片、16 条中英文查询，组成 8 对相同检索意图。用途是比较冻结预训练模型的整图文字检索；不用于训练，不评价精确型号、同一实物、多视图几何或以图搜图准确率。

## 图片与标注

图片构成：13 张太阳能逆变器照片（含 1 张中央逆变站远景）、1 张 MPPT 充电控制器、8 张配电/控制设备、7 张变压器、6 张 UPS、5 张无明显逆变器的光伏板场景。13 张逆变器图中包括 3 张微型逆变器、5 张外壳闭合的墙挂逆变器、2 张黄色逆变器与光伏板的安装图、1 张打开外壳图、1 张中央逆变器近景和 1 张站房远景。中央逆变器近景也能看到背后的光伏板，因此“安装在光伏板下/背面”意图有 3 张相关图。

- [manifest.json](manifest.json)：逐图作者、文件页及固定 revision、许可、原始与下载 URL、源 SHA-1、本地 SHA-256、实际尺寸和可见属性。
- [ATTRIBUTION.md](ATTRIBUTION.md)：可读署名清单；图片许可独立于代码许可。
- [sources/commons-api-snapshot.json](sources/commons-api-snapshot.json)：入选图片的原始 Commons API 页面元数据快照。
- [queries.json](queries.json)、[qrels.json](qrels.json)：查询与完整二元正例；中文和英文同意图使用同一标签。
- [annotation-notes.json](annotation-notes.json)：8 个意图的相关性规则、边界案例与候选排除理由。
- [duplicate-review.json](duplicate-review.json)：源哈希、规范化像素哈希、64 位 dHash 检查。阈值内候选另做目视检查；感知哈希不能证明不同实物或完全消除近重复。
- [label-review.json](label-review.json)：第二助手逐图/意图盲审记录、320 对相关性检查与边界限制。
- [freeze.json](freeze.json)：运行目标模型前的数据/查询/标签哈希与冻结时间，另记录不改数据的序列化修复及实现哈希。
- [results-summary.json](results-summary.json)、[完整报告](../../docs/equipment-benchmark-v1.md)：本轮实际指标、计时、逐条错误与后续问题。

查询由助手根据可见设备类别/属性编写，并非采集到的真实用户查询日志。标签由助手逐图目视并结合来源说明整理，另有第二助手盲审；**未经过独立人类标注审核**。未用待比较模型生成标签或据其排序修改标签。不是官方或通用逆变器基准。

MPPT 控制器原在 Commons 的逆变器类别中，但可见 DC 端子和标签支持其为充电控制器，故列为干扰项。类别页/文件名不直接充当真值。中央站房只按来源识别功能，不作为墙挂、微型或打开外壳查询正例。Enphase 图为安装过程中，面板尚未覆盖其上方，因此不满足“在光伏板下方/背面已安装”的场景查询。两张被外罩遮住的变压器图不满足“可见套管或散热片”的属性查询；带锁通风格栅不当作裸露散热片。

图像使用 Commons 的等比缩略版本（小图可能为原始版本），没有本地裁剪、压缩或合成。Commons 缩略尺寸档位可能大于请求的 1280，使用 manifest 中的实际尺寸。单张厂家历史宣传照片允许入选；广告海报和多产品/多视角合成图不入选。已存在的源裁剪图保留说明，其母图不重复计数。

## 复现

在仓库根目录运行，先安装 `uv sync --frozen --extra dev`：

```bash
uv run python -m siglip_tools.commons \
  --manifest benchmarks/equipment-v1/manifest.json --data-root data/equipment-v1

# 两个 checkpoint 预先下载；这一步不纳入计时。
uv run python -m siglip_tools.benchmark prepare --model base
uv run python -m siglip_tools.benchmark prepare --model quality

# 分别启动独立进程，依次执行。模型仅从本地缓存加载。
uv run python -m siglip_tools.benchmark run --model base --device mps \
  --manifest benchmarks/equipment-v1/manifest.json \
  --queries benchmarks/equipment-v1/queries.json --qrels benchmarks/equipment-v1/qrels.json \
  --data-root data/equipment-v1 --output artifacts/equipment-benchmark/base-v1

uv run python -m siglip_tools.benchmark run --model quality --device mps \
  --manifest benchmarks/equipment-v1/manifest.json \
  --queries benchmarks/equipment-v1/queries.json --qrels benchmarks/equipment-v1/qrels.json \
  --data-root data/equipment-v1 --output artifacts/equipment-benchmark/quality-v1

uv run python -m siglip_tools.benchmark compare \
  --base artifacts/equipment-benchmark/base-v1/result.json \
  --quality artifacts/equipment-benchmark/quality-v1/result.json \
  --output artifacts/equipment-benchmark/comparison.json

uv run python examples/equipment_review.py \
  --result artifacts/equipment-benchmark/base-v1/result.json \
  --result artifacts/equipment-benchmark/quality-v1/result.json
```

其他硬件可同时将两条 run 命令改为 `--device cuda` 或 `--device cpu`；不要将不同设备结果混作公平耗时比较。默认图片 batch=4、文本 batch=1，文本预热 3 次，每条重复 10 次。输出目录已有结果会拒绝覆盖；复跑使用新目录。下载器默认逐张间隔 7 秒、遵守 HTTP Retry-After，存在且哈希相同的文件跳过，哈希不同则停止。来源或缩略服务发生变化导致哈希不同，应审查后建立新版本，不能静默替换 v1。

不传 `--result` 运行审查脚本可先生成标签/正例与全部图库的本地 HTML。HTML 引用本机图片，不能脱离本机图库独立分享；脚本输出、图片、索引和原始计时均保留在 Git 忽略目录。

## 指标与耗时口径

`Hit@K` 为 Top-K 中至少命中一张的查询比例；`Recall@K` 为每条查询找到的相关图数量除以其全部相关图数量，再做查询平均。相关图有 1–6 张，Hit@1 不等于 Recall@1。MRR 为首个正例倒数排名均值；mAP 对完整 40 张排序计算，保留全部正例。

分别报告总体、中文、英文及配对意图。16 条查询只是 8 个意图的语言对照，不是 16 个独立任务；不从这一规模推导稳定显著性或产品精度。

计时排除网络下载、Python 启动及 benchmark 模块导入。缓存模型加载包含构造器内的模型/processor 类惰性导入、缓存文件读取和设备搬运，不是磁盘冷缓存测量。单列首次图片 batch、预热后的完整图库建索引、常驻文本查询及缓存向量搜索。完整建库包括读图、哈希、预处理、模型、归一化、CPU 搬运及压缩保存/fsync；不是纯 GPU kernel 时间。常驻查询包括分词、文字编码、CPU 搬运和图库精确排序；前后同步 MPS/CUDA。只有一次完整建库/模型加载测量，查询分位数来自每个模型 160 次顺序热查询，不是 160 个独立任务。

Base 使用原生 224，So400m 使用原生 384。比较的是完整部署配置，无法将结果差异单独归因于容量或分辨率。公开照片可能与模型的 WebLI 预训练来源重叠；本项目无法核实或保证无污染。后续应补充用户自有、采集日期和设备身份明确的现场照片与真实用户查询，另建人工审核留出集。
