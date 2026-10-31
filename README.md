# SigLIP Embedding Tool

使用预训练 SigLIP / SigLIP 2 进行本地图片批量编码、文字搜图和以图搜图。默认直接加载训练好的模型，不需要先训练。输出是 L2 归一化的 CPU float32 NumPy 向量；索引会记录权重版本和预处理身份，阻止不兼容向量混用。

默认模型是 `google/siglip2-base-patch16-224`。另提供 `quality` 选项加载 `google/siglip2-so400m-patch14-384`；这个名称表示更高资源需求的候选配置，不表示已证明它对你的设备图片更好。[模型选择与版本](docs/models.md)

## 安装

推荐 Python 3.12 和 uv，使用仓库中的依赖锁文件：

```bash
uv sync --frozen --extra dev
uv run siglip-tools --help
```

也可以使用现有 Python 环境安装：

```bash
python3.12 -m venv .venv
./.venv/bin/python -m pip install -e '.[dev]'
./.venv/bin/siglip-tools --help
```

首次加载模型会从 Hugging Face 下载权重；后续复用本机缓存。`--local-files-only` 要求完整模型已经缓存。模型使用 safetensors，关闭远程自定义代码。`--device auto` 按 CUDA、MPS、CPU 的顺序选择；也可以明确指定 `--device cpu`、`--device mps` 或 `--device cuda`。

## 批量提取图片 embedding 并建库

```bash
uv run siglip-tools index \
  --images /path/to/images \
  --output artifacts/gallery.npz \
  --batch-size 8
```

`embed-images` 是 `index` 的别名，输出文件同时包含向量与检索所需元数据。默认递归扫描 JPG、JPEG、PNG、WebP、BMP、TIF、TIFF，按路径排序。按 EXIF 方向读取图片，将透明背景合成为白色并统一为 RGB，再执行模型预处理。

规范化后像素完全相同的图片只编码一次，其他路径保存为 aliases。读取时将文件哈希和像素绑定到同一字节快照，编码已核对的 RGB 快照，发布前重新检查所有原图及别名；发现输入变化则中止并保留旧索引。坏图会明确报错，已有索引默认不覆盖；确定要重建时加 `--overwrite`。近重复图片尚不自动合并。

选择其他预训练模型：

```bash
uv run siglip-tools index --images /path/to/images \
  --output artifacts/gallery-quality.npz --model quality --batch-size 2
```

`--model` 也接受 Hugging Face SigLIP 模型 ID 或完整本地 checkpoint 目录。更换模型应创建新索引。

## 文字搜图

```bash
uv run siglip-tools search-text \
  --index artifacts/gallery.npz \
  --query "安装在立柱上的白色光伏逆变器" \
  --top-k 5
```

查询会自动使用索引中记录的 checkpoint commit，并检查模型、processor 和运行版本指纹。输出 JSON，包含 rank、相似度、相对路径和完整路径。相似度是余弦分数，不是经过校准的匹配概率。

文本上限来自 checkpoint，已验证的默认模型为 64 tokens。过长查询会报错，避免静默截断。长文档的纯文本检索应使用单独的文本 embedding 模型。

## 以图搜图

```bash
uv run siglip-tools search-image \
  --index artifacts/gallery.npz \
  --image /path/to/query.jpg \
  --top-k 5
```

默认排除与查询图规范化像素完全相同的记录，避免把找到自己当成检索质量。需要查看自身匹配时加 `--include-self`。JPEG 重新压缩后的近重复仍可能出现，应在正式评估中单独处理。

图库移动后，可在查询时用 `--image-root /new/gallery/root` 改变返回路径的根目录，无需重新计算向量。

## Python API

```python
from pathlib import Path
from siglip_tools.encoder import SigLIPEmbedder
from siglip_tools.index import ImageIndex

root = Path("/path/to/images")
encoder = SigLIPEmbedder(device="auto")
index = ImageIndex.from_directory(encoder, root, batch_size=8)
vectors = index.vectors  # 可直接用于后续分析的 [N, D] NumPy 数组
index.save("artifacts/gallery.npz")

# 在同一个进程中复用 encoder，避免每次查询都重新加载模型。
text_vector = encoder.encode_texts(["a white solar inverter"])[0]
index.assert_compatible(encoder.identity)
for match in index.search(text_vector, top_k=5):
    print(match.rank, match.score, root / match.path)

image_vector = encoder.encode_images(["/path/to/query.jpg"])[0]
for match in index.search(image_vector, top_k=5):
    print(match.path, match.score)
```

低层 `ImageIndex.search` 不自动知道查询图片的身份。若要像 CLI 一样排除自身，可用 `pixel_sha256(load_rgb(query_path))`，传给 `exclude_pixel_sha256` 参数。

重新加载索引时，使用其中的模型身份初始化编码器，并在查询前调用兼容性检查：

```python
index = ImageIndex.load("artifacts/gallery.npz")
identity = index.model_identity
encoder = SigLIPEmbedder(identity["model_id"], revision=identity["revision"])
index.assert_compatible(encoder.identity)
```

## 检查索引

```bash
uv run siglip-tools inspect --index artifacts/gallery.npz
```

`inspect` 读取图片数、向量维度、去重数量和模型身份，不加载预训练权重。`.npz` 使用 `allow_pickle=False` 读取，并检查元数据、跨平台安全的相对路径、向量维度、有限值和单位范数。

索引是某次建库的快照，之后改变图库内容需要重建。`ImageIndex.from_directory` 提供输入一致性检查；直接调用低层 `ImageIndex(vectors, records, ...)` 时，调用者负责确保向量与记录来自同一批输入。

当前使用 NumPy 精确搜索，适合先建立可解释的小图库基线。大规模图库的 ANN、流式编码、增量更新与 OCR 融合见 [改进点](docs/improvements.md)。

## 已有数据能做什么

首次检查时，2Dto3D runtime 的 10 张 JPG 只有 2 种图片内容；实现过程中新增了一张室内设备图，复核为 11 张 JPG、3 种内容，仍有 8 个重复副本。原始候选照片目录有两张图片。可以检查真实模型的编码、索引和查询流程，但不能据此判断一般检索准确率。原图、模型权重和索引均不进入 Git。

需要补充不同设备、相似外观干扰项和人工确认的查询标签。具体检查结果和起步数据方案见 [数据评估](docs/data-assessment.md)，本机验证见 [验证记录](docs/validation.md)。

## 保留的训练代码与学习资料

- [学习笔记](SigLIP学习笔记.md)：介绍、损失公式、猫狗数值例子与梯度。
- [原始技术问答](对话记录.md)。
- [原始损失与梯度演示](clip_siglip_demo.py)，原有五项测试保留。
- [可复用 CLIP 与 SigLIP 损失](src/siglip_tools/losses.py)。
- [备用微调入口](training/finetune.py)：读取显式提供的图文 JSONL，微调预训练模型并保存新的 checkpoint。不会由编码或检索命令自动调用。

训练说明、数据格式与本次未训练的边界见 [训练备用说明](training/README.md)。优先用冻结的预训练模型建立检索评估，再决定是否训练。

## 验证命令

```bash
uv run pytest -m 'not integration'
uv run ruff check src tests training
uv run python clip_siglip_demo.py
```

默认模型完整缓存后，可以额外运行真实模型检查；这些检查只做推理，不执行训练：

```bash
SIGLIP_RUN_MODEL_TESTS=1 uv run pytest
```

## 仓库中的数据边界

`artifacts/`、`data/`、`local/`、`models/`、`checkpoints/`、图片、权重和向量文件均已加入 `.gitignore`。这些目录用于本机数据与结果。第三方模型权重遵循各自的模型许可，仓库不重新分发权重。原有问答中出现的绝对路径是历史记录，当前工具不依赖那些路径。
