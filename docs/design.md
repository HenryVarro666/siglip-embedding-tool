# SigLIP embedding 工具设计

## 用户已确认的目标

在本目录实现可复用的预训练 SigLIP embedding 工具，支持批量编码、文字搜图、以图搜图。优先直接使用他人训练好的模型，不优先训练；保留已有损失函数教学代码，并提供显式调用才会运行的备用微调入口。整理为私有 GitHub 仓库，保存使用说明、模型选择依据和改进点。

## 实现选择

- Python 3.12，PyTorch、Transformers、NumPy、Pillow，Python API 与 CLI 共用实现。
- 默认 `google/siglip2-base-patch16-224`，另支持 `google/siglip2-so400m-patch14-384` 及兼容的原始 SigLIP。Base 便于先完成本地验证；So400m 是更高资源需求的候选，不提前声明它在本地设备数据上更好。
- 下载模型时解析并固定 Hugging Face commit，模型和 processor 使用相同 revision。支持从本地完整 checkpoint 加载，不运行远程自定义代码。
- `SigLIPEmbedder.encode_images(images, batch_size=8)` 与 `encode_texts(texts, batch_size=8)` 返回二维、L2 归一化的 CPU float32 NumPy 数组。
- 推理使用 `eval()`、`inference_mode()` 和 float32。`device=auto` 按 CUDA、MPS、CPU 顺序选择；明确指定但不可用的设备报错。Text 超过 checkpoint token 上限时报错，避免静默截断。
- 图片按 EXIF 方向读取，透明背景合成为白色，统一为 RGB，再交给 checkpoint processor。文件夹扫描使用固定排序；坏图默认明确报错。
- 建索引时按规范化 RGB 像素哈希去重，保留重复路径别名与原文件哈希。不同 JPEG 压缩引起的近重复仍是后续问题。
- 索引保存为单个 `.npz`，包括 embedding、相对路径、文件及像素哈希、完整模型身份和处理版本。读取使用 `allow_pickle=False`，检查维度、数量、有限值、单位范数和路径。
- 查询时严格比较模型身份指纹，不接受仅维度相同的不同模型向量。模型权重 revision、processor、文本配置和处理实现版本都参与身份。
- 小图库先用 NumPy 精确余弦检索，稳定排序，返回图片路径及相似度。索引保存须完整后原子替换；默认不覆盖已有文件。
- CLI 提供 `index`（别名 `embed-images`）、`search-text`、`search-image`、`inspect`。以图搜图默认排除规范化像素完全相同的查询图，`--include-self` 可以保留。
- 备用微调入口读取用户显式提供的图文 JSONL，调用预训练模型和保留的 CLIP/SigLIP 损失，只在用户运行训练命令时更新参数。本次不执行训练。

## 数据与验证

用户指定 `/Users/caochao/Documents/2Dto3D/data/`。其 runtime 中 10 张 JPG 仅有 2 种文件内容和 RGB 像素内容，分别重复 8 次、2 次。原图位于同项目 `photos/string_inverters/`，共两张，来源许可与实物关系未确认，不能从文件名推断型号。

仅用两张原图检查真实模型能否编码、建库、文字查询和图片查询；记录原始排名与耗时，不据此宣称检索准确率。数据、权重、向量、训练输出和本机报告保存在 Git 忽略目录，不上传原图。补充数据和标注需求写入 `docs/data-assessment.md` 与 `docs/improvements.md`。

## 验收

1. 原有猫狗损失与梯度测试保持通过。
2. 新测试覆盖错误模型身份、非有限与零向量、持久化校验、稳定 Top-K、去重和图像预处理。
3. 真实预训练模型在两张用户图片上完成批量编码、索引保存重载、中文和英文文字查询、排除自己的图片查询。
4. 从新的 clone 可以安装、运行离线测试、查看 CLI 帮助，模型推理命令首次运行下载权重。
5. 文档明确区分已验证行为、未测试模型和后续研究点。

## 本次不实现

不收集或发布新的用户数据，不执行训练，不把 SigLIP 视为精确 SKU 识别器或 3D 重建器。不实现 OCR 融合、ANN 数据库、HTTP 服务、图形界面或几何解码器；这些作为有评估标准的后续事项记录。
