# 训练备用代码

默认工具使用冻结的预训练模型。本目录只是备用入口，不由索引或检索命令自动执行；本次实现与验证没有进行训练。

保留的内容包括根目录 `clip_siglip_demo.py` 的教学示例、`siglip_tools.losses` 中的两个损失函数，以及 [包内实现](../src/siglip_tools/finetune.py) 中从预训练模型开始的最小微调流程。`training/finetune.py` 保留为兼容脚本入口。这里没有实现从零预训练 SigLIP，也没有实现 SigLIP 2 的完整多目标训练配方。

## 数据格式

用户显式提供 JSONL，每行一对图片与文字。相对图片路径以 JSONL 所在目录为基准。

```json
{"image": "images/device_a.jpg", "text": "a white solar inverter mounted on a pole"}
{"image": "images/device_b.jpg", "text": "an electrical control cabinet indoors"}
```

训练 batch 至少包含两对样本，并使用完整 batch。应在运行前完成去重、相关性核对和按物理设备划分的训练/验证/测试分区；入口不会自动替你建立评估集。

## 仅查看参数

```bash
uv run python training/finetune.py --help
```

安装后，也可以从任意工作目录调用 `python -m siglip_tools.finetune --help`，不依赖仓库根目录进入 Python 导入路径。

## 将来明确决定训练后运行

以下命令会更新模型参数，仅作为未来使用示例，本次没有执行：

```bash
uv run python training/finetune.py \
  --pairs data/train.jsonl \
  --output checkpoints/domain-v1 \
  --loss siglip \
  --batch-size 4 \
  --max-steps 100 \
  --learning-rate 0.00001
```

模型输出的 `logits_per_image` 已包含温度与偏置，不能在损失里重复添加。SigLIP 损失按全部组合求和后除以原始配对数 N；CLIP 损失对两个检索方向求平均。

输出包含 safetensors 权重、processor 配置与训练摘要，可作为 `--model checkpoints/domain-v1` 的本地 checkpoint。新的权重需要重新建立索引；训练摘要中的 loss 是训练过程值，不是检索准确率。

当前入口不提供训练断点恢复、学习率调度、数据增强、自动验证或最佳 checkpoint 选择。真正开始微调前，应根据数据和评估目标补充这些内容。两个候选设备图片不作为有效训练集。
