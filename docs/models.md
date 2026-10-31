# 预训练模型选择

## 默认选择

直接加载训练好的图文编码器。默认使用多语言 SigLIP 2 Base 224，先建立冻结模型的检索基线。官方将 SigLIP 2 用于零样本分类、图文检索和视觉特征提取。[官方模型说明](https://huggingface.co/google/siglip2-base-patch16-224)

| CLI 配置 | Checkpoint | 权重文件大小 | 本机状态 |
|---|---|---:|---|
| `base` | `google/siglip2-base-patch16-224` | 1,500,800,904 bytes，约 1.50 GB | 已完成原图 smoke test 和 40 图 / 16 查询评估，768 维 |
| `quality` | `google/siglip2-so400m-patch14-384` | 4,544,143,072 bytes，约 4.54 GB | 已完成同集评估，1152 维；本机质量提高、耗时增加 |

这些是 safetensors 下载文件的大小，不是运行时内存或显存需求。So400m 名称描述视觉主干规模，不能直接当作整个图文模型的总参数量或内存预算。[So400m 模型说明](https://huggingface.co/google/siglip2-so400m-patch14-384)

2026 年 10 月 31 日核对的 commit：

```text
Base 224   75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2
So400m 384 e8e487298228002f3d8a82e0cd5c8ea9c567f57f
```

运行时未指定 `--revision` 会先解析当前配置对应的 immutable commit，然后用同一个 commit 加载模型和 processor。具体索引里保存实际使用的 commit；将来默认分支更新不会使旧索引自动改用新权重。

## 为什么同时保留两个选择

Base 提供较轻量的可复现基线。2026-10-31 已在冻结的同一份 40 图 / 16 查询（8 个意图）评估集中完成两者比较：Base / So400m 的 Hit@1 为 68.75% / 87.50%，mAP 为 0.693 / 0.824；M4 Pro / MPS 常驻文本查询 p50 为 6.84 / 21.29 ms，完整 40 图建库为 0.98 / 5.96 秒。So400m 有 3 条中文查询 AP 退步，不能据平均分推断所有任务都更好。

这比较了各自原生 224 / 384 的完整配置，查询与标签由助手整理并盲审，未经过独立人类审核；公开图像的预训练重叠未知。默认仍为 Base，优先质量时可选 So400m，下一步用用户现场图和真实查询复核。完整指标、逐条错误及计时定义见 [评估报告](equipment-benchmark-v1.md)。

SigLIP 2 论文报告了相对初代模型的多项改进，但论文结果不能证明哪个 checkpoint 最适合当前逆变器图片。[SigLIP 2 论文](https://arxiv.org/abs/2502.14786)

## 模型与 embedding 的兼容性

图像和文字 embedding 必须来自相互匹配的模型。相同维度仅表示数组形状一致。

当前索引身份记录：模型 ID、权重 revision、本地 config 文件哈希、processor 与 tokenizer 配方哈希、预处理实现版本、pooling 方式、归一化方式、文字长度和关键运行库版本。查询比较完整身份指纹，任一项改变都要求使用原环境或重建索引。

这是一种保守的重现策略：即使某次库升级可能不改变结果，也不会默认允许旧索引与新处理流程混用。

## API 与处理细节

当前验证环境为 Transformers 5.22.1。部分 feature API 返回 tensor，部分返回带 `pooler_output` 的 ModelOutput；适配层统一提取 checkpoint 的 pooled features，不把 patch token 数组误当成整图 embedding。[Transformers SigLIP 2 接口](https://huggingface.co/docs/transformers/model_doc/siglip2)

默认模型提供 768 维向量和 64-token 文本输入。其他模型的输出维度由实际模型决定，工具不把 768 写成所有 checkpoint 的固定维度。文字过长时明确报错；图像预处理由加载的 processor 决定。

代码允许兼容的 `siglip`、`siglip2` model type，包括完整本地 checkpoint。其他 checkpoints 和 NaFlex 变体尚未逐一在本机验证，不能视为已经认证的配置。

## 后续比较原则

先保留默认模型的原始 Top-K，再测试更大的预训练模型。若错误主要来自整图背景、铭牌文字或对象尺度，优先做裁剪、OCR 或局部特征的对照实验。只有在固定评估集上确认预训练方案不足，再考虑微调。
