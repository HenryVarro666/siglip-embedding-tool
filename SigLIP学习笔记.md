# SigLIP 学习笔记

整理日期：2026 年 10 月 31 日。

SigLIP 通过图像与文本的联合预训练，学习能够相互比较的语义特征。它与 CLIP 的主要区别在训练目标：CLIP 对每行、每列做 Softmax 多分类，SigLIP 对每个图文组合做 Sigmoid 二分类。本笔记将技术介绍、损失函数、猫狗数值例子、梯度推导和可运行的 PyTorch 代码放在一起，方便从概念一路验证到实现。

## 文件与阅读顺序

| 文件 | 内容 |
|---|---|
| [SigLIP学习笔记.md](SigLIP学习笔记.md) | 整理后的技术介绍、公式、例子、完整代码与运行方法 |
| [对话记录.md](对话记录.md) | 前三轮技术问答原文，保留当时的解释与代码片段 |
| [clip_siglip_demo.py](clip_siglip_demo.py) | 可运行的损失函数、猫狗演示、解析梯度检查和有限差分检查 |
| [test_clip_siglip_demo.py](test_clip_siglip_demo.py) | 5 项固定数值测试，包含非对称矩阵检查 |
| [requirements.txt](requirements.txt) | 复现环境的 PyTorch 版本 |
| [demo_output.txt](demo_output.txt) | 在此目录运行演示脚本的实际输出 |
| [test_results.txt](test_results.txt) | 在此目录运行测试的实际输出 |

第一次学习时，可以按“技术介绍 → 统一记号 → 两种损失 → 猫狗例子 → 偏置与梯度 → 代码运行”的顺序阅读。代码直接使用构造的分数矩阵，不需要下载图片、数据集或预训练模型。

## SigLIP 的基本思路

SigLIP 的名称来自 **Sigmoid Loss for Language Image Pre-Training**。Google DeepMind 在 2023 年提出这一方法，名称也常用来指采用该方法训练出的图文模型。[SigLIP 原论文](https://arxiv.org/abs/2303.15343)

模型通常使用两个编码器：图像编码器采用 Vision Transformer，文本编码器采用 Transformer。图像与文字分别编码成向量，再通过相似度计算衡量是否匹配。

```text
图片 → 图像编码器 → 图像向量 ─┐
                             ├→ 图文匹配分数
文字 → 文本编码器 → 文本向量 ─┘
```

例如，“小狗在草地上奔跑”的图片应与这段描述更接近，与“飞机正在起飞”的描述更远。训练会同时调整编码器，使匹配图文在特征空间中更加接近。[论文方法部分](https://arxiv.org/pdf/2303.15343)

常见用途包括图文检索、零样本图像分类，以及将图像编码器作为多模态大模型的视觉特征提取模块。零样本分类可以把类别写成文字描述，再比较图片与各描述的匹配分数。[模型使用示例](https://huggingface.co/google/siglip-base-patch16-256)、[SigLIP 2 论文](https://arxiv.org/abs/2502.14786)

## 统一记号

假设一个 batch 有 $N$ 对图文：

$$
(I_1,T_1),\ldots,(I_N,T_N).
$$

图像和文本特征经过 L2 归一化：

$$
u_i=\frac{f(I_i)}{\|f(I_i)\|_2},\qquad
v_j=\frac{g(T_j)}{\|g(T_j)\|_2}.
$$

定义缩放后的相似度分数：

$$
s_{ij}=\alpha\,u_i^\top v_j,\qquad \alpha>0.
$$

$\alpha$ 是可学习的缩放系数，也常写为 $1/\tau$。实现时可用 $\alpha=\exp(\text{log\_scale})$ 保证它为正。

| 符号 | 含义 |
|---|---|
| $N$ | 原始配对样本数量 |
| $s_{ij}$ | 第 $i$ 张图片与第 $j$ 条文字的缩放后分数 |
| $q_{ij}$ | 二分类目标标签，配对为 1，其他组合为 0 |
| $b$ | SigLIP 可学习的标量偏置 |
| $a_{ij}=s_{ij}+b$ | SigLIP 送入 Sigmoid 的 logit |
| $\sigma(x)=1/(1+e^{-x})$ | Sigmoid 函数 |

按照标准的一一配对设定：

$$
q_{ij}=\mathbf 1[i=j].
$$

因此，$N\times N$ 分数矩阵的对角线是正样本，其余位置作为负样本。余弦相似度本身位于 $[-1,1]$，但缩放后的 $s_{ij}$ 可以大于 1。后面的 2 和 1 都是 logit，不是未经缩放的余弦相似度。

## CLIP 的损失函数

### 从图片选择文字

固定第 $i$ 张图片，在 batch 的 $N$ 条文字中选择对应文字。对分数矩阵逐行做 Softmax：

$$
p^{\mathrm{row}}_{ij}
=\frac{e^{s_{ij}}}{\sum_{k=1}^{N}e^{s_{ik}}}.
$$

交叉熵为：

$$
\mathcal L_{I\to T}
=-\frac{1}{N}\sum_{i=1}^{N}\log p^{\mathrm{row}}_{ii}.
$$

### 从文字选择图片

固定第 $j$ 条文字，在 batch 的 $N$ 张图片中选择对应图片。对分数矩阵逐列做 Softmax：

$$
p^{\mathrm{col}}_{ij}
=\frac{e^{s_{ij}}}{\sum_{k=1}^{N}e^{s_{kj}}}.
$$

对应损失为：

$$
\mathcal L_{T\to I}
=-\frac{1}{N}\sum_{i=1}^{N}\log p^{\mathrm{col}}_{ii}.
$$

### 双向平均与梯度

CLIP 对两个方向取平均：

$$
\boxed{
\mathcal L_{\mathrm{CLIP}}
=-\frac{1}{2N}\sum_{i=1}^{N}
\left[\log p^{\mathrm{row}}_{ii}+\log p^{\mathrm{col}}_{ii}\right]
}.
$$

一对图文的概率取决于它与其他候选的相对分数，同一行或同一列的候选通过分母相互影响。[CLIP 原论文](https://arxiv.org/pdf/2103.00020)

从上述公式求导可得：

$$
\boxed{
\frac{\partial\mathcal L_{\mathrm{CLIP}}}{\partial s_{ij}}
=\frac{p^{\mathrm{row}}_{ij}+p^{\mathrm{col}}_{ij}-2q_{ij}}{2N}
}.
$$

单行交叉熵的梯度是 $p-q$，这里再把行方向与列方向的贡献相加，并保留 $1/(2N)$ 的归一化系数。

## SigLIP 的损失函数

### 每一对图文做二分类

SigLIP 将每个位置单独看作一个二分类样本：

$$
a_{ij}=s_{ij}+b,\qquad p^{\mathrm{sig}}_{ij}=\sigma(a_{ij}).
$$

单对图文的二元交叉熵为：

$$
\ell_{ij}
=-q_{ij}\log\sigma(a_{ij})
-(1-q_{ij})\log\left(1-\sigma(a_{ij})\right).
$$

正样本希望分数上升，负样本希望分数下降。每一项无需其他图文对参与归一化。

原论文使用的整体损失是：

$$
\boxed{
\mathcal L_{\mathrm{SigLIP}}
=\frac{1}{N}\sum_{i=1}^{N}\sum_{j=1}^{N}\ell_{ij}
}.
$$

也可以令 $y_{ij}=2q_{ij}-1\in\{-1,+1\}$，写成：

$$
\boxed{
\mathcal L_{\mathrm{SigLIP}}
=-\frac{1}{N}\sum_{i,j}\log\sigma(y_{ij}a_{ij})
}.
$$

这里将偏置写成 $a=s+b$，与原论文算法 1 的 `logits = dot(...) * t + b` 一致。[SigLIP 论文方法与算法](https://arxiv.org/pdf/2303.15343)

### 归一化系数

原论文是“全部 $N^2$ 个组合求和，再除以 $N$”。PyTorch 中应使用：

```python
F.binary_cross_entropy_with_logits(logits, targets, reduction="sum") / n
```

如果直接使用默认的 `reduction="mean"`，则会除以 $N^2$。与原论文相比，损失及其梯度都会缩小为 $1/N$。这个差别会影响与其他损失组合时的权重和训练步长。

### 分数梯度与偏置梯度

由二元交叉熵可得：

$$
\boxed{
\frac{\partial\mathcal L_{\mathrm{SigLIP}}}{\partial s_{ij}}
=\frac{\sigma(s_{ij}+b)-q_{ij}}{N}
}.
$$

由于同一个标量偏置加到了所有位置上：

$$
\boxed{
\frac{\partial\mathcal L_{\mathrm{SigLIP}}}{\partial b}
=\frac{1}{N}\sum_{i,j}
\left[\sigma(s_{ij}+b)-q_{ij}\right]
}.
$$

给定当前分数和偏置时，单个分数的梯度不需要同行、同列其他分数参与计算。所有样本仍共享编码器参数，训练过程中依然会相互影响。

## 猫狗数值例子

以下数值是为理解损失而构造的例子，并非模型训练或推理的测量结果。所有对数均为自然对数。

假设 batch 中只有两对图文，$N=2$：

| $s_{ij}$ | 文字 一只猫 | 文字 一只狗 |
|---|---:|---:|
| 猫的图片 | **2** | 1 |
| 狗的图片 | 1 | **2** |

因此：

$$
S=\begin{bmatrix}2&1\\1&2\end{bmatrix},\qquad
Q=\begin{bmatrix}1&0\\0&1\end{bmatrix}.
$$

### CLIP 的损失值

猫图与猫文字的行 Softmax 概率为：

$$
p^{\mathrm{row}}_{11}
=\frac{e^2}{e^2+e^1}
\approx0.731058579.
$$

对应交叉熵：

$$
-\log(0.731058579)\approx0.313261688.
$$

由于矩阵对称，两个行方向和两个列方向的正样本概率均相同，最终得到：

$$
\boxed{\mathcal L_{\mathrm{CLIP}}=0.3132616875182228}.
$$

### SigLIP 的损失值

先令 $b=0$。逐元素计算 Sigmoid：

$$
P^{\mathrm{sig}}
=\begin{bmatrix}\sigma(2)&\sigma(1)\\\sigma(1)&\sigma(2)\end{bmatrix}
\approx\begin{bmatrix}0.880797078&0.731058579\\0.731058579&0.880797078\end{bmatrix}.
$$

每个正样本的损失为：

$$
\ell_+=-\log\sigma(2)\approx0.126928011.
$$

每个负样本的损失为：

$$
\ell_-=-\log\left(1-\sigma(1)\right)\approx1.313261688.
$$

总损失为：

$$
\boxed{
\mathcal L_{\mathrm{SigLIP}}
=\frac{2\ell_++2\ell_-}{2}
=1.4401896985611953
}.
$$

负样本虽然比正样本低 1 分，但它的 Sigmoid 输出仍达到约 0.7311，因此受到明显惩罚。CLIP 则根据候选之间的相对差距计算损失，并继续推动正样本相对负样本的分数差扩大。两个目标的损失数值不应直接用于比较模型优劣。

### 两种损失的分数梯度

代入解析公式得到：

$$
\nabla_S\mathcal L_{\mathrm{CLIP}}
\approx\begin{bmatrix}
-0.134470711&0.134470711\\
0.134470711&-0.134470711
\end{bmatrix}.
$$

$$
\nabla_S\mathcal L_{\mathrm{SigLIP}}
\approx\begin{bmatrix}
-0.059601461&0.365529289\\
0.365529289&-0.059601461
\end{bmatrix}.
$$

梯度下降使用 $S\leftarrow S-\eta\nabla_SL$。对角线的负梯度会提高正样本分数，非对角线的正梯度会降低负样本分数。这个例子中，SigLIP 对负样本的梯度较大，与其较高的错误匹配输出相对应。

这里分析的是对分数矩阵的梯度。真实编码器训练时，PyTorch 会继续通过链式法则把梯度传播到特征向量和网络参数；分数矩阵的元素并不是彼此独立的模型参数。

## 偏置与整体平移

把所有 logit 统一减去 1.5，矩阵变成：

$$
A=\begin{bmatrix}0.5&-0.5\\-0.5&0.5\end{bmatrix}.
$$

### CLIP 对共同偏置不敏感

对任意共同常数 $c$：

$$
\frac{e^{s_{ij}+c}}{\sum_k e^{s_{ik}+c}}
=\frac{e^{s_{ij}}}{\sum_k e^{s_{ik}}}.
$$

列方向也成立。因此，对整个矩阵加相同常数时，CLIP 的损失和分数梯度均不变。代码给 CLIP 加偏置仅用于验证这个性质，原始 CLIP 损失不需要这一标量偏置。

### SigLIP 的偏置调整匹配门槛

对 SigLIP，保持 $S$ 不变而设置 $b=-1.5$，即得到上面的 $A$。此时：

$$
\sigma(0.5)\approx0.622459331,\qquad
\sigma(-0.5)\approx0.377540669.
$$

正样本置信度降低了一些，负样本的错误匹配输出也明显下降，总损失变小：

| 项目 | $b=0$ | 共同偏置为 $-1.5$ |
|---|---:|---:|
| CLIP 损失 | 0.313261688 | 0.313261688 |
| SigLIP 损失 | 1.440189699 | 0.948153968 |
| SigLIP 正样本 Sigmoid 输出 | 0.880797078 | 0.622459331 |
| SigLIP 负样本 Sigmoid 输出 | 0.731058579 | 0.377540669 |
| SigLIP 偏置梯度 | 0.611855657 | 0 |

SigLIP 在 $b=-1.5$ 时的分数梯度为：

$$
\nabla_S\mathcal L_{\mathrm{SigLIP}}
\approx\begin{bmatrix}
-0.188770334&0.188770334\\
0.188770334&-0.188770334
\end{bmatrix}.
$$

$b=0$ 时偏置梯度为正，梯度下降会减小 $b$。在这个固定分数的对称例子中，$b=-1.5$ 时正负贡献恰好抵消，偏置梯度为 0；它不是所有数据通用的偏置取值。

标准 batch 中有 $N$ 个正样本和 $N(N-1)$ 个负样本。可学习偏置有助于处理这种数量不平衡，原论文也通过偏置初始化避免训练开始时负样本产生过大的修正。[SigLIP 原论文](https://arxiv.org/pdf/2303.15343)

## PyTorch 完整实现

下面代码与同目录的 [clip_siglip_demo.py](clip_siglip_demo.py) 完全一致。它同时验证 $b=0$ 和 $b=-1.5$ 的猫狗例子，并在非对称的 $3\times3$ 分数矩阵上做有限差分 `gradcheck`。

```python
"""CLIP and SigLIP losses, with a runnable cat/dog gradient demonstration.

Inputs are floating-point N x N logits: rows are images, columns are texts,
and the diagonal contains matching pairs. Scores already include temperature
scaling; SigLIP's scalar bias is supplied separately.
"""

import torch
from torch import Tensor
from torch.nn import functional as F


def clip_loss(scores: Tensor) -> Tensor:
    """Return the symmetric image-to-text / text-to-image loss."""
    n = scores.shape[0]
    targets = torch.arange(n, device=scores.device)
    image_to_text = F.cross_entropy(scores, targets)
    text_to_image = F.cross_entropy(scores.T, targets)
    return (image_to_text + text_to_image) / 2


def siglip_loss(scores: Tensor, bias: float | Tensor = 0.0) -> Tensor:
    """Return the pairwise sigmoid loss, normalized by N, not N squared."""
    n = scores.shape[0]
    targets = torch.eye(n, device=scores.device, dtype=scores.dtype)
    return F.binary_cross_entropy_with_logits(
        scores + bias, targets, reduction="sum"
    ) / n


def main() -> None:
    torch.set_printoptions(precision=6, sci_mode=False)
    base = torch.tensor([[2.0, 1.0], [1.0, 2.0]], dtype=torch.float64)
    n = base.shape[0]
    targets = torch.eye(n, dtype=base.dtype)
    print(f"PyTorch {torch.__version__}; CPU; float64")
    print("Rows: [cat image, dog image]; columns: [cat text, dog text]")

    for bias_value in (0.0, -1.5):
        print(f"\nShared logit bias = {bias_value}")
        for name in ("CLIP", "SigLIP"):
            # Fresh leaf tensors keep gradients from different runs separate.
            scores = base.clone().requires_grad_(True)
            bias = torch.tensor(bias_value, dtype=base.dtype, requires_grad=True)
            if name == "CLIP":
                # CLIP has no bias parameter; this tests common-shift invariance.
                loss = clip_loss(scores + bias)
                logits = (scores + bias).detach()
                expected = (
                    logits.softmax(dim=1) + logits.softmax(dim=0) - 2 * targets
                ) / (2 * n)
            else:
                loss = siglip_loss(scores, bias)
                expected = (
                    (scores + bias).detach().sigmoid() - targets
                ) / n

            loss.backward()
            torch.testing.assert_close(scores.grad, expected, rtol=0, atol=1e-12)
            torch.testing.assert_close(
                bias.grad, expected.sum(), rtol=0, atol=1e-12
            )
            print(f"{name}: loss = {loss.item():.9f}")
            print("dL/dscores =")
            print(scores.grad)
            print(f"dL/dbias = {bias.grad.item():.9f}")
            print("Autograd matches analytic gradients.")

    # Also compare autograd to finite differences away from the symmetric case.
    probe = torch.tensor(
        [[2.0, 0.1, -0.4], [0.7, 1.5, 0.3], [-0.2, 0.9, 1.8]],
        dtype=torch.float64,
        requires_grad=True,
    )
    probe_bias = torch.tensor(-0.7, dtype=torch.float64, requires_grad=True)
    torch.autograd.gradcheck(clip_loss, (probe,))
    torch.autograd.gradcheck(siglip_loss, (probe, probe_bias))
    print("\nFinite-difference gradcheck passed for both losses (asymmetric N=3).")


if __name__ == "__main__":
    main()
```

### 实现与公式的对应关系

`torch.arange(n)` 生成 CLIP 的类别索引 `[0, 1, ..., n-1]`。第 $i$ 行正确匹配的是第 $i$ 列，因此目标就是 $i$。转置分数矩阵后，同一个标签向量可以用于文本到图像方向。

`torch.eye(n)` 生成 SigLIP 的二分类标签矩阵，对角线为 1，其余位置为 0。二元交叉熵需要与 logits 同形状的浮点标签。

`F.cross_entropy` 直接接收未归一化 logits，内部处理对应计算；不要预先调用 Softmax。[PyTorch 文档](https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.cross_entropy.html)

`F.binary_cross_entropy_with_logits` 直接接收 logits，不要预先调用 Sigmoid。该函数将 Sigmoid 与二元交叉熵结合计算。[PyTorch 文档](https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.binary_cross_entropy_with_logits.html)

`requires_grad_(True)` 让分数矩阵成为需要求导的叶子张量；`loss.backward()` 将梯度写入 `.grad`。每次实验都创建新叶子，避免不同实验的梯度累加。

传入 SigLIP 的 `bias` 可以是普通浮点数，也可以是需要求导的标量张量或 `nn.Parameter`。要学习偏置，应保持它与损失的计算图连接，不能先调用 `.item()` 变成普通数值。

### 接到图像与文本编码器

损失函数接收的 `scores` 已包含温度缩放。实际训练时，分数通常这样构造：

```python
image_vectors = F.normalize(image_features, dim=-1)
text_vectors = F.normalize(text_features, dim=-1)
scores = log_scale.exp() * (image_vectors @ text_vectors.T)

# log_scale 和 bias 应为训练中的可学习参数
loss = siglip_loss(scores, bias)
loss.backward()
```

这里的 `image_features`、`text_features`、`log_scale`、`bias` 由训练模型提供，这个片段说明模块之间的连接方式。独立演示脚本只验证损失与梯度，不包含编码器训练。

## 环境与运行方法

保存到此目录后，已用 Python 3.12.13、PyTorch 2.15.0、CPU 和 `float64` 重新运行。双精度便于检查到较小误差，不代表实际训练必须使用双精度。

### 在自己的 Python 环境中复现

如果需要新建环境，可以在终端依次运行：

```bash
cd "/Users/caochao/Documents/SigLIP"
python3.12 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python clip_siglip_demo.py
./.venv/bin/python -m unittest discover -v
```

这些命令用于后续自行创建环境。本次保存与验证复用了机器上已有的 PyTorch 环境，没有另外安装依赖。

### 演示的实际运行输出

原始输出也保存为 [demo_output.txt](demo_output.txt)。

```text
PyTorch 2.15.0; CPU; float64
Rows: [cat image, dog image]; columns: [cat text, dog text]

Shared logit bias = 0.0
CLIP: loss = 0.313261688
dL/dscores =
tensor([[-0.134471,  0.134471],
        [ 0.134471, -0.134471]], dtype=torch.float64)
dL/dbias = 0.000000000
Autograd matches analytic gradients.
SigLIP: loss = 1.440189699
dL/dscores =
tensor([[-0.059601,  0.365529],
        [ 0.365529, -0.059601]], dtype=torch.float64)
dL/dbias = 0.611855657
Autograd matches analytic gradients.

Shared logit bias = -1.5
CLIP: loss = 0.313261688
dL/dscores =
tensor([[-0.134471,  0.134471],
        [ 0.134471, -0.134471]], dtype=torch.float64)
dL/dbias = 0.000000000
Autograd matches analytic gradients.
SigLIP: loss = 0.948153968
dL/dscores =
tensor([[-0.188770,  0.188770],
        [ 0.188770, -0.188770]], dtype=torch.float64)
dL/dbias = 0.000000000
Autograd matches analytic gradients.

Finite-difference gradcheck passed for both losses (asymmetric N=3).
```

### 测试覆盖与结果

5 项测试分别覆盖：

1. CLIP 猫狗例子的损失值和分数梯度。
2. SigLIP 猫狗例子的损失值、分数梯度和偏置梯度。
3. 非对称矩阵上的 CLIP 双向平均，防止漏算转置方向。
4. 共同偏置下 CLIP 的损失、分数梯度不变，偏置梯度为 0。
5. SigLIP 在 $b=-1.5$ 时的损失、分数梯度和偏置梯度。

非对称的 $2\times2$ 测试矩阵为：

$$
S_{\mathrm{asym}}=\begin{bmatrix}2&0\\1&3\end{bmatrix}.
$$

对应 CLIP 损失为 $0.15392626529447748$。只检查对称的猫狗矩阵，无法发现某些漏算反向损失的错误，因此需要这个补充例子。

演示脚本还使用 `torch.autograd.gradcheck`，在非对称 $3\times3$ 输入上将自动求导结果与有限差分结果对照。两种损失均通过检查。

测试原始输出保存为 [test_results.txt](test_results.txt)。5 项测试全部通过，测试命令退出码为 0。

## 容易混淆的地方

- **SigLIP 仍使用负样本。** 它把负组合放进二分类损失，而不是取消负样本。
- **独立指损失项可以逐对计算。** 编码器、温度和偏置仍是共享参数，不能理解成每对图文各自训练一个模型。
- **共同偏置与温度缩放不同。** CLIP 对全矩阵的共同加法平移不敏感，但改变缩放系数会改变 Softmax 的分布和损失。
- **不用全局 Softmax 不代表没有 $N^2$ 个组合。** 计算全部图文对时依然有平方数量的组合；可分块累计损失，避免一次性保存完整矩阵。
- **Sigmoid 输出不是跨类别归一化的分布。** 一行输出不要求加起来等于 1，也不能仅凭数值就认定为校准良好的现实概率。
- **默认的负样本标签来自配对规则。** 同 batch 中不同配对的内容也可能语义相关，SigLIP 不会自动解决这种假负样本问题。
- **较低的损失数值不等于更好的模型。** CLIP 与 SigLIP 的目标形式不同，应使用相同下游任务、数据和评估方法比较效果。

## SigLIP 2 的延伸

SigLIP 2 发表于 2025 年。在原有图文训练目标之外，它整合了图像描述预训练、自蒸馏、掩码预测和在线数据筛选等方法，提升多语言理解、图文检索、定位和局部视觉特征能力。部分变体支持多分辨率并保留输入图像原有宽高比。本文的损失函数代码对应这里讨论的基础图文目标，不是 SigLIP 2 的完整训练配方。[SigLIP 2 论文](https://arxiv.org/abs/2502.14786)

## 参考资料

1. Zhai 等，2023，[Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343)。损失公式、算法 1、偏置与分块计算。
2. Radford 等，2021，[Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)。CLIP 双向对比学习。
3. Tschannen 等，2025，[SigLIP 2](https://arxiv.org/abs/2502.14786)。后续模型与完整训练配方。
4. [google siglip base patch16 256 模型说明](https://huggingface.co/google/siglip-base-patch16-256)。模型使用与零样本分类示例。
5. [PyTorch cross_entropy](https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.cross_entropy.html)。
6. [PyTorch binary_cross_entropy_with_logits](https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.binary_cross_entropy_with_logits.html)。
