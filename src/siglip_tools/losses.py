"""Training losses retained separately from pretrained embedding inference."""

import torch
from torch.nn import functional as F


def _batch_size(scores: torch.Tensor) -> int:
    if scores.ndim != 2 or scores.shape[0] == 0 or scores.shape[0] != scores.shape[1]:
        raise ValueError("Training scores must be a nonempty square matrix")
    return scores.shape[0]


def clip_loss(scores: torch.Tensor) -> torch.Tensor:
    n = _batch_size(scores)
    labels = torch.arange(n, device=scores.device)
    return (F.cross_entropy(scores, labels) + F.cross_entropy(scores.T, labels)) / 2


def siglip_loss(scores: torch.Tensor, bias: float | torch.Tensor = 0.0) -> torch.Tensor:
    n = _batch_size(scores)
    labels = torch.eye(n, dtype=scores.dtype, device=scores.device)
    return F.binary_cross_entropy_with_logits(scores + bias, labels, reduction="sum") / n
