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
