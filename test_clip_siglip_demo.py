"""Check the hand-derived losses, score gradients, and scalar bias gradients."""

import unittest

import torch

from clip_siglip_demo import clip_loss, siglip_loss


class LossTests(unittest.TestCase):
    def setUp(self):
        self.scores = torch.tensor(
            [[2.0, 1.0], [1.0, 2.0]], dtype=torch.float64, requires_grad=True
        )

    def test_clip_cat_dog_loss_and_score_gradient(self):
        # Catches wrong targets, sign, or loss scaling.
        loss = clip_loss(self.scores)
        self.assertAlmostEqual(loss.item(), 0.3132616875182228, places=12)
        loss.backward()
        expected = torch.tensor(
            [[-0.13447071068499755, 0.13447071068499755],
             [0.13447071068499755, -0.13447071068499755]],
            dtype=torch.float64,
        )
        torch.testing.assert_close(self.scores.grad, expected, rtol=0, atol=1e-12)

    def test_siglip_cat_dog_loss_score_and_bias_gradients(self):
        # Catches mean over N^2 instead of sum / N, and detached bias.
        bias = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        loss = siglip_loss(self.scores, bias)
        self.assertAlmostEqual(loss.item(), 1.4401896985611953, places=12)
        loss.backward()
        expected = torch.tensor(
            [[-0.05960146101105884, 0.36552928931500245],
             [0.36552928931500245, -0.05960146101105884]],
            dtype=torch.float64,
        )
        torch.testing.assert_close(self.scores.grad, expected, rtol=0, atol=1e-12)
        self.assertAlmostEqual(bias.grad.item(), 0.6118556566078872, places=12)

    def test_clip_averages_both_directions(self):
        # A symmetric matrix cannot detect a missing transpose/direction.
        scores = torch.tensor(
            [[2.0, 0.0], [1.0, 3.0]], dtype=torch.float64, requires_grad=True
        )
        loss = clip_loss(scores)
        self.assertAlmostEqual(loss.item(), 0.15392626529447748, places=12)
        loss.backward()
        expected = torch.tensor(
            [[-0.0970360858480282, 0.04165719879992108],
             [0.0970360858480282, -0.04165719879992108]],
            dtype=torch.float64,
        )
        torch.testing.assert_close(scores.grad, expected, rtol=0, atol=1e-12)

    def test_shared_shift_leaves_clip_loss_and_score_gradient_unchanged(self):
        bias = torch.tensor(-1.5, dtype=torch.float64, requires_grad=True)
        loss = clip_loss(self.scores + bias)
        self.assertAlmostEqual(loss.item(), 0.3132616875182228, places=12)
        loss.backward()
        expected = torch.tensor(
            [[-0.13447071068499755, 0.13447071068499755],
             [0.13447071068499755, -0.13447071068499755]],
            dtype=torch.float64,
        )
        torch.testing.assert_close(self.scores.grad, expected, rtol=0, atol=1e-12)
        self.assertAlmostEqual(bias.grad.item(), 0.0, places=12)

    def test_siglip_bias_changes_loss_and_score_gradient(self):
        bias = torch.tensor(-1.5, dtype=torch.float64, requires_grad=True)
        loss = siglip_loss(self.scores, bias)
        self.assertAlmostEqual(loss.item(), 0.9481539683602134, places=12)
        loss.backward()
        expected = torch.tensor(
            [[-0.1887703343990727, 0.1887703343990727],
             [0.1887703343990727, -0.1887703343990727]],
            dtype=torch.float64,
        )
        torch.testing.assert_close(self.scores.grad, expected, rtol=0, atol=1e-12)
        self.assertAlmostEqual(bias.grad.item(), 0.0, places=12)


if __name__ == "__main__":
    unittest.main()
