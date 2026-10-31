import json

import pytest
import torch
from PIL import Image

from siglip_tools.finetune import read_pairs
from siglip_tools.losses import clip_loss, siglip_loss


@pytest.mark.parametrize(
    "loss_fn, expected",
    [
        (clip_loss, 0.3132616875182228),
        (siglip_loss, 1.4401896985611953),
    ],
)
def test_packaged_training_losses_preserve_original_values(loss_fn, expected):
    scores = torch.tensor([[2.0, 1.0], [1.0, 2.0]], dtype=torch.float64, requires_grad=True)
    loss = loss_fn(scores)
    assert loss.item() == pytest.approx(expected, abs=1e-12)
    loss.backward()
    assert scores.grad[0, 0] < 0
    assert scores.grad[0, 1] > 0


@pytest.mark.parametrize("loss_fn", [clip_loss, siglip_loss])
@pytest.mark.parametrize("shape", [(0, 0), (2, 3), (2, 2, 1)])
def test_training_losses_require_nonempty_square_scores(loss_fn, shape):
    with pytest.raises(ValueError, match="square"):
        loss_fn(torch.zeros(shape))


def test_training_pairs_resolve_images_relative_to_manifest(tmp_path):
    image = tmp_path / "a.png"
    Image.new("RGB", (2, 2)).save(image)
    manifest = tmp_path / "pairs.jsonl"
    manifest.write_text(json.dumps({"image": "a.png", "text": "a red cabinet"}) + "\n")
    pairs = read_pairs(manifest)
    assert len(pairs) == 1
    assert pairs[0][0] == image.resolve()
    assert pairs[0][1] == "a red cabinet"


@pytest.mark.parametrize(
    "record",
    [{"image": "missing.png", "text": "cat"}, {"image": "a.png", "text": ""}, {"text": "cat"}],
)
def test_training_pairs_reject_missing_images_or_captions(tmp_path, record):
    manifest = tmp_path / "bad.jsonl"
    manifest.write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="line 1"):
        read_pairs(manifest)
