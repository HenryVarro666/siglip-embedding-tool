"""Opt-in real-model checks. No training, private photos, or quality claims."""

import os

import numpy as np
import pytest
from PIL import Image

from siglip_tools.encoder import DEFAULT_MODEL, SigLIPEmbedder

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("SIGLIP_RUN_MODEL_TESTS") != "1",
        reason="Set SIGLIP_RUN_MODEL_TESTS=1 after caching the pinned pretrained model",
    ),
]


@pytest.fixture(scope="module")
def encoder():
    return SigLIPEmbedder(
        DEFAULT_MODEL,
        revision="75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
        device="cpu",
        local_files_only=True,
    )


def test_real_model_returns_aligned_normalized_batch_embeddings(encoder):
    images = [Image.new("RGB", (64, 80), "red"), Image.new("RGB", (80, 64), "blue")]
    vectors = encoder.encode_images(images, batch_size=1)
    together = encoder.encode_images(images, batch_size=2)
    texts = encoder.encode_texts(["a red image", "蓝色的图片"], batch_size=1)
    assert vectors.shape == texts.shape == (2, 768)
    assert vectors.dtype == texts.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(texts, axis=1), 1, atol=1e-6)
    np.testing.assert_allclose(vectors, together, atol=1e-5)


def test_real_model_accepts_single_text_without_encoding_characters(encoder):
    text = "a solar inverter"
    batched = encoder.encode_texts([text])
    single = encoder.encode_texts(text)
    assert single.shape == batched.shape == (1, 768)
    np.testing.assert_allclose(single, batched, atol=1e-6)
    with pytest.raises(ValueError, match="tokens"):
        encoder.encode_texts(["solar inverter " * 100])
    with pytest.raises(ValueError, match="at least one"):
        encoder.encode_texts([])
