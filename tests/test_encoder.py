from types import SimpleNamespace

import numpy as np
import pytest
import torch

from siglip_tools.encoder import feature_matrix, identity_fingerprint, pooled_features


def test_feature_matrix_returns_normalized_cpu_float32():
    result = feature_matrix(torch.tensor([[3.0, 4.0], [0.0, 2.0]], dtype=torch.float64))
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, [[0.6, 0.8], [0.0, 1.0]], atol=1e-7)


@pytest.mark.parametrize("values", [[[0.0, 0.0]], [[float("nan"), 1]], [[float("inf"), 1]]])
def test_feature_matrix_rejects_zero_or_nonfinite_vectors(values):
    with pytest.raises(ValueError, match="[Ff]inite|[Zz]ero"):
        feature_matrix(torch.tensor(values))


def test_feature_matrix_rejects_patch_tokens_as_global_embeddings():
    with pytest.raises(ValueError, match="two-dimensional"):
        feature_matrix(torch.ones(2, 4, 8))


def test_pooled_features_handles_tensor_and_model_output():
    pooled = torch.tensor([[3.0, 4.0]])
    assert pooled_features(pooled) is pooled
    result = SimpleNamespace(pooler_output=pooled, last_hidden_state=torch.zeros(1, 5, 2))
    assert pooled_features(result) is pooled


def test_pooled_features_requires_an_actual_pooled_embedding():
    with pytest.raises(ValueError, match="pooled"):
        pooled_features(SimpleNamespace(last_hidden_state=torch.ones(1, 5, 2)))


def test_identity_fingerprint_is_order_independent_but_checkpoint_sensitive():
    first = {"revision": "aaa", "model_id": "google/example"}
    reordered = {"model_id": "google/example", "revision": "aaa"}
    other = {"model_id": "google/example", "revision": "bbb"}
    assert identity_fingerprint(first) == identity_fingerprint(reordered)
    assert identity_fingerprint(first) != identity_fingerprint(other)
