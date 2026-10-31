import json

import numpy as np
import pytest

from siglip_tools.images import ImageRecord
from siglip_tools.index import ImageIndex

IDENTITY = {"model_id": "test/model", "revision": "frozen", "processor_sha256": "recipe"}
GALLERY = np.array([[1, 0], [0, 1], [0.6, 0.8]], dtype=np.float32)


def index_at(root):
    return ImageIndex(
        GALLERY,
        [ImageRecord(f"{i}.png", str(i + 1) * 64, str(i + 4) * 64) for i in range(3)],
        IDENTITY,
        root,
    )


class Encoder:
    identity = IDENTITY
    device = "cpu"

    def encode_texts(self, texts, batch_size=8):
        vectors = {"中文": [1, 0], "English": [0, 1]}
        return np.array([vectors[text] for text in texts], dtype=np.float32)


def test_equal_fusion_returns_mean_cosines_without_rescaling(tmp_path):
    from siglip_tools.fusion import search_texts

    matches = search_texts(index_at(tmp_path), Encoder(), ["中文", "English"], top_k=3)
    assert [match.path for match in matches] == ["2.png", "0.png", "1.png"]
    np.testing.assert_allclose([match.score for match in matches], [0.7, 0.5, 0.5], atol=1e-7)
    assert [match.rank for match in matches] == [1, 2, 3]


def test_single_language_matches_existing_cosine_search(tmp_path):
    from siglip_tools.fusion import search_texts

    index = index_at(tmp_path)
    assert search_texts(index, Encoder(), ["中文"], top_k=3) == index.search([1, 0], top_k=3)


def test_weights_apply_to_normalized_queries_and_are_normalized_by_sum():
    from siglip_tools.fusion import weighted_query_vector

    vector = weighted_query_vector([[3, 0], [0, 4]], [1, 3])
    np.testing.assert_allclose(vector, [0.25, 0.75])


def test_cancelling_queries_produce_stable_zero_score_ties(tmp_path):
    from siglip_tools.fusion import search_texts

    class OppositeEncoder(Encoder):
        def encode_texts(self, texts, batch_size=8):
            return np.array([[1, 0], [-1, 0]], dtype=np.float32)

    matches = search_texts(index_at(tmp_path), OppositeEncoder(), ["中文", "English"], top_k=3)
    assert [match.path for match in matches] == ["0.png", "1.png", "2.png"]
    assert [match.score for match in matches] == [0, 0, 0]


@pytest.mark.parametrize("weights", [[0, 0], [-1, 2], [1], [np.nan, 1], [np.inf, 1]])
def test_invalid_weights_are_rejected(weights):
    from siglip_tools.fusion import weighted_query_vector

    with pytest.raises(ValueError):
        weighted_query_vector([[1, 0], [0, 1]], weights)


@pytest.mark.parametrize("vectors", [[], [1, 0], [[0, 0]], [[np.nan, 1]], [[np.inf, 1]]])
def test_invalid_query_vectors_are_rejected(vectors):
    from siglip_tools.fusion import weighted_query_vector

    with pytest.raises(ValueError):
        weighted_query_vector(vectors)


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5])
def test_invalid_top_k_is_rejected(tmp_path, top_k):
    from siglip_tools.fusion import search_texts

    with pytest.raises(ValueError):
        search_texts(index_at(tmp_path), Encoder(), ["中文"], top_k=top_k)


@pytest.mark.parametrize("texts", [[], "中文", None, [""], ["中文", 1]])
def test_invalid_text_collections_are_rejected(tmp_path, texts):
    from siglip_tools.fusion import search_texts

    with pytest.raises(ValueError):
        search_texts(index_at(tmp_path), Encoder(), texts)


def test_same_dimension_does_not_bypass_identity(tmp_path):
    from siglip_tools.fusion import search_texts

    class WrongEncoder(Encoder):
        identity = {**IDENTITY, "revision": "other"}

        def encode_texts(self, *args, **kwargs):
            pytest.fail("Identity must be checked before encoding")

    with pytest.raises(ValueError, match="Incompatible"):
        search_texts(index_at(tmp_path), WrongEncoder(), ["中文", "English"])


def test_cli_dual_query_uses_fusion_and_persists_mean_score(tmp_path, capsys, monkeypatch):
    from siglip_tools import cli

    path = tmp_path / "gallery.npz"
    index_at(tmp_path).save(path)
    monkeypatch.setattr(cli, "SigLIPEmbedder", lambda *args: Encoder())
    assert (
        cli.main(
            [
                "search-text",
                "--index",
                str(path),
                "--query",
                "中文",
                "--english-query",
                "English",
                "--top-k",
                "3",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert [row["path"] for row in payload["results"]] == ["2.png", "0.png", "1.png"]
    assert payload["results"][0]["score"] == pytest.approx(0.7)
    assert payload["query_fusion"]["method"] == "mean_cosine"
