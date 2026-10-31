import json

import numpy as np
import pytest

from siglip_tools.images import ImageRecord
from siglip_tools.index import ImageIndex

IDENTITY = {"model_id": "google/example", "revision": "abc123", "pipeline_version": 1}
RECORDS = [
    ImageRecord("a.png", "1" * 64, "a" * 64, ("copy.png",)),
    ImageRecord("b.png", "2" * 64, "b" * 64),
    ImageRecord("c.png", "3" * 64, "c" * 64),
]
VECTORS = np.array([[1, 0], [0, 1], [0, 1]], dtype=np.float32)


def make_index(tmp_path):
    return ImageIndex(VECTORS, RECORDS, IDENTITY, tmp_path)


def test_exact_cosine_search_is_stable_for_ties_and_caps_k(tmp_path):
    result = make_index(tmp_path).search(np.array([0, 3]), top_k=10)
    assert [item.path for item in result] == ["b.png", "c.png", "a.png"]
    assert [item.rank for item in result] == [1, 2, 3]
    np.testing.assert_allclose([item.score for item in result], [1, 1, 0])


def test_search_excludes_query_pixels_including_aliases(tmp_path):
    result = make_index(tmp_path).search([1, 0], top_k=3, exclude_pixel_sha256="a" * 64)
    assert [item.path for item in result] == ["b.png", "c.png"]


@pytest.mark.parametrize("query", [[0, 0], [float("nan"), 1], [1, 0, 0], [[1, 0], [0, 1]]])
def test_search_rejects_invalid_query_vectors(tmp_path, query):
    with pytest.raises(ValueError):
        make_index(tmp_path).search(query)


@pytest.mark.parametrize("k", [0, -1, 1.5])
def test_search_requires_positive_integer_top_k(tmp_path, k):
    with pytest.raises(ValueError):
        make_index(tmp_path).search([1, 0], top_k=k)


def test_same_dimensions_do_not_make_different_checkpoints_compatible(tmp_path):
    index = make_index(tmp_path)
    index.assert_compatible(dict(IDENTITY))
    with pytest.raises(ValueError, match="[Ii]ncompatible"):
        index.assert_compatible({**IDENTITY, "revision": "different"})


@pytest.mark.parametrize(
    "vectors",
    [
        np.array([[0, 0], [0, 1], [0, 1]]),
        np.array([[2, 0], [0, 1], [0, 1]]),
        np.array([[float("nan"), 0], [0, 1], [0, 1]]),
        np.ones((2, 2)),
        np.ones((3, 2, 1)),
    ],
)
def test_index_rejects_invalid_vectors_or_record_count(tmp_path, vectors):
    with pytest.raises(ValueError):
        ImageIndex(vectors, RECORDS, IDENTITY, tmp_path)


@pytest.mark.parametrize(
    "bad_path",
    [
        "../outside.png",
        "/absolute.png",
        "a/../../outside.png",
        "C:/outside.png",
        "C:outside.png",
        "folder/\x00.png",
    ],
)
def test_index_rejects_unsafe_record_paths(tmp_path, bad_path):
    records = [ImageRecord(bad_path, "1" * 64, "a" * 64)]
    with pytest.raises(ValueError, match="relative"):
        ImageIndex(np.array([[1, 0]]), records, IDENTITY, tmp_path)


def test_index_round_trip_preserves_vectors_records_and_identity(tmp_path):
    path = tmp_path / "gallery.npz"
    make_index(tmp_path).save(path)
    loaded = ImageIndex.load(path)
    assert isinstance(loaded, ImageIndex)
    np.testing.assert_array_equal(loaded.vectors, VECTORS)
    assert loaded.records == tuple(RECORDS)
    assert loaded.model_identity == IDENTITY
    assert loaded.search([1, 0])[0].path == "a.png"


def test_save_refuses_to_overwrite_existing_file(tmp_path):
    path = tmp_path / "gallery.npz"
    path.write_bytes(b"keep this file")
    with pytest.raises(FileExistsError):
        make_index(tmp_path).save(path)
    assert path.read_bytes() == b"keep this file"


def test_explicit_overwrite_produces_a_complete_readable_index(tmp_path):
    path = tmp_path / "gallery.npz"
    path.write_bytes(b"old")
    make_index(tmp_path).save(path, overwrite=True)
    loaded = ImageIndex.load(path)
    assert isinstance(loaded, ImageIndex)
    assert len(loaded.records) == 3
    assert not list(tmp_path.glob(".siglip-*.tmp"))


def test_load_rejects_unknown_schema(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez(path, embeddings=VECTORS, metadata=np.array(json.dumps({"schema_version": 999})))
    with pytest.raises(ValueError, match="schema"):
        ImageIndex.load(path)


def test_load_rejects_pickle_arrays(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez(path, embeddings=np.array([object()], dtype=object), metadata=np.array("{}"))
    with pytest.raises(ValueError):
        ImageIndex.load(path)


@pytest.mark.parametrize("bad_path", ["C:/outside.png", "folder/\x00.png"])
def test_load_rejects_cross_platform_unsafe_paths(tmp_path, bad_path):
    path = tmp_path / "tampered.npz"
    original = tmp_path / "original.npz"
    make_index(tmp_path).save(original)
    with np.load(original, allow_pickle=False) as stored:
        metadata = json.loads(stored["metadata"].item())
    metadata["records"][0]["path"] = bad_path
    np.savez(path, embeddings=VECTORS, metadata=np.array(json.dumps(metadata)))
    with pytest.raises(ValueError, match="relative"):
        ImageIndex.load(path)
