"""Retrieve with weighted means of matching-model cosine scores."""

from numbers import Integral

import numpy as np

from .index import SearchResult


def _weights(weights, count):
    values = (
        np.ones(count, dtype=np.float32)
        if weights is None
        else np.asarray(weights, dtype=np.float32)
    )
    if (
        values.shape != (count,)
        or not np.isfinite(values).all()
        or (values < 0).any()
        or not np.isfinite(values.sum())
        or values.sum() <= 0
    ):
        raise ValueError(
            "Weights must be finite, nonnegative and match the queries, with positive sum"
        )
    return values / values.sum()


def weighted_query_vector(query_vectors, weights=None):
    """Average L2-normalized queries; keep the resulting vector unnormalized.

    Dot products with unit gallery vectors then equal weighted mean cosines.
    Opposite queries can yield zero: all gallery scores tie, which is valid.
    """
    vectors = np.asarray(query_vectors, dtype=np.float32)
    if vectors.ndim != 2 or min(vectors.shape) == 0 or not np.isfinite(vectors).all():
        raise ValueError("Queries must form a nonempty finite matrix")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or (norms == 0).any():
        raise ValueError("Every query must have a finite positive norm")
    normalized_weights = _weights(weights, len(vectors))
    return np.sum((vectors / norms) * normalized_weights[:, None], axis=0, dtype=np.float32)


def search_texts(index, encoder, texts, weights=None, top_k=5):
    """Encode supplied queries together and rank by their weighted mean cosine.

    The caller supplies the English text; this function does not translate.
    All texts and gallery vectors use the same checkpoint/processor identity.
    """
    if (
        not isinstance(texts, (list, tuple))
        or not texts
        or any(not isinstance(text, str) or not text.strip() for text in texts)
    ):
        raise ValueError("Texts must be a nonempty list of nonempty strings")
    if isinstance(top_k, bool) or not isinstance(top_k, Integral) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    normalized_weights = _weights(weights, len(texts))
    index.assert_compatible(encoder.identity)
    vectors = encoder.encode_texts(list(texts), batch_size=len(texts))
    if vectors.shape != (len(texts), index.vectors.shape[1]):
        raise ValueError("Query embeddings must match the text count and gallery dimension")
    query = weighted_query_vector(vectors, normalized_weights)
    scores = index.vectors @ query
    ordered = np.argsort(-scores, kind="stable")[:top_k]
    return [
        SearchResult(
            index.records[position].path,
            float(scores[position]),
            rank,
            index.records[position].aliases,
        )
        for rank, position in enumerate(ordered, 1)
    ]
