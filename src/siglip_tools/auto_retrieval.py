"""Actual per-request translation plus image retrieval with stage timing."""

from time import perf_counter

from .benchmark import _sync
from .fusion import search_texts

MODES = ("zh", "auto_en", "auto_fusion")


def timed_search(index, encoder, translator, chinese, mode="auto_en", top_k=5):
    """Time a resident-model request, executing translation on every auto call.

    Translation includes tokenization/generation/decoding. Total is measured
    directly and includes translation, text encoding, score fusion and sorting.
    Model/index loading is performed by the caller, outside this request.
    """
    if mode not in MODES or (mode != "zh" and translator is None):
        raise ValueError("Expected zh, auto_en or auto_fusion and an automatic translator")
    _sync(encoder.device)
    if mode != "zh":
        _sync(translator.device)
    started = perf_counter()
    english = None
    translation_ms = 0.0
    if mode != "zh":
        english = translator.translate(chinese)
        _sync(translator.device)
        translation_ms = (perf_counter() - started) * 1000
    texts = [chinese] if mode == "zh" else ([english] if mode == "auto_en" else [chinese, english])
    retrieval_started = perf_counter()
    matches = search_texts(index, encoder, texts, top_k=top_k)
    _sync(encoder.device)
    finished = perf_counter()
    return {
        "matches": matches,
        "translation": english,
        "timing": {
            "translation_ms": translation_ms,
            "retrieval_ms": (finished - retrieval_started) * 1000,
            "total_ms": (finished - started) * 1000,
        },
    }
