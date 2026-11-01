"""Time an actual constrained attempt and an explicit rejection recovery route."""

from copy import deepcopy
from numbers import Integral
from time import perf_counter

from .benchmark import _sync
from .fusion import search_texts
from .translation import _CHINESE


def _rejected_output(trace, previous, chinese):
    """Accept only a newly recorded decoded generation from this request."""
    return (
        isinstance(trace, dict)
        and trace is not previous
        and trace.get("source_text") == chinese
        and trace.get("status") == "failed"
        and isinstance(trace.get("sequence"), list)
        and bool(trace["sequence"])
        and isinstance(trace.get("output"), str)
    )


def _failure(stage, exc):
    return {"stage": stage, "type": type(exc).__name__, "message": str(exc)}


def timed_fallback_search(
    index, encoder, constrained, original, chinese, mode="auto_en", fallback="error", top_k=5
):
    """Run fresh translation and retrieval, retaining rejected-output evidence.

    Loading is the caller's responsibility. Total is an independent synchronized
    wall-clock interval including rejection handling and recovery. Stage times
    cover each model operation; trace copies happen after the total ends.
    Input, infrastructure and program errors propagate. A retrieval ValueError
    is a terminal failure and never causes another translation attempt.
    """
    if mode not in ("auto_en", "auto_fusion") or fallback not in ("error", "opus", "chinese"):
        raise ValueError("Expected auto_en/auto_fusion and error/opus/chinese fallback")
    if constrained is None or (fallback == "opus" and original is None):
        raise ValueError(
            "A constrained translator and the selected fallback translator are required"
        )
    if not isinstance(chinese, str) or not chinese.strip() or not _CHINESE.search(chinese):
        raise ValueError("Translation requires a nonempty query containing Chinese text")
    if isinstance(top_k, bool) or not isinstance(top_k, Integral) or top_k < 1:
        raise ValueError("top_k must be a positive integer")

    _sync(encoder.device)
    _sync(constrained.device)
    previous = getattr(constrained, "last_trace", None)
    started = perf_counter()
    english, matches, texts, route = None, [], [], None
    primary_failure, failure, fallback_trace = None, None, None
    fallback_triggered = False
    fallback_ms, retrieval_ms = 0.0, 0.0
    suffix = "en" if mode == "auto_en" else "fusion"
    try:
        english = constrained.translate(chinese)
    except ValueError as exc:
        _sync(constrained.device)
        constraint_ms = (perf_counter() - started) * 1000
        constraint_trace = getattr(constrained, "last_trace", None)
        if not _rejected_output(constraint_trace, previous, chinese):
            raise
        primary_failure = _failure("constraint_translation", exc)
        failure = primary_failure
    else:
        _sync(constrained.device)
        constraint_ms = (perf_counter() - started) * 1000
        constraint_trace = getattr(constrained, "last_trace", None)
        route = f"constrained_{suffix}"
        texts = [english] if mode == "auto_en" else [chinese, english]

    if primary_failure is not None and fallback != "error":
        fallback_triggered = True
        if fallback == "chinese":
            texts, route, failure = [chinese], "zh", None
        else:
            previous = getattr(original, "last_trace", None)
            fallback_started = perf_counter()
            _sync(original.device)
            try:
                english = original.translate(chinese)
            except ValueError as exc:
                _sync(original.device)
                fallback_ms = (perf_counter() - fallback_started) * 1000
                fallback_trace = getattr(original, "last_trace", None)
                if not _rejected_output(fallback_trace, previous, chinese):
                    raise
                failure = _failure("fallback_translation", exc)
            else:
                _sync(original.device)
                fallback_ms = (perf_counter() - fallback_started) * 1000
                fallback_trace = getattr(original, "last_trace", None)
                failure, route = None, f"original_{suffix}"
                texts = [english] if mode == "auto_en" else [chinese, english]

    if texts:
        retrieval_started = perf_counter()
        try:
            matches = search_texts(index, encoder, texts, top_k=top_k)
        except ValueError as exc:
            _sync(encoder.device)
            retrieval_ms = (perf_counter() - retrieval_started) * 1000
            failure = _failure("retrieval", exc)
        else:
            _sync(encoder.device)
            retrieval_ms = (perf_counter() - retrieval_started) * 1000
    finished = perf_counter()
    return {
        "ok": failure is None,
        "translation": english,
        "matches": matches,
        "failure": deepcopy(failure),
        "primary_failure": deepcopy(primary_failure),
        "constraint_trace": deepcopy(constraint_trace),
        "fallback_trace": deepcopy(fallback_trace),
        "fallback_triggered": fallback_triggered,
        "effective_route": route,
        "query_texts": list(texts),
        "timing": {
            "constraint_translation_ms": constraint_ms,
            "fallback_translation_ms": fallback_ms,
            "translation_ms": constraint_ms + fallback_ms,
            "retrieval_ms": retrieval_ms,
            "total_ms": (finished - started) * 1000,
        },
    }
