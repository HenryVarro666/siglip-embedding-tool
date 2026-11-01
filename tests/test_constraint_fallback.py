from copy import deepcopy

import numpy as np
import pytest
import torch
from test_fusion import Encoder as FixtureEncoder
from test_fusion import index_at


class Clock:
    def __init__(self):
        self.now = 0.0

    def advance(self, seconds):
        self.now += seconds


class Encoder(FixtureEncoder):
    device = torch.device("cpu")

    def __init__(self, clock=None):
        self.clock = clock
        self.queries = []

    def encode_texts(self, texts, batch_size=8):
        self.queries.append(list(texts))
        if self.clock:
            self.clock.advance(0.010)
        values = {"中文": [1, 0], "English": [0, 1], "fresh English": [1, 0]}
        return np.array([values[text] for text in texts], dtype=np.float32)


class Translator:
    device = torch.device("cpu")
    identity = {"source_model": "fixture/opus", "source_revision": "fixed"}

    def __init__(self, *, reject=False, outputs=("English",), clock=None, duration=0.030):
        self.reject = reject
        self.outputs = outputs
        self.clock = clock
        self.duration = duration
        self.calls = []
        self.last_trace = {}

    def translate(self, text):
        output = self.outputs[min(len(self.calls), len(self.outputs) - 1)]
        self.calls.append(text)
        if self.clock:
            self.clock.advance(self.duration)
        self.last_trace = {
            "source_text": text,
            "sequence": [0, 71, 72] if self.reject else [0, 71, 2],
            "raw_decoded_output": output,
            "output": output,
            "eos_terminated": not self.reject,
            "status": "failed" if self.reject else "success",
        }
        if self.reject:
            self.last_trace["error"] = "missing EOS"
            raise ValueError("missing EOS")
        return output


def search(index, encoder, constrained, original, **options):
    from siglip_tools.constraint_fallback import timed_fallback_search

    return timed_fallback_search(index, encoder, constrained, original, "中文", **options)


@pytest.mark.parametrize(
    "mode,route,path,queries",
    [
        ("auto_en", "original_en", "1.png", ["English"]),
        ("auto_fusion", "original_fusion", "2.png", ["中文", "English"]),
    ],
)
def test_rejected_output_uses_fresh_original_translation_and_real_scores(
    tmp_path, mode, route, path, queries
):
    encoder = Encoder()
    constrained = Translator(reject=True, outputs=("rejected English",))
    original = Translator()
    result = search(index_at(tmp_path), encoder, constrained, original, mode=mode, fallback="opus")
    assert result["ok"] is True
    assert result["translation"] == "English"
    assert result["matches"][0].path == path
    assert result["effective_route"] == route
    assert result["fallback_triggered"] is True
    assert result["primary_failure"] == {
        "stage": "constraint_translation",
        "type": "ValueError",
        "message": "missing EOS",
    }
    assert result["failure"] is None
    assert result["constraint_trace"]["output"] == "rejected English"
    assert result["fallback_trace"]["output"] == "English"
    assert result["query_texts"] == queries
    assert encoder.queries == [queries]


def test_repeated_request_does_not_cache_original_translation(tmp_path):
    original = Translator(outputs=("English", "fresh English"))
    constrained = Translator(reject=True)
    results = [
        search(index_at(tmp_path), Encoder(), constrained, original, fallback="opus")
        for _ in range(2)
    ]
    assert [result["translation"] for result in results] == ["English", "fresh English"]
    assert [result["matches"][0].path for result in results] == ["1.png", "0.png"]


@pytest.mark.parametrize("mode", ["auto_en", "auto_fusion"])
def test_chinese_recovery_encodes_only_one_query_and_retains_rejection(tmp_path, mode):
    encoder = Encoder()
    result = search(
        index_at(tmp_path),
        encoder,
        Translator(reject=True, outputs=("rejected English",)),
        None,
        mode=mode,
        fallback="chinese",
    )
    assert result["ok"] is True
    assert result["translation"] is None
    assert result["matches"][0].path == "0.png"
    assert result["effective_route"] == "zh"
    assert result["query_texts"] == ["中文"]
    assert encoder.queries == [["中文"]]
    assert result["constraint_trace"]["output"] == "rejected English"
    assert result["fallback_trace"] is None
    assert result["fallback_triggered"] is True
    assert result["timing"]["fallback_translation_ms"] == 0.0


@pytest.mark.parametrize("fallback", ["error", "opus", "chinese"])
def test_primary_success_skips_fallback_and_keeps_requested_route(tmp_path, fallback):
    class Unused(Translator):
        def translate(self, text):
            pytest.fail("Successful constrained translation must skip fallback")

    result = search(
        index_at(tmp_path), Encoder(), Translator(), Unused(), mode="auto_fusion", fallback=fallback
    )
    assert result["ok"] is True
    assert result["matches"][0].score == pytest.approx(0.7)
    assert result["effective_route"] == "constrained_fusion"
    assert result["primary_failure"] is None
    assert result["fallback_triggered"] is False
    assert result["fallback_trace"] is None
    assert result["timing"]["fallback_translation_ms"] == 0.0


def test_default_error_returns_failure_without_searching_rejected_english(tmp_path):
    encoder = Encoder()
    result = search(index_at(tmp_path), encoder, Translator(reject=True), None)
    assert result["ok"] is False
    assert result["matches"] == []
    assert result["query_texts"] == []
    assert result["translation"] is None
    assert result["effective_route"] is None
    assert result["failure"] == result["primary_failure"]
    assert result["fallback_triggered"] is False
    assert encoder.queries == []
    assert result["timing"]["retrieval_ms"] == 0.0


def test_original_rejection_terminates_without_cascading_to_chinese(tmp_path):
    encoder = Encoder()
    result = search(
        index_at(tmp_path),
        encoder,
        Translator(reject=True),
        Translator(reject=True, outputs=("fallback fragment",)),
        fallback="opus",
    )
    assert result["ok"] is False
    assert result["failure"]["stage"] == "fallback_translation"
    assert result["primary_failure"]["stage"] == "constraint_translation"
    assert result["fallback_trace"]["output"] == "fallback fragment"
    assert result["translation"] is None
    assert result["effective_route"] is None
    assert result["query_texts"] == []
    assert result["matches"] == []
    assert encoder.queries == []


def test_total_includes_both_attempts_and_handling_but_excludes_diagnostic_copy(
    tmp_path, monkeypatch
):
    from siglip_tools import constraint_fallback

    clock = Clock()
    monkeypatch.setattr(constraint_fallback, "perf_counter", lambda: clock.now)

    class TimedError(ValueError):
        def __str__(self):
            clock.advance(0.007)
            return "missing EOS"

    class Trace(dict):
        def __deepcopy__(self, memo):
            clock.advance(0.500)
            return deepcopy(dict(self), memo)

    class Rejected(Translator):
        def translate(self, text):
            try:
                return super().translate(text)
            except ValueError:
                self.last_trace = Trace(self.last_trace)
                raise TimedError() from None

    constrained = Rejected(reject=True, clock=clock)
    result = search(
        index_at(tmp_path),
        Encoder(clock),
        constrained,
        Translator(clock=clock, duration=0.020),
        fallback="opus",
    )
    assert result["timing"] == pytest.approx(
        {
            "constraint_translation_ms": 30.0,
            "fallback_translation_ms": 20.0,
            "translation_ms": 50.0,
            "retrieval_ms": 10.0,
            "total_ms": 67.0,
        }
    )
    assert clock.now == pytest.approx(0.567)
    constrained.last_trace["output"] = "changed after request"
    assert result["constraint_trace"]["output"] == "English"


@pytest.mark.parametrize("chinese", [None, 3, "", "  ", "English only"])
def test_invalid_input_is_rejected_before_translation_even_with_a_stale_rejected_trace(
    tmp_path, chinese
):
    from siglip_tools.constraint_fallback import timed_fallback_search

    constrained = Translator(reject=True)
    constrained.last_trace = {
        "source_text": chinese,
        "status": "failed",
        "output": "old",
        "sequence": [0, 1],
    }
    with pytest.raises(ValueError):
        timed_fallback_search(
            index_at(tmp_path), Encoder(), constrained, Translator(), chinese, fallback="opus"
        )
    assert constrained.calls == []


@pytest.mark.parametrize("variant", ["stale", "wrong_source", "no_sequence", "no_output"])
def test_only_current_generated_rejection_can_trigger_fallback(tmp_path, variant):
    class Broken(Translator):
        def __init__(self):
            super().__init__()
            self.last_trace = {
                "source_text": "中文",
                "status": "failed",
                "output": "old",
                "sequence": [0, 1],
            }

        def translate(self, text):
            if variant != "stale":
                self.last_trace = dict(self.last_trace)
            if variant == "wrong_source":
                self.last_trace["source_text"] = "其他中文"
            if variant == "no_sequence":
                self.last_trace["sequence"] = None
            if variant == "no_output":
                self.last_trace["output"] = None
            raise ValueError("input or infrastructure failure")

    with pytest.raises(ValueError, match="input or infrastructure failure"):
        search(index_at(tmp_path), Encoder(), Broken(), Translator(), fallback="opus")


@pytest.mark.parametrize(
    "options",
    [
        {"mode": "zh"},
        {"mode": "bad"},
        {"fallback": "other"},
        {"top_k": 0},
        {"top_k": True},
        {"top_k": 1.5},
    ],
)
def test_invalid_options_do_not_start_translation(tmp_path, options):
    constrained = Translator()
    with pytest.raises(ValueError):
        search(index_at(tmp_path), Encoder(), constrained, Translator(), **options)
    assert constrained.calls == []


@pytest.mark.parametrize("constrained,original", [(None, Translator()), (Translator(), None)])
def test_opus_recovery_requires_both_translators(tmp_path, constrained, original):
    with pytest.raises(ValueError):
        search(index_at(tmp_path), Encoder(), constrained, original, fallback="opus")


@pytest.mark.parametrize("reject", [False, True])
def test_retrieval_rejection_does_not_trigger_another_translation(tmp_path, reject):
    class BadEncoder(Encoder):
        def encode_texts(self, texts, batch_size=8):
            raise ValueError("query too long")

    original = Translator()
    result = search(
        index_at(tmp_path), BadEncoder(), Translator(reject=reject), original, fallback="opus"
    )
    assert result["ok"] is False
    assert result["failure"]["stage"] == "retrieval"
    assert result["failure"]["message"] == "query too long"
    assert result["fallback_triggered"] is reject
    assert original.calls == (["中文"] if reject else [])
    assert result["translation"] == "English"
    assert result["query_texts"] == ["English"]


@pytest.mark.parametrize("stage", ["constraint", "fallback", "retrieval"])
def test_runtime_errors_propagate_in_every_stage(tmp_path, stage):
    class BrokenTranslator(Translator):
        def translate(self, text):
            raise RuntimeError("device unavailable")

    class BrokenEncoder(Encoder):
        def encode_texts(self, texts, batch_size=8):
            raise RuntimeError("device unavailable")

    with pytest.raises(RuntimeError, match="device unavailable"):
        search(
            index_at(tmp_path),
            BrokenEncoder() if stage == "retrieval" else Encoder(),
            BrokenTranslator() if stage == "constraint" else Translator(reject=stage == "fallback"),
            BrokenTranslator() if stage == "fallback" else Translator(),
            fallback="opus",
        )


def test_fallback_infrastructure_value_error_propagates(tmp_path):
    class Broken(Translator):
        def translate(self, text):
            self.last_trace = {
                "source_text": text,
                "status": "failed",
                "output": None,
                "sequence": None,
                "error": "tokenizer missing",
            }
            raise ValueError("tokenizer missing")

    with pytest.raises(ValueError, match="tokenizer missing"):
        search(index_at(tmp_path), Encoder(), Translator(reject=True), Broken(), fallback="opus")
