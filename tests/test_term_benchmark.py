from copy import deepcopy
from importlib import import_module

import pytest
import torch
from test_fusion import Encoder as BaseEncoder
from test_fusion import index_at


class Encoder(BaseEncoder):
    device = torch.device("cpu")


class Translator:
    device = torch.device("cpu")

    def __init__(self):
        self.calls = 0

    def translate(self, text):
        assert text == "中文"
        self.calls += 1
        return "English"


def test_timed_attempt_measures_actual_total_and_translates_every_time(tmp_path, monkeypatch):
    runner = import_module("siglip_tools.term_benchmark")
    translator = Translator()
    ticks = iter([10, 11, 12, 15, 20, 21, 22, 25])
    monkeypatch.setattr(runner, "perf_counter", lambda: next(ticks))
    for _ in range(2):
        result = runner.timed_attempt(
            index_at(tmp_path), Encoder(), translator, "中文", "auto_en", 3
        )
        assert result["ok"] is True
        assert result["translation"] == "English"
        assert result["matches"][0].path == "1.png"
        assert result["timing"] == {
            "translation_ms": 1000,
            "retrieval_ms": 3000,
            "total_ms": 5000,
        }
    assert translator.calls == 2


def test_translation_failure_is_explicit_and_does_not_retrieve(tmp_path, monkeypatch):
    runner = import_module("siglip_tools.term_benchmark")

    class Fails(Translator):
        def translate(self, text):
            raise ValueError("missing required bushing")

    ticks = iter([10, 11])
    monkeypatch.setattr(runner, "perf_counter", lambda: next(ticks))
    result = runner.timed_attempt(index_at(tmp_path), Encoder(), Fails(), "中文", "auto_en", 3)
    assert result["ok"] is False
    assert result["failure"]["stage"] == "translation"
    assert result["failure"]["message"] == "missing required bushing"
    assert result["matches"] == []
    assert result["timing"] == {
        "translation_ms": 1000,
        "retrieval_ms": 0,
        "total_ms": 1000,
    }


def test_failure_stays_in_unconditional_metric_denominator():
    runner = import_module("siglip_tools.term_benchmark")
    result = runner.evaluate_attempts(
        {"success": ["b", "a"], "failure": None},
        {"success": ["b"], "failure": ["a"]},
        ["a", "b"],
    )
    assert result["metrics"]["aggregate"]["average_precision"] == 0.5
    assert result["metrics"]["per_query"]["failure"]["average_precision"] == 0
    assert result["success_conditional_metrics"]["average_precision"] == 1
    assert result["successful_intents"] == 1
    assert result["failed_intents"] == 1


def test_all_failures_are_zero_with_no_conditional_claim():
    runner = import_module("siglip_tools.term_benchmark")
    result = runner.evaluate_attempts({"q": None}, {"q": ["a"]}, ["a", "b"])
    assert all(v == 0 for v in result["metrics"]["aggregate"].values())
    assert result["success_conditional_metrics"] is None


def test_invalid_partial_ranking_is_not_reclassified_as_a_failure():
    runner = import_module("siglip_tools.term_benchmark")
    with pytest.raises(ValueError, match="gallery"):
        runner.evaluate_attempts({"q": ["a"]}, {"q": ["a"]}, ["a", "b"])


def comparison_fixture(alias):
    runner = import_module("siglip_tools.term_benchmark")
    from siglip_tools.benchmark import MODEL_REVISIONS

    model_id, revision = MODEL_REVISIONS[alias]
    return {
        "model_alias": alias,
        "model_identity": {"model_id": model_id, "revision": revision, "runtime_versions": {}},
        "dataset_hashes": {"queries": "same"},
        "protocol": {
            "source_indexes": {"base": "base-index", "quality": "quality-index"},
            "implementation_sha256": "c",
        },
        "protocol_sha256": "p",
        "implementation_sha256": "c",
        "source_index_sha256": f"{alias}-index",
        "constraint_traces": {v: {"t01": None} for v in ("original", "beam", "constrained")},
        "environment": {},
        "translator_identities": {
            "original": {"id": "opus"},
            "beam": {"id": "beam"},
            "constrained": {"id": "constrained"},
        },
        "translations": {v: {"t01": "same"} for v in ("original", "beam", "constrained")},
        "strategies": {
            p: {
                "metrics": {"aggregate": {"average_precision": 0.5}},
                "timing": {"total_ms": {"p50": 10}},
            }
            for p in runner.POLICIES
        },
    }


def test_comparison_preserves_all_decoding_controls_and_rejects_drift():
    runner = import_module("siglip_tools.term_benchmark")
    b, q = comparison_fixture("base"), comparison_fixture("quality")
    result = runner.compare_results(b, q)
    assert result["quality_minus_base"]["constrained_en"]["average_precision"] == 0
    assert result["within_model"]["base"]["constrained_minus_beam"]["en"]["average_precision"] == 0
    for key in (
        "translations",
        "translator_identities",
        "implementation_sha256",
        "environment",
        "constraint_traces",
    ):
        wrong = deepcopy(q)
        wrong[key] = {"different": True}
        with pytest.raises(ValueError):
            runner.compare_results(b, wrong)


def term_fixture(tmp_path):
    import json
    from types import SimpleNamespace

    from test_fusion_benchmark import fixture as gallery_fixture

    from siglip_tools import term_benchmark as runner
    from siglip_tools.images import file_sha256

    args, protocol, identity = gallery_fixture(tmp_path)
    args.translation_dir = tmp_path / "mt"
    args.decoder_dir = tmp_path / "decoder"
    args.translation_dir.mkdir()
    args.decoder_dir.mkdir()
    (args.translation_dir / "provenance.json").write_text("{}")
    (args.decoder_dir / "provenance.json").write_text("{}")
    for filename in ("annotation-notes.json", "review-rules.json", "label-review.json"):
        (args.metadata / filename).write_text("{}")
    (args.metadata / "glossary.json").write_bytes(runner.GLOSSARY_PATH.read_bytes())
    protocol.update(
        {
            "policies": runner.POLICIES,
            "translation_model": runner.MODEL_ID,
            "translation_revision": runner.REVISION,
            "translator_device": "cpu",
            "fusion_weights": [0.5, 0.5],
            "translation_text_cache": False,
            "source_token_limit": 256,
            "max_new_tokens": 64,
            "num_beams": {
                "original": 1,
                "beam": 4,
                "constrained": 4,
            },
            "do_sample": False,
            "use_cache": True,
            "score_atol": 1e-5,
            "glossary_sha256": file_sha256(runner.GLOSSARY_PATH),
            "implementation_sha256": "fixture-code",
            "translator_files": runner.translator_snapshot(args.translation_dir),
            "decoder_files": runner.translator_snapshot(args.decoder_dir),
            "supporting_hashes": {
                n: file_sha256(args.metadata / n)
                for n in (
                    "annotation-notes.json",
                    "review-rules.json",
                    "glossary.json",
                    "label-review.json",
                )
            },
            "source_indexes": {"base": file_sha256(args.index)},
        }
    )
    args.protocol.write_text(json.dumps(protocol))

    class FixtureEncoder:
        device = torch.device("cpu")
        processor = SimpleNamespace(
            image_processor=SimpleNamespace(size={"height": 224, "width": 224})
        )

        def __init__(self, *a, **kw):
            self.identity = identity

        def encode_texts(self, texts, batch_size=8):
            import numpy as np

            return np.array(
                [{"中文": [1, 0], "English": [0, 1]}[t] for t in texts], dtype="float32"
            )

    class FixtureTranslator(Translator):
        model_dir = args.translation_dir

        def __init__(self, variant):
            super().__init__()
            self.identity = {
                "source_model": runner.MODEL_ID,
                "source_revision": runner.REVISION,
                "device": "cpu",
                "dtype": "float32",
                "generation": {
                    "do_sample": False,
                    "num_beams": 1 if variant == "original" else 4,
                    "max_new_tokens": 64,
                    "use_cache": True,
                },
            }
            self.variant = variant

        def translate(self, text):
            result = super().translate(text)
            if self.variant == "constrained":
                raise ValueError("test constraint unavailable")
            return result

    translators = {v: FixtureTranslator(v) for v in runner.VARIANTS}
    return args, protocol, FixtureEncoder, translators


def test_complete_seven_policy_run_keeps_failed_queries_and_all_live_samples(tmp_path, monkeypatch):
    import json

    from siglip_tools import term_benchmark as runner

    args, _, encoder, translators = term_fixture(tmp_path)
    monkeypatch.setattr(runner, "SigLIPEmbedder", encoder)
    monkeypatch.setattr(runner, "build_translators", lambda *a: translators)
    monkeypatch.setattr(runner, "implementation_hash", lambda: "fixture-code")
    runner.run_benchmark(args)
    result = json.loads((args.output / "result.json").read_text())
    assert list(result["strategies"]) == list(runner.POLICIES)
    assert result["strategies"]["original_en"]["metrics"]["aggregate"]["average_precision"] == 0.5
    assert result["strategies"]["constrained_en"]["metrics"]["aggregate"]["average_precision"] == 0
    assert result["strategies"]["constrained_en"]["failed_intents"] == 1
    assert result["strategies"]["constrained_en"]["failed_warm_requests"] == 2
    assert result["translations"]["constrained"]["i"] is None
    assert all(t.calls == 9 for t in translators.values())
    for strategy in result["strategies"].values():
        assert strategy["timing"]["total_ms"]["count"] == 2
    assert result["training_executed"] is False


@pytest.mark.parametrize(
    "field,value",
    [("translation_text_cache", True), ("max_new_tokens", 128), ("fusion_weights", [0.8, 0.2])],
)
def test_frozen_protocol_rejected_before_model_load(tmp_path, monkeypatch, field, value):
    import json

    from siglip_tools import term_benchmark as runner

    args, protocol, _, _ = term_fixture(tmp_path)
    protocol[field] = value
    args.protocol.write_text(json.dumps(protocol))
    monkeypatch.setattr(runner, "implementation_hash", lambda: "fixture-code")
    monkeypatch.setattr(runner, "SigLIPEmbedder", lambda *a, **kw: pytest.fail("model allocated"))
    with pytest.raises(ValueError):
        runner.run_benchmark(args)


def test_module_compare_entrypoint_executes_after_definitions(tmp_path):
    import json
    import subprocess
    import sys

    for alias in ("base", "quality"):
        (tmp_path / f"{alias}.json").write_text(json.dumps(comparison_fixture(alias)))
    output = tmp_path / "comparison.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "siglip_tools.term_benchmark",
            "compare",
            "--base",
            str(tmp_path / "base.json"),
            "--quality",
            str(tmp_path / "quality.json"),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(output.read_text())["quality_minus_base"]["zh"]["average_precision"] == 0


def test_comparison_rejects_index_that_contradicts_predeclared_slot():
    from siglip_tools import term_benchmark as runner

    b, q = comparison_fixture("base"), comparison_fixture("quality")
    q["source_index_sha256"] = "different-index"
    with pytest.raises(ValueError, match="index"):
        runner.compare_results(b, q)


def test_predeclared_code_mismatch_rejected_before_model_allocation(tmp_path, monkeypatch):
    import json

    from siglip_tools import term_benchmark as runner

    args, protocol, _, _ = term_fixture(tmp_path)
    protocol["implementation_sha256"] = "predeclared-code"
    args.protocol.write_text(json.dumps(protocol))
    monkeypatch.setattr(runner, "implementation_hash", lambda: "other-code")
    monkeypatch.setattr(runner, "SigLIPEmbedder", lambda *a, **kw: pytest.fail("model allocated"))
    with pytest.raises(ValueError, match="implementation"):
        runner.run_benchmark(args)
