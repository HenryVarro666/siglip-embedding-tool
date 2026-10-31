import json
from types import SimpleNamespace

import pytest
import torch
from test_fusion_benchmark import fixture as gallery_fixture


def inputs(tmp_path):
    args, protocol, identity = gallery_fixture(tmp_path)
    protocol.update(
        {
            "strategies": ["zh", "auto_en", "auto_fusion"],
            "translation_model": "Helsinki-NLP/opus-mt-zh-en",
            "translation_revision": "cf109095479db38d6df799875e34039d4938aaa6",
            "translator_device": "cpu",
            "translation_text_cache": False,
            "translation_generation": {
                "do_sample": False,
                "num_beams": 1,
                "max_new_tokens": 64,
                "use_cache": True,
            },
            "fusion_weights": [0.5, 0.5],
        }
    )
    args.protocol.write_text(json.dumps(protocol))
    args.translation_dir = tmp_path / "mt"
    return args, protocol, identity


def test_complete_live_translation_run_persists_all_stages(tmp_path, monkeypatch):
    import numpy as np

    from siglip_tools import auto_benchmark as runner

    args, _, identity = inputs(tmp_path)

    class Encoder:
        device = torch.device("cpu")
        processor = SimpleNamespace(
            image_processor=SimpleNamespace(size={"height": 224, "width": 224})
        )

        def __init__(self, *a, **kw):
            self.identity = identity

        def encode_texts(self, texts, batch_size=8):
            return np.array([{"中文": [1, 0], "auto": [0, 1]}[t] for t in texts], dtype=np.float32)

    class Translator:
        device = torch.device("cpu")
        model_dir = args.translation_dir

        def __init__(self, *a, **kw):
            self.identity = {
                "source_model": "Helsinki-NLP/opus-mt-zh-en",
                "source_revision": "cf109095479db38d6df799875e34039d4938aaa6",
                "generation": {
                    "do_sample": False,
                    "num_beams": 1,
                    "max_new_tokens": 64,
                    "use_cache": True,
                },
                "artifact_hashes": {},
                "device": "cpu",
                "dtype": "float32",
            }
            self.calls = 0

        def translate(self, text):
            assert text == "中文"  # Curated English must not enter the translator.
            self.calls += 1
            return "auto"

    translator = Translator()
    monkeypatch.setattr(runner, "SigLIPEmbedder", Encoder)
    monkeypatch.setattr(runner, "LocalTranslator", lambda *a, **kw: translator)
    runner.run_benchmark(args)
    result = json.loads((args.output / "result.json").read_text())
    assert result["translations"] == {"i": "auto"}
    assert result["independent_intents"] == 1
    assert result["strategies"]["auto_fusion"]["metrics"]["aggregate"]["average_precision"] == 1
    assert result["strategies"]["auto_en"]["metrics"]["aggregate"]["average_precision"] == 0.5
    assert result["strategies"]["zh"]["metrics"]["aggregate"]["average_precision"] == 0.5
    assert translator.calls == 9  # First1 + auto warm2 + initial2 + live repeats4.
    for policy in result["strategies"].values():
        for key in ["translation_ms", "retrieval_ms", "total_ms"]:
            assert policy["timing"][key]["count"] == 2
    assert result["training_executed"] is False


@pytest.mark.parametrize(
    "field,value",
    [("translation_text_cache", True), ("fusion_weights", [0.2, 0.8]), ("source_token_limit", 512)],
)
def test_invalid_protocol_is_rejected_before_allocating_models(tmp_path, monkeypatch, field, value):
    from siglip_tools import auto_benchmark as runner

    args, protocol, _ = inputs(tmp_path)
    protocol[field] = value
    args.protocol.write_text(json.dumps(protocol))
    monkeypatch.setattr(
        runner, "SigLIPEmbedder", lambda *a, **kw: pytest.fail("Invalid inputs reached model")
    )
    with pytest.raises(ValueError):
        runner.run_benchmark(args)


def test_comparison_requires_identical_translator_and_verbatim_outputs():
    from siglip_tools.auto_benchmark import compare_results
    from siglip_tools.benchmark import MODEL_REVISIONS

    def result(alias):
        mid, rev = MODEL_REVISIONS[alias]
        return {
            "model_alias": alias,
            "model_identity": {"model_id": mid, "revision": rev, "runtime_versions": {}},
            "translator_identity": {"model": "fixed"},
            "translations": {"i": "auto"},
            "dataset_hashes": {},
            "protocol_sha256": "p",
            "protocol": {},
            "implementation_sha256": "c",
            "environment": {},
            "strategies": {
                s: {
                    "metrics": {"aggregate": {"average_precision": 0.5}},
                    "timing": {"total_ms": {"p50": 10}},
                }
                for s in ["zh", "auto_en", "auto_fusion"]
            },
        }

    b, q = result("base"), result("quality")
    assert compare_results(b, q)["quality_minus_base"]["auto_en"]["average_precision"] == 0
    for field in ["translator_identity", "translations", "implementation_sha256"]:
        with pytest.raises(ValueError):
            compare_results(b, {**q, field: {"changed": True}})


@pytest.mark.parametrize("change", ["provenance", "new_file"])
def test_translator_file_guard_detects_provenance_and_inventory_drift(tmp_path, change):
    from siglip_tools.auto_benchmark import check_translator_files, translator_snapshot

    provenance = tmp_path / "translation_provenance.json"
    provenance.write_text("original")
    frozen = translator_snapshot(tmp_path)
    if change == "provenance":
        provenance.write_text("changed")
    else:
        (tmp_path / "extra.json").write_text("new")
    with pytest.raises(ValueError, match="Translator"):
        check_translator_files(tmp_path, frozen)
