import json

import pytest
from test_auto_benchmark import inputs


def test_nllb_protocol_is_accepted_only_when_selected(tmp_path):
    from siglip_tools import auto_benchmark as runner

    args, protocol, _ = inputs(tmp_path)
    args.translator = "nllb"
    protocol.update(
        {
            "translation_backend": "nllb",
            "translation_model": "facebook/nllb-200-1.3B",
            "translation_revision": "b0de46b488af0cf31749cd8da5ed3171e11b2309",
        }
    )
    args.protocol.write_text(json.dumps(protocol))
    assert len(runner.load_inputs(args)[-1]) == 1
    args.translator = "opus"
    with pytest.raises(ValueError):
        runner.load_inputs(args)


def comparison_result(translator):
    from siglip_tools.benchmark import MODEL_REVISIONS

    sources = {
        "opus": ("Helsinki-NLP/opus-mt-zh-en", "cf109095479db38d6df799875e34039d4938aaa6"),
        "nllb": ("facebook/nllb-200-1.3B", "b0de46b488af0cf31749cd8da5ed3171e11b2309"),
    }
    mid, rev = MODEL_REVISIONS["base"]
    mtid, mtrev = sources[translator]
    return {
        "model_alias": "base",
        "model_identity": {"model_id": mid, "revision": rev},
        "translator_alias": translator,
        "translator_identity": {
            "source_model": mtid,
            "source_revision": mtrev,
            "runtime_versions": {"sentencepiece": "fixed"},
        },
        "dataset_hashes": {},
        "implementation_sha256": "code",
        "environment": {},
        "source_index_sha256": "same-gallery-vectors",
        "protocol": {
            "dataset_hashes": {},
            "strategies": ["zh", "auto_en", "auto_fusion"],
            "device": "cpu",
            "translator_device": "cpu",
            "dtype": "float32",
            "query_warmup": 3,
            "query_repeats": 10,
            "source_token_limit": 256,
            "fusion_weights": [0.5, 0.5],
            "score_atol": 1e-5,
            "translation_text_cache": False,
            "training": False,
            "translation_generation": {
                "do_sample": False,
                "num_beams": 1,
                "max_new_tokens": 64,
                "use_cache": True,
            },
            "translation_model": mtid,
            "translation_revision": mtrev,
        },
        "strategies": {
            s: {
                "metrics": {
                    "aggregate": {
                        "average_precision": 0.5 if translator == "opus" or s == "zh" else 0.75
                    }
                },
                "timing": {"total_ms": {"p50": 10 if translator == "opus" else 20}},
                "rankings": {"i": [{"image_id": "a", "score": 0.1}]},
            }
            for s in ["zh", "auto_en", "auto_fusion"]
        },
    }


def test_cross_translator_comparison_preserves_retrieval_identity_and_common_protocol():
    from siglip_tools.auto_benchmark import compare_translators

    opus, nllb = comparison_result("opus"), comparison_result("nllb")
    result = compare_translators(opus, nllb)
    assert result["nllb_minus_opus"]["auto_en"]["average_precision"] == 0.25
    assert result["nllb_to_opus_total_p50_ratio"]["auto_en"] == 2
    for field in ["model_identity", "source_index_sha256", "environment"]:
        with pytest.raises(ValueError):
            compare_translators(opus, {**nllb, field: "changed"})
    nllb["protocol"]["query_repeats"] = 1
    with pytest.raises(ValueError):
        compare_translators(opus, nllb)


def test_cross_translator_comparison_rejects_chinese_baseline_drift():
    from siglip_tools.auto_benchmark import compare_translators

    opus, nllb = comparison_result("opus"), comparison_result("nllb")
    nllb["strategies"]["zh"]["rankings"]["i"][0]["score"] = 0.5
    with pytest.raises(ValueError, match="score"):
        compare_translators(opus, nllb)


def test_cross_translator_comparison_rejects_translation_runtime_drift():
    from siglip_tools.auto_benchmark import compare_translators

    opus, nllb = comparison_result("opus"), comparison_result("nllb")
    nllb["translator_identity"]["runtime_versions"]["sentencepiece"] = "changed"
    with pytest.raises(ValueError, match="runtime"):
        compare_translators(opus, nllb)


def test_cli_exposes_translator_selection_without_model_loading(capsys):
    from siglip_tools.cli import main

    with pytest.raises(SystemExit) as err:
        main(["search-text", "--help"])
    assert err.value.code == 0
    assert "--translator" in capsys.readouterr().out
