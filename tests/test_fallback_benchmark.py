"""Contracts: frozen inputs, live fallbacks, evidence and fair denominators."""

import json
from copy import deepcopy
from importlib import import_module

import pytest
from test_term_benchmark import term_fixture


def fixture(tmp_path, monkeypatch):
    runner = import_module("siglip_tools.fallback_benchmark")
    args, protocol, encoder, translators = term_fixture(tmp_path)
    original = translators["original"]
    constrained = translators["constrained"]
    translate = original.translate

    def primary(text):
        constrained.calls += 1
        constrained.last_trace = {
            "source_text": text,
            "status": "failed",
            "sequence": [9, 10],
            "output": "rejected fragment",
            "eos_terminated": False,
            "satisfied": True,
            "error": "test constraint unavailable",
        }
        raise ValueError("test constraint unavailable")

    def fallback(text):
        output = translate(text)
        original.last_trace = {
            "source_text": text,
            "status": "ok",
            "sequence": [9, 2],
            "output": output,
            "eos_terminated": True,
        }
        return output

    constrained.translate = primary
    original.translate = fallback
    protocol.update(
        {
            "policies": runner.POLICIES,
            "num_beams": {"original": 1, "constrained": 4},
            "retrieval_weights": {"base": {"sha256": "fixture-weights"}},
            "failure_policy": runner.FAILURE_POLICY,
            "warm_evidence": runner.WARM_EVIDENCE,
            "implementation_git_commit": "fixture-commit",
            "evaluation_inputs_sha256": runner.identity_fingerprint(
                {
                    "gallery_ids": ["a", "b", "c"],
                    "qrels": {"i": ["c"]},
                }
            ),
        }
    )
    if hasattr(runner, "SOURCE_DATA_HASHES"):
        monkeypatch.setattr(runner, "SOURCE_DATA_HASHES", deepcopy(protocol["dataset_hashes"]))
        monkeypatch.setattr(
            runner, "SOURCE_SUPPORT_HASHES", deepcopy(protocol["supporting_hashes"])
        )
    if hasattr(runner, "QUERY_REPEATS"):
        monkeypatch.setattr(runner, "QUERY_REPEATS", 2)
        monkeypatch.setattr(runner, "QUERY_WARMUP", 1)
    protocol["source_indexes"]["quality"] = protocol["source_indexes"]["base"]
    args.protocol.write_text(json.dumps(protocol))
    (args.metadata / "freeze.json").write_text(
        json.dumps(
            {
                "protocol_sha256": runner.file_sha256(args.protocol),
                "implementation_sha256": "fixture-code",
                "dataset_hashes": protocol["dataset_hashes"],
                "supporting_hashes": protocol["supporting_hashes"],
            }
        )
    )
    monkeypatch.setattr(runner, "implementation_hash", lambda: "fixture-code")
    monkeypatch.setattr(
        runner, "retrieval_weight_snapshot", lambda alias: {"sha256": "fixture-weights"}
    )
    monkeypatch.setattr(runner, "SigLIPEmbedder", encoder)
    monkeypatch.setattr(
        runner,
        "build_translators",
        lambda *a: {
            "original": original,
            "constrained": constrained,
        },
    )
    return runner, args, protocol, original, constrained


def test_live_nine_policy_run_preserves_failed_attempts_and_fallback_evidence(
    tmp_path, monkeypatch
):
    runner, args, _, original, constrained = fixture(tmp_path, monkeypatch)
    runner.run_benchmark(args)
    result = json.loads((args.output / "result.json").read_text())
    s = result["strategies"]
    assert len(s) == 9
    assert s["constrained_en"]["failed_intents"] == 1
    assert s["constrained_en"]["metrics"]["aggregate"]["average_precision"] == 0
    assert s["fallback_opus_en"]["metrics"]["aggregate"]["average_precision"] == 0.5
    assert s["fallback_zh_fusion"]["metrics"] == s["zh"]["metrics"]
    for name in ("fallback_opus_en", "fallback_zh_fusion"):
        strategy = s[name]
        assert strategy["rates"]["primary_rejection"]["rate"] == 1
        assert strategy["rates"]["fallback_trigger"]["rate"] == 1
        assert strategy["rates"]["terminal_failure"]["rate"] == 0
        assert strategy["rates"]["fallback_success"]["denominator"] == 1
        sample = strategy["warm_samples"]["i"][0]
        assert sample["fallback_triggered"] is True
        assert len(sample["evidence_sha256"]) == 64
        assert sample["timing"]["total_ms"] >= sample["timing"]["constraint_translation_ms"]
        assert strategy["timing"]["total_ms"]["count"] == 2
    final = s["fallback_zh_fusion"]["attempts"]["i"]
    assert final["translation"] is None
    assert final["query_texts"] == ["中文"]
    assert final["constraint_trace"]["output"] == "rejected fragment"
    assert original.calls > 6 and constrained.calls > 10  # live, no translation cache
    assert result["training_executed"] is False
    assert result["branch_audit"]["checked"] > 0


@pytest.mark.parametrize("changed", ["protocol", "support", "weights", "code", "index"])
def test_preflight_rejects_drift_before_allocating_model(tmp_path, monkeypatch, changed):
    runner, args, protocol, _, _ = fixture(tmp_path, monkeypatch)
    if changed == "protocol":
        protocol["query_repeats"] += 1
        args.protocol.write_text(json.dumps(protocol))
    elif changed == "support":
        (args.metadata / "review-rules.json").write_text('{"changed": true}')
    elif changed == "weights":
        monkeypatch.setattr(
            runner, "retrieval_weight_snapshot", lambda alias: {"sha256": "changed"}
        )
    elif changed == "code":
        monkeypatch.setattr(runner, "implementation_hash", lambda: "changed")
    else:
        args.index.write_bytes(b"changed")
    monkeypatch.setattr(runner, "SigLIPEmbedder", lambda *a, **kw: pytest.fail("model allocated"))
    with pytest.raises(ValueError):
        runner.run_benchmark(args)


def test_branch_audit_detects_wrong_fallback_rank_even_with_same_terminal_status(
    tmp_path, monkeypatch
):
    runner, args, _, _, _ = fixture(tmp_path, monkeypatch)
    runner.run_benchmark(args)
    result = json.loads((args.output / "result.json").read_text())
    bad = deepcopy(result["strategies"])
    bad["fallback_opus_en"]["attempts"]["i"]["rankings"].reverse()
    with pytest.raises(ValueError, match="rank|branch"):
        runner.audit_branches(bad)


@pytest.mark.parametrize("changed", ["constraint_trace", "primary_failure", "fallback_trace"])
def test_branch_audit_checks_actual_primary_and_fallback_evidence(tmp_path, monkeypatch, changed):
    runner, args, _, _, _ = fixture(tmp_path, monkeypatch)
    runner.run_benchmark(args)
    strategies = json.loads((args.output / "result.json").read_text())["strategies"]
    strategies["fallback_opus_en"]["attempts"]["i"][changed] = {"different": True}
    with pytest.raises(ValueError, match="primary|trace|branch|evidence"):
        runner.audit_branches(strategies)


@pytest.mark.parametrize(
    "changed", ["metrics", "semantics", "warm_hash", "warm_count", "warm_status", "timing"]
)
def test_saved_result_validation_reconstructs_evidence_and_statistics(
    tmp_path, monkeypatch, changed
):
    runner, args, _, _, _ = fixture(tmp_path, monkeypatch)
    runner.run_benchmark(args)
    result = json.loads((args.output / "result.json").read_text())
    strategy = result["strategies"]["fallback_opus_en"]
    if changed == "metrics":
        strategy["metrics"]["aggregate"]["average_precision"] = 0.9
    elif changed == "semantics":
        strategy["attempts"]["i"]["translation"] = "not the saved semantic record"
    elif changed == "warm_hash":
        strategy["warm_samples"]["i"][0]["evidence_sha256"] = "0" * 64
    elif changed == "warm_count":
        strategy["warm_samples"]["i"].pop()
    elif changed == "warm_status":
        strategy["warm_samples"]["i"][0]["fallback_triggered"] = False
    else:
        strategy["timing"]["total_ms"]["p50"] += 10
    with pytest.raises(ValueError):
        runner.validate_saved_result(result)


def test_no_rejection_has_null_conditional_fallback_rate():
    runner = import_module("siglip_tools.fallback_benchmark")
    rates = runner.request_rates(
        [
            {
                "ok": True,
                "primary_failure": None,
                "fallback_triggered": False,
            }
        ]
    )
    assert rates["terminal_failure"] == {"numerator": 0, "denominator": 1, "rate": 0}
    assert rates["fallback_success"] == {"numerator": 0, "denominator": 0, "rate": None}


def test_coherently_changed_query_protocol_and_freeze_cannot_replace_fixed_source(
    tmp_path, monkeypatch
):
    runner, args, protocol, _, _ = fixture(tmp_path, monkeypatch)
    query_path = args.metadata / "queries.json"
    document = json.loads(query_path.read_text())
    document["queries"][0]["text"] = "different source"
    query_path.write_text(json.dumps(document))
    protocol["dataset_hashes"]["queries"] = runner.file_sha256(query_path)
    args.protocol.write_text(json.dumps(protocol))
    freeze_path = args.metadata / "freeze.json"
    freeze = json.loads(freeze_path.read_text())
    freeze.update(
        {
            "dataset_hashes": protocol["dataset_hashes"],
            "protocol_sha256": runner.file_sha256(args.protocol),
        }
    )
    freeze_path.write_text(json.dumps(freeze))
    with pytest.raises(ValueError, match="source|fixed"):
        runner.validate_frozen_inputs(args.metadata, args.protocol)


def comparison_fixture(tmp_path, monkeypatch):
    runner, args, _, _, _ = fixture(tmp_path, monkeypatch)
    runner.run_benchmark(args)
    base = json.loads((args.output / "result.json").read_text())
    quality = deepcopy(base)
    quality["model_alias"] = "quality"
    model_id, revision = runner.MODEL_REVISIONS["quality"]
    quality["model_identity"].update({"model_id": model_id, "revision": revision})
    return base, quality


@pytest.mark.parametrize(
    "changed", ["source_index_sha256", "implementation_sha256", "request_semantics", "protocol"]
)
def test_comparison_rejects_model_slot_code_translation_or_protocol_drift(
    tmp_path, monkeypatch, changed
):
    runner = import_module("siglip_tools.fallback_benchmark")
    b, q = comparison_fixture(tmp_path, monkeypatch)
    assert runner.compare_results(b, q)["quality_minus_base"]["zh"]["average_precision"] == 0
    q[changed] = "changed"
    with pytest.raises(ValueError):
        runner.compare_results(b, q)
