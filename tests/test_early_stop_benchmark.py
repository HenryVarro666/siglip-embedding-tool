"""Three live fusion branches retain frozen inputs, rejected evidence and costs."""

import json
from copy import deepcopy
from importlib import import_module

import pytest
from test_term_benchmark import term_fixture


def fixture(tmp_path, monkeypatch, *, legacy_success=False, guard_trigger=True):
    runner = import_module("siglip_tools.early_stop_benchmark")
    args, protocol, encoder, translators = term_fixture(tmp_path)
    original = translators["original"]
    legacy = translators["constrained"]
    guarded = deepcopy(legacy)
    profile = {
        "schema_version": 1,
        "profile_id": "suffix-n8-r3-min24-q4",
        "algorithm": "suffix_consecutive_token_cycles",
        "num_beams": 4,
        "check_every": 4,
        "max_new_tokens": 64,
        "max_ngram_size": 8,
        "repeat_count": 3,
        "min_new_tokens": 24,
        "min_repeating_beams": 4,
    }
    profile_path = tmp_path / "repetition_profile_v1.json"
    profile_path.write_text(json.dumps(profile))
    (args.metadata / "repetition_profile_v1.json").write_bytes(profile_path.read_bytes())
    monkeypatch.setattr(runner, "PROFILE_PATH", profile_path)

    def translate_original(text):
        original.calls += 1
        original.last_trace = {
            "source_text": text,
            "status": "success",
            "sequence": [9, 2],
            "output": "English",
            "eos_terminated": True,
        }
        return "English"

    def primary(translator, early):
        def translate(text):
            translator.calls += 1
            rejected = early and guard_trigger
            failed = rejected or not legacy_success
            translator.last_trace = {
                "source_text": text,
                "status": "failed" if failed else "success",
                "sequence": [9] + [10] * 24 if rejected else [9, 10] if failed else [9, 2],
                "output": "rejected fragment" if failed else "English",
                "eos_terminated": not failed,
                "constraints_applied": True,
                "satisfied": not failed,
            }
            if failed:
                translator.last_trace["error"] = (
                    "repetition rejected" if rejected else "cap rejected"
                )
            if early:
                translator.last_trace["repetition_stop"] = {
                    "triggered": rejected,
                    "profile": deepcopy(profile),
                    "profile_sha256": runner.identity_fingerprint(profile),
                    "special_ids": [2, 9],
                    "observe_only": False,
                    "enabled": True,
                    "trigger_new_tokens": 24 if rejected else None,
                    "generated_new_tokens": 24 if rejected else 1,
                    "observed_generation_steps": 24 if rejected else 1,
                    "observed_steps_complete": rejected,
                    "length_scope": "partial_beam_sequence" if rejected else "finalized_sequence",
                    "snapshots": [
                        {
                            "new_tokens": 24,
                            "beams": [[9] + [10] * 24 for _ in range(4)],
                            "eligible": True,
                            "repeating_beams": 4,
                            "triggered": True,
                            "beam_evidence": [
                                {
                                    "beam_index": i,
                                    "period": 1,
                                    "cycle": [10],
                                    "repeat_count": 24,
                                    "suffix_start": 1,
                                }
                                for i in range(4)
                            ],
                        }
                    ]
                    if rejected
                    else [],
                }
            if failed:
                raise ValueError(translator.last_trace["error"])
            return "English"

        return translate

    original.translate = translate_original
    legacy.translate = primary(legacy, False)
    guarded.translate = primary(guarded, True)
    development_hashes = {}
    for name in runner.DEVELOPMENT_FILES:
        (args.metadata / name).write_text(json.dumps({"fixture": name}))
        development_hashes[name] = runner.file_sha256(args.metadata / name)
    protocol.update(
        {
            "policies": runner.POLICIES,
            "num_beams": {"original": 1, "constrained": 4, "guarded": 4},
            "retrieval_weights": {"base": {"sha256": "fixture-weights"}},
            "failure_policy": runner.FAILURE_POLICY,
            "warm_evidence": runner.WARM_EVIDENCE,
            "implementation_git_commit": "fixture-commit",
            "independent_intents": 1,
            "gallery_images": 3,
            "repetition_profile_sha256": runner.file_sha256(profile_path),
            "development_hashes": development_hashes,
            "evaluation_inputs_sha256": runner.identity_fingerprint(
                {"gallery_ids": ["a", "b", "c"], "qrels": {"i": ["c"]}}
            ),
        }
    )
    protocol["source_indexes"]["quality"] = protocol["source_indexes"]["base"]
    monkeypatch.setattr(runner, "SOURCE_DATA_HASHES", deepcopy(protocol["dataset_hashes"]))
    monkeypatch.setattr(runner, "SOURCE_SUPPORT_HASHES", deepcopy(protocol["supporting_hashes"]))
    monkeypatch.setattr(runner, "QUERY_REPEATS", 2)
    monkeypatch.setattr(runner, "QUERY_WARMUP", 1)
    monkeypatch.setattr(runner, "INTENT_COUNT", 1)
    monkeypatch.setattr(runner, "GALLERY_COUNT", 3)
    monkeypatch.setattr(runner, "validate_development_evidence", lambda *a: True, raising=False)
    monkeypatch.setattr(runner, "implementation_hash", lambda: "fixture-code")
    monkeypatch.setattr(
        runner, "retrieval_weight_snapshot", lambda alias: {"sha256": "fixture-weights"}
    )
    monkeypatch.setattr(runner, "SigLIPEmbedder", encoder)
    monkeypatch.setattr(
        runner,
        "build_translators",
        lambda *a: {"original": original, "constrained": legacy, "guarded": guarded},
    )
    save_freeze(runner, args, protocol)
    return runner, args, protocol, original, legacy, guarded


def save_freeze(runner, args, protocol):
    args.protocol.write_text(json.dumps(protocol))
    (args.metadata / "freeze.json").write_text(
        json.dumps(
            {
                "protocol_sha256": runner.file_sha256(args.protocol),
                "implementation_sha256": "fixture-code",
                **{
                    k: protocol[k]
                    for k in (
                        "dataset_hashes",
                        "supporting_hashes",
                        "repetition_profile_sha256",
                        "development_hashes",
                    )
                },
            }
        )
    )


def run_fixture(tmp_path, monkeypatch, **kwargs):
    runner, args, protocol, original, legacy, guarded = fixture(tmp_path, monkeypatch, **kwargs)
    runner.run_benchmark(args)
    result = json.loads((args.output / "result.json").read_text())
    return runner, args, protocol, result, original, legacy, guarded


def test_three_policies_execute_fresh_primary_and_fallback_with_reconstructable_evidence(
    tmp_path, monkeypatch
):
    runner, _, _, result, original, legacy, guarded = run_fixture(tmp_path, monkeypatch)
    assert list(result["strategies"]) == [
        "original_fusion",
        "fallback_opus_fusion",
        "early_fallback_fusion",
    ]
    assert legacy.calls == guarded.calls == 5
    assert original.calls == 15
    early = result["strategies"]["early_fallback_fusion"]
    assert early["rates"]["early_rejection"] == {"numerator": 1, "denominator": 1, "rate": 1}
    assert early["warm_rates"]["fallback_success"] == {"numerator": 2, "denominator": 2, "rate": 1}
    assert early["metrics"]["aggregate"]["average_precision"] == 1
    assert early["attempts"]["i"]["query_texts"] == ["中文", "English"]
    assert early["attempts"]["i"]["constraint_trace"]["output"] == "rejected fragment"
    assert set(early["warm_samples"]["i"][0]["timing"]) == set(runner.TIMING_KEYS)
    assert result["early_stop_analysis"]["caught_legacy_failure_ids"] == ["i"]
    assert result["training_executed"] is False
    assert runner.validate_saved_result(result)


def test_early_rejection_of_legacy_accepted_is_reported_and_allowed(tmp_path, monkeypatch):
    _, _, _, result, _, _, _ = run_fixture(tmp_path, monkeypatch, legacy_success=True)
    assert result["early_stop_analysis"]["early_rejected_legacy_accepted_ids"] == ["i"]
    assert result["early_stop_analysis"]["caught_legacy_failure_ids"] == []
    assert result["strategies"]["fallback_opus_fusion"]["rates"]["primary_rejection"]["rate"] == 0


def test_nontrigger_guard_preserves_legacy_trace_and_missed_failure_accounting(
    tmp_path, monkeypatch
):
    runner, _, _, result, _, _, _ = run_fixture(tmp_path, monkeypatch, guard_trigger=False)
    assert result["early_stop_analysis"]["missed_legacy_failure_ids"] == ["i"]
    assert runner.audit_branches(result["strategies"])["checked"] > 0
    trace = result["strategies"]["early_fallback_fusion"]["attempts"]["i"]["constraint_trace"]
    trace["sequence"] = [42]
    with pytest.raises(ValueError, match="primary|trace|evidence"):
        runner.audit_branches(result["strategies"])


@pytest.mark.parametrize(
    "changed", ["profile", "development", "support", "weights", "code", "index"]
)
def test_preflight_rejects_frozen_artifact_drift_before_model_allocation(
    tmp_path, monkeypatch, changed
):
    runner, args, _, _, _, _ = fixture(tmp_path, monkeypatch)
    if changed == "profile":
        runner.PROFILE_PATH.write_text("{}")
    elif changed == "development":
        (args.metadata / "dev-selection.json").write_text("{}")
    elif changed == "support":
        (args.metadata / "review-rules.json").write_text("{} changed")
    elif changed == "weights":
        monkeypatch.setattr(runner, "retrieval_weight_snapshot", lambda alias: {"sha256": "wrong"})
    elif changed == "code":
        monkeypatch.setattr(runner, "implementation_hash", lambda: "wrong")
    else:
        args.index.write_bytes(b"wrong")
    monkeypatch.setattr(runner, "SigLIPEmbedder", lambda *a, **kw: pytest.fail("allocated model"))
    with pytest.raises(ValueError):
        runner.run_benchmark(args)


def test_coherent_replacement_of_formal_queries_cannot_replace_original_source(
    tmp_path, monkeypatch
):
    runner, args, protocol, _, _, _ = fixture(tmp_path, monkeypatch)
    path = args.metadata / "queries.json"
    document = json.loads(path.read_text())
    document["queries"][0]["text"] = "替换查询"
    path.write_text(json.dumps(document))
    protocol["dataset_hashes"]["queries"] = runner.file_sha256(path)
    save_freeze(runner, args, protocol)
    with pytest.raises(ValueError, match="fixed|source"):
        runner.validate_frozen_inputs(args.metadata, args.protocol)


@pytest.mark.parametrize(
    "changed",
    ["metrics", "warm_digest", "warm_timing", "warm_status", "analysis", "fallback_trace", "rank"],
)
def test_saved_validation_reconstructs_numbers_branches_and_warm_evidence(
    tmp_path, monkeypatch, changed
):
    runner, _, _, result, _, _, _ = run_fixture(tmp_path, monkeypatch)
    early = result["strategies"]["early_fallback_fusion"]
    if changed == "metrics":
        early["metrics"]["aggregate"]["average_precision"] = 0
    elif changed == "warm_digest":
        early["warm_samples"]["i"][0]["evidence_sha256"] = "0" * 64
    elif changed == "warm_timing":
        early["timing"]["total_ms"]["p50"] += 1
    elif changed == "warm_status":
        early["warm_samples"]["i"][0]["early_rejected"] = False
    elif changed == "analysis":
        result["early_stop_analysis"]["caught_legacy_failure_ids"] = []
    elif changed == "fallback_trace":
        early["attempts"]["i"]["fallback_trace"]["sequence"] = [1, 2]
    else:
        early["attempts"]["i"]["rankings"].reverse()
    with pytest.raises(ValueError):
        runner.validate_saved_result(result)


def test_all_failures_remain_zero_and_no_trigger_has_null_conditional_rate(tmp_path, monkeypatch):
    runner, args, _, original, _, _ = fixture(tmp_path, monkeypatch)

    def failed(text):
        original.calls += 1
        original.last_trace = {
            "source_text": text,
            "status": "failed",
            "sequence": [9, 8],
            "output": "bad original",
            "eos_terminated": False,
            "error": "original cap",
        }
        raise ValueError("original cap")

    original.translate = failed
    runner.run_benchmark(args)
    result = json.loads((args.output / "result.json").read_text())
    for strategy in result["strategies"].values():
        assert strategy["failed_intents"] == 1
        assert strategy["success_conditional_metrics"] is None
        assert all(v == 0 for v in strategy["metrics"]["aggregate"].values())
    assert result["strategies"]["original_fusion"]["rates"]["fallback_success"] == {
        "numerator": 0,
        "denominator": 0,
        "rate": None,
    }


def test_comparison_rejects_cross_model_semantic_and_model_slot_drift(tmp_path, monkeypatch):
    runner, _, _, base, _, _, _ = run_fixture(tmp_path, monkeypatch)
    quality = deepcopy(base)
    quality["model_alias"] = "quality"
    model_id, revision = runner.MODEL_REVISIONS["quality"]
    quality["model_identity"].update({"model_id": model_id, "revision": revision})
    assert (
        runner.compare_results(base, quality)["quality_minus_base"]["original_fusion"][
            "average_precision"
        ]
        == 0
    )
    for key in ("request_semantics", "source_index_sha256", "repetition_profile_sha256"):
        bad = deepcopy(quality)
        bad[key] = "wrong"
        with pytest.raises(ValueError):
            runner.compare_results(base, bad)


def test_invalid_coherently_frozen_profile_is_rejected_before_model_allocation(
    tmp_path, monkeypatch
):
    runner, args, protocol, _, _, _ = fixture(tmp_path, monkeypatch)
    profile = json.loads(runner.PROFILE_PATH.read_text())
    profile["min_repeating_beams"] = 1
    runner.PROFILE_PATH.write_text(json.dumps(profile))
    (args.metadata / "repetition_profile_v1.json").write_bytes(runner.PROFILE_PATH.read_bytes())
    protocol["repetition_profile_sha256"] = runner.file_sha256(runner.PROFILE_PATH)
    save_freeze(runner, args, protocol)
    monkeypatch.setattr(runner, "SigLIPEmbedder", lambda *a, **kw: pytest.fail("allocated model"))
    with pytest.raises(ValueError, match="profile|grid"):
        runner.run_benchmark(args)


@pytest.mark.parametrize("changed", ["development", "source", "profile_hash", "code"])
def test_saved_result_cannot_rebind_frozen_artifact_identity(tmp_path, monkeypatch, changed):
    runner, _, _, result, _, _, _ = run_fixture(tmp_path, monkeypatch)
    if changed == "development":
        result["development_hashes"]["dev-selection.json"] = "wrong"
    elif changed == "source":
        result["dataset_hashes"]["queries"] = "wrong"
        result["protocol"]["dataset_hashes"]["queries"] = "wrong"
    elif changed == "profile_hash":
        result["repetition_profile_sha256"] = "wrong"
        result["protocol"]["repetition_profile_sha256"] = "wrong"
    else:
        result["implementation_sha256"] = "wrong"
        result["protocol"]["implementation_sha256"] = "wrong"
    with pytest.raises(ValueError):
        runner.validate_saved_result(result)


@pytest.mark.parametrize("changed", ["quorum", "step", "missing", "disabled"])
def test_guard_validation_replays_actual_partial_trigger_evidence(tmp_path, monkeypatch, changed):
    runner, _, _, result, _, _, _ = run_fixture(tmp_path, monkeypatch)
    attempt = result["strategies"]["early_fallback_fusion"]["attempts"]["i"]
    trace = attempt["constraint_trace"]["repetition_stop"]
    if changed == "quorum":
        trace["snapshots"][0]["beams"][3] = [9] + list(range(20, 44))
    elif changed == "step":
        trace["trigger_new_tokens"] = 64
    elif changed == "missing":
        trace["snapshots"] = []
    else:
        trace["enabled"] = False
    with pytest.raises(ValueError):
        runner.validate_saved_result(result)


def test_branch_audit_rejects_a_false_effective_route(tmp_path, monkeypatch):
    runner, _, _, result, _, _, _ = run_fixture(tmp_path, monkeypatch)
    strategies = result["strategies"]
    strategies["early_fallback_fusion"]["attempts"]["i"]["effective_route"] = "constrained_fusion"
    with pytest.raises(ValueError, match="route"):
        runner.audit_branches(strategies)


def test_subset_timing_reconstruction_has_stable_intent_order():
    runner = import_module("siglip_tools.early_stop_benchmark")

    class Intent(str):
        def __hash__(self):
            return {"a": 2, "b": 3, "c": 0}[str(self)]

    attempts = {
        Intent(i): {
            "primary_failure": {},
            "early_rejected": True,
            "constraint_trace": {"sequence": [1, 2]},
        }
        for i in ("a", "b", "c")
    }
    samples = {
        i: [{"timing": {k: value for k in runner.TIMING_KEYS}}]
        for i, value in zip(attempts, (1.0, 1.0, 1e16))
    }
    strategies = {p: {"attempts": attempts, "warm_samples": samples} for p in runner.POLICIES}
    result = runner.early_stop_analysis(strategies)
    # a,b,c order preserves the two unit samples before adding the large sample.
    assert (
        result["warm_timing_by_subset"]["legacy_failure"]["original_fusion"]["total_ms"]["mean"]
        == 3333333333333334.0
    )


@pytest.mark.parametrize("changed", ["sequence_count", "missing_checks", "scope", "complete"])
def test_nontrigger_token_evidence_cannot_fabricate_lengths_or_omit_observed_checks(
    tmp_path, monkeypatch, changed
):
    runner, _, _, result, _, _, _ = run_fixture(tmp_path, monkeypatch, guard_trigger=False)
    attempt = result["strategies"]["early_fallback_fusion"]["attempts"]["i"]
    record = attempt["constraint_trace"]["repetition_stop"]
    if changed == "sequence_count":
        record["generated_new_tokens"] = 63
    elif changed == "missing_checks":
        record["observed_generation_steps"] = 60
    elif changed == "scope":
        record["length_scope"] = "partial_beam_sequence"
    else:
        record["observed_steps_complete"] = True
    with pytest.raises(ValueError):
        runner._validate_guard(attempt, result["repetition_profile"])


def test_semantically_invalid_development_selection_rejects_before_model_allocation(
    tmp_path, monkeypatch
):
    runner, args, _, _, _, _ = fixture(tmp_path, monkeypatch)

    def reject_selection(metadata, profile):
        raise ValueError("Selected profile differs from development replay")

    monkeypatch.setattr(runner, "validate_development_evidence", reject_selection)
    monkeypatch.setattr(runner, "SigLIPEmbedder", lambda *a, **kw: pytest.fail("allocated model"))
    with pytest.raises(ValueError, match="Selected profile"):
        runner.run_benchmark(args)
