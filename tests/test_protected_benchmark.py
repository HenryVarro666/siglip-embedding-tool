"""Four fresh request branches bind protection, complete cost and reconstructable evidence."""

import importlib
import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest
from test_term_benchmark import term_fixture


def runner_module():
    assert importlib.util.find_spec("siglip_tools.protected_benchmark") is not None, (
        "The separate completed-protection benchmark is not implemented"
    )
    return importlib.import_module("siglip_tools.protected_benchmark")


def test_protection_rate_does_not_count_suppressed_detection_as_actual_rejection():
    runner = runner_module()
    attempts = [
        {
            "primary_failure": None,
            "fallback_triggered": False,
            "ok": True,
            "early_rejected": False,
            "candidate_protected": True,
        },
        {
            "primary_failure": {"stage": "constraint_translation"},
            "fallback_triggered": True,
            "ok": False,
            "early_rejected": True,
            "candidate_protected": False,
        },
    ]
    rates = runner.request_rates(attempts)
    assert rates["candidate_protection"] == {"numerator": 1, "denominator": 2, "rate": 0.5}
    assert rates["early_rejection"] == {"numerator": 1, "denominator": 2, "rate": 0.5}
    assert rates["fallback_success"] == {"numerator": 0, "denominator": 1, "rate": 0.0}


def _protected_trace(runner, *, protected=True, abort=False):
    """A single genuine EOS at step 5; it is retained when the raw step-32 check runs."""
    from siglip_tools.repetition_stop import detect_repetition

    profile = json.loads(runner.PROFILE_PATH.read_text())
    beams = [[1] + [7, 8, 10, 8] * 8 for _ in range(4)]
    snapshot = {
        "new_tokens": 32,
        "beams": beams,
        **detect_repetition(beams, profile, [0, 1]),
        "pool_candidate_ids": [1] if protected else [],
        "completed_candidates": [],
        "candidate_pool": "inspected",
        "protected": protected,
        "abort": abort,
    }
    trace = {
        "source_text": "一台逆变器",
        "constrained": True,
        "status": "failed" if abort else "success",
        "sequence": beams[0] if abort else [1, 7, 8, 10, 8, 0],
        "output": "rejected fragment" if abort else "inverter",
        "eos_terminated": not abort,
        "constraints_applied": True,
        "satisfied": not abort,
        "requirements": [
            {
                "id": "inverter",
                "source_terms": ["逆变器"],
                "target_forms": ["inverter", "inverters", "Inverter", "Inverters"],
                "token_alternatives": [[7, 8], [7, 9], [10, 8], [10, 9]],
                "decoder_token_alternatives": [[7, 8], [7, 9], [10, 8], [10, 9]],
            }
        ],
        "repetition_stop": {
            "profile": profile,
            "profile_sha256": runner.identity_fingerprint(profile),
            "special_ids": [0, 1],
            "observe_only": False,
            "enabled": True,
            "triggered": abort,
            "trigger_new_tokens": 32 if abort else None,
            "generated_new_tokens": 32 if abort else 5,
            "observed_generation_steps": 32,
            "observed_steps_complete": abort,
            "length_scope": "partial_beam_sequence" if abort else "finalized_sequence",
            "snapshots": [snapshot],
            "protected_trigger_steps": [32] if protected else [],
            "candidate_protection": {
                "schema_version": 1,
                "rule": json.loads(runner.RULE_PATH.read_text()),
                "rule_sha256": runner.identity_fingerprint(
                    json.loads(runner.RULE_PATH.read_text())
                ),
                "source": "request_local_scorer_process_selected_eos",
                "inspection": "raw_trigger_only",
                "decoder_prompt_len": 1,
                "batch_size": 1,
                "num_beams": 4,
                "eos_token_ids": [0],
                "length_penalty": 1.0,
                "decoder_start_token_id": 1,
                "finalize_entered": not abort,
                "scorer_done_before_finalize": None if abort else False,
                "eos_candidates": [],
                "pool_history": [
                    {
                        "new_tokens": step,
                        "phase": "process",
                        "pool_candidate_ids": [1] if protected and step >= 5 else [],
                    }
                    for step in range(1, 33)
                ],
            },
        },
    }
    trace["required_token_alternatives"] = [
        deepcopy(requirement["token_alternatives"]) for requirement in trace["requirements"]
    ]
    for requirement in trace["requirements"]:
        requirement.update(tokens_satisfied=True, phrase_satisfied=not abort, satisfied=not abort)
    if abort:
        trace["error"] = "repetition rejected"
    if protected:
        validation = {
            "output": "inverter",
            "nonempty": True,
            "satisfied": True,
            "valid": True,
            "requirements": [
                {
                    **deepcopy(trace["requirements"][0]),
                    "tokens_satisfied": True,
                    "phrase_satisfied": True,
                    "satisfied": True,
                }
            ],
        }
        candidate = {
            "candidate_id": 1,
            "prefix": [1, 7, 8, 10, 8],
            "eos_step": 5,
            "normalized_score": -0.2,
            "beam_indices": None,
            "eos_token_ids": [0],
            "witnesses": [
                {
                    "phase": "process",
                    "process_new_tokens": 5,
                    "eos_token_id": 0,
                    "rank": 0,
                    "parent_beam_index": 0,
                    "next_score": -1.0,
                    "parent_prefix": [1, 7, 8, 10, 8],
                    "generated_len": 5,
                    "beam_indices": None,
                }
            ],
        }
        trace["repetition_stop"]["candidate_protection"]["eos_candidates"] = [candidate]
        snapshot["completed_candidates"] = [
            {
                "candidate_id": 1,
                "pool_rank": 0,
                "phase": "process",
                "current_step": 32,
                "retained": True,
                "validation": validation,
            }
        ]
    return trace


def save_freeze(runner, args, protocol):
    args.protocol.write_text(json.dumps(protocol))
    (args.metadata / "freeze.json").write_text(
        json.dumps(
            {
                "protocol_sha256": runner.file_sha256(args.protocol),
                "implementation_sha256": "fixture-code",
                **{
                    key: protocol[key]
                    for key in (
                        "dataset_hashes",
                        "supporting_hashes",
                        "development_hashes",
                        "repetition_profile_sha256",
                        "protection_rule_sha256",
                    )
                },
            }
        )
    )


def fixture(tmp_path, monkeypatch, *, legacy_success=True, protect=True, guard_trigger=True):
    runner = runner_module()
    args, protocol, encoder_class, translators = term_fixture(tmp_path)
    original, legacy = translators["original"], translators["constrained"]
    guarded, protected = deepcopy(legacy), deepcopy(legacy)
    queries_path = args.metadata / "queries.json"
    queries = json.loads(queries_path.read_text())
    queries["queries"][0]["text"] = "一台逆变器"
    queries_path.write_text(json.dumps(queries))
    protocol["dataset_hashes"]["queries"] = runner.file_sha256(queries_path)

    class Encoder(encoder_class):
        def encode_texts(self, texts, batch_size=8):
            import numpy as np

            return np.array(
                [[0, 1] if text == "English" else [1, 0] for text in texts], dtype="float32"
            )

    source_profile = Path(runner.__file__).with_name("repetition_profile_v1.json")
    source_rule = Path(runner.__file__).with_name("completed_protection_v1.json")
    profile_path, rule_path = tmp_path / source_profile.name, tmp_path / source_rule.name
    profile_path.write_bytes(source_profile.read_bytes())
    rule_path.write_bytes(source_rule.read_bytes())
    for path in (profile_path, rule_path):
        (args.metadata / path.name).write_bytes(path.read_bytes())
    monkeypatch.setattr(runner, "PROFILE_PATH", profile_path)
    monkeypatch.setattr(runner, "RULE_PATH", rule_path)
    protected_trace = _protected_trace(runner, protected=protect, abort=not protect)
    legacy_trace = {
        k: deepcopy(v) for k, v in _protected_trace(runner).items() if k != "repetition_stop"
    }
    if not legacy_success:
        legacy_trace.update(
            status="failed",
            satisfied=False,
            eos_terminated=False,
            sequence=[1, 7],
            output="legacy rejected",
            error="cap rejected",
        )
        for requirement in legacy_trace["requirements"]:
            requirement.update(tokens_satisfied=False, phrase_satisfied=False, satisfied=False)
    if protect:
        protected_trace = {
            **deepcopy(legacy_trace),
            "repetition_stop": protected_trace["repetition_stop"],
        }
        protected_trace["repetition_stop"]["generated_new_tokens"] = (
            len(legacy_trace["sequence"]) - 1
        )
    guarded_trace = deepcopy(legacy_trace)
    if guard_trigger:
        guarded_trace = _protected_trace(runner, protected=False, abort=True)
        record = guarded_trace["repetition_stop"]
        record.pop("candidate_protection")
        record.pop("protected_trigger_steps")
        for snapshot in record["snapshots"]:
            for key in (
                "pool_candidate_ids",
                "completed_candidates",
                "candidate_pool",
                "protected",
                "abort",
            ):
                snapshot.pop(key)
    else:
        guarded_trace["repetition_stop"] = {
            "profile": json.loads(profile_path.read_text()),
            "profile_sha256": runner.identity_fingerprint(json.loads(profile_path.read_text())),
            "special_ids": [0, 1],
            "observe_only": False,
            "enabled": True,
            "triggered": False,
            "trigger_new_tokens": None,
            "generated_new_tokens": len(legacy_trace["sequence"]) - 1,
            "observed_generation_steps": 1,
            "observed_steps_complete": False,
            "length_scope": "finalized_sequence",
            "snapshots": [],
        }

    def translate_original(text):
        assert text == "一台逆变器"
        original.calls += 1
        original.last_trace = {
            "source_text": text,
            "constrained": False,
            "constraints_applied": False,
            "requirements": [],
            "required_token_alternatives": [],
            "satisfied": True,
            "status": "success",
            "sequence": [1, 0],
            "output": "English",
            "raw_decoded_output": "English",
            "eos_terminated": True,
        }
        return "English"

    def translate_primary(translator, template):
        def translate(text):
            assert text == "一台逆变器"
            translator.calls += 1
            translator.last_trace = deepcopy(template)
            if template["status"] == "failed":
                raise ValueError(template["error"])
            return template["output"]

        return translate

    original.translate = translate_original
    for translator, template in (
        (legacy, legacy_trace),
        (guarded, guarded_trace),
        (protected, protected_trace),
    ):
        translator.translate = translate_primary(translator, template)
    development_hashes = {}
    for name in runner.DEVELOPMENT_FILES:
        (args.metadata / name).write_text(json.dumps({"fixture": name}))
        development_hashes[name] = runner.file_sha256(args.metadata / name)
    protocol.update(
        {
            "policies": runner.POLICIES,
            "num_beams": {"original": 1, "constrained": 4, "guarded": 4, "protected": 4},
            "retrieval_weights": {"base": {"sha256": "fixture-weights"}},
            "failure_policy": runner.FAILURE_POLICY,
            "warm_evidence": runner.WARM_EVIDENCE,
            "implementation_git_commit": "fixture-commit",
            "independent_intents": 1,
            "gallery_images": 3,
            "repetition_profile_sha256": runner.file_sha256(profile_path),
            "protection_rule_sha256": runner.file_sha256(rule_path),
            "development_hashes": development_hashes,
            "evaluation_inputs_sha256": runner.identity_fingerprint(
                {"gallery_ids": ["a", "b", "c"], "qrels": {"i": ["c"]}}
            ),
        }
    )
    protocol["source_indexes"]["quality"] = protocol["source_indexes"]["base"]
    monkeypatch.setattr(runner, "SOURCE_DATA_HASHES", deepcopy(protocol["dataset_hashes"]))
    monkeypatch.setattr(runner, "SOURCE_SUPPORT_HASHES", deepcopy(protocol["supporting_hashes"]))
    for name, value in (
        ("QUERY_REPEATS", 2),
        ("QUERY_WARMUP", 1),
        ("INTENT_COUNT", 1),
        ("GALLERY_COUNT", 3),
    ):
        monkeypatch.setattr(runner, name, value)
    monkeypatch.setattr(runner, "validate_development_evidence", lambda *a: True)
    monkeypatch.setattr(runner, "implementation_hash", lambda: "fixture-code")
    monkeypatch.setattr(runner, "previous_core_hash", lambda: runner.PREVIOUS_CORE_SHA256)
    monkeypatch.setattr(
        runner, "retrieval_weight_snapshot", lambda alias: {"sha256": "fixture-weights"}
    )
    monkeypatch.setattr(runner, "SigLIPEmbedder", Encoder)
    monkeypatch.setattr(
        runner,
        "build_translators",
        lambda *a: {
            "original": original,
            "constrained": legacy,
            "guarded": guarded,
            "protected": protected,
        },
    )
    save_freeze(runner, args, protocol)
    return runner, args, protocol, original, legacy, guarded, protected


def run_fixture(tmp_path, monkeypatch, **kwargs):
    runner, args, protocol, original, legacy, guarded, protected = fixture(
        tmp_path, monkeypatch, **kwargs
    )
    runner.run_benchmark(args)
    result = json.loads((args.output / "result.json").read_text())
    return runner, args, protocol, result, original, legacy, guarded, protected


def test_four_policies_execute_fresh_primary_and_retained_candidate_does_not_abort(
    tmp_path, monkeypatch
):
    runner, _, _, result, original, legacy, guarded, protected = run_fixture(tmp_path, monkeypatch)
    assert list(result["strategies"]) == [
        "original_fusion",
        "fallback_opus_fusion",
        "early_fallback_fusion",
        "protected_fallback_fusion",
    ]
    assert legacy.calls == guarded.calls == protected.calls == 5
    assert original.calls == 10
    early, new = (
        result["strategies"][p] for p in ("early_fallback_fusion", "protected_fallback_fusion")
    )
    assert new["rates"]["candidate_protection"] == {"numerator": 1, "denominator": 1, "rate": 1}
    assert new["rates"]["early_rejection"]["rate"] == 0
    assert early["rates"]["early_rejection"]["rate"] == 1
    assert (
        new["attempts"]["i"]["constraint_trace"]["repetition_stop"]["snapshots"][0]["triggered"]
        is True
    )
    assert new["attempts"]["i"]["early_rejected"] is False
    assert new["attempts"]["i"]["query_texts"] == ["一台逆变器", "inverter"]
    assert result["protection_analysis"]["restored_old_accepted_ids"] == ["i"]
    assert result["branch_audit"]["checked"] == 3
    assert result["startup"]["first_request_order"] == [
        "original",
        "constrained",
        "guarded",
        "protected",
    ]
    assert set(result["startup"]["first_requests_per_decoder"]) == {
        "original",
        "constrained",
        "guarded",
        "protected",
    }
    assert runner.validate_saved_result(result)


def test_restored_legacy_output_can_change_english_ranking_and_metrics_vs_old_guard(
    tmp_path, monkeypatch
):
    _, _, _, result, *_ = run_fixture(tmp_path, monkeypatch)
    old, new = (
        result["strategies"][p] for p in ("early_fallback_fusion", "protected_fallback_fusion")
    )
    assert old["attempts"]["i"]["translation"] == "English"
    assert new["attempts"]["i"]["translation"] == "inverter"
    assert old["metrics"]["aggregate"]["average_precision"] == 1
    assert new["metrics"]["aggregate"]["average_precision"] == 0.5
    for policy in result["strategies"]:
        assert (
            result["protection_analysis"]["warm_timing_by_subset"]["restored_old_accepted"][policy][
                "total_ms"
            ]["count"]
            == 2
        )


def test_absent_valid_completion_really_aborts_then_runs_fresh_original(tmp_path, monkeypatch):
    _, _, _, result, original, legacy, guarded, protected = run_fixture(
        tmp_path, monkeypatch, protect=False
    )
    assert legacy.calls == guarded.calls == protected.calls == 5
    assert original.calls == 15
    new = result["strategies"]["protected_fallback_fusion"]
    assert new["rates"]["candidate_protection"]["rate"] == 0
    assert new["rates"]["early_rejection"]["rate"] == 1
    assert (
        new["attempts"]["i"]["fallback_trace"]
        == result["strategies"]["original_fusion"]["attempts"]["i"]["original_trace"]
    )
    assert result["protection_analysis"]["new_early_rejected_legacy_accepted_ids"] == ["i"]


def test_protection_can_be_recorded_even_when_final_primary_still_fails(tmp_path, monkeypatch):
    _, _, _, result, *_ = run_fixture(tmp_path, monkeypatch, legacy_success=False)
    new = result["strategies"]["protected_fallback_fusion"]
    assert new["rates"]["candidate_protection"]["rate"] == 1
    assert new["rates"]["primary_rejection"]["rate"] == 1
    assert new["rates"]["early_rejection"]["rate"] == 0
    assert result["protection_analysis"]["protection_without_primary_success_ids"] == ["i"]
    assert result["protection_analysis"]["remaining_legacy_failure_ids"] == ["i"]


@pytest.mark.parametrize(
    "changed", ["profile", "rule", "development", "support", "weights", "code", "old_core", "index"]
)
def test_preflight_rejects_frozen_drift_before_model_allocation(tmp_path, monkeypatch, changed):
    runner, args, _, *_ = fixture(tmp_path, monkeypatch)
    if changed in ("profile", "rule"):
        getattr(runner, "PROFILE_PATH" if changed == "profile" else "RULE_PATH").write_text("{}")
    elif changed in ("development", "support"):
        (
            args.metadata
            / ("dev-validation.json" if changed == "development" else "review-rules.json")
        ).write_text("{} changed")
    elif changed == "weights":
        monkeypatch.setattr(runner, "retrieval_weight_snapshot", lambda alias: {"sha256": "wrong"})
    elif changed in ("code", "old_core"):
        monkeypatch.setattr(
            runner,
            "implementation_hash" if changed == "code" else "previous_core_hash",
            lambda: "wrong",
        )
    else:
        args.index.write_bytes(b"wrong")
    monkeypatch.setattr(runner, "SigLIPEmbedder", lambda *a, **kw: pytest.fail("allocated model"))
    with pytest.raises(ValueError):
        runner.run_benchmark(args)


def test_dev_semantic_replay_happens_before_model_allocation(tmp_path, monkeypatch):
    runner, args, _, *_ = fixture(tmp_path, monkeypatch)

    def reject(metadata, profile):
        raise ValueError("Invalid frozen development replay")

    monkeypatch.setattr(runner, "validate_development_evidence", reject)
    monkeypatch.setattr(runner, "SigLIPEmbedder", lambda *a, **kw: pytest.fail("allocated model"))
    with pytest.raises(ValueError, match="development replay"):
        runner.run_benchmark(args)


@pytest.mark.parametrize(
    "changed",
    [
        "metrics",
        "rates",
        "candidate_flag",
        "early_flag",
        "warm_flag",
        "warm_digest",
        "warm_timing",
        "analysis",
        "rule",
        "branch",
        "candidate",
        "eos",
        "pool",
        "raw_beams",
        "rank",
    ],
)
def test_saved_validation_replays_metrics_branches_protection_and_all_warm_evidence(
    tmp_path, monkeypatch, changed
):
    runner, _, _, result, *_ = run_fixture(tmp_path, monkeypatch)
    new = result["strategies"]["protected_fallback_fusion"]
    attempt = new["attempts"]["i"]
    if changed == "metrics":
        new["metrics"]["aggregate"]["average_precision"] = 0
    elif changed == "rates":
        new["rates"]["candidate_protection"]["numerator"] = 0
    elif changed in ("candidate_flag", "early_flag"):
        attempt["candidate_protected" if changed == "candidate_flag" else "early_rejected"] = (
            changed == "early_flag"
        )
    elif changed == "warm_flag":
        new["warm_samples"]["i"][0]["candidate_protected"] = False
    elif changed == "warm_digest":
        new["warm_samples"]["i"][0]["evidence_sha256"] = "0" * 64
    elif changed == "warm_timing":
        new["timing"]["total_ms"]["p50"] += 1
    elif changed == "analysis":
        result["protection_analysis"]["restored_old_accepted_ids"] = []
    elif changed == "rule":
        result["protection_rule"]["decoder_prompt_len"] = 2
    elif changed == "branch":
        attempt["effective_route"] = "original_fusion"
    elif changed == "rank":
        attempt["rankings"].reverse()
    else:
        record = attempt["constraint_trace"]["repetition_stop"]
        if changed == "candidate":
            record["snapshots"][0]["completed_candidates"][0]["validation"]["output"] = (
                "microinverter"
            )
        elif changed == "eos":
            record["candidate_protection"]["eos_candidates"][0]["witnesses"][0]["phase"] = (
                "finalize"
            )
        elif changed == "pool":
            record["snapshots"][0]["pool_candidate_ids"] = []
        else:
            record["snapshots"][0]["beams"][0] = list(range(33))
    with pytest.raises(ValueError):
        runner.validate_saved_result(result)


def test_comparison_rejects_semantic_rule_source_and_model_slot_drift(tmp_path, monkeypatch):
    runner, _, _, base, *_ = run_fixture(tmp_path, monkeypatch)
    quality = deepcopy(base)
    quality["model_alias"] = "quality"
    model_id, revision = runner.MODEL_REVISIONS["quality"]
    quality["model_identity"].update(model_id=model_id, revision=revision)
    assert (
        runner.compare_results(base, quality)["quality_minus_base"]["protected_fallback_fusion"][
            "average_precision"
        ]
        == 0
    )
    for key in (
        "request_semantics",
        "source_index_sha256",
        "protection_rule_sha256",
        "development_hashes",
    ):
        wrong = deepcopy(quality)
        wrong[key] = "wrong"
        with pytest.raises(ValueError):
            runner.compare_results(base, wrong)


@pytest.mark.parametrize("field,value", [("candidate_protected", 1), ("early_rejected", 0)])
def test_saved_warm_flags_are_boolean_evidence_not_numeric_lookalikes(
    tmp_path, monkeypatch, field, value
):
    runner, _, _, result, *_ = run_fixture(tmp_path, monkeypatch)
    result["strategies"]["protected_fallback_fusion"]["warm_samples"]["i"][0][field] = value
    with pytest.raises(ValueError, match="bool|status|branch"):
        runner.validate_saved_result(result)


@pytest.mark.parametrize("change", ["order", "missing", "protected_trace", "total"])
def test_saved_startup_retains_four_actual_first_request_controls(tmp_path, monkeypatch, change):
    runner, _, _, result, *_ = run_fixture(tmp_path, monkeypatch)
    startup = result["startup"]
    if change == "order":
        startup["first_request_order"].reverse()
    elif change == "missing":
        del startup["first_requests_per_decoder"]["protected"]
    elif change == "total":
        startup["first_requests_per_decoder"]["protected"]["timing"]["total_ms"] = 0
    else:
        startup["first_requests_per_decoder"]["protected"]["constraint_trace"]["repetition_stop"][
            "candidate_protection"
        ]["eos_candidates"][0]["witnesses"][0]["phase"] = "finalize"
    with pytest.raises(ValueError):
        runner.validate_saved_result(result)


def test_full_total_includes_translation_return_but_excludes_recorded_audit_replay(
    tmp_path, monkeypatch
):
    runner, args, _, _, _, _, protected = fixture(tmp_path, monkeypatch)
    clock = {"seconds": 0.0}
    audit_observations = []
    translate, validate = protected.translate, runner._validate_protected

    def now():
        clock["seconds"] += 1.0
        return clock["seconds"]

    def production_translation(text):
        # The translator's observer, lexical decisions, cleanup and return are request work.
        clock["seconds"] += 2.0
        return translate(text)

    def recorded_audit(attempt):
        audit_observations.append((protected.calls, attempt["timing"]["total_ms"]))
        clock["seconds"] += 100.0
        return validate(attempt)

    monkeypatch.setattr(runner, "perf_counter", now)
    monkeypatch.setattr(protected, "translate", production_translation)
    monkeypatch.setattr(runner, "_validate_protected", recorded_audit)
    runner.run_benchmark(args)
    result = json.loads((args.output / "result.json").read_text())
    measured = result["strategies"]["protected_fallback_fusion"]
    assert measured["attempts"]["i"]["timing"]["total_ms"] == 3000.0
    assert [sample["timing"]["total_ms"] for sample in measured["warm_samples"]["i"]] == [
        3000.0,
        3000.0,
    ]
    assert (
        result["startup"]["first_requests_per_decoder"]["protected"]["timing"]["total_ms"] == 3000.0
    )
    # Every fresh request is still audited, after its complete measured interval is saved.
    assert audit_observations[:5] == [
        (1, 3000.0),
        (2, 3000.0),
        (3, 3000.0),
        (4, 3000.0),
        (5, 3000.0),
    ]
    assert all(total == 3000.0 for _, total in audit_observations)
