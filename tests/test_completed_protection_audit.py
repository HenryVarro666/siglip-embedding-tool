"""Independent recorded-data replay rejects forged EOS protection and evidence."""

import hashlib
import importlib.util
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples/audit_completed_protection.py"


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def auditor():
    assert SCRIPT.exists(), "Independent completed-protection auditor is missing"
    return load(SCRIPT)


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def protected_trace(protected=True):
    """One hand-checked 32-step raw trigger, with one genuine 5-step completion."""
    prior = load(ROOT / "tests/test_early_stop_audit.py")
    raw = prior.guard(
        {
            "source_text": "逆变器",
            "status": "success",
            "sequence": [0, 31, 32, 33, 34, 2],
            "output": "inverter",
            "satisfied": True,
            "requirements": [
                {
                    "id": "inverter",
                    "source_terms": ["逆变器"],
                    "target_forms": ["inverter", "Inverter"],
                    "token_alternatives": [[31, 32], [35, 36]],
                    "decoder_token_alternatives": [[31, 32], [35, 36]],
                    "tokens_satisfied": True,
                    "phrase_satisfied": True,
                    "satisfied": True,
                }
            ],
            "required_token_alternatives": [[[31, 32], [35, 36]]],
        },
        True,
    )
    rule = json.loads((ROOT / "src/siglip_tools/completed_protection_v1.json").read_text())
    stop = raw["repetition_stop"]
    candidate = {
        "candidate_id": 1,
        "prefix": [0, 31, 32, 33, 34],
        "eos_step": 5,
        "normalized_score": -1.0,
        "beam_indices": None,
        "eos_token_ids": [2],
        "witnesses": [
            {
                "phase": "process",
                "process_new_tokens": 5,
                "rank": 0,
                "parent_beam_index": 1,
                "parent_prefix": [0, 31, 32, 33, 34],
                "eos_token_id": 2,
                "next_score": -5.0,
                "generated_len": 5,
                "beam_indices": None,
            }
        ],
    }
    stop["candidate_protection"] = {
        "schema_version": 1,
        "rule": rule,
        "rule_sha256": fingerprint(rule),
        "source": "request_local_scorer_process_selected_eos",
        "inspection": "raw_trigger_only",
        "decoder_prompt_len": 1,
        "batch_size": 1,
        "num_beams": 4,
        "eos_token_ids": [2],
        "decoder_start_token_id": 0,
        "length_penalty": 1.0,
        "finalize_entered": protected,
        "scorer_done_before_finalize": False if protected else None,
        "eos_candidates": [candidate],
        "pool_history": [
            {
                "new_tokens": step,
                "phase": "process",
                "pool_candidate_ids": [1] if step >= 5 else [],
            }
            for step in range(1, 33)
        ],
    }
    snapshot = stop["snapshots"][0]
    snapshot.update(
        pool_candidate_ids=[1],
        completed_candidates=[
            {
                "candidate_id": 1,
                "pool_rank": 0,
                "current_step": 32,
                "retained": True,
                "phase": "process",
                "validation": {
                    "output": "inverter" if protected else "inverterish",
                    "nonempty": True,
                    "requirements": deepcopy(raw["requirements"]),
                    "satisfied": protected,
                    "valid": protected,
                },
            }
        ],
        candidate_pool="inspected",
        protected=protected,
        abort=not protected,
    )
    if not protected:
        requirement = snapshot["completed_candidates"][0]["validation"]["requirements"][0]
        requirement.update(phrase_satisfied=False, satisfied=False)
    stop["protected_trigger_steps"] = [32] if protected else []
    stop.update(
        triggered=not protected,
        trigger_new_tokens=None if protected else 32,
        observed_steps_complete=not protected,
        length_scope="finalized_sequence" if protected else "partial_beam_sequence",
    )
    if protected:
        raw.update(
            status="success",
            sequence=[0, 31, 32, 33, 34, 2],
            output="inverter",
            eos_terminated=True,
        )
        raw.pop("sequence_role")
        stop["generated_new_tokens"] = 5
    return raw


@pytest.mark.parametrize("protected", [True, False])
def test_replays_retained_eos_protection_and_actual_abort(protected):
    audit = auditor()
    report = audit.validate_protected_guard(protected_trace(protected), audit.Checks(), "synthetic")
    assert report == {
        "raw_trigger_steps": [32],
        "protected_trigger_steps": [32] if protected else [],
        "abort_step": None if protected else 32,
        "eos_candidates": 1,
    }


@pytest.mark.parametrize(
    "corruption",
    [
        "score",
        "tiny_score",
        "eos_rank",
        "parent",
        "eos_step",
        "finalize_origin",
        "prompt",
        "special",
        "evicted",
        "future",
        "pool_rank",
        "word_boundary",
        "missing_alternative",
        "missing_requirement",
        "nonempty",
        "raw_cycle",
        "actual_abort",
        "missing_check",
        "observed_lower_bound",
        "missing_process",
        "new_id_order",
        "self_hashed_rule",
        "duplicate_rank",
        "constraints_flag",
        "false_terminal_skip",
    ],
)
def test_rejects_coherently_forged_candidate_and_guard_records(corruption):
    trace = protected_trace()
    stop = trace["repetition_stop"]
    observer = stop["candidate_protection"]
    event = observer["eos_candidates"][0]
    witness = event["witnesses"][0]
    snapshot = stop["snapshots"][0]
    checked = snapshot["completed_candidates"][0]
    if corruption == "score":
        event["normalized_score"] = -0.9
    elif corruption == "tiny_score":
        event["normalized_score"] += 1e-10
    elif corruption == "eos_rank":
        witness["rank"] = 4
    elif corruption == "parent":
        witness["parent_beam_index"] = 4
    elif corruption == "eos_step":
        witness["process_new_tokens"] = 6
    elif corruption == "finalize_origin":
        witness["phase"] = "finalize"
    elif corruption == "prompt":
        event["prefix"][0] = 99
        witness["parent_prefix"][0] = 99
    elif corruption == "special":
        event["prefix"][2] = 2
        witness["parent_prefix"][2] = 2
    elif corruption == "evicted":
        observer["pool_history"][-1]["pool_candidate_ids"] = []
    elif corruption == "future":
        observer["pool_history"][3]["pool_candidate_ids"] = [1]
    elif corruption == "pool_rank":
        checked["pool_rank"] = 1
    elif corruption == "word_boundary":
        checked["validation"]["output"] = "inverterish"
    elif corruption == "missing_alternative":
        checked["validation"]["requirements"][0]["token_alternatives"] = [[31, 32]]
    elif corruption == "missing_requirement":
        checked["validation"]["requirements"] = []
    elif corruption == "nonempty":
        checked["validation"]["output"] = " "
    elif corruption == "raw_cycle":
        snapshot["beam_evidence"][0]["period"] = 8
    elif corruption == "actual_abort":
        stop["triggered"] = True
    elif corruption == "missing_check":
        stop["snapshots"] = []
    elif corruption == "observed_lower_bound":
        stop["observed_generation_steps"] = 31
    elif corruption == "missing_process":
        observer["pool_history"].pop(5)
    elif corruption == "new_id_order":
        event["candidate_id"] = 2
    elif corruption == "duplicate_rank":
        duplicate = deepcopy(witness)
        duplicate["parent_beam_index"] = 2
        event["witnesses"].append(duplicate)
    elif corruption == "constraints_flag":
        trace["requirements"] = []
    elif corruption == "false_terminal_skip":
        observer["scorer_done_before_finalize"] = True
    else:
        observer["rule"]["training"] = True
        observer["rule_sha256"] = fingerprint(observer["rule"])
    with pytest.raises(ValueError):
        auditor().validate_protected_guard(trace, auditor().Checks(), "forged")


def test_evicted_genuine_completion_does_not_protect_current_pool():
    trace = protected_trace(False)
    stop = trace["repetition_stop"]
    for record in stop["candidate_protection"]["pool_history"][-2:]:
        record["pool_candidate_ids"] = []
    stop["snapshots"][0].update(pool_candidate_ids=[], completed_candidates=[])
    assert auditor().validate_protected_guard(trace)["abort_step"] == 32


def test_tied_selected_eos_witnesses_preserve_all_actual_ids():
    trace = protected_trace()
    observer = trace["repetition_stop"]["candidate_protection"]
    observer["eos_token_ids"] = [2, 3]
    event = observer["eos_candidates"][0]
    event["eos_token_ids"] = [2, 3]
    tied = deepcopy(event["witnesses"][0])
    tied.update(rank=1, parent_beam_index=2, eos_token_id=3)
    event["witnesses"].append(tied)
    assert auditor().validate_protected_guard(trace)["protected_trigger_steps"] == [32]


def test_no_term_request_has_no_observer_or_protection_state():
    trace = protected_trace()
    trace.update(requirements=[], required_token_alternatives=[], constraints_applied=False)
    stop = trace["repetition_stop"]
    stop.update(
        enabled=False, snapshots=[], protected_trigger_steps=[], observed_generation_steps=0
    )
    stop["candidate_protection"].update(
        eos_candidates=[],
        pool_history=[],
        finalize_entered=False,
        length_penalty=None,
        scorer_done_before_finalize=None,
    )
    report = auditor().validate_protected_guard(trace)
    assert report == {
        "raw_trigger_steps": [],
        "protected_trigger_steps": [],
        "abort_step": None,
        "eos_candidates": 0,
    }


def test_completed_scorer_can_skip_exactly_one_terminal_callback():
    trace = protected_trace()
    observer = trace["repetition_stop"]["candidate_protection"]
    observer["scorer_done_before_finalize"] = True
    observer["pool_history"].append(
        {"new_tokens": 33, "phase": "process", "pool_candidate_ids": [1]}
    )
    assert auditor().validate_protected_guard(trace)["protected_trigger_steps"] == [32]


def test_independent_source_requirement_replay_keeps_longest_context_and_exclusions():
    glossary = {
        "entries": [
            {"id": "micro", "sources": ["微型逆变器"], "target_forms": ["microinverter"]},
            {"id": "inverter", "sources": ["逆变器"], "target_forms": ["inverter"]},
            {
                "id": "wire",
                "sources": ["线"],
                "target_forms": ["wire"],
                "required_context": "电气",
                "exclude_within": ["线槽"],
            },
        ]
    }
    audit = auditor()
    assert callable(getattr(audit, "source_requirements", None)), (
        "Source requirement replay is missing"
    )
    assert audit.source_requirements("微型逆变器和逆变器；电气线槽外有线。", glossary) == [
        {
            "id": "micro",
            "source_terms": ["微型逆变器"],
            "target_forms": ["microinverter", "Microinverter"],
        },
        {"id": "inverter", "source_terms": ["逆变器"], "target_forms": ["inverter", "Inverter"]},
        {"id": "wire", "source_terms": ["线"], "target_forms": ["wire", "Wire"]},
    ]
    assert audit.source_requirements("线槽外有线。", glossary) == []


def test_source_binding_rejects_omitted_original_term_even_with_rehashed_records():
    audit = auditor()
    assert callable(getattr(audit, "bind_trace_requirements", None)), (
        "Original term binding is missing"
    )
    trace = protected_trace()
    glossary = {
        "entries": [{"id": "inverter", "sources": ["逆变器"], "target_forms": ["inverter"]}]
    }
    audit.bind_trace_requirements(trace, glossary, [0, 2], audit.Checks(), "source")
    trace.update(
        requirements=[], required_token_alternatives=[], constraints_applied=False, satisfied=True
    )
    with pytest.raises(ValueError):
        audit.bind_trace_requirements(trace, glossary, [0, 2], audit.Checks(), "source")


@pytest.mark.parametrize("corruption", ["missing_eos", "empty", "final_lexical", "invented_eos"])
def test_protected_candidate_does_not_make_invalid_final_output_successful(corruption):
    trace = protected_trace()
    if corruption == "missing_eos":
        trace["sequence"][-1] = 40
    elif corruption == "empty":
        trace["output"] = " "
    elif corruption == "final_lexical":
        trace["output"] = "inverterish"
    else:
        trace["eos_terminated"] = False
    with pytest.raises(ValueError):
        auditor().validate_protected_guard(trace)


def test_atomic_output_and_failure_record_preserve_existing_evidence(tmp_path):
    audit = auditor()
    output = tmp_path / "audit.json"
    assert (
        audit.main(
            [
                "--base",
                "missing",
                "--quality",
                "missing",
                "--metadata",
                str(tmp_path),
                "--output",
                str(output),
            ]
        )
        == 1
    )
    failed = json.loads(output.read_text())
    assert failed["passed"] is False
    assert failed["failures"]
    preserved = output.read_bytes()
    with pytest.raises(FileExistsError):
        audit.main(["--base", "missing", "--quality", "missing", "--output", str(output)])
    with pytest.raises(FileExistsError):
        audit.save_new(output, {"passed": True})
    assert output.read_bytes() == preserved


def test_log_snapshot_excludes_current_stdout_inode_and_hashes_closed_logs(tmp_path):
    auditor()
    stable, active = tmp_path / "closed.log", tmp_path / "active.log"
    stable.write_text("closed evidence")
    code = (
        "import importlib.util,json,sys; from pathlib import Path; "
        "s=importlib.util.spec_from_file_location('audit',sys.argv[1]); "
        "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
        "print(json.dumps(m.evidence_log_snapshot(Path(sys.argv[2]))))"
    )
    with active.open("w") as stream:
        result = subprocess.run(
            [sys.executable, "-c", code, str(SCRIPT), str(tmp_path)],
            stdout=stream,
            stderr=subprocess.PIPE,
            text=True,
        )
    assert result.returncode == 0, result.stderr
    assert json.loads(active.read_text()) == {
        str(stable): hashlib.sha256(b"closed evidence").hexdigest()
    }


def synthetic_result():
    """Two intents: protection restores an old acceptance; a lexical miss still aborts."""
    prior_tests = load(ROOT / "tests/test_early_stop_audit.py")
    prior_audit = load(ROOT / "examples/audit_early_stop_benchmark.py")
    helper = prior_audit.legacy
    result, inputs, chinese = prior_tests.synthetic_result(early_accept=True)
    result["protocol"]["policies"]["protected_fallback_fusion"] = {
        "decoder": "protected",
        "mode": "auto_fusion",
        "fallback": "opus",
    }
    rule = json.loads((ROOT / "src/siglip_tools/completed_protection_v1.json").read_text())
    result.update(
        protection_rule=rule,
        protection_rule_sha256=hashlib.sha256(
            (ROOT / "src/siglip_tools/completed_protection_v1.json").read_bytes()
        ).hexdigest(),
    )
    strategies = result["strategies"]
    legacy = strategies["fallback_opus_fusion"]["attempts"]["i1"]
    long_trace = protected_trace()
    long_trace["source_text"] = chinese["i1"]
    legacy["constraint_trace"] = {
        k: deepcopy(v) for k, v in long_trace.items() if k != "repetition_stop"
    }
    legacy.update(translation="inverter", query_texts=[chinese["i1"], "inverter"])
    strategies["protected_fallback_fusion"] = deepcopy(strategies["early_fallback_fusion"])
    protected = strategies["protected_fallback_fusion"]["attempts"]
    protected["i1"] = deepcopy(legacy)
    protected["i1"]["constraint_trace"] = long_trace
    protected["i1"]["early_rejected"] = False
    protected["i2"]["constraint_trace"] = protected_trace(False)
    protected["i2"]["constraint_trace"]["source_text"] = chinese["i2"]
    result["translator_identities"] = {
        name: {"generation": {"eos_token_id": 2}}
        for name in ("original", "constrained", "guarded", "protected")
    }
    failed_legacy = strategies["fallback_opus_fusion"]["attempts"]["i2"]["constraint_trace"]
    failed_legacy.update(
        requirements=[], required_token_alternatives=[], constraints_applied=False, satisfied=True
    )
    for intent, attempt in strategies["original_fusion"]["attempts"].items():
        attempt["original_trace"].update(eos_terminated=True, raw_decoded_output="original en")
        for strategy in strategies.values():
            if strategy["attempts"][intent]["fallback_triggered"]:
                strategy["attempts"][intent]["fallback_trace"] = deepcopy(attempt["original_trace"])
    for policy, strategy in strategies.items():
        for intent, attempt in strategy["attempts"].items():
            attempt["candidate_protected"] = (
                policy == "protected_fallback_fusion" and intent == "i1"
            )
            strategy["warm_samples"][intent] = [
                {
                    **{
                        k: deepcopy(attempt[k])
                        for k in (*helper.STATUS, "early_rejected", "candidate_protected", "timing")
                    },
                    "ranking_override": None,
                    "evidence_sha256": fingerprint(
                        {"semantics": helper.semantics(attempt), "rankings": attempt["rankings"]}
                    ),
                }
                for _ in range(2)
            ]
        attempts = list(strategy["attempts"].values())
        warm = [s for values in strategy["warm_samples"].values() for s in values]
        for name, rows in (("rates", attempts), ("warm_rates", warm)):
            strategy[name] = prior_audit.rates(rows)
            strategy[name]["candidate_protection"] = {
                "numerator": sum(a["candidate_protected"] for a in rows),
                "denominator": len(rows),
                "rate": sum(a["candidate_protected"] for a in rows) / len(rows),
            }
        strategy["timing"] = prior_audit.timing(warm)
        strategy["timing_by_branch"] = {
            name: prior_audit.timing([s for s in warm if s["fallback_triggered"] == flag])
            for name, flag in (("fallback_triggered", True), ("no_fallback", False))
            if any(s["fallback_triggered"] == flag for s in warm)
        }
    # The protected control now has AP 1/2 and 1/3, the same rankings as legacy.
    for key in ("metrics", "success_conditional_metrics", "successful_intents", "failed_intents"):
        strategies["protected_fallback_fusion"][key] = deepcopy(
            strategies["fallback_opus_fusion"][key]
        )
    result["request_semantics"] = {
        p: {i: helper.semantics(a) for i, a in s["attempts"].items()} for p, s in strategies.items()
    }
    result["branch_audit"]["checked"] = 6
    result["early_stop_analysis"] = prior_audit.early_analysis(strategies)
    groups = {
        "legacy_failure": ["i2"],
        "old_early_rejected": ["i1", "i2"],
        "protected_early_rejected": ["i2"],
        "restored_old_accepted": ["i1"],
        "candidate_protected": ["i1"],
        "remaining_legacy_failure": [],
        "new_early_rejected_legacy_accepted": [],
        "protected_primary_accepted": ["i1"],
        "protection_without_primary_success": [],
    }
    result["protection_analysis"] = {
        **{k + "_ids": v for k, v in groups.items()},
        "rates": {
            "candidate_protection": {"numerator": 1, "denominator": 2, "rate": 0.5},
            "restored_old_accepted": {"numerator": 1, "denominator": 1, "rate": 1.0},
            "remaining_legacy_failure": {"numerator": 0, "denominator": 1, "rate": 0.0},
            "new_early_rejected_legacy_accepted": {"numerator": 0, "denominator": 1, "rate": 0.0},
        },
        "scored_primary_generated_tokens": {
            p: {
                i: len((a["constraint_trace"] or a["original_trace"])["sequence"]) - 1
                for i, a in s["attempts"].items()
            }
            for p, s in strategies.items()
        },
        "warm_timing_by_subset": {
            group: {
                p: prior_audit.timing([v for i in ids for v in s["warm_samples"][i]])
                for p, s in strategies.items()
            }
            for group, ids in {"all": ["i1", "i2"], **groups}.items()
            if ids
        },
        "protected_vs_old_timing": {
            "old_early_fallback_fusion": deepcopy(strategies["early_fallback_fusion"]["timing"]),
            "protected_fallback_fusion": deepcopy(
                strategies["protected_fallback_fusion"]["timing"]
            ),
            "protected_minus_old_ms": {
                k: {
                    stat: (-0.5 if k in ("fallback_translation_ms", "translation_ms") else 0.0)
                    for stat in ("mean", "p50", "p95")
                }
                for k in helper.TIMES
            },
        },
    }
    # Four samples [0,0,1,1] vs four [1,1,1,1]: p95 has no difference.
    for key in ("fallback_translation_ms", "translation_ms"):
        result["protection_analysis"]["protected_vs_old_timing"]["protected_minus_old_ms"][key][
            "p95"
        ] = 0.0
    result["startup"] = {
        "index_load_seconds": 0.1,
        "retrieval_model_load_seconds": 0.2,
        "shared_translator_and_decoder_load_seconds": 0.3,
        "validation_and_models_seconds": 0.7,
        "first_request_order": ["original", "constrained", "guarded", "protected"],
        "first_requests_per_decoder": {
            recipe["decoder"]: deepcopy(strategies[policy]["attempts"]["i1"])
            for policy, recipe in result["protocol"]["policies"].items()
        },
    }
    return result, inputs, chinese


def test_four_policy_audit_recomputes_restoration_and_all_warm_requests():
    result, inputs, chinese = synthetic_result()
    audit = auditor()
    assert callable(getattr(audit, "audit_result", None)), "Four-policy audit is missing"
    report = audit.audit_result(result, inputs, chinese)
    assert report["scored_requests"] == 8
    assert report["warm_requests"] == 16
    assert report["branch_audit"]["checked"] == 6
    assert report["protection_analysis"]["restored_old_accepted_ids"] == ["i1"]
    assert report["strategies"]["protected_fallback_fusion"]["rates"]["candidate_protection"] == {
        "numerator": 1,
        "denominator": 2,
        "rate": 0.5,
    }


@pytest.mark.parametrize(
    "corruption",
    [
        "metric",
        "timing",
        "warm_digest",
        "warm_flag",
        "protected_flag",
        "route",
        "legacy_trace",
        "fallback_trace",
        "analysis_ids",
        "analysis_time",
        "rule",
        "startup_semantics",
        "startup_score",
        "startup_timing",
        "startup_order",
        "startup_loading",
    ],
)
def test_four_policy_audit_rejects_recorded_corruption(corruption):
    result, inputs, chinese = synthetic_result()
    strategy = result["strategies"]["protected_fallback_fusion"]
    if corruption == "metric":
        strategy["metrics"]["aggregate"]["average_precision"] += 0.1
    elif corruption == "timing":
        strategy["timing"]["total_ms"]["p95"] += 1
    elif corruption == "warm_digest":
        strategy["warm_samples"]["i1"][0]["evidence_sha256"] = "0" * 64
    elif corruption == "warm_flag":
        strategy["warm_samples"]["i1"][0]["candidate_protected"] = False
    elif corruption == "protected_flag":
        strategy["attempts"]["i1"]["candidate_protected"] = False
    elif corruption == "route":
        strategy["attempts"]["i2"]["query_texts"][1] = "rejected partial"
    elif corruption == "legacy_trace":
        strategy["attempts"]["i1"]["constraint_trace"]["output"] = "edited final output"
    elif corruption == "fallback_trace":
        strategy["attempts"]["i2"]["fallback_trace"]["output"] = "edited backup"
    elif corruption == "analysis_ids":
        result["protection_analysis"]["restored_old_accepted_ids"] = []
    elif corruption == "analysis_time":
        result["protection_analysis"]["warm_timing_by_subset"]["restored_old_accepted"][
            "protected_fallback_fusion"
        ]["total_ms"]["mean"] += 1
    elif corruption == "startup_semantics":
        result["startup"]["first_requests_per_decoder"]["protected"]["candidate_protected"] = False
    elif corruption == "startup_score":
        result["startup"]["first_requests_per_decoder"]["original"]["rankings"][0]["score"] += 0.1
    elif corruption == "startup_timing":
        result["startup"]["first_requests_per_decoder"]["protected"]["timing"]["total_ms"] = 0.0
    elif corruption == "startup_order":
        result["startup"]["first_request_order"] = [
            "protected",
            "original",
            "constrained",
            "guarded",
        ]
    elif corruption == "startup_loading":
        result["startup"]["validation_and_models_seconds"] = -1.0
    else:
        result["protection_rule"]["training"] = True
    with pytest.raises(ValueError):
        auditor().audit_result(result, inputs, chinese)


def development_fixture(tmp_path):
    sources = json.loads((ROOT / "benchmarks/completed-protection-v1/dev-queries.json").read_text())
    protocol = {
        "queries_sha256": "",
        "development_implementation_sha256": "a" * 64,
        "repetition_profile_sha256": (
            "e93ee35b4b81fe94a7b2d97c25fa3317bce599e6976d1a54a6a2d90d764b5777"
        ),
        "protection_rule_sha256": (
            "a8fb1d276fc53ace8c59b76dabd13a24035c323aa637f184c768b20db2464cdc"
        ),
        "fixed_rule": json.loads(
            (ROOT / "src/siglip_tools/completed_protection_v1.json").read_text()
        ),
        "methods": ["original", "constrained", "guarded", "protected"],
        "training": False,
        "formal_queries_or_retrieval_used": False,
    }
    prior = load(ROOT / "tests/test_early_stop_audit.py")
    rows = []
    for source in sources["queries"]:
        baseline = {k: v for k, v in protected_trace().items() if k != "repetition_stop"}
        baseline["source_text"] = source["text"]
        old = prior.guard(baseline, True)
        guarded = protected_trace()
        guarded["source_text"] = source["text"]
        original = {
            "source_text": source["text"],
            "status": "success",
            "sequence": [0, 2],
            "output": "original en",
            "raw_decoded_output": "original en",
            "eos_terminated": True,
        }
        methods = {}
        for name, trace in (
            ("original", original),
            ("constrained", baseline),
            ("guarded", old),
            ("protected", guarded),
        ):
            ok = name != "guarded"
            primary = {
                "ok": ok,
                "english": trace["output"] if ok else None,
                "trace": deepcopy(trace),
                "translation_ms": 1.0,
                "failure": None if ok else {"type": "RepetitionAbort", "message": "repeat"},
            }
            backup = (
                {
                    "ok": True,
                    "english": "original en",
                    "trace": deepcopy(original),
                    "translation_ms": 0.5,
                    "failure": None,
                }
                if not ok
                else None
            )
            methods[name] = {
                "primary": primary,
                "fallback": backup,
                "ok": True,
                "total_translation_ms": 2.0 if backup else 1.5,
            }
        rows.append({**source, "input_tokens": 60, "methods": methods})
    observed = {
        "protocol": protocol,
        "protocol_sha256": "",
        "rows": rows,
        "translator_identities": {},
        "development_only": True,
        "training_executed": False,
    }
    summary = {}
    for group, values in {
        "all": rows,
        **{
            g: [r for r in rows if r["group"] == g]
            for g in ("old_development", "long_normal", "legitimate_repetition")
        },
    }.items():
        ids = sorted(r["query_id"] for r in values)
        summary[group] = {
            "count": len(values),
            "baseline_accepted_ids": ids,
            "accepted_eligible_ids": ids,
            "old_early_rejected_ids": ids,
            "new_early_rejected_ids": [],
            "protected_ids": ids,
            "old_accepted_new_rejected_ids": [],
        }
    validation = {
        "protocol_sha256": "",
        "observations_sha256": "",
        "repetition_profile_sha256": protocol["repetition_profile_sha256"],
        "protection_rule_sha256": protocol["protection_rule_sha256"],
        "development_only": True,
        "training_executed": False,
        "summary": summary,
    }

    def write(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        return hashlib.sha256(path.read_bytes()).hexdigest()

    protocol["queries_sha256"] = write("dev-queries.json", sources)
    observed["protocol_sha256"] = validation["protocol_sha256"] = write(
        "dev-protocol.json", protocol
    )
    validation["observations_sha256"] = write("dev-observations.json", observed)
    write("dev-validation.json", validation)
    return protocol, observed, validation, write


def test_development_audit_replays_all_24_sources_four_methods_and_coverage(tmp_path):
    protocol, _, _, _ = development_fixture(tmp_path)
    audit = auditor()
    assert callable(getattr(audit, "audit_development", None)), "Development replay is missing"
    report = audit.audit_development(
        tmp_path, protocol["fixed_rule"]["repetition_profile"], "a" * 64, audit.Checks()
    )
    assert report["sources"] == 24
    assert report["method_requests"] == 96
    assert report["summary"]["long_normal"]["count"] == 8
    assert len(report["summary"]["long_normal"]["accepted_eligible_ids"]) == 8


@pytest.mark.parametrize(
    "corruption", ["self_hashed_source", "coverage", "fallback", "time", "guard", "code"]
)
def test_development_audit_rejects_coherent_source_and_observation_changes(tmp_path, corruption):
    protocol, observed, validation, write = development_fixture(tmp_path)
    if corruption == "self_hashed_source":
        sources = json.loads((tmp_path / "dev-queries.json").read_text())
        sources["queries"][8]["text"] += "修改"
        protocol["queries_sha256"] = write("dev-queries.json", sources)
        observed["rows"][8]["text"] = sources["queries"][8]["text"]
    elif corruption == "coverage":
        validation["summary"]["long_normal"]["accepted_eligible_ids"] = []
    elif corruption == "fallback":
        observed["rows"][0]["methods"]["guarded"]["fallback"]["trace"]["output"] = "changed"
    elif corruption == "time":
        observed["rows"][0]["methods"]["protected"]["total_translation_ms"] = 0.5
    elif corruption == "guard":
        observed["rows"][0]["methods"]["protected"]["primary"]["trace"]["repetition_stop"][
            "snapshots"
        ][0]["protected"] = False
    else:
        protocol["development_implementation_sha256"] = "b" * 64
    observed["protocol_sha256"] = validation["protocol_sha256"] = write(
        "dev-protocol.json", protocol
    )
    validation["observations_sha256"] = write("dev-observations.json", observed)
    write("dev-validation.json", validation)
    with pytest.raises(ValueError):
        auditor().audit_development(
            tmp_path, protocol["fixed_rule"]["repetition_profile"], "a" * 64, auditor().Checks()
        )
