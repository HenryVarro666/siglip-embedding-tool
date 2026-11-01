"""Independent replay must reject coherent metric, guard and branch corruption."""

import hashlib
import importlib.util
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ROOT = Path(__file__).resolve().parents[1]


def auditor():
    path = ROOT / "examples/audit_early_stop_benchmark.py"
    assert path.exists(), "Independent early-stop auditor is missing"
    return load(path)


def profile():
    return {
        "schema_version": 2,
        "profile_id": "bounded-n16-r2-min32-q4-tail4",
        "algorithm": "bounded_tail_consecutive_token_cycles",
        "max_new_tokens": 64,
        "num_beams": 4,
        "check_every": 4,
        "max_ngram_size": 16,
        "repeat_count": 2,
        "min_new_tokens": 32,
        "min_repeating_beams": 4,
        "min_cycle_tokens": 4,
        "tail_tolerance": 4,
    }


def beams():
    return [[0, *range(100, 120), 10, 11, 12, 13, 10, 11, 12, 13, 90, 91, 92, 93]] * 4


def guard(trace, triggered):
    helper = load(ROOT / "examples/audit_fallback_benchmark.py")
    trace = deepcopy(trace)
    trace["constraints_applied"] = True
    trace["eos_terminated"] = not triggered
    snapshot = {
        "beams": beams(),
        "new_tokens": 32,
        "eligible": True,
        "repeating_beams": 4,
        "triggered": True,
        "beam_evidence": [
            {
                "beam_index": n,
                "period": 4,
                "cycle": [10, 11, 12, 13],
                "repeat_count": 2,
                "suffix_start": 21,
                "tail_tokens": 4,
            }
            for n in range(4)
        ],
    }
    if triggered:
        trace.update(
            sequence=beams()[0],
            status="failed",
            output="rejected partial",
            sequence_role="active_first_beam_diagnostic_not_final",
        )
    count = len(trace["sequence"]) - 1
    trace["repetition_stop"] = {
        "profile": profile(),
        "profile_sha256": helper.fingerprint(profile()),
        "special_ids": [0, 2],
        "observe_only": False,
        "enabled": True,
        "triggered": triggered,
        "trigger_new_tokens": 32 if triggered else None,
        "snapshots": [snapshot] if triggered else [],
        "generated_new_tokens": count,
        "observed_generation_steps": count,
        "observed_steps_complete": triggered,
        "length_scope": "partial_beam_sequence" if triggered else "finalized_sequence",
    }
    return trace


def synthetic_result(early_accept=False):
    previous = load(ROOT / "tests/test_fallback_audit.py")
    helper = load(ROOT / "examples/audit_fallback_benchmark.py")
    result, inputs, chinese = previous.synthetic_result()
    policies = {
        "original_fusion": {"decoder": "original", "mode": "auto_fusion", "fallback": "error"},
        "fallback_opus_fusion": {
            "decoder": "constrained",
            "mode": "auto_fusion",
            "fallback": "opus",
        },
        "early_fallback_fusion": {"decoder": "guarded", "mode": "auto_fusion", "fallback": "opus"},
    }
    result["protocol"]["policies"] = policies
    result["repetition_profile"] = profile()
    result["strategies"] = {
        p: result["strategies"][p] for p in policies if p != "early_fallback_fusion"
    }
    legacy = result["strategies"]["fallback_opus_fusion"]
    for attempt in legacy["attempts"].values():
        attempt["constraint_trace"].update(
            constraints_applied=True, eos_terminated=attempt["primary_failure"] is None
        )
    result["strategies"]["early_fallback_fusion"] = deepcopy(legacy)
    for p, strategy in result["strategies"].items():
        for i, attempt in strategy["attempts"].items():
            early = p == "early_fallback_fusion" and (i == "i2" or early_accept)
            if early:
                original = result["strategies"]["original_fusion"]["attempts"][i]
                for k in ("translation", "query_texts", "rankings", "effective_route"):
                    attempt[k] = deepcopy(original[k])
                attempt["primary_failure"] = {
                    "stage": "constraint_translation",
                    "type": "RepetitionAbort",
                    "message": "repeat",
                }
                attempt["fallback_trace"] = deepcopy(original["original_trace"])
                attempt["fallback_triggered"] = True
                attempt["timing"]["fallback_translation_ms"] = 1.0
                attempt["timing"]["translation_ms"] = 2.0
            attempt["early_rejected"] = early
            if p == "early_fallback_fusion":
                attempt["constraint_trace"] = guard(attempt["constraint_trace"], early)
            strategy["warm_samples"][i] = [
                {
                    **{
                        k: deepcopy(attempt[k])
                        for k in (*helper.STATUS, "early_rejected", "timing")
                    },
                    "ranking_override": None,
                    "evidence_sha256": helper.fingerprint(
                        {"semantics": helper.semantics(attempt), "rankings": attempt["rankings"]}
                    ),
                }
                for _ in range(2)
            ]
        attempts = list(strategy["attempts"].values())
        warm = [s for values in strategy["warm_samples"].values() for s in values]
        ids = {i: [r["image_id"] for r in a["rankings"]] for i, a in strategy["attempts"].items()}
        strategy.update(helper.retrieval_metrics(ids, inputs["qrels"], inputs["gallery_ids"]))
        for key, values in (("rates", attempts), ("warm_rates", warm)):
            strategy[key] = helper.request_rates(values)
            strategy[key]["early_rejection"] = {
                "numerator": sum(a["early_rejected"] for a in values),
                "denominator": len(values),
                "rate": sum(a["early_rejected"] for a in values) / len(values),
            }
        strategy["timing"] = {
            k: helper.statistics([s["timing"][k] for s in warm]) for k in helper.TIMES
        }
        strategy["timing_by_branch"] = {
            key: {
                k: helper.statistics(
                    [s["timing"][k] for s in warm if s["fallback_triggered"] == trigger]
                )
                for k in helper.TIMES
            }
            for key, trigger in (("fallback_triggered", True), ("no_fallback", False))
            if any(s["fallback_triggered"] == trigger for s in warm)
        }
    result["request_semantics"] = {
        p: {i: helper.semantics(a) for i, a in s["attempts"].items()}
        for p, s in result["strategies"].items()
    }
    result["branch_audit"]["checked"] = 4
    caught, false = ["i2"], ["i1"] if early_accept else []
    groups = {
        "legacy_failure": caught,
        "early_rejected": sorted(caught + false),
        "caught_legacy_failure": caught,
        "missed_legacy_failure": [],
        "early_rejected_legacy_accepted": false,
    }
    result["early_stop_analysis"] = {f"{k}_ids": v for k, v in groups.items()}
    result["early_stop_analysis"].update(
        rates={
            "early_rejection": {
                "numerator": 1 + len(false),
                "denominator": 2,
                "rate": (1 + len(false)) / 2,
            },
            "caught_legacy_failure": {"numerator": 1, "denominator": 1, "rate": 1.0},
            "missed_legacy_failure": {"numerator": 0, "denominator": 1, "rate": 0.0},
            "early_rejected_legacy_accepted": {
                "numerator": len(false),
                "denominator": 1,
                "rate": float(bool(false)),
            },
        },
        scored_primary_generated_tokens={
            p: {
                i: len((a["constraint_trace"] or a["original_trace"])["sequence"]) - 1
                for i, a in s["attempts"].items()
            }
            for p, s in result["strategies"].items()
        },
        warm_timing_by_subset={
            g: {
                p: {
                    k: helper.statistics(
                        [v["timing"][k] for i in ids for v in s["warm_samples"][i]]
                    )
                    for k in helper.TIMES
                }
                for p, s in result["strategies"].items()
            }
            for g, ids in groups.items()
            if ids
        },
    )
    return result, inputs, chinese


def test_bounded_tail_evidence_requires_all_beams_and_ignores_special_cycles():
    audit = auditor()
    detected = audit.detect_repetition(beams(), profile(), [0, 2])
    assert detected["triggered"]
    assert detected["beam_evidence"][0] == {
        "beam_index": 0,
        "period": 4,
        "cycle": [10, 11, 12, 13],
        "repeat_count": 2,
        "suffix_start": 21,
        "tail_tokens": 4,
    }
    fewer = deepcopy(beams())
    fewer[3] = [0, *range(100, 132)]
    assert not audit.detect_repetition(fewer, profile(), [0, 2])["triggered"]
    assert not audit.detect_repetition(beams(), profile(), [0, 2, 10])["triggered"]


def test_clean_synthetic_reports_legacy_accepted_early_rejection():
    result, inputs, chinese = synthetic_result(early_accept=True)
    report = auditor().audit_result(result, inputs, chinese)
    assert report["checks"] > 100
    assert report["early_stop_analysis"]["early_rejected_legacy_accepted_ids"] == ["i1"]
    assert "additional_at_3" not in report["strategies"]["original_fusion"]


@pytest.mark.parametrize(
    "corruption",
    ["metric", "timing", "digest", "query", "cycle", "missing_check", "length", "primary"],
)
def test_corrupt_metrics_timing_guard_and_route_are_rejected(corruption):
    result, inputs, chinese = synthetic_result()
    strategy = result["strategies"]["early_fallback_fusion"]
    attempt = strategy["attempts"]["i2"]
    if corruption == "metric":
        strategy["metrics"]["aggregate"]["average_precision"] += 0.1
    elif corruption == "timing":
        strategy["timing"]["total_ms"]["p95"] += 1
    elif corruption == "digest":
        strategy["warm_samples"]["i2"][0]["evidence_sha256"] = "0" * 64
    elif corruption == "query":
        attempt["query_texts"] = [chinese["i2"], attempt["constraint_trace"]["output"]]
    elif corruption == "cycle":
        attempt["constraint_trace"]["repetition_stop"]["snapshots"][0]["beam_evidence"][0][
            "period"
        ] = 8
    elif corruption == "missing_check":
        attempt["constraint_trace"]["repetition_stop"]["snapshots"] = []
    elif corruption == "length":
        attempt["constraint_trace"]["repetition_stop"]["generated_new_tokens"] = 31
    else:
        result["strategies"]["early_fallback_fusion"]["attempts"]["i1"]["constraint_trace"][
            "output"
        ] = "different legacy output"
    with pytest.raises(ValueError):
        auditor().audit_result(result, inputs, chinese)


def test_rehashed_ranking_override_cannot_hide_score_drift():
    result, inputs, chinese = synthetic_result()
    audit = auditor()
    strategy = result["strategies"]["original_fusion"]
    attempt = strategy["attempts"]["i1"]
    sample = strategy["warm_samples"]["i1"][0]
    sample["ranking_override"] = deepcopy(attempt["rankings"])
    sample["ranking_override"][0]["score"] += 0.01
    sample["evidence_sha256"] = audit.legacy.fingerprint(
        {"semantics": audit.legacy.semantics(attempt), "rankings": sample["ranking_override"]}
    )
    with pytest.raises(ValueError):
        audit.audit_result(result, inputs, chinese)


def test_finalized_diagnostic_length_is_not_observed_decoder_step_count():
    result, inputs, chinese = synthetic_result()
    trace = result["strategies"]["early_fallback_fusion"]["attempts"]["i1"]["constraint_trace"]
    trace["repetition_stop"]["observed_generation_steps"] = 3
    audit = auditor()
    audit.validate_guard(trace, profile(), audit.legacy.Checks(), "guard")


def test_development_selection_is_recomputed_from_all_candidates():
    audit = auditor()
    candidates = audit.candidate_profiles(2)
    long_cycle = [[0, *range(100, 108), *list(range(10, 22)) * 2]] * 4
    rows = [
        {
            "query_id": "i07_zh",
            "baseline": {"ok": False},
            "observer": {
                "trace": {
                    "repetition_stop": {"snapshots": [{"beams": long_cycle, "new_tokens": 32}]}
                }
            },
        }
    ]
    selected = audit.select_development(candidates, rows, [0, 2])
    assert selected["selected_profile"] == profile()
    assert len(selected["candidates"]) == 16
    assert selected["selected_row"]["trigger_new_tokens"] == {"i07_zh": 32}


def test_output_refuses_existing_evidence_before_read_and_on_atomic_publish(tmp_path):
    output = tmp_path / "audit.json"
    output.write_text("preserve evidence")
    audit = auditor()
    with pytest.raises(FileExistsError):
        audit.main(["--base", "missing", "--quality", "missing", "--output", str(output)])
    with pytest.raises(FileExistsError):
        audit.save_new(output, {"replaced": True})
    assert output.read_text() == "preserve evidence"


def test_active_stdout_log_is_excluded_but_closed_evidence_is_hashed(tmp_path):
    stable = tmp_path / "closed.log"
    stable.write_text("stable")
    active = tmp_path / "active.log"
    script = ROOT / "examples/audit_early_stop_benchmark.py"
    code = (
        "import importlib.util,json,sys; from pathlib import Path; "
        "s=importlib.util.spec_from_file_location('a',sys.argv[1]); "
        "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
        "print(json.dumps(m.legacy.evidence_log_snapshot(Path(sys.argv[2]))))"
    )
    with active.open("w") as stream:
        result = subprocess.run(
            [sys.executable, "-c", code, str(script), str(tmp_path)],
            stdout=stream,
            stderr=subprocess.PIPE,
            text=True,
        )
    assert result.returncode == 0, result.stderr
    assert json.loads(active.read_text()) == {str(stable): hashlib.sha256(b"stable").hexdigest()}
