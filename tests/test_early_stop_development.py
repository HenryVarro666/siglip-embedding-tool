"""Development rule selection never consumes formal retrieval outcomes."""

import json
from copy import deepcopy
from importlib import import_module
from pathlib import Path

import pytest


def beam(tokens, length=24):
    return [99] + list(range(100, 100 + length - len(tokens))) + tokens


def observations(include_failure_loop=True, accepted_repeats=False):
    normal = [beam([31 + i, 41 + i]) for i in range(4)]
    loop = beam([7, 8] * 6)
    accepted = [loop, loop, loop, normal[3]] if accepted_repeats else normal
    return {
        "special_ids": [0, 99],
        "rows": [
            {
                "query_id": "i01_zh",
                "baseline": {"ok": True},
                "observer": {
                    "trace": {
                        "repetition_stop": {
                            "snapshots": [
                                {"new_tokens": 24, "beams": accepted},
                            ]
                        }
                    }
                },
            },
            {
                "query_id": "i07_zh",
                "baseline": {"ok": False},
                "observer": {
                    "trace": {
                        "repetition_stop": {
                            "snapshots": [
                                {
                                    "new_tokens": 24,
                                    "beams": [loop] * 4 if include_failure_loop else normal,
                                },
                                {
                                    "new_tokens": 32,
                                    "beams": [beam([7, 8] * 6, 32)] * 4
                                    if include_failure_loop
                                    else [beam([31 + i, 41 + i], 32) for i in range(4)],
                                },
                            ]
                        }
                    }
                },
            },
        ],
    }


def test_selection_prefers_consensus_then_earliest_and_conservative_tiebreaks():
    dev = import_module("siglip_tools.early_stop_development")
    from siglip_tools.repetition_stop import candidate_profiles

    result = dev.select_profile(list(reversed(candidate_profiles())), observations())
    chosen = result["selected_profile"]
    assert chosen["min_repeating_beams"] == 4
    assert chosen["min_new_tokens"] == 24
    assert chosen["repeat_count"] == 4
    assert chosen["max_ngram_size"] == 8
    assert result["viable"] is True
    assert result["selected_row"]["early_rejected_baseline_accepted_ids"] == []
    assert result["selected_row"]["caught_baseline_failure_ids"] == ["i07_zh"]


def test_rules_that_reject_accepted_development_output_are_ineligible():
    dev = import_module("siglip_tools.early_stop_development")
    from siglip_tools.repetition_stop import candidate_profiles

    result = dev.select_profile(candidate_profiles(), observations(accepted_repeats=True))
    q3 = [
        r
        for r in result["candidates"]
        if r["profile"]["min_repeating_beams"] == 3 and r["profile"]["min_new_tokens"] == 24
    ]
    assert all(r["early_rejected_baseline_accepted_ids"] == ["i01_zh"] for r in q3)
    assert result["selected_profile"]["min_repeating_beams"] == 4


def test_no_detectable_failure_returns_explicitly_nonviable_conservative_profile():
    dev = import_module("siglip_tools.early_stop_development")
    from siglip_tools.repetition_stop import candidate_profiles

    result = dev.select_profile(candidate_profiles(), observations(include_failure_loop=False))
    assert result["viable"] is False
    p = result["selected_profile"]
    assert (
        p["max_ngram_size"],
        p["repeat_count"],
        p["min_new_tokens"],
        p["min_repeating_beams"],
    ) == (8, 4, 32, 4)


def test_formal_or_changed_queries_rejected_before_translation(tmp_path):
    dev = import_module("siglip_tools.early_stop_development")
    original = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "benchmarks/repetition-early-stop-v1/dev-queries.json"
        ).read_text()
    )
    path = tmp_path / "queries.json"
    for key, value in [("query_id", "t01_zh"), ("text", "另一个开发问题")]:
        doc = deepcopy(original)
        doc["queries"][0][key] = value
        path.write_text(json.dumps(doc))
        with pytest.raises(ValueError, match="development|fixed"):
            dev.load_development_queries(path)


def test_valid_fixed_development_queries_are_accepted():
    dev = import_module("siglip_tools.early_stop_development")
    rows = dev.load_development_queries(
        Path(__file__).resolve().parents[1] / "benchmarks/repetition-early-stop-v1/dev-queries.json"
    )
    assert len(rows) == 8 and rows[-1]["query_id"] == "i08_zh"


def development_bundle(tmp_path):
    dev = import_module("siglip_tools.early_stop_development")
    from siglip_tools.images import file_sha256
    from siglip_tools.repetition_stop import detect_repetition

    source = (
        Path(__file__).resolve().parents[1] / "benchmarks/repetition-early-stop-v1/dev-queries.json"
    )
    (tmp_path / "dev-queries.json").write_bytes(source.read_bytes())
    candidates = dev.development_candidates()
    observer_profile = next(
        p
        for p in candidates
        if (p["max_ngram_size"], p["repeat_count"], p["min_new_tokens"], p["min_repeating_beams"])
        == (16, 2, 24, 3)
    )
    protocol = {
        "queries_sha256": file_sha256(tmp_path / "dev-queries.json"),
        "candidate_profiles": candidates,
        "selection_rule": dev.SELECTION_RULE,
        "development_implementation_sha256": dev.development_implementation_hash(),
        "training": False,
        "formal_queries_or_retrieval_used": False,
    }

    def write(name, value):
        p = tmp_path / name
        p.write_text(json.dumps(value))
        return file_sha256(p)

    ph = write("dev-protocol.json", protocol)
    obs = {
        "protocol": protocol,
        "protocol_sha256": ph,
        "special_ids": [0, 99],
        "rows": [],
        "parameters_require_grad": False,
    }
    for query in dev.load_development_queries(source):
        ok = query["query_id"] != "i07_zh"
        trace = {
            "source_text": query["text"],
            "sequence": [99, 1, 0] if ok else [99] + [7, 8] * 32,
            "output": "English",
            "status": "success" if ok else "failed",
            "eos_terminated": ok,
        }
        baseline = {
            "ok": ok,
            "english": "English" if ok else None,
            "failure": None if ok else {"type": "ValueError", "message": "cap"},
            "trace": trace,
        }
        beams = [beam([7, 8] * 6)] * 4
        snapshots = (
            [] if ok else [{"beams": beams, **detect_repetition(beams, observer_profile, [0, 99])}]
        )
        guard = {
            "profile": observer_profile,
            "special_ids": [0, 99],
            "observe_only": True,
            "enabled": not ok,
            "snapshots": snapshots,
        }
        observed = deepcopy(baseline)
        observed["trace"]["repetition_stop"] = guard
        obs["rows"].append({**query, "baseline": baseline, "observer": observed})
    oh = write("dev-observations.json", obs)
    selected = dev.select_profile(candidates, obs)
    selected.update(protocol_sha256=ph, observations_sha256=oh)
    sh = write("dev-selection.json", selected)
    profile = selected["selected_profile"]
    fh = write("repetition_profile_v1.json", profile)
    validation = {
        "protocol_sha256": ph,
        "selection_sha256": sh,
        "profile_byte_sha256": fh,
        "profile": profile,
        "active_stops_match_replayed_predictions": True,
        "rows": [],
    }
    for row in obs["rows"]:
        trigger = selected["selected_row"]["trigger_new_tokens"].get(row["query_id"])
        actual = deepcopy(row["baseline"])
        guard = {
            "profile": profile,
            "special_ids": [0, 99],
            "observe_only": False,
            "triggered": trigger is not None,
            "trigger_new_tokens": trigger,
            "snapshots": [],
        }
        if trigger:
            beams = [beam([7, 8] * 6)] * 4
            snapshot = {"beams": beams, **detect_repetition(beams, profile, [0, 99])}
            guard["snapshots"] = [snapshot]
            actual["trace"].update(
                sequence=beams[0], output="Partial English", status="failed", eos_terminated=False
            )
        actual["trace"]["repetition_stop"] = guard
        backup = (
            {
                "ok": True,
                "english": "Backup",
                "trace": {
                    "source_text": row["text"],
                    "status": "success",
                    "sequence": [99, 1, 0],
                    "output": "Backup",
                    "eos_terminated": True,
                },
            }
            if not actual["ok"]
            else None
        )
        validation["rows"].append(
            {
                "query_id": row["query_id"],
                "text": row["text"],
                "baseline": deepcopy(row["baseline"]),
                "guarded": actual,
                "fallback": backup,
            }
        )
    write("dev-validation.json", validation)
    return dev, profile


def test_saved_development_evidence_replays_selection_and_validation(tmp_path):
    dev, profile = development_bundle(tmp_path)
    assert dev.validate_development_evidence(tmp_path, profile) is True


@pytest.mark.parametrize("corruption", ["selection", "active_prediction", "observation_source"])
def test_coherent_hashes_cannot_hide_wrong_development_semantics(tmp_path, corruption):
    dev, profile = development_bundle(tmp_path)
    if corruption == "selection":
        p = tmp_path / "dev-selection.json"
        doc = json.loads(p.read_text())
        doc["selected_profile"]["repeat_count"] = (
            2 if doc["selected_profile"]["repeat_count"] == 3 else 3
        )
        p.write_text(json.dumps(doc))
    elif corruption == "active_prediction":
        p = tmp_path / "dev-validation.json"
        doc = json.loads(p.read_text())
        doc["rows"][6]["guarded"]["trace"]["repetition_stop"]["trigger_new_tokens"] = 32
        p.write_text(json.dumps(doc))
    else:
        p = tmp_path / "dev-observations.json"
        doc = json.loads(p.read_text())
        doc["rows"][0]["text"] = "not the fixed development source"
        p.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        dev.validate_development_evidence(tmp_path, profile)
