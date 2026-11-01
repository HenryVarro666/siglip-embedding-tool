"""Development source integrity and observed coverage, without pretrained inference."""

import importlib
import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest


def module():
    assert importlib.util.find_spec("siglip_tools.protected_development") is not None, (
        "Fixed extended-development validation is not implemented"
    )
    return importlib.import_module("siglip_tools.protected_development")


def test_declared_sources_reject_formal_ids_and_changed_text_before_inference(tmp_path):
    m = module()
    source = Path(__file__).parents[1] / "benchmarks/completed-protection-v1/dev-queries.json"
    doc = json.loads(source.read_text())
    assert len(m.load_development_queries(source)) == 24
    for field, value in (("text", "另一个中文查询"), ("query_id", "t09_zh")):
        bad = deepcopy(doc)
        bad["queries"][0][field] = value
        path = tmp_path / (field + ".json")
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match="development"):
            m.load_development_queries(path)


def sample(group, name, *, accepted, eligible, old_abort=False, new_abort=False, protected=False):
    return {
        "query_id": name,
        "group": group,
        "methods": {
            "constrained": {"primary": {"ok": accepted}},
            "guarded": {"primary": {"trace": {"repetition_stop": {"triggered": old_abort}}}},
            "protected": {
                "primary": {
                    "ok": not new_abort,
                    "trace": {
                        "repetition_stop": {
                            "triggered": new_abort,
                            "snapshots": [{"new_tokens": 32}] if eligible else [],
                            "protected_trigger_steps": [32] if protected else [],
                        }
                    },
                }
            },
        },
    }


def test_coverage_counts_actual_accepted_eligible_controls_and_new_rejections():
    m = module()
    rows = [
        sample("long_normal", "l01", accepted=True, eligible=False),
        sample("long_normal", "l02", accepted=True, eligible=True, old_abort=True, protected=True),
        sample("legitimate_repetition", "r01", accepted=True, eligible=True, new_abort=True),
        sample(
            "old_development", "i07", accepted=False, eligible=True, old_abort=True, new_abort=True
        ),
    ]
    report = m.development_summary(rows)
    assert report["all"]["baseline_accepted_ids"] == ["l01", "l02", "r01"]
    assert report["all"]["accepted_eligible_ids"] == ["l02", "r01"]
    assert report["all"]["protected_ids"] == ["l02"]
    assert report["all"]["old_accepted_new_rejected_ids"] == ["r01"]
    assert report["long_normal"]["accepted_eligible_ids"] == ["l02"]
    assert report["legitimate_repetition"]["accepted_eligible_ids"] == ["r01"]


def test_disabling_guard_does_not_count_intended_long_control_as_exercised():
    report = module().development_summary(
        [
            sample("long_normal", "l01", accepted=True, eligible=False),
            sample("legitimate_repetition", "r01", accepted=False, eligible=False),
        ]
    )
    assert report["all"]["accepted_eligible_ids"] == []
    assert report["long_normal"]["count"] == 1
    assert report["legitimate_repetition"]["baseline_accepted_ids"] == []


@pytest.mark.parametrize("change", ["disabled", "empty", "renamed_target", "empty_tokens"])
def test_source_constraint_erasure_cannot_claim_development_coverage(change):
    m = module()
    assert hasattr(m, "validate_source_constraints"), "Source glossary coverage is not bound"
    trace = {
        "constraints_applied": True,
        "requirements": [
            {
                "id": "transformer_bushing",
                "source_terms": ["套管"],
                "target_forms": ["bushing", "bushings", "Bushing", "Bushings"],
                "token_alternatives": [[13], [14], [15], [16]],
            }
        ],
    }
    assert m.validate_source_constraints(trace, "变压器顶部的套管") is True
    if change == "disabled":
        trace["constraints_applied"] = False
    elif change == "empty":
        trace["requirements"] = []
    elif change == "renamed_target":
        trace["requirements"][0]["target_forms"] = ["sleeve"]
    else:
        trace["requirements"][0]["token_alternatives"] = []
    with pytest.raises(ValueError, match="source.*constraints"):
        m.validate_source_constraints(trace, "变压器顶部的套管")


def test_custom_output_paths_validate_those_files_instead_of_missing_canonical_files(tmp_path):
    m = module()
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    source = Path(__file__).parents[1] / "benchmarks/completed-protection-v1/dev-queries.json"
    (metadata / "dev-queries.json").write_bytes(source.read_bytes())
    protocol = {
        "queries_sha256": m.file_sha256(source),
        "development_implementation_sha256": m.development_implementation_hash(),
        "fixed_rule": json.loads(m.RULE_PATH.read_text()),
        "repetition_profile_sha256": m.file_sha256(m.PROFILE_PATH),
        "protection_rule_sha256": m.file_sha256(m.RULE_PATH),
        "methods": list(m.METHODS),
        "training": False,
        "formal_queries_or_retrieval_used": False,
    }
    (metadata / "dev-protocol.json").write_text(json.dumps(protocol))
    custom_observations = tmp_path / "own-observations.json"
    custom_validation = tmp_path / "own-validation.json"
    custom_observations.write_text(
        json.dumps(
            {
                "protocol": protocol,
                "protocol_sha256": m.file_sha256(metadata / "dev-protocol.json"),
                "development_only": True,
                "training_executed": False,
                "rows": [],
            }
        )
    )
    custom_validation.write_text("{}")
    with pytest.raises(ValueError, match="Saved development sources"):
        m.validate_development_evidence(
            metadata,
            json.loads(m.PROFILE_PATH.read_text()),
            observations_path=custom_observations,
            validation_path=custom_validation,
        )
    assert not (metadata / "dev-observations.json").exists()
    assert not (metadata / "dev-validation.json").exists()


def test_primary_acceptance_rejects_empty_non_eos_success_before_counting_coverage():
    m = module()
    assert hasattr(m, "validate_primary_acceptance"), "Development final acceptance is not replayed"
    primary = {
        "ok": True,
        "english": "English",
        "failure": None,
        "trace": {
            "source_text": "查询",
            "status": "success",
            "sequence": [1, 9, 2],
            "eos_terminated": True,
            "raw_decoded_output": " English ",
            "output": "English",
        },
    }
    identity = {"generation": {"eos_token_id": 2}}
    assert m.validate_primary_acceptance(primary, "查询", identity, constrained=False) is True
    primary["trace"].update(sequence=[1, 9], eos_terminated=False, raw_decoded_output="", output="")
    primary["english"] = ""
    with pytest.raises(ValueError, match="acceptance"):
        m.validate_primary_acceptance(primary, "查询", identity, constrained=False)
