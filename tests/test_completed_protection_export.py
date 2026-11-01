"""Curated evidence export refuses overwrite before any input reads."""

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "examples/summarize_completed_protection.py"


def exporter():
    assert SCRIPT.exists(), "Protected-candidate result exporter is not implemented"
    spec = importlib.util.spec_from_file_location("completed_export", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("existing", ["requests.json", "results-summary.json"])
def test_existing_either_export_file_prevents_partial_writes_and_input_reads(tmp_path, existing):
    module = exporter()
    output = tmp_path / "export"
    output.mkdir()
    (output / existing).write_text("retain original evidence")
    with pytest.raises(FileExistsError):
        module.export_results(
            tmp_path / "absent-base.json",
            tmp_path / "absent-so.json",
            output,
            tmp_path / "absent-metadata",
        )
    assert list(output.iterdir()) == [output / existing]
    assert (output / existing).read_text() == "retain original evidence"


def results(tmp_path, monkeypatch):
    from test_protected_benchmark import run_fixture

    runner, args, _, base, *_ = run_fixture(tmp_path, monkeypatch)
    quality = deepcopy(base)
    quality["model_alias"] = "quality"
    model_id, revision = runner.MODEL_REVISIONS["quality"]
    quality["model_identity"].update(model_id=model_id, revision=revision)
    return args, base, quality


def test_export_preserves_protection_and_changed_retrieval_instead_of_forcing_old_metrics(
    tmp_path, monkeypatch
):
    args, base, quality = results(tmp_path, monkeypatch)
    summary, requests = exporter().build_exports(base, quality, args.metadata, args.protocol)
    model = summary["models"]["base"]
    assert len(model["strategies"]) == 4
    assert "protection_analysis" in model
    attempt = requests["models"]["base"]["protected_fallback_fusion"]["attempts"]["i"]
    assert attempt["candidate_protected"] is True
    assert attempt["early_rejected"] is False
    assert attempt["translation"] == "inverter"
    assert attempt["constraint_trace"]["repetition_stop"]["candidate_protection"]["eos_candidates"]
    protected = model["strategies"]["protected_fallback_fusion"]["metrics"]["aggregate"]
    old = model["strategies"]["early_fallback_fusion"]["metrics"]["aggregate"]
    assert protected["average_precision"] == 0.5
    assert old["average_precision"] == 1.0


@pytest.mark.parametrize("change", ["metric", "protocol", "protected_status"])
def test_export_rejects_tampered_metrics_protocol_or_protected_status(
    tmp_path, monkeypatch, change
):
    args, base, quality = results(tmp_path, monkeypatch)
    if change == "metric":
        quality["strategies"]["protected_fallback_fusion"]["metrics"]["aggregate"][
            "average_precision"
        ] = 0.123
    elif change == "protocol":
        for result in (base, quality):
            result["protocol"]["query_repeats"] = 9
    else:
        quality["strategies"]["protected_fallback_fusion"]["attempts"]["i"][
            "candidate_protected"
        ] = False
    with pytest.raises(ValueError):
        exporter().build_exports(base, quality, args.metadata, args.protocol)


def test_successful_export_writes_both_curated_files_and_source_hashes(tmp_path, monkeypatch):
    args, base, quality = results(tmp_path, monkeypatch)
    for name, value in (("base", base), ("quality", quality)):
        (tmp_path / (name + ".json")).write_text(json.dumps(value))
    target = tmp_path / "curated"
    summary = exporter().export_results(
        tmp_path / "base.json", tmp_path / "quality.json", target, args.metadata, args.protocol
    )
    assert len(summary["source_results"]["base"]["sha256"]) == 64
    requests = json.loads((target / "requests.json").read_text())
    assert len(requests["models"]["quality"]["protected_fallback_fusion"]["warm_samples"]["i"]) == 2
    assert (target / "results-summary.json").is_file()
