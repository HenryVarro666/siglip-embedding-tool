import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest
from test_fallback_benchmark import comparison_fixture

SCRIPT = Path(__file__).resolve().parents[1] / "examples/summarize_fallback_benchmark.py"


def module():
    spec = importlib.util.spec_from_file_location("fallback_export", SCRIPT)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_export_keeps_failure_denominators_raw_traces_full_ranks_and_warm_evidence(
    tmp_path, monkeypatch
):
    b, q = comparison_fixture(tmp_path, monkeypatch)
    summary, requests = module().build_exports(
        b, q, tmp_path / "metadata", tmp_path / "protocol.json"
    )
    base = summary["models"]["base"]["strategies"]
    assert base["constrained_en"]["failed_intents"] == 1
    assert base["constrained_en"]["metrics"]["aggregate"]["average_precision"] == 0
    attempt = requests["models"]["base"]["fallback_zh_fusion"]["attempts"]["i"]
    assert len(attempt["rankings"]) == 3
    assert attempt["translation"] is None
    assert attempt["constraint_trace"]["output"] == "rejected fragment"
    assert len(requests["models"]["quality"]["original_en"]["warm_samples"]["i"]) == 2
    assert summary["training_executed"] is False


def test_export_rejects_changed_metrics_before_emitting_summary(tmp_path, monkeypatch):
    b, q = comparison_fixture(tmp_path, monkeypatch)
    q["strategies"]["fallback_opus_en"]["metrics"]["aggregate"]["average_precision"] = 0.9
    with pytest.raises(ValueError):
        module().build_exports(b, q, tmp_path / "metadata", tmp_path / "protocol.json")


def test_export_rejects_coherently_changed_result_protocol(tmp_path, monkeypatch):
    b, q = comparison_fixture(tmp_path, monkeypatch)
    altered = deepcopy(b["protocol"])
    altered["query_repeats"] = 99
    for result in (b, q):
        result["protocol"] = deepcopy(altered)
    with pytest.raises(ValueError, match="protocol|frozen"):
        module().build_exports(b, q, tmp_path / "metadata", tmp_path / "protocol.json")


def test_export_refuses_existing_outputs_before_any_partial_write(tmp_path, monkeypatch):
    b, q = comparison_fixture(tmp_path, monkeypatch)
    for name, value in (("base", b), ("quality", q)):
        (tmp_path / f"{name}.json").write_text(json.dumps(value))
    out = tmp_path / "exported"
    out.mkdir()
    (out / "requests.json").write_text("preserve")
    with pytest.raises(FileExistsError):
        module().export_results(
            tmp_path / "base.json",
            tmp_path / "quality.json",
            out,
            tmp_path / "metadata",
            tmp_path / "protocol.json",
        )
    assert not (out / "results-summary.json").exists()
    assert (out / "requests.json").read_text() == "preserve"
