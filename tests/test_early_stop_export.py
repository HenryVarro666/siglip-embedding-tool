import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest
from test_early_stop_benchmark import fixture

SCRIPT = Path(__file__).resolve().parents[1] / "examples/summarize_early_stop_benchmark.py"


def exporter():
    spec = importlib.util.spec_from_file_location("early_export", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def results(tmp_path, monkeypatch):
    runner, args, *_ = fixture(tmp_path, monkeypatch)
    runner.run_benchmark(args)
    base = json.loads((args.output / "result.json").read_text())
    quality = deepcopy(base)
    quality["model_alias"] = "quality"
    model_id, revision = runner.MODEL_REVISIONS["quality"]
    quality["model_identity"].update(model_id=model_id, revision=revision)
    return args, base, quality


def test_export_retains_early_rejection_partial_evidence_full_ranks_and_actual_warm_times(
    tmp_path, monkeypatch
):
    args, b, q = results(tmp_path, monkeypatch)
    s, r = exporter().build_exports(b, q, args.metadata, args.protocol)
    st = s["models"]["base"]["strategies"]["early_fallback_fusion"]
    assert st["failed_intents"] == 0
    a = r["models"]["base"]["early_fallback_fusion"]["attempts"]["i"]
    assert a["early_rejected"] and a["translation"] == "English"
    assert a["constraint_trace"]["output"] == "rejected fragment"
    assert len(a["rankings"]) == 3
    assert len(r["models"]["quality"]["early_fallback_fusion"]["warm_samples"]["i"]) == 2
    assert s["models"]["base"]["early_stop_analysis"]["early_rejected_ids"] == ["i"]


@pytest.mark.parametrize("change", ["metrics", "protocol"])
def test_export_rejects_changed_saved_statistics_or_frozen_protocol(tmp_path, monkeypatch, change):
    args, b, q = results(tmp_path, monkeypatch)
    if change == "metrics":
        q["strategies"]["early_fallback_fusion"]["metrics"]["aggregate"]["average_precision"] = 0.9
    else:
        for v in (b, q):
            v["protocol"]["query_repeats"] = 99
    with pytest.raises(ValueError):
        exporter().build_exports(b, q, args.metadata, args.protocol)


def test_export_existing_second_file_prevents_partial_write(tmp_path, monkeypatch):
    args, b, q = results(tmp_path, monkeypatch)
    for name, value in [("base", b), ("quality", q)]:
        (tmp_path / f"{name}.json").write_text(json.dumps(value))
    out = tmp_path / "export"
    out.mkdir()
    (out / "requests.json").write_text("keep")
    with pytest.raises(FileExistsError):
        exporter().export_results(
            tmp_path / "base.json", tmp_path / "quality.json", out, args.metadata, args.protocol
        )
    assert not (out / "results-summary.json").exists()
    assert (out / "requests.json").read_text() == "keep"
