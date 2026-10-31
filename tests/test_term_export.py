import importlib.util
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from shutil import copyfile

import pytest

from siglip_tools.benchmark import MODEL_REVISIONS
from siglip_tools.images import file_sha256

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "examples/summarize_term_benchmark.py"
METADATA = REPO / "benchmarks/terminology-constraints-v1"


def fixture(tmp_path):
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    for name in (
        "protocol.json",
        "freeze.json",
        "manifest.json",
        "queries.json",
        "qrels.json",
        "annotation-notes.json",
        "review-rules.json",
        "label-review.json",
        "glossary.json",
    ):
        copyfile(METADATA / name, metadata / name)
    protocol = json.loads((metadata / "protocol.json").read_text())
    queries = [
        q
        for q in json.loads((metadata / "queries.json").read_text())["queries"]
        if q["language"] == "zh"
    ]
    positives = json.loads((metadata / "qrels.json").read_text())["qrels"]
    images = [r["image_id"] for r in json.loads((metadata / "manifest.json").read_text())["images"]]
    ranks = {
        q["intent_id"]: [
            {
                "image_id": i,
                "rank": j,
                "score": 1 - j / 100,
                "relevant": i in positives[q["query_id"]],
            }
            for j, i in enumerate(images, 1)
        ]
        for q in queries
    }
    samples = {
        q["intent_id"]: [{"ok": True, "translation_ms": 1, "retrieval_ms": 1, "total_ms": 2.1}]
        for q in queries
    }
    stats = {"count": 16, "p50": 2.1, "p95": 2.1, "mean": 2.1, "min": 2.1, "max": 2.1}
    strategy = {
        "metrics": {"aggregate": {"average_precision": 0.5}},
        "rankings": ranks,
        "failures": {},
        "failed_intents": 0,
        "warm_samples": samples,
        "timing": {k: stats for k in ("translation_ms", "retrieval_ms", "total_ms")},
    }
    outputs = {}
    for alias, (mid, rev) in MODEL_REVISIONS.items():
        outputs[alias] = {
            "experiment_id": protocol["experiment_id"],
            "model_alias": alias,
            "model_identity": {"model_id": mid, "revision": rev, "runtime_versions": {}},
            "dataset_hashes": protocol["dataset_hashes"],
            "protocol": protocol,
            "protocol_sha256": file_sha256(metadata / "protocol.json"),
            "implementation_sha256": protocol["implementation_sha256"],
            "source_index_sha256": protocol["source_indexes"][alias],
            "environment": {},
            "translator_identities": {"same": True},
            "translations": {
                v: {q["intent_id"]: "English" for q in queries}
                for v in ("original", "beam", "constrained")
            },
            "constraint_traces": {
                v: {q["intent_id"]: {"constraints_applied": False} for q in queries}
                for v in ("original", "beam", "constrained")
            },
            "startup": {"first_requests_per_decoder": {}},
            "strategies": {k: deepcopy(strategy) for k in protocol["policies"]},
        }
    return metadata, outputs


def exporter():
    spec = importlib.util.spec_from_file_location("term_exporter", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_valid_fixed_protocol_and_rules_are_accepted(tmp_path):
    metadata, results = fixture(tmp_path)
    hashes = exporter().validate_fixed_inputs(results, metadata)
    assert hashes["queries"] == "19a1da3d4fa9d961f379af0d56dad9274e698895af0d2800d35ad44dfad445db"


def test_coherent_changed_protocol_cannot_export_under_frozen_reference(tmp_path):
    metadata, results = fixture(tmp_path)
    for alias, r in results.items():
        r["protocol"]["query_repeats"] = 11
        r["protocol_sha256"] = "changed-common-protocol"
        (tmp_path / f"{alias}.json").write_text(json.dumps(r))
    output = tmp_path / "summary.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--base",
            str(tmp_path / "base.json"),
            "--quality",
            str(tmp_path / "quality.json"),
            "--metadata",
            str(metadata),
            "--output",
            str(output),
            "--views-output",
            str(tmp_path / "views"),
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "frozen protocol" in completed.stderr.lower()
    assert not output.exists()
    assert not (tmp_path / "views").exists()


@pytest.mark.parametrize("change", ["supporting", "packaged_glossary"])
def test_current_support_and_packaged_glossary_must_match_frozen_bytes(
    tmp_path, monkeypatch, change
):
    metadata, results = fixture(tmp_path)
    module = exporter()
    if change == "supporting":
        (metadata / "label-review.json").write_text("changed")
    else:
        glossary = tmp_path / "different-glossary.json"
        glossary.write_text("{}")
        monkeypatch.setattr(module, "GLOSSARY_PATH", glossary)
    with pytest.raises(ValueError):
        module.validate_fixed_inputs(results, metadata)
