"""Frozen benchmarks bind the measured source snapshot, not the working tree (SIG-073)."""

import json
import shutil
from pathlib import Path

import pytest

from siglip_tools import frozen_source
from siglip_tools.images import file_sha256

REPO = Path(__file__).resolve().parents[1]
BENCHMARKS = sorted(p for p in (REPO / "benchmarks").iterdir() if (p / "freeze.json").is_file())


@pytest.mark.parametrize("metadata", BENCHMARKS, ids=[p.name for p in BENCHMARKS])
def test_every_frozen_benchmark_reproduces_its_bindings_from_the_snapshot(metadata):
    report = frozen_source.verify(metadata)
    freeze = json.loads((metadata / "freeze.json").read_text())
    assert report["bindings"]["implementation_sha256"] == freeze["implementation_sha256"]
    assert len(report["source_commit"]) == 40 and report["files"] > 0


def test_live_hash_is_used_when_nothing_is_bound_and_the_table_when_bound():
    live = frozen_source.source_sha256("images.py")
    assert live == file_sha256(REPO / "src/siglip_tools/images.py")
    table = frozen_source.TableResolver({"images.py": "0" * 64})
    with frozen_source.bound(table):
        assert frozen_source.source_sha256("images.py") == "0" * 64
        with pytest.raises(ValueError, match="lacks"):
            frozen_source.source_sha256("missing.py")
    assert table.requested == {"images.py", "missing.py"}
    assert frozen_source.source_sha256("images.py") == live


def test_tampered_snapshot_is_rejected_but_a_changed_working_tree_is_reported_not_fatal(tmp_path):
    source = REPO / "benchmarks/terminology-constraints-v1"
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    for name in ("freeze.json", "protocol.json", "frozen-source.json"):
        shutil.copyfile(source / name, metadata / name)
    from siglip_tools.term_benchmark import implementation_hash

    freeze = json.loads((metadata / "freeze.json").read_text())
    expected = freeze["implementation_sha256"]
    assert (
        frozen_source.reference_implementation(metadata, implementation_hash, expected) == expected
    )
    drift = frozen_source.execution_drift(metadata, implementation_hash)
    assert drift["frozen"] == expected and isinstance(drift["identical"], bool)
    snapshot = json.loads((metadata / "frozen-source.json").read_text())
    snapshot["files"]["translation.py"] = "f" * 64
    (metadata / "frozen-source.json").write_text(json.dumps(snapshot))
    with pytest.raises(ValueError, match="does not reproduce"):
        frozen_source.reference_implementation(metadata, implementation_hash, expected)
    with pytest.raises(ValueError, match="does not reproduce"):
        frozen_source.verify(metadata)


def test_missing_snapshot_falls_back_to_the_live_formula_for_fixtures(tmp_path):
    from siglip_tools.term_benchmark import implementation_hash

    assert not frozen_source.has_snapshot(tmp_path)
    assert (
        frozen_source.reference_implementation(tmp_path, implementation_hash)
        == implementation_hash()
    )
    with pytest.raises(ValueError, match="Missing or invalid"):
        frozen_source.load_snapshot(tmp_path)


def test_git_resolver_hashes_committed_blobs_not_the_working_tree():
    resolver = frozen_source.GitResolver("HEAD", REPO)
    digest = resolver("evaluation.py")
    assert len(digest) == 64 and resolver.files == {"evaluation.py": digest}
    with pytest.raises(ValueError, match="not present"):
        resolver("does-not-exist.py")


def test_verify_cli_reports_each_directory_and_fails_on_a_broken_one(tmp_path, capsys):
    assert frozen_source.main(["verify", str(BENCHMARKS[0])]) == 0
    assert BENCHMARKS[0].name in capsys.readouterr().out
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "freeze.json").write_text("{}")
    assert frozen_source.main(["verify", str(broken)]) == 1
    assert "error:" in capsys.readouterr().err


def test_executing_a_frozen_protocol_requires_the_measured_code(tmp_path, monkeypatch):
    from siglip_tools import fallback_benchmark as runner

    source = REPO / "benchmarks/constraint-fallback-v1"
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    for name in ("freeze.json", "protocol.json", "frozen-source.json"):
        shutil.copyfile(source / name, metadata / name)
    monkeypatch.setattr(
        runner,
        "SOURCE_DATA_HASHES",
        json.loads((metadata / "protocol.json").read_text())["dataset_hashes"],
    )
    monkeypatch.setattr(
        runner,
        "SOURCE_SUPPORT_HASHES",
        json.loads((metadata / "protocol.json").read_text())["supporting_hashes"],
    )
    for name in runner.SOURCE_DATA_HASHES:
        shutil.copyfile(source / f"{name}.json", metadata / f"{name}.json")
    for name in runner.SOURCE_SUPPORT_HASHES:
        shutil.copyfile(source / name, metadata / name)
    # Export/audit validation binds the snapshot and succeeds on any working tree.
    protocol, _ = runner.validate_frozen_inputs(metadata, metadata / "protocol.json")
    assert protocol["experiment_id"] == "equipment-constraint-fallback-v1"
    # Executing the frozen protocol additionally requires the live code to equal it:
    # simulate an evolved working tree by pointing the live lookup at an edited copy.
    package = tmp_path / "package"
    shutil.copytree(REPO / "src/siglip_tools", package)
    (package / "translation.py").write_text(
        (package / "translation.py").read_text() + "\n# evolved after the measurement\n"
    )
    monkeypatch.setattr(frozen_source, "PACKAGE_ROOT", package)
    protocol, _ = runner.validate_frozen_inputs(metadata, metadata / "protocol.json")
    assert protocol["experiment_id"] == "equipment-constraint-fallback-v1"
    with pytest.raises(ValueError, match="Executing implementation"):
        runner.validate_frozen_inputs(metadata, metadata / "protocol.json", executing=True)
