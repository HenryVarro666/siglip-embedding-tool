import json

import pytest

from siglip_tools.cli import main
from siglip_tools.images import ImageRecord
from siglip_tools.index import ImageIndex


def test_cli_help_exposes_the_three_requested_workflows(capsys):
    with pytest.raises(SystemExit) as error:
        main(["--help"])
    assert error.value.code == 0
    output = capsys.readouterr().out
    for command in ("index", "embed-images", "search-text", "search-image"):
        assert command in output


def test_inspect_reads_an_index_without_loading_a_model(tmp_path, capsys):
    path = tmp_path / "index.npz"
    ImageIndex(
        [[1, 0]],
        [ImageRecord("a.png", "a" * 64, "b" * 64)],
        {"model_id": "not-a-real-model", "revision": "offline-test"},
        tmp_path,
    ).save(path)
    assert main(["inspect", "--index", str(path)]) == 0
    output = capsys.readouterr().out
    assert "dimension" in output
    payload = json.loads(output)
    assert payload["images"] == 1
    assert payload["dimension"] == 2


def test_cli_rejects_invalid_top_k_before_loading_a_model(capsys):
    with pytest.raises(SystemExit) as error:
        main(["search-text", "--index", "missing.npz", "--query", "cat", "--top-k", "0"])
    assert error.value.code == 2
    assert "positive" in capsys.readouterr().err


def test_index_reports_empty_input_before_loading_a_model(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        main(["index", "--images", str(tmp_path), "--output", str(tmp_path / "out.npz")])
    assert error.value.code == 2
    assert "No supported images" in capsys.readouterr().err


def test_index_refuses_existing_output_before_loading_a_model(tmp_path, capsys):
    path = tmp_path / "out.npz"
    path.write_bytes(b"keep")
    with pytest.raises(SystemExit) as error:
        main(["index", "--images", str(tmp_path), "--output", str(path)])
    assert error.value.code == 2
    assert "already exists" in capsys.readouterr().err
    assert path.read_bytes() == b"keep"
