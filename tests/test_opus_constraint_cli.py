import json

import pytest
import torch
from test_fusion import Encoder as FixtureEncoder
from test_fusion import index_at


@pytest.mark.parametrize("extra", [[], ["--auto-translate", "english", "--translator", "nllb"]])
def test_opus_decoding_requires_automatic_opus_before_any_model_load(monkeypatch, capsys, extra):
    from siglip_tools import cli

    monkeypatch.setattr(cli, "SigLIPEmbedder", lambda *a, **kw: pytest.fail("model allocated"))
    with pytest.raises(SystemExit) as error:
        cli.main(
            [
                "search-text",
                "--index",
                "absent.npz",
                "--query",
                "中文",
                "--opus-decoding",
                "constrained",
                *extra,
            ]
        )
    assert error.value.code == 2
    assert "requires --auto-translate and --translator opus" in capsys.readouterr().err


def test_constrained_cli_returns_exact_translation_and_constraint_trace(
    tmp_path, monkeypatch, capsys
):
    from siglip_tools import cli

    class Encoder(FixtureEncoder):
        device = torch.device("cpu")

    class Base:
        device = torch.device("cpu")
        identity = {"id": "base"}

    class Wrapper:
        device = torch.device("cpu")
        identity = {"id": "constrained"}
        last_trace = {"output": "English", "satisfied": True}

        def __init__(self, base, *, constrained=True, decoder_dir=None):
            assert isinstance(base, Base)
            assert constrained is True

        def translate(self, text):
            assert text == "中文"
            return "English"

    index = tmp_path / "gallery.npz"
    index_at(tmp_path).save(index)
    monkeypatch.setattr(cli, "SigLIPEmbedder", lambda *a: Encoder())
    monkeypatch.setattr(cli, "LocalTranslator", lambda *a, **kw: Base())
    monkeypatch.setattr("siglip_tools.opus_constraints.OPUSConstraintTranslator", Wrapper)
    assert (
        cli.main(
            [
                "search-text",
                "--index",
                str(index),
                "--query",
                "中文",
                "--auto-translate",
                "english",
                "--opus-decoding",
                "constrained",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["automatic_translation"]["opus_decoding"] == "constrained"
    assert result["automatic_translation"]["constraint_trace"]["satisfied"] is True
    assert result["automatic_translation"]["english"] == "English"
    assert result["results"][0]["path"] == "1.png"
