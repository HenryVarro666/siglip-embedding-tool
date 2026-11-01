import json

import pytest
from test_constraint_fallback import Encoder, Translator
from test_fusion import index_at


def install_models(monkeypatch, *, reject=True, original_reject=False):
    from siglip_tools import cli

    original = Translator(reject=original_reject)
    constrained = Translator(reject=reject, outputs=("rejected English" if reject else "English",))
    monkeypatch.setattr(cli, "SigLIPEmbedder", lambda *a, **kw: Encoder())
    monkeypatch.setattr(cli, "LocalTranslator", lambda *a, **kw: original)

    def wrapper(base, *, constrained=True, decoder_dir=None):
        assert base is original
        assert constrained is True
        return constraint_translator

    constraint_translator = constrained
    monkeypatch.setattr("siglip_tools.opus_constraints.OPUSConstraintTranslator", wrapper)
    return constrained, original


def arguments(tmp_path, *, fallback="opus", mode="fusion"):
    path = tmp_path / "gallery.npz"
    index_at(tmp_path).save(path)
    return [
        "search-text",
        "--index",
        str(path),
        "--query",
        "中文",
        "--auto-translate",
        mode,
        "--translator",
        "opus",
        "--opus-decoding",
        "constrained",
        "--constraint-fallback",
        fallback,
        "--top-k",
        "3",
    ]


@pytest.mark.parametrize(
    "extra",
    [
        [],
        ["--auto-translate", "english"],
        ["--auto-translate", "fusion", "--opus-decoding", "beam"],
        ["--auto-translate", "english", "--translator", "nllb"],
        ["--auto-translate", "english", "--translator", "nllb", "--opus-decoding", "constrained"],
    ],
)
def test_nonerror_fallback_requires_constrained_automatic_opus_before_allocation(
    monkeypatch, capsys, extra
):
    from siglip_tools import cli

    monkeypatch.setattr(cli.ImageIndex, "load", lambda *a: pytest.fail("index was loaded"))
    monkeypatch.setattr(cli, "SigLIPEmbedder", lambda *a, **kw: pytest.fail("model allocated"))
    with pytest.raises(SystemExit) as error:
        cli.main(
            [
                "search-text",
                "--index",
                "absent.npz",
                "--query",
                "中文",
                "--constraint-fallback",
                "opus",
                *extra,
            ]
        )
    assert error.value.code == 2
    assert "--constraint-fallback requires" in capsys.readouterr().err


@pytest.mark.parametrize(
    "fallback,route,english,path,queries",
    [
        ("opus", "original_fusion", "English", "2.png", ["中文", "English"]),
        ("chinese", "zh", None, "0.png", ["中文"]),
    ],
)
def test_cli_reports_actual_recovery_route_texts_traces_and_independent_timing(
    tmp_path, monkeypatch, capsys, fallback, route, english, path, queries
):
    from siglip_tools import cli

    install_models(monkeypatch)
    assert cli.main(arguments(tmp_path, fallback=fallback)) == 0
    payload = json.loads(capsys.readouterr().out)
    automatic = payload["automatic_translation"]
    assert automatic["ok"] is True
    assert automatic["failure"] is None
    assert automatic["primary_failure"]["stage"] == "constraint_translation"
    assert automatic["english"] == english
    assert automatic["constraint_fallback"] == fallback
    assert automatic["effective_route"] == route
    assert automatic["query_texts"] == queries
    assert automatic["constraint_trace"]["output"] == "rejected English"
    assert automatic["fallback_triggered"] is True
    if fallback == "opus":
        assert automatic["fallback_trace"]["output"] == "English"
    else:
        assert automatic["fallback_trace"] is None
    assert payload["results"][0]["path"] == path
    timing = payload["query_timing_ms"]
    assert timing["total_ms"] >= timing["translation_ms"]
    assert timing["translation_ms"] == pytest.approx(
        timing["constraint_translation_ms"] + timing["fallback_translation_ms"]
    )
    assert payload["seconds"]["query"] == pytest.approx(timing["total_ms"] / 1000)
    assert payload["seconds"]["constraint_translation"] == pytest.approx(
        timing["constraint_translation_ms"] / 1000
    )
    assert payload["seconds"]["fallback_translation"] == pytest.approx(
        timing["fallback_translation_ms"] / 1000
    )


def test_cli_terminal_fallback_failure_preserves_json_and_returns_nonzero(
    tmp_path, monkeypatch, capsys
):
    from siglip_tools import cli

    install_models(monkeypatch, original_reject=True)
    assert cli.main(arguments(tmp_path)) == 2
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    automatic = payload["automatic_translation"]
    assert automatic["ok"] is False
    assert automatic["failure"]["stage"] == "fallback_translation"
    assert automatic["constraint_trace"]["output"] == "rejected English"
    assert automatic["fallback_trace"]["output"] == "English"
    assert automatic["query_texts"] == []
    assert payload["results"] == []
    assert captured.err == ""


def test_cli_successful_constraint_output_skips_the_original_fallback(
    tmp_path, monkeypatch, capsys
):
    from siglip_tools import cli

    _, original = install_models(monkeypatch, reject=False, original_reject=True)
    assert cli.main(arguments(tmp_path, mode="english")) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["automatic_translation"]["fallback_triggered"] is False
    assert payload["automatic_translation"]["effective_route"] == "constrained_en"
    assert payload["results"][0]["path"] == "1.png"
    assert original.calls == []


@pytest.mark.parametrize("explicit", [False, True])
def test_cli_default_or_error_policy_retains_existing_error_behavior(
    tmp_path, monkeypatch, capsys, explicit
):
    from siglip_tools import cli

    install_models(monkeypatch)
    argv = arguments(tmp_path, fallback="error")
    if not explicit:
        offset = argv.index("--constraint-fallback")
        del argv[offset : offset + 2]
    with pytest.raises(SystemExit) as error:
        cli.main(argv)
    assert error.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "missing EOS" in captured.err
