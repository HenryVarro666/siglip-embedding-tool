from types import SimpleNamespace

import numpy as np
import pytest
import torch

from siglip_tools.images import ImageRecord
from siglip_tools.index import ImageIndex

IDENTITY = {"model_id": "test", "revision": "fixed"}


def index_at(tmp_path):
    return ImageIndex(
        [[1, 0], [0, 1], [0.6, 0.8]],
        [ImageRecord(f"{i}.jpg", str(i + 1) * 64, str(i + 4) * 64) for i in range(3)],
        IDENTITY,
        tmp_path,
    )


class Encoder:
    device = torch.device("cpu")
    identity = IDENTITY

    def encode_texts(self, texts, batch_size=8):
        values = {"中文": [1, 0], "automatic English": [0, 1]}
        return np.array([values[text] for text in texts], dtype=np.float32)


class Translator:
    device = torch.device("cpu")
    identity = {"source_model": "test/mt", "source_revision": "fixed"}

    def __init__(self):
        self.calls = []

    def translate(self, text):
        self.calls.append(text)
        return "automatic English"


@pytest.mark.parametrize(
    "mode,want", [("zh", "0.jpg"), ("auto_en", "1.jpg"), ("auto_fusion", "2.jpg")]
)
def test_policy_uses_actual_translation_and_mean_scores(tmp_path, mode, want):
    from siglip_tools.auto_retrieval import timed_search

    translator = Translator()
    result = timed_search(index_at(tmp_path), Encoder(), translator, "中文", mode, top_k=3)
    assert result["matches"][0].path == want
    assert result["translation"] == (None if mode == "zh" else "automatic English")
    assert translator.calls == ([] if mode == "zh" else ["中文"])
    assert result["timing"]["total_ms"] >= result["timing"]["translation_ms"]
    assert result["timing"]["retrieval_ms"] >= 0
    if mode == "auto_fusion":
        assert result["matches"][0].score == pytest.approx(0.7)


def test_auto_translation_is_reexecuted_and_included_in_total(tmp_path, monkeypatch):
    from siglip_tools import auto_retrieval

    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(auto_retrieval, "perf_counter", lambda: clock.now)

    class TimedTranslator(Translator):
        def translate(self, text):
            clock.now += 0.050
            return super().translate(text)

    translator = TimedTranslator()
    for _ in range(2):
        r = auto_retrieval.timed_search(
            index_at(tmp_path), Encoder(), translator, "中文", "auto_en"
        )
        assert r["timing"]["translation_ms"] == pytest.approx(50)
        assert r["timing"]["total_ms"] >= 50 - 1e-8
    assert translator.calls == ["中文", "中文"]


def test_unknown_policy_or_missing_translator_is_rejected(tmp_path):
    from siglip_tools.auto_retrieval import timed_search

    for mode, translator in [("unknown", Translator()), ("auto_en", None)]:
        with pytest.raises(ValueError):
            timed_search(index_at(tmp_path), Encoder(), translator, "中文", mode)


def test_cli_auto_translation_reports_real_english_and_stages(tmp_path, capsys, monkeypatch):
    import json

    from siglip_tools import cli

    index_path = tmp_path / "index.npz"
    index_at(tmp_path).save(index_path)
    monkeypatch.setattr(cli, "SigLIPEmbedder", lambda *a: Encoder())
    monkeypatch.setattr(cli, "LocalTranslator", lambda *a, **kw: Translator(), raising=False)
    assert (
        cli.main(
            [
                "search-text",
                "--index",
                str(index_path),
                "--query",
                "中文",
                "--auto-translate",
                "fusion",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["results"][0]["path"] == "2.jpg"
    assert payload["automatic_translation"]["english"] == "automatic English"
    assert payload["seconds"]["query"] >= payload["seconds"]["translation"]


def test_cli_rejects_supplied_english_with_auto_translation_before_loading(capsys):
    from siglip_tools.cli import main

    with pytest.raises(SystemExit) as error:
        main(
            [
                "search-text",
                "--index",
                "missing.npz",
                "--query",
                "中文",
                "--english-query",
                "supplied",
                "--auto-translate",
                "english",
            ]
        )
    assert error.value.code == 2
    assert "not allowed" in capsys.readouterr().err
