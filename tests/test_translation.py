import json
from hashlib import sha256
from pathlib import Path

import pytest
import torch
import transformers
from safetensors.torch import save_file
from transformers import GenerationConfig


class TinyTokenizer:
    eos_token_id = 0
    special_tokens_map = {"eos_token": "</s>"}
    input_length = 3
    decoded = None

    def __call__(self, text, **kwargs):
        assert kwargs["truncation"] is False
        assert kwargs["return_tensors"] == "pt"
        return {"input_ids": torch.ones((1, self.input_length), dtype=torch.long)}

    def decode(self, sequence, *, skip_special_tokens):
        assert skip_special_tokens
        if self.decoded is not None:
            return self.decoded
        return "a red car" if 21 in sequence else "a blue car"

    def save_pretrained(self, path):
        for filename in ("source.spm", "target.spm", "vocab.json", "tokenizer_config.json"):
            (Path(path) / filename).write_text("{}", encoding="utf-8")


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
        self.generation_config = GenerationConfig(
            eos_token_id=0, pad_token_id=2, decoder_start_token_id=2, forced_eos_token_id=0
        )
        self.calls = []
        self.unterminated = False

    def generate(self, **kwargs):
        assert torch.is_inference_mode_enabled()
        assert not self.training
        assert self.weight.dtype == torch.float32
        settings = kwargs["generation_config"]
        assert settings.do_sample is False
        assert settings.num_beams == 1
        assert settings.max_new_tokens == 64
        assert settings.use_cache is True
        assert settings.forced_eos_token_id is None
        self.calls.append(kwargs)
        if self.unterminated:
            return torch.tensor([[2] + [21] * 64])
        return torch.tensor([[2, 21 if len(self.calls) % 2 else 22, 0]])

    def save_pretrained(self, path, *, safe_serialization):
        assert safe_serialization is True
        save_file({"weight": self.weight.detach()}, str(Path(path) / "model.safetensors"))
        (Path(path) / "config.json").write_text('{"model_type":"marian"}', encoding="utf-8")
        self.generation_config.save_pretrained(path)


@pytest.fixture
def boundaries(monkeypatch, tmp_path):
    """Only replace the remote download and heavyweight pretrained loaders."""
    import huggingface_hub

    source = tmp_path / "snapshot"
    source.mkdir()
    (source / "pytorch_model.bin").write_bytes(b"official pinned source weights")
    tokenizer = TinyTokenizer()
    model = TinyModel()
    calls = {"download": [], "model": [], "tokenizer": []}

    def download(**kwargs):
        calls["download"].append(kwargs)
        return str(source)

    def load_model(path, **kwargs):
        calls["model"].append((str(path), kwargs))
        return model

    def load_tokenizer(path, **kwargs):
        calls["tokenizer"].append((str(path), kwargs))
        return tokenizer

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    monkeypatch.setattr(transformers.AutoModelForSeq2SeqLM, "from_pretrained", load_model)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", load_tokenizer)
    return tokenizer, model, calls, source


def prepare(tmp_path, boundaries):
    from siglip_tools.translation import prepare_translator

    return prepare_translator(tmp_path / "converted")


def test_prepare_converts_fixed_official_weights_and_records_hashes(tmp_path, boundaries):
    from siglip_tools.translation import MODEL_ID, REVISION

    output = prepare(tmp_path, boundaries)
    _, _, calls, source = boundaries
    request = calls["download"][0]
    assert request["repo_id"] == MODEL_ID == "Helsinki-NLP/opus-mt-zh-en"
    assert request["revision"] == REVISION == "cf109095479db38d6df799875e34039d4938aaa6"
    load = calls["model"][0][1]
    assert load["local_files_only"] is True
    assert load["use_safetensors"] is False
    assert load["weights_only"] is True
    assert load["trust_remote_code"] is False
    provenance = json.loads((output / "translation_provenance.json").read_text())
    assert provenance["source"]["license"] == "cc-by-4.0"
    assert (
        provenance["source"]["weight_sha256"]
        == sha256((source / "pytorch_model.bin").read_bytes()).hexdigest()
    )
    assert "pytorch_model.bin" not in provenance["artifacts"]
    for filename, digest in provenance["artifacts"].items():
        assert digest == sha256((output / filename).read_bytes()).hexdigest()
    assert "model.safetensors" in provenance["artifacts"]


@pytest.mark.parametrize("filename", ["model.safetensors", "config.json", "source.spm"])
def test_changed_local_artifact_rejected_before_model_load(tmp_path, boundaries, filename):
    from siglip_tools.translation import LocalTranslator

    output = prepare(tmp_path, boundaries)
    calls = boundaries[2]
    calls["model"].clear()
    (output / filename).write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash|checksum"):
        LocalTranslator(output, device="cpu")
    assert calls["model"] == []


def test_wrong_source_or_untracked_artifact_rejected_before_load(tmp_path, boundaries):
    from siglip_tools.translation import LocalTranslator

    output = prepare(tmp_path, boundaries)
    path = output / "translation_provenance.json"
    provenance = json.loads(path.read_text())
    provenance["source"]["revision"] = "main"
    path.write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="source|revision"):
        LocalTranslator(output, device="cpu")
    provenance["source"]["revision"] = "cf109095479db38d6df799875e34039d4938aaa6"
    path.write_text(json.dumps(provenance))
    (output / "unexpected.json").write_text("{}")
    with pytest.raises(ValueError, match="artifact"):
        LocalTranslator(output, device="cpu")


def test_nested_provenance_name_does_not_hide_an_untracked_artifact(tmp_path, boundaries):
    from siglip_tools.translation import LocalTranslator

    output = prepare(tmp_path, boundaries)
    (output / "nested").mkdir()
    (output / "nested" / "translation_provenance.json").write_text("{}")
    with pytest.raises(ValueError, match="artifact"):
        LocalTranslator(output, device="cpu")


def test_local_inference_identity_and_no_query_memoization(tmp_path, boundaries):
    from siglip_tools.translation import LocalTranslator

    translator = LocalTranslator(prepare(tmp_path, boundaries), device="cpu")
    assert translator.translate("红色汽车") == "a red car"
    assert translator.translate("红色汽车") == "a blue car"
    assert len(boundaries[1].calls) == 2
    load = boundaries[2]["model"][-1][1]
    assert load["local_files_only"] is True
    assert load["use_safetensors"] is True
    assert load["trust_remote_code"] is False
    identity = translator.identity
    assert identity["model_id"] == "Helsinki-NLP/opus-mt-zh-en"
    assert identity["device"] == "cpu"
    assert identity["generation_config"]["max_new_tokens"] == 64
    assert identity["artifact_fingerprint"]
    assert identity["tokenizer_sha256"]
    assert identity["runtime_versions"]["torch"]
    assert identity["source_model"] == "Helsinki-NLP/opus-mt-zh-en"
    assert identity["source_revision"] == "cf109095479db38d6df799875e34039d4938aaa6"
    assert identity["generation"]["do_sample"] is False
    assert identity["generation"]["num_beams"] == 1
    for filename, digest in identity["artifact_hashes"].items():
        assert digest == sha256((translator.model_dir / filename).read_bytes()).hexdigest()
    json.dumps(identity, allow_nan=False)
    identity["generation_config"]["max_new_tokens"] = 3
    assert translator.identity["generation_config"]["max_new_tokens"] == 64


@pytest.mark.parametrize("text", ["", "  ", "an English car", None, 4])
def test_invalid_source_is_rejected_without_generation(tmp_path, boundaries, text):
    from siglip_tools.translation import LocalTranslator

    translator = LocalTranslator(prepare(tmp_path, boundaries), device="cpu")
    with pytest.raises(ValueError, match="Chinese|nonempty"):
        translator.translate(text)
    assert boundaries[1].calls == []


def test_oversized_source_is_not_silently_truncated(tmp_path, boundaries):
    from siglip_tools.translation import LocalTranslator

    translator = LocalTranslator(prepare(tmp_path, boundaries), device="cpu")
    boundaries[0].input_length = 257
    with pytest.raises(ValueError, match="256"):
        translator.translate("很长的中文查询")
    assert boundaries[1].calls == []
    boundaries[0].input_length = 256
    assert translator.translate("中文查询") == "a red car"


def test_unterminated_or_empty_translation_is_rejected(tmp_path, boundaries):
    from siglip_tools.translation import LocalTranslator

    translator = LocalTranslator(prepare(tmp_path, boundaries), device="cpu")
    boundaries[1].unterminated = True
    with pytest.raises(ValueError, match="EOS|terminate|truncat"):
        translator.translate("中文查询")
    boundaries[1].unterminated = False
    boundaries[0].decoded = " "
    with pytest.raises(ValueError, match="empty"):
        translator.translate("中文查询")


def test_prepare_never_overwrites_an_existing_output(tmp_path, boundaries):
    from siglip_tools.translation import prepare_translator

    output = prepare(tmp_path, boundaries)
    before = (output / "model.safetensors").read_bytes()
    with pytest.raises(FileExistsError):
        prepare_translator(output)
    assert (output / "model.safetensors").read_bytes() == before


def test_prepare_module_command_uses_same_provenance_path(tmp_path, boundaries, capsys):
    from siglip_tools.translation import main

    output = tmp_path / "command-output"
    assert main(["prepare", "--output", str(output)]) == 0
    assert (output / "translation_provenance.json").is_file()
    assert str(output) in capsys.readouterr().out
