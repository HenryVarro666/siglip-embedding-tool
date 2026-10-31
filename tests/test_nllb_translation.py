import json
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

import pytest
import torch
import transformers
from safetensors.torch import load_file, save_file
from transformers import GenerationConfig


class TinyTokenizer:
    unk_token_id = 3
    src_lang = "eng_Latn"
    input_length = 3
    decoded = None

    def __init__(self):
        self.languages = {"zho_Hans": 4, "eng_Latn": 5}
        self.calls = []

    def convert_tokens_to_ids(self, token):
        return self.languages.get(token, self.unk_token_id)

    def convert_ids_to_tokens(self, token_id):
        return next((name for name, value in self.languages.items() if value == token_id), "<unk>")

    def __call__(self, text, **kwargs):
        self.calls.append((text, self.src_lang, kwargs))
        return {"input_ids": torch.ones((1, self.input_length), dtype=torch.long)}

    def decode(self, sequence, *, skip_special_tokens):
        assert skip_special_tokens
        if self.decoded is not None:
            return self.decoded
        return "  A Red CAR, beside a pole.  " if 21 in sequence else "A blue car."

    def save_pretrained(self, path):
        for name in ("tokenizer.json", "special_tokens_map.json"):
            (Path(path) / name).write_text("{}", encoding="utf-8")
        (Path(path) / "tokenizer_config.json").write_text(
            json.dumps({"src_lang": self.src_lang}), encoding="utf-8"
        )


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
        self.generation_config = GenerationConfig(
            eos_token_id=2,
            pad_token_id=1,
            decoder_start_token_id=2,
            forced_bos_token_id=6,
            forced_eos_token_id=2,
            max_length=200,
            do_sample=True,
            num_beams=5,
        )
        self.calls = []
        self.unterminated = False

    def generate(self, **kwargs):
        assert torch.is_inference_mode_enabled()
        assert not self.training
        assert self.weight.dtype == torch.float32
        self.calls.append(kwargs)
        if self.unterminated:
            return torch.tensor([[2, 5] + [21] * 63])
        return torch.tensor([[2, 5, 21 if len(self.calls) % 2 else 22, 2]])

    def save_pretrained(self, path, *, safe_serialization, max_shard_size):
        assert safe_serialization is True
        assert max_shard_size == "8GB"
        save_file({"weight": self.weight.detach()}, str(Path(path) / "model.safetensors"))
        (Path(path) / "config.json").write_text('{"model_type":"m2m_100"}', encoding="utf-8")
        self.generation_config.save_pretrained(path)


@pytest.fixture
def boundaries(monkeypatch, tmp_path):
    """Replace the download and heavyweight loaders, retaining real local artifacts."""
    import huggingface_hub

    source = tmp_path / "snapshot"
    source.mkdir()
    weight_bytes = b"official pinned NLLB source weights"
    files = {
        "pytorch_model.bin": weight_bytes,
        "config.json": b'{"model_type":"m2m_100"}',
        "generation_config.json": b'{"eos_token_id":2}',
        "sentencepiece.bpe.model": b"sentencepiece fixture",
        "tokenizer.json": b"{}",
        "tokenizer_config.json": b"{}",
        "special_tokens_map.json": b"{}",
        "README.md": b"---\nlicense: cc-by-nc-4.0\n---\nResearch translation model.\n",
    }
    for name, content in files.items():
        (source / name).write_bytes(content)
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
        tokenizer.src_lang = kwargs.get("src_lang", "eng_Latn")
        return tokenizer

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    monkeypatch.setattr(transformers.AutoModelForSeq2SeqLM, "from_pretrained", load_model)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", load_tokenizer)
    return {
        "source": source,
        "source_files": files,
        "weight_bytes": weight_bytes,
        "tokenizer": tokenizer,
        "model": model,
        "calls": calls,
        "monkeypatch": monkeypatch,
    }


def prepare(tmp_path, boundaries):
    from siglip_tools import nllb_translation

    # The remote boundary uses tiny bytes in place of the 5.48 GB official file.
    boundaries["monkeypatch"].setattr(
        nllb_translation, "_SOURCE_WEIGHT_SHA256", sha256(boundaries["weight_bytes"]).hexdigest()
    )
    boundaries["monkeypatch"].setattr(
        nllb_translation, "_SOURCE_WEIGHT_SIZE_BYTES", len(boundaries["weight_bytes"])
    )
    return nllb_translation.prepare_translator(tmp_path / "converted")


def edit_provenance(output, edit):
    path = output / "translation_provenance.json"
    provenance = json.loads(path.read_text())
    edit(provenance)
    path.write_text(json.dumps(provenance), encoding="utf-8")


def test_prepare_pins_official_source_and_converts_cpu_float32_safetensors(tmp_path, boundaries):
    from siglip_tools.nllb_translation import MODEL_ID, REVISION

    output = prepare(tmp_path, boundaries)
    calls = boundaries["calls"]
    request = calls["download"][0]
    assert request["repo_id"] == MODEL_ID == "facebook/nllb-200-1.3B"
    assert request["revision"] == REVISION == "b0de46b488af0cf31749cd8da5ed3171e11b2309"
    assert set(request["allow_patterns"]) == set(boundaries["source_files"])
    load = calls["model"][0][1]
    assert load["local_files_only"] is True
    assert load["trust_remote_code"] is False
    assert load["weights_only"] is True
    assert load["use_safetensors"] is False
    assert load["dtype"] == torch.float32
    assert boundaries["model"].weight.device == torch.device("cpu")
    assert not boundaries["model"].weight.requires_grad
    assert load_file(str(output / "model.safetensors"))["weight"].dtype == torch.float32
    assert list(output.glob("*.safetensors")) == [output / "model.safetensors"]
    assert not list(output.glob("*.bin"))
    assert (output / "README.md").read_bytes() == boundaries["source_files"]["README.md"]
    assert calls["tokenizer"][0][1]["src_lang"] == "zho_Hans"


def test_preparation_records_complete_source_converted_and_runtime_hashes(tmp_path, boundaries):
    output = prepare(tmp_path, boundaries)
    provenance = json.loads((output / "translation_provenance.json").read_text())
    source = provenance["source"]
    assert source["license"] == "cc-by-nc-4.0"
    assert source["weight_size_bytes"] == len(boundaries["weight_bytes"])
    assert source["weight_sha256"] == sha256(boundaries["weight_bytes"]).hexdigest()
    assert set(source["artifact_hashes"]) == set(boundaries["source_files"])
    for name, content in boundaries["source_files"].items():
        assert source["artifact_hashes"][name] == sha256(content).hexdigest()
    conversion = provenance["conversion"]
    assert conversion["format"] == "safetensors"
    assert conversion["dtype"] == "float32"
    assert conversion["training"] is False
    assert conversion["runtime_versions"]["torch"]
    expected = {
        path.name: sha256(path.read_bytes()).hexdigest()
        for path in output.iterdir()
        if path.name != "translation_provenance.json"
    }
    assert provenance["artifacts"] == expected


@pytest.mark.parametrize("failure", ["checksum", "size"])
def test_source_weight_checksum_and_size_are_checked_before_loading(tmp_path, boundaries, failure):
    corruption = b"x" * len(boundaries["weight_bytes"]) if failure == "checksum" else b"x"
    (boundaries["source"] / "pytorch_model.bin").write_bytes(corruption)
    with pytest.raises(ValueError, match="source|checksum|hash|size"):
        prepare(tmp_path, boundaries)
    assert boundaries["calls"]["model"] == []
    assert not (tmp_path / "converted").exists()


@pytest.mark.parametrize(
    "filename", ["model.safetensors", "config.json", "tokenizer.json", "README.md"]
)
def test_changed_artifact_is_rejected_before_local_model_loading(tmp_path, boundaries, filename):
    from siglip_tools.nllb_translation import NLLBTranslator

    output = prepare(tmp_path, boundaries)
    boundaries["calls"]["model"].clear()
    (output / filename).write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash|checksum"):
        NLLBTranslator(output, device="cpu")
    assert boundaries["calls"]["model"] == []


@pytest.mark.parametrize(
    ("key", "invalid"),
    [
        ("model_id", "Helsinki-NLP/opus-mt-zh-en"),
        ("revision", "main"),
        ("license", "cc-by-4.0"),
        ("weight_filename", "model.safetensors"),
        ("weight_sha256", "0" * 64),
        ("weight_size_bytes", 1),
    ],
)
def test_unexpected_pinned_source_is_rejected_before_loading(tmp_path, boundaries, key, invalid):
    from siglip_tools.nllb_translation import NLLBTranslator

    output = prepare(tmp_path, boundaries)
    boundaries["calls"]["model"].clear()
    edit_provenance(output, lambda data: data["source"].__setitem__(key, invalid))
    with pytest.raises(ValueError, match="source|revision"):
        NLLBTranslator(output, device="cpu")
    assert boundaries["calls"]["model"] == []


def test_missing_source_hash_inventory_is_rejected_before_loading(tmp_path, boundaries):
    from siglip_tools.nllb_translation import NLLBTranslator

    output = prepare(tmp_path, boundaries)
    boundaries["calls"]["model"].clear()
    edit_provenance(output, lambda data: data["source"]["artifact_hashes"].pop("tokenizer.json"))
    with pytest.raises(ValueError, match="source|inventory"):
        NLLBTranslator(output, device="cpu")
    assert boundaries["calls"]["model"] == []


@pytest.mark.parametrize(("key", "invalid"), [("training", True), ("dtype", "float16")])
def test_invalid_conversion_record_is_rejected_before_loading(tmp_path, boundaries, key, invalid):
    from siglip_tools.nllb_translation import NLLBTranslator

    output = prepare(tmp_path, boundaries)
    boundaries["calls"]["model"].clear()
    edit_provenance(output, lambda data: data["conversion"].__setitem__(key, invalid))
    with pytest.raises(ValueError, match="conversion"):
        NLLBTranslator(output, device="cpu")
    assert boundaries["calls"]["model"] == []


def test_untracked_nested_provenance_is_not_hidden_from_inventory(tmp_path, boundaries):
    from siglip_tools.nllb_translation import NLLBTranslator

    output = prepare(tmp_path, boundaries)
    boundaries["calls"]["model"].clear()
    (output / "nested").mkdir()
    (output / "nested" / "translation_provenance.json").write_text("{}")
    with pytest.raises(ValueError, match="inventory"):
        NLLBTranslator(output, device="cpu")
    assert boundaries["calls"]["model"] == []


@pytest.mark.parametrize("filename", ["translation_provenance.json", "tokenizer.json"])
def test_symlinked_manifest_or_artifact_is_rejected_before_loading(tmp_path, boundaries, filename):
    from siglip_tools.nllb_translation import NLLBTranslator

    output = prepare(tmp_path, boundaries)
    boundaries["calls"]["model"].clear()
    target = tmp_path / filename
    target.write_bytes((output / filename).read_bytes())
    (output / filename).unlink()
    (output / filename).symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        NLLBTranslator(output, device="cpu")
    assert boundaries["calls"]["model"] == []


@pytest.mark.parametrize("filename", ["pytorch_model.bin", "model-00002.safetensors"])
def test_extra_weight_format_is_rejected_even_if_manifest_tracks_it(tmp_path, boundaries, filename):
    from siglip_tools.nllb_translation import NLLBTranslator

    output = prepare(tmp_path, boundaries)
    boundaries["calls"]["model"].clear()
    (output / filename).write_bytes(b"unexpected weight")
    edit_provenance(
        output,
        lambda data: data["artifacts"].__setitem__(
            filename, sha256(b"unexpected weight").hexdigest()
        ),
    )
    with pytest.raises(ValueError, match="safetensors|single"):
        NLLBTranslator(output, device="cpu")
    assert boundaries["calls"]["model"] == []


def test_local_translation_forces_language_and_preserves_raw_english_without_memoization(
    tmp_path, boundaries
):
    from siglip_tools.nllb_translation import NLLBTranslator

    output = prepare(tmp_path, boundaries)
    boundaries["tokenizer"].src_lang = "eng_Latn"
    translator = NLLBTranslator(output, device="cpu")
    assert translator.model_dir == output
    assert translator.device == torch.device("cpu")
    assert translator.translate("  红色汽车  ") == "A Red CAR, beside a pole."
    assert translator.translate("  红色汽车  ") == "A blue car."
    assert len(boundaries["model"].calls) == 2
    for text, src_lang, recipe in boundaries["tokenizer"].calls:
        assert text == "红色汽车"
        assert src_lang == "zho_Hans"
        assert recipe == {"padding": False, "truncation": False, "return_tensors": "pt"}
    for call in boundaries["model"].calls:
        settings = call["generation_config"]
        assert settings.forced_bos_token_id == 5
        assert settings.do_sample is False
        assert settings.num_beams == 1
        assert settings.num_return_sequences == 1
        assert settings.max_new_tokens == 64
        assert settings.max_length is None
        assert settings.forced_eos_token_id is None
        assert settings.use_cache is True
    load = boundaries["calls"]["model"][-1][1]
    assert load["local_files_only"] is True
    assert load["trust_remote_code"] is False
    assert load["use_safetensors"] is True
    assert load["dtype"] == torch.float32
    assert boundaries["calls"]["tokenizer"][-1][1]["src_lang"] == "zho_Hans"


def test_identity_records_language_ids_encoder_recipe_and_all_generation_settings(
    tmp_path, boundaries
):
    from siglip_tools.nllb_translation import NLLBTranslator

    translator = NLLBTranslator(prepare(tmp_path, boundaries), device="cpu")
    identity = translator.identity
    assert identity["source_model"] == identity["model_id"] == "facebook/nllb-200-1.3B"
    assert (
        identity["source_revision"]
        == identity["revision"]
        == "b0de46b488af0cf31749cd8da5ed3171e11b2309"
    )
    assert identity["source"]["license"] == "cc-by-nc-4.0"
    assert identity["source_language_code"] == "zho_Hans"
    assert identity["target_language_code"] == "eng_Latn"
    assert identity["source_language_token_id"] == 4
    assert identity["target_language_token_id"] == 5
    assert identity["generation"] == identity["generation_config"]
    assert identity["generation"]["forced_bos_token_id"] == 5
    assert identity["generation"]["max_length"] is None
    assert identity["generation"]["forced_eos_token_id"] is None
    assert identity["encoder_tokenization"] == {
        "src_lang": "zho_Hans",
        "legacy_behaviour": False,
        "text_preprocessing": "strip",
        "add_special_tokens": True,
        "padding": False,
        "truncation": False,
        "return_tensors": "pt",
    }
    assert identity["max_input_tokens"] == 256
    assert identity["dtype"] == "float32"
    assert identity["device"] == "cpu"
    assert identity["artifact_fingerprint"]
    assert identity["tokenizer_sha256"]
    assert identity["runtime_versions"]["transformers"]
    for name, digest in identity["artifact_hashes"].items():
        assert digest == sha256((translator.model_dir / name).read_bytes()).hexdigest()
    json.dumps(identity, allow_nan=False)
    identity["generation"]["forced_bos_token_id"] = 6
    assert translator.identity["generation"]["forced_bos_token_id"] == 5


def test_upstream_generation_fallback_cannot_restore_forced_eos_or_length_cap(tmp_path, boundaries):
    from transformers.generation.utils import GenerationMixin

    from siglip_tools.nllb_translation import NLLBTranslator

    translator = NLLBTranslator(prepare(tmp_path, boundaries), device="cpu")
    effective, _ = GenerationMixin._prepare_generation_config(
        translator.model, translator._generation_config
    )
    assert effective.forced_bos_token_id == 5
    assert effective.forced_eos_token_id is None
    # Upstream fills a global max_length=20, then max_new_tokens takes precedence.
    effective = GenerationMixin._prepare_generated_length(
        translator.model,
        effective,
        has_default_max_length=True,
        has_default_min_length=True,
        model_input_name="input_ids",
        input_ids_length=1,
        inputs_tensor=torch.ones((1, 3), dtype=torch.long),
    )
    assert effective.max_new_tokens == 64
    assert effective.max_length == 65
    assert translator.model.generation_config.forced_eos_token_id is None
    assert translator.model.generation_config.max_length is None


@pytest.mark.parametrize("language", ["zho_Hans", "eng_Latn"])
def test_unknown_language_token_is_rejected_before_model_loading(tmp_path, boundaries, language):
    from siglip_tools.nllb_translation import NLLBTranslator

    output = prepare(tmp_path, boundaries)
    boundaries["calls"]["model"].clear()
    boundaries["tokenizer"].languages.pop(language)
    with pytest.raises(ValueError, match="language|token"):
        NLLBTranslator(output, device="cpu")
    assert boundaries["calls"]["model"] == []


@pytest.mark.parametrize("text", ["", " ", "an English car", None, 4])
def test_invalid_source_is_rejected_without_generation(tmp_path, boundaries, text):
    from siglip_tools.nllb_translation import NLLBTranslator

    translator = NLLBTranslator(prepare(tmp_path, boundaries), device="cpu")
    with pytest.raises(ValueError, match="Chinese|nonempty"):
        translator.translate(text)
    assert boundaries["model"].calls == []


def test_input_limit_rejects_truncation_and_accepts_exact_boundary(tmp_path, boundaries):
    from siglip_tools.nllb_translation import NLLBTranslator

    translator = NLLBTranslator(prepare(tmp_path, boundaries), device="cpu")
    boundaries["tokenizer"].input_length = 257
    with pytest.raises(ValueError, match="256"):
        translator.translate("很长的中文查询")
    assert boundaries["model"].calls == []
    boundaries["tokenizer"].input_length = 256
    assert translator.translate("中文查询") == "A Red CAR, beside a pole."


def test_unterminated_and_empty_outputs_are_rejected(tmp_path, boundaries):
    from siglip_tools.nllb_translation import NLLBTranslator

    translator = NLLBTranslator(prepare(tmp_path, boundaries), device="cpu")
    boundaries["model"].unterminated = True
    with pytest.raises(ValueError, match="EOS|terminate|truncat"):
        translator.translate("中文查询")
    boundaries["model"].unterminated = False
    boundaries["tokenizer"].decoded = " "
    with pytest.raises(ValueError, match="empty"):
        translator.translate("中文查询")


def test_prepare_never_overwrites_existing_output(tmp_path, boundaries):
    from siglip_tools.nllb_translation import prepare_translator

    output = prepare(tmp_path, boundaries)
    before = (output / "model.safetensors").read_bytes()
    with pytest.raises(FileExistsError):
        prepare_translator(output)
    assert len(boundaries["calls"]["download"]) == 1
    assert (output / "model.safetensors").read_bytes() == before


def test_prepare_command_uses_validated_conversion_path(tmp_path, boundaries, capsys):
    from siglip_tools.nllb_translation import main

    prepare(tmp_path, boundaries)
    output = tmp_path / "command-output"
    assert main(["prepare", "--output", str(output)]) == 0
    assert (output / "model.safetensors").is_file()
    assert (output / "translation_provenance.json").is_file()
    assert str(output) in capsys.readouterr().out


def test_module_entrypoint_exposes_preparation_command():
    result = subprocess.run(
        [sys.executable, "-m", "siglip_tools.nllb_translation", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "prepare" in result.stdout
