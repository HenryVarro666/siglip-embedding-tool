import json
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import pytest
import torch
from transformers import GenerationConfig, MarianConfig, MarianMTModel


def test_microinverter_source_match_does_not_also_require_inverter():
    """A nested generic source match would impose a second, wrong constraint."""
    from siglip_tools.opus_constraints import glossary_requirements

    assert glossary_requirements("板下的微型光伏逆变器") == [
        {
            "id": "microinverter",
            "source_terms": ["微型光伏逆变器"],
            "target_forms": ["microinverter", "microinverters", "Microinverter", "Microinverters"],
        }
    ]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("车库里的电缆套管", []),
        ("变压器旁的线缆套管", []),
        ("普通套管", []),
        ("变压器上的套管", ["transformer_bushing"]),
        ("变压器旁的电缆套管和绝缘套管", ["transformer_bushing"]),
        ("微逆旁边的逆变器", ["microinverter", "inverter"]),
        ("逆变器的散热鳍片和冷却鳍片", ["inverter", "cooling_fins"]),
        ("打开门能看见接线的电气柜", []),
    ],
)
def test_source_context_exclusions_and_separate_term_families(text, expected):
    """Context-free sleeve or cable sleeve must not force transformer terminology."""
    from siglip_tools.opus_constraints import glossary_requirements

    assert [entry["id"] for entry in glossary_requirements(text)] == expected


@pytest.fixture
def cached_decoder(monkeypatch):
    """The only external boundary is the four already downloaded upstream files."""
    import huggingface_hub

    repo = "transformers-community/constrained-beam-search"
    revision = "07b2f120b7db38f1d7bac617ad65ea130508f297"
    files = [
        "README.md",
        "custom_generate/generate.py",
        "custom_generate/beam_constraints.py",
        "custom_generate/beam_search.py",
    ]
    paths = [
        huggingface_hub.try_to_load_from_cache(repo, file, revision=revision) for file in files
    ]
    if not all(isinstance(path, str) for path in paths):
        pytest.skip("Pinned constrained decoder is absent from the offline HF cache")
    snapshot = Path(paths[0]).parent

    def download(**kwargs):
        assert kwargs == {"repo_id": repo, "revision": revision, "allow_patterns": files}
        return str(snapshot)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    return snapshot


def test_prepare_downloads_only_fixed_code_and_verifies_the_local_inventory(
    tmp_path, cached_decoder
):
    """Changing the code, accepting a new file, or downloading weights must fail."""
    from siglip_tools.opus_constraints import DECODER_HASHES, prepare_decoder

    output = prepare_decoder(tmp_path / "decoder")
    provenance = json.loads((output / "decoder_provenance.json").read_text())
    assert provenance["training"] is False
    assert provenance["source"]["repository_license"] is None
    assert provenance["source"]["revision"] == "07b2f120b7db38f1d7bac617ad65ea130508f297"
    assert set(provenance["artifacts"]) == {
        "README.md",
        "custom_generate/generate.py",
        "custom_generate/beam_constraints.py",
        "custom_generate/beam_search.py",
    }
    assert provenance["artifacts"] == DECODER_HASHES
    for name, digest in provenance["artifacts"].items():
        assert sha256((output / name).read_bytes()).hexdigest() == digest
        assert (output / name).read_bytes() == (cached_decoder / name).read_bytes()
        assert not (output / name).is_symlink()
    with pytest.raises(FileExistsError):
        prepare_decoder(output)


class QueryTokenizer:
    """A target/source vocabulary split catches accidental source-SPM constraints."""

    all_special_ids = [0, 1, 2]
    target_ids = {
        "inverter": [7, 8],
        "inverters": [7, 9],
        "Inverter": [10, 8],
        "Inverters": [10, 9],
        "microinverter": [11, 8],
        "microinverters": [11, 9],
        "Microinverter": [12, 8],
        "Microinverters": [12, 9],
        "bushing": [13],
        "bushings": [14],
        "Bushing": [15],
        "Bushings": [16],
        "cooling fins": [17, 18],
        "radiator fins": [19, 18],
        "Cooling fins": [20, 18],
        "Radiator fins": [21, 18],
    }

    def __init__(self):
        self.target_calls = []
        self.input_length = 3

    def __call__(self, text=None, *, text_target=None, **kwargs):
        if text_target is not None:
            assert text is None
            assert kwargs == {"add_special_tokens": False}
            self.target_calls.append(text_target)
            return {"input_ids": list(self.target_ids[text_target])}
        assert kwargs == {"padding": False, "truncation": False, "return_tensors": "pt"}
        return {"input_ids": torch.full((1, self.input_length), 4, dtype=torch.long)}

    def decode(self, sequence, *, skip_special_tokens):
        assert skip_special_tokens
        words = []
        offset = 0
        while offset < len(sequence):
            if sequence[offset] in self.all_special_ids:
                offset += 1
                continue
            for phrase, ids in self.target_ids.items():
                if sequence[offset : offset + len(ids)] == ids:
                    words.append(phrase)
                    offset += len(ids)
                    break
            else:
                words.append(f"token{sequence[offset]}")
                offset += 1
        return " ".join(words)


@pytest.fixture
def tiny_base(tmp_path):
    """Random test-fixture parameters only: no fitting or pretrained model load."""
    from siglip_tools.translation import LocalTranslator

    with torch.random.fork_rng():
        torch.manual_seed(123)
        model = (
            MarianMTModel(
                MarianConfig(
                    vocab_size=32,
                    d_model=8,
                    encoder_layers=1,
                    decoder_layers=1,
                    encoder_attention_heads=1,
                    decoder_attention_heads=1,
                    encoder_ffn_dim=16,
                    decoder_ffn_dim=16,
                    max_position_embeddings=256,
                    pad_token_id=1,
                    eos_token_id=0,
                    decoder_start_token_id=1,
                )
            )
            .eval()
            .requires_grad_(False)
        )
    settings = GenerationConfig(
        do_sample=False,
        num_beams=1,
        num_return_sequences=1,
        max_new_tokens=64,
        max_length=None,
        use_cache=True,
        forced_eos_token_id=None,
        eos_token_id=0,
        pad_token_id=1,
        decoder_start_token_id=1,
        sequence_bias={(0,): 8.0},
        min_new_tokens=1,
    )
    model.generation_config = deepcopy(settings)
    base = object.__new__(LocalTranslator)
    base.model_dir, base.device = tmp_path, torch.device("cpu")
    base.tokenizer, base.model = QueryTokenizer(), model
    base._generation_config, base._eos_ids = settings, {0}
    base._identity = {
        "model_id": "Helsinki-NLP/opus-mt-zh-en",
        "dtype": "float32",
        "generation_config": settings.to_dict(),
        "generation": settings.to_dict(),
    }
    return base


def test_no_target_uses_ordinary_beam4_and_preserves_shared_base(
    tiny_base, tmp_path, cached_decoder
):
    """Empty constraints must never invoke the custom decoder or alter beam1."""
    from siglip_tools.opus_constraints import OPUSConstraintTranslator, prepare_decoder

    before = deepcopy(tiny_base.model.generation_config.to_dict())
    original = tiny_base.translate("打开的柜门")
    beam = OPUSConstraintTranslator(tiny_base, constrained=False)
    constrained = OPUSConstraintTranslator(
        tiny_base, decoder_dir=prepare_decoder(tmp_path / "decoder")
    )
    assert constrained.translate("打开的柜门") == beam.translate("打开的柜门")
    assert beam.last_trace["constraints_applied"] is False
    assert constrained.last_trace["requirements"] == []
    assert constrained.last_trace["eos_terminated"] is True
    assert constrained.model is tiny_base.model
    assert constrained.tokenizer is tiny_base.tokenizer
    assert beam.identity["generation"]["num_beams"] == 4
    assert tiny_base.identity["generation"]["num_beams"] == 1
    assert tiny_base.model.generation_config.to_dict() == before
    assert tiny_base.translate("打开的柜门") == original
    assert all(not parameter.requires_grad for parameter in tiny_base.model.parameters())


def test_actual_upstream_callable_uses_target_tokens_and_satisfies_multiple_requirements(
    tiny_base, tmp_path, cached_decoder
):
    """Source-SPM ids or double generation preparation cannot satisfy these phrases."""
    from siglip_tools.opus_constraints import OPUSConstraintTranslator, prepare_decoder

    before = deepcopy(tiny_base.model.generation_config.to_dict())
    parameters = [parameter.detach().clone() for parameter in tiny_base.model.parameters()]
    translator = OPUSConstraintTranslator(
        tiny_base, decoder_dir=prepare_decoder(tmp_path / "decoder")
    )
    result = translator.translate("微逆与逆变器旁的变压器套管和散热片")
    trace = translator.last_trace
    assert trace["output"] == result
    assert trace["constraints_applied"] is True
    assert trace["satisfied"] is True
    assert trace["eos_terminated"] is True
    assert trace["required_token_alternatives"] == [
        [[11, 8], [11, 9], [12, 8], [12, 9]],
        [[7, 8], [7, 9], [10, 8], [10, 9]],
        [[13], [14], [15], [16]],
        [[17, 18], [19, 18], [20, 18], [21, 18]],
    ]
    assert all(requirement["satisfied"] for requirement in trace["requirements"])
    for forms in [
        ["microinverter", "microinverters", "Microinverter", "Microinverters"],
        ["inverter", "inverters", "Inverter", "Inverters"],
        ["bushing", "bushings", "Bushing", "Bushings"],
        ["cooling fins", "radiator fins", "Cooling fins", "Radiator fins"],
    ]:
        assert any(form in result for form in forms)
    assert len(tiny_base.tokenizer.target_calls) == 16
    assert tiny_base.model.generation_config.to_dict() == before
    assert all(
        torch.equal(parameter, previous)
        for parameter, previous in zip(tiny_base.model.parameters(), parameters, strict=True)
    )
    json.dumps(trace, allow_nan=False)


def test_truncation_trace_still_records_term_satisfaction_and_exact_output(
    tiny_base, tmp_path, cached_decoder
):
    """The audit must not lose output or coverage merely because EOS was absent."""
    from siglip_tools.opus_constraints import OPUSConstraintTranslator, prepare_decoder

    tiny_base._generation_config.suppress_tokens = [0]
    translator = OPUSConstraintTranslator(
        tiny_base, decoder_dir=prepare_decoder(tmp_path / "decoder")
    )
    with pytest.raises(ValueError, match="EOS"):
        translator.translate("一台逆变器")
    trace = translator.last_trace
    assert trace["status"] == "failed"
    assert trace["eos_terminated"] is False
    assert trace["output"] == tiny_base.tokenizer.decode(
        trace["sequence"], skip_special_tokens=True
    )
    assert isinstance(trace["requirements"][0]["satisfied"], bool)
    json.dumps(trace, allow_nan=False)


@pytest.mark.parametrize(
    "drift",
    [
        "changed_code",
        "unexpected_file",
        "unexpected_directory",
        "symlink_file",
        "symlink_directory",
        "symlink_root",
        "changed_manifest",
        "changed_hash_manifest",
        "weights_declaration",
    ],
)
def test_decoder_inventory_and_hash_drift_rejected_even_after_loading(
    tiny_base, tmp_path, cached_decoder, drift
):
    """A cached callable must not allow a later wrapper to bless a changed bundle."""
    from siglip_tools.opus_constraints import OPUSConstraintTranslator, prepare_decoder

    directory = prepare_decoder(tmp_path / "decoder")
    OPUSConstraintTranslator(tiny_base, decoder_dir=directory)
    provenance_path = directory / "decoder_provenance.json"
    if drift == "changed_code":
        (directory / "custom_generate/generate.py").write_text(
            "raise RuntimeError('drift executed')"
        )
    elif drift == "unexpected_file":
        (directory / "model.safetensors").write_bytes(b"weights must never be accepted")
    elif drift == "unexpected_directory":
        (directory / "extra").mkdir()
    elif drift == "symlink_file":
        file = directory / "README.md"
        file.unlink()
        file.symlink_to(cached_decoder / "README.md")
    elif drift == "symlink_directory":
        (directory / "alias").symlink_to(directory / "custom_generate", target_is_directory=True)
    elif drift == "symlink_root":
        alias = tmp_path / "alias"
        alias.symlink_to(directory, target_is_directory=True)
        directory = alias
    else:
        provenance = json.loads(provenance_path.read_text())
        if drift == "changed_manifest":
            provenance["source"]["revision"] = "main"
        elif drift == "changed_hash_manifest":
            file = "custom_generate/generate.py"
            (directory / file).write_bytes(b"raise RuntimeError('drift executed')")
            provenance["artifacts"][file] = sha256((directory / file).read_bytes()).hexdigest()
        else:
            provenance["weights_downloaded"] = True
        provenance_path.write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="decoder|symlink|inventory|provenance|hash"):
        OPUSConstraintTranslator(tiny_base, decoder_dir=directory)


def test_token_prefixes_and_duplicate_case_forms_keep_the_same_disjunction(
    tiny_base, tmp_path, cached_decoder
):
    """Upstream rejects a plural that extends a singular's full token sequence."""
    from siglip_tools.opus_constraints import OPUSConstraintTranslator, prepare_decoder

    tiny_base.tokenizer.target_ids = {
        **tiny_base.tokenizer.target_ids,
        "inverters": [7, 8, 22],
        "Inverter": [7, 8],
        "Inverters": [7, 8, 22],
    }
    translator = OPUSConstraintTranslator(
        tiny_base, decoder_dir=prepare_decoder(tmp_path / "decoder")
    )
    assert "inverter" in translator.translate("一台逆变器")
    requirement = translator.last_trace["requirements"][0]
    assert requirement["token_alternatives"] == [[7, 8], [7, 8, 22], [7, 8], [7, 8, 22]]
    assert requirement["decoder_token_alternatives"] == [[7, 8]]
    assert requirement["satisfied"] is True


def test_substring_phrase_does_not_satisfy_a_complete_inverter_word(
    tiny_base, tmp_path, cached_decoder, monkeypatch
):
    """The literal 'inverter' substring inside 'microinverter' is insufficient."""
    from siglip_tools.opus_constraints import OPUSConstraintTranslator, prepare_decoder

    translator = OPUSConstraintTranslator(
        tiny_base, decoder_dir=prepare_decoder(tmp_path / "decoder")
    )
    monkeypatch.setattr(tiny_base.tokenizer, "decode", lambda *args, **kwargs: "microinverter")
    with pytest.raises(ValueError, match="lexical constraints.*inverter"):
        translator.translate("一台逆变器")
    assert translator.last_trace["output"] == "microinverter"
    assert translator.last_trace["satisfied"] is False
    requirement = translator.last_trace["requirements"][0]
    assert requirement["tokens_satisfied"] is True
    assert requirement["phrase_satisfied"] is False


@pytest.mark.parametrize("bad_tokens", [[], [2], [0], [-1], ["7"]])
def test_unrepresentable_target_forms_fail_before_generation(
    tiny_base, tmp_path, cached_decoder, bad_tokens
):
    """An unknown/special/empty token sequence cannot encode a real constraint."""
    from siglip_tools.opus_constraints import OPUSConstraintTranslator, prepare_decoder

    tiny_base.tokenizer.target_ids = {**tiny_base.tokenizer.target_ids, "inverter": bad_tokens}
    translator = OPUSConstraintTranslator(
        tiny_base, decoder_dir=prepare_decoder(tmp_path / "decoder")
    )
    with pytest.raises(ValueError, match="encode constraint phrase"):
        translator.translate("一台逆变器")
    assert translator.last_trace["status"] == "failed"
    assert translator.last_trace["output"] is None


@pytest.mark.parametrize("text", [None, 4, "", "  ", "an English cabinet"])
def test_beam_control_rejects_invalid_source_without_a_decoder(tiny_base, tmp_path, text):
    from siglip_tools.opus_constraints import OPUSConstraintTranslator

    translator = OPUSConstraintTranslator(
        tiny_base, constrained=False, decoder_dir=tmp_path / "absent"
    )
    with pytest.raises(ValueError, match="Chinese|nonempty"):
        translator.translate(text)
    assert translator.last_trace["status"] == "failed"
    assert translator.last_trace["output"] is None


def test_source_limit_and_exact_returned_output_without_a_decoder(tiny_base, monkeypatch):
    from siglip_tools.opus_constraints import OPUSConstraintTranslator

    translator = OPUSConstraintTranslator(tiny_base, constrained=False)
    tiny_base.tokenizer.input_length = 257
    with pytest.raises(ValueError, match="256"):
        translator.translate("中文柜门")
    tiny_base.tokenizer.input_length = 256
    monkeypatch.setattr(
        tiny_base.tokenizer, "decode", lambda *args, **kwargs: "  An open cabinet.\n"
    )
    assert translator.translate("中文柜门") == "  An open cabinet.\n"
    assert translator.last_trace["output"] == "  An open cabinet.\n"
    assert translator.last_trace["status"] == "success"


def test_repeated_constraint_requests_generate_afresh_and_keep_previous_traces(
    tiny_base, tmp_path, cached_decoder
):
    from siglip_tools.opus_constraints import OPUSConstraintTranslator, prepare_decoder

    translator = OPUSConstraintTranslator(
        tiny_base, decoder_dir=prepare_decoder(tmp_path / "decoder")
    )
    calls = []
    hook = tiny_base.model.register_forward_pre_hook(lambda *args: calls.append(True))
    try:
        first = translator.translate("变压器的套管")
        first_trace, first_calls = translator.last_trace, len(calls)
        assert translator.translate("变压器的套管") == first
        assert len(calls) > first_calls
        assert translator.last_trace is not first_trace
        assert first_trace["source_text"] == "变压器的套管"
        assert first_trace["status"] == "success"
        translator.translate("打开的柜门")
        assert translator.last_trace["requirements"] == []
        assert first_trace["requirements"][0]["id"] == "transformer_bushing"
    finally:
        hook.remove()


def test_prepare_command_produces_the_hash_validated_local_bundle(tmp_path, cached_decoder, capsys):
    from siglip_tools.opus_constraints import main

    output = tmp_path / "cli-decoder"
    assert main(["prepare", "--output", str(output)]) == 0
    assert str(output) in capsys.readouterr().out
    assert (output / "decoder_provenance.json").is_file()
