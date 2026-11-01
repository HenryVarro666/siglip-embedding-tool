"""Guard contracts using token fixtures and the pinned decoder, never pretrained loads."""

import json
from copy import deepcopy
from pathlib import Path

import pytest
import test_opus_constraints as opus_fixtures
import torch
from test_constraint_fallback import Encoder as RetrievalEncoder
from test_constraint_fallback import Translator as OriginalTranslator
from test_fusion import Encoder as FusionEncoder
from test_fusion import index_at
from transformers import StoppingCriteriaList

cached_decoder = opus_fixtures.cached_decoder
tiny_base = opus_fixtures.tiny_base


PROFILE = {
    "schema_version": 1,
    "profile_id": "suffix-n8-r3-min24-q3",
    "algorithm": "suffix_consecutive_token_cycles",
    "max_new_tokens": 64,
    "num_beams": 4,
    "check_every": 4,
    "max_ngram_size": 8,
    "repeat_count": 3,
    "min_new_tokens": 24,
    "min_repeating_beams": 3,
}


def beams_at(new_tokens, repeating=4, *, period=(7, 8)):
    repeated = ([23] * new_tokens + list(period) * (new_tokens // len(period) + 1))[-new_tokens:]
    ordinary = list(range(100, 100 + new_tokens))
    return [[1] + (repeated if beam < repeating else ordinary) for beam in range(4)]


def guard_module():
    # An assertion gives RED the intended missing-feature failure, not a collection error.
    from siglip_tools import repetition_stop

    return repetition_stop


def test_candidates_cover_the_predeclared_grid_without_sharing_mutable_profiles():
    """Dropping a candidate or coupling profiles would invalidate development selection."""
    module = guard_module()
    profiles = module.candidate_profiles()
    assert len(profiles) == 16
    assert len({p["profile_id"] for p in profiles}) == 16
    assert {
        (p["max_ngram_size"], p["repeat_count"], p["min_new_tokens"], p["min_repeating_beams"])
        for p in profiles
    } == {
        (n, r, minimum, quorum)
        for n in (8, 12)
        for r in (3, 4)
        for minimum in (24, 32)
        for quorum in (3, 4)
    }
    profiles[0]["max_new_tokens"] = 5
    assert all(p["max_new_tokens"] == 64 for p in module.candidate_profiles())


def test_suffix_cycle_reports_exact_period_and_beam_quorum():
    """A minority repeating beam must never satisfy a three-beam rule."""
    module = guard_module()
    evidence = module.detect_repetition(beams_at(24, 3), PROFILE, {0, 1, 2})
    assert evidence["new_tokens"] == 24
    assert evidence["eligible"] is True
    assert evidence["repeating_beams"] == 3
    assert evidence["triggered"] is True
    assert evidence["beam_evidence"][0] == {
        "beam_index": 0,
        "period": 2,
        "cycle": [7, 8],
        "repeat_count": 12,
        "suffix_start": 1,
    }
    assert evidence["beam_evidence"][3]["period"] is None
    assert module.detect_repetition(beams_at(24, 2), PROFILE, {0, 1, 2})["triggered"] is False
    strict = {**PROFILE, "profile_id": "suffix-n8-r3-min24-q4", "min_repeating_beams": 4}
    assert module.detect_repetition(beams_at(24, 3), strict, {0, 1, 2})["triggered"] is False


@pytest.mark.parametrize(
    "step,want",
    [(20, False), (23, False), (24, True), (26, False), (28, True), (60, True), (64, False)],
)
def test_only_predeclared_check_steps_can_trigger(step, want):
    assert guard_module().detect_repetition(beams_at(step), PROFILE, {0, 1, 2})["triggered"] is want


@pytest.mark.parametrize("special", [0, 1, 2, 31])
def test_special_tokens_mixed_into_cycle_do_not_trigger(special):
    evidence = guard_module().detect_repetition(
        beams_at(24, period=(7, special)), PROFILE, {0, 1, 2, 31}
    )
    assert evidence["repeating_beams"] == 0
    assert evidence["triggered"] is False


def test_repeat_must_be_consecutive_at_suffix_and_fit_max_period():
    module = guard_module()
    # A repeated prefix followed by novel text is not a repeating suffix.
    prefix = [[1] + [7, 8] * 8 + list(range(100, 108)) for _ in range(4)]
    assert module.detect_repetition(prefix, PROFILE, {0, 1})["triggered"] is False
    nine = beams_at(36, period=tuple(range(10, 19)))
    assert module.detect_repetition(nine, PROFILE, {0, 1})["triggered"] is False
    larger = {**PROFILE, "profile_id": "suffix-n12-r3-min24-q3", "max_ngram_size": 12}
    assert module.detect_repetition(nine, larger, {0, 1})["triggered"] is True


@pytest.mark.parametrize(
    "field,value",
    [
        ("num_beams", 1),
        ("max_new_tokens", 63),
        ("check_every", True),
        ("min_new_tokens", 28),
        ("repeat_count", 2),
        ("min_repeating_beams", 1),
        ("max_ngram_size", 99),
        ("algorithm", "other"),
        ("profile_id", "renamed"),
        ("schema_version", 2),
    ],
)
def test_invalid_profile_fails_before_decoder_or_model_allocation(field, value, monkeypatch):
    module = guard_module()
    from siglip_tools import opus_constraints

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid profile must fail before decoder loading")

    monkeypatch.setattr(opus_constraints, "_load_decoder", forbidden)
    with pytest.raises(ValueError, match="profile"):
        module.GuardedOPUSConstraintTranslator(object(), profile={**PROFILE, field: value})


class SteppingDecoder:
    """Model-boundary double that executes all criteria inside a generation loop."""

    def __init__(self, *, repeating=4, fail=None):
        self.repeating = repeating
        self.fail = fail
        self.steps = []
        self.criteria = []
        self.finalized = False

    def __call__(self, model, *, input_ids, generation_config, stopping_criteria):
        assert generation_config.num_beams == 4
        assert generation_config.max_new_tokens == 64
        assert isinstance(stopping_criteria, StoppingCriteriaList)
        assert generation_config.force_words_ids == [[[7, 8], [7, 9], [10, 8], [10, 9]]]
        self.criteria.append(stopping_criteria[0])
        if self.fail:
            raise self.fail
        for step in range(1, 65):
            self.steps.append(step)
            ids = torch.tensor(beams_at(step, self.repeating), dtype=torch.long)
            if any(stopping_criteria(ids, None)):
                break
        self.finalized = True
        # Deliberately emulate a finalizer appending EOS; active guard must abort first.
        return torch.tensor([[1, 7, 8, 0]], dtype=torch.long)


@pytest.fixture
def scripted_guard(monkeypatch, tiny_base):
    from siglip_tools import opus_constraints

    decoder = SteppingDecoder()
    monkeypatch.setattr(
        opus_constraints, "_load_decoder", lambda directory: (decoder, {"fixture": True})
    )
    guard_class = guard_module().GuardedOPUSConstraintTranslator

    def construct(*, profile=None, observe_only=False):
        return guard_class(tiny_base, profile=profile or PROFILE, observe_only=observe_only)

    return construct, decoder, tiny_base


def test_active_callback_aborts_before_finalization_and_retains_four_raw_beams(scripted_guard):
    construct, decoder, base = scripted_guard
    translator = construct()
    with pytest.raises(guard_module().RepetitionAbort, match="repetition"):
        translator.translate("一台逆变器")
    assert decoder.steps[-1] == 24
    assert decoder.finalized is False
    trace = translator.last_trace
    assert trace["status"] == "failed"
    assert trace["eos_terminated"] is False
    assert trace["sequence"] == [1] + [7, 8] * 12
    assert trace["output"] == " ".join(["inverter"] * 12)
    assert trace["requirements"][0]["satisfied"] is True
    assert trace["satisfied"] is True
    stop = trace["repetition_stop"]
    assert stop["profile"] == PROFILE
    assert stop["special_ids"] == [0, 1, 2]
    assert stop["enabled"] is True
    assert stop["observe_only"] is False
    assert stop["triggered"] is True
    assert stop["trigger_new_tokens"] == stop["generated_new_tokens"] == 24
    assert stop["observed_generation_steps"] == 24
    assert stop["observed_steps_complete"] is True
    assert stop["length_scope"] == "partial_beam_sequence"
    assert stop["snapshots"][-1]["beams"] == beams_at(24)
    assert stop["snapshots"][-1]["triggered"] is True
    from siglip_tools.encoder import identity_fingerprint

    assert stop["profile_sha256"] == identity_fingerprint(PROFILE)
    json.dumps(trace, allow_nan=False)


def test_observation_records_every_check_and_does_not_stop_generation(scripted_guard):
    construct, decoder, _ = scripted_guard
    translator = construct(observe_only=True)
    assert translator.translate("一台逆变器") == "inverter"
    assert decoder.finalized is True
    assert decoder.steps[-1] == 64
    stop = translator.last_trace["repetition_stop"]
    assert [s["new_tokens"] for s in stop["snapshots"]] == list(range(24, 61, 4))
    assert stop["triggered"] is True
    assert stop["trigger_new_tokens"] == 24
    # A selected finalized sequence can be shorter than the observed beam work;
    # neither sequence length nor callback coverage proves physical forward cost.
    assert stop["generated_new_tokens"] == 3
    assert stop["observed_generation_steps"] == 64
    assert stop["observed_steps_complete"] is False
    assert stop["length_scope"] == "finalized_sequence"
    assert translator.last_trace["eos_terminated"] is True


def test_nonrepeating_minority_does_not_interrupt_stricter_quorum(scripted_guard):
    construct, decoder, _ = scripted_guard
    decoder.repeating = 3
    translator = construct(
        profile={**PROFILE, "profile_id": "suffix-n8-r3-min24-q4", "min_repeating_beams": 4}
    )
    assert translator.translate("一台逆变器") == "inverter"
    assert translator.last_trace["repetition_stop"]["triggered"] is False


def test_successive_requests_reset_criteria_without_mutating_base_or_prior_trace(scripted_guard):
    construct, decoder, base = scripted_guard
    before = deepcopy(base.identity), deepcopy(base._generation_config.to_dict())
    translator = construct()
    for _ in range(2):
        with pytest.raises(guard_module().RepetitionAbort):
            translator.translate("一台逆变器")
        if len(decoder.criteria) == 1:
            first = translator.last_trace
    assert translator.last_trace is not first
    assert translator.last_trace == first
    assert decoder.criteria[0] is not decoder.criteria[1]
    assert base.identity == before[0]
    assert base._generation_config.to_dict() == before[1]
    assert base.model.generation_config.to_dict() == before[1]
    identity = translator.identity
    assert identity["repetition_stop"]["profile"] == PROFILE
    identity["repetition_stop"]["profile"]["max_new_tokens"] = 1
    assert translator.identity["repetition_stop"]["profile"]["max_new_tokens"] == 64


def test_no_target_path_remains_ordinary_beam4_with_no_stale_guard(scripted_guard):
    construct, decoder, base = scripted_guard
    from siglip_tools.opus_constraints import OPUSConstraintTranslator

    translator = construct()
    with pytest.raises(guard_module().RepetitionAbort):
        translator.translate("一台逆变器")
    old = translator.last_trace
    ordinary = OPUSConstraintTranslator(base)
    assert translator.translate("打开的柜门") == ordinary.translate("打开的柜门")
    assert translator.last_trace["repetition_stop"]["enabled"] is False
    assert translator.last_trace["repetition_stop"]["snapshots"] == []
    assert translator.last_trace["repetition_stop"]["triggered"] is False
    stripped = deepcopy(translator.last_trace)
    stripped.pop("repetition_stop")
    assert stripped == ordinary.last_trace
    assert old["repetition_stop"]["triggered"] is True


@pytest.mark.parametrize(
    "error", [RuntimeError("device unavailable"), ValueError("invalid decoder input")]
)
def test_infrastructure_errors_propagate_without_rejected_generation(scripted_guard, error):
    construct, decoder, _ = scripted_guard
    decoder.fail = error
    translator = construct()
    with pytest.raises(type(error), match=str(error)):
        translator.translate("一台逆变器")
    assert translator.last_trace.get("sequence") is None
    assert translator.last_trace["output"] is None


def test_existing_fallback_discards_partial_beam_and_generates_original_afresh(
    tmp_path, scripted_guard
):
    construct, decoder, _ = scripted_guard
    from siglip_tools.constraint_fallback import timed_fallback_search

    constrained, original = construct(), OriginalTranslator(outputs=("English", "fresh English"))
    encoder = RetrievalEncoder()
    results = [
        timed_fallback_search(
            index_at(tmp_path),
            encoder,
            constrained,
            original,
            "一台逆变器",
            mode="auto_en",
            fallback="opus",
        )
        for _ in range(2)
    ]
    assert [result["translation"] for result in results] == ["English", "fresh English"]
    assert encoder.queries == [["English"], ["fresh English"]]
    assert [result["matches"][0].path for result in results] == ["1.png", "0.png"]
    assert all(result["fallback_triggered"] for result in results)
    assert all(result["primary_failure"]["type"] == "RepetitionAbort" for result in results)
    assert all(result["constraint_trace"]["eos_terminated"] is False for result in results)


def test_actual_pinned_decoder_aborts_inside_real_generation(tiny_base, tmp_path, cached_decoder):
    from siglip_tools.opus_constraints import prepare_decoder

    tiny_base._generation_config.suppress_tokens = [0]
    tiny_base._generation_config.sequence_bias = {(7,): 20.0}
    translator = guard_module().GuardedOPUSConstraintTranslator(
        tiny_base, profile=PROFILE, decoder_dir=prepare_decoder(tmp_path / "decoder")
    )
    with pytest.raises(guard_module().RepetitionAbort):
        translator.translate("一台逆变器")
    assert 24 <= translator.last_trace["repetition_stop"]["trigger_new_tokens"] <= 60
    assert translator.last_trace["eos_terminated"] is False


def cli_fixture(tmp_path, monkeypatch, scripted_guard):
    module = guard_module()
    construct, decoder, base = scripted_guard
    # The tiny test model's tuple-key bias is not part of the saved OPUS identity.
    base._generation_config.sequence_bias = None
    path = tmp_path / "gallery.npz"
    index_at(tmp_path).save(path)
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps(PROFILE))

    def encoder(model_id, revision, device, local_files_only):
        assert (model_id, revision, device, local_files_only) == (
            "test/model",
            "frozen",
            "cpu",
            True,
        )
        class QueryEncoder(FusionEncoder):
            def encode_texts(self, texts, batch_size=8):
                return super().encode_texts(
                    ["中文" if text == "一台逆变器" else text for text in texts], batch_size
                )

        result = QueryEncoder()
        result.device = torch.device("cpu")
        return result

    class FreshOriginal(OriginalTranslator):
        def __init__(self, directory, *, device):
            assert directory == Path("models/opus-mt-zh-en")
            assert device == "cpu"
            super().__init__()

    monkeypatch.setattr(module, "SigLIPEmbedder", encoder)
    monkeypatch.setattr(module, "LocalTranslator", FreshOriginal)
    monkeypatch.setattr(
        module, "GuardedOPUSConstraintTranslator", lambda *args, **kwargs: construct()
    )
    return module, [
        "search",
        "--index",
        str(path),
        "--query",
        "一台逆变器",
        "--profile",
        str(profile),
        "--device",
        "cpu",
        "--top-k",
        "3",
        "--local-files-only",
    ]


def test_module_cli_uses_pinned_index_and_real_fusion_with_fresh_fallback(
    tmp_path, monkeypatch, scripted_guard, capsys
):
    module, argv = cli_fixture(tmp_path, monkeypatch, scripted_guard)
    assert module.main(argv) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["ok"] is True
    assert result["automatic_translation"]["effective_route"] == "original_fusion"
    assert result["automatic_translation"]["query_texts"] == ["一台逆变器", "English"]
    assert result["results"][0]["path"] == "2.png"
    assert result["results"][0]["score"] == pytest.approx(0.7)
    assert result["query_timing_ms"]["constraint_translation_ms"] > 0


def test_module_cli_expected_rejection_is_json_with_exit_two(
    tmp_path, monkeypatch, scripted_guard, capsys
):
    module, argv = cli_fixture(tmp_path, monkeypatch, scripted_guard)
    assert module.main(argv + ["--fallback", "error"]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["ok"] is False
    assert result["results"] == []
    assert result["automatic_translation"]["constraint_trace"]["repetition_stop"]["triggered"]
    assert result["automatic_translation"]["query_texts"] == []


@pytest.mark.parametrize(
    "extra",
    [["--top-k", "0"], ["--mode", "unknown"], ["--device", "bogus"], ["--query", "English"]],
)
def test_invalid_cli_options_fail_before_index_or_model_load(monkeypatch, extra, capsys):
    module = guard_module()

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid query options must fail before allocation")

    monkeypatch.setattr(module.ImageIndex, "load", forbidden)
    monkeypatch.setattr(module, "SigLIPEmbedder", forbidden)
    try:
        status = module.main(["search", "--index", "absent.npz", "--query", "中文"] + extra)
    except SystemExit as exc:
        status = exc.code
    assert status == 2


def test_invalid_profile_cli_reports_json_before_allocating_weights(tmp_path, monkeypatch, capsys):
    module = guard_module()
    path = tmp_path / "bad.json"
    path.write_text('{"schema_version": 1}')
    monkeypatch.setattr(
        module, "SigLIPEmbedder", lambda *a, **kw: pytest.fail("Unexpected weights")
    )
    assert (
        module.main(["search", "--index", "absent.npz", "--query", "中文", "--profile", str(path)])
        == 2
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["results"] == []
