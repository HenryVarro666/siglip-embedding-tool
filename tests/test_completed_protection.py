"""EOS protection at the real pinned scorer boundary, using offline tiny fixtures."""

import importlib
import importlib.util
import json
from copy import deepcopy
from pathlib import Path
from types import FunctionType, SimpleNamespace

import pytest
import test_opus_constraints as opus_fixtures
import torch
from test_constraint_fallback import Encoder as RetrievalEncoder
from test_constraint_fallback import Translator as OriginalTranslator
from test_fusion import index_at

cached_decoder = opus_fixtures.cached_decoder
tiny_base = opus_fixtures.tiny_base

PROFILE = {
    "schema_version": 2,
    "profile_id": "bounded-n16-r2-min32-q4-tail4",
    "algorithm": "bounded_tail_consecutive_token_cycles",
    "max_new_tokens": 64,
    "num_beams": 4,
    "check_every": 4,
    "max_ngram_size": 16,
    "repeat_count": 2,
    "min_new_tokens": 32,
    "min_repeating_beams": 4,
    "min_cycle_tokens": 4,
    "tail_tolerance": 4,
}
REQUIREMENTS = [
    {
        "id": "inverter",
        "source_terms": ["逆变器"],
        "target_forms": ["inverter", "inverters", "Inverter", "Inverters"],
        "token_alternatives": [[7, 8], [7, 9], [10, 8], [10, 9]],
        "decoder_token_alternatives": [[7, 8], [7, 9], [10, 8], [10, 9]],
    }
]


def protection_module():
    # RED is an assertion about the absent feature, rather than a collection error.
    assert importlib.util.find_spec("siglip_tools.completed_protection") is not None, (
        "Completed-candidate protection is not implemented"
    )
    return importlib.import_module("siglip_tools.completed_protection")


def active_tokens(step, *, repeating=True):
    if repeating:
        return [1] + ([7, 8, 10, 8] * 17)[:step]
    return [1] + ([7, 8] + [11 + token % 20 for token in range(step)])[:step]


def _driver_generate(model, *args, **kwargs):
    return _constrained_beam_search(model, *args, **kwargs)  # noqa: F821


def _driver_loop(model, *, input_ids, generation_config, stopping_criteria=()):
    """Control proposal tensors while keeping the actual pinned scorer intact."""
    state = DRIVER  # noqa: F821
    scorer = ConstrainedBeamSearchScorer(  # noqa: F821
        batch_size=1,
        num_beams=4,
        constraints=[DisjunctiveConstraint(words) for words in generation_config.force_words_ids],  # noqa: F821
        device=input_ids.device,
        length_penalty=generation_config.length_penalty,
        do_early_stopping="never",
        max_length=65,
    )
    state.scorers.append(scorer)
    state.criteria.append(stopping_criteria)
    for step in range(1, 64):
        prefix = torch.tensor([active_tokens(step - 1)] * 4)
        proposals = state.events.get(step, [])
        tokens = [token for token, _, _ in proposals] + [7, 8, 10, 8, 7, 8, 10, 8]
        parents = [parent for _, parent, _ in proposals] + [0, 1, 2, 3, 0, 1, 2, 3]
        scores = [score for _, _, score in proposals] + [-20.0 - rank for rank in range(8)]
        output = scorer.process(
            prefix,
            torch.tensor([scores]),
            torch.tensor([tokens]),
            torch.tensor([parents]),
            torch.full((4, 32), -30.0),
            pad_token_id=1,
            eos_token_id=state.eos_ids,
            decoder_prompt_len=1,
        )
        state.process_outputs.append({key: value.tolist() for key, value in output.items()})
        state.steps.append(step)
        active = torch.tensor([active_tokens(step, repeating=state.repeating)] * 4)
        if stopping_criteria:
            if any(stopping_criteria(active, None)):
                break
    state.finalized = True
    result = scorer.finalize(
        active,
        torch.tensor([-64.0] * 4),
        torch.tensor([7, 8, 10, 8]),
        torch.tensor([0, 1, 2, 3]),
        max_length=65,
        pad_token_id=1,
        eos_token_id=state.eos_ids,
        decoder_prompt_len=1,
    )
    return result["sequences"]


@pytest.fixture
def driver(monkeypatch, tiny_base, tmp_path, cached_decoder):
    from siglip_tools import opus_constraints

    tiny_base._generation_config.length_penalty = 1.0
    directory = opus_constraints.prepare_decoder(tmp_path / "decoder")
    validated, provenance = opus_constraints._load_decoder(directory)
    original_globals = validated.__globals__
    state = SimpleNamespace(
        events={5: [(0, 0, -1.0)]},
        eos_ids=[0],
        repeating=True,
        scorers=[],
        criteria=[],
        steps=[],
        process_outputs=[],
        finalized=False,
    )
    loop_globals = dict(_driver_loop.__globals__)
    loop_globals.update(
        DRIVER=state,
        ConstrainedBeamSearchScorer=original_globals["ConstrainedBeamSearchScorer"],
        DisjunctiveConstraint=original_globals["DisjunctiveConstraint"],
    )
    loop = FunctionType(_driver_loop.__code__, loop_globals)
    loop.__kwdefaults__ = _driver_loop.__kwdefaults__
    outer_globals = dict(_driver_generate.__globals__)
    outer_globals.update(_constrained_beam_search=loop)
    outer = FunctionType(_driver_generate.__code__, outer_globals)
    # The validated outer normally has this name too; keep both original maps separate.
    outer_globals["ConstrainedBeamSearchScorer"] = original_globals["ConstrainedBeamSearchScorer"]
    monkeypatch.setattr(opus_constraints, "_load_decoder", lambda directory: (outer, provenance))
    state.outer, state.loop = outer, loop
    state.original_globals = original_globals
    state.base = tiny_base
    state.directory = directory
    return state


def construct(driver):
    return protection_module().CompletedProtectedTranslator(
        driver.base, profile=PROFILE, decoder_dir=driver.directory
    )


def test_candidate_requires_original_tokens_whole_words_and_nonempty_text():
    # Dropping either lexical check would admit the literal invalid cases below.
    module = protection_module()
    tokenizer = opus_fixtures.QueryTokenizer()
    valid = module.validate_candidate([1, 7, 8], REQUIREMENTS, tokenizer)
    assert valid["output"] == "inverter"
    assert valid["nonempty"] is valid["satisfied"] is valid["valid"] is True
    assert valid["requirements"][0]["token_alternatives"] == REQUIREMENTS[0]["token_alternatives"]
    for text, prefix, want_tokens, want_phrase in [
        ("microinverter", [1, 7, 8], True, False),
        ("inverter", [1, 11, 8], False, True),
        ("", [1, 7, 8], True, False),
        ("inverters", [1, 7, 9], True, True),
    ]:
        tokenizer.decode = lambda sequence, *, skip_special_tokens, text=text: text
        result = module.validate_candidate(prefix, REQUIREMENTS, tokenizer)
        assert result["requirements"][0]["tokens_satisfied"] is want_tokens
        assert result["requirements"][0]["phrase_satisfied"] is want_phrase
        assert result["valid"] is (want_tokens and want_phrase and bool(text))


def test_real_process_eos_protects_and_preserves_original_search_and_finalization(driver):
    # Returning a completion early or modifying scorer output changes this continuation.
    from siglip_tools.opus_constraints import OPUSConstraintTranslator

    baseline = OPUSConstraintTranslator(driver.base, decoder_dir=driver.directory)
    baseline_english = baseline.translate("一台逆变器")
    baseline_sequence = baseline.last_trace["sequence"]
    baseline_outputs = deepcopy(driver.process_outputs)
    driver.process_outputs.clear()
    translator = construct(driver)
    assert translator.translate("一台逆变器") == baseline_english
    assert translator.last_trace["sequence"] == baseline_sequence
    assert driver.process_outputs == baseline_outputs
    assert driver.finalized is True
    stop = translator.last_trace["repetition_stop"]
    assert stop["triggered"] is False
    assert stop["trigger_new_tokens"] is None
    assert stop["protected_trigger_steps"] == list(range(32, 64, 4))
    snapshot = stop["snapshots"][0]
    assert snapshot["triggered"] is snapshot["protected"] is True
    assert snapshot["abort"] is False
    assert snapshot["candidate_pool"] == "inspected"
    candidate = snapshot["completed_candidates"][0]
    assert candidate["current_step"] == 32
    assert candidate["retained"] is True
    assert candidate["validation"]["valid"] is True
    event = stop["candidate_protection"]["eos_candidates"][0]
    assert event["prefix"] == [1, 7, 8, 10, 8]
    assert event["eos_step"] == 5
    assert event["normalized_score"] == -0.2
    assert event["witnesses"][0]["eos_token_id"] == 0
    assert event["witnesses"][0]["rank"] == 0
    assert event["witnesses"][0]["parent_beam_index"] == 0
    assert module_validator(translator.last_trace)
    assert translator._active_observer is None


def module_validator(trace):
    return protection_module().validate_protection_trace(trace)


def test_finalizer_only_eos_cannot_prevent_real_repetition_abort(driver):
    # Counting finalizer-added EOS as genuine completion would turn this into success.
    from siglip_tools.opus_constraints import OPUSConstraintTranslator
    from siglip_tools.repetition_stop import RepetitionAbort

    driver.events.clear()
    ordinary = OPUSConstraintTranslator(driver.base, decoder_dir=driver.directory)
    assert ordinary.translate("一台逆变器")
    assert ordinary.last_trace["eos_terminated"] is True
    driver.finalized = False
    translator = construct(driver)
    with pytest.raises(RepetitionAbort):
        translator.translate("一台逆变器")
    stop = translator.last_trace["repetition_stop"]
    assert driver.finalized is False
    assert stop["triggered"] is True
    assert stop["trigger_new_tokens"] == stop["generated_new_tokens"] == 32
    assert stop["snapshots"][-1]["completed_candidates"] == []
    assert stop["snapshots"][-1]["abort"] is True
    assert stop["candidate_protection"]["eos_candidates"] == []
    assert module_validator(translator.last_trace)


def test_evicted_valid_completion_does_not_protect_current_invalid_pool(driver):
    # An ever-seen-completion flag would protect after the valid completion is evicted.
    from siglip_tools.repetition_stop import RepetitionAbort

    original_decode = driver.base.tokenizer.decode
    driver.base.tokenizer.decode = lambda ids, **kw: (
        original_decode(ids, **kw) if len(ids) <= 5 else "microinverter"
    )
    driver.events = {5: [(0, 0, -50.0)], 10: [(0, parent, -1.0 - parent) for parent in range(4)]}
    translator = construct(driver)
    with pytest.raises(RepetitionAbort):
        translator.translate("一台逆变器")
    stop = translator.last_trace["repetition_stop"]
    event_ids = [event["candidate_id"] for event in stop["candidate_protection"]["eos_candidates"]]
    assert len(event_ids) == 5
    assert event_ids[0] not in stop["snapshots"][-1]["pool_candidate_ids"]
    assert len(stop["snapshots"][-1]["completed_candidates"]) == 4
    assert all(not c["validation"]["valid"] for c in stop["snapshots"][-1]["completed_candidates"])
    assert stop["snapshots"][-1]["protected"] is False
    assert module_validator(translator.last_trace)


@pytest.mark.parametrize("text", ["microinverter", "", "inverterish"])
def test_token_complete_but_lexically_invalid_pool_does_not_protect(driver, text):
    # Upstream token-complete is insufficient to pass original lexical validation.
    from siglip_tools.repetition_stop import RepetitionAbort

    driver.base.tokenizer.decode = lambda ids, **kwargs: text
    translator = construct(driver)
    with pytest.raises(RepetitionAbort):
        translator.translate("一台逆变器")
    assert translator.last_trace["repetition_stop"]["snapshots"][-1]["protected"] is False
    assert module_validator(translator.last_trace)


def test_all_tied_eos_witnesses_and_actual_nonfirst_eos_id_are_preserved(driver):
    # Choosing EOS[0] or dropping tied matches would manufacture the witness.
    driver.eos_ids = [0, 2]
    driver.base._eos_ids = {0, 2}
    driver.base._generation_config.eos_token_id = [0, 2]
    driver.events = {5: [(2, 0, -1.0), (0, 0, -1.0)]}
    translator = construct(driver)
    assert translator.translate("一台逆变器")
    events = translator.last_trace["repetition_stop"]["candidate_protection"]["eos_candidates"]
    assert len(events) == 2
    assert all([w["eos_token_id"] for w in event["witnesses"]] == [2, 0] for event in events)
    assert module_validator(translator.last_trace)


def test_nonrepeating_long_pattern_records_checks_without_inspecting_candidates(driver):
    # The repetition detector must still distinguish long ordinary active sequences.
    driver.events.clear()
    driver.repeating = False
    translator = construct(driver)
    assert translator.translate("一台逆变器")
    stop = translator.last_trace["repetition_stop"]
    assert [s["new_tokens"] for s in stop["snapshots"]] == list(range(32, 64, 4))
    assert all(not s["triggered"] and not s["abort"] for s in stop["snapshots"])
    assert all(s["candidate_pool"] == "not_requested_no_raw_trigger" for s in stop["snapshots"])
    assert module_validator(translator.last_trace)


def test_requests_reset_pool_and_no_term_path_preserves_ordinary_beam4(driver):
    # Stale retained candidates must not survive into another or unconstrained request.
    from siglip_tools.opus_constraints import OPUSConstraintTranslator
    from siglip_tools.repetition_stop import RepetitionAbort

    translator = construct(driver)
    translator.translate("一台逆变器")
    first = deepcopy(translator.last_trace)
    driver.events.clear()
    with pytest.raises(RepetitionAbort):
        translator.translate("一台逆变器")
    assert translator.last_trace["repetition_stop"]["candidate_protection"]["eos_candidates"] == []
    assert first["repetition_stop"]["protected_trigger_steps"]
    ordinary = OPUSConstraintTranslator(driver.base)
    assert translator.translate("打开的柜门") == ordinary.translate("打开的柜门")
    stop = translator.last_trace["repetition_stop"]
    assert stop["enabled"] is stop["triggered"] is False
    assert stop["snapshots"] == stop["protected_trigger_steps"] == []
    assert stop["candidate_protection"]["pool_history"] == []
    assert module_validator(translator.last_trace)
    assert translator._active_observer is None


def test_observer_is_request_local_and_does_not_mutate_source_globals_or_configs(driver):
    # Patching a cached module/global scorer contaminates all following translators.
    before_base = deepcopy(driver.base.identity), deepcopy(driver.base._generation_config.to_dict())
    before_model = deepcopy(driver.base.model.generation_config.to_dict())
    before_outer, before_loop = dict(driver.outer.__globals__), dict(driver.loop.__globals__)
    before_source = {p: p.read_bytes() for p in driver.directory.rglob("*") if p.is_file()}
    translator = construct(driver)
    translator.translate("一台逆变器")
    assert driver.outer.__globals__ == before_outer
    assert driver.loop.__globals__ == before_loop
    assert {p: p.read_bytes() for p in before_source} == before_source
    assert driver.base.identity == before_base[0]
    assert driver.base._generation_config.to_dict() == before_base[1]
    assert driver.base.model.generation_config.to_dict() == before_model
    assert driver.scorers[-1].__class__ is not before_loop["ConstrainedBeamSearchScorer"]
    assert driver.scorers[-1].__class__.__mro__[1] is before_loop["ConstrainedBeamSearchScorer"]
    identity = translator.identity
    assert identity["opus_decoding"] == "completed_candidate_protected_constrained_beam4"
    identity["candidate_protection"]["rule"]["training"] = True
    assert translator.identity["candidate_protection"]["rule"]["training"] is False


def test_contract_mismatch_propagates_as_infrastructure_error_and_clears_state(driver):
    # A missing observer seam must never silently fall back to unprotected generation.
    driver.outer.__globals__.pop("_constrained_beam_search")
    translator = construct(driver)
    with pytest.raises(RuntimeError, match="observer contract"):
        translator.translate("一台逆变器")
    assert translator.last_trace["output"] is None
    assert translator._active_observer is None


def test_actual_pinned_tiny_generation_aborts_without_eos(tiny_base, tmp_path, cached_decoder):
    # Exercise the real loop in addition to controlled scorer proposal characterization.
    from siglip_tools.opus_constraints import prepare_decoder
    from siglip_tools.repetition_stop import RepetitionAbort

    tiny_base._generation_config.suppress_tokens = [0]
    tiny_base._generation_config.sequence_bias = {(7,): 20.0}
    translator = protection_module().CompletedProtectedTranslator(
        tiny_base, profile=PROFILE, decoder_dir=prepare_decoder(tmp_path / "decoder")
    )
    with pytest.raises(RepetitionAbort):
        translator.translate("一台逆变器")
    assert 32 <= translator.last_trace["repetition_stop"]["trigger_new_tokens"] <= 60
    assert module_validator(translator.last_trace)


def test_real_abort_retrieves_only_fresh_fallback_and_never_rejected_partial(driver, tmp_path):
    # Reusing partial diagnostic English would leak a rejected hypothesis into retrieval.
    from siglip_tools.constraint_fallback import timed_fallback_search

    driver.events.clear()
    original = OriginalTranslator(outputs=("English", "fresh English"))
    encoder = RetrievalEncoder()
    translator = construct(driver)
    outcomes = [
        timed_fallback_search(
            index_at(tmp_path),
            encoder,
            translator,
            original,
            "一台逆变器",
            "auto_en",
            fallback="opus",
        )
        for _ in range(2)
    ]
    assert [outcome["translation"] for outcome in outcomes] == ["English", "fresh English"]
    assert encoder.queries == [["English"], ["fresh English"]]
    assert all(outcome["primary_failure"]["type"] == "RepetitionAbort" for outcome in outcomes)
    assert all(outcome["constraint_trace"]["eos_terminated"] is False for outcome in outcomes)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda t: t["repetition_stop"]["snapshots"][0].update(protected=False),
        lambda t: t["repetition_stop"]["snapshots"][0].update(abort=True),
        lambda t: t["repetition_stop"].update(triggered=True, trigger_new_tokens=32),
        lambda t: t["repetition_stop"]["candidate_protection"]["eos_candidates"][0].update(
            normalized_score=0.0
        ),
        lambda t: t["repetition_stop"]["candidate_protection"]["eos_candidates"][0]["witnesses"][
            0
        ].update(rank=4),
        lambda t: t["repetition_stop"]["candidate_protection"]["eos_candidates"][0]["witnesses"][
            0
        ].update(phase="finalize"),
        lambda t: t["repetition_stop"]["candidate_protection"]["eos_candidates"][0].update(
            eos_step=33
        ),
        lambda t: t["repetition_stop"]["snapshots"][0]["completed_candidates"][0][
            "validation"
        ].update(output="microinverter"),
        lambda t: t["repetition_stop"]["snapshots"][0]["completed_candidates"][0].update(
            candidate_id=999
        ),
        lambda t: t["repetition_stop"]["candidate_protection"]["pool_history"][31].update(
            pool_candidate_ids=[]
        ),
    ],
)
def test_recorded_replay_rejects_inconsistent_or_inserted_candidate_evidence(driver, mutate):
    # Changing any audit decision without matching actual recorded evidence must fail replay.
    translator = construct(driver)
    translator.translate("一台逆变器")
    changed = deepcopy(translator.last_trace)
    mutate(changed)
    with pytest.raises(ValueError):
        module_validator(changed)


@pytest.mark.parametrize("extra", [["--top-k", "0"], ["--device", "bogus"], ["--query", "English"]])
def test_invalid_cli_fails_before_index_or_model_allocation(monkeypatch, extra):
    module = protection_module()
    monkeypatch.setattr(module.ImageIndex, "load", lambda *a: pytest.fail("Unexpected index"))
    monkeypatch.setattr(module, "SigLIPEmbedder", lambda *a, **kw: pytest.fail("Unexpected model"))
    try:
        status = module.main(["search", "--index", "missing.npz", "--query", "中文"] + extra)
    except SystemExit as exc:
        status = exc.code
    assert status == 2


def test_invalid_profile_cli_is_json_without_allocation(tmp_path, monkeypatch, capsys):
    module = protection_module()
    path = tmp_path / "bad-profile.json"
    path.write_text(json.dumps({**PROFILE, "min_new_tokens": 24}))
    monkeypatch.setattr(module, "SigLIPEmbedder", lambda *a, **kw: pytest.fail("Unexpected model"))
    assert (
        module.main(["search", "--index", "missing.npz", "--query", "中文", "--profile", str(path)])
        == 2
    )
    assert json.loads(capsys.readouterr().out)["ok"] is False


def test_replay_rejects_claimed_nonabort_that_never_entered_finalizer(driver):
    # A fabricated successful result with no finalizer must not pass the audit.
    translator = construct(driver)
    translator.translate("一台逆变器")
    changed = deepcopy(translator.last_trace)
    changed["repetition_stop"]["candidate_protection"]["finalize_entered"] = False
    with pytest.raises(ValueError, match="finaliz"):
        module_validator(changed)


def test_all_original_requirements_are_checked_independently():
    # Checking only the first required term would pass this missing-bushing candidate.
    module = protection_module()
    requirements = REQUIREMENTS + [
        {
            "id": "transformer_bushing",
            "source_terms": ["套管"],
            "target_forms": ["bushing"],
            "token_alternatives": [[13]],
            "decoder_token_alternatives": [[13]],
        }
    ]
    tokenizer = opus_fixtures.QueryTokenizer()
    missing = module.validate_candidate([1, 7, 8], requirements, tokenizer)
    assert missing["requirements"][0]["satisfied"] is True
    assert missing["requirements"][1]["satisfied"] is False
    assert missing["valid"] is False
    assert module.validate_candidate([1, 7, 8, 13], requirements, tokenizer)["valid"] is True


def test_prompt_token_alone_cannot_satisfy_requirement():
    # Searching prompt-inclusive IDs for lexical alternatives would accept this prefix.
    requirements = [{"id": "word", "target_forms": ["token7"], "token_alternatives": [[7]]}]
    result = protection_module().validate_candidate(
        [7], requirements, opus_fixtures.QueryTokenizer()
    )
    assert result["requirements"][0]["tokens_satisfied"] is False
    assert result["valid"] is False


def test_rank_four_eos_proposal_is_not_a_genuine_retained_completion(driver):
    # An EOS proposal alone is insufficient when it falls outside the selected beam ranks.
    from siglip_tools.repetition_stop import RepetitionAbort

    driver.events = {5: [(7, 0, -0.1), (8, 1, -0.2), (10, 2, -0.3), (8, 3, -0.4), (0, 0, -0.5)]}
    translator = construct(driver)
    with pytest.raises(RepetitionAbort):
        translator.translate("一台逆变器")
    assert translator.last_trace["repetition_stop"]["candidate_protection"]["eos_candidates"] == []
    assert module_validator(translator.last_trace)


def test_actual_tiny_model_retained_eos_protects_without_changing_legacy_output(
    tiny_base, tmp_path, cached_decoder, monkeypatch
):
    # This drives the unmodified pinned loop/model forward with deterministic score inputs.
    from transformers import LogitsProcessor

    from siglip_tools.opus_constraints import OPUSConstraintTranslator, prepare_decoder

    class ControlledScores(LogitsProcessor):
        def __call__(self, input_ids, scores):
            result = torch.full_like(scores, -1000.0)
            token = [7, 8, 10, 8][(input_ids.shape[-1] - 1) % 4]
            result[:, token] = -1.0
            result[:, 22] = -20.0
            result[:, 23] = -21.0
            result[:, 24] = -22.0
            if input_ids.shape[-1] == 5:
                result[0, 0] = -0.9
            return result

    original = tiny_base.model._get_logits_processor

    def processors(*args, **kwargs):
        result = original(*args, **kwargs)
        result.append(ControlledScores())
        return result

    monkeypatch.setattr(tiny_base.model, "_get_logits_processor", processors)
    directory = prepare_decoder(tmp_path / "decoder")
    legacy = OPUSConstraintTranslator(tiny_base, decoder_dir=directory)
    assert legacy.translate("一台逆变器") == "inverter Inverter"
    assert legacy.last_trace["sequence"] == [1, 7, 8, 10, 8, 0]
    translator = protection_module().CompletedProtectedTranslator(
        tiny_base, profile=PROFILE, decoder_dir=directory
    )
    assert translator.translate("一台逆变器") == legacy.last_trace["output"]
    assert translator.last_trace["sequence"] == legacy.last_trace["sequence"]
    stop = translator.last_trace["repetition_stop"]
    assert stop["protected_trigger_steps"] == list(range(32, 64, 4))
    assert stop["triggered"] is False
    assert module_validator(translator.last_trace)


def protected_cli_fixture(tmp_path, monkeypatch, driver):
    from test_fusion import Encoder as FusionEncoder

    module = protection_module()
    driver.base._generation_config.sequence_bias = None
    translator = construct(driver)
    path = tmp_path / "gallery.npz"
    index_at(tmp_path).save(path)

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
                    ["中文" if t == "一台逆变器" else t for t in texts], batch_size
                )

        result = QueryEncoder()
        result.device = torch.device("cpu")
        return result

    class FreshOriginal(OriginalTranslator):
        def __init__(self, directory, *, device):
            assert directory == Path("models/opus-mt-zh-en") and device == "cpu"
            super().__init__()

    monkeypatch.setattr(module, "SigLIPEmbedder", encoder)
    monkeypatch.setattr(module, "LocalTranslator", FreshOriginal)
    monkeypatch.setattr(module, "CompletedProtectedTranslator", lambda *a, **kw: translator)
    return (
        module,
        translator,
        [
            "search",
            "--index",
            str(path),
            "--query",
            "一台逆变器",
            "--device",
            "cpu",
            "--local-files-only",
        ],
    )


def test_module_cli_defaults_to_fresh_opus_fallback_and_reports_protected_identity(
    tmp_path, monkeypatch, driver, capsys
):
    driver.events.clear()
    module, translator, argv = protected_cli_fixture(tmp_path, monkeypatch, driver)
    assert module.main(argv) == 0
    result = json.loads(capsys.readouterr().out)
    automatic = result["automatic_translation"]
    assert automatic["opus_decoding"] == "completed_candidate_protected_constrained_beam4"
    assert automatic["constraint_fallback"] == "opus"
    assert automatic["query_texts"] == ["一台逆变器", "English"]
    assert automatic["effective_route"] == "original_fusion"
    assert result["results"][0]["path"] == "2.png"
    assert result["results"][0]["score"] == pytest.approx(0.7)
    assert module_validator(translator.last_trace)


def test_module_cli_expected_abort_with_error_policy_returns_json_exit_two(
    tmp_path, monkeypatch, driver, capsys
):
    driver.events.clear()
    module, _, argv = protected_cli_fixture(tmp_path, monkeypatch, driver)
    assert module.main(argv + ["--fallback", "error"]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["ok"] is False
    assert result["results"] == []
    assert (
        result["automatic_translation"]["constraint_trace"]["repetition_stop"]["triggered"] is True
    )


def test_replay_rejects_duplicate_witness_rank_and_disabled_constraint_pool(driver):
    # Duplicating a proposal or detaching constraints from an enabled guard forges provenance.
    translator = construct(driver)
    translator.translate("一台逆变器")
    changed = deepcopy(translator.last_trace)
    event = changed["repetition_stop"]["candidate_protection"]["eos_candidates"][0]
    event["witnesses"].append(deepcopy(event["witnesses"][0]))
    with pytest.raises(ValueError):
        module_validator(changed)
    changed = deepcopy(translator.last_trace)
    changed["constraints_applied"] = False
    with pytest.raises(ValueError):
        module_validator(changed)


def test_replay_binds_every_callback_to_terminal_scorer_done_evidence(driver):
    # Erasing coverage cannot be excused by arbitrary observed<=history counts.
    translator = construct(driver)
    translator.translate("一台逆变器")
    stop = translator.last_trace["repetition_stop"]
    evidence = stop["candidate_protection"]
    assert evidence.get("scorer_done_before_finalize", "missing") is False
    assert stop["observed_generation_steps"] == len(evidence["pool_history"]) == 63
    for done, observed in [(False, 0), (False, 62), (True, 0), (True, 63)]:
        changed = deepcopy(translator.last_trace)
        changed_stop = changed["repetition_stop"]
        changed_stop["candidate_protection"]["scorer_done_before_finalize"] = done
        changed_stop["observed_generation_steps"] = observed
        if observed == 0:
            changed_stop["snapshots"] = []
            changed_stop["protected_trigger_steps"] = []
        with pytest.raises(ValueError, match="callback|observed"):
            module_validator(changed)
    # A hand fixture may omit exactly the final callback when is_done short-circuits.
    changed = deepcopy(translator.last_trace)
    changed["repetition_stop"]["candidate_protection"]["scorer_done_before_finalize"] = True
    changed["repetition_stop"]["observed_generation_steps"] = 62
    assert module_validator(changed)


def test_actual_tiny_scorer_done_omits_only_final_callback(tiny_base, tmp_path, cached_decoder):
    # The pinned OR expression omits criteria on the real terminal scorer-done step.
    from siglip_tools.opus_constraints import prepare_decoder

    translator = protection_module().CompletedProtectedTranslator(
        tiny_base, profile=PROFILE, decoder_dir=prepare_decoder(tmp_path / "decoder")
    )
    assert translator.translate("一台逆变器")
    stop = translator.last_trace["repetition_stop"]
    evidence = stop["candidate_protection"]
    assert evidence.get("scorer_done_before_finalize", "missing") is True
    assert stop["observed_generation_steps"] + 1 == len(evidence["pool_history"])
    assert module_validator(translator.last_trace)


def test_abort_and_no_term_have_no_final_scorer_done_state(driver):
    from siglip_tools.repetition_stop import RepetitionAbort

    driver.events.clear()
    translator = construct(driver)
    with pytest.raises(RepetitionAbort):
        translator.translate("一台逆变器")
    assert (
        translator.last_trace["repetition_stop"]["candidate_protection"].get(
            "scorer_done_before_finalize", "missing"
        )
        is None
    )
    translator.translate("打开的柜门")
    assert (
        translator.last_trace["repetition_stop"]["candidate_protection"].get(
            "scorer_done_before_finalize", "missing"
        )
        is None
    )


def accepted_constraint_trace():
    return {
        "sequence": [1, 7, 8, 0],
        "output": "inverter",
        "eos_terminated": True,
        "status": "success",
        "requirements": [
            {
                **deepcopy(REQUIREMENTS[0]),
                "tokens_satisfied": True,
                "phrase_satisfied": True,
                "satisfied": True,
            }
        ],
        "satisfied": True,
    }


def final_acceptance(trace, *, constrained=True):
    module = protection_module()
    assert hasattr(module, "validate_final_acceptance"), (
        "Final acceptance replay is not implemented"
    )
    return module.validate_final_acceptance(trace, {0}, constrained=constrained)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda trace: trace.update(eos_terminated=False),
        lambda trace: trace.update(output=""),
        lambda trace: trace.update(output="microinverter"),
        lambda trace: trace.update(sequence=[1, 11, 8, 0]),
        lambda trace: trace.update(satisfied=False),
        lambda trace: trace["requirements"][0].update(tokens_satisfied=False),
        lambda trace: trace["requirements"][0].update(phrase_satisfied=False),
        lambda trace: trace["requirements"][0].update(satisfied=False),
        lambda trace: trace.update(status="failed", error="fabricated rejection"),
    ],
)
def test_final_acceptance_rejects_inconsistent_eos_text_tokens_or_status(mutate):
    # A successful final result must prove every original acceptance condition.
    trace = accepted_constraint_trace()
    assert final_acceptance(trace)
    mutate(trace)
    with pytest.raises(ValueError):
        final_acceptance(trace)


def test_final_acceptance_replays_legitimate_rejections_and_genuine_guard_abort():
    # Failed finalization can have valid terms but no EOS; active abort can have valid terms.
    no_eos = accepted_constraint_trace()
    no_eos.update(sequence=[1, 7, 8], eos_terminated=False, status="failed", error="missing EOS")
    assert final_acceptance(no_eos)
    lexical = accepted_constraint_trace()
    lexical.update(
        output="microinverter", status="failed", error="missing whole word", satisfied=False
    )
    lexical["requirements"][0].update(phrase_satisfied=False, satisfied=False)
    assert final_acceptance(lexical)
    aborted = deepcopy(no_eos)
    aborted["repetition_stop"] = {"triggered": True, "observe_only": False}
    assert final_acceptance(aborted)
    aborted["status"] = "success"
    with pytest.raises(ValueError):
        final_acceptance(aborted)


def test_final_acceptance_preserves_original_raw_text_trimming_and_pre_generation_failure():
    # The original translator strips successful raw output, retaining failed raw text verbatim.
    original = {
        "sequence": [1, 22, 0],
        "raw_decoded_output": " English ",
        "output": "English",
        "eos_terminated": True,
        "status": "success",
    }
    assert final_acceptance(original, constrained=False)
    original["output"] = " English "
    with pytest.raises(ValueError):
        final_acceptance(original, constrained=False)
    failed = {
        "sequence": None,
        "output": None,
        "raw_decoded_output": None,
        "eos_terminated": False,
        "status": "failed",
        "error": "input too long",
    }
    assert final_acceptance(failed, constrained=False)
    failed["status"] = "success"
    with pytest.raises(ValueError):
        final_acceptance(failed, constrained=False)


def test_protection_replay_rejects_self_declared_success_without_final_acceptance(driver):
    # Recomputed guard decisions cannot excuse an invalid returned final translation.
    translator = construct(driver)
    translator.translate("一台逆变器")
    changed = deepcopy(translator.last_trace)
    changed.update(sequence=[1, 22], output="", eos_terminated=False, satisfied=False)
    changed["repetition_stop"]["generated_new_tokens"] = 1
    changed["requirements"][0].update(
        tokens_satisfied=False, phrase_satisfied=False, satisfied=False
    )
    with pytest.raises(ValueError):
        module_validator(changed)


def test_callback_coverage_cannot_be_erased_while_history_remains(driver):
    # This directly reproduces the reviewed bug without relying on the new metadata field.
    translator = construct(driver)
    translator.translate("一台逆变器")
    changed = deepcopy(translator.last_trace)
    changed["repetition_stop"].update(
        observed_generation_steps=0, snapshots=[], protected_trigger_steps=[]
    )
    with pytest.raises(ValueError):
        module_validator(changed)
