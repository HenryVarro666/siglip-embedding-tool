"""Protect retained genuine EOS completions while preserving the pinned search.

Only a selected EOS insertion from scorer.process is a completion witness.
Finalization, active prefixes and evicted hypotheses never protect a raw
repetition trigger. Request-local FunctionType copies replace the scorer only
inside copied globals; the original decoder, search and finalizer stay intact.

Trace schema v1 adds ``candidate_protection`` to ``repetition_stop``. Its
``eos_candidates`` records immutable prefix/score/proposal witnesses, while
``pool_history`` records current IDs after every process call. The final scorer
state binds whether its terminal callback was short-circuited. Eligible snapshots
record current IDs and, only on a raw trigger, lexical validation of every
retained candidate. Snapshot ``triggered`` is raw detection; top-level
``triggered`` and snapshot ``abort`` describe an actual rejection. The pure
validator checks recorded consistency, not independent raw tensor provenance.
"""

import argparse
import json
import math
import re
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from types import FunctionType

from transformers import StoppingCriteria, StoppingCriteriaList

from .constraint_fallback import timed_fallback_search
from .encoder import SigLIPEmbedder, identity_fingerprint
from .index import ImageIndex
from .opus_constraints import DEFAULT_DECODER_DIR, _has_tokens
from .repetition_stop import (
    DEFAULT_PROFILE_PATH,
    GuardedOPUSConstraintTranslator,
    RepetitionAbort,
    _positive,
    detect_repetition,
    profile_sha256,
    validate_profile,
)
from .translation import _CHINESE, LocalTranslator

PROTECTION_PATH = Path(__file__).with_name("completed_protection_v1.json")
_RULE = json.loads(PROTECTION_PATH.read_text(encoding="utf-8"))
_RULE_SHA256 = identity_fingerprint(_RULE)
_SOURCE = "request_local_scorer_process_selected_eos"
_DECODING = "completed_candidate_protected_constrained_beam4"
_CHECK_KEYS = {"tokens_satisfied", "phrase_satisfied", "satisfied"}


def _selected_profile(profile):
    result = validate_profile(profile)
    if result != _RULE["repetition_profile"]:
        raise ValueError("Protection v1 requires the unchanged selected repetition profile")
    return result


def _lexical_validation(prefix, requirements, output):
    if (
        not isinstance(prefix, list)
        or not prefix
        or any(type(token) is not int or token < 0 for token in prefix)
        or not isinstance(requirements, list)
        or not isinstance(output, str)
    ):
        raise ValueError("Candidate validation requires a prompt-inclusive prefix and raw text")
    checked = []
    for requirement in requirements:
        original = {
            key: deepcopy(value) for key, value in requirement.items() if key not in _CHECK_KEYS
        }
        alternatives, forms = original["token_alternatives"], original["target_forms"]
        if (
            not isinstance(alternatives, list)
            or not alternatives
            or any(
                not isinstance(tokens, list)
                or not tokens
                or any(type(token) is not int or token < 0 for token in tokens)
                for tokens in alternatives
            )
            or not isinstance(forms, list)
            or not forms
            or any(not isinstance(form, str) or not form for form in forms)
        ):
            raise ValueError(
                "Candidate requirements must retain full token alternatives and target forms"
            )
        token_ok = any(_has_tokens(prefix[1:], tokens) for tokens in alternatives)
        phrase_ok = any(
            re.search(r"(?<!\w)" + re.escape(form) + r"(?!\w)", output) for form in forms
        )
        checked.append(
            {
                **original,
                "tokens_satisfied": token_ok,
                "phrase_satisfied": phrase_ok,
                "satisfied": bool(token_ok and phrase_ok),
            }
        )
    satisfied = all(requirement["satisfied"] for requirement in checked)
    return {
        "output": output,
        "nonempty": bool(output.strip()),
        "requirements": checked,
        "satisfied": satisfied,
        "valid": bool(output.strip()) and satisfied,
    }


def validate_candidate(prefix: list[int], requirements: list[dict], tokenizer) -> dict:
    """Check all original lexical alternatives on a prefix excluding actual EOS.

    ``prefix`` contains the single decoder prompt, which is excluded from token
    matching. Decode verbatim with special tokens skipped; return raw ``output``,
    ``nonempty``, full checked ``requirements``, lexical ``satisfied`` and ``valid``.
    No pretrained component is loaded by this function.
    """
    return _lexical_validation(
        prefix, requirements, tokenizer.decode(prefix, skip_special_tokens=True)
    )


def _protection_trace(eos_ids, settings):
    return {
        "schema_version": 1,
        "rule": deepcopy(_RULE),
        "rule_sha256": _RULE_SHA256,
        "source": _SOURCE,
        "inspection": "raw_trigger_only",
        "decoder_prompt_len": 1,
        "batch_size": 1,
        "num_beams": 4,
        "eos_token_ids": sorted(eos_ids),
        "decoder_start_token_id": settings.decoder_start_token_id,
        "length_penalty": settings.length_penalty,
        "finalize_entered": False,
        "scorer_done_before_finalize": None,
        "eos_candidates": [],
        "pool_history": [],
    }


def _contract(condition, detail):
    if not condition:
        raise RuntimeError(f"Completed-candidate observer contract: {detail}")


def _cpu_ids(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    return [value] if type(value) is int else list(value)


def _ancestry(value):
    return None if value is None else [int(item) for item in value]


def _clone_function(function, namespace):
    _contract(isinstance(function, FunctionType), "validated callable is not a Python function")
    result = FunctionType(
        function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__
    )
    result.__kwdefaults__ = function.__kwdefaults__
    result.__annotations__ = function.__annotations__.copy()
    result.__dict__.update(function.__dict__)
    return result


class _CandidateObserver:
    def __init__(self, trace):
        self.trace = trace
        self.scorer = None
        self.phase = "process"
        self.step = 0
        # Holding the tuples also holds their tensors and prevents object-ID reuse.
        self.references = {}
        self.records = {}
        self.pool_ids = []

    def bind(self, scorer):
        _contract(self.scorer is None, "multiple scorer instances in one request")
        _contract(
            len(scorer._beam_hyps) == 1
            and scorer.num_beams == scorer.group_size == 4
            and scorer.num_beam_groups == 1,
            "requires batch1, beam4 and one beam group",
        )
        _contract(math.isfinite(scorer.length_penalty), "nonfinite length penalty")
        self.trace["length_penalty"] = float(scorer.length_penalty)
        self.scorer = scorer

    def after_process(
        self,
        scorer,
        input_ids,
        next_scores,
        next_tokens,
        next_indices,
        eos_token_id,
        beam_indices,
        decoder_prompt_len,
    ):
        _contract(self.scorer is scorer and self.phase == "process", "process after finalize")
        _contract(
            decoder_prompt_len == 1 and input_ids.ndim == 2 and input_ids.shape[0] == 4,
            "requires one decoder prompt token and four input beams",
        )
        step = input_ids.shape[-1]
        _contract(step == self.step + 1 and step <= 64, "nonconsecutive process steps")
        self.step = step
        pool = scorer._beam_hyps[0].beams
        _contract(len(pool) <= 4 and all(len(hyp) == 3 for hyp in pool), "unexpected beam pool")
        new = [hyp for hyp in pool if id(hyp) not in self.records]
        if new:
            _contract(
                next_scores.ndim == next_tokens.ndim == next_indices.ndim == 2
                and next_scores.shape == next_tokens.shape == next_indices.shape
                and next_tokens.shape[0] == 1
                and next_tokens.shape[1] >= 4,
                "unexpected EOS proposal tensor shape",
            )
            eos_ids = _cpu_ids(eos_token_id)
            _contract(sorted(eos_ids) == self.trace["eos_token_ids"], "EOS identity mismatch")
            # Copy only selected proposal slots and their needed parent prefixes,
            # and only when a new retained hypothesis actually appeared.
            tokens = next_tokens[0, :4].detach().cpu().tolist()
            scores = next_scores[0, :4].detach().cpu().tolist()
            parents = next_indices[0, :4].detach().cpu().tolist()
            proposals = []
            for rank, (token, score, parent) in enumerate(zip(tokens, scores, parents)):
                if token not in eos_ids:
                    continue
                _contract(0 <= parent < 4, "EOS parent outside beam4")
                prefix = input_ids[parent].detach().cpu().tolist()
                ancestry = (
                    None if beam_indices is None else _ancestry(beam_indices[parent]) + [parent]
                )
                proposals.append(
                    {
                        "phase": "process",
                        "process_new_tokens": step,
                        "rank": rank,
                        "parent_beam_index": parent,
                        "parent_prefix": prefix,
                        "eos_token_id": token,
                        "next_score": score,
                        "generated_len": step,
                        "beam_indices": ancestry,
                    }
                )
            for hypothesis in new:
                score, tensor, ancestry = hypothesis
                prefix = tensor.detach().cpu().tolist()
                ancestry = _ancestry(ancestry)
                _contract(
                    len(prefix) == step
                    and prefix[0] == self.trace["decoder_start_token_id"]
                    and not set(prefix[1:]).intersection(eos_ids)
                    and math.isfinite(score),
                    "unexpected retained prefix or normalized score",
                )
                witnesses = [
                    proposal
                    for proposal in proposals
                    if proposal["parent_prefix"] == prefix
                    and proposal["beam_indices"] == ancestry
                    and math.isclose(
                        score,
                        proposal["next_score"] / (step ** self.trace["length_penalty"]),
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                ]
                _contract(
                    bool(witnesses), "retained hypothesis has no selected genuine EOS witness"
                )
                candidate_id = len(self.trace["eos_candidates"]) + 1
                record = {
                    "candidate_id": candidate_id,
                    "prefix": prefix,
                    "eos_step": step,
                    "normalized_score": score,
                    "beam_indices": ancestry,
                    "eos_token_ids": sorted({witness["eos_token_id"] for witness in witnesses}),
                    "witnesses": deepcopy(witnesses),
                }
                self.trace["eos_candidates"].append(record)
                self.records[id(hypothesis)] = record
                self.references[id(hypothesis)] = hypothesis
        self.pool_ids = [self.records[id(hypothesis)]["candidate_id"] for hypothesis in pool]
        self.trace["pool_history"].append(
            {"new_tokens": step, "phase": "process", "pool_candidate_ids": list(self.pool_ids)}
        )

    def current_pool(self, step):
        _contract(
            self.phase == "process" and self.step == step, "callback is not after current process"
        )
        _contract(self.scorer is not None, "missing live scorer")
        actual = [self.records[id(hyp)]["candidate_id"] for hyp in self.scorer._beam_hyps[0].beams]
        _contract(actual == self.pool_ids, "pool changed between process and callback")
        return list(actual)

    def clear(self):
        self.scorer = None
        self.references.clear()
        self.records.clear()
        self.pool_ids.clear()


def _observed_decoder(validated, observer):
    _contract(isinstance(validated, FunctionType), "missing validated outer function")
    loop = validated.__globals__.get("_constrained_beam_search")
    _contract(isinstance(loop, FunctionType), "missing constrained search callable")
    scorer_class = loop.__globals__.get("ConstrainedBeamSearchScorer")
    _contract(isinstance(scorer_class, type), "missing constrained beam scorer class")

    class RequestScorer(scorer_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            observer.bind(self)

        def process(
            self,
            input_ids,
            next_scores,
            next_tokens,
            next_indices,
            scores_for_all_vocab,
            pad_token_id=None,
            eos_token_id=None,
            beam_indices=None,
            decoder_prompt_len=0,
        ):
            output = super().process(
                input_ids,
                next_scores,
                next_tokens,
                next_indices,
                scores_for_all_vocab,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                beam_indices=beam_indices,
                decoder_prompt_len=decoder_prompt_len,
            )
            observer.after_process(
                self,
                input_ids,
                next_scores,
                next_tokens,
                next_indices,
                eos_token_id,
                beam_indices,
                decoder_prompt_len,
            )
            return output

        def finalize(self, *args, **kwargs):
            observer.trace["scorer_done_before_finalize"] = bool(self.is_done)
            observer.phase = "finalize"
            observer.trace["finalize_entered"] = True
            return super().finalize(*args, **kwargs)

    loop_globals = dict(loop.__globals__)
    loop_globals["ConstrainedBeamSearchScorer"] = RequestScorer
    copied_loop = _clone_function(loop, loop_globals)
    outer_globals = dict(validated.__globals__)
    outer_globals["_constrained_beam_search"] = copied_loop
    return _clone_function(validated, outer_globals)


class _ProtectedMonitor(StoppingCriteria):
    def __init__(self, monitor, observer, requirements, tokenizer):
        self.monitor, self.observer = monitor, observer
        self.requirements, self.tokenizer = requirements, tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        trace, profile = self.monitor.trace, self.monitor.profile
        step = input_ids.shape[-1] - 1
        trace["observed_generation_steps"] = step
        if not (profile["min_new_tokens"] <= step < 64 and step % profile["check_every"] == 0):
            return False
        beams = input_ids.detach().cpu().tolist()
        snapshot = {"beams": beams, **detect_repetition(beams, profile, trace["special_ids"])}
        pool_ids = self.observer.current_pool(step)
        completed = []
        if snapshot["triggered"]:
            for rank, candidate_id in enumerate(pool_ids):
                event = self.observer.trace["eos_candidates"][candidate_id - 1]
                completed.append(
                    {
                        "candidate_id": candidate_id,
                        "pool_rank": rank,
                        "current_step": step,
                        "retained": True,
                        "phase": "process",
                        "validation": validate_candidate(
                            event["prefix"], self.requirements, self.tokenizer
                        ),
                    }
                )
        protected = snapshot["triggered"] and any(c["validation"]["valid"] for c in completed)
        abort = snapshot["triggered"] and not protected
        snapshot.update(
            pool_candidate_ids=pool_ids,
            completed_candidates=completed,
            candidate_pool="inspected" if snapshot["triggered"] else "not_requested_no_raw_trigger",
            protected=protected,
            abort=abort,
        )
        trace["snapshots"].append(snapshot)
        if protected:
            trace["protected_trigger_steps"].append(step)
        if abort:
            trace.update(
                triggered=True,
                trigger_new_tokens=step,
                observed_steps_complete=True,
                length_scope="partial_beam_sequence",
            )
            raise RepetitionAbort(snapshot)
        return False


class CompletedProtectedTranslator(GuardedOPUSConstraintTranslator):
    """Use the selected old guard, suppressing rejection for valid live completions."""

    def __init__(self, base, *, profile: dict, decoder_dir=DEFAULT_DECODER_DIR):
        super().__init__(base, profile=_selected_profile(profile), decoder_dir=decoder_dir)
        self._active_observer = None
        self._identity["opus_decoding"] = _DECODING
        self._identity["candidate_protection"] = {
            "rule": deepcopy(_RULE),
            "rule_sha256": _RULE_SHA256,
        }

    def _guarded_generate(self, model, **kwargs):
        monitor = self._active_monitor
        _contract(monitor is not None, "requires fresh request monitor")
        monitor.trace["enabled"] = True
        monitor.trace["protected_trigger_steps"] = []
        evidence = _protection_trace(self._eos_ids, kwargs["generation_config"])
        monitor.trace["candidate_protection"] = evidence
        observer = _CandidateObserver(evidence)
        self._active_observer = observer
        try:
            decoder = _observed_decoder(self._validated_decoder_generate, observer)
            criteria = _ProtectedMonitor(
                monitor, observer, self.last_trace["requirements"], self.tokenizer
            )
            return decoder(model, **kwargs, stopping_criteria=StoppingCriteriaList([criteria]))
        finally:
            observer.clear()
            self._active_observer = None

    def translate(self, text: str) -> str:
        try:
            return super().translate(text)
        finally:
            stop = self.last_trace.get("repetition_stop")
            if stop is not None:
                stop.setdefault("protected_trigger_steps", [])
                stop.setdefault(
                    "candidate_protection",
                    _protection_trace(self._eos_ids, self._generation_config),
                )


def _check(condition, detail):
    if not condition:
        raise ValueError(f"Invalid completed-candidate protection trace: {detail}")


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_final_acceptance(trace: dict, eos_ids, *, constrained: bool = True) -> bool:
    """Validate recorded final acceptance or rejection without decoding tokens.

    Return True for a consistent success *or failure* record, and raise ValueError
    for inconsistent evidence. ``constrained=True`` uses the constraint wrapper's
    full requirements, lexical flags and unedited output (also for its no-term
    path). False uses LocalTranslator's raw_decoded_output and success-only trim.
    A real guard abort remains a rejection even when its diagnostic terms pass.
    Stored text origin remains outside this recorded-data check.
    """
    try:
        _check(type(constrained) is bool, "final acceptance mode")
        eos_ids = set(eos_ids)
        _check(
            bool(eos_ids) and all(type(token) is int and token >= 0 for token in eos_ids),
            "final EOS identity",
        )
        sequence, output, status = trace.get("sequence"), trace["output"], trace["status"]
        if sequence is None:
            _check(
                output is None and trace["eos_terminated"] is False and status == "failed",
                "pre-generation failure acceptance",
            )
            if constrained:
                _check(
                    trace["satisfied"] is None
                    and all(
                        not _CHECK_KEYS.intersection(requirement)
                        for requirement in trace["requirements"]
                    ),
                    "pre-generation lexical checks",
                )
            else:
                _check(trace["raw_decoded_output"] is None, "pre-generation raw text")
        else:
            _check(
                isinstance(sequence, list)
                and bool(sequence)
                and all(type(token) is int and token >= 0 for token in sequence)
                and isinstance(output, str),
                "final sequence and output",
            )
            eos_terminated = any(token in eos_ids for token in sequence[1:])
            _check(trace["eos_terminated"] is eos_terminated, "final EOS termination")
            if constrained:
                replay = _lexical_validation(sequence, trace["requirements"], output)
                for requirement, checked in zip(trace["requirements"], replay["requirements"]):
                    _check(
                        all(requirement[key] is checked[key] for key in _CHECK_KEYS),
                        "final original lexical flags",
                    )
                _check(trace["satisfied"] is replay["satisfied"], "final lexical satisfaction")
                nonempty, lexical = replay["nonempty"], replay["satisfied"]
            else:
                raw = trace["raw_decoded_output"]
                _check(isinstance(raw, str), "final raw decoded output")
                _check(
                    output == (raw.strip() if status == "success" else raw),
                    "original output trimming",
                )
                nonempty, lexical = bool(raw.strip()), True
            stop = trace.get("repetition_stop", {})
            aborted = stop.get("triggered") is True and stop.get("observe_only") is False
            accepted = eos_terminated and nonempty and lexical and not aborted
            _check(status == ("success" if accepted else "failed"), "final acceptance status")
        if status == "failed":
            _check(
                isinstance(trace.get("error"), str) and bool(trace["error"].strip()),
                "rejection error evidence",
            )
        return True
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise ValueError("Invalid completed-candidate final acceptance trace structure") from exc


def _replay_trace(trace):
    stop = trace["repetition_stop"]
    profile = _selected_profile(stop["profile"])
    _check(stop["profile_sha256"] == profile_sha256(profile), "profile digest")
    _check(stop["observe_only"] is False, "observe-only is not protected mode")
    _check(
        type(stop["enabled"]) is bool
        and stop["enabled"] is trace["constraints_applied"]
        and stop["enabled"] is bool(trace["requirements"]),
        "guard enablement and original constraints",
    )
    evidence = stop["candidate_protection"]
    _check(
        evidence["schema_version"] == 1
        and evidence["rule"] == _RULE
        and evidence["rule_sha256"] == _RULE_SHA256
        and evidence["source"] == _SOURCE
        and evidence["inspection"] == "raw_trigger_only"
        and evidence["decoder_prompt_len"] == evidence["batch_size"] == 1
        and evidence["num_beams"] == 4,
        "rule or observer identity",
    )
    eos_ids = evidence["eos_token_ids"]
    _check(
        isinstance(eos_ids, list)
        and eos_ids
        and eos_ids == sorted(set(eos_ids))
        and all(type(token) is int and token >= 0 for token in eos_ids),
        "EOS IDs",
    )
    events = evidence["eos_candidates"]
    _check(isinstance(events, list), "EOS candidates")
    _check(type(evidence["finalize_entered"]) is bool, "finalize phase")
    if stop["enabled"]:
        _check(_finite(evidence["length_penalty"]), "length penalty")
    by_step = {}
    for candidate_id, event in enumerate(events, 1):
        prefix, step, score = event["prefix"], event["eos_step"], event["normalized_score"]
        _check(
            event["candidate_id"] == candidate_id
            and isinstance(prefix, list)
            and bool(prefix)
            and all(type(token) is int and token >= 0 for token in prefix)
            and prefix[0] == evidence["decoder_start_token_id"]
            and not set(prefix[1:]).intersection(eos_ids)
            and type(step) is int
            and 1 <= step <= 64
            and len(prefix) == step
            and _finite(score),
            "candidate prefix, step or score",
        )
        witnesses = event["witnesses"]
        _check(isinstance(witnesses, list) and bool(witnesses), "missing EOS witnesses")
        ranks = [witness["rank"] for witness in witnesses]
        _check(ranks == sorted(set(ranks)), "duplicate or unordered EOS proposal ranks")
        for witness in witnesses:
            _check(
                witness["phase"] == "process"
                and witness["process_new_tokens"] == step
                and type(witness["rank"]) is int
                and 0 <= witness["rank"] < 4
                and type(witness["parent_beam_index"]) is int
                and 0 <= witness["parent_beam_index"] < 4
                and witness["parent_prefix"] == prefix
                and witness["eos_token_id"] in eos_ids
                and witness["generated_len"] == step
                and witness["beam_indices"] == event["beam_indices"]
                and _finite(witness["next_score"]),
                "selected process EOS witness",
            )
            _check(
                math.isclose(
                    score,
                    witness["next_score"] / (step ** evidence["length_penalty"]),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ),
                "normalized score",
            )
            ancestry = witness["beam_indices"]
            if ancestry is not None:
                _check(
                    isinstance(ancestry, list)
                    and len(ancestry) == step
                    and all(type(parent) is int and 0 <= parent < 4 for parent in ancestry)
                    and ancestry[-1] == witness["parent_beam_index"],
                    "EOS beam ancestry",
                )
        _check(
            event["eos_token_ids"] == sorted({w["eos_token_id"] for w in witnesses}),
            "actual EOS identity",
        )
        by_step.setdefault(step, []).append(candidate_id)
    history = evidence["pool_history"]
    _check(isinstance(history, list) and len(history) <= 64, "pool history")
    current = []
    for step, record in enumerate(history, 1):
        _check(
            record["new_tokens"] == step and record["phase"] == "process", "process history step"
        )
        for candidate_id in by_step.get(step, []):
            score = events[candidate_id - 1]["normalized_score"]
            _check(
                len(current) < 4 or score > min(events[c - 1]["normalized_score"] for c in current),
                "unretained candidate insertion",
            )
            current.append(candidate_id)
            if len(current) > 4:
                worst = min(
                    range(len(current)),
                    key=lambda i: (events[current[i] - 1]["normalized_score"], i),
                )
                current.pop(worst)
        _check(record["pool_candidate_ids"] == current, "current retained pool membership")
    _check(not by_step or max(by_step) <= len(history), "candidate outside process history")
    observed = stop["observed_generation_steps"]
    _check(type(observed) is int and 0 <= observed <= len(history), "observed step count")
    done = evidence["scorer_done_before_finalize"]
    if stop["enabled"] and not stop["triggered"]:
        _check(
            type(done) is bool and observed == len(history) - (1 if done else 0),
            "observed callback coverage versus terminal scorer done state",
        )
    else:
        _check(done is None, "scorer done state outside normal finalization")
    snapshots = stop["snapshots"]
    expected_steps = list(
        range(profile["min_new_tokens"], min(observed + 1, 64), profile["check_every"])
    )
    _check(
        [snapshot["new_tokens"] for snapshot in snapshots] == expected_steps,
        "eligible snapshot coverage",
    )
    protected_steps, aborted = [], []
    for snapshot in snapshots:
        step = snapshot["new_tokens"]
        raw = detect_repetition(snapshot["beams"], profile, stop["special_ids"])
        _check(all(snapshot[key] == value for key, value in raw.items()), "raw repetition replay")
        pool_ids = history[step - 1]["pool_candidate_ids"]
        _check(snapshot["pool_candidate_ids"] == pool_ids, "snapshot retained pool")
        candidates = snapshot["completed_candidates"]
        want_ids = pool_ids if raw["triggered"] else []
        _check([c["candidate_id"] for c in candidates] == want_ids, "inspected candidate IDs")
        valid = []
        for rank, candidate in enumerate(candidates):
            _check(
                candidate["pool_rank"] == rank
                and candidate["current_step"] == step
                and candidate["retained"] is True
                and candidate["phase"] == "process",
                "candidate membership step or phase",
            )
            event = events[candidate["candidate_id"] - 1]
            _check(event["eos_step"] <= step, "future candidate")
            validation = candidate["validation"]
            replay = _lexical_validation(
                event["prefix"], trace["requirements"], validation["output"]
            )
            _check(validation == replay, "original lexical and nonempty validation")
            valid.append(validation["valid"])
        protected = bool(raw["triggered"] and any(valid))
        abort = bool(raw["triggered"] and not protected)
        _check(
            snapshot["protected"] is protected
            and snapshot["abort"] is abort
            and snapshot["candidate_pool"]
            == ("inspected" if raw["triggered"] else "not_requested_no_raw_trigger"),
            "protection or actual abort decision",
        )
        if protected:
            protected_steps.append(step)
        if abort:
            aborted.append(step)
    _check(stop["protected_trigger_steps"] == protected_steps, "protected steps")
    _check(
        len(aborted) <= 1
        and stop["triggered"] is bool(aborted)
        and stop["trigger_new_tokens"] == (aborted[0] if aborted else None),
        "actual abort status",
    )
    if aborted:
        _check(
            aborted[0] == observed == len(history)
            and snapshots[-1]["abort"] is True
            and stop["observed_steps_complete"] is True
            and stop["length_scope"] == "partial_beam_sequence"
            and evidence["finalize_entered"] is False
            and trace["status"] == "failed"
            and trace["eos_terminated"] is False
            and trace["sequence"] == snapshots[-1]["beams"][0],
            "abort phase and diagnostic output",
        )
    else:
        _check(
            not stop["enabled"] or evidence["finalize_entered"] is True,
            "nonabort finalization phase",
        )
        _check(
            stop["observed_steps_complete"] is False
            and stop["length_scope"] == "finalized_sequence",
            "nonabort generation scope",
        )
    if not stop["enabled"]:
        _check(
            not history and not events and not snapshots and not evidence["finalize_entered"],
            "disabled guard evidence",
        )
    sequence = trace.get("sequence")
    _check(
        stop["generated_new_tokens"] == (len(sequence) - 1 if sequence is not None else 0),
        "diagnostic/final sequence length",
    )
    validate_final_acceptance(trace, eos_ids, constrained=True)
    return True


def validate_protection_trace(trace: dict) -> bool:
    """Replay all recorded protection decisions without loading a model/tokenizer.

    Returns True, or raises ValueError on inconsistent EOS, score, process/current
    pool, original lexical, repetition or abort evidence. Recorded raw English
    is used for regex checks; this audit cannot independently prove that stored
    text/tensors originated in the frozen decoder or tokenizer.
    """
    try:
        return _replay_trace(trace)
    except (KeyError, IndexError, TypeError, ArithmeticError, AttributeError) as exc:
        raise ValueError("Invalid completed-candidate protection trace structure") from exc


def _search(args):
    if not isinstance(args.query, str) or not args.query.strip() or not _CHINESE.search(args.query):
        raise ValueError("Translation requires a nonempty query containing Chinese text")
    profile = _selected_profile(json.loads(args.profile.read_text(encoding="utf-8")))
    started = perf_counter()
    index = ImageIndex.load(args.index)
    identity = index.model_identity
    encoder = SigLIPEmbedder(
        identity["model_id"], identity["revision"], args.device, args.local_files_only
    )
    index.assert_compatible(encoder.identity)
    original = LocalTranslator(args.translation_dir, device=str(encoder.device))
    guarded = CompletedProtectedTranslator(original, profile=profile, decoder_dir=args.decoder_dir)
    loaded = perf_counter()
    measured = timed_fallback_search(
        index, encoder, guarded, original, args.query, args.mode, args.fallback, args.top_k
    )
    return {
        "ok": measured["ok"],
        "query_type": "search",
        "index": str(args.index.resolve()),
        "fingerprint": index.fingerprint,
        "device": str(encoder.device),
        "automatic_translation": {
            "english": measured["translation"],
            "mode": args.mode,
            "translator": "opus",
            "opus_decoding": _DECODING,
            "constraint_fallback": args.fallback,
            "translator_fingerprint": identity_fingerprint(guarded.identity),
            "translator_identity": guarded.identity,
            **{
                key: measured[key]
                for key in (
                    "ok",
                    "failure",
                    "primary_failure",
                    "constraint_trace",
                    "fallback_trace",
                    "fallback_triggered",
                    "effective_route",
                    "query_texts",
                )
            },
        },
        "query_timing_ms": measured["timing"],
        "results": [
            {**asdict(match), "full_path": str(index.image_root / match.path)}
            for match in measured["matches"]
        ],
        "seconds": {
            "load": loaded - started,
            "query": measured["timing"]["total_ms"] / 1000,
            "total": perf_counter() - started,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    search = commands.add_parser("search", help="Completed-protected OPUS with explicit fallback")
    search.add_argument("--index", required=True, type=Path)
    search.add_argument("--query", required=True)
    search.add_argument("--profile", type=Path, default=DEFAULT_PROFILE_PATH)
    search.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    search.add_argument("--top-k", type=_positive, default=5)
    search.add_argument("--decoder-dir", type=Path, default=DEFAULT_DECODER_DIR)
    search.add_argument("--translation-dir", type=Path, default=Path("models/opus-mt-zh-en"))
    search.add_argument("--local-files-only", action="store_true")
    search.add_argument("--mode", choices=("auto_en", "auto_fusion"), default="auto_fusion")
    search.add_argument("--fallback", choices=("error", "opus", "chinese"), default="opus")
    args = parser.parse_args(argv)
    try:
        payload = _search(args)
    except (ValueError, OSError, RuntimeError) as exc:
        payload = {
            "ok": False,
            "failure": {"stage": "search", "type": type(exc).__name__, "message": str(exc)},
            "results": [],
        }
    print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
