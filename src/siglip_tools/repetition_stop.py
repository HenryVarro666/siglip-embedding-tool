"""Reject repeating constrained OPUS generations before finalization, without training.

The guard is an explicit, separate entry point. Its callback raises a dedicated
exception: returning a successful stop flag could let the beam finalizer append
EOS to an incomplete fragment. Instances, like the original wrapper, are for
serial requests; every request receives its own monitor and immutable evidence.
"""

import argparse
import json
import re
from copy import deepcopy
from dataclasses import asdict
from itertools import product
from pathlib import Path
from time import perf_counter

from transformers import StoppingCriteria, StoppingCriteriaList

from .constraint_fallback import timed_fallback_search
from .encoder import SigLIPEmbedder, identity_fingerprint
from .index import ImageIndex
from .opus_constraints import DEFAULT_DECODER_DIR, OPUSConstraintTranslator, _has_tokens
from .translation import _CHINESE, LocalTranslator

DEFAULT_PROFILE_PATH = Path(__file__).with_name("repetition_profile_v1.json")
_ALGORITHM = "suffix_consecutive_token_cycles"


def candidate_profiles() -> list[dict]:
    """Return fresh profiles for the complete predeclared sixteen-candidate grid."""
    return [
        {
            "schema_version": 1,
            "profile_id": f"suffix-n{size}-r{count}-min{minimum}-q{quorum}",
            "algorithm": _ALGORITHM,
            "max_new_tokens": 64,
            "num_beams": 4,
            "check_every": 4,
            "max_ngram_size": size,
            "repeat_count": count,
            "min_new_tokens": minimum,
            "min_repeating_beams": quorum,
        }
        for size, count, minimum, quorum in product((8, 12), (3, 4), (24, 32), (3, 4))
    ]


def validate_profile(profile: dict) -> dict:
    """Validate the frozen algorithm/grid and return an independent profile."""
    if not isinstance(profile, dict):
        raise ValueError("Repetition profile must be an object")
    expected_keys = set(candidate_profiles()[0])
    if set(profile) != expected_keys:
        raise ValueError("Repetition profile has missing or unexpected fields")
    numeric = expected_keys - {"profile_id", "algorithm"}
    if any(type(profile[key]) is not int for key in numeric):
        raise ValueError("Repetition profile numeric fields must be integers")
    if profile not in candidate_profiles():
        raise ValueError("Repetition profile differs from the predeclared candidate grid")
    return deepcopy(profile)


def profile_sha256(profile: dict) -> str:
    """Return the repository's canonical semantic digest, distinct from file SHA."""
    return identity_fingerprint(validate_profile(profile))


def detect_repetition(beams, profile: dict, special_ids) -> dict:
    """Replay one prompt-inclusive four-beam snapshot using a frozen profile.

    A match uses the smallest qualifying suffix period. ``repeat_count`` is the
    actual consecutive count, and ``suffix_start`` indexes the raw sequence,
    including its single prompt token. A nonmatch has period/start ``None``, an
    empty cycle and count zero. Repetition evidence is reported at every supplied
    step; eligibility and trigger respect the predeclared early-check interval.
    """
    profile = validate_profile(profile)
    if (
        not isinstance(beams, (list, tuple))
        or len(beams) != profile["num_beams"]
        or any(not isinstance(beam, (list, tuple)) or not beam for beam in beams)
        or len({len(beam) for beam in beams}) != 1
        or any(type(token) is not int or token < 0 for beam in beams for token in beam)
    ):
        raise ValueError("Repetition detection requires four equal, nonempty token sequences")
    special_ids = set(special_ids)
    new_tokens = len(beams[0]) - 1
    eligible = (
        profile["min_new_tokens"] <= new_tokens < profile["max_new_tokens"]
        and new_tokens % profile["check_every"] == 0
    )
    evidence = []
    for index, beam in enumerate(beams):
        tokens = list(beam[1:])
        match = {
            "beam_index": index,
            "period": None,
            "cycle": [],
            "repeat_count": 0,
            "suffix_start": None,
        }
        maximum = min(profile["max_ngram_size"], len(tokens) // profile["repeat_count"])
        for period in range(1, maximum + 1):
            cycle = tokens[-period:]
            if special_ids.intersection(cycle):
                continue
            count = 1
            while (count + 1) * period <= len(tokens):
                start = len(tokens) - (count + 1) * period
                if tokens[start : start + period] != cycle:
                    break
                count += 1
            if count >= profile["repeat_count"]:
                match.update(
                    period=period,
                    cycle=cycle,
                    repeat_count=count,
                    suffix_start=len(beam) - count * period,
                )
                break
        evidence.append(match)
    repeating = sum(match["period"] is not None for match in evidence)
    return {
        "new_tokens": new_tokens,
        "eligible": eligible,
        "repeating_beams": repeating,
        "triggered": eligible and repeating >= profile["min_repeating_beams"],
        "beam_evidence": evidence,
    }


class RepetitionAbort(ValueError):
    """An actual in-loop repetition rejection, never a successful generation stop."""

    def __init__(self, snapshot: dict):
        self.snapshot = deepcopy(snapshot)
        super().__init__(
            f"Constrained generation rejected for repetition at {snapshot['new_tokens']} "
            f"new tokens ({snapshot['repeating_beams']}/4 repeating beams)"
        )


class _RepetitionMonitor(StoppingCriteria):
    def __init__(self, profile: dict, observe_only: bool, special_ids):
        self.profile = profile
        self.trace = {
            "profile": deepcopy(profile),
            "profile_sha256": profile_sha256(profile),
            "special_ids": sorted(special_ids),
            "observe_only": observe_only,
            "enabled": False,
            "triggered": False,
            "trigger_new_tokens": None,
            "snapshots": [],
            "generated_new_tokens": 0,
            "observed_generation_steps": 0,
            "observed_steps_complete": False,
            "length_scope": "finalized_sequence",
        }

    def __call__(self, input_ids, scores, **kwargs):
        step = input_ids.shape[-1] - 1
        self.trace["observed_generation_steps"] = step
        if not (
            self.profile["min_new_tokens"] <= step < self.profile["max_new_tokens"]
            and step % self.profile["check_every"] == 0
        ):
            return False
        beams = input_ids.detach().cpu().tolist()
        snapshot = {
            "beams": beams,
            **detect_repetition(beams, self.profile, self.trace["special_ids"]),
        }
        self.trace["snapshots"].append(snapshot)
        if snapshot["triggered"]:
            if not self.trace["triggered"]:
                self.trace.update(triggered=True, trigger_new_tokens=step)
            if not self.trace["observe_only"]:
                self.trace.update(
                    observed_steps_complete=True, length_scope="partial_beam_sequence"
                )
                raise RepetitionAbort(snapshot)
        return False


class GuardedOPUSConstraintTranslator(OPUSConstraintTranslator):
    """Explicit repetition guard around the existing validated local callable."""

    def __init__(
        self,
        base: LocalTranslator,
        *,
        profile: dict,
        observe_only: bool = False,
        decoder_dir: Path = DEFAULT_DECODER_DIR,
    ):
        self._profile = validate_profile(profile)
        if type(observe_only) is not bool:
            raise ValueError("observe_only must be a boolean")
        self.observe_only = observe_only
        super().__init__(base, constrained=True, decoder_dir=decoder_dir)
        self._validated_decoder_generate = self._decoder_generate
        self._decoder_generate = self._guarded_generate
        self._active_monitor = None
        self._identity["repetition_stop"] = {
            "profile": deepcopy(self._profile),
            "profile_sha256": profile_sha256(self._profile),
            "observe_only": observe_only,
        }

    def _guarded_generate(self, model, **kwargs):
        monitor = self._active_monitor
        if monitor is None:
            raise RuntimeError("Guarded generation requires a fresh request monitor")
        monitor.trace["enabled"] = True
        # The inherited translator supplies a fresh generation config. Neither
        # model settings nor the validated decoder are modified by this hook.
        return self._validated_decoder_generate(
            model, **kwargs, stopping_criteria=StoppingCriteriaList([monitor])
        )

    def _special_ids(self) -> set[int]:
        ids = set(self.tokenizer.all_special_ids) | self._eos_ids
        for field in ("decoder_start_token_id", "pad_token_id", "eos_token_id"):
            value = getattr(self._generation_config, field, None)
            if value is not None:
                ids.update(value if isinstance(value, (list, tuple)) else [value])
        return ids

    def translate(self, text: str) -> str:
        monitor = _RepetitionMonitor(self._profile, self.observe_only, self._special_ids())
        self._active_monitor = monitor
        try:
            return super().translate(text)
        except RepetitionAbort as exc:
            # These are active beam evidence, not the decoder's selected final
            # hypothesis. Preserve the first beam only as a rejected diagnostic.
            sequence = list(exc.snapshot["beams"][0])
            english = self.tokenizer.decode(sequence, skip_special_tokens=True)
            self.last_trace.update(
                sequence=sequence,
                output=english,
                sequence_role="active_first_beam_diagnostic_not_final",
                eos_terminated=False,
                status="failed",
                error=str(exc),
            )
            for requirement in self.last_trace["requirements"]:
                requirement["tokens_satisfied"] = any(
                    _has_tokens(sequence[1:], tokens)
                    for tokens in requirement["token_alternatives"]
                )
                requirement["phrase_satisfied"] = any(
                    re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", english)
                    for phrase in requirement["target_forms"]
                )
                requirement["satisfied"] = bool(
                    requirement["tokens_satisfied"] and requirement["phrase_satisfied"]
                )
            self.last_trace["satisfied"] = all(
                requirement["satisfied"] for requirement in self.last_trace["requirements"]
            )
            raise
        finally:
            sequence = self.last_trace.get("sequence")
            if sequence is not None:
                monitor.trace["generated_new_tokens"] = len(sequence) - 1
            self.last_trace["repetition_stop"] = monitor.trace
            self._active_monitor = None


def _positive(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _search(args) -> dict:
    if not isinstance(args.query, str) or not args.query.strip() or not _CHINESE.search(args.query):
        raise ValueError("Translation requires a nonempty query containing Chinese text")
    profile = validate_profile(json.loads(args.profile.read_text(encoding="utf-8")))
    started = perf_counter()
    index = ImageIndex.load(args.index)
    identity = index.model_identity
    encoder = SigLIPEmbedder(
        identity["model_id"], identity["revision"], args.device, args.local_files_only
    )
    index.assert_compatible(encoder.identity)
    original = LocalTranslator(args.translation_dir, device=str(encoder.device))
    guarded = GuardedOPUSConstraintTranslator(
        original, profile=profile, decoder_dir=args.decoder_dir
    )
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
            "opus_decoding": "repetition_guarded_constrained_beam4",
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
    search = commands.add_parser("search", help="Guarded constrained OPUS with explicit fallback")
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
