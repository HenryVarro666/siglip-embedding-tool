"""Observe eight fixed development queries and freeze a repetition rule before evaluation."""

import argparse
from copy import deepcopy
from math import fsum
from pathlib import Path
from time import perf_counter

from .auto_benchmark import check_translator_files, translator_snapshot
from .benchmark import _json_snapshot, _sync
from .encoder import identity_fingerprint
from .fallback_benchmark import implementation_hash as legacy_hash
from .frozen_source import reference_implementation, source_sha256
from .fusion_benchmark import _save_new
from .images import file_sha256
from .opus_constraints import OPUSConstraintTranslator
from .translation import LocalTranslator

DEVELOPMENT_ROWS_SHA256 = "53bfaf65a00cae94a2c578907870c6694b36f3f50c85515038930d19c0de045b"
SELECTION_RULE = (
    "Exclude candidates rejecting any baseline-accepted dev output; maximize caught baseline "
    "failures, then higher beam quorum, lower mean trigger tokens, more copies, smaller maximum "
    "ngram, larger minimum tokens, lexical profile ID. If no viable candidate, freeze "
    "n8/max_declared_repeat/min32/q4 and mark nonviable."
)


def development_candidates():
    """Stage2 grid declared after preserving unsuccessful stage1 development."""
    from .repetition_stop import candidate_profiles_v2

    return candidate_profiles_v2()


def development_implementation_hash():
    return identity_fingerprint(
        {
            "legacy": legacy_hash(),
            **{n: source_sha256(n) for n in ("repetition_stop.py", "early_stop_development.py")},
        }
    )


def load_development_queries(path):
    doc, _ = _json_snapshot(path)
    rows = doc.get("queries")
    if (
        doc.get("role") != "development_only"
        or identity_fingerprint(rows) != DEVELOPMENT_ROWS_SHA256
        or [r.get("query_id") for r in rows] != [f"i{i:02}_zh" for i in range(1, 9)]
    ):
        raise ValueError(
            "Expected the eight fixed development source strings, never formal queries"
        )
    return rows


def select_profile(candidates, observations):
    """Replay declared rules on complete dev beam histories with no retrieval outcomes."""
    from .repetition_stop import detect_repetition, validate_profile

    rows = observations["rows"]
    if not rows or any(r["query_id"] not in {f"i{i:02}_zh" for i in range(1, 9)} for r in rows):
        raise ValueError("Selection accepts development IDs only")
    matrix = []
    for raw_profile in candidates:
        profile = validate_profile(raw_profile)
        triggers = {}
        accepted, caught = [], []
        for row in rows:
            snapshots = row["observer"]["trace"]["repetition_stop"]["snapshots"]
            for snapshot in snapshots:
                actual = detect_repetition(snapshot["beams"], profile, observations["special_ids"])
                if actual["new_tokens"] != snapshot["new_tokens"]:
                    raise ValueError("Development snapshot length differs from beam evidence")
                if actual["triggered"]:
                    triggers[row["query_id"]] = actual["new_tokens"]
                    (accepted if row["baseline"]["ok"] else caught).append(row["query_id"])
                    break
        matrix.append(
            {
                "profile": profile,
                "trigger_new_tokens": triggers,
                "early_rejected_baseline_accepted_ids": sorted(accepted),
                "caught_baseline_failure_ids": sorted(caught),
                "mean_failure_trigger_tokens": fsum(triggers[i] for i in caught) / len(caught)
                if caught
                else None,
            }
        )
    viable = [
        r
        for r in matrix
        if not r["early_rejected_baseline_accepted_ids"] and r["caught_baseline_failure_ids"]
    ]

    def priority(row):
        p = row["profile"]
        return (
            -len(row["caught_baseline_failure_ids"]),
            -p["min_repeating_beams"],
            row["mean_failure_trigger_tokens"],
            -p["repeat_count"],
            p["max_ngram_size"],
            -p["min_new_tokens"],
            p["profile_id"],
        )

    if viable:
        selected = min(viable, key=priority)
    else:
        selected = next(
            r
            for r in matrix
            if (
                r["profile"]["max_ngram_size"],
                r["profile"]["repeat_count"],
                r["profile"]["min_new_tokens"],
                r["profile"]["min_repeating_beams"],
            )
            == (8, max(p["repeat_count"] for p in candidates), 32, 4)
        )
    return {
        "viable": bool(viable),
        "selection_rule": SELECTION_RULE,
        "selected_profile": deepcopy(selected["profile"]),
        "selected_row": deepcopy(selected),
        "candidates": sorted(matrix, key=lambda r: r["profile"]["profile_id"]),
        "training_executed": False,
        "formal_queries_or_retrieval_used": False,
    }


def validate_development_evidence(metadata, profile):
    """Bind the active profile to replayed fixed-development observations and validation."""
    from .repetition_stop import detect_repetition, validate_profile

    root = Path(metadata)
    profile = validate_profile(profile)
    queries = load_development_queries(root / "dev-queries.json")
    protocol, protocol_hash = _json_snapshot(root / "dev-protocol.json")
    observations, observed_hash = _json_snapshot(root / "dev-observations.json")
    selection, selection_hash = _json_snapshot(root / "dev-selection.json")
    validation, _ = _json_snapshot(root / "dev-validation.json")
    candidates = development_candidates()
    expected = {
        "queries_sha256": file_sha256(root / "dev-queries.json"),
        "candidate_profiles": candidates,
        "selection_rule": SELECTION_RULE,
        "development_implementation_sha256": reference_implementation(
            root, development_implementation_hash
        ),
        "training": False,
        "formal_queries_or_retrieval_used": False,
    }
    if any(protocol.get(k) != v for k, v in expected.items()):
        raise ValueError("Development evidence differs from fixed source/code/rule protocol")
    if (
        observations.get("protocol") != protocol
        or observations.get("protocol_sha256") != protocol_hash
    ):
        raise ValueError("Development observations differ from protocol")
    if observations.get("parameters_require_grad") is not False:
        raise ValueError("Development parameters were not frozen")

    def same_sources(rows):
        return [{"query_id": r["query_id"], "text": r["text"]} for r in rows] == queries

    if not same_sources(observations["rows"]) or not same_sources(validation["rows"]):
        raise ValueError("Development evidence replaced fixed source strings")

    def replay_snapshots(trace, expected_profile, observe_only):
        guard = trace["repetition_stop"]
        if (
            guard["profile"] != expected_profile
            or guard["observe_only"] != observe_only
            or guard["special_ids"] != observations["special_ids"]
        ):
            raise ValueError("Development monitor used a different profile or special IDs")
        for snapshot in guard["snapshots"]:
            computed = {
                "beams": snapshot["beams"],
                **detect_repetition(
                    snapshot["beams"], expected_profile, observations["special_ids"]
                ),
            }
            if computed != snapshot:
                raise ValueError("Development repetition evidence differs from beam history")
        return guard

    observer_profile = next(
        p
        for p in candidates
        if (p["max_ngram_size"], p["repeat_count"], p["min_new_tokens"], p["min_repeating_beams"])
        == (16, 2, 24, 3)
    )
    for query, row in zip(queries, observations["rows"]):
        baseline, observed = row["baseline"], row["observer"]
        if (
            baseline["trace"]["source_text"] != query["text"]
            or {k: v for k, v in observed["trace"].items() if k != "repetition_stop"}
            != baseline["trace"]
            or any(observed[k] != baseline[k] for k in ("ok", "english", "failure"))
        ):
            raise ValueError("Development observation changed legacy generation")
        replay_snapshots(observed["trace"], observer_profile, True)
    recomputed = select_profile(candidates, observations)
    recomputed.update(protocol_sha256=protocol_hash, observations_sha256=observed_hash)
    if selection != recomputed or profile != recomputed["selected_profile"]:
        raise ValueError("Active profile is not the rule selected by fixed development replay")
    if (
        validation["protocol_sha256"] != protocol_hash
        or validation["selection_sha256"] != selection_hash
        or validation["profile"] != profile
        or validation["profile_byte_sha256"] != file_sha256(root / "repetition_profile_v1.json")
        or _json_snapshot(root / "repetition_profile_v1.json")[0] != profile
        or validation.get("active_stops_match_replayed_predictions") is not True
    ):
        raise ValueError("Active development validation differs from selected profile")
    predicted = selection["selected_row"]["trigger_new_tokens"]
    for reference, row in zip(observations["rows"], validation["rows"]):
        if any(
            reference["baseline"][k] != row["baseline"][k]
            for k in ("ok", "english", "failure", "trace")
        ):
            raise ValueError("Active validation changed legacy baseline")
        actual = row["guarded"]
        guard = replay_snapshots(actual["trace"], profile, False)
        step = predicted.get(row["query_id"])
        if guard["triggered"] != (step is not None) or guard["trigger_new_tokens"] != step:
            raise ValueError("Actual development abort differs from replay prediction")
        if step is not None:
            if (
                actual["ok"]
                or not guard["snapshots"]
                or guard["snapshots"][-1]["new_tokens"] != step
                or guard["snapshots"][-1]["triggered"] is not True
                or actual["trace"]["sequence"] != guard["snapshots"][-1]["beams"][0]
                or actual["trace"]["eos_terminated"] is not False
            ):
                raise ValueError("Development abort lacks its actual rejected partial evidence")
        elif {k: v for k, v in actual["trace"].items() if k != "repetition_stop"} != reference[
            "baseline"
        ]["trace"]:
            raise ValueError("Non-stopped development output differs from legacy")
        backup = row["fallback"]
        if actual["ok"] != (backup is None):
            raise ValueError("Development fallback trigger differs from primary rejection")
        if backup is not None and backup["trace"]["source_text"] != row["text"]:
            raise ValueError("Development fallback used a different source")
    return True


def load_protocol(args):
    rows = load_development_queries(args.queries)
    protocol, digest = _json_snapshot(args.protocol)
    expected = {
        "queries_sha256": file_sha256(args.queries),
        "candidate_profiles": development_candidates(),
        "selection_rule": SELECTION_RULE,
        "development_implementation_sha256": development_implementation_hash(),
        "training": False,
        "formal_queries_or_retrieval_used": False,
    }
    if any(protocol.get(k) != v for k, v in expected.items()):
        raise ValueError("Development protocol/code/queries differ from predeclared rules")
    for directory, key in (
        (args.translation_dir, "translator_files"),
        (args.decoder_dir, "decoder_files"),
    ):
        if translator_snapshot(directory) != protocol[key]:
            raise ValueError("Development translator/decoder differs from protocol")
    return rows, protocol, digest


def _translate(translator, text):
    _sync(translator.device)
    started = perf_counter()
    try:
        english = translator.translate(text)
        failure, ok = None, True
    except ValueError as exc:
        trace = translator.last_trace
        if not isinstance(trace.get("sequence"), list) or not isinstance(trace.get("output"), str):
            raise
        english, failure, ok = None, {"type": type(exc).__name__, "message": str(exc)}, False
    _sync(translator.device)
    finished = perf_counter()
    return {
        "ok": ok,
        "english": english,
        "failure": failure,
        "translation_ms": (finished - started) * 1000,
        "trace": deepcopy(translator.last_trace),
    }


def run_observation(args):
    from .repetition_stop import GuardedOPUSConstraintTranslator

    if args.output.exists():
        raise FileExistsError("Development observations exist; use a new output path")
    rows, protocol, digest = load_protocol(args)
    profile = next(
        p
        for p in development_candidates()
        if (p["max_ngram_size"], p["repeat_count"], p["min_new_tokens"], p["min_repeating_beams"])
        == (16, 2, 24, 3)
    )
    original = LocalTranslator(args.translation_dir, device=protocol["device"])
    baseline = OPUSConstraintTranslator(original, decoder_dir=args.decoder_dir)
    observer = GuardedOPUSConstraintTranslator(
        original, profile=profile, observe_only=True, decoder_dir=args.decoder_dir
    )
    identities = {
        "original": original.identity,
        "baseline": baseline.identity,
        "observer": observer.identity,
    }
    results = []
    special_ids = sorted(set(original.tokenizer.all_special_ids) | set(original._eos_ids))
    for query in rows:
        reference, actual = _translate(baseline, query["text"]), _translate(observer, query["text"])
        stripped = {k: v for k, v in actual["trace"].items() if k != "repetition_stop"}
        if (
            stripped != reference["trace"]
            or actual["ok"] != reference["ok"]
            or actual["english"] != reference["english"]
        ):
            raise ValueError("Observation changed legacy generation; no selection is valid")
        results.append({**query, "baseline": reference, "observer": actual})
        print(f"Observed {query['query_id']}: baseline_ok={reference['ok']}", flush=True)
    load_protocol(args)
    if {
        "original": original.identity,
        "baseline": baseline.identity,
        "observer": observer.identity,
    } != identities:
        raise ValueError("Development translator configuration mutated")
    for directory, key in (
        (args.translation_dir, "translator_files"),
        (args.decoder_dir, "decoder_files"),
    ):
        check_translator_files(directory, protocol[key])
    _save_new(
        args.output,
        {
            "protocol_sha256": digest,
            "protocol": protocol,
            "special_ids": special_ids,
            "rows": results,
            "translator_identities": identities,
            "development_only": True,
            "training_executed": False,
            "parameters_require_grad": any(p.requires_grad for p in original.model.parameters()),
        },
    )


def run_selection(args):
    if args.output.exists() or args.profile_output.exists():
        raise FileExistsError("Selection/profile exists; do not overwrite frozen development")
    rows, protocol, digest = load_protocol(args)
    observations, observations_hash = _json_snapshot(args.observations)
    if observations["protocol_sha256"] != digest or observations["protocol"] != protocol:
        raise ValueError("Observations differ from frozen development protocol")
    if [{"query_id": r["query_id"], "text": r["text"]} for r in observations["rows"]] != rows:
        raise ValueError("Observations differ from development source strings")
    selected = select_profile(development_candidates(), observations)
    selected.update({"protocol_sha256": digest, "observations_sha256": observations_hash})
    _save_new(args.output, selected)
    _save_new(args.profile_output, selected["selected_profile"])
    print(selected["selected_profile"], flush=True)


def run_validation(args):
    from .repetition_stop import GuardedOPUSConstraintTranslator

    if args.output.exists():
        raise FileExistsError("Development validation exists; use a new path")
    rows, protocol, digest = load_protocol(args)
    selection, selection_hash = _json_snapshot(args.selection)
    profile, profile_hash = _json_snapshot(args.profile)
    if profile != selection["selected_profile"] or selection["protocol_sha256"] != digest:
        raise ValueError("Active rule differs from selected development profile")
    original = LocalTranslator(args.translation_dir, device=protocol["device"])
    baseline = OPUSConstraintTranslator(original, decoder_dir=args.decoder_dir)
    guarded = GuardedOPUSConstraintTranslator(
        original, profile=profile, decoder_dir=args.decoder_dir
    )
    identities = {
        "original": original.identity,
        "baseline": baseline.identity,
        "guarded": guarded.identity,
    }
    results = []
    predicted = selection["selected_row"]["trigger_new_tokens"]
    for query in rows:
        reference = _translate(baseline, query["text"])
        actual = _translate(guarded, query["text"])
        guard_trace = actual["trace"]["repetition_stop"]
        expected_step = predicted.get(query["query_id"])
        if (
            guard_trace["triggered"] != (expected_step is not None)
            or guard_trace["trigger_new_tokens"] != expected_step
        ):
            raise ValueError("Active generation stop disagrees with developer-history replay")
        if not guard_trace["triggered"]:
            if {k: v for k, v in actual["trace"].items() if k != "repetition_stop"} != reference[
                "trace"
            ]:
                raise ValueError("Non-stopped generation differs from legacy")
        backup = _translate(original, query["text"]) if not actual["ok"] else None
        results.append({**query, "baseline": reference, "guarded": actual, "fallback": backup})
        print(
            f"Validated {query['query_id']}: early={guard_trace['triggered']}, "
            f"step={expected_step}",
            flush=True,
        )
    load_protocol(args)
    if identities != {
        "original": original.identity,
        "baseline": baseline.identity,
        "guarded": guarded.identity,
    }:
        raise ValueError("Active development mutated translator config")
    _save_new(
        args.output,
        {
            "protocol_sha256": digest,
            "selection_sha256": selection_hash,
            "profile_byte_sha256": profile_hash,
            "profile": profile,
            "rows": results,
            "translator_identities": identities,
            "development_only": True,
            "training_executed": False,
            "active_stops_match_replayed_predictions": True,
        },
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("observe", "select", "validate"):
        sub = commands.add_parser(command)
        defaults = {
            "queries": "benchmarks/repetition-early-stop-v1/dev-queries.json",
            "protocol": "benchmarks/repetition-early-stop-v1/dev-protocol.json",
            "translation-dir": "models/opus-mt-zh-en",
            "decoder-dir": "models/opus-constrained-decoder",
        }
        for name, default in defaults.items():
            sub.add_argument(f"--{name}", type=Path, default=Path(default))
        sub.add_argument("--output", type=Path, required=True)
        if command == "select":
            sub.add_argument("--observations", type=Path, required=True)
            sub.add_argument("--profile-output", type=Path, required=True)
        if command == "validate":
            sub.add_argument("--selection", type=Path, required=True)
            sub.add_argument("--profile", type=Path, required=True)
    args = parser.parse_args(argv)
    {"observe": run_observation, "select": run_selection, "validate": run_validation}[args.command](
        args
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
