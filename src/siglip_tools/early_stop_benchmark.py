"""Dev-frozen repetition rejection: three fresh fusion controls, no training."""

import argparse
from copy import deepcopy
from math import isfinite
from pathlib import Path
from time import perf_counter

from .auto_benchmark import check_translator_files, translator_snapshot
from .benchmark import (
    MODEL_REVISIONS,
    _environment,
    _json_snapshot,
    _stats,
    _sync,
    load_evaluation_data,
)
from .constraint_fallback import timed_fallback_search
from .early_stop_development import validate_development_evidence
from .encoder import SigLIPEmbedder, identity_fingerprint
from .fallback_benchmark import (
    SOURCE_DATA_HASHES,
    SOURCE_SUPPORT_HASHES,
    TIMING_KEYS,
    WARM_EVIDENCE,
    retrieval_weight_snapshot,
    semantics,
)
from .fallback_benchmark import implementation_hash as previous_core_hash
from .fallback_benchmark import (
    request_rates as legacy_rates,
)
from .frozen_source import reference_implementation, source_sha256
from .fusion_benchmark import _save_new, check_ranking_and_scores
from .images import file_sha256
from .index import ImageIndex
from .term_benchmark import GLOSSARY_PATH, evaluate_attempts, timed_attempt
from .translation import MODEL_ID, REVISION, LocalTranslator

POLICIES = {
    "original_fusion": {"decoder": "original", "mode": "auto_fusion", "fallback": "error"},
    "fallback_opus_fusion": {"decoder": "constrained", "mode": "auto_fusion", "fallback": "opus"},
    "early_fallback_fusion": {"decoder": "guarded", "mode": "auto_fusion", "fallback": "opus"},
}
FAILURE_POLICY = (
    "Only current decoded constraint-output rejection triggers fresh original beam1 OPUS; "
    "no cascade or rejected-English retrieval; failed intents score zero. "
    "Frozen repetition rejection may reject a legacy accepted output and must be reported."
)
PROFILE_PATH = Path(__file__).with_name("repetition_profile_v1.json")
DEVELOPMENT_FILES = (
    "dev-queries.json",
    "dev-protocol.json",
    "dev-observations.json",
    "dev-selection.json",
    "dev-validation.json",
)
QUERY_WARMUP, QUERY_REPEATS = 3, 5
INTENT_COUNT, GALLERY_COUNT = 16, 40
PREVIOUS_CORE_SHA256 = "e377efdeb1bcfe1ff44ac93eca1ee8489d1a5a2032fcba0eabbfdeae958f8675"


def implementation_hash():
    return identity_fingerprint(
        {
            "previous_core": previous_core_hash(),
            **{
                n: source_sha256(n)
                for n in (
                    "repetition_stop.py",
                    "early_stop_benchmark.py",
                    "early_stop_development.py",
                    "repetition_profile_v1.json",
                )
            },
        }
    )


def validate_frozen_inputs(metadata, protocol_path, *, executing=False):
    """Bind the record to its measured source and profile; require live code only when executing."""
    from .repetition_stop import validate_profile

    metadata = Path(metadata)
    protocol, digest = _json_snapshot(protocol_path)
    freeze, _ = _json_snapshot(metadata / "freeze.json")
    code = reference_implementation(metadata, implementation_hash)
    if reference_implementation(metadata, previous_core_hash) != PREVIOUS_CORE_SHA256:
        raise ValueError("Historical fallback implementation changed")
    if freeze.get("protocol_sha256") != digest:
        raise ValueError("Protocol differs from the committed freeze")
    if freeze.get("implementation_sha256") != code or protocol.get("implementation_sha256") != code:
        raise ValueError("Frozen record does not match its measured implementation")
    if executing and implementation_hash() != code:
        raise ValueError("Executing implementation differs from frozen implementation")
    for key in (
        "dataset_hashes",
        "supporting_hashes",
        "development_hashes",
        "repetition_profile_sha256",
    ):
        if freeze.get(key) != protocol.get(key):
            raise ValueError(f"Frozen {key} differs from protocol")
    if (
        protocol.get("dataset_hashes") != SOURCE_DATA_HASHES
        or protocol.get("supporting_hashes") != SOURCE_SUPPORT_HASHES
    ):
        raise ValueError("Experiment must preserve fixed terminology-v1 source bytes")
    if set(protocol.get("development_hashes", {})) != set(DEVELOPMENT_FILES):
        raise ValueError(
            "Missing frozen development protocol, observations, selection or validation"
        )
    hashes = {
        **{f"{s}.json": h for s, h in protocol["dataset_hashes"].items()},
        **protocol["supporting_hashes"],
        **protocol["development_hashes"],
        "repetition_profile_v1.json": protocol["repetition_profile_sha256"],
    }
    for name, expected in hashes.items():
        if file_sha256(metadata / name) != expected:
            raise ValueError(f"Changed frozen metadata: {name}")
    if file_sha256(metadata / "glossary.json") != file_sha256(GLOSSARY_PATH):
        raise ValueError("Benchmark and packaged glossary differ")
    if file_sha256(PROFILE_PATH) != protocol["repetition_profile_sha256"]:
        raise ValueError("Benchmark and packaged repetition profile differ")
    profile = validate_profile(_json_snapshot(PROFILE_PATH)[0])
    validate_development_evidence(metadata, profile)
    return protocol, digest


def _fixed_protocol(protocol, hashes):
    fixed = {
        "policies": POLICIES,
        "translation_model": MODEL_ID,
        "translation_revision": REVISION,
        "dtype": "float32",
        "translation_text_cache": False,
        "training": False,
        "fusion_weights": [0.5, 0.5],
        "source_token_limit": 256,
        "max_new_tokens": 64,
        "num_beams": {"original": 1, "constrained": 4, "guarded": 4},
        "do_sample": False,
        "use_cache": True,
        "score_atol": 1e-5,
        "dataset_hashes": hashes,
        "glossary_sha256": file_sha256(GLOSSARY_PATH),
        "failure_policy": FAILURE_POLICY,
        "warm_evidence": WARM_EVIDENCE,
        "query_warmup": QUERY_WARMUP,
        "query_repeats": QUERY_REPEATS,
        "independent_intents": INTENT_COUNT,
        "gallery_images": GALLERY_COUNT,
    }
    if any(protocol.get(k) != v for k, v in fixed.items()):
        raise ValueError("Protocol differs from fixed policies, data or decoding")


def load_inputs(args):
    protocol, protocol_hash = validate_frozen_inputs(args.metadata, args.protocol, executing=True)
    data = load_evaluation_data(
        args.metadata / "manifest.json",
        args.metadata / "queries.json",
        args.metadata / "qrels.json",
        args.data_root,
    )
    _fixed_protocol(protocol, data["hashes"])
    for directory, key in (
        (args.translation_dir, "translator_files"),
        (args.decoder_dir, "decoder_files"),
    ):
        if translator_snapshot(directory) != protocol[key]:
            raise ValueError("Translator or decoder artifacts differ from protocol")
    if retrieval_weight_snapshot(args.model) != protocol["retrieval_weights"][args.model]:
        raise ValueError("Retrieval weight bytes differ from protocol")
    started = perf_counter()
    index_hash = file_sha256(args.index)
    if index_hash != protocol["source_indexes"][args.model]:
        raise ValueError("Source index differs from frozen model slot")
    index = ImageIndex.load(args.index)
    if file_sha256(args.index) != index_hash:
        raise ValueError("Index changed during loading")
    index_seconds = perf_counter() - started
    expected = {r["path"]: r["sha256"] for r in data["images"]}
    if (
        {r.path: r.sha256 for r in index.records} != expected
        or len(index.records) != len(expected)
        or any(r.aliases for r in index.records)
    ):
        raise ValueError("Index must exactly match frozen gallery without aliases")
    if (index.model_identity["model_id"], index.model_identity["revision"]) != MODEL_REVISIONS[
        args.model
    ]:
        raise ValueError("Index uses wrong pinned model")
    pairs = {}
    for query in data["queries"]:
        pairs.setdefault(query["intent_id"], {})[query["language"]] = query
    inputs = {
        "gallery_ids": data["image_ids"],
        "qrels": {i: data["qrels"][p["zh"]["query_id"]] for i, p in pairs.items()},
    }
    if len(pairs) != INTENT_COUNT or len(expected) != GALLERY_COUNT:
        raise ValueError("Expected fixed sixteen intents and complete forty-image gallery")
    if protocol.get("evaluation_inputs_sha256") != identity_fingerprint(inputs):
        raise ValueError("Evaluation inputs differ from protocol")
    return (
        data,
        protocol,
        protocol_hash,
        index,
        index_hash,
        index_seconds,
        dict(sorted(pairs.items())),
    )


def build_translators(args, protocol):
    from .opus_constraints import OPUSConstraintTranslator
    from .repetition_stop import GuardedOPUSConstraintTranslator

    profile, _ = _json_snapshot(PROFILE_PATH)
    original = LocalTranslator(args.translation_dir, device=protocol["translator_device"])
    return {
        "original": original,
        "constrained": OPUSConstraintTranslator(original, decoder_dir=args.decoder_dir),
        "guarded": GuardedOPUSConstraintTranslator(
            original,
            profile=profile,
            decoder_dir=args.decoder_dir,
        ),
    }


def _fraction(n, d):
    return {"numerator": n, "denominator": d, "rate": n / d if d else None}


def request_rates(attempts):
    return {
        **legacy_rates(attempts),
        "early_rejection": _fraction(sum(a["early_rejected"] for a in attempts), len(attempts)),
    }


def _without_guard(trace):
    return {k: v for k, v in trace.items() if k != "repetition_stop"} if trace else trace


def _check_control(actual, expected):
    for key in ("ok", "translation", "query_texts", "effective_route"):
        if actual[key] != expected[key]:
            raise ValueError(f"Branch {key} differs from fresh control")
    if actual["ok"]:
        return check_ranking_and_scores(actual["rankings"], expected["rankings"])
    return 0.0


def audit_branches(strategies):
    """Allow early rejection of legacy accepted outputs; audit every actual route."""
    checked, maximum = 0, 0.0
    originals = strategies["original_fusion"]["attempts"]
    legacy = strategies["fallback_opus_fusion"]["attempts"]
    for name in ("fallback_opus_fusion", "early_fallback_fusion"):
        for intent, actual in strategies[name]["attempts"].items():
            primary = actual["constraint_trace"]
            if (
                not isinstance(primary, dict)
                or actual["fallback_triggered"] != (actual["primary_failure"] is not None)
                or (actual["primary_failure"] is None and primary.get("status") != "success")
                or (actual["primary_failure"] is not None and primary.get("status") != "failed")
            ):
                raise ValueError("Invalid primary trace or rejection evidence")
            early = primary.get("repetition_stop", {}).get("triggered", False)
            if actual["early_rejected"] != early or (early and actual["primary_failure"] is None):
                raise ValueError("Early rejection disagrees with primary evidence")
            if name == "early_fallback_fusion" and not early:
                control = legacy[intent]
                if (
                    _without_guard(primary) != control["constraint_trace"]
                    or actual["primary_failure"] != control["primary_failure"]
                ):
                    raise ValueError(
                        "Nontrigger primary trace differs from legacy control evidence"
                    )
                maximum = max(maximum, _check_control(actual, control))
            if actual["fallback_triggered"]:
                control = originals[intent]
                if actual["fallback_trace"] != control["original_trace"]:
                    raise ValueError("Fallback trace differs from fresh original control evidence")
                maximum = max(maximum, _check_control(actual, control))
            elif actual["fallback_trace"] is not None:
                raise ValueError("Unexpected fallback trace without rejection")
            checked += 1
    return {"checked": checked, "max_absolute_score_difference": maximum, "rank_disagreements": 0}


def _validate_guard(attempt, profile):
    """Replay stored in-loop checks instead of trusting the saved trigger flag."""
    from .repetition_stop import detect_repetition

    trace = attempt["constraint_trace"]
    record = trace.get("repetition_stop") if trace else None
    if not isinstance(record, dict):
        raise ValueError("Missing guard evidence")
    if (
        record.get("profile") != profile
        or record.get("profile_sha256") != identity_fingerprint(profile)
        or record.get("observe_only") is not False
        or record.get("triggered") != attempt["early_rejected"]
    ):
        raise ValueError("Guard profile or active-mode evidence differs")
    snapshots = record.get("snapshots")
    if not isinstance(snapshots, list):
        raise ValueError("Missing beam snapshots")
    last, triggered = 0, []
    for snapshot in snapshots:
        recomputed = detect_repetition(snapshot["beams"], profile, record["special_ids"])
        if any(snapshot.get(k) != v for k, v in recomputed.items()):
            raise ValueError(
                "Repetition trigger, quorum or cycle evidence differs from beam snapshot"
            )
        if not recomputed["eligible"] or snapshot["new_tokens"] <= last:
            raise ValueError("Guard snapshot is outside ordered eligible token checks")
        last = snapshot["new_tokens"]
        if snapshot["triggered"]:
            triggered.append(snapshot["new_tokens"])
    count = record.get("generated_new_tokens")
    observed = record.get("observed_generation_steps")
    sequence = trace.get("sequence")
    if (
        type(count) is not int
        or not 0 <= count <= 64
        or not isinstance(sequence, list)
        or count != len(sequence) - 1
        or type(observed) is not int
        or not 0 <= observed <= 64
        or type(record.get("enabled")) is not bool
        or record["enabled"] != trace.get("constraints_applied")
        or record.get("observed_steps_complete") is not record["triggered"]
        or record.get("length_scope")
        != ("partial_beam_sequence" if record["triggered"] else "finalized_sequence")
    ):
        raise ValueError("Invalid generated token count")
    expected_steps = list(range(profile["min_new_tokens"], min(observed, 60) + 1, 4))
    if [s["new_tokens"] for s in snapshots] != expected_steps or (
        not record["enabled"] and (observed or snapshots)
    ):
        raise ValueError("Missing eligible observed generation checks")
    if record["triggered"]:
        if (
            not record.get("enabled")
            or triggered != [record.get("trigger_new_tokens")]
            or last != count
            or observed != count
            or last >= 64
            or not snapshots[-1]["triggered"]
            or trace.get("status") != "failed"
            or trace.get("eos_terminated") is not False
            or trace.get("sequence") != snapshots[-1]["beams"][0]
        ):
            raise ValueError("Early rejection lacks final partial trigger-step evidence")
    elif triggered or record.get("trigger_new_tokens") is not None:
        raise ValueError("Nontrigger request contains active rejection evidence")


def _timing_summary(samples):
    return {k: _stats([s["timing"][k] for s in samples]) for k in TIMING_KEYS}


def _branch_timing(samples):
    return {
        name: _timing_summary([s for s in samples if s["fallback_triggered"] == trigger])
        for name, trigger in (("fallback_triggered", True), ("no_fallback", False))
        if any(s["fallback_triggered"] == trigger for s in samples)
    }


def _tokens(attempt):
    trace = attempt["constraint_trace"] or attempt["original_trace"] or {}
    guard = trace.get("repetition_stop", {})
    return guard.get("generated_new_tokens", max(0, len(trace.get("sequence", [])) - 1))


def early_stop_analysis(strategies):
    legacy = strategies["fallback_opus_fusion"]["attempts"]
    early = strategies["early_fallback_fusion"]["attempts"]
    old_fail = {i for i, a in legacy.items() if a["primary_failure"] is not None}
    rejected = {i for i, a in early.items() if a["early_rejected"]}
    groups = {
        "legacy_failure": old_fail,
        "early_rejected": rejected,
        "caught_legacy_failure": old_fail & rejected,
        "missed_legacy_failure": old_fail - rejected,
        "early_rejected_legacy_accepted": rejected - old_fail,
    }
    output = {f"{name}_ids": sorted(ids) for name, ids in groups.items()}
    output["rates"] = {
        "early_rejection": _fraction(len(rejected), len(early)),
        "caught_legacy_failure": _fraction(len(old_fail & rejected), len(old_fail)),
        "missed_legacy_failure": _fraction(len(old_fail - rejected), len(old_fail)),
        "early_rejected_legacy_accepted": _fraction(
            len(rejected - old_fail), len(early) - len(old_fail)
        ),
    }
    output["scored_primary_generated_tokens"] = {
        p: {i: _tokens(a) for i, a in s["attempts"].items()} for p, s in strategies.items()
    }
    output["warm_timing_by_subset"] = {
        group: {
            p: _timing_summary([v for i in sorted(ids) for v in s["warm_samples"][i]])
            for p, s in strategies.items()
        }
        for group, ids in groups.items()
        if ids
    }
    return output


def _validate_timing(timing):
    stages = ("constraint_translation_ms", "original_translation_ms", "fallback_translation_ms")
    if (
        set(timing) != set(TIMING_KEYS)
        or any(type(v) not in (int, float) or not isfinite(v) or v < 0 for v in timing.values())
        or abs(timing["translation_ms"] - sum(timing[k] for k in stages)) > 1e-6
        or timing["total_ms"] + 1e-6 < sum(timing[k] for k in (*stages, "retrieval_ms"))
    ):
        raise ValueError("Invalid timing or hidden primary/detector/fallback cost")


def validate_saved_result(result, *, executing=False):
    """Rebuild metrics, rates, warm digests/timings and the three-branch audit."""
    protocol, inputs = result["protocol"], result["evaluation_inputs"]
    _fixed_protocol(protocol, SOURCE_DATA_HASHES)
    if (
        identity_fingerprint(inputs) != protocol.get("evaluation_inputs_sha256")
        or set(result["strategies"]) != set(POLICIES)
        or result.get("dataset_hashes") != SOURCE_DATA_HASHES
        or result.get("development_hashes") != protocol.get("development_hashes")
        or result.get("implementation_sha256") != protocol.get("implementation_sha256")
        or result.get("repetition_profile_sha256") != protocol.get("repetition_profile_sha256")
        or result.get("repetition_profile_sha256") != file_sha256(PROFILE_PATH)
    ):
        raise ValueError("Saved inputs, profile binding or three policies differ from protocol")
    if executing and result.get("implementation_sha256") != implementation_hash():
        raise ValueError("Saved result was not produced by the executing implementation")
    gallery, qrels = inputs["gallery_ids"], inputs["qrels"]
    if len(gallery) != GALLERY_COUNT or len(qrels) != INTENT_COUNT:
        raise ValueError("Saved gallery/intents differ from fixed input counts")
    profile = result["repetition_profile"]
    if profile != _json_snapshot(PROFILE_PATH)[0]:
        raise ValueError("Saved profile differs from packaged frozen profile")
    reconstructed = {}
    for policy, strategy in result["strategies"].items():
        attempts = strategy["attempts"]
        if set(attempts) != set(qrels) or set(strategy["warm_samples"]) != set(qrels):
            raise ValueError("Saved intents differ from fixed labels")
        reconstructed[policy] = {i: semantics(a) for i, a in attempts.items()}
        for intent, attempt in attempts.items():
            _validate_timing(attempt["timing"])
            if (
                type(attempt["ok"]) is not bool
                or type(attempt["early_rejected"]) is not bool
                or attempt["ok"] != (attempt["failure"] is None)
                or (not attempt["ok"] and attempt["rankings"] is not None)
            ):
                raise ValueError("Saved terminal status contradicts failure/ranking")
            for rank, row in enumerate(attempt["rankings"] or [], 1):
                if (
                    row["rank"] != rank
                    or not isfinite(row["score"])
                    or row["relevant"] != (row["image_id"] in qrels[intent])
                ):
                    raise ValueError("Invalid ranking rank, score or relevance")
            if policy == "early_fallback_fusion":
                _validate_guard(attempt, profile)
            elif attempt["early_rejected"]:
                raise ValueError("Early rejection outside guarded policy")
        ids = {
            i: [r["image_id"] for r in a["rankings"]] if a["ok"] else None
            for i, a in attempts.items()
        }
        if any(strategy.get(k) != v for k, v in evaluate_attempts(ids, qrels, gallery).items()):
            raise ValueError("Saved retrieval metrics differ from complete rankings")
        if strategy["rates"] != request_rates(list(attempts.values())):
            raise ValueError("Saved scored rates differ from attempts")
        warm, maximum = [], 0.0
        for intent, values in strategy["warm_samples"].items():
            if len(values) != protocol["query_repeats"]:
                raise ValueError("Warm request count differs from protocol")
            expected = attempts[intent]
            for sample in values:
                for field in (
                    "ok",
                    "failure",
                    "primary_failure",
                    "fallback_triggered",
                    "effective_route",
                    "early_rejected",
                ):
                    if sample[field] != expected[field]:
                        raise ValueError("Warm status/branch differs from scored reference")
                ranking = (
                    sample["ranking_override"]
                    if sample["ranking_override"] is not None
                    else expected["rankings"]
                )
                if sample["evidence_sha256"] != identity_fingerprint(
                    {"semantics": semantics(expected), "rankings": ranking}
                ):
                    raise ValueError("Warm digest differs from reconstructable evidence")
                if expected["ok"]:
                    maximum = max(maximum, check_ranking_and_scores(ranking, expected["rankings"]))
                elif ranking is not None:
                    raise ValueError("Failed warm request contains rankings")
                _validate_timing(sample["timing"])
                warm.append(sample)
        if (
            strategy["warm_rates"] != request_rates(warm)
            or strategy["timing"] != _timing_summary(warm)
            or strategy["timing_by_branch"] != _branch_timing(warm)
            or strategy["max_repeated_absolute_score_difference"] != maximum
            or strategy["rank_disagreements"] != 0
        ):
            raise ValueError("Saved warm rates, timing or drift summary differs from evidence")
    if reconstructed != result["request_semantics"]:
        raise ValueError("Duplicated request semantics differ from attempts")
    if audit_branches(result["strategies"]) != result["branch_audit"]:
        raise ValueError("Saved branch audit differs from attempts")
    if early_stop_analysis(result["strategies"]) != result["early_stop_analysis"]:
        raise ValueError("Saved early rejection analysis differs from actual branches")
    return True


def run_benchmark(args):
    output = Path(args.output) / "result.json"
    if output.exists():
        raise FileExistsError("Result exists; use a new output directory")
    startup = perf_counter()
    data, protocol, protocol_hash, index, index_hash, index_seconds, pairs = load_inputs(args)
    code_hash = implementation_hash()
    started = perf_counter()
    encoder = SigLIPEmbedder(
        *MODEL_REVISIONS[args.model], protocol["device"], local_files_only=True
    )
    _sync(encoder.device)
    encoder_seconds = perf_counter() - started
    index.assert_compatible(encoder.identity)
    started = perf_counter()
    translators = build_translators(args, protocol)
    _sync(translators["original"].device)
    translator_seconds = perf_counter() - started
    identities = {v: t.identity for v, t in translators.items()}
    for variant, translator in translators.items():
        identity = identities[variant]
        expected = {
            "num_beams": protocol["num_beams"][variant],
            "do_sample": False,
            "max_new_tokens": 64,
            "use_cache": True,
        }
        if (
            identity["source_model"] != MODEL_ID
            or identity["source_revision"] != REVISION
            or identity["dtype"] != "float32"
            or str(translator.device) != protocol["translator_device"]
            or any(identity["generation"].get(k) != v for k, v in expected.items())
        ):
            raise ValueError("Translator identity/configuration differs from protocol")
    if str(encoder.device) != protocol["device"]:
        raise ValueError("Retrieval device differs from protocol")
    startup_seconds = perf_counter() - startup
    by_path = {r["path"]: r["image_id"] for r in data["images"]}
    relevance = {i: data["qrels"][p["zh"]["query_id"]] for i, p in pairs.items()}
    intents = list(pairs)

    def retrieve(policy, intent):
        recipe, chinese = POLICIES[policy], pairs[intent]["zh"]["text"]
        if recipe["decoder"] != "original":
            result = timed_fallback_search(
                index,
                encoder,
                translators[recipe["decoder"]],
                translators["original"],
                chinese,
                "auto_fusion",
                "opus",
                len(data["images"]),
            )
            result["original_trace"] = None
        else:
            result = timed_attempt(
                index, encoder, translators["original"], chinese, "auto_fusion", len(data["images"])
            )
            trace = result.pop("constraint_trace")
            texts = [chinese, result["translation"]] if result["translation"] is not None else []
            result.update(
                {
                    "primary_failure": None,
                    "constraint_trace": None,
                    "fallback_trace": None,
                    "original_trace": trace,
                    "fallback_triggered": False,
                    "effective_route": "original_fusion" if texts else None,
                    "query_texts": texts,
                }
            )
            result["timing"]["original_translation_ms"] = result["timing"]["translation_ms"]
        result["early_rejected"] = bool(
            (result["constraint_trace"] or {}).get("repetition_stop", {}).get("triggered", False)
        )
        for key in TIMING_KEYS:
            result["timing"].setdefault(key, 0.0)
        matches = result.pop("matches")
        result["rankings"] = (
            [
                {
                    "image_id": by_path[m.path],
                    "rank": m.rank,
                    "score": m.score,
                    "relevant": by_path[m.path] in relevance[intent],
                }
                for m in matches
            ]
            if result["ok"]
            else None
        )
        return result

    first_requests = {POLICIES[p]["decoder"]: retrieve(p, intents[0]) for p in POLICIES}
    attempts = {p: {} for p in POLICIES}
    for policy in POLICIES:
        for j in range(protocol["query_warmup"]):
            retrieve(policy, intents[j % len(intents)])
        for intent in intents:
            attempts[policy][intent] = retrieve(policy, intent)
    samples = {p: {i: [] for i in intents} for p in POLICIES}
    drift = {p: 0.0 for p in POLICIES}
    order = list(POLICIES)
    for round_no in range(protocol["query_repeats"]):
        for j, intent in enumerate(intents):
            offset = (round_no + j) % len(order)
            for policy in order[offset:] + order[:offset]:
                actual, expected = retrieve(policy, intent), attempts[policy][intent]
                if semantics(actual) != semantics(expected):
                    raise ValueError("Repeated output/trace/status/branch changed; discard run")
                difference = (
                    check_ranking_and_scores(actual["rankings"], expected["rankings"])
                    if actual["ok"]
                    else 0.0
                )
                drift[policy] = max(drift[policy], difference)
                samples[policy][intent].append(
                    {
                        **{
                            k: actual[k]
                            for k in (
                                "timing",
                                "ok",
                                "failure",
                                "primary_failure",
                                "fallback_triggered",
                                "effective_route",
                                "early_rejected",
                            )
                        },
                        "evidence_sha256": identity_fingerprint(
                            {"semantics": semantics(actual), "rankings": actual["rankings"]}
                        ),
                        "ranking_override": actual["rankings"] if difference else None,
                    }
                )
        print(f"Completed warm round {round_no + 1}/{protocol['query_repeats']}", flush=True)
    strategies = {}
    for policy, scored in attempts.items():
        ids = {
            i: [r["image_id"] for r in a["rankings"]] if a["ok"] else None
            for i, a in scored.items()
        }
        warm = [s for values in samples[policy].values() for s in values]
        strategies[policy] = {
            **evaluate_attempts(ids, relevance, data["image_ids"]),
            "attempts": scored,
            "rates": request_rates(list(scored.values())),
            "warm_samples": samples[policy],
            "warm_rates": request_rates(warm),
            "timing": _timing_summary(warm),
            "timing_by_branch": _branch_timing(warm),
            "max_repeated_absolute_score_difference": drift[policy],
            "rank_disagreements": 0,
        }
    validate_frozen_inputs(args.metadata, args.protocol)
    if (
        file_sha256(args.index) != index_hash
        or any(file_sha256(p) != r["sha256"] for p, r in zip(data["paths"], data["images"]))
        or retrieval_weight_snapshot(args.model) != protocol["retrieval_weights"][args.model]
        or {v: t.identity for v, t in translators.items()} != identities
    ):
        raise ValueError("Frozen source artifacts or translator identities changed during run")
    check_translator_files(args.translation_dir, protocol["translator_files"])
    check_translator_files(args.decoder_dir, protocol["decoder_files"])
    profile, _ = _json_snapshot(PROFILE_PATH)
    payload = {
        "schema_version": 1,
        "experiment_id": protocol["experiment_id"],
        "model_alias": args.model,
        "model_identity": encoder.identity,
        "translator_identities": identities,
        "environment": _environment(encoder.device),
        "dataset_hashes": data["hashes"],
        "protocol": protocol,
        "protocol_sha256": protocol_hash,
        "implementation_sha256": code_hash,
        "source_index_sha256": index_hash,
        "gallery_images": len(data["images"]),
        "independent_intents": len(intents),
        "training_executed": False,
        "repetition_profile": profile,
        "repetition_profile_sha256": protocol["repetition_profile_sha256"],
        "development_hashes": deepcopy(protocol["development_hashes"]),
        "evaluation_inputs": {"gallery_ids": data["image_ids"], "qrels": relevance},
        "request_semantics": {
            p: {i: semantics(a) for i, a in a.items()} for p, a in attempts.items()
        },
        "strategies": strategies,
        "branch_audit": audit_branches(strategies),
        "early_stop_analysis": early_stop_analysis(strategies),
        "startup": {
            "index_load_seconds": index_seconds,
            "retrieval_model_load_seconds": encoder_seconds,
            "shared_translator_and_decoder_load_seconds": translator_seconds,
            "validation_and_models_seconds": startup_seconds,
            "first_requests_per_decoder": first_requests,
            "first_request_order": ["original", "constrained", "guarded"],
        },
    }
    validate_saved_result(payload, executing=True)
    _save_new(output, payload)
    print(
        {
            p: {
                "mAP": s["metrics"]["aggregate"]["average_precision"],
                "rates": s["rates"],
                "total_ms": s["timing"]["total_ms"],
            }
            for p, s in strategies.items()
        },
        flush=True,
    )


def compare_results(base, quality):
    for alias, result in (("base", base), ("quality", quality)):
        validate_saved_result(result)
        identity, protocol = result["model_identity"], result["protocol"]
        if (
            result["model_alias"] != alias
            or (identity["model_id"], identity["revision"]) != MODEL_REVISIONS[alias]
            or result["source_index_sha256"] != protocol["source_indexes"].get(alias)
            or result["implementation_sha256"] != protocol["implementation_sha256"]
        ):
            raise ValueError(
                "Result model slot, source index or implementation differs from protocol"
            )
    for key in (
        "dataset_hashes",
        "protocol",
        "protocol_sha256",
        "implementation_sha256",
        "environment",
        "translator_identities",
        "request_semantics",
        "repetition_profile",
        "repetition_profile_sha256",
        "development_hashes",
    ):
        if base[key] != quality[key]:
            raise ValueError(f"Comparison requires same {key}")
    if base["model_identity"]["runtime_versions"] != quality["model_identity"]["runtime_versions"]:
        raise ValueError("Retrieval runtime differs")
    return {
        "quality_minus_base": {
            p: {
                k: quality["strategies"][p]["metrics"]["aggregate"][k] - v
                for k, v in base["strategies"][p]["metrics"]["aggregate"].items()
            }
            for p in POLICIES
        },
        "quality_to_base_p50_ratio": {
            p: quality["strategies"][p]["timing"]["total_ms"]["p50"]
            / base["strategies"][p]["timing"]["total_ms"]["p50"]
            for p in POLICIES
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--model", choices=list(MODEL_REVISIONS), required=True)
    for name in ("index", "output"):
        run.add_argument(f"--{name}", type=Path, required=True)
    for name, default in (
        ("metadata", "benchmarks/repetition-early-stop-v1"),
        ("data-root", "data/equipment-v1"),
        ("protocol", "benchmarks/repetition-early-stop-v1/protocol.json"),
        ("translation-dir", "models/opus-mt-zh-en"),
        ("decoder-dir", "models/opus-constrained-decoder"),
    ):
        run.add_argument(f"--{name}", type=Path, default=Path(default))
    compare = commands.add_parser("compare")
    for name in ("base", "quality", "output"):
        compare.add_argument(f"--{name}", type=Path, required=True)
    compare.add_argument(
        "--metadata", type=Path, default=Path("benchmarks/repetition-early-stop-v1")
    )
    args = parser.parse_args(argv)
    if args.command == "run":
        run_benchmark(args)
    else:
        protocol, digest = validate_frozen_inputs(args.metadata, args.metadata / "protocol.json")
        results = [_json_snapshot(p)[0] for p in (args.base, args.quality)]
        if any(r["protocol"] != protocol or r["protocol_sha256"] != digest for r in results):
            raise ValueError("Result differs from frozen protocol")
        _save_new(args.output, compare_results(*results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
