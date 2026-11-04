"""Frozen live constraint-rejection fallback evaluation; no training or text cache."""

import argparse
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
from .encoder import SigLIPEmbedder, identity_fingerprint
from .frozen_source import reference_implementation, source_sha256
from .fusion_benchmark import _save_new, check_ranking_and_scores
from .images import file_sha256
from .index import ImageIndex
from .term_benchmark import GLOSSARY_PATH, evaluate_attempts, timed_attempt
from .term_benchmark import implementation_hash as previous_core_hash
from .translation import MODEL_ID, REVISION, LocalTranslator

POLICIES = {
    "zh": {"decoder": None, "mode": "zh", "fallback": "error"},
    **{
        f"{variant}_{suffix}": {"decoder": variant, "mode": mode, "fallback": "error"}
        for variant in ("original", "constrained")
        for suffix, mode in (("en", "auto_en"), ("fusion", "auto_fusion"))
    },
    **{
        f"fallback_{name}_{suffix}": {
            "decoder": "constrained",
            "mode": mode,
            "fallback": fallback,
        }
        for name, fallback in (("opus", "opus"), ("zh", "chinese"))
        for suffix, mode in (("en", "auto_en"), ("fusion", "auto_fusion"))
    },
}
FAILURE_POLICY = (
    "Only current decoded constraint-output rejection triggers explicit fallback; "
    "OPUS fallback failure is terminal; Chinese fallback searches Chinese once; "
    "failed intents score zero."
)
WARM_EVIDENCE = (
    "Per-request semantic+ranking digest; full scored reference, "
    "with full ranking override for nonzero tolerated score drift."
)
QUERY_WARMUP = 3
QUERY_REPEATS = 5
TIMING_KEYS = (
    "constraint_translation_ms",
    "original_translation_ms",
    "fallback_translation_ms",
    "translation_ms",
    "retrieval_ms",
    "total_ms",
)
SOURCE_DATA_HASHES = {
    "manifest": "3487fdf62b90971619ec098b2ec5bbad00e5af2d22ec9183d258bc1a24916975",
    "queries": "19a1da3d4fa9d961f379af0d56dad9274e698895af0d2800d35ad44dfad445db",
    "qrels": "0a6c9486084c5108439f642f4440d8ec4191ff8221a2ee0c553f8589f12daf95",
}
SOURCE_SUPPORT_HASHES = {
    "annotation-notes.json": "717d4ff5bcaeeadb395f14109fb1e6438db2eb5497cb740b5e68d188ed170c6d",
    "review-rules.json": "1feccdda7885c3d6f830c63b5613fbf23dbae8b22e9b09a5da839eb3e859829a",
    "glossary.json": "20fef944133fe8453e9b60d5d7a6ccabb93ce3a4ffe0565038caaf8f51e3c42e",
    "label-review.json": "f721ca9cc296ccf2fec0c2aa10be121982f85736c69e1242e83229c1803460e7",
}


def implementation_hash():
    return identity_fingerprint(
        {
            "previous_core": previous_core_hash(),
            **{
                n: source_sha256(n)
                for n in (
                    "constraint_fallback.py",
                    "fallback_benchmark.py",
                    "cli.py",
                )
            },
        }
    )


def retrieval_weight_snapshot(alias):
    """Hash the exact locally cached pinned safetensors before model allocation."""
    from huggingface_hub import hf_hub_download

    model_id, revision = MODEL_REVISIONS[alias]
    path = Path(
        hf_hub_download(
            model_id,
            "model.safetensors",
            revision=revision,
            local_files_only=True,
        )
    )
    return {
        "model_id": model_id,
        "revision": revision,
        "filename": "model.safetensors",
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def validate_frozen_inputs(metadata, protocol_path, *, executing=False):
    """Bind the frozen record to its measured source; require live code only when executing."""
    metadata = Path(metadata)
    protocol, digest = _json_snapshot(protocol_path)
    freeze, _ = _json_snapshot(metadata / "freeze.json")
    code = reference_implementation(metadata, implementation_hash)
    if freeze.get("protocol_sha256") != digest:
        raise ValueError("Protocol differs from the committed freeze")
    if freeze.get("implementation_sha256") != code or protocol.get("implementation_sha256") != code:
        raise ValueError("Frozen record does not match its measured implementation")
    if executing and implementation_hash() != code:
        raise ValueError("Executing implementation differs from the frozen implementation")
    for key in ("dataset_hashes", "supporting_hashes"):
        if freeze.get(key) != protocol.get(key):
            raise ValueError(f"Frozen {key} differs from protocol")
    if (
        protocol["dataset_hashes"] != SOURCE_DATA_HASHES
        or protocol["supporting_hashes"] != SOURCE_SUPPORT_HASHES
    ):
        raise ValueError("Experiment must preserve the fixed terminology-v1 source bytes")
    for stem, expected in protocol["dataset_hashes"].items():
        if file_sha256(metadata / f"{stem}.json") != expected:
            raise ValueError(f"Changed dataset: {stem}")
    for filename, expected in protocol["supporting_hashes"].items():
        if file_sha256(metadata / filename) != expected:
            raise ValueError(f"Changed supporting metadata: {filename}")
    if file_sha256(metadata / "glossary.json") != file_sha256(GLOSSARY_PATH):
        raise ValueError("Benchmark and packaged glossary differ")
    return protocol, digest


def load_inputs(args):
    protocol, protocol_hash = validate_frozen_inputs(args.metadata, args.protocol, executing=True)
    data = load_evaluation_data(
        args.metadata / "manifest.json",
        args.metadata / "queries.json",
        args.metadata / "qrels.json",
        args.data_root,
    )
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
        "num_beams": {"original": 1, "constrained": 4},
        "do_sample": False,
        "use_cache": True,
        "score_atol": 1e-5,
        "dataset_hashes": data["hashes"],
        "glossary_sha256": file_sha256(GLOSSARY_PATH),
        "failure_policy": FAILURE_POLICY,
        "warm_evidence": WARM_EVIDENCE,
        "query_warmup": QUERY_WARMUP,
        "query_repeats": QUERY_REPEATS,
    }
    if any(protocol.get(k) != v for k, v in fixed.items()):
        raise ValueError("Protocol differs from fixed policies, data or decoding")
    for key, minimum in (("query_warmup", 0), ("query_repeats", 1)):
        if type(protocol.get(key)) is not int or protocol[key] < minimum:
            raise ValueError("Invalid warmup/repeat protocol")
    if set(protocol["supporting_hashes"]) != {
        "annotation-notes.json",
        "review-rules.json",
        "glossary.json",
        "label-review.json",
    }:
        raise ValueError("Missing fixed supporting metadata")
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
    evaluation_inputs = {
        "gallery_ids": data["image_ids"],
        "qrels": {i: data["qrels"][p["zh"]["query_id"]] for i, p in pairs.items()},
    }
    if protocol.get("evaluation_inputs_sha256") != identity_fingerprint(evaluation_inputs):
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

    original = LocalTranslator(args.translation_dir, device=protocol["translator_device"])
    return {
        "original": original,
        "constrained": OPUSConstraintTranslator(original, decoder_dir=args.decoder_dir),
    }


def request_rates(attempts):
    def fraction(n, d):
        return {"numerator": n, "denominator": d, "rate": n / d if d else None}

    n = len(attempts)
    triggered = sum(a["fallback_triggered"] for a in attempts)
    return {
        "primary_rejection": fraction(sum(a["primary_failure"] is not None for a in attempts), n),
        "fallback_trigger": fraction(triggered, n),
        "fallback_success": fraction(
            sum(a["fallback_triggered"] and a["ok"] for a in attempts), triggered
        ),
        "terminal_failure": fraction(sum(not a["ok"] for a in attempts), n),
    }


def semantics(attempt):
    return {k: v for k, v in attempt.items() if k not in {"matches", "rankings", "timing"}}


def audit_branches(strategies):
    checked, max_difference = 0, 0.0
    for policy, recipe in POLICIES.items():
        if recipe["fallback"] == "error":
            continue
        suffix = "en" if recipe["mode"] == "auto_en" else "fusion"
        for intent, actual in strategies[policy]["attempts"].items():
            primary = strategies[f"constrained_{suffix}"]["attempts"][intent]
            if (
                actual["constraint_trace"] != primary["constraint_trace"]
                or actual["primary_failure"] != primary["primary_failure"]
                or actual["fallback_triggered"] != (primary["primary_failure"] is not None)
            ):
                raise ValueError(
                    "Fallback branch primary evidence differs from constrained control"
                )
            control = (
                f"constrained_{suffix}"
                if not actual["fallback_triggered"]
                else f"original_{suffix}"
                if recipe["fallback"] == "opus"
                else "zh"
            )
            expected = strategies[control]["attempts"][intent]
            if (
                actual["fallback_triggered"]
                and recipe["fallback"] == "opus"
                and actual["fallback_trace"] != expected["original_trace"]
            ):
                raise ValueError("Fallback trace differs from original control evidence")
            if actual["ok"] != expected["ok"]:
                raise ValueError("Fallback branch terminal status differs from control")
            if actual["query_texts"] != expected["query_texts"]:
                raise ValueError("Fallback branch queried different texts from control")
            if actual["translation"] != expected["translation"]:
                raise ValueError("Fallback branch used English differs from control")
            if actual["ok"]:
                max_difference = max(
                    max_difference,
                    check_ranking_and_scores(actual["rankings"], expected["rankings"]),
                )
            checked += 1
    return {
        "checked": checked,
        "max_absolute_score_difference": max_difference,
        "rank_disagreements": 0,
    }


def validate_saved_result(result, *, executing=False):
    """Reconstruct saved evidence, metrics, rates and timing before publishing comparisons."""
    protocol = result["protocol"]
    inputs = result["evaluation_inputs"]
    if identity_fingerprint(inputs) != protocol.get("evaluation_inputs_sha256"):
        raise ValueError("Saved evaluation inputs differ from protocol")
    gallery, qrels = inputs["gallery_ids"], inputs["qrels"]
    if set(result["strategies"]) != set(POLICIES):
        raise ValueError("Saved result lacks nine policies")
    reconstructed = {}
    for policy, strategy in result["strategies"].items():
        attempts = strategy["attempts"]
        if set(attempts) != set(qrels) or set(strategy["warm_samples"]) != set(qrels):
            raise ValueError("Saved intents differ from fixed labels")
        reconstructed[policy] = {i: semantics(a) for i, a in attempts.items()}
        ids = {
            i: [r["image_id"] for r in a["rankings"]] if a["ok"] else None
            for i, a in attempts.items()
        }
        for i, a in attempts.items():
            if a["ok"] != (a["failure"] is None) or (not a["ok"] and a["rankings"] is not None):
                raise ValueError("Saved terminal status contradicts failure/ranking")
            for rank, row in enumerate(a["rankings"] or [], 1):
                if (
                    row["rank"] != rank
                    or not isfinite(row["score"])
                    or row["relevant"] != (row["image_id"] in qrels[i])
                ):
                    raise ValueError("Saved ranking has invalid rank, score or relevance")
        metrics = evaluate_attempts(ids, qrels, gallery)
        if any(strategy.get(k) != v for k, v in metrics.items()):
            raise ValueError("Saved retrieval metrics differ from complete rankings")
        if strategy["rates"] != request_rates(list(attempts.values())):
            raise ValueError("Saved scored failure rates differ from attempts")
        warm, maximum = [], 0.0
        for intent, values in strategy["warm_samples"].items():
            if len(values) != protocol["query_repeats"]:
                raise ValueError("Saved warm request count differs from protocol")
            expected = attempts[intent]
            for sample in values:
                for field in (
                    "ok",
                    "failure",
                    "primary_failure",
                    "fallback_triggered",
                    "effective_route",
                ):
                    if sample[field] != expected[field]:
                        raise ValueError("Warm status/branch differs from scored reference")
                ranking = sample["ranking_override"] or expected["rankings"]
                digest = identity_fingerprint(
                    {"semantics": semantics(expected), "rankings": ranking}
                )
                if sample["evidence_sha256"] != digest:
                    raise ValueError("Warm evidence digest differs from reconstructable record")
                if expected["ok"]:
                    maximum = max(maximum, check_ranking_and_scores(ranking, expected["rankings"]))
                timing = sample["timing"]
                if (
                    set(timing) != set(TIMING_KEYS)
                    or any(not isfinite(v) or v < 0 for v in timing.values())
                    or timing["total_ms"] + 1e-6
                    < sum(
                        timing[k]
                        for k in (
                            "constraint_translation_ms",
                            "original_translation_ms",
                            "fallback_translation_ms",
                            "retrieval_ms",
                        )
                    )
                ):
                    raise ValueError("Invalid warm timing or hidden failed-attempt cost")
                warm.append(sample)
        if strategy["warm_rates"] != request_rates(warm):
            raise ValueError("Saved warm rates differ from requests")
        if strategy["timing"] != {k: _stats([s["timing"][k] for s in warm]) for k in TIMING_KEYS}:
            raise ValueError("Saved timing statistics differ from live samples")
        branches = {
            branch: {
                k: _stats([s["timing"][k] for s in warm if s["fallback_triggered"] == trigger])
                for k in TIMING_KEYS
            }
            for branch, trigger in (("fallback_triggered", True), ("no_fallback", False))
            if any(s["fallback_triggered"] == trigger for s in warm)
        }
        if (
            strategy["timing_by_branch"] != branches
            or strategy["max_repeated_absolute_score_difference"] != maximum
            or strategy["rank_disagreements"] != 0
        ):
            raise ValueError("Saved branch timing or drift summary differs from evidence")
    if reconstructed != result["request_semantics"]:
        raise ValueError("Saved duplicated request semantics differ from attempts")
    if audit_branches(result["strategies"]) != result["branch_audit"]:
        raise ValueError("Saved branch audit differs from attempts")
    return True


def run_benchmark(args):
    from .constraint_fallback import timed_fallback_search

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
        if recipe["decoder"] == "constrained":
            result = timed_fallback_search(
                index,
                encoder,
                translators["constrained"],
                translators["original"],
                chinese,
                recipe["mode"],
                recipe["fallback"],
                len(data["images"]),
            )
            result["original_trace"] = None
        else:
            translator = translators["original"] if recipe["decoder"] else None
            result = timed_attempt(
                index, encoder, translator, chinese, recipe["mode"], len(data["images"])
            )
            trace = result.pop("constraint_trace")
            result.update(
                {
                    "primary_failure": None,
                    "constraint_trace": None,
                    "fallback_trace": None,
                    "original_trace": trace,
                    "fallback_triggered": False,
                    "effective_route": policy if result["ok"] else None,
                    "query_texts": (
                        [chinese]
                        if policy == "zh"
                        else [result["translation"]]
                        if recipe["mode"] == "auto_en"
                        else [chinese, result["translation"]]
                    )
                    if result["ok"]
                    else [],
                }
            )
            result["timing"]["original_translation_ms"] = result["timing"]["translation_ms"]
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

    first_requests = {v: retrieve(f"{v}_en", intents[0]) for v in ("original", "constrained")}
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
                    raise ValueError("Repeated output/trace/branch changed; discard run")
                difference = (
                    check_ranking_and_scores(actual["rankings"], expected["rankings"])
                    if actual["ok"]
                    else 0.0
                )
                drift[policy] = max(drift[policy], difference)
                samples[policy][intent].append(
                    {
                        "timing": actual["timing"],
                        "ok": actual["ok"],
                        "failure": actual["failure"],
                        "primary_failure": actual["primary_failure"],
                        "fallback_triggered": actual["fallback_triggered"],
                        "effective_route": actual["effective_route"],
                        "evidence_sha256": identity_fingerprint(
                            {"semantics": semantics(actual), "rankings": actual["rankings"]}
                        ),
                        "ranking_override": actual["rankings"] if difference else None,
                    }
                )
        print(f"Completed warm round {round_no + 1}/{protocol['query_repeats']}", flush=True)
    strategies = {}
    for policy in POLICIES:
        scored = attempts[policy]
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
            "timing": {k: _stats([s["timing"][k] for s in warm]) for k in TIMING_KEYS},
            "timing_by_branch": {
                branch: {
                    k: _stats([s["timing"][k] for s in warm if s["fallback_triggered"] == trigger])
                    for k in TIMING_KEYS
                }
                for branch, trigger in (("fallback_triggered", True), ("no_fallback", False))
                if any(s["fallback_triggered"] == trigger for s in warm)
            },
            "max_repeated_absolute_score_difference": drift[policy],
            "rank_disagreements": 0,
        }
    branch_audit = audit_branches(strategies)
    validate_frozen_inputs(args.metadata, args.protocol)
    if (
        file_sha256(args.index) != index_hash
        or any(
            file_sha256(path) != row["sha256"] for path, row in zip(data["paths"], data["images"])
        )
        or retrieval_weight_snapshot(args.model) != protocol["retrieval_weights"][args.model]
        or {v: t.identity for v, t in translators.items()} != identities
    ):
        raise ValueError("Frozen artifacts or translator identities changed during run")
    check_translator_files(args.translation_dir, protocol["translator_files"])
    check_translator_files(args.decoder_dir, protocol["decoder_files"])
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
        "request_semantics": {
            p: {i: semantics(a) for i, a in scored.items()} for p, scored in attempts.items()
        },
        "strategies": strategies,
        "branch_audit": branch_audit,
        "startup": {
            "index_load_seconds": index_seconds,
            "retrieval_model_load_seconds": encoder_seconds,
            "shared_translator_and_decoder_load_seconds": translator_seconds,
            "validation_and_models_seconds": startup_seconds,
            "first_requests_per_decoder": first_requests,
            "first_request_order": ["original", "constrained"],
        },
    }
    payload["evaluation_inputs"] = {"gallery_ids": data["image_ids"], "qrels": relevance}
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
        identity, protocol = result["model_identity"], result["protocol"]
        if not isinstance(protocol, dict):
            raise ValueError("Invalid protocol")
        validate_saved_result(result)
        if (
            result["model_alias"] != alias
            or (identity["model_id"], identity["revision"]) != MODEL_REVISIONS[alias]
        ):
            raise ValueError("Expected pinned Base and So400m in named slots")
        if result["source_index_sha256"] != protocol["source_indexes"].get(alias):
            raise ValueError("Result index differs from predeclared model slot")
        if result["implementation_sha256"] != protocol["implementation_sha256"]:
            raise ValueError("Result implementation differs from protocol")
        if set(result["strategies"]) != set(POLICIES):
            raise ValueError("Result must contain all nine policies")
    for key in (
        "dataset_hashes",
        "protocol",
        "protocol_sha256",
        "implementation_sha256",
        "environment",
        "translator_identities",
        "request_semantics",
    ):
        if base[key] != quality[key]:
            raise ValueError(f"Comparison requires same {key}")
    if base["model_identity"]["runtime_versions"] != quality["model_identity"]["runtime_versions"]:
        raise ValueError("Retrieval runtime differs")

    def delta(a, b):
        return {k: a[k] - v for k, v in b.items()}

    return {
        "quality_minus_base": {
            p: delta(
                quality["strategies"][p]["metrics"]["aggregate"],
                base["strategies"][p]["metrics"]["aggregate"],
            )
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
        ("metadata", "benchmarks/constraint-fallback-v1"),
        ("data-root", "data/equipment-v1"),
        ("protocol", "benchmarks/constraint-fallback-v1/protocol.json"),
        ("translation-dir", "models/opus-mt-zh-en"),
        ("decoder-dir", "models/opus-constrained-decoder"),
    ):
        run.add_argument(f"--{name}", type=Path, default=Path(default))
    compare = commands.add_parser("compare")
    for name in ("base", "quality", "output"):
        compare.add_argument(f"--{name}", type=Path, required=True)
    compare.add_argument("--metadata", type=Path, default=Path("benchmarks/constraint-fallback-v1"))
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
