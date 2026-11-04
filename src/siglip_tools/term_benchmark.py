"""Frozen OPUS original/beam/lexical-constraint retrieval evaluation."""

import argparse
from copy import deepcopy
from math import fsum
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
from .evaluation import evaluate_rankings
from .frozen_source import source_sha256
from .fusion import search_texts
from .fusion_benchmark import _save_new, check_ranking_and_scores
from .fusion_benchmark import implementation_hash as fusion_hash
from .images import file_sha256
from .index import ImageIndex
from .translation import MODEL_ID, REVISION, LocalTranslator

VARIANTS = ("original", "beam", "constrained")
GLOSSARY_PATH = Path(__file__).with_name("power_glossary_v1.json")
POLICIES = {
    "zh": {"decoder": None, "mode": "zh"},
    **{
        f"{variant}_{suffix}": {"decoder": variant, "mode": mode}
        for variant in VARIANTS
        for suffix, mode in (("en", "auto_en"), ("fusion", "auto_fusion"))
    },
}


def timed_attempt(index, encoder, translator, chinese, mode="auto_en", top_k=5):
    """Measure a complete request, retaining expected failures and their cost."""
    if mode not in {"zh", "auto_en", "auto_fusion"} or (mode != "zh" and translator is None):
        raise ValueError("Invalid request mode or missing translator")
    _sync(encoder.device)
    if mode != "zh":
        _sync(translator.device)
    started = perf_counter()
    english, stage, translation_ms = None, "translation", 0.0
    try:
        if mode != "zh":
            english = translator.translate(chinese)
            _sync(translator.device)
            translation_ms = (perf_counter() - started) * 1000
        texts = (
            [chinese] if mode == "zh" else [english] if mode == "auto_en" else [chinese, english]
        )
        stage = "retrieval"
        retrieval_started = perf_counter()
        matches = search_texts(index, encoder, texts, top_k=top_k)
        _sync(encoder.device)
        finished = perf_counter()
        failure = None
        retrieval_ms = (finished - retrieval_started) * 1000
    except ValueError as exc:
        # Programmer/system errors propagate; a rejected input/output is explicit data.
        _sync(translator.device if stage == "translation" else encoder.device)
        finished = perf_counter()
        if stage == "translation":
            translation_ms = (finished - started) * 1000
            retrieval_ms = 0.0
        else:
            retrieval_ms = (finished - retrieval_started) * 1000
        matches = []
        failure = {"stage": stage, "type": type(exc).__name__, "message": str(exc)}
    return {
        "ok": failure is None,
        "translation": english,
        "matches": matches,
        "failure": failure,
        "constraint_trace": (
            deepcopy(getattr(translator, "last_trace", None)) if mode != "zh" else None
        ),
        "timing": {
            "translation_ms": translation_ms,
            "retrieval_ms": retrieval_ms,
            "total_ms": (finished - started) * 1000,
        },
    }


def evaluate_attempts(rankings, qrels, gallery_ids):
    """Score failed intents as zero, while keeping strict complete successful ranks."""
    if not gallery_ids or len(set(gallery_ids)) != len(gallery_ids):
        raise ValueError("Expected a unique nonempty gallery")
    gallery = set(gallery_ids)
    for ranking in rankings.values():
        if ranking is not None and (len(ranking) != len(gallery_ids) or set(ranking) != gallery):
            raise ValueError("Successful rankings must contain the complete gallery")
    temporary = {i: gallery_ids if rows is None else rows for i, rows in rankings.items()}
    metrics = evaluate_rankings(temporary, qrels)
    failed = {i for i, rows in rankings.items() if rows is None}
    for intent in failed:
        metrics["per_query"][intent] = {k: 0.0 for k in metrics["aggregate"]}
    metrics["aggregate"] = {
        key: fsum(row[key] for row in metrics["per_query"].values()) / len(rankings)
        for key in metrics["aggregate"]
    }
    successful = [m for i, m in metrics["per_query"].items() if i not in failed]
    return {
        "metrics": metrics,
        "success_conditional_metrics": (
            {k: fsum(m[k] for m in successful) / len(successful) for k in metrics["aggregate"]}
            if successful
            else None
        ),
        "successful_intents": len(successful),
        "failed_intents": len(failed),
    }


def implementation_hash():
    return identity_fingerprint(
        {
            "fusion": fusion_hash(),
            **{
                name: source_sha256(name)
                for name in (
                    "translation.py",
                    "auto_benchmark.py",
                    "term_benchmark.py",
                    "opus_constraints.py",
                    "power_glossary_v1.json",
                )
            },
        }
    )


def load_inputs(args):
    metadata = Path(args.metadata)
    data = load_evaluation_data(
        metadata / "manifest.json",
        metadata / "queries.json",
        metadata / "qrels.json",
        args.data_root,
    )
    protocol, protocol_hash = _json_snapshot(args.protocol)
    if protocol.get("implementation_sha256") != implementation_hash():
        raise ValueError("Executing implementation differs from predeclared implementation")
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
        "num_beams": {"original": 1, "beam": 4, "constrained": 4},
        "do_sample": False,
        "use_cache": True,
        "score_atol": 1e-5,
        "dataset_hashes": data["hashes"],
        "glossary_sha256": file_sha256(GLOSSARY_PATH),
    }
    if any(protocol.get(key) != value for key, value in fixed.items()):
        raise ValueError("Protocol differs from fixed data, glossary, decoding or policies")
    for key, minimum in (("query_warmup", 0), ("query_repeats", 1)):
        if type(protocol.get(key)) is not int or protocol[key] < minimum:
            raise ValueError("Invalid warmup/repeat protocol")
    if set(protocol.get("supporting_hashes", {})) != {
        "annotation-notes.json",
        "review-rules.json",
        "glossary.json",
        "label-review.json",
    }:
        raise ValueError("Protocol must include annotation, review and glossary hashes")
    for filename, digest in protocol["supporting_hashes"].items():
        if file_sha256(metadata / filename) != digest:
            raise ValueError(f"Changed supporting metadata: {filename}")
    if file_sha256(metadata / "glossary.json") != fixed["glossary_sha256"]:
        raise ValueError("Benchmark glossary differs from packaged glossary")
    for directory, frozen in (
        (args.translation_dir, protocol["translator_files"]),
        (args.decoder_dir, protocol["decoder_files"]),
    ):
        if translator_snapshot(directory) != frozen:
            raise ValueError("Translator/decoder file inventory differs from protocol")
    started = perf_counter()
    index_hash = file_sha256(args.index)
    index = ImageIndex.load(args.index)
    if (
        file_sha256(args.index) != index_hash
        or index_hash != protocol["source_indexes"][args.model]
    ):
        raise ValueError("Source index differs from frozen protocol")
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
        raise ValueError("Index uses the wrong pinned retrieval model")
    pairs = {}
    for query in data["queries"]:
        pairs.setdefault(query["intent_id"], {})[query["language"]] = query
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

    base = LocalTranslator(args.translation_dir, device=protocol["translator_device"])
    return {
        "original": base,
        "beam": OPUSConstraintTranslator(base, constrained=False, decoder_dir=args.decoder_dir),
        "constrained": OPUSConstraintTranslator(base, decoder_dir=args.decoder_dir),
    }


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
    identities = {variant: translator.identity for variant, translator in translators.items()}
    for variant, translator in translators.items():
        identity = identities[variant]
        expected_generation = {
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
            or any(identity["generation"].get(k) != v for k, v in expected_generation.items())
        ):
            raise ValueError("Translator identity, generation or device differs from protocol")
    if str(encoder.device) != protocol["device"]:
        raise ValueError("Retrieval device differs from protocol")
    startup_seconds = perf_counter() - startup
    by_path = {r["path"]: r["image_id"] for r in data["images"]}
    relevance = {i: data["qrels"][p["zh"]["query_id"]] for i, p in pairs.items()}
    translations = {variant: {} for variant in VARIANTS}
    traces = {variant: {} for variant in VARIANTS}
    intents = list(pairs)

    def retrieve(policy, intent):
        recipe = POLICIES[policy]
        variant = recipe["decoder"]
        translator = translators[variant] if variant is not None else None
        result = timed_attempt(
            index,
            encoder,
            translator,
            pairs[intent]["zh"]["text"],
            recipe["mode"],
            top_k=len(data["images"]),
        )
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
        if variant is not None:
            english = result["translation"]
            if english is None and result["constraint_trace"]:
                english = result["constraint_trace"].get("output")
            if intent in translations[variant] and translations[variant][intent] != english:
                raise ValueError("Repeated translation changed; discard run")
            translations[variant][intent] = english
            trace = result["constraint_trace"]
            if intent in traces[variant] and traces[variant][intent] != trace:
                raise ValueError("Repeated constraint trace changed; discard run")
            traces[variant][intent] = trace
        return result

    # These are the first calls per decoder in this process, not independent cold starts.
    first_requests = {v: retrieve(f"{v}_en", intents[0]) for v in VARIANTS}
    scored = {policy: {} for policy in POLICIES}
    for policy in POLICIES:
        for j in range(protocol["query_warmup"]):
            retrieve(policy, intents[j % len(intents)])
        for intent in intents:
            scored[policy][intent] = retrieve(policy, intent)
    samples = {policy: {i: [] for i in intents} for policy in POLICIES}
    drift = {policy: 0.0 for policy in POLICIES}
    order = list(POLICIES)
    for round_no in range(protocol["query_repeats"]):
        for j, intent in enumerate(intents):
            offset = (round_no + j) % len(order)
            for policy in order[offset:] + order[:offset]:
                actual = retrieve(policy, intent)
                expected = scored[policy][intent]
                if actual["ok"] != expected["ok"] or actual["failure"] != expected["failure"]:
                    raise ValueError("Repeated request status changed; discard run")
                if actual["ok"]:
                    drift[policy] = max(
                        drift[policy],
                        check_ranking_and_scores(actual["rankings"], expected["rankings"]),
                    )
                samples[policy][intent].append(
                    {**actual["timing"], "ok": actual["ok"], "failure": actual["failure"]}
                )
        print(f"Completed warm round {round_no + 1}/{protocol['query_repeats']}", flush=True)
    strategies = {}
    for policy in POLICIES:
        rankings = {i: r["rankings"] for i, r in scored[policy].items()}
        ids = {
            i: [r["image_id"] for r in rows] if rows is not None else None
            for i, rows in rankings.items()
        }
        warm = [sample for values in samples[policy].values() for sample in values]
        strategies[policy] = {
            **evaluate_attempts(ids, relevance, data["image_ids"]),
            "rankings": rankings,
            "failures": {i: r["failure"] for i, r in scored[policy].items() if not r["ok"]},
            "warm_samples": samples[policy],
            "failed_warm_requests": sum(not s["ok"] for s in warm),
            "timing": {
                k: _stats([s[k] for s in warm])
                for k in ("translation_ms", "retrieval_ms", "total_ms")
            },
            "max_repeated_absolute_score_difference": drift[policy],
            "rank_disagreements": 0,
        }
    changed = (
        implementation_hash() != code_hash
        or file_sha256(args.protocol) != protocol_hash
        or file_sha256(args.index) != index_hash
        or any(
            file_sha256(Path(args.metadata) / f"{n}.json") != h for n, h in data["hashes"].items()
        )
        or any(
            file_sha256(Path(args.metadata) / n) != h
            for n, h in protocol["supporting_hashes"].items()
        )
        or any(
            file_sha256(path) != row["sha256"] for path, row in zip(data["paths"], data["images"])
        )
    )
    if changed or {v: t.identity for v, t in translators.items()} != identities:
        raise ValueError("Frozen inputs, implementation or translator configuration changed")
    check_translator_files(args.translation_dir, protocol["translator_files"])
    check_translator_files(args.decoder_dir, protocol["decoder_files"])
    payload = {
        "schema_version": 1,
        "experiment_id": protocol["experiment_id"],
        "model_alias": args.model,
        "model_identity": encoder.identity,
        "translator_identities": identities,
        "translations": translations,
        "constraint_traces": traces,
        "environment": _environment(encoder.device),
        "dataset_hashes": data["hashes"],
        "protocol": protocol,
        "protocol_sha256": protocol_hash,
        "implementation_sha256": code_hash,
        "source_index_sha256": index_hash,
        "gallery_images": len(data["images"]),
        "independent_intents": len(intents),
        "strategies": strategies,
        "training_executed": False,
        "startup": {
            "index_load_seconds": index_seconds,
            "retrieval_model_load_seconds": encoder_seconds,
            "shared_translator_and_decoder_load_seconds": translator_seconds,
            "validation_and_models_seconds": startup_seconds,
            "first_requests_per_decoder": first_requests,
            "first_request_order": list(VARIANTS),
        },
    }
    _save_new(output, payload)
    print(
        {
            p: {
                "metrics": s["metrics"]["aggregate"],
                "failed_intents": s["failed_intents"],
                "total_ms": s["timing"]["total_ms"],
            }
            for p, s in strategies.items()
        },
        flush=True,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--model", choices=list(MODEL_REVISIONS), required=True)
    for name in ("index", "output"):
        run.add_argument(f"--{name}", type=Path, required=True)
    for name, default in (
        ("metadata", "benchmarks/terminology-constraints-v1"),
        ("data-root", "data/equipment-v1"),
        ("protocol", "benchmarks/terminology-constraints-v1/protocol.json"),
        ("translation-dir", "models/opus-mt-zh-en"),
        ("decoder-dir", "models/opus-constrained-decoder"),
    ):
        run.add_argument(f"--{name}", type=Path, default=Path(default))
    compare = commands.add_parser("compare")
    for name in ("base", "quality", "output"):
        compare.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        run_benchmark(args)
    else:
        _save_new(
            args.output,
            compare_results(_json_snapshot(args.base)[0], _json_snapshot(args.quality)[0]),
        )
    return 0


def compare_results(base, quality):
    for alias, result in (("base", base), ("quality", quality)):
        identity = result["model_identity"]
        if (
            result["model_alias"] != alias
            or (identity["model_id"], identity["revision"]) != MODEL_REVISIONS[alias]
        ):
            raise ValueError("Expected pinned Base and So400m in named slots")
        if result["source_index_sha256"] != result["protocol"]["source_indexes"].get(alias):
            raise ValueError("Result index differs from its predeclared model slot")
        if result["implementation_sha256"] != result["protocol"]["implementation_sha256"]:
            raise ValueError("Result implementation differs from its predeclared protocol")
    for key in (
        "dataset_hashes",
        "protocol",
        "protocol_sha256",
        "implementation_sha256",
        "environment",
        "translator_identities",
        "translations",
        "constraint_traces",
    ):
        if base[key] != quality[key]:
            raise ValueError(f"Comparison requires the same {key}")
    if base["model_identity"]["runtime_versions"] != quality["model_identity"]["runtime_versions"]:
        raise ValueError("Comparison requires the same retrieval runtime")

    def delta(left, right):
        return {key: left[key] - value for key, value in right.items()}

    within = {}
    for alias, result in (("base", base), ("quality", quality)):
        strategies = result["strategies"]
        within[alias] = {}
        for new, old in (
            ("constrained", "original"),
            ("constrained", "beam"),
            ("beam", "original"),
        ):
            within[alias][f"{new}_minus_{old}"] = {
                suffix: delta(
                    strategies[f"{new}_{suffix}"]["metrics"]["aggregate"],
                    strategies[f"{old}_{suffix}"]["metrics"]["aggregate"],
                )
                for suffix in ("en", "fusion")
            }
            within[alias][f"{new}_to_{old}_p50_ratio"] = {
                suffix: strategies[f"{new}_{suffix}"]["timing"]["total_ms"]["p50"]
                / strategies[f"{old}_{suffix}"]["timing"]["total_ms"]["p50"]
                for suffix in ("en", "fusion")
            }
    return {
        "quality_minus_base": {
            policy: delta(
                quality["strategies"][policy]["metrics"]["aggregate"],
                base["strategies"][policy]["metrics"]["aggregate"],
            )
            for policy in POLICIES
        },
        "quality_to_base_total_p50_ratio": {
            policy: quality["strategies"][policy]["timing"]["total_ms"]["p50"]
            / base["strategies"][policy]["timing"]["total_ms"]["p50"]
            for policy in POLICIES
        },
        "within_model": within,
    }


if __name__ == "__main__":
    raise SystemExit(main())
