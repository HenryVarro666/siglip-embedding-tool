"""Frozen Chinese/English/equal-cosine retrieval comparison, without training."""

import argparse
import json
import os
import tempfile
from collections import defaultdict
from math import isfinite
from pathlib import Path
from time import perf_counter

from .benchmark import (
    MODEL_REVISIONS,
    _environment,
    _json_snapshot,
    _stats,
    _sync,
    load_evaluation_data,
)
from .benchmark import (
    _implementation_hash as baseline_implementation_hash,
)
from .encoder import SigLIPEmbedder, identity_fingerprint
from .evaluation import evaluate_rankings
from .fusion import search_texts
from .images import file_sha256
from .index import ImageIndex

STRATEGIES = {
    "zh": {"languages": ["zh"], "weights": [1.0]},
    "en": {"languages": ["en"], "weights": [1.0]},
    "fusion": {"languages": ["zh", "en"], "weights": [0.5, 0.5]},
}
SCORE_ATOL = 1e-5


def check_ranking_and_scores(actual, expected):
    """Require identical order and float32-scale score agreement."""
    if [row["image_id"] for row in actual] != [row["image_id"] for row in expected]:
        raise ValueError("Repeated or historical ranking changed")
    maximum = max(abs(a["score"] - b["score"]) for a, b in zip(actual, expected))
    if not isfinite(maximum) or maximum > SCORE_ATOL:
        raise ValueError("Repeated or historical score drift exceeds frozen tolerance")
    return maximum


def implementation_hash():
    root = Path(__file__).parent
    return identity_fingerprint(
        {
            "baseline": baseline_implementation_hash(),
            **{name: file_sha256(root / name) for name in ("fusion.py", "fusion_benchmark.py")},
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
    if (
        protocol["dataset_hashes"] != data["hashes"]
        or protocol["dataset_id"] != data["manifest"]["dataset_id"]
        or protocol["strategies"] != STRATEGIES
        or protocol["query_batch_sizes"] != {"zh": 1, "en": 1, "fusion": 2}
        or protocol["dtype"] != "float32"
        or protocol["training"] is not False
        or protocol["weight_tuning"] is not False
        or protocol.get("score_atol", SCORE_ATOL) != SCORE_ATOL
    ):
        raise ValueError("Protocol must match frozen data and fixed equal-cosine strategies")
    for name, minimum in (("query_warmup", 0), ("query_repeats", 1)):
        if type(protocol[name]) is not int or protocol[name] < minimum:
            raise ValueError(f"Invalid protocol {name}")
    started = perf_counter()
    index_hash = file_sha256(args.index)
    index = ImageIndex.load(args.index)
    if file_sha256(args.index) != index_hash:
        raise ValueError("Index changed while loading")
    index_seconds = perf_counter() - started
    expected = {row["path"]: row["sha256"] for row in data["images"]}
    if (
        {record.path: record.sha256 for record in index.records} != expected
        or len(index.records) != len(expected)
        or any(record.aliases for record in index.records)
    ):
        raise ValueError("Index records must exactly match the frozen gallery")
    mid, revision = MODEL_REVISIONS[args.model]
    if (index.model_identity["model_id"], index.model_identity["revision"]) != (mid, revision):
        raise ValueError("Index must use the requested pinned checkpoint")
    pairs = defaultdict(dict)
    for query in data["queries"]:
        pairs[query["intent_id"]][query["language"]] = query
    return (
        data,
        protocol,
        protocol_hash,
        index,
        index_hash,
        index_seconds,
        dict(sorted(pairs.items())),
    )


def _save_new(path, payload):
    content = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content.encode())
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _check_reference(path, payload, pairs):
    if path is None:
        return None
    reference, digest = _json_snapshot(path)
    if (
        reference["model_identity"] != payload["model_identity"]
        or reference["dataset_hashes"] != payload["dataset_hashes"]
        or reference["implementation_sha256"] != baseline_implementation_hash()
    ):
        raise ValueError("Historical reference identity, data or baseline implementation differs")
    max_error = 0.0
    count = 0
    for language in ("zh", "en"):
        for intent, pair in pairs.items():
            old = reference["rankings"][pair[language]["query_id"]]
            new = payload["strategies"][language]["rankings"][intent]
            max_error = max(max_error, check_ranking_and_scores(new, old))
            count += 1
    return {"sha256": digest, "rankings_matched": count, "max_absolute_score_difference": max_error}


def run_benchmark(args):
    output = Path(args.output) / "result.json"
    if output.exists():
        raise FileExistsError("Result exists; use a new output directory")
    data, protocol, protocol_hash, index, index_hash, index_seconds, pairs = load_inputs(args)
    code_hash = implementation_hash()
    mid, revision = MODEL_REVISIONS[args.model]
    started = perf_counter()
    encoder = SigLIPEmbedder(mid, revision, protocol["device"], local_files_only=True)
    _sync(encoder.device)
    model_seconds = perf_counter() - started
    index.assert_compatible(encoder.identity)
    if str(encoder.device) != protocol["device"]:
        raise ValueError("Actual device differs from frozen protocol")
    by_path = {row["path"]: row["image_id"] for row in data["images"]}
    relevance = {intent: data["qrels"][pair["zh"]["query_id"]] for intent, pair in pairs.items()}
    intents = list(pairs)

    def retrieve(strategy, intent):
        recipe = STRATEGIES[strategy]
        texts = [pairs[intent][lang]["text"] for lang in recipe["languages"]]
        return search_texts(index, encoder, texts, recipe["weights"], top_k=len(data["images"]))

    scored = {strategy: {} for strategy in STRATEGIES}
    for strategy in STRATEGIES:
        for i in range(protocol["query_warmup"]):
            retrieve(strategy, intents[i % len(intents)])
        _sync(encoder.device)
        for intent in intents:
            scored[strategy][intent] = [
                {
                    "image_id": by_path[match.path],
                    "rank": match.rank,
                    "score": match.score,
                    "relevant": by_path[match.path] in relevance[intent],
                }
                for match in retrieve(strategy, intent)
            ]
    samples = {strategy: defaultdict(list) for strategy in STRATEGIES}
    max_score_errors = {strategy: 0.0 for strategy in STRATEGIES}
    names = list(STRATEGIES)
    for round_number in range(protocol["query_repeats"]):
        for position, intent in enumerate(intents):
            offset = (round_number + position) % len(names)
            for strategy in names[offset:] + names[:offset]:
                _sync(encoder.device)
                started = perf_counter()
                matches = retrieve(strategy, intent)
                _sync(encoder.device)
                samples[strategy][intent].append((perf_counter() - started) * 1000)
                repeated = [
                    {"image_id": by_path[match.path], "score": match.score} for match in matches
                ]
                max_score_errors[strategy] = max(
                    max_score_errors[strategy],
                    check_ranking_and_scores(repeated, scored[strategy][intent]),
                )
    diagnostics = {}
    for intent, pair in pairs.items():
        texts = [pair[lang]["text"] for lang in ("zh", "en")]
        vectors = encoder.encode_texts(texts, batch_size=2)
        tokens = encoder.processor.tokenizer(texts, padding=False, truncation=False)["input_ids"]
        diagnostics[intent] = {
            "query_ids": {lang: pair[lang]["query_id"] for lang in ("zh", "en")},
            "token_counts": {lang: len(ids) for lang, ids in zip(("zh", "en"), tokens)},
            "text_query_cosine": float(vectors[0] @ vectors[1]),
        }
    strategies = {}
    for strategy in STRATEGIES:
        rankings = {
            intent: [row["image_id"] for row in rows] for intent, rows in scored[strategy].items()
        }
        strategies[strategy] = {
            "metrics": evaluate_rankings(rankings, relevance),
            "rankings": scored[strategy],
            "timing": _stats(
                [sample for values in samples[strategy].values() for sample in values]
            ),
            "warm_samples_ms": dict(samples[strategy]),
            "rank_disagreements": 0,
            "max_repeated_absolute_score_difference": max_score_errors[strategy],
        }
    payload = {
        "schema_version": 1,
        "experiment_id": protocol["experiment_id"],
        "model_alias": args.model,
        "model_identity": encoder.identity,
        "environment": _environment(encoder.device),
        "dataset_hashes": data["hashes"],
        "dataset_id": data["manifest"]["dataset_id"],
        "protocol": protocol,
        "protocol_sha256": protocol_hash,
        "implementation_sha256": code_hash,
        "source_index_sha256": index_hash,
        "gallery_images": len(data["images"]),
        "independent_intents": len(pairs),
        "embedding_dimension": index.vectors.shape[1],
        "processor_image_size": dict(encoder.processor.image_processor.size),
        "timing": {"index_load_seconds": index_seconds, "cached_model_load_seconds": model_seconds},
        "strategies": strategies,
        "diagnostics": diagnostics,
        "training_executed": False,
    }
    payload["historical_reference"] = _check_reference(args.reference_result, payload, pairs)
    if (
        any(
            file_sha256(Path(args.metadata) / f"{name}.json") != digest
            for name, digest in data["hashes"].items()
        )
        or file_sha256(args.protocol) != protocol_hash
        or file_sha256(args.index) != index_hash
        or implementation_hash() != code_hash
        or any(
            file_sha256(path) != row["sha256"] for path, row in zip(data["paths"], data["images"])
        )
    ):
        raise ValueError("Frozen data, protocol, index or implementation changed during execution")
    if args.reference_result is not None:
        if file_sha256(args.reference_result) != payload["historical_reference"]["sha256"]:
            raise ValueError("Historical reference changed during validation")
    _save_new(output, payload)
    print(
        json.dumps(
            {
                strategy: {"metrics": value["metrics"]["aggregate"], "timing": value["timing"]}
                for strategy, value in strategies.items()
            },
            indent=2,
        )
    )


def compare_results(base, quality):
    for alias, result in (("base", base), ("quality", quality)):
        if (
            result["model_alias"] != alias
            or (result["model_identity"]["model_id"], result["model_identity"]["revision"])
            != MODEL_REVISIONS[alias]
        ):
            raise ValueError("Expected pinned Base/So400m in the named slots")
    for name in (
        "dataset_hashes",
        "protocol_sha256",
        "protocol",
        "implementation_sha256",
        "environment",
    ):
        if base[name] != quality[name]:
            raise ValueError(f"Comparison requires identical {name}")
    if base["model_identity"]["runtime_versions"] != quality["model_identity"]["runtime_versions"]:
        raise ValueError("Comparison requires identical runtime libraries")
    deltas, within, ratios = {}, {}, {}
    for strategy in STRATEGIES:
        b = base["strategies"][strategy]
        q = quality["strategies"][strategy]
        deltas[strategy] = {
            key: q["metrics"]["aggregate"][key] - value
            for key, value in b["metrics"]["aggregate"].items()
        }
        ratios[strategy] = q["timing"]["p50"] / b["timing"]["p50"]
    for alias, result in (("base", base), ("quality", quality)):
        policies = result["strategies"]
        within[alias] = {}
        for language in ("zh", "en"):
            fused = policies["fusion"]["metrics"]["aggregate"]
            single = policies[language]["metrics"]["aggregate"]
            within[alias][f"fusion_minus_{language}"] = {
                key: fused[key] - single[key] for key in fused
            }
            within[alias][f"fusion_to_{language}_query_p50_ratio"] = (
                policies["fusion"]["timing"]["p50"] / policies[language]["timing"]["p50"]
            )
    return {
        "quality_minus_base": deltas,
        "within_model": within,
        "quality_to_base_query_p50_ratio": ratios,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--model", choices=tuple(MODEL_REVISIONS), required=True)
    run.add_argument("--index", required=True, type=Path)
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--metadata", type=Path, default=Path("benchmarks/equipment-v1"))
    run.add_argument("--data-root", type=Path, default=Path("data/equipment-v1"))
    run.add_argument(
        "--protocol", type=Path, default=Path("benchmarks/bilingual-fusion-v1/protocol.json")
    )
    run.add_argument("--reference-result", type=Path)
    compare = sub.add_parser("compare")
    for name in ("base", "quality", "output"):
        compare.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        run_benchmark(args)
    else:
        result = compare_results(_json_snapshot(args.base)[0], _json_snapshot(args.quality)[0])
        _save_new(args.output, result)
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
