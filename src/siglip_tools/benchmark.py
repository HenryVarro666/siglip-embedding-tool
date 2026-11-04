"""Frozen, reproducible comparison of pretrained SigLIP retrieval configurations."""

import argparse
import json
import os
import platform
import subprocess
import sys
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .commons import validate_source_record
from .encoder import DEFAULT_MODEL, QUALITY_MODEL, SigLIPEmbedder, identity_fingerprint
from .evaluation import evaluate_rankings
from .frozen_source import source_sha256
from .images import file_sha256, scan_images
from .index import ImageIndex

MODEL_REVISIONS = {
    "base": (DEFAULT_MODEL, "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"),
    "quality": (QUALITY_MODEL, "e8e487298228002f3d8a82e0cd5c8ea9c567f57f"),
}


def _json(path):
    return _json_snapshot(path)[0]


def _json_snapshot(path):
    content = Path(path).read_bytes()
    return json.loads(content), sha256(content).hexdigest()


def load_evaluation_data(manifest_path, queries_path, qrels_path, data_root):
    root = Path(data_root).resolve()
    manifest, manifest_hash = _json_snapshot(manifest_path)
    query_document, query_hash = _json_snapshot(queries_path)
    relevance_document, relevance_hash = _json_snapshot(qrels_path)
    images = manifest["images"]
    queries = query_document["queries"]
    qrels = relevance_document["qrels"]
    image_ids = [item["image_id"] for item in images]
    query_ids = [query["query_id"] for query in queries]
    if not image_ids or len(set(image_ids)) != len(image_ids):
        raise ValueError("Gallery IDs must be nonempty and unique")
    if not query_ids or len(set(query_ids)) != len(query_ids) or set(query_ids) != set(qrels):
        raise ValueError("Query IDs and relevance IDs must match and be unique")
    paths = []
    for item in images:
        validate_source_record(item)
        path = (root / item["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Image path escapes data root")
        if file_sha256(path) != item["sha256"]:
            raise ValueError(f"Image hash mismatch: {item['image_id']}")
        paths.append(path)
    if len(set(paths)) != len(paths):
        raise ValueError("Gallery paths must be unique")
    by_intent = defaultdict(dict)
    for query in queries:
        if query["language"] not in {"en", "zh"} or not query["text"].strip():
            raise ValueError("Queries require nonempty English or Chinese text")
        language = query["language"]
        intent = query["intent_id"]
        if language in by_intent[intent]:
            raise ValueError("Each intent requires exactly one English/Chinese pair")
        by_intent[intent][language] = query["query_id"]
    # This independent evaluator validates every positive and the fixed-gallery contract.
    evaluate_rankings({query_id: image_ids for query_id in query_ids}, qrels)
    for pair in by_intent.values():
        if set(pair) != {"en", "zh"} or set(qrels[pair["en"]]) != set(qrels[pair["zh"]]):
            raise ValueError("Language pairs must have the same relevance labels")
    return {
        "manifest": manifest,
        "images": images,
        "image_ids": image_ids,
        "paths": paths,
        "queries": queries,
        "qrels": qrels,
        "root": root,
        "hashes": {"manifest": manifest_hash, "queries": query_hash, "qrels": relevance_hash},
    }


def compare_results(base, quality):
    if (
        base["dataset_hashes"] != quality["dataset_hashes"]
        or base["protocol"] != quality["protocol"]
    ):
        raise ValueError("Comparisons require the same frozen dataset and timing protocol/device")
    for alias, result in (("base", base), ("quality", quality)):
        expected_id, expected_revision = MODEL_REVISIONS[alias]
        identity = result["model_identity"]
        if (
            result["model_alias"] != alias
            or identity["model_id"] != expected_id
            or identity["revision"] != expected_revision
        ):
            raise ValueError("Expected the pinned Base and So400m results in their named slots")
    if (
        base["environment"] != quality["environment"]
        or base["implementation_sha256"] != quality["implementation_sha256"]
        or base["model_identity"]["runtime_versions"]
        != quality["model_identity"]["runtime_versions"]
    ):
        raise ValueError("Comparisons require the same hardware, runtime and implementation")
    first = base["metrics"]["aggregate"]
    second = quality["metrics"]["aggregate"]
    if first.keys() != second.keys():
        raise ValueError("Comparisons require the same metric definitions")
    return {
        "quality_minus_base": {key: second[key] - first[key] for key in first},
        "quality_to_base_query_p50_ratio": (
            quality["timing"]["query_warm_ms"]["p50"] / base["timing"]["query_warm_ms"]["p50"]
        ),
        "quality_to_base_gallery_time_ratio": (
            quality["timing"]["gallery_index_seconds"] / base["timing"]["gallery_index_seconds"]
        ),
    }


def _sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _stats(samples):
    values = np.asarray(samples, dtype=np.float64)
    return {
        "count": len(samples),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "mean": float(values.mean()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _implementation_hash():
    return identity_fingerprint(
        {
            name: source_sha256(name)
            for name in (
                "benchmark.py",
                "evaluation.py",
                "commons.py",
                "encoder.py",
                "index.py",
                "images.py",
            )
        }
    )


def _environment(device):
    result = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
    }
    if sys.platform == "darwin":
        for field, key in (
            ("cpu_model", "machdep.cpu.brand_string"),
            ("physical_memory_bytes", "hw.memsize"),
        ):
            result[field] = subprocess.check_output(["sysctl", "-n", key], text=True).strip()
    else:
        result["cpu_model"] = platform.processor()
        if Path("/proc/cpuinfo").is_file():
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    result["cpu_model"] = line.split(":", 1)[1].strip()
                    break
        if hasattr(os, "sysconf"):
            result["physical_memory_bytes"] = str(
                os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            )
    if device.type == "cuda":
        result["cuda_device"] = torch.cuda.get_device_name(device)
    return result


def run_benchmark(args):
    if args.batch_size < 1 or args.query_repeats < 1 or args.query_warmup < 0:
        raise ValueError("Batch size/repeats must be positive; warmup must be nonnegative")
    output = Path(args.output)
    if (output / "result.json").exists() and not args.overwrite:
        raise FileExistsError("Result exists; use a new output directory or --overwrite")
    data = load_evaluation_data(args.manifest, args.queries, args.qrels, args.data_root)
    records = scan_images(data["root"])
    declared = {item["path"] for item in data["images"]}
    if {record.path for record in records} != declared or any(record.aliases for record in records):
        raise ValueError("Data root contains extra images, missing images, or pixel duplicates")
    manifest_hashes = {item["path"]: item["sha256"] for item in data["images"]}
    if any(record.sha256 != manifest_hashes[record.path] for record in records):
        raise ValueError("Image hash changed between frozen validation and scan")
    implementation_hash = _implementation_hash()
    model_id, revision = MODEL_REVISIONS[args.model]
    started = perf_counter()
    # Network/cache preparation is a separate command and is never timed as inference.
    encoder = SigLIPEmbedder(model_id, revision, args.device, local_files_only=True)
    _sync(encoder.device)
    load_seconds = perf_counter() - started
    first_paths = data["paths"][: args.batch_size]
    _sync(encoder.device)
    started = perf_counter()
    encoder.encode_images(first_paths, batch_size=args.batch_size)
    _sync(encoder.device)
    first_batch_seconds = perf_counter() - started
    # One extra image warmup after the measured first execution.
    encoder.encode_images(first_paths, batch_size=args.batch_size)
    _sync(encoder.device)
    started = perf_counter()
    index = ImageIndex.from_directory(
        encoder, data["root"], batch_size=args.batch_size, records=records
    )
    output.mkdir(parents=True, exist_ok=True)
    index.save(output / "index.npz", overwrite=args.overwrite)
    _sync(encoder.device)
    gallery_seconds = perf_counter() - started
    by_path = {item["path"]: item["image_id"] for item in data["images"]}
    for i in range(args.query_warmup):
        encoder.encode_texts(data["queries"][i % len(data["queries"])]["text"])
    _sync(encoder.device)
    rankings, scored, text_vectors = {}, {}, {}
    first_query_samples = []
    for query in data["queries"]:
        started = perf_counter()
        vector = encoder.encode_texts(query["text"])[0]
        matches = index.search(vector, top_k=len(data["images"]))
        _sync(encoder.device)
        first_query_samples.append((perf_counter() - started) * 1000)
        query_id = query["query_id"]
        text_vectors[query_id] = vector
        rankings[query_id] = [by_path[match.path] for match in matches]
        scored[query_id] = [
            {
                "image_id": by_path[match.path],
                "score": match.score,
                "rank": match.rank,
                "relevant": by_path[match.path] in data["qrels"][query_id],
            }
            for match in matches
        ]
    warm_samples = defaultdict(list)
    for _ in range(args.query_repeats):
        for query in data["queries"]:
            _sync(encoder.device)
            started = perf_counter()
            index.search(encoder.encode_texts(query["text"])[0], top_k=len(data["images"]))
            _sync(encoder.device)
            warm_samples[query["query_id"]].append((perf_counter() - started) * 1000)
    search_samples = []
    for _ in range(args.query_repeats):
        for vector in text_vectors.values():
            started = perf_counter()
            index.search(vector, top_k=len(data["images"]))
            search_samples.append((perf_counter() - started) * 1000)
    metrics = evaluate_rankings(rankings, data["qrels"])
    by_language = {}
    for language in ("en", "zh"):
        ids = [q["query_id"] for q in data["queries"] if q["language"] == language]
        by_language[language] = evaluate_rankings(
            {key: rankings[key] for key in ids}, {key: data["qrels"][key] for key in ids}
        )["aggregate"]
    current_hashes = {
        name: file_sha256(path)
        for name, path in (
            ("manifest", args.manifest),
            ("queries", args.queries),
            ("qrels", args.qrels),
        )
    }
    if current_hashes != data["hashes"]:
        raise ValueError("Frozen metadata changed during evaluation; discard this run")
    if any(
        file_sha256(path) != item["sha256"] for path, item in zip(data["paths"], data["images"])
    ):
        raise ValueError("Frozen image content changed during evaluation; discard this run")
    if _implementation_hash() != implementation_hash:
        raise ValueError("Benchmark implementation changed during evaluation; discard this run")
    payload = {
        "schema_version": 1,
        "model_alias": args.model,
        "model_identity": encoder.identity,
        "dataset_hashes": data["hashes"],
        "dataset_id": data["manifest"]["dataset_id"],
        "gallery_images": len(data["images"]),
        "queries": len(data["queries"]),
        "independent_intents": len({q["intent_id"] for q in data["queries"]}),
        "embedding_dimension": index.vectors.shape[1],
        "processor_image_size": dict(encoder.processor.image_processor.size),
        "protocol": {
            "device": str(encoder.device),
            "dtype": "float32",
            "batch_size": args.batch_size,
            "query_batch_size": 1,
            "query_warmup": args.query_warmup,
            "query_repeats": args.query_repeats,
            "image_extra_warmup": 1,
            "download_excluded": True,
        },
        "environment": _environment(encoder.device),
        "implementation_sha256": implementation_hash,
        "timing": {
            "cached_model_load_seconds": load_seconds,
            "first_image_batch_seconds": first_batch_seconds,
            "first_image_batch_size": len(first_paths),
            "gallery_index_seconds": gallery_seconds,
            "gallery_images_per_second": len(data["images"]) / gallery_seconds,
            "first_query_pass_ms": _stats(first_query_samples),
            "query_warm_ms": _stats([v for values in warm_samples.values() for v in values]),
            "query_warm_samples_ms": dict(warm_samples),
            "cached_vector_search_ms": _stats(search_samples),
        },
        "metrics": metrics,
        "metrics_by_language": by_language,
        "rankings": scored,
        "training_executed": False,
    }
    (output / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "model": args.model,
                "metrics": metrics["aggregate"],
                "query_warm_ms": payload["timing"]["query_warm_ms"],
                "gallery_index_seconds": gallery_seconds,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="Cache one pinned model before timed inference")
    prepare.add_argument("--model", choices=tuple(MODEL_REVISIONS), required=True)
    run = sub.add_parser("run", help="Run one pretrained model on a frozen evaluation set")
    for name in ("manifest", "queries", "qrels", "data-root", "output"):
        run.add_argument(f"--{name}", required=True, type=Path)
    run.add_argument("--model", choices=tuple(MODEL_REVISIONS), required=True)
    run.add_argument("--device", default="auto")
    run.add_argument("--batch-size", type=int, default=4)
    run.add_argument("--query-warmup", type=int, default=3)
    run.add_argument("--query-repeats", type=int, default=10)
    run.add_argument("--overwrite", action="store_true")
    compare = sub.add_parser("compare", help="Compare results with identical frozen inputs")
    compare.add_argument("--base", required=True, type=Path)
    compare.add_argument("--quality", required=True, type=Path)
    compare.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        from huggingface_hub import snapshot_download

        model_id, revision = MODEL_REVISIONS[args.model]
        cached = snapshot_download(
            model_id,
            revision=revision,
            max_workers=2,
            allow_patterns=[
                "config.json",
                "preprocessor_config.json",
                "processor_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "tokenizer.model",
                "spiece.model",
                "model.safetensors",
                "model.safetensors.index.json",
                "model-*.safetensors",
                "chat_template.jinja",
            ],
        )
        print(cached)
    elif args.command == "run":
        run_benchmark(args)
    else:
        result = compare_results(_json(args.base), _json(args.quality))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
