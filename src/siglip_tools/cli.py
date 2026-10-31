"""Command-line entry point."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from .encoder import DEFAULT_MODEL, QUALITY_MODEL, SigLIPEmbedder
from .images import file_sha256, load_rgb, pixel_sha256, scan_images
from .index import ImageIndex


def _positive(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local pretrained SigLIP image embeddings/retrieval"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser(
        "index", aliases=["embed-images"], help="Batch-encode a picture directory"
    )
    build.add_argument("--images", required=True, type=Path)
    build.add_argument("--output", required=True, type=Path)
    build.add_argument(
        "--model", default="base", help="base, quality, HF model ID, or local checkpoint"
    )
    build.add_argument("--revision", help="HF commit/tag to resolve and pin")
    build.add_argument("--batch-size", type=_positive, default=8)
    build.add_argument("--overwrite", action="store_true")
    build.add_argument("--device", default="auto")
    build.add_argument("--local-files-only", action="store_true")
    build.set_defaults(handler=_build)

    for command, query_flag in (("search-text", "--query"), ("search-image", "--image")):
        query = sub.add_parser(command, help="Query the index with its pinned checkpoint")
        query.add_argument("--index", required=True, type=Path)
        query.add_argument(query_flag, required=True)
        query.add_argument("--top-k", type=_positive, default=5)
        query.add_argument(
            "--image-root", type=Path, help="Relocate paths without rebuilding vectors"
        )
        query.add_argument("--device", default="auto")
        query.add_argument("--local-files-only", action="store_true")
        if command == "search-image":
            query.add_argument("--include-self", action="store_true")
        query.set_defaults(handler=_search)

    inspect = sub.add_parser("inspect", help="Inspect index metadata without downloading a model")
    inspect.add_argument("--index", required=True, type=Path)
    inspect.set_defaults(handler=_inspect)
    return parser


def _print(payload: dict) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))


def _build(args) -> None:
    started = perf_counter()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Index already exists: {args.output}; pass --overwrite to replace it"
        )
    records = scan_images(args.images)
    scanned = perf_counter()
    model_id = {"base": DEFAULT_MODEL, "quality": QUALITY_MODEL}.get(args.model, args.model)
    encoder = SigLIPEmbedder(model_id, args.revision, args.device, args.local_files_only)
    loaded = perf_counter()
    root = args.images.resolve()
    vectors = encoder.encode_images([root / record.path for record in records], args.batch_size)
    # Refuse to bind an embedding to stale file provenance if an input changed during encoding.
    for record in records:
        if file_sha256(root / record.path) != record.sha256:
            raise ValueError(f"Image changed during indexing: {record.path}; retry")
    encoded = perf_counter()
    index = ImageIndex(vectors, records, encoder.identity, root)
    index.save(args.output, overwrite=args.overwrite)
    _print(
        {
            "index": str(args.output.resolve()),
            "images": len(records),
            "duplicate_aliases": sum(len(r.aliases) for r in records),
            "dimension": vectors.shape[1],
            "device": str(encoder.device),
            "model_identity": encoder.identity,
            "fingerprint": index.fingerprint,
            "seconds": {
                "scan": scanned - started,
                "model_load": loaded - scanned,
                "encode": encoded - loaded,
                "total": perf_counter() - started,
            },
        }
    )


def _search(args) -> None:
    started = perf_counter()
    index = ImageIndex.load(args.index)
    identity = index.model_identity
    encoder = SigLIPEmbedder(
        identity["model_id"], identity["revision"], args.device, args.local_files_only
    )
    index.assert_compatible(encoder.identity)
    loaded = perf_counter()
    exclusion = None
    if args.command == "search-text":
        vector = encoder.encode_texts([args.query])[0]
    else:
        image = load_rgb(args.image)
        vector = encoder.encode_images([image])[0]
        if not args.include_self:
            exclusion = pixel_sha256(image)
    matches = index.search(vector, args.top_k, exclude_pixel_sha256=exclusion)
    image_root = args.image_root.resolve() if args.image_root else index.image_root
    _print(
        {
            "query_type": args.command,
            "index": str(args.index.resolve()),
            "fingerprint": index.fingerprint,
            "device": str(encoder.device),
            "results": [
                {**asdict(match), "full_path": str(image_root / match.path)} for match in matches
            ],
            "seconds": {
                "load": loaded - started,
                "query": perf_counter() - loaded,
                "total": perf_counter() - started,
            },
        }
    )


def _inspect(args) -> None:
    index = ImageIndex.load(args.index)
    _print(
        {
            "images": len(index.records),
            "dimension": index.vectors.shape[1],
            "duplicate_aliases": sum(len(r.aliases) for r in index.records),
            "image_root": str(index.image_root),
            "model_identity": index.model_identity,
            "fingerprint": index.fingerprint,
        }
    )


def main(argv=None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        args.handler(args)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0
