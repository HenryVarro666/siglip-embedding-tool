"""Export validated fallback measurements into reviewable Git metadata."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from siglip_tools.benchmark import _json_snapshot
from siglip_tools.fallback_benchmark import compare_results, validate_frozen_inputs
from siglip_tools.fusion_benchmark import _save_new


def build_exports(base, quality, metadata, protocol_path=None):
    protocol, digest = validate_frozen_inputs(
        metadata, protocol_path or Path(metadata) / "protocol.json"
    )
    for result in (base, quality):
        if result["protocol"] != protocol or result["protocol_sha256"] != digest:
            raise ValueError("Result does not match the frozen protocol")
    comparison = compare_results(base, quality)
    summary = {
        "experiment_id": protocol["experiment_id"],
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_git_commit": protocol["implementation_git_commit"],
        "implementation_sha256": protocol["implementation_sha256"],
        "protocol_sha256": digest,
        "dataset_hashes": protocol["dataset_hashes"],
        "evaluation_inputs_sha256": protocol["evaluation_inputs_sha256"],
        "training_executed": False,
        "models": {},
        "comparison": comparison,
        "scope": (
            "Same previously diagnosed16queries/40images; assistant labels, no new holdout; "
            "fresh live attempts, no training or translation cache."
        ),
    }
    requests = {
        "experiment_id": protocol["experiment_id"],
        "protocol_sha256": digest,
        "evaluation_inputs": deepcopy(base["evaluation_inputs"]),
        "models": {},
    }
    for alias, result in (("base", base), ("quality", quality)):
        summary["models"][alias] = {
            key: deepcopy(result[key])
            for key in (
                "model_identity",
                "translator_identities",
                "environment",
                "startup",
                "source_index_sha256",
                "branch_audit",
                "gallery_images",
                "independent_intents",
            )
        }
        summary["models"][alias]["strategies"] = {
            policy: {
                k: deepcopy(v) for k, v in strategy.items() if k not in {"attempts", "warm_samples"}
            }
            for policy, strategy in result["strategies"].items()
        }
        requests["models"][alias] = {
            policy: {key: deepcopy(strategy[key]) for key in ("attempts", "warm_samples")}
            for policy, strategy in result["strategies"].items()
        }
    return summary, requests


def export_results(base_path, quality_path, output, metadata, protocol_path=None):
    output = Path(output)
    targets = [output / n for n in ("results-summary.json", "requests.json")]
    if any(p.exists() for p in targets):
        raise FileExistsError("Export exists; use a new output directory")
    base, base_hash = _json_snapshot(base_path)
    quality, quality_hash = _json_snapshot(quality_path)
    summary, requests = build_exports(base, quality, metadata, protocol_path)
    summary["source_results"] = {
        "base": {"path": str(base_path), "sha256": base_hash},
        "quality": {"path": str(quality_path), "sha256": quality_hash},
    }
    for target, value in zip(targets, (summary, requests)):
        _save_new(target, value)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "quality", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=Path("benchmarks/constraint-fallback-v1"))
    args = parser.parse_args(argv)
    export_results(args.base, args.quality, args.output, args.metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
