"""Export frozen four-policy completed-candidate-protection measurements for repository review."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from siglip_tools.benchmark import _json_snapshot
from siglip_tools.fusion_benchmark import _save_new


def build_exports(base, quality, metadata, protocol_path=None):
    from siglip_tools.protected_benchmark import compare_results, validate_frozen_inputs

    protocol, digest = validate_frozen_inputs(
        metadata, protocol_path or Path(metadata) / "protocol.json"
    )
    if any(
        result["protocol"] != protocol or result["protocol_sha256"] != digest
        for result in (base, quality)
    ):
        raise ValueError("Results differ from the committed frozen protocol")
    comparison = compare_results(base, quality)
    summary = {
        "experiment_id": protocol["experiment_id"],
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_git_commit": protocol["implementation_git_commit"],
        "implementation_sha256": protocol["implementation_sha256"],
        "protocol_sha256": digest,
        "dataset_hashes": protocol["dataset_hashes"],
        "development_hashes": protocol["development_hashes"],
        "repetition_profile": base["repetition_profile"],
        "protection_rule": base["protection_rule"],
        "protection_rule_sha256": protocol["protection_rule_sha256"],
        "repetition_profile_sha256": protocol["repetition_profile_sha256"],
        "training_executed": False,
        "models": {},
        "comparison": comparison,
        "scope": (
            "Fixed protection validated on24declared development controls, then the same "
            "alreadydiagnosed16queries/40gallery; "
            "no new independentholdout or training. Actual live constraint attempt and "
            "freshfallback included in each complete timing."
        ),
    }
    requests = {
        "experiment_id": protocol["experiment_id"],
        "protocol_sha256": digest,
        "evaluation_inputs": deepcopy(base["evaluation_inputs"]),
        "models": {},
    }
    for alias, result in [("base", base), ("quality", quality)]:
        summary["models"][alias] = {
            key: deepcopy(result[key])
            for key in (
                "model_identity",
                "translator_identities",
                "environment",
                "startup",
                "source_index_sha256",
                "branch_audit",
                "early_stop_analysis",
                "protection_analysis",
                "gallery_images",
                "independent_intents",
            )
        }
        summary["models"][alias]["strategies"] = {
            p: {k: deepcopy(v) for k, v in st.items() if k not in {"attempts", "warm_samples"}}
            for p, st in result["strategies"].items()
        }
        requests["models"][alias] = {
            p: {k: deepcopy(st[k]) for k in ("attempts", "warm_samples")}
            for p, st in result["strategies"].items()
        }
    return summary, requests


def export_results(base_path, quality_path, output, metadata, protocol_path=None):
    output = Path(output)
    targets = [output / n for n in ["results-summary.json", "requests.json"]]
    if any(p.exists() for p in targets):
        raise FileExistsError("Export exists; use a new output directory")
    base, bh = _json_snapshot(base_path)
    quality, qh = _json_snapshot(quality_path)
    summary, requests = build_exports(base, quality, metadata, protocol_path)
    summary["source_results"] = {
        "base": {"path": str(base_path), "sha256": bh},
        "quality": {"path": str(quality_path), "sha256": qh},
    }
    for p, value in zip(targets, [summary, requests]):
        _save_new(p, value)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["base", "quality", "output"]:
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=Path("benchmarks/completed-protection-v1"))
    args = parser.parse_args(argv)
    export_results(args.base, args.quality, args.output, args.metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
