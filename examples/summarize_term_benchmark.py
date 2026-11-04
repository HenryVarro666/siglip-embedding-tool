"""Export auditable summaries and HTML inputs from the frozen terminology pilot."""

import argparse
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from siglip_tools.benchmark import _stats
from siglip_tools.frozen_source import execution_drift, reference_implementation
from siglip_tools.fusion_benchmark import _save_new
from siglip_tools.images import file_sha256
from siglip_tools.opus_constraints import GLOSSARY_PATH, glossary_requirements
from siglip_tools.term_benchmark import (
    VARIANTS,
    compare_results,
    evaluate_attempts,
    implementation_hash,
)


def validate_fixed_inputs(results, metadata):
    """Bind this after-run export to the declared v1 protocol and source rules."""
    freeze = json.loads((metadata / "freeze.json").read_text())
    protocol = json.loads((metadata / "protocol.json").read_text())
    hashes = {n: file_sha256(metadata / f"{n}.json") for n in ("manifest", "queries", "qrels")}
    if file_sha256(metadata / "protocol.json") != freeze["protocol_sha256"]:
        raise ValueError("Current file differs from the frozen protocol")
    reference_implementation(
        metadata, implementation_hash, expected=freeze["implementation_sha256"]
    )
    for result in results.values():
        if result["protocol_sha256"] != freeze["protocol_sha256"] or result["protocol"] != protocol:
            raise ValueError("Result differs from the frozen protocol")
        if (
            result["experiment_id"] != "equipment-terminology-constraints-v1"
            or result["dataset_hashes"] != hashes
            or hashes != freeze["dataset_hashes"]
            or result["implementation_sha256"] != freeze["implementation_sha256"]
        ):
            raise ValueError("Metadata differs from the measured dataset")
    if protocol["supporting_hashes"] != freeze["supporting_hashes"]:
        raise ValueError("Supporting declarations differ from freeze")
    for name, digest in freeze["supporting_hashes"].items():
        if file_sha256(metadata / name) != digest:
            raise ValueError("Current supporting metadata differs from frozen bytes")
    if file_sha256(GLOSSARY_PATH) != freeze["supporting_hashes"]["glossary.json"]:
        raise ValueError("Packaged glossary differs from frozen bytes")
    return hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--quality", type=Path, required=True)
    parser.add_argument(
        "--metadata", type=Path, default=Path("benchmarks/terminology-constraints-v1")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--views-output", type=Path, required=True)
    args = parser.parse_args()
    paths = {"base": args.base, "quality": args.quality}
    results = {a: json.loads(p.read_text()) for a, p in paths.items()}
    comparison = compare_results(results["base"], results["quality"])
    hashes = validate_fixed_inputs(results, args.metadata)
    queries = json.loads((args.metadata / "queries.json").read_text())["queries"]
    chinese = {q["intent_id"]: q["text"] for q in queries if q["language"] == "zh"}
    qrels = json.loads((args.metadata / "qrels.json").read_text())["qrels"]
    relevance = {q["intent_id"]: qrels[q["query_id"]] for q in queries if q["language"] == "zh"}
    gallery = [
        r["image_id"] for r in json.loads((args.metadata / "manifest.json").read_text())["images"]
    ]
    targeted = [i for i, text in chinese.items() if glossary_requirements(text)]
    controls = [i for i in chinese if i not in targeted]
    summary = {
        "experiment_id": results["base"]["experiment_id"],
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_reference_commit": "b68cb99",
        "implementation_sha256": results["base"]["implementation_sha256"],
        "analysis_helpers": execution_drift(args.metadata, implementation_hash),
        "protocol_sha256": results["base"]["protocol_sha256"],
        "dataset_hashes": hashes,
        "gallery_images": len(gallery),
        "independent_intents": len(chinese),
        "term_opportunities": sum(len(glossary_requirements(t)) for t in chinese.values()),
        "source_triggered_intents": targeted,
        "no_target_controls": controls,
        "training_executed": False,
        "translation_text_cache": False,
        "translator_identities": results["base"]["translator_identities"],
        "models": {},
        "comparison": comparison,
        "subset_notes": {
            "constrained_accepted": (
                "All policies on the same constrained-success subset; descriptive selection "
                "after execution, not the primary all-intent result."
            ),
            "without_t16": (
                "Post-review sensitivity: t16 below versus inside lower cabinet is ambiguous. "
                "Primary frozen labels remain unchanged."
            ),
        },
    }
    for alias, result in results.items():
        accepted = [
            i for i in chinese if i not in result["strategies"]["constrained_en"]["failures"]
        ]
        row = {
            "source_result": {"path": str(paths[alias]), "sha256": file_sha256(paths[alias])},
            "model_identity": result["model_identity"],
            "environment": result["environment"],
            "source_index_sha256": result["source_index_sha256"],
            "startup": deepcopy(result["startup"]),
            "strategies": {},
            "subsets": {},
            "control_checks": {},
        }
        for request in row["startup"]["first_requests_per_decoder"].values():
            request.pop("rankings", None)
        for policy, strategy in result["strategies"].items():
            exported = {
                k: deepcopy(v) for k, v in strategy.items() if k not in {"rankings", "warm_samples"}
            }
            exported["per_intent"] = {
                i: {
                    "ok": ranks is not None,
                    "failure": strategy["failures"].get(i),
                    "positive_image_ranks": {
                        r["image_id"]: r["rank"] for r in (ranks or []) if r["relevant"]
                    },
                    "top5": (ranks or [])[:5],
                }
                for i, ranks in strategy["rankings"].items()
            }
            warm = [s for samples in strategy["warm_samples"].values() for s in samples]
            exported["success_only_timing"] = {
                key: _stats([s[key] for s in warm if s["ok"]])
                if any(s["ok"] for s in warm)
                else None
                for key in ("translation_ms", "retrieval_ms", "total_ms")
            }
            exported["failure_only_total_ms"] = (
                _stats([s["total_ms"] for s in warm if not s["ok"]])
                if any(not s["ok"] for s in warm)
                else None
            )
            row["strategies"][policy] = exported
        for name, ids in (
            ("source_triggered", targeted),
            ("controls", controls),
            ("constrained_accepted", accepted),
            ("without_t16", [i for i in chinese if i != "t16"]),
        ):
            row["subsets"][name] = {"intent_ids": ids, "strategies": {}}
            if ids:
                for policy, strategy in result["strategies"].items():
                    rankings = {
                        i: [r["image_id"] for r in strategy["rankings"][i]]
                        if strategy["rankings"][i] is not None
                        else None
                        for i in ids
                    }
                    row["subsets"][name]["strategies"][policy] = evaluate_attempts(
                        rankings, {i: relevance[i] for i in ids}, gallery
                    )
        for i in controls:
            if result["translations"]["beam"][i] != result["translations"]["constrained"][i]:
                raise ValueError("No-target control translation differs")
            if result["constraint_traces"]["constrained"][i]["constraints_applied"]:
                raise ValueError("No-target control received a constraint")
            for suffix in ("en", "fusion"):
                if (
                    result["strategies"][f"beam_{suffix}"]["rankings"][i]
                    != result["strategies"][f"constrained_{suffix}"]["rankings"][i]
                ):
                    raise ValueError("No-target control ranking differs")
            row["control_checks"][i] = (
                "exact beam/constrained translation, rankings and scores; no constraint"
            )
        summary["models"][alias] = row
        for variant in VARIANTS:
            view = {
                "model_alias": alias,
                "translator_alias": f"opus-{variant}",
                "dataset_hashes": hashes,
                "translations": result["translations"][variant],
                "strategies": {
                    "zh": result["strategies"]["zh"],
                    "auto_en": result["strategies"][f"{variant}_en"],
                    "auto_fusion": result["strategies"][f"{variant}_fusion"],
                },
            }
            _save_new(args.views_output / f"{alias}-{variant}.json", view)
    _save_new(args.output, summary)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
