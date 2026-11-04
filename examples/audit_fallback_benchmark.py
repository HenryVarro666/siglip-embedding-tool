"""Standard-library after-run audit; imports no retrieval, metric or timing code."""

import argparse
import hashlib
import json
import math
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

TIMES = (
    "constraint_translation_ms",
    "original_translation_ms",
    "fallback_translation_ms",
    "translation_ms",
    "retrieval_ms",
    "total_ms",
)
STATUS = ("ok", "failure", "primary_failure", "fallback_triggered", "effective_route")


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")
    ).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def evidence_log_snapshot(directory):
    """Exclude this process's open stdout/stderr from whole-file attestations."""
    active = set()
    for descriptor in (1, 2):
        try:
            stat = os.fstat(descriptor)
        except OSError:
            continue
        active.add((stat.st_dev, stat.st_ino))
    snapshot = {}
    for path in sorted(Path(directory).glob("*.log")):
        stat = path.stat()
        if (stat.st_dev, stat.st_ino) not in active:
            snapshot[str(path)] = file_hash(path)
    return snapshot


class Checks:
    def __init__(self):
        self.count = self.numeric = 0
        self.max_error = 0.0

    def same(self, actual, expected, label):
        self.count += 1
        if isinstance(expected, dict):
            self.require(isinstance(actual, dict) and actual.keys() == expected.keys(), label)
            for key, value in expected.items():
                self.same(actual[key], value, f"{label}.{key}")
        elif isinstance(expected, list):
            self.require(isinstance(actual, list) and len(actual) == len(expected), label)
            for n, value in enumerate(expected):
                self.same(actual[n], value, f"{label}[{n}]")
        elif type(expected) in (int, float) and type(actual) in (int, float):
            difference = abs(actual - expected)
            self.numeric += 1
            self.max_error = max(self.max_error, difference)
            self.require(
                math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-9),
                f"{label}: saved={actual!r}, recomputed={expected!r}",
            )
        else:
            self.require(type(actual) is type(expected) and actual == expected, label)

    def require(self, condition, label):
        self.count += 1
        if not condition:
            raise ValueError(label)


def statistics(samples):
    values = sorted(samples)
    if not values or any(
        type(v) not in (float, int) or not math.isfinite(v) or v < 0 for v in values
    ):
        raise ValueError("Timing requires nonempty finite nonnegative samples")

    def percentile(fraction):
        position = (len(values) - 1) * fraction
        lower = math.floor(position)
        return values[lower] + (values[math.ceil(position)] - values[lower]) * (position - lower)

    return {
        "count": len(values),
        "p50": percentile(0.5),
        "p95": percentile(0.95),
        "mean": math.fsum(values) / len(values),
        "min": values[0],
        "max": values[-1],
    }


def retrieval_metrics(rankings, qrels, gallery, cutoffs=(1, 5, 10)):
    rows = {}
    for intent in sorted(qrels):
        relevant, ranking = set(qrels[intent]), rankings[intent]
        if not relevant or len(relevant) != len(qrels[intent]) or not relevant <= set(gallery):
            raise ValueError("Invalid positive labels")
        if ranking is not None and (len(ranking) != len(gallery) or set(ranking) != set(gallery)):
            raise ValueError("Successful ranking is not the complete unique gallery")
        positive_ranks = [n for n, image in enumerate(ranking or [], 1) if image in relevant]
        row = {}
        for k in cutoffs:
            hits = sum(n <= k for n in positive_ranks)
            row[f"hit_at_{k}"] = float(hits > 0)
            row[f"recall_at_{k}"] = hits / len(relevant)
        row["mrr"] = 1 / positive_ranks[0] if positive_ranks else 0.0
        row["average_precision"] = math.fsum(
            n / rank for n, rank in enumerate(positive_ranks, 1)
        ) / len(relevant)
        rows[intent] = row
    successful = [row for i, row in rows.items() if rankings[i] is not None]

    def mean(values):
        return {
            k: math.fsum(row[k] for row in values) / len(values) for k in next(iter(rows.values()))
        }

    return {
        "metrics": {"per_query": rows, "aggregate": mean(list(rows.values()))},
        "success_conditional_metrics": mean(successful) if successful else None,
        "successful_intents": len(successful),
        "failed_intents": len(rows) - len(successful),
    }


def request_rates(attempts):
    triggered = sum(a["fallback_triggered"] for a in attempts)

    def fraction(n, d):
        return {"numerator": n, "denominator": d, "rate": n / d if d else None}

    return {
        "primary_rejection": fraction(
            sum(a["primary_failure"] is not None for a in attempts), len(attempts)
        ),
        "fallback_trigger": fraction(triggered, len(attempts)),
        "fallback_success": fraction(
            sum(a["ok"] and a["fallback_triggered"] for a in attempts), triggered
        ),
        "terminal_failure": fraction(sum(not a["ok"] for a in attempts), len(attempts)),
    }


def semantics(attempt):
    return {k: v for k, v in attempt.items() if k not in {"matches", "rankings", "timing"}}


def ranking_difference(actual, expected, qrels, gallery, checks, label, tolerance):
    checks.require(
        len(actual) == len(gallery) and {r["image_id"] for r in actual} == set(gallery), label
    )
    maximum = 0.0
    for n, (a, b) in enumerate(zip(actual, expected), 1):
        checks.same(a["rank"], n, label + ".rank")
        checks.same(a["image_id"], b["image_id"], label + ".order")
        checks.same(a["relevant"], a["image_id"] in qrels, label + ".relevance")
        checks.require(
            type(a["score"]) in (int, float) and math.isfinite(a["score"]), label + ".score"
        )
        maximum = max(maximum, abs(a["score"] - b["score"]))
    checks.require(maximum <= tolerance, label + ".score_drift")
    return maximum


def validate_timing(timing, checks, label):
    checks.same(sorted(timing), sorted(TIMES), label + ".keys")
    checks.require(
        all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in timing.values()),
        label + ".finite_nonnegative",
    )
    stages = sum(timing[k] for k in TIMES[:3])
    checks.same(timing["translation_ms"], stages, label + ".translation_sum")
    checks.require(
        timing["total_ms"] + 1e-6 >= stages + timing["retrieval_ms"], label + ".hidden_cost"
    )


def audit_result(result, inputs, chinese):
    c, policies = Checks(), result["protocol"]["policies"]
    gallery, qrels = inputs["gallery_ids"], inputs["qrels"]
    c.same(result["evaluation_inputs"], inputs, "frozen_evaluation_inputs")
    c.same(fingerprint(inputs), result["protocol"]["evaluation_inputs_sha256"], "input_digest")
    c.same(sorted(result["strategies"]), sorted(policies), "policy_set")
    summaries, duplicate_semantics = {}, {}
    for policy, recipe in policies.items():
        strategy = result["strategies"][policy]
        attempts, warm_samples = strategy["attempts"], strategy["warm_samples"]
        c.same(sorted(attempts), sorted(qrels), policy + ".intents")
        c.same(sorted(warm_samples), sorted(qrels), policy + ".warm_intents")
        rankings, warm, maximum = {}, [], 0.0
        for intent, attempt in attempts.items():
            label = f"{policy}.{intent}"
            c.same(attempt["ok"], attempt["failure"] is None, label + ".terminal_status")
            ranking = attempt["rankings"]
            if not attempt["ok"]:
                c.same(ranking, None, label + ".failed_ranking")
            else:
                ranking_difference(ranking, ranking, qrels[intent], gallery, c, label, 0)
            rankings[intent] = [r["image_id"] for r in ranking] if ranking is not None else None
            validate_timing(attempt["timing"], c, label + ".scored_timing")
            expected_trigger = (
                attempt["primary_failure"] is not None and recipe["fallback"] != "error"
            )
            c.same(attempt["fallback_triggered"], expected_trigger, label + ".trigger_contract")
            if recipe["decoder"] != "constrained":
                c.same(attempt["primary_failure"], None, label + ".nonconstraint_primary")
                c.same(attempt["constraint_trace"], None, label + ".nonconstraint_trace")
                c.same(
                    attempt["timing"]["constraint_translation_ms"], 0.0, label + ".constraint_time"
                )
            else:
                c.same(attempt["original_trace"], None, label + ".original_trace")
                c.same(attempt["timing"]["original_translation_ms"], 0.0, label + ".original_time")
            if not expected_trigger or recipe["fallback"] != "opus":
                c.same(attempt["fallback_trace"], None, label + ".unused_fallback_trace")
                c.same(
                    attempt["timing"]["fallback_translation_ms"],
                    0.0,
                    label + ".unused_fallback_time",
                )
            if attempt["ok"]:
                route = attempt["effective_route"]
                texts = (
                    [chinese[intent]]
                    if route == "zh"
                    else [attempt["translation"]]
                    if recipe["mode"] == "auto_en"
                    else [chinese[intent], attempt["translation"]]
                )
                c.same(attempt["query_texts"], texts, label + ".actual_query_texts")
            for field in ("constraint_trace", "original_trace", "fallback_trace"):
                trace = attempt[field]
                if trace is not None:
                    c.same(trace["source_text"], chinese[intent], label + "." + field + ".source")
                    c.require(trace.get("status") in {"success", "failed"}, label + ".trace_status")
                    if trace["status"] == "success":
                        c.same(
                            trace["output"], attempt["translation"], label + "." + field + ".output"
                        )
            if attempt["primary_failure"] is not None:
                trace = attempt["constraint_trace"]
                c.require(
                    isinstance(trace, dict)
                    and trace.get("status") == "failed"
                    and isinstance(trace.get("sequence"), list)
                    and bool(trace["sequence"])
                    and isinstance(trace.get("output"), str),
                    label + ".decoded_rejection",
                )
            c.same(
                len(warm_samples[intent]),
                result["protocol"]["query_repeats"],
                label + ".repeat_count",
            )
            for sample in warm_samples[intent]:
                for field in STATUS:
                    c.same(sample[field], attempt[field], label + ".warm." + field)
                actual_ranking = sample["ranking_override"]
                if actual_ranking is None:
                    actual_ranking = ranking
                c.same(
                    sample["evidence_sha256"],
                    fingerprint({"semantics": semantics(attempt), "rankings": actual_ranking}),
                    label + ".warm.evidence",
                )
                if attempt["ok"]:
                    maximum = max(
                        maximum,
                        ranking_difference(
                            actual_ranking,
                            ranking,
                            qrels[intent],
                            gallery,
                            c,
                            label + ".override",
                            result["protocol"]["score_atol"],
                        ),
                    )
                else:
                    c.same(actual_ranking, None, label + ".failed_override")
                validate_timing(sample["timing"], c, label + ".warm_timing")
                for stage in TIMES[:3]:
                    if attempt["timing"][stage] == 0:
                        c.same(sample["timing"][stage], 0.0, label + ".warm.unused." + stage)
                warm.append(sample)
        computed = retrieval_metrics(rankings, qrels, gallery)
        for key, value in computed.items():
            c.same(strategy[key], value, policy + "." + key)
        computed["rates"], computed["warm_rates"] = (
            request_rates(list(attempts.values())),
            request_rates(warm),
        )
        computed["timing"] = {k: statistics([s["timing"][k] for s in warm]) for k in TIMES}
        computed["timing_by_branch"] = {
            branch: {
                k: statistics([s["timing"][k] for s in warm if s["fallback_triggered"] == trigger])
                for k in TIMES
            }
            for branch, trigger in (("fallback_triggered", True), ("no_fallback", False))
            if any(s["fallback_triggered"] == trigger for s in warm)
        }
        computed.update(max_repeated_absolute_score_difference=maximum, rank_disagreements=0)
        for key in computed.keys() - {
            "metrics",
            "success_conditional_metrics",
            "successful_intents",
            "failed_intents",
        }:
            c.same(strategy[key], computed[key], policy + "." + key)
        computed["additional_at_3"] = retrieval_metrics(rankings, qrels, gallery, (3,))["metrics"]
        summaries[policy] = computed
        duplicate_semantics[policy] = {i: semantics(a) for i, a in attempts.items()}
    branch_count, branch_max = 0, 0.0
    for policy, recipe in policies.items():
        if recipe["fallback"] == "error":
            continue
        suffix = "en" if recipe["mode"] == "auto_en" else "fusion"
        for intent, actual in result["strategies"][policy]["attempts"].items():
            primary = result["strategies"][f"constrained_{suffix}"]["attempts"][intent]
            for field in ("constraint_trace", "primary_failure"):
                c.same(actual[field], primary[field], policy + ".primary." + field)
            triggered = primary["primary_failure"] is not None
            c.same(actual["fallback_triggered"], triggered, policy + ".trigger")
            route = (
                (f"original_{suffix}" if recipe["fallback"] == "opus" else "zh")
                if triggered
                else f"constrained_{suffix}"
            )
            control = result["strategies"][route]["attempts"][intent]
            for field in ("ok", "translation", "query_texts"):
                c.same(actual[field], control[field], policy + ".control." + field)
            c.same(
                actual["effective_route"],
                route if control["ok"] else control["effective_route"],
                policy + ".route",
            )
            c.same(
                actual["fallback_trace"],
                control["original_trace"] if triggered and recipe["fallback"] == "opus" else None,
                policy + ".fallback_trace",
            )
            if actual["ok"]:
                branch_max = max(
                    branch_max,
                    ranking_difference(
                        actual["rankings"],
                        control["rankings"],
                        qrels[intent],
                        gallery,
                        c,
                        policy + ".branch_ranking",
                        result["protocol"]["score_atol"],
                    ),
                )
            branch_count += 1
    branch = {
        "checked": branch_count,
        "max_absolute_score_difference": branch_max,
        "rank_disagreements": 0,
    }
    c.same(result["branch_audit"], branch, "branch_audit")
    c.same(result["request_semantics"], duplicate_semantics, "request_semantics")
    return {
        "checks": c.count,
        "numeric_comparisons": c.numeric,
        "maximum_numeric_absolute_difference": c.max_error,
        "branch_audit": branch,
        "strategies": summaries,
    }


def source_hashes(root, metadata=None):
    """Hashes of the bound package files: from the frozen-source snapshot of ``metadata``
    when one exists (the measured blobs), otherwise from the working tree."""
    source = root / "src/siglip_tools"
    hashes = {}
    table = None
    if metadata is not None and (Path(metadata) / "frozen-source.json").is_file():
        table = read_json(Path(metadata) / "frozen-source.json")["files"]

    def group(files, **dependencies):
        values = {
            name: hashes.setdefault(
                name, table[name] if table is not None else file_hash(source / name)
            )
            for name in files.split()
        }
        return fingerprint({**dependencies, **values})

    baseline = group("benchmark.py evaluation.py commons.py encoder.py index.py images.py")
    fusion = group("fusion.py fusion_benchmark.py", baseline=baseline)
    previous = group(
        "translation.py auto_benchmark.py term_benchmark.py opus_constraints.py "
        "power_glossary_v1.json",
        fusion=fusion,
    )
    current = group("constraint_fallback.py fallback_benchmark.py cli.py", previous_core=previous)
    return {"files": hashes, "previous_core_sha256": previous, "implementation_sha256": current}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "quality", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=Path("benchmarks/constraint-fallback-v1"))
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Audit output exists; choose a new path")
    metadata, root = args.metadata.resolve(), Path(__file__).resolve().parents[1]
    c, report = Checks(), {"schema_version": 1, "passed": False, "failures": []}
    try:
        protocol, freeze = (
            read_json(metadata / "protocol.json"),
            read_json(metadata / "freeze.json"),
        )
        sources = source_hashes(root, metadata)
        c.same(file_hash(metadata / "protocol.json"), freeze["protocol_sha256"], "protocol_hash")
        for key in (
            "implementation_sha256",
            "dataset_hashes",
            "supporting_hashes",
            "source_indexes",
            "retrieval_weights",
        ):
            c.same(freeze[key], protocol[key], "freeze." + key)
        c.same(sources["implementation_sha256"], protocol["implementation_sha256"], "source_code")
        c.same(sources["previous_core_sha256"], freeze["previous_core_sha256"], "preserved_core")
        bound = {
            "protocol.json": file_hash(metadata / "protocol.json"),
            "freeze.json": file_hash(metadata / "freeze.json"),
        }
        for files in (
            {f"{k}.json": v for k, v in protocol["dataset_hashes"].items()},
            protocol["supporting_hashes"],
        ):
            for name, expected in files.items():
                bound[name] = file_hash(metadata / name)
                c.same(bound[name], expected, "input." + name)
                c.same(
                    bound[name],
                    file_hash(root / "benchmarks/terminology-constraints-v1" / name),
                    "unchanged_prior_input." + name,
                )
        manifest, queries, labels = (
            read_json(metadata / f"{name}.json") for name in ("manifest", "queries", "qrels")
        )
        chinese = {q["intent_id"]: q["text"] for q in queries["queries"] if q["language"] == "zh"}
        qrels = {
            q["intent_id"]: labels["qrels"][q["query_id"]]
            for q in queries["queries"]
            if q["language"] == "zh"
        }
        inputs = {"gallery_ids": [r["image_id"] for r in manifest["images"]], "qrels": qrels}
        c.same(len(inputs["gallery_ids"]), 40, "gallery_size")
        c.same(len(set(inputs["gallery_ids"])), 40, "unique_gallery")
        c.same(len(qrels), 16, "intent_count")
        c.same(len(protocol["policies"]), 9, "nine_policies")
        results, audits, result_files = {}, {}, {}
        for alias in ("base", "quality"):
            path, result = getattr(args, alias), read_json(getattr(args, alias))
            for key, expected in (
                ("model_alias", alias),
                ("protocol", protocol),
                ("protocol_sha256", freeze["protocol_sha256"]),
                ("implementation_sha256", protocol["implementation_sha256"]),
                ("source_index_sha256", protocol["source_indexes"][alias]),
                ("dataset_hashes", protocol["dataset_hashes"]),
                ("gallery_images", 40),
                ("independent_intents", 16),
                ("training_executed", False),
            ):
                c.same(result[key], expected, alias + "." + key)
            for key, expected in protocol["retrieval_models"][alias].items():
                c.same(result["model_identity"][key], expected, alias + ".model." + key)
            audits[alias], results[alias] = audit_result(result, inputs, chinese), result
            result_files[alias] = {"path": str(path.resolve()), "sha256": file_hash(path)}
        for key in ("request_semantics", "translator_identities", "environment"):
            c.same(results["base"][key], results["quality"][key], "cross_model." + key)
        provenance = {
            name: read_json(metadata / f"{name}-provenance.json")
            for name in ("translator", "decoder")
        }
        for variant, identity in results["base"]["translator_identities"].items():
            for key, expected in (
                ("source_model", protocol["translation_model"]),
                ("source_revision", protocol["translation_revision"]),
                ("artifacts", provenance["translator"]["artifacts"]),
                ("dtype", protocol["dtype"]),
                ("device", protocol["translator_device"]),
                ("source", provenance["translator"]["source"]),
            ):
                c.same(identity[key], expected, "translator_identity." + variant + "." + key)
            for key, expected in (
                ("num_beams", protocol["num_beams"][variant]),
                ("do_sample", False),
                ("max_new_tokens", 64),
                ("use_cache", True),
            ):
                c.same(identity["generation"][key], expected, "generation." + variant + "." + key)
            if variant == "constrained":
                c.same(identity["decoder"], provenance["decoder"], "decoder_identity")
                c.same(
                    identity["glossary"]["sha256"], protocol["glossary_sha256"], "glossary_identity"
                )
        artifacts, historical = {}, {}
        c.same(
            provenance["translator"]["artifacts"],
            {
                name: digest
                for name, digest in protocol["translator_files"].items()
                if name != "translation_provenance.json"
            },
            "translator_provenance_artifacts",
        )
        for row in manifest["images"]:
            path = root / "data/equipment-v1" / row["path"]
            artifacts[str(path.relative_to(root))] = file_hash(path)
            c.same(
                artifacts[str(path.relative_to(root))], row["sha256"], "gallery." + row["image_id"]
            )
        for dirname, key in (
            ("models/opus-mt-zh-en", "translator_files"),
            ("models/opus-constrained-decoder", "decoder_files"),
        ):
            directory = root / dirname
            c.same(
                sorted(
                    p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()
                ),
                sorted(protocol[key]),
                "artifact_inventory." + key,
            )
            for name, expected in protocol[key].items():
                path = root / dirname / name
                artifacts[str(path.relative_to(root))] = file_hash(path)
                c.same(file_hash(path), expected, "artifact." + name)
        for alias, entry in protocol["retrieval_weights"].items():
            cache = Path(
                os.environ.get(
                    "HF_HUB_CACHE",
                    Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub",
                )
            )
            path = (
                cache
                / ("models--" + entry["model_id"].replace("/", "--"))
                / "snapshots"
                / entry["revision"]
                / entry["filename"]
            )
            artifacts[str(path)] = file_hash(path)
            c.same(artifacts[str(path)], entry["sha256"], alias + ".weight_bytes")
            c.same(path.stat().st_size, entry["size_bytes"], alias + ".weight_size")
            index = root / "artifacts/equipment-benchmark" / f"{alias}-v1/index.npz"
            artifacts[str(index.relative_to(root))] = file_hash(index)
            c.same(
                artifacts[str(index.relative_to(root))],
                protocol["source_indexes"][alias],
                alias + ".index_bytes",
            )
            old_ref = protocol["previous_result_reference"][alias]
            old_path = root / old_ref["path"]
            c.same(file_hash(old_path), old_ref["sha256"], alias + ".historical_result_hash")
            old = read_json(old_path)
            historical[alias] = {
                "path": old_ref["path"],
                "sha256": old_ref["sha256"],
                "controls": {},
            }
            for policy in (
                "zh",
                "original_en",
                "original_fusion",
                "constrained_en",
                "constrained_fusion",
            ):
                previous, current = old["strategies"][policy], results[alias]["strategies"][policy]
                c.same(
                    current["metrics"],
                    previous["metrics"],
                    alias + ".historical." + policy + ".metrics",
                )
                max_difference = 0.0
                c.same(
                    sorted(i for i, a in current["attempts"].items() if not a["ok"]),
                    sorted(previous["failures"]),
                    alias + ".historical." + policy + ".failures",
                )
                for intent, attempt in current["attempts"].items():
                    if attempt["ok"]:
                        max_difference = max(
                            max_difference,
                            ranking_difference(
                                attempt["rankings"],
                                previous["rankings"][intent],
                                qrels[intent],
                                inputs["gallery_ids"],
                                c,
                                alias + ".historical." + policy,
                                protocol["score_atol"],
                            ),
                        )
                    if policy != "zh":
                        variant = policy.split("_", 1)[0]
                        trace = (
                            attempt["original_trace"]
                            if variant == "original"
                            else attempt["constraint_trace"]
                        )
                        c.same(
                            trace,
                            old["constraint_traces"][variant][intent],
                            alias + ".historical.trace",
                        )
                        english = (
                            attempt["translation"]
                            if attempt["translation"] is not None
                            else trace["output"]
                        )
                        c.same(
                            english,
                            old["translations"][variant][intent],
                            alias + ".historical.output",
                        )
                historical[alias]["controls"][policy] = {
                    "retrieval_metrics_unchanged": True,
                    "outputs_and_traces_unchanged": True,
                    "maximum_absolute_score_difference": max_difference,
                    "previous_total_ms_p50": previous["timing"]["total_ms"]["p50"],
                    "current_total_ms_p50": current["timing"]["total_ms"]["p50"],
                    "note": "Separate new timing protocol; old timings were not reused.",
                }
        for name in ("translator-provenance.json", "decoder-provenance.json"):
            bound[name] = file_hash(metadata / name)
            c.same(
                bound[name],
                file_hash(root / "benchmarks/terminology-constraints-v1" / name),
                "provenance." + name,
            )
        for name, key, local in (
            ("translator-provenance.json", "translator_files", "translation_provenance.json"),
            ("decoder-provenance.json", "decoder_files", "decoder_provenance.json"),
        ):
            c.same(bound[name], protocol[key][local], "bound_provenance." + name)
        report.update(
            passed=True,
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            protocol=protocol,
            freeze=freeze,
            frozen_source_identifiers={
                "implementation_git_commit": protocol["implementation_git_commit"],
                "freeze_git_commit": subprocess.check_output(
                    ["git", "log", "-1", "--format=%H", "--", str(metadata / "freeze.json")],
                    cwd=root,
                    text=True,
                ).strip(),
                **sources,
            },
            external_provenance=provenance,
            input_and_support_sha256=bound,
            local_artifact_sha256=artifacts,
            result_files=result_files,
            checks=c.count + sum(a["checks"] for a in audits.values()),
            numeric_comparisons=c.numeric + sum(a["numeric_comparisons"] for a in audits.values()),
            maximum_numeric_absolute_difference=max(
                [c.max_error] + [a["maximum_numeric_absolute_difference"] for a in audits.values()]
            ),
            models=audits,
            historical_comparison=historical,
            evidence_file_sha256={
                str(Path(p).relative_to(root)): digest
                for p, digest in evidence_log_snapshot(
                    root / "artifacts/constraint-fallback"
                ).items()
            },
            evidence_log_scope="Completed logs snapshot; current stdout/stderr files excluded.",
            auditor_sha256=file_hash(__file__),
            limits=[
                "Earlier diagnostic results were known; "
                "this is no new independent holdout or human gold.",
                "No training, per-query tuning, external service, license change "
                "or human semantic validation.",
                "Fresh MPS/CPU workstation measurements do not establish concurrency, "
                "peak resources or other hardware performance.",
                "Warm semantics reconstruct from scored reference plus declared ranking overrides; "
                "digests attest recorded consistency, not independent inference replay.",
            ],
        )
    except (ValueError, KeyError, TypeError, OSError) as exc:
        report["failures"].append(f"{type(exc).__name__}: {exc}")
        report.update(completed_top_level_checks=c.count, completed_numeric_comparisons=c.numeric)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {"passed": report["passed"], "failures": report["failures"], "output": str(args.output)}
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
