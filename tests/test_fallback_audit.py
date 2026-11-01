"""Independent audit rejects numeric, timing, evidence and routing corruption."""

import hashlib
import importlib.util
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest


def load_audit():
    path = Path(__file__).resolve().parents[1] / "examples/audit_fallback_benchmark.py"
    assert path.exists(), "Independent audit implementation is missing"
    spec = importlib.util.spec_from_file_location("fallback_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def synthetic_result():
    inputs = {"gallery_ids": ["a", "b", "c"], "qrels": {"i1": ["c"], "i2": ["c"]}}
    policies = {"zh": {"decoder": None, "mode": "zh", "fallback": "error"}}
    for name in ("original", "constrained", "fallback_opus", "fallback_zh"):
        for suffix in ("en", "fusion"):
            policies[f"{name}_{suffix}"] = {
                "decoder": "original" if name == "original" else "constrained",
                "mode": "auto_en" if suffix == "en" else "auto_fusion",
                "fallback": {"fallback_opus": "opus", "fallback_zh": "chinese"}.get(name, "error"),
            }
    result = {
        "protocol": {"policies": policies, "query_repeats": 2, "score_atol": 1e-5},
        "evaluation_inputs": inputs,
        "strategies": {},
        "request_semantics": {},
        "branch_audit": {
            "checked": 8,
            "max_absolute_score_difference": 0.0,
            "rank_disagreements": 0,
        },
    }
    result["protocol"]["evaluation_inputs_sha256"] = fingerprint(inputs)
    chinese = {"i1": "中文一", "i2": "中文二"}
    metric_names = (
        "hit_at_1",
        "recall_at_1",
        "hit_at_5",
        "recall_at_5",
        "hit_at_10",
        "recall_at_10",
        "mrr",
        "average_precision",
    )

    def metrics(rank):
        if rank is None:
            return dict.fromkeys(metric_names, 0.0)
        return dict(
            zip(
                metric_names,
                [float(rank == 1), float(rank == 1), 1.0, 1.0, 1.0, 1.0, 1 / rank, 1 / rank],
            )
        )

    def rates(primary, triggered, failed):
        def fraction(n, d):
            return {"numerator": n, "denominator": d, "rate": n / d if d else None}

        return {
            "primary_rejection": fraction(primary, 2),
            "fallback_trigger": fraction(triggered, 2),
            "fallback_success": fraction(triggered - failed if triggered else 0, triggered),
            "terminal_failure": fraction(failed, 2),
        }

    def stats(values):
        lo, hi = min(values), max(values)
        return {
            "count": len(values),
            "p50": (lo + hi) / 2,
            "p95": hi,
            "mean": sum(values) / len(values),
            "min": lo,
            "max": hi,
        }

    for policy, recipe in policies.items():
        attempts, ranks = {}, {}
        for intent in chinese:
            constrained = recipe["decoder"] == "constrained"
            rejected = constrained and intent == "i2"
            trigger = rejected and recipe["fallback"] != "error"
            route = (
                (
                    "zh"
                    if recipe["fallback"] == "chinese"
                    else policy.replace("fallback_opus", "original")
                )
                if trigger
                else policy
            )
            if constrained and not rejected:
                route = "constrained_" + policy.rsplit("_", 1)[1]
            ok = not rejected or trigger
            english = (
                None
                if route == "zh" or not ok
                else ("original en" if route.startswith("original") else "valid en")
            )
            primary = (
                {"stage": "constraint_translation", "type": "ValueError", "message": "rejected"}
                if rejected
                else None
            )
            trace = {
                "source_text": chinese[intent],
                "status": "failed" if rejected else "success",
                "sequence": [9, 10] if rejected else [9, 2],
                "output": "rejected fragment" if rejected else "valid en",
            }
            original_trace = {
                "source_text": chinese[intent],
                "status": "success",
                "sequence": [8, 2],
                "output": "original en",
            }
            rank = (
                None if not ok else 1 if route == "zh" else 3 if route.startswith("original") else 2
            )
            ranks[intent] = rank
            ids = (
                None
                if rank is None
                else {1: ["c", "a", "b"], 2: ["a", "c", "b"], 3: ["a", "b", "c"]}[rank]
            )
            timing = {
                "constraint_translation_ms": float(constrained),
                "original_translation_ms": float(recipe["decoder"] == "original"),
                "fallback_translation_ms": float(trigger and recipe["fallback"] == "opus"),
                "retrieval_ms": float(ok),
                "total_ms": 10.0,
            }
            timing["translation_ms"] = sum(
                timing[k]
                for k in (
                    "constraint_translation_ms",
                    "original_translation_ms",
                    "fallback_translation_ms",
                )
            )
            attempts[intent] = {
                "ok": ok,
                "translation": english,
                "failure": None if ok else primary,
                "primary_failure": primary,
                "constraint_trace": trace if constrained else None,
                "fallback_trace": original_trace
                if trigger and recipe["fallback"] == "opus"
                else None,
                "original_trace": original_trace if recipe["decoder"] == "original" else None,
                "fallback_triggered": trigger,
                "effective_route": route if ok else None,
                "query_texts": []
                if not ok
                else [chinese[intent]]
                if route == "zh"
                else [english]
                if recipe["mode"] == "auto_en"
                else [chinese[intent], english],
                "rankings": None
                if ids is None
                else [
                    {"image_id": image, "rank": n, "score": 1.0 / n, "relevant": image == "c"}
                    for n, image in enumerate(ids, 1)
                ],
                "timing": timing,
            }
        per_query = {i: metrics(rank) for i, rank in ranks.items()}
        failed = sum(r is None for r in ranks.values())
        success = [row for i, row in per_query.items() if ranks[i] is not None]
        scored_rates = rates(
            int(recipe["decoder"] == "constrained"), int(recipe["fallback"] != "error"), failed
        )
        warm_rates = deepcopy(scored_rates)
        for fraction in warm_rates.values():
            fraction["numerator"] *= 2
            fraction["denominator"] *= 2
        warm = {}
        for i, attempt in attempts.items():
            semantic = {k: v for k, v in attempt.items() if k not in {"rankings", "timing"}}
            warm[i] = [
                {
                    **{
                        k: attempt[k]
                        for k in (
                            "ok",
                            "failure",
                            "primary_failure",
                            "fallback_triggered",
                            "effective_route",
                        )
                    },
                    "timing": deepcopy(attempt["timing"]),
                    "ranking_override": None,
                    "evidence_sha256": fingerprint(
                        {"semantics": semantic, "rankings": attempt["rankings"]}
                    ),
                }
                for _ in range(2)
            ]
        samples = [s for values in warm.values() for s in values]
        result["strategies"][policy] = {
            "metrics": {
                "per_query": per_query,
                "aggregate": {
                    k: sum(row[k] for row in per_query.values()) / 2 for k in metric_names
                },
            },
            "success_conditional_metrics": {
                k: sum(row[k] for row in success) / len(success) for k in metric_names
            },
            "successful_intents": 2 - failed,
            "failed_intents": failed,
            "attempts": attempts,
            "rates": scored_rates,
            "warm_samples": warm,
            "warm_rates": warm_rates,
            "timing": {
                k: stats([s["timing"][k] for s in samples]) for k in attempts["i1"]["timing"]
            },
            "timing_by_branch": {
                branch: {
                    k: stats([s["timing"][k] for s in samples if s["fallback_triggered"] == flag])
                    for k in attempts["i1"]["timing"]
                }
                for branch, flag in (("fallback_triggered", True), ("no_fallback", False))
                if any(s["fallback_triggered"] == flag for s in samples)
            },
            "max_repeated_absolute_score_difference": 0.0,
            "rank_disagreements": 0,
        }
        result["request_semantics"][policy] = {
            i: {k: v for k, v in a.items() if k not in {"rankings", "timing"}}
            for i, a in attempts.items()
        }
    return result, inputs, chinese


def test_full_ranking_ap_and_failed_intent_remain_in_denominator():
    audit = load_audit()
    gallery = [f"e{i}" for i in range(40)]
    measured = audit.retrieval_metrics(
        {"a": gallery, "b": None}, {"a": ["e39"], "b": ["e0"]}, gallery
    )
    assert measured["metrics"]["aggregate"]["average_precision"] == 1 / 80
    assert measured["metrics"]["per_query"]["b"]["recall_at_10"] == 0


def test_percentiles_use_linear_interpolation():
    assert load_audit().statistics([0, 10, 20, 30]) == pytest.approx(
        {"count": 4, "p50": 15.0, "p95": 28.5, "mean": 15.0, "min": 0, "max": 30}
    )


def test_clean_synthetic_evidence_and_branch_controls_pass():
    result, inputs, chinese = synthetic_result()
    report = load_audit().audit_result(result, inputs, chinese)
    assert report["checks"] > 100
    assert report["branch_audit"]["checked"] == 8


@pytest.mark.parametrize("corruption", ["metric", "timing", "evidence", "branch"])
def test_tampering_is_rejected(corruption):
    audit = load_audit()
    result, inputs, chinese = synthetic_result()
    strategy = result["strategies"]["fallback_opus_en"]
    if corruption == "metric":
        strategy["metrics"]["aggregate"]["average_precision"] = 0.99
    elif corruption == "timing":
        strategy["timing"]["total_ms"]["p95"] = 99.0
    elif corruption == "evidence":
        strategy["warm_samples"]["i2"][0]["evidence_sha256"] = "0" * 64
    else:
        strategy["attempts"]["i2"]["query_texts"] = ["rejected fragment"]
    with pytest.raises(ValueError):
        audit.audit_result(result, inputs, chinese)


def test_ranking_override_cannot_hide_score_drift_after_rehash():
    audit = load_audit()
    result, inputs, chinese = synthetic_result()
    attempt = result["strategies"]["zh"]["attempts"]["i1"]
    sample = result["strategies"]["zh"]["warm_samples"]["i1"][0]
    sample["ranking_override"] = deepcopy(attempt["rankings"])
    sample["ranking_override"][0]["score"] += 0.01
    semantic = {k: v for k, v in attempt.items() if k not in {"rankings", "timing"}}
    sample["evidence_sha256"] = fingerprint(
        {"semantics": semantic, "rankings": sample["ranking_override"]}
    )
    with pytest.raises(ValueError):
        audit.audit_result(result, inputs, chinese)


def test_source_text_cannot_be_replaced_by_a_coherent_saved_trace():
    result, inputs, chinese = synthetic_result()
    result["strategies"]["constrained_en"]["attempts"]["i2"]["constraint_trace"]["source_text"] = (
        "别的中文"
    )
    with pytest.raises(ValueError):
        load_audit().audit_result(result, inputs, chinese)


def test_existing_output_is_preserved_before_any_result_is_read(tmp_path):
    output = tmp_path / "audit.json"
    output.write_text("preserve evidence")
    with pytest.raises(FileExistsError):
        load_audit().main(["--base", "missing", "--quality", "missing", "--output", str(output)])
    assert output.read_text() == "preserve evidence"


def test_open_stdout_log_is_excluded_while_closed_evidence_remains_hashable(tmp_path):
    stable = tmp_path / "stable.log"
    stable.write_text("closed evidence")
    active = tmp_path / "active.log"
    script = Path(__file__).resolve().parents[1] / "examples/audit_fallback_benchmark.py"
    code = (
        "import importlib.util,json,sys; from pathlib import Path; "
        "s=importlib.util.spec_from_file_location('audit',sys.argv[1]); "
        "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
        "print(json.dumps(m.evidence_log_snapshot(Path(sys.argv[2]))))"
    )
    with active.open("w") as stream:
        result = subprocess.run(
            [sys.executable, "-c", code, str(script), str(tmp_path)],
            stdout=stream,
            stderr=subprocess.PIPE,
            text=True,
        )
    assert result.returncode == 0, result.stderr
    captured = json.loads(active.read_text())
    assert captured == {str(stable): hashlib.sha256(b"closed evidence").hexdigest()}
