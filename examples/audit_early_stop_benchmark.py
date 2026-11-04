"""Independent standard-library audit of frozen repetition rejection and fresh fallback."""

import argparse
import importlib.util
import json
import os
import tempfile
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "independent_fallback_audit", Path(__file__).with_name("audit_fallback_benchmark.py")
)
legacy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(legacy)
POLICIES = {
    "original_fusion": {"decoder": "original", "mode": "auto_fusion", "fallback": "error"},
    "fallback_opus_fusion": {"decoder": "constrained", "mode": "auto_fusion", "fallback": "opus"},
    "early_fallback_fusion": {"decoder": "guarded", "mode": "auto_fusion", "fallback": "opus"},
}
SELECTION_RULE = (
    "Exclude candidates rejecting any baseline-accepted dev output; maximize caught baseline "
    "failures, then higher beam quorum, lower mean trigger tokens, more copies, smaller maximum "
    "ngram, larger minimum tokens, lexical profile ID. If no viable candidate, freeze "
    "n8/max_declared_repeat/min32/q4 and mark nonviable."
)


def candidate_profiles(version):
    bounded = version == 2
    return [
        {
            "schema_version": version,
            "profile_id": f"bounded-n{n}-r{r}-min{m}-q{q}-tail4"
            if bounded
            else f"suffix-n{n}-r{r}-min{m}-q{q}",
            "algorithm": "bounded_tail_consecutive_token_cycles"
            if bounded
            else "suffix_consecutive_token_cycles",
            "max_new_tokens": 64,
            "num_beams": 4,
            "check_every": 4,
            "max_ngram_size": n,
            "repeat_count": r,
            "min_new_tokens": m,
            "min_repeating_beams": q,
            **({"min_cycle_tokens": 4, "tail_tolerance": 4} if bounded else {}),
        }
        for n, r, m, q in product(
            (8, 16) if bounded else (8, 12), (2, 3) if bounded else (3, 4), (24, 32), (3, 4)
        )
    ]


def detect_repetition(beams, profile, special_ids):
    """Find repeated blocks independently by comparing whole repeated slices."""
    if profile not in candidate_profiles(1) + candidate_profiles(2):
        raise ValueError("Profile is outside both declared development grids")
    if (
        len(beams) != 4
        or any(not isinstance(b, list) or not b for b in beams)
        or len({len(b) for b in beams}) != 1
        or any(type(t) is not int or t < 0 for b in beams for t in b)
    ):
        raise ValueError("Expected four equal, nonempty raw token beams")
    bounded, evidence = profile["schema_version"] == 2, []
    for index, beam in enumerate(beams):
        match = dict(beam_index=index, period=None, cycle=[], repeat_count=0, suffix_start=None)
        if bounded:
            match["tail_tokens"] = None
        for tail in range(profile.get("tail_tolerance", 0) + 1):
            end = len(beam) - tail
            for period in range(profile.get("min_cycle_tokens", 1), profile["max_ngram_size"] + 1):
                copies = profile["repeat_count"]
                if end - copies * period < 1:
                    continue
                cycle = beam[end - period : end]
                if (
                    set(cycle) & set(special_ids)
                    or beam[end - copies * period : end] != cycle * copies
                ):
                    continue
                counts = [
                    n
                    for n in range(copies, (end - 1) // period + 1)
                    if beam[end - n * period : end] == cycle * n
                ]
                count = max(counts)
                match.update(
                    period=period,
                    cycle=cycle,
                    repeat_count=count,
                    suffix_start=end - count * period,
                )
                if bounded:
                    match["tail_tokens"] = tail
                break
            if match["period"] is not None:
                break
        evidence.append(match)
    step, repeating = len(beams[0]) - 1, sum(e["period"] is not None for e in evidence)
    eligible = profile["min_new_tokens"] <= step < 64 and step % 4 == 0
    return dict(
        new_tokens=step,
        eligible=eligible,
        repeating_beams=repeating,
        triggered=eligible and repeating >= profile["min_repeating_beams"],
        beam_evidence=evidence,
    )


def validate_guard(trace, profile, c, label, observe_only=False):
    record = trace["repetition_stop"]
    c.require(profile in candidate_profiles(1) + candidate_profiles(2), label + ".declared_profile")
    c.require(
        all(type(v) is int for k, v in profile.items() if k not in {"profile_id", "algorithm"}),
        label + ".profile_integer_fields",
    )
    c.require(
        type(record["enabled"]) is bool and type(trace["constraints_applied"]) is bool,
        label + ".enabled_boolean",
    )
    for key, expected in (
        ("profile", profile),
        ("profile_sha256", legacy.fingerprint(profile)),
        ("observe_only", observe_only),
        ("enabled", trace["constraints_applied"]),
    ):
        c.same(record[key], expected, label + "." + key)
    snapshots, triggers = record["snapshots"], []
    for snapshot in snapshots:
        replay = detect_repetition(snapshot["beams"], profile, record["special_ids"])
        c.same(snapshot, {"beams": snapshot["beams"], **replay}, label + ".raw_beam_replay")
        c.require(replay["eligible"], label + ".eligible_check")
        if replay["triggered"]:
            triggers.append(replay["new_tokens"])
    count, observed = record["generated_new_tokens"], record["observed_generation_steps"]
    c.require(
        type(count) is int and 0 <= count <= 64 and count == len(trace["sequence"]) - 1,
        label + ".diagnostic_sequence_length",
    )
    c.require(type(observed) is int and 0 <= observed <= 64, label + ".observed_step_count")
    c.same(
        [s["new_tokens"] for s in snapshots],
        list(range(profile["min_new_tokens"], min(observed, 60) + 1, 4)),
        label + ".complete_check_coverage",
    )
    c.same(record["triggered"], bool(triggers), label + ".triggered")
    c.same(
        record["trigger_new_tokens"], triggers[0] if triggers else None, label + ".first_trigger"
    )
    aborted = bool(triggers) and not observe_only
    c.same(record["observed_steps_complete"], aborted, label + ".observation_scope")
    c.same(
        record["length_scope"],
        "partial_beam_sequence" if aborted else "finalized_sequence",
        label + ".length_scope",
    )
    if not record["enabled"]:
        c.same(observed, 0, label + ".disabled_observations")
    if aborted:
        c.same(triggers, [count], label + ".single_active_abort")
        c.same(observed, count, label + ".actual_abort_step")
        c.same(trace["sequence"], snapshots[-1]["beams"][0], label + ".rejected_first_beam")
        c.same(trace["status"], "failed", label + ".rejected_status")
        c.same(trace["eos_terminated"], False, label + ".no_synthetic_eos")
        c.same(
            trace["sequence_role"],
            "active_first_beam_diagnostic_not_final",
            label + ".diagnostic_role",
        )
        c.require(record["enabled"] and count < 64, label + ".real_early_abort")


def fraction(n, d):
    return dict(numerator=n, denominator=d, rate=n / d if d else None)


def rates(attempts):
    return {
        **legacy.request_rates(attempts),
        "early_rejection": fraction(sum(a["early_rejected"] for a in attempts), len(attempts)),
    }


def timing(samples):
    return {k: legacy.statistics([s["timing"][k] for s in samples]) for k in legacy.TIMES}


def early_analysis(strategies):
    previous, guarded = (
        strategies[p]["attempts"] for p in ("fallback_opus_fusion", "early_fallback_fusion")
    )
    old = {i for i, a in previous.items() if a["primary_failure"] is not None}
    rejected = {i for i, a in guarded.items() if a["early_rejected"]}
    groups = dict(
        legacy_failure=old,
        early_rejected=rejected,
        caught_legacy_failure=old & rejected,
        missed_legacy_failure=old - rejected,
        early_rejected_legacy_accepted=rejected - old,
    )
    return {
        **{k + "_ids": sorted(v) for k, v in groups.items()},
        "rates": dict(
            early_rejection=fraction(len(rejected), len(guarded)),
            caught_legacy_failure=fraction(len(old & rejected), len(old)),
            missed_legacy_failure=fraction(len(old - rejected), len(old)),
            early_rejected_legacy_accepted=fraction(len(rejected - old), len(guarded) - len(old)),
        ),
        "scored_primary_generated_tokens": {
            p: {
                i: len((a["constraint_trace"] or a["original_trace"])["sequence"]) - 1
                for i, a in s["attempts"].items()
            }
            for p, s in strategies.items()
        },
        "warm_timing_by_subset": {
            g: {
                p: timing([v for i in sorted(ids) for v in s["warm_samples"][i]])
                for p, s in strategies.items()
            }
            for g, ids in groups.items()
            if ids
        },
    }


def audit_result(result, inputs, chinese):
    c, summaries, reconstructed = legacy.Checks(), {}, {}
    c.same(result["evaluation_inputs"], inputs, "frozen_inputs")
    c.same(
        legacy.fingerprint(inputs), result["protocol"]["evaluation_inputs_sha256"], "input_digest"
    )
    c.same(result["protocol"]["policies"], POLICIES, "three_fusion_policies")
    c.same(sorted(result["strategies"]), sorted(POLICIES), "policy_set")
    gallery, qrels, atol = inputs["gallery_ids"], inputs["qrels"], result["protocol"]["score_atol"]
    for policy, strategy in result["strategies"].items():
        attempts, warm, ranks, maximum = strategy["attempts"], [], {}, 0.0
        c.same(sorted(attempts), sorted(qrels), policy + ".intents")
        c.same(sorted(strategy["warm_samples"]), sorted(qrels), policy + ".warm_intents")
        for intent, a in attempts.items():
            label = policy + "." + intent
            c.require(
                type(a["ok"]) is bool and type(a["early_rejected"]) is bool, label + ".booleans"
            )
            c.same(a["ok"], a["failure"] is None, label + ".terminal_status")
            legacy.validate_timing(a["timing"], c, label)
            primary, fallback = a["primary_failure"] is not None, a["fallback_triggered"]
            c.same(fallback, primary, label + ".fresh_fallback_required")
            original = policy == "original_fusion"
            c.same(a["original_trace"] is not None, original, label + ".original_control")
            c.same(a["constraint_trace"] is None, original, label + ".constraint_branch")
            if original:
                c.same(primary, False, label + ".no_primary_rejection")
            else:
                c.same(
                    a["constraint_trace"]["status"],
                    "failed" if primary else "success",
                    label + ".primary_status",
                )
            if policy == "early_fallback_fusion":
                validate_guard(a["constraint_trace"], result["repetition_profile"], c, label)
                c.same(
                    a["early_rejected"],
                    a["constraint_trace"]["repetition_stop"]["triggered"],
                    label + ".early_flag",
                )
                if a["early_rejected"]:
                    c.same(a["primary_failure"]["type"], "RepetitionAbort", label + ".abort_type")
            else:
                c.same(a["early_rejected"], False, label + ".unguarded")
            c.same(a["fallback_trace"] is not None, fallback, label + ".fallback_trace_presence")
            route = "original_fusion" if original or fallback else "constrained_fusion"
            c.same(a["effective_route"], route if a["ok"] else None, label + ".effective_route")
            c.same(
                a["query_texts"],
                [chinese[intent], a["translation"]] if a["ok"] else [],
                label + ".retrieved_texts",
            )
            final_trace = (
                a["original_trace"]
                if original
                else a["fallback_trace"]
                if fallback
                else a["constraint_trace"]
            )
            c.same(
                final_trace["status"],
                "success" if a["ok"] else "failed",
                label + ".final_trace_status",
            )
            if primary:
                c.require(
                    isinstance(a["constraint_trace"]["sequence"], list)
                    and bool(a["constraint_trace"]["sequence"])
                    and isinstance(a["constraint_trace"]["output"], str),
                    label + ".decoded_rejection",
                )
            for field in ("original_trace", "constraint_trace", "fallback_trace"):
                trace = a[field]
                if trace is not None:
                    c.same(trace["source_text"], chinese[intent], label + ".source")
                    c.require(trace["status"] in {"success", "failed"}, label + ".trace_status")
                    if trace["status"] == "success":
                        c.same(trace["output"], a["translation"], label + ".accepted_output")
            unused = ["constraint_translation_ms"] if original else ["original_translation_ms"]
            if not fallback:
                unused.append("fallback_translation_ms")
            for stage in unused:
                c.same(a["timing"][stage], 0.0, label + ".unused_stage")
            if a["ok"]:
                legacy.ranking_difference(
                    a["rankings"], a["rankings"], qrels[intent], gallery, c, label, 0
                )
            else:
                c.same(a["rankings"], None, label + ".failed_ranking")
            ranks[intent] = [r["image_id"] for r in a["rankings"]] if a["ok"] else None
            samples = strategy["warm_samples"][intent]
            c.same(len(samples), result["protocol"]["query_repeats"], label + ".repeats")
            for sample in samples:
                for key in (*legacy.STATUS, "early_rejected"):
                    c.same(sample[key], a[key], label + ".warm." + key)
                ranking = (
                    a["rankings"]
                    if sample["ranking_override"] is None
                    else sample["ranking_override"]
                )
                c.same(
                    sample["evidence_sha256"],
                    legacy.fingerprint({"semantics": legacy.semantics(a), "rankings": ranking}),
                    label + ".digest",
                )
                if a["ok"]:
                    maximum = max(
                        maximum,
                        legacy.ranking_difference(
                            ranking, a["rankings"], qrels[intent], gallery, c, label, atol
                        ),
                    )
                else:
                    c.same(ranking, None, label + ".failed_override")
                legacy.validate_timing(sample["timing"], c, label + ".warm")
                for stage in unused:
                    c.same(sample["timing"][stage], 0.0, label + ".warm_unused_stage")
                warm.append(sample)
        computed = {
            **legacy.retrieval_metrics(ranks, qrels, gallery),
            "rates": rates(list(attempts.values())),
            "warm_rates": rates(warm),
            "timing": timing(warm),
            "timing_by_branch": {
                b: timing([s for s in warm if s["fallback_triggered"] == flag])
                for b, flag in (("fallback_triggered", True), ("no_fallback", False))
                if any(s["fallback_triggered"] == flag for s in warm)
            },
            "max_repeated_absolute_score_difference": maximum,
            "rank_disagreements": 0,
        }
        for key, expected in computed.items():
            c.same(strategy[key], expected, policy + "." + key)
        summaries[policy], reconstructed[policy] = (
            computed,
            {i: legacy.semantics(a) for i, a in attempts.items()},
        )
    maximum = 0.0
    for policy in ("fallback_opus_fusion", "early_fallback_fusion"):
        for intent, actual in result["strategies"][policy]["attempts"].items():
            control = result["strategies"]["fallback_opus_fusion"]["attempts"][intent]
            if policy == "early_fallback_fusion" and not actual["early_rejected"]:
                c.same(
                    {k: v for k, v in actual["constraint_trace"].items() if k != "repetition_stop"},
                    control["constraint_trace"],
                    policy + ".unchanged_legacy_trace",
                )
                c.same(
                    actual["primary_failure"],
                    control["primary_failure"],
                    policy + ".unchanged_rejection",
                )
            if actual["fallback_triggered"]:
                control = result["strategies"]["original_fusion"]["attempts"][intent]
                c.same(
                    actual["fallback_trace"],
                    control["original_trace"],
                    policy + ".fresh_original_trace",
                )
            for key in ("ok", "translation", "query_texts", "effective_route"):
                c.same(actual[key], control[key], policy + ".control." + key)
            if actual["ok"]:
                maximum = max(
                    maximum,
                    legacy.ranking_difference(
                        actual["rankings"],
                        control["rankings"],
                        qrels[intent],
                        gallery,
                        c,
                        policy,
                        atol,
                    ),
                )
    branch = dict(
        checked=2 * len(qrels), max_absolute_score_difference=maximum, rank_disagreements=0
    )
    analysis = early_analysis(result["strategies"])
    c.same(result["request_semantics"], reconstructed, "request_semantics")
    c.same(result["branch_audit"], branch, "branch_audit")
    c.same(result["early_stop_analysis"], analysis, "early_stop_analysis")
    return dict(
        checks=c.count,
        numeric_comparisons=c.numeric,
        maximum_numeric_absolute_difference=c.max_error,
        branch_audit=branch,
        strategies=summaries,
        early_stop_analysis=analysis,
    )


def select_development(candidates, rows, special_ids):
    matrix = []
    for profile in candidates:
        triggers, accepted, caught = {}, [], []
        for row in rows:
            for snapshot in row["observer"]["trace"]["repetition_stop"]["snapshots"]:
                detected = detect_repetition(snapshot["beams"], profile, special_ids)
                if detected["new_tokens"] != snapshot["new_tokens"]:
                    raise ValueError("Development beam length differs from recorded step")
                if detected["triggered"]:
                    triggers[row["query_id"]] = detected["new_tokens"]
                    (accepted if row["baseline"]["ok"] else caught).append(row["query_id"])
                    break
        matrix.append(
            dict(
                profile=profile,
                trigger_new_tokens=triggers,
                early_rejected_baseline_accepted_ids=sorted(accepted),
                caught_baseline_failure_ids=sorted(caught),
                mean_failure_trigger_tokens=sum(triggers[i] for i in caught) / len(caught)
                if caught
                else None,
            )
        )
    viable = [
        r
        for r in matrix
        if not r["early_rejected_baseline_accepted_ids"] and r["caught_baseline_failure_ids"]
    ]
    if viable:

        def priority(row):
            p = row["profile"]
            return (
                -len(row["caught_baseline_failure_ids"]),
                -p["min_repeating_beams"],
                row["mean_failure_trigger_tokens"],
                -p["repeat_count"],
                p["max_ngram_size"],
                -p["min_new_tokens"],
                p["profile_id"],
            )

        selected = min(viable, key=priority)
    else:
        selected = next(
            r
            for r in matrix
            if (
                r["profile"]["max_ngram_size"],
                r["profile"]["repeat_count"],
                r["profile"]["min_new_tokens"],
                r["profile"]["min_repeating_beams"],
            )
            == (8, max(p["repeat_count"] for p in candidates), 32, 4)
        )
    return dict(
        viable=bool(viable),
        selection_rule=SELECTION_RULE,
        selected_profile=selected["profile"],
        selected_row=selected,
        candidates=sorted(matrix, key=lambda r: r["profile"]["profile_id"]),
        training_executed=False,
        formal_queries_or_retrieval_used=False,
    )


def audit_development(metadata, profile, code_hash, c):
    read = legacy.read_json
    queries, protocol, observed, selection, validation = (
        read(metadata / f"dev-{name}.json")
        for name in ("queries", "protocol", "observations", "selection", "validation")
    )
    query_rows, rows = queries["queries"], observed["rows"]
    c.same(queries["role"], "development_only", "dev.role")
    c.same(
        legacy.fingerprint(query_rows),
        "53bfaf65a00cae94a2c578907870c6694b36f3f50c85515038930d19c0de045b",
        "dev.fixed_source_strings",
    )
    c.same(
        [q["query_id"] for q in query_rows], [f"i{i:02}_zh" for i in range(1, 9)], "dev.eight_ids"
    )
    for key, expected in (
        ("queries_sha256", legacy.file_hash(metadata / "dev-queries.json")),
        ("candidate_profiles", candidate_profiles(2)),
        ("selection_rule", SELECTION_RULE),
        ("development_implementation_sha256", code_hash),
        ("training", False),
        ("formal_queries_or_retrieval_used", False),
    ):
        c.same(protocol[key], expected, "dev.protocol." + key)
    protocol_hash, observed_hash = (
        legacy.file_hash(metadata / name) for name in ("dev-protocol.json", "dev-observations.json")
    )
    c.same(observed["protocol"], protocol, "dev.observed_protocol")
    c.same(observed["protocol_sha256"], protocol_hash, "dev.protocol_hash")
    c.same(observed["parameters_require_grad"], False, "dev.frozen_parameters")
    c.same(observed["training_executed"], False, "dev.no_training")
    c.same(observed["development_only"], True, "dev.only")
    prior = protocol["prior_development"]
    c.same(prior["archive"], "development-stage1", "dev.preserved_stage1_directory")
    c.same(prior["source_commit"], "7fb0973", "dev.preserved_stage1_revision")
    for name, digest in prior["sha256"].items():
        c.same(legacy.file_hash(metadata / prior["archive"] / name), digest, "dev.stage1." + name)
    observer_profile = next(
        p
        for p in candidate_profiles(2)
        if (p["max_ngram_size"], p["repeat_count"], p["min_new_tokens"], p["min_repeating_beams"])
        == (16, 2, 24, 3)
    )
    for name, values in (("observed", rows), ("validation", validation["rows"])):
        c.same(
            [{k: r[k] for k in ("query_id", "text")} for r in values],
            query_rows,
            "dev." + name + ".sources",
        )
    for row in rows:
        baseline, observer = row["baseline"], row["observer"]
        c.same(baseline["trace"]["source_text"], row["text"], "dev.baseline_source")
        c.same(
            {k: v for k, v in observer["trace"].items() if k != "repetition_stop"},
            baseline["trace"],
            "dev.unchanged_observer_trace",
        )
        for key in ("ok", "english", "failure"):
            c.same(observer[key], baseline[key], "dev.observer." + key)
        c.same(
            observer["trace"]["repetition_stop"]["special_ids"],
            observed["special_ids"],
            "dev.special_ids",
        )
        validate_guard(observer["trace"], observer_profile, c, "dev.observer", observe_only=True)
    replayed = {
        **select_development(candidate_profiles(2), rows, observed["special_ids"]),
        "protocol_sha256": protocol_hash,
        "observations_sha256": observed_hash,
    }
    c.same(selection, replayed, "dev.recomputed_all_sixteen_candidates")
    c.same(profile, replayed["selected_profile"], "dev.selected_profile")
    for key, expected in (
        ("protocol_sha256", protocol_hash),
        ("selection_sha256", legacy.file_hash(metadata / "dev-selection.json")),
        ("profile", profile),
        ("profile_byte_sha256", legacy.file_hash(metadata / "repetition_profile_v1.json")),
        ("active_stops_match_replayed_predictions", True),
    ):
        c.same(validation[key], expected, "dev.validation." + key)
    for previous, row in zip(rows, validation["rows"]):
        for key in ("ok", "english", "failure", "trace"):
            c.same(row["baseline"][key], previous["baseline"][key], "dev.active_baseline." + key)
        guarded, backup = row["guarded"], row["fallback"]
        validate_guard(guarded["trace"], profile, c, "dev.active")
        guard = guarded["trace"]["repetition_stop"]
        c.same(guard["special_ids"], observed["special_ids"], "dev.active_specials")
        step = selection["selected_row"]["trigger_new_tokens"].get(row["query_id"])
        c.same(guard["trigger_new_tokens"], step, "dev.predicted_actual_step")
        c.same(guard["triggered"], step is not None, "dev.predicted_actual_abort")
        if step is None:
            c.same(
                {k: v for k, v in guarded["trace"].items() if k != "repetition_stop"},
                previous["baseline"]["trace"],
                "dev.unchanged_active_trace",
            )
        else:
            c.same(guarded["ok"], False, "dev.rejected_partial")
        c.same(backup is None, guarded["ok"], "dev.fresh_fallback_branch")
        if backup is not None:
            c.same(backup["trace"]["source_text"], row["text"], "dev.fallback_source")
    return dict(
        candidate_count=len(replayed["candidates"]),
        selected_profile=profile,
        selected_row=replayed["selected_row"],
        special_ids=observed["special_ids"],
    )


def _bound_file_hash(root, metadata, name):
    """Measured blob hash from the frozen-source snapshot when present, else the working tree."""
    if metadata is not None and (Path(metadata) / "frozen-source.json").is_file():
        return legacy.read_json(Path(metadata) / "frozen-source.json")["files"][name]
    return legacy.file_hash(root / "src/siglip_tools" / name)


def source_hashes(root, metadata=None):
    prior = legacy.source_hashes(root, metadata)
    files = {
        name: _bound_file_hash(root, metadata, name)
        for name in (
            "repetition_stop.py",
            "early_stop_development.py",
            "early_stop_benchmark.py",
            "repetition_profile_v1.json",
        )
    }
    return {
        "historical_files": prior["files"],
        "files": files,
        "previous_core_sha256": prior["implementation_sha256"],
        "implementation_sha256": legacy.fingerprint(
            {"previous_core": prior["implementation_sha256"], **files}
        ),
        "development_implementation_sha256": legacy.fingerprint(
            {
                "legacy": prior["implementation_sha256"],
                **{k: files[k] for k in ("repetition_stop.py", "early_stop_development.py")},
            }
        ),
    }


def bind_metadata(metadata, root, c):
    protocol, freeze = (legacy.read_json(metadata / n) for n in ("protocol.json", "freeze.json"))
    source = source_hashes(root, metadata)
    c.same(
        source["previous_core_sha256"],
        "e377efdeb1bcfe1ff44ac93eca1ee8489d1a5a2032fcba0eabbfdeae958f8675",
        "preserved_fallback_core",
    )
    c.same(legacy.file_hash(metadata / "protocol.json"), freeze["protocol_sha256"], "protocol_hash")
    for key in (
        "implementation_sha256",
        "dataset_hashes",
        "supporting_hashes",
        "development_hashes",
        "repetition_profile_sha256",
        "source_indexes",
        "retrieval_weights",
    ):
        c.same(freeze[key], protocol[key], "freeze." + key)
    c.same(source["implementation_sha256"], protocol["implementation_sha256"], "source_code")
    c.same(source["previous_core_sha256"], freeze["previous_core_sha256"], "freeze.preserved_core")
    bound = {n: legacy.file_hash(metadata / n) for n in ("protocol.json", "freeze.json")}
    c.same(
        sorted(protocol["development_hashes"]),
        sorted(
            f"dev-{n}.json"
            for n in ("queries", "protocol", "observations", "selection", "validation")
        ),
        "complete_development_files",
    )
    for files, prior in (
        ({k + ".json": v for k, v in protocol["dataset_hashes"].items()}, True),
        (protocol["supporting_hashes"], True),
        (protocol["development_hashes"], False),
        ({"repetition_profile_v1.json": protocol["repetition_profile_sha256"]}, False),
    ):
        for name, digest in files.items():
            bound[name] = legacy.file_hash(metadata / name)
            c.same(bound[name], digest, "input." + name)
            if prior:
                c.same(
                    bound[name],
                    legacy.file_hash(root / "benchmarks/terminology-constraints-v1" / name),
                    "unchanged_prior_input." + name,
                )
    c.same(
        legacy.file_hash(root / "src/siglip_tools/repetition_profile_v1.json"),
        protocol["repetition_profile_sha256"],
        "packaged_profile",
    )
    c.same(
        legacy.file_hash(root / "src/siglip_tools/power_glossary_v1.json"),
        protocol["glossary_sha256"],
        "packaged_glossary",
    )
    fixed = dict(
        policies=POLICIES,
        query_warmup=3,
        query_repeats=5,
        independent_intents=16,
        gallery_images=40,
        source_token_limit=256,
        max_new_tokens=64,
        num_beams=dict(original=1, constrained=4, guarded=4),
        do_sample=False,
        use_cache=True,
        dtype="float32",
        translation_text_cache=False,
        training=False,
        fusion_weights=[0.5, 0.5],
        score_atol=1e-5,
    )
    for key, expected in fixed.items():
        c.same(protocol[key], expected, "fixed_protocol." + key)
    profile = legacy.read_json(metadata / "repetition_profile_v1.json")
    development = audit_development(
        metadata, profile, source["development_implementation_sha256"], c
    )
    validation = legacy.read_json(metadata / "dev-validation.json")
    accepted = [r for r in validation["rows"] if r["baseline"]["ok"]]
    coverage = protocol["development_coverage"]
    c.same(
        coverage["baseline_accepted_ids"],
        [r["query_id"] for r in accepted],
        "dev.coverage.accepted",
    )
    c.same(
        coverage["accepted_ids_reaching_active_check"],
        [r["query_id"] for r in accepted if r["guarded"]["trace"]["repetition_stop"]["snapshots"]],
        "dev.coverage.active_checks",
    )
    development["coverage"] = coverage
    manifest, queries, labels = (
        legacy.read_json(metadata / f"{n}.json") for n in ("manifest", "queries", "qrels")
    )
    chinese = {q["intent_id"]: q["text"] for q in queries["queries"] if q["language"] == "zh"}
    inputs = dict(
        gallery_ids=[r["image_id"] for r in manifest["images"]],
        qrels={
            q["intent_id"]: labels["qrels"][q["query_id"]]
            for q in queries["queries"]
            if q["language"] == "zh"
        },
    )
    c.same(len(inputs["gallery_ids"]), 40, "gallery_count")
    c.same(len(set(inputs["gallery_ids"])), 40, "unique_gallery")
    c.same(len(chinese), 16, "sixteen_intents")
    c.same(
        legacy.fingerprint(inputs), protocol["evaluation_inputs_sha256"], "evaluation_input_digest"
    )
    return protocol, freeze, source, bound, profile, development, manifest, inputs, chinese


def bind_artifacts(protocol, root, manifest, c):
    artifacts = {}
    for row in manifest["images"]:
        path = root / "data/equipment-v1" / row["path"]
        artifacts[str(path)] = legacy.file_hash(path)
        c.same(artifacts[str(path)], row["sha256"], "image." + row["image_id"])
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
            path = directory / name
            artifacts[str(path)] = legacy.file_hash(path)
            c.same(artifacts[str(path)], expected, "artifact." + name)
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
        artifacts[str(path)] = legacy.file_hash(path)
        c.same(artifacts[str(path)], entry["sha256"], alias + ".weight_bytes")
        c.same(path.stat().st_size, entry["size_bytes"], alias + ".weight_size")
        index = root / "artifacts/equipment-benchmark" / f"{alias}-v1/index.npz"
        artifacts[str(index)] = legacy.file_hash(index)
        c.same(artifacts[str(index)], protocol["source_indexes"][alias], alias + ".source_index")
    return artifacts


def bind_identities(result, protocol, metadata, profile, c):
    provenance = {
        name: legacy.read_json(metadata / f"{name}-provenance.json")
        for name in ("translator", "decoder")
    }
    c.same(
        sorted(result["translator_identities"]),
        ["constrained", "guarded", "original"],
        "translator_variants",
    )
    for variant, identity in result["translator_identities"].items():
        for key, expected in (
            ("source_model", protocol["translation_model"]),
            ("source_revision", protocol["translation_revision"]),
            ("artifacts", provenance["translator"]["artifacts"]),
            ("source", provenance["translator"]["source"]),
            ("dtype", protocol["dtype"]),
            ("device", protocol["translator_device"]),
        ):
            c.same(identity[key], expected, "translator." + variant + "." + key)
        for key, expected in (
            ("num_beams", protocol["num_beams"][variant]),
            ("do_sample", False),
            ("max_new_tokens", 64),
            ("use_cache", True),
        ):
            c.same(identity["generation"][key], expected, "generation." + variant + "." + key)
        if variant != "original":
            c.same(identity["decoder"], provenance["decoder"], "decoder_identity")
            c.same(identity["glossary"]["sha256"], protocol["glossary_sha256"], "glossary_identity")
        if variant == "guarded":
            c.same(
                identity["repetition_stop"],
                dict(
                    profile=profile, profile_sha256=legacy.fingerprint(profile), observe_only=False
                ),
                "guarded_identity",
            )
    for name, key, filename in (
        ("translator", "translator_files", "translation_provenance.json"),
        ("decoder", "decoder_files", "decoder_provenance.json"),
    ):
        c.same(
            legacy.file_hash(metadata / f"{name}-provenance.json"),
            protocol[key][filename],
            name + ".provenance_bytes",
        )
    c.same(
        provenance["translator"]["artifacts"],
        {
            k: v
            for k, v in protocol["translator_files"].items()
            if k != "translation_provenance.json"
        },
        "translator_provenance_artifacts",
    )


def save_new(path, payload):
    """Publish complete JSON atomically, refusing a concurrent or preexisting output."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=".audit-", dir=path.parent)
    try:
        with os.fdopen(handle, "w") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "quality", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--metadata", type=Path, default=Path("benchmarks/repetition-early-stop-v1")
    )
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Audit output exists; choose a new path")
    c, report = legacy.Checks(), dict(schema_version=1, passed=False, failures=[])
    root, metadata = Path(__file__).resolve().parents[1], args.metadata.resolve()
    try:
        protocol, freeze, source, bound, profile, development, manifest, inputs, chinese = (
            bind_metadata(metadata, root, c)
        )
        results, audits, result_files = {}, {}, {}
        for alias in ("base", "quality"):
            path = getattr(args, alias)
            result = legacy.read_json(path)
            for key, expected in (
                ("model_alias", alias),
                ("protocol", protocol),
                ("protocol_sha256", freeze["protocol_sha256"]),
                ("implementation_sha256", protocol["implementation_sha256"]),
                ("source_index_sha256", protocol["source_indexes"][alias]),
                ("dataset_hashes", protocol["dataset_hashes"]),
                ("development_hashes", protocol["development_hashes"]),
                ("repetition_profile", profile),
                ("repetition_profile_sha256", protocol["repetition_profile_sha256"]),
                ("gallery_images", 40),
                ("independent_intents", 16),
                ("training_executed", False),
            ):
                c.same(result[key], expected, alias + "." + key)
            for key, expected in protocol["retrieval_models"][alias].items():
                c.same(result["model_identity"][key], expected, alias + ".identity." + key)
            bind_identities(result, protocol, metadata, profile, c)
            for attempt in result["strategies"]["early_fallback_fusion"]["attempts"].values():
                c.same(
                    attempt["constraint_trace"]["repetition_stop"]["special_ids"],
                    development["special_ids"],
                    alias + ".guard_special_ids",
                )
            audits[alias], results[alias] = audit_result(result, inputs, chinese), result
            result_files[alias] = dict(path=str(path.resolve()), sha256=legacy.file_hash(path))
        for key in ("request_semantics", "translator_identities", "environment"):
            c.same(results["base"][key], results["quality"][key], "cross_model." + key)
        c.same(
            results["base"]["model_identity"]["runtime_versions"],
            results["quality"]["model_identity"]["runtime_versions"],
            "cross_model.runtime_versions",
        )
        report.update(
            passed=True,
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            protocol=protocol,
            freeze=freeze,
            frozen_source_identifiers=source,
            input_and_support_sha256=bound,
            development_replay=development,
            local_artifact_sha256=bind_artifacts(protocol, root, manifest, c),
            result_files=result_files,
            models=audits,
            checks=c.count + sum(a["checks"] for a in audits.values()),
            numeric_comparisons=c.numeric + sum(a["numeric_comparisons"] for a in audits.values()),
            maximum_numeric_absolute_difference=max(
                [c.max_error] + [a["maximum_numeric_absolute_difference"] for a in audits.values()]
            ),
            evidence_file_sha256=legacy.evidence_log_snapshot(
                root / "artifacts/repetition-early-stop"
            ),
            evidence_log_scope="Closed logs only; current stdout/stderr inode excluded (SIG-043).",
            auditor_sha256=legacy.file_hash(__file__),
            helper_sha256=legacy.file_hash(legacy.__file__),
            limits=[
                "Independent recorded-data recomputation; no neural replay, training "
                "or human semantic validation.",
                "The diagnosed sixteen intents and gallery are not a new independent holdout.",
                "Warm semantics reconstruct from scored attempts plus ranking overrides.",
                "Finalized token lengths differ from observed generation callback steps.",
                "Development cases without eligible checks do not test long-query false rejection.",
            ],
        )
    except (ValueError, KeyError, TypeError, OSError, IndexError) as exc:
        report["failures"].append(f"{type(exc).__name__}: {exc}")
        report.update(
            passed=False,
            completed_top_level_checks=c.count,
            completed_numeric_comparisons=c.numeric,
        )
    save_new(args.output, report)
    print(
        json.dumps(
            dict(passed=report["passed"], failures=report["failures"], output=str(args.output))
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
