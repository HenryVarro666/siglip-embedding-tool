"""Independent standard-library replay of retained-EOS protection and four controls.

This audits recorded consistency. It neither imports inference/scoring code nor
replays a pretrained model. Tensor origin and decoded text remain recorded
claims, bounded by pinned source/artifact hashes and scorer characterization.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "independent_previous_guard_audit", Path(__file__).with_name("audit_early_stop_benchmark.py")
)
previous = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(previous)
legacy = previous.legacy
Checks = legacy.Checks
save_new = previous.save_new
evidence_log_snapshot = legacy.evidence_log_snapshot

PREVIOUS_CORE_SHA256 = "5508fc7372887770f584065949d19debe77b81fd6d09cf022a432de164d7bc12"
PREVIOUS_PROTOCOL_SHA256 = "cfdc70f0d5c75bb5cd2fb242c5eaed033b7b7c975f72f273028659c13ab59229"
RULE_BYTE_SHA256 = "a8fb1d276fc53ace8c59b76dabd13a24035c323aa637f184c768b20db2464cdc"
PROFILE_BYTE_SHA256 = "e93ee35b4b81fe94a7b2d97c25fa3317bce599e6976d1a54a6a2d90d764b5777"
DEVELOPMENT_ROWS_SHA256 = "2917c1aea48f166ad2af5c07664b979902fe11ef6ac59e28843fc09744e9919c"
POLICIES = {
    **previous.POLICIES,
    "protected_fallback_fusion": {
        "decoder": "protected",
        "mode": "auto_fusion",
        "fallback": "opus",
    },
}
FAILURE_POLICY = (
    "Only current decoded constraint-output rejection triggers fresh original beam1 OPUS; "
    "no cascade or rejected-English retrieval; failed intents score zero. Frozen repetition "
    "rejection may reject a legacy accepted output and must be reported. A retained genuine "
    "EOS-completed candidate passing full original lexical and nonempty checks suppresses "
    "rejection; protection does not imply final output success."
)
_VALIDATION_FLAGS = {"tokens_satisfied", "phrase_satisfied", "satisfied"}
DEVELOPMENT_FILES = (
    "dev-queries.json",
    "dev-protocol.json",
    "dev-observations.json",
    "dev-validation.json",
)
DATA_HASHES = {
    "manifest": "3487fdf62b90971619ec098b2ec5bbad00e5af2d22ec9183d258bc1a24916975",
    "queries": "19a1da3d4fa9d961f379af0d56dad9274e698895af0d2800d35ad44dfad445db",
    "qrels": "0a6c9486084c5108439f642f4440d8ec4191ff8221a2ee0c553f8589f12daf95",
}
SUPPORT_HASHES = {
    "annotation-notes.json": "717d4ff5bcaeeadb395f14109fb1e6438db2eb5497cb740b5e68d188ed170c6d",
    "review-rules.json": "1feccdda7885c3d6f830c63b5613fbf23dbae8b22e9b09a5da839eb3e859829a",
    "glossary.json": "20fef944133fe8453e9b60d5d7a6ccabb93ce3a4ffe0565038caaf8f51e3c42e",
    "label-review.json": "f721ca9cc296ccf2fec0c2aa10be121982f85736c69e1242e83229c1803460e7",
}


def expected_rule():
    path = Path(__file__).resolve().parents[1] / "src/siglip_tools/completed_protection_v1.json"
    if legacy.file_hash(path) != RULE_BYTE_SHA256:
        raise ValueError("Packaged protection rule differs from the predeclared v1 byte hash")
    return legacy.read_json(path)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def token_list(value, nonempty=True):
    return (
        isinstance(value, list)
        and (bool(value) or not nonempty)
        and all(type(item) is int and item >= 0 for item in value)
    )


def lexical_check(prefix, requirements, output, c, label):
    """Use full original alternatives, excluding the decoder prompt from tokens."""
    c.require(token_list(prefix), label + ".candidate_tokens")
    c.require(isinstance(output, str), label + ".decoded_text")
    c.require(isinstance(requirements, list), label + ".original_requirements")
    checked = []
    for index, requirement in enumerate(requirements):
        original = {k: v for k, v in requirement.items() if k not in _VALIDATION_FLAGS}
        alternatives, forms = original["token_alternatives"], original["target_forms"]
        c.require(
            isinstance(alternatives, list)
            and bool(alternatives)
            and all(token_list(tokens) for tokens in alternatives),
            label + f".requirement{index}.full_alternatives",
        )
        c.require(
            isinstance(forms, list)
            and bool(forms)
            and all(isinstance(form, str) and form for form in forms),
            label + f".requirement{index}.target_forms",
        )
        available = prefix[1:]
        token_ok = any(
            any(
                available[start : start + len(tokens)] == tokens
                for start in range(len(available) - len(tokens) + 1)
            )
            for tokens in alternatives
        )
        phrase_ok = any(
            re.search(r"(?<!\w)" + re.escape(form) + r"(?!\w)", output) is not None
            for form in forms
        )
        checked.append(
            {
                **original,
                "tokens_satisfied": token_ok,
                "phrase_satisfied": phrase_ok,
                "satisfied": token_ok and phrase_ok,
            }
        )
    satisfied, nonempty = all(r["satisfied"] for r in checked), bool(output.strip())
    return {
        "output": output,
        "nonempty": nonempty,
        "requirements": checked,
        "satisfied": satisfied,
        "valid": nonempty and satisfied,
    }


def eos_ids_from_identity(identity, c, label):
    ids = identity["generation"]["eos_token_id"]
    if type(ids) is int:
        ids = [ids]
    c.require(token_list(ids) and len(ids) == len(set(ids)), label + ".configured_eos_ids")
    return sorted(ids)


def source_requirements(text, glossary):
    """Replay frozen longest-source selection with context and excluded spans."""
    occupied, matched = set(), {}
    sources = [(source, entry) for entry in glossary["entries"] for source in entry["sources"]]
    for source, entry in sorted(sources, key=lambda pair: -len(pair[0])):
        if entry.get("required_context", "") not in text:
            continue
        excluded = set()
        for phrase in entry.get("exclude_within", []):
            for match in re.finditer("(?=" + re.escape(phrase) + ")", text):
                excluded.update(range(match.start(), match.start() + len(phrase)))
        for match in re.finditer("(?=" + re.escape(source) + ")", text):
            positions = set(range(match.start(), match.start() + len(source)))
            if positions.isdisjoint(occupied | excluded):
                occupied.update(positions)
                matched.setdefault(entry["id"], []).append((match.start(), source))
    return [
        {
            "id": entry["id"],
            "source_terms": list(dict.fromkeys(term for _, term in sorted(matched[entry["id"]]))),
            "target_forms": entry["target_forms"]
            + [form[:1].upper() + form[1:] for form in entry["target_forms"]],
        }
        for entry in glossary["entries"]
        if entry["id"] in matched
    ]


def bind_trace_requirements(trace, glossary, special_ids, c, label):
    expected = source_requirements(trace["source_text"], glossary)
    actual = trace["requirements"]
    c.same(
        [{key: r[key] for key in ("id", "source_terms", "target_forms")} for r in actual],
        expected,
        label + ".all_source_glossary_requirements",
    )
    c.same(trace["constraints_applied"], bool(expected), label + ".source_requires_constraints")
    for requirement in actual:
        alternatives = requirement["token_alternatives"]
        c.require(
            isinstance(alternatives, list)
            and len(alternatives) == len(requirement["target_forms"])
            and all(
                token_list(tokens) and not set(tokens) & set(special_ids) for tokens in alternatives
            ),
            label + ".full_nonspecial_alternatives",
        )
        unique = list(dict.fromkeys(tuple(tokens) for tokens in alternatives))
        decoder = [
            list(tokens)
            for tokens in unique
            if not any(
                len(other) < len(tokens) and tokens[: len(other)] == other for other in unique
            )
        ]
        c.same(
            requirement["decoder_token_alternatives"],
            decoder,
            label + ".unchanged_decoder_alternatives",
        )
    c.same(
        trace["required_token_alternatives"],
        [r["token_alternatives"] for r in actual],
        label + ".complete_original_token_requirements",
    )


def validate_final_acceptance(trace, eos_ids, c, label, constrained=True, aborted=False):
    """Recompute final acceptance independently of candidate protection or flags."""
    sequence, output = trace["sequence"], trace["output"]
    c.require(token_list(sequence), label + ".generated_sequence")
    c.require(isinstance(output, str), label + ".decoded_output")
    ended = any(token in eos_ids for token in sequence[1:])
    c.same(trace["eos_terminated"], ended, label + ".actual_eos_in_generated_tokens")
    if aborted:
        c.same(trace["status"], "failed", label + ".aborted_not_accepted")
        c.same(ended, False, label + ".active_prefix_without_eos")
        return
    if constrained:
        requirements = trace["requirements"]
        c.same(trace["constraints_applied"], bool(requirements), label + ".requirements_applied")
        c.same(
            trace["required_token_alternatives"],
            [r["token_alternatives"] for r in requirements],
            label + ".all_original_alternatives",
        )
        replay = lexical_check(sequence, requirements, output, c, label + ".final_lexical")
        c.same(requirements, replay["requirements"], label + ".final_full_lexical_flags")
        c.same(trace["satisfied"], replay["satisfied"], label + ".final_satisfied")
        accepted = ended and replay["valid"]
    else:
        raw = trace["raw_decoded_output"]
        c.require(isinstance(raw, str), label + ".raw_original_output")
        accepted = ended and bool(raw.strip())
        c.same(output, raw.strip() if accepted else raw, label + ".original_trim_contract")
    c.same(
        trace["status"], "success" if accepted else "failed", label + ".recomputed_final_acceptance"
    )


def _validate_protected_guard(trace, c, label):
    stop = trace["repetition_stop"]
    rule, evidence = expected_rule(), stop["candidate_protection"]
    profile = rule["repetition_profile"]
    for key, expected in (
        ("profile", profile),
        ("profile_sha256", legacy.fingerprint(profile)),
        ("observe_only", False),
        ("enabled", trace["constraints_applied"]),
    ):
        c.same(stop[key], expected, label + "." + key)
    c.require(type(stop["enabled"]) is bool, label + ".enabled_boolean")
    c.same(stop["enabled"], bool(trace["requirements"]), label + ".full_original_requirements")
    for key, expected in (
        ("schema_version", 1),
        ("rule", rule),
        ("rule_sha256", legacy.fingerprint(rule)),
        ("source", "request_local_scorer_process_selected_eos"),
        ("inspection", "raw_trigger_only"),
        ("decoder_prompt_len", 1),
        ("batch_size", 1),
        ("num_beams", 4),
    ):
        c.same(evidence[key], expected, label + ".observer." + key)
    eos_ids = evidence["eos_token_ids"]
    c.require(token_list(eos_ids) and eos_ids == sorted(set(eos_ids)), label + ".eos_identity")
    c.require(type(evidence["decoder_start_token_id"]) is int, label + ".prompt_identity")
    c.require(type(evidence["finalize_entered"]) is bool, label + ".finalize_boolean")
    if stop["enabled"]:
        c.require(finite(evidence["length_penalty"]), label + ".finite_length_penalty")
    events, history = evidence["eos_candidates"], evidence["pool_history"]
    c.require(isinstance(events, list) and isinstance(history, list), label + ".observer_lists")
    c.require(len(history) <= 64, label + ".process_budget")
    introduced = {}
    for candidate_id, event in enumerate(events, 1):
        prefix, step, score = event["prefix"], event["eos_step"], event["normalized_score"]
        c.same(event["candidate_id"], candidate_id, label + ".candidate_id")
        c.require(token_list(prefix), label + ".candidate_prefix")
        c.same(prefix[0], evidence["decoder_start_token_id"], label + ".single_prompt")
        c.require(not set(prefix[1:]) & set(eos_ids), label + ".no_eos_inside_active_prefix")
        c.require(type(step) is int and 1 <= step <= 64, label + ".completion_step")
        c.same(len(prefix), step, label + ".eos_prefix_length")
        c.require(finite(score), label + ".normalized_score_finite")
        witnesses = event["witnesses"]
        c.require(isinstance(witnesses, list) and bool(witnesses), label + ".selected_eos_witness")
        ranks = [w["rank"] for w in witnesses]
        c.require(
            all(type(rank) is int for rank in ranks) and ranks == sorted(set(ranks)),
            label + ".unique_ordered_selected_ranks",
        )
        witness_keys = set()
        for witness in witnesses:
            for key, expected in (
                ("phase", "process"),
                ("process_new_tokens", step),
                ("parent_prefix", prefix),
                ("generated_len", step),
                ("beam_indices", event["beam_indices"]),
            ):
                c.same(witness[key], expected, label + ".witness." + key)
            c.require(
                type(witness["rank"]) is int and 0 <= witness["rank"] < 4,
                label + ".selected_eos_rank",
            )
            parent = witness["parent_beam_index"]
            c.require(type(parent) is int and 0 <= parent < 4, label + ".parent_beam")
            c.require(
                type(witness["eos_token_id"]) is int and witness["eos_token_id"] in eos_ids,
                label + ".genuine_eos_token",
            )
            c.require(finite(witness["next_score"]), label + ".proposal_score_finite")
            normalized = witness["next_score"] / (step ** evidence["length_penalty"])
            c.same(score, normalized, label + ".score_length_normalization")
            c.require(
                math.isclose(score, normalized, rel_tol=1e-12, abs_tol=1e-12),
                label + ".exact_scorer_normalization_tolerance",
            )
            ancestry = witness["beam_indices"]
            if ancestry is not None:
                c.require(
                    isinstance(ancestry, list)
                    and len(ancestry) == step
                    and all(type(a) is int and 0 <= a < 4 for a in ancestry),
                    label + ".complete_ancestry",
                )
                c.same(ancestry[-1], parent, label + ".ancestry_parent")
            key = legacy.fingerprint(witness)
            c.require(key not in witness_keys, label + ".duplicate_witness")
            witness_keys.add(key)
        c.same(
            event["eos_token_ids"],
            sorted({w["eos_token_id"] for w in witnesses}),
            label + ".actual_eos_proposals",
        )
        introduced.setdefault(step, []).append(candidate_id)
    known, retired, previous_pool, pools = set(), set(), [], {}
    for step, record in enumerate(history, 1):
        c.same(record["new_tokens"], step, label + ".consecutive_process_step")
        c.same(record["phase"], "process", label + ".pool_origin")
        pool = record["pool_candidate_ids"]
        c.require(
            isinstance(pool, list)
            and len(pool) <= 4
            and all(type(i) is int for i in pool)
            and len(pool) == len(set(pool)),
            label + ".current_pool_ids",
        )
        new = set(introduced.get(step, []))
        known.update(new)
        c.require(new <= set(pool), label + ".newly_retained_completion")
        c.require(set(pool) <= known, label + ".no_future_or_unknown_completion")
        retired.update(set(previous_pool) - set(pool))
        c.require(not set(pool) & retired, label + ".evicted_completion_cannot_return")
        previous_pool, pools[step] = pool, pool
    c.same(known, set(range(1, len(events) + 1)), label + ".all_events_retained_in_process")
    observed, count = stop["observed_generation_steps"], stop["generated_new_tokens"]
    c.require(type(observed) is int and 0 <= observed <= 64, label + ".observed_loop_steps")
    c.require(token_list(trace["sequence"]), label + ".diagnostic_sequence")
    c.require(type(count) is int and 0 <= count <= 64, label + ".diagnostic_count")
    c.same(count, len(trace["sequence"]) - 1, label + ".diagnostic_sequence_length")
    if stop["enabled"]:
        done = evidence["scorer_done_before_finalize"]
        if stop["triggered"]:
            c.same(done, None, label + ".abort_without_finalizer")
            c.same(observed, len(history), label + ".abort_observed_every_process")
        else:
            # Only scorer.is_done can short-circuit the final stopping callback.
            c.require(type(done) is bool, label + ".recorded_terminal_scorer_state")
            c.same(observed, len(history) - int(done), label + ".exact_observed_process_count")
    else:
        c.same(observed, 0, label + ".disabled_observed_steps")
        c.same(history, [], label + ".disabled_history")
        c.same(events, [], label + ".disabled_events")
        c.same(evidence["scorer_done_before_finalize"], None, label + ".disabled_terminal_state")
    snapshots = stop["snapshots"]
    c.require(isinstance(snapshots, list), label + ".snapshots")
    c.same(
        [s["new_tokens"] for s in snapshots],
        list(range(32, min(observed, 60) + 1, 4)),
        label + ".complete_check_coverage",
    )
    raw_triggers, protected_steps, aborts = [], [], []
    for snapshot in snapshots:
        replay = previous.detect_repetition(snapshot["beams"], profile, stop["special_ids"])
        c.same(
            {k: snapshot[k] for k in ("beams", *replay)},
            {"beams": snapshot["beams"], **replay},
            label + ".raw_detector_replay",
        )
        c.require(replay["eligible"], label + ".eligible_check")
        step, raw = replay["new_tokens"], replay["triggered"]
        pool = pools[step]
        c.same(snapshot["pool_candidate_ids"], pool, label + ".current_retained_pool")
        c.same(
            snapshot["candidate_pool"],
            "inspected" if raw else "not_requested_no_raw_trigger",
            label + ".lazy_inspection",
        )
        completed = snapshot["completed_candidates"]
        c.same(len(completed), len(pool) if raw else 0, label + ".inspect_complete_live_pool")
        valid = False
        for rank, checked in enumerate(completed):
            candidate_id = pool[rank]
            for key, expected in (
                ("candidate_id", candidate_id),
                ("pool_rank", rank),
                ("current_step", step),
                ("retained", True),
                ("phase", "process"),
            ):
                c.same(checked[key], expected, label + ".inspection." + key)
            event = events[candidate_id - 1]
            validation = lexical_check(
                event["prefix"], trace["requirements"], checked["validation"]["output"], c, label
            )
            c.same(checked["validation"], validation, label + ".original_full_lexical_validation")
            valid |= validation["valid"]
        protected, abort = raw and valid, raw and not valid
        c.same(snapshot["protected"], protected, label + ".candidate_protection_decision")
        c.same(snapshot["abort"], abort, label + ".actual_abort_decision")
        if raw:
            raw_triggers.append(step)
        if protected:
            protected_steps.append(step)
        if abort:
            aborts.append(step)
    c.same(stop["protected_trigger_steps"], protected_steps, label + ".protected_steps")
    c.same(stop["triggered"], bool(aborts), label + ".actual_top_abort")
    c.same(stop["trigger_new_tokens"], aborts[0] if aborts else None, label + ".first_actual_abort")
    c.same(stop["observed_steps_complete"], bool(aborts), label + ".observation_scope")
    c.same(
        stop["length_scope"],
        "partial_beam_sequence" if aborts else "finalized_sequence",
        label + ".length_scope",
    )
    c.same(
        evidence["finalize_entered"],
        bool(stop["enabled"] and not aborts),
        label + ".real_finalization_phase",
    )
    if aborts:
        c.same(aborts, [observed], label + ".single_immediate_abort")
        c.same(len(history), observed, label + ".abort_before_finalize")
        c.same(count, observed, label + ".abort_diagnostic_length")
        c.same(trace["sequence"], snapshots[-1]["beams"][0], label + ".rejected_active_beam")
        c.same(trace["status"], "failed", label + ".abort_status")
        c.same(trace["eos_terminated"], False, label + ".no_fabricated_final_eos")
        c.same(
            trace["sequence_role"],
            "active_first_beam_diagnostic_not_final",
            label + ".diagnostic_role",
        )
        c.require(observed < 64, label + ".real_early_abort")
    validate_final_acceptance(trace, eos_ids, c, label + ".final", aborted=bool(aborts))
    return {
        "raw_trigger_steps": raw_triggers,
        "protected_trigger_steps": protected_steps,
        "abort_step": aborts[0] if aborts else None,
        "eos_candidates": len(events),
    }


def validate_protected_guard(trace, checks=None, label="protected"):
    c = checks if checks is not None else Checks()
    try:
        return _validate_protected_guard(trace, c, label)
    except (KeyError, TypeError, IndexError, OverflowError, ZeroDivisionError) as exc:
        raise ValueError(f"{label}: malformed protection evidence: {exc}") from exc


def rates(rows):
    return {
        **previous.rates(rows),
        "candidate_protection": previous.fraction(
            sum(a["candidate_protected"] for a in rows), len(rows)
        ),
    }


def protection_analysis(strategies):
    legacy_rows = strategies["fallback_opus_fusion"]["attempts"]
    old = strategies["early_fallback_fusion"]["attempts"]
    new = strategies["protected_fallback_fusion"]["attempts"]
    failed = {i for i in legacy_rows if legacy_rows[i]["primary_failure"] is not None}
    old_abort = {i for i in old if old[i]["early_rejected"]}
    new_abort = {i for i in new if new[i]["early_rejected"]}
    protected = {i for i in new if new[i]["candidate_protected"]}
    accepted = {i for i in new if new[i]["primary_failure"] is None}
    groups = {
        "legacy_failure": failed,
        "old_early_rejected": old_abort,
        "protected_early_rejected": new_abort,
        "restored_old_accepted": (old_abort - failed) & accepted,
        "candidate_protected": protected,
        "remaining_legacy_failure": failed - new_abort,
        "new_early_rejected_legacy_accepted": new_abort - failed,
        "protected_primary_accepted": accepted,
        "protection_without_primary_success": protected - accepted,
    }
    timing = {
        group: {
            p: previous.timing([v for i in sorted(ids) for v in s["warm_samples"][i]])
            for p, s in strategies.items()
        }
        for group, ids in {"all": set(legacy_rows), **groups}.items()
        if ids
    }
    old_cost = timing["all"]["early_fallback_fusion"]
    new_cost = timing["all"]["protected_fallback_fusion"]
    return {
        **{key + "_ids": sorted(ids) for key, ids in groups.items()},
        "rates": {
            "candidate_protection": previous.fraction(len(protected), len(new)),
            "restored_old_accepted": previous.fraction(
                len(groups["restored_old_accepted"]), len(old_abort - failed)
            ),
            "remaining_legacy_failure": previous.fraction(len(failed - new_abort), len(failed)),
            "new_early_rejected_legacy_accepted": previous.fraction(
                len(new_abort - failed), len(new) - len(failed)
            ),
        },
        "scored_primary_generated_tokens": {
            p: {
                i: len((a["constraint_trace"] or a["original_trace"])["sequence"]) - 1
                for i, a in s["attempts"].items()
            }
            for p, s in strategies.items()
        },
        "warm_timing_by_subset": timing,
        "protected_vs_old_timing": {
            "old_early_fallback_fusion": old_cost,
            "protected_fallback_fusion": new_cost,
            "protected_minus_old_ms": {
                key: {
                    stat: new_cost[key][stat] - old_cost[key][stat]
                    for stat in ("mean", "p50", "p95")
                }
                for key in legacy.TIMES
            },
        },
    }


def audit_result(result, inputs, chinese, checks=None):
    c = checks if checks is not None else Checks()
    summaries, reconstructed, guard_replays = {}, {}, {}
    c.same(result["evaluation_inputs"], inputs, "frozen_inputs")
    c.same(
        legacy.fingerprint(inputs), result["protocol"]["evaluation_inputs_sha256"], "input_digest"
    )
    c.same(result["protocol"]["policies"], POLICIES, "four_fusion_policies")
    c.same(list(result["protocol"]["policies"]), list(POLICIES), "declared_policy_order")
    c.same(sorted(result["strategies"]), sorted(POLICIES), "strategy_set")
    c.same(result["repetition_profile"], expected_rule()["repetition_profile"], "fixed_profile")
    c.same(result["protection_rule"], expected_rule(), "fixed_protection_rule")
    c.same(result["protection_rule_sha256"], RULE_BYTE_SHA256, "rule_byte_hash")
    gallery, qrels = inputs["gallery_ids"], inputs["qrels"]
    atol, scored_count, warm_count = result["protocol"]["score_atol"], 0, 0
    c.require(finite(atol) and 0 <= atol <= 1e-5, "bounded_score_tolerance")
    for policy, strategy in result["strategies"].items():
        attempts, warm, ranks, maximum = strategy["attempts"], [], {}, 0.0
        c.same(sorted(attempts), sorted(qrels), policy + ".intents")
        c.same(sorted(strategy["warm_samples"]), sorted(qrels), policy + ".warm_intents")
        for intent, a in attempts.items():
            label = policy + "." + intent
            scored_count += 1
            c.require(
                all(
                    type(a[k]) is bool
                    for k in ("ok", "early_rejected", "candidate_protected", "fallback_triggered")
                ),
                label + ".boolean_status",
            )
            c.same(a["ok"], a["failure"] is None, label + ".terminal_status")
            legacy.validate_timing(a["timing"], c, label)
            primary, fallback = a["primary_failure"] is not None, a["fallback_triggered"]
            c.same(fallback, primary, label + ".fresh_original_fallback_required")
            original = policy == "original_fusion"
            c.same(a["original_trace"] is not None, original, label + ".original_control")
            c.same(a["constraint_trace"] is None, original, label + ".constraint_branch")
            if original:
                c.same(primary, False, label + ".no_constraint_rejection")
            else:
                c.same(
                    a["constraint_trace"]["status"],
                    "failed" if primary else "success",
                    label + ".primary_status",
                )
            if policy in {"early_fallback_fusion", "protected_fallback_fusion"}:
                trace = a["constraint_trace"]
                if policy == "protected_fallback_fusion":
                    guard_replays[intent] = validate_protected_guard(trace, c, label)
                else:
                    previous.validate_guard(trace, result["repetition_profile"], c, label)
                c.same(
                    a["early_rejected"],
                    trace["repetition_stop"]["triggered"],
                    label + ".actual_early_abort_flag",
                )
                if a["early_rejected"]:
                    c.same(
                        a["primary_failure"]["type"],
                        "RepetitionAbort",
                        label + ".dedicated_abort_type",
                    )
            else:
                c.same(a["early_rejected"], False, label + ".no_unguarded_abort")
            c.same(
                a["candidate_protected"],
                bool(a["constraint_trace"]["repetition_stop"]["protected_trigger_steps"])
                if policy == "protected_fallback_fusion"
                else False,
                label + ".actual_candidate_protection_flag",
            )
            c.same(a["fallback_trace"] is not None, fallback, label + ".fallback_trace_presence")
            route = "original_fusion" if original or fallback else "constrained_fusion"
            c.same(a["effective_route"], route if a["ok"] else None, label + ".effective_route")
            c.same(
                a["query_texts"],
                [chinese[intent], a["translation"]] if a["ok"] else [],
                label + ".actual_retrieval_texts",
            )
            final_trace = (
                a["original_trace"]
                if original
                else a["fallback_trace"]
                if fallback
                else a["constraint_trace"]
            )
            c.same(
                final_trace["status"], "success" if a["ok"] else "failed", label + ".final_status"
            )
            if primary:
                c.require(
                    token_list(a["constraint_trace"]["sequence"])
                    and isinstance(a["constraint_trace"]["output"], str),
                    label + ".decoded_rejection_preserved",
                )
            for field in ("original_trace", "constraint_trace", "fallback_trace"):
                trace = a[field]
                if trace is not None:
                    c.same(trace["source_text"], chinese[intent], label + ".source")
                    c.require(trace["status"] in {"success", "failed"}, label + ".trace_status")
                    if trace["status"] == "success":
                        c.same(trace["output"], a["translation"], label + ".accepted_output")
                    variant = (
                        "original" if field != "constraint_trace" else POLICIES[policy]["decoder"]
                    )
                    eos_ids = eos_ids_from_identity(
                        result["translator_identities"][variant], c, label
                    )
                    validate_final_acceptance(
                        trace,
                        eos_ids,
                        c,
                        label + "." + field,
                        constrained=field == "constraint_trace",
                        aborted=field == "constraint_trace" and a["early_rejected"],
                    )
                    if policy == "protected_fallback_fusion" and field == "constraint_trace":
                        c.same(
                            trace["repetition_stop"]["candidate_protection"]["eos_token_ids"],
                            eos_ids,
                            label + ".observer_configured_eos_identity",
                        )
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
            c.same(len(samples), result["protocol"]["query_repeats"], label + ".full_warm_rounds")
            for sample in samples:
                for key in (*legacy.STATUS, "early_rejected", "candidate_protected"):
                    c.same(sample[key], a[key], label + ".warm." + key)
                ranking = (
                    a["rankings"]
                    if sample["ranking_override"] is None
                    else sample["ranking_override"]
                )
                c.same(
                    sample["evidence_sha256"],
                    legacy.fingerprint({"semantics": legacy.semantics(a), "rankings": ranking}),
                    label + ".warm_digest",
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
                warm_count += 1
        computed = {
            **legacy.retrieval_metrics(ranks, qrels, gallery),
            "rates": rates(list(attempts.values())),
            "warm_rates": rates(warm),
            "timing": previous.timing(warm),
            "timing_by_branch": {
                b: previous.timing([s for s in warm if s["fallback_triggered"] == flag])
                for b, flag in (("fallback_triggered", True), ("no_fallback", False))
                if any(s["fallback_triggered"] == flag for s in warm)
            },
            "max_repeated_absolute_score_difference": maximum,
            "rank_disagreements": 0,
        }
        for key, expected in computed.items():
            c.same(strategy[key], expected, policy + "." + key)
        summaries[policy] = computed
        reconstructed[policy] = {i: legacy.semantics(a) for i, a in attempts.items()}
    maximum = 0.0
    for policy in list(POLICIES)[1:]:
        for intent, actual in result["strategies"][policy]["attempts"].items():
            control = result["strategies"]["fallback_opus_fusion"]["attempts"][intent]
            if policy != "fallback_opus_fusion" and not actual["early_rejected"]:
                c.same(
                    {k: v for k, v in actual["constraint_trace"].items() if k != "repetition_stop"},
                    control["constraint_trace"],
                    policy + ".unchanged_legacy_trace",
                )
                c.same(
                    actual["primary_failure"],
                    control["primary_failure"],
                    policy + ".unchanged_primary_outcome",
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
    branch = {
        "checked": 3 * len(qrels),
        "max_absolute_score_difference": maximum,
        "rank_disagreements": 0,
    }
    early, analysis = (
        previous.early_analysis(result["strategies"]),
        protection_analysis(result["strategies"]),
    )
    c.same(result["request_semantics"], reconstructed, "reconstructed_request_semantics")
    c.same(result["branch_audit"], branch, "branch_audit")
    c.same(result["early_stop_analysis"], early, "early_stop_analysis")
    c.same(result["protection_analysis"], analysis, "protection_analysis")
    startup = audit_startup(result, inputs, c)
    return {
        "checks": c.count,
        "numeric_comparisons": c.numeric,
        "maximum_numeric_absolute_difference": c.max_error,
        "scored_requests": scored_count,
        "warm_requests": warm_count,
        "branch_audit": branch,
        "strategies": summaries,
        "early_stop_analysis": early,
        "protection_analysis": analysis,
        "independent_protection_replay": guard_replays,
        "startup": startup,
    }


def audit_startup(result, inputs, c):
    startup = result["startup"]
    loading = (
        "index_load_seconds",
        "retrieval_model_load_seconds",
        "shared_translator_and_decoder_load_seconds",
        "validation_and_models_seconds",
    )
    for key in loading:
        c.require(finite(startup[key]) and startup[key] >= 0, "startup." + key)
    c.require(
        startup["validation_and_models_seconds"] + 1e-6 >= sum(startup[k] for k in loading[:3]),
        "startup.complete_loading_cost",
    )
    decoders = [recipe["decoder"] for recipe in POLICIES.values()]
    c.same(startup["first_request_order"], decoders, "startup.first_request_order")
    first = startup["first_requests_per_decoder"]
    c.same(sorted(first), sorted(decoders), "startup.four_first_requests")
    intent = sorted(inputs["qrels"])[0]
    maximum = 0.0
    for policy, recipe in POLICIES.items():
        actual, expected = (
            first[recipe["decoder"]],
            result["strategies"][policy]["attempts"][intent],
        )
        legacy.validate_timing(actual["timing"], c, "startup." + policy)
        c.same(
            legacy.semantics(actual), legacy.semantics(expected), "startup." + policy + ".semantics"
        )
        if policy == "protected_fallback_fusion":
            validate_protected_guard(actual["constraint_trace"], c, "startup.protected")
        elif policy == "early_fallback_fusion":
            previous.validate_guard(
                actual["constraint_trace"], result["repetition_profile"], c, "startup.guarded"
            )
        if actual["ok"]:
            maximum = max(
                maximum,
                legacy.ranking_difference(
                    actual["rankings"],
                    expected["rankings"],
                    inputs["qrels"][intent],
                    inputs["gallery_ids"],
                    c,
                    "startup." + policy,
                    result["protocol"]["score_atol"],
                ),
            )
        else:
            c.same(actual["rankings"], None, "startup.failed_ranking")
    return {
        "first_requests": 4,
        "maximum_absolute_score_difference": maximum,
        "loading_seconds": {key: startup[key] for key in loading},
    }


def development_summary(rows):
    result = {}
    groups = {
        "all": rows,
        **{
            group: [r for r in rows if r["group"] == group]
            for group in ("old_development", "long_normal", "legitimate_repetition")
        },
    }
    for group, values in groups.items():
        sets = {
            key: []
            for key in (
                "baseline_accepted_ids",
                "accepted_eligible_ids",
                "old_early_rejected_ids",
                "new_early_rejected_ids",
                "protected_ids",
                "old_accepted_new_rejected_ids",
            )
        }
        for row in values:
            methods, query_id = row["methods"], row["query_id"]
            accepted = methods["constrained"]["primary"]["ok"]
            old = methods["guarded"]["primary"]["trace"]["repetition_stop"]
            new = methods["protected"]["primary"]["trace"]["repetition_stop"]
            conditions = (
                accepted,
                accepted and bool(new["snapshots"]),
                old["triggered"],
                new["triggered"],
                bool(new["protected_trigger_steps"]),
                accepted and new["triggered"],
            )
            for key, condition in zip(sets, conditions):
                if condition:
                    sets[key].append(query_id)
        result[group] = {"count": len(values), **{k: sorted(v) for k, v in sets.items()}}
    return result


def audit_development(metadata, profile, implementation_hash, c):
    queries, protocol, observed, validation = (
        legacy.read_json(metadata / name) for name in DEVELOPMENT_FILES
    )
    rows = queries["queries"]
    c.same(queries["role"], "development_only", "dev.source_role")
    c.same(legacy.fingerprint(rows), DEVELOPMENT_ROWS_SHA256, "dev.predeclared_24_source_rows")
    c.same(
        [r["query_id"] for r in rows],
        [f"{prefix}{n:02}_zh" for prefix in ("i", "l", "r") for n in range(1, 9)],
        "dev.source_ids",
    )
    c.same(
        [r["group"] for r in rows],
        [
            group
            for group in ("old_development", "long_normal", "legitimate_repetition")
            for _ in range(8)
        ],
        "dev.source_groups",
    )
    protocol_hash, observed_hash = (
        legacy.file_hash(metadata / n) for n in ("dev-protocol.json", "dev-observations.json")
    )
    fixed = {
        "queries_sha256": legacy.file_hash(metadata / "dev-queries.json"),
        "development_implementation_sha256": implementation_hash,
        "repetition_profile_sha256": PROFILE_BYTE_SHA256,
        "protection_rule_sha256": RULE_BYTE_SHA256,
        "fixed_rule": expected_rule(),
        "methods": ["original", "constrained", "guarded", "protected"],
        "training": False,
        "formal_queries_or_retrieval_used": False,
    }
    for key, expected in fixed.items():
        c.same(protocol[key], expected, "dev.protocol." + key)
    c.same(profile, expected_rule()["repetition_profile"], "dev.unchanged_profile")
    c.same(observed["protocol"], protocol, "dev.observed_protocol")
    c.same(observed["protocol_sha256"], protocol_hash, "dev.observed_protocol_hash")
    c.same(observed["development_only"], True, "dev.development_only")
    c.same(observed["training_executed"], False, "dev.no_training")
    c.same(
        [{k: r[k] for k in ("query_id", "text", "group")} for r in observed["rows"]],
        rows,
        "dev.observed_sources",
    )
    replays, special_ids, requests = {}, None, 0
    for row in observed["rows"]:
        label, methods = "dev." + row["query_id"], row["methods"]
        c.require(
            type(row["input_tokens"]) is int and 0 < row["input_tokens"] <= 256,
            label + ".untruncated_source_tokens",
        )
        c.same(sorted(methods), sorted(fixed["methods"]), label + ".four_methods")
        baseline = methods["constrained"]["primary"]
        original = methods["original"]["primary"]
        eos_ids = methods["protected"]["primary"]["trace"]["repetition_stop"][
            "candidate_protection"
        ]["eos_token_ids"]
        for method, record in methods.items():
            requests += 1
            primary, backup = record["primary"], record["fallback"]
            trace = primary["trace"]
            for outcome, branch in ((primary, "primary"), (backup, "fallback")):
                if outcome is None:
                    continue
                c.require(type(outcome["ok"]) is bool, label + "." + branch + ".ok_boolean")
                c.same(
                    outcome["ok"],
                    outcome["failure"] is None,
                    label + "." + branch + ".failure_status",
                )
                c.same(
                    outcome["trace"]["source_text"], row["text"], label + "." + branch + ".source"
                )
                c.same(
                    outcome["trace"]["status"],
                    "success" if outcome["ok"] else "failed",
                    label + "." + branch + ".trace_status",
                )
                c.same(
                    outcome["english"],
                    outcome["trace"]["output"] if outcome["ok"] else None,
                    label + "." + branch + ".raw_english",
                )
                c.require(
                    finite(outcome["translation_ms"]) and outcome["translation_ms"] >= 0,
                    label + "." + branch + ".translation_time",
                )
            c.same(
                backup is not None,
                method != "original" and not primary["ok"],
                label + ".fresh_backup_presence",
            )
            c.same(
                record["ok"],
                primary["ok"] or bool(backup and backup["ok"]),
                label + ".final_status",
            )
            c.require(
                finite(record["total_translation_ms"])
                and record["total_translation_ms"] + 1e-6
                >= primary["translation_ms"] + (backup["translation_ms"] if backup else 0),
                label + ".complete_translation_time",
            )
            if backup is not None:
                for key in ("ok", "english", "failure", "trace"):
                    c.same(backup[key], original[key], label + ".fresh_original." + key)
            if method == "guarded":
                previous.validate_guard(trace, profile, c, label + ".guarded")
            elif method == "protected":
                replays[row["query_id"]] = validate_protected_guard(trace, c, label + ".protected")
            if method in {"guarded", "protected"}:
                stop = trace["repetition_stop"]
                if special_ids is None:
                    special_ids = stop["special_ids"]
                c.same(stop["special_ids"], special_ids, label + ".special_ids")
                if stop["triggered"]:
                    c.same(primary["ok"], False, label + ".reject_active_fragment")
                    c.same(primary["failure"]["type"], "RepetitionAbort", label + ".abort_error")
                else:
                    c.same(
                        {k: v for k, v in trace.items() if k != "repetition_stop"},
                        baseline["trace"],
                        label + ".unchanged_legacy_trace",
                    )
                    for key in ("ok", "english", "failure"):
                        c.same(primary[key], baseline[key], label + ".unchanged_legacy." + key)
            validate_final_acceptance(
                trace,
                eos_ids,
                c,
                label + "." + method + ".final",
                constrained=method != "original",
                aborted=method in {"guarded", "protected"}
                and trace["repetition_stop"]["triggered"],
            )
            if backup is not None:
                validate_final_acceptance(
                    backup["trace"],
                    eos_ids,
                    c,
                    label + "." + method + ".fallback_final",
                    constrained=False,
                )
    summary = development_summary(observed["rows"])
    c.same(
        validation,
        {
            "protocol_sha256": protocol_hash,
            "observations_sha256": observed_hash,
            "repetition_profile_sha256": PROFILE_BYTE_SHA256,
            "protection_rule_sha256": RULE_BYTE_SHA256,
            "development_only": True,
            "training_executed": False,
            "summary": summary,
        },
        "dev.recomputed_validation_and_coverage",
    )
    return {
        "sources": len(rows),
        "method_requests": requests,
        "summary": summary,
        "special_ids": special_ids,
        "protection_replay": replays,
    }


def source_hashes(root, metadata=None):
    prior = previous.source_hashes(root, metadata)
    files = {
        name: previous._bound_file_hash(root, metadata, name)
        for name in (
            "completed_protection.py",
            "protected_development.py",
            "protected_benchmark.py",
            "completed_protection_v1.json",
        )
    }
    core = prior["implementation_sha256"]
    return {
        "historical_files": {**prior["historical_files"], **prior["files"]},
        "files": files,
        "previous_core_sha256": core,
        "implementation_sha256": legacy.fingerprint({"previous_core": core, **files}),
        "development_implementation_sha256": legacy.fingerprint(
            {
                "previous_core": core,
                **{k: v for k, v in files.items() if k != "protected_benchmark.py"},
            }
        ),
    }


def bind_metadata(metadata, root, c):
    protocol, freeze = (legacy.read_json(metadata / n) for n in ("protocol.json", "freeze.json"))
    source = source_hashes(root, metadata)
    c.same(
        source["previous_core_sha256"], PREVIOUS_CORE_SHA256, "unchanged_measured_early_stop_core"
    )
    c.same(
        legacy.file_hash(metadata / "protocol.json"),
        freeze["protocol_sha256"],
        "protocol_byte_hash",
    )
    for key in (
        "implementation_sha256",
        "dataset_hashes",
        "supporting_hashes",
        "development_hashes",
        "repetition_profile_sha256",
        "protection_rule_sha256",
        "source_indexes",
        "retrieval_weights",
    ):
        c.same(freeze[key], protocol[key], "freeze." + key)
    c.same(
        source["implementation_sha256"], protocol["implementation_sha256"], "executed_source_code"
    )
    c.same(freeze["previous_core_sha256"], PREVIOUS_CORE_SHA256, "freeze.preserved_core")
    prior_protocol_path = root / "benchmarks/repetition-early-stop-v1/protocol.json"
    c.same(
        legacy.file_hash(prior_protocol_path),
        PREVIOUS_PROTOCOL_SHA256,
        "preserved_previous_protocol_bytes",
    )
    prior_protocol = legacy.read_json(prior_protocol_path)
    for key in (
        "translator_files",
        "decoder_files",
        "source_indexes",
        "retrieval_weights",
        "retrieval_models",
        "device",
        "translator_device",
    ):
        c.same(protocol[key], prior_protocol[key], "preserved_artifact_and_model_slots." + key)
    fixed = {
        "policies": POLICIES,
        "query_warmup": 3,
        "query_repeats": 5,
        "independent_intents": 16,
        "gallery_images": 40,
        "source_token_limit": 256,
        "max_new_tokens": 64,
        "num_beams": {"original": 1, "constrained": 4, "guarded": 4, "protected": 4},
        "do_sample": False,
        "use_cache": True,
        "dtype": "float32",
        "translation_text_cache": False,
        "training": False,
        "fusion_weights": [0.5, 0.5],
        "score_atol": 1e-5,
        "failure_policy": FAILURE_POLICY,
        "dataset_hashes": DATA_HASHES,
        "supporting_hashes": SUPPORT_HASHES,
        "repetition_profile_sha256": PROFILE_BYTE_SHA256,
        "protection_rule_sha256": RULE_BYTE_SHA256,
        "translation_model": "Helsinki-NLP/opus-mt-zh-en",
        "translation_revision": "cf109095479db38d6df799875e34039d4938aaa6",
    }
    for key, expected in fixed.items():
        c.same(protocol[key], expected, "fixed_protocol." + key)
    c.same(
        sorted(protocol["development_hashes"]),
        sorted(DEVELOPMENT_FILES),
        "complete_development_binding",
    )
    bound = {name: legacy.file_hash(metadata / name) for name in ("protocol.json", "freeze.json")}
    for name, key, artifact in (
        ("translator-provenance.json", "translator_files", "translation_provenance.json"),
        ("decoder-provenance.json", "decoder_files", "decoder_provenance.json"),
    ):
        bound[name] = legacy.file_hash(metadata / name)
        c.same(bound[name], protocol[key][artifact], "metadata_provenance." + name)
        c.same(
            bound[name],
            legacy.file_hash(root / "benchmarks/terminology-constraints-v1" / name),
            "preserved_metadata_provenance." + name,
        )
    for files, prior in (
        ({k + ".json": v for k, v in DATA_HASHES.items()}, True),
        (SUPPORT_HASHES, True),
        (protocol["development_hashes"], False),
        (
            {
                "repetition_profile_v1.json": PROFILE_BYTE_SHA256,
                "completed_protection_v1.json": RULE_BYTE_SHA256,
            },
            False,
        ),
    ):
        for name, expected in files.items():
            bound[name] = legacy.file_hash(metadata / name)
            c.same(bound[name], expected, "input." + name)
            if prior:
                c.same(
                    bound[name],
                    legacy.file_hash(root / "benchmarks/terminology-constraints-v1" / name),
                    "preserved_historical_input." + name,
                )
    c.same(
        legacy.file_hash(root / "src/siglip_tools/repetition_profile_v1.json"),
        PROFILE_BYTE_SHA256,
        "packaged_selected_profile",
    )
    c.same(
        legacy.file_hash(root / "src/siglip_tools/completed_protection_v1.json"),
        RULE_BYTE_SHA256,
        "packaged_protection_rule",
    )
    c.same(
        legacy.file_hash(root / "src/siglip_tools/power_glossary_v1.json"),
        protocol["glossary_sha256"],
        "packaged_glossary",
    )
    c.same(protocol["glossary_sha256"], SUPPORT_HASHES["glossary.json"], "frozen_glossary")
    profile = expected_rule()["repetition_profile"]
    c.same(
        legacy.read_json(metadata / "repetition_profile_v1.json"),
        profile,
        "selected_profile_contents",
    )
    development = audit_development(
        metadata, profile, source["development_implementation_sha256"], c
    )
    dev_protocol = legacy.read_json(metadata / "dev-protocol.json")
    for key in (
        "translation_model",
        "translation_revision",
        "translator_device",
        "dtype",
        "translator_files",
        "decoder_files",
    ):
        c.same(dev_protocol[key], protocol[key], "development_formal_binding." + key)
    manifest, queries, labels = (
        legacy.read_json(metadata / f"{name}.json") for name in ("manifest", "queries", "qrels")
    )
    chinese = {q["intent_id"]: q["text"] for q in queries["queries"] if q["language"] == "zh"}
    inputs = {
        "gallery_ids": [r["image_id"] for r in manifest["images"]],
        "qrels": {
            q["intent_id"]: labels["qrels"][q["query_id"]]
            for q in queries["queries"]
            if q["language"] == "zh"
        },
    }
    c.same(len(inputs["gallery_ids"]), 40, "forty_gallery_images")
    c.same(len(set(inputs["gallery_ids"])), 40, "unique_gallery_images")
    c.same(len(chinese), 16, "sixteen_intents")
    c.same(
        legacy.fingerprint(inputs), protocol["evaluation_inputs_sha256"], "evaluation_input_digest"
    )
    return protocol, freeze, source, bound, profile, development, manifest, inputs, chinese


def bind_identities(result, protocol, metadata, profile, c, label):
    provenance = {
        name: legacy.read_json(metadata / f"{name}-provenance.json")
        for name in ("translator", "decoder")
    }
    c.same(
        sorted(result["translator_identities"]),
        ["constrained", "guarded", "original", "protected"],
        label + ".four_translators",
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
            c.same(identity[key], expected, label + ".translator." + variant + "." + key)
        for key, expected in (
            ("num_beams", protocol["num_beams"][variant]),
            ("do_sample", False),
            ("max_new_tokens", 64),
            ("use_cache", True),
        ):
            c.same(
                identity["generation"][key], expected, label + ".generation." + variant + "." + key
            )
        c.same(
            identity["generation_config"],
            identity["generation"],
            label + ".generation_configuration",
        )
        if variant != "original":
            c.same(identity["decoder"], provenance["decoder"], label + ".decoder_identity")
            c.same(
                identity["glossary"]["sha256"],
                protocol["glossary_sha256"],
                label + ".glossary_identity",
            )
            c.same(
                identity["glossary"]["profile"],
                legacy.read_json(metadata / "glossary.json"),
                label + ".glossary_content",
            )
        if variant in {"guarded", "protected"}:
            c.same(
                identity["repetition_stop"],
                {
                    "profile": profile,
                    "profile_sha256": legacy.fingerprint(profile),
                    "observe_only": False,
                },
                label + ".guard_identity",
            )
        if variant == "protected":
            c.same(
                identity["candidate_protection"],
                {"rule": expected_rule(), "rule_sha256": legacy.fingerprint(expected_rule())},
                label + ".protection_identity",
            )
            c.same(
                identity["opus_decoding"],
                "completed_candidate_protected_constrained_beam4",
                label + ".protected_decoder",
            )
    for name, key, filename in (
        ("translator", "translator_files", "translation_provenance.json"),
        ("decoder", "decoder_files", "decoder_provenance.json"),
    ):
        c.same(
            legacy.file_hash(metadata / f"{name}-provenance.json"),
            protocol[key][filename],
            label + ".provenance_bytes." + name,
        )
    c.same(
        provenance["translator"]["artifacts"],
        {
            k: v
            for k, v in protocol["translator_files"].items()
            if k != "translation_provenance.json"
        },
        label + ".translator_provenance_artifacts",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "quality", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=Path("benchmarks/completed-protection-v1"))
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Audit output exists; choose a new path")
    c, report = Checks(), {"schema_version": 1, "passed": False, "failures": []}
    root, metadata = Path(__file__).resolve().parents[1], args.metadata.resolve()
    try:
        protocol, freeze, source, bound, profile, development, manifest, inputs, chinese = (
            bind_metadata(metadata, root, c)
        )
        dev_observed = legacy.read_json(metadata / "dev-observations.json")
        bind_identities(dev_observed, protocol, metadata, profile, c, "development")
        glossary = legacy.read_json(metadata / "glossary.json")
        for row in dev_observed["rows"]:
            for method in ("constrained", "guarded", "protected"):
                bind_trace_requirements(
                    row["methods"][method]["primary"]["trace"],
                    glossary,
                    development["special_ids"],
                    c,
                    "development_source." + row["query_id"] + "." + method,
                )
        results, audits, result_files = {}, {}, {}
        for alias in ("base", "quality"):
            path = getattr(args, alias).resolve()
            raw = path.read_bytes()
            result = json.loads(raw)
            result_files[alias] = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}
            for key, expected in (
                ("model_alias", alias),
                ("protocol", protocol),
                ("protocol_sha256", freeze["protocol_sha256"]),
                ("implementation_sha256", protocol["implementation_sha256"]),
                ("source_index_sha256", protocol["source_indexes"][alias]),
                ("dataset_hashes", protocol["dataset_hashes"]),
                ("development_hashes", protocol["development_hashes"]),
                ("repetition_profile", profile),
                ("repetition_profile_sha256", PROFILE_BYTE_SHA256),
                ("protection_rule", expected_rule()),
                ("protection_rule_sha256", RULE_BYTE_SHA256),
                ("gallery_images", 40),
                ("independent_intents", 16),
                ("training_executed", False),
            ):
                c.same(result[key], expected, alias + "." + key)
            for key, expected in protocol["retrieval_models"][alias].items():
                c.same(result["model_identity"][key], expected, alias + ".identity." + key)
            bind_identities(result, protocol, metadata, profile, c, alias)
            for policy, strategy in result["strategies"].items():
                if policy == "original_fusion":
                    continue
                for intent, attempt in strategy["attempts"].items():
                    bind_trace_requirements(
                        attempt["constraint_trace"],
                        glossary,
                        development["special_ids"],
                        c,
                        alias + ".source." + policy + "." + intent,
                    )
            for decoder in ("constrained", "guarded", "protected"):
                bind_trace_requirements(
                    result["startup"]["first_requests_per_decoder"][decoder]["constraint_trace"],
                    glossary,
                    development["special_ids"],
                    c,
                    alias + ".startup_source." + decoder,
                )
            c.same(
                result["translator_identities"],
                dev_observed["translator_identities"],
                alias + ".unchanged_development_identities",
            )
            for policy in ("early_fallback_fusion", "protected_fallback_fusion"):
                for attempt in result["strategies"][policy]["attempts"].values():
                    c.same(
                        attempt["constraint_trace"]["repetition_stop"]["special_ids"],
                        development["special_ids"],
                        alias + ".guard_special_ids",
                    )
            audits[alias] = audit_result(result, inputs, chinese)
            c.same(audits[alias]["scored_requests"], 64, alias + ".all_scored_requests")
            c.same(audits[alias]["warm_requests"], 320, alias + ".all_warm_requests")
            results[alias] = result
        for key in ("request_semantics", "translator_identities", "environment"):
            c.same(results["base"][key], results["quality"][key], "cross_model." + key)
        c.same(
            results["base"]["model_identity"]["runtime_versions"],
            results["quality"]["model_identity"]["runtime_versions"],
            "cross_model.runtime_versions",
        )
        artifacts = previous.bind_artifacts(protocol, root, manifest, c)
        for alias, entry in result_files.items():
            c.same(
                legacy.file_hash(entry["path"]),
                entry["sha256"],
                alias + ".result_unchanged_during_audit",
            )
        for name, digest in bound.items():
            c.same(
                legacy.file_hash(metadata / name), digest, "metadata_unchanged_during_audit." + name
            )
        report.update(
            passed=True,
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            protocol=protocol,
            freeze=freeze,
            frozen_source_identifiers=source,
            input_and_support_sha256=bound,
            development_replay=development,
            local_artifact_sha256=artifacts,
            result_files=result_files,
            models=audits,
            checks=c.count + sum(a["checks"] for a in audits.values()),
            numeric_comparisons=c.numeric + sum(a["numeric_comparisons"] for a in audits.values()),
            maximum_numeric_absolute_difference=max(
                [c.max_error] + [a["maximum_numeric_absolute_difference"] for a in audits.values()]
            ),
            evidence_file_sha256=evidence_log_snapshot(root / "artifacts/completed-protection"),
            evidence_log_scope=(
                "Caller must finish every other writer first; "
                "current stdout/stderr inode excluded (SIG-043/050)."
            ),
            auditor_sha256=legacy.file_hash(__file__),
            helper_sha256={
                Path(module.__file__).name: legacy.file_hash(module.__file__)
                for module in (previous, legacy)
            },
            limits=[
                "Independent recorded-data recomputation; no pretrained neural replay, "
                "training or human semantic validation.",
                "Recorded EOS tensor origin, tokenizer decoding and source-token counts "
                "are not cryptographic proof; pinned code/artifacts and actual-scorer "
                "characterization bound that claim.",
                "All original recorded lexical alternatives are replayed; lexical acceptance "
                "does not establish wiring, negation or mounting semantics.",
                "The diagnosed sixteen intents, known t09 regression and shared gallery "
                "are not a new independent holdout.",
                "Warm semantics reconstruct from scored attempts plus ranking overrides; "
                "every saved timing/status/digest is checked, not independently reinferred.",
                "Finalized diagnostic sequence length may be shorter than actual process/callback "
                "steps; scorer.is_done may skip the final callback.",
                "Protection suppresses an abort and continues the unchanged search; "
                "it does not guarantee the final selected output succeeds.",
                "Closed-log hashes depend on all external writers finishing; "
                "this auditor excludes only its own stdout/stderr inodes.",
            ],
        )
    except (ValueError, KeyError, TypeError, OSError, IndexError, OverflowError) as exc:
        report["failures"].append(f"{type(exc).__name__}: {exc}")
        report.update(
            passed=False,
            completed_top_level_checks=c.count,
            completed_numeric_comparisons=c.numeric,
        )
    save_new(args.output, report)
    print(
        json.dumps(
            {"passed": report["passed"], "failures": report["failures"], "output": str(args.output)}
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
