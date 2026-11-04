"""Twenty-four declared development controls for a fixed EOS-pool protection rule."""

import argparse
from pathlib import Path
from time import perf_counter

from .auto_benchmark import check_translator_files, translator_snapshot
from .benchmark import _json_snapshot, _sync
from .early_stop_benchmark import _validate_guard, _without_guard
from .early_stop_benchmark import implementation_hash as previous_hash
from .early_stop_development import _translate
from .encoder import identity_fingerprint
from .frozen_source import reference_implementation, source_sha256
from .fusion_benchmark import _save_new
from .images import file_sha256
from .opus_constraints import OPUSConstraintTranslator, glossary_requirements
from .repetition_stop import GuardedOPUSConstraintTranslator
from .translation import LocalTranslator

METADATA = Path("benchmarks/completed-protection-v1")
PROFILE_PATH = Path(__file__).with_name("repetition_profile_v1.json")
RULE_PATH = Path(__file__).with_name("completed_protection_v1.json")
DEVELOPMENT_ROWS_SHA256 = "2917c1aea48f166ad2af5c07664b979902fe11ef6ac59e28843fc09744e9919c"
PREVIOUS_CORE_SHA256 = "5508fc7372887770f584065949d19debe77b81fd6d09cf022a432de164d7bc12"
METHODS = ("original", "constrained", "guarded", "protected")
DEVELOPMENT_FILES = (
    "dev-queries.json",
    "dev-protocol.json",
    "dev-observations.json",
    "dev-validation.json",
)


def development_implementation_hash():
    return identity_fingerprint(
        {
            "previous_core": previous_hash(),
            **{
                n: source_sha256(n)
                for n in (
                    "completed_protection.py",
                    "completed_protection_v1.json",
                    "protected_development.py",
                )
            },
        }
    )


def load_development_queries(path):
    doc, _ = _json_snapshot(path)
    rows = doc.get("queries")
    expected = (
        [f"i{i:02}_zh" for i in range(1, 9)]
        + [f"l{i:02}_zh" for i in range(1, 9)]
        + [f"r{i:02}_zh" for i in range(1, 9)]
    )
    if (
        doc.get("role") != "development_only"
        or not isinstance(rows, list)
        or identity_fingerprint(rows) != DEVELOPMENT_ROWS_SHA256
        or [r.get("query_id") for r in rows] != expected
    ):
        raise ValueError("Expected the fixed twenty-four development sources, never formal IDs")
    return rows


def development_summary(rows):
    report = {}
    groups = {
        "all": rows,
        **{
            group: [r for r in rows if r["group"] == group]
            for group in ("old_development", "long_normal", "legitimate_repetition")
        },
    }
    for group, values in groups.items():

        def ids(predicate):
            return sorted(r["query_id"] for r in values if predicate(r["methods"]))

        def trace(m, key):
            return m[key]["primary"]["trace"].get("repetition_stop", {})

        report[group] = {
            "count": len(values),
            "baseline_accepted_ids": ids(lambda m: m["constrained"]["primary"]["ok"]),
            "accepted_eligible_ids": ids(
                lambda m: (
                    m["constrained"]["primary"]["ok"]
                    and bool(trace(m, "protected").get("snapshots"))
                )
            ),
            "old_early_rejected_ids": ids(lambda m: trace(m, "guarded").get("triggered", False)),
            "new_early_rejected_ids": ids(lambda m: trace(m, "protected").get("triggered", False)),
            "protected_ids": ids(
                lambda m: bool(trace(m, "protected").get("protected_trigger_steps"))
            ),
            "old_accepted_new_rejected_ids": ids(
                lambda m: (
                    m["constrained"]["primary"]["ok"] and trace(m, "protected").get("triggered")
                )
            ),
        }
    return report


def validate_source_constraints(trace, source):
    """A saved disabled/empty requirement set cannot masquerade as source coverage."""
    expected = glossary_requirements(source)
    actual = trace.get("requirements")
    if (
        trace.get("constraints_applied") is not bool(expected)
        or not isinstance(actual, list)
        or [{k: r.get(k) for k in ("id", "source_terms", "target_forms")} for r in actual]
        != expected
        or any(
            not r.get("token_alternatives")
            or any(
                not alt or any(type(t) is not int or t < 0 for t in alt)
                for alt in r["token_alternatives"]
            )
            for r in actual
        )
    ):
        raise ValueError("Saved source glossary constraints differ from fixed matching")
    return True


def validate_primary_acceptance(primary, source, identity, *, constrained):
    """Count acceptance only after replaying actual final EOS/text/lexical checks."""
    from .completed_protection import validate_final_acceptance

    eos = identity.get("generation", {}).get("eos_token_id")
    eos_ids = [eos] if type(eos) is int else eos
    if (
        not isinstance(eos_ids, list)
        or not eos_ids
        or any(type(token) is not int or token < 0 for token in eos_ids)
    ):
        raise ValueError("Development final acceptance has invalid frozen EOS identity")
    trace = primary["trace"]
    try:
        validate_final_acceptance(trace, eos_ids, constrained=constrained)
    except ValueError as exc:
        raise ValueError("Development final acceptance differs from final sequence/text") from exc
    if constrained:
        validate_source_constraints(trace, source)
    if (
        trace.get("source_text") != source
        or type(primary["ok"]) is not bool
        or primary["ok"] != (primary["failure"] is None)
        or primary["ok"] != (trace["status"] == "success")
        or primary["english"] != (trace["output"] if primary["ok"] else None)
    ):
        raise ValueError("Development final acceptance status or source differs")
    return True


def load_protocol(args):
    rows = load_development_queries(args.queries)
    protocol, digest = _json_snapshot(args.protocol)
    expected = {
        "queries_sha256": file_sha256(args.queries),
        "development_implementation_sha256": development_implementation_hash(),
        "repetition_profile_sha256": file_sha256(PROFILE_PATH),
        "protection_rule_sha256": file_sha256(RULE_PATH),
        "fixed_rule": _json_snapshot(RULE_PATH)[0],
        "methods": list(METHODS),
        "training": False,
        "formal_queries_or_retrieval_used": False,
    }
    if previous_hash() != PREVIOUS_CORE_SHA256 or any(
        protocol.get(k) != v for k, v in expected.items()
    ):
        raise ValueError("Development code/rules/queries differ from the predeclared protocol")
    for directory, key in (
        (args.translation_dir, "translator_files"),
        (args.decoder_dir, "decoder_files"),
    ):
        if translator_snapshot(directory) != protocol[key]:
            raise ValueError("Development artifacts differ from the frozen protocol")
    return rows, protocol, digest


def validate_development_evidence(
    metadata, profile, *, observations_path=None, validation_path=None
):
    """Replay actual coverage/acceptance and protection before formal model allocation."""
    from .completed_protection import validate_protection_trace

    metadata = Path(metadata)
    sources = load_development_queries(metadata / "dev-queries.json")
    protocol, protocol_hash = _json_snapshot(metadata / "dev-protocol.json")
    observed, observed_hash = _json_snapshot(
        observations_path or metadata / "dev-observations.json"
    )
    saved, _ = _json_snapshot(validation_path or metadata / "dev-validation.json")
    if (
        protocol.get("queries_sha256") != file_sha256(metadata / "dev-queries.json")
        or protocol.get("development_implementation_sha256")
        != reference_implementation(metadata, development_implementation_hash)
        or protocol.get("fixed_rule") != _json_snapshot(RULE_PATH)[0]
        or protocol.get("repetition_profile_sha256") != file_sha256(PROFILE_PATH)
        or protocol.get("protection_rule_sha256") != file_sha256(RULE_PATH)
        or protocol.get("methods") != list(METHODS)
        or protocol.get("training") is not False
        or protocol.get("formal_queries_or_retrieval_used") is not False
        or observed.get("protocol") != protocol
        or observed.get("protocol_sha256") != protocol_hash
        or observed.get("development_only") is not True
        or observed.get("training_executed") is not False
        or [{k: r[k] for k in ("query_id", "text", "group")} for r in observed["rows"]] != sources
    ):
        raise ValueError("Saved development sources, rule, implementation or protocol changed")
    for row in observed["rows"]:
        if set(row["methods"]) != set(METHODS) or not 0 < row["input_tokens"] <= 256:
            raise ValueError("Development method coverage or input-token contract changed")
        baseline = row["methods"]["constrained"]["primary"]
        for method, record in row["methods"].items():
            primary, backup = record["primary"], record["fallback"]
            trace = primary["trace"]
            validate_primary_acceptance(
                primary,
                row["text"],
                observed["translator_identities"][method],
                constrained=method != "original",
            )
            if backup is not None:
                validate_primary_acceptance(
                    backup,
                    row["text"],
                    observed["translator_identities"]["original"],
                    constrained=False,
                )
            if (
                trace.get("source_text") != row["text"]
                or primary["ok"] != (primary["failure"] is None)
                or primary["ok"] != (trace["status"] == "success")
                or primary["english"] != (trace["output"] if primary["ok"] else None)
                or (backup is not None) != (method != "original" and not primary["ok"])
                or record["ok"] != (primary["ok"] or bool(backup and backup["ok"]))
                or record["total_translation_ms"] + 1e-6
                < primary["translation_ms"] + (backup["translation_ms"] if backup else 0)
            ):
                raise ValueError("Development source, failure, fallback or full time changed")
            if (
                backup is not None
                and backup["trace"] != row["methods"]["original"]["primary"]["trace"]
            ):
                raise ValueError("Development backup differs from fresh original control")
            if method == "guarded":
                _validate_guard(
                    {
                        "constraint_trace": trace,
                        "early_rejected": trace["repetition_stop"]["triggered"],
                    },
                    profile,
                )
            if method == "protected":
                validate_protection_trace(trace)
            if method in ("guarded", "protected") and not trace["repetition_stop"]["triggered"]:
                if _without_guard(trace) != baseline["trace"]:
                    raise ValueError("Non-aborted development primary differs from legacy control")
    expected = {
        "protocol_sha256": protocol_hash,
        "observations_sha256": observed_hash,
        "repetition_profile_sha256": file_sha256(PROFILE_PATH),
        "protection_rule_sha256": file_sha256(RULE_PATH),
        "development_only": True,
        "training_executed": False,
        "summary": development_summary(observed["rows"]),
    }
    if saved != expected:
        raise ValueError("Saved development validation differs from actual evidence")
    return True


def run_development(args):
    from .completed_protection import CompletedProtectedTranslator

    if args.output.exists() or args.validation.exists():
        raise FileExistsError("Development output exists; use new paths")
    rows, protocol, digest = load_protocol(args)
    profile, _ = _json_snapshot(PROFILE_PATH)
    original = LocalTranslator(args.translation_dir, device=protocol["device"])
    translators = {
        "original": original,
        "constrained": OPUSConstraintTranslator(original, decoder_dir=args.decoder_dir),
        "guarded": GuardedOPUSConstraintTranslator(
            original, profile=profile, decoder_dir=args.decoder_dir
        ),
        "protected": CompletedProtectedTranslator(
            original, profile=profile, decoder_dir=args.decoder_dir
        ),
    }
    identities = {k: t.identity for k, t in translators.items()}
    results = []
    for query in rows:
        length = len(original.tokenizer(query["text"], add_special_tokens=True)["input_ids"])
        if not 0 < length <= 256:
            raise ValueError("Declared development source exceeds untruncated input budget")
        methods = {}
        for name, translator in translators.items():
            _sync(original.device)
            started = perf_counter()
            primary = _translate(translator, query["text"])
            backup = (
                _translate(original, query["text"])
                if name != "original" and not primary["ok"]
                else None
            )
            _sync(original.device)
            elapsed = (perf_counter() - started) * 1000
            methods[name] = {
                "primary": primary,
                "fallback": backup,
                "ok": primary["ok"] or bool(backup and backup["ok"]),
                "total_translation_ms": elapsed,
            }
        results.append({**query, "input_tokens": length, "methods": methods})
        print(
            query["query_id"],
            {
                k: (
                    v["primary"]["ok"],
                    v["primary"]["trace"].get("repetition_stop", {}).get("triggered", False),
                )
                for k, v in methods.items()
            },
            flush=True,
        )
    load_protocol(args)
    if identities != {k: t.identity for k, t in translators.items()}:
        raise ValueError("Development mutated shared model configuration or identity")
    for directory, key in (
        (args.translation_dir, "translator_files"),
        (args.decoder_dir, "decoder_files"),
    ):
        check_translator_files(directory, protocol[key])
    observed = {
        "protocol": protocol,
        "protocol_sha256": digest,
        "rows": results,
        "translator_identities": identities,
        "development_only": True,
        "training_executed": False,
    }
    _save_new(args.output, observed)
    validation = {
        "protocol_sha256": digest,
        "observations_sha256": file_sha256(args.output),
        "repetition_profile_sha256": file_sha256(PROFILE_PATH),
        "protection_rule_sha256": file_sha256(RULE_PATH),
        "development_only": True,
        "training_executed": False,
        "summary": development_summary(results),
    }
    _save_new(args.validation, validation)
    validate_development_evidence(
        args.queries.parent,
        profile,
        observations_path=args.output,
        validation_path=args.validation,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in (
        ("queries", METADATA / "dev-queries.json"),
        ("protocol", METADATA / "dev-protocol.json"),
        ("output", METADATA / "dev-observations.json"),
        ("validation", METADATA / "dev-validation.json"),
        ("translation-dir", Path("models/opus-mt-zh-en")),
        ("decoder-dir", Path("models/opus-constrained-decoder")),
    ):
        parser.add_argument(f"--{name}", type=Path, default=default)
    run_development(parser.parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
