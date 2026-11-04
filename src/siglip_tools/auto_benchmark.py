"""Live translation/retrieval benchmark with synchronized end-to-end timing."""

import argparse
from collections import defaultdict
from pathlib import Path
from time import perf_counter

from .auto_retrieval import MODES, timed_search
from .benchmark import (
    MODEL_REVISIONS,
    _environment,
    _json_snapshot,
    _stats,
    _sync,
    load_evaluation_data,
)
from .encoder import SigLIPEmbedder, identity_fingerprint
from .evaluation import evaluate_rankings
from .frozen_source import source_sha256
from .fusion_benchmark import _save_new, check_ranking_and_scores
from .fusion_benchmark import implementation_hash as fusion_hash
from .images import file_sha256
from .index import ImageIndex
from .translation import MAX_INPUT_TOKENS, MODEL_ID, REVISION, LocalTranslator


def translator_spec(alias):
    """Return the fixed loader, source, revision and default local directory."""
    if alias == "opus":
        return LocalTranslator, MODEL_ID, REVISION, Path("models/opus-mt-zh-en")
    if alias == "nllb":
        from .nllb_translation import MODEL_ID as nllb_id
        from .nllb_translation import REVISION as nllb_revision
        from .nllb_translation import NLLBTranslator

        return NLLBTranslator, nllb_id, nllb_revision, Path("models/nllb-200-1.3B")
    raise ValueError("Expected opus or nllb translator")


def translator_snapshot(directory):
    files = sorted(Path(directory).rglob("*"))
    if any(p.is_symlink() for p in files):
        raise ValueError("Translator files must not contain symlinks")
    return {p.relative_to(directory).as_posix(): file_sha256(p) for p in files if p.is_file()}


def check_translator_files(directory, frozen):
    if translator_snapshot(directory) != frozen:
        raise ValueError("Translator provenance, artifacts or inventory changed during execution")


def implementation_hash():
    return identity_fingerprint(
        {
            "fusion": fusion_hash(),
            **{
                name: source_sha256(name)
                for name in [
                    "translation.py",
                    "nllb_translation.py",
                    "auto_retrieval.py",
                    "auto_benchmark.py",
                ]
            },
        }
    )


def load_inputs(args):
    meta = Path(args.metadata)
    data = load_evaluation_data(
        meta / "manifest.json", meta / "queries.json", meta / "qrels.json", args.data_root
    )
    protocol, digest = _json_snapshot(args.protocol)
    _, translator_id, translator_revision, _ = translator_spec(getattr(args, "translator", "opus"))
    if (
        protocol["dataset_hashes"] != data["hashes"]
        or protocol["strategies"] != list(MODES)
        or protocol["translation_model"] != translator_id
        or protocol["translation_revision"] != translator_revision
        or protocol.get("translation_backend", "opus") != getattr(args, "translator", "opus")
        or protocol["fusion_weights"] != [0.5, 0.5]
        or protocol["translation_text_cache"] is not False
        or protocol["training"] is not False
        or protocol["dtype"] != "float32"
        or protocol["translation_generation"]
        != {"do_sample": False, "num_beams": 1, "max_new_tokens": 64, "use_cache": True}
        or protocol.get("score_atol", 1e-5) != 1e-5
        or protocol.get("source_token_limit", MAX_INPUT_TOKENS) != MAX_INPUT_TOKENS
    ):
        raise ValueError("Protocol must retain fixed translator, generation, data and policies")
    for key, minimum in [("query_warmup", 0), ("query_repeats", 1)]:
        if type(protocol[key]) is not int or protocol[key] < minimum:
            raise ValueError("Invalid repeat/warmup")
    started = perf_counter()
    index_hash = file_sha256(args.index)
    index = ImageIndex.load(args.index)
    if file_sha256(args.index) != index_hash:
        raise ValueError("Index changed during load")
    index_seconds = perf_counter() - started
    expected = {r["path"]: r["sha256"] for r in data["images"]}
    if (
        {r.path: r.sha256 for r in index.records} != expected
        or len(index.records) != len(expected)
        or any(r.aliases for r in index.records)
    ):
        raise ValueError("Index must exactly match frozen image records")
    if (index.model_identity["model_id"], index.model_identity["revision"]) != MODEL_REVISIONS[
        args.model
    ]:
        raise ValueError("Index must use requested pinned model")
    pairs = defaultdict(dict)
    for q in data["queries"]:
        pairs[q["intent_id"]][q["language"]] = q
    return data, protocol, digest, index, index_hash, index_seconds, dict(sorted(pairs.items()))


def run_benchmark(args):
    output = Path(args.output) / "result.json"
    if output.exists():
        raise FileExistsError("Result exists; use a new output directory")
    startup = perf_counter()
    data, protocol, protocol_hash, index, index_hash, index_seconds, pairs = load_inputs(args)
    code_hash = implementation_hash()
    started = perf_counter()
    encoder = SigLIPEmbedder(
        *MODEL_REVISIONS[args.model], protocol["device"], local_files_only=True
    )
    _sync(encoder.device)
    encoder_seconds = perf_counter() - started
    index.assert_compatible(encoder.identity)
    started = perf_counter()
    alias = getattr(args, "translator", "opus")
    loader, translator_id, translator_revision, default_dir = translator_spec(alias)
    translator = loader(args.translation_dir or default_dir, device=protocol["translator_device"])
    _sync(translator.device)
    translator_seconds = perf_counter() - started
    identity = translator.identity
    translator_files = translator_snapshot(translator.model_dir)
    if (
        identity["source_model"] != translator_id
        or identity["source_revision"] != translator_revision
        or identity["dtype"] != "float32"
    ):
        raise ValueError("Translator must match fixed checkpoint and dtype")
    if (
        str(encoder.device) != protocol["device"]
        or str(translator.device) != protocol["translator_device"]
    ):
        raise ValueError("Device differs from protocol")
    if any(
        identity["generation"].get(k) != v for k, v in protocol["translation_generation"].items()
    ):
        raise ValueError("Translator generation differs from protocol")
    startup_seconds = perf_counter() - startup
    by_path = {r["path"]: r["image_id"] for r in data["images"]}
    relevance = {i: data["qrels"][p["zh"]["query_id"]] for i, p in pairs.items()}
    translations = {}
    intents = list(pairs)

    def retrieve(mode, intent):
        result = timed_search(
            index, encoder, translator, pairs[intent]["zh"]["text"], mode, top_k=len(data["images"])
        )
        english = result["translation"]
        if mode != "zh":
            if intent in translations and translations[intent] != english:
                raise ValueError("Automatic translation changed across requests; discard run")
            translations[intent] = english
        rows = [
            {
                "image_id": by_path[m.path],
                "score": m.score,
                "rank": m.rank,
                "relevant": by_path[m.path] in relevance[intent],
            }
            for m in result["matches"]
        ]
        return rows, result["timing"]

    first_rows, first_timing = retrieve("auto_en", intents[0])
    first_request = {
        "intent_id": intents[0],
        "translation": translations[intents[0]],
        "timing": first_timing,
    }
    scored = {mode: {} for mode in MODES}
    for mode in MODES:
        for j in range(protocol["query_warmup"]):
            retrieve(mode, intents[j % len(intents)])
        for intent in intents:
            scored[mode][intent] = retrieve(mode, intent)[0]
    samples = {m: defaultdict(list) for m in MODES}
    max_drift = {m: 0.0 for m in MODES}
    for round_no in range(protocol["query_repeats"]):
        for j, intent in enumerate(intents):
            offset = (round_no + j) % len(MODES)
            order = MODES[offset:] + MODES[:offset]
            for mode in order:
                rows, timing = retrieve(mode, intent)
                max_drift[mode] = max(
                    max_drift[mode], check_ranking_and_scores(rows, scored[mode][intent])
                )
                samples[mode][intent].append(timing)
    policies = {}
    for mode in MODES:
        ranking = {i: [r["image_id"] for r in rows] for i, rows in scored[mode].items()}
        policies[mode] = {
            "rankings": scored[mode],
            "metrics": evaluate_rankings(ranking, relevance),
            "timing": {
                key: _stats([t[key] for values in samples[mode].values() for t in values])
                for key in ["translation_ms", "retrieval_ms", "total_ms"]
            },
            "warm_samples": dict(samples[mode]),
            "max_repeated_absolute_score_difference": max_drift[mode],
            "rank_disagreements": 0,
        }
    # All loaded artifact and source snapshots must remain identical until publication.
    if (
        implementation_hash() != code_hash
        or file_sha256(args.protocol) != protocol_hash
        or file_sha256(args.index) != index_hash
        or any(
            file_sha256(Path(args.metadata) / f"{name}.json") != digest
            for name, digest in data["hashes"].items()
        )
        or any(
            file_sha256(path) != row["sha256"] for path, row in zip(data["paths"], data["images"])
        )
    ):
        raise ValueError("Frozen inputs, translator artifacts or code changed during execution")
    check_translator_files(translator.model_dir, translator_files)
    payload = {
        "schema_version": 1,
        "experiment_id": protocol["experiment_id"],
        "model_alias": args.model,
        "translator_alias": alias,
        "model_identity": encoder.identity,
        "translator_identity": identity,
        "environment": _environment(encoder.device),
        "dataset_hashes": data["hashes"],
        "protocol": protocol,
        "protocol_sha256": protocol_hash,
        "implementation_sha256": code_hash,
        "source_index_sha256": index_hash,
        "gallery_images": len(data["images"]),
        "independent_intents": len(intents),
        "processor_image_size": dict(encoder.processor.image_processor.size),
        "translations": translations,
        "strategies": policies,
        "training_executed": False,
        "startup": {
            "index_load_seconds": index_seconds,
            "retrieval_model_load_seconds": encoder_seconds,
            "translator_load_seconds": translator_seconds,
            "validation_and_models_seconds": startup_seconds,
            "first_auto_en_request": first_request,
        },
    }
    _save_new(output, payload)
    print(
        {
            m: {"metrics": p["metrics"]["aggregate"], "total_ms": p["timing"]["total_ms"]}
            for m, p in policies.items()
        }
    )


def compare_results(base, quality):
    for alias, r in [("base", base), ("quality", quality)]:
        if (
            r["model_alias"] != alias
            or (r["model_identity"]["model_id"], r["model_identity"]["revision"])
            != MODEL_REVISIONS[alias]
        ):
            raise ValueError("Expected pinned Base and So400m in named slots")
    for key in [
        "dataset_hashes",
        "protocol_sha256",
        "protocol",
        "implementation_sha256",
        "environment",
        "translator_identity",
        "translations",
    ]:
        if base[key] != quality[key]:
            raise ValueError(f"Comparison requires same {key}")
    if base["model_identity"]["runtime_versions"] != quality["model_identity"]["runtime_versions"]:
        raise ValueError("Runtime libraries differ")
    return {
        "quality_minus_base": {
            m: {
                k: quality["strategies"][m]["metrics"]["aggregate"][k] - v
                for k, v in base["strategies"][m]["metrics"]["aggregate"].items()
            }
            for m in MODES
        },
        "quality_to_base_total_p50_ratio": {
            m: quality["strategies"][m]["timing"]["total_ms"]["p50"]
            / base["strategies"][m]["timing"]["total_ms"]["p50"]
            for m in MODES
        },
    }


def compare_translators(opus, nllb):
    """Compare translator choices while keeping the retrieval configuration fixed."""
    for alias, result in (("opus", opus), ("nllb", nllb)):
        _, source, revision, _ = translator_spec(alias)
        if (
            result["translator_alias"] != alias
            or result["translator_identity"]["source_model"] != source
            or result["translator_identity"]["source_revision"] != revision
        ):
            raise ValueError("Expected pinned OPUS and NLLB in the named slots")
    if (
        opus["translator_identity"]["runtime_versions"]
        != nllb["translator_identity"]["runtime_versions"]
    ):
        raise ValueError("Cross-translator comparison requires the same translation runtime")
    for key in (
        "model_alias",
        "model_identity",
        "dataset_hashes",
        "source_index_sha256",
        "implementation_sha256",
        "environment",
    ):
        if opus[key] != nllb[key]:
            raise ValueError(f"Cross-translator comparison requires same {key}")
    for key in (
        "dataset_hashes",
        "strategies",
        "device",
        "translator_device",
        "dtype",
        "query_warmup",
        "query_repeats",
        "source_token_limit",
        "fusion_weights",
        "score_atol",
        "translation_text_cache",
        "training",
        "translation_generation",
    ):
        if opus["protocol"][key] != nllb["protocol"][key]:
            raise ValueError(f"Cross-translator comparison requires same protocol {key}")
    old_zh, new_zh = opus["strategies"]["zh"], nllb["strategies"]["zh"]
    if old_zh["rankings"].keys() != new_zh["rankings"].keys():
        raise ValueError("Chinese baseline intent sets differ")
    for intent in old_zh["rankings"]:
        check_ranking_and_scores(new_zh["rankings"][intent], old_zh["rankings"][intent])
    if old_zh["metrics"] != new_zh["metrics"]:
        raise ValueError("Chinese baseline metrics differ")
    return {
        "nllb_minus_opus": {
            mode: {
                key: nllb["strategies"][mode]["metrics"]["aggregate"][key] - value
                for key, value in opus["strategies"][mode]["metrics"]["aggregate"].items()
            }
            for mode in MODES
        },
        "nllb_to_opus_total_p50_ratio": {
            mode: nllb["strategies"][mode]["timing"]["total_ms"]["p50"]
            / opus["strategies"][mode]["timing"]["total_ms"]["p50"]
            for mode in MODES
        },
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", choices=list(MODEL_REVISIONS), required=True)
    r.add_argument("--translator", choices=("opus", "nllb"), default="opus")
    r.add_argument("--translation-dir", type=Path)
    for name in ["index", "output"]:
        r.add_argument(f"--{name}", type=Path, required=True)
    for name, default in [
        ("metadata", "benchmarks/equipment-v1"),
        ("data-root", "data/equipment-v1"),
        ("protocol", "benchmarks/auto-translation-v1/protocol.json"),
    ]:
        r.add_argument(f"--{name}", type=Path, default=Path(default))
    c = sub.add_parser("compare")
    for name in ["base", "quality", "output"]:
        c.add_argument(f"--{name}", type=Path, required=True)
    cross = sub.add_parser("compare-translators")
    for name in ("opus", "nllb", "output"):
        cross.add_argument(f"--{name}", type=Path, required=True)
    a = p.parse_args(argv)
    if a.command == "run":
        run_benchmark(a)
    elif a.command == "compare":
        _save_new(
            a.output, compare_results(_json_snapshot(a.base)[0], _json_snapshot(a.quality)[0])
        )
    else:
        _save_new(
            a.output, compare_translators(_json_snapshot(a.opus)[0], _json_snapshot(a.nllb)[0])
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
