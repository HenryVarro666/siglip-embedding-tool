"""Pinned NLLB-200 Chinese-to-English inference for research comparisons.

The official checkpoint is CC-BY-NC-4.0. Preparation verifies its pinned
PyTorch weights and converts them on CPU; inference uses validated local
safetensors only. Neither path trains or fine-tunes the model.
"""

import argparse
import json
import re
from copy import deepcopy
from numbers import Integral
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

import torch

from .encoder import _device, identity_fingerprint
from .images import file_sha256
from .translation import (
    MAX_INPUT_TOKENS,
    MAX_NEW_TOKENS,
    PROVENANCE_FILENAME,
    LocalTranslator,
    _artifact_hashes,
    _runtime_versions,
)

MODEL_ID = "facebook/nllb-200-1.3B"
REVISION = "b0de46b488af0cf31749cd8da5ed3171e11b2309"
SOURCE_LANGUAGE = "zho_Hans"
TARGET_LANGUAGE = "eng_Latn"
_SOURCE_WEIGHT_SHA256 = "8a134f26de3677e0b7c47d3e98c199fdb4c85a8c6577448d861af4ff24e2ad3f"
_SOURCE_WEIGHT_SIZE_BYTES = 5482882236
_SOURCE_FILES = [
    "pytorch_model.bin",
    "config.json",
    "generation_config.json",
    "sentencepiece.bpe.model",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "README.md",
]


def _valid_hashes(hashes: dict) -> bool:
    return isinstance(hashes, dict) and all(
        isinstance(name, str)
        and isinstance(digest, str)
        and re.fullmatch(r"[0-9a-f]{64}", digest)
        for name, digest in hashes.items()
    )


def _validated_provenance(directory: Path) -> dict:
    """Validate this adapter's source and complete local inventory before loading."""
    try:
        manifest = directory / PROVENANCE_FILENAME
        if manifest.is_symlink():
            raise ValueError("Translator provenance must not be a symlink")
        provenance = json.loads(manifest.read_text(encoding="utf-8"))
        source = provenance["source"]
        if (
            provenance["schema_version"] != 1
            or source["model_id"] != MODEL_ID
            or source["revision"] != REVISION
            or source["license"] != "cc-by-nc-4.0"
            or source["weight_filename"] != "pytorch_model.bin"
            or source["weight_sha256"] != _SOURCE_WEIGHT_SHA256
            or source["weight_size_bytes"] != _SOURCE_WEIGHT_SIZE_BYTES
        ):
            raise ValueError("NLLB translator provenance has an unexpected source or revision")
        source_hashes = source["artifact_hashes"]
        if (
            not _valid_hashes(source_hashes)
            or set(source_hashes) != set(_SOURCE_FILES)
            or source_hashes["pytorch_model.bin"] != _SOURCE_WEIGHT_SHA256
        ):
            raise ValueError("NLLB source artifact hash inventory is invalid")
        conversion = provenance["conversion"]
        if (
            conversion["format"] != "safetensors"
            or conversion["dtype"] != "float32"
            or conversion["training"] is not False
            or not isinstance(conversion["runtime_versions"], dict)
            or not conversion["runtime_versions"]
            or not all(
                isinstance(value, str) and value
                for value in conversion["runtime_versions"].values()
            )
        ):
            raise ValueError("NLLB translator conversion record is invalid")
        artifacts = provenance["artifacts"]
        required = {
            "model.safetensors",
            "config.json",
            "generation_config.json",
            "tokenizer_config.json",
            "README.md",
        }
        if not _valid_hashes(artifacts) or not required.issubset(artifacts):
            raise ValueError("NLLB translator provenance is missing required artifact hashes")
        if not {"tokenizer.json", "sentencepiece.bpe.model"}.intersection(artifacts):
            raise ValueError("NLLB translator provenance is missing tokenizer artifacts")
        if any(
            name.endswith(".bin") or (name.endswith(".safetensors") and name != "model.safetensors")
            for name in artifacts
        ):
            raise ValueError("Local NLLB weights must use a single model.safetensors file")
        actual = _artifact_hashes(directory)
        if actual.keys() != artifacts.keys():
            raise ValueError("NLLB translator artifact inventory differs from its provenance")
        if actual != artifacts:
            raise ValueError("NLLB translator artifact hash mismatch; prepare a fresh checkpoint")
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Missing or invalid NLLB translator provenance") from exc
    return provenance


def _language_token_id(tokenizer, language: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(language)
    if (
        not isinstance(token_id, Integral)
        or isinstance(token_id, bool)
        or token_id < 0
        or token_id == tokenizer.unk_token_id
        or tokenizer.convert_ids_to_tokens(token_id) != language
    ):
        raise ValueError(f"NLLB tokenizer has no valid language token for {language}")
    return int(token_id)


def prepare_translator(output: Path) -> Path:
    """Convert the exact official research checkpoint on CPU without training."""
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Translator output already exists: {output}")
    source = Path(
        snapshot_download(repo_id=MODEL_ID, revision=REVISION, allow_patterns=_SOURCE_FILES)
    )
    if (source / "pytorch_model.bin").stat().st_size != _SOURCE_WEIGHT_SIZE_BYTES:
        raise ValueError("Pinned NLLB source weight size mismatch")
    # Hugging Face snapshot files are cache symlinks; converted artifacts must
    # be regular local files, checked separately by _artifact_hashes.
    source_hashes = {name: file_sha256(source / name) for name in _SOURCE_FILES}
    if source_hashes["pytorch_model.bin"] != _SOURCE_WEIGHT_SHA256:
        raise ValueError("Pinned NLLB source weight SHA-256 checksum mismatch")
    common = {"local_files_only": True, "trust_remote_code": False}
    tokenizer = AutoTokenizer.from_pretrained(
        str(source), src_lang=SOURCE_LANGUAGE, legacy_behaviour=False, **common
    )
    tokenizer.src_lang = SOURCE_LANGUAGE
    _language_token_id(tokenizer, SOURCE_LANGUAGE)
    _language_token_id(tokenizer, TARGET_LANGUAGE)
    model = (
        AutoModelForSeq2SeqLM.from_pretrained(
            str(source), use_safetensors=False, weights_only=True, dtype=torch.float32, **common
        )
        .to(device="cpu", dtype=torch.float32)
        .eval()
    )
    model.requires_grad_(False)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".nllb-translator-", dir=output.parent) as temporary:
        staging = Path(temporary) / "checkpoint"
        staging.mkdir()
        model.save_pretrained(staging, safe_serialization=True, max_shard_size="8GB")
        tokenizer.save_pretrained(staging)
        copyfile(source / "README.md", staging / "README.md")
        provenance = {
            "schema_version": 1,
            "source": {
                "model_id": MODEL_ID,
                "revision": REVISION,
                "license": "cc-by-nc-4.0",
                "weight_filename": "pytorch_model.bin",
                "weight_sha256": source_hashes["pytorch_model.bin"],
                "weight_size_bytes": _SOURCE_WEIGHT_SIZE_BYTES,
                "artifact_hashes": source_hashes,
            },
            "conversion": {
                "format": "safetensors",
                "dtype": "float32",
                "training": False,
                "runtime_versions": _runtime_versions(),
            },
            "artifacts": _artifact_hashes(staging),
        }
        (staging / PROVENANCE_FILENAME).write_text(
            json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        _validated_provenance(staging)
        staging.rename(output)
    return output


class NLLBTranslator(LocalTranslator):
    """Fixed zho_Hans→eng_Latn translation with fresh generation every request."""

    def __init__(self, model_dir: Path, device: str = "mps"):
        directory = Path(model_dir).expanduser().resolve()
        provenance = _validated_provenance(directory)
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.model_dir = directory
        self.device = _device(device)
        common = {"local_files_only": True, "trust_remote_code": False}
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(directory), src_lang=SOURCE_LANGUAGE, legacy_behaviour=False, **common
        )
        self.tokenizer.src_lang = SOURCE_LANGUAGE
        source_token_id = _language_token_id(self.tokenizer, SOURCE_LANGUAGE)
        target_token_id = _language_token_id(self.tokenizer, TARGET_LANGUAGE)
        self.model = (
            AutoModelForSeq2SeqLM.from_pretrained(
                str(directory), use_safetensors=True, dtype=torch.float32, **common
            )
            .to(device=self.device, dtype=torch.float32)
            .eval()
        )
        self.model.requires_grad_(False)
        settings = deepcopy(self.model.generation_config)
        settings.do_sample = False
        settings.num_beams = 1
        settings.num_return_sequences = 1
        settings.max_new_tokens = MAX_NEW_TOKENS
        settings.max_length = None
        settings.use_cache = True
        settings.forced_bos_token_id = target_token_id
        # A forced EOS at the token cap would disguise a truncated translation.
        settings.forced_eos_token_id = None
        settings.return_dict_in_generate = False
        settings.output_scores = False
        settings.output_logits = False
        # Transformers fills None fields from this fallback even when a separate
        # per-call generation config is supplied by LocalTranslator.translate.
        self.model.generation_config = deepcopy(settings)
        self._generation_config = settings
        eos = settings.eos_token_id
        if eos is None:
            raise ValueError("NLLB translator configuration must define an EOS token")
        self._eos_ids = {eos} if isinstance(eos, int) else set(eos)
        artifacts = provenance["artifacts"]
        tokenizer_artifacts = {
            name: digest
            for name, digest in artifacts.items()
            if name
            not in {"model.safetensors", "config.json", "generation_config.json", "README.md"}
        }
        self._identity = {
            "model_id": MODEL_ID,
            "revision": REVISION,
            "source_model": MODEL_ID,
            "source_revision": REVISION,
            "source": provenance["source"],
            "conversion": provenance["conversion"],
            "artifact_fingerprint": identity_fingerprint(artifacts),
            "artifacts": artifacts,
            "artifact_hashes": artifacts,
            "tokenizer": type(self.tokenizer).__name__,
            "tokenizer_sha256": identity_fingerprint(tokenizer_artifacts),
            "runtime_versions": _runtime_versions(),
            "device": str(self.device),
            "dtype": "float32",
            "source_language": "zh",
            "target_language": "en",
            "source_language_code": SOURCE_LANGUAGE,
            "target_language_code": TARGET_LANGUAGE,
            "source_language_token_id": source_token_id,
            "target_language_token_id": target_token_id,
            "encoder_tokenization": {
                "src_lang": SOURCE_LANGUAGE,
                "legacy_behaviour": False,
                "text_preprocessing": "strip",
                "add_special_tokens": True,
                "padding": False,
                "truncation": False,
                "return_tensors": "pt",
            },
            "max_input_tokens": MAX_INPUT_TOKENS,
            "generation_config": settings.to_dict(),
            "generation": settings.to_dict(),
            "pipeline_version": 1,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="Convert the pinned official pretrained model")
    prepare.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(prepare_translator(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
