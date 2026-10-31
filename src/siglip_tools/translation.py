"""Pinned, local Chinese-to-English pretrained translation for retrieval queries.

Preparation converts the official PyTorch checkpoint to safetensors once. Query
inference only reads the resulting local, hash-validated artifacts.
"""

import argparse
import json
import re
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

import torch

from .encoder import _device, identity_fingerprint
from .images import file_sha256

MODEL_ID = "Helsinki-NLP/opus-mt-zh-en"
REVISION = "cf109095479db38d6df799875e34039d4938aaa6"
PROVENANCE_FILENAME = "translation_provenance.json"
MAX_INPUT_TOKENS = 256
MAX_NEW_TOKENS = 64
_SOURCE_FILES = [
    "pytorch_model.bin",
    "config.json",
    "generation_config.json",
    "source.spm",
    "target.spm",
    "vocab.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "README.md",
]
_CHINESE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U000323af]")


def _runtime_versions() -> dict:
    return {
        name: version(name)
        for name in ("torch", "transformers", "tokenizers", "sentencepiece", "sacremoses")
    }


def _artifact_hashes(directory: Path) -> dict:
    files = sorted(directory.rglob("*"))
    if any(path.is_symlink() for path in files):
        raise ValueError("Translator artifacts must not contain symlinks")
    return {
        path.relative_to(directory).as_posix(): file_sha256(path)
        for path in files
        if path.is_file() and path != directory / PROVENANCE_FILENAME
    }


def _validated_provenance(directory: Path) -> dict:
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
            or source["license"] != "cc-by-4.0"
            or source["weight_filename"] != "pytorch_model.bin"
            or not re.fullmatch(r"[0-9a-f]{64}", source["weight_sha256"])
        ):
            raise ValueError("Translator provenance has an unexpected source or revision")
        artifacts = provenance["artifacts"]
        required = {"model.safetensors", "config.json", "tokenizer_config.json"}
        if not isinstance(artifacts, dict) or not required.issubset(artifacts):
            raise ValueError("Translator provenance is missing required artifacts")
        if any(
            not isinstance(name, str)
            or not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
            for name, digest in artifacts.items()
        ):
            raise ValueError("Translator provenance contains invalid artifact hashes")
        if any(name.endswith(".bin") for name in artifacts):
            raise ValueError("Local translator artifacts must use converted safetensors")
        actual = _artifact_hashes(directory)
        if actual.keys() != artifacts.keys():
            raise ValueError("Translator artifact inventory differs from its provenance")
        if actual != artifacts:
            raise ValueError("Translator artifact hash mismatch; prepare a fresh checkpoint")
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Missing or invalid translator provenance") from exc
    return provenance


def prepare_translator(output: Path) -> Path:
    """Download the fixed official revision and convert weights, without training."""
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Translator output already exists: {output}")
    source = Path(
        snapshot_download(repo_id=MODEL_ID, revision=REVISION, allow_patterns=_SOURCE_FILES)
    )
    original_sha256 = file_sha256(source / "pytorch_model.bin")
    common = {"local_files_only": True, "trust_remote_code": False}
    tokenizer = AutoTokenizer.from_pretrained(str(source), **common)
    model = (
        AutoModelForSeq2SeqLM.from_pretrained(
            str(source), use_safetensors=False, weights_only=True, dtype=torch.float32, **common
        )
        .to(device="cpu", dtype=torch.float32)
        .eval()
    )
    model.requires_grad_(False)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".translator-", dir=output.parent) as temporary:
        staging = Path(temporary) / "checkpoint"
        staging.mkdir()
        model.save_pretrained(staging, safe_serialization=True)
        tokenizer.save_pretrained(staging)
        if (source / "README.md").is_file():
            copyfile(source / "README.md", staging / "README.md")
        provenance = {
            "schema_version": 1,
            "source": {
                "model_id": MODEL_ID,
                "revision": REVISION,
                "license": "cc-by-4.0",
                "weight_filename": "pytorch_model.bin",
                "weight_sha256": original_sha256,
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


class LocalTranslator:
    """Fixed pretrained zh→en translation with no query or output memoization."""

    def __init__(self, model_dir: Path, device: str = "mps"):
        directory = Path(model_dir).expanduser().resolve()
        provenance = _validated_provenance(directory)
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.model_dir = directory
        self.device = _device(device)
        common = {"local_files_only": True, "trust_remote_code": False}
        self.tokenizer = AutoTokenizer.from_pretrained(str(directory), **common)
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
        settings.use_cache = True
        # Marian normally forces EOS at the generation cap, hiding truncation.
        settings.forced_eos_token_id = None
        settings.return_dict_in_generate = False
        settings.output_scores = False
        settings.output_logits = False
        self._generation_config = settings
        eos = settings.eos_token_id
        if eos is None:
            raise ValueError("Translator configuration must define an EOS token")
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
            "max_input_tokens": MAX_INPUT_TOKENS,
            "generation_config": settings.to_dict(),
            "generation": settings.to_dict(),
            "pipeline_version": 1,
        }

    @property
    def identity(self) -> dict:
        return deepcopy(self._identity)

    def translate(self, text: str) -> str:
        """Translate a Chinese query, rejecting silent input or output truncation."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Translation requires a nonempty Chinese query")
        if not _CHINESE.search(text):
            raise ValueError("Translation source must contain Chinese text")
        inputs = self.tokenizer(text.strip(), padding=False, truncation=False, return_tensors="pt")
        if inputs["input_ids"].shape[-1] > MAX_INPUT_TOKENS:
            raise ValueError(f"Translation input exceeds {MAX_INPUT_TOKENS} tokens")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            result = self.model.generate(
                **inputs, generation_config=deepcopy(self._generation_config)
            )
        sequence = result[0].detach().cpu().tolist()
        if not any(token in self._eos_ids for token in sequence[1:]):
            raise ValueError(
                f"Translation did not terminate with EOS within {MAX_NEW_TOKENS} tokens"
            )
        english = self.tokenizer.decode(sequence, skip_special_tokens=True).strip()
        if not english:
            raise ValueError("Translator returned an empty translation")
        return english


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
