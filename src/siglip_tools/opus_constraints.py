"""Source-triggered OPUS lexical constraints with a pinned local beam decoder."""

import argparse
import json
import re
import sys
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from types import ModuleType
from typing import Callable

import torch

from .translation import _CHINESE, MAX_INPUT_TOKENS, MAX_NEW_TOKENS, LocalTranslator

DECODER_REPO_ID = "transformers-community/constrained-beam-search"
DECODER_REVISION = "07b2f120b7db38f1d7bac617ad65ea130508f297"
DEFAULT_DECODER_DIR = Path("models/opus-constrained-decoder")
DECODER_PROVENANCE_FILENAME = "decoder_provenance.json"
DECODER_FILES = (
    "README.md",
    "custom_generate/generate.py",
    "custom_generate/beam_constraints.py",
    "custom_generate/beam_search.py",
)
DECODER_HASHES = {
    "README.md": "0b550f52f5ffb16b8f36a34b2e5c9cb40076994063dd72b283552370eb43d903",
    "custom_generate/generate.py": (
        "3f2975a90ef9f506077d8e54d0e145cfd735088c9c082c595d546cfe5cd7eb64"
    ),
    "custom_generate/beam_constraints.py": (
        "72da73e8f601167c895974bbb6064f1be7d14f33247f6a8ea0acc6c162b62704"
    ),
    "custom_generate/beam_search.py": (
        "247baefee037b9cebda29140abcc298ed2d4fdb32b79a40550a06b004a03bda5"
    ),
}

GLOSSARY_PATH = Path(__file__).with_name("power_glossary_v1.json")
_GLOSSARY = json.loads(GLOSSARY_PATH.read_text(encoding="utf-8"))
GLOSSARY_SHA256 = sha256(GLOSSARY_PATH.read_bytes()).hexdigest()
_DECODER_CACHE: dict[str, Callable] = {}
_DECODER_LOCK = Lock()


def _read_decoder(directory: Path) -> tuple[dict, dict[str, bytes]]:
    try:
        if directory.is_symlink():
            raise ValueError("Constrained decoder directory must not be a symlink")
        paths = list(directory.rglob("*"))
        if any(path.is_symlink() for path in paths):
            raise ValueError("Constrained decoder artifacts must not contain symlinks")
        inventory = {path.relative_to(directory).as_posix() for path in paths}
        expected = set(DECODER_FILES) | {DECODER_PROVENANCE_FILENAME, "custom_generate"}
        if inventory != expected:
            raise ValueError("Constrained decoder inventory differs from the pinned files")
        provenance = json.loads(
            (directory / DECODER_PROVENANCE_FILENAME).read_text(encoding="utf-8")
        )
        source = provenance["source"]
        if (
            provenance["schema_version"] != 1
            or provenance["training"] is not False
            or provenance["weights_downloaded"] is not False
            or source["repo_id"] != DECODER_REPO_ID
            or source["revision"] != DECODER_REVISION
            or source["repository_license"] is not None
            or provenance["artifacts"] != DECODER_HASHES
        ):
            raise ValueError("Constrained decoder provenance differs from the pinned source")
        contents = {name: (directory / name).read_bytes() for name in DECODER_FILES}
        if {name: sha256(data).hexdigest() for name, data in contents.items()} != DECODER_HASHES:
            raise ValueError("Constrained decoder hash mismatch; prepare a fresh decoder")
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Missing or invalid constrained decoder provenance") from exc
    return provenance, contents


def prepare_decoder(output: Path) -> Path:
    """Prepare only four pinned source files locally, without weights or training."""
    from huggingface_hub import snapshot_download

    output = Path(output).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Constrained decoder output already exists: {output}")
    source = Path(
        snapshot_download(
            repo_id=DECODER_REPO_ID, revision=DECODER_REVISION, allow_patterns=list(DECODER_FILES)
        )
    )
    contents = {name: (source / name).read_bytes() for name in DECODER_FILES}
    if {name: sha256(data).hexdigest() for name, data in contents.items()} != DECODER_HASHES:
        raise ValueError("Downloaded constrained decoder hash mismatch")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".opus-decoder-", dir=output.parent) as temporary:
        staging = Path(temporary) / "decoder"
        (staging / "custom_generate").mkdir(parents=True)
        for name, data in contents.items():
            (staging / name).write_bytes(data)
        provenance = {
            "schema_version": 1,
            "source": {
                "repo_id": DECODER_REPO_ID,
                "revision": DECODER_REVISION,
                "repository_license": None,
                "license_note": "No repository license; beam_search.py retains Apache-2.0 header.",
                "url": f"https://huggingface.co/{DECODER_REPO_ID}/tree/{DECODER_REVISION}",
            },
            "training": False,
            "weights_downloaded": False,
            "redistribution": "Local ignored artifact; external Python is not vendored in Git.",
            "artifacts": DECODER_HASHES,
        }
        (staging / DECODER_PROVENANCE_FILENAME).write_text(
            json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
        )
        _read_decoder(staging)
        staging.rename(output)
    return output.resolve()


def _load_decoder(directory: Path) -> tuple[Callable, dict]:
    # Validate even when code from this directory was already imported. Execute
    # the exact validated byte snapshot, without SourceFileLoader/pycache writes.
    provenance, contents = _read_decoder(directory)
    key = str(directory.resolve())
    with _DECODER_LOCK:
        if key not in _DECODER_CACHE:
            package_name = "_siglip_opus_decoder_" + sha256(key.encode()).hexdigest()
            package = ModuleType(package_name)
            package.__path__ = []
            sys.modules[package_name] = package
            loaded = [package_name]
            try:
                for stem in ("beam_constraints", "beam_search", "generate"):
                    name = f"{package_name}.{stem}"
                    filename = str(directory / f"custom_generate/{stem}.py")
                    module = ModuleType(name)
                    module.__file__, module.__package__ = filename, package_name
                    sys.modules[name] = module
                    loaded.append(name)
                    exec(
                        compile(contents[f"custom_generate/{stem}.py"], filename, "exec"),
                        module.__dict__,
                    )
                _DECODER_CACHE[key] = sys.modules[f"{package_name}.generate"].generate
            except Exception:
                for name in loaded:
                    sys.modules.pop(name, None)
                raise
    return _DECODER_CACHE[key], provenance


def _target_alternatives(tokenizer, forms: list[str]) -> list[list[int]]:
    alternatives = []
    special_ids = set(tokenizer.all_special_ids)
    for phrase in forms:
        # Marian has distinct source and target SentencePiece models. Ordinary
        # tokenizer(phrase) would silently constrain the wrong target tokens.
        token_ids = tokenizer(text_target=phrase, add_special_tokens=False)["input_ids"]
        if (
            not isinstance(token_ids, list)
            or not token_ids
            or any(
                type(token) is not int or token < 0 or token in special_ids for token in token_ids
            )
        ):
            raise ValueError(
                f"Cannot encode constraint phrase without special/unknown tokens: {phrase}"
            )
        alternatives.append(token_ids)
    return alternatives


def _decoder_alternatives(alternatives: list[list[int]]) -> list[list[int]]:
    # Upstream's disjunctive trie rejects duplicate and prefix-subset lists.
    # At token-inclusion level any longer prefix alternative is redundant.
    unique = list(dict.fromkeys(tuple(tokens) for tokens in alternatives))
    return [
        list(tokens)
        for tokens in unique
        if not any(len(other) < len(tokens) and tokens[: len(other)] == other for other in unique)
    ]


def _has_tokens(sequence: list[int], tokens: list[int]) -> bool:
    return any(
        sequence[start : start + len(tokens)] == tokens
        for start in range(len(sequence) - len(tokens) + 1)
    )


def glossary_requirements(text: str) -> list[dict]:
    """Return fresh longest-match requirements from the frozen power glossary."""
    occupied: set[int] = set()
    matches: dict[str, list[tuple[int, str]]] = {}
    sources = sorted(
        ((source, entry) for entry in _GLOSSARY["entries"] for source in entry["sources"]),
        key=lambda item: -len(item[0]),
    )
    for source, entry in sources:
        if entry.get("required_context", "") not in text:
            continue
        excluded: set[int] = set()
        for phrase in entry.get("exclude_within", []):
            offset = text.find(phrase)
            while offset != -1:
                excluded.update(range(offset, offset + len(phrase)))
                offset = text.find(phrase, offset + 1)
        start = text.find(source)
        while start != -1:
            positions = set(range(start, start + len(source)))
            if not positions & (occupied | excluded):
                occupied.update(positions)
                matches.setdefault(entry["id"], []).append((start, source))
            start = text.find(source, start + 1)
    return [
        {
            "id": entry["id"],
            "source_terms": list(dict.fromkeys(term for _, term in sorted(matches[entry["id"]]))),
            "target_forms": entry["target_forms"]
            + [term[0].upper() + term[1:] for term in entry["target_forms"]],
        }
        for entry in _GLOSSARY["entries"]
        if entry["id"] in matches
    ]


class OPUSConstraintTranslator:
    """Share frozen OPUS weights with ordinary or lexically constrained beam4."""

    def __init__(
        self,
        base: LocalTranslator,
        *,
        constrained: bool = True,
        decoder_dir: Path = DEFAULT_DECODER_DIR,
    ):
        self.model_dir, self.device = base.model_dir, base.device
        self.model, self.tokenizer = base.model, base.tokenizer
        self.constrained = constrained
        self.decoder_dir = Path(decoder_dir).expanduser().absolute()
        self._generation_config = deepcopy(base._generation_config)
        self._generation_config.num_beams = 4
        self._eos_ids = set(base._eos_ids)
        self._identity = base.identity
        self._identity.update(
            {
                "opus_decoding": "constrained_beam4" if constrained else "beam4",
                "constrained": constrained,
                "generation_config": self._generation_config.to_dict(),
                "generation": self._generation_config.to_dict(),
            }
        )
        self._decoder_generate = None
        if constrained:
            self._decoder_generate, provenance = _load_decoder(self.decoder_dir)
            self._identity.update(
                {
                    "decoder": provenance,
                    "glossary": {
                        "profile_id": _GLOSSARY["profile_id"],
                        "sha256": GLOSSARY_SHA256,
                        "profile": deepcopy(_GLOSSARY),
                    },
                }
            )
        self.last_trace: dict = {}

    @property
    def identity(self) -> dict:
        return deepcopy(self._identity)

    def translate(self, text: str) -> str:
        """Translate afresh and preserve the unedited English and audit trace."""
        self.last_trace = {
            "source_text": text,
            "constrained": self.constrained,
            "constraints_applied": False,
            "requirements": [],
            "required_token_alternatives": [],
            "output": None,
            "eos_terminated": False,
            "satisfied": None,
            "status": "pending",
        }
        try:
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Translation requires a nonempty Chinese query")
            if not _CHINESE.search(text):
                raise ValueError("Translation source must contain Chinese text")
            requirements = glossary_requirements(text) if self.constrained else []
            self.last_trace["requirements"] = requirements
            for requirement in requirements:
                requirement["token_alternatives"] = _target_alternatives(
                    self.tokenizer, requirement["target_forms"]
                )
                requirement["decoder_token_alternatives"] = _decoder_alternatives(
                    requirement["token_alternatives"]
                )
            self.last_trace["required_token_alternatives"] = [
                requirement["token_alternatives"] for requirement in requirements
            ]
            self.last_trace["constraints_applied"] = bool(requirements)
            inputs = self.tokenizer(
                text.strip(), padding=False, truncation=False, return_tensors="pt"
            )
            if inputs["input_ids"].shape[-1] > MAX_INPUT_TOKENS:
                raise ValueError(f"Translation input exceeds {MAX_INPUT_TOKENS} tokens")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            settings = deepcopy(self._generation_config)
            with torch.inference_mode():
                if requirements:
                    settings.constraints = None
                    settings.force_words_ids = [
                        requirement["decoder_token_alternatives"] for requirement in requirements
                    ]
                    # The public upstream function prepares and expands inputs
                    # once, then delegates to its private beam-search callable.
                    result = self._decoder_generate(
                        self.model, **inputs, generation_config=settings
                    )
                else:
                    result = self.model.generate(**inputs, generation_config=settings)
            sequence = result[0].detach().cpu().tolist()
            self.last_trace["sequence"] = sequence
            self.last_trace["eos_terminated"] = any(
                token in self._eos_ids for token in sequence[1:]
            )
            english = self.tokenizer.decode(sequence, skip_special_tokens=True)
            self.last_trace["output"] = english
            for requirement in requirements:
                requirement["tokens_satisfied"] = any(
                    _has_tokens(sequence[1:], tokens)
                    for tokens in requirement["token_alternatives"]
                )
                requirement["phrase_satisfied"] = any(
                    re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", english)
                    for phrase in requirement["target_forms"]
                )
                requirement["satisfied"] = bool(
                    requirement["tokens_satisfied"] and requirement["phrase_satisfied"]
                )
            self.last_trace["satisfied"] = all(
                requirement["satisfied"] for requirement in requirements
            )
            if not self.last_trace["eos_terminated"]:
                raise ValueError(
                    f"Translation did not terminate with EOS within {MAX_NEW_TOKENS} tokens"
                )
            if not english.strip():
                raise ValueError("Translator returned an empty translation")
            if any(not requirement["satisfied"] for requirement in requirements):
                self.last_trace["satisfied"] = False
                missing = [
                    requirement["id"]
                    for requirement in requirements
                    if not requirement["satisfied"]
                ]
                raise ValueError(f"Translation did not satisfy lexical constraints: {missing}")
            self.last_trace.update({"status": "success", "satisfied": True})
            return english
        except ValueError as exc:
            self.last_trace.update({"status": "failed", "error": str(exc)})
            raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser(
        "prepare", help="Prepare only pinned constrained decoder source"
    )
    prepare.add_argument("--output", type=Path, default=DEFAULT_DECODER_DIR)
    args = parser.parse_args(argv)
    print(prepare_decoder(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
