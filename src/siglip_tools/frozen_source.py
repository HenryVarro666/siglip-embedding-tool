"""Bind frozen benchmark records to the measured source blobs instead of the working tree.

Every benchmark exporter computes an implementation fingerprint over named files of
this package. Before this module existed, those files were always read from the
working tree, so a frozen record could only be validated while the checkout was
byte-identical to the measurement. Now each benchmark directory carries
``frozen-source.json``: the SHA-256 of every bound file at the measurement commit.
Validation of a frozen record evaluates the same fingerprint formula over that table,
which proves the record is internally consistent with the measured code, while the
working tree is free to evolve. Executing a measurement still requires the live code
to match the protocol it declares.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path

from .encoder import identity_fingerprint
from .images import file_sha256

PACKAGE_DIRECTORY = "src/siglip_tools"
PACKAGE_ROOT = Path(__file__).resolve().parent
SNAPSHOT_FILENAME = "frozen-source.json"
SCHEMA_VERSION = 1
_RESOLVER: ContextVar[Callable[[str], str] | None] = ContextVar("frozen_source", default=None)


def source_sha256(name: str) -> str:
    """SHA-256 of a package file: live by default, from the bound snapshot inside ``bound``."""
    resolver = _RESOLVER.get()
    if resolver is None:
        return file_sha256(PACKAGE_ROOT / name)
    return resolver(name)


@contextmanager
def bound(resolver: Callable[[str], str]):
    token = _RESOLVER.set(resolver)
    try:
        yield resolver
    finally:
        _RESOLVER.reset(token)


class TableResolver:
    """Serve hashes from a snapshot table and remember which files were requested."""

    def __init__(self, files: dict[str, str]):
        self.files = dict(files)
        self.requested: set[str] = set()

    def __call__(self, name: str) -> str:
        self.requested.add(name)
        try:
            return self.files[name]
        except KeyError as exc:
            raise ValueError(f"frozen-source snapshot lacks {name}") from exc


class GitResolver:
    """Hash package files as committed at one revision, without touching the working tree."""

    def __init__(self, commit: str, repository: Path | None = None):
        self.commit = commit
        self.repository = Path(repository or Path(__file__).resolve().parents[2])
        self.files: dict[str, str] = {}

    def __call__(self, name: str) -> str:
        if name not in self.files:
            completed = subprocess.run(
                [
                    "git",
                    "-C",
                    str(self.repository),
                    "show",
                    f"{self.commit}:{PACKAGE_DIRECTORY}/{name}",
                ],
                capture_output=True,
                check=False,
            )
            if completed.returncode != 0:
                raise ValueError(f"{name} is not present at {self.commit}")
            from hashlib import sha256

            self.files[name] = sha256(completed.stdout).hexdigest()
        return self.files[name]


def snapshot_path(metadata: str | Path) -> Path:
    return Path(metadata) / SNAPSHOT_FILENAME


def load_snapshot(metadata: str | Path) -> dict:
    path = snapshot_path(metadata)
    try:
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        if (
            snapshot["schema_version"] != SCHEMA_VERSION
            or snapshot["package_directory"] != PACKAGE_DIRECTORY
            or not isinstance(snapshot["files"], dict)
            or not isinstance(snapshot["bindings"], dict)
            or any(
                not isinstance(name, str) or not isinstance(digest, str) or len(digest) != 64
                for name, digest in snapshot["files"].items()
            )
        ):
            raise ValueError(f"Unsupported frozen-source snapshot: {path}")
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Missing or invalid frozen-source snapshot: {path}") from exc
    return snapshot


def has_snapshot(metadata: str | Path) -> bool:
    return snapshot_path(metadata).is_file()


def frozen_hash(metadata: str | Path, formula: Callable[[], str]) -> str:
    """Evaluate a fingerprint formula over the measured source table."""
    with bound(TableResolver(load_snapshot(metadata)["files"])):
        return formula()


def reference_implementation(
    metadata: str | Path, formula: Callable[[], str], expected: str | None = None
) -> str:
    """The implementation fingerprint a frozen record is bound to.

    With a snapshot present this is the fingerprint of the measured source blobs; a
    table that does not reproduce ``expected`` means the record was tampered with.
    Without a snapshot (development fixtures) the live formula is used, as before.
    """
    value = frozen_hash(metadata, formula) if has_snapshot(metadata) else formula()
    if expected is not None and value != expected:
        raise ValueError("frozen-source snapshot does not reproduce the frozen implementation")
    return value


def execution_drift(metadata: str | Path, formula: Callable[[], str]) -> dict:
    """Report whether the executing code equals the measured code for this record."""
    frozen = reference_implementation(metadata, formula)
    live = formula()
    return {"frozen": frozen, "executing": live, "identical": frozen == live}


def auto_translation_v1_implementation_hash() -> str:
    """auto_benchmark.implementation_hash as it was before NLLB joined the bound files."""
    from .fusion_benchmark import implementation_hash as fusion_hash

    return identity_fingerprint(
        {
            "fusion": fusion_hash(),
            **{
                name: source_sha256(name)
                for name in ["translation.py", "auto_retrieval.py", "auto_benchmark.py"]
            },
        }
    )


def _load_formula(spec: str) -> Callable[[], str]:
    module_name, _, function_name = spec.partition(":")
    if not module_name or not function_name:
        raise ValueError("Formula must be written as module:function")
    module = import_module(f"siglip_tools.{module_name}")
    return getattr(module, function_name)


def _git_commits(repository: Path) -> list[str]:
    completed = subprocess.run(
        ["git", "-C", str(repository), "rev-list", "--all"],
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.split()


def discover(
    metadata: Path, formula: Callable[[], str], expected: str, repository: Path
) -> GitResolver:
    """Find the newest commit whose bound files reproduce ``expected``."""
    for commit in _git_commits(repository):
        resolver = GitResolver(commit, repository)
        try:
            with bound(resolver):
                value = formula()
        except ValueError:
            continue
        if value == expected:
            return resolver
    raise ValueError(f"No commit reproduces {expected} for {metadata}")


def write_snapshot(
    metadata: Path,
    primary: str,
    extras: dict[str, tuple[str, str]],
    repository: Path,
) -> dict:
    """Create ``frozen-source.json`` from git history and verify every declared binding.

    ``primary`` is the formula bound to ``freeze.json`` ``implementation_sha256``;
    ``extras`` maps a binding name to (formula spec, expected value).
    """
    freeze = json.loads((metadata / "freeze.json").read_text(encoding="utf-8"))
    expected = freeze["implementation_sha256"]
    formula = _load_formula(primary)
    resolver = discover(metadata, formula, expected, repository)
    bindings = {"implementation_sha256": {"formula": primary, "value": expected}}
    with bound(resolver):
        for name, (spec, value) in extras.items():
            actual = _load_formula(spec)()
            if actual != value:
                raise ValueError(
                    f"{name} does not reproduce at {resolver.commit}: {actual} != {value}"
                )
            bindings[name] = {"formula": spec, "value": value}
    snapshot = {
        "schema_version": SCHEMA_VERSION,
        "package_directory": PACKAGE_DIRECTORY,
        "source_commit": resolver.commit,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "note": (
            "SHA-256 of every package file bound by the frozen implementation fingerprint, taken "
            "from the measured commit. Validation evaluates the fingerprint over this table so the "
            "working tree may change without relabelling the measurement."
        ),
        "files": dict(sorted(resolver.files.items())),
        "bindings": bindings,
    }
    path = snapshot_path(metadata)
    path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return snapshot


def verify(metadata: Path) -> dict:
    """Recompute every recorded binding from the table; raise on any mismatch."""
    snapshot = load_snapshot(metadata)
    checks = {}
    with bound(TableResolver(snapshot["files"])):
        for name, binding in snapshot["bindings"].items():
            actual = _load_formula(binding["formula"])()
            if actual != binding["value"]:
                raise ValueError(f"{metadata}: {name} does not reproduce from frozen-source.json")
            checks[name] = actual
    freeze = json.loads((metadata / "freeze.json").read_text(encoding="utf-8"))
    if freeze.get("implementation_sha256") != checks["implementation_sha256"]:
        raise ValueError(
            f"{metadata}: freeze.json implementation differs from the snapshot binding"
        )
    return {
        "source_commit": snapshot["source_commit"],
        "bindings": checks,
        "files": len(snapshot["files"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("snapshot", help="Create frozen-source.json for one benchmark directory")
    make.add_argument("--metadata", type=Path, required=True)
    make.add_argument(
        "--formula", required=True, help="module:function bound to implementation_sha256"
    )
    make.add_argument(
        "--extra",
        action="append",
        default=[],
        metavar="NAME=module:function=VALUE",
        help="Additional binding that must reproduce at the same commit",
    )
    make.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[2])
    check = sub.add_parser("verify", help="Verify frozen-source.json in one or more directories")
    check.add_argument("metadata", type=Path, nargs="+")
    args = parser.parse_args(argv)
    if args.command == "snapshot":
        extras = {}
        for item in args.extra:
            name, spec, value = item.split("=", 2)
            extras[name] = (spec, value)
        snapshot = write_snapshot(args.metadata, args.formula, extras, args.repository)
        print(
            json.dumps(
                {"source_commit": snapshot["source_commit"], "files": len(snapshot["files"])}
            )
        )
        return 0
    failures = 0
    for metadata in args.metadata:
        try:
            print(json.dumps({str(metadata): verify(metadata)}))
        except ValueError as exc:
            failures += 1
            print(f"error: {exc}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    # ``python -m`` executes this file as ``__main__``; the formulas import the canonical
    # module, so delegate to it or the bound resolver would live in a different module.
    from siglip_tools.frozen_source import main as canonical_main

    raise SystemExit(canonical_main())
