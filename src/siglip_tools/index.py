"""Exact cosine image retrieval and single-file index persistence."""

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from numbers import Integral
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np

from .encoder import identity_fingerprint
from .images import ImageRecord, pixel_sha256, read_image_snapshot, scan_images, validate_records

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SearchResult:
    path: str
    score: float
    rank: int
    aliases: tuple[str, ...]


def _relative_path(path: str) -> None:
    if not isinstance(path, str) or not path or "\\" in path or "\x00" in path:
        raise ValueError("Image paths must be safe relative POSIX paths")
    parsed = PurePosixPath(path)
    if (
        parsed.is_absolute()
        or ".." in parsed.parts
        or path in {".", ""}
        or PureWindowsPath(path).drive
        or PureWindowsPath(path).root
    ):
        raise ValueError("Image paths must be safe relative paths")


class ImageIndex:
    @classmethod
    def from_directory(cls, encoder, image_root, batch_size=8, *, records=None):
        """Encode verified RGB snapshots, then validate every canonical/alias path."""
        if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        root = Path(image_root).expanduser().resolve()
        records = scan_images(root) if records is None else list(records)
        if not records:
            raise ValueError("No supported images to encode")
        for record in records:
            for path in (record.path, *record.aliases):
                _relative_path(path)
        matrices = []
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            snapshots = []
            for record in batch:
                image, digest = read_image_snapshot(root / record.path)
                if digest != record.sha256 or pixel_sha256(image) != record.pixel_sha256:
                    raise ValueError(f"Image changed during indexing: {record.path}; retry")
                snapshots.append(image)
            matrices.append(encoder.encode_images(snapshots, batch_size=batch_size))
        validate_records(root, records)
        return cls(np.concatenate(matrices), records, encoder.identity, root)

    def __init__(self, vectors, records, model_identity: dict, image_root: str | Path):
        vectors = np.array(vectors, dtype=np.float32, copy=True)
        if vectors.ndim != 2 or min(vectors.shape) == 0:
            raise ValueError("Index embeddings must be a nonempty two-dimensional matrix")
        if len(records) != vectors.shape[0]:
            raise ValueError("Embedding rows and image records must have the same count")
        if not np.isfinite(vectors).all():
            raise ValueError("Index embeddings must be finite")
        if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4):
            raise ValueError("Index embeddings must be L2 normalized")
        seen_paths = set()
        for record in records:
            for path in (record.path, *record.aliases):
                _relative_path(path)
                if path in seen_paths:
                    raise ValueError(f"Duplicate path in index records: {path}")
                seen_paths.add(path)
            for digest in (record.sha256, record.pixel_sha256):
                if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                    raise ValueError("Image records require SHA-256 hashes")
        if not isinstance(model_identity, dict) or any(
            not isinstance(model_identity.get(key), str) or not model_identity[key]
            for key in ("model_id", "revision")
        ):
            raise ValueError("Index requires model_id and revision provenance")
        self.model_identity = json.loads(json.dumps(model_identity, allow_nan=False))
        self.fingerprint = identity_fingerprint(self.model_identity)
        vectors.setflags(write=False)
        self.vectors = vectors
        self.records = tuple(records)
        self.image_root = Path(image_root).expanduser().resolve()

    def search(self, query, top_k: int = 5, exclude_pixel_sha256: str | None = None):
        if isinstance(top_k, bool) or not isinstance(top_k, Integral) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        query = np.asarray(query, dtype=np.float32)
        if query.shape == (1, self.vectors.shape[1]):
            query = query[0]
        if query.shape != (self.vectors.shape[1],) or not np.isfinite(query).all():
            raise ValueError("Query must be a finite vector matching the index dimension")
        norm = np.linalg.norm(query)
        if not np.isfinite(norm) or norm == 0:
            raise ValueError("Query must have a finite, nonzero norm")
        scores = self.vectors @ (query / norm)
        ordered = np.argsort(-scores, kind="stable")
        results = []
        for position in ordered:
            record = self.records[position]
            if record.pixel_sha256 == exclude_pixel_sha256:
                continue
            results.append(
                SearchResult(record.path, float(scores[position]), len(results) + 1, record.aliases)
            )
            if len(results) == top_k:
                break
        return results

    def assert_compatible(self, identity: dict) -> None:
        if self.fingerprint != identity_fingerprint(identity):
            raise ValueError(
                "Incompatible model or preprocessing identity. Use the index's pinned "
                "checkpoint and environment, or rebuild the index; dimension alone is insufficient."
            )

    def save(self, path: str | Path, overwrite: bool = False) -> None:
        path = Path(path)
        if path.exists() and not overwrite:
            raise FileExistsError(f"Index already exists: {path}; use overwrite=True")
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "model_identity": self.model_identity,
            "fingerprint": self.fingerprint,
            "image_root": str(self.image_root),
            "records": [asdict(record) for record in self.records],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=path.parent, prefix=".siglip-", suffix=".tmp", delete=False
            ) as stream:
                temporary = Path(stream.name)
                np.savez_compressed(
                    stream,
                    embeddings=self.vectors,
                    metadata=np.array(json.dumps(metadata, ensure_ascii=False, allow_nan=False)),
                )
                stream.flush()
                os.fsync(stream.fileno())
            if overwrite:
                os.replace(temporary, path)
            else:
                # Atomic no-clobber publication, even if another writer races us.
                os.link(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: str | Path):
        try:
            with np.load(path, allow_pickle=False) as stored:
                vectors = stored["embeddings"].copy()
                metadata = json.loads(stored["metadata"].item())
            if not isinstance(metadata, dict) or metadata.get("schema_version") != SCHEMA_VERSION:
                raise ValueError("Unsupported index schema")
            records = []
            for record in metadata["records"]:
                if not isinstance(record.get("aliases", []), list):
                    raise ValueError("Record aliases must be a list")
                records.append(
                    ImageRecord(**{**record, "aliases": tuple(record.get("aliases", []))})
                )
            index = cls(vectors, records, metadata["model_identity"], metadata["image_root"])
            if metadata["fingerprint"] != index.fingerprint:
                raise ValueError("Index model identity fingerprint is inconsistent")
            return index
        except (KeyError, TypeError, AttributeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid index metadata: {exc}") from exc
