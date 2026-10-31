# SigLIP Embedding Tool Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task by task. The user's feature request authorizes execution; routine choices are recorded here. A fresh read-only reviewer checks the completed implementation.

**Goal:** Provide reusable pretrained image/text embeddings, batch image indexing, text-to-image and image-to-image retrieval, and preserved optional training code.

**Architecture:** A local Python package separates preprocessing, pretrained encoding, index persistence/search, and a small CLI. The encoder's immutable model identity travels with every index. Existing notes and numerical examples remain available.

**Tech Stack:** Python 3.12, PyTorch, Transformers, Hugging Face Hub, NumPy, Pillow, pytest, Ruff.

**Spec:** `docs/design.md`

## Global Constraints

- Work in `/Users/caochao/Documents/SigLIP/`; preserve all existing notes and numerical example files.
- Use pretrained models first; do not execute training.
- Default to a private GitHub repository; exclude local photos, weights, vectors, caches, and local reports.
- No claimed retrieval quality from a two-image smoke test.
- All returned embeddings are finite, nonzero, normalized CPU float32 matrices.
- Model and processor use the same immutable revision; incompatible indexes fail clearly.

## Review Focus

- Same-dimensional embeddings from different checkpoints must be rejected: index compatibility tests.
- Metadata, EXIF and transparency variants must preserve deterministic row/path mapping: preprocessing and duplicate tests.
- Partial failures and existing index files must not be silently overwritten: atomic persistence tests.
- Index load must reject malformed metadata, shape mismatch, nonfinite vectors and unsafe relative paths: validation tests.
- Cached Transformers feature APIs vary between tensors and pooled outputs: explicit adapter tests plus real pretrained inference.

## Task 1 Package and image preprocessing

**Files:** `pyproject.toml`, `src/siglip_tools/images.py`, `tests/test_images.py`.

**Interfaces:** `load_rgb(path) -> PIL.Image.Image`; `scan_images(root) -> list[ImageRecord]`, where each record contains relative path, file hash, pixel hash, and aliases.

- [x] Write tests for RGB/EXIF/alpha behavior, exact-pixel deduplication, empty input and bad image errors; run RED.
- [x] Implement the preprocessing and deterministic scan; run GREEN.
- [x] Install the package into a local environment and retain a reproducible dependency lock.

## Task 2 Encoder and index

**Files:** `src/siglip_tools/encoder.py`, `src/siglip_tools/index.py`, `tests/test_encoder.py`, `tests/test_index.py`.

**Interfaces:** `SigLIPEmbedder(model_id=DEFAULT_MODEL, revision=None, device='auto', local_files_only=False)`; `encode_images`, `encode_texts`, `identity`. `ImageIndex(vectors, records, model_identity, image_root)`; `save`, `load`, `assert_compatible`, `search`.

- [x] Write failing tests for normalized matrix outputs, pooled-output adaptation, identity mismatch, ranking, exclusions and persistence errors.
- [x] Implement pretrained loading with pinned revision, robust feature extraction and finite/shape checks.
- [x] Implement exact search and safe single-file persistence; run all relevant tests.

## Task 3 CLI and training backup

**Files:** `src/siglip_tools/cli.py`, `src/siglip_tools/losses.py`, `training/finetune.py`, `tests/test_cli.py`, `tests/test_training.py`.

**Interfaces:** `siglip-tools index/embed-images`, `search-text`, `search-image`, `inspect`; optional training JSONL `{image, text}`.

- [x] Write failing tests for argument errors and training manifest/loss behavior; implement and run GREEN.
- [x] Preserve existing root loss demo and tests; no training is executed.
- [x] Run the real model against the user's two original images and added runtime image, save local-only outputs and capture latency/rankings.

## Task 4 Documentation and GitHub delivery

**Files:** `README.md`, `docs/models.md`, `docs/improvements.md`, `docs/data-assessment.md`, CI workflow, `.gitignore`.

- [x] Document API/CLI commands, cache/device behavior, model identity guarantees, dataset limitations and acceptance criteria for improvements.
- [x] Run full pytest, existing unittest demo, lint, packaging and clean-clone checks.
- [x] Dispatch a fresh read-only whole-repository reviewer; address correctness findings with regression tests.
- [ ] Commit and push a private repository using the authenticated GitHub account; verify its default branch and privacy.
