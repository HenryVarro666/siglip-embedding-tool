# Project guidance

- The default workflow is pretrained inference. Never start training just because backup training code exists.
- Read README.md for install and commands. Use `uv sync --frozen --extra dev` and Python 3.12.
- Run the full offline suite with `uv run pytest -m 'not integration'`; lint with `uv run ruff check src tests training`.
- Real-model tests are opt-in with `SIGLIP_RUN_MODEL_TESTS=1`, require the pinned default model cached locally, and do inference only.
- Preserve the original learning notes, conversation archive, loss demo and original tests.
- Model and processor revision, preprocessing and runtime identity travel with every index. Never bypass compatibility checks just because dimensions match.
- Keep raw images, weights, local indexes and local results out of Git. The repository is private by default.
- The two existing inverter photos are smoke-test references, not a retrieval benchmark. Filenames are not evidence of an exact SKU or physical object identity.
- Record limitations and future work in docs/improvements.md. Do not silently expand into OCR, a web service or 3D geometry generation.
