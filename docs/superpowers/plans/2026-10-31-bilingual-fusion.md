# Bilingual Query Fusion Implementation Plan

> **For agentic workers:** Implement with TDD in this session, then request an independent whole-branch review. Preserve the existing equipment-v1 inputs and reports.

**Goal:** Explain the three Chinese AP regressions and measure fixed equal-weight bilingual retrieval for Base and So400m, without training.

**Architecture:** Reuse each checkpoint's compatible frozen image index. Encode the existing Chinese/English pairs, average their cosine scores, and compare Chinese-only, English-only and fusion over the same eight intents. Save full local rankings/timings and a tracked report/curated summary.

**Tech stack:** Existing locked PyTorch/Transformers, NumPy, pytest; no new dependency.

**Spec:** User request in this chat and `benchmarks/bilingual-fusion-v1/protocol.json`.

## Global constraints

- Preserve equipment-v1 images, queries, qrels and their frozen hashes. No training, prompt rewriting, per-query weight selection or new OCR/crop model.
- Use each model's own index and pinned checkpoint. Native Base224 / So400m384; MPS, float32, text batch1 for single-language and2 for fusion.
- Fusion score is `0.5*cos(image,zh)+0.5*cos(image,en)`. Do not renormalize the mean query vector when reporting scores.
- Eight intent-level rankings per strategy. Every English query is already supplied; runtime translation is not implemented or included in latency.
- Sequential model processes,3 warmups per strategy,10 repeats per intent/policy; rotate policy order to reduce measurement-order bias. Time tokenization, text encoding, CPU transfer, score fusion and full ranking with accelerator synchronization.
- Keep images, indexes, complete rankings and individual timing samples ignored. Track protocol/freeze, code/tests, curated numerical summary, diagnosis and improvements.

## Review focus

- Opposite query vectors legitimately produce zero average scores and stable gallery-order ties.
- A matching embedding dimension cannot bypass model/processor identity.
- Index paths/hash/aliases must exactly match the frozen gallery; reject drift before model allocation and after timing.
- One bilingual fused result must not count as two independent intents; English-only is an explicit control.
- Cached English pairs and cached image indexes do not represent an automatic Chinese-to-English service; timings must name excluded work.

## Task1: Fusion API and existing CLI

- [x] Write `tests/test_fusion.py` hand-derived rank/score tests: equal-weight two-dimensional queries, one query, cancelling queries, invalid weights/shapes/top-k and incompatible identity. Add CLI dual-query behavior test.
- [x] Run focused tests and observe expected failures before implementation:25 RED, then25 GREEN.
- [x] Add `src/siglip_tools/fusion.py`: `weighted_query_vector(query_vectors, weights)` and `search_texts(index, encoder, texts, weights=None, top_k=5)` returning existing SearchResult objects.
- [x] Add optional `--english-query` to existing search-text; preserve single-query behavior. Full offline suite175passed before runner addition; lint clean.

## Task2: Frozen comparison runner

- [x] Write tests for frozen index validation, protocol/dataset drift, same-checkpoint comparison, intent pairing and a complete mocked-model persisted run:7 RED then7 GREEN.
- [x] Add `src/siglip_tools/fusion_benchmark.py` run/compare CLI. Reuse evaluation metrics and baseline synchronization/environment utilities. Record single-language baselines, bilingual full rankings, token counts, query-vector cosine, timing samples and identity/input/code hashes.
- [ ] Verify tests, commit implementation/protocol, save a pre-run freeze of data/protocol/code/index hashes.
- [ ] Run both models sequentially using existing cached checkpoints and own indexes. Independently compare new zh/en baseline ranks to historical results; do not alter inputs or weight after results.

## Task3: Analysis, reporting and publication

- [ ] Inspect the three Chinese regressions, preserve rank evidence, English controls, visual/source boundaries and uncertainty about causes.
- [ ] Report all eight intents and strategies, AP/Recall changes and per-query latency p50/p95. Check if fusion beats Chinese-only and whether English-only is already better.
- [ ] Save tracked summary/report, reproducible commands, local Top-K review and updated improvements/README. Keep the previous experiment intact.
- [ ] Independent read-only code/results audit, full offline tests/lint, clean-clone check, push to the existing private repo and verify CI.
