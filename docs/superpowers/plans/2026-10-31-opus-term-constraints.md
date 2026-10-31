# OPUS terminology constraints implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compare original and terminology-constrained pretrained OPUS translation on new queries, preserving full evidence and continuous issue documentation.

**Architecture:** A wrapper shares the existing frozen OPUS weights and adds optional beam4/source-triggered constraints through a pinned local callable. A separate benchmark interleaves seven request policies on each existing SigLIP index; previous measured implementations remain unchanged. Versioned metadata and a durable issue ledger connect historical results, current work and future handoff.

**Tech Stack:** Python3.12, PyTorch2.15, Transformers5.22.1, existing local pretrained checkpoints and NumPy exact search.

**Spec:** ../specs/2026-10-31-opus-term-constraints-design.md

## Global constraints

- No training, no post-editing English, no changing previous inputs; no tuning on new queries.
- Pinned source glossary and context rubric commit3e0da59, external decoder revision07b2f120b7db38f1d7bac617ad65ea130508f297, existing OPUS and both SigLIP immutable revisions.
- Compare original beam1, ordinary beam4, constrained beam4; same max_new_tokens64/source256/f32/MPS, 0.5/0.5 fusion; every auto request translates.
- Source target-token alternatives use Marian target SPM. Verify constraint completion and EOS; no silent fallback or appended terms.
-16 new intents with bilingual metadata and shared40-image gallery; AI annotation, not human/field gold. Freeze before any new-query generation or scoring.
- Every observed problem/fix is persisted with evidence in docs/issue-log.md (historical detail linked separately); update continuation and doc index at handoff.

## Review focus

- A source microinverter must not require an extra generic inverter phrase; contextual bushing excludes cable conduits.
- Queries without glossary terms must follow ordinary beam4, with no injected term.
- Word inclusion can lose negation, wiring, support or under/behind relationships; grade these separately, before viewing retrieval rankings.
- A failed generation/retrieval must stay in the metric denominator, with explicit failure timing/status.
- Preserve base generation configuration and include code/profile/decoder/runtime/artifact/index identity in all comparisons.

### Task1: OPUS wrapper and local decoder preparation

**Files:** Create src/siglip_tools/opus_constraints.py, tests/test_opus_constraints.py; consume frozen src/siglip_tools/power_glossary_v1.json.
**Interfaces:** OPUSConstraintTranslator(base, constrained=True, decoder_dir=...); identity/model_dir/device/translate/last_trace; glossary_requirements(text), prepare_decoder(output).

- [ ] Write and observe failing tests for matching, target tokenization, no-term path, configuration immutability, missing terms/EOS, inventory drift and real tiny Marian callable flow.
- [ ] Implement the wrapper/verified local loader; only decoder source allowlist, never attached Qwen weights.
- [ ] Run focused and full offline tests; save discovered problems and fixes in ledger.

### Task2: New query/label metadata and text-review rules

**Files:** New benchmark queries.json/qrels.json/annotation-notes.json, byte-identical existing manifest copy; frozen glossary/review-rules already committed.
**Interfaces:**16 intent IDs t01..t16, zh/en paired labels; expected term/context slots for independent review; never feed reference EN to translation.

- [ ] Fresh curator checks source photos, all positives and close negatives without historical queries or model results.
- [ ] Confirm label completeness/visible evidence; validate dataset loader, freeze hashes and annotate shared-gallery limits.

### Task3: Benchmark, CLI and evidence capture

**Files:** Create src/siglip_tools/term_benchmark.py, tests/test_term_benchmark.py; modify cli.py/test_cli.py for explicit --opus-decoding, decoder directory.
**Interfaces:** timed_attempt(index, encoder, translator, chinese, mode, top_k), evaluate_attempts(rankings,qrels,gallery_ids), run, compare and exports for existing HTML renderer. Record zero-score failed attempts and raw per-request status/timing.

- [ ] Write failing tests for real total/stages, failure accounting, incompatible comparisons, frozen source checks and CLI exclusivity/default preservation.
- [ ] Implement7-policy interleaved runner with shared OPUS, immutable wrappers, startup/first request separate, full ranking and raw warm samples, identities and drift checks.
- [ ] Test wrappers on OLD development queries only; log any engineering fixes, then freeze code/artifacts/protocol and commit.
- [ ] Run Base then So400m in separate processes; no simultaneous model jobs. Never change rules to improve new test scores.

### Task4: Results, persistent documentation and delivery

**Files:** docs/terminology-constraints-v1.md, docs/{README,issue-log,historical-issues,continuation,improvements,models}.md, README/AGENTS, benchmark summaries/translation/review/audit/provenance/freeze.

- [ ] Review original/beam/constrained English terms and context without retrieval scores; independent relevance audit; preserve uncertain cases.
- [ ] Recompute all policy rankings/metrics and timing percentiles, report per-intent tradeoffs/failure rates and use matched beam comparison.
- [ ] Link every old/current issue and future recording rule; save reproduction commands and Top5 page.
- [ ] Full offline tests, Ruff, original loss demo, wheel/sdist/package resources, independent checkout; independent review, commit, private push and GitHub CI.
