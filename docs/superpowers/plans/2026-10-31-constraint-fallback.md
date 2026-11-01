# Constraint Failure Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Real timed rejection fallback, frozen16-query evaluation on Base/So400m without training.

**Architecture:** Add a focused engine, a separate runner and explicit CLI flags. Reuse indexing/provenance/evaluation while leaving the v1 measured core unmodified.

**Tech Stack:** Python3.12, PyTorch/MPS, Transformers, pytest/Ruff.

**Spec:** docs/superpowers/specs/2026-10-31-constraint-fallback-design.md

## Global Constraints

- No training; fixed16queries/40images, beam1/beam4,64newtokens,equal cosine fusion.
- No edits to old translation.py/opus_constraints.py/term_benchmark.py, glossary, labels or protocols.
- Fresh translation per request; independently timed failed attempt plus fallback.
- Freeze code/artifacts/index/data/protocol before scoring; failures stay zero in main denominator.

## Review Focus

- Stale traces/invalid input must not trigger fallback.
- Retrieval errors and infrastructure errors must not trigger translation fallback.
- OPUS fallback failure terminates and retains both traces, without hidden Chinese cascade.
- Chinese fallback encodes once in fusion mode; rejected English remains evidence only.
- Coherently changed protocol/support files and mismatched model slots must fail validation/export.

### Task 1: Timed fallback engine

Files: constraint_fallback.py, tests/test_constraint_fallback.py. Produces timed_fallback_search(index, encoder, constrained, original, chinese, mode='auto_en', fallback='error', top_k=5) and the spec's returned fields.

- [ ] Write failing tests for rejection→OPUS/Chinese timing, primary success, no cascade, stale trace/input/retrieval/system errors and invalid policies.
- [ ] Run focused pytest RED; implement minimal engine with search_texts/sync/perf_counter and post-timing diagnostics.
- [ ] Run focused pytest GREEN and lint; integrate engine/tests after independent review.

### Task 2: Benchmark and CLI

Files: fallback_benchmark.py/tests; cli.py/tests. Consumes Task1 API, produces9-policy attempts/samples and frozen comparison.

- [ ] Tests RED for preflight data/source/index binding, policies, repeats, comparison, denominator and CLI validation.
- [ ] Implement runner and CLI; GREEN plus full offline suite/Ruff.
- [ ] Freeze copied metadata, code and protocol with5repeats/3warmups; commit before scoring.
- [ ] Run Base thenSo400m in separate processes, preserving logs/results; compare validated results only.

### Task 3: Results, audit and documentation

Files: benchmarks/constraint-fallback-v1 metadata/results/audit; docs/constraint-fallback-v1.md; README,AGENTS,docs/{README,issue-log,improvements,continuation}.

- [ ] Independently recompute all metrics/rates/percentiles and branch equality from saved raw evidence.
- [ ] Report16-intent metrics, branch/failure costs, p50/p95, loading and semantic limits.
- [ ] Fresh code/result review; regression RED/GREEN for findings; suite/Ruff, real CLI and package checks.
- [ ] Commit/push private repo, verify exact-commit CI, update ledger/handoff.

## Execution decisions

Proceed under the user's explicit execution request without an extra approval gate for these reversible repository changes. Persist decisions and evidence as work proceeds. Planned tasks remain unchecked until verified.
