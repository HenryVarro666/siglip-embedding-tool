# Completed Candidate Protection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Track each task below.

**Goal:** Protect genuine completed candidates from repetition rejection and measure quality/failure/full cost without training.

**Architecture:** Request-local pinned-scorer observer plus additive guarded translator; separate development, four-policy benchmark and independent audit. Old measured cores remain unchanged.

**Tech Stack:** Existing Python3.12/PyTorch/Transformers/MPS, pytest/Ruff, pinned decoder code.

**Spec:** docs/superpowers/specs/2026-11-01-completed-protection-design.md

## Global Constraints

No training; fixed old profile and 24 declared dev sources; unchanged16/40 formal inputs; four fresh fusion controls,3 warmups/5 rounds. Original artifacts/identities/defaults preserved. Explicit pre-finalize EOS witnesses, no candidate substitution or silent observer fallback.

## Review Focus

Finalizer EOS and evicted candidates cannot protect; every requirement needs token and word-boundary checks; duplicate witnesses stay honest; request isolation/no-term behavior and infrastructure errors; saved-rule/dev/trace/timing/metric drift fail export.

### Task1: Request-local observer and protected translator

Owned files: src/siglip_tools/completed_protection.py, tests/test_completed_protection.py; rule completed_protection_v1.json.

- [ ] RED→GREEN for actual EOS/retained pool vs finalizer/active/evicted candidates, full lexical checks and protected continuation.
- [ ] Implement `CompletedProtectedTranslator(base, *, profile, decoder_dir=...)` and module CLI with fresh OPUS fallback. Preserve old measured files. `last_trace.repetition_stop` keeps raw snapshots plus protection/actual-abort decisions and immutable EOS witnesses.
- [ ] Characterize pinned decoder boundary with tiny inference fixtures; state reset, config/source immutability, failure/CLI cases; scoped tests/lint.

### Task2: Extended development and freeze

Owned files: src/siglip_tools/protected_development.py, tests/test_protected_development.py, benchmarks/completed-protection-v1/dev-*.json.

- [ ] RED→GREEN fixed24 input binding, rule and semantic validation, complete observed eligibility and primary-acceptance comparisons.
- [ ] Commit source/rules before new model calls; run four fresh MT controls on24 dev cases. Preserve all controls/errors/evidence; no retrieval or formal queries.
- [ ] Validate fixed protection and coverage, save decisions/identities/times and commit before formal outputs.

### Task3: Four-policy measured benchmark

Owned files: src/siglip_tools/protected_benchmark.py, tests/test_protected_benchmark.py, protocol/freeze and metadata.

- [ ] RED→GREEN preflight/source/weight/index/dev binding, rates/denominators, guard/protection witnesses, branch controls and saved-evidence tamper guards.
- [ ] Reuse pure legacy evaluator/timer without changing measured cores; save all128 scored and640 warm records/digests, actual complete times and protected/missed/accepted subsets.
- [ ] Review and full suite/lint, commit frozen source/protocol, then Base→So400m sequential inference.

### Task4: Independent audit, report and continuation

Owned files: examples/audit_completed_protection.py, examples/summarize_completed_protection.py and tests; curated JSON; docs/README, issue-log, improvements, continuation, rootREADME/AGENTS.

- [ ] Independently replay recorded EOS/lexical/protection, metrics/rates/full timing/digests, branch/cross-model evidence. Preserve negatives and all tail samples.
- [ ] Real CLI, wheel/package import, original loss demo, full suite and clean clone; collect log hashes only after all writers finish. Fresh review/fixes as needed.
- [ ] Save report/results/limits, update SIG IDs/index/improvements/continuation; private push/main fast-forward and exact-head CI receipts.

## Execution ledger

Baseline source7386423; graph queried empty (SIG-021). In-place feature branch preserves shared local artifacts, authorized by current task. Decoder mechanics read-only subagent completed; implementation/review delegation follows scoped file ownership. All decisions/failed attempts are also persisted in docs/issue-log.md.
