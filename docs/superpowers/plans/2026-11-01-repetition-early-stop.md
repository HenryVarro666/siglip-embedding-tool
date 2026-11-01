# Repetition Early Stop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce failed constrained-generation cost with dev-frozen repetition rejection and fresh OPUS fallback, without training.

**Architecture:** Independent guard wrapper and development observer; a new3-policy runner reuses the existing timed fallback/index/evaluator. Old measured cores remain unchanged. Separate export/audit and versioned metadata retain old evidence.

**Tech Stack:** Python3.12/PyTorch/MPS/Transformers, pinned local constrained callable, pytest/Ruff.

**Spec:** docs/superpowers/specs/2026-11-01-repetition-early-stop-design.md

## Global Constraints

- No training, old source cores/CLI/data/protocol untouched; same16formalqueries/40gallery.
- Dev8i01–i08 only for rules;16predeclared candidates, deterministic selection, profile frozen before formalrun.
- Real in-loop abort, no falseEOS acceptance or use of rejected fragments, fresh OPUS every fallback.
- Threefusionpolicies,3warmups/5rounds/16intents, actual complete timing and no failure censoring.

## Review Focus

- A repeating minority beam must not cause a more stringent quorum rule to abort.
- Normal repeated nouns/lists and a legacy acceptable candidate may still be early-rejected; retain/report rather than hide it.
- Callback exception must preserve current partial evidence and original config, and must not swallow infrastructure errors.
- No-constraint path and successive requests must not reuse a stale monitor or mutate base generation settings.
- Coherent protocol/profile/dev/support changes or saved-statistic drift must fail comparison/export before attribution to frozenexperiment.

### Task1: Guarded translator and module CLI

Files: src/siglip_tools/repetition_stop.py, tests/test_repetition_stop.py.
Produces GuardedOPUSConstraintTranslator(base, *, profile, observe_only=False, decoder_dir=...), candidate_profiles(), detect_repetition(beams, profile, special_ids) and replayable last_trace.repetition_stop.

- [ ] TDD: suffix cycles/quorum/specials/checkinterval/boundaries, state resets, real callback abort before64, partial evidence rejected by fallback, no-term unchanged, base config/identity unchanged and errorspropagate.
- [ ] Implement new subclass/callback without altering pinned decoder/old code. GREEN and scopedlint.
- [ ] ModuleCLI search loads/reuses pinned index/model and selected packagedprofile with defaultOPUSfallback; invalid inputs preallocation, JSONstates/timings; test/real smoke later.

### Task2: Development freeze and rule selection

Files: src/siglip_tools/early_stop_development.py/tests; benchmarks/repetition-early-stop-v1/dev-{queries,protocol,results,selection}.json; selected packagedprofile.

- [ ] TDD deterministic candidate ranking and fallback-to-conservative, no evalquery inputs; freeze8queries/16grid/selection before observer calls.
- [ ] Observe real devbeam histories; replay candidates; select frozenprofile; rerun8withactive guard/fallback to validate actual stop and preserve acceptedcases.
- [ ] Save full development outputs, caps/errors/detector times/identities; commit profile before formalresults.

### Task3: Threepolicy benchmark and validations

Files: src/siglip_tools/early_stop_benchmark.py/tests; separate protocol/freeze.

- [ ] TDD preflight/hash/index/profile/dev binding; denominators, schema/evidence, no stringcache, branch control and changed oldaccepted accounting.
- [ ] Implement fresh shared MT/guard + timed fallback; save all96scored attempts/480warm samples, complete wall clock and generatedtoken/cause summaries.
- [ ] Review code; fullsuite/lint; freeze actualsource/weights/profile/data; commit before Base→So runs.

### Task4: Results and continuation

Files: newversioned report/curatedresult/export/independent audit; docs/{README,issue-log,improvements,continuation},README,AGENTS.

- [ ] Run bothmodels, no tuning after formaloutputs. Independently recompute metrics/rates/quantiles/digests/branch/control and savedtoken evidence; preserve any failed attempts.
- [ ] Explain full/conditional costs and overhead, falseearly-rejection vs mechanical success, semantic limits, and rules/devscope honestly.
- [ ] Verify realmoduleCLI, package, fullsuite and clone; commit/push private repo, verify percommitCI and update handoff.

## Execution decision

The user explicitly requested implementation and real experiments; proceed within this reversible authorizedscope without another approval gate. Builder/code review may identify clarifications resolved from repo and thisspec. Never tune on formal16 outputs. Record deviations and failed attempts immediately in issue-log.
