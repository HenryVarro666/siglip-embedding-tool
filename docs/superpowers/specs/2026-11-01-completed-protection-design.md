# Completed-candidate protection experiment

User requests a pretrained-only protection against erroneous repetition rejection, extended long-normal/legitimate-repetition development controls, and a fixed-query Base/So400m comparison with complete timing and durable issue/continuation records.

## Frozen design before new inference

- Preserve old measured cores, old guard/profile/CLI/decoder/weights and all historical results. New namespace `completed-protection-v1`; feature branch `feat/completed-candidate-protection` in the existing checkout reuses ignored artifacts. No training or downloads/upgrades.
- Use the already selected `bounded-n16-r2-min32-q4-tail4` repetition rule unchanged. The new protection has no numerical candidate grid: a currently retained genuine EOS-completed candidate that passes the original full lexical and nonempty checks suppresses rejection. Continue the original search and finalizer unchanged; do not return/rescue/reorder candidates or accept active fragments.
- Genuine EOS originates only from the pinned scorer.process branch, not finalize. Prefix tensors in the live pool exclude EOS. Record actual EOS proposals/witnesses and current pool membership; candidate prefix/EOS step, normalized score and full lexical checks must be reconstructable. Assert one prompt token, batch1/beam4. Reject observer-contract failure as infrastructure error; do not silently call it normal translation rejection.
- Isolate observer per request using FunctionType clones of validated outer/loop globals and a request-local scorer subclass. No original module/class/cache/global mutation, no pinned source edits. No-term paths preserve ordinary beam4. Release live tensors/instance state after each request.
- Freeze 24 development sources before current inference: old i01–i08, eight long normal l01–l08, eight legitimate repetitive r01–r08. Four methods run afresh, no reference English or retrieval used. Preserve all source lengths, raw outputs, errors, beam checks, EOS/protection evidence, generation/complete times and identities. These are assistant-written controls; classify baseline acceptance and actual >=32 eligibility from observations, never assume all intended long texts qualify.
- Validate fixed protection on development, freeze code/rule/observations/validation before new formal runs. If new controls miss relevant behavior, report coverage gap rather than tune on formal t09. t09 is a known engineering regression, not unseen validation; formal16/gallery40 are unchanged historical inputs and assistant labels.
- Formal four fusion controls: original OPUS; legacy constraint→fresh original OPUS; existing repetition early-stop→fresh OPUS; protected early-stop→fresh OPUS. Each 3 warmups, full16x40 scored ranks, five rounds per intent; fresh translations each request, no cache. Base then So400m processes, same weights/indexes and 0.5/0.5 fusion. 128 scored and640 warm requests total, failed intents remain zero in denominator16.
- Complete synchronized wall time includes source handling, observer bookkeeping/beam copies, detection, failed attempt, validation/exception/cleanup, fresh fallback, fusion and retrieval; serialization outside. Loading/first calls separate. Keep every tail sample. Report primary rejection, actual early rejection, protected pattern detection, fallback trigger/conditional success and final failure separately.
- Compare protected vs old guard/legacy for saved accepted outputs, remaining missed failures, same-subset complete costs, aggregate mAP/Hit@1/Recall@5/MRR and p50/p95/mean. Changes to rankings/English and statistical limits must be explicit. EOS/lexical acceptance does not establish wiring/negation/mounting semantics.
- Before export/comparison validate frozen code, source data, current rules, development binding, artifact identity, metrics/rates/timings and warm digests. Independent recorded-data audit replays protection decisions and checks both model outputs. Never hash a shared log glob while any writer remains active (SIG-050).

## Review focus

Finalizer-only EOS, evicted completion, lexical-substring-only candidate, prompt/special-token confusion, duplicate EOS/tied proposals, observer failure, repeated requests and no-term state leakage. A protection decision is not a promise the finalizer selects that candidate. No model/config/identity mutation or formal-ID routing.

## Decisions

The user explicitly requests execution, real experiments and persistence. Proceed with a new in-place feature branch; reuse local weights, caches and indexes, avoiding extra setup/approval. Existing private-push authorization persists. No default-method change is planned; any measured limitation remains in the report/ledger.
