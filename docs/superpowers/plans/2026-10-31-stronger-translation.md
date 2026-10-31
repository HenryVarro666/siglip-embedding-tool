# Stronger pretrained translation comparison

Compare pinned NLLB-200 1.3B with OPUS-MT for Base224 and So400m384 on the existing 40 images / 8 Chinese intents. Do not train, edit generated English, relabel data or tune weights on this pilot.

## Protocol

- NLLB source: `facebook/nllb-200-1.3B`, revision `b0de46b488af0cf31749cd8da5ed3171e11b2309`, source `zho_Hans` / target `eng_Latn`. Record its research scope and CC-BY-NC-4.0 license.
- Use the same greedy beam1 / max_new_tokens64 / source cap256 / float32 / MPS settings as OPUS. Sanitize generation-config fallbacks, force the required NLLB English BOS and record it. Every automatic request translates afresh; no translation text cache.
- Four independent sequential processes, each with zh / auto_en / auto_fusion. Three warmups, 10 repeats × 8 intents; measure stages and actual wall total. List model startup and first request separately.
- Run fresh OPUS controls under the same implementation and hardware. Preserve previous published reports and their historical timings.
- Freeze data, model artifacts, configuration, code and index hashes before scoring. Compare Base/So400m with the same translator, and OPUS/NLLB with the same retrieval checkpoint and common protocol.
- Term review covers inverter/microinverter in i01–i04 and power-transformer bushing / visible cooling fins in i07. Distinguish clear, partial and incorrect across six opportunities. Broader radiator / heat-sink wording does not establish visible fins. Review context separately, without retrieval rankings. This is assistant review, not an independent human translation benchmark.

## Work

1. Implement and test the NLLB adapter, preparation, language IDs, provenance and generation guards. Download and convert fixed source weights without updating parameters.
2. Implement and test translator selection, benchmark comparisons and CLI options, preserving OPUS defaults and old protocols. Complete the offline suite, lint and independent preflight review.
3. Commit and freeze. Run all four cases sequentially with no other model jobs.
4. Audit terms and context, all 12 policy rows / 8 intents, and stage / end-to-end percentiles. Do not infer that improved terminology causes retrieval gains or that a larger translator must be more accurate.
5. Save summaries, reports, reproduction instructions, Top5 review and improvements. Independently review actual results, run the full suite / build / clean-clone checks, push privately and verify CI. Keep raw images, weights, indexes and raw runs local.

## Progress

Steps 1–3 implemented at measurement commit `f023fa2`; the pre-run suite passed 259 offline tests. Code and protocol hashes were frozen before NLLB translations or scores existed. The final report records the actual outcomes and verification.
