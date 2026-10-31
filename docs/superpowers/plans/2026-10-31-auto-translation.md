# Automatic query translation plan

User request: compare direct Chinese, automatic English, and Chinese+automatic-English retrieval for both frozen SigLIP2 configurations; include translation in end-to-end timing and do not train.

## Fixed design

- Reuse equipment-v1's40images/8intents and own matching Base/So400m indexes. Preserve all previous data/results.
- Use one compact local pretrained OPUS-MT Chinese→English checkpoint, immutable revision in protocol.json. Convert official PyTorch weights to local safetensors during preparation; record original and converted hashes. No weight update.
- Greedy generation, no sampling, max64newtokens; source input<=256tokens. Reject truncation/empty output. Keep translations verbatim and deterministic; do not supply the previous curated English or edit output.
- Three policies: zh; auto_en; auto_fusion with0.5/0.5mean cosine. Each timed auto request performs fresh translation; no translation text cache. Model-internal generation KV cache is per request.
- Local models resident on MPS/float32. Stage translation/retrieval/actual wall-total are timed with synchronization; startup and first automatic request separately recorded. Rotate policies,3warmups,10repeats×8intents=80samples per policy/model.
- Compare translator identity/output across both runs, all original image/query/qrels hashes, repeated scores/ranks/translations. Preserve raw outputs locally and curated report/provenance in Git.

## Implementation and verification

1. TDD translation wrapper and preparation; optional dependency extra. Independent worker owns translation.py and its tests, parent owns other modules.
2. TDD auto-retrieval API and CLI mutually exclusive with supplied-English query. Explicit stage timing and no silent fallback.
3. TDD frozen auto_benchmark runner and compatibility comparison. Use existing metrics/stability helpers; metadata/code/index/translation provenance frozen before target scoring.
4. Sequential real Base and So400m runs. Inspect all8unedited translations, all6ranking sets and all480requests; compare against last round's supplied-English controls with clear denominators.
5. Independent review, full offline suite/lint/build/clean clone; docs, Top5 review, private GitHub push and CI. No training.
