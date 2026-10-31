# Stronger pretrained translation comparison

Compare fixed NLLB-2001.3B with OPUS-MT for Base224 andSo400m384 on the existing40images/8Chineseintents. Do not train, editgeneratedEnglish, relabeldata or tune weights onthispilot.

## Protocol

- NLLBsource `facebook/nllb-200-1.3B` revision`b0de46b488af0cf31749cd8da5ed3171e11b2309`, source`zho_Hans` / target`eng_Latn`. Research-modelCC-BY-NC4license is recorded; no production endorsement.
- Same greedybeam1/maxnew64/sourcecap256/f32/MPS policies asOPUS. Sanitize generationconfigfallbacks, force requiredNLLBEnglishBOS andrecord it. No translationtextcache: eachauto request translatesafresh.
- Four independent sequential processes (translator×retrievalmodel), eachhas zh/auto_en/auto_fusion. Threewarmups,10repeats×8intents; directstageandwall-total timing,modelstartup/firstrequest separate.
- Fresh OPUScontrols under the sameimplementation/hardware, rather than mixing previous timings with newNLLB measurements. Preservepreviouspublishedreports.
- Freeze data, model/artifact/config/code/index hashes before scoring. Compare Base/So with same translator andOPUS/NLLBwith same retrievalcheckpoint/commonprotocol.
- Termreview: inverter/microinverter in i01–i04; power-transformerbushing andvisiblecoolingfins in i07. Clear/partial/wrong distinctions are assistant review on6opportunities, not an independenthuman translationbenchmark. Radiator/heat-sink are broaderthan explicitfin visibility. Save primary source terminology references and unchangedEnglish.

## Work

1. TDD NLLBadapter/preparation+languageID/provenance/generationguards; downloadandconvertfixedsource locally withoutupdatingweights.
2. TDD translatorregistry, benchmarkselection/comparison andCLIselection, preservingOPUSdefaults/oldprotocols. Fulltests/lint andindependentpreflightreview.
3. Commitandfreeze; runall4cases sequentially. Trackprogress/latency andavoidothermodeljobs.
4. Audit termsagainstsourcecontext andfullrankings, all12policyrows/8intents, stage+E2Epercentiles. No causalclaimfromterminologyalone andno conclusionthatlargerismoreaccurate withoutresults.
5. Save curatedsummary, reports/reproduction/Top5view, improvements; independentresultsreview,fullsuite/build/cleanclone/privatepush/CI. Rawimages/weights/resultsstaylocal.
