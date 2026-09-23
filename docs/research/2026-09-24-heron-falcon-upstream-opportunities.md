# Heron and Falcon improvement decisions

Accessed 2026-09-24.

## Decision

Keep the production candidate lean:

`300 DPI page -> Heron-101 proposals and features -> learned Falcon plain/layout route -> Falcon-OCR 1.5 -> owned selective Qwen correction -> canonical assembly`

Do not add Docling, TableFormer, TATR, Nemotron, TrOCR, or Granite-Docling to the default graph. Use Docling issues as evidence about Heron failure modes, then repair the responsible Heron or Falcon boundary.

The current matched clinical development result is:

| Falcon action | Exact snippets |
|---|---:|
| Layout route | 35/146 |
| Native full page | 33/146, with 3 failed pages |
| Per-page oracle | 48/146 |
| Historical GPT-5.4-mini | 48/146 |

The oracle shows complementary Falcon modes. It does not establish learned-router performance. The six source documents are development data and cannot support a final generalization claim.

The implemented grouped Ridge experiment confirms that limitation:

| Policy | Exact snippets | Recorded Qwen cost | Falcon actions |
|---|---:|---:|---:|
| Always layout | 35/146 | $0.00342114 | 32 layout |
| Always plain | 33/146 | $0 | 32 plain |
| Development-selected router | 43/146 | $0.00198318 | 21 layout, 11 plain |
| Nested source-group estimate | 30/146 | $0.00050459 | 3 layout, 29 plain |

The 43/146 result selects policy parameters on the same development outcomes and is optimistic. The honest nested estimate regresses. The service therefore rejects the development artifact and accepts only a separately calibrated router. Learned routing remains disabled by default.

## Source-backed changes worth testing

### 1. Learned Falcon mode routing

Falcon-OCR 1.5 exposes native plain and layout decoding. The release recommends plain OCR for handwriting, scans, receipts, and most pages, with layout OCR for dense layouts. The local oracle makes a learned page-level decision the smallest measured next step. The route must use only pre-decode Heron and image features and execute exactly one Falcon decoder path.

Source: [Falcon-OCR 1.5 release commit](https://github.com/tiiuae/Falcon-Perception/commit/777f85c284313086dab8743e0c1eec2640f0f055).

### 2. Heron-specific overlap preservation in Falcon

Falcon currently contains overlap suppression designed around PP-DocLayoutV3. Falcon [PR 32](https://github.com/tiiuae/Falcon-Perception/pull/32) reports that applying it to Heron can suppress legitimate cross-category regions and randomly choose same-category duplicates. The maintainer requested model-specific layout processors rather than replacing the default behavior.

Test a Heron policy that preserves cross-category regions and picture boxes while selecting same-category duplicates by score, area, and stable source order. Keep the existing policy for other detectors. Enable the Heron policy only after paired local results show better coverage without more duplicate physical readings.

### 3. Reversible image-padding ablation

Docling [issue 2965](https://github.com/docling-project/docling/issues/2965) reports that tables near page edges can be missed before downstream structure processing, and that white image padding can recover them. The report also documents regressions and latency from rewriting PDFs. The valid local experiment is image-only padding before Heron with an exact inverse coordinate transform, not PDF rewriting. This remains an ablation until paired form and table scores improve.

### 4. Public Heron hard-negative fine-tuning

Current Heron failure reports include grid-lined forms classified as tables, tables classified as forms, short isolated text omissions, checkbox grouping, and wrapper overlap. [Issue 3219](https://github.com/docling-project/docling/issues/3219) also reports a hardcoded table-over-key-value conflict in Docling postprocessing. That Docling rule is not part of this pipeline, but the underlying detector confusion is relevant.

Heron is a 17-class RT-DETRv2-R101 detector with selected-checkbox, unselected-checkbox, form, and key-value classes. Its strongest reported training mixture includes non-public data, and official training scripts are not available. A publishable fine-tune therefore needs public or generated form/table, checkbox, small-text, rotation, and degraded-scan hard negatives. Private clinical pages remain evaluation-only.

Sources: [Heron report](https://arxiv.org/abs/2509.11720), [Heron-101 model card](https://huggingface.co/docling-project/docling-layout-heron-101), [fine-tuning issue 3016](https://github.com/docling-project/docling/issues/3016).

## Rejected defaults

- Always-full-page Falcon: 33/146 with three failures, below the 35/146 layout route.
- Both Falcon modes on every page: useful only for oracle measurement because it doubles decoding work.
- Lowering the global Heron threshold: may increase recall and duplicates without repairing form/table confusion.
- Full Docling integration: adds excluded components and reproduces the same form, table, and reading-order failure classes.
- Falcon decoder fine-tuning: official training code is unavailable.

## Acceptance rule

Adopt a change only when a source-grouped, failure-inclusive comparison shows a positive paired gain over the current route, no observed critical-field regression, no reduction in form or table score, no increase in duplicate physical readings, and lower Qwen use than always-Qwen. Report distance to the 48/146 oracle. An inconclusive result remains inconclusive.

## Tool record

Jina Reader was used to inspect the official issue and pull-request pages above. Jev was attempted for a bounded public-evidence choice, but its MCP process did not have `OPENROUTER_API_KEY`, so it returned no decision and did not influence the ranking.
