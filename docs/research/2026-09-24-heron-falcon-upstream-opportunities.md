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

### 5. Downstream-quality routing and owned correction

Three routing papers sharpen the target for the next phase:

| Stage | Source and inspected implementation | Transfer to this pipeline |
|---|---|---|
| Pre-routing | [VL-RouterBench](https://arxiv.org/abs/2512.23562), commit `f5a41ce8`, `tools/build_matrices.py`, `routers/linear/train_and_eval.py`, and `routers/utils/fusion.py` | Build the paired quality and cost matrix first, keep the oracle as a ceiling, then train the smallest source-grouped visual router. The current dual Ridge experiment follows this structure but lacks enough independent groups for activation. |
| Post-routing | [ViLU](https://arxiv.org/abs/2507.07620), commit `e4bf5723`, `vilu/models/vilu_attn.py`, `vilu/engine/trainer.py`, and `vilu/losses/bce.py` | Predict actual OCR failure from visual evidence and emitted text together. Raw decoder confidence is only one feature. Train against corrected and uncorrected outcomes, then calibrate on separate source groups. |
| Crop selection | [CropVLM](https://arxiv.org/abs/2511.19820), commit `9c9cb913`, `grpo.py` and `trainer.py` | Optimize the crop for downstream reader improvement. The paper reports that box IoU can be a poor proxy for final task quality, so this pipeline should label crops by the change in unique page reconstruction instead of overlap alone. |

Implementation status:

- The VL-RouterBench transfer is implemented in two experimental paths. The Falcon plain-versus-layout router trains from a paired model-win matrix and reports its oracle ceiling. The new Heron region-role router uses source-grouped Heron class scores to decide whether a detected table also needs a form/control reading. Both loaders reject development artifacts, so neither router can be enabled without separate calibration.
- The ViLU transfer has an offline public-data prototype, not a live clinical integration. A source-separated visual classifier trained on same-call Falcon post states retained Falcon for 1,319 of 4,509 development line crops, reducing Qwen calls by 29.3%. Its routed CER was 33.80% versus 34.79% for always-Qwen, but the paired source bootstrap interval for the difference crossed zero. A later visual-plus-transcript model found no threshold that safely saved even one calibration call. The production path therefore still uses deterministic, fail-closed correction validation and does not load either experimental post-router.
- The CropVLM transfer is not implemented as a learned crop policy. The new region-role router selects existing Heron regions and measures downstream benchmark change. It does not learn box coordinates or optimize a crop reward.

The production candidate does not add the repositories or their model stacks. It reuses Falcon and Heron evidence and trains compact heads only after paired outcomes exist.

### Heron region-role experiment

The implemented classifier uses all 17 Heron class scores and a regularized logistic regression. It never replaces a table. A positive decision appends a form-source companion with the same physical owner, and malformed features or an uncalibrated artifact leave the current output unchanged.

On 71 private development observations from source-separated pages, full Heron class scores ranked form/control-bearing regions better than the smaller score summary and visual-plus-output alternatives. The fixed held-out source groups still did not support a usable calibration threshold, and model-assisted labels are development evidence rather than adjudicated truth. The runtime therefore remains disabled.

Two selectively routed pages were read by Qwen for $0.00222593. On the selected eight-form development cohort, their owned controls improve case-macro state accuracy from 66.74% to 68.24% and reduce missing controls from 286 to 284. Historical GPT-5.4-mini remains higher at 71.77%. The initial mixed adaptation reduced the three-table score from 45.25% to 27.55% because it replaced the stronger saved table predictions. That violates the append-only design, so the accepted development composition keeps the existing five-table result unchanged at 42.22% and claims no table gain.

## Structured form ownership result

The current selected eight-form run improves substantially over the older Falcon adapter, while remaining below the historical GPT output:

| Saved output | Case-macro state accuracy | Missing controls |
|---|---:|---:|
| Structured selective Qwen, saved evaluation | 66.74% | 286 |
| Historical full-page Qwen | 67.78% | 238 |
| Historical GPT-5.4-mini | 71.77% | 192 |
| Older Falcon adapter | 38.05% | 475 |

The current adapter replay is 66.53% with 285 missing controls because two ambiguous duplicate-label assignments differ from the saved evaluation. The eight pages were used during development, so neither number is an untouched estimate. Historical SmartOCR has no submitted predictions for these eight cases; its zero is missing baseline coverage and is not a quality comparison.

Manual review found a shared production bug: a valid Qwen correction to an existing checkbox label converted the region into plain text and discarded the physical control state. The repair preserves checkbox kind, state, structure, provenance, and Markdown rendering. It fixes canonical output integrity but does not change the saved aggregate because a second form extraction already recovered that label.

An offline conflict-policy ablation compared keeping Falcon state, promoting every spatially aligned Qwen state, promoting only same-label states, and marking conflicts unresolved. Promotion improved case macro by only 0.12 percentage points on these eight development forms. It also included one spatial collision between different labels. That evidence is too weak to replace the conservative policy. A learned post-router needs more source-grouped conflict outcomes.

## Rejected defaults

- Always-full-page Falcon: 33/146 with three failures, below the 35/146 layout route.
- Both Falcon modes on every page: useful only for oracle measurement because it doubles decoding work.
- Lowering the global Heron threshold: may increase recall and duplicates without repairing form/table confusion.
- Full Docling integration: adds excluded components and reproduces the same form, table, and reading-order failure classes.
- Falcon decoder fine-tuning: official training code is unavailable.

## Acceptance rule

Adopt a change only when a source-grouped, failure-inclusive comparison shows a positive paired gain over the current route, no observed critical-field regression, no reduction in form or table score, no increase in duplicate physical readings, and lower Qwen use than always-Qwen. Report distance to the 48/146 oracle. An inconclusive result remains inconclusive.

## Tool record

Jina Reader was used to inspect the official papers, issue, and pull-request pages above. The canonical routing repositories were inspected at the pinned commits listed in the transfer table. Search stopped at depth 2 because the next sources repeated the same three decisions: paired quality-cost labels, visual-plus-text failure prediction, and downstream-quality crop rewards. Jev was attempted for a bounded public-evidence choice, but its MCP process did not have `OPENROUTER_API_KEY`, so it returned no decision and did not influence the ranking.
