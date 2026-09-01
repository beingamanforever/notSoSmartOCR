# Scalable, Accurate, Cost-Aware OCR Pipeline

Research and execution plan, 2026-09-01

## 1. Goal and evidence standard

Build a research pipeline that turns PDFs and images into an evidence-linked structured document while preserving literal text, geometry, reading order, tables, controls, formulas, and cross-page relationships. The system must be modular enough to exchange the local parser, hosted escalator, route policy, and output adapter without changing the core document contract.

The target is accuracy under explicit cost and latency constraints. Cost is optimized only after quality and safety requirements are met. Failed pages, invalid responses, abstentions, and unavailable cases remain visible and remain in evaluation denominators.

This work is research-only. Licenses and model provenance are recorded separately from technical quality. Research permission does not imply production, privacy, PHI, or commercial approval.

The deployable pipeline now has a strict lineage constraint: it must not use a Chinese-origin model, backbone, or teacher. PaddleOCR-VL, GLM-OCR, Qwen, DeepSeek, MinerU, olmOCR, and Qwen-derived systems remain useful public-data research comparators, but they are not eligible production components. This lineage rule supersedes the earlier candidate roles retained below as historical experiment context.

Evidence labels used in this report:

- **Reported**: a result published by a model or benchmark author.
- **Verified source fact**: confirmed in an official paper, repository, model card, dataset card, or API documentation.
- **Measured here**: produced by this workspace using a recorded caller-boundary command.
- **Unverified**: requires a local experiment, live provider call, or legal review.

Local measurements are reported below. No hosted-model ranking, hosted page-cost claim, or superiority claim is treated as measured until the pinned live experiments run.

### 1.1 Implemented research system

The repository now contains:

- ordered PDF, multi-frame TIFF, and image ingestion with explicit render, reader, empty-output, and invalid-input failures;
- lossless EXIF orientation normalization into a temporary PNG without modifying the source image;
- swappable Tesseract, NVIDIA Nemotron OCR v2, IBM Granite Docling, full PaddleOCR-VL-1.6, full GLM-OCR SDK, and direct GLM-OCR model-only readers;
- one typed evidence contract for page size, regions, geometry, reading order, literal text, confidence when available, provider, route, and failures;
- deterministic detection of empty content, invalid geometry, invalid reading order, malformed tables, and repeated multi-token hallucination loops;
- selective crop or full-page repair that protects region ID, geometry, kind, and local geometry-provider provenance;
- independent visual verification: a proposed patch is merged only when a second image model, which never sees the first candidate, independently returns the same literal text and the deterministic risk set shrinks;
- a strict OpenRouter adapter with separate production and public-benchmark model roles, exact response-model validation, provider pinning, structured output, no-data-collection and ZDR requests, bounded retries, truncation rejection, actual response metadata, and environment-only credentials;
- failure-inclusive public transcription, direct hosted, stitched cascade, and OmniDocBench export harnesses.

Nemotron OCR v2 is the current eligible literal-OCR candidate, pending its frozen quality and operational evaluation. IBM Heron is integrated only as a research layout comparator: its RT-DETRv2 lineage excludes it from the strict deployable stack even though the released weights and code are Apache-2.0. Granite Docling is a small structured-parser ablation, not literal truth. Paddle and GLM remain public-data research comparators under the lineage rule. The direct GLM reader is explicitly a model-only ablation.

"Evidence-Patch Cascade" is a project synthesis label, not a novelty claim. SAFE-Cascade supplies a selective-routing prior, GRC and Consensus Entropy supply stability and disagreement priors, and the local contribution being tested is the combination of typed OCR evidence, deterministic failure detection, independently generated visual candidates, protected patches, and explicit abstention.

### 1.2 Measurements completed here

| Caller-boundary run | Cases | Coverage | Micro CER | Micro WER | p50 | p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Tesseract, ClinOCR v1.0 evaluation | 328 | 91.2% | 0.488293 | 0.611352 | 1.844 s | 5.990 s |
| Tesseract, FUNSD original test | 50 | 100% | 0.565042 | 0.790239 | 1.663 s | 2.976 s |
| Nemotron OCR v2 plus rectification and direct OSD, ClinOCR rotated evaluation | 56 | 100% | 0.103958 | 0.122140 | 2.209 s | 10.443 s |
| Nemotron OCR v2 selective OSD fallback, full ClinOCR evaluation | 328 | 86.0% | 0.331423 | 0.417869 | 1.862 s | 2.355 s |
| PaddleOCR-VL-1.6, one page from each ClinOCR subset | 6 | 100% | 0.1986 | 0.2313 | 11.20 s | 79.19 s |
| Direct GLM-OCR model-only, same six development pages | 6 | 100% | 0.1093 | 0.1178 | 9.36 s | 18.34 s |
| PaddleOCR-VL-1.6 base, full ClinOCR rotated subset | 56 | 100% | 0.3137 | 0.3780 | 12.82 s | 41.29 s |
| PaddleOCR-VL-1.6 with document unwarping, full ClinOCR rotated subset | 56 | 100% | 0.0607 | 0.0810 | 13.19 s | 20.47 s |
| Direct GLM-OCR model-only, full ClinOCR rotated subset | 56 | 100% | 0.0790 | 0.0837 | 10.79 s | 15.63 s |

The full ClinOCR Tesseract run attempted all 328 evaluation pages, covered 299, and retained 29 failures. Its normalized edit distance was 0.486593, missed-text rate 0.331191, and hallucinated-text rate 0.027070. The FUNSD run attempted and covered all 50 test pages; its normalized edit distance was 0.563301, missed-text rate 0.236543, and hallucinated-text rate 0.086457.

The eligible Nemotron arm attempted all 56 frozen rotated pages with no failures. Its rectification plus direct Tesseract OSD policy was fixed on the separate eight-page exemplar set. On the matched evaluation panel, Nemotron was worse than direct GLM by 0.024938 CER, cluster bootstrap interval [0.005010, 0.048507], and 0.038408 WER, interval [0.017509, 0.063548]. It was worse than Paddle plus unwarping by 0.043221 CER, interval [0.025719, 0.059437], and 0.041130 WER, interval [0.031233, 0.051845]. Those Chinese-origin arms are research comparators rather than deployable candidates, but the accuracy gap remains real and must not be hidden by the lineage rule.

Manual review of three high-error Nemotron pages found distinct failure mechanisms. One oblique page lost leading characters and changed a visible phone suffix from `6700` to `6709`, consistent with edge damage after rectification. One inverted form returned the impossible date `06106/99999` for a clearly printed `06/15/1999`. A rotated pathology report preserved much of the body but missed or reordered demographic fields and sections. These cases justify protected critical-field validation, crop-boundary checks, and alternate-view abstention. They do not justify semantic correction.

On all 328 ClinOCR evaluation pages, strict OSD covered 277 pages with CER 0.346732 and WER 0.431763. The frozen selective fallback covered 282 pages and reached CER 0.331423 and WER 0.417869. Cluster bootstrap intervals for the selective deltas excluded zero, but only five cases changed and the Holm-adjusted two-sided sign-flip p-value was 0.256974 for both metrics. The policy therefore remains an informative ablation, not a promoted default.

The same frozen selective candidate was paired against the failure-inclusive Tesseract floor on all 328 evaluation pages and 16 template clusters. Nemotron reduced micro CER by 0.156870, template-cluster 95 percent interval [-0.185387, -0.113176], and micro WER by 0.193483, interval [-0.223458, -0.151314]. Its Holm-adjusted sign-flip values were 0.0026 for CER and 0.0004 for WER. Coverage was lower, 86.0 percent versus 91.2 percent, while both arms retained failed cases as empty predictions. This establishes a transcription improvement over the CPU floor, not superiority to an eligible hosted frontier model.

Completed structural evidence:

| Capability | Frozen public panel | Measured result |
| --- | ---: | --- |
| Forms | FUNSD 50 test pages, 2,332 entities, 837 key-value links | Official oracle and empty controls validate field exact match, normalized value accuracy, and key-value relation F1 at 1.0 and 0.0. |
| Layout | OmniDocBench v1.5 balanced 61 pages, 1,136 evaluated boxes | Research-only Heron covered 61 of 61 pages: COCO mAP 0.342071, AP50 0.433447, and project micro P/R/F1 0.640424 / 0.744718 / 0.688645. Official oracle and empty controls score 1.0 and 0.0. |
| Tables | PubTables-1M 60 test tables | TATR v1.1-Pub produced 60 valid results: GriTS Top 0.990225, Con 0.991262, Loc 0.984719, cell exact F1 0.977380, p50 96.354 ms, p95 996.692 ms, and 573.365 MiB peak allocated GPU memory. |
| Controls | PulseBench-Select 60 public development cases | Empty control retains every case and scores state macro-F1 and label association F1 at 0.0. |
| Orientation | ClinOCR 56 rotated evaluation pages | The gold-free docTR and OSD two-view guard covered 56/56, selected a minimum-CER cached view on 56/56, reached case-mean CER 0.100211 and WER 0.121439, and had zero measured regret. Aggregate guarded latency is unavailable. |
| Multi-page | MPDocBench 420 documents and 3,135 pages | Official control ceiling: HeadTEDS 0.995730, text relation F1 0.992288, table relation F1 0.971698, merged-table TEDS 1.0, and exact continuation accuracy 1.0. |

The non-perfect MPDoc oracle scores are the measured ceiling of the official heading and relation detectors. The empty HeadTEDS floor is 0.153148 rather than zero, another evaluator property that must remain visible. These controls validate the metric paths but do not establish model quality. TATR and Heron are real component-model results, but Heron remains research-only because of its RT-DETRv2 lineage. Hosted paired baselines and the official full OmniDocBench end-to-end model score remain incomplete, so no frontier-superiority claim is made.

Private specialist evidence remains separate from the public table. The geometric control stage scored state macro-F1 1.0 and label association F1 0.9903 on a twice-reviewed 52-control clear panel, but dense ruled grids generated false proposals, so the stage is review-only. On the same 46 reviewed handwritten crops, PyLaia achieved 0 strict exact fields, 94.58 percent CER, and 117.39 percent WER; TrOCR base achieved 1 strict exact field, 82.51 percent CER, and 136.23 percent WER; Nemotron Parse achieved 5 punctuation-insensitive exact fields, 117.49 percent CER, and 108.33 percent WER with one failure. All three routes were rejected.

The expanded hard panel completed 44 of 44 requests and routed all 44 pages to review. It contains the frozen 41 generated clinical pages plus one handwriting-heavy form and two application-layout failures. Manual inspection judged 0 of 44 pages complete enough for unattended use. This corrects the earlier 40 false accepts and rejects both newly observed page-sized false tables, but it does not solve the underlying handwriting, table-association, checkbox-grid, faint-text, or multi-pane reading-order errors. Operational `success` means schema-valid output, not correctness.

The Heron run used the released `8f39ad3c0b4c58e9c2d2c84a38465abf757272d8` revision, raw threshold 0.6, the same frozen 61-page panel as both controls, and the A10G. It produced 1,321 mapped detections with no failed or abstained page, p50 52.844 ms, and p95 142.725 ms. Nineteen detections were outside the official mapping: one unselected checkbox, 11 forms, and seven key-value regions. Manual review of the largest-error pages found a blank ruled form fragmented into many text boxes, a stitched two-page mathematics spread with title and text misses, and a dense newspaper where table captions were not represented. These are measured failure directions for routing and class mapping, not evidence that Heron is deployable.

The six-page rows are development traces spanning only two independent ClinOCR templates, not ranking evidence. Direct GLM is model-only and lacks the complete SDK's layout stage. The full rotated rows are matched on all 56 cases and eight template clusters. Unwarping reduced Paddle micro CER by 0.2530 with a template-cluster bootstrap 95 percent interval of [-0.3060, -0.2001], and micro WER by 0.2970 with an interval of [-0.3690, -0.2211]. The Holm-adjusted two-sided sign-flip p-value was 0.0156 for both metrics. It won 52 of 56 pages on CER and 50 of 56 on WER.

Direct GLM versus Paddle base reduced micro CER by 0.2347 with interval [-0.2977, -0.1754] and micro WER by 0.2943 with interval [-0.3669, -0.2184]; both Holm-adjusted p-values were 0.0156. Against Paddle plus unwarping, however, direct GLM's CER delta was +0.0183 with interval [-0.0028, 0.0357], and WER delta was +0.0027 with interval [-0.0220, 0.0224]. Those intervals cross zero and do not establish the locked 0.5-point non-inferiority margin. A point sample during the direct GLM run used 4,884 MiB, while a point sample during Paddle unwarping used 22,206 MiB of the A10G's 23,028 MiB. These are not peaks. The GLM ablation is operationally promising for literal transcription, but it cannot replace structured layout evaluation.

The saved Paddle GPU JSON files predate the `run_config` serializer. Their matched commands and option differences are recorded here, but those two artifacts are not independently self-describing and must be regenerated before external publication. New public, direct hosted, cascade, and OmniDocBench artifacts serialize all behavior-affecting reader and run options without secrets.

Recorded caller arguments for the matched Paddle pair:

```bash
PYTHONPATH=src python experiments/public_benchmark.py \
  clinocr CLINOCR_ROOT paddle-vl-1.6-rotated-full-base.json \
  --reader paddleocr-vl --workers 1 --subset rotated \
  --backend native --device gpu:0

PYTHONPATH=src python experiments/public_benchmark.py \
  clinocr CLINOCR_ROOT paddle-vl-1.6-rotated-full-unwarp.json \
  --reader paddleocr-vl --workers 1 --subset rotated \
  --backend native --device gpu:0 --use-doc-unwarping
```

Both used the same ClinOCR v1.0 bytes, PaddleOCR-VL-1.6 adapter, A10G, failure denominator, and normalization. The paired report was recomputed from raw predictions with 10,000 template-cluster resamples and seed 0.

On the matched six-page trace, enabling Paddle's document unwarping improved aggregate micro CER from 0.1986 to 0.1687, but harmed the handwriting page from 0.0890 to 0.2824 and the normal page from 0.0853 to 0.1805. It greatly improved the rotated page from 0.4635 to 0.1007. The full rotated result confirms a large effect only for that fixed degradation. The option loads orientation classification plus UVDoc, so it is a combined preprocessing arm. It must be routed from observable image or stability signals, never from the dataset subset name. Global enablement is not supported by this evidence.

Paddle region confidence in the current adapter comes from the matched layout detector box, not recognition-token confidence. It must not be described or routed as OCR confidence.

No live OpenRouter request was made with the credential pasted into chat. That credential is treated as compromised. Direct Muse, Gemma, Luna, Sol, Qwen, DeepSeek, and stitched superiority remain unmeasured until a fresh environment key and exact provider pins are supplied.

## 2. Research conclusion

The smallest eligible stack worth testing is a routed specialist pipeline:

1. Preserve valid native PDF text and the original rendered page as separate evidence.
2. Run [NVIDIA Nemotron OCR v2](https://huggingface.co/nvidia/nemotron-ocr-v2) multilingual for literal text, boxes, confidence, and relational reading order.
3. Keep layout detection behind the existing swappable interface. Use raw [IBM Heron](https://huggingface.co/docling-project/docling-layout-heron) only to measure the value of a separate layout pass and to source architecture ideas, then select a lineage-eligible detector on the same frozen panel before deployment.
4. Detect observable risk from missing text, low confidence, geometry conflict, unsupported insertions, table or control structure, view instability, and failed reading-order invariants.
5. Route only affected evidence: Microsoft Table Transformer for table grids, IBM CodeFormulaV2 for formulas, and the measured geometric control stage as review evidence. Keep handwriting unresolved. Generic PyLaia, TrOCR, and Nemotron Parse 2.0 all failed the fixed clinical-field panel and are excluded from routing.
6. Generate at most one task-specific transformed view after a risk signal. Compare it with the raw page and roll it back unless heterogeneous evidence and deterministic structure improve.
7. Use Meta Muse Glimmer 30B and Google Gemma 4 31B only on selected public-data crops for independent repair or verification. Merge only protected, literal agreement. Otherwise retain the original evidence and abstain.
8. Compare direct Mistral OCR 4.1, GPT-5.6 Luna, and the actual frontier GPT-5.6 Sol on identical frozen pages. Keep Qwen and DeepSeek as public benchmark-only external comparators.

Why this is the current candidate stack:

- Nemotron OCR v2 is a 54M or 84M detector-recognizer-relational pipeline, explicitly supports A10G, exposes text confidence and geometry, and is designed for batching. NVIDIA's reported OmniDocBench crop results skip empty predictions, so this project must rescore failure-inclusively.
- Heron is a 42.9M Apache-2.0 layout detector with text, tables, formulas, forms, selected and unselected checkboxes, and key-value regions. Its RT-DETRv2 lineage makes it research-only under this project's stricter origin rule.
- Mistral OCR 4.1 is a strong eligible hosted specialist at a list price of USD 4 per 1,000 pages, but it is proprietary and must be measured through its separate API.
- Muse and Gemma have independent non-Chinese lineages and can verify selected visual evidence. Same-model self-verification and multiple views of one model are correlated evidence, not independent support.
- Direct 30B inference on every page weakens cost, attribution, and reliability. The 30B model earns a production role only if routed calls improve a frozen ambiguity slice after call rate, failures, latency, and cost are counted.
- Recent visual-grounding studies show that VLM OCR can replace visible character sequences with fluent, unsupported text. Script masking and generic contrastive decoding are not reliably corrective. Cross-paradigm disagreement, explicit unsupported-insertion metrics, and abstention are required.
- Tesseract, PaddleOCR-VL, and GLM-OCR remain valuable floors and diagnostic comparators. Their prior measurements are preserved, but they do not define the eligible final architecture.

## 3. Reusable ideas and code boundaries

### 3.1 Systems worth reusing or comparing

| System | Reuse in this project | License or boundary | Current evidence status |
| --- | --- | --- | --- |
| [NVIDIA Nemotron OCR v2](https://huggingface.co/nvidia/nemotron-ocr-v2) | Eligible literal OCR, boxes, confidence, and reading order | NVIDIA Open Model License; A10G supported | Frozen failure-inclusive ClinOCR results now exist for 56 rotated pages and all 328 evaluation pages. It remains behind the best research comparators on the matched rotated panel. |
| [IBM Heron](https://huggingface.co/docling-project/docling-layout-heron) | Research-only raw layout detector and component-router ablation | Apache-2.0 release, RT-DETRv2 lineage | 42.9M and 17 classes. Raw and postprocessed outputs require separate ablations; it cannot enter the strict deployable stack. |
| [Mistral OCR 4.1](https://docs.mistral.ai/models/ocr-4-1) | Eligible hosted specialist baseline | Proprietary service, USD 4 per 1,000 pages list price | Paragraph boxes, labels, confidence, and annotations are verified API capabilities, not local accuracy results. |
| [NVIDIA Nemotron Parse 2.0](https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-2.0) | Isolated structured-parser research adapter | OpenMDW-1.1; custom runtime; legal review | Rejected as a handwriting route: 5/46 punctuation-insensitive exact, 117.49% CER, 108.33% WER, and one failure. |
| [PaddleOCR-VL-1.6](https://github.com/PaddlePaddle/PaddleOCR) | Public benchmark comparator and prior preprocessing evidence | Chinese lineage, benchmark-only | Author reports 96.33 on OmniDocBench v1.6. Local rotated-page results are recorded above. |
| [PP-DocLayoutV3](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3) | Public research comparator for regions, polygons, classes, and reading order | Chinese lineage, benchmark-only | Reuse the component boundary as an idea, not the model. |
| [GLM-OCR](https://github.com/zai-org/GLM-OCR) | Public benchmark recognizer comparator | Chinese lineage, benchmark-only | Author reports 94.62 on OmniDocBench v1.5. Different benchmark revisions are not directly comparable. |
| [PP-StructureV3](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/pipeline_usage/PP-StructureV3.en.md) | Public research ablations for OCR, tables, formulas, charts, cells, and reconstruction | Chinese lineage, benchmark-only | Reuse task decomposition ideas only. |
| [Docling](https://github.com/docling-project/docling) | Ideas for a lossless document representation, native PDF preservation, adapters, and serializers | MIT | Reuse the contract ideas before adding the dependency. |
| [MinerU](https://github.com/opendatalab/MinerU) | Cross-page tables, truncated paragraph continuation, hybrid parsing comparison | Chinese lineage and custom Apache-derived terms | Public research comparison only. |
| [olmOCR](https://github.com/allenai/olmocr) | Unit-test benchmark ideas only under the strict rule | Qwen2.5-VL lineage, benchmark-only | Its Western organization does not change its ineligible backbone lineage. |
| [Surya](https://github.com/datalab-to/surya) | Compact research comparison for OCR, layout, reading order, and tables | Code and weight terms differ; weights have use limits | Research comparison only until the exact terms are accepted. |
| [NaviDC-OCR](https://github.com/caipeng328/NaviDC-OCR) | Research ideas for geometry-aware decoding, camera documents, render checks, and content-structure separation | Repository license unresolved in this review | Do not copy source until a clear license exists. |
| [OvisOCR2](https://huggingface.co/ATH-MaaS/OvisOCR2) | Compact end-to-end Markdown comparator | Chinese lineage, benchmark-only | Author results only; deployment and parser reliability remain unverified. |
| [Consensus Entropy](https://github.com/Aslan-yulong/consensus-entropy) | Training-free string disagreement, risk-coverage evaluation, and diverse-model verification | Confirm repository terms before copying code | Strong methodological prior for independent disagreement routing. Agreement is not correctness. |
| [GRC](https://github.com/phare111/GRC) | Multi-view generation, stability scoring, accept-or-abstain behavior, and length bounds | MIT | Reuse the measured-stability idea, not gold or cohort routing. |
| [SAFE-Cascade](https://www.alphaxiv.org/abs/2606.19646) | Quality-first cascade design and call-rate reporting | Paper method | ChartQA routing evidence does not establish OCR accuracy. |
| [Uncertainty-aware OCR](https://github.com/NikoGuan/Uncertainty_OCR) | Explicit uncertain-span labels, Blur-OCR degradation suite, and uncertainty-tag evaluation | Verify code and data terms before reuse | Useful annotation and evaluation representation before considering its RL recipe. |
| [Infinity-Parser2](https://github.com/infly-ai/INF-MLLM) | Synthetic data taxonomy, verifiable rewards, and multi-task document parsing comparison | Official repository terms require exact component review | Author-reported data and training methodology, not a local result. |
| [FastOCR](https://www.alphaxiv.org/abs/2605.17447) | Visual-token importance and attention-latency research | Paper method | Token pruning is approximate and excluded from the default lossless path. |
| [Tesseract](https://github.com/tesseract-ocr/tesseract) | CPU floor, local development, deterministic failure tests | Apache-2.0 | Installed locally. It does not solve semantic layout or table reconstruction. |

The design borrows stable ideas, not repository-specific orchestration. It does not copy code from projects with unresolved or restrictive terms.

### 3.2 Thin interfaces

Only interfaces with real alternative implementations are introduced:

```text
LocalReader.read(page) -> page regions and local evidence
Escalator.extract(request) -> validated extraction, usage, latency, status
RoutePolicy.choose(context) -> accept local, escalate regions, escalate page, review
```

Composition stays explicit. There is no plugin discovery, registry, factory hierarchy, or class per stage. A provider swap changes one adapter. A benchmark swap changes one experiment adapter. The core does not import benchmark-specific code.

### 3.3 Canonical document contract

Each document result must contain:

- document and page identifiers in original order;
- page dimensions and source type;
- regions with type, polygon or bounding box, reading order, literal text, confidence when available, and provider provenance;
- structured fields with `present`, `absent`, `unreadable`, `conflicting`, or `not_applicable` status;
- evidence links from every field or cell to its source page and regions;
- tables with cells, row and column spans, labels, and source geometry;
- controls with state, label association, and source geometry;
- route decisions and observable reasons;
- latency, token usage, provider cost, and explicit failures;
- warnings for unsupported, conflicting, truncated, or cross-page content.

Unreadable values are never completed from domain plausibility. A provider timeout, invalid JSON, empty response, budget limit, or validation failure is never converted into silent success.

## 4. Implemented spine and planned structure

```text
PDF or images
  -> ordered pages
  -> valid native text plus EXIF-normalized lossless page image
  -> docTR orientation proposal plus OSD disagreement guard
  -> Nemotron OCR v2 literal text, boxes, confidence, and reading order
  -> swappable layout regions, with Heron only in the research comparator arm
  -> deterministic evidence merge and validators
  -> table, formula, checkbox, handwriting, or relation specialist on risky regions
  -> optional Muse and Gemma independent crop agreement
  -> protected patch or explicit abstention
  -> structured JSON plus Markdown or task adapter

Planned after measured need:
  -> Mistral OCR 4.1 hosted specialist arm
  -> license-cleared clinical handwriting challenger
  -> Arctic-TILT existing-ID multipage relation validation
  -> padded context only for measured association failures
```

Routing uses observable signals only:

- missing text inside a detected content region;
- low or inconsistent recognition confidence, only where the reader exposes it;
- invalid table geometry, spans, or row alignment;
- ambiguous checkbox state or label association;
- conflicting repeated identifiers or values;
- failed layout, reading order, or cross-page continuation;
- invalid provider schema or parser output;
- remaining document budget.

The current implementation uses exact region crops and creates an explicit full-page target only when a reader cleanly returns zero regions. Reader exceptions remain failures instead of becoming speculative repair. Padding, nearby-label context, general full-page escalation, and cross-page reconstruction remain controlled ablations rather than claimed behavior. Same-model or same-weight endpoint agreement is not independent evidence. The current merge requires two different visual models, strict schemas, a normal non-truncated finish reason, literal agreement, protected geometry, recorded text provenance, and deterministic risk reduction. It still needs held-out false-accept measurement before being called evidence-preserving.

### 4.1 Selective preprocessing and hallucination control

The original page is always the primary evidence. Preprocessing produces an alternate view, never a replacement source. The first experiment is the missing Paddle public-comparator 2 by 2 ablation on development-only rotated cases: neither orientation nor unwarping, orientation only, unwarping only, and both. This separates the geometry effect that the current combined arm cannot attribute. It does not make Paddle production-eligible.

The eligible-core view ladder is evaluated one transform at a time:

1. Raw page or exact crop.
2. White crop padding of `max(8 px, 2 percent of the short side)`.
3. Deterministic 2x Lanczos enlargement for low-resolution crops.
4. Grayscale plus autocontrast for visibly degraded text.
5. One fixed Sauvola configuration for a separate degraded-text ablation.
6. At most two mild shifted or zoomed views as stability probes.

No transform is global by default. Binarization is not applied to handwriting, faint controls, or colored evidence unless its own held-out result supports it. Learned restoration such as DocRes is isolated until the deterministic ladder fails. Super-resolution, diffusion restoration, semantic correction, and any process that can invent glyph strokes are excluded from the default path.

Selection must be gold-free. A transformed output is accepted only when an observable raw-page risk triggered the view, the original and transformed provenance remain separate, deterministic geometry or structure improves, and an architecturally different eligible reader supports the literal result. Same-model agreement across transformed views is only a risk signal. If support is absent, the pipeline retains raw evidence, escalates, or abstains.

The primary hallucination controls are literal prompts, exact structured schemas, returned-model identity checks, protected critical fields, heterogeneous verification, unsupported-insertion and missed-text metrics, and explicit `partial`, `illegible`, `conflicting`, or `abstained` states. Recent studies show that a low CER can coexist with plausible semantic rewriting, and explicit requests not to guess do not reliably prevent it. Domain plausibility is never permitted to overwrite visible characters.

A calibration-free research diagnostic on the frozen 56-page rotated panel used Nemotron as the primary reader and Paddle plus GLM only as public-data disagreement comparators. Consensus disagreement reached severe-error AUROC 0.966346 for cases above 0.2 CER, versus 0.451923 for the implemented literal date heuristics. At 80.36 percent retained coverage, the consensus ordering contained no severe case and had mean case CER 0.073950. This supports heterogeneous disagreement as a routing direction, not a deployable policy: the alternative readers are lineage-ineligible, correlated agreement can still be wrong, and reference text was used only to evaluate the ordering.

## 5. Hosted model integration

Hosted roles are separated by lineage, privacy, and claim type. Qwen and DeepSeek may be benchmarked only on public pages. The deployable repair pair is Muse plus Gemma, and a genuine frontier claim must include GPT-5.6 Sol because OpenAI positions Luna as its cost-sensitive high-volume tier. Prices and ZDR endpoints are a 2026-09-01 snapshot and must be recorded again at execution.

| Model | Verified role | Current list price | Experiment decision |
| --- | --- | --- | --- |
| [`meta/muse-glimmer-30b`](https://openrouter.ai/meta/muse-glimmer-30b) | Multimodal text and image input with strict schema support | About USD 0.30 input and USD 1.20 output per million tokens on the selected BF16 endpoint | Eligible primary crop repair candidate. Pin a ZDR BF16 provider and disable fallback. |
| [`google/gemma-4-31b-it`](https://openrouter.ai/google/gemma-4-31b-it) | 30.7B dense multimodal model with schema support | About USD 0.14 input and USD 0.40 output per million tokens on the selected BF16 endpoint | Eligible independent visual verifier and direct 30B baseline. |
| [`openai/gpt-5.6-luna`](https://openrouter.ai/openai/gpt-5.6-luna-20260709) | Image and PDF input, structured output, cost-sensitive tier | USD 0.20 input and USD 1.20 output per million tokens | Cost-tier direct baseline, not the frontier baseline. |
| `openai/gpt-5.6-sol` | OpenAI flagship comparator | Recheck immediately before the run | Required for a broad frontier superiority claim. |
| [`qwen/qwen3.8-flash`](https://openrouter.ai/qwen/qwen3.8-flash) | Multimodal model with schema support | USD 0.15 input and USD 0.47 output per million tokens | Chinese public-data comparator only. Its only current provider does not advertise ZDR, so never send private pages. |
| [`deepseek/deepseek-v4-flash-vision-exp`](https://openrouter.ai/deepseek/deepseek-v4-flash-vision-exp) | Experimental image model without strict JSON-schema enforcement on current endpoints | USD 0.44 input and USD 1.32 output per million tokens | Chinese public-data comparator only. |

The request builder now enforces explicit production and public-benchmark roles. Production crop repair accepts only Muse Glimmer and Gemma. Qwen, DeepSeek, Luna, and Sol require an explicit public-benchmark flag. The returned model must exactly match the requested model, so an unexpected route cannot silently satisfy a production call. Mistral OCR 4.1 still needs a separate thin adapter because it uses Mistral's OCR API rather than OpenRouter chat completions.

The implemented adapter:

- read `OPENROUTER_API_KEY` only from the environment;
- send image data through the official multimodal chat-completions format;
- require a strict JSON Schema response;
- request providers that support required parameters;
- request no data collection and zero-data-retention routing when available;
- use bounded retries and a capped `Retry-After` delay;
- reject missing, length-limited, or otherwise non-normal finish reasons;
- require the actual returned model to match the requested model and record provider when supplied, usage, cost, attempts, latency, and normalized and native finish reasons without estimating missing provider costs;
- never persist authorization headers, the key, or private input in logs;
- stay disabled for private clinical images unless PHI, region, contract, and retention approval exists.

Relevant official documentation: [structured outputs](https://openrouter.ai/docs/features/structured-outputs), [provider routing](https://openrouter.ai/docs/guides/routing/provider-selection), [zero data retention](https://openrouter.ai/docs/guides/features/zdr), [multimodal images](https://openrouter.ai/docs/features/multimodal/images), and [usage accounting](https://openrouter.ai/docs/cookbook/administration/usage-accounting).

The key pasted into chat is treated as compromised. It will not be copied into source, commands, reports, or configuration. A rotated key must be exported in the execution environment before a live hosted benchmark.

## 6. Benchmark suite

No single aggregate score will mix transcription, layout, reading order, tables, forms, controls, and multi-page parsing. Each capability keeps its native denominator and metric family.

### 6.1 End-to-end and clinical benchmarks

| Benchmark | Verified public scale | Role | Important limitation |
| --- | --- | --- | --- |
| [ClinOCR-Bench](https://huggingface.co/datasets/ClinOCR-Bench/ClinOCR-Bench) and [paper](https://www.alphaxiv.org/abs/2607.03650) | 384 synthetic clinical documents across 6 subsets and 16 templates; 56 train exemplars and 328 scored test documents | First public clinical transcription benchmark | Synthetic and template-reused; does not prove real clinical safety. |
| [OmniDocBench](https://github.com/opendatalab/OmniDocBench) and [paper](https://www.alphaxiv.org/abs/2412.07626) | Acquired v1.6 has exactly 1,651 images and 1,651 annotation records | General text, layout, formulas, tables, and reading order | CC BY-NC and revision-dependent scoring. The evaluator must be pinned to v1.6 before reporting results. |
| [MPDocBench-Parse](https://github.com/Tongyi-Zhiwen/Qwen-Doc/tree/main/MPDocBench) and [paper](https://www.alphaxiv.org/abs/2605.22100) | Public release: 420 PDFs and 3,135 pages; paper: 433 documents and 3,246 pages before removals | Multi-page hierarchy, continuation, reading order, figures, tables, and formulas | Official annotation-only controls now cover all released documents and pages. Research terms, no train split, and formula dependencies still limit use. |
| [olmOCR-Bench](https://github.com/allenai/olmocr) | More than 7,000 unit tests over about 1,400 documents in the project description | Broad parser regression and failure taxonomy | Primarily English and unit-test oriented; benchmark artifact terms need confirmation before redistribution. |

### 6.2 Component benchmarks

| Benchmark | Verified scale | Capability and metrics | Acquisition decision |
| --- | --- | --- | --- |
| [DocLayNet](https://github.com/DS4SD/DocLayNet) | 80,863 pages, with 69,375 train, 6,489 validation, and 4,999 test pages; 11 layout classes | Layout precision, recall, F1, and COCO mAP | Use test or a fixed test slice later. It is large and has no end-to-end transcription truth. |
| [FUNSD](https://guillaumejaume.github.io/FUNSD/) | Acquired 199 noisy forms, split 149 train and 50 test | Word transcription plus completed field, normalized-value, and key-value relation scoring on all 50 test pages | Small and useful, but non-commercial research and education terms apply. |
| [CheckboxQA](https://github.com/Snowflake-Labs/CheckboxQA) | Acquired 88 documents, 2,048 retained PDF pages, and 579 QA pairs | Official ANLS* for downstream checkable-content understanding | Evaluation-only CC BY-NC plus underlying DocumentCloud terms. It is not direct checkbox-detection ground truth. |
| [IAM Handwriting](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database) | 1,539 pages from 657 writers, 13,353 lines, and 115,320 words | CER and WER for handwriting | Registered official access is required. Do not use an unofficial mirror. |
| [PubTables-1M](https://github.com/microsoft/table-transformer) | 575,305 page images and 947,642 structure tables | GriTS topology, content, location, and project cell exact match | A pinned 60-table public test panel now has oracle, empty, and real TATR v1.1-Pub results. The full dataset remains about 117 GB. |
| [ICDAR 2019 cTDaR](https://zenodo.org/records/3239032) | Separate modern and historical tracks with overlapping detection and structure subsets | Table detection and adjacency-based structure F1 across IoU thresholds | Good archival and handwritten holdout. Confirm archive terms before use. |
| [CORD v2](https://github.com/clovaai/cord) | 1,000 public receipts, split 800, 100, and 100 | Receipt OCR and hierarchical extraction | Useful extra document-domain stress test under CC BY 4.0. |
| [SROIE](https://arxiv.org/abs/2103.10213) | 1,000 receipts, 600 train or validation and 400 test | Localization, OCR, and four-field extraction F1 | Verify current RRC access and use terms before acquisition. |

### 6.3 Existing internal benchmark

The current `internal-clinical-ocr-benchmark` contains 432 cases, of which 414 are ready and 18 are unavailable: 162 forms, 30 handwriting cases, and 222 tables. One handwriting target is quarantined for independent human adjudication while its source pixels remain preserved. New systems write adapter predictions that satisfy the current format. Post-quarantine baselines were regenerated, and explicitly named pre-quarantine directories preserve the prior artifacts.

The private suite is local-only until external processing receives explicit PHI approval. Missing and invalid submissions remain in the current denominator.

## 7. Metrics and reporting

### 7.1 Quality

- Transcription: CER, WER, normalized edit distance, substitutions, insertions, deletions, missed-text rate, hallucinated-text rate.
- Layout: class-wise precision, recall, F1, and mAP.
- Reading order: sequence edit distance and pairwise or edge F1.
- Tables: TEDS, GriTS topology, content, and location, cell exact match, row and column F1, merged-cell accuracy.
- Forms: field exact match, normalized value accuracy, entity F1, and key-value relation F1.
- Controls: state macro-F1 and control-to-label association F1.
- Multi-page: HeadTEDS, cross-page relation F1, continuation accuracy, figure association.
- Safety: evidence-supported fact accuracy, unsupported-fact rate, unresolved rate, conflict detection, and abstention quality.

### 7.2 Operations

- Page and document p50 and p95 end-to-end latency.
- Queue, preprocessing, layout, recognition, escalation, and merge time.
- Throughput, batch size, crop count, peak GPU memory, and failures by stage.
- Hosted input and output tokens, actual returned cost, cost per page, cost per document, and cost per correct supported field.
- Escalated-region and escalated-page rates.
- Provider invalid response, timeout, rate-limit, and abstention rates.

Report paired case differences and 95 percent bootstrap intervals where the metric supports pairing. The paired harness recomputes CER and WER from raw predictions and references rather than trusting stored edit counts. ClinOCR uses template clusters as the primary bootstrap and sign-flip unit; page resampling is labeled exploratory. Never compare author scores across different benchmark revisions as if they were a controlled experiment.

## 8. Experiments and decision rules

### 8.1 First controlled comparison

Run identical pages through:

1. Tesseract CPU floor.
2. Nemotron OCR v2 multilingual literal OCR.
3. Nemotron OCR v2 plus raw IBM Heron layout regions as a research-only marginal-value ablation.
4. Granite Docling 258M as a compact structured-parser ablation.
5. The earlier Paddle and GLM arms as public-data historical comparators only.
6. Direct Mistral OCR 4.1, Muse Glimmer 30B, Gemma 4 31B, GPT-5.6 Luna, and GPT-5.6 Sol on identical public pages.
7. Direct Qwen3.8-Flash and DeepSeek vision as Chinese-lineage external comparators on public pages only.
8. Nemotron plus routed table, control, formula, and structure specialists.
9. The routed system plus Muse crop repair independently verified by Gemma, and the reverse role assignment.
10. A hosted-budget-matched cascade arm when several crop calls can exceed one direct page call.
11. Gold-route and gold-candidate oracles for diagnosis only, never as deployable results.

Run BF16 before testing one quantized configuration. A quantized model is accepted only if its paired quality loss is within the locked task bounds and its operational gain is real.

Required ablations:

- full routed pipeline versus each direct VLM and hosted OCR specialist;
- full page versus layout crops;
- crops versus crops with neighboring label context;
- Nemotron alone versus the research-only Nemotron plus Heron ablation, and core plus one eligible specialist at a time;
- native PDF text preservation on versus off;
- Muse and Gemma escalation off versus independently verified repair;
- heuristic-only, disagreement-only, and combined routing;
- tight crop, padded crop with nearby labels, and full-page context;
- BF16 versus one selected quantization;
- page-level versus document-level reconstruction.

Direct and stitched arms must share source bytes, prompt intent, strict schema, output limit, provider and quantization pin, failure denominator, and scoring normalization. Record the actual returned model and provider because a model slug alone does not establish the serving implementation.

The 30B plus pipeline USP is established only if the routed core plus selective 30B repair:

- beats or is non-inferior to direct Muse on the same held-out cases;
- beats or is non-inferior to direct Gemma and Luna;
- beats GPT-5.6 Sol before making a broad frontier-superiority claim;
- reports Qwen3.8-Flash and DeepSeek as public external comparisons without using them inside the deployable system; and
- reaches those results through fewer hosted pages or a better measured quality-cost point without increasing unsupported digits, identifiers, entities, controls, or table cells.

Until those four conditions hold under paired confidence intervals, the architecture is a testable synthesis, not a frontier-model superiority claim.

### 8.2 Routing experiment

Reuse identical cached stage outputs to compare:

1. local-only;
2. each eligible hosted model on every page;
3. heuristic-only selective repair;
4. independent-disagreement selective repair;
5. combined routing;
6. gold-oracle routing for research diagnosis only.

Calibrate routing only on the 56 ClinOCR exemplars and a rights-cleared document-grouped development set. Lock the policy before the 328 ClinOCR evaluation pages and the exact OmniDocBench release. Report gate precision, recall, AUPRC, risk-coverage, recoverable-error recall, escalation rate, false-accept rate, abstention rate, and cost completeness.

Stop routing work if oracle escalation improves the primary metric by less than 1 absolute point and recovers fewer than 10 percent of local errors.

Accept automatic routing only if it:

- captures at least 90 percent of oracle-recoverable gain;
- costs at most 40 percent of the cheapest all-page hosted policy that meets the quality floor;
- does not increase unsupported critical facts;
- regresses neither control-state F1 nor table-position accuracy by more than 0.5 points.

For the broad architecture claim, require non-inferiority to the better direct hosted arm on transcription with a one-sided paired 95 percent lower bound above -0.5 absolute points. Require at least a 2-point table-TEDS or reading-order improvement with the paired lower bound above zero. The local transcription-only harness cannot establish either structural condition; the version-pinned OmniDocBench evaluator is required.

Do not optimize a blended quality-cost score. Discard every policy that violates a quality, safety, privacy, or reliability constraint, then choose the cheapest remaining policy.

### 8.3 Core and specialist selection

Adopt Nemotron OCR v2 as the literal core only if, on frozen public and internal holdouts:

- failure-inclusive transcription is non-inferior to the best eligible baseline within 0.5 percentage points;
- at least 99 percent of eligible pages receive a valid result or an explicit abstention;
- critical digits, dates, negation, and identifiers do not regress;
- normalized geometry, confidence, and reading order remain valid at the caller boundary;
- measured p95 latency, throughput, and A10G memory meet the selected operating target.

Use Heron only to quantify recoverable layout errors. Add a deployable layout detector or other specialist only after its own lineage review and only if oracle stage analysis finds recoverable errors and its routed held-out run improves the task-native metric without hurting literal transcription or critical subsets. Add Muse or Gemma repair only if independent verification improves evidence-supported critical-field recovery by at least 2 points at no more than the locked escalation rate and satisfies provider privacy requirements.

These thresholds are research defaults. Product owners must lock critical-field precision, maximum cost, latency, and privacy constraints before held-out selection.

## 9. Phasewise execution

| Phase | Current status |
| --- | --- |
| 0. Contract and spine | Completed and covered through the real CLI boundary. |
| 1. Public transcription floor | Completed for full ClinOCR evaluation and FUNSD test with failure-inclusive Tesseract results. |
| 2. GPU baseline | In progress. Nemotron OCR v2 has frozen 56-page and 328-page ClinOCR evidence on the A10G. The repaired Granite Docling adapter produced one valid page but needs the full frozen panel. Earlier Paddle and GLM runs remain public comparator evidence only. |
| 3. Hosted adapters | Production and public roles, exact returned-model validation, privacy controls, and injected end-to-end tests are complete. Live public calls are blocked on a rotated key and provider pins. |
| 4. Routing and structural evaluation | Partial. Public controls validate FUNSD forms, Omni layout, PubTables tables, Pulse controls, MPDoc hierarchy and continuation, and the 18-page Omni end-to-end path. TATR has a real 60-table score, docTR has a guarded 56-page orientation result, and the expanded 44-page private hard panel is fully review-routed after manual inspection. Hosted comparisons and a manually acceptable full-stack result remain. |
| 5. Internal bridge | Implemented against the post-quarantine 414-case ready set. Ambiguous form identities and unavailable truth abstain while staying in failure-inclusive accounting. External calls remain disabled without a dataset-bound provider approval record. |
| 6. Audit | In progress. Independent review found benchmark provenance and failed-case accounting defects; completion requires their repair, a clean full verification run, and a final independent review. |

### Phase 0: Pipeline contract and end-to-end spine

Deliver ordered image and PDF ingestion, canonical evidence-linked JSON, an explicit local reader, an explicit escalator interface, routing states, and a CLI. Test the real CLI over a multi-page fixture. Page order, geometry, abstentions, and failures must survive serialization.

Exit condition: achieved. The complete local fixture path passes from user input to serialized result with page order, geometry, evidence, failures, and nonzero partial status preserved.

### Phase 1: Local floor and public transcription

Add the Tesseract floor adapter and public evaluation harness. Download ClinOCR-Bench and FUNSD from official sources. Attempt every eligible case and retain failures in denominators.

Exit condition: achieved for the transcription floor. All 328 ClinOCR evaluation cases and 50 FUNSD test cases were attempted, with failures retained.

### Phase 2: Eligible GPU core

On the authorized A10G, install Nemotron OCR v2 in an isolated Python 3.12, Torch 2.8, CUDA 12.8 environment on the NVMe volume. Keep caches and results on NVMe. Sync source to `/home/ubuntu/aman/notSoSmartOCR` without deleting existing files. Run the multilingual BF16 release first and record valid boxes, confidence, reading order, peak VRAM, throughput, stage latency, and failure-inclusive quality. Run Granite Docling separately as a compact parser ablation. Do not mix the older Paddle or GLM comparator results into the deployable lineage.

Exit condition: not yet achieved. Nemotron compiled for SM 8.6, its CUDA extension imported successfully, and frozen failure-inclusive ClinOCR scores now exist. A native batch sweep measured output changes, so batching is not accepted as lossless. Granite Docling produced one valid repaired page after correcting a Docling Core API mismatch, but that single page is only a caller-boundary check. The Granite frozen panel, full-stack peak-memory sampling, and a lineage-eligible layout replacement for the measured Heron comparator remain.

### Phase 3: Hosted comparison and repair

Implement strict structured output, production and benchmark model roles, exact response-model validation, provider constraints, retries, usage accounting, and route reasons. Use a rotated environment key. Test only public images first.

Exit condition: blocked on a fresh key and explicit provider pins. The adapter behavior is tested with injected caller-boundary responses, but no provider validity, accuracy, latency, or cost percentage is measured.

### Phase 4: Routing and multi-capability evaluation

Calibrate deterministic route signals on development data. Run local-only, every eligible all-page hosted arm, oracle, and automatic policies over the same outputs. Run Qwen and DeepSeek only as external public comparators. Add CheckboxQA and selected OmniDocBench, DocLayNet, PubTables-1M, and cTDaR slices only after source terms and evaluators are confirmed.

Exit condition: not yet achieved. Official failure-inclusive controls validate the forms, layout, table, control, and multi-page metric paths. TATR has a real 60-table result, research-only Heron has a real 61-page layout result, and the guarded orientation selector has a real 56-page result. The expanded private 44-page run prevents silent acceptance, but manual review found no complete page. A rejected, truncated, same-model, or disagreeing patch leaves local evidence unchanged and records a distinct abstention. Provider exceptions remain explicit failures. Real hosted comparisons and a manually acceptable full-stack result still remain.

### Phase 5: Existing benchmark bridge

Write a separate adapter into a new run directory and evaluate all 414 ready internal cases with the existing evaluator unchanged. Skip the 18 declared unavailable cases. External escalation stays disabled without a dataset-bound provider approval record.

Exit condition: achieved. All expected identifiers are valid, coverage and missing units are reported, clean dataset and baseline rebuilds match the canonical post-quarantine artifacts, and the 56-test private suite plus 10 subtests passes.

### Phase 6: Audit and independent verification

Review the final diff for correctness, leakage, privacy, unsupported fallbacks, over-engineering, license boundaries, and unused dependencies. Re-run formatting, linting, the full root end-to-end suite, the existing benchmark tests, public benchmarks, and reproducible remote caller-boundary commands.

Exit condition: achieved for the private benchmark evidence boundary. Its 56 tests plus 10 subtests pass after clean dataset and baseline rebuilds. Root verification must be rerun after the remaining public-policy work, and blocked claims remain labeled.

## 10. Current file layout

Start small and split only after a file has distinct responsibilities:

```text
src/ocr_pipeline/
  cascade.py
  contracts.py
  demo.py
  layout.py
  operations.py
  openrouter.py
  pipeline.py
  providers.py
  repair.py
  cli.py
experiments/
  cascade_benchmark.py
  frontier_benchmark.py
  funsd_forms_benchmark.py
  mpdocbench_export.py
  omnidocbench_export.py
  omnidocbench_layout_benchmark.py
  paired_comparison.py
  pubtables_benchmark.py
  public_benchmark.py
  pulsebench_benchmark.py
data/
  README.md
tests/
  end-to-end and adapter tests
```

The pipeline package has no dependency on experiment code or the existing internal evaluator. Experiments import the pipeline. The internal adapter imports the existing evaluator only at the experiment edge.

No downloaded dataset, model weight, API secret, private image, or provider payload is committed.

## 11. Immediate stop conditions and open decisions

Stop or narrow the affected phase when:

- an official dataset license or access boundary cannot be confirmed;
- output validity is below the phase requirement;
- provider identity, retention, training use, or region is unclear;
- a model silently fills unreadable values;
- quality gains disappear when failures remain in denominators;
- benchmark release drift prevents a valid comparison;
- routing learns from gold labels, dataset cohort names, or test outcomes;
- a new abstraction has only one implementation and no immediate testing benefit.

Still required before a production recommendation:

- PHI, DPA or BAA, region, retention, and training-use approval;
- explicit critical-field precision and unsupported-fact limits;
- maximum cost per page or document;
- latency and throughput service levels;
- license review for restricted research comparators;
- measured results on real deployment-like documents.

## 12. Research sources

Primary papers read through the OpenResearch and alphaXiv workflow:

- [PaddleOCR-VL-1.6](https://www.alphaxiv.org/abs/2606.03264)
- [ClinOCR-Bench](https://www.alphaxiv.org/abs/2607.03650)
- [OmniDocBench](https://www.alphaxiv.org/abs/2412.07626)
- [Docling](https://www.alphaxiv.org/abs/2408.09869)
- [MPDocBench-Parse](https://www.alphaxiv.org/abs/2605.22100)
- [GLM-OCR](https://www.alphaxiv.org/abs/2603.10910)
- [FireRed-OCR](https://www.alphaxiv.org/abs/2603.01840)
- [OCRVerse](https://www.alphaxiv.org/abs/2601.21639)
- [SAFE-Cascade](https://www.alphaxiv.org/abs/2606.19646)
- [GRC](https://www.alphaxiv.org/abs/2603.19790)
- [FastOCR](https://www.alphaxiv.org/abs/2605.17447)
- [Infinity-Parser2](https://arxiv.org/abs/2607.07836)
- [Uncertainty-Aware OCR](https://openreview.net/forum?id=zyCjizqOxB)
- [Consensus Entropy](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_Consensus_Entropy_Harnessing_Multi-VLM_Agreement_for_Self-Verifying_and_Self-Improving_OCR_CVPR_2026_paper.html)
- [When Low CER Is Not Enough](https://www.alphaxiv.org/abs/2607.24077)
- [Reading or Guessing?](https://www.alphaxiv.org/abs/2605.27750)
- [DocIntent](https://www.alphaxiv.org/abs/2608.29037)
- [TADoc](https://www.alphaxiv.org/abs/2508.06988)
- [InstructTable](https://www.alphaxiv.org/abs/2604.02880)
- [Hierarchical Speculative Decoding for MLLMs](https://www.alphaxiv.org/abs/2602.12957)
- [JapanDocReader](https://www.alphaxiv.org/abs/2608.06758)
- [OvisOCR2](https://www.alphaxiv.org/abs/2607.13639)
- [HunyuanOCR with DFlash](https://www.alphaxiv.org/abs/2607.04884)

Official repositories, documentation, dataset cards, and model cards are linked in the relevant sections above. Repository-reported scores remain hypotheses until reproduced by this workspace.

## 13. Lossless inference optimization plan

Optimization is split into two tracks. A candidate stays in the lossless track only when every deterministic output matches the BF16 eager baseline across repeated runs. Any changed text, class, reading order, evidence link, cell, control state, or cross-page relation moves the candidate into the approximate track and requires full quality evaluation.

Low-risk operational candidates, still subject to exact-output regression tests:

- keep layout and recognition services resident;
- render each page once and reuse the same lossless pixel buffer for crops;
- preserve stable document, page, and block identifiers through bounded concurrency;
- batch independent layout pages and recognition crops, then restore original order deterministically;
- apply queue backpressure instead of unlimited requests;
- keep weights, compilation caches, and temporary renders on the A10G NVMe volume.

Behavior-preserving candidates that still require exact-output tests:

- CPU decode and image preparation overlapped with a single bounded GPU inference queue;
- pinned host buffers and nonblocking transfers where the official runtime supports them;
- CUDA graphs or `torch.compile` only around supported fixed-shape paths;
- resident Nemotron and a lineage-eligible layout worker, with research-only Heron residency measured separately;
- target-verified speculative decoding for a locally served Muse or Gemma repair model on larger hardware, never as an assumption about a hosted endpoint;
- HSD and DFlash as research-only serving candidates until the exact target model, runtime, hardware, and frozen OCR outputs are verified locally.

Nemotron native page batching has already failed the lossless criterion on the eight-page rotated exemplar panel. Batch size 1 reached median 10.937632 pages per second and 688,535,040 peak allocated bytes; batch size 8 reached 12.098005 pages per second and 3,292,832,256 bytes. Exact case-run equality within a batch size was only 0.90625 to 0.9375, and no case matched across every repeated batch and request permutation. Batching therefore stays out of the lossless path until the source of nondeterminism is removed and the full frozen panel matches exactly.

Approximate and excluded from the default path:

- FP8, INT8, INT4, GGUF quantization, or changing BF16 to FP16;
- lower image resolution, JPEG recompression, visual token pruning, KV compression, or sparse attention;
- skipping orientation or unwarping from a learned confidence threshold;
- model switching or text correction that can change visually supported content;
- unpinned OpenRouter provider or provider quantization changes.
- native PDF text bypass, until its extraction and visual-fallback policy are proven equivalent on held-out documents.

Ranked A10G experiment:

1. One lossless render per page with immutable crop reuse.
2. Diagnose Nemotron batch-dependent output changes before reconsidering batch sizes 2, 4, or 8; retain batch size 1 as the evidence baseline.
3. CPU page preparation overlapped with one GPU queue while restoring source order deterministically.
4. Page batching for the selected eligible layout detector, with Heron measured only as a separate research comparator, plus bounded specialist crop queues with stable evidence IDs.
5. `skip_relational` and `detector_only` only as explicit quality-changing ablations, not as lossless speedups.
6. CUDA graph, compilation, transfer, and supported operator ablations one at a time.
7. Separate quantized or reduced-resolution experiments only after the BF16 baseline is complete.
8. Evaluate native-PDF text preservation as its own quality-changing policy, not as an assumed lossless shortcut.

Adopt a lossless optimization only if exact outputs match across three runs, batch sizes 1, 4, and 8, request permutations, and supported concurrency schedules. Failures must not increase, peak A10G memory must stay below 90 percent, and throughput or p95 must improve by at least 10 percent. vLLM does not guarantee reproducibility by default, so a serving optimization is never called lossless from architecture alone. Stop raising concurrency when throughput improves by less than 5 percent, p95 worsens by more than 10 percent, or failures increase.

The earlier Paddle comparator reached 22,206 MiB in a point sample, or 96.4 percent of the A10G's available 23,028 MiB. That is one reason it is unsuitable as the deployable core under the new lineage rule. Nemotron's native sweep measured allocated memory on eight exemplars, but full-panel resident and process-level peak memory still need measurement rather than inference from its small parameter count.

Primary sources: [Nemotron OCR v2](https://huggingface.co/nvidia/nemotron-ocr-v2), [vLLM optimization](https://docs.vllm.ai/en/stable/configuration/optimization/), [vLLM speculative decoding](https://docs.vllm.ai/en/latest/features/speculative_decoding/), [FlashAttention](https://www.alphaxiv.org/abs/2205.14135), [FlashAttention-2](https://www.alphaxiv.org/abs/2307.08691), [PagedAttention](https://www.alphaxiv.org/abs/2309.06180), [HSD](https://www.alphaxiv.org/abs/2602.12957), and [DFlash](https://www.alphaxiv.org/abs/2602.06036). HSD and DFlash are Chinese-origin research directions, not deployable model components. Their author-reported speedups are not local OCR measurements, and distribution-preserving mechanisms still require exact greedy-output checks on the pinned stack.

## 14. Fine-tuning and annotation decision

Do not fine-tune before untuned Nemotron, Granite, eligible hosted arms, specialist oracles, and routing results identify the remaining bottleneck. Recognition tuning cannot repair layout misses, wrong reading order, checkbox-label association, or cross-page joins.

### 14.1 What recent OCR systems actually trained on

These are author-reported recipes from full technical reports read through OpenResearch and alphaXiv. They are methodological evidence, not local results.

| System | Reported scale | Training strategy | Relevant lesson |
| --- | --- | --- | --- |
| [Nemotron OCR v2](https://huggingface.co/nvidia/nemotron-ocr-v2) | About 12M images: roughly 680K real and more than 11M synthetic | Detector, recognizer, and relational reading-order training with word, line, paragraph, quad, and relation supervision | Foundation-scale evidence, but the released package is inference-only and removes training methods. Keep it frozen unless NVIDIA releases a supported recipe. |
| [NVIDIA OCR Synthetic Multilingual v1](https://huggingface.co/datasets/nvidia/OCR-Synthetic-Multilingual-v1) | 12,258,146 released synthetic pages | Full-page text plus word, line, paragraph boxes, quads, and relation graphs | Reuse its schema and relation representation. Audit source and commercial terms before any training mix. |
| [SmolDocling](https://www.alphaxiv.org/abs/2503.11576) | 60K human-reviewed layout pages, 63K WordScape pages, 250K synthetic layout pages, 2.5M charts, 9.3M code images, and 5.5M formulas | Add DocTags with the vision encoder frozen, unfreeze for document pretraining and task data, then tune on the combined mix | This is a foundation curriculum, not a domain-adapter sample requirement. It supports decoder and projector tuning before unfreezing vision. |
| [Granite Docling 258M](https://huggingface.co/ibm-granite/granite-docling-258M) and [two-stage variant](https://huggingface.co/docling-project/granite-docling-2stage-258m) | Named dataset cards expose 1,270,911 DoclingMatix, 8,400,838 SynthCodeNet, and 1,981,157 SynthChartNet rows, about 11.65M available rows total; actual sampled mixture, epochs, and SynthFormula contribution are undisclosed | nanoVLM-based image-to-DocTags training; the two-stage release adds layout-derived dynamic prompts and reports full-page edit distance improving from 0.45 to 0.27 | Best first eligible adaptation target if DocTags, reading order, or tables remain the measured bottleneck. Do not claim every available row was sampled. |
| [JapanDocReader](https://www.alphaxiv.org/abs/2608.06758) | About 94K synthetic VQA examples, 25K structured-parse SFT examples, and 7K filtered RL prompts | Mixed SFT limits forgetting; DAPO improves parsing while VQA can drift | The Stockmark and NVIDIA base is non-Chinese, but the reported data-generation recipe used a Qwen teacher. Under the strict teacher-origin rule this is methodological evidence only, not an eligible training recipe. |
| [BanglaWild](https://www.alphaxiv.org/abs/2608.03884) | 1,268 images | LoRA adaptation with character-level adjudication and an explicit uncertainty marker | About 1K examples can teach output behavior, but the strongest base model slightly regressed. Strong models must earn tuning through a held-out learning curve. |
| [Iterative manuscript fine-tuning](https://www.alphaxiv.org/abs/2608.18696) | One to three corrected pages per manuscript | Writer or manuscript-specific iterative adaptation | Few-page gains apply only to a stable manuscript distribution and do not support broad clinical generalization. |

The scale lesson is not "collect millions first." Foundation systems and domain adapters solve different problems. Start with a rights-cleared, adjudicated learning curve and expand only while independently grouped validation continues to improve.

Nemotron is not currently a practical fine-tuning target because NVIDIA's released package is inference-only. The first eligible tuning candidate is Granite Docling 258M, but only if oracle stage analysis shows that DocTags, layout, reading order, or table reconstruction is the dominant residual error. Start with assistant-only supervised next-token loss, freeze the vision encoder, and tune the decoder plus multimodal projector. Unfreeze vision only if scanner, blur, font, or handwriting appearance errors remain after the frozen-vision curve plateaus.

A 30B model is not the first tuning target. Official Muse Glimmer BF16 LoRA guidance calls for one 80 GB H100, and Gemma 4 31B BF16 weights alone exceed the A10G. If hosted crop repair remains the bottleneck, select the stronger untuned model, adapt only that model on larger hardware, and keep the other frozen as the independent verifier. QLoRA on 24 GB is an engineering experiment, not an accuracy-preserving default. Use supervised fine-tuning before any RL or GRPO experiment.

### 14.2 Canonical label record

Keep one evidence-linked annotation record and derive model-specific training examples from it. At minimum, record:

- document, source family, rights, page ID, image size, language, degradation, rotation, and layout type;
- blocks with stable ID, type, polygon, reading index, verbatim text, separate normalized text, legibility, and parent or caption links;
- text status from `legible`, `partly_legible`, `unreadable`, or `ambiguous` so annotators never guess;
- uncertain spans with exact character boundaries, degradation type, and whether the uncertainty is visual or annotation-induced;
- table IDs, rows, columns, cells, blank cells, row and column spans, headers, geometry, and continuation links;
- form key and value block IDs, normalized value, relation, and field status;
- control state from `checked`, `unchecked`, `indeterminate`, or `unreadable`, plus label and group links;
- reading-next, heading-parent, figure-caption, paragraph-continuation, and table-continuation relations;
- adjudication status, annotator uncertainty, and evidence block IDs.

Store model error analysis separately from truth using omission, unsupported insertion, substitution, structure, association, reading-order, and wrong-abstention labels.

Use region examples for recognition failures and full-page or multi-page examples only when global structure is required. Hard examples should vary one digit, date, unit, negation, state, faint entry, span, or page continuation while preserving the rest of the document.

### 14.3 Selection, split, and quality control

Practical labeling loop:

1. Run the untuned local parser to prefill blocks, text, tables, controls, and relations.
2. Show the source image and prefill together, but require character-level correction rather than acceptance by default.
3. Assign a second blind reviewer to every critical field, unreadable span, table span, control state, and relation. Randomly double-review the remaining pages.
4. Send disagreements to adjudication with the source pixels visible. Never resolve from domain plausibility alone.
5. Run deterministic checks for geometry bounds, duplicate IDs, reading-order permutations, table rectangularity and spans, control-label links, cross-page references, and exact rendered structure.
6. Keep immutable truth revisions and separate model-error tags. Derive DocTags, literal text, Markdown, HTML, and task-specific training records from the canonical truth instead of editing separate labels.
7. Measure character agreement, block IoU, table-cell agreement, relation agreement, unreadable agreement, and adjudication rate by source family before releasing a training slice.

- Group every page, crop, perturbation, and derivative from the same source document into one split.
- Hold out customer, provider, time, template, and document family where relevant.
- Never train on the 328 ClinOCR evaluation pages or general benchmark evaluation annotations.
- Use human adjudication for high-risk fields and unresolved model disagreement.
- Render tables and formulas back for deterministic structure checks.
- Start learning curves at 250, 500, 1,000, and 2,000 independent documents or high-value regions, with document-group holdouts and at least three seeds. These are checkpoints, not claimed sufficiency thresholds.
- Compare random sampling with uncertainty plus diversity sampling. Stop labeling when two successive increments improve the target by less than 0.5 absolute points.

Proceed to supervised adaptation only when a recurring model-stage error is at least 20 percent of material errors, replacing only that stage with human truth recovers at least 2 end-to-end points or 5 critical-slice points, preprocessing and routing fixes do not recover the gap, and at least 250 rights-cleared adjudicated examples cover multiple document families.

Adopt the adapter only when its paired 95 percent interval excludes zero, unsupported insertions do not rise, no untouched task regresses by more than 0.5 points, and latency and memory stay inside the operating envelope. Do not tune when errors are mainly layout, association, reconstruction, ambiguous truth, service failures, or isolated document families.

## 15. Independent audit and remaining evidence gaps

The independent correctness review reproduced and repaired these material defects before the final commit:

- multi-frame TIFFs now expand into ordered pages instead of silently losing frames;
- clean zero-region reader results now create an explicit full-page repair target, while reader exceptions remain failures;
- table regions cannot be repaired into flat text, and every cell span is validated before rowspan-aware rectangularity abstention;
- primary and verifier roles must return different actual model identifiers;
- OpenRouter responses must return the exact requested model, so a benchmark-only or unexpected route cannot pass production role checks;
- Nemotron's normalized `left`, `right`, `upper`, and `lower` coordinates are converted to pixel geometry using the source image size and the official lower-to-upper vertical convention;
- Granite DocTags are converted with Docling's visible-text export before CER or WER scoring, so Markdown punctuation is not counted as OCR content;
- Granite checks for Docling Core before generation rather than wasting inference and mislabeling a missing dependency as an output failure;
- stored benchmark edit counts cannot manufacture gains because paired CER and WER are recomputed from raw text;
- exact bit-parallel Levenshtein recomputation is cross-checked against the full edit alignment and makes full-page paired evaluation practical without changing scores;
- ClinOCR paired inference uses template clusters rather than treating transformed pages as independent;
- length-limited or otherwise abnormal hosted finish reasons are rejected;
- new run artifacts serialize every behavior-affecting reader and run option; pre-serializer Paddle artifacts remain explicitly labeled;
- accepted hosted text records both model and provider identities while retaining local geometry provenance;
- expected provider failures keep structured status, attempts, and latency, while unexpected programming exceptions escape;
- multi-frame, empty-page, special-token, provider-error, provenance, and cluster regressions cross caller boundaries in tests.

Ponytail over-engineering audit:

`delete:` unused Qwen3-32B test-only relation-call surface. Keep it in the research plan until a real caller and evaluation exist. `[src/ocr_pipeline/openrouter.py]`

`net: -48 production lines, -0 dependencies possible.`

The remaining evidence gaps are not hidden by passing tests:

- no fresh-key, provider-pinned OpenRouter accuracy, cost, latency, provider-identity, or ZDR run exists;
- an eligible replacement for the measured research-only Heron arm and the official full version-matched OmniDocBench end-to-end model score remain unmeasured;
- external-upload privacy is a documented operating boundary, not a technical PHI detector;
- plausible wrong text, missed layout regions, controls, and cross-page relationships exceed the current deterministic router;
- two-model literal agreement is supporting evidence, not ground truth, and still requires held-out false-accept and risk-coverage measurement;
- point GPU memory is not peak memory; Nemotron batching and the full routed stack still need measured peaks and stable p95 throughput.
