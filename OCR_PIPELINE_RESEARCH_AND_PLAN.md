# Scalable, Accurate, Cost-Aware OCR Pipeline

Research and execution plan, 2026-09-01

## 1. Goal and evidence standard

Build a research pipeline that turns PDFs and images into an evidence-linked structured document while preserving literal text, geometry, reading order, tables, controls, formulas, and cross-page relationships. The system must be modular enough to exchange the local parser, hosted escalator, route policy, and output adapter without changing the core document contract.

The target is accuracy under explicit cost and latency constraints. Cost is optimized only after quality and safety requirements are met. Failed pages, invalid responses, abstentions, and unavailable cases remain visible and remain in evaluation denominators.

This work is research-only. Licenses and model provenance are recorded separately from technical quality. Research permission does not imply production, privacy, PHI, or commercial approval.

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
- swappable Tesseract, full PaddleOCR-VL-1.6, full GLM-OCR SDK, and direct GLM-OCR model-only readers;
- one typed evidence contract for page size, regions, geometry, reading order, literal text, confidence when available, provider, route, and failures;
- deterministic detection of empty content, invalid geometry, invalid reading order, malformed tables, and repeated multi-token hallucination loops;
- selective crop or full-page repair that protects region ID, geometry, kind, and local geometry-provider provenance;
- independent visual verification: a proposed patch is merged only when a second image model, which never sees the first candidate, independently returns the same literal text and the deterministic risk set shrinks;
- a strict OpenRouter adapter for Qwen3.8-Flash and Muse Glimmer visual calls, with provider pinning, structured output, no-data-collection and ZDR requests, bounded retries, truncation rejection, actual response metadata, and environment-only credentials;
- failure-inclusive public transcription, direct hosted, stitched cascade, and OmniDocBench export harnesses.

The direct GLM reader is explicitly a model-only ablation. Qwen3-32B remains a planned text-only relation stage, not an implemented OCR reader or unused provider surface. The complete GLM SDK remains the preferred GLM comparison because it supplies layout analysis and structured parsing.

"Evidence-Patch Cascade" is a project synthesis label, not a novelty claim. SAFE-Cascade supplies a selective-routing prior, GRC and Consensus Entropy supply stability and disagreement priors, and the local contribution being tested is the combination of typed OCR evidence, deterministic failure detection, independently generated visual candidates, protected patches, and explicit abstention.

### 1.2 Measurements completed here

| Caller-boundary run | Cases | Coverage | Micro CER | Micro WER | p50 | p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Tesseract, ClinOCR v1.0 evaluation | 328 | 91.2% | 0.4514 | 0.5677 | 0.94 s | 1.66 s |
| Tesseract, FUNSD original test | 50 | 100% | 0.5650 | 0.7902 | 0.58 s | 0.95 s |
| PaddleOCR-VL-1.6, one page from each ClinOCR subset | 6 | 100% | 0.1986 | 0.2313 | 11.20 s | 79.19 s |
| Direct GLM-OCR model-only, same six development pages | 6 | 100% | 0.1093 | 0.1178 | 9.36 s | 18.34 s |
| PaddleOCR-VL-1.6 base, full ClinOCR rotated subset | 56 | 100% | 0.3137 | 0.3780 | 12.82 s | 41.29 s |
| PaddleOCR-VL-1.6 with document unwarping, full ClinOCR rotated subset | 56 | 100% | 0.0607 | 0.0810 | 13.19 s | 20.47 s |
| Direct GLM-OCR model-only, full ClinOCR rotated subset | 56 | 100% | 0.0790 | 0.0837 | 10.79 s | 15.63 s |

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

No live OpenRouter request was made with the credential pasted into chat. That credential is treated as compromised. Direct Qwen, direct Muse, and stitched superiority remain unmeasured until a fresh environment key and exact provider pins are supplied.

## 2. Research conclusion

The smallest strong first system is the Evidence-Patch Cascade:

1. Use the full [PaddleOCR-VL-1.6](https://github.com/PaddlePaddle/PaddleOCR) pipeline as the first local structured baseline.
2. Retain its regions, literal text, geometry, reading order, and provider provenance.
3. Detect only observable structural or transcription risks, never benchmark cohort names, reference text, or downstream scores.
4. Ask one hosted visual model to read only a risky crop.
5. Ask a different visual model to read the same crop independently without revealing the first candidate.
6. Merge only strict literal agreement that reduces the original deterministic risk; otherwise preserve the local result and abstain.
7. Compare [GLM-OCR](https://github.com/zai-org/GLM-OCR) under its complete SDK and as a clearly labeled model-only ablation.
8. Add validated native PDF text preservation and document-level relation repair only after their own end-to-end evidence exists.

The Paddle documentation explicitly distinguishes the complete pipeline from direct VLM inference. The complete pipeline adds layout detection, region-level recognition, and structured merge. A VLM-only run is therefore an ablation, not the baseline. See the official [PaddleOCR-VL pipeline guide](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html) and the [PaddleOCR-VL-1.6 paper](https://www.alphaxiv.org/abs/2606.03264).

Why this is the first baseline:

- PaddleOCR-VL-1.6 is a compact 0.9B local model with an Apache-2.0 code and model path.
- PP-DocLayoutV3 supplies a reusable structure boundary instead of forcing the hosted model to rediscover page geometry.
- GLM-OCR is also compact and uses the same layout family, enabling a controlled recognizer comparison.
- Paddle and GLM agreement is not independent layout evidence because both complete pipelines use PP-DocLayoutV3. Layout failures need separate checks or a genuinely different detector.
- Qwen3.8-Flash and Muse Glimmer are useful visual candidates, but their OCR accuracy, page cost, provider behavior, and relative value are unverified here.
- Two-model agreement is stronger support than heuristic disappearance, but agreement is not ground truth. Correlated mistakes remain possible and must be measured through false-accept and risk-coverage curves.
- Tesseract remains a cheap CPU floor and integration check, not the expected complex-document winner.

## 3. Reusable ideas and code boundaries

### 3.1 Systems worth reusing or comparing

| System | Reuse in this project | License or boundary | Current evidence status |
| --- | --- | --- | --- |
| [PaddleOCR-VL-1.6](https://github.com/PaddlePaddle/PaddleOCR) | Primary full local pipeline, layout-aware crops, ordered structured output | Apache-2.0 code and model path | Author reports 96.33 on OmniDocBench v1.6. Must reproduce on selected public and private data. |
| [PP-DocLayoutV3](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3) | Shared region, polygon, class, and reading-order stage | Apache-2.0 | Strong reusable component boundary. |
| [GLM-OCR](https://github.com/zai-org/GLM-OCR) | Paired recognizer comparison under the same layout stage | Apache-2.0 code, MIT model | Author reports 94.62 on OmniDocBench v1.5. Different benchmark revisions are not directly comparable. |
| [PP-StructureV3](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/pipeline_usage/PP-StructureV3.en.md) | Non-VLM component ablations for OCR, tables, formulas, charts, cells, and reconstruction | Apache-2.0 | Useful only when a failure needs stage-level isolation. |
| [Docling](https://github.com/docling-project/docling) | Ideas for a lossless document representation, native PDF preservation, adapters, and serializers | MIT | Reuse the contract ideas before adding the dependency. |
| [MinerU](https://github.com/opendatalab/MinerU) | Cross-page tables, truncated paragraph continuation, hybrid parsing comparison | Custom Apache-derived terms | Research comparison only until legal review. |
| [olmOCR](https://github.com/allenai/olmocr) | Broad parser comparator and unit-test-oriented benchmark design | Apache-2.0 | Larger than the 0.9B candidates and not the cheapest default. |
| [Surya](https://github.com/datalab-to/surya) | Compact research comparison for OCR, layout, reading order, and tables | Code and weight terms differ; weights have use limits | Research comparison only until the exact terms are accepted. |
| [NaviDC-OCR](https://github.com/caipeng328/NaviDC-OCR) | Research ideas for geometry-aware decoding, camera documents, render checks, and content-structure separation | Repository license unresolved in this review | Do not copy source until a clear license exists. |
| [OvisOCR2](https://huggingface.co/ATH-MaaS/OvisOCR2) | Compact end-to-end Markdown comparator | Apache-2.0 model card | Author results only; deployment and parser reliability remain unverified. |
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
  -> EXIF-normalized lossless page image
  -> PP-DocLayoutV3 structure and reading order
  -> PaddleOCR-VL-1.6 region recognition
  -> deterministic merge and validators
  -> route selected risky regions to Qwen3.8-Flash or Muse
  -> independent second-model visual agreement
  -> protected patch or explicit abstention
  -> structured JSON plus Markdown or task adapter

Planned after measured need:
  -> validated native PDF text preservation
  -> padded or full-page context for relationship failures
  -> Qwen3-32B existing-ID relation repair
  -> cross-page continuation and hierarchy repair
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

## 5. OpenRouter integration

The required hosted model is `qwen/qwen3.8-flash`. The official OpenRouter model page currently describes multimodal input, structured output support, a 1M-token context, and pricing of USD 0.15 per million input tokens and USD 0.47 per million output tokens. These are current list prices, not measured page costs.

Additional user-selected comparators have distinct roles:

| Model | Verified role | Current list price | Experiment decision |
| --- | --- | --- | --- |
| [`qwen/qwen3.8-flash`](https://openrouter.ai/qwen/qwen3.8-flash) | Multimodal text, image, and video input with structured output | USD 0.15 input and USD 0.47 output per million tokens | Primary hosted OCR and document-understanding escalator. |
| [`meta/muse-glimmer-30b`](https://openrouter.ai/meta/muse-glimmer-30b) | Multimodal text and image input with structured output | USD 0.30 input and USD 1.10 output per million tokens | Higher-cost visual challenger on the same fixed public pages. It replaces Qwen only if the paired quality gain satisfies the quality-first decision rules. |
| [`qwen/qwen3-32b`](https://openrouter.ai/qwen/qwen3-32b) | Dense causal language model with structured output; the current model page does not advertise image input | USD 0.08 input and USD 0.28 output per million tokens | Planned text-only schema extraction or existing-ID relation repair after OCR. It is not wired into the current pipeline and cannot be ranked as an image OCR engine unless endpoint capabilities change and are reverified. |

The model slug is configuration, not a new adapter. The current request builder accepts only the two verified image-capable model roles. A future text-only relation stage must use a separate request boundary so an image is never silently dropped.

The implemented adapter:

- read `OPENROUTER_API_KEY` only from the environment;
- send image data through the official multimodal chat-completions format;
- require a strict JSON Schema response;
- request providers that support required parameters;
- request no data collection and zero-data-retention routing when available;
- use bounded retries and a capped `Retry-After` delay;
- reject missing, length-limited, or otherwise non-normal finish reasons;
- record actual returned model, provider when supplied, usage, cost, attempts, latency, and normalized and native finish reasons without estimating missing provider costs;
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
| [MPDocBench-Parse](https://github.com/Tongyi-Zhiwen/Qwen-Doc/tree/main/MPDocBench) and [paper](https://www.alphaxiv.org/abs/2605.22100) | Public release: 420 PDFs and 3,135 pages; paper: 433 documents and 3,246 pages before removals | Multi-page hierarchy, continuation, reading order, figures, tables, and formulas | Research terms, no train split, additional formula dependencies. Use released counts, not pre-release paper counts. |
| [olmOCR-Bench](https://github.com/allenai/olmocr) | More than 7,000 unit tests over about 1,400 documents in the project description | Broad parser regression and failure taxonomy | Primarily English and unit-test oriented; benchmark artifact terms need confirmation before redistribution. |

### 6.2 Component benchmarks

| Benchmark | Verified scale | Capability and metrics | Acquisition decision |
| --- | --- | --- | --- |
| [DocLayNet](https://github.com/DS4SD/DocLayNet) | 80,863 pages, with 69,375 train, 6,489 validation, and 4,999 test pages; 11 layout classes | Layout precision, recall, F1, and COCO mAP | Use test or a fixed test slice later. It is large and has no end-to-end transcription truth. |
| [FUNSD](https://guillaumejaume.github.io/FUNSD/) | Acquired 199 noisy forms, split 149 train and 50 test | Word transcription first; entity and key-value relations need a separate adapter | Small and useful, but non-commercial research and education terms apply. |
| [CheckboxQA](https://github.com/Snowflake-Labs/CheckboxQA) | Acquired 88 documents, 2,048 retained PDF pages, and 579 QA pairs | Official ANLS* for downstream checkable-content understanding | Evaluation-only CC BY-NC plus underlying DocumentCloud terms. It is not direct checkbox-detection ground truth. |
| [IAM Handwriting](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database) | 1,539 pages from 657 writers, 13,353 lines, and 115,320 words | CER and WER for handwriting | Registered official access is required. Do not use an unofficial mirror. |
| [PubTables-1M](https://github.com/microsoft/table-transformer) | 575,305 page images and 947,642 structure tables | Detection AP and AR, GriTS topology, content, and location | About 117 GB. Start with a fixed stratified test slice and official evaluator. |
| [ICDAR 2019 cTDaR](https://zenodo.org/records/3239032) | Separate modern and historical tracks with overlapping detection and structure subsets | Table detection and adjacency-based structure F1 across IoU thresholds | Good archival and handwritten holdout. Confirm archive terms before use. |
| [CORD v2](https://github.com/clovaai/cord) | 1,000 public receipts, split 800, 100, and 100 | Receipt OCR and hierarchical extraction | Useful extra document-domain stress test under CC BY 4.0. |
| [SROIE](https://arxiv.org/abs/2103.10213) | 1,000 receipts, 600 train or validation and 400 test | Localization, OCR, and four-field extraction F1 | Verify current RRC access and use terms before acquisition. |

### 6.3 Existing internal benchmark

The existing `internal-clinical-ocr-benchmark` remains unchanged. It contains 432 cases, of which 415 are ready and 17 are unavailable, across forms, handwriting, and tables. New systems write adapter predictions that satisfy its current format. Existing sources, evaluator code, tests, historical predictions, results, and reports are not rewritten.

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
2. Full PaddleOCR-VL-1.6 BF16 pipeline.
3. Direct GLM-OCR model-only ablation.
4. Complete GLM-OCR SDK pipeline.
5. Direct full-page Qwen3.8-Flash with an exact provider pin.
6. Direct full-page Muse Glimmer 30B with an exact provider pin.
7. Paddle plus Qwen repair independently verified by Muse.
8. Paddle plus Muse repair independently verified by Qwen.
9. A hosted-budget-matched cascade arm when multiple crop calls can exceed one direct page call.
10. Gold-route and gold-candidate oracles for diagnosis only, never as deployable results.

Run BF16 before testing one quantized configuration. A quantized model is accepted only if its paired quality loss is within the locked task bounds and its operational gain is real.

Required ablations:

- full pipeline versus VLM-only;
- full page versus layout crops;
- crops versus crops with neighboring label context;
- Paddle recognizer versus GLM recognizer with the same layout stage;
- native PDF text preservation on versus off;
- Qwen and Muse escalation off versus independently verified repair;
- heuristic-only, disagreement-only, and combined routing;
- tight crop, padded crop with nearby labels, and full-page context;
- BF16 versus one selected quantization;
- page-level versus document-level reconstruction.

Direct and stitched arms must share source bytes, prompt intent, strict schema, output limit, provider and quantization pin, failure denominator, and scoring normalization. Record the actual returned model and provider because a model slug alone does not establish the serving implementation.

The 30B plus pipeline USP is established only if stitched Muse:

- beats or is non-inferior to direct Muse on the same held-out cases;
- beats or is non-inferior to direct Qwen3.8-Flash;
- adds value over the otherwise identical stitched Qwen arm; and
- reaches those results through fewer hosted pages or lower reported hosted cost without increasing unsupported digits, identifiers, entities, controls, or table cells.

Until those four conditions hold under paired confidence intervals, the architecture is a testable synthesis, not a frontier-model superiority claim.

### 8.2 Routing experiment

Reuse identical cached stage outputs to compare:

1. local-only;
2. each hosted model on every eligible page;
3. heuristic-only selective repair;
4. independent-disagreement selective repair;
5. combined routing;
6. gold-oracle routing for research diagnosis only.

Calibrate routing only on the 56 ClinOCR exemplars and a rights-cleared document-grouped development set. Lock the policy before the 328 ClinOCR evaluation pages and the exact OmniDocBench release. Report gate precision, recall, AUPRC, risk-coverage, recoverable-error recall, escalation rate, false-accept rate, abstention rate, and cost completeness.

Stop routing work if oracle escalation improves the primary metric by less than 1 absolute point and recovers fewer than 10 percent of local errors.

Accept automatic routing only if it:

- captures at least 90 percent of oracle-recoverable gain;
- costs at most 40 percent of the all-Qwen policy;
- does not increase unsupported critical facts;
- regresses neither control-state F1 nor table-position accuracy by more than 0.5 points.

For the broad architecture claim, require non-inferiority to the better direct hosted arm on transcription with a one-sided paired 95 percent lower bound above -0.5 absolute points. Require at least a 2-point table-TEDS or reading-order improvement with the paired lower bound above zero. The local transcription-only harness cannot establish either structural condition; the version-pinned OmniDocBench evaluator is required.

Do not optimize a blended quality-cost score. Discard every policy that violates a quality, safety, privacy, or reliability constraint, then choose the cheapest remaining policy.

### 8.3 Parser selection

Adopt PaddleOCR-VL-1.6 as the primary parser only if, on the full internal benchmark:

- coverage-adjusted task macro is non-inferior to the best current baseline within 1 percentage point;
- at least one safety-critical task improves by 2 points or more;
- no task or cohort regresses by more than 2 points;
- unsupported outputs and unresolved controls or cells do not increase;
- measured p95 latency, throughput, and infrastructure cost meet the selected operating target.

Adopt GLM-OCR instead only if its paired gain over Paddle is meaningful with the same layout stage. Enable hosted Qwen escalation only if it improves evidence-supported critical-field recovery by at least 2 points at no more than a 10 percent escalation rate and satisfies provider privacy requirements.

These thresholds are research defaults. Product owners must lock critical-field precision, maximum cost, latency, and privacy constraints before held-out selection.

## 9. Phasewise execution

| Phase | Current status |
| --- | --- |
| 0. Contract and spine | Completed and covered through the real CLI boundary. |
| 1. Public transcription floor | Completed for full ClinOCR evaluation and FUNSD test with failure-inclusive Tesseract results. |
| 2. GPU baseline | In progress. Matched 56-page Paddle preprocessing and direct GLM model-only runs are measured; complete GLM and broader public structure runs remain. |
| 3. Hosted adapters | Code and injected end-to-end tests completed. Live public calls are blocked on a rotated key and provider pins. |
| 4. Routing and structural evaluation | Partial. Validators, independent visual agreement, direct and stitched harnesses, CheckboxQA data, and OmniDocBench export exist; held-out hosted and official structural scores do not. |
| 5. Internal bridge | Not implemented. The existing private benchmark and its files remain untouched. |
| 6. Audit | Completed for the current code. Independent review findings were repaired and caller-boundary verification passed; hosted superiority remains an evidence blocker. |

### Phase 0: Pipeline contract and end-to-end spine

Deliver ordered image and PDF ingestion, canonical evidence-linked JSON, an explicit local reader, an explicit escalator interface, routing states, and a CLI. Test the real CLI over a multi-page fixture. Page order, geometry, abstentions, and failures must survive serialization.

Exit condition: achieved. The complete local fixture path passes from user input to serialized result with page order, geometry, evidence, failures, and nonzero partial status preserved.

### Phase 1: Local floor and public transcription

Add the Tesseract floor adapter and public evaluation harness. Download ClinOCR-Bench and FUNSD from official sources. Attempt every eligible case and retain failures in denominators.

Exit condition: achieved for the transcription floor. All 328 ClinOCR evaluation cases and 50 FUNSD test cases were attempted, with failures retained.

### Phase 2: GPU structure-aware baseline

On the authorized A10G, install the full PaddleOCR-VL-1.6 pipeline in an isolated environment stored on the NVMe volume. Keep model caches on NVMe because the root filesystem has limited free space. Sync source to `/home/ubuntu/aman/notSoSmartOCR` without deleting existing files. Run BF16 first and record peak VRAM, throughput, stage latency, and output validity.

Exit condition: not yet achieved. Full Paddle base, Paddle unwarping, and direct GLM model-only runs are matched on the 56-page rotated subset. The complete GLM SDK comparison, official structural scoring, peak-memory sampling, and broader held-out runs remain.

### Phase 3: Hosted Qwen escalator

Implement strict structured output, provider constraints, retries, usage accounting, and route reasons. Use a rotated environment key. Test only public images first.

Exit condition: blocked on a fresh key and explicit provider pins. The adapter behavior is tested with injected caller-boundary responses, but no provider validity, accuracy, latency, or cost percentage is measured.

### Phase 4: Routing and multi-capability evaluation

Calibrate deterministic route signals on development data. Run local-only, all-Qwen, oracle, and automatic policies over the same outputs. Add CheckboxQA and selected OmniDocBench, DocLayNet, PubTables-1M, and cTDaR slices only after source terms and evaluators are confirmed.

Exit condition: not yet achieved. A rejected, truncated, same-model, or disagreeing patch now leaves local evidence unchanged and records a distinct abstention. Clean zero-region pages receive a full-page repair target; provider exceptions remain explicit failures.

### Phase 5: Existing benchmark bridge

Write a separate adapter into a new run directory and evaluate all 415 ready internal cases with the existing evaluator unchanged. Skip the 17 declared unavailable cases. External escalation stays disabled without PHI approval.

Exit condition: all expected identifiers are valid, coverage and missing units are reported, and the original 49-test suite still passes.

### Phase 6: Audit and independent verification

Review the final diff for correctness, leakage, privacy, unsupported fallbacks, over-engineering, license boundaries, and unused dependencies. Re-run formatting, linting, the full root end-to-end suite, the existing benchmark tests, public benchmarks, and reproducible remote caller-boundary commands.

Exit condition: achieved for the current code and evidence boundary. All material review findings were repaired, 75 root tests and the untouched private benchmark's 49 tests plus 10 subtests pass, and blocked claims remain labeled.

## 10. Current file layout

Start small and split only after a file has distinct responsibilities:

```text
src/ocr_pipeline/
  cascade.py
  contracts.py
  openrouter.py
  pipeline.py
  providers.py
  repair.py
  cli.py
experiments/
  cascade_benchmark.py
  frontier_benchmark.py
  omnidocbench_export.py
  paired_comparison.py
  public_benchmark.py
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

- vLLM or FastDeploy continuous batching and PagedAttention;
- chunked prefill, prefix cache, and multimodal processor cache;
- CUDA graphs, `torch.compile`, FlashAttention, and nonblocking transfers;
- PP-DocLayoutV3 TensorRT FP32 execution;
- target-verified speculative decoding or GLM-OCR multi-token prediction under greedy decoding.

Approximate and excluded from the default path:

- FP8, INT8, INT4, GGUF quantization, or changing BF16 to FP16;
- lower image resolution, JPEG recompression, visual token pruning, KV compression, or sparse attention;
- skipping orientation or unwarping from a learned confidence threshold;
- model switching or text correction that can change visually supported content;
- unpinned OpenRouter provider or provider quantization changes.
- native PDF text bypass, until its extraction and visual-fallback policy are proven equivalent on held-out documents.

Ranked A10G experiment:

1. One lossless render per page with immutable crop reuse.
2. Layout page batches and bounded crop concurrency with stable merge order.
3. Resident PaddleOCR-VL-1.6 BF16 service with vLLM controls for GPU utilization, sequence count, batched tokens, and client concurrency.
4. Paddle vLLM versus FastDeploy using identical weights and decoding.
5. GLM-OCR vLLM versus SGLang using identical weights and decoding.
6. CUDA graph, compilation, attention, transfer, and verified speculative ablations one at a time.
7. Separate approximate experiments only after the BF16 baseline is complete.
8. Evaluate native-PDF text preservation as its own quality-changing policy, not as an assumed lossless shortcut.

Adopt a lossless optimization only if exact outputs match across three runs, batch sizes 1, 4, and 8, request permutations, and supported concurrency schedules. Failures must not increase, peak A10G memory must stay below 90 percent, and throughput or p95 must improve by at least 10 percent. vLLM does not guarantee reproducibility by default, so a serving optimization is never called lossless from architecture alone. Stop raising concurrency when throughput improves by less than 5 percent, p95 worsens by more than 10 percent, or failures increase.

The first native Paddle run already reached 22,206 MiB in a point sample, or 96.4 percent of the A10G's available 23,028 MiB. Before concurrency work, reduce the configured memory envelope or change the serving path while holding BF16 weights, source bytes, decoding, and outputs fixed.

Primary sources: [PaddleOCR-VL deployment](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/pipeline_usage/PaddleOCR-VL.en.md), [Paddle high-performance configuration](https://github.com/PaddlePaddle/PaddleOCR/blob/main/deploy/paddleocr_vl_docker/pipeline_config_fastdeploy.yaml), [vLLM optimization](https://docs.vllm.ai/en/stable/configuration/optimization/), [GLM-OCR serving](https://github.com/zai-org/GLM-OCR), [FlashAttention](https://www.alphaxiv.org/abs/2205.14135), [FlashAttention-2](https://www.alphaxiv.org/abs/2307.08691), [PagedAttention](https://www.alphaxiv.org/abs/2309.06180), and [DFlash](https://www.alphaxiv.org/abs/2602.06036).

## 14. Fine-tuning and annotation decision

Do not fine-tune before untuned Paddle, GLM, hosted escalation, and routing results identify the remaining bottleneck. Recognition tuning cannot repair layout misses, wrong reading order, checkbox-label association, or cross-page joins.

### 14.1 What recent OCR systems actually trained on

These are author-reported recipes from full technical reports read through OpenResearch and alphaXiv. They are methodological evidence, not local results.

| System | Reported scale | Training strategy | Relevant lesson |
| --- | --- | --- | --- |
| [PaddleOCR-VL-1.6](https://www.alphaxiv.org/abs/2606.03264) | 16.8M continued-pretraining samples, 7.3M hard SFT samples, and 49K RL samples | One CPT epoch and one SFT epoch with all parameters unfrozen; 16 rollouts per RL candidate; filter reward-flat, over-easy, over-hard, and low-headroom groups; select top 8K per task for two GRPO epochs | Most reported gain comes from CPT and SFT. Mine unstable and mislabeled regions, repair supervision, then use a small high-value RL set. |
| [GLM-OCR](https://www.alphaxiv.org/abs/2603.10910) | Vision training is described as tens of billions of image-text pairs; OCR SFT and RL sample counts are not disclosed | Vision training, multimodal pretraining, MTP pretraining, balanced OCR SFT with MTP, then difficulty-stratified GRPO using NED, CDM, TEDS, field F1, closure, JSON, and repetition rewards | MTP must remain aligned across training and inference. Do not invent undisclosed OCR-stage counts. |
| [FireRed-OCR](https://www.alphaxiv.org/abs/2603.01840) | About 1.3M multi-task pre-alignment samples, 400K document-to-Markdown SFT pairs, and 50K GRPO samples | Geometry plus semantics data factory, coarse-to-fine labels, SFT at global batch 256 and learning rate 3e-5, GRPO at 5e-7, balanced 1:1:1 text/table/formula rewards, and iterative SFT-GRPO | Balance modalities and reintroduce SFT to counter structural reward hacking and empty or repetitive outputs. |
| [OCRVerse](https://www.alphaxiv.org/abs/2601.21639) | The report describes large-scale eight-domain mixtures but does not disclose a total SFT or RL count | Mix text, document, table, formula, chart, webpage, scientific-plot, and graphics data for SFT; select hard RL data with entropy and quality filters; use domain-specific rule or visual-fidelity rewards | Keep task-native output formats and rewards. Do not report a total that the authors did not publish. |
| [Infinity-Parser2](https://arxiv.org/abs/2607.07836) | About 5M balanced SFT samples and about 220K RL samples across eight tasks | Weakness taxonomy from held-out errors, disjoint acquisition, controllable browser rendering with exact DOM labels, expert pseudo-label filtering, balanced SFT, then joint verifiable multi-task RL | Benchmark cases remain diagnostic only. Convert their weakness tags into new disjoint real or synthetic data rather than training on the benchmark. |

The scale lesson is not "collect millions first." Those systems build general OCR foundation models. For domain adaptation of an existing 0.9B model, start with a rights-cleared, adjudicated learning curve and expand only when held-out error reduction has not saturated.

If repeated domain recognition errors remain dominant, the first tuning experiment is GLM-OCR 0.9B LoRA SFT. Its official [fine-tuning guide](https://github.com/zai-org/GLM-OCR/blob/main/examples/finetune/README.md) states that LoRA can run with at least 8 GB while full tuning is around the A10G's entire 24 GB budget. Start with BF16 LoRA and rank 8. The published configuration targets `all`, so log every trainable parameter name and verify the intended vision and language modules instead of assuming the vision tower is frozen. QLoRA is unnecessary for a 0.9B first experiment. Full tuning and RL stay deferred.

PaddleOCR-VL-1.6 is the second SFT candidate if it wins the untuned baseline. The [official guide](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/pipeline_usage/PaddleOCR-VL.en.md) supports VLM SFT but not tuning its layout or ranking stages. A 30B model is not the first tuning target: Muse Glimmer official BF16 LoRA guidance exceeds one A10G, and Qwen3-32B is text-only.

Recent evidence supports hard-sample selection and SFT before RL. PaddleOCR-VL-1.6 reports most of its improvement before the final RL step, while [GLM-OCR](https://www.alphaxiv.org/abs/2603.10910), [FireRed-OCR](https://www.alphaxiv.org/abs/2603.01840), and [OCRVerse](https://www.alphaxiv.org/abs/2601.21639) use task-specific format and fidelity rewards only after supervised adaptation. These are author-reported results, not measurements here.

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
6. Keep immutable truth revisions and separate model-error tags. Derive GLM, Paddle, Markdown, HTML, and task-specific training records from the canonical truth instead of editing separate labels.
7. Measure character agreement, block IoU, table-cell agreement, relation agreement, unreadable agreement, and adjudication rate by source family before releasing a training slice.

- Group every page, crop, perturbation, and derivative from the same source document into one split.
- Hold out customer, provider, time, template, and document family where relevant.
- Never train on the 328 ClinOCR evaluation pages or general benchmark evaluation annotations.
- Use human adjudication for high-risk fields and unresolved model disagreement.
- Render tables and formulas back for deterministic structure checks.
- Start learning curves at 100, 250, 500, 1,000, and 2,000 adjudicated examples, with document-group holdouts and at least three seeds.
- Compare random sampling with uncertainty plus diversity sampling. Stop labeling when two successive increments improve the target by less than 0.5 absolute points.

Proceed to LoRA SFT only when a recurring recognition error is at least 20 percent of material errors, replacing only that recognition with human truth recovers at least 2 end-to-end points or 5 critical-slice points, cheaper pipeline fixes do not recover the gap, and at least 250 rights-cleared adjudicated examples cover multiple document families.

Adopt the adapter only when its paired 95 percent interval excludes zero, unsupported insertions do not rise, no untouched task regresses by more than 0.5 points, and latency and memory stay inside the operating envelope. Do not tune when errors are mainly layout, association, reconstruction, ambiguous truth, service failures, or isolated document families.

## 15. Independent audit and remaining evidence gaps

The independent correctness review reproduced and repaired these material defects before the final commit:

- multi-frame TIFFs now expand into ordered pages instead of silently losing frames;
- clean zero-region reader results now create an explicit full-page repair target, while reader exceptions remain failures;
- table regions cannot be repaired into flat text, and every cell span is validated before rowspan-aware rectangularity abstention;
- primary and verifier roles must return different actual model identifiers;
- stored benchmark edit counts cannot manufacture gains because paired CER and WER are recomputed from raw text;
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
- the complete GLM-OCR SDK caller boundary and official version-matched OmniDocBench evaluator remain unmeasured;
- external-upload privacy is a documented operating boundary, not a technical PHI detector;
- plausible wrong text, missed layout regions, controls, and cross-page relationships exceed the current deterministic router;
- two-model literal agreement is supporting evidence, not ground truth, and still requires held-out false-accept and risk-coverage measurement;
- point GPU memory is not peak memory, and Paddle unwarping currently exceeds the planned headroom target.
