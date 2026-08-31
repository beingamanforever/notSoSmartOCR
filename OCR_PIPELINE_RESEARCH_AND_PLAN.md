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

No model ranking or cost claim is treated as measured here until the experiments in this plan run.

## 2. Research conclusion

The smallest strong first system is:

1. Preserve native PDF text when it passes deterministic reliability checks.
2. Use the full [PaddleOCR-VL-1.6](https://github.com/PaddlePaddle/PaddleOCR) pipeline as the primary local baseline.
3. Retain PP-DocLayoutV3 regions, polygons, reading order, and nearby label context.
4. Recognize regions with PaddleOCR-VL-1.6 and merge them deterministically.
5. Compare [GLM-OCR](https://github.com/zai-org/GLM-OCR) by changing only the recognizer while keeping the same layout stage.
6. Route only observable failures or ambiguities to OpenRouter model [`qwen/qwen3.8-flash`](https://openrouter.ai/qwen/qwen3.8-flash).
7. Validate the merged result, retain evidence for every extracted fact, and abstain instead of inventing unreadable content.

The Paddle documentation explicitly distinguishes the complete pipeline from direct VLM inference. The complete pipeline adds layout detection, region-level recognition, and structured merge. A VLM-only run is therefore an ablation, not the baseline. See the official [PaddleOCR-VL pipeline guide](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html) and the [PaddleOCR-VL-1.6 paper](https://www.alphaxiv.org/abs/2606.03264).

Why this is the first baseline:

- PaddleOCR-VL-1.6 is a compact 0.9B local model with an Apache-2.0 code and model path.
- PP-DocLayoutV3 supplies a reusable structure boundary instead of forcing the hosted model to rediscover page geometry.
- GLM-OCR is also compact and uses the same layout family, enabling a controlled recognizer comparison.
- Paddle and GLM agreement is not independent layout evidence because both complete pipelines use PP-DocLayoutV3. Layout failures need separate checks or a genuinely different detector.
- Hosted Qwen is valuable for difficult regions and document-level context, but its OCR accuracy, page cost, structured validity, provider behavior, and privacy suitability are unverified here.
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

## 4. Proposed pipeline

```text
PDF or images
  -> ordered pages
  -> native PDF text check
  -> PP-DocLayoutV3 structure and reading order
  -> PaddleOCR-VL-1.6 region recognition
  -> deterministic merge and validators
  -> route selected regions or pages to Qwen3.8-Flash
  -> schema validation and evidence-preserving merge
  -> document-level continuation and hierarchy repair
  -> structured JSON plus Markdown or task adapter
```

Routing uses observable signals only:

- missing text inside a detected content region;
- low or inconsistent OCR confidence;
- invalid table geometry, spans, or row alignment;
- ambiguous checkbox state or label association;
- conflicting repeated identifiers or values;
- failed layout, reading order, or cross-page continuation;
- invalid provider schema or parser output;
- remaining document budget.

Region crops include nearby labels and page coordinates. Full-page escalation is reserved for failed layout, reading order, multi-region relationships, or cross-page reconstruction. Same-model self-critique is not independent evidence. Prefer deterministic validation, evidence consistency, and render-back checks for tables or formulas.

## 5. OpenRouter integration

The required hosted model is `qwen/qwen3.8-flash`. The official OpenRouter model page currently describes multimodal input, structured output support, a 1M-token context, and pricing of USD 0.15 per million input tokens and USD 0.47 per million output tokens. These are current list prices, not measured page costs.

Additional user-selected comparators have distinct roles:

| Model | Verified role | Current list price | Experiment decision |
| --- | --- | --- | --- |
| [`qwen/qwen3.8-flash`](https://openrouter.ai/qwen/qwen3.8-flash) | Multimodal text, image, and video input with structured output | USD 0.15 input and USD 0.47 output per million tokens | Primary hosted OCR and document-understanding escalator. |
| [`meta/muse-glimmer-30b`](https://openrouter.ai/meta/muse-glimmer-30b) | Multimodal text and image input with structured output | USD 0.30 input and USD 1.10 output per million tokens | Higher-cost visual challenger on the same fixed public pages. It replaces Qwen only if the paired quality gain satisfies the quality-first decision rules. |
| [`qwen/qwen3-32b`](https://openrouter.ai/qwen/qwen3-32b) | Dense causal language model with structured output; the current model page does not advertise image input | USD 0.08 input and USD 0.28 output per million tokens | Text-only schema extraction or evidence-preserving repair after OCR. It never receives a raw image and cannot be ranked as an image OCR engine unless endpoint capabilities change and are reverified. |

The model slug is configuration, not a new adapter. Visual and text-only models use separate request builders so an image is never silently dropped for a text-only model.

The adapter will:

- read `OPENROUTER_API_KEY` only from the environment;
- send image data through the official multimodal chat-completions format;
- require a strict JSON Schema response;
- request providers that support required parameters;
- request no data collection and zero-data-retention routing when available;
- use bounded retries and a capped `Retry-After` delay;
- record provider, status, prompt tokens, completion tokens, actual returned cost, and latency;
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
| [OmniDocBench](https://github.com/opendatalab/OmniDocBench) and [paper](https://www.alphaxiv.org/abs/2412.07626) | Official prose says 1,651 pages; current HF viewer count differs | General text, layout, formulas, tables, and reading order | Research-only terms and revision-dependent scoring. Count downloaded ground-truth pages before reporting a denominator. |
| [MPDocBench-Parse](https://github.com/Tongyi-Zhiwen/Qwen-Doc/tree/main/MPDocBench) and [paper](https://www.alphaxiv.org/abs/2605.22100) | Public release: 420 PDFs and 3,135 pages; paper: 433 documents and 3,246 pages before removals | Multi-page hierarchy, continuation, reading order, figures, tables, and formulas | Research terms, no train split, additional formula dependencies. Use released counts, not pre-release paper counts. |
| [olmOCR-Bench](https://github.com/allenai/olmocr) | More than 7,000 unit tests over about 1,400 documents in the project description | Broad parser regression and failure taxonomy | Primarily English and unit-test oriented; benchmark artifact terms need confirmation before redistribution. |

### 6.2 Component benchmarks

| Benchmark | Verified scale | Capability and metrics | Acquisition decision |
| --- | --- | --- | --- |
| [DocLayNet](https://github.com/DS4SD/DocLayNet) | 80,863 pages, with 69,375 train, 6,489 validation, and 4,999 test pages; 11 layout classes | Layout precision, recall, F1, and COCO mAP | Use test or a fixed test slice later. It is large and has no end-to-end transcription truth. |
| [FUNSD](https://guillaumejaume.github.io/FUNSD/) | 199 noisy forms, split 149 train and 50 test | Word, entity, and key-value relation evaluation | Download first. Small and useful, but non-commercial research and education terms apply. |
| [CheckboxQA](https://github.com/Snowflake-Labs/CheckboxQA) | 88 English multi-page documents and 579 QA pairs | Official ANLS* plus derived control-state and association analysis where annotations allow | Evaluation-only and CC BY-NC under current repository terms. Do not claim it is direct checkbox-detection ground truth. |
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

Report paired case differences and 95 percent bootstrap intervals where the metric supports pairing. Never compare author scores across different benchmark revisions as if they were a controlled experiment.

## 8. Experiments and decision rules

### 8.1 First controlled comparison

Run identical pages through:

1. Tesseract CPU floor.
2. Full PaddleOCR-VL-1.6 BF16 pipeline.
3. PaddleOCR-VL-1.6 VLM-only ablation.
4. PP-DocLayoutV3 plus GLM-OCR BF16.
5. Full local winner plus Qwen escalation.
6. Qwen on every eligible public page as a cost and quality ceiling.

Run BF16 before testing one quantized configuration. A quantized model is accepted only if its paired quality loss is within the locked task bounds and its operational gain is real.

Required ablations:

- full pipeline versus VLM-only;
- full page versus layout crops;
- crops versus crops with neighboring label context;
- Paddle recognizer versus GLM recognizer with the same layout stage;
- native PDF text preservation on versus off;
- Qwen escalation off versus on;
- BF16 versus one selected quantization;
- page-level versus document-level reconstruction.

### 8.2 Routing experiment

Reuse identical cached stage outputs to compare:

1. local-only;
2. Qwen on every eligible page;
3. gold-oracle routing for research diagnosis only;
4. automatic cost-aware routing.

Stop routing work if oracle escalation improves the primary metric by less than 1 absolute point and recovers fewer than 10 percent of local errors.

Accept automatic routing only if it:

- captures at least 90 percent of oracle-recoverable gain;
- costs at most 40 percent of the all-Qwen policy;
- does not increase unsupported critical facts;
- regresses neither control-state F1 nor table-position accuracy by more than 0.5 points.

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

### Phase 0: Pipeline contract and end-to-end spine

Deliver ordered image and PDF ingestion, canonical evidence-linked JSON, an explicit local reader, an explicit escalator interface, routing states, and a CLI. Test the real CLI over a multi-page fixture. Page order, geometry, abstentions, and failures must survive serialization.

Exit condition: the complete local fixture path passes from user input to serialized result. Do not start model integration before this works.

### Phase 1: Local floor and public transcription

Add the Tesseract floor adapter and public evaluation harness. Download ClinOCR-Bench and FUNSD from official sources. Attempt every eligible case and retain failures in denominators.

Exit condition: reproducible caller-boundary metrics and failure accounting for the full selected splits.

### Phase 2: GPU structure-aware baseline

On the authorized A10G, install the full PaddleOCR-VL-1.6 pipeline in an isolated environment stored on the NVMe volume. Keep model caches on NVMe because the root filesystem has limited free space. Sync source to `/home/ubuntu/aman/notSoSmartOCR` without deleting existing files. Run BF16 first and record peak VRAM, throughput, stage latency, and output validity.

Exit condition: full Paddle pipeline runs end to end over fixtures and the public first suite. Then run the VLM-only ablation and GLM recognizer comparison in separate environments.

### Phase 3: Hosted Qwen escalator

Implement strict structured output, provider constraints, retries, usage accounting, and route reasons. Use a rotated environment key. Test only public images first.

Exit condition: at least 98 percent valid structured responses on the selected public sample, identifiable provider, complete usage and cost accounting, no credential exposure, and explicit failure results.

### Phase 4: Routing and multi-capability evaluation

Calibrate deterministic route signals on development data. Run local-only, all-Qwen, oracle, and automatic policies over the same outputs. Add CheckboxQA and selected OmniDocBench, DocLayNet, PubTables-1M, and cTDaR slices only after source terms and evaluators are confirmed.

Exit condition: routing meets the quality-first rules or is rejected. A rejected router leaves the strong local parser plus explicit review path intact.

### Phase 5: Existing benchmark bridge

Write a separate adapter into a new run directory and evaluate all 415 ready internal cases with the existing evaluator unchanged. Skip the 17 declared unavailable cases. External escalation stays disabled without PHI approval.

Exit condition: all expected identifiers are valid, coverage and missing units are reported, and the original 49-test suite still passes.

### Phase 6: Audit and independent verification

Review the final diff for correctness, leakage, privacy, unsupported fallbacks, over-engineering, license boundaries, and unused dependencies. Re-run formatting, linting, the full root end-to-end suite, the existing benchmark tests, public benchmarks, and reproducible remote caller-boundary commands.

Exit condition: every material review finding is repaired and affected checks rerun. Any unmeasured or blocked claim remains labeled.

## 10. Initial file layout

Start small and split only after a file has distinct responsibilities:

```text
src/ocr_pipeline/
  contracts.py
  pipeline.py
  providers.py
  cli.py
experiments/
  public_benchmark.py
  internal_benchmark.py
data/
  README.md
tests/
  test_end_to_end.py
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

Official repositories, documentation, dataset cards, and model cards are linked in the relevant sections above. Repository-reported scores remain hypotheses until reproduced by this workspace.

## 13. Lossless inference optimization plan

Optimization is split into two tracks. A candidate stays in the lossless track only when every deterministic output matches the BF16 eager baseline across repeated runs. Any changed text, class, reading order, evidence link, cell, control state, or cross-page relation moves the candidate into the approximate track and requires full quality evaluation.

Safe by construction:

- keep layout and recognition services resident;
- render each page once and reuse the same lossless pixel buffer for crops;
- preserve stable document, page, and block identifiers through bounded concurrency;
- batch independent layout pages and recognition crops, then restore original order deterministically;
- apply queue backpressure instead of unlimited requests;
- use a validated native PDF text path with visual fallback;
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

Ranked A10G experiment:

1. One render per page and validated native-text bypass.
2. Layout page batches and bounded crop concurrency with stable merge order.
3. Resident PaddleOCR-VL-1.6 BF16 service with vLLM controls for GPU utilization, sequence count, batched tokens, and client concurrency.
4. Paddle vLLM versus FastDeploy using identical weights and decoding.
5. GLM-OCR vLLM versus SGLang using identical weights and decoding.
6. CUDA graph, compilation, attention, transfer, and verified speculative ablations one at a time.
7. Separate approximate experiments only after the BF16 baseline is complete.

Adopt a lossless optimization only if exact outputs match across three runs, failures do not increase, peak A10G memory stays below 90 percent, and throughput or p95 improves by at least 10 percent. Stop raising concurrency when throughput improves by less than 5 percent, p95 worsens by more than 10 percent, or failures increase.

Primary sources: [PaddleOCR-VL deployment](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/pipeline_usage/PaddleOCR-VL.en.md), [Paddle high-performance configuration](https://github.com/PaddlePaddle/PaddleOCR/blob/main/deploy/paddleocr_vl_docker/pipeline_config_fastdeploy.yaml), [vLLM optimization](https://docs.vllm.ai/en/stable/configuration/optimization/), [GLM-OCR serving](https://github.com/zai-org/GLM-OCR), [FlashAttention](https://www.alphaxiv.org/abs/2205.14135), [FlashAttention-2](https://www.alphaxiv.org/abs/2307.08691), [PagedAttention](https://www.alphaxiv.org/abs/2309.06180), and [DFlash](https://www.alphaxiv.org/abs/2602.06036).

## 14. Fine-tuning and annotation decision

Do not fine-tune before untuned Paddle, GLM, hosted escalation, and routing results identify the remaining bottleneck. Recognition tuning cannot repair layout misses, wrong reading order, checkbox-label association, or cross-page joins.

If repeated domain recognition errors remain dominant, the first tuning experiment is GLM-OCR 0.9B LoRA SFT. Its official [fine-tuning guide](https://github.com/zai-org/GLM-OCR/blob/main/examples/finetune/README.md) states that LoRA can run with at least 8 GB while full tuning is around the A10G's entire 24 GB budget. Start with BF16 LoRA, rank 8, and the vision tower unchanged. QLoRA is unnecessary for a 0.9B first experiment. Full tuning and RL stay deferred.

PaddleOCR-VL-1.6 is the second SFT candidate if it wins the untuned baseline. The [official guide](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/pipeline_usage/PaddleOCR-VL.en.md) supports VLM SFT but not tuning its layout or ranking stages. A 30B model is not the first tuning target: Muse Glimmer official BF16 LoRA guidance exceeds one A10G, and Qwen3-32B is text-only.

Recent evidence supports hard-sample selection and SFT before RL. PaddleOCR-VL-1.6 reports most of its improvement before the final RL step, while [GLM-OCR](https://www.alphaxiv.org/abs/2603.10910), [FireRed-OCR](https://www.alphaxiv.org/abs/2603.01840), and [OCRVerse](https://www.alphaxiv.org/abs/2601.21639) use task-specific format and fidelity rewards only after supervised adaptation. These are author-reported results, not measurements here.

### 14.1 Canonical label record

Keep one evidence-linked annotation record and derive model-specific training examples from it. At minimum, record:

- document, source family, rights, page ID, image size, language, degradation, rotation, and layout type;
- blocks with stable ID, type, polygon, reading index, verbatim text, separate normalized text, legibility, and parent or caption links;
- text status from `legible`, `partly_legible`, `unreadable`, or `ambiguous` so annotators never guess;
- table IDs, rows, columns, cells, blank cells, row and column spans, headers, geometry, and continuation links;
- form key and value block IDs, normalized value, relation, and field status;
- control state from `checked`, `unchecked`, `indeterminate`, or `unreadable`, plus label and group links;
- reading-next, heading-parent, figure-caption, paragraph-continuation, and table-continuation relations;
- adjudication status, annotator uncertainty, and evidence block IDs.

Store model error analysis separately from truth using omission, unsupported insertion, substitution, structure, association, reading-order, and wrong-abstention labels.

Use region examples for recognition failures and full-page or multi-page examples only when global structure is required. Hard examples should vary one digit, date, unit, negation, state, faint entry, span, or page continuation while preserving the rest of the document.

### 14.2 Selection, split, and quality control

- Group every page, crop, perturbation, and derivative from the same source document into one split.
- Hold out customer, provider, time, template, and document family where relevant.
- Never train on the 328 ClinOCR evaluation pages or general benchmark evaluation annotations.
- Use human adjudication for high-risk fields and unresolved model disagreement.
- Render tables and formulas back for deterministic structure checks.
- Start learning curves at 100, 250, 500, 1,000, and 2,000 adjudicated examples, with document-group holdouts and at least three seeds.
- Compare random sampling with uncertainty plus diversity sampling. Stop labeling when two successive increments improve the target by less than 0.5 absolute points.

Proceed to LoRA SFT only when a recurring recognition error is at least 20 percent of material errors, replacing only that recognition with human truth recovers at least 2 end-to-end points or 5 critical-slice points, cheaper pipeline fixes do not recover the gap, and at least 250 rights-cleared adjudicated examples cover multiple document families.

Adopt the adapter only when its paired 95 percent interval excludes zero, unsupported insertions do not rise, no untouched task regresses by more than 0.5 points, and latency and memory stay inside the operating envelope. Do not tune when errors are mainly layout, association, reconstruction, ambiguous truth, service failures, or isolated document families.
