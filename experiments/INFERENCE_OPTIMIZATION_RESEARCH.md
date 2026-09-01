# Lossless inference optimization research

Date: 2026-09-01

## Decision

Optimize the measured cascade before replacing models. The highest-value speed fix is to stop sending obvious non-table screen captures through the complete table path, while preserving the OCR, orientation, table, control, and evidence outputs whenever their routes are justified. Keep Nemotron at batch size 1 until its batch-dependent output changes are explained. Do not use quantization, reduced resolution, `skip_relational`, `detector_only`, TensorRT, CUDA graphs, or `torch.compile` in the default path until exact structured outputs pass the fixed end-to-end panels.

This is a research plan, not a performance claim. Source-reported speedups, existing local measurements, new manual observations, and untested hypotheses are separated below.

## Current evidence

### Local measurements

| Evidence | Result | Interpretation |
| --- | --- | --- |
| Generated clinical hard panel, 41 pages | 41/41 operational success, 0/41 manually complete, p50 1.227 s, p95 4.245 s, max 14.632 s | Schema success is not extraction correctness. The maximum was the first cold request. |
| Tiny-text route on the same panel | 3,780 crop alternatives, 235 unsupported tile-only candidates, 0 promoted | Tiling preserves the primary output and exposes missed evidence, but has not yet recovered text safely. |
| Orientation classifier | 41/41 private orientations correct; 15.243 ms/page preparation plus classifier | Keep the docTR guard. It is not the latency bottleneck. |
| Public rotated panel | 56/56 pages covered; selected the minimum-CER cached view on 56/56 | This is a diagnostic selection result. Aggregate guarded latency is unavailable. |
| Table detection, 60 public tables | p50 38.212 ms, p95 56.559 ms; precision 0.983607, recall 1.0 | TATR detection itself is fast on true tables. Unnecessary table routing and downstream structure/challenger work remain the likely waste. |
| Table structure, 60 public tables | GriTS Top 0.990225, Con 0.991262, Loc 0.984719, cell exact 0.977380 | Any routing optimization must preserve this table quality. |
| Geometric controls, 21 pages | p50 11.82 ms, p95 25.08 ms | Control detection is cheap, but dense rules still cause false positives and must remain review-only. |
| Nemotron native batching, eight pages | Batch 1 median 10.937632 pages/s; batch 8 median 12.098005 pages/s, but exact equality was only 0.90625 to 0.9375 within a batch size | The throughput gain is not lossless. Batch size 1 remains the reference. |

Sources: [private hard-case evaluation](PRIVATE_HARD_CASE_EVALUATION.md), [orientation evaluation](ORIENTATION_HARD_CASE_EVALUATION.md), [PubTables detection](PUBTABLES_DETECTION_EVALUATION.md), [checkbox specialist](CHECKBOX_SPECIALIST_EVALUATION.md), and [workspace research plan](../OCR_PIPELINE_RESEARCH_AND_PLAN.md#13-lossless-inference-optimization-plan).

### New manual observations

These are visual findings from the user-supplied images, not benchmark scores:

- A clinical form required 22.329 s and still corrupted or omitted handwritten names, address, date, phone, diagnosis, checked services, signature, and small footer text.
- Two application screenshots were each enclosed by one large false table region. Sidebar and footer fragments were emitted as cells, the central conversation was partly omitted, and text from separate panes was interleaved.
- The application screenshots are text-layout adversarial cases, not tables. They expose a routing error that can harm both accuracy and latency.
- Faint text, tiny text, handwriting, dense ruled forms, checked controls, rotated pages, and real tables remain separate failure classes. A faster wrong route does not count as an improvement.

The observed 22.329 s is a single user-visible request, so it must be reproduced with per-stage timing before being used as a latency baseline.

### Verified current flow

The demo keeps one Nemotron pipeline resident and uses batch size 1. A tiny-text page first receives a full-page Nemotron call and then three sequential tile calls. Orientation may evaluate more than one view. TATR detection and structure are serialized, and low-confidence table crops can trigger raw and Sauvola Tesseract challengers. This explains how a false full-page table route can multiply work even when no table exists.

Relevant code: [`create_verified_app`](serve_gpu_demo.py), [`TiledReader`](../src/ocr_pipeline/preprocessing.py), [`NemotronOCRV2Reader`](../src/ocr_pipeline/providers.py), [`OrientationReader`](../src/ocr_pipeline/orientation.py), and [`TatrTableStage`](../src/ocr_pipeline/tables.py).

## Source-reported options

These are upstream claims, not local OCR results:

| Option | Upstream statement | Local decision |
| --- | --- | --- |
| Nemotron `detector_only` | NVIDIA reports about 37% less GPU memory and 20% faster inference | Do not use for primary OCR because recognition is required. It may be profiled only as a table-routing signal. |
| Nemotron `skip_relational` | NVIDIA reports about 35% less GPU memory and 8% faster inference | Do not use globally because reading order is part of the output contract. Test only on isolated crops where relational output is unused. |
| Nemotron `verbose_post=True` | NVIDIA documents CUDA-synchronized phase profiling | Use first to identify detector, recognizer, relational, and postprocessing cost. Profiling output must not alter results. |
| `torch.compile(mode="reduce-overhead")` | PyTorch says the mode can reduce Python overhead with CUDA graphs and can use more memory | Test only after fixed-shape buckets are identified. It is not automatically lossless. |
| CUDA graphs | PyTorch says graphs reduce CPU and kernel-launch overhead but require graph-safe, static work and fixed memory addresses | Restrict to stable shapes and control flow. Dynamic OCR pages and crop counts are not a direct fit. |
| Automatic mixed precision | PyTorch documents lower-precision inference through autocast | Treat as approximate until every output field matches. The current BF16 reference must not silently change. |
| TensorRT dynamic shapes | NVIDIA requires explicit minimum, optimum, and maximum shape profiles | Last-resort experiment after export compatibility and dominant GPU phases are proven. Do not rewrite the service speculatively. |

Primary sources: [NVIDIA Nemotron OCR v2 model card](https://huggingface.co/nvidia/nemotron-ocr-v2), [NVIDIA Open Model License](https://huggingface.co/nvidia/nemotron-ocr-v2/blob/main/LICENSE), [PyTorch `torch.compile`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html), [PyTorch CUDA semantics](https://docs.pytorch.org/docs/main/notes/cuda.html), [PyTorch AMP](https://docs.pytorch.org/docs/stable/accelerator/amp.html), and [TensorRT performance guidance](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/benchmarking.html).

## Ranked reversible experiments

### 1. Reproduce and profile the two false-table screenshots

Add stage timers for decode, orientation, baseline OCR, each tile, TATR detection, each structure crop, each challenger, controls, risk, serialization, and total request time. Use NVIDIA phase profiling inside Nemotron for this experiment only.

Acceptance:

- The two screen captures reproduce the false-table region and report complete stage timings.
- Timers add less than 2% warm p50 overhead when disabled.
- Text, boxes, reading order, alternatives, route decisions, failures, and review state are unchanged.

Rollback: remove the timing instrumentation if disabled-mode overhead exceeds 2% or any structured output changes.

### 2. Add a conservative table-route precheck

Hypothesis: skipping TATR on obvious application screenshots will remove the large false table and avoid structure and challenger calls. The precheck must abstain when uncertain. Candidate evidence includes repeated ruled lines, aligned cell boundaries, table-like whitespace intersections, and a minimum set of row and column separators. A screen frame, sidebar, or large rectangular panel is not sufficient.

Acceptance:

- Zero table regions on the two supplied application screenshots.
- No omitted central text attributable to the table stage.
- No regression on all 60 public table detection and structure cases or the targeted 187-cell financial-table panel.
- At least 10% lower warm p95 on the two screen captures, with no increase in failures.

Rollback: route every page through the current TATR stage if any known table is skipped or any table metric declines.

### 3. Overlap independent CPU work with one GPU queue

Hypothesis: page decode, lossless crop creation, docTR preparation, geometric control proposal, and Tesseract challenger preparation can overlap without concurrent calls to the same GPU model. Preserve one bounded Nemotron queue and one bounded TATR queue. Complete label association and final ordering only after primary OCR returns.

Acceptance:

- Exact structured equality across three repeated runs of the 41-page hard panel, 56 rotated pages, 60 public tables, 21 control pages, and the supplied adversarial images.
- At least 10% better warm p95 or throughput at the caller boundary.
- No increase in failure or review-routing counts and peak GPU memory below 90% of the A10G.

Rollback: disable overlap if outputs, ordering, failures, or memory limits change.

### 4. Diagnose tile and page batching before using it

The current three tiles are sequential, but the existing Nemotron batching experiment changed outputs. First isolate whether changes come from padding, request order, merge behavior, nondeterministic kernels, or postprocessing. Keep the baseline page and tile evidence identifiers stable.

Acceptance:

- Exact structured equality across three runs, request permutations, batch sizes 1, 2, 4, and supported concurrency schedules.
- At least 10% throughput or p95 improvement with no new failures and memory below 90%.

Rollback: retain sequential batch size 1 if any character, box, order, confidence, alternative, or failure differs.

### 5. Compile only a measured fixed-shape hotspot

After profiling, test `torch.compile` and then CUDA graphs on one stable detector or recognizer shape bucket. Record compilation time separately from warm latency. Do not graph dynamic page routing or variable table counts.

Acceptance:

- Exact structured equality on all fixed panels and three repeats.
- No graph breaks or repeated recompilation in the measured bucket.
- At least 10% warm p95 improvement after amortizing compile cost over the declared service horizon.
- Peak GPU memory below 90%.

Rollback: return to eager BF16 if compilation changes outputs, increases failures, exceeds memory, or provides less than 10% benefit.

### 6. Test quality-changing fast paths only as isolated ablations

`skip_relational`, `detector_only`, AMP changes, TensorRT, FP8, INT8, lower resolution, JPEG recompression, and visual token pruning are not lossless by design or evidence. Evaluate them separately only after the preceding experiments.

Acceptance: full quality evaluation by capability, including CER, WER, missed and unsupported text, reading order, tables, forms, controls, handwriting, coverage, abstention, latency, and memory. Do not promote an ablation on latency alone.

Rollback: keep it research-only if any clinically meaningful evidence or structural metric declines.

## End-to-end comparison rules

Each experiment must include failures and abstentions in its denominator and report cold and warm results separately.

Required panels:

- 328-page ClinOCR evaluation for transcription and coverage
- 56-page rotated ClinOCR subset
- 60-table PubTables detection and structure set
- 21-page control set and its clear-control subset
- targeted 187-cell financial-table set
- 41-page generated clinical hard panel
- the supplied handwritten clinical form and two application screenshots

Required output comparison:

- literal text and normalization
- bounding boxes and coordinate system
- reading order and parent-child relations
- table rows, columns, cells, spans, and content
- control state and label association
- provider, route, alternative, provenance, uncertainty, failure, and abstention fields

Required operational metrics:

- p50, p95, maximum, throughput, first-request latency, peak GPU memory, failures, and abstentions
- concurrency 1 first, then bounded concurrency 2 and 4 only while throughput improves
- stop increasing concurrency when throughput improves by less than 5%, p95 worsens by more than 10%, or failures increase

## Origin and license boundary

| Component | Origin and license evidence | Status |
| --- | --- | --- |
| Nemotron OCR v2 | NVIDIA model card; NVIDIA Open Model License; RegNetX detector and Transformer recognizer | Eligible local OCR candidate, subject to license review |
| Table Transformer | Microsoft repository; MIT license; DETR-based table detection and structure models | Eligible table specialist |
| docTR | Mindee code under Apache-2.0; torchvision MobileNetV3 orientation backbone | Code eligible; exact weight lineage and terms still require confirmation |
| Tesseract | Open-source project under Apache-2.0 | Eligible CPU challenger |
| OpenCV controls | OpenCV under Apache-2.0; no learned control model | Eligible geometric specialist |

Microsoft sources: [Table Transformer repository](https://github.com/microsoft/table-transformer) and [MIT license](https://github.com/microsoft/table-transformer/blob/main/LICENSE). Chinese-origin models and backbones remain excluded from the deployable path. Research ideas may be studied without adopting their weights or backbones.

## Recommended order

1. Reproduce the 22.329 s request and collect per-stage and Nemotron phase timings.
2. Repair false-table routing and recheck the two application screenshots manually.
3. Overlap independent CPU preparation with the serialized GPU queues.
4. Diagnose Nemotron nondeterminism before any tile or page batching.
5. Compile one proven fixed-shape hotspot.
6. Consider TensorRT or quality-changing fast paths only if the measured bottleneck remains and full evaluation justifies them.

The fastest credible path is routing less unnecessary work, not making every model approximate. Handwriting, faint text, tiny text, dense controls, and tables still need capability-specific quality improvements, and no inference optimization may hide those failures behind a lower latency number.
