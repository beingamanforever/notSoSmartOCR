# Not So Smart OCR

**Evidence-linked OCR for clinical and structured documents.**
Text stays tied to the pixels it came from, and nothing in the pipeline is allowed to overwrite a reading it did not produce.

[Why evidence-linked](#why-evidence-linked) ·
[Quick start](#quick-start) ·
[How it works](#how-it-works) ·
[Results](#results) ·
[Limits](#what-is-and-is-not-supported) ·
[Technical report](artifacts/ocr-technical-report.md)

> **Research status.** This is a research snapshot, not a product. It has not
> demonstrated superiority over a frozen frontier baseline, and it is not safe
> for unattended clinical use. Every page of the private hard panel routes to
> human review. Measured gains, negative results, and open gaps are reported in
> the same denominator.

![Evidence-linked OCR architecture](artifacts/research/figures/pipeline_architecture.svg)

*The accented block is the only contract every component shares. Specialists may add readings to it; none of them may delete one.*

## Why evidence-linked

A typical document parser takes a page and returns Markdown.
That output has no seam at which a reader can tell recovered text from invented text.

When a faint gray band at the top of a scan is silently dropped, the Markdown looks complete.
When a handwritten drug name is read as the printed label a few millimeters away, the Markdown looks confident.
Neither failure is visible in the artifact a downstream system consumes.

This pipeline makes the region, not the string, the unit of output.

```json
{
  "id": "p1-word-1",
  "kind": "word",
  "text": "Sertraline",
  "confidence": 0.91,
  "bounding_box": {"left": 314, "top": 902, "right": 512, "bottom": 926},
  "reading_order": 1,
  "provider": "tesseract",
  "resolution": "resolved",
  "alternatives": [
    {"text": "Sertralino", "confidence": 0.74, "provider": "tesseract-routed-tiled"}
  ],
  "structure": null
}
```

Four fields carry the design.
`provider` makes every reading attributable, so a mixed page can be audited by component.
`alternatives` is where rejected readings go, which is what makes a non-destructive challenge possible at all.
`resolution` is the only way a stage can say "I could not settle this", and it is what routing reads.
`structure` holds whatever a specialist needs to add: a cell grid, a control state and its label links, or the reasons a coverage risk was raised.

Rendering is a projection of these records, so every rendered character traces back to the region that produced it.
Failures are kept in the result rather than smoothed away.

## Quick start

The lean path needs Python, Pillow, Tesseract, and Poppler, and nothing else.
The repository does not ship a package manifest.
GPU specialists run in their own model-specific environments and are not part of this install.

System tools on macOS:

```bash
brew install tesseract poppler
```

On Debian or Ubuntu:

```bash
sudo apt-get install tesseract-ocr poppler-utils
```

The Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install Pillow fastapi uvicorn python-multipart
```

Extract evidence-linked JSON from an image or PDF:

```bash
PYTHONPATH=src python -m ocr_pipeline.cli page.png --reader tesseract --output result.json
```

Start the local review workbench, then open `http://127.0.0.1:8000`:

```bash
PYTHONPATH=src python -m ocr_pipeline.demo
```

### What each entry point actually composes

The recovery guards and specialist stages are opt-in.
Reaching for the CLI and expecting the full stack is the easiest way to misread a result, so the composition is spelled out here.

| Entry point | Reader | Stages |
| --- | --- | --- |
| `ocr_pipeline.cli` | one bare reader, chosen by `--reader` | none |
| `ocr_pipeline.demo` | reduced workbench with `RoutedTesseractReader` as primary OCR | none |
| `experiments/serve_gpu_demo.py` | page-frame and orientation guards, over selective wide-band recovery, over tiny-text tiling, over Nemotron OCR v2 at word merge level and batch size one | tables with two Tesseract crop challengers, controls, then evidence risk; optional Phi-4 is manual reread only |

The verified GPU composition uses three of the four guards described under [gated recovery](#gated-recovery).
Its primary recognizer is Nemotron OCR v2 at batch size one.
Tesseract is limited to orientation OSD, selective wide-band fallback and confirmation, and raw and Sauvola table-crop challengers.
Page-frame isolation wraps the full reader, and evidence risk runs last so that it sees what every other stage produced.
The always-visible banner and `GET /api/composition` report this static configuration before a request; per-page stage execution in the process response reports what actually ran.

The GPU composition needs local model paths.
Run it with `--help` for the full list; the required arguments are the Nemotron model directory, the pinned Table Transformer source and its detection and structure checkpoints, and the Tesseract executable.

```bash
PYTHONPATH=src python experiments/serve_gpu_demo.py --help
```

The optional Phi-4 handwriting adapter runs as a separate loopback service in its own environment:

```bash
PYTHONPATH=src /path/to/phi4/python \
  experiments/serve_phi4_handwriting.py \
  --adapter /private/path/vision_decoder_lora-runtime.pt \
  --torch-site /path/to/torch/site-packages --warmup

PYTHONPATH=src python experiments/serve_gpu_demo.py \
  <required local model arguments> \
  --phi4-handwriting-url http://127.0.0.1:8083
```

Both processes bind to loopback, crop bytes stay in memory, and the service is contacted only when a reviewer selects a region and asks for a handwriting reread.

## How it works

### The contract

`process_document(source, reader, stages=(...))` is the whole entry point.
It prepares pages, reads them, runs the ordered stages, restores coordinates, computes a route, and returns a `DocumentResult` at schema version 2.

| Layer | Contract | Current implementations |
| --- | --- | --- |
| Reader | `read(image_path, page_number) -> list[TextRegion]` | Tesseract, Nemotron OCR v2, Nemotron Parse 2.0, Granite Docling, Ministral OCR |
| Region stage | `apply(image_path, page_number, regions) -> list[TextRegion]` | tables, controls, evidence risk |
| Manual crop reviewer | `review_region(image_path, page_number, region) -> TextRegion` | Phi-4 handwriting reread |
| Result | pages, regions, alternatives, provenance, failures, route | `DocumentResult`, `schema_version = 2` |
| Workbench | inspect the result without changing the model route | page overlays, evidence detail, stage timings, JSON and Markdown export |

A reader may additionally expose `read_batch`, `stage_view`, `restore_regions`, `page_needs_review`, and `coverage_assessment`.
The orientation guard uses four of those, which is how it can read a rotated page, show specialists the upright view, and still return boxes in the original page's coordinate space.

A stage receives a deep copy of the regions.
A stage that raises produces a recorded failure and the previous regions survive, so one broken specialist cannot destroy a page.

### Gated recovery

Preprocessing here is not a set of global options.
It is four composable readers, each wrapping the next, each firing on a measured condition and passing its input through unchanged when it does not fire.

![Gated reading stack](artifacts/research/figures/reading_stack.svg)

*Columns run left to right in the order a page meets the guards. Rows run from the condition that triggers a guard, through what it computes, to what it is permitted to change.*

Page preparation applies nothing globally: no deskewing, no denoising, no contrast normalization, and no resampling.
Every enhancement in the system is scoped to a bounded region, behind a gate, with the original retained.
Global thresholding was measured and rejected for exactly this reason, because it recovered selected faint text and regressed broader transcription.

The wide-band gate is the only place on the automatic path where a heavier recognizer runs, and it sees a band rather than a page.
Its replacement rule is deliberately hard to pass, and a candidate that fails it is attached as alternative evidence instead of being discarded.

### One resolution rule

Tile fusion, table cells, and the handwriting sidecar all resolve competing readings the same way.
Adoption demotes the incumbent to an alternative; it never deletes it.
An equally confident disagreement marks the region `conflicting`, which routes the page to review.

The full rule set, its thresholds, and its provenance handling are in [Section 4.6 of the technical report](artifacts/ocr-technical-report.md#46-one-resolution-rule).

### Verification and routing

Verification is deterministic and training-free, because a learned confidence model would need calibration data from the clinical distribution the system cannot yet read.

It checks impossible and truncated dates, empty content, invalid or out-of-page geometry, duplicate and out-of-range reading order, malformed and non-rectangular table markup, repeated-phrase decoder loops, and exact edit-distance disagreement between candidates.
The route is two-valued: `accept_local`, or `review` if the page accumulated any failure, the reader asked for review, or any region is unresolved, conflicting, or contradicted by one of its own alternatives.

### Reading a result

Start at `pages[i].route`.
On `review`, the reason is always locatable rather than inferred: check `pages[i].failure_ids` against the document's `failures`, then scan that page's regions for a `resolution` other than `resolved` and for any region whose `alternatives` disagree with its `text`.
A `coverage_risk` region names its own reasons and metrics in `structure`, and a rejected table detection is kept as a `table_candidate` region rather than dropped.
`pages[i].text.evidence_ids` lists exactly which regions produced the rendered string, so anything absent from that list is evidence the renderer deliberately withheld.

## Results

Failed, missing, invalid, and abstained cases stay in the denominator unless a row explicitly describes an already-localized crop subset.
Every component row is a single run of a fixed panel with no confidence interval, except the paired ClinOCR comparison.

### Public transcription

The strongest public result is selective Nemotron routing against the Tesseract floor, measured on all 328 ClinOCR evaluation pages.

| Reader | Coverage | CER | WER | Page latency |
| --- | ---: | ---: | ---: | ---: |
| Tesseract | 299/328 | 0.488293 | 0.611352 | p50 1.844 s, p95 5.990 s |
| Selective Nemotron | 282/328 | **0.331423** | **0.417869** | p50 1.862 s, p95 2.355 s |

The paired CER change is -0.156870 with a cluster 95% interval of [-0.185387, -0.113176], and the WER change is -0.193483 with [-0.223458, -0.151314].
Holm-adjusted sign-flip p-values are 0.0026 and 0.0004, over 16 template clusters and 10,000 bootstrap draws at seed 0.

This establishes improvement over the local floor and says nothing about a hosted frontier model.
The coverage regression is the honest cost: 17 fewer pages served, all of them still counted.

![Public OCR comparison](artifacts/research/figures/public_ocr_comparison.svg)

*Coverage is plotted beside the error rates so the improvement cannot hide the pages that went unserved.*

### Components

| Capability | Fixed panel | Measured result | Decision |
| --- | --- | --- | --- |
| Orientation | 56 rotated ClinOCR pages | 56/56 covered, 55/56 detector agreement, micro CER 0.104300 | Keep the guard |
| Table detection | 60 PubTables tables | precision 0.983607, recall 1.0, F1 0.991736 at IoU 0.50 and 0.75 | Keep the detector |
| Table structure | 60 PubTables tables | GriTS topology 0.990225, content 0.991262, cell exact 0.977380 | Keep it, broaden clinical validation |
| Clear controls | 52 controls on two pages | 52/52 detection, no false positives, label association F1 0.9903 | Keep as narrow review evidence |
| Table text fusion | 187 cells, two financial tables | Tesseract 163, Sauvola 164, Nemotron 182, all three fused 187 exact | Offline experiment, not the shipped challenger set |
| General forms | 50 FUNSD test forms | 50/50 covered, micro CER 0.565042, WER 0.790239 | Public floor, no structure claim |
| Layout | 61 OmniDocBench pages | mAP 0.342071, AP50 0.433447 | Research-only baseline |

### Handwriting

The adapter measures recognition on crops that have already been localized.
Localization is excluded, which makes this a narrower claim than the numbers suggest.

| Arm | Exact | CER | Hallucinated | Missed |
| --- | ---: | ---: | ---: | ---: |
| Stock Phi-4 | 32/88 | 0.401471 | 0.301471 | 0.020588 |
| Adapter v4 | **50/88** | **0.135294** | **0.044118** | 0.029412 |

Native exact match is 24/44, below the predeclared 36/44 adoption threshold, and 28 of 78 critical field-view pairs still contain substitutions.
That is why the adapter is exposed only behind a user selection and why its output is retained as alternative evidence rather than adopted.

The missing step is localization, and it is not solved.
An automatic handwriting classifier was built and rejected: 18 proposals, 8 matches, 9/44 box recall at warm p50 7.497 s.

![Handwriting adapter comparison](artifacts/research/figures/handwriting_adapter_comparison.svg)

*Each arm is 44 native and 44 scaled inferences over the same fixed C14 panel. One run per arm, no confidence intervals.*

### Hard documents

This is the gap between component success and useful document parsing.

| Track | Denominator | Result |
| --- | ---: | --- |
| Frozen private hard route | 44 pages | 44 operational successes, **0 manually complete**, 44 review routes; p50 0.895 s, p95 4.178 s |
| Challenging-formats aggregate | 169 annotated cases | 167 covered, 2 orientation failures; p50 4.504 s, p95 12.255 s |

On the 169-case panel, handwriting exact recovery is 35/228 legible spans, the control specialist safely matches 21 of 1,521 annotations, and table presence F1 is 0.862191 while row-count accuracy is 0.017699 and column-count accuracy is 0.049242.

Operational success means the pipeline returned a valid result, not that the page was transcribed correctly.

![Specialist limits](artifacts/research/figures/specialist_limits.svg)

*Clean-slice strength does not transfer. The panels use different units and protocols, so they are shown separately rather than averaged.*

## What is and is not supported

**Supported**

- The schema v2 pipeline preserves evidence across stages and records specialist failures rather than hiding them.
- Selective Nemotron improves ClinOCR CER and WER against the Tesseract floor, with paired cluster intervals excluding zero.
- The orientation guard, the PubTables detection and structure specialists, and the clear-control specialist are strong on their stated fixed panels.
- The Phi-4 adapter substantially improves already-localized handwriting crops.
- Review routing removes silent acceptance on the 44-page hard panel.

**Not supported**

- Frontier-model superiority. No frozen frontier baseline was beaten end to end.
- Unattended clinical use. Every hard-panel page routes to review, and none is manually complete.
- End-to-end handwriting improvement. The adapter is measured after localization, and localization is the unsolved step.
- General clinical table, control, or layout accuracy. Strong public component numbers do not transfer to the broad clinical panel.
- Any service-level latency, throughput, GPU-memory, or cost guarantee.

## Model eligibility and privacy

The eligible deployment path excludes Chinese-origin models and backbones.
Such models are reachable only as research comparators on public data, behind an explicit `--public-comparator` flag whose absence is a hard argument error rather than a warning.
Model origin, license, revision, and dataset notes are tracked in [data/README.md](data/README.md).

- Private clinical pages, crops, labels, and predictions stay local or on the authorized private GPU host.
- Private material is not sent to OpenRouter, hosted comparators, or any other third-party service.
- The optional hosted repair cascade accepts public data only, may replace text only in regions it was explicitly authorized to touch, cannot alter `kind`, `bounding_box`, or `provider`, and its patches are rejected unless they strictly reduce the region's risk set.
- The local demo binds to loopback and deletes its temporary session files on exit.
- Held-out private evaluation data stays isolated from training and tuning data.

Do not expose the demo to a network, and do not use it for unattended clinical decisions.

## Repository layout

```text
src/ocr_pipeline/        22 modules, 11,521 lines
  pipeline.py            page preparation, stage orchestration, routing
  contracts.py           TextRegion, DocumentResult, Failure, schema v2
  providers.py           readers: Tesseract, Nemotron, Granite, Ministral, Phi-4
  preprocessing.py       routed view, tiny-text tiling, wide-band recovery
  orientation.py         docTR and OSD guard, view scoring, box restoration
  tables.py              Table Transformer geometry, per-cell candidate fusion
  controls.py            geometric checkbox and radio extraction
  handwriting.py         user-triggered crop reread, two views, abstention
  risk.py                deterministic coverage-risk signals
  verification.py        edit distance, consensus, literal date risks
  cascade.py, repair.py  evidence-scoped patch authorization
  rendering.py           evidence-preserving text and Markdown projection
  demo.py, cli.py        local workbench and one-shot CLI
  layout.py              swappable layout detection, research only
  nemotron_parse.py      Nemotron Parse 2.0 output grammar and box transform
  openrouter*.py         hosted comparator and batch clients, public data only
  operations.py          CUDA peak-memory measurement for benchmarks
tests/                   71 test files
experiments/             49 benchmark and training scripts, 9 evaluation reports
experiments/results/     fixed-panel result JSON
artifacts/               reports and figures
data/                    public benchmark slices and provenance notes
```

## Tests and lint

In a prepared development environment:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  python -m pytest -p no:cacheprovider tests -q

ruff check src experiments tests
ruff format --check src experiments tests
```

## Research record

The reports carry the protocols, denominators, limitations, and evidence paths, so this page can stay navigable.

- [Technical report: architecture, mechanisms, thresholds, results](artifacts/ocr-technical-report.md)
- [Evidence report: claim ledger with per-panel denominators](artifacts/ocr-evidence-report.md)
- [Figure captions, sources, and regeneration commands](artifacts/research/figures/captions.md)
- [Frozen hard-panel backend selection](artifacts/research/backend_selection.md)
- [Phi-4 handwriting adapter evaluation](artifacts/research/phi4_finetuning_cost.md)
- [Fine-tuning plan and execution status](artifacts/research/finetuning_plan.md)
- [Handwriting proposal benchmark](artifacts/research/doctr_proposal_benchmark.md)
- [Private hard-case visual evaluation](experiments/PRIVATE_HARD_CASE_EVALUATION.md)

Each report is a snapshot.
Read every claim with its stated panel, date, and evidence boundary.
