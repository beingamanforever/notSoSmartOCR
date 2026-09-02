# Not So Smart OCR

**Evidence-linked OCR for clinical and structured documents.**

[Architecture](#architecture) · [Results](#results-at-a-glance) ·
[Quick start](#local-quick-start) · [Research report](artifacts/ocr-evidence-report.md)

Not So Smart OCR is a modular research pipeline that keeps text tied to source pixels.
It preserves geometry, reading order, provider provenance, alternatives, structure,
and explicit failures in one JSON schema so a specialist can challenge a reading
without silently erasing the original evidence.

> Research status: the current system is not production-ready and has not
> demonstrated superiority over a frozen frontier baseline. The report keeps
> measured gains, negative results, and remaining gaps in the same denominator.

![Evidence-linked OCR pipeline](artifacts/research/figures/pipeline_architecture.svg)

## Architecture

```text
PDF, TIFF, or image
  -> ordered page preparation
  -> swappable literal reader
  -> ordered region stages
  -> schema v2 document result
  -> local inspection and review routing
```

The core boundary is deliberately small:

| Layer | Contract | Current examples |
| --- | --- | --- |
| Reader | Produces literal, positioned `TextRegion` evidence | Tesseract, NVIDIA Nemotron OCR v2, IBM Granite Docling |
| Region stage | Receives regions and returns enriched regions | tables, controls, handwriting alternatives, evidence-risk review |
| Result | Keeps pages, regions, alternatives, provenance, failures, and route | JSON-serializable `DocumentResult`, schema version 2 |
| Workbench | Inspects the result without changing the model route | page overlays, evidence details, stage timing, JSON and Markdown export |

Readers implement `LocalReader`; enrichment components implement `RegionStage`.
`process_document(..., reader, stages=(...))` composes them in order. A failed
stage is recorded instead of being hidden, and unresolved or conflicting evidence
is routed to review.

The eligible path excludes Chinese-origin models and backbones. Such models may
appear only as research comparators on public data. Model origin, license, and
dataset notes are tracked in [data/README.md](data/README.md).

## Local quick start

The lean demo uses Python, Pillow, Tesseract, and Poppler. The repository does
not currently provide a package manifest. GPU specialists run in separate
model-specific environments.

Install system tools on macOS:

```bash
brew install tesseract poppler
```

On Debian or Ubuntu:

```bash
sudo apt-get install tesseract-ocr poppler-utils
```

Create the lean Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install Pillow fastapi uvicorn python-multipart
```

Extract evidence-linked JSON from an image or PDF:

```bash
PYTHONPATH=src python -m ocr_pipeline.cli page.png \
  --reader tesseract --output result.json
```

Start the local review workbench, then open `http://127.0.0.1:8000`:

```bash
PYTHONPATH=src python -m ocr_pipeline.demo
```

The default workbench uses the local routed Tesseract reader. Optional GPU
readers and stages are assembled by
[`experiments/serve_gpu_demo.py`](experiments/serve_gpu_demo.py); run it with
`--help` to see the required local model paths. Those environments are not part
of the lean install above.

Keep the optional Phi-4 adapter warm in its compatible environment, then point
the GPU demo at the loopback service:

```bash
PYTHONPATH=src /path/to/phi4/python \
  experiments/serve_phi4_handwriting.py \
  --adapter /private/path/vision_decoder_lora-runtime.pt \
  --torch-site /path/to/torch/site-packages --warmup

PYTHONPATH=src python experiments/serve_gpu_demo.py \
  <required local model arguments> \
  --phi4-handwriting-url http://127.0.0.1:8083
```

Both processes bind to loopback. Crop bytes remain in memory and the Phi-4
service is contacted only when a user selects a region and requests a
handwriting reread. The browser is a private review surface; the configured
backend may run on the same machine or on an authorized GPU reached through a
private tunnel.

## Swappable stages and safety boundary

The verified GPU composition can add orientation selection, tiny-text tiling,
wide-band recovery, Table Transformer geometry, control extraction, and
deterministic evidence-risk routing. Each component keeps its own provenance.

The Phi-4 handwriting adapter is an optional, on-demand research stage. A user
must select an existing bounded region before it runs. The adapter can add a
candidate or corroborate an existing reading, but disagreement is retained as
alternative evidence. The rejected automatic classifier is available only for
offline experiments and is not enabled in the demo. Adapter weights stay on the
authorized private GPU host and outside Git.

## Results at a glance

The strongest public result is selective Nemotron versus the Tesseract floor on
all 328 ClinOCR evaluation pages. Failures remain in the denominator.

| Reader | Coverage | CER | WER | Page latency |
| --- | ---: | ---: | ---: | ---: |
| Tesseract | 299/328 | 0.488293 | 0.611352 | p50 1.844 s, p95 5.990 s |
| Selective Nemotron | 282/328 | **0.331423** | **0.417869** | p50 1.862 s, p95 2.355 s |

The paired CER change is -0.156870 with a cluster 95% interval of
[-0.185387, -0.113176]. This establishes improvement over the local floor,
not a frontier-model win.

![Public OCR comparison](artifacts/research/figures/public_ocr_comparison.svg)

The five-page hard panel is a development snapshot. Its redacted per-page
artifact was not retained in the tracked evidence tree, so it is not the
primary reproducibility result.

| Five-page configuration | CER | WER | Hallucination rate | Missed-character rate | Completion | Warm latency |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Frozen old baseline | 0.559161 | 0.766033 | 0.28249 | 0.146711 | Not reported here | Not reported here |
| Final safe wide-band v5 | **0.550873** | **0.760095** | **0.281256** | 0.154117 | 5/5 success | p50 2.786 s, p95 6.523 s |

The final route slightly improves CER, WER, and hallucination rate while
increasing missed-character rate. Operational success means the pipeline
returned a valid result, not that the page was transcribed correctly.

| Component check | Fixed evidence | Decision |
| --- | --- | --- |
| Tables | Presence F1 1.0 on 5 pages; structure descriptors remain poor | Keep structure review-routed |
| Phi-4 handwriting adapter | Already-localized C14 crops: 50/88 exact and 0.135294 CER, versus stock 32/88 and 0.401471 CER | Crop specialist only; no measured end-to-end handwriting gain |
| Automatic handwriting classifier | 18 proposals, 8 matches, 9/44 box recall; warm p50 7.497 s | Rejected for automatic routing |

![Handwriting adapter comparison](artifacts/research/figures/handwriting_adapter_comparison.svg)

The warmed Phi-4 process took 0.606 s on a blank 320 by 96 crop after a 24.964 s
cold load and occupied 11,646 MiB on the measured A10G. These are single-host
observations, not general service guarantees.

The adapter result measures recognition after localization. It does not measure
page-level handwriting detection, end-to-end recall, or unattended safety. The
classifier result is insufficient to supply that missing localization step.

## Privacy

- Private clinical pages, crops, labels, and predictions stay local or on the
  authorized private GPU host.
- Private material is not sent to Jina, OpenRouter, hosted comparators, or other
  third-party services.
- The local demo binds to loopback by default and deletes its temporary session
  files when the process exits.
- Held-out private evaluation data must remain isolated from training and tuning
  data.

Do not expose the demo to a network or use it for unattended clinical decisions.

## Test

In a prepared development environment:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  python -m pytest -p no:cacheprovider tests -q

ruff check src experiments tests
ruff format --check src experiments tests
```

## Research record

The reports carry protocols, denominators, limitations, and evidence paths so
the README can stay concise:

- [Complete OCR evidence report with plots](artifacts/ocr-evidence-report.md)
- [Frozen hard-panel backend selection](artifacts/research/backend_selection.md)
- [Phi-4 handwriting adapter evaluation](artifacts/research/phi4_finetuning_cost.md)
- [Fine-tuning plan and execution status](artifacts/research/finetuning_plan.md)
- [Handwriting proposal benchmark](artifacts/research/doctr_proposal_benchmark.md)
- [Private hard-case visual evaluation](experiments/PRIVATE_HARD_CASE_EVALUATION.md)
- [Demo design QA](design-qa.md)

These reports are research snapshots. Claims should be read with their stated
panel, date, and evidence boundary.
