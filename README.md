<div align="center">

<img src="artifacts/brand/notsosmartocr-logo.png" alt="Not So Smart OCR logo" width="120">

# Not So Smart OCR

**Document extraction you can inspect.**

Text · Tables · Handwriting · Formulas · Source-linked review

[Examples](#examples) · [Pipeline](#pipeline) · [Serving](#serving-and-dashboard) · [Walkthroughs](#walkthroughs)

</div>

![Not So Smart OCR workbench](artifacts/demo/workbench.png)

OCR turns text in images, PDFs, and scanned or handwritten documents into editable text.

![Images, PDFs, and scanned documents converted to text with OCR](artifacts/demo/ocr-overview.png)

## Features

- Extract text and structure from PDFs and images.
- Inspect colored layout boxes and download full-resolution source crops.
- Review formulas, tables, images, and detected checkbox states.
- Explore text, visual, Markdown, and JSON views of the same evidence.
- Select output to locate its source, inspect regional scores, and edit a transcription in a multiline editor with its crop alongside it.
- Saved edits update Text output, Markdown, and JSON together while retaining the original reading in the evidence history.
- Fit the full page, navigate multi-page documents, and use the responsive light or dark interface.
- Save feedback with its image and result, retaining 1,000 records locally.
- Watch long PDFs fill in page by page, and see your place in line while other documents are read.
- Follow usage, routing, latency, spend, and the live queue in a separate, token-protected admin dashboard.

## Examples

| Type | Example | View |
| --- | --- | --- |
| Paper | Attention Is All You Need | [Output](artifacts/demo/paper-ui.png) |
| Handwriting and math | Physics notes | [Output](artifacts/demo/notes-ui.png) |
| Printed math | Scanned calculus formulas | [Output](artifacts/demo/printed-math-ui.png) |
| Tables | Public financial report | [Output](artifacts/demo/table-ui.png) |
| Screenshot | GSoC Final Evaluation | [Output](artifacts/demo/gsoc-ui.png) |

| Academic layout | Handwritten formulas |
| --- | --- |
| ![Academic paper with detected regions](artifacts/demo/paper-detection.png) | ![Handwritten notes and model output](artifacts/demo/notes-ui.png) |
| **Text and images** | **Printed mathematics** |
| ![GSoC Final Evaluation with source-linked OCR output](artifacts/demo/gsoc-ui.png) | ![Scanned calculus formulas with detected regions and rendered math output](artifacts/demo/printed-math-ui.png) |

| Inspect a source region | Edit a transcription |
| --- | --- |
| ![Visual inspector with the source crop, transcription, and score provenance](artifacts/demo/visual-inspector.png) | ![Source-linked editing with an option to keep uncertain text unresolved](artifacts/demo/region-editing.png) |

The workbench and inspector screenshots were updated on 30 September 2026; the academic paper and source-selection screenshots on 29 September 2026.
Screenshots show actual output, including errors.
Overlapping regions can still produce duplicates; handwriting, numeric fields, and reading order remain imperfect.
Model scores are uncalibrated, and a hosted reread may not provide a recognition score.
Private documents and datasets are excluded.

## Pipeline

```mermaid
flowchart LR
    A[PDF / image] --> B[Orientation]
    B --> C[Heron layout]
    C --> D[Falcon OCR crops]
    D --> E[Evidence checks]
    E --> F[Optional crop refinement]
    F --> G[Canonical evidence]
    G --> H[Text / Visual / Markdown / JSON]
    H --> I[Review and feedback]
```

The demo verified on 29 September 2026 uses docTR orientation and skew estimation, Heron layout inside the native Falcon server, complete-region Falcon OCR, and evidence checks.
Optional Qwen3.7 Flash crop refinement is enabled in the updated demo screenshots; it requires a separately configured hosted API and spending limit.
Source pixels and canonical evidence are retained, with local KaTeX for math.
The demo composition endpoint reports the configuration of each running service.

Image-decode reuse, lighter response transfer, and overlapping local OCR with hosted refinement reduce overhead while preserving original OCR inputs and canonical exports.
Parallel page rereads and CPU layout tiles reduced one warm three-page run from 84.6 to 52.5 seconds on an A10G, with partial results retained.
Four paired native OCR comparisons preserved the complete response except timing metadata.
Single-page hosted latency remains a bottleneck; see the [measurements and preservation limits](docs/walkthroughs/latency-preservation.md).
The UI loads source crops on demand; downloaded JSON retains their original PNG data.

## Serving and dashboard

Parses reach the GPU one at a time, in arrival order.
Waiting uploads see how many documents are ahead of them; past 32 running or waiting parses, new uploads get a `503` with `Retry-After`.
Long PDFs show each page as soon as it is read.
In a burst of ten concurrent uploads on the A10G host, all ten were served in order, and the page still loaded in under 0.09 s.

Two in-memory, byte-bounded caches with a time limit reuse page readings and complete LLM replies for byte-identical input; repeated parses took 0.1 to 0.8 s instead of 9 to 16 s, with identical results.
An LLM reply that is cut off or has one malformed record keeps every crop it answered correctly; a provider failure in the middle of a streamed reply is retried.

`/admin` is off unless an admin token is configured.
It shows requests, pages, users, latency split into OCR, LLM, and waiting time, LLM spend, cache use, routing reasons, and the live queue, with each request's page, routed regions, and confidence scores.

![Admin dashboard](artifacts/demo/admin-dashboard.png)

```bash
OPENROUTER_API_KEY=... OCR_ADMIN_TOKEN=... python experiments/serve_gpu_demo.py --falcon-layout-url http://127.0.0.1:8196 \
  --qwen-crops --qwen-budget-usd 5 --usage-root ~/ocr-state/usage --feedback-root ~/ocr-state/feedback \
  --page-cache-mb 1024 --qwen-reply-cache-mb 64 --max-active-parses 32
```

Keep usage records, feedback, and the spending ledger on persistent storage; instance storage is erased when a cloud instance stops.
See [Serving under load](docs/walkthroughs/09-serving-queue-cache-dashboard.md) for measurements, the failure that motivated these changes, and operating advice.

## Walkthroughs

Walkthroughs grounded in the code, technical reports, and measured runs:

| Read | Covers |
| --- | --- |
| [From pixels to text](docs/walkthroughs/01-from-image-to-text.md) | The running pipeline and model choices |
| [Layout, tables, and Falcon contributions](docs/walkthroughs/02-layout-tables-and-crops.md) | Geometry, crops, duplicates, and upstream fixes |
| [Confidence, Markdown, and review](docs/walkthroughs/03-confidence-exports-and-review.md) | Token scores, rendering, exports, and feedback |
| [Model internals and hardware](docs/walkthroughs/04-model-internals-and-hardware.md) | DETR, orientation, table recognition, and VRAM |
| [Inside Falcon-OCR](docs/walkthroughs/05-falcon-technical-report.md) | Early fusion, attention, training, and native serving |
| [Document-parsing strategies](docs/walkthroughs/06-document-parsing-strategies.md) | Seven OCR systems and their architectural trade-offs |
| [From research to implementation](docs/walkthroughs/07-research-to-implementation.md) | What we reused, what each paper taught us, and observed improvements |
| [Fidelity repairs and routing results](docs/walkthroughs/08-fidelity-and-routing.md) | Current runtime, preserved table content, paired Qwen comparison, and remaining failures |
| [Serving under load](docs/walkthroughs/09-serving-queue-cache-dashboard.md) | Fair queue, page streaming, caching, robust LLM replies, the admin dashboard, and operations |
| [Routing between Falcon and Qwen](docs/walkthroughs/10-routing-evaluation.md) | What each routing signal is worth, controlled A/B arms, cost per 1,000 pages, and what was rejected |

**Hardware:** exercised on a 24 GB-class NVIDIA A10G. Minimum VRAM is not established; see the [memory breakdown](docs/walkthroughs/04-model-internals-and-hardware.md#vram-and-performance).

## Acknowledgements

Thank you to the teams behind [Falcon-Perception](https://github.com/tiiuae/Falcon-Perception), [IBM's Heron layout model](https://huggingface.co/docling-project/docling-layout-heron-101), [DocTR](https://github.com/mindee/doctr), [Tesseract](https://github.com/tesseract-ocr/tesseract), [Poppler](https://gitlab.freedesktop.org/poppler/poppler), and [KaTeX](https://github.com/KaTeX/KaTeX). Heron comes from the Docling project. Their respective licenses apply.
