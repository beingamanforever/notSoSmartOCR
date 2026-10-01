# Public benchmarks: OmniDocBench (English) and ParseBench

Measured on 1 October 2026 against the deployed build (Falcon OCR with Heron layout, evidence checks and Qwen 3.7 Flash routing), on one NVIDIA A10G.
Both runs went through a side web app sharing the live Falcon service, one page at a time.

## OmniDocBench v1.6, all 755 English pages

Scored with the maintainers' own evaluation image (`ghcr.io/zeng-weijun/omnidocbench-eval:repro-ubuntu2204`: TeX Live 2025, ImageMagick 7.1.1-47, Ghostscript 9.55), end2end config with quick_match.
Rescoring an earlier 150-page run in this image reproduced its text edit exactly (0.0481), so the Docker scorer agrees with the earlier local one; it adds formula CDM, which needs LaTeX.

| Text edit ↓ | Formula CDM ↑ | Formula edit ↓ | Table TEDS ↑ | Table TEDS-S ↑ | Reading-order edit ↓ | Overall ↑ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0545 | 91.59 | 0.154 | 78.50 | 85.35 | 0.128 | **88.21** |

Overall is the leaderboard's formula, ((1 − text edit) × 100 + TEDS + CDM) / 3.
All 755 pages have predictions; 1,905 formulas and 243 tables were scored, with no matcher, TEDS or CDM timeouts.
Three bilingual textbook pages produced no text and count against the score; 150 pages are partial only because some routed crops kept their OCR reading.

The public leaderboard averages all 1,651 pages, Chinese included, so this English-only score places the pipeline only approximately: between OCRVerse (88.60) and HunyuanOCR (89.95), above GPT-5.2 (86.59), olmOCR (85.74) and Mistral OCR (85.66), below dots.ocr (90.77) and the leading 1B-class specialized models (95-97).

## ParseBench, all 2,079 documents

[ParseBench](https://github.com/run-llama/ParseBench) (LlamaIndex, April 2026) scores enterprise PDFs on five dimensions; Overall is their mean.
The pipeline is connected through `experiments/parsebench_provider`, modelled on ParseBench's Falcon-OCR provider: each PDF goes to `/api/process`, the page Markdown is the service's own export, and every evidence region becomes a layout item in Heron's labels.
All five dimensions evaluated every document (no failures).

| | Overall | Tables | Charts | Content faithfulness | Semantic formatting | Visual grounding |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| This pipeline | 50.99 | 53.84 | 1.26 | 86.62 | 48.36 | 64.88 |
| Falcon-OCR alone (leaderboard) | 53.08 | 74.70 | 0.82 | 78.59 | 47.91 | 63.37 |

Content faithfulness, which counts omissions, hallucinations and reading-order violations, is 8 points above Falcon alone; grounding and formatting are slightly above.
Tables are 21 points below: 45 of 503 table documents had no table region and 220 had fewer tables than the ground truth.
ParseBench's Falcon-OCR entry uses Falcon's own layout model, where this pipeline uses Heron, which misses more tables on these pages; table detection is the next target.
Charts are returned as images rather than data, as with most OCR models on that leaderboard.

One adapter correction was made during scoring: the export writes simple tables as Markdown pipe tables, and ParseBench's table metrics read HTML only.
The adapter now applies the same pipe-table conversion ParseBench's own Falcon-OCR provider applies; the saved outputs were rebuilt from their raw page Markdown and rescored without new inference, which moved Tables from 34.64 to 53.84.

## Reproducing

```bash
# OmniDocBench: inference on the side app, then the official image
bash ops/live_bench.sh NAME odb_all
# ParseBench, from a ParseBench checkout with `uv sync --extra runners`
NSSOCR_SERVER_URL=http://127.0.0.1:8194 .venv/bin/python ../provider/run.py run nssocr --max_concurrent 2
```
