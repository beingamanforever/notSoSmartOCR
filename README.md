# notSoSmartOCR

A research-first OCR pipeline for evidence-linked document extraction. It preserves page order, word geometry, confidence, provenance, and explicit failures instead of returning text alone.

The current implemented floor supports PDF and image input through Poppler and Tesseract. The planned structure-aware path uses the full PaddleOCR-VL-1.6 pipeline, compares GLM-OCR under the same layout stage, and escalates only observable failures to multimodal hosted models.

## Current path

```text
PDF or image
  -> ordered page render
  -> local OCR regions
  -> evidence-linked document JSON
  -> public benchmark adapters
```

Blank output, renderer errors, OCR errors, and invalid images remain explicit failures. The CLI returns a nonzero status when the document is partial or failed.

## Run local OCR

Requirements currently used by the local floor:

- Python 3.12 or later
- Pillow
- Tesseract
- Poppler `pdftoppm` for PDFs

```bash
PYTHONPATH=src python -m ocr_pipeline.cli document.pdf --output result.json
```

## Run tests

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest -p no:cacheprovider tests -q
ruff check src/ocr_pipeline experiments/public_benchmark.py tests
ruff format --check src/ocr_pipeline experiments/public_benchmark.py tests
```

The tests cross the real CLI boundary, render a multi-page PDF, invoke Tesseract TSV, verify geometry and evidence links, and retain both forced and empty-output failures.

## Public benchmarks

Dataset sources, terms, splits, and exclusions are recorded in [data/README.md](data/README.md). Downloaded data and run outputs stay outside Git.

```bash
PYTHONPATH=src python experiments/public_benchmark.py \
  clinocr data/public/clinocr-bench-v1.0/ClinOCR-Bench \
  experiments/results/tesseract-clinocr-v1.0.json --workers 8

PYTHONPATH=src python experiments/public_benchmark.py \
  funsd data/public/funsd/dataset \
  experiments/results/tesseract-funsd-original.json --workers 8
```

Measured Tesseract floor on this machine:

| Dataset | Cases | Coverage | Micro CER | Micro WER | p50 page latency | p95 page latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ClinOCR-Bench v1.0 eval | 328 | 91.2% | 0.451 | 0.568 | 943 ms | 1,658 ms |
| FUNSD original test | 50 | 100% | 0.565 | 0.790 | 576 ms | 952 ms |

All 29 empty ClinOCR outputs remain in the denominator. These runs are a cheap CPU floor, not evidence that Tesseract is the preferred parser.

## Research and execution plan

The evidence review, architecture, model and dataset matrix, metrics, ablations, routing rules, and phase exit conditions are in [OCR_PIPELINE_RESEARCH_AND_PLAN.md](OCR_PIPELINE_RESEARCH_AND_PLAN.md).

Private clinical images are never sent to an external provider without separate PHI, region, retention, and contract approval. API keys are read from the environment and are never stored in the repository.
