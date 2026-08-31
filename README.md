# notSoSmartOCR

A research-first OCR pipeline for accurate, scalable, evidence-linked document parsing. It preserves page order, literal text, geometry, reading order, provider provenance, and explicit failures instead of returning an untraceable text blob.

The implemented system supports:

- ordered PDF, multi-frame TIFF, and image ingestion with lossless EXIF orientation normalization;
- Tesseract, PaddleOCR-VL-1.6, complete GLM-OCR SDK, and direct GLM-OCR model-only readers;
- deterministic validation for missing text, geometry, reading order, malformed tables, and repeated hallucination loops;
- selective OpenRouter crop repair using Qwen3.8-Flash or Muse Glimmer 30B;
- independent second-model visual agreement before any hosted text replaces local evidence;
- direct, stitched, public transcription, OmniDocBench export, and paired statistical benchmark harnesses.

The current innovation hypothesis is the Evidence-Patch Cascade: use a strong compact local parser, detect observable risks, ask one visual model for a crop-level candidate, require a different visual model to independently agree, then merge only the agreed text while protecting local geometry and recording both hosted text sources. Disagreement abstains. This is implemented, but it is not yet a proven frontier-model win.

## Pipeline

```text
PDF or image
  -> ordered, EXIF-normalized pages
  -> swappable local structured reader
  -> evidence-linked regions
  -> deterministic risk detection
  -> selective visual candidate
  -> independent visual verification
  -> protected patch or abstention
  -> JSON, Markdown, or benchmark adapter
```

## Run local OCR

The Tesseract path needs Pillow, Tesseract, and Poppler `pdftoppm`. Paddle and GLM use their official isolated runtime dependencies.

```bash
PYTHONPATH=src python -m ocr_pipeline.cli document.pdf \
  --reader tesseract --output result.json

PYTHONPATH=src python -m ocr_pipeline.cli page.png \
  --reader paddleocr-vl --device gpu:0 --output result.json

PYTHONPATH=src python -m ocr_pipeline.cli page.png \
  --reader glm-ocr-direct --max-new-tokens 8192 --output result.json
```

`glm-ocr` uses the official complete SDK and recognition server. `glm-ocr-direct` is a model-only ablation without layout analysis.

## Public experiments

Dataset sources, acquired revisions, terms, counts, and evaluation roles are recorded in [data/README.md](data/README.md). Downloaded data and run outputs remain outside Git.

```bash
# Failure-inclusive local transcription
PYTHONPATH=src python experiments/public_benchmark.py \
  clinocr data/public/clinocr-bench-v1.0/ClinOCR-Bench \
  experiments/results/paddle-clinocr.json \
  --reader paddleocr-vl --device gpu:0 --workers 1

# Direct hosted full-page baseline
PYTHONPATH=src python experiments/frontier_benchmark.py \
  clinocr data/public/clinocr-bench-v1.0/ClinOCR-Bench \
  experiments/results/qwen-direct.json \
  --model qwen/qwen3.8-flash --provider PROVIDER_SLUG

# Selective Muse repair with independent Qwen verification
PYTHONPATH=src python experiments/cascade_benchmark.py \
  clinocr data/public/clinocr-bench-v1.0/ClinOCR-Bench \
  experiments/results/muse-cascade.json \
  --reader paddleocr-vl --device gpu:0 \
  --model meta/muse-glimmer-30b --provider PRIMARY_PROVIDER \
  --verifier-model qwen/qwen3.8-flash \
  --verifier-provider VERIFIER_PROVIDER

# Export predictions for the official version-matched OmniDocBench evaluator
PYTHONPATH=src python experiments/omnidocbench_export.py \
  data/public/omnidocbench-v1.6/OmniDocBench.json \
  data/public/omnidocbench-v1.6 \
  experiments/results/omnidocbench-paddle \
  --dataset-revision v1.6 --reader paddleocr-vl --device gpu:0

# Paired, failure-inclusive confidence intervals and corrected tests
PYTHONPATH=src python experiments/paired_comparison.py \
  experiments/results/baseline.json experiments/results/candidate.json \
  experiments/results/paired.json --resamples 10000 --seed 0
```

Hosted experiments require a fresh `OPENROUTER_API_KEY` in the environment. Pin providers for comparisons. The key pasted into chat was not used and should be rotated.

## Measurements so far

| Run | Cases | Coverage | Micro CER | Micro WER | p50 | p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Tesseract, ClinOCR v1.0 eval | 328 | 91.2% | 0.451 | 0.568 | 0.94 s | 1.66 s |
| Tesseract, FUNSD original test | 50 | 100% | 0.565 | 0.790 | 0.58 s | 0.95 s |
| PaddleOCR-VL-1.6, balanced six-page development trace | 6 | 100% | 0.199 | 0.231 | 11.20 s | 79.19 s |
| GLM-OCR direct model-only, same six development pages | 6 | 100% | 0.109 | 0.118 | 9.36 s | 18.34 s |
| PaddleOCR-VL-1.6 base, ClinOCR rotated subset | 56 | 100% | 0.314 | 0.378 | 12.82 s | 41.29 s |
| PaddleOCR-VL-1.6 plus unwarping, ClinOCR rotated subset | 56 | 100% | 0.061 | 0.081 | 13.19 s | 20.47 s |
| GLM-OCR direct model-only, ClinOCR rotated subset | 56 | 100% | 0.079 | 0.084 | 10.79 s | 15.63 s |

The six-page rows cover only two independent ClinOCR templates and are development signals, not ranking evidence. On the matched 56-page rotated subset, unwarping reduced micro CER by 0.2530 with a template-cluster 95 percent interval of [-0.3060, -0.2001], and micro WER by 0.2970 with an interval of [-0.3690, -0.2211]. Direct GLM also beat Paddle base, but versus Paddle plus unwarping its CER delta was +0.0183 with interval [-0.0028, 0.0357] and WER delta was +0.0027 with interval [-0.0220, 0.0224]. The difference is inconclusive and does not establish the locked non-inferiority margin. GLM is a text-only whole-page ablation here, not a structured-parser replacement. A point sample used 4.9 GB for GLM versus 22.2 GB for Paddle unwarping, but neither is a measured peak. Hosted direct and stitched arms remain unmeasured.

The saved Paddle GPU artifacts predate the `run_config` serializer. Their exact commands and option differences are recorded in the research report, but the files are not independently self-describing and should be regenerated before external publication. New public, cascade, frontier, and OmniDocBench artifacts serialize behavior-affecting options.

## Verify the project

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  python -m pytest -p no:cacheprovider tests -q
ruff check src/ocr_pipeline experiments tests
ruff format --check src/ocr_pipeline experiments tests
```

The tests cross real CLI and experiment boundaries, render a multi-page PDF and multi-frame TIFF, invoke Tesseract TSV, verify evidence and geometry, inject exact Paddle, GLM, and OpenRouter contracts, retain failures in denominators, recompute paired metrics from raw text, and prove that truncated, same-model, arbitrary, or disagreeing hosted patches cannot replace local evidence.

## Research report

The paper and repository review, architecture rationale, measured evidence, benchmark matrix, statistical decision rules, lossless inference plan, fine-tuning gate, and labeling schema are in [OCR_PIPELINE_RESEARCH_AND_PLAN.md](OCR_PIPELINE_RESEARCH_AND_PLAN.md).

Private clinical images are never sent to an external provider without separate PHI, region, retention, training-use, and contract approval. API keys are read only from the environment and never stored in the repository.
