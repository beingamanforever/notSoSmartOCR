# Figure captions

## Five-page hard-panel comparison

**Selective routing yields small net changes on the five-page hard panel.** Final safe v5 improves CER, WER, and hallucination rate, while missed-character rate rises. One run per configuration; no confidence intervals.

Source: the measured table in `README.md`. The generating script reads that table directly. Operational success and latency are intentionally omitted because the baseline row does not report them.

## Handwriting adapter comparison

**The adapter improves already-localized handwriting recognition.** On the fixed 88-inference C14 panel, the v4 adapter raises exact match and sharply reduces CER and hallucinated characters. One paired run per arm; no confidence intervals; localization is excluded.

Source: the fixed C14 holdout table in `artifacts/research/phi4_finetuning_cost.md`. Each arm contains 44 native and 44 scaled crop inferences. The figure does not imply page-level detection or end-to-end gain.

## Evidence-linked architecture

**The pipeline preserves evidence through specialist stages.** Every page becomes ordered, provenance-carrying `TextRegion` records. Table and control specialists enrich those records; the on-demand handwriting sidecar adds alternative evidence without overwriting the incumbent, and verification drives routing and rendering.

Source: the executable flow in `src/ocr_pipeline/pipeline.py`, `src/ocr_pipeline/contracts.py`, `src/ocr_pipeline/handwriting.py`, `src/ocr_pipeline/verification.py`, and `src/ocr_pipeline/rendering.py`. The handwriting stage is the only accent because it is optional and user-triggered.

## Public ClinOCR comparison

**Selective Nemotron lowers transcription error while serving fewer pages.** CER and WER are failure-inclusive over all 328 ClinOCR evaluation pages. Coverage is plotted beside them so the improvement cannot hide a drop from 299 to 282 served pages. The paired panel shows the directly available 95% cluster-bootstrap intervals for error-rate deltas only. Those intervals resample 16 template clusters with 10,000 draws at seed 0; they are not repeated-inference uncertainty.

Source: the public-transcription table and paired-delta paragraph in `artifacts/ocr-evidence-report.md`. The generating script reads both directly. Negative deltas favor selective Nemotron.

## Specialist limits

**Strong narrow-slice behavior does not imply broad structured extraction.** The control panel contrasts detection coverage on 52 clear controls from two pages with exact safe-match coverage on 1,521 annotations across a 169-page challenging-formats panel. These protocols are intentionally labeled separately. On the same broad panel, page-level table-presence F1 remains high while exact row-count and column-count accuracy are low. Presence F1 and descriptor accuracy are different statistics and are not averaged.

Sources: `experiments/CHECKBOX_SPECIALIST_EVALUATION.md` for the clear-control result and `artifacts/research/figures/specialist_limits.csv` for the redacted broad controls and tables aggregate. The CSV was copied from the private 169-case evaluation; raw pages, annotations, and predictions remain outside Git. One fixed execution per protocol; no confidence intervals.

## Reproduction

Run from this directory:

```bash
MPLCONFIGDIR=/tmp/not-so-smart-ocr-matplotlib python hard_panel_comparison.py
MPLCONFIGDIR=/tmp/not-so-smart-ocr-matplotlib python handwriting_adapter_comparison.py
MPLCONFIGDIR=/tmp/not-so-smart-ocr-matplotlib python pipeline_architecture.py
MPLCONFIGDIR=/tmp/not-so-smart-ocr-matplotlib python public_ocr_comparison.py
MPLCONFIGDIR=/tmp/not-so-smart-ocr-matplotlib python specialist_limits.py
```

Each script writes PDF and SVG at a 6.75-inch printed width and prints the OpenResearch audit result.
