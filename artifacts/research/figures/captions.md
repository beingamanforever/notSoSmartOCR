# Figure captions

Schematics are TikZ; data plots are matplotlib. Both obey the same rules: built
at the printed width, exported as vector, and captioned rather than titled.

## Evidence-linked architecture

**Evidence survives every stage.** The prepared page passes through one nested
`LocalReader`, becomes an ordered `TextRegion[]` record, and is enriched by
region stages that may add readings but never delete them. The accented block
is the intermediate representation, which is the contribution: it is the only
contract shared by the reader, the specialists, verification, and rendering.
The orange dashed edge is the optional handwriting sidecar, which returns a
candidate as an alternative rather than as a replacement.

Source: the executable flow in `src/ocr_pipeline/pipeline.py`,
`src/ocr_pipeline/contracts.py`, `src/ocr_pipeline/preprocessing.py`,
`src/ocr_pipeline/handwriting.py`, `src/ocr_pipeline/verification.py`, and
`src/ocr_pipeline/rendering.py`.

## Gated reading stack

**Each recovery guard has an explicit gate, and a closed gate changes nothing.**
The four guards run in order inside a single reader. The gate row states the
measured condition that triggers the guard, the computation row states what it
runs, and the accented row states what it is permitted to change. No guard
deletes a region it did not create; a rejected candidate is retained as
alternative evidence and every fired gate is recorded in the page coverage
assessment.

Source: `RoutedTesseractReader`, `TiledReader`, and `WideBandFallbackReader` in
`src/ocr_pipeline/preprocessing.py`, and `OrientationReader` in
`src/ocr_pipeline/orientation.py`. Thresholds in the figure are the shipped
defaults, not tuned per document.

## Candidate resolution

**Adoption demotes the incumbent; it never erases it.** One ordered rule set
decides whether a challenger reading replaces the surviving text, and the same
rule serves tile fusion, table cells, and the handwriting sidecar. Whichever
branch fires, the record keeps the merged evidence IDs of every supporting
candidate, every distinct rejected reading, and a `conflicting` marker when an
equally confident challenger still disagrees.

Source: `_resolve_cell` and `_best_challenger` in `src/ocr_pipeline/tables.py`,
`_fuse_tiled_view` and `_resolve_tile_only` in
`src/ocr_pipeline/preprocessing.py`, and `_apply_candidate` in
`src/ocr_pipeline/handwriting.py`.

## Public ClinOCR comparison

**Selective Nemotron lowers transcription error while serving fewer pages.**
CER and WER are failure-inclusive over all 328 ClinOCR evaluation pages.
Coverage is plotted beside them so the improvement cannot hide a drop from 299
to 282 served pages. The paired panel shows the directly available 95% cluster
bootstrap intervals for error-rate deltas only. Those intervals resample 16
template clusters with 10,000 draws at seed 0; they are not repeated-inference
uncertainty.

Source: the public-transcription table and paired-delta paragraph in
`artifacts/ocr-evidence-report.md`. The generating script reads both directly.
Negative deltas favor selective Nemotron.

## Specialist limits

**Strong narrow-slice behavior does not imply broad structured extraction.**
The control panel contrasts detection coverage on 52 clear controls from two
pages with exact safe-match coverage on 1,521 annotations across a 169-page
challenging-formats panel. These protocols are intentionally labeled
separately. On the same broad panel, page-level table-presence F1 remains high
while exact row-count and column-count accuracy are low. Presence F1 and
descriptor accuracy are different statistics and are not averaged.

Sources: `experiments/CHECKBOX_SPECIALIST_EVALUATION.md` for the clear-control
result and `artifacts/research/figures/specialist_limits.csv` for the redacted
broad controls and tables aggregate. The CSV was copied from the private
169-case evaluation; raw pages, annotations, and predictions remain outside
Git. One fixed execution per protocol; no confidence intervals.

## Handwriting adapter comparison

**The adapter improves already-localized handwriting recognition.** On the
fixed 88-inference C14 panel, the v4 adapter raises exact match and sharply
reduces CER and hallucinated characters. One paired run per arm; no confidence
intervals; localization is excluded.

Source: the fixed C14 holdout table in
`artifacts/research/phi4_finetuning_cost.md`. Each arm contains 44 native and
44 scaled crop inferences. The figure does not imply page-level detection or
end-to-end gain.

## Five-page hard-panel comparison

**Selective routing yields small net changes on the five-page hard panel.**
Final safe v5 improves CER, WER, and hallucination rate, while
missed-character rate rises. One run per configuration; no confidence
intervals.

Source: the measured table in `README.md`. The generating script reads that
table directly. Operational success and latency are intentionally omitted
because the baseline row does not report them.

## Reproduction

Run from this directory. The three schematics are TikZ and compile with any
LaTeX engine; `tectonic` needs no local TeX tree. `pdftocairo` ships with
Poppler, which the pipeline already requires.

```bash
for figure in pipeline_architecture reading_stack evidence_resolution; do
  tectonic -X compile "$figure.tex" && pdftocairo -svg "$figure.pdf" "$figure.svg"
done
```

```bash
MPLCONFIGDIR=/tmp/not-so-smart-ocr-matplotlib python public_ocr_comparison.py
MPLCONFIGDIR=/tmp/not-so-smart-ocr-matplotlib python specialist_limits.py
MPLCONFIGDIR=/tmp/not-so-smart-ocr-matplotlib python handwriting_adapter_comparison.py
MPLCONFIGDIR=/tmp/not-so-smart-ocr-matplotlib python hard_panel_comparison.py
```

The plots write PDF and SVG at a 6.75-inch printed width and print the
OpenResearch audit result. The schematics compile to a tightly cropped PDF
between 6.5 and 6.8 inches wide; `orx-tikz-preamble.tex` is the vendored
scaffold they share, so the diagram palette matches the plot palette.
