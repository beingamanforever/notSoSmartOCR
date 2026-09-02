# docTR FAST missing-text proposal benchmark

Date: 2026-09-02

## Decision

Do not integrate docTR `fast_base` as a generic missing-handwriting proposal
detector.

The union recovered only 1 of the 9 known C14 misses at target coverage 0.50,
for a gain of 2.273 absolute recall points. It emitted 494 unmatched proposals,
including 444 on 20 pages annotated with no handwriting. This is 0.202%
target-specific proposal precision and 22.2 false proposals per negative page on
average. The measured recall gain misses the required 5-point floor, and no
tested confidence threshold retained the one recovery while controlling the
flood.

The model is also ineligible under the project's non-Chinese-origin model and
backbone rule. The primary FAST paper attributes the architecture and searched
TextNet backbone to authors at Nanjing University, Shanghai AI Laboratory, and
the Chinese University of Hong Kong. This benchmark is a research falsification,
not an approval to ship the model.

No OCR text was generated or replaced. No production code or live OCR
environment was changed.

## Protocol

Local measurement:

- Fixed target pages: `C14-D001-P001` and `C14-D002-P001`.
- Target truth: 44 legible handwriting boxes, 30 on the first page and 14 on the
  second.
- Hard negatives: 20 deterministically selected pages across 9 eligible
  non-handwriting categories. Every selected annotation has an empty handwriting
  list.
- Current regions: every `text` or `word` box in the current OCR output,
  including table-source regions.
- A docTR detection is a novel proposal when less than 50% of its area is
  covered by the union of current OCR regions.
- Proposals are matched by descending score, one-to-one, only to targets missed
  by current OCR. The two matching views are IoU at 0.50 and directional target
  coverage, intersection divided by target area, at 0.50.
- Failed or absent predictions remain in the denominator with zero proposals.
- Default docTR detector thresholds were used: binarization 0.10 and box score
  0.10. Inference used straight boxes, aspect-ratio preservation, symmetric
  padding, batch size 1, and FP32.

The negative annotation establishes absence of handwriting, not exhaustive
absence of all printed text. Therefore, proposal precision is specifically for
the missing-handwriting target. Overall OCR precision loss is not estimable from
this panel. Manual review below checks what the target-false proposals contain.

## Results

All 22 pages completed successfully.

| Match rule | Current OCR | Current + docTR | Absolute gain | Proposal TP / FP / FN | Proposal precision | Recall of baseline misses |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| IoU >= 0.50 | 32 / 44, 72.727% | 33 / 44, 75.000% | +2.273 points | 1 / 494 / 11 | 0.202% | 1 / 12, 8.333% |
| Target coverage >= 0.50 | 35 / 44, 79.545% | 36 / 44, 81.818% | +2.273 points | 1 / 494 / 8 | 0.202% | 1 / 9, 11.111% |

The same box, target index 12 on `C14-D001-P001`, was the only recovery under
both match rules. docTR produced 6,016 raw boxes. After the current-region
novelty filter, 495 remained: 51 on the two targets and 444 on the 20 negatives.

Negative-page proposal load at score 0.10:

| Pages | Total | Mean | p50 | p95 | Maximum |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 444 | 22.2 | 6.5 | 115.45 | 124 |

The four worst negative pages had 124, 115, 48, and 26 novel proposals.

### Confidence sweep

| Minimum score | Novel proposals | Negative proposals | Negative mean/page | Coverage recoveries | Coverage gain |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.10 | 495 | 444 | 22.20 | 1 / 9 | +2.273 points |
| 0.25 | 495 | 444 | 22.20 | 1 / 9 | +2.273 points |
| 0.50 | 488 | 438 | 21.90 | 1 / 9 | +2.273 points |
| 0.75 | 172 | 141 | 7.05 | 0 / 9 | 0 points |
| 0.90 | 0 | 0 | 0 | 0 / 9 | 0 points |

Confidence is not a usable gate here. The thresholds that keep the one true
proposal also keep at least 438 target-false proposals. Score 0.75 removes the
recovery but still emits 141 negative proposals.

## Runtime and failures

The two target pages ran in an isolated environment on an NVIDIA A10G with
python-doctr v1.0.1, PyTorch 2.7.1+cu126, CUDA 12.6, and FP32. After one warmup
page, detector inference was p50 90.749 ms and p95 98.241 ms over 2 measured
pages. Peak PyTorch process memory was 372.601 MiB allocated and 516.0 MiB
reserved. Model load took 5,266.145 ms and both pages succeeded.

The 20 negatives ran locally with the same docTR release and public checkpoint
on CPU because authorization did not permit copying those private pages to the
A10G. They all succeeded. CPU inference was p50 1,031.246 ms and p95 1,238.389
ms. These CPU numbers are coverage evidence only and are not comparable to the
A10G latency. The A10G p95 has only two observations and is not a stable service
tail estimate.

## Manual overlay review

The overlay legend is blue for current OCR, orange for already covered target
truth, red for missed target truth, and green for novel docTR proposals. Review
covered 11 pages: both targets, the four highest-proposal negatives, four
negative pages sampled with seed `20260902`, and the zero-proposal negative.

Observed locally:

- On both C14 pages, most missed handwriting stayed red. Green proposals landed
  mainly on printed labels, instruction fragments, digits, check or control
  marks, and footer elements.
- The two worst negatives were flooded by ordinary printed table values,
  checkbox labels, small glyph fragments, and dense footer text.
- A calendar-style negative produced 48 proposals on printed dates, schedule
  fragments, and grid-adjacent marks.
- Other sampled pages showed proposals on printed table cells, checkmarks,
  electronic-signature text, URLs, and small footer or icon fragments.
- The barcode-bearing C14 page did not yield a separable handwriting signal.
  The sample does not support a barcode- or logo-specific false-positive rate,
  so none is claimed.

The dominant problem is granularity disagreement between docTR detections and
current OCR boxes, not evidence of new handwritten regions. The visual sample
does not reveal a simple geometry or confidence rule that retains the one useful
box while excluding controls, rules, printed fragments, footers, and URLs.

## Evidence boundary and provenance

Source-reported evidence, accessed 2026-09-02:

- The [official docTR predictor source](https://mindee.github.io/doctr/_modules/doctr/models/detection/zoo.html)
  lists `fast_base`, straight-box operation, aspect-ratio preservation, and
  symmetric padding. It also states that docTR source code is Apache-2.0.
- The [official docTR FAST implementation](https://mindee.github.io/doctr/latest/_modules/doctr/models/detection/fast/pytorch.html)
  identifies the exact public `fast_base` checkpoint URL and the 0.10
  binarization and box-score defaults.
- The [primary FAST paper](https://arxiv.org/html/2111.02394v2) identifies the
  authors' institutions and describes FAST-B as using the searched TextNet-B
  backbone. Those affiliations make the architecture ineligible under this
  project's origin policy.

The Apache-2.0 repository license is not treated as proof of a separate
checkpoint license or training-data clearance. Those remain unverified, but no
further provenance work is warranted for a model that already fails both policy
and quality criteria.

Local evidence is limited to this fixed 22-page panel and the stated hardware
split. Raw predictions, evaluation JSON, and overlays remain under the
gitignored `internal-clinical-ocr-benchmark/challenging-formats-20260902/runs/doctr-proposals-v1/`
directory. The tracked report contains case IDs and aggregate measurements only,
not protected text or images.

Hypothesis, not measured: a handwriting-specific, field-anchored proposal model
could reduce printed-text flood. This result does not justify implementing that
route or adapting docTR FAST.

## Reproduction and decision rule

The benchmark runner is `experiments/benchmark_doctr_proposals.py`. It supports
detector-only inference, deterministic panel selection, failure-inclusive
evaluation, confidence sweeps, and local overlays. Focused end-to-end tests cover
split saved prediction runs, scoring, overlay production, box normalization,
union-area novelty, and exclusion of private text from result JSON.

Verification completed locally:

- Ruff check passed.
- Ruff format check passed.
- 20 focused and adjacent tests passed, covering this benchmark, the challenge
  evaluator, and the existing docTR orientation benchmark.
- Both raw evidence files checked are ignored by the repository, and no private
  benchmark path is tracked by Git.

The required integration rule was at least +5 absolute recall points and at
most 1 percentage point of precision loss, or a justified conservative gate.
The measured gain was +2.273 points, overall precision loss was not identifiable,
and the available target-specific evidence showed 0.202% proposal precision
with no viable confidence gate. The decision is final for this candidate:
reject and do not integrate.
