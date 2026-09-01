# Orientation specialist evaluation

Date: 2026-09-01

## Decision

Use Mindee docTR page orientation as a cheap proposal and use Tesseract OSD as
a disagreement signal. When the proposed angles differ, compare only those two
lossless OCR views with the production confidence and coverage policy. Keep any
nonzero rotation or failed view review-routed.

This replaces the earlier four-view fallback on the evaluated panels. It does
not establish complete OCR quality: all 41 private hard pages still require
review for handwriting, table, control, or faint-text failures.

## Implementation

The wrapper performs EXIF normalization, classifies the page, records weak OSD
evidence, and evaluates the distinct proposed views. It prefers the usual OCR
confidence rank unless a view has at least twice the character coverage, at
least 10 supporting words, and mean confidence within 0.03 of the narrower
view. Selected region boxes and nested table-cell boxes are mapped back to the
original pixel coordinates.

The classifier is `mobilenet_v3_small_page_orientation` from python-doctr
1.0.1. The recorded checkpoint is published by Mindee and the code repository
is Apache-2.0. The architecture uses torchvision MobileNetV3. No Chinese-origin
model or backbone appears in the inspected runtime lineage. Weight-specific
license confirmation and an explicit official corporate-country statement are
still required before production approval.

Primary sources, accessed 2026-09-01:

- [docTR repository](https://github.com/mindee/doctr)
- [docTR model documentation](https://mindee.github.io/doctr/using_doctr/using_models.html)
- [Mindee company site](https://www.mindee.com/fr)

## Private orientation panel

All 41 supplied generated clinical-style pages were inspected at original
resolution before reading predictions. The frozen manual labels contain 36
upright pages, two pages requiring 270 degrees, and three pages requiring 180
degrees.

| Measure | Result |
| --- | ---: |
| Manually correct classifier angles | 41 / 41 |
| Upright / 180 / 270 degree predictions | 36 / 3 / 2 |
| Warm page preparation plus classifier | 15.243 ms/page |
| Warm classifier inference | 7.716 ms/page |
| Checkpoint size | 6,233,146 bytes |

Tesseract OSD was not a viable primary classifier on the same pages: it
returned an angle on 22 of 41, failed on 19, and was manually correct on 7 of
41. No page reached confidence 15. OSD remains useful only as independent weak
evidence.

One private page exposed a confidence-only selection failure. The wrong upright
view contained 188 characters at mean confidence 0.715096, while the correct
180-degree view contained 2,593 characters at 0.713687. The coverage rule chose
the correct view and the final 41-page run used the five manually correct
nonzero rotations.

## Frozen public guard evaluation

The guarded selector was replayed from cached caller-boundary OCR on all 56
ClinOCR rotated evaluation pages. The deployable selector never used the
reference text. References were used only after selection to measure error and
oracle regret.

| Measure | Result |
| --- | ---: |
| Attempted / covered / failed | 56 / 56 / 0 |
| Exact minimum-CER view selection | 56 / 56 |
| Mean / maximum CER regret | 0 / 0 |
| docTR and OSD angle agreement | 55 / 56 |
| Case-mean / micro CER | 0.100211 / 0.104300 |
| Case-mean / micro WER | 0.121439 / 0.122360 |
| Micro missed / hallucinated text rate | 0.039526 / 0.032178 |

The exact-oracle count is a diagnostic, not a deployable oracle claim. It means
the gold-free guarded choice happened to match a minimum-CER cached view on
every fixed case. Aggregate guarded latency is unavailable because orientation
signals and OCR views were measured in separate runs. The latency row above is
only the resident private classifier measurement.

## Historical five-page recovery

Before docTR integration, a four-view Nemotron fallback recovered useful text
from the two sideways and three upside-down supplied pages, including one page
where angle zero failed. It also selected a physically wrong angle on one page
and required four sequential OCR calls whenever OSD was weak or absent. That
experiment established the value of alternate views but was not promoted.

## Claim boundary

The current evidence supports accurate angle proposal and lossless recovery on
the two fixed orientation panels. It does not establish handwriting accuracy,
table relationship accuracy, reading order, concurrency behavior, or end-to-end
latency for the guarded policy. These remain separate measured requirements.

Evidence:

- Runner: `experiments/orientation_guard_benchmark.py`
- Public output: `experiments/results/orientation-guard-rotated-eval-frozen-v1.json`
- Private hard-panel report: `experiments/PRIVATE_HARD_CASE_EVALUATION.md`
- Production wrapper: `src/ocr_pipeline/orientation.py`
