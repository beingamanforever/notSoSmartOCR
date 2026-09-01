# Private hard-case pipeline evaluation

Date: 2026-09-01

## Decision

Keep the specialist pipeline as a review-first research system. It converts the
original near-silent failure mode into explicit evidence and review routing,
but it is not ready for unattended clinical extraction. The frozen 41-page
clinical panel and all three newly supplied pages completed and routed to
review. Independent visual inspection judged every page partial or failed
rather than complete.

Raw images, outputs, and page-level notes stay in the ignored private benchmark
tree. No private page was sent to Jina, OpenRouter, or another hosted service.

## Frozen panel

The 41-page panel supplied on 2026-09-01 covers mixed handwriting, fax forms,
sideways and upside-down pages, faint and tiny text, dense tables, multi-column
records, checkbox grids, signatures, and OASIS-style forms. This is a diverse
failure panel, not a statistically representative clinical distribution and
not transcription ground truth.

Each source was compared visually with the final text, geometry, controls,
tables, and route. A page was counted as complete only if no clinically or
structurally relevant content was visibly missed, corrupted, or mis-associated.

## Expanded hard set

Three exact source images were added to the ignored local benchmark as cases
42 through 44: a handwriting-heavy home-care form and two application-layout
screenshots. Specialist-v7 ran all 44 pages with the same pipeline and no
case-specific rules.

| Measure | Verified result |
| --- | ---: |
| Attempted / operational success / request failure | 44 / 44 / 0 |
| Manually complete / partial or failed | 0 / 44 |
| Review-routed / accepted | 44 / 0 |
| Selected orientation | 39 at 0, 3 at 180, 2 at 270 degrees |
| Evidence-risk regions | 38 |
| Control / table / rejected table-candidate regions | 832 / 19 / 2 |
| End-to-end latency p50 / p95 / max | 0.895 / 4.178 / 4.511 s |
| Reader stage p50 / p95 | 0.514 / 1.181 s |
| Table stage p50 / p95 | 0.203 / 2.903 s |

Cases 43 and 44 previously produced page-sized false tables that hid most of
the useful text. The generic near-page low-complexity check now preserves the
OCR text, emits an unresolved `table_candidate`, and routes both pages to
review. Warm end-to-end measurements were 1.150 seconds for case 43 and 1.369
seconds for case 44, versus 8.164 and 7.522 seconds before the fix. The output
still has substitutions and interleaved reading order across application panes,
so this is a false-structure and latency fix rather than a complete parse.

Case 42 remains materially incomplete on handwritten identity, date, phone,
diagnosis, and order fields. A fixed 46-field handwriting panel rejected a
four-view same-model consensus: it was slower and less accurate than the best
single view. The experiment therefore remains documented rather than enabled.

On the original 41 pages, specialist-v7 retained all 41 review routes and the
same 832 controls and 19 tables. Reader text is not byte-stable across repeated
GPU runs, so the comparison uses these structural and routing invariants rather
than claiming exact transcription equality.

## Pipeline under test

1. NVIDIA Nemotron OCR v2 for literal text and geometry.
2. A four-tile tiny-text rerun that preserves every baseline region, attaches
   crop alternatives, and leaves unsupported tile-only text unresolved.
3. Mindee docTR page orientation with weak Tesseract OSD as a disagreement
   signal and a two-view OCR comparison when needed.
4. Microsoft Table Transformer detection and structure with Nemotron,
   Tesseract raw, and Sauvola cell evidence. Near-identical nested detections
   are suppressed before recognition.
5. Geometric checkbox detection and nearest-line label association. Controls
   remain structured evidence and are not duplicated into rendered text.
6. Deterministic evidence-risk signals for weak text, tiny text, unusually
   large low-confidence regions, and uncertain table-source text.

The final stage does not rewrite text. It adds an unresolved full-page risk
region, so the caller can distinguish operational success from safe acceptance.

## Frozen 41-page specialist-v6 result

| Measure | Verified result |
| --- | ---: |
| Attempted / operational success / request failure | 41 / 41 / 0 |
| Manually complete / partial or failed | 0 / 41 |
| Review-routed / accepted | 41 / 0 |
| Selected orientation | 36 at 0, 3 at 180, 2 at 270 degrees |
| Evidence-risk pages | 37 / 41 |
| Control regions | 832 |
| Table regions | 19 |
| Tiled baseline alternatives / unsupported tile-only candidates | 3,780 / 235 |
| Tile-only candidates promoted into primary text | 0 |
| End-to-end latency mean / p50 / p95 / max | 1.920 / 1.227 / 4.245 / 14.632 s |

Risk reasons are not mutually exclusive: small-text evidence occurred on 21
pages, table-text uncertainty on 16, low mean confidence on 13, and large
low-confidence regions on 6. One page used coverage recovery to select 2,593
characters from the correct 180-degree view instead of 188 characters from a
slightly higher-confidence wrong view.

The original verified cascade returned 40 operational successes, one failure,
40 local accepts, one review route, no structured controls, and 19 table
regions. Manual review found no complete page. Specialist-v6 therefore fixes
a measured false-accept problem and exposes more crop evidence. It does not
establish better handwriting accuracy.

## Specialist-v6 ablation

Specialist-v6 adds the tiny-text rerun and two output-integrity repairs. The
rerun preserved the primary text of every baseline region. It attached 3,780
crop alternatives and exposed 235 tile-only candidates, but promoted none
because no candidate had independent repeated support. Manual inspection found
some better alternatives and some worse ones, so conservative non-promotion is
the correct result.

Rendered text changed on 26 of 41 pages because structured checkbox labels are
no longer repeated as transcription and two nested duplicate table detections
were suppressed. Table count fell from 21 to 19 without removing a distinct
visible table. The 14.632 second maximum is the first cold request; warm p95 is
4.245 seconds. These changes improve evidence clarity and output integrity, not
the manual complete-page denominator, which remains 0 of 41.

## Manual findings

- Orientation: all five visibly rotated pages were restored to the manually
  correct angle. The earlier empty upside-down page returned useful text.
- Tables: dense row and column evidence improved, and two previously accepted
  obstetric table pages now route to review. Exact associations remain wrong in
  places, so token count is not used as a quality claim.
- Controls: clean checkboxes can be useful, but dense ruled grids create false
  proposals. The component remains review-only outside its 52-control frozen
  clear panel.
- Handwriting: dates, identifiers, medication names, measurements, signatures,
  and sparse notes remain unreliable. Generic PyLaia and TrOCR challengers were
  rejected on the same fixed 46-field crop panel.
- Faint and tiny text: more content is recovered, but substitutions remain
  clinically material. Risk routing is an abstention mechanism, not a repair.

## Claim boundary and next experiment

This panel proves that the final route no longer silently accepts the supplied
hard failures and that the new false-table guard fixes the two application
screenshots without changing the frozen panel's table, control, or route counts.
It does not prove field accuracy, clinical safety, generalization, or
superiority over a frontier model. The next high-value work is independent
field and relationship annotation for at least 30 pages, followed by identical
page-level comparisons against pinned frontier baselines when a fresh inherited
OpenRouter credential is available.

Local evidence:

- Expanded outputs: `internal-clinical-ocr-benchmark/hard-cases-20260901/outputs/specialist-v7`
- Frozen outputs: `internal-clinical-ocr-benchmark/hard-cases-20260901/outputs/specialist-v6`
- Page-level visual ledger: `internal-clinical-ocr-benchmark/hard-cases-20260901/REVIEW_NOTES.md`
- Risk implementation: `src/ocr_pipeline/risk.py`
- End-to-end regression: `tests/test_risk.py`
