# Routing between Falcon and Qwen: what each signal is worth

Measured on 30 September 2026 on one NVIDIA A10G, against the deployed build.
All arms ran the same pages on a side web app sharing the live Falcon service; Qwen 3.7 Flash at temperature 0, which still varies between runs by about ±2 handwriting snippets.

## The policy

Falcon reads every region.
A region goes to Qwen only when a specific signal says Falcon's reading is doubtful; everything else keeps Falcon's text.

| Reason | Signal | Basis |
| --- | --- | --- |
| `likely_misread_text` | Falcon's calibrated probability of a correct read is below 0.5 | Bayes decision with equal costs; also where Qwen beat Falcon on 1,699 NAF test regions |
| `second_reader_doubt` | docTR's printed-text reader doubts a word, or surely reads one Falcon's text lacks | Falcon's likelihood stays high on confident handwriting misreads, which a monotone calibration cannot reorder |
| `unresolved_recognition`, `sampled_retry_read`, repetition reasons | The decoder terminated, looped, or needed a sampled retry | No calibrated probability exists for these reads |
| `letter_among_digits` | A lone letter inside a run of digits | Character-class consistency |
| `visual_region`, `owned_form_row`, failed tables | Figures, form rows owned by a form, and tables whose cells failed | Content Falcon's text reader does not cover |

Crops travel in parallel requests of at most 12, and each crop's answer is accepted or rejected on its own (see [serving under load](09-serving-queue-cache-dashboard.md)).

## What each reason is worth on print

Each Qwen-refined region on the OmniDocBench English pages was matched to the ground-truth blocks whose centres it holds, and Falcon's original reading (kept in the region's provenance) and Qwen's reading were scored against them.
The absolute error rates include matching noise; the comparison between the two readers on the same regions is the signal.

| Pages | Reason | Regions | Falcon CER | Qwen CER | Qwen better / worse |
| --- | --- | ---: | ---: | ---: | ---: |
| Text | `second_reader_doubt` only | 501 | 0.225 | 0.236 | 87 / 141 |
| Text | `unresolved_recognition` (alone or with others) | 24 | 0.093 | 0.043 | 9 / 0 |
| Text | `letter_among_digits` | 14 | 0.742 | 0.723 | 9 / 3 |
| Tables | `second_reader_doubt` only | 350 | 0.118 | 0.120 | 39 / 59 |
| Tables | `likely_misread_text` | 39 | 0.311 | 0.162 | 14 / 3 |

Falcon's own calibrated doubt and failed recognitions are where Qwen helps.
On print, docTR's doubt alone is not: Qwen's reading is no better than Falcon's there.

## Controlled arms

### OmniDocBench English text, 150 pages (official evaluator)

| Arm | Text edit ↓ | Reading-order edit ↓ | Formula edit ↓ | Qwen crops | Output tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| A: all reasons (deployed) | 0.0481 | 0.0667 | 0.171 | 971 | 103k |
| B: without second-reader doubt | 0.0492 | 0.0644 | 0.163 | 203 | 11k |
| C: Falcon only | 0.0686 | 0.0815 | 0.163 | 0 | 0 |

Routing lowers the text error by 28% (C to B) with 203 crops.
The second reader adds 768 crops for 0.001 text edit.

### Handwritten cohort (6 source PDFs, structured forms and tables)

| Arm | Snippets exact | Format-insensitive | Forms pooled accuracy | Missing controls | Qwen crops | Output tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A: all reasons (deployed) | 45 / 146 | 74 | 0.733 | 127 | 1,464 | 223k |
| B: without second-reader doubt | 42 / 146 | 71 | 0.718 | 137 | 794 | 171k |

On handwriting the second reader finds confident misreads; the earlier offline analysis sent 42 of 81 missed snippets to Qwen with it and 22 without.

### Decision

The second reader stays.
It is worth about three snippets and 1.5 points of form accuracy on handwriting, the documents this service is for, and on print it costs about $0.02 per 1,000 pages.
Gating it by writing style would keep the gain without the print cost, but the style has to be detected reliably first (below).

## Cost per 1,000 pages (deployed routing)

| Benchmark | Regions routed | Pages with any crop | Crops kept | Qwen cost |
| --- | ---: | ---: | ---: | ---: |
| OmniDocBench text | 37.9% | 90% | 95% | $0.18 |
| OmniDocBench tables | 23.8% | 84% | 96% | $0.12 |
| Domain checkbox forms (392 pages) | 13.2% | 75% | 93% | $0.30 |
| Handwritten snippet pages | 24.4% | 96% | 89% | $0.43 |

## Tried and not adopted

- **A handwriting style classifier to gate the second reader.** A logistic regression on docTR's recognition features, trained on NAF form words (CDLA-Permissive-1.0) with its official split by form type, reached test AUC 0.983 and 94% accuracy on form types it never saw. On our documents it called 40% of docTR's doubted words on printed OmniDocBench pages handwriting, and 43% of those on the handwritten forms: printed doubt there is formula fragments, stylized fonts and tiny text, unlike NAF's typewritten forms. It needs in-domain training data before it can gate routing.
- **Unexplained-ink detection for tick marks without printed boxes.** Pixels left after removing ruling lines and docTR word boxes flagged glyph fragments on a clean financial table and missed ticks that touch words. Tick-only lists need a learned mark detector.
- **Letting Qwen mark a box Heron left unselected.** Right in 6 of 7 handwritten-form cases where it was adopted, right in 1 of 18 on the domain forms. Withdrawn; the detector keeps its state.

## A Falcon-only caveat

With Qwen off, three English pages that also carry Chinese (OmniDocBench textbook pages) lose a region each: the foreign-script guard marks it doubtful so Qwen rereads it, and without Qwen doubtful text is not rendered.
That accounts for the whole Falcon-only difference from the 29 September code (0.0686 against 0.0540 on identical pages).
With Qwen on, as deployed, all three pages score 0.0 to 0.025.
