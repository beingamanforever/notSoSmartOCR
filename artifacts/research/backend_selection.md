# OCR backend selection on the frozen hard-page panel

Date: 2026-09-02

## Decision

Keep the Nemotron-backed specialist pipeline as the live incumbent. None of the
stock challengers passes the replacement gate on the same five difficult pages
and localized C14 crop checks. Phi-4 Multimodal is the only candidate worth
supervised adaptation, but it must remain outside the runtime until a
family-held-out crop evaluation shows a large literal-recovery gain with zero
critical substitutions and no unsupported-text increase.

## Frozen panel

The page panel contains two upright handwritten clinical forms, one dense mixed
form, one tiny dense scoring grid, and one 180-degree rotated form:

- `C14-D001-P001`
- `C14-D002-P001`
- `C08-D003-P001`
- `C08-D007-P007`
- `C08-D017-P003`

All pages and failures remain in the denominator. Only the three pages with
complete transcription references contribute to CER and WER. Handwriting
promotion uses localized field crops or exact text plus annotated-box overlap,
not page-wide phrase search.

## Outcome

| Candidate | Best useful evidence | Disqualifying evidence | Decision |
| --- | --- | --- | --- |
| Nemotron specialist v10 | 5 / 5 pages covered; 0.3534 CER and 0.5143 WER on 3 complete pages; preserves regions, controls, tables, reading order, and provenance | Localized C14 handwriting is only 15 / 44; controls and table structure still need improvement | Keep incumbent |
| Phi-4 page reader | 0.3262 CER and 0.4613 WER on the 3 complete pages | 42% higher insertion rate, p50 33.470 s, p95 228.857 s, no geometry or structure, and repetition on the rotated page | Reject page replacement |
| Phi-4 crop reader | 15 / 44 native and 16 / 44 scaled normalized exact | 23 critical substitutions and 97 inserted characters in the better scaled arm; misses the 36 / 44 gate by 20 fields | Fine-tune candidate only |
| GOT-OCR2.0 | Two readable flat page transcriptions | 3 / 5 pages unusable or token-limited, 2.2209 CER, p50 268.344 s, no geometry or structure | Reject |
| Granite Docling 258M | Small memory footprint | 2 / 44 exact, 3.2000 CER, 16 empty outputs, 9 repetition loops | Reject |
| Ministral 3B crop reader | 9 / 44 normalized exact | 1.0941 CER and substantial context completion | Reject stock model |
| Florence-2 Large crop reader | 5 / 44 normalized exact with low latency | Too little literal recovery and 23.5% inserted-character rate | Reject |
| docTR FAST proposals | Recovered 1 additional C14 miss | 494 unmatched proposals, including 444 on 20 negative pages | Reject generic proposal route |

The five-page incumbent p50 was 3.699 seconds and p95 was 13.366 seconds. These
are small-panel diagnostics, not service-level latency estimates. The candidate
tests nevertheless establish clear domination: the generative page challengers
are slower, less structured, less reliable on rotation, or more hallucinatory.

## Replacement gate

A challenger may replace Nemotron only after it passes all of the following on
the frozen panel and an expanded family-held-out set:

1. Every attempted page remains in coverage, failure, and latency denominators.
2. CER, WER, missed-text rate, and localized handwriting exact recovery improve.
3. Unsupported insertions and critical literal substitutions do not increase.
4. Rotation, tables, checkboxes, reading order, bounding boxes, and provenance
   remain available through the canonical `TextRegion` representation.
5. Manual review confirms the gain on at least handwriting, tiny text, faint
   text, dense tables, and rotated forms.

## Next experiment

Compile only geometry-backed, family-separated handwriting crops from non-C14
development families, then run a short Phi-4 vision-adapter LoRA canary. Keep
C14 untouched as the frozen promotion set. Compare the adapted checkpoint with
the stock model and incumbent on tight and padded crops. Stop unless two seeds
improve exact localized recovery without any critical substitution or insertion
regression.

## Evidence

- `artifacts/research/phi4_triage_review.md`
- `artifacts/research/phi4_crop_review.md`
- `artifacts/research/got_triage_review.md`
- `artifacts/research/granite_crop_benchmark.md`
- `artifacts/research/doctr_proposal_benchmark.md`
- `internal-clinical-ocr-benchmark/challenging-formats-20260902/runs/specialist-v10/c14-evaluation-spatial.json`

Private source pages, crops, annotations, and raw predictions stayed on the local
machine and the authorized private A10G host.
