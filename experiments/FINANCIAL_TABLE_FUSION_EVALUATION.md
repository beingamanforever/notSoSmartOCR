# Financial Table Fusion Evaluation

Date: 2026-09-01

## Scope

This paired diagnostic covers two manually transcribed financial tables with 187
cells. Both source crops and every remaining disagreement were reviewed at
original resolution. This is targeted failure evidence, not a general benchmark
or a production-readiness claim.

## Method

The experiment uses the 17-row structure recovered by TATR-v1.1-All from the
enhanced view. Cell assignment uses non-overlapping row and column anchors from
the recognized span centers because some logical TATR cell boxes overlap. Raw
tokens remain authoritative. Enhanced tokens are used only when raw text is
absent, or when raw mean confidence is below 0.7 and the enhanced candidate is
more confident.

Cell comparison applies Unicode NFKC, case folding, whitespace removal, and
removal of `$` or `§` currency glyphs. Signs, parentheses, percentages, digits,
commas, and other characters remain significant. Source rows are aligned
monotonically by their first-column labels, so omitted rows and blank cells stay
in the failure-inclusive denominator.

The production-path comparison uses full-page TATR detection and
TATR-v1.1-All cell geometry with Nemotron tokens as the primary evidence.
Tesseract raw and Sauvola token boxes are translated from their recorded source
crop into the detected crop with `source origin - (detection origin - 5 px)`.
They can replace primary text only for missing text, mean confidence below 0.9
with agreeing stronger evidence, or overlapping nested primary tokens with a
matching value signature.

## Results

| Method | Correct cells | Exact match | Shape matches | Structural misses | Wrong text |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tesseract raw | 163 / 187 | 87.17% | 0 / 2 | 22 | 2 |
| Tesseract enhanced | 164 / 187 | 87.70% | 2 / 2 | 0 | 21 |
| Tesseract fused | 186 / 187 | 99.47% | 2 / 2 | 0 | 1 |
| Nemotron raw | 182 / 187 | 97.33% | 2 / 2 | 0 | 5 |
| Tri-source fused | 187 / 187 | 100.00% | 2 / 2 | 0 | 0 |

Per table:

| Table | Raw | Enhanced | Fused | Nemotron raw | Tri-source |
| --- | ---: | ---: | ---: | ---: | ---: |
| Projected 2026 | 85.88% | 100.00% | 100.00% | 96.47% | 100.00% |
| Three year | 88.24% | 77.45% | 99.02% | 98.04% | 100.00% |

## Manual review

- Raw structure omitted both light gray margin rows in each table.
- Enhancement recovered all four missing rows, but introduced 21 text errors on
  the three-year table, mainly false prefixes before correct currency values.
- Confidence-gated fusion corrected the raw readings `$ =. 7,500,542` and
  `$ 81,755,451` with visibly correct enhanced readings.
- Every numeric and percentage value in the fused output matched the reviewed
  sources.
- Full-page Nemotron plus TATR recovered the exact 17 by 5 and 17 by 6 shapes
  without enhancement in 1.010 and 0.515 seconds. Four of its five errors were
  duplicated numeric fragments. The fifth changed `(6,247,596)` to
  `(6,247,590`.
- Tri-source fusion retained Nemotron for 182 cells and changed exactly those
  five reviewed errors. Three overlapping-fragment errors used stronger
  Tesseract or Sauvola evidence on the projected table. The three-year table used
  Tesseract raw for one overlapping fragment and the low-confidence final-digit
  error. All five replacements match the source images.
- The only remaining error in the earlier two-source Tesseract fusion is the
  light row label `SaaS GM%`, read as `aaS GM%` in the three-year table. The
  tri-source result preserves Nemotron's correct label and has no remaining
  disagreement on this panel.

## Decision

The tri-source rule is the best measured candidate for the table route. It keeps
the strongest single-view output as primary and uses preprocessing only as local
evidence. It beats the earlier fusion by one cell without value regressions and
beats Nemotron raw by five cells. Promotion still requires a larger fixed table
panel because both confidence thresholds were measured on only these two failure
cases.

Evidence:

- Experiment: `experiments/financial_table_fusion.py`
- Tests: `tests/test_financial_table_fusion.py`
- Measured output: `/private/tmp/notso-ocr-financial-fusion-r7.json`
- Input result: `/private/tmp/notso-ocr-financial-tatr-results.json`
- Nemotron result: `/private/tmp/tatr-fullpage-results.raw`
- Model implementation: <https://github.com/microsoft/table-transformer>
