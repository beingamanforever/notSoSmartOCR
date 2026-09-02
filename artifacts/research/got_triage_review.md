# GOT-OCR2.0 hard-page triage review

Date: 2026-09-02

## Decision

Reject GOT-OCR2.0 as a replacement for the Nemotron-backed specialist pipeline.
Only two of five pages produced usable flat transcription. Three dense pages
either reached the 4,096-token generation limit or returned only a page header,
and the model provides neither evidence geometry nor structured tables and
controls.

GOT remains a research-only comparator. The official implementation derives
from Qwen2 and Vary, so it is outside the deployable model-origin boundary even
if its accuracy were competitive.

## Frozen five-page result

| Measure | GOT-OCR2.0 | Specialist v10 / Nemotron | Judgment |
| --- | ---: | ---: | --- |
| Pages attempted | 5 | 5 | Paired |
| Manually usable page transcriptions | 2 / 5 | 5 / 5 covered | GOT fails the gate |
| Raw micro CER, all five diagnostic references | 2.2209 | Not comparable across partial references | GOT dominated by insertions |
| Raw hallucinated-text rate | 1.8032 | Not comparable across partial references | GOT dominated by repeated output |
| Page-wide legible handwriting phrase presence | 15 / 79, 19.0% | 20 / 79, 25.3% | Diagnostic only; GOT is worse |
| Page latency p50 | 268.344 s | 3.699 s | GOT 72.5x slower |
| Page latency p95 | 334.867 s | 13.366 s | GOT 25.1x slower |
| End-to-end throughput | 0.0039 pages/s | Not recorded in the same run | Too slow for routing |

CER on the two partial-reference C08 pages is diagnostic only. The handwriting
row also uses the original page-wide phrase matcher, which can credit text
outside the annotated handwriting box. The replacement decision is instead
supported by the complete-reference pages, token-limit failures, manual source
comparison, missing structure, and latency.

## Manual source comparison

All five private pages and predictions were inspected at source resolution.
The detailed literals stay in the ignored benchmark. The review found:

- both upright handwriting pages had low span recovery, character splitting,
  clinically material substitutions, and missing control associations;
- both dense pages lost row, column, score, and checkbox relationships, with
  one page producing no valid orientation candidate;
- the 180-degree page reached the generation limit on the correct orientation
  and returned no useful structured content.

Three pages showed token-limit behavior. This makes the candidate a hard
failure for the fixed panel regardless of its aggregate edit score.

## Instrumentation correction

The completed raw artifact was generated just before the runner's token-limit
status fix, so its five top-level records say `success` even when the chosen
candidate has `possible_token_limit: true`. The runner now records such a
candidate as `got_token_limit`, excludes it from orientation selection, and
fails a page when no valid candidate remains. The raw predictions are retained
for diagnosis; they are not counted as usable OCR.

Focused verification:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. python -m pytest \
  -p no:cacheprovider tests/test_benchmark_got.py -q
4 passed
```

## Sources

All sources accessed 2026-09-02.

- [GOT paper](https://arxiv.org/abs/2409.01704)
- [Official GOT repository](https://github.com/Ucas-HaoranWei/GOT-OCR2.0)
- [Released GOT checkpoint](https://huggingface.co/stepfun-ai/GOT-OCR2_0)
- Local raw result: `internal-clinical-ocr-benchmark/challenging-formats-20260902/runs/got-triage-v1/results.json`
