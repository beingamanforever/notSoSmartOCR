# Phi-4 unresolved handwriting review

Date: 2026-09-02

## Outcome

Frozen Phi-4 is not a replacement for the live Nemotron reader. It recovered more
characters from the 202 unresolved review crops, but frequently transcribed printed
neighbours instead of the referenced handwritten field. The result supports the
planned context-copying and abstention training examples.

## Fixed run

- Model: `microsoft/Phi-4-multimodal-instruct`
- Revision: `93f923e1a7727d1c4f446756212d9d3e8fcc5d81`
- Runtime: BF16, SDPA, Torch 2.7.1+cu126, CUDA 12.6
- Input: all 202 records whose conservative Nemotron match failed
- Privacy: local A10G inference only; references were not passed to the model or
  serialized in its result
- Denominator: all 202 records, including one CUDA out-of-memory failure

## Results

| Measure | Frozen Phi-4 | Nemotron top-1 candidate |
| --- | ---: | ---: |
| Fields | 202 | 202 |
| Normalized exact | 3 | 0 |
| Normalized exact rate | 1.49% | 0.00% |
| Character error rate | 3.6388 | 0.6705 |
| Missed-character rate | 0.0957 | 0.3661 |
| Extra-character rate | 3.2347 | 0.1015 |
| Substitution rate | 0.3084 | 0.2029 |
| Mean normalized edit distance | 0.7756 | 0.5418 |

Phi-4 operations were 201/202 successful, with 1 out-of-memory failure. Latency
was 1.338 s p50, 8.448 s p95, and 8.670 s maximum after model loading.

## Interpretation boundary

This is a combined localization and recognition stress test, not an isolated crop
recognition benchmark. Each crop is centered on Nemotron's closest region after the
true handwriting span failed conservative localization. Some crops therefore omit
the target or contain substantially more printed context. That makes the absolute
recognition scores unsuitable for a model leaderboard.

The paired failure pattern is still useful. Phi-4's lower deletion rate and extreme
insertion rate show that it reads surrounding content instead of returning an empty
or tightly scoped answer. The next training set must therefore contain:

1. A tight and a context-padded view of the same resolved field with the same target.
2. Blank fields, printed-only crops, and stray marks labeled `<NO_HANDWRITING>`.
3. Genuinely unreadable handwriting labeled `<UNREADABLE>`.
4. Real-only family-disjoint development and test sets.

No backend promotion follows from this run. Model selection remains gated on the
fixed hard pages, exact literal recovery, character errors, unsupported additions,
failures, and latency.
