# Phi-4 Multimodal hard-page triage review

Date: 2026-09-02

## Decision

**Reject Phi-4 Multimodal as a replacement for the Nemotron-backed specialist pipeline. Do not spend the full 22-page promotion run on this configuration.**

Phi-4 is promising only as a selective upright handwriting rereader. On the three complete-reference pages it slightly improves transcription and substantially improves exact handwriting recovery, but it increases unsupported insertions, has no region geometry or structured table output, does not preserve checkbox states, and catastrophically fails the 180-degree page. A crop-level experiment may still be worthwhile, provided every candidate span remains evidence-linked and the incumbent output is retained when Phi-4 disagrees.

## Measured comparison

Both systems were scored against the same five source-only annotations. Only three pages have complete transcription references, so CER and WER promotion claims use those three pages. The two partial-reference pages remain in manual review and failure analysis.

| Measure | Phi-4 | Specialist v10 / Nemotron | Judgment |
| --- | ---: | ---: | --- |
| Complete-reference micro CER, 3 pages | 0.3262 | 0.3534 | Phi-4 better by 0.0272 absolute |
| Complete-reference micro WER, 3 pages | 0.4613 | 0.5143 | Phi-4 better by 0.0530 absolute |
| Complete-reference missed-text rate | 0.1392 | 0.1766 | Phi-4 better |
| Complete-reference hallucinated-text rate | 0.1483 | 0.1043 | Phi-4 worse by 42% relative |
| Page-wide legible handwriting phrase presence, 5 pages | 40 / 79, 50.6% | 20 / 79, 25.3% | Diagnostic only; not localized evidence and concentrated on upright pages |
| Page latency p50 | 33.470 s | 3.699 s | Phi-4 9.0x slower |
| Page latency p95 | 228.857 s | 13.366 s | Phi-4 17.1x slower |
| Maximum page latency | 267.368 s | 14.322 s | Phi-4 18.7x slower |
| Structured controls | None | Present, but over-detected | Phi-4 cannot replace this stage |
| Structured tables and cell geometry | None | Present, with imperfect structure | Phi-4 cannot replace this stage |
| Phi-4 CUDA peak allocated / reserved | 14,245.5 / 15,142.0 MiB | Not recorded in the paired run | Fits A10G, no paired memory claim |

The Phi-4 result file reports handwriting as 40 / 81. Its denominator includes two partial or illegible placeholders on the rotated page. The legible denominator is 79, yielding 40 / 79. A later evaluator audit found that this old page-wide matcher can credit a phrase that appears outside the annotated handwriting box. It is retained only as a diagnostic and is not promotion evidence.

On the two fully annotated C14 pages alone, Phi-4 is stronger at page transcription: micro CER is 0.1849 versus 0.2632 and micro WER is 0.2351 versus 0.4123 under the original scorer. Its 24 / 44 handwriting figure is unlocalized page-wide phrase presence, while the incumbent's audited localized score is 15 / 44. They are not a like-for-like localization comparison. The later crop experiment supplies the valid field-localized comparison and rejects stock Phi-4 at 15 / 44 native and 16 / 44 scaled with unsafe substitutions.

## Manual review by failure class

All five source pages and predictions were inspected at source resolution. The
private literals stay in the ignored benchmark. The review found:

- upright forms: stronger handwriting recall but clinically material character
  substitutions, copied neighboring labels, and lost checkbox associations;
- dense forms: higher handwriting recall did not compensate for higher CER and
  unsupported row generation;
- tiny control grids: flattened rows and columns, missing checkbox state, and
  poor recovery of short handwritten scores;
- the 180-degree form: a repetition loop reached the 4,096-token cap and
  omitted the page's useful structured content.

One diagnostic page took 74.812 seconds and the rotated failure took 267.368
seconds. These are single-run diagnostics, not service-level latency claims.

The run records this token-limit loop as `status: success`, so the displayed 100% coverage and 0% failure rate overstate useful coverage. Any future candidate evaluator should classify a repetition loop or token-limit termination without a valid complete output as a failed or abstained page.

## Minimal next experiment

Do not integrate Phi-4 as a page reader or renderer. If GPU time permits, run only the frozen 44 native and 44 scaled handwriting crops and compare exact span recovery, unsupported insertions, latency, and memory against the existing crop baselines. Advance it only as a selective rereader when:

1. the proposed crop is localized by existing evidence,
2. the returned literal is short and non-repetitive,
3. it agrees with another reader or passes deterministic field constraints,
4. disagreement remains visible rather than overwriting the incumbent, and
5. the full pipeline still preserves boxes, tables, controls, reading order, and provenance.

## Evidence inspected

- `internal-clinical-ocr-benchmark/challenging-formats-20260902/runs/phi4-triage-v1/results.json`
- `internal-clinical-ocr-benchmark/challenging-formats-20260902/runs/specialist-v10/model-output/`
- `internal-clinical-ocr-benchmark/challenging-formats-20260902/runs/specialist-v10/c14-evaluation.json`
- `internal-clinical-ocr-benchmark/challenging-formats-20260902/runs/specialist-v10/c08-evaluation-content.json`
- Five private source PNGs and matching source-only annotations

The paired five-page incumbent metrics were recomputed with `experiments/evaluate_challenge_set.py` over only these five cases. No model was rerun and no private source left the local or authorized GPU environment.
