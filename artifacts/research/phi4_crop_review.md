# Phi-4 Multimodal C14 crop review

Date: 2026-09-02

## Status update

This report is the frozen stock-model baseline. A later v4 decoder-LoRA run
materially improved crop recognition, but did not pass the safety or
localization requirements for automatic use. See
[Phi-4 handwriting adapter v4](phi4_finetuning_cost.md) for the paired result.

## Decision

Reject stock Phi-4 Multimodal as a live handwriting crop reader. Keep the
Nemotron-backed specialist pipeline as the incumbent and use the later adapted
Phi-4 only as a selective, review-routed crop specialist.

The paired crop run completes quickly and improves over the rejected stock
Ministral, Florence, and Granite crop readers, but it does not improve enough
over the incumbent extraction to justify another model call. Native Phi-4
matches 15 of 44 fields after normalization. The 3x crop matches 16 of 44,
only one more than the incumbent's 15 of 44 full-page handwriting recovery.
It also makes 23 critical substitutions and inserts 97 unsupported characters.

## Frozen result

| Reader and view | Normalized exact | Critical substitutions | Inserted-character rate | p50 latency |
| --- | ---: | ---: | ---: | ---: |
| Phi-4 native crop | 15 / 44, 34.1% | 24 / 39 | 29.4% | 0.486 s |
| Phi-4 3x crop | 16 / 44, 36.4% | 23 / 39 | 28.5% | 0.491 s |
| Specialist v10 full page | 15 / 44, 34.1% | Not scored in this crop format | Not scored in this crop format | Included in page latency |
| Ministral 3B native crop | 9 / 44, 20.5% | Not scored in the older artifact | 90.0% | 0.154 s |
| Florence-2 Large native crop | 5 / 44, 11.4% | Not scored in the older artifact | 23.5% | 0.129 s |
| Granite Docling crop | 2 / 44, 4.5% | 26 / 39 | 262.1% | 3.370 s |

The adoption target was at least 36 of 44 exact fields without unsafe
insertions or critical substitutions. Phi-4 misses that target by 20 fields.
The paired benchmark took 46.955 seconds after model load. Peak CUDA memory was
11,381 MiB allocated and 11,724 MiB reserved.

## Manual review

Four representative crops and their persisted predictions were inspected at
source resolution. The review found four recurring error classes: copied
printed context, unsupported prefixes, clinically material character
substitutions, and omission of signature-like writing. Upscaling changed some
errors but did not consistently correct them. The detailed literals and source
images remain only in the ignored private benchmark.

## Interpretation

The crop experiment isolates the main problem: localization alone is not
sufficient. Tight crops often retain form labels or adjacent rows, and the
stock decoder copies that context or completes plausible clinical words. A live
Phi-4 route would add about 0.5 seconds per crop and almost 11 GiB of resident
weights for a one-field gain with unsafe substitutions.

The useful next step is supervised adaptation on the ordered local crop targets,
with hard negatives for adjacent labels and rows. Keep literal output separate
from normalization, train assistant tokens only, and evaluate field-family
holdouts before any RL stage. Do not integrate this stock checkpoint into the
runtime cascade.

## Evidence

- Private paired result:
  `internal-clinical-ocr-benchmark/challenging-formats-20260902/runs/phi4-crops-v1/results.json`
- Frozen crops and annotations:
  `internal-clinical-ocr-benchmark/challenging-formats-20260902/runs/ministral-c14-crops-source-only/`
- Incumbent C14 result:
  `internal-clinical-ocr-benchmark/challenging-formats-20260902/runs/specialist-v10/c14-evaluation.json`
- Earlier candidate artifacts in the same frozen crop directory

Raw crops and predictions remained on the local machine and the authorized
private A10G server. They were not sent to a public service.
