# Handwriting and control recovery review

Date: 2026-09-02

## Recommendation

Use an evidence-preserving three-stage path:

1. Localize missing ink around printed labels, underlines, boxes, and form cells.
2. Recover broken checkbox borders, X marks, ticks, and line-style selections with deterministic image evidence.
3. Route only owned handwritten crops to a specialist recognizer.

TrOCR-small is the best first research challenger and TrOCR-base is the accuracy comparison arm. Neither should be promoted as a commercially clear checkpoint because both use IAM, whose official terms restrict commercial use. LightOnOCR-2-1B is not eligible under this project's backbone-origin rule because its official configuration declares a Qwen3 decoder.

No reviewed pretrained modern-English handwriting checkpoint was simultaneously strong, non-Chinese-origin, and clearly commercially deployable.

## Candidate comparison

| Candidate | Size | Boundary | Decision |
| --- | ---: | --- | --- |
| TrOCR-small handwritten | 62M | Microsoft model; IAM training-data restriction | Research crop challenger |
| TrOCR-base handwritten | 334M | Same provenance limitation | Research accuracy arm |
| PyLaia IAM | Small CTC model | MIT toolkit; IAM restriction | Latency baseline only |
| Kraken with McCATMuS | 16.2 MB model | Engine is Apache-2.0; weight license needs final review | Historical-writing falsifier |
| LightOnOCR-2-1B | 1B | Apache-2.0 surface metadata; Qwen3 decoder | Reject under origin rule |
| BoxDetect | OpenCV heuristics | MIT; no convincing public accuracy result | Comparison only |

## Primary evidence

Sources accessed 2026-09-02.

- The [TrOCR paper](https://arxiv.org/pdf/2109.10282) reports IAM cased CER of 4.22 for the 62M small model, 3.42 for the 334M base model, and 2.89 for the 558M large model. It assumes line crops and therefore does not solve localization. Its augmentation included rotation, blur, erosion, dilation, downscaling, and underlines.
- The [Microsoft TrOCR repository](https://github.com/microsoft/unilm/tree/master/trocr) is MIT licensed. IAM checkpoint provenance remains questioned in [issue 1620](https://github.com/microsoft/unilm/issues/1620) and [issue 1659](https://github.com/microsoft/unilm/issues/1659).
- The [official IAM database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database) limits use to non-commercial research. Repository or model-card metadata does not override dataset rights.
- The [PyLaia IAM model card](https://huggingface.co/Teklia/pylaia-iam) reports CER 8.44 without a language model and 7.50 with a character language model on the RWTH IAM split.
- [Kraken](https://github.com/mittagessen/kraken) is an Apache-2.0 recognition framework. The older [McCATMuS model](https://zenodo.org/records/13788177) is small, but clinical quality and complete weight-license clarity remain unresolved.
- The [LightOnOCR-2-1B configuration](https://huggingface.co/lightonai/LightOnOCR-2-1B/raw/main/config.json) declares `Qwen3ForCausalLM` and `model_type: qwen3` in `text_config`.
- [CheckboxQA](https://arxiv.org/pdf/2504.10419) evaluates checkbox-dependent document understanding, but its [repository](https://github.com/Snowflake-Labs/CheckboxQA) does not provide detector-training truth and is non-commercial.
- [BoxDetect](https://github.com/karolzak/boxdetect) uses geometry and pixel-density thresholds but provides no sufficient general accuracy benchmark.
- [Microsoft form recipes](https://github.com/microsoft/knowledge-extraction-recipes-forms) provide useful OMR patterns while explicitly requiring adaptation, testing, and profiling.
- [NIST Special Database 19](https://www.nist.gov/srd/nist-special-database-19) contains about 810,000 isolated handwritten character images from 3,600 writers. It is relevant to handprint and mark stress tests, not clinical cursive-line validation.

## Local evidence

The frozen 46-field clinical panel already rejects generic handwriting replacement:

| Model | Exact | CER | WER |
| --- | ---: | ---: | ---: |
| PyLaia | 0 / 46 | 94.58% | 117.39% |
| TrOCR-base | 1 / 46 | 82.51% | 136.23% |

Both models remained healthy on a fixed 30-line IAM control, which rules out a broken decoder and confirms domain mismatch. TrOCR-base took about 29 ms per crop resident at batch 8 on A10G, but low latency does not justify clinically incorrect replacement. Full results are in `experiments/HANDWRITING_SPECIALIST_EVALUATION.md`.

## Implementation order

### Missing-ink proposals

- Remove long printed rules morphologically.
- Find residual stroke components near labels, underlines, empty cells, and form fields.
- Merge nearby components into line-like crops.
- Emit candidate evidence even when recognition fails.

### Control recovery

- Keep the intact-square detector.
- Recover broken borders with closing or line intersections.
- Remove estimated borders and rules before state classification.
- Measure diagonal strokes, center residual ink, border crossings, and connected components.
- Search for X and tick proposals only in label-anchored regions.
- Never infer selected state without pixel evidence.

### Crop recognition

- Compare TrOCR-small and base only as research challengers.
- Preserve aspect ratio with white padding.
- Batch crops by width or aspect ratio.
- Compare greedy, beam 4, and beam 10 decoding.
- Preserve the primary output as an alternative instead of silently replacing it.

Approximate FP16 weight storage is 0.12 GB for TrOCR-small and 0.67 GB for TrOCR-base, excluding framework and activation memory.

## Acceptance criteria

Promote deterministic proposal and control repair only if the frozen hard set shows:

- at least 10 points of absolute handwritten-region recall gain;
- no more than 2 false handwriting proposals per page;
- at least 5 points of checkbox macro-F1 gain;
- false selected-state rate at or below 0.5%;
- no decline in label-association F1.

Enable a crop recognizer only if paired CER improvement has a 95% confidence interval excluding zero, hallucinated-character rate rises by no more than 0.5 point, no hard category declines by more than 2 points, and routed-page p95 remains within the selected service budget. Preserve crop, geometry, provider, alternatives, and unresolved evidence.

## Risks

- A recognizer cannot recover handwriting that was never proposed.
- Square resizing distorts long fields.
- Beam search may improve CER while worsening tail latency.
- IAM quality does not establish clinical-form quality.
- Majority voting can reinforce stable errors.
- Full-page handwriting routing increases latency, hallucination exposure, and signature or logo false positives.
- Signatures, scribbles, ticks, and clinical values need different labels.
- A code license does not establish training-data clearance.
