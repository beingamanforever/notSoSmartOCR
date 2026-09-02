# Control and handwriting specialist review

Date: 2026-09-02

## Decision

Keep the current contour control detector and test one targeted extension:
label and row-anchored proposals for broken or overwritten controls, followed by
border and rule removal and conservative residual-ink classification. Do not add
a full-page detector or VLM until this deterministic recovery is measured on an
exhaustively annotated degraded panel.

Keep handwriting review-routed. The frozen 46-field clinical panel rejected
generic PyLaia and TrOCR-base checkpoints even though both were healthy on IAM.
No reviewed non-Chinese-origin pretrained handwriting checkpoint is currently
both clinically strong and provenance-clear. Florence-2-base is the smallest
license-clear fixed challenger. TrOCR-small-stage1 and PARSeq are research
adaptation arms, not deployment-ready specialists.

## Why control recovery should start before classification

The current `detect_controls` path proposes near-square four-corner contours
with an enclosed child. A destructive X, tick, scribble, or broken ring can
erase the proposal before state classification runs. The next experiment should
therefore improve proposal recall rather than replace the already accurate state
classifier:

1. Preserve intact contour proposals.
2. Estimate expected control slots from readable labels and intact row peers.
3. Remove long horizontal and vertical rules inside only those slots.
4. Classify remaining ink as selected, unselected, or ambiguous.
5. Never emit unselected for a borderless empty slot.

The broader 21-page control set is not fully annotated, so raw prediction counts
cannot support false-positive claims. Bounding-box metrics must honor annotation
scope: exhaustive pages support precision and recall, while selected-only pages
support checked recall only.

## Evidence

Local measurements:

- The clear 52-control panel has complete detection coverage, zero false
  positives, state macro-F1 1.0, and label-association F1 0.9903.
- The control stage costs p50 11.82 ms and p95 25.08 ms per page. A heavyweight
  model is not justified by the current measured failure.
- PyLaia recovered 0 of 46 clinical handwriting fields exactly at 94.58% CER.
- TrOCR-base recovered 1 of 46 exactly at 82.51% CER and 136.23% WER, despite
  5.47% CER on the 30-line IAM health check.
- Manual review found corrupted medication names, dates, measurements, and fax
  identifiers. These are unsafe silent substitutions.

Primary external sources, accessed 2026-09-02:

- [RescueOMR](https://github.com/EuracBiomedicalResearch/RescueOMR) masks
  orthogonal checkbox borders and retains unknown, empty, checked, and filled
  states. Its AGPL and template-specific implementation make it recipe-only.
- [Clinical checkbox detection paper](https://arxiv.org/abs/2504.20220) supports
  localized checkbox clusters and contextual crops, but evaluates private forms.
  The associated [repository](https://github.com/ReMeDi-Blut/Checkbox-Detection-in-Clinical-Documents)
  does not provide a clear top-level license or a reusable detector artifact.
- [LynnHaDo checkbox detector](https://github.com/LynnHaDo/Checkbox-Detection)
  uses copy-paste synthetic marks and a human-annotated validation set. Its AGPL
  YOLO dependency, incomplete training artifacts, and limited reported metrics
  make it recipe-only.
- [MobileNetV3-Small](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.mobilenet_v3_small.html)
  is a possible three-class crop-classifier fallback at 2.54M parameters and
  0.06 GFLOPS, trained from scratch on owned data.
- [TrOCR-small-stage1](https://huggingface.co/microsoft/trocr-small-stage1) is a
  62M single-line recognizer from Microsoft. Its model card does not provide a
  sufficiently complete checkpoint license and pretraining-data rights audit.
- [Florence-2-base](https://huggingface.co/microsoft/Florence-2-base) is a
  0.23B Microsoft model under MIT and supports OCR with regions, but reports no
  handwriting-specific result.
- [PARSeq](https://github.com/baudm/parseq) is an Apache-2.0, 23.8M scene-text
  architecture. Its public weights are not clinical handwriting specialists.
- [Granite Vision 3.3 2B](https://huggingface.co/ibm-granite/granite-vision-3.3-2b)
  is Apache-2.0 but too heavy for a crop route without evidence of handwriting
  gains.

## Promotion rules

Adopt deterministic control recovery only if it gains at least 5 absolute state
macro-F1 points on an exhaustive degraded panel, keeps false-selected rate at or
below 0.5%, does not reduce label-association F1, adds no more than two false
proposals per page, preserves the clear panel, and adds less than 10 ms to p95.

Adopt a handwriting specialist only if strict field exact match reaches 80%, CER
is at most 10%, a paired 95% interval for CER improvement excludes zero, manual
review finds no critical identifier, medication, date, or measurement
substitutions, and the failure-inclusive latency budget still passes.

## Data and training guidance

If a learned control classifier becomes necessary, split by document family and
template rather than by crop. Include empty and selected squares, radio rings,
dots, X marks, ticks, filled and overwritten boxes, broken borders, rule
crossings, blur, compression, rotation, erosion, dilation, faint scans, and
dense-table negatives. Fit abstention thresholds only on development families.

For every specialist decision, retain the source crop, source bounding box, base
text, specialist text, model revision, confidence, disagreement, and final
selection. Never silently replace a clinical literal using language plausibility.
