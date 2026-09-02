# Clinical control specialist options

Date: 2026-09-02

## Decision

Keep Nemotron and the text pipeline unchanged. The checked-control regression is
inside the deterministic control stage, so replacing the page OCR backend would
add cost without targeting the failure.

First measure the raw high-recall proposal union before text, label, or state
filters. If it covers at least 18 of 19 checked controls and 57 of 63 total
controls on the four-page fail-fast set, validate those proposals with a small
MobileNetV3-Small binary crop classifier. Low-score or disputed proposals remain
ambiguous. They must never be converted silently to unchecked controls.

## Minimal experiment

1. Add bounding boxes to the exhaustively reviewed control references.
2. Compare raw v10-style square and anchored-mark proposals with current v11 at
   IoU 0.5. Report checked recall, total recall, proposals per page, false
   proposals per page, latency, and memory.
3. If proposal recall passes, train a `control` versus `not_control` validator on
   128 by 128 crops with 3x context. Use document-family splits, all positives,
   and three to five hard negatives per positive. Exclude proposals with IoU in
   the uncertain 0.1 to 0.5 band.
4. Compare raw proposals, proposals plus validator, and v11 on at least 30
   family-held-out pages. Inspect every false positive and false negative.

Adopt only if checked recall is at least 17 of 19, total recall is at least 57 of
63, exhaustive precision is at least 0.90, state macro-F1 and label association
do not regress, no unsupported selected state is emitted, and whole-pipeline p95
increases by no more than 5 percent.

If raw proposal recall fails, use `pdf-net/ffdetr-filled` at 1024 pixels only as
an isolated proposal comparator. Do not add a full detector dependency before
the existing proposal oracle is measured.

## Why MobileNetV3-Small

MobileNetV3-Small is a Google-origin, peer-reviewed low-resource classifier with
a maintained torchvision implementation. The documented model has about 2.54M
parameters, 0.06 GFLOPS at 224 pixels, and a 9.8 MB weight file. Batched page
crops should therefore add materially less cost than a second OCR or VLM pass.

CommonForms and the filled-form FFDetr pilot support two reusable ideas: small
controls need higher-resolution detection, and completed forms need filled-form
training rather than blank-widget supervision. They do not establish checked
state accuracy on real clinical scans.

## Evidence

All sources accessed 2026-09-02.

- [CommonForms paper](https://arxiv.org/abs/2509.16506)
- [CommonForms repository](https://github.com/jbarrow/commonforms)
- [FFDNet-L model card](https://huggingface.co/jbarrow/FFDNet-L)
- [FFDetr model card](https://huggingface.co/jbarrow/FFDetr)
- [Filled-form FFDetr pilot](https://huggingface.co/pdf-net/ffdetr-filled)
- [RF-DETR repository](https://github.com/roboflow/rf-detr)
- [Wendy checkbox detector](https://huggingface.co/wendys-llc/checkbox-detector)
- [MobileNetV3 paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Howard_Searching_for_MobileNetV3_ICCV_2019_paper.pdf)
- [Torchvision MobileNetV3-Small](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.mobilenet_v3_small.html)
- [Torchvision license](https://github.com/pytorch/vision/blob/main/LICENSE)

## Current evidence boundary

The four existing hard-page references name control labels and states but lack
control boxes. Existing label-only matching covers too few controls to establish
proposal recall. The proposal-oracle gate therefore requires local box
annotation before classifier work. Current v11 remains an experiment, not a
replacement for the measured v10 result.
