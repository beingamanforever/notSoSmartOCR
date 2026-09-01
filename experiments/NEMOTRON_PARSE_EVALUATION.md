# NVIDIA Nemotron Parse 2.0 evaluation

Date: 2026-09-01

## Decision

Reject NVIDIA Nemotron Parse 2.0 as a handwriting or unresolved-page route.
Keep its isolated adapter for research, but do not register it in production or
merge its text into OCR output. The completed private run had 5 of 46
punctuation-insensitive exact fields, 117.49% CER, 108.33% WER, and one retained
failure.

## Adapter

`NemotronParseReader` follows NVIDIA's published prompt and tag grammar. It
preserves generated markdown, semantic class, reading order, normalized and
source-image boxes, model lineage, licenses, and prompt. The model does not
publish region confidence, so the adapter records `None`. Truncated tag streams,
unknown residual output, invalid geometry, image failures, initialization
failures, and prediction failures remain explicit.

The runner is lazy, pinned, and local-only by default. A two-argument generator
can inject output from an isolated Transformers or vLLM service without
changing parsing. No production registration or merge behavior was added.

## Origin and license review

- The official Parse 2.0 card identifies NVIDIA as publisher, a NVIDIA C-RADIO
  ViT-H vision encoder, and a 10-block mBART decoder.
- The Parse repository applies OpenMDW-1.1 to binary model and source files and
  CC-BY-4.0 to its tokenizer.
- NVIDIA publishes the named C-RADIO backbone under the NVIDIA Open Model
  License Agreement. Facebook Research's upstream fairseq mBART implementation
  is MIT, but that does not change the distributed Parse artifact license.

The identified inference components are NVIDIA and Facebook Research rather
than Chinese-origin models or backbones. Technical eligibility does not imply
production or commercial approval.

Primary sources, accessed 2026-09-01:

- [Nemotron Parse 2.0 model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-2.0)
- [Parse configuration](https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-2.0/blob/main/config.json)
- [Parse postprocessing](https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-2.0/blob/main/postprocessing.py)
- [Parse license](https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-2.0/blob/main/LICENSE)
- [NVIDIA C-RADIOv2-H model card](https://huggingface.co/nvidia/C-RADIOv2-H)
- [Facebook Research mBART](https://github.com/facebookresearch/fairseq/tree/main/examples/mbart)

## Fixed private evaluation

The panel contains 46 twice-reviewed handwritten fields from four generated
clinical-style pages. One ambiguous contact surname was excluded before the
run. The model revision was
`b6742064f4a8cf22a10383ece5e7fbead355ac04`. Inference ran locally on the
authorized A10G and all failures remained in the denominator.

Normalization applies Unicode NFKC, case folding, and alphanumeric token
extraction. The exact count is therefore punctuation-insensitive and should not
be interpreted as strict clinical field correctness.

| Measure | Result |
| --- | ---: |
| Attempted / covered / failed | 46 / 45 / 1 |
| Punctuation-insensitive exact | 5 / 46, 10.87% |
| Micro CER | 117.49% |
| Micro WER | 108.33% |
| Latency p50 / p95 / max | 1.015 / 1.320 / 72.835 s |
| Mean normalized gold / prediction length | 8.83 / 12.28 characters |

The single long failure remains in the latency and quality denominator. The
prediction length inflation and CER above 100% show that this structured-page
parser is poorly matched to isolated handwritten fields. Occasional exact
values do not offset the insertion and substitution failures.

## Comparison and claim boundary

On the same private 46-field panel, generic PyLaia and TrOCR were also rejected.
Parse has more punctuation-insensitive exact fields than either model, but much
worse CER than both and one 72.8-second failure. These systems use different
decoding contracts, so no model is promoted from the rank ordering.

This evaluation does not measure Parse on full pages, tables, or its intended
structured-document distribution. It is enough to reject the proposed
handwriting route, not to claim that the model has no other research value.

## Verification and evidence

- Adapter tests: `tests/test_nemotron_parse.py`
- Evaluator: `/opt/dlami/nvme/aman/notso-ocr-private/handwriting-panel-20260901/parse_handwriting_bench.py`
- Complete result: `/opt/dlami/nvme/aman/notso-ocr-results/parse-handwriting-20260901/results-complete.json`
- Adapter: `src/ocr_pipeline/nemotron_parse.py`
