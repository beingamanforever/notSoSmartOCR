# Handwriting specialist evaluation

Date: 2026-09-01

## Decision

Do not integrate generic PyLaia, TrOCR, Doc-UFCN, or same-model multiview
consensus into the clinical pipeline.
PyLaia and TrOCR are healthy on the public IAM control but fail the reviewed
clinical form crops. Doc-UFCN detects generic text lines and floods clinical
forms with printed-text and rule proposals.

Handwriting remains unresolved and review-routed. A larger model is not
justified by the current evidence: TrOCR base improves character error over
PyLaia but still corrupts identifiers, measurements, and medication names.

## Protocol

- Private panel: 46 twice-reviewed handwritten value crops from four generated
  clinical-style pages. One ambiguous surname was excluded before scoring.
- Public health check: fixed rows 0 through 29 of the pinned IAM-line test split.
- PyLaia: grayscale, fixed height 128 pixels, aspect ratio preserved, literal
  greedy CTC, and no language model.
- TrOCR base: standard 384 by 384 processor, batch 8, FP32, deterministic greedy
  decoding with one beam and at most 64 new tokens.
- Metrics: failure-inclusive micro CER and WER, strict normalized exact match,
  punctuation-insensitive exact match, token recall, token precision, failures,
  and resident latency where measured.
- Manual review: 12 private crop and output pairs plus four Doc-UFCN overlays.
- Privacy: private images stayed on local storage and the authorized private
  A10G.

Strict exact match case-folds and normalizes whitespace but preserves visible
punctuation. The punctuation-insensitive score is reported only to explain the
legacy evaluator and must not be treated as clinical correctness.

## Results

| Model and panel | Items | Strict exact | Punctuation-insensitive exact | Token recall / precision | CER | WER | Failures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PyLaia, private | 46 | 0 / 46 | 1 / 46 | 2.08% / 3.70% | 94.58% | 117.39% | 0 |
| TrOCR base, private | 46 | 1 / 46 | 4 / 46 | 22.92% / 27.16% | 82.51% | 136.23% | 0 |
| PyLaia, IAM test | 30 | 2 / 30 | 2 / 30 | 74.92% / 76.78% | 7.60% | 27.84% | 0 |
| TrOCR base, IAM test | 30 | 6 / 30 | 9 / 30 | 86.71% / 88.58% | 5.47% | 16.75% | 0 |

TrOCR resident private inference took 28.68 ms/item wall time. Generation
p50 and p95 were 31.42 and 33.98 ms/item at batch 8. Cached load time was 1.02
seconds and peak allocated and reserved GPU memory were 1,473.8 and 1,580 MiB.
These are resident single-process measurements, not a concurrency benchmark.

The strong public controls rule out corrupt checkpoints or broken decoders.
The private domain mismatch is material. Manual review found `8.3` read as
`813.`, `130 68` read as `130 168`, medication names absorbing adjacent printed
labels, and numeric fax identifiers gaining or changing punctuation. Only
`Hypertension` was strictly exact for TrOCR. PyLaia also changed clear numbers
and medication names, and its confidence did not separate severe failures.

Doc-UFCN loosely overlapped 41 of 46 reviewed fields, but 120 of 180 detections
fell outside reviewed handwriting. Overlays showed detections on printed
labels, logos, and horizontal rules plus five missed light fields. It is not a
safe handwriting router.

## Same-model multiview rejection

Nemotron OCR v2 was also tested on the same fixed 46 fields with four local
views: grayscale at 1x and 2x, line removal at 2x, and Otsu thresholding at 2x.
The whole document pipeline processed each crop so this experiment measures the
actual reader and routing stack, not an isolated decoder.

| Selection | Strict exact | Punctuation-insensitive exact | CER | WER |
| --- | ---: | ---: | ---: | ---: |
| Grayscale 1x | 7 / 46 | 11 / 46 | 67.00% | 81.25% |
| Grayscale 2x | 9 / 46 | 12 / 46 | 74.88% | 87.50% |
| Majority consensus with 1x fallback | 7 / 46 | 10 / 46 | 70.94% | 84.38% |

An unattainable ground-truth oracle over the four views reached only 16 of 46
punctuation-insensitive exact fields, 52.71% CER, and 69.79% WER. The best
strict output present in any view existed for only 11 of 46 fields. Four-view
processing took 2.538 seconds p50 and 4.592 seconds p95 per field through the
resident endpoint.

The failure pattern is clinically important. Scaling recovered `8.3` from `8`,
but changed the exact fax date `3-6-25` to `3-6-22` and the exact service date
`3/18/2025` to `3/18/2005`. Majority selection replaced one exact service date
with an empty result because two thresholded views agreed on missing text. A
full-page two-scale stage on the supplied homecare form added 0.495 seconds to
the warm request and changed no region. The candidate was removed rather than
adding a zero-gain production path.

These results reject the hypothesis that correlated preprocessing views provide
independent support. They remain useful as review evidence, but must not select
clinical literals without a separately validated field detector and specialist.

## Provenance and license boundary

- PyLaia IAM revision
  `9c22c3e4ae7e8455a7fb7c6784a7c372c47270db`, with MIT code and model card.
- TrOCR base revision
  `eaacaf452b06415df8f10bb6fad3a4c11e609406`, published under Microsoft.
  Its Hugging Face card declares MIT but says Hugging Face wrote the card.
- TrOCR large revision
  `e68501f437cd2587ae5d68ee457964cac824ddee` has no license field and was not
  downloaded or evaluated.
- IAM is restricted by its official site to non-commercial research. Both
  handwriting checkpoints are IAM-derived, so commercial checkpoint use needs
  legal review even if the surrounding code is permissive.
- Doc-UFCN revision
  `3338539ef366fa0d963e87de675bc431183a7c80`, MIT.

Primary sources, accessed 2026-09-01:

- [PyLaia IAM model card](https://huggingface.co/Teklia/pylaia-iam)
- [TrOCR base model card](https://huggingface.co/microsoft/trocr-base-handwritten)
- [TrOCR large model card](https://huggingface.co/microsoft/trocr-large-handwritten)
- [Microsoft TrOCR repository](https://github.com/microsoft/unilm/tree/master/trocr)
- [Official IAM database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database)
- [Doc-UFCN model card](https://huggingface.co/Teklia/doc-ufcn-generic-historical-line)

## Next challenger

Test only a license-cleared model with clinical-form adaptation or a verified
handwriting classifier and tightly owned single-line crops. Promotion requires
at least 80% strict field exact match, CER at most 10%, zero critical identifier
or medication substitutions during manual review, zero silent failures, and a
matched improvement over the contextual reader on the same 46 crops. Preserve
the source crop, geometry, base output, specialist output, and disagreement;
never apply semantic correction.

Evidence is retained in the ignored local benchmark under
`internal-clinical-ocr-benchmark/hard-cases-20260901/outputs/handwriting-panel`:

- Nemotron multiview outputs: `nemotron_views_20260901.json`
- PyLaia score: `specialist-controls/private_score.json`
- TrOCR score: `specialist-controls/trocr-base-handwritten-score.json`
- Public controls: `specialist-controls/public_score.json`
- Doc-UFCN detections: `specialist-controls/docufcn_score.json`
