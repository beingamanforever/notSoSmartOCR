# Phi-4 handwriting adapter v4

Date: 2026-09-02

## Decision

Keep the v4 Microsoft Phi-4 Multimodal decoder-LoRA as a selective handwriting
crop specialist for review. Do not use it as a full-page parser, handwriting
detector, or automatic replacement for the incumbent reader.

Supervised adaptation produced a large paired improvement on
already-localized real crops. It did not satisfy the safety threshold on the
fixed C14 holdout, and the experiment does not measure localization or
end-to-end page recall. The useful result is a stronger handwriting challenger
inside the evidence-preserving cascade, not a completed handwriting solution.

## Training run

The run updated only the checkpoint's released vision-conditioned decoder LoRA
for `microsoft/Phi-4-multimodal-instruct` at pinned revision
`93f923e1a7727d1c4f446756212d9d3e8fcc5d81`.

| Property | Measured or recorded value |
| --- | ---: |
| Training fields | 356 |
| Training families | 68 |
| Development fields | 47 |
| Development families | 6 |
| C14 fields in train or development | 0 |
| Trainable tensors | 256 |
| Trainable parameters | 369,098,752 |
| Effective batch | 8 |
| Precision and attention | BF16, SDPA |
| Training runtime | 1,028.09 s, or 17.13 min |
| Training throughput | 1.731 samples/s |
| Reported train loss | 1.580020 |
| Save and reload prediction parity | Passed |

The five-epoch job used learning rate `3e-5`, microbatch 1, gradient
accumulation 8, `dynamic_hd=1`, a 1,024-token limit, no quantization, and seed
17. The runtime-ready adapter is about 704 MiB. Adapter artifacts remain on the
authorized private GPU host and are intentionally excluded from Git.

## Family-held development result

The development set is real-only and family-disjoint from training. Its 47
examples comprise 43 resolved text targets and 4 abstention targets. The exact
rate emitted by the evaluator uses the 43 resolved targets as its denominator.

| Metric | Stock Phi-4 | v4 adapter |
| --- | ---: | ---: |
| Resolved exact | 11 / 43, 25.6% | 37 / 43, 86.0% |
| Abstentions correct | 0 / 4 | 3 / 4 |
| CER | 2.529630 | 0.200000 |
| Hallucinated-character rate | 2.448148 | 0.029630 |
| Missed-character rate | 0.003704 | 0.044444 |
| Critical substitutions | 32 | 6 |
| p50 latency | 650.277 ms | 327.249 ms |
| p95 latency | 1,920.376 ms | 711.630 ms |
| Failed fields | 0 | 0 |

The lower adapter latency was measured after a warmup for each arm in one
resident process. It is not a concurrency or end-to-end service benchmark.

## Fixed C14 holdout

C14 contains 44 real handwriting fields. Each field was evaluated in native
and scaled views, producing 88 paired inferences per arm. C14 was not used for
training, development, or augmentation. It was opened only for the fixed
post-training evaluation.

| Metric | Stock Phi-4 | v4 adapter |
| --- | ---: | ---: |
| Exact | 32 / 88, 36.4% | 50 / 88, 56.8% |
| CER | 0.401471 | 0.135294 |
| Hallucinated-character rate | 0.301471 | 0.044118 |
| Missed-character rate | 0.020588 | 0.029412 |
| Critical substitutions | 46 / 78 | 28 / 78 |
| p50 latency | 455.314 ms | 445.145 ms |
| p95 latency | 733.867 ms | 611.376 ms |
| Failed fields | 0 | 0 |

The native adapter view reached 24/44 exact and the scaled view reached 26/44.
This misses the predeclared native threshold of 36/44. The adapter reduced
insertions and substitutions substantially, but its 28 remaining critical
substitutions and slightly higher missed-character rate make automatic
promotion unsafe.

## Runtime boundary

The adapter accepts an existing crop and returns literal handwriting. It does
not locate handwriting, infer page structure, read tables, or render a page.
The crop benchmark therefore says nothing about detection recall. Earlier
proposal evaluation found that a generic handwriting detector added hundreds
of unmatched regions for one extra C14 recovery, so it was rejected.

Use the adapter only when an independently configured stage has already
localized a likely handwriting crop. Preserve the incumbent output, adapter
output, crop geometry, provider, and disagreement. Promote neither output
without the pipeline's existing evidence rule. Empty fields, uncertain crops,
and critical clinical literals stay review-routed.

## What remains unverified

- End-to-end page recall with the adapter enabled.
- Handwriting localization on the full hard-page set.
- Paired latency and peak memory through the production endpoint.
- Family-clustered uncertainty on a larger real holdout.
- Zero critical substitutions, which the current C14 result does not meet.

No claim is made that the adapter beats a frontier model or improves the whole
OCR pipeline. The measured claim is limited to paired recognition of already
localized crops.

## Evidence boundary

The aggregate metrics above were read from the completed private-safe run
summary and paired family-held and C14 evaluation summaries on the authorized
GPU host. Those summaries persist no private OCR text, predictions, exception
messages, or case identifiers. Raw images, labels, and row-level predictions
remain outside the repository.

Architecture references:

- [Phi-4 Multimodal model card](https://huggingface.co/microsoft/Phi-4-multimodal-instruct/blob/main/README.md)
- [Official Phi-4 vision fine-tuning example](https://huggingface.co/microsoft/Phi-4-multimodal-instruct/blob/main/sample_finetune_vision.py)
- [Phi-4 Multimodal technical report](https://arxiv.org/abs/2503.01743)
