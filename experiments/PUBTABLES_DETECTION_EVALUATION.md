# PubTables-1M table detection evaluation

Accessed and measured on 2026-09-01.

## Setup

- Dataset: `bsmock/pubtables-1m`, test split, revision
  `35b1c097807e0b07ec5313879b85956b7b3890db`,
  CDLA-Permissive-2.0.
- Panel: 60 deterministic, evenly spaced cases from the prepared public
  200-case panel. Every missing, invalid, failed, or abstained case remains in
  the denominator.
- Model: [Microsoft Table Transformer detection](https://huggingface.co/microsoft/table-transformer-detection),
  revision `2357cbe2b5a5d1c03e54f32764f06058933b65ab`, MIT.
- Inference source: [Microsoft Table Transformer](https://github.com/microsoft/table-transformer),
  revision `16d124f616109746b7785f03085100f1f6247575`, MIT.
- Hardware: NVIDIA A10G 24 GB. Confidence threshold: 0.5.
- Scoring: local protocol v1, score-descending one-to-one matching to the
  highest-IoU unmatched truth table. The local repository has no declared
  license for this scorer.

The tracked runner is `experiments/pubtables_detection_benchmark.py`. The A10G
command was:

```bash
PYTHONPATH=. python experiments/pubtables_detection_benchmark.py \
  /path/to/pubtables-public-200 /path/to/result.json \
  --tatr-checkpoint /path/to/pubtables1m_detection_detr_r18.pth \
  --source-root /path/to/table-transformer \
  --source-revision 16d124f616109746b7785f03085100f1f6247575 \
  --checkpoint-revision 2357cbe2b5a5d1c03e54f32764f06058933b65ab \
  --device cuda --limit 60
```

## Failure-inclusive results

| Measure | Result |
| --- | ---: |
| Attempted / valid / failed | 60 / 60 / 0 |
| Precision at IoU 0.50 | 0.983607 |
| Recall at IoU 0.50 | 1.000000 |
| F1 at IoU 0.50 | 0.991736 |
| Precision at IoU 0.75 | 0.983607 |
| Recall at IoU 0.75 | 1.000000 |
| F1 at IoU 0.75 | 0.991736 |
| Mean best IoU | 0.981263 |
| Latency p50 / p95 | 38.212 / 56.559 ms |
| Model load time | 4,156.228 ms |

The model found all 60 truth tables at both IoU thresholds. It produced one
extra detection, so the precision and F1 are below 1.0 even though recall is
perfect. This corrects the earlier recall-only probe, which selected only the
highest-scoring box and therefore could not count false positives.

## Manual review

The extra detection is in `PMC1414031_table_2`. The correct table is localized
at IoU 0.970220. The additional box is a narrow 9.6 px strip on the left page
edge, where the source crop visibly contains a clipped neighboring table
fragment. It is not a second usable table. The detector should stay unchanged
for now: a downstream minimum usable width or successful structure check can
reject this artifact without risking the measured recall.

This result supports Table Transformer as the current table-routing arm on
this public PubTables slice. It does not establish clinical-domain table
accuracy or end-to-end transcription quality; those remain separate pipeline
measurements.

The research environment emitted a SciPy and NumPy compatibility warning plus
torchvision deprecation warnings while loading the pinned upstream code. They
did not cause a measured failure, but the serving environment should use a
compatible dependency set before production deployment.
