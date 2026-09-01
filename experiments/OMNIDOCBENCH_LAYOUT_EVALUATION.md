# OmniDocBench layout evaluation

Accessed 2026-09-01.

## Protocol

The harness uses the official OmniDocBench `v1_5` layout adapter and
`mmeval==0.2.1` `COCODetection` implementation. It adds an explicit class-wise
precision, recall, and F1 calculation using score-descending, one-to-one
matching within each page and class at IoU 0.5. Official COCO mAP remains
unchanged.

The control panel contains 61 pages from the v1.5 base subset of the public
v1.6 release. It selects up to four English and four simplified Chinese pages
per document source. The panel covers all nine document sources, all five
layout types, 29 English pages, 32 Chinese pages, all ten official evaluation
classes, and 1,136 evaluated boxes. Missing or empty predictions remain in the
61-page denominator.

## Control results

| Control | Pages | Coverage | COCO mAP | COCO AP50 | Micro P/R/F1 | Macro P/R/F1 |
|---|---:|---:|---:|---:|---:|---:|
| Oracle | 61 | 1.0 | 1.0 | 1.0 | 1.0 / 1.0 / 1.0 | 1.0 / 1.0 / 1.0 |
| Empty | 61 | 0.0 | 0.0 | 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |

The official `mmeval` wrapper returns no metric keys when the complete result
is empty. The harness records that condition as `raw_result_empty: true` and
reports zero for metrics whose panel has ground truth. This makes the empty
control failure-inclusive without disguising the evaluator behavior.

## Research-only model result

Raw IBM Heron detections at threshold 0.6 covered all 61 pages. The released
`8f39ad3c0b4c58e9c2d2c84a38465abf757272d8` revision scored 0.342071 COCO mAP,
0.433447 AP50, and 0.640424 / 0.744718 / 0.688645 project micro precision,
recall, and F1. It produced 1,321 mapped boxes with p50 52.844 ms and p95
142.725 ms on the A10G. Nineteen raw detections had no official class mapping:
one unselected checkbox, 11 forms, and seven key-value regions.

This is a component-model result, not a deployable-stack result. Heron's
RT-DETRv2 lineage makes it research-only under the project origin rule. Manual
review found severe text over-segmentation on a blank ruled form, title and
text misses on a stitched two-page mathematics spread, and missing table
captions on a dense newspaper page.

The official v1.5 category config excludes 45 panel boxes: 23 `reference`
boxes because the config key is misspelled as `refernece`, 14
`figure_footnote` boxes because that mapped class is not evaluated, and eight
mask classes. The official adapter also does not apply the annotation-level
`ignore` flag. These behaviors are preserved and disclosed so local results
remain comparable to the official protocol.

## Commands

```bash
PYTHONPATH=src /private/tmp/omni-layout-venv313/bin/python experiments/omnidocbench_layout_benchmark.py data/public/omnidocbench-v1.6/OmniDocBench.json experiments/results/omnidocbench-layout-v1.5-balanced-61-oracle-v2.json --evaluator-root /private/tmp/omnidocbench-v1_5 --control oracle --base-per-language 4

PYTHONPATH=src /private/tmp/omni-layout-venv313/bin/python experiments/omnidocbench_layout_benchmark.py data/public/omnidocbench-v1.6/OmniDocBench.json experiments/results/omnidocbench-layout-v1.5-balanced-61-empty-v2.json --evaluator-root /private/tmp/omnidocbench-v1_5 --control empty --base-per-language 4
```

## License and use boundary

- [OmniDocBench evaluator](https://github.com/opendatalab/OmniDocBench/tree/v1_5): Apache-2.0.
- [OmniDocBench v1.6 dataset](https://huggingface.co/datasets/MinerU25Pro-NIPS26/OmniDocBench-v1.6): CC BY-NC 4.0, so this panel is research evaluation only.
- The harness is model-independent. A model and its backbone still require a
  separate origin and license review before deployment.
