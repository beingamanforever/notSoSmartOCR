# Checkbox specialist evaluation

Accessed 2026-09-01.

## Decision

Keep the geometric control stage as review evidence, not an unattended
production extractor. It adds pixel-space state, label evidence, and provenance
without a model call. The frozen clear panel is strong, but manual review of
dense ruled grids found false checkbox proposals. The broader panels are not
fully annotated, so neither recall nor false-positive rate is known.

This is still a measured improvement over the prior cascade, which emitted no
structured controls and silently accepted the supplied checkbox failures. It
does not justify accepting a page because the stage returned many controls.
Ambiguous controls, missing labels, isolated groups, and dense-grid output must
route to review until the broader panel is completely annotated.

## Method

- Threshold the page with Otsu and a faint-line view.
- Keep near-square, four-corner contours with an enclosed interior.
- Reject contours joined horizontally to text and runs of character-entry boxes.
- Classify the cleaned interior as selected, unselected, or ambiguous.
- Link each control to the nearest readable OCR region on the same line.
- Preserve the control box, state, label evidence ID, ink ratio, and method.
- Route ambiguous, unlabeled, or groups smaller than six controls to review.

No learned model or backbone is used. The implementation uses OpenCV contour
and threshold operations. OpenCV 4.5 and later are Apache-2.0 licensed according
to the [official license page](https://opencv.org/license/). The contour design
follows the hierarchy semantics in the [official shape documentation](https://docs.opencv.org/4.13.0/d3/dc0/group__imgproc__shape.html).

## Frozen manual panel

The panel contains 52 manually verified controls from two supplied private hard
pages: all 37 controls on the plan-of-care page and the 15 controls in the header
and intervention section of the physical-therapy progress page. Raw private
images and annotations remain outside Git.

| Metric | Result | Denominator |
|---|---:|---:|
| Detection coverage | 100.00% | 52 / 52 controls |
| False positives | 0 | 52 predicted controls |
| Selected-class F1 | 1.0000 | 16 selected controls |
| Unselected-class F1 | 1.0000 | 36 unselected controls |
| State macro-F1 | 1.0000 | 2 classes |
| Label-association precision | 1.0000 | 51 predicted edges |
| Label-association recall | 0.9808 | 52 gold edges |
| Label-association F1 | 0.9903 | 52 gold edges |

The single missing label edge is an unchecked header control whose base OCR did
not produce readable label evidence. The control is retained as unresolved and
routes the page to review. Manual overlay inspection found correct boxes and
states for all 52 controls.

## Broader hard-track behavior

On 21 supplied checkbox-focused pages, the stage produced 664 control regions.
Sixteen states were ambiguous and 40 controls lacked readable label evidence.
The earlier policy routed 18 pages to review. These counts measure routing
behavior, not ground-truth recall, because the full 21-page set is not completely
annotated. Manual review found that dense table and OASIS rules can be mistaken
for controls, so the unannotated count cannot be interpreted as coverage.

In the final 41-page specialist run, the stage returned 832 control regions and
the combined policy routed all 41 pages to review. That result closes the silent
acceptance failure but does not establish the correctness of those 832 regions.

Warm CPU latency over five passes of all 21 pages, after one warm-up pass:

| Metric | Result |
|---|---:|
| Samples | 105 pages |
| p50 | 11.82 ms/page |
| p95 | 25.08 ms/page |
| Maximum | 26.63 ms/page |

## Remaining risks

- Label quality cannot exceed the base OCR text. The relationship is preserved,
  but a misspelled OCR label remains misspelled.
- Very small or line-connected controls can be missed. The review policy lowers
  silent-accept risk but does not prove exhaustive coverage.
- Checkbox-to-label evaluation currently covers two clear layouts. Add complete
  annotations for degraded OASIS, fax, and handwritten forms before broadening
  the claim or allowing any control-bearing page to be accepted locally.
