"""Detect table crops with IBM's Heron-101 general layout detector.

Table Transformer is trained on a narrower distribution and misses full-page
financial tables (domain shift). Heron-101 (RT-DETRv2, Apache-2.0) is a general
17-class layout detector, so it fires on those pages instead. Its output feeds
`tables.py`'s existing layout-proposal channel as plain crops - the detector's
score is a layout confidence, not a recognition confidence.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Sequence
from functools import cache
from typing import Any

from PIL import Image

from .contracts import BoundingBox
from .providers import ReaderError

DEFAULT_MODEL_ID = "ds4sd/docling-layout-heron-101"
DEFAULT_MODEL_REVISION = "2e4993cf6bb211112084a2f80938f26138008917"
MODEL_ARCHITECTURE = "RTDetrV2ForObjectDetection"
MODEL_LICENSE = "apache-2.0"
MODEL_ORIGIN = "IBM Research"
# From id2label in https://huggingface.co/ds4sd/docling-layout-heron-101/raw/main/config.json
# (id 8 is "table"). id 11 "document_index" is Docling's table-of-contents label, not a
# data table, so it is excluded.
TABLE_LABELS = frozenset({"table"})
# A glyph-sized control marks state inside text, so it competes only with controls.
CONTROL_LABELS = frozenset({"checkbox_selected", "checkbox_unselected"})
# SAHI's default slice overlap ratio.
SLICE_OVERLAP = 0.2
# SAHI's greedy merge threshold (intersection over the smaller box) for one element.
SAME_CONTROL_OVERLAP = 0.5
# Standard NMS IoU for one checkbox proposed twice (page pass and tiles, either state).
SAME_CONTROL_IOU = 0.5
# Where small controls live: tiles overlapping these proposals are sliced.
CONTROL_REGION_LABELS = CONTROL_LABELS | {"form", "key_value_region", "table"}
# Docling's own threshold for two proposals being the same element.
SAME_ELEMENT_IOU = 0.8
# Classes added by the fine-tuned Heron. Docling has no label for them, so they are
# resolved as their own group, as pictures, and never compete with text.
NEW_LABELS = frozenset({"signature", "hand_mark"})

Detection = dict[str, Any]


def resolve_layout(
    detections: Sequence[Detection],
    width: int,
    height: int,
    cells: Sequence[dict[str, Any]] | None = None,
) -> list[Detection]:
    """Resolve overlapping Heron proposals with Docling's official postprocessor.

    Table and picture children are read by their container. On a born-digital page
    picture children stay top-level: their text comes exactly from the text layer.
    A born-digital page's text cells go through Docling's own cell assignment, as in
    its standard PDF pipeline: clusters regrow to their cells, unassigned cells become
    text clusters, and each cluster carries its cells' text as `cell_text`.
    Checkboxes are resolved apart from Docling's overlap groups (`_resolve_controls`).
    """
    added = [i for i, item in enumerate(detections) if item["label"] in NEW_LABELS]
    postprocessor = _group_union_postprocessor()
    marks = _postprocess(postprocessor, detections, added, width, height, marks=True)
    detections = [item for item in detections if item["label"] not in NEW_LABELS]
    if cells:
        return _postprocess_with_cells(detections, cells, width, height) + marks
    # A control covering a whole line is the same element as that line's text.
    controls = [
        i
        for i, item in enumerate(detections)
        if item["label"] in CONTROL_LABELS
        and not any(
            other["label"] not in CONTROL_LABELS
            and _iou(item["box"], other["box"]) > SAME_ELEMENT_IOU
            for other in detections
        )
    ]
    content = [i for i in range(len(detections)) if i not in set(controls)]
    resolved = [
        *_postprocess(postprocessor, detections, content, width, height),
        *_resolve_controls([detections[i] for i in controls]),
    ]
    return resolved + marks


def _resolve_controls(controls: Sequence[Detection]) -> list[Detection]:
    """Class-aware NMS over checkbox proposals, both states pooled.

    Docling groups overlapping clusters transitively and keeps the larger box within a
    0.05 confidence gap, so a proposal spanning two adjacent options swallowed both.
    A box holding two or more smaller kept boxes is such a span and is dropped.
    """
    from docling.utils.layout_postprocessor import LayoutPostprocessor
    from docling_core.types.doc import DocItemLabel

    kept: list[Detection] = []
    for item in sorted(controls, key=lambda item: -item["score"]):
        threshold = LayoutPostprocessor.CONFIDENCE_THRESHOLDS[DocItemLabel(item["label"])]
        if item["score"] >= threshold and all(
            _iou(item["box"], other["box"]) < SAME_CONTROL_IOU for other in kept
        ):
            kept.append(item)
    return [
        item
        for item in kept
        if sum(
            other is not item and _overlap_of_smaller(item["box"], other["box"]) >= 0.8
            and _area(other["box"]) < 0.6 * _area(item["box"])
            for other in kept
        )
        < 2
    ]


def _area(box: Sequence[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _overlap_of_smaller(first: Sequence[float], second: Sequence[float]) -> float:
    width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    smaller = min(
        (first[2] - first[0]) * (first[3] - first[1]),
        (second[2] - second[0]) * (second[3] - second[1]),
    )
    return width * height / smaller if smaller > 0 else 0.0


def _iou(first: Sequence[float], second: Sequence[float]) -> float:
    width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    overlap = width * height
    union = (
        (first[2] - first[0]) * (first[3] - first[1])
        + (second[2] - second[0]) * (second[3] - second[1])
        - overlap
    )
    return overlap / union if union > 0 else 0.0


def _postprocess(
    postprocessor: type,
    detections: Sequence[Detection],
    indices: list[int],
    width: int,
    height: int,
    *,
    marks: bool = False,
) -> list[Detection]:
    from docling.datamodel.base_models import Cluster, Page
    from docling.datamodel.pipeline_options import BaseLayoutPostprocessorOptions
    from docling_core.types.doc import BoundingBox as Box
    from docling_core.types.doc import DocItemLabel, Size

    clusters = [
        Cluster(
            id=index,
            label=DocItemLabel.PICTURE
            if detections[index]["label"] in NEW_LABELS
            else DocItemLabel(detections[index]["label"]),
            confidence=detections[index]["score"],
            bbox=Box(
                l=detections[index]["box"][0],
                t=detections[index]["box"][1],
                r=detections[index]["box"][2],
                b=detections[index]["box"][3],
            ),
        )
        for index in indices
    ]
    page = Page(page_no=0, size=Size(width=width, height=height))
    processor = postprocessor(
        page, clusters, BaseLayoutPostprocessorOptions(skip_cell_assignment=True)
    )
    # Marks use picture overlap resolution, but are not page-background pictures.
    kept = (
        processor._remove_overlapping_clusters(
            [
                c
                for c in clusters
                if c.confidence >= processor.CONFIDENCE_THRESHOLDS[c.label]
            ],
            "picture",
        )
        if marks
        else processor.postprocess()
    )
    if marks:
        kept = processor._sort_clusters(kept, mode="id")
    # A picture is transcribed whole, so the text regions inside it are its own content,
    # as in Docling; reading them too printed their text twice.
    return [
        {**detections[cluster.id], "box": (cluster.bbox.l, cluster.bbox.t, cluster.bbox.r, cluster.bbox.b)}
        for cluster in kept
    ]


def _postprocess_with_cells(
    detections: Sequence[Detection],
    cells: Sequence[dict[str, Any]],
    width: int,
    height: int,
) -> list[Detection]:
    from docling.datamodel.base_models import Cluster, Page
    from docling.datamodel.pipeline_options import BaseLayoutPostprocessorOptions
    from docling.models.stages.page_assemble.page_assemble_model import (
        PageAssembleModel,
        PageAssembleOptions,
    )
    from docling_core.types.doc import BoundingBox as Box
    from docling_core.types.doc import DocItemLabel, Size
    from docling_core.types.doc.page import (
        BoundingRectangle,
        SegmentedPdfPage,
        TextCell,
    )

    text_cells = [
        TextCell(
            index=index,
            text=cell["text"],
            orig=cell["text"],
            rect=BoundingRectangle.from_bounding_box(
                Box(l=left, t=top, r=right, b=bottom)
            ),
            from_ocr=False,
        )
        for index, cell in enumerate(cells)
        for left, top, right, bottom in [cell["bbox"]]
    ]
    clusters = [
        Cluster(
            id=index,
            label=DocItemLabel(item["label"]),
            confidence=item["score"],
            bbox=Box(
                l=item["box"][0], t=item["box"][1], r=item["box"][2], b=item["box"][3]
            ),
        )
        for index, item in enumerate(detections)
    ]
    page = Page(page_no=0, size=Size(width=width, height=height))
    # Only textline cells are read by the postprocessor.
    page.parsed_page = SegmentedPdfPage.model_construct(
        textline_cells=text_cells, char_cells=[], word_cells=[]
    )
    kept = _group_union_postprocessor()(
        page, clusters, BaseLayoutPostprocessorOptions(skip_cell_assignment=False)
    ).postprocess()
    assemble = PageAssembleModel(options=PageAssembleOptions())
    resolved = []
    for cluster in kept:
        members = cluster.children if cluster.label == DocItemLabel.PICTURE else []
        for item in (cluster, *members):
            source = detections[item.id] if item.id < len(detections) else {}
            lines = [cell.text.strip() for cell in item.cells if cell.text.strip()]
            resolved.append(
                {
                    **source,
                    "label": str(item.label.value),
                    "score": float(item.confidence),
                    "box": (item.bbox.l, item.bbox.t, item.bbox.r, item.bbox.b),
                    **({"cell_text": assemble.sanitize_text(lines)} if lines else {}),
                }
            )
    return resolved


@cache
def _group_union_postprocessor() -> type:
    from docling.utils.layout_postprocessor import LayoutPostprocessor
    from docling_core.types.doc import BoundingBox as Box

    class GroupUnionPostprocessor(LayoutPostprocessor):
        def _set_cluster_children(self, cluster, children):
            original = cluster.bbox
            super()._set_cluster_children(cluster, children)
            if cluster.label in self.CONTAINER_TYPES:
                # Text children do not describe all the ink in a source form.
                cluster.bbox = Box(
                    l=min(original.l, cluster.bbox.l),
                    t=min(original.t, cluster.bbox.t),
                    r=max(original.r, cluster.bbox.r),
                    b=max(original.b, cluster.bbox.b),
                )

        # Docling regrows a winner from its merged text cells. Image-only input has
        # no cells, so the winner takes its overlap group's union instead.
        def _select_best_cluster_from_group(self, group_clusters, params):
            best = super()._select_best_cluster_from_group(group_clusters, params)
            if (
                not self.options.skip_cell_assignment
                and best.label not in self.CONTAINER_TYPES
            ):
                return best
            best.bbox = Box(
                l=min(item.bbox.l for item in group_clusters),
                t=min(item.bbox.t for item in group_clusters),
                r=max(item.bbox.r for item in group_clusters),
                b=max(item.bbox.b for item in group_clusters),
            )
            return best

    return GroupUnionPostprocessor


Detector = Callable[[Image.Image], list[Detection]]


class HeronTableDetector:
    """Table-crop proposals from the Heron-101 general layout detector."""

    name = "heron-101"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        revision: str | None = DEFAULT_MODEL_REVISION,
        device: str = "cuda",
        score_threshold: float = 0.5,
        detector: Detector | None = None,
    ) -> None:
        if not 0 <= score_threshold <= 1:
            raise ValueError("score_threshold must be from 0 to 1")
        self.model_id = model_id
        self.revision = revision
        self.device = device
        self.score_threshold = score_threshold
        self._detector = detector
        self._lock = threading.Lock()

    def detect(self, image: Image.Image) -> list[BoundingBox]:
        return [
            _clamped_box(item["box"], *image.size)
            for item in self.detect_elements(image)
            if item["label"] in TABLE_LABELS
        ]

    def detect_elements(self, image: Image.Image) -> list[Detection]:
        """Return all learned layout classes using the same detector pass."""
        width, height = image.size
        with self._lock:
            detector = self._get_detector()
            try:
                detections = detector(image)
            except Exception as error:
                raise ReaderError("heron_detect_failed", str(error)) from error

        boxes = []
        for item in detections:
            score = float(item.get("score", 0.0))
            if not math.isfinite(score) or not all(
                math.isfinite(part) for part in item.get("box") or ()
            ):
                raise ReaderError(
                    "heron_invalid_detection", f"nonfinite detection {item}"
                )
            if score < self.score_threshold:
                continue
            box = _clamped_box(item.get("box"), width, height)
            if box is not None:
                boxes.append(
                    {
                        "label": str(item["label"]).lower(),
                        "score": score,
                        "box": (box.left, box.top, box.right, box.bottom),
                        **{
                            key: item[key]
                            for key in ("query_id", "class_scores")
                            if key in item
                        },
                    }
                )
        return boxes

    def detect_sliced_controls(
        self,
        image: Image.Image,
        grid: int,
        regions: Sequence[Sequence[float]] | None = None,
        page_controls: Sequence[Detection] = (),
        overlap: float = SLICE_OVERLAP,
    ) -> list[Detection]:
        """Control proposals from overlapping tiles, in page coordinates.

        Sliced inference (SAHI, Akyon et al., ICIP 2022): a checkbox is a few pixels at
        the detector's page input, so its score falls below Docling's class threshold.
        With `regions`, only tiles overlapping one run (ROI-gated SAHI, arXiv 2608.23923).
        Slices only add controls the page pass missed: merged into a page-pass control,
        a wider tile box took the next field's words into its label.
        """
        width, height = image.size
        step = 1 - overlap
        tile_width = width / (grid - (grid - 1) * overlap)
        tile_height = height / (grid - (grid - 1) * overlap)
        found = []
        for row in range(grid):
            for column in range(grid):
                left, top = int(column * tile_width * step), int(row * tile_height * step)
                right = min(width, int(left + tile_width))
                bottom = min(height, int(top + tile_height))
                if regions is not None and not any(
                    box[0] < right and box[2] > left and box[1] < bottom and box[3] > top
                    for box in regions
                ):
                    continue
                for item in self.detect_elements(image.crop((left, top, right, bottom))):
                    if item["label"] not in CONTROL_LABELS:
                        continue
                    x0, y0, x1, y1 = item["box"]
                    box = (x0 + left, y0 + top, x1 + left, y1 + top)
                    if any(
                        _overlap_of_smaller(box, other["box"]) >= SAME_CONTROL_OVERLAP
                        for other in page_controls
                    ):
                        continue
                    found.append({**item, "box": box})
        return found

    def model_provenance(self) -> dict[str, Any]:
        return {
            "id": self.model_id,
            "revision": self.revision,
            "architecture": MODEL_ARCHITECTURE,
            "license": MODEL_LICENSE,
            "origin": MODEL_ORIGIN,
            "score_meaning": "layout detection confidence, not recognition confidence",
        }

    def _get_detector(self) -> Detector:
        if self._detector is not None:
            return self._detector
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForObjectDetection

            processor = AutoImageProcessor.from_pretrained(
                self.model_id, revision=self.revision
            )
            model = AutoModelForObjectDetection.from_pretrained(
                self.model_id, revision=self.revision
            )
            model.to(self.device)
            model.eval()
            id2label = model.config.id2label
            device = self.device
            score_threshold = self.score_threshold

            def run(image: Image.Image) -> list[Detection]:
                inputs = processor(images=[image], return_tensors="pt").to(device)
                with torch.inference_mode():
                    outputs = model(**inputs)
                # Heron's targets assign one class per query. Flattened focal top-k
                # emits several alternative labels for one box and drops other queries.
                probabilities = outputs.logits[0].sigmoid().float().cpu()
                scores, labels = probabilities.max(dim=-1)
                centers, sizes = outputs.pred_boxes[0].float().cpu().split(2, dim=-1)
                boxes = torch.cat((centers - sizes / 2, centers + sizes / 2), dim=-1)
                boxes *= torch.tensor([*image.size, *image.size])
                queries = scores.argsort(descending=True, stable=True)
                queries = queries[scores[queries] >= score_threshold]
                return [
                    {
                        "label": id2label[label],
                        "score": score,
                        "box": tuple(box),
                        "query_id": query,
                        "class_scores": {
                            id2label[index]: probability
                            for index, probability in enumerate(class_scores)
                        },
                    }
                    for query, label, score, box, class_scores in zip(
                        queries.tolist(),
                        labels[queries].tolist(),
                        scores[queries].tolist(),
                        boxes[queries].tolist(),
                        probabilities[queries].tolist(),
                        strict=True,
                    )
                ]
        except Exception as error:
            raise ReaderError("heron_load_failed", str(error)) from error
        self._detector = run
        return self._detector


def _clamped_box(
    value: Sequence[float] | None, width: int, height: int
) -> BoundingBox | None:
    if value is None or len(value) != 4:
        return None
    if value[2] <= value[0] or value[3] <= value[1]:
        return None
    left, top, right, bottom = (int(round(part)) for part in value)
    left = max(0, min(width, left))
    top = max(0, min(height, top))
    right = max(0, min(width, right))
    bottom = max(0, min(height, bottom))
    if right <= left or bottom <= top:
        return None
    return BoundingBox(left, top, right, bottom)
