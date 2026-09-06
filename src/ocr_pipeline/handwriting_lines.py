"""Page-level handwriting route: propose detected text lines for specialist rereading.

The handwriting specialist is a repair path that only reaches regions the primary reader
already resolved. A page written entirely by hand never qualifies, so its lines are never
offered to the specialist at all. This stage proposes those lines from page geometry.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Protocol

from PIL import Image, UnidentifiedImageError

from .contracts import BoundingBox, TextRegion
from .providers import ReaderError

PROVIDER = "doctr-line-segmentation"
CANDIDATE_SOURCE = "line_segmentation"
MODEL = {
    "id": "db_resnet50",
    "library": "python-doctr",
    "origin": "Mindee",
    "license": "Apache-2.0",
}
MEASURED_KINDS = frozenset({"text", "word"})
# Evidence the table reread path and the formula specialist already own.
EXCLUDED_OWNERS = frozenset(
    {
        "equation",
        "formula",
        "math",
        "table",
        "table_candidate",
        "table_cell",
        "table_source",
    }
)


class LineDetector(Protocol):
    name: str

    @property
    def provenance(self) -> dict[str, Any]: ...

    def detect(self, image_path: Path) -> list[BoundingBox]: ...


class DocTRLineDetector:
    """Group docTR word boxes into text lines at source resolution."""

    name = PROVIDER

    def __init__(
        self, *, device: str = "cuda:0", architecture: str = "db_resnet50"
    ) -> None:
        self.device = device
        self.architecture = architecture
        self._predictor: Any | None = None
        self._lock = threading.Lock()

    @property
    def provenance(self) -> dict[str, Any]:
        return {**MODEL, "id": self.architecture}

    def detect(self, image_path: Path) -> list[BoundingBox]:
        predictor = self._load()
        try:
            import numpy as np

            with Image.open(image_path) as opened:
                page = opened.convert("RGB")
                width, height = page.size
                predicted = predictor([np.asarray(page)])
        except (OSError, UnidentifiedImageError, ValueError, RuntimeError) as error:
            raise ReaderError("line_detection_failed", str(error)) from error

        entry = predicted[0]
        raw = entry.get("words") if isinstance(entry, dict) else entry
        boxes = []
        for item in raw:
            try:
                left, top, right, bottom = (float(value) for value in item[:4])
            except (TypeError, ValueError):
                continue
            box = BoundingBox(
                max(0, round(left * width)),
                max(0, round(top * height)),
                min(width, round(right * width)),
                min(height, round(bottom * height)),
            )
            if box.right > box.left and box.bottom > box.top:
                boxes.append(box)
        return _group_lines(boxes)

    def _load(self) -> Any:
        with self._lock:
            if self._predictor is not None:
                return self._predictor
            return self._build()

    def _build(self) -> Any:
        try:
            from doctr.models import detection_predictor

            predictor = detection_predictor(
                arch=self.architecture,
                pretrained=True,
                assume_straight_pages=True,
            )
            if self.device.startswith("cuda"):
                predictor = predictor.cuda()
        except Exception as error:
            raise ReaderError(
                "line_detection_unavailable",
                "docTR line detection is unavailable",
            ) from error
        self._predictor = predictor
        return predictor


class HandwritingLineStage:
    """Propose line crops wherever the primary reader read poorly.

    The decision is per region, not per page. A page-wide ratio would classify mixed
    clinical forms as printed and never offer their handwritten field values to the
    specialist, which is the case that matters most.
    """

    name = "handwriting-lines"

    def __init__(
        self,
        detector: LineDetector,
        *,
        text_provider: str | None = None,
        confidence_threshold: float = 0.75,
        max_lines: int = 24,
    ) -> None:
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be from 0 to 1")
        if max_lines <= 0:
            raise ValueError("max_lines must be positive")
        self.detector = detector
        self.text_provider = text_provider
        self.confidence_threshold = confidence_threshold
        self.max_lines = max_lines

    def apply(
        self,
        image_path: Path,
        page_number: int,
        regions: list[TextRegion],
    ) -> list[TextRegion]:
        low = self._poorly_read_regions(regions)
        if not low:
            return regions

        lines = [
            line
            for line in self.detector.detect(image_path)
            if _overlaps_any(line, low)
        ]
        if not lines:
            return regions

        order = max((region.reading_order for region in regions), default=0) + 1
        proposals = [
            _proposal(line, page_number, index, order + index)
            for index, line in enumerate(
                sorted(lines, key=lambda box: (box.top, box.left))[: self.max_lines],
                start=1,
            )
        ]
        return [*regions, *proposals]

    def _poorly_read_regions(self, regions: list[TextRegion]) -> list[TextRegion]:
        """Low-confidence text the table and formula specialists do not already own."""
        return [
            region
            for region in regions
            if region.kind in MEASURED_KINDS
            and region.confidence is not None
            and region.confidence < self.confidence_threshold
            and bool(region.text.strip())
            and not _owned_by_another_specialist(region)
            and (self.text_provider is None or region.provider == self.text_provider)
        ]


def _owned_by_another_specialist(region: TextRegion) -> bool:
    structure = region.structure if isinstance(region.structure, dict) else {}
    labels = {
        str(structure.get(name, "")).strip().casefold()
        for name in ("role", "layout_owner_type", "block_type", "semantic_class")
    }
    return bool(labels & EXCLUDED_OWNERS)


def _proposal(
    line: BoundingBox,
    page_number: int,
    index: int,
    reading_order: int,
) -> TextRegion:
    return TextRegion(
        id=f"p{page_number}-handwriting-line-{index}",
        kind="handwriting",
        text="",
        confidence=None,
        bounding_box=line,
        reading_order=reading_order,
        provider=PROVIDER,
        text_provenance={
            "method": "detected_text_line",
            "model": MODEL,
        },
        resolution="unreadable",
        structure={
            "handwriting_candidate": True,
            "handwriting_candidate_source": CANDIDATE_SOURCE,
        },
    )


def _group_lines(boxes: list[BoundingBox]) -> list[BoundingBox]:
    """Merge word boxes whose vertical centre falls inside a growing line band."""
    lines: list[dict[str, Any]] = []
    for box in sorted(boxes, key=lambda item: (item.top, item.left)):
        centre = (box.top + box.bottom) / 2
        placed = False
        for line in lines:
            if line["top"] <= centre <= line["bottom"]:
                line["top"] = min(line["top"], box.top)
                line["bottom"] = max(line["bottom"], box.bottom)
                line["left"] = min(line["left"], box.left)
                line["right"] = max(line["right"], box.right)
                placed = True
                break
        if not placed:
            lines.append(
                {
                    "top": box.top,
                    "bottom": box.bottom,
                    "left": box.left,
                    "right": box.right,
                }
            )
    return [
        BoundingBox(line["left"], line["top"], line["right"], line["bottom"])
        for line in lines
    ]


def _overlaps_any(line: BoundingBox, regions: list[TextRegion]) -> bool:
    return any(_intersects(line, region.bounding_box) for region in regions)


def _intersects(first: BoundingBox, second: BoundingBox) -> bool:
    return min(first.right, second.right) > max(first.left, second.left) and min(
        first.bottom, second.bottom
    ) > max(first.top, second.top)
