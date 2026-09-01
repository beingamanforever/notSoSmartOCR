from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from ocr_pipeline.contracts import BoundingBox, TextRegion
from ocr_pipeline.controls import GeometricControlStage, detect_controls
from ocr_pipeline.pipeline import process_document


def test_detect_controls_classifies_selected_and_unselected(tmp_path: Path) -> None:
    source = tmp_path / "controls.png"
    image = Image.new("L", (240, 100), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 34, 34), outline="black", width=2)
    draw.rectangle((20, 55, 34, 69), outline="black", width=2)
    draw.line((23, 58, 31, 66), fill="black", width=2)
    draw.line((31, 58, 23, 66), fill="black", width=2)
    image.save(source)

    detections = detect_controls(source)

    assert [item.state for item in detections] == ["unselected", "selected"]
    assert [item.bounding_box for item in detections] == [
        BoundingBox(20, 20, 35, 35),
        BoundingBox(20, 55, 35, 70),
    ]


def test_control_stage_associates_labels_and_preserves_provenance(
    tmp_path: Path,
) -> None:
    source = _control_image(tmp_path, selected=True)
    stage = GeometricControlStage(minimum_group_size=1)
    marker = _region("marker", "x", (18, 18, 37, 37), 3)
    label = _region("label", "Fall prevention", (45, 20, 140, 35), 4)

    output = stage.apply(source, 2, [marker, label])

    control = next(region for region in output if region.kind == "checkbox")
    assert control.id == "p2-controls-checkbox-1"
    assert control.text == "[x] Fall prevention"
    assert control.resolution == "resolved"
    assert control.bounding_box == BoundingBox(20, 20, 35, 35)
    assert control.reading_order == 4
    assert control.text_provenance == {
        "method": "square_contour_with_line_cleanup",
        "label_evidence_ids": ["label"],
    }
    assert control.structure["role"] == "control"
    assert control.structure["state"] == "selected"
    assert control.structure["label"] == "Fall prevention"
    assert control.structure["model"]["origin"] == "Open Source Vision Foundation"


def test_unlabeled_or_ambiguous_control_routes_page_to_review(tmp_path: Path) -> None:
    source = _control_image(tmp_path, selected=False)

    result = process_document(source, FixedReader([]), stages=[GeometricControlStage()])

    control = next(
        region for region in result.pages[0].regions if region.kind == "checkbox"
    )
    assert control.structure["state"] == "unselected"
    assert control.structure["label"] is None
    assert control.resolution == "unreadable"
    assert result.pages[0].route == "review"


def test_small_control_group_routes_coverage_review(tmp_path: Path) -> None:
    source = _control_image(tmp_path, selected=True)
    label = _region("label", "Fall prevention", (45, 20, 140, 35), 4)

    result = process_document(
        source,
        FixedReader([label]),
        stages=[GeometricControlStage()],
    )

    control = next(
        region for region in result.pages[0].regions if region.kind == "checkbox"
    )
    assert control.structure["coverage_status"] == "insufficient_control_group"
    assert control.resolution == "unreadable"
    assert result.pages[0].route == "review"


def test_line_cleanup_rejects_square_letter_inside_word(tmp_path: Path) -> None:
    source = tmp_path / "joined.png"
    image = np.full((80, 180), 255, dtype=np.uint8)
    cv2.putText(
        image,
        "DENSE",
        (10, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        0,
        2,
        cv2.LINE_8,
    )
    cv2.imwrite(str(source), image)

    assert detect_controls(source) == []


def test_adjacent_character_grid_is_not_reported_as_checkboxes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "grid.png"
    image = Image.new("L", (160, 80), "white")
    draw = ImageDraw.Draw(image)
    for left in (20, 36, 52, 68):
        draw.rectangle((left, 20, left + 14, 34), outline="black", width=2)
    image.save(source)

    assert detect_controls(source) == []


def _control_image(tmp_path: Path, *, selected: bool) -> Path:
    source = tmp_path / "page.png"
    image = Image.new("L", (180, 80), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 34, 34), outline="black", width=2)
    if selected:
        draw.line((23, 23, 31, 31), fill="black", width=2)
        draw.line((31, 23, 23, 31), fill="black", width=2)
    image.save(source)
    return source


def _region(
    region_id: str,
    text: str,
    box: tuple[int, int, int, int],
    order: int,
) -> TextRegion:
    return TextRegion(
        id=region_id,
        kind="word",
        text=text,
        confidence=0.98,
        bounding_box=BoundingBox(*box),
        reading_order=order,
        provider="fixed",
    )


class FixedReader:
    name = "fixed"

    def __init__(self, regions: list[TextRegion]) -> None:
        self.regions = regions

    def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
        return self.regions
