from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
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


def test_detect_controls_recovers_peer_supported_deformed_square(
    tmp_path: Path,
) -> None:
    source = tmp_path / "deformed-checkbox.png"
    image = np.full((400, 420), 255, dtype=np.uint8)
    cv2.rectangle(image, (30, 50), (46, 66), 0, 2)
    cv2.rectangle(image, (100, 50), (116, 66), 0, 2)
    cv2.polylines(
        image,
        [np.array([(102, 58), (107, 63), (120, 45)], dtype=np.int32)],
        False,
        0,
        2,
    )
    cv2.putText(
        image,
        "Subsequent exam",
        (118, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        0,
        1,
        cv2.LINE_8,
    )
    cv2.ellipse(image, (260, 100), (9, 6), 0, 0, 360, 0, 2)
    cv2.line(image, (254, 106), (266, 94), 0, 2)
    cv2.imwrite(str(source), image)

    detections = detect_controls(source)

    assert sorted(
        ((item.bounding_box, item.state) for item in detections),
        key=lambda item: item[0].left,
    ) == [
        (BoundingBox(29, 49, 48, 68), "unselected"),
        (BoundingBox(99, 44, 122, 68), "selected"),
    ]


def test_thick_checkbox_border_does_not_imply_selected(tmp_path: Path) -> None:
    source = tmp_path / "thick-border.png"
    image = Image.new("L", (400, 400), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 30, 59, 59), outline="black", width=6)
    image.save(source)

    detections = detect_controls(source)

    assert len(detections) == 1
    assert detections[0].state == "unselected"


def test_thin_checkbox_next_to_label_is_preserved(tmp_path: Path) -> None:
    source = tmp_path / "thin-checkbox.png"
    image = np.full((70, 160), 255, dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (34, 34), 0, 1)
    cv2.putText(
        image,
        "Dose",
        (37, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        0,
        1,
        cv2.LINE_8,
    )
    cv2.imwrite(str(source), image)

    detections = detect_controls(source)

    assert [(item.bounding_box, item.state) for item in detections] == [
        (BoundingBox(20, 20, 35, 35), "unselected")
    ]


def test_colored_background_does_not_imply_selected(tmp_path: Path) -> None:
    source = tmp_path / "colored-background.png"
    image = np.full((100, 180), 225, dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (36, 36), 0, 2)
    cv2.rectangle(image, (20, 60), (36, 76), 0, 2)
    cv2.line(image, (24, 64), (32, 72), 0, 2)
    cv2.line(image, (32, 64), (24, 72), 0, 2)
    cv2.imwrite(str(source), image)

    detections = detect_controls(source)

    assert [item.state for item in detections] == ["unselected", "selected"]


def test_tiny_selected_candidate_on_large_page_is_preserved_as_ambiguous(
    tmp_path: Path,
) -> None:
    source = tmp_path / "tiny-selected.png"
    image = Image.new("L", (2000, 2000), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 30, 44, 44), outline="black", width=2)
    draw.line((33, 33, 41, 41), fill="black", width=2)
    draw.line((41, 33, 33, 41), fill="black", width=2)
    image.save(source)
    label = _region("label", "Tiny option", (50, 30, 170, 45), 1)

    result = process_document(
        source,
        FixedReader([label]),
        stages=[GeometricControlStage(minimum_group_size=1)],
    )

    control = next(
        region for region in result.pages[0].regions if region.kind == "checkbox"
    )
    assert control.text == "[?] Tiny option"
    assert control.structure["state"] == "ambiguous"
    assert control.structure["state_confidence"] is None
    assert control.structure["observed_mark_confidence"] == 0.5
    assert control.confidence is None
    assert control.structure["association_status"] == "linked"
    assert control.structure["association_confidence"] is None
    assert control.resolution == "unreadable"
    assert result.pages[0].route == "review"


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


def test_unlabeled_selected_geometry_is_not_asserted(tmp_path: Path) -> None:
    source = _control_image(tmp_path, selected=True)

    result = process_document(
        source,
        FixedReader([]),
        stages=[GeometricControlStage(minimum_group_size=1)],
    )

    control = next(
        region for region in result.pages[0].regions if region.kind == "checkbox"
    )
    assert control.text == "[?]"
    assert control.structure["state"] == "ambiguous"
    assert control.structure["observed_state"] == "selected"
    assert control.resolution == "unreadable"


def test_small_control_group_keeps_readable_state_with_coverage_warning(
    tmp_path: Path,
) -> None:
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
    assert control.resolution == "resolved"
    assert result.pages[0].route == "review"


def test_colon_rich_form_keeps_sparse_control_marks_for_review(
    tmp_path: Path,
) -> None:
    source = _control_image(tmp_path, selected=True)
    labels = [
        _region(
            f"field-{index}",
            f"Field {index}:",
            (45, 20, 120, 35),
            index + 1,
        )
        for index in range(6)
    ]

    result = process_document(
        source,
        FixedReader(labels),
        stages=[GeometricControlStage(minimum_group_size=6)],
    )

    controls = [
        region for region in result.pages[0].regions if region.kind == "checkbox"
    ]
    assert len(controls) == 1
    assert controls[0].resolution == "resolved"
    assert controls[0].structure["coverage_status"] == "insufficient_control_group"


def test_sparse_math_glyphs_are_not_promoted_to_selected_controls(
    tmp_path: Path,
) -> None:
    source = tmp_path / "math-glyphs.png"
    image = Image.new("L", (400, 400), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 40, 55, 65), outline="black", width=2)
    draw.line((36, 46, 49, 59), fill="black", width=2)
    draw.line((49, 46, 36, 59), fill="black", width=2)
    image.save(source)
    label = _region("math", "x x x", (65, 42, 115, 64), 1)

    result = process_document(
        source,
        FixedReader([label]),
        stages=[GeometricControlStage()],
    )

    assert all(region.kind != "checkbox" for region in result.pages[0].regions)


@pytest.mark.parametrize(
    ("selected_label", "other_label"), [("M", "F"), ("Y", "N"), ("1", "2")]
)
def test_selected_short_option_is_preserved_by_its_control_group(
    tmp_path: Path,
    selected_label: str,
    other_label: str,
) -> None:
    source = tmp_path / f"short-options-{selected_label}.png"
    image = Image.new("L", (220, 90), "white")
    draw = ImageDraw.Draw(image)
    for top, selected in ((15, True), (50, False)):
        draw.rectangle((20, top, 34, top + 14), outline="black", width=2)
        if selected:
            draw.line((23, top + 3, 31, top + 11), fill="black", width=2)
            draw.line((31, top + 3, 23, top + 11), fill="black", width=2)
    image.save(source)
    labels = [
        _region("selected", selected_label, (42, 15, 58, 30), 1),
        _region("other", other_label, (42, 50, 58, 65), 2),
    ]

    result = process_document(
        source,
        FixedReader(labels),
        stages=[GeometricControlStage()],
    )

    assert any(
        region.kind == "checkbox" and region.text == f"[x] {selected_label}"
        for region in result.pages[0].regions
    )


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


def test_hash_glyph_is_not_reported_as_selected_control(tmp_path: Path) -> None:
    source = tmp_path / "hash.png"
    image = Image.new("L", (120, 120), "white")
    draw = ImageDraw.Draw(image)
    draw.line((35, 25, 28, 85), fill="black", width=5)
    draw.line((58, 25, 51, 85), fill="black", width=5)
    draw.line((22, 43, 66, 49), fill="black", width=5)
    draw.line((20, 63, 64, 69), fill="black", width=5)
    image.save(source)

    assert detect_controls(source) == []


def test_control_stage_rejects_square_inside_reader_text(tmp_path: Path) -> None:
    source = tmp_path / "inside-text.png"
    image = Image.new("L", (240, 100), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 34, 34), outline="black", width=2)
    image.save(source)
    text = _region("word", "Printed token", (10, 12, 90, 42), 1)

    result = process_document(
        source,
        FixedReader([text]),
        stages=[GeometricControlStage(minimum_group_size=1)],
    )

    assert all(region.kind != "checkbox" for region in result.pages[0].regions)


def test_control_stage_keeps_leading_control_overlapped_by_reader_box(
    tmp_path: Path,
) -> None:
    source = tmp_path / "leading-control.png"
    image = Image.new("L", (240, 100), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 34, 34), outline="black", width=2)
    image.save(source)
    label = _region("label", "Review option", (28, 18, 120, 38), 1)

    result = process_document(
        source,
        FixedReader([label]),
        stages=[GeometricControlStage(minimum_group_size=1)],
    )

    control = next(
        region for region in result.pages[0].regions if region.kind == "checkbox"
    )
    assert control.structure["state"] == "unselected"
    assert control.structure["label"] == "Review option"


def test_control_stage_keeps_controls_at_edges_of_line_level_reader_boxes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "line-level-boxes.png"
    image = Image.new("L", (240, 100), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 20, 24, 34), outline="black", width=2)
    draw.rectangle((180, 60, 194, 74), outline="black", width=2)
    image.save(source)
    labels = [
        _region("leading", "Leading option", (8, 18, 150, 38), 1),
        _region("trailing", "Trailing option", (50, 58, 196, 78), 2),
    ]

    result = process_document(
        source,
        FixedReader(labels),
        stages=[GeometricControlStage(minimum_group_size=1)],
    )

    controls = [
        region for region in result.pages[0].regions if region.kind == "checkbox"
    ]
    assert [control.structure["label"] for control in controls] == [
        "Leading option",
        "Trailing option",
    ]


def test_control_stage_rejects_box_like_table_glyph_and_keeps_checkbox(
    tmp_path: Path,
) -> None:
    source = tmp_path / "table-controls.png"
    image = np.full((140, 280), 255, dtype=np.uint8)
    cv2.putText(
        image,
        "D",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        0,
        2,
        cv2.LINE_8,
    )
    cv2.putText(
        image,
        "ate",
        (42, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        0,
        2,
        cv2.LINE_8,
    )
    cv2.rectangle(image, (20, 75), (34, 89), 0, 2)
    cv2.imwrite(str(source), image)
    date = _region("date", "Date", (18, 26, 110, 48), 1)
    date.structure = {"role": "table_source"}
    option = _region("option", "Genuine option", (45, 73, 180, 93), 2)
    option.structure = {"role": "table_source"}

    result = process_document(
        source,
        FixedReader([date, option]),
        stages=[GeometricControlStage(minimum_group_size=1)],
    )

    controls = [
        region for region in result.pages[0].regions if region.kind == "checkbox"
    ]
    assert [control.text for control in controls] == ["[ ] Genuine option"]


def test_control_stage_does_not_infer_unboxed_marks_from_table_text(
    tmp_path: Path,
) -> None:
    source = tmp_path / "table-mark.png"
    image = Image.new("L", (280, 100), "white")
    draw = ImageDraw.Draw(image)
    draw.line((30, 42, 40, 52), fill="black", width=2)
    draw.line((40, 42, 30, 52), fill="black", width=2)
    image.save(source)
    value = _region("value", "25,971,901", (52, 38, 150, 58), 1)
    value.structure = {"role": "table_source"}

    result = process_document(
        source,
        FixedReader([value]),
        stages=[GeometricControlStage(minimum_group_size=1)],
    )

    assert all(region.kind != "checkbox" for region in result.pages[0].regions)


def test_control_stage_recovers_unboxed_mark_for_table_label(tmp_path: Path) -> None:
    source = tmp_path / "table-option.png"
    image = Image.new("L", (280, 100), "white")
    draw = ImageDraw.Draw(image)
    draw.line((30, 42, 40, 52), fill="black", width=2)
    draw.line((40, 42, 30, 52), fill="black", width=2)
    image.save(source)
    option = _region("option", "Fall prevention", (52, 38, 180, 58), 1)
    option.structure = {"role": "table_source"}

    result = process_document(
        source,
        FixedReader([option]),
        stages=[GeometricControlStage(minimum_group_size=1)],
    )

    controls = [
        region for region in result.pages[0].regions if region.kind == "checkbox"
    ]
    assert [control.text for control in controls] == ["[?] Fall prevention"]


def test_control_stage_rejects_box_like_glyph_between_vertical_text(
    tmp_path: Path,
) -> None:
    source = tmp_path / "vertical-text.png"
    line = np.full((60, 180), 255, dtype=np.uint8)
    cv2.putText(
        line,
        "ADA",
        (5, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        0,
        2,
        cv2.LINE_8,
    )
    image = cv2.rotate(line, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(str(source), image)

    result = process_document(
        source,
        FixedReader([]),
        stages=[GeometricControlStage(minimum_group_size=1)],
    )

    assert all(region.kind != "checkbox" for region in result.pages[0].regions)


def test_control_stage_rejects_inline_mark_inside_low_confidence_peer_label(
    tmp_path: Path,
) -> None:
    source = tmp_path / "broken-selected-peer.png"
    image = Image.new("L", (440, 100), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 40, 36, 56), outline="black", width=2)
    draw.line((220, 42, 232, 54), fill="black", width=2)
    draw.line((232, 42, 220, 54), fill="black", width=2)
    image.save(source)
    peer_label = _region("peer", "Initial exam", (42, 38, 160, 58), 1)
    target_label = _region("target", "Subsequent exam", (216, 38, 390, 58), 2)
    target_label.confidence = 0.83

    result = process_document(
        source,
        FixedReader([peer_label, target_label]),
        stages=[GeometricControlStage(minimum_group_size=1)],
    )

    controls = [
        region for region in result.pages[0].regions if region.kind == "checkbox"
    ]
    assert [control.text for control in controls] == ["[ ] Initial exam"]


def test_control_stage_ignores_unlabeled_empty_square_artifacts(
    tmp_path: Path,
) -> None:
    source = tmp_path / "empty-square-artifact.png"
    image = Image.new("L", (400, 400), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 34, 34), outline="black", width=2)
    draw.rectangle((300, 300, 314, 314), outline="black", width=2)
    image.save(source)
    label = _region("label", "Review option", (45, 18, 140, 38), 1)

    result = process_document(
        source,
        FixedReader([label]),
        stages=[GeometricControlStage(minimum_group_size=1)],
    )

    controls = [
        region for region in result.pages[0].regions if region.kind == "checkbox"
    ]
    assert [control.structure["label"] for control in controls] == ["Review option"]


def test_control_stage_ignores_all_unlabeled_artifacts_when_labels_exist(
    tmp_path: Path,
) -> None:
    source = tmp_path / "unlabeled-artifacts.png"
    image = Image.new("L", (2000, 2000), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 30, 59, 59), outline="black", width=2)
    draw.rectangle((300, 300, 329, 329), outline="black", width=2)
    draw.line((305, 305, 324, 324), fill="black", width=3)
    draw.line((324, 305, 305, 324), fill="black", width=3)
    draw.rectangle((600, 600, 614, 614), outline="black", width=2)
    image.save(source)
    label = _region("label", "Review option", (70, 35, 170, 55), 1)

    result = process_document(
        source,
        FixedReader([label]),
        stages=[GeometricControlStage(minimum_group_size=1)],
    )

    controls = [
        region for region in result.pages[0].regions if region.kind == "checkbox"
    ]
    assert [control.structure["label"] for control in controls] == ["Review option"]


def test_control_stage_keeps_grouped_unmatched_control_for_review(
    tmp_path: Path,
) -> None:
    source = tmp_path / "mixed-controls.png"
    image = Image.new("L", (600, 600), "white")
    draw = ImageDraw.Draw(image)
    tops = (30, 90, 150, 210, 270, 330, 390)
    for top in tops:
        draw.rectangle((30, top, 59, top + 29), outline="black", width=2)
    draw.line((36, 396, 53, 413), fill="black", width=3)
    draw.line((53, 396, 36, 413), fill="black", width=3)
    image.save(source)
    labels = [
        _region(
            f"label-{index}",
            f"Option {index}",
            (70, top, 170, top + 30),
            index,
        )
        for index, top in enumerate(tops[:6], start=1)
    ]

    result = process_document(
        source,
        FixedReader(labels),
        stages=[GeometricControlStage(minimum_group_size=6)],
    )

    controls = [
        region for region in result.pages[0].regions if region.kind == "checkbox"
    ]
    assert len(controls) == 7
    unmatched = controls[-1]
    assert unmatched.text == "[?]"
    assert unmatched.structure["label"] is None
    assert unmatched.structure["coverage_status"] == "unmatched_label"
    assert unmatched.structure["observed_state"] == "selected"
    assert unmatched.structure["state"] == "ambiguous"
    assert unmatched.resolution == "unreadable"
    assert result.pages[0].route == "review"


def test_control_stage_detects_large_checkbox(tmp_path: Path) -> None:
    source = tmp_path / "large-control.png"
    image = Image.new("L", (400, 400), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 40, 55, 65), outline="black", width=2)
    image.save(source)
    label = _region("label", "Review", (65, 42, 125, 64), 1)

    result = process_document(
        source,
        FixedReader([label]),
        stages=[GeometricControlStage(minimum_group_size=1)],
    )

    control = next(
        region for region in result.pages[0].regions if region.kind == "checkbox"
    )
    assert control.bounding_box == BoundingBox(30, 40, 56, 66)
    assert control.structure["state"] == "unselected"


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


def test_control_stage_preserves_labeled_mixed_size_vertical_controls(
    tmp_path: Path,
) -> None:
    source = tmp_path / "vertical-input-grid.png"
    image = Image.new("L", (1000, 1000), "white")
    draw = ImageDraw.Draw(image)
    for top in (30, 100, 170, 240):
        draw.rectangle((30, top, 59, top + 29), outline="black", width=2)
    for top in (400, 525):
        draw.rectangle((400, top, 459, top + 59), outline="black", width=2)
        draw.line((408, top + 8, 451, top + 51), fill="black", width=5)
        draw.line((451, top + 8, 408, top + 51), fill="black", width=5)
    image.save(source)
    labels = [
        *[
            _region(f"small-{index}", f"Small {index}", (70, top, 170, top + 30), index)
            for index, top in enumerate((30, 100, 170, 240), start=1)
        ],
        _region("large-a", "Large A", (475, 410, 575, 450), 5),
        _region("large-b", "Large B", (475, 535, 575, 575), 6),
    ]

    result = process_document(
        source,
        FixedReader(labels),
        stages=[GeometricControlStage(minimum_group_size=1)],
    )

    controls = [
        region for region in result.pages[0].regions if region.kind == "checkbox"
    ]
    assert [control.structure["label"] for control in controls] == [
        "Small 1",
        "Small 2",
        "Small 3",
        "Small 4",
        "Large A",
        "Large B",
    ]
    assert [control.structure["state"] for control in controls[-2:]] == [
        "ambiguous",
        "ambiguous",
    ]


def test_control_stage_recovers_label_anchored_x_and_tick(tmp_path: Path) -> None:
    source = tmp_path / "selected-marks.png"
    image = Image.new("L", (320, 150), "white")
    draw = ImageDraw.Draw(image)
    draw.line((25, 31, 35, 41), fill="black", width=2)
    draw.line((35, 31, 25, 41), fill="black", width=2)
    draw.line((232, 94, 238, 102), fill="black", width=2)
    draw.line((238, 102, 248, 87), fill="black", width=2)
    image.save(source)
    labels = [
        _region("left-label", "First option:", (43, 29, 120, 43), 2),
        _region("right-label", "Second option:", (130, 90, 225, 106), 3),
    ]

    result = process_document(
        source,
        FixedReader(labels),
        stages=[GeometricControlStage(minimum_group_size=1)],
    )

    controls = [
        region for region in result.pages[0].regions if region.kind == "checkbox"
    ]
    assert [control.text for control in controls] == [
        "[?] First option",
        "[?] Second option",
    ]
    assert all(control.resolution == "unreadable" for control in controls)
    assert result.pages[0].route == "review"
    assert all(
        control.text_provenance["method"] == "label_anchored_residual_ink"
        for control in controls
    )
    assert all(
        control.structure["model"]["id"] == "opencv-label-anchored-residual-v1"
        for control in controls
    )


def test_control_stage_reports_label_anchored_slashed_circle_for_review(
    tmp_path: Path,
) -> None:
    source = tmp_path / "null-mark.png"
    image = Image.new("L", (420, 100), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((330, 38, 348, 54), outline="black", width=2)
    draw.line((332, 53, 346, 39), fill="black", width=2)
    image.save(source)
    label = _region(
        "taps",
        "TAPS SCORE (Substance Abuse Disorder):",
        (60, 36, 325, 56),
        1,
    )

    result = process_document(
        source,
        FixedReader([label]),
        stages=[GeometricControlStage(minimum_group_size=1)],
    )

    controls = [
        region for region in result.pages[0].regions if region.kind == "checkbox"
    ]
    # The glyph records what was drawn. Its meaning stays unresolved until the form's
    # conventions or a reviewer establish it, so this is an annotation, not a selection.
    assert [control.text for control in controls] == [
        "∅ TAPS SCORE (Substance Abuse Disorder)"
    ]
    assert controls[0].structure["control_type"] == "annotation"
    assert controls[0].structure["annotation_shape"] == "slashed_loop"
    assert controls[0].structure["interpretation_status"] == "unresolved"
    assert controls[0].resolution == "unreadable"
    assert controls[0].text_provenance["method"] == "label_anchored_null_glyph"
    assert controls[0].text_provenance["label_evidence_ids"] == ["taps"]
    assert result.pages[0].route == "review"


def test_control_stage_rejects_empty_checkbox_outline_as_null_mark(
    tmp_path: Path,
) -> None:
    source = tmp_path / "empty-box.png"
    image = Image.new("L", (420, 100), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((330, 38, 348, 56), outline="black", width=2)
    image.save(source)
    label = _region("alcohol", "Alcohol:", (60, 36, 325, 56), 1)

    result = process_document(
        source,
        FixedReader([label]),
        stages=[GeometricControlStage(minimum_group_size=1)],
    )

    controls = [
        region for region in result.pages[0].regions if region.kind == "checkbox"
    ]
    assert all(
        control.text_provenance["method"] != "label_anchored_residual_ink"
        for control in controls
    )


def test_ring_over_one_value_labels_the_enclosed_words_not_the_whole_row(
    tmp_path: Path,
) -> None:
    """The JPMorgan page circles single values on long rows. Two rings on one row must
    not each repeat the row's full text - that rendered every such line twice."""
    source = tmp_path / "circled-values.png"
    image = Image.new("L", (900, 100), "white")
    draw = ImageDraw.Draw(image)
    draw.text((60, 44), "Average deposits", fill="black")
    draw.text((420, 44), "$187", fill="black")
    draw.text((640, 44), "$1,064", fill="black")
    draw.ellipse((410, 36, 480, 64), outline="black", width=2)
    draw.ellipse((630, 36, 710, 64), outline="black", width=2)
    image.save(source)
    row = _region("row", "Average deposits $187 $1,064", (60, 42, 780, 62), 1)
    row.structure = {
        "word_evidence": [
            {"text": "Average", "bbox": {"left": 60, "top": 44, "right": 115, "bottom": 60}, "confidence": 0.99, "in_region_text": True},
            {"text": "deposits", "bbox": {"left": 120, "top": 44, "right": 180, "bottom": 60}, "confidence": 0.98, "in_region_text": True},
            {"text": "$187", "bbox": {"left": 420, "top": 44, "right": 460, "bottom": 60}, "confidence": 0.97, "in_region_text": True},
            {"text": "$1,064", "bbox": {"left": 640, "top": 44, "right": 700, "bottom": 60}, "confidence": 0.96, "in_region_text": True},
        ]
    }

    result = process_document(
        source,
        FixedReader([row]),
        stages=[GeometricControlStage(minimum_group_size=1)],
    )

    rings = [
        region
        for region in result.pages[0].regions
        if region.kind == "checkbox"
        and region.text_provenance["method"] == "enclosing_ring_annotation"
    ]
    assert len(rings) == 2
    labels = sorted(ring.structure["label"] for ring in rings)
    assert labels == ["$1,064", "$187"]


def test_control_stage_reports_ring_drawn_over_printed_option(
    tmp_path: Path,
) -> None:
    source = tmp_path / "circled-option.png"
    image = Image.new("L", (420, 100), "white")
    draw = ImageDraw.Draw(image)
    draw.text((60, 44), "Non-Smoker / Smoker", fill="black")
    draw.ellipse((56, 38, 135, 62), outline="black", width=2)
    image.save(source)
    label = _region("social", "Non-Smoker / Smoker", (60, 42, 200, 60), 1)

    result = process_document(
        source,
        FixedReader([label]),
        stages=[GeometricControlStage(minimum_group_size=1)],
    )

    rings = [
        region
        for region in result.pages[0].regions
        if region.kind == "checkbox"
        and region.text_provenance["method"] == "enclosing_ring_annotation"
    ]
    assert len(rings) == 1
    assert rings[0].text_provenance["label_evidence_ids"] == ["social"]
    assert rings[0].resolution == "unreadable"
    assert result.pages[0].route == "review"
    # A ring means the enclosed option was chosen, so the target is a relationship to the
    # region it surrounds, not just a label sitting next to a mark.
    assert rings[0].structure["control_type"] == "annotation"
    assert rings[0].structure["annotation_shape"] == "ring"
    assert rings[0].structure["annotation_target"] == {
        "relation": "encloses",
        "evidence_ids": ["social"],
    }
    assert rings[0].structure["interpretation_status"] == "unresolved"


def test_control_stage_rejects_ruled_rectangle_as_ring(tmp_path: Path) -> None:
    source = tmp_path / "ruled-box.png"
    image = Image.new("L", (420, 140), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 30, 380, 120), outline="black", width=2)
    image.save(source)
    label = _region("comments", "Comments:", (60, 40, 160, 58), 1)

    result = process_document(
        source,
        FixedReader([label]),
        stages=[GeometricControlStage(minimum_group_size=1)],
    )

    assert all(
        region.kind != "checkbox"
        or region.text_provenance["method"] != "enclosing_ring_annotation"
        for region in result.pages[0].regions
    )


def test_control_stage_recovers_anchored_marks_without_label_punctuation(
    tmp_path: Path,
) -> None:
    source = tmp_path / "plain-label-marks.png"
    image = Image.new("L", (320, 150), "white")
    draw = ImageDraw.Draw(image)
    draw.line((25, 31, 35, 41), fill="black", width=2)
    draw.line((35, 31, 25, 41), fill="black", width=2)
    draw.line((232, 94, 238, 102), fill="black", width=2)
    draw.line((238, 102, 248, 87), fill="black", width=2)
    image.save(source)
    labels = [
        _region("left-label", "First option", (43, 29, 120, 43), 2),
        _region("right-label", "Second option", (130, 90, 225, 106), 3),
    ]

    result = process_document(
        source,
        FixedReader(labels),
        stages=[GeometricControlStage(minimum_group_size=1)],
    )

    controls = [
        region for region in result.pages[0].regions if region.kind == "checkbox"
    ]
    assert [control.text for control in controls] == [
        "[?] First option",
        "[?] Second option",
    ]
    assert all(control.resolution == "unreadable" for control in controls)
    assert result.pages[0].route == "review"


def test_control_stage_does_not_count_squares_toward_anchored_mark_group(
    tmp_path: Path,
) -> None:
    source = tmp_path / "mixed-control-sources.png"
    image = Image.new("L", (1000, 1000), "white")
    draw = ImageDraw.Draw(image)
    square_tops = (30, 100, 170, 240, 310)
    for top in square_tops:
        draw.rectangle((30, top, 59, top + 29), outline="black", width=2)
    draw.line((600, 510, 612, 522), fill="black", width=2)
    draw.line((612, 510, 600, 522), fill="black", width=2)
    image.save(source)
    labels = [
        *[
            _region(
                f"square-{index}",
                f"Option {index}",
                (70, top, 170, top + 30),
                index,
            )
            for index, top in enumerate(square_tops, start=1)
        ],
        _region("prose", "contribution", (625, 507, 760, 525), 6),
    ]
    for label in labels[:-1]:
        label.kind = "form_field"

    result = process_document(
        source,
        FixedReader(labels),
        stages=[GeometricControlStage(minimum_group_size=6)],
    )

    controls = [
        region for region in result.pages[0].regions if region.kind == "checkbox"
    ]
    assert [control.structure["label"] for control in controls] == [
        "Option 1",
        "Option 2",
        "Option 3",
        "Option 4",
        "Option 5",
    ]
    assert all(
        control.text_provenance["method"] == "square_contour_with_line_cleanup"
        for control in controls
    )


def test_solid_square_list_markers_are_not_reported_as_controls(
    tmp_path: Path,
) -> None:
    source = tmp_path / "solid-list-markers.png"
    image = Image.new("L", (620, 320), "white")
    draw = ImageDraw.Draw(image)
    regions = []
    for index in range(6):
        top = 25 + index * 42
        draw.rectangle((28, top + 4, 35, top + 11), fill="black")
        regions.append(
            _region(
                f"list-item-{index}",
                f"Financial statement bullet {index + 1}",
                (48, top, 560, top + 18),
                index + 1,
            )
        )
    image.save(source)

    result = process_document(
        source,
        FixedReader(regions),
        stages=[GeometricControlStage()],
    )

    assert not any(region.kind == "checkbox" for region in result.pages[0].regions)


def test_control_stage_preserves_numbered_narrative_mark_for_review(
    tmp_path: Path,
) -> None:
    source = tmp_path / "numbered-narrative.png"
    image = Image.new("L", (360, 100), "white")
    draw = ImageDraw.Draw(image)
    draw.line((25, 31, 35, 41), fill="black", width=2)
    draw.line((35, 31, 25, 41), fill="black", width=2)
    image.save(source)
    label = _region(
        "narrative",
        "2. Medication regimen (purpose, side effects)",
        (43, 29, 330, 43),
        2,
    )

    result = process_document(
        source,
        FixedReader([label]),
        stages=[GeometricControlStage(minimum_group_size=1)],
    )

    control = next(
        region for region in result.pages[0].regions if region.kind == "checkbox"
    )
    assert control.structure["observed_state"] == "selected"
    assert control.structure["state"] == "ambiguous"
    assert control.resolution == "unreadable"
    assert result.pages[0].route == "review"


def test_narrative_marks_route_review_without_asserting_selected_controls(
    tmp_path: Path,
) -> None:
    source = tmp_path / "narrative-marks.png"
    image = Image.new("L", (620, 320), "white")
    draw = ImageDraw.Draw(image)
    lines = [
        "Patient continues to require supervision during mobility",
        "Medication teaching completed with patient and caregiver",
        "Therapist reviewed transfer safety during the home visit",
        "The patient tolerated treatment without reported discomfort",
        "Care coordination was completed with the nursing team",
        "Follow up assessment remains planned for next week",
    ]
    regions = []
    for index, text in enumerate(lines):
        top = 25 + index * 42
        draw.line((28, top + 2, 40, top + 14), fill="black", width=2)
        draw.line((40, top + 2, 28, top + 14), fill="black", width=2)
        regions.append(
            _region(
                f"narrative-{index}",
                text,
                (52, top, 570, top + 18),
                index + 1,
            )
        )
    image.save(source)

    result = process_document(
        source,
        FixedReader(regions),
        stages=[GeometricControlStage()],
    )

    controls = [
        region for region in result.pages[0].regions if region.kind == "checkbox"
    ]
    assert len(controls) == 6
    assert all(
        control.structure["observed_state"] == "selected" for control in controls
    )
    assert all(control.structure["state"] == "ambiguous" for control in controls)
    assert all(control.resolution == "unreadable" for control in controls)
    assert result.pages[0].route == "review"


def test_short_clinical_rows_cannot_resolve_unboxed_marks(
    tmp_path: Path,
) -> None:
    source = tmp_path / "short-clinical-rows.png"
    image = Image.new("L", (680, 320), "white")
    draw = ImageDraw.Draw(image)
    lines = [
        "No acute distress",
        "Patient reports no pain",
        "Assessment:",
        "Plan:",
        "No medication changes",
        "Follow up option",
    ]
    regions = []
    for index, text in enumerate(lines):
        top = 25 + index * 42
        draw.line((28, top + 2, 40, top + 14), fill="black", width=2)
        draw.line((40, top + 2, 28, top + 14), fill="black", width=2)
        regions.append(
            _region(
                f"clinical-no-{index}",
                text,
                (52, top, 630, top + 18),
                index + 1,
            )
        )
    image.save(source)

    result = process_document(
        source,
        FixedReader(regions),
        stages=[GeometricControlStage()],
    )

    controls = [
        region for region in result.pages[0].regions if region.kind == "checkbox"
    ]
    assert len(controls) == 6
    assert all(control.structure["state"] == "ambiguous" for control in controls)
    assert all(control.resolution == "unreadable" for control in controls)
    assert result.pages[0].route == "review"


def test_control_stage_emits_semantic_labels_with_complete_evidence(
    tmp_path: Path,
) -> None:
    source = tmp_path / "semantic-labels.png"
    image = Image.new("L", (420, 180), "white")
    draw = ImageDraw.Draw(image)
    for left, top in ((24, 30), (204, 88), (230, 136)):
        draw.rectangle((left, top, left + 14, top + 14), outline="black", width=1)
        draw.line((left + 3, top + 3, left + 11, top + 11), fill="black", width=2)
        draw.line((left + 11, top + 3, left + 3, top + 11), fill="black", width=2)
    image.save(source)
    labels = [
        _region(
            "short-label",
            "Foley (specify type and specific orders)",
            (43, 29, 190, 43),
            2,
        ),
        _region("section", "Physical Therapy:", (90, 89, 198, 104), 3),
        _region("action", "Evaluate and Treat", (225, 90, 340, 105), 4),
        _region("trailing-colon", "History and Physical:", (95, 139, 225, 154), 5),
    ]

    result = process_document(
        source,
        FixedReader(labels),
        stages=[GeometricControlStage(minimum_group_size=1)],
    )

    controls = [
        region for region in result.pages[0].regions if region.kind == "checkbox"
    ]
    assert [control.structure["label"] for control in controls] == [
        "Foley (specify type and specific orders)",
        "Physical Therapy Evaluate and Treat",
        "History and Physical",
    ]
    assert [control.structure["label_evidence_ids"] for control in controls] == [
        ["short-label"],
        ["section", "action"],
        ["trailing-colon"],
    ]
    assert [control.text_provenance["label_evidence_ids"] for control in controls] == [
        ["short-label"],
        ["section", "action"],
        ["trailing-colon"],
    ]
    assert [control.reading_order for control in controls] == [2, 3, 5]


def test_large_filled_input_box_is_not_asserted_as_checkbox(tmp_path: Path) -> None:
    source = tmp_path / "filled-input.png"
    image = Image.new("L", (1000, 1000), "white")
    draw = ImageDraw.Draw(image)
    for left, top in ((30, 30), (30, 90)):
        draw.rectangle((left, top, left + 29, top + 29), outline="black", width=2)
    draw.rectangle((300, 300, 359, 359), outline="black", width=2)
    draw.line((308, 308, 351, 351), fill="black", width=5)
    draw.line((351, 308, 308, 351), fill="black", width=5)
    image.save(source)
    labels = [
        _region("small-a", "Option A", (70, 35, 150, 55), 1),
        _region("small-b", "Option B", (70, 95, 150, 115), 2),
        _region("large", "Code entry", (375, 315, 500, 345), 3),
    ]

    result = process_document(
        source,
        FixedReader(labels),
        stages=[GeometricControlStage(minimum_group_size=1)],
    )

    large = next(
        region
        for region in result.pages[0].regions
        if region.kind == "checkbox" and region.structure["label"] == "Code entry"
    )
    assert large.structure["state"] == "ambiguous"
    assert large.structure["observed_state"] == "selected"
    assert large.structure["selection_supported"] is False


def test_isolated_large_selected_geometry_is_ambiguous(tmp_path: Path) -> None:
    source = tmp_path / "isolated-large-selected.png"
    image = Image.new("L", (1000, 1000), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((300, 300, 359, 359), outline="black", width=2)
    draw.line((308, 308, 351, 351), fill="black", width=5)
    draw.line((351, 308, 308, 351), fill="black", width=5)
    image.save(source)
    label = _region("label", "Isolated option", (375, 315, 520, 345), 1)

    result = process_document(
        source,
        FixedReader([label]),
        stages=[GeometricControlStage(minimum_group_size=1)],
    )

    control = next(
        region for region in result.pages[0].regions if region.kind == "checkbox"
    )
    assert control.text == "[?] Isolated option"
    assert control.structure["observed_state"] == "selected"
    assert control.structure["selection_supported"] is False
    assert control.resolution == "unreadable"


def test_large_selected_checkbox_with_row_peer_remains_selected(
    tmp_path: Path,
) -> None:
    source = tmp_path / "large-pair.png"
    image = Image.new("L", (1000, 1000), "white")
    draw = ImageDraw.Draw(image)
    for left, top in ((30, 30), (30, 90)):
        draw.rectangle((left, top, left + 29, top + 29), outline="black", width=2)
    draw.rectangle((300, 300, 359, 359), outline="black", width=2)
    draw.line((308, 308, 351, 351), fill="black", width=5)
    draw.line((351, 308, 308, 351), fill="black", width=5)
    draw.rectangle((600, 300, 659, 359), outline="black", width=2)
    image.save(source)
    labels = [
        _region("small-a", "Option A", (70, 35, 150, 55), 1),
        _region("small-b", "Option B", (70, 95, 150, 115), 2),
        _region("selected", "Plan", (375, 315, 480, 345), 3),
        _region("peer", "Update", (675, 315, 790, 345), 4),
    ]

    result = process_document(
        source,
        FixedReader(labels),
        stages=[GeometricControlStage(minimum_group_size=1)],
    )

    selected = next(
        region
        for region in result.pages[0].regions
        if region.kind == "checkbox" and region.structure["label"] == "Plan"
    )
    assert selected.structure["state"] == "selected"
    assert selected.structure["selection_supported"] is True


def test_control_stage_ignores_underlines_and_single_diagonal_noise(
    tmp_path: Path,
) -> None:
    source = tmp_path / "not-selected.png"
    image = Image.new("L", (260, 100), "white")
    draw = ImageDraw.Draw(image)
    draw.line((20, 45, 80, 45), fill="black", width=2)
    draw.line((190, 24, 204, 39), fill="black", width=2)
    image.save(source)
    labels = [
        _region("left-label", "First option", (88, 37, 160, 50), 2),
        _region("right-label", "Second option", (105, 60, 185, 74), 3),
    ]

    result = process_document(
        source,
        FixedReader(labels),
        stages=[GeometricControlStage(minimum_group_size=1)],
    )

    assert all(region.kind != "checkbox" for region in result.pages[0].regions)


def test_control_stage_rejects_text_bearing_icon_and_keeps_real_boxes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "screen-controls.png"
    image = Image.new("L", (1000, 1000), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((80, 80, 149, 149), outline="black", width=2)
    draw.text((91, 103), "People", fill="black")
    draw.rectangle((80, 260, 109, 289), outline="black", width=2)
    draw.rectangle((80, 330, 109, 359), outline="black", width=2)
    image.save(source)
    regions = [
        _region("icon-label", "People", (91, 103, 137, 119), 1),
        _region("saas", "SaaS", (122, 265, 190, 285), 2),
        _region("one-time", "One-Time", (122, 335, 220, 355), 3),
    ]

    result = process_document(
        source,
        FixedReader(regions),
        stages=[GeometricControlStage(minimum_group_size=1)],
    )

    controls = [
        region for region in result.pages[0].regions if region.kind == "checkbox"
    ]
    assert [control.text for control in controls] == ["[ ] SaaS", "[ ] One-Time"]


def test_control_stage_rejects_sparse_unselected_list_boxes(tmp_path: Path) -> None:
    source = tmp_path / "sparse-list-boxes.png"
    image = Image.new("L", (1000, 1000), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((80, 260, 109, 289), outline="black", width=2)
    draw.rectangle((80, 330, 109, 359), outline="black", width=2)
    image.save(source)
    regions = [
        _region("first-item", "First list item", (122, 265, 250, 285), 2),
        _region("second-item", "Second list item", (122, 335, 270, 355), 3),
    ]

    result = process_document(
        source,
        FixedReader(regions),
        stages=[GeometricControlStage()],
    )

    assert all(region.kind != "checkbox" for region in result.pages[0].regions)


def test_control_stage_preserves_single_unselected_form_field(tmp_path: Path) -> None:
    source = tmp_path / "single-form-control.png"
    image = Image.new("L", (400, 400), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 40, 55, 65), outline="black", width=2)
    image.save(source)
    label = _region("consent", "Patient consent", (65, 42, 210, 64), 1)
    label.kind = "form_field"

    result = process_document(
        source,
        FixedReader([label]),
        stages=[GeometricControlStage()],
    )

    control = next(
        region for region in result.pages[0].regions if region.kind == "checkbox"
    )
    assert control.text == "[ ] Patient consent"
    assert control.structure["coverage_status"] == "insufficient_control_group"


def test_control_stage_recovers_selected_marks_inside_semantic_table_cells(
    tmp_path: Path,
) -> None:
    source = tmp_path / "table-check.png"
    image = Image.new("L", (600, 260), "white")
    draw = ImageDraw.Draw(image)
    draw.line((365, 106, 374, 115), fill="black", width=3)
    draw.line((374, 115, 394, 88), fill="black", width=3)
    image.save(source)
    cells = [
        _cell("header-task", 0, 0, "TASK", (20, 40, 180, 80)),
        _cell("header-mon", 0, 1, "MON", (180, 40, 260, 80)),
        _cell("header-tues", 0, 2, "TUES", (260, 40, 340, 80)),
        _cell("header-wed", 0, 3, "WED", (340, 40, 420, 80)),
        _cell("row-label", 1, 0, "Bed Bath", (20, 80, 180, 125)),
        _cell("row-mon", 1, 1, "", (180, 80, 260, 125)),
        _cell("row-tues", 1, 2, "", (260, 80, 340, 125)),
        _cell("row-wed", 1, 3, "hm", (340, 80, 420, 125)) | {"confidence": 0.25},
    ]
    table = TextRegion(
        id="table",
        kind="table",
        text="",
        confidence=0.96,
        bounding_box=BoundingBox(20, 40, 420, 125),
        reading_order=10,
        provider="table-transformer",
        structure={"role": "table", "cells": cells},
    )

    result = process_document(
        source,
        FixedReader([table]),
        stages=[GeometricControlStage()],
    )

    control = next(
        region for region in result.pages[0].regions if region.kind == "checkbox"
    )
    assert control.text == "[x] Bed Bath (WED)"
    assert control.resolution == "resolved"
    assert control.structure["selection_supported"] is True
    assert control.structure["coverage_status"] == "detected"
    assert control.text_provenance == {
        "method": "table_cell_residual_ink",
        "label_evidence_ids": ["row-label", "header-wed"],
        "source_evidence_ids": ["table", "row-wed"],
    }
    assert control.structure["source_evidence_ids"] == ["table", "row-wed"]


def test_control_stage_rejects_single_diagonal_table_text_fragments(
    tmp_path: Path,
) -> None:
    source = tmp_path / "table-text-fragments.png"
    image = Image.new("L", (500, 180), "white")
    draw = ImageDraw.Draw(image)
    draw.line((215, 100, 230, 116), fill="black", width=3)
    draw.line((295, 100, 310, 116), fill="black", width=3)
    image.save(source)
    cells = [
        _cell("header-task", 0, 0, "TASK", (20, 40, 180, 80)),
        _cell("header-mon", 0, 1, "MON", (180, 40, 260, 80)),
        _cell("header-tues", 0, 2, "TUES", (260, 40, 340, 80)),
        _cell("header-wed", 0, 3, "WED", (340, 40, 420, 80)),
        _cell("row-label", 1, 0, "Financial values", (20, 80, 180, 125)),
        _cell("row-mon", 1, 1, "", (180, 80, 260, 125)),
        _cell("row-tues", 1, 2, "", (260, 80, 340, 125)),
        _cell("row-wed", 1, 3, "", (340, 80, 420, 125)),
    ]
    table = TextRegion(
        id="table",
        kind="table",
        text="",
        confidence=0.96,
        bounding_box=BoundingBox(20, 40, 420, 125),
        reading_order=10,
        provider="table-transformer",
        structure={"role": "table", "cells": cells},
    )

    result = process_document(
        source,
        FixedReader([table]),
        stages=[GeometricControlStage()],
    )

    assert all(region.kind != "checkbox" for region in result.pages[0].regions)


def test_control_stage_does_not_treat_financial_values_as_control_headers(
    tmp_path: Path,
) -> None:
    source = tmp_path / "financial-table.png"
    image = Image.new("L", (760, 360), "white")
    draw = ImageDraw.Draw(image)
    for x, y in ((235, 155), (535, 155), (315, 305), (535, 305)):
        draw.line((x, y, x + 8, y + 8), fill="black", width=3)
        draw.line((x + 8, y + 8, x + 24, y - 12), fill="black", width=3)
    image.save(source)

    table_rows = [
        ["", "2005", "2014", "2023", "2024"],
        ["Revenue", "$187", "$487", "$1,127", "$1,064"],
        ["Total payments volume", "NA", "$1.6", "$5.9", "SE"],
        ["Credit card loans market share", "", "17%", "17%", ""],
    ]
    rank_rows = [
        ["", "2006", "", "", ""],
        ["Total Markets revenue", "#8", "dl", "#1", "CA"],
        ["Global Investment banking fees", "#2", "ud", "#1", "a"],
    ]
    tables = []
    for table_id, top, rows in (
        ("consumer-table", 30, table_rows),
        ("markets-table", 210, rank_rows),
    ):
        cells = []
        for row, values in enumerate(rows):
            for column, text in enumerate(values):
                left = 20 if column == 0 else 200 + (column - 1) * 100
                right = 200 if column == 0 else left + 100
                cell = _cell(
                    f"{table_id}-{row}-{column}",
                    row,
                    column,
                    text,
                    (left, top + row * 40, right, top + (row + 1) * 40),
                )
                cells.append(cell | {"confidence": 0.4})
        tables.append(
            TextRegion(
                id=table_id,
                kind="table",
                text="",
                confidence=0.96,
                bounding_box=BoundingBox(20, top, 600, top + len(rows) * 40),
                reading_order=10,
                provider="table-transformer",
                structure={"role": "table", "cells": cells},
            )
        )

    result = process_document(
        source,
        FixedReader(tables),
        stages=[GeometricControlStage()],
    )

    assert all(region.kind != "checkbox" for region in result.pages[0].regions)


def test_financial_artifact_does_not_promote_scattered_glyphs_as_controls() -> None:
    source = Path(__file__).parents[1] / "artifacts/demo/financial_table.png"
    labels = [
        _region("serve", "Serve", (732, 179, 767, 207), 1),
        _region("deposit", "deposit", (751, 307, 795, 320), 2),
        _region("of-1", "of", (210, 462, 220, 471), 3),
        _region("of-2", "of", (210, 480, 220, 489), 4),
        _region("first", "first", (863, 760, 886, 770), 5),
        _region("morgan", "Morgan", (756, 829, 800, 842), 6),
        _region("of-3", "of", (210, 915, 221, 925), 7),
        _region("payments", "Payments", (314, 949, 372, 962), 8),
        _region("of-4", "of", (210, 1035, 221, 1045), 9),
        _region("of-5", "of", (210, 1265, 221, 1275), 10),
    ]

    result = process_document(
        source,
        FixedReader(labels),
        stages=[GeometricControlStage()],
    )

    assert all(region.kind != "checkbox" for region in result.pages[0].regions)


def test_historical_formula_region_owns_math_marks_before_control_detection() -> None:
    source = Path(__file__).parents[1] / "artifacts/demo/formula_scan.png"
    formula = TextRegion(
        id="formula",
        kind="formula",
        text="integral equation x dx = result",
        confidence=0.99,
        bounding_box=BoundingBox(80, 370, 430, 535),
        reading_order=1,
        provider="nemotron-ocr-v2",
        structure={"semantic_class": "formula"},
    )

    result = process_document(
        source,
        FixedReader([formula]),
        stages=[GeometricControlStage(minimum_group_size=1)],
    )

    assert all(region.kind != "checkbox" for region in result.pages[0].regions)


def test_control_stage_recovers_same_cell_controls_from_ruled_form(
    tmp_path: Path,
) -> None:
    source = tmp_path / "ruled-form-controls.png"
    image = np.full((150, 620), 255, dtype=np.uint8)
    for x in (20, 210, 400, 590):
        cv2.line(image, (x, 30), (x, 120), 0, 2)
    for y in (30, 120):
        cv2.line(image, (20, y), (590, y), 0, 2)
    _draw_grid_connected_control(image, 20, selected=True)
    _draw_grid_connected_control(image, 210, selected=False)
    cv2.putText(
        image,
        "Notes only",
        (420, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        0,
        1,
        cv2.LINE_8,
    )
    cv2.imwrite(str(source), image)
    source_pixels = image.copy()
    cells = [
        _cell("inpatient", 0, 0, "[x] Inpatient", (20, 30, 210, 120)),
        _cell("outpatient", 0, 1, "[ ] Outpatient", (210, 30, 400, 120)),
        _cell("notes", 0, 2, "Notes only", (400, 30, 590, 120)),
    ]
    table = TextRegion(
        id="form-table",
        kind="table",
        text="",
        confidence=0.97,
        bounding_box=BoundingBox(20, 30, 590, 120),
        reading_order=7,
        provider="table-transformer",
        structure={"role": "table", "cells": cells},
    )

    result = process_document(
        source,
        FixedReader([table]),
        stages=[GeometricControlStage()],
    )

    controls = [
        region for region in result.pages[0].regions if region.kind == "checkbox"
    ]
    assert [control.text for control in controls] == [
        "[x] Inpatient",
        "[ ] Outpatient",
    ]
    assert [control.structure["label_evidence_ids"] for control in controls] == [
        ["inpatient"],
        ["outpatient"],
    ]
    assert [control.structure["source_evidence_ids"] for control in controls] == [
        ["form-table", "inpatient"],
        ["form-table", "outpatient"],
    ]
    assert [control.reading_order for control in controls] == [7, 7]
    assert all(
        control.text_provenance["method"] == "table_cell_residual_ink"
        for control in controls
    )
    assert np.array_equal(cv2.imread(str(source), cv2.IMREAD_GRAYSCALE), source_pixels)


def test_control_stage_rejects_multiple_boxes_in_same_table_cell(
    tmp_path: Path,
) -> None:
    source = tmp_path / "ambiguous-ruled-form-cell.png"
    image = np.full((150, 240), 255, dtype=np.uint8)
    cv2.rectangle(image, (20, 30), (220, 120), 0, 2)
    cv2.rectangle(image, (35, 30), (53, 48), 0, 2)
    cv2.rectangle(image, (65, 30), (83, 48), 0, 2)
    cv2.putText(
        image,
        "Select one",
        (95, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        0,
        1,
        cv2.LINE_8,
    )
    cv2.imwrite(str(source), image)
    cell = _cell("ambiguous", 0, 0, "[ ] Select one", (20, 30, 220, 120))
    table = TextRegion(
        id="form-table",
        kind="table",
        text="",
        confidence=0.97,
        bounding_box=BoundingBox(20, 30, 220, 120),
        reading_order=1,
        provider="table-transformer",
        structure={"role": "table", "cells": [cell]},
    )

    result = process_document(
        source,
        FixedReader([table]),
        stages=[GeometricControlStage()],
    )

    assert not any(region.kind == "checkbox" for region in result.pages[0].regions)


def test_control_stage_recovers_labeled_controls_from_rejected_table_candidate(
    tmp_path: Path,
) -> None:
    source = tmp_path / "rejected-form-table.png"
    image = np.full((150, 620), 255, dtype=np.uint8)
    for x in (20, 210, 400, 590):
        cv2.line(image, (x, 30), (x, 120), 0, 2)
    for y in (30, 55, 120):
        cv2.line(image, (20, y), (590, y), 0, 2)
    labels = []
    for index, (left, text) in enumerate(
        ((20, "Bed mobility"), (210, "Gait training"), (400, "Home exercise")),
        start=1,
    ):
        cv2.rectangle(image, (left + 8, 55), (left + 26, 73), 0, 2)
        cv2.putText(
            image,
            text,
            (left + 30, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            0,
            1,
            cv2.LINE_8,
        )
        labels.append(
            _region(f"label-{index}", text, (left + 30, 58, left + 150, 75), index)
        )
    cv2.line(image, (32, 59), (42, 69), 0, 2)
    cv2.line(image, (42, 59), (32, 69), 0, 2)
    image[65:66, 415:421] = 210
    cv2.imwrite(str(source), image)
    source_pixels = image.copy()
    candidate = TextRegion(
        id="rejected-form",
        kind="table_candidate",
        text="",
        confidence=None,
        bounding_box=BoundingBox(20, 30, 590, 120),
        reading_order=10,
        provider="table-transformer",
        text_provenance={
            "method": "table_semantics_rejection",
            "source_region_ids": [label.id for label in labels],
        },
        resolution="unreadable",
        structure={
            "role": "table_candidate",
            "status": "rejected",
            "reason": "unsupported_spanning_layout",
        },
    )

    result = process_document(
        source,
        FixedReader([*labels, candidate]),
        stages=[GeometricControlStage()],
    )

    controls = [
        region for region in result.pages[0].regions if region.kind == "checkbox"
    ]
    assert [control.text for control in controls] == [
        "[x] Bed mobility",
        "[ ] Gait training",
        "[?] Home exercise",
    ]
    assert [control.resolution for control in controls] == [
        "resolved",
        "resolved",
        "unreadable",
    ]
    assert [control.structure["label_evidence_ids"] for control in controls] == [
        ["label-1"],
        ["label-2"],
        ["label-3"],
    ]
    assert all(
        control.structure["source_evidence_ids"] == ["rejected-form"]
        for control in controls
    )
    assert np.array_equal(cv2.imread(str(source), cv2.IMREAD_GRAYSCALE), source_pixels)


def test_control_stage_rejects_ambiguous_and_repeated_candidate_labels(
    tmp_path: Path,
) -> None:
    source = tmp_path / "rejected-numeric-grid.png"
    image = np.full((180, 260), 255, dtype=np.uint8)
    for y in (20, 60, 100, 140):
        cv2.line(image, (20, y), (240, y), 0, 2)
    cv2.line(image, (20, 20), (20, 140), 0, 2)
    cv2.line(image, (240, 20), (240, 140), 0, 2)
    labels = []
    for index, top in enumerate((20, 60, 100), start=1):
        cv2.rectangle(image, (30, top), (48, top + 18), 0, 2)
        cv2.putText(
            image,
            "Max",
            (52, top + 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            0,
            1,
            cv2.LINE_8,
        )
        labels.append(
            _region(f"max-{index}", "Max", (52, top + 3, 85, top + 20), index)
        )
    cv2.rectangle(image, (110, 100), (128, 118), 0, 2)
    unique = _region("unique", "Transfer", (132, 103, 205, 120), 4)
    cv2.imwrite(str(source), image)
    candidate = TextRegion(
        id="numeric-grid",
        kind="table_candidate",
        text="",
        confidence=None,
        bounding_box=BoundingBox(20, 20, 240, 140),
        reading_order=10,
        provider="table-transformer",
        text_provenance={
            "method": "table_semantics_rejection",
            "source_region_ids": [region.id for region in [*labels, unique]],
        },
        resolution="unreadable",
        structure={"role": "table_candidate", "status": "rejected"},
    )

    result = process_document(
        source,
        FixedReader([*labels, unique, candidate]),
        stages=[GeometricControlStage()],
    )

    assert not any(region.kind == "checkbox" for region in result.pages[0].regions)


def test_control_stage_does_not_treat_short_table_text_as_a_mark(
    tmp_path: Path,
) -> None:
    source = tmp_path / "short-cell-text.png"
    image = Image.new("L", (600, 260), "white")
    draw = ImageDraw.Draw(image)
    draw.text((355, 92), "ST", fill="black")
    image.save(source)
    cells = [
        _cell("header-task", 0, 0, "TASK", (20, 40, 180, 80)),
        _cell("header-mon", 0, 1, "MON", (180, 40, 260, 80)),
        _cell("header-tues", 0, 2, "TUES", (260, 40, 340, 80)),
        _cell("header-wed", 0, 3, "WED", (340, 40, 420, 80)),
        _cell("row-label", 1, 0, "Bed Bath", (20, 80, 180, 125)),
        _cell("row-wed", 1, 3, "ST", (340, 80, 420, 125)) | {"confidence": 0.95},
    ]
    table = TextRegion(
        id="table",
        kind="table",
        text="",
        confidence=0.96,
        bounding_box=BoundingBox(20, 40, 420, 125),
        reading_order=10,
        provider="table-transformer",
        structure={"role": "table", "cells": cells},
    )

    result = process_document(
        source,
        FixedReader([table]),
        stages=[GeometricControlStage()],
    )

    assert not any(region.kind == "checkbox" for region in result.pages[0].regions)


def test_control_stage_rejects_diagonal_stroke_in_generic_results_table(
    tmp_path: Path,
) -> None:
    source = tmp_path / "lab-results.png"
    image = Image.new("L", (600, 240), "white")
    draw = ImageDraw.Draw(image)
    draw.line((245, 106, 254, 115), fill="black", width=3)
    draw.line((254, 115, 274, 88), fill="black", width=3)
    image.save(source)
    cells = [
        _cell("test-header", 0, 0, "TEST", (20, 40, 220, 80)),
        _cell("result-header", 0, 1, "RESULT", (220, 40, 300, 80)),
        _cell("unit-header", 0, 2, "UNIT", (300, 40, 420, 80)),
        _cell("test", 1, 0, "Hemoglobin", (20, 80, 220, 125)),
        _cell("result", 1, 1, "", (220, 80, 300, 125)) | {"confidence": 0.2},
        _cell("unit", 1, 2, "g/dL", (300, 80, 420, 125)),
    ]
    table = TextRegion(
        id="lab-table",
        kind="table",
        text="",
        confidence=0.98,
        bounding_box=BoundingBox(20, 40, 420, 125),
        reading_order=1,
        provider="table-transformer",
        structure={"role": "table", "cells": cells},
    )

    result = process_document(
        source,
        FixedReader([table]),
        stages=[GeometricControlStage()],
    )

    assert not any(region.kind == "checkbox" for region in result.pages[0].regions)


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


def _cell(
    cell_id: str,
    row: int,
    column: int,
    text: str,
    box: tuple[int, int, int, int],
) -> dict[str, object]:
    return {
        "id": cell_id,
        "bbox": dict(zip(("left", "top", "right", "bottom"), box, strict=True)),
        "row_nums": [row],
        "column_nums": [column],
        "text": text,
    }


def _draw_grid_connected_control(
    image: np.ndarray,
    cell_left: int,
    *,
    selected: bool,
) -> None:
    cv2.rectangle(image, (cell_left, 62), (cell_left + 18, 80), 0, 2)
    if selected:
        cv2.line(image, (cell_left + 4, 66), (cell_left + 14, 76), 0, 2)
        cv2.line(image, (cell_left + 14, 66), (cell_left + 4, 76), 0, 2)
    cv2.putText(
        image,
        "Inpatient" if selected else "Outpatient",
        (cell_left + 28, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        0,
        1,
        cv2.LINE_8,
    )


class FixedReader:
    name = "fixed"

    def __init__(self, regions: list[TextRegion]) -> None:
        self.regions = regions

    def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
        return self.regions


def test_a_capital_letter_is_not_a_ring_annotation(tmp_path: Path) -> None:
    """At scan resolution a capital O satisfies every ring test except its shape:
    an annotation encircles words, so it is wider than it is tall."""
    source = tmp_path / "prose.png"
    image = Image.new("L", (600, 200), "white")
    draw = ImageDraw.Draw(image)
    # a letter-shaped ring: roughly square, hollow, with an enclosed hole
    draw.ellipse((60, 60, 96, 100), outline="black", width=4)
    image.save(source)
    label = _region("line", "Agreement between the parties", (40, 55, 560, 105), 1)

    result = process_document(
        source,
        FixedReader([label]),
        stages=[GeometricControlStage(minimum_group_size=1)],
    )

    assert all(
        region.kind != "checkbox"
        or region.text_provenance["method"] != "enclosing_ring_annotation"
        for region in result.pages[0].regions
    )


def test_mark_shape_vocabulary_names_ticks_crosses_and_null_glyphs() -> None:
    """A tick, a cross and a slashed loop must be told apart, not merged into "a mark"."""
    cv2 = pytest.importorskip("cv2")
    numpy = pytest.importorskip("numpy")
    from ocr_pipeline.controls import _mark_shape

    def canvas():
        return numpy.zeros((40, 40), dtype="uint8")

    tick = canvas()
    cv2.line(tick, (8, 20), (16, 30), 255, 3)
    cv2.line(tick, (16, 30), (32, 8), 255, 3)

    cross = canvas()
    cv2.line(cross, (6, 6), (34, 34), 255, 3)
    cv2.line(cross, (34, 6), (6, 34), 255, 3)

    slashed = canvas()
    cv2.ellipse(slashed, (20, 20), (13, 9), 0, 0, 360, 255, 3)
    cv2.line(slashed, (8, 30), (32, 10), 255, 3)

    assert _mark_shape(tick, cv2, numpy) == "tick"
    assert _mark_shape(cross, cv2, numpy) == "cross"
    assert _mark_shape(slashed, cv2, numpy) == "slashed_loop"
    assert _mark_shape(canvas(), cv2, numpy) == "unknown"


def test_boxed_marks_record_the_glyph_without_breaking_task_list_syntax() -> None:
    """A ticked box and a crossed box differ in structure, not in the markdown text."""
    cv2 = pytest.importorskip("cv2")
    numpy = pytest.importorskip("numpy")
    from ocr_pipeline.controls import BOXED_GLYPHS, _boxed_mark_shape

    def inner(draw):
        canvas = numpy.full((40, 40), 255, dtype="uint8")
        draw(canvas)
        return canvas

    ticked = inner(
        lambda c: (
            cv2.line(c, (8, 20), (16, 30), 0, 3),
            cv2.line(c, (16, 30), (32, 8), 0, 3),
        )
    )
    crossed = inner(
        lambda c: (
            cv2.line(c, (6, 6), (34, 34), 0, 3),
            cv2.line(c, (34, 6), (6, 34), 0, 3),
        )
    )

    assert _boxed_mark_shape(ticked, 255.0, cv2, numpy) == "tick"
    assert _boxed_mark_shape(crossed, 255.0, cv2, numpy) == "cross"
    assert BOXED_GLYPHS["tick"] == "☑"
    assert BOXED_GLYPHS["cross"] == "☒"
    assert BOXED_GLYPHS["empty"] == "☐"


def test_prose_with_incidental_colons_is_not_treated_as_a_form() -> None:
    """A contract clause mentioning times and sections must not lose mark corroboration.

    `_form_like_regions` drops the required mark-group size from six to one, so declaring
    prose "form-like" is what let a single spurious mark become a checked box.
    """
    from ocr_pipeline.controls import _form_like_regions

    def prose(identifier: str, text: str) -> TextRegion:
        return TextRegion(
            id=identifier,
            kind="word",
            text=text,
            confidence=0.95,
            bounding_box=BoundingBox(0, 0, 40, 12),
            reading_order=1,
            provider="reader",
        )

    incidental = [
        prose(f"c{index}", text)
        for index, text in enumerate(
            ["9:00", "3:1", "Section:", "10:30", "A:B", "ratio:", "1:2", "see:"]
        )
    ]
    assert not _form_like_regions(incidental, "reader"), (
        "colons inside prose are not field labels"
    )

    real_labels = [
        prose(f"f{index}", label)
        for index, label in enumerate(
            [
                "Patient Name:",
                "DOB:",
                "Allergies:",
                "Gender:",
                "Date of Service:",
                "Member Id No:",
            ]
        )
    ]
    assert _form_like_regions(real_labels, "reader"), "trailing colons are field labels"


def test_controls_stand_down_when_the_reader_already_wrote_the_mark(tmp_path) -> None:
    """Detecting a box the reader transcribed renders it twice, as "Colorado ☐ [ ]"."""
    from ocr_pipeline.controls import GeometricControlStage

    image_path = tmp_path / "page.png"
    Image.new("RGB", (400, 200), "white").save(image_path)
    regions = [
        TextRegion(
            id="p1-falcon-1",
            kind="text",
            text="☐Ambulance ☒Skilled Nursing Facility",
            confidence=0.6,
            bounding_box=BoundingBox(10, 10, 380, 40),
            reading_order=1,
            provider="falcon-perception",
        )
    ]

    result = GeometricControlStage().apply(image_path, 1, regions)

    assert result == regions
    assert not [region for region in result if region.kind == "checkbox"]
