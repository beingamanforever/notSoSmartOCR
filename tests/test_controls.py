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


def test_thick_checkbox_border_does_not_imply_selected(tmp_path: Path) -> None:
    source = tmp_path / "thick-border.png"
    image = Image.new("L", (400, 400), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 30, 59, 59), outline="black", width=6)
    image.save(source)

    detections = detect_controls(source)

    assert len(detections) == 1
    assert detections[0].state == "unselected"


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
