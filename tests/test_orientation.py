from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from PIL import Image

from ocr_pipeline.contracts import BoundingBox, TextRegion
from ocr_pipeline.orientation import (
    DocTROrientationDetector,
    OrientationReader,
    detect_tesseract_orientation,
)
from ocr_pipeline.pipeline import process_document
from ocr_pipeline.providers import ReaderError


class AngleReader:
    name = "angle-reader"

    def __init__(
        self,
        confidence_by_angle: dict[int, float] | None = None,
        words_by_angle: dict[int, int] | None = None,
        failures: set[int] | None = None,
    ) -> None:
        self.confidence_by_angle = confidence_by_angle or {}
        self.words_by_angle = words_by_angle or {}
        self.failures = failures or set()
        self.calls: list[int] = []

    def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
        angle = int(image_path.stem.rsplit("-", 1)[1])
        self.calls.append(angle)
        if angle in self.failures:
            raise ReaderError("view_failed", f"angle {angle} failed")
        return [
            TextRegion(
                id=f"p{page_number}-source-{index + 1}",
                kind="word",
                text=f"text at {angle} item {index}",
                confidence=self.confidence_by_angle.get(angle, 0.7),
                bounding_box=BoundingBox(2, 3, 8, 7),
                reading_order=index + 1,
                provider=self.name,
                text_provenance={"method": "controlled"},
            )
            for index in range(self.words_by_angle.get(angle, 1))
        ]


class ReviewAngleReader(AngleReader):
    def page_needs_review(self, page_number: int) -> bool:
        return True


class StructureStage:
    name = "structure"

    def __init__(self) -> None:
        self.image_sizes: list[tuple[int, int]] = []

    def apply(
        self,
        image_path: Path,
        page_number: int,
        regions: list[TextRegion],
    ) -> list[TextRegion]:
        with Image.open(image_path) as image:
            self.image_sizes.append(image.size)
        return regions + [
            TextRegion(
                id=f"p{page_number}-table-1",
                kind="table",
                text="cell",
                confidence=0.9,
                bounding_box=BoundingBox(1, 1, 4, 4),
                reading_order=2,
                provider=self.name,
                structure={
                    "role": "table",
                    "cells": [
                        {
                            "bbox": {"left": 1, "top": 1, "right": 4, "bottom": 4},
                            "span_bboxes": [
                                {"left": 1, "top": 1, "right": 4, "bottom": 4}
                            ],
                        }
                    ],
                },
            )
        ]


@pytest.mark.parametrize(
    ("angle", "expected"),
    [
        (0, BoundingBox(2, 3, 8, 7)),
        (90, BoundingBox(13, 2, 17, 8)),
        (180, BoundingBox(12, 3, 18, 7)),
        (270, BoundingBox(3, 2, 7, 8)),
    ],
)
def test_confident_osd_restores_each_rotation_box(
    tmp_path: Path,
    angle: int,
    expected: BoundingBox,
) -> None:
    image_path = _image(tmp_path)
    source = AngleReader()
    reader = OrientationReader(source, osd_detector=lambda _: _osd(angle, 20.0))

    regions = reader.read(image_path, 1)

    assert source.calls == [angle]
    assert regions[0].id == "p1-source-1"
    assert regions[0].text == f"text at {angle} item 0"
    assert regions[0].provider == source.name
    assert regions[0].bounding_box == expected
    assert regions[0].text_provenance == {
        "method": "controlled",
        "orientation": {
            "angle": angle,
            "selector": "tesseract_osd",
            "score_margin": 1.0,
            "original_size": [20, 10],
            "rotated_size": [20, 10] if angle in {0, 180} else [10, 20],
            "osd": _osd(angle, 20.0),
            "osd_failure": None,
            "view_failures": {},
        },
    }


def test_osd_failure_runs_all_views_and_preserves_failure(tmp_path: Path) -> None:
    image_path = _image(tmp_path)
    source = AngleReader({0: 0.6, 90: 0.7, 180: 0.95, 270: 0.65})

    def fail_osd(_: Path) -> dict[str, object]:
        raise ReaderError("osd_failed", "no orientation")

    reader = OrientationReader(source, osd_detector=fail_osd)

    regions = reader.read(image_path, 1)

    assert source.calls == [0, 90, 180, 270]
    assert regions[0].text == "text at 180 item 0"
    orientation = regions[0].text_provenance["orientation"]
    assert orientation["selector"] == "evidence_fallback"
    assert orientation["osd_failure"] == {
        "code": "osd_failed",
        "message": "no orientation",
    }
    assessment = reader.coverage_assessment(1)
    assert assessment["status"] == "review_recommended"
    assert assessment["pages"][0]["angle"] == 180


def test_weak_osd_runs_four_angle_fallback(tmp_path: Path) -> None:
    image_path = _image(tmp_path)
    source = AngleReader({0: 0.6, 90: 0.9, 180: 0.7, 270: 0.65})
    reader = OrientationReader(source, osd_detector=lambda _: _osd(180, 2.0))

    regions = reader.read(image_path, 1)

    assert source.calls == [0, 90, 180, 270]
    assert regions[0].text == "text at 90 item 0"
    page = reader.coverage_assessment(1)["pages"][0]
    assert page["osd_status"] == "weak"
    assert page["selector"] == "evidence_fallback"


def test_surviving_view_retains_other_reader_failure(tmp_path: Path) -> None:
    image_path = _image(tmp_path)
    source = AngleReader(
        {0: 0.6, 90: 0.7, 180: 0.95, 270: 0.65},
        failures={90},
    )
    reader = OrientationReader(source, osd_executable=None)

    regions = reader.read(image_path, 1)

    assert regions[0].text == "text at 180 item 0"
    assert regions[0].text_provenance["orientation"]["view_failures"] == {
        "90": {"code": "view_failed", "message": "angle 90 failed"}
    }


def test_classifier_zero_skips_osd_and_uses_one_ocr_call(tmp_path: Path) -> None:
    image_path = _image(tmp_path)
    source = AngleReader()

    def unexpected_osd(_: Path) -> dict[str, object]:
        raise AssertionError("OSD must be skipped for a zero-degree prediction")

    reader = OrientationReader(
        source,
        orientation_detector=lambda _: {"angle": 0, "confidence": 0.98},
        osd_detector=unexpected_osd,
    )

    regions = reader.read(image_path, 1)

    assert source.calls == [0]
    assert regions[0].text == "text at 0 item 0"
    page = reader.coverage_assessment(1)["pages"][0]
    assert page["selector"] == "orientation_classifier"
    assert page["osd_status"] == "skipped_zero_prediction"


def test_orientation_propagates_selected_reader_review(tmp_path: Path) -> None:
    image_path = _image(tmp_path)
    source = ReviewAngleReader()
    reader = OrientationReader(
        source,
        orientation_detector=lambda _: {"angle": 0, "confidence": 0.98},
    )

    result = process_document(image_path, reader)

    assert result.pages[0].route == "review"
    page = reader.coverage_assessment(1)["pages"][0]
    assert page["nested_reader_reviews"] == {"0": True}
    assert result.pages[0].regions[0].text_provenance["orientation"][
        "nested_reader_reviews"
    ] == {"0": True}


def test_uncertain_zero_prediction_recovers_sideways_text_coverage(
    tmp_path: Path,
) -> None:
    image_path = _image(tmp_path)
    source = AngleReader(
        {0: 0.72, 90: 0.71, 180: 0.71, 270: 0.7},
        words_by_angle={0: 10, 90: 100, 180: 120, 270: 10},
    )
    reader = OrientationReader(
        source,
        orientation_detector=lambda _: {"angle": 0, "confidence": 0.82},
        osd_detector=lambda _: _osd(0, 20.0),
    )

    result = process_document(image_path, reader)

    assert source.calls == [0, 90, 180, 270]
    assert result.status == "success"
    assert result.pages[0].route == "review"
    assert len(result.pages[0].regions) == 120
    assert result.pages[0].regions[0].bounding_box == BoundingBox(12, 3, 18, 7)
    page = reader.coverage_assessment(1)["pages"][0]
    assert page["angle"] == 180
    assert page["selector"] == "orientation_evidence_fallback"
    assert page["view_selection_reason"] == "coverage_recovery"


def test_classifier_osd_disagreement_scores_only_two_views(tmp_path: Path) -> None:
    image_path = _image(tmp_path)
    source = AngleReader({0: 0.6, 270: 0.95})
    reader = OrientationReader(
        source,
        orientation_detector=lambda _: {"angle": 270, "confidence": 0.99},
        osd_detector=lambda _: _osd(0, 20.0),
    )

    regions = reader.read(image_path, 1)

    assert source.calls == [270, 0]
    assert regions[0].text == "text at 270 item 0"
    page = reader.coverage_assessment(1)["pages"][0]
    assert page["selector"] == "orientation_evidence_fallback"
    assert page["status"] == "review_recommended"


def test_classifier_fallback_recovers_materially_broader_view(
    tmp_path: Path,
) -> None:
    image_path = _image(tmp_path)
    source = AngleReader(
        {0: 0.72, 180: 0.71},
        words_by_angle={0: 10, 180: 100},
    )
    reader = OrientationReader(
        source,
        orientation_detector=lambda _: {"angle": 180, "confidence": 0.99},
        osd_detector=lambda _: _osd(0, 1.0),
    )

    regions = reader.read(image_path, 1)

    assert source.calls == [180, 0]
    assert len(regions) == 100
    page = reader.coverage_assessment(1)["pages"][0]
    assert page["angle"] == 180
    assert page["view_selection_reason"] == "coverage_recovery"
    assert page["score_margin_metric"] == "character_coverage"


def test_classifier_fallback_keeps_clear_confidence_winner(tmp_path: Path) -> None:
    image_path = _image(tmp_path)
    source = AngleReader(
        {0: 0.9, 180: 0.7},
        words_by_angle={0: 10, 180: 100},
    )
    reader = OrientationReader(
        source,
        orientation_detector=lambda _: {"angle": 180, "confidence": 0.99},
        osd_detector=lambda _: _osd(0, 1.0),
    )

    regions = reader.read(image_path, 1)

    assert len(regions) == 10
    page = reader.coverage_assessment(1)["pages"][0]
    assert page["angle"] == 0
    assert page["view_selection_reason"] == "confidence_evidence"


def test_deferred_restore_runs_structure_stage_on_oriented_page(
    tmp_path: Path,
) -> None:
    image_path = _image(tmp_path)
    source = AngleReader({0: 0.6, 180: 0.95})
    stage = StructureStage()
    reader = OrientationReader(
        source,
        orientation_detector=lambda _: {"angle": 180, "confidence": 0.99},
        osd_detector=lambda _: _osd(0, 1.0),
        defer_restore=True,
    )

    result = process_document(image_path, reader, stages=(stage,))

    assert result.status == "success"
    assert result.pages[0].route == "review"
    assert stage.image_sizes == [(20, 10)]
    assert result.pages[0].regions[0].bounding_box == BoundingBox(12, 3, 18, 7)
    table = result.pages[0].regions[1]
    assert table.bounding_box == BoundingBox(16, 6, 19, 9)
    assert table.structure["cells"][0]["bbox"] == {
        "left": 16,
        "top": 6,
        "right": 19,
        "bottom": 9,
    }
    assert table.structure["cells"][0]["span_bboxes"] == [
        {"left": 16, "top": 6, "right": 19, "bottom": 9}
    ]


def test_all_view_failures_reach_the_caller(tmp_path: Path) -> None:
    image_path = _image(tmp_path)
    source = AngleReader(failures={0, 90, 180, 270})
    reader = OrientationReader(source, osd_executable=None)

    with pytest.raises(ReaderError, match="0=view_failed") as raised:
        reader.read(image_path, 1)

    assert raised.value.code == "orientation_views_failed"
    assessment = reader.coverage_assessment(1)
    assert assessment["status"] == "review_recommended"
    assert assessment["pages"][0]["status"] == "failed"
    assert set(assessment["pages"][0]["view_failures"]) == {
        "0",
        "90",
        "180",
        "270",
    }


def test_process_document_uses_restored_source_geometry(tmp_path: Path) -> None:
    image_path = _image(tmp_path)
    source = AngleReader({0: 0.6, 90: 0.7, 180: 0.95, 270: 0.65})
    reader = OrientationReader(source, osd_detector=lambda _: _osd(0, 1.0))

    result = process_document(image_path, reader)

    assert result.status == "success"
    assert result.failures == []
    assert result.pages[0].width == 20
    assert result.pages[0].height == 10
    assert result.pages[0].text.value == "text at 180 item 0"
    assert result.pages[0].regions[0].bounding_box == BoundingBox(12, 3, 18, 7)


def test_tesseract_osd_parses_clockwise_rotation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = _image(tmp_path)
    output = "\n".join(
        [
            "Orientation in degrees: 270",
            "Rotate: 90",
            "Orientation confidence: 22.5",
            "Script: Latin",
            "Script confidence: 9.2",
        ]
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, output, ""),
    )

    result = detect_tesseract_orientation(image_path, executable="fake")

    assert result == {
        "angle": 270,
        "rotate_clockwise": 90,
        "confidence": 22.5,
        "script": "Latin",
        "script_confidence": 9.2,
    }


def test_doctr_detector_parses_single_prediction(tmp_path: Path) -> None:
    detector = DocTROrientationDetector(
        device="cpu",
        predictor=lambda _: ([2], [-90], [0.97]),
    )

    result = detector(_image(tmp_path))

    assert result["angle"] == 270
    assert result["confidence"] == 0.97
    assert result["model"]["publisher"] == "Mindee"


def _image(tmp_path: Path) -> Path:
    path = tmp_path / "page.png"
    Image.new("RGB", (20, 10), "white").save(path)
    return path


def _osd(angle: int, confidence: float) -> dict[str, object]:
    return {
        "angle": angle,
        "rotate_clockwise": (-angle) % 360,
        "confidence": confidence,
        "script": "Latin",
        "script_confidence": 10.0,
    }
