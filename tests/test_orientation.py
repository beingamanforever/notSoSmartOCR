from __future__ import annotations

import subprocess
import time
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
        box_by_angle: dict[int, BoundingBox] | None = None,
        failures: set[int] | None = None,
        region_kind: str = "word",
    ) -> None:
        self.confidence_by_angle = confidence_by_angle or {}
        self.words_by_angle = words_by_angle or {}
        self.box_by_angle = box_by_angle or {}
        self.failures = failures or set()
        self.region_kind = region_kind
        self.calls: list[int] = []

    def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
        angle = int(image_path.stem.rsplit("-", 1)[1])
        self.calls.append(angle)
        if angle in self.failures:
            raise ReaderError("view_failed", f"angle {angle} failed")
        return [
            TextRegion(
                id=f"p{page_number}-source-{index + 1}",
                kind=self.region_kind,
                text=f"text at {angle} item {index}",
                confidence=self.confidence_by_angle.get(angle, 0.7),
                bounding_box=self.box_by_angle.get(
                    angle,
                    BoundingBox(2, 3, 8, 7),
                ),
                reading_order=index + 1,
                provider=self.name,
                text_provenance=(
                    {"merge_level": "word"}
                    if self.region_kind == "text"
                    else {"method": "controlled"}
                ),
            )
            for index in range(self.words_by_angle.get(angle, 1))
        ]


class ReviewAngleReader(AngleReader):
    def page_needs_review(self, page_number: int) -> bool:
        return True


class SlowAngleReader(AngleReader):
    def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
        time.sleep(0.01)
        return super().read(image_path, page_number)


class ObservableAngleReader(AngleReader):
    """Model the nested reader state that each orientation call overwrites."""

    def __init__(self) -> None:
        super().__init__({0: 0.95, 90: 0.8}, words_by_angle={0: 12, 90: 12})
        self.last_angle: int | None = None

    def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
        self.last_angle = int(image_path.stem.rsplit("-", 1)[1])
        return super().read(image_path, page_number)

    def coverage_assessment(self, page_count: int) -> dict[str, object]:
        fallback_runs = int(self.last_angle == 90)
        return {
            "status": "review_recommended" if fallback_runs else "not_assessed",
            "message": "controlled nested execution",
            "pages": [
                {
                    "page_number": page_number,
                    "ran": True,
                    "status": "uncertain" if fallback_runs else "not_routed",
                    "fallback_reader": "tesseract",
                    "fallback_reader_runs": fallback_runs,
                    "fallback_candidates": 3 * fallback_runs,
                    "replaced_bands": fallback_runs,
                    "nested_reader": {
                        "ran": True,
                        "tile_views_run": 3,
                        "tiled_candidates": 7 + int(self.last_angle or 0),
                    },
                }
                for page_number in range(1, page_count + 1)
            ],
        }


class ResidualAngleReader:
    name = "residual-angle-reader"

    def __init__(
        self,
        confidence: float = 0.96,
        *,
        selected_angle: int = 0,
        residual_angle: int = 90,
        residual_box: BoundingBox = BoundingBox(20, 90, 150, 97),
    ) -> None:
        self.confidence = confidence
        self.selected_angle = selected_angle
        self.residual_angle = residual_angle
        self.residual_box = residual_box
        self.calls: list[int] = []

    def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
        angle = int(image_path.stem.rsplit("-", 1)[1])
        self.calls.append(angle)
        if angle == self.selected_angle:
            return [
                _region(
                    f"p{page_number}-body-{index + 1}",
                    f"canonical body line {index + 1}",
                    BoundingBox(25, 12 + index * 10, 85, 18 + index * 10),
                    index + 1,
                    0.95,
                    self.name,
                )
                for index in range(12)
            ]
        if angle == self.residual_angle:
            return [
                _region(
                    f"p{page_number}-side",
                    "vertical publication identifier",
                    self.residual_box,
                    1,
                    self.confidence,
                    self.name,
                ),
                _region(
                    f"p{page_number}-wrong-body",
                    "wrong orientation body text",
                    BoundingBox(40, 30, 50, 70),
                    2,
                    0.99,
                    self.name,
                ),
            ]
        return [
            _region(
                f"p{page_number}-noise-{angle}",
                "noise",
                BoundingBox(5, 5, 9, 12),
                1,
                0.4,
                self.name,
            )
        ]


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

    assert source.calls == ([0] if angle == 0 else [angle, 0])
    assert regions[0].id == "p1-source-1"
    assert regions[0].text == f"text at {angle} item 0"
    assert regions[0].provider == source.name
    assert regions[0].bounding_box == expected
    assert regions[0].text_provenance == {
        "method": "controlled",
        "orientation": {
            "angle": angle,
            "selector": (
                "tesseract_osd" if angle == 0 else "tesseract_osd_evidence_check"
            ),
            "score_margin": 1.0 if angle == 0 else 0.0,
            "original_size": [20, 10],
            "rotated_size": [20, 10] if angle in {0, 180} else [10, 20],
            "osd": _osd(angle, 20.0),
            "osd_failure": None,
            "view_failures": {},
        },
    }


def test_confident_wrong_osd_must_beat_the_upright_ocr_evidence(
    tmp_path: Path,
) -> None:
    image_path = _image(tmp_path)
    source = AngleReader(
        {0: 0.94, 270: 0.72},
        words_by_angle={0: 12, 270: 12},
    )
    reader = OrientationReader(source, osd_detector=lambda _: _osd(270, 25.0))

    regions = reader.read(image_path, 1)

    assert source.calls == [270, 0]
    assert regions[0].text == "text at 0 item 0"
    page = reader.coverage_assessment(1)["pages"][0]
    assert page["angle"] == 0
    assert page["selector"] == "tesseract_osd_evidence_check"


def test_orientation_preserves_nested_execution_for_every_scored_view(
    tmp_path: Path,
) -> None:
    source = ObservableAngleReader()
    reader = OrientationReader(
        source,
        orientation_detector=lambda _: {"angle": 90, "confidence": 0.99},
        osd_detector=lambda _: _osd(0, 20.0),
    )

    regions = reader.read(_image(tmp_path), 1)

    page = reader.coverage_assessment(1)["pages"][0]
    assert page["angle"] == 0
    assert page["review_reasons"] == ["orientation_evidence_fallback"]
    assert page["view_reader_execution"] == {
        "90": {
            "reader": "angle-reader",
            "ran": True,
            "status": "uncertain",
            "fallback_reader": "tesseract",
            "fallback_reader_runs": 1,
            "fallback_candidates": 3,
            "replaced_bands": 1,
            "nested_reader": {
                "ran": True,
                "tile_views_run": 3,
                "tiled_candidates": 97,
            },
        },
        "0": {
            "reader": "angle-reader",
            "ran": True,
            "status": "not_routed",
            "fallback_reader": "tesseract",
            "fallback_reader_runs": 0,
            "fallback_candidates": 0,
            "replaced_bands": 0,
            "nested_reader": {
                "ran": True,
                "tile_views_run": 3,
                "tiled_candidates": 7,
            },
        },
    }
    assert (
        regions[0].text_provenance["orientation"]["reader_execution"]
        == page["view_reader_execution"]["0"]
    )


def test_horizontal_word_geometry_breaks_a_narrow_confidence_tie(
    tmp_path: Path,
) -> None:
    image_path = _image(tmp_path)
    source = AngleReader(
        {0: 0.8809, 270: 0.8857},
        words_by_angle={0: 12, 270: 12},
        box_by_angle={
            0: BoundingBox(2, 3, 8, 7),
            270: BoundingBox(2, 1, 5, 8),
        },
        region_kind="text",
    )
    reader = OrientationReader(source, osd_detector=lambda _: _osd(270, 25.0))

    regions = reader.read(image_path, 1)

    assert regions[0].text == "text at 0 item 0"
    page = reader.coverage_assessment(1)["pages"][0]
    assert page["angle"] == 0
    assert page["view_selection_reason"] == "word_box_geometry"
    assert page["score_margin_metric"] == "horizontal_word_fraction"
    assert page["view_scores"]["0"]["horizontal_word_fraction"] == 1.0
    assert page["view_scores"]["270"]["horizontal_word_fraction"] == 0.0


def test_word_geometry_cannot_overturn_materially_stronger_ocr_evidence(
    tmp_path: Path,
) -> None:
    image_path = _image(tmp_path)
    source = AngleReader(
        {0: 0.88, 270: 0.92},
        words_by_angle={0: 12, 270: 12},
        box_by_angle={
            0: BoundingBox(2, 3, 8, 7),
            270: BoundingBox(2, 1, 5, 8),
        },
    )
    reader = OrientationReader(source, osd_detector=lambda _: _osd(270, 25.0))

    regions = reader.read(image_path, 1)

    assert regions[0].text == "text at 270 item 0"
    page = reader.coverage_assessment(1)["pages"][0]
    assert page["angle"] == 270
    assert page["view_selection_reason"] == "confidence_evidence"


def test_geometry_selects_the_rotated_view_that_makes_words_horizontal(
    tmp_path: Path,
) -> None:
    image_path = _image(tmp_path)
    source = AngleReader(
        {0: 0.95, 90: 0.94},
        words_by_angle={0: 12, 90: 12},
        box_by_angle={
            0: BoundingBox(2, 1, 5, 8),
            90: BoundingBox(2, 3, 8, 7),
        },
    )
    reader = OrientationReader(
        source,
        orientation_detector=lambda _: {"angle": 90, "confidence": 0.99},
        osd_detector=lambda _: _osd(0, 20.0),
    )

    regions = reader.read(image_path, 1)

    assert source.calls == [90, 0]
    assert regions[0].text == "text at 90 item 0"
    page = reader.coverage_assessment(1)["pages"][0]
    assert page["angle"] == 90
    assert page["view_selection_reason"] == "word_box_geometry"


def test_recovers_high_confidence_vertical_margin_from_scored_view(
    tmp_path: Path,
) -> None:
    source = ResidualAngleReader()
    reader = OrientationReader(source, osd_detector=lambda _: _osd(0, 1.0))

    regions = reader.read(_large_image(tmp_path), 1)

    assert source.calls == [0, 90, 180, 270]
    assert [region.id for region in regions[:12]] == [
        f"p1-body-{index}" for index in range(1, 13)
    ]
    assert [region.reading_order for region in regions[:12]] == list(range(1, 13))
    assert len(regions) == 13
    residual = regions[-1]
    assert residual.id == "p1-orientation-residual-1"
    assert residual.text == "vertical publication identifier"
    assert residual.bounding_box == BoundingBox(3, 20, 10, 150)
    assert residual.reading_order == 13
    assert residual.provider == source.name
    assert residual.text_provenance["orientation_residual"] == {
        "method": "orthogonal_margin_residual",
        "source_view_angle": 90,
        "selected_view_angle": 0,
        "original_provider": source.name,
    }
    page = reader.coverage_assessment(1)["pages"][0]
    assert page["angle"] == 0
    assert page["vertical_residual_recovery"] == {
        "method": "orthogonal_margin_residual",
        "recovered_regions": 1,
        "source_view_angles": [90],
    }


def test_does_not_recover_central_wrong_orientation_body_text(
    tmp_path: Path,
) -> None:
    source = ResidualAngleReader()
    reader = OrientationReader(source, osd_detector=lambda _: _osd(0, 1.0))

    regions = reader.read(_large_image(tmp_path), 1)

    assert all(region.text != "wrong orientation body text" for region in regions)


def test_does_not_turn_rotated_page_header_into_vertical_residual(
    tmp_path: Path,
) -> None:
    source = ResidualAngleReader(
        selected_angle=270,
        residual_angle=0,
        residual_box=BoundingBox(20, 3, 150, 10),
    )
    reader = OrientationReader(source, osd_detector=lambda _: _osd(0, 1.0))

    regions = reader.read(_wide_image(tmp_path), 1)

    assert len(regions) == 12
    assert all(region.text != "vertical publication identifier" for region in regions)
    page = reader.coverage_assessment(1)["pages"][0]
    assert page["angle"] == 270
    assert page["vertical_residual_recovery"]["recovered_regions"] == 0


def test_does_not_recover_low_confidence_vertical_margin_text(
    tmp_path: Path,
) -> None:
    source = ResidualAngleReader(confidence=0.84)
    reader = OrientationReader(source, osd_detector=lambda _: _osd(0, 1.0))

    regions = reader.read(_large_image(tmp_path), 1)

    assert len(regions) == 12
    page = reader.coverage_assessment(1)["pages"][0]
    assert page["vertical_residual_recovery"]["recovered_regions"] == 0


def test_deferred_vertical_residual_uses_selected_view_then_restores(
    tmp_path: Path,
) -> None:
    source = ResidualAngleReader(selected_angle=180)
    reader = OrientationReader(
        source,
        osd_detector=lambda _: _osd(0, 1.0),
        defer_restore=True,
    )

    selected_view = reader.read(_large_image(tmp_path), 1)

    residual = selected_view[-1]
    assert residual.bounding_box == BoundingBox(90, 50, 97, 180)
    restored = reader.restore_regions(selected_view, 1)
    assert restored[-1].bounding_box == BoundingBox(3, 20, 10, 150)
    assert (
        restored[-1].text_provenance["orientation_residual"]["source_view_angle"] == 90
    )


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


def test_uncertain_zero_accepts_strong_upright_ocr_evidence(tmp_path: Path) -> None:
    image_path = _image(tmp_path)
    adaptive_source = AngleReader({0: 0.9}, words_by_angle={0: 12})
    direct_source = AngleReader({0: 0.9}, words_by_angle={0: 12})
    adaptive = OrientationReader(
        adaptive_source,
        orientation_detector=lambda _: {"angle": 0, "confidence": 0.55},
    )
    direct = OrientationReader(
        direct_source,
        orientation_detector=lambda _: {"angle": 0, "confidence": 0.98},
    )

    adaptive_result = process_document(image_path, adaptive)
    direct_result = process_document(image_path, direct)

    assert adaptive_source.calls == [0]
    assert adaptive_result.pages[0].text.value == direct_result.pages[0].text.value
    assert [
        (
            region.kind,
            region.text,
            region.confidence,
            region.bounding_box,
            region.reading_order,
            region.provider,
        )
        for region in adaptive_result.pages[0].regions
    ] == [
        (
            region.kind,
            region.text,
            region.confidence,
            region.bounding_box,
            region.reading_order,
            region.provider,
        )
        for region in direct_result.pages[0].regions
    ]
    page = adaptive.coverage_assessment(1)["pages"][0]
    assert page["selector"] == "orientation_evidence_fallback"
    assert page["status"] == "review_recommended"
    assert page["adaptive_orientation"]["status"] == "accepted_upright"


def test_adaptive_orientation_reduces_caller_visible_reader_time(
    tmp_path: Path,
) -> None:
    image_path = _image(tmp_path)
    adaptive_source = SlowAngleReader({0: 0.9}, words_by_angle={0: 12})
    fallback_source = SlowAngleReader(
        {0: 0.79, 90: 0.7, 180: 0.7, 270: 0.7},
        words_by_angle={0: 12, 90: 12, 180: 12, 270: 12},
    )
    adaptive = OrientationReader(
        adaptive_source,
        orientation_detector=lambda _: {"angle": 0, "confidence": 0.55},
    )
    fallback = OrientationReader(
        fallback_source,
        orientation_detector=lambda _: {"angle": 0, "confidence": 0.55},
    )
    adaptive_timings: dict[str, float] = {}
    fallback_timings: dict[str, float] = {}

    adaptive_result = process_document(
        image_path,
        adaptive,
        timings=adaptive_timings,
    )
    fallback_result = process_document(
        image_path,
        fallback,
        timings=fallback_timings,
    )

    assert adaptive_source.calls == [0]
    assert fallback_source.calls == [0, 90, 180, 270]
    assert adaptive_result.pages[0].text.value == fallback_result.pages[0].text.value
    assert adaptive_timings["reader"] < fallback_timings["reader"] / 2
    page = fallback.coverage_assessment(1)["pages"][0]
    assert page["adaptive_orientation"]["status"] == "expanded"


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


def _large_image(tmp_path: Path) -> Path:
    path = tmp_path / "large-page.png"
    Image.new("RGB", (100, 200), "white").save(path)
    return path


def _wide_image(tmp_path: Path) -> Path:
    path = tmp_path / "wide-page.png"
    Image.new("RGB", (200, 100), "white").save(path)
    return path


def _region(
    identifier: str,
    text: str,
    box: BoundingBox,
    order: int,
    confidence: float,
    provider: str,
) -> TextRegion:
    return TextRegion(
        id=identifier,
        kind="text",
        text=text,
        confidence=confidence,
        bounding_box=box,
        reading_order=order,
        provider=provider,
        text_provenance={"merge_level": "word"},
    )


def _osd(angle: int, confidence: float) -> dict[str, object]:
    return {
        "angle": angle,
        "rotate_clockwise": (-angle) % 360,
        "confidence": confidence,
        "script": "Latin",
        "script_confidence": 10.0,
    }
