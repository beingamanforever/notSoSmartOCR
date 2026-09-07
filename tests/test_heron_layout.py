from __future__ import annotations

from PIL import Image

from ocr_pipeline.contracts import BoundingBox
from ocr_pipeline.heron_layout import HeronTableDetector

IMAGE_SIZE = (200, 300)


def _fake_detector(detections):
    def run(image):
        assert image.size == IMAGE_SIZE
        return detections

    return run


def test_detect_keeps_only_confident_table_boxes() -> None:
    image = Image.new("RGB", IMAGE_SIZE)
    detections = [
        {"label": "table", "score": 0.9, "box": (10, 10, 100, 100)},
        {"label": "picture", "score": 0.95, "box": (5, 5, 50, 50)},
        {"label": "table", "score": 0.3, "box": (0, 0, 20, 20)},
        {"label": "table", "score": 0.8, "box": (150, 250, 500, 500)},
        {"label": "table", "score": 0.7, "box": (30, 30, 30, 60)},
    ]
    detector = HeronTableDetector(detector=_fake_detector(detections))

    boxes = detector.detect(image)

    assert boxes == [
        BoundingBox(10, 10, 100, 100),
        BoundingBox(150, 250, 200, 300),
    ]


def test_score_threshold_is_respected() -> None:
    image = Image.new("RGB", IMAGE_SIZE)
    detections = [{"label": "table", "score": 0.6, "box": (10, 10, 100, 100)}]

    default_detector = HeronTableDetector(detector=_fake_detector(detections))
    assert default_detector.detect(image) == [BoundingBox(10, 10, 100, 100)]

    strict_detector = HeronTableDetector(
        detector=_fake_detector(detections), score_threshold=0.7
    )
    assert strict_detector.detect(image) == []


def test_model_provenance() -> None:
    detector = HeronTableDetector(detector=_fake_detector([]))

    provenance = detector.model_provenance()

    assert provenance == {
        "id": "ds4sd/docling-layout-heron-101",
        "architecture": "RTDetrV2ForObjectDetection",
        "license": "apache-2.0",
        "origin": "IBM Research",
        "score_meaning": "layout detection confidence, not recognition confidence",
    }
