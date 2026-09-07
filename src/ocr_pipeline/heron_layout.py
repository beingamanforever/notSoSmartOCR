"""Detect table crops with IBM's Heron-101 general layout detector.

Table Transformer is trained on a narrower distribution and misses full-page
financial tables (domain shift). Heron-101 (RT-DETRv2, Apache-2.0) is a general
17-class layout detector, so it fires on those pages instead. Its output feeds
`tables.py`'s existing layout-proposal channel as plain crops - the detector's
score is a layout confidence, not a recognition confidence.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from typing import Any

from PIL import Image

from .contracts import BoundingBox
from .providers import ReaderError

DEFAULT_MODEL_ID = "ds4sd/docling-layout-heron-101"
MODEL_ARCHITECTURE = "RTDetrV2ForObjectDetection"
MODEL_LICENSE = "apache-2.0"
MODEL_ORIGIN = "IBM Research"
# From id2label in https://huggingface.co/ds4sd/docling-layout-heron-101/raw/main/config.json
# (id 8 is "table"). id 11 "document_index" is Docling's table-of-contents label, not a
# data table, so it is excluded.
TABLE_LABELS = frozenset({"table"})

Detection = dict[str, Any]
Detector = Callable[[Image.Image], list[Detection]]


class HeronTableDetector:
    """Table-crop proposals from the Heron-101 general layout detector."""

    name = "heron-101"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        device: str = "cuda",
        score_threshold: float = 0.5,
        detector: Detector | None = None,
    ) -> None:
        if not 0 <= score_threshold <= 1:
            raise ValueError("score_threshold must be from 0 to 1")
        self.model_id = model_id
        self.device = device
        self.score_threshold = score_threshold
        self._detector = detector
        self._lock = threading.Lock()

    def detect(self, image: Image.Image) -> list[BoundingBox]:
        width, height = image.size
        with self._lock:
            detector = self._get_detector()
            try:
                detections = detector(image)
            except Exception as error:
                raise ReaderError("heron_detect_failed", str(error)) from error

        boxes = []
        for item in detections:
            if item.get("label") not in TABLE_LABELS:
                continue
            if float(item.get("score", 0.0)) < self.score_threshold:
                continue
            box = _clamped_box(item.get("box"), width, height)
            if box is not None:
                boxes.append(box)
        return boxes

    def model_provenance(self) -> dict[str, Any]:
        return {
            "id": self.model_id,
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

            processor = AutoImageProcessor.from_pretrained(self.model_id)
            model = AutoModelForObjectDetection.from_pretrained(self.model_id)
            model.to(self.device)
            model.eval()
            id2label = model.config.id2label
            device = self.device
            score_threshold = self.score_threshold

            def run(image: Image.Image) -> list[Detection]:
                inputs = processor(images=[image], return_tensors="pt").to(device)
                with torch.inference_mode():
                    outputs = model(**inputs)
                result = processor.post_process_object_detection(
                    outputs,
                    target_sizes=torch.tensor([image.size[::-1]]),
                    threshold=score_threshold,
                )[0]
                return [
                    {
                        "label": id2label[int(label_id)],
                        "score": float(score),
                        "box": tuple(float(value) for value in box.tolist()),
                    }
                    for score, label_id, box in zip(
                        result["scores"], result["labels"], result["boxes"]
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
    left, top, right, bottom = (int(round(part)) for part in value)
    left = max(0, min(width, left))
    top = max(0, min(height, top))
    right = max(0, min(width, right))
    bottom = max(0, min(height, bottom))
    if right <= left or bottom <= top:
        return None
    return BoundingBox(left, top, right, bottom)
