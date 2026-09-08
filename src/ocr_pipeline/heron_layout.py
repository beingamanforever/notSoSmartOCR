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
            if float(item.get("score", 0.0)) < self.score_threshold:
                continue
            box = _clamped_box(item.get("box"), width, height)
            if box is not None:
                boxes.append(
                    {
                        "label": str(item["label"]).lower(),
                        "score": float(item["score"]),
                        "box": (box.left, box.top, box.right, box.bottom),
                        **{
                            key: item[key]
                            for key in ("query_id", "class_scores")
                            if key in item
                        },
                    }
                )
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
                # Heron's targets assign one class per query. Flattened focal top-k
                # emits several alternative labels for one box and drops other queries.
                probabilities = outputs.logits[0].sigmoid().float().cpu()
                scores, labels = probabilities.max(dim=-1)
                centers, sizes = outputs.pred_boxes[0].float().cpu().split(2, dim=-1)
                boxes = torch.cat((centers - sizes / 2, centers + sizes / 2), dim=-1)
                boxes *= torch.tensor([*image.size, *image.size])
                return [
                    {
                        "label": id2label[int(labels[query])],
                        "score": float(scores[query]),
                        "box": tuple(boxes[query].tolist()),
                        "query_id": int(query),
                        "class_scores": {
                            id2label[label]: float(score)
                            for label, score in enumerate(probabilities[query])
                        },
                    }
                    for query in scores.argsort(descending=True, stable=True)
                    if scores[query] >= score_threshold
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
