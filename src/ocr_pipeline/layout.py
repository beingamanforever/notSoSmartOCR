"""Swappable document-layout detection."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from PIL import Image


@dataclass(frozen=True)
class LayoutDetection:
    label: str
    confidence: float
    box: tuple[float, float, float, float]
    provider: str


class LayoutDetector(Protocol):
    name: str

    def detect(self, image_path: Path) -> list[LayoutDetection]: ...


class HeronLayoutDetector:
    """Raw RT-DETR predictions from IBM's Docling layout Heron model."""

    name = "docling-layout-heron"

    def __init__(
        self,
        *,
        model_name: str = "docling-project/docling-layout-heron",
        model_revision: str,
        device: str = "cuda",
        threshold: float = 0.6,
    ) -> None:
        if not model_revision.strip():
            raise ValueError("Heron model revision must not be empty")
        if not 0 <= threshold <= 1:
            raise ValueError("Heron threshold must be between 0 and 1")
        self.model_name = model_name
        self.model_revision = model_revision
        self.device = device
        self.threshold = threshold
        self._processor: Any = None
        self._model: Any = None
        self._torch: Any = None

    def detect(self, image_path: Path) -> list[LayoutDetection]:
        self._load()
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        inputs = self._processor(images=[image], return_tensors="pt")
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        with self._torch.inference_mode():
            outputs = self._model(**inputs)
        result = self._processor.post_process_object_detection(
            outputs,
            target_sizes=self._torch.tensor([image.size[::-1]]),
            threshold=self.threshold,
        )[0]
        labels = self._model.config.id2label
        detections = []
        for score, label_id, box in zip(
            result["scores"],
            result["labels"],
            result["boxes"],
            strict=True,
        ):
            values = tuple(float(value) for value in box.tolist())
            confidence = float(score.item())
            if len(values) != 4 or not all(math.isfinite(value) for value in values):
                raise ValueError("Heron returned an invalid bounding box")
            if not math.isfinite(confidence):
                raise ValueError("Heron returned an invalid confidence")
            raw_label = labels.get(int(label_id.item()))
            if raw_label is None:
                raw_label = labels.get(str(int(label_id.item())))
            if not isinstance(raw_label, str) or not raw_label.strip():
                raise ValueError("Heron returned an unknown label")
            detections.append(
                LayoutDetection(
                    label=_normalize_label(raw_label),
                    confidence=confidence,
                    box=values,
                    provider=self.name,
                )
            )
        return detections

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection
        except ImportError as error:
            raise RuntimeError(
                "Heron requires torch and transformers with RT-DETR v2 support"
            ) from error

        self._torch = torch
        self._processor = RTDetrImageProcessor.from_pretrained(
            self.model_name,
            revision=self.model_revision,
        )
        self._model = RTDetrV2ForObjectDetection.from_pretrained(
            self.model_name,
            revision=self.model_revision,
        ).to(self.device)
        self._model.eval()


def _normalize_label(label: str) -> str:
    return "_".join(label.strip().casefold().replace("-", " ").split())
