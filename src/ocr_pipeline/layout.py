"""Swappable document-layout detection."""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from PIL import Image

NEMOTRON_PAGE_ELEMENTS_MODEL = "nvidia/nemotron-page-elements-v3"


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
        self.model_revision_enforced = True
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


class NemotronPageElementsV3Detector:
    """NVIDIA Nemotron Page Elements v3 benchmark adapter."""

    name = "nemotron-page-elements-v3"

    def __init__(
        self,
        *,
        model_name: str = NEMOTRON_PAGE_ELEMENTS_MODEL,
        model_revision: str | None = None,
        device: str = "cuda",
        model: Any = None,
        postprocessor: Any = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError("Nemotron Page Elements model name must not be empty")
        if model is None and model_name != NEMOTRON_PAGE_ELEMENTS_MODEL:
            raise ValueError(
                "Nemotron Page Elements official loader supports only its official model"
            )
        if model_revision is not None:
            raise ValueError(
                "Nemotron Page Elements official loader cannot enforce model revisions"
            )
        if not device.strip():
            raise ValueError("Nemotron Page Elements device must not be empty")
        self.model_name = model_name
        self.model_revision = None
        self.model_revision_enforced = False
        self.device = device
        self.threshold = getattr(model, "thresholds_per_class", None)
        self._model = model
        self._postprocessor = postprocessor
        self._torch: Any = None

    def detect(self, image_path: Path) -> list[LayoutDetection]:
        self._load()
        import numpy as np

        with Image.open(image_path) as source:
            image = np.array(source.convert("RGB"))

        inference_context = (
            self._torch.inference_mode() if self._torch is not None else nullcontext()
        )
        with inference_context:
            inputs = self._model.preprocess(image)
            predictions = self._model(inputs, image.shape)[0]
        boxes, labels, scores = self._postprocessor(
            predictions,
            self._model.thresholds_per_class,
            self._model.labels,
        )
        try:
            result_count = len(boxes)
            lengths_match = result_count == len(labels) == len(scores)
        except TypeError as error:
            raise ValueError(
                "Nemotron Page Elements returned invalid results"
            ) from error
        if not lengths_match:
            raise ValueError("Nemotron Page Elements returned mismatched results")

        height, width = image.shape[:2]
        detections = []
        for box, label, score in zip(boxes, labels, scores, strict=True):
            detections.append(
                LayoutDetection(
                    label=_nemotron_label(label, self._model.labels),
                    confidence=_nemotron_confidence(score),
                    box=_nemotron_pixel_box(box, width, height),
                    provider=self.name,
                )
            )
        return detections

    def _load(self) -> None:
        if self._model is not None and self._postprocessor is not None:
            self.threshold = getattr(self._model, "thresholds_per_class", None)
            return
        try:
            import torch
            from nemotron_page_elements_v3.model import define_model
            from nemotron_page_elements_v3.utils import (
                postprocess_preds_page_element,
            )
        except (ImportError, OSError) as error:
            raise RuntimeError(
                "Nemotron Page Elements v3 requires its official Python package"
            ) from error

        self._torch = torch
        if self._model is None:
            self._model = define_model("page_element_v3").to(self.device)
            self._model.device = self.device
            self._model.eval()
        if self._postprocessor is None:
            self._postprocessor = postprocess_preds_page_element
        self.threshold = getattr(self._model, "thresholds_per_class", None)


def _nemotron_pixel_box(
    box: Any,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    try:
        values = tuple(float(value) for value in box)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Nemotron Page Elements returned an invalid bounding box"
        ) from error
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        raise ValueError("Nemotron Page Elements returned an invalid bounding box")
    left, top, right, bottom = values
    if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
        raise ValueError(
            "Nemotron Page Elements returned an out-of-bounds bounding box"
        )
    return left * width, top * height, right * width, bottom * height


def _nemotron_label(label: Any, labels: Any) -> str:
    if isinstance(label, str):
        raw_label = label
    else:
        try:
            index = int(label)
            if index < 0 or float(label) != index:
                raise ValueError
            raw_label = labels[index]
        except (IndexError, TypeError, ValueError) as error:
            raise ValueError(
                "Nemotron Page Elements returned an unknown label"
            ) from error
    if not isinstance(raw_label, str) or not raw_label.strip():
        raise ValueError("Nemotron Page Elements returned an unknown label")
    return _normalize_label(raw_label)


def _nemotron_confidence(score: Any) -> float:
    try:
        confidence = float(score)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Nemotron Page Elements returned an invalid confidence"
        ) from error
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("Nemotron Page Elements returned an invalid confidence")
    return confidence


def _normalize_label(label: str) -> str:
    return "_".join(label.strip().casefold().replace("-", " ").split())
