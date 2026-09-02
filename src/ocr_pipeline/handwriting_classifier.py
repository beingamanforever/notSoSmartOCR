"""Conservative handwriting proposals for existing OCR regions."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from PIL import Image, UnidentifiedImageError

from .contracts import BoundingBox, TextRegion
from .handwriting import ELIGIBLE_KINDS, EXCLUDED_ROLES
from .providers import ReaderError

OVERLAP_KINDS = frozenset({"checkbox", "control", "table", "table_candidate"})
OVERLAP_ROLES = frozenset(
    {"checkbox", "control", "table", "table_candidate", "table_source"}
)


class HandwritingClassifier(Protocol):
    name: str

    @property
    def provenance(self) -> dict[str, Any]: ...

    def score_batch(self, images: Sequence[Image.Image]) -> list[float]: ...


class HandwritingClassifierStage:
    """Mark only dual-view classifier agreements as handwriting candidates."""

    name = "handwriting-classifier"

    def __init__(
        self,
        classifier: HandwritingClassifier,
        *,
        text_provider: str,
        score_threshold: float = 0.9,
        confidence_threshold: float = 0.75,
        max_characters: int = 64,
        max_regions: int = 16,
        context_padding: int = 12,
        overlap_threshold: float = 0.2,
    ) -> None:
        if not text_provider.strip():
            raise ValueError("text_provider must be non-empty")
        if not 0 <= score_threshold <= 1:
            raise ValueError("score_threshold must be from 0 to 1")
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be from 0 to 1")
        if max_characters <= 0 or max_regions <= 0 or context_padding <= 0:
            raise ValueError("classifier limits must be positive")
        if not 0 < overlap_threshold <= 1:
            raise ValueError(
                "overlap_threshold must be greater than zero and at most one"
            )
        self.classifier = classifier
        self.text_provider = text_provider
        self.score_threshold = score_threshold
        self.confidence_threshold = confidence_threshold
        self.max_characters = max_characters
        self.max_regions = max_regions
        self.context_padding = context_padding
        self.overlap_threshold = overlap_threshold

    def apply(
        self,
        image_path: Path,
        page_number: int,
        regions: list[TextRegion],
    ) -> list[TextRegion]:
        selected = [
            index
            for index, region in enumerate(regions)
            if self._eligible(region, index, regions)
        ]
        selected.sort(
            key=lambda index: (
                regions[index].confidence,
                regions[index].reading_order,
                index,
            )
        )
        selected = selected[: self.max_regions]
        if not selected:
            return regions

        crops: list[Image.Image] = []
        boxes: list[tuple[BoundingBox, BoundingBox]] = []
        try:
            with Image.open(image_path) as source:
                page = source.convert("RGB")
            try:
                for index in selected:
                    tight, context = _crop_boxes(
                        regions[index].bounding_box,
                        page.size,
                        self.context_padding,
                    )
                    boxes.append((tight, context))
                    crops.extend(
                        (
                            page.crop(_box_tuple(tight)),
                            page.crop(_box_tuple(context)),
                        )
                    )
            finally:
                page.close()
        except (OSError, UnidentifiedImageError, ValueError) as error:
            for crop in crops:
                crop.close()
            raise ReaderError(
                "handwriting_classifier_crop_failed",
                "Handwriting classifier crops could not be prepared",
            ) from error

        try:
            scores = self.classifier.score_batch(crops)
        finally:
            for crop in crops:
                crop.close()
        if len(scores) != len(crops) or any(
            not _valid_score(score) for score in scores
        ):
            raise ReaderError(
                "invalid_handwriting_classifier_output",
                "Handwriting classifier returned invalid scores",
            )

        for position, index in enumerate(selected):
            tight_score, context_score = scores[position * 2 : position * 2 + 2]
            tight, context = boxes[position]
            evidence = {
                "method": "dual_crop_classifier_agreement",
                "page_number": page_number,
                "threshold": self.score_threshold,
                "scores": {
                    "tight": round(tight_score, 6),
                    "context": round(context_score, 6),
                },
                "crops": {
                    "tight": {"bounding_box": list(_box_tuple(tight))},
                    "context": {"bounding_box": list(_box_tuple(context))},
                },
                "model": self.classifier.provenance,
            }
            accepted = (
                tight_score >= self.score_threshold
                and context_score >= self.score_threshold
            )
            disagreed = (tight_score >= self.score_threshold) != (
                context_score >= self.score_threshold
            )
            if accepted:
                structure = dict(regions[index].structure or {})
                structure["handwriting_candidate"] = True
                structure["handwriting_classifier"] = {
                    **evidence,
                    "decision": "candidate",
                }
                regions[index].structure = structure
            elif disagreed:
                structure = dict(regions[index].structure or {})
                structure["handwriting_classifier"] = {
                    **evidence,
                    "decision": "view_disagreement",
                    "review": {"required": False, "reason": "view_disagreement"},
                }
                regions[index].structure = structure
        return regions

    def _eligible(
        self,
        region: TextRegion,
        index: int,
        regions: list[TextRegion],
    ) -> bool:
        text = region.text.strip()
        structure = region.structure or {}
        box = region.bounding_box
        return (
            region.resolution == "resolved"
            and region.kind in ELIGIBLE_KINDS
            and region.kind not in EXCLUDED_ROLES
            and structure.get("role") not in EXCLUDED_ROLES
            and region.provider == self.text_provider
            and bool(text)
            and "\n" not in text
            and len(text) <= self.max_characters
            and region.confidence is not None
            and region.confidence < self.confidence_threshold
            and _valid_box(box)
            and not _has_material_overlap(
                box,
                index,
                regions,
                self.overlap_threshold,
            )
        )


class MobileNetHandwritingClassifier:
    """Lazy local MobileNetV3-small binary classifier."""

    name = "mobilenet-v3-small-handwriting"

    def __init__(
        self,
        checkpoint: Path,
        *,
        device: str = "cpu",
        batch_size: int = 32,
        image_size: int = 224,
    ) -> None:
        if batch_size <= 0 or image_size <= 0:
            raise ValueError("batch_size and image_size must be positive")
        self.checkpoint = checkpoint
        self.device = device
        self.batch_size = batch_size
        self.image_size = image_size
        self._runtime: tuple[Any, Any, Any] | None = None

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "id": self.name,
            "architecture": "torchvision/mobilenet_v3_small",
            "checkpoint": self.checkpoint.name,
        }

    def score_batch(self, images: Sequence[Image.Image]) -> list[float]:
        if not images:
            return []
        torch, model, transform = self._load()
        scores: list[float] = []
        with torch.inference_mode():
            for start in range(0, len(images), self.batch_size):
                batch = torch.stack(
                    [
                        transform(image.convert("RGB"))
                        for image in images[start : start + self.batch_size]
                    ]
                ).to(self.device)
                logits = model(batch).flatten()
                scores.extend(torch.sigmoid(logits).cpu().tolist())
        return scores

    def _load(self) -> tuple[Any, Any, Any]:
        if self._runtime is not None:
            return self._runtime
        if not self.checkpoint.is_file():
            raise ReaderError(
                "handwriting_classifier_checkpoint_missing",
                f"Handwriting classifier checkpoint not found: {self.checkpoint}",
            )
        try:
            import torch
            from torchvision import models, transforms
        except ImportError as error:
            raise ReaderError(
                "handwriting_classifier_dependency_unavailable",
                "Handwriting classification requires torch and torchvision",
            ) from error

        model = models.mobilenet_v3_small(weights=None)
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, 1)
        payload = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        state = payload
        if isinstance(payload, dict) and "model_state_dict" in payload:
            state = payload["model_state_dict"]
        if not isinstance(state, dict):
            raise ReaderError(
                "invalid_handwriting_classifier_checkpoint",
                "Handwriting classifier checkpoint has no model state",
            )
        try:
            model.load_state_dict(state)
        except (RuntimeError, TypeError, ValueError) as error:
            raise ReaderError(
                "invalid_handwriting_classifier_checkpoint",
                "Handwriting classifier checkpoint is incompatible",
            ) from error
        model.to(self.device).eval()
        transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )
        self._runtime = (torch, model, transform)
        return self._runtime


def _has_material_overlap(
    box: BoundingBox,
    source_index: int,
    regions: list[TextRegion],
    threshold: float,
) -> bool:
    area = _box_area(box)
    for index, other in enumerate(regions):
        if index == source_index:
            continue
        structure = other.structure or {}
        if (
            other.kind not in OVERLAP_KINDS
            and structure.get("role") not in OVERLAP_ROLES
        ):
            continue
        other_box = other.bounding_box
        if not _valid_box(other_box):
            continue
        if _intersection_area(box, other_box) / area >= threshold:
            return True
    return False


def _crop_boxes(
    box: BoundingBox,
    page_size: tuple[int, int],
    padding: int,
) -> tuple[BoundingBox, BoundingBox]:
    width, height = page_size
    tight = BoundingBox(
        max(0, box.left),
        max(0, box.top),
        min(width, box.right),
        min(height, box.bottom),
    )
    if not _valid_box(tight):
        raise ValueError("classifier region lies outside the page")
    margin = max(padding, math.ceil((tight.bottom - tight.top) / 2))
    context = BoundingBox(
        max(0, tight.left - margin),
        max(0, tight.top - margin),
        min(width, tight.right + margin),
        min(height, tight.bottom + margin),
    )
    return tight, context


def _valid_score(score: object) -> bool:
    return (
        not isinstance(score, bool)
        and isinstance(score, (float, int))
        and math.isfinite(score)
        and 0 <= score <= 1
    )


def _valid_box(box: object) -> bool:
    return (
        isinstance(box, BoundingBox) and box.right > box.left and box.bottom > box.top
    )


def _intersection_area(left: BoundingBox, right: BoundingBox) -> int:
    return max(0, min(left.right, right.right) - max(left.left, right.left)) * max(
        0,
        min(left.bottom, right.bottom) - max(left.top, right.top),
    )


def _box_area(box: BoundingBox) -> int:
    return (box.right - box.left) * (box.bottom - box.top)


def _box_tuple(box: BoundingBox) -> tuple[int, int, int, int]:
    return box.left, box.top, box.right, box.bottom
