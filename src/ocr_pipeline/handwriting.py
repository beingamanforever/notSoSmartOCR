"""Conservative crop-level handwriting specialization."""

from __future__ import annotations

import copy
import math
import unicodedata
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from PIL import Image, UnidentifiedImageError

from .contracts import BoundingBox, TextAlternative, TextRegion
from .providers import ReaderError

ABSTENTION_OUTPUTS = frozenset({"<no_handwriting>", "<unreadable>"})
ELIGIBLE_KINDS = frozenset({"handwriting", "text", "word"})
EXCLUDED_ROLES = frozenset(
    {
        "control",
        "coverage_risk",
        "footer",
        "header",
        "heading",
        "table",
        "table_candidate",
        "table_source",
        "tiny_text_candidate",
        "title",
    }
)


class HandwritingCropReader(Protocol):
    name: str
    max_batch_items: int

    @property
    def provenance(self) -> dict[str, Any]: ...

    def transcribe_batch(self, images: Sequence[Image.Image]) -> list[str]: ...


class HandwritingStage:
    """Reread explicitly marked, bounded handwriting regions using two crop views."""

    name = "handwriting"

    def __init__(
        self,
        reader: HandwritingCropReader,
        *,
        text_provider: str | None = None,
        confidence_threshold: float = 0.75,
        max_regions: int = 8,
        max_incumbent_characters: int = 64,
        max_candidate_characters: int = 96,
        context_padding: int = 12,
    ) -> None:
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be from 0 to 1")
        if max_regions <= 0:
            raise ValueError("max_regions must be positive")
        if max_incumbent_characters <= 0 or max_candidate_characters <= 0:
            raise ValueError("handwriting text limits must be positive")
        if context_padding <= 0:
            raise ValueError("context_padding must be positive")
        if reader.max_batch_items < max_regions * 2:
            raise ValueError("handwriting reader batch limit is too small")
        self.reader = reader
        self.text_provider = text_provider
        self.confidence_threshold = confidence_threshold
        self.max_regions = max_regions
        self.max_incumbent_characters = max_incumbent_characters
        self.max_candidate_characters = max_candidate_characters
        self.context_padding = context_padding

    def apply(
        self,
        image_path: Path,
        page_number: int,
        regions: list[TextRegion],
    ) -> list[TextRegion]:
        selected = self._selected_indices(regions)
        if not selected:
            return regions

        crops: list[Image.Image] = []
        crop_boxes: list[tuple[BoundingBox, BoundingBox]] = []
        try:
            with Image.open(image_path) as opened:
                page = opened.convert("RGB")
            try:
                for index in selected:
                    tight, context = _crop_boxes(
                        regions[index].bounding_box,
                        page.size,
                        self.context_padding,
                    )
                    crop_boxes.append((tight, context))
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
                "handwriting_crop_failed",
                "Handwriting crops could not be prepared",
            ) from error

        try:
            candidates = self.reader.transcribe_batch(crops)
        finally:
            for crop in crops:
                crop.close()
        if len(candidates) != len(crops) or any(
            not isinstance(candidate, str) for candidate in candidates
        ):
            raise ReaderError(
                "invalid_handwriting_output",
                "Handwriting reader returned an invalid crop batch",
            )

        model_provenance = self.reader.provenance
        for position, index in enumerate(selected):
            tight_text = candidates[position * 2].strip()
            context_text = candidates[position * 2 + 1].strip()
            tight_box, context_box = crop_boxes[position]
            provenance = _crop_provenance(
                page_number,
                tight_box,
                context_box,
                model_provenance,
            )
            self._apply_candidate(
                regions[index],
                tight_text,
                context_text,
                provenance,
            )
        return regions

    def review_region(
        self,
        image_path: Path,
        page_number: int,
        region: TextRegion,
    ) -> TextRegion:
        """Reread one user-selected region without enabling automatic routing."""
        candidate = copy.deepcopy(region)
        structure = dict(candidate.structure or {})
        structure["handwriting_candidate"] = True
        structure["handwriting_candidate_source"] = "manual"
        candidate.structure = structure
        if not self._is_eligible(candidate):
            raise ReaderError(
                "handwriting_region_ineligible",
                "The selected region is not a bounded text crop",
            )
        return self.apply(image_path, page_number, [candidate])[0]

    def _selected_indices(self, regions: list[TextRegion]) -> list[int]:
        candidates = [
            index for index, region in enumerate(regions) if self._is_eligible(region)
        ]
        candidates.sort(
            key=lambda index: (
                regions[index].confidence,
                regions[index].reading_order,
                index,
            )
        )
        return candidates[: self.max_regions]

    def _is_eligible(self, region: TextRegion) -> bool:
        text = region.text.strip()
        structure = region.structure or {}
        manual = structure.get("handwriting_candidate_source") == "manual"
        handwriting_signal = (
            region.kind == "handwriting"
            or structure.get("handwriting_candidate") is True
        )
        return (
            region.resolution == "resolved"
            and region.kind in ELIGIBLE_KINDS
            and (manual or structure.get("role") not in EXCLUDED_ROLES)
            and handwriting_signal
            and bool(text)
            and "\n" not in text
            and len(text) <= self.max_incumbent_characters
            and region.confidence is not None
            and (manual or region.confidence < self.confidence_threshold)
            and (self.text_provider is None or region.provider == self.text_provider)
        )

    def _apply_candidate(
        self,
        region: TextRegion,
        tight_text: str,
        context_text: str,
        provenance: dict[str, Any],
    ) -> None:
        if tight_text != context_text:
            for view, text in (("tight", tight_text), ("context", context_text)):
                if (
                    text != region.text
                    and _literal_rejection(
                        text,
                        region.text,
                        self.max_candidate_characters,
                    )
                    is None
                ):
                    region.alternatives.append(
                        TextAlternative(
                            text=text,
                            confidence=None,
                            provider=f"{self.reader.name}:{view}",
                            text_provenance={**provenance, "view": view},
                        )
                    )
            _mark_review(region, "crop_disagreement", provenance)
            return

        rejection = _literal_rejection(
            tight_text,
            region.text,
            self.max_candidate_characters,
        )
        if rejection is not None:
            _mark_review(region, rejection, provenance)
            return

        if tight_text == region.text:
            current = dict(region.text_provenance or {})
            current["handwriting_specialist"] = {
                **provenance,
                "decision": "corroborated",
            }
            region.text_provenance = current
            return

        support = _independent_support(region, tight_text, self.reader.name)
        if support is not None:
            incumbent = TextAlternative(
                text=region.text,
                confidence=region.confidence,
                provider=region.provider,
                text_provenance=region.text_provenance,
            )
            region.text = tight_text
            region.confidence = support.confidence
            region.provider = self.reader.name
            region.text_provenance = {
                **provenance,
                "decision": "independently_corroborated_replacement",
                "supporting_provider": support.provider,
            }
            region.alternatives = [
                incumbent,
                *[item for item in region.alternatives if item is not support],
            ]
            _mark_review(region, "corroborated_replacement", provenance)
            return

        region.alternatives.append(
            TextAlternative(
                text=tight_text,
                confidence=None,
                provider=self.reader.name,
                text_provenance={**provenance, "view": "agreed"},
            )
        )
        _mark_review(region, "specialist_candidate", provenance)


def _literal_rejection(
    candidate: str,
    incumbent: str,
    max_candidate_characters: int,
) -> str | None:
    if not candidate:
        return "empty_candidate"
    if candidate.casefold() in ABSTENTION_OUTPUTS:
        return "abstention_candidate"
    if len(candidate) > max_candidate_characters:
        return "candidate_too_long"
    relative_limit = max(12, len(incumbent.strip()) * 3 + 8)
    if len(candidate) > relative_limit:
        return "candidate_expanded_context"
    if any(unicodedata.category(character).startswith("C") for character in candidate):
        return "candidate_control_characters"
    if "<|" in candidate or "|>" in candidate:
        return "candidate_control_tokens"
    return None


def _independent_support(
    region: TextRegion,
    candidate: str,
    specialist_provider: str,
) -> TextAlternative | None:
    normalized = _normalized_text(candidate)
    for alternative in region.alternatives:
        if alternative.provider.startswith(specialist_provider):
            continue
        if _normalized_text(alternative.text) == normalized:
            return alternative
    return None


def _normalized_text(text: str) -> str:
    return " ".join(text.casefold().split())


def _crop_boxes(
    box: BoundingBox,
    page_size: tuple[int, int],
    context_padding: int,
) -> tuple[BoundingBox, BoundingBox]:
    width, height = page_size
    tight = BoundingBox(
        max(0, box.left),
        max(0, box.top),
        min(width, box.right),
        min(height, box.bottom),
    )
    if tight.right <= tight.left or tight.bottom <= tight.top:
        raise ValueError("handwriting region lies outside the page")
    region_height = tight.bottom - tight.top
    padding = max(context_padding, math.ceil(region_height / 2))
    context = BoundingBox(
        max(0, tight.left - padding),
        max(0, tight.top - padding),
        min(width, tight.right + padding),
        min(height, tight.bottom + padding),
    )
    return tight, context


def _crop_provenance(
    page_number: int,
    tight: BoundingBox,
    context: BoundingBox,
    model: dict[str, Any],
) -> dict[str, Any]:
    return {
        "method": "dual_crop_exact_agreement",
        "page_number": page_number,
        "crops": {
            "tight": {"bounding_box": list(_box_tuple(tight))},
            "context": {"bounding_box": list(_box_tuple(context))},
        },
        "model": model,
    }


def _mark_review(
    region: TextRegion,
    reason: str,
    provenance: dict[str, Any],
) -> None:
    structure = dict(region.structure or {})
    structure["handwriting_review"] = {
        "required": True,
        "reason": reason,
        "provenance": provenance,
    }
    region.structure = structure


def _box_tuple(box: BoundingBox) -> tuple[int, int, int, int]:
    return box.left, box.top, box.right, box.bottom
