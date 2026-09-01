"""Conservative page-level review signals from OCR region evidence."""

from __future__ import annotations

import statistics
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from .contracts import BoundingBox, TextRegion
from .providers import ReaderError

PROVIDER = "deterministic-evidence-risk"


class EvidenceRiskStage:
    """Flag weak, tiny, or handwriting-like evidence without changing OCR text."""

    name = "evidence-risk"

    def __init__(
        self,
        *,
        text_provider: str | None = None,
        minimum_mean_confidence: float = 0.75,
        table_mean_confidence: float = 0.92,
        small_text_height: int = 10,
        small_text_mean_confidence: float = 0.9,
        large_region_ratio: float = 2.0,
        large_region_confidence: float = 0.8,
        minimum_large_regions: int = 2,
    ) -> None:
        if not 0 <= minimum_mean_confidence <= 1:
            raise ValueError("minimum_mean_confidence must be from 0 to 1")
        if not 0 <= table_mean_confidence <= 1:
            raise ValueError("table_mean_confidence must be from 0 to 1")
        if small_text_height < 1:
            raise ValueError("small_text_height must be positive")
        if not 0 <= small_text_mean_confidence <= 1:
            raise ValueError("small_text_mean_confidence must be from 0 to 1")
        if large_region_ratio <= 1:
            raise ValueError("large_region_ratio must be greater than 1")
        if not 0 <= large_region_confidence <= 1:
            raise ValueError("large_region_confidence must be from 0 to 1")
        if minimum_large_regions < 1:
            raise ValueError("minimum_large_regions must be positive")
        self.text_provider = text_provider
        self.minimum_mean_confidence = minimum_mean_confidence
        self.table_mean_confidence = table_mean_confidence
        self.small_text_height = small_text_height
        self.small_text_mean_confidence = small_text_mean_confidence
        self.large_region_ratio = large_region_ratio
        self.large_region_confidence = large_region_confidence
        self.minimum_large_regions = minimum_large_regions

    def apply(
        self,
        image_path: Path,
        page_number: int,
        regions: list[TextRegion],
    ) -> list[TextRegion]:
        evidence = [region for region in regions if self._is_primary_text(region)]
        if not evidence:
            return regions

        heights = [
            region.bounding_box.bottom - region.bounding_box.top for region in evidence
        ]
        mean_confidence = statistics.fmean(
            region.confidence for region in evidence if region.confidence is not None
        )
        median_height = statistics.median(heights)
        large_threshold = median_height * self.large_region_ratio
        large_low_confidence = sum(
            height >= large_threshold
            and region.confidence is not None
            and region.confidence < self.large_region_confidence
            for region, height in zip(evidence, heights, strict=True)
        )

        reasons = []
        if mean_confidence < self.minimum_mean_confidence:
            reasons.append("low_mean_confidence")
        if (
            any(region.kind == "table" for region in regions)
            and mean_confidence < self.table_mean_confidence
        ):
            reasons.append("table_text_uncertainty")
        if (
            median_height <= self.small_text_height
            and mean_confidence < self.small_text_mean_confidence
        ):
            reasons.append("small_text_evidence")
        if large_low_confidence >= self.minimum_large_regions:
            reasons.append("large_low_confidence_regions")
        if not reasons:
            return regions

        try:
            with Image.open(image_path) as image:
                width, height = image.size
        except (OSError, UnidentifiedImageError) as error:
            raise ReaderError("risk_image_failed", str(error)) from error

        risk = TextRegion(
            id=f"p{page_number}-evidence-risk-1",
            kind="coverage_risk",
            text="",
            confidence=None,
            bounding_box=BoundingBox(0, 0, width, height),
            reading_order=max((region.reading_order for region in regions), default=0)
            + 1,
            provider=PROVIDER,
            resolution="unreadable",
            structure={
                "role": "coverage_risk",
                "reasons": reasons,
                "metrics": {
                    "primary_regions": len(evidence),
                    "mean_confidence": round(mean_confidence, 6),
                    "median_height": median_height,
                    "large_low_confidence_regions": large_low_confidence,
                },
            },
        )
        return [*regions, risk]

    def _is_primary_text(self, region: TextRegion) -> bool:
        if not region.text.strip() or region.confidence is None:
            return False
        if region.kind in {"checkbox", "table", "coverage_risk"}:
            return False
        return self.text_provider is None or region.provider == self.text_provider
