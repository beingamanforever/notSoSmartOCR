"""Page review from calibrated region confidence."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, UnidentifiedImageError

from .contracts import BoundingBox, TextRegion
from .falcon_layout import CALIBRATED_CONFIDENCE
from .providers import ReaderError

PROVIDER = "deterministic-evidence-risk"
REASON = "likely_misread_text"
RETRY_REASON = "sampled_retry_read"
SECOND_READER_REASON = "second_reader_doubt"
# docTR's printed-text reader doubts a word it holds. Falcon's likelihood stays high on
# confident handwriting misreads, and a monotone calibration cannot reorder those; routing
# on this too sent 42 of 81 missed handwritten snippets to Qwen instead of 22, for 19.5%
# of text regions instead of 13.5% (hw cohort, 2026-09-29).
SECOND_READER_DOUBT = 0.5


class EvidenceRiskStage:
    """Flag regions more likely misread than read, without changing OCR text."""

    name = "evidence-risk"

    def apply(
        self,
        image_path: Path,
        page_number: int,
        regions: list[TextRegion],
    ) -> list[TextRegion]:
        risks = [(region.id, reason) for region in regions if (reason := _risk(region))]
        if not risks:
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
                "reasons": sorted({reason for _, reason in risks}),
                "region_risks": [
                    {"region_id": region_id, "reasons": [reason]}
                    for region_id, reason in risks
                ],
            },
        )
        return [*regions, risk]


def _risk(region: TextRegion) -> str | None:
    provenance = region.text_provenance or {}
    if provenance.get("read_retries"):
        return RETRY_REASON  # a sampled retry has no calibrated probability
    calibrated = provenance.get("confidence_meaning") == CALIBRATED_CONFIDENCE
    # Bayes decision with equal costs on a calibrated probability of a correct read. It is
    # also the Qwen deferral point (CAUC, arXiv 2609.11446): on 1,699 NAF test regions
    # Qwen beat Falcon below 0.5 (0.42 vs 0.24, 0.72 vs 0.50) and not above it.
    if calibrated and region.confidence is not None and region.confidence < 0.5:
        return REASON
    second = provenance.get("second_reader") or {}
    if second.get("min_word_confidence", 1.0) < SECOND_READER_DOUBT:
        return SECOND_READER_REASON
    return None
