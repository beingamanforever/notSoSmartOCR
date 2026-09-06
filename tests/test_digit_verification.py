from __future__ import annotations

from pathlib import Path

from PIL import Image
import pytest

from ocr_pipeline.contracts import BoundingBox, TextRegion
from ocr_pipeline.digit_verification import DigitVerificationStage
from ocr_pipeline.providers import ReaderError


class ScriptedReader:
    name = "tesseract"

    def __init__(self, replies: list[str] | ReaderError) -> None:
        self.replies = replies
        self.calls = 0

    def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
        if isinstance(self.replies, ReaderError):
            raise self.replies
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return [
            TextRegion(
                id="c-1",
                kind="text",
                text=reply,
                confidence=0.9,
                bounding_box=BoundingBox(0, 0, 10, 10),
                reading_order=1,
                provider=self.name,
                text_provenance={},
                resolution="resolved",
                structure={},
            )
        ]


def _region(identifier: str, text: str, **structure: object) -> TextRegion:
    return TextRegion(
        id=identifier,
        kind="text",
        text=text,
        confidence=0.93,
        bounding_box=BoundingBox(10, 10, 120, 34),
        reading_order=1,
        provider="reader",
        text_provenance={},
        resolution="resolved",
        structure=dict(structure),
    )


def _page(tmp_path: Path) -> Path:
    source = tmp_path / "page.png"
    Image.new("RGB", (400, 200), "white").save(source)
    return source


def test_disagreement_becomes_review_evidence_not_a_silent_replacement(
    tmp_path: Path,
) -> None:
    region = _region("r-1", "77899")
    reader = ScriptedReader(["77099"])

    DigitVerificationStage(reader).apply(_page(tmp_path), 1, [region])

    assert region.text == "77899", "the primary reading must not be overwritten"
    assert region.resolution == "conflicting"
    assert region.structure["digit_verification"]["outcome"] == "disagreed"
    assert region.structure["review_required"] is True
    assert [alt.text for alt in region.alternatives] == ["77099"]
    assert region.alternatives[0].provider == "tesseract"


def test_agreement_is_recorded_as_independent_confirmation(tmp_path: Path) -> None:
    region = _region("r-1", "77099")

    DigitVerificationStage(ScriptedReader(["77099"])).apply(
        _page(tmp_path), 1, [region]
    )

    assert region.resolution == "resolved"
    assert region.structure["digit_verification"]["outcome"] == "confirmed"
    assert region.alternatives == []


def test_punctuation_and_spacing_differences_are_not_disagreement(
    tmp_path: Path,
) -> None:
    region = _region("r-1", "346-901-9997")

    DigitVerificationStage(ScriptedReader(["346 901 9997"])).apply(
        _page(tmp_path), 1, [region]
    )

    assert region.structure["digit_verification"]["outcome"] == "confirmed"
    assert region.resolution == "resolved"


def test_regions_without_digits_are_never_reread(tmp_path: Path) -> None:
    region = _region("r-1", "Patient Street Address")
    reader = ScriptedReader(["anything"])

    DigitVerificationStage(reader).apply(_page(tmp_path), 1, [region])

    assert reader.calls == 0
    assert "digit_verification" not in (region.structure or {})


def test_table_and_formula_evidence_is_left_to_its_own_specialist(
    tmp_path: Path,
) -> None:
    reader = ScriptedReader(["999"])
    regions = [
        _region("r-1", "77099", role="table_source"),
        _region("r-2", "77099", layout_owner_type="formula"),
    ]

    DigitVerificationStage(reader).apply(_page(tmp_path), 1, regions)

    assert reader.calls == 0


def test_an_empty_or_failed_reread_leaves_the_region_untouched(
    tmp_path: Path,
) -> None:
    for reader in (
        ScriptedReader([""]),
        ScriptedReader(ReaderError("reader_failed", "boom")),
    ):
        region = _region("r-1", "77099")
        DigitVerificationStage(reader).apply(_page(tmp_path), 1, [region])
        assert region.resolution == "resolved"
        assert "digit_verification" not in (region.structure or {})


def test_the_longest_digit_runs_are_verified_first_within_the_budget(
    tmp_path: Path,
) -> None:
    reader = ScriptedReader(["0"])
    regions = [_region("r-short", "7"), _region("r-long", "1234567890")]

    DigitVerificationStage(reader, max_regions=1).apply(_page(tmp_path), 1, regions)

    assert reader.calls == 1
    assert "digit_verification" in (regions[1].structure or {})
    assert "digit_verification" not in (regions[0].structure or {})


def test_stage_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="max_regions"):
        DigitVerificationStage(ScriptedReader([]), max_regions=0)
    with pytest.raises(ValueError, match="padding"):
        DigitVerificationStage(ScriptedReader([]), padding=-1)
