from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from ocr_pipeline.contracts import BoundingBox, TextAlternative, TextRegion
from ocr_pipeline.faint_text import FaintTinyTextStage, _find_proposals
from ocr_pipeline.pipeline import process_document
from ocr_pipeline.providers import ReaderError


class FixedReader:
    name = "first-pass"

    def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
        return [_heading()]


class CropReader:
    name = "crop-reader"

    def __init__(self, *, confidence: float = 0.96) -> None:
        self.confidence = confidence
        self.calls: list[tuple[int, int]] = []

    def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
        with Image.open(image_path) as image:
            width, height = image.size
        self.calls.append((width, height))
        return [
            TextRegion(
                id="reread-1",
                kind="text",
                text="fax ref 2048",
                confidence=self.confidence,
                bounding_box=BoundingBox(0, 0, width, height),
                reading_order=1,
                provider=self.name,
                text_provenance={"method": "controlled-reread"},
            )
        ]


def test_selective_high_resolution_recovers_pixels_missed_by_first_pass(
    tmp_path: Path,
) -> None:
    source = _page(tmp_path / "faint-tiny.png")
    reader = CropReader()

    baseline = process_document(source, FixedReader())
    recovered = process_document(
        source,
        FixedReader(),
        stages=[FaintTinyTextStage(reader)],
    )

    assert baseline.pages[0].text.value == "DISCHARGE SUMMARY"
    assert recovered.pages[0].text.value == "DISCHARGE SUMMARY fax ref 2048"
    assert len(reader.calls) == 1
    assert reader.calls[0][0] < 500 * 3
    region = recovered.pages[0].regions[-1]
    assert region.text == "fax ref 2048"
    assert region.resolution == "resolved"
    assert region.provider == "crop-reader"
    assert region.bounding_box.left <= 315
    assert region.bounding_box.right >= 368
    provenance = region.text_provenance
    assert provenance["method"] == ("unowned_pixel_proposal_and_high_resolution_reread")
    assert provenance["page_number"] == 1
    assert provenance["view"] == "selective_crop_high_resolution"
    assert provenance["scale"] == 3
    assert provenance["reader"] == "crop-reader"
    assert provenance["proposal"]["component_count"] >= 2
    assert provenance["proposal"]["ink_pixels"] > 0
    assert provenance["source_region_id"] == "reread-1"
    assert provenance["source_provider"] == "crop-reader"
    assert provenance["source_text_provenance"]["method"] == "controlled-reread"
    assert provenance["source_text_provenance"]["faint_tiny_view"]["scale"] == 3


def test_native_global_and_selective_views_are_ablatable_on_same_page(
    tmp_path: Path,
) -> None:
    source = _page(tmp_path / "same-page.png")
    observed = {}

    class AblationReader(CropReader):
        def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
            with Image.open(image_path) as image:
                width, height = image.size
            self.calls.append((width, height))
            if (width, height) == (500, 220):
                box = BoundingBox(315, 188, 369, 199)
            elif (width, height) == (1500, 660):
                box = BoundingBox(945, 564, 1107, 597)
            else:
                box = BoundingBox(0, 0, width, height)
            return [
                TextRegion(
                    id="reread-1",
                    kind="text",
                    text="fax ref 2048",
                    confidence=0.96,
                    bounding_box=box,
                    reading_order=1,
                    provider=self.name,
                )
            ]

    for view in (
        "native",
        "global_high_resolution",
        "selective_crop_high_resolution",
    ):
        reader = AblationReader()
        result = process_document(
            source,
            FixedReader(),
            stages=[FaintTinyTextStage(reader, view=view)],
        )
        observed[view] = {
            "text": result.pages[0].text.value,
            "size": reader.calls[0],
            "view": result.pages[0].regions[-1].structure["recovery_view"],
        }

    assert {item["text"] for item in observed.values()} == {
        "DISCHARGE SUMMARY fax ref 2048"
    }
    assert observed["native"]["size"] == (500, 220)
    assert observed["global_high_resolution"]["size"] == (1500, 660)
    assert observed["selective_crop_high_resolution"]["size"][0] < 1500
    assert [observed[view]["view"] for view in observed] == list(observed)


def test_blank_page_and_ruling_do_not_create_text_proposals(tmp_path: Path) -> None:
    for name, ruling in (("blank.png", False), ("ruling.png", True)):
        source = tmp_path / name
        image = Image.new("L", (500, 220), "white")
        draw = ImageDraw.Draw(image)
        draw.text((24, 26), "DISCHARGE SUMMARY", fill=0)
        if ruling:
            draw.line((260, 188, 460, 188), fill=175, width=1)
        image.save(source)
        reader = CropReader()

        result = process_document(
            source,
            FixedReader(),
            stages=[FaintTinyTextStage(reader)],
        )

        assert result.pages[0].regions == [_heading()]
        assert reader.calls == []


def test_low_confidence_reread_remains_unreadable_alternative(tmp_path: Path) -> None:
    source = _page(tmp_path / "uncertain.png")

    result = process_document(
        source,
        FixedReader(),
        stages=[FaintTinyTextStage(CropReader(confidence=0.6))],
    )

    candidate = result.pages[0].regions[-1]
    assert result.pages[0].route == "review"
    assert result.pages[0].text.value == "DISCHARGE SUMMARY"
    assert candidate.text == ""
    assert candidate.confidence is None
    assert candidate.resolution == "unreadable"
    assert [
        (alternative.text, alternative.confidence)
        for alternative in candidate.alternatives
    ] == [("fax ref 2048", 0.6)]


def test_reread_failure_preserves_first_pass_and_is_reported(tmp_path: Path) -> None:
    source = _page(tmp_path / "failure.png")

    class FailingReader:
        name = "failed-reread"

        def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
            raise ReaderError("reread_unavailable", "reader failed")

    executions: list[dict[str, object]] = []
    result = process_document(
        source,
        FixedReader(),
        stages=[FaintTinyTextStage(FailingReader())],
        stage_execution=executions,
    )

    assert result.pages[0].text.value == "DISCHARGE SUMMARY"
    assert result.pages[0].regions == [_heading()]
    assert result.pages[0].route == "review"
    assert [(failure.stage, failure.code) for failure in result.failures] == [
        ("faint-tiny-text", "reread_unavailable")
    ]
    assert executions[0]["status"] == "failed"
    assert executions[0]["failure_code"] == "reread_unavailable"


def test_selective_crops_use_one_reader_batch(tmp_path: Path) -> None:
    source = tmp_path / "two-lines.png"
    image = Image.new("L", (500, 240), "white")
    draw = ImageDraw.Draw(image)
    draw.text((24, 26), "DISCHARGE SUMMARY", fill=0)
    draw.text((315, 170), "fax ref 2048", fill=185)
    draw.text((36, 210), "copy to care", fill=185)
    image.save(source)

    class BatchReader(CropReader):
        batch_size = 8

        def __init__(self) -> None:
            super().__init__()
            self.batches: list[int] = []

        def read_batch(
            self,
            paths: list[Path],
            page_numbers: list[int],
        ) -> list[list[TextRegion]]:
            self.batches.append(len(paths))
            assert page_numbers == [1] * len(paths)
            return [self.read(path, 1) for path in paths]

    reader = BatchReader()
    result = process_document(
        source,
        FixedReader(),
        stages=[FaintTinyTextStage(reader)],
    )

    assert reader.batches == [2]
    assert len(result.pages[0].regions) == 3


def test_empty_reader_recovery_preserves_failure_and_empty_evidence(
    tmp_path: Path,
) -> None:
    source = _faint_only_page(tmp_path / "empty-reader.png")

    class EmptyReader:
        name = "empty-reader"

        def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
            return []

    result = process_document(
        source,
        EmptyReader(),
        stages=[FaintTinyTextStage(CropReader())],
    )

    page = result.pages[0]
    assert [failure.code for failure in result.failures] == ["no_text_detected"]
    assert page.failure_ids == [result.failures[0].id]
    assert page.route == "review"
    assert page.regions[0] == TextRegion(
        id="p1-empty-page-1",
        kind="page_text",
        text="",
        confidence=None,
        bounding_box=BoundingBox(0, 0, 500, 220),
        reading_order=1,
        provider="empty-reader",
    )
    assert page.regions[1].text == "fax ref 2048"
    assert page.text.value == " fax ref 2048"


def test_unresolved_full_page_region_does_not_claim_residual_pixels(
    tmp_path: Path,
) -> None:
    source = _faint_only_page(tmp_path / "unresolved-page.png")

    class UnresolvedReader:
        name = "unresolved-reader"

        def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
            return [
                TextRegion(
                    id="unresolved-page",
                    kind="text",
                    text="unreadable evidence",
                    confidence=None,
                    bounding_box=BoundingBox(0, 0, 500, 220),
                    reading_order=1,
                    provider=self.name,
                    resolution="unreadable",
                )
            ]

    result = process_document(
        source,
        UnresolvedReader(),
        stages=[FaintTinyTextStage(CropReader())],
    )

    assert [region.id for region in result.pages[0].regions] == [
        "unresolved-page",
        "p1-faint-tiny-1-1",
    ]
    assert result.pages[0].regions[0].resolution == "unreadable"
    assert result.pages[0].regions[1].text == "fax ref 2048"


def test_recovered_region_is_inserted_between_existing_reading_order(
    tmp_path: Path,
) -> None:
    source = tmp_path / "middle.png"
    image = Image.new("L", (500, 220), "white")
    draw = ImageDraw.Draw(image)
    draw.text((24, 26), "DISCHARGE SUMMARY", fill=0)
    draw.text((315, 105), "fax ref 2048", fill=185)
    draw.text((24, 188), "END OF PAGE", fill=0)
    image.save(source)

    class TopBottomReader:
        name = "top-bottom"

        def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
            return [
                _heading(),
                TextRegion(
                    id="footer",
                    kind="text",
                    text="END OF PAGE",
                    confidence=0.99,
                    bounding_box=BoundingBox(24, 188, 130, 204),
                    reading_order=2,
                    provider=self.name,
                ),
            ]

    result = process_document(
        source,
        TopBottomReader(),
        stages=[FaintTinyTextStage(CropReader())],
    )

    page = result.pages[0]
    assert [region.id for region in page.regions] == [
        "heading",
        "p1-faint-tiny-1-1",
        "footer",
    ]
    assert [region.reading_order for region in page.regions] == [1, 2, 2]
    assert page.text.value == "DISCHARGE SUMMARY fax ref 2048 END OF PAGE"


def test_unresolved_reread_preserves_and_deduplicates_all_alternatives(
    tmp_path: Path,
) -> None:
    source = _page(tmp_path / "alternatives.png")

    class AlternativeReader(CropReader):
        def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
            with Image.open(image_path) as image:
                width, height = image.size
            return [
                TextRegion(
                    id="reread-1",
                    kind="text",
                    text="",
                    confidence=0.6,
                    bounding_box=BoundingBox(0, 0, width, height),
                    reading_order=1,
                    provider=self.name,
                    text_provenance={"method": "primary"},
                    resolution="unreadable",
                    alternatives=[
                        TextAlternative(
                            "fax ref 2048",
                            0.6,
                            self.name,
                            {"method": "primary"},
                        ),
                        TextAlternative(
                            "FAX  REF  2048",
                            0.5,
                            self.name,
                            {"method": "duplicate"},
                        ),
                        TextAlternative(
                            "fax ref 204B",
                            0.4,
                            "second-reader",
                            {"method": "alternate"},
                        ),
                    ],
                )
            ]

    result = process_document(
        source,
        FixedReader(),
        stages=[FaintTinyTextStage(AlternativeReader())],
    )

    recovered = result.pages[0].regions[-1]
    assert recovered.resolution == "unreadable"
    assert [(item.text, item.provider) for item in recovered.alternatives] == [
        ("fax ref 2048", "crop-reader"),
        ("fax ref 204B", "second-reader"),
    ]
    assert recovered.alternatives[0].text_provenance["method"] == "primary"
    assert recovered.alternatives[1].text_provenance["method"] == "alternate"
    assert all(
        "faint_tiny_view" in item.text_provenance
        and "faint_tiny_recovery" in item.text_provenance
        for item in recovered.alternatives
    )


def test_proposal_search_returns_only_one_over_the_configured_limit() -> None:
    image = Image.new("L", (600, 420), "white")
    draw = ImageDraw.Draw(image)
    for top in range(20, 380, 30):
        draw.text((40, top), f"faint line {top}", fill=175)

    proposals = _find_proposals(image.convert("RGB"), [], proposal_limit=2)

    assert len(proposals) == 3


def test_proposal_limit_fails_before_any_reread_and_keeps_prior_evidence(
    tmp_path: Path,
) -> None:
    source = tmp_path / "proposal-limit.png"
    image = Image.new("L", (600, 420), "white")
    draw = ImageDraw.Draw(image)
    for top in range(20, 380, 30):
        draw.text((40, top), f"faint line {top}", fill=175)
    image.save(source)
    rereader = CropReader()

    result = process_document(
        source,
        FixedReader(),
        stages=[FaintTinyTextStage(rereader, max_proposals=2)],
    )

    assert result.pages[0].regions == [_heading()]
    assert result.pages[0].text.value == "DISCHARGE SUMMARY"
    assert rereader.calls == []
    assert [failure.code for failure in result.failures] == [
        "faint_tiny_proposal_limit_exceeded"
    ]


def _page(path: Path) -> Path:
    image = Image.new("L", (500, 220), "white")
    draw = ImageDraw.Draw(image)
    draw.text((24, 26), "DISCHARGE SUMMARY", fill=0)
    draw.text((315, 188), "fax ref 2048", fill=185)
    image.save(path)
    return path


def _faint_only_page(path: Path) -> Path:
    image = Image.new("L", (500, 220), "white")
    ImageDraw.Draw(image).text((315, 188), "fax ref 2048", fill=185)
    image.save(path)
    return path


def _heading() -> TextRegion:
    return TextRegion(
        id="heading",
        kind="text",
        text="DISCHARGE SUMMARY",
        confidence=0.99,
        bounding_box=BoundingBox(24, 26, 210, 42),
        reading_order=1,
        provider="first-pass",
    )
