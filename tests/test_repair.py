from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from ocr_pipeline.contracts import (
    BoundingBox,
    DocumentResult,
    EvidenceText,
    Failure,
    PageResult,
    TextAlternative,
    TextRegion,
)
from ocr_pipeline.openrouter import OpenRouterError
from ocr_pipeline.repair import repair_risky_regions


def test_selective_repair_abstains_when_candidate_is_unverified(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (100, 80), "white").save(image_path)
    document = _document()

    def repair(crop: Path, prompt: str, schema: dict[str, object]) -> SimpleNamespace:
        with Image.open(crop) as image:
            assert image.format == "PNG"
            assert image.size == (20, 20)
        assert "risky" in prompt
        assert schema["properties"]["id"]["enum"] == ["risky"]
        return SimpleNamespace(
            content={"id": "risky", "text": "recovered"},
            provider="test-provider",
            usage={"prompt_tokens": 7},
            cost=0.0001,
        )

    repaired, records = repair_risky_regions(document, {1: image_path}, repair)

    assert document.pages[0].regions[0].text == ""
    assert repaired is document
    assert records == [
        {
            "region_id": "risky",
            "page_number": 1,
            "status": "abstained",
            "reason": "unverified_candidate",
            "primary": {
                "model": None,
                "provider": "test-provider",
                "usage": {"prompt_tokens": 7},
                "cost": 0.0001,
            },
            "verifier": None,
            "reported_cost": 0.0001,
        }
    ]


def test_selective_repair_abstains_when_independent_reads_disagree(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (100, 80), "white").save(image_path)
    document = _document()

    def primary(*_args: object) -> SimpleNamespace:
        return _result("arbitrary text", model="primary-model", cost=0.1)

    def verifier(*_args: object) -> SimpleNamespace:
        return _result("visual text", model="verifier-model", cost=0.2)

    repaired, records = repair_risky_regions(
        document,
        {1: image_path},
        primary,
        verifier,
    )

    assert repaired is document
    assert records[0]["reason"] == "verifier_disagreement"
    assert records[0]["primary"]["model"] == "primary-model"
    assert records[0]["primary"]["provider"] == "primary-model-provider"
    assert records[0]["primary"]["usage"] == {"total_tokens": 10}
    assert records[0]["verifier"]["model"] == "verifier-model"
    assert records[0]["verifier"]["provider"] == "verifier-model-provider"
    assert records[0]["verifier"]["usage"] == {"total_tokens": 10}
    assert records[0]["reported_cost"] == 0.3


def test_selective_repair_accepts_only_independent_literal_agreement(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (100, 80), "white").save(image_path)
    document = _document()
    document.pages[0].regions[0].text_provenance = {"method": "initial-local"}
    primary_crop: Path | None = None

    def primary(crop: Path, prompt: str, schema: dict[str, object]) -> SimpleNamespace:
        nonlocal primary_crop
        primary_crop = crop
        assert schema == _strict_schema()
        return _result("  Cafe\u0301\r\nNorth  ", model="primary-model", cost=0.1)

    def verifier(crop: Path, prompt: str, schema: dict[str, object]) -> SimpleNamespace:
        assert crop == primary_crop
        assert schema == _strict_schema()
        assert "Cafe" not in prompt
        assert "candidate" not in prompt.casefold()
        assert "reference" not in prompt.casefold()
        assert "subset" not in prompt.casefold()
        return _result("Caf\u00e9\nNorth", model="verifier-model", cost=0.2)

    repaired, records = repair_risky_regions(
        document,
        {1: image_path},
        primary,
        verifier,
    )

    assert repaired.pages[0].regions[0].text == "  Cafe\u0301\r\nNorth  "
    assert repaired.pages[0].regions[0].confidence is None
    assert repaired.pages[0].regions[0].provider == "primary-model-provider"
    assert repaired.pages[0].regions[0].text_provenance == {
        "mode": "hosted_patch",
        "accepted_correction": {
            "provider": "primary-model-provider",
            "model": "primary-model",
        },
        "primary_model": "primary-model",
        "primary_provider": "primary-model-provider",
        "verifier_model": "verifier-model",
        "verifier_provider": "verifier-model-provider",
    }
    assert repaired.pages[0].regions[0].alternatives[0].text == ""
    assert repaired.pages[0].regions[0].alternatives[0].confidence == 0.2
    assert repaired.pages[0].regions[0].alternatives[0].provider == "local"
    assert repaired.pages[0].regions[0].alternatives[0].text_provenance == {
        "method": "initial-local"
    }
    assert repaired.pages[0].regions[0].alternatives[0].decision_state == "superseded"
    assert repaired.pages[0].route == "accept_hosted_patch"
    assert records[0]["status"] == "accepted"
    assert records[0]["reason"] == "independent_agreement"
    assert records[0]["reported_cost"] == 0.3


def test_selective_repair_preserves_history_and_rejects_only_pending_candidates(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (100, 80), "white").save(image_path)
    document = _document("Accession No. Accession No. Accession No.")
    document.pages[0].regions[0].alternatives = [
        TextAlternative("older", 0.7, "reader-a", decision_state="accepted"),
        TextAlternative("candidate", 0.8, "reader-b", decision_state="pending"),
    ]

    def primary(*_args: object) -> SimpleNamespace:
        return _result("Accession No.", model="primary-model", cost=0.0)

    def verifier(*_args: object) -> SimpleNamespace:
        return _result("Accession No.", model="verifier-model", cost=0.0)

    repaired, records = repair_risky_regions(
        document,
        {1: image_path},
        primary,
        verifier,
    )

    assert records[0]["status"] == "accepted"
    assert [
        (alternative.text, alternative.decision_state)
        for alternative in repaired.pages[0].regions[0].alternatives
    ] == [
        ("older", "accepted"),
        ("candidate", "rejected"),
        ("Accession No. Accession No. Accession No.", "superseded"),
    ]


def test_selective_repair_abstains_without_improvement(tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (100, 80), "white").save(image_path)
    document = _document("<table><tr><td>broken</table>", kind="table")

    def primary(*_args: object) -> SimpleNamespace:
        return _result("<table><tr><td>broken</table>", model="primary-model", cost=0.0)

    def verifier(*_args: object) -> SimpleNamespace:
        return _result(
            "<table><tr><td>broken</table>", model="verifier-model", cost=0.0
        )

    repaired, records = repair_risky_regions(
        document,
        {1: image_path},
        primary,
        verifier,
    )

    assert repaired is document
    assert records[0]["status"] == "abstained"
    assert records[0]["reason"] == "no_deterministic_improvement"


def test_selective_repair_accepts_repeated_text_replacement(tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (100, 80), "white").save(image_path)
    document = _document("Accession No. Accession No. Accession No.")
    document.pages[0].regions[0].text_provenance = {"method": "initial-local"}

    def primary(*_args: object) -> SimpleNamespace:
        return _result("Accession No.", model="primary-model", cost=0.0)

    def verifier(*_args: object) -> SimpleNamespace:
        return _result("Accession No.", model="verifier-model", cost=0.0)

    repaired, records = repair_risky_regions(
        document,
        {1: image_path},
        primary,
        verifier,
    )

    assert repaired.pages[0].regions[0].text == "Accession No."
    assert repaired.pages[0].regions[0].provider == "primary-model-provider"
    assert [
        (
            alternative.text,
            alternative.provider,
            alternative.text_provenance,
            alternative.decision_state,
        )
        for alternative in repaired.pages[0].regions[0].alternatives
    ] == [
        (
            "Accession No. Accession No. Accession No.",
            "local",
            {"method": "initial-local"},
            "superseded",
        )
    ]
    assert records[0]["status"] == "accepted"


def test_selective_repair_does_not_call_provider_for_invalid_bbox() -> None:
    document = _document()
    document.pages[0].regions[0].bounding_box.right = 101

    def repair(*_args: object) -> SimpleNamespace:
        raise AssertionError("repair must not be called")

    repaired, records = repair_risky_regions(document, {}, repair)

    assert repaired is document
    assert records[0]["status"] == "abstained"
    assert records[0]["reason"] == "invalid_bbox"


def test_empty_table_requires_structured_output(tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (100, 80), "white").save(image_path)
    document = _document(kind="table")

    def primary(*_args: object) -> SimpleNamespace:
        return _result("A B", model="primary-model", cost=0.0)

    def verifier(*_args: object) -> SimpleNamespace:
        return _result("A B", model="verifier-model", cost=0.0)

    repaired, records = repair_risky_regions(
        document, {1: image_path}, primary, verifier
    )

    assert repaired is document
    assert records[0]["reason"] == "invalid_table_repair"


def test_same_actual_model_is_not_independent(tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (100, 80), "white").save(image_path)
    document = _document()

    def response(*_args: object) -> SimpleNamespace:
        return _result("recovered", model="same-model", cost=0.0)

    repaired, records = repair_risky_regions(
        document, {1: image_path}, response, response
    )

    assert repaired is document
    assert records[0]["reason"] == "non_independent_models"


def test_provider_failures_are_structured_and_programming_errors_escape(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (100, 80), "white").save(image_path)
    document = _document()

    def forbidden(*_args: object) -> SimpleNamespace:
        raise OpenRouterError(
            "denied",
            code="http_403",
            status_code=403,
            attempts=1,
            latency_ms=12.5,
        )

    repaired, records = repair_risky_regions(document, {1: image_path}, forbidden)
    assert repaired is document
    assert records[0]["reason"] == "primary_http_403"
    assert records[0]["primary_error"] == {
        "code": "http_403",
        "message": "denied",
        "status_code": 403,
        "attempts": 1,
        "latency_ms": 12.5,
    }

    def bug(*_args: object) -> SimpleNamespace:
        raise TypeError("implementation defect")

    try:
        repair_risky_regions(document, {1: image_path}, bug)
    except TypeError as error:
        assert str(error) == "implementation defect"
    else:
        raise AssertionError("unexpected implementation errors must not be hidden")


def test_full_page_repair_resolves_no_text_failure(tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (100, 80), "white").save(image_path)
    document = _document(kind="page_text")
    document.status = "failed"
    document.failures = [
        Failure(
            id="failure-1",
            stage="ocr",
            code="no_text_detected",
            message="The reader returned no text regions",
            page_number=1,
        )
    ]
    document.pages[0].failure_ids = ["failure-1"]

    def primary(*_args: object) -> SimpleNamespace:
        return _result("Recovered page", model="primary-model", cost=0.0)

    def verifier(*_args: object) -> SimpleNamespace:
        return _result("Recovered page", model="verifier-model", cost=0.0)

    repaired, records = repair_risky_regions(
        document, {1: image_path}, primary, verifier
    )

    assert records[0]["status"] == "accepted"
    assert repaired.pages[0].text.value == "Recovered page"
    assert repaired.pages[0].failure_ids == []
    assert repaired.failures == []
    assert repaired.status == "success"


def _document(text: str = "", *, kind: str = "word") -> DocumentResult:
    region = TextRegion(
        id="risky",
        kind=kind,
        text=text,
        confidence=0.2,
        bounding_box=BoundingBox(10, 20, 30, 40),
        reading_order=1,
        provider="local",
    )
    return DocumentResult(
        document_id="doc",
        source={"name": "page.png", "kind": "image"},
        status="success",
        pages=[
            PageResult(
                page_number=1,
                width=100,
                height=80,
                reader="local",
                route="review",
                text=EvidenceText(text, [region.id]),
                regions=[region],
            )
        ],
    )


def _result(text: str, *, model: str, cost: float) -> SimpleNamespace:
    return SimpleNamespace(
        content={"id": "risky", "text": text},
        model=model,
        provider=f"{model}-provider",
        usage={"total_tokens": 10},
        cost=cost,
    )


def _strict_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "id": {"type": "string", "enum": ["risky"]},
            "text": {"type": "string"},
        },
        "required": ["id", "text"],
        "additionalProperties": False,
    }
