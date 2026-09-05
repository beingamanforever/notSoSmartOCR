from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from ocr_pipeline.contracts import BoundingBox, TextAlternative, TextRegion
from ocr_pipeline.dispute_resolution import DisputeResolutionStage
from ocr_pipeline.openrouter import OpenRouterError
from ocr_pipeline.pipeline import process_document


def _decision(
    item_id: str,
    action: str,
    candidate_id: str | None = None,
    *,
    text: str | None = None,
) -> dict[str, object]:
    return {
        "id": item_id,
        "action": action,
        "candidate_id": candidate_id,
        "text": text,
    }


class RecordingCall:
    def __init__(self, result: SimpleNamespace) -> None:
        self.result = result
        self.calls = 0
        self.prompts: list[str] = []
        self.schemas: list[dict[str, object]] = []

    def __call__(
        self,
        image_path: Path,
        prompt: str,
        schema: dict[str, object],
    ) -> SimpleNamespace:
        self.calls += 1
        self.prompts.append(prompt)
        self.schemas.append(schema)
        with Image.open(image_path) as image:
            assert image.format == "PNG"
            assert image.mode == "RGB"
            assert image.width > 0
            assert image.height > 0
        return self.result


class FixedReader:
    name = "fixed"

    def __init__(self, regions: list[TextRegion]) -> None:
        self.regions = regions

    def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
        return self.regions


def test_two_eligible_models_resolve_many_region_and_table_conflicts_in_one_call_each(
    tmp_path: Path,
) -> None:
    source = tmp_path / "page.png"
    image = np.full((120, 200, 3), 240, dtype=np.uint8)
    Image.fromarray(image).save(source)
    first = _region("first", "Alpha", "conflicting", BoundingBox(10, 10, 40, 30))
    first.alternatives.append(TextAlternative("Beta", 0.8, "reader-b"))
    second = _region("second", "Gamma", "unreadable", BoundingBox(60, 10, 90, 30))
    table = _table_region()
    primary = RecordingCall(
        _result(
            [
                _decision("region:first", "select", "region:first:candidate:1"),
                _decision("region:second", "select", "region:second:candidate:0"),
                _decision(
                    "cell:table:cell-1",
                    "select",
                    "cell:table:cell-1:candidate:1",
                ),
            ],
            model="local-primary",
            provider="local-runtime",
        )
    )
    verifier = RecordingCall(
        _result(
            [
                _decision("region:first", "select", "region:first:candidate:1"),
                _decision("region:second", "select", "region:second:candidate:0"),
                _decision(
                    "cell:table:cell-1",
                    "select",
                    "cell:table:cell-1:candidate:1",
                ),
            ],
            model="local-verifier",
            provider="local-runtime",
        )
    )
    stage = DisputeResolutionStage(
        primary,
        verifier,
        crop_padding=6,
        eligible_local_models={"local-primary", "local-verifier"},
    )

    document = process_document(
        source,
        FixedReader([first, second, table]),
        stages=[stage],
    )
    result = document.pages[0].regions

    assert primary.calls == verifier.calls == 1
    assert all(
        item_id in primary.prompts[0]
        for item_id in (
            "region:first",
            "region:second",
            "cell:table:cell-1",
        )
    )
    decision_schema = primary.schemas[0]["properties"]["decisions"]
    assert decision_schema["minItems"] == 3
    assert decision_schema["maxItems"] == 3
    assert len(decision_schema["items"]["oneOf"]) == 3
    assert result[0].text == "Beta"
    assert result[0].provider == "reader-b"
    assert result[0].resolution == "resolved"
    assert [(item.text, item.provider) for item in result[0].alternatives] == [
        ("Alpha", "reader-a")
    ]
    assert result[1].text == "Gamma"
    assert result[1].resolution == "unreadable"
    assert result[1].structure["dispute_resolution"]["decision"] == "review_required"
    cell = result[2].structure["cells"][0]
    assert cell["text"] == "48"
    assert cell["source"] == "reader-b"
    assert cell["resolution"] == "resolved"
    assert [item["text"] for item in cell["alternatives"]] == ["43"]
    assert cell["dispute_resolution"]["primary"] == {
        "model": "local-primary",
        "provider": "local-runtime",
    }
    assert cell["dispute_resolution"]["verifier"] == {
        "model": "local-verifier",
        "provider": "local-runtime",
    }
    with Image.open(source) as unchanged:
        assert np.array_equal(np.asarray(unchanged), image)


def test_table_cell_selection_swaps_complete_candidate_evidence_atomically(
    tmp_path: Path,
) -> None:
    source = _page(tmp_path)
    table = _table_region()
    cell = table.structure["cells"][0]
    cell.update(
        {
            "evidence_ids": ["primary-43"],
            "supporters": [{"provider": "reader-a", "evidence_ids": ["primary-43"]}],
            "decision": "strong_disagreement",
            "text_provenance": {
                "method": "assigned_cell_text",
                "evidence_ids": ["primary-43"],
            },
        }
    )
    cell["alternatives"][0]["text_provenance"] = {
        "method": "challenger_cell_text",
        "evidence_ids": ["challenger-48"],
    }
    decision = _decision(
        "cell:table:cell-1",
        "select",
        "cell:table:cell-1:candidate:1",
    )
    primary = RecordingCall(_result([decision], model="local-primary"))
    verifier = RecordingCall(_result([decision], model="local-verifier"))

    document = process_document(
        source,
        FixedReader([table]),
        stages=[
            DisputeResolutionStage(
                primary,
                verifier,
                eligible_local_models={"local-primary", "local-verifier"},
            )
        ],
    )

    selected = document.pages[0].regions[0].structure["cells"][0]
    assert selected["text"] == "48"
    assert selected["confidence"] == 0.8
    assert selected["source"] == "reader-b"
    assert selected["evidence_ids"] == ["challenger-48"]
    assert selected["supporters"] == [
        {"provider": "reader-b", "evidence_ids": ["challenger-48"]}
    ]
    assert selected["decision"] == "independent_dispute_agreement"
    assert selected["resolution"] == "resolved"
    assert selected["text_provenance"] == {
        "method": "challenger_cell_text",
        "evidence_ids": ["challenger-48"],
    }
    assert selected["dispute_resolution"]["decision"] == ("selected_existing_candidate")
    assert selected["alternatives"] == [
        {
            "text": "43",
            "confidence": 0.7,
            "provider": "reader-a",
            "text_provenance": {
                "method": "assigned_cell_text",
                "evidence_ids": ["primary-43"],
            },
            "evidence_ids": ["primary-43"],
            "supporters": [{"provider": "reader-a", "evidence_ids": ["primary-43"]}],
            "decision": "strong_disagreement",
            "resolution": "conflicting",
        }
    ]


def test_unverified_new_text_remains_an_alternative(tmp_path: Path) -> None:
    source = _page(tmp_path)
    region = _region(
        "unclear",
        "old reading",
        "unreadable",
        BoundingBox(10, 10, 40, 30),
    )
    call = RecordingCall(
        _result(
            [_decision("region:unclear", "transcribe", text="new visual text")],
            model="primary-model",
            provider="primary-provider",
        )
    )

    result = DisputeResolutionStage(call).apply(source, 1, [region])

    assert call.calls == 1
    assert result[0].text == "old reading"
    assert result[0].resolution == "unreadable"
    assert [(item.text, item.provider) for item in result[0].alternatives] == [
        ("new visual text", "primary-provider")
    ]
    assert result[0].structure["dispute_resolution"]["decision"] == ("review_required")


def test_two_models_never_auto_apply_new_transcription(
    tmp_path: Path,
) -> None:
    source = _page(tmp_path)
    region = _region("unclear", "old", "unreadable", BoundingBox(10, 10, 40, 30))
    primary = RecordingCall(
        _result(
            [_decision("region:unclear", "transcribe", text="  Cafe\u0301 North  ")],
            model="primary-model",
            provider="primary-provider",
        )
    )
    verifier = RecordingCall(
        _result(
            [_decision("region:unclear", "transcribe", text="Caf\u00e9\nNorth")],
            model="verifier-model",
            provider="verifier-provider",
        )
    )

    result = DisputeResolutionStage(
        primary,
        verifier,
        eligible_local_models={"primary-model", "verifier-model"},
    ).apply(source, 1, [region])

    assert primary.calls == verifier.calls == 1
    assert result[0].text == "old"
    assert result[0].resolution == "unreadable"
    assert [(item.text, item.provider) for item in result[0].alternatives] == [
        ("  Cafe\u0301 North  ", "primary-provider")
    ]
    provenance = result[0].structure["dispute_resolution"]
    assert provenance["decision"] == "review_required"
    assert provenance["primary"] == {
        "model": "primary-model",
        "provider": "primary-provider",
    }
    assert provenance["verifier"] == {
        "model": "verifier-model",
        "provider": "verifier-provider",
    }
    alternative_provenance = result[0].alternatives[0].text_provenance
    assert alternative_provenance["decision"] == "visual_transcription_review"
    assert alternative_provenance["primary"] == provenance["primary"]
    assert alternative_provenance["verifier"] == provenance["verifier"]


def test_one_model_selection_is_review_metadata_without_duplicate_alternative(
    tmp_path: Path,
) -> None:
    source = _page(tmp_path)
    region = _region("conflict", "Alpha", "conflicting", BoundingBox(5, 5, 25, 20))
    region.alternatives.append(TextAlternative("Beta", 0.8, "reader-b"))
    primary = RecordingCall(
        _result(
            [_decision("region:conflict", "select", "region:conflict:candidate:1")],
            model="local-primary",
            provider="local-runtime",
        )
    )

    result = DisputeResolutionStage(
        primary,
        eligible_local_models={"local-primary"},
    ).apply(source, 1, [region])

    assert primary.calls == 1
    assert result[0].text == "Alpha"
    assert result[0].resolution == "conflicting"
    assert [item.text for item in result[0].alternatives] == ["Beta"]
    provenance = result[0].structure["dispute_resolution"]
    assert provenance["decision"] == "review_required"
    assert provenance["primary_decision"] == {
        "action": "select",
        "candidate_id": "region:conflict:candidate:1",
    }
    assert provenance["verifier"] is None


@pytest.mark.parametrize(
    (
        "primary_model",
        "verifier_model",
        "primary_provider",
        "verifier_provider",
        "eligible_models",
    ),
    [
        ("same-model", "same-model", "runtime-a", "runtime-b", {"same-model"}),
        ("local-a", "remote-b", "runtime-a", "runtime-b", {"local-a"}),
        ("", "", "local-a", "local-b", {"local-a", "local-b"}),
    ],
)
def test_non_independent_or_ineligible_models_cannot_resolve(
    tmp_path: Path,
    primary_model: str,
    verifier_model: str,
    primary_provider: str,
    verifier_provider: str,
    eligible_models: set[str],
) -> None:
    source = _page(tmp_path)
    region = _region("conflict", "Alpha", "conflicting", BoundingBox(5, 5, 25, 20))
    region.alternatives.append(TextAlternative("Beta", 0.8, "reader-b"))
    decision = _decision(
        "region:conflict",
        "select",
        "region:conflict:candidate:1",
    )
    primary = RecordingCall(
        _result([decision], model=primary_model, provider=primary_provider)
    )
    verifier = RecordingCall(
        _result([decision], model=verifier_model, provider=verifier_provider)
    )

    result = DisputeResolutionStage(
        primary,
        verifier,
        eligible_local_models=eligible_models,
    ).apply(source, 1, [region])

    assert primary.calls == verifier.calls == 1
    assert result[0].text == "Alpha"
    assert result[0].resolution == "conflicting"
    assert [item.text for item in result[0].alternatives] == ["Beta"]
    assert result[0].structure["dispute_resolution"]["decision"] == "review_required"


@pytest.mark.parametrize(
    "decisions",
    [
        [_decision("region:first", "abstain")],
        [
            _decision("region:first", "abstain"),
            _decision("region:first", "abstain"),
        ],
        [
            _decision("region:first", "abstain"),
            _decision("region:extra", "abstain"),
        ],
    ],
)
def test_malformed_bulk_responses_fail_closed(
    tmp_path: Path,
    decisions: list[dict[str, object]],
) -> None:
    source = _page(tmp_path)
    regions = [
        _region("first", "one", "unreadable", BoundingBox(5, 5, 25, 20)),
        _region("second", "two", "conflicting", BoundingBox(30, 5, 50, 20)),
    ]
    call = RecordingCall(_result(decisions))

    result = DisputeResolutionStage(call).apply(source, 1, regions)

    assert call.calls == 1
    assert result is regions
    assert [region.resolution for region in result] == ["unreadable", "conflicting"]


def test_conflicting_item_cannot_introduce_new_text(tmp_path: Path) -> None:
    source = _page(tmp_path)
    regions = [_region("conflict", "one", "conflicting", BoundingBox(5, 5, 25, 20))]
    call = RecordingCall(
        _result([_decision("region:conflict", "transcribe", text="invented")])
    )

    result = DisputeResolutionStage(call).apply(source, 1, regions)

    assert result is regions
    assert regions[0].text == "one"


def test_invalid_bounds_and_provider_errors_fail_closed(tmp_path: Path) -> None:
    source = _page(tmp_path)
    invalid = _region(
        "invalid",
        "old",
        "unreadable",
        BoundingBox(5, 5, 101, 20),
    )
    forbidden = RecordingCall(_result([]))

    assert DisputeResolutionStage(forbidden).apply(source, 1, [invalid]) == [invalid]
    assert forbidden.calls == 0

    valid = _region("valid", "old", "unreadable", BoundingBox(5, 5, 25, 20))

    def provider_error(*_args: object) -> SimpleNamespace:
        raise OpenRouterError("provider unavailable")

    assert DisputeResolutionStage(provider_error).apply(source, 1, [valid]) == [valid]


def test_contact_sheet_limits_fail_closed_before_any_model_call(
    tmp_path: Path,
) -> None:
    source = _page(tmp_path)
    regions = [
        _region("first", "one", "unreadable", BoundingBox(5, 5, 25, 20)),
        _region("second", "two", "unreadable", BoundingBox(30, 5, 50, 20)),
    ]
    call = RecordingCall(_result([]))

    too_many = DisputeResolutionStage(call, max_disputes=1).apply(source, 1, regions)
    too_wide = DisputeResolutionStage(call, max_sheet_width=27).apply(
        source,
        1,
        [regions[0]],
    )
    too_tall = DisputeResolutionStage(call, max_sheet_height=42).apply(
        source,
        1,
        [regions[0]],
    )

    assert too_many is regions
    assert too_wide[0] is regions[0]
    assert too_tall[0] is regions[0]
    assert call.calls == 0


@pytest.mark.parametrize(
    ("kind", "role"),
    [
        ("checkbox", "control"),
        ("coverage_risk", "coverage_risk"),
        ("table_candidate", "table_candidate"),
    ],
)
def test_non_text_uncertainty_is_not_sent_to_text_dispute_resolution(
    tmp_path: Path,
    kind: str,
    role: str,
) -> None:
    source = _page(tmp_path)
    region = _region("state", "[?] Fall risk", "unreadable", BoundingBox(5, 5, 25, 20))
    region.kind = kind
    region.structure = {"role": role}
    call = RecordingCall(_result([]))

    result = DisputeResolutionStage(call).apply(source, 1, [region])

    assert result[0] is region
    assert call.calls == 0


def test_no_disputes_does_not_call_provider(tmp_path: Path) -> None:
    source = _page(tmp_path)
    resolved = _region("clear", "clear", "resolved", BoundingBox(5, 5, 25, 20))
    call = RecordingCall(_result([]))

    result = DisputeResolutionStage(call).apply(source, 1, [resolved])

    assert result is not None
    assert result[0] is resolved
    assert call.calls == 0


def _page(tmp_path: Path) -> Path:
    source = tmp_path / "page.png"
    Image.new("RGB", (100, 80), "white").save(source)
    return source


def _region(
    region_id: str,
    text: str,
    resolution: str,
    box: BoundingBox,
) -> TextRegion:
    return TextRegion(
        id=region_id,
        kind="word",
        text=text,
        confidence=0.6,
        bounding_box=box,
        reading_order=1,
        provider="reader-a",
        resolution=resolution,
    )


def _table_region() -> TextRegion:
    return TextRegion(
        id="table",
        kind="table",
        text="| value |",
        confidence=0.9,
        bounding_box=BoundingBox(10, 50, 100, 90),
        reading_order=3,
        provider="table-reader",
        structure={
            "role": "table",
            "cells": [
                {
                    "id": "cell-1",
                    "bbox": {"left": 15, "top": 55, "right": 45, "bottom": 75},
                    "text": "43",
                    "confidence": 0.7,
                    "source": "reader-a",
                    "resolution": "conflicting",
                    "alternatives": [
                        {
                            "text": "48",
                            "confidence": 0.8,
                            "provider": "reader-b",
                            "text_provenance": {"method": "fixture"},
                        }
                    ],
                }
            ],
        },
    )


def _result(
    decisions: list[dict[str, object]],
    *,
    model: str = "model",
    provider: str = "provider",
) -> SimpleNamespace:
    return SimpleNamespace(
        content={"decisions": decisions},
        model=model,
        provider=provider,
    )
