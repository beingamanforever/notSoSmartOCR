from __future__ import annotations

from contextlib import nullcontext
import io
import json
from pathlib import Path
import threading
from types import SimpleNamespace

from PIL import Image
import pytest
from fastapi.testclient import TestClient

from ocr_pipeline.contracts import BoundingBox, TextAlternative, TextRegion
from ocr_pipeline.demo import create_app
from ocr_pipeline.handwriting import HandwritingStage
from ocr_pipeline.pipeline import process_document
from ocr_pipeline.providers import (
    PHI4_ADAPTER_FORMAT,
    Phi4HandwritingReader,
    ReaderError,
    _force_phi4_sdpa,
)


class FixedReader:
    name = "incumbent"

    def __init__(self, regions: list[TextRegion]) -> None:
        self.regions = regions

    def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
        return self.regions


class CropReader:
    name = "phi4-handwriting"
    max_batch_items = 16
    provenance = {
        "id": "microsoft/Phi-4-multimodal-instruct",
        "revision": "pinned",
        "adapter": {"format": PHI4_ADAPTER_FORMAT, "validated": True},
    }

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.crop_sizes: list[tuple[int, int]] = []
        self.crop_modes: list[str] = []
        self.calls = 0

    def transcribe_batch(self, images: list[Image.Image]) -> list[str]:
        self.calls += 1
        self.crop_sizes = [image.size for image in images]
        self.crop_modes = [image.mode for image in images]
        return self.outputs


def test_stage_batches_only_explicit_bounded_handwriting_regions(
    tmp_path: Path,
) -> None:
    source = tmp_path / "page.png"
    Image.new("RGB", (100, 80), "white").save(source)
    regions = [
        _region("first", "first", 0.4, BoundingBox(10, 10, 30, 20), 1),
        _region(
            "weakest",
            "weakest",
            0.2,
            BoundingBox(60, 30, 80, 50),
            2,
            structure={"handwriting_candidate": True},
        ),
        _region(
            "table",
            "printed table",
            0.1,
            BoundingBox(40, 5, 80, 28),
            3,
            structure={"role": "table_source"},
        ),
        _region("strong", "strong", 0.95, BoundingBox(5, 60, 25, 70), 3),
    ]
    regions[1].alternatives.append(
        TextAlternative("corrected", 0.7, "tesseract", {"method": "fixture"})
    )
    crop_reader = CropReader(["corrected", "corrected"])
    stage = HandwritingStage(
        crop_reader,
        text_provider="incumbent",
        max_regions=1,
        context_padding=12,
    )

    result = process_document(source, FixedReader(regions), stages=[stage])

    page = result.pages[0]
    assert crop_reader.calls == 1
    assert crop_reader.crop_sizes == [(20, 20), (44, 44)]
    assert page.regions[0].text == "first"
    assert page.regions[2].text == "printed table"
    assert page.regions[3].text == "strong"
    replaced = page.regions[1]
    assert replaced.text == "corrected"
    assert replaced.provider == "phi4-handwriting"
    assert replaced.confidence == 0.7
    assert replaced.resolution == "resolved"
    assert [
        (item.text, item.confidence, item.provider) for item in replaced.alternatives
    ] == [("weakest", 0.2, "incumbent")]
    assert replaced.text_provenance == {
        "method": "dual_crop_exact_agreement",
        "page_number": 1,
        "crops": {
            "tight": {"bounding_box": [60, 30, 80, 50]},
            "context": {"bounding_box": [48, 18, 92, 62]},
        },
        "model": crop_reader.provenance,
        "decision": "independently_corroborated_replacement",
        "supporting_provider": "tesseract",
    }
    assert replaced.structure["handwriting_review"]["reason"] == (
        "corroborated_replacement"
    )
    assert page.route == "review"
    assert "corrected" in page.text.value


def test_crop_disagreement_keeps_incumbent_and_marks_conflict(tmp_path: Path) -> None:
    source = tmp_path / "page.png"
    Image.new("RGB", (60, 40), "white").save(source)
    crop_reader = CropReader(["tight result", "context result"])
    stage = HandwritingStage(crop_reader, text_provider="incumbent")

    result = process_document(
        source,
        FixedReader(
            [
                _region(
                    "field",
                    "incumbent",
                    0.1,
                    BoundingBox(5, 5, 30, 20),
                    1,
                    kind="handwriting",
                )
            ]
        ),
        stages=[stage],
    )

    region = result.pages[0].regions[0]
    assert region.text == "incumbent"
    assert region.provider == "incumbent"
    assert region.resolution == "resolved"
    assert [(item.text, item.provider) for item in region.alternatives] == [
        ("tight result", "phi4-handwriting:tight"),
        ("context result", "phi4-handwriting:context"),
    ]
    assert region.structure["handwriting_review"]["reason"] == "crop_disagreement"
    assert result.pages[0].route == "review"
    assert result.pages[0].text.value == "incumbent"


def test_agreed_specialist_candidate_preserves_incumbent_without_support(
    tmp_path: Path,
) -> None:
    source = tmp_path / "page.png"
    Image.new("RGB", (60, 40), "white").save(source)
    stage = HandwritingStage(
        CropReader(["candidate", "candidate"]),
        text_provider="incumbent",
    )

    result = process_document(
        source,
        FixedReader(
            [
                _region(
                    "field",
                    "incumbent",
                    0.1,
                    BoundingBox(5, 5, 30, 20),
                    1,
                    kind="handwriting",
                )
            ]
        ),
        stages=[stage],
    )

    region = result.pages[0].regions[0]
    assert region.text == "incumbent"
    assert region.provider == "incumbent"
    assert region.resolution == "resolved"
    assert [(item.text, item.provider) for item in region.alternatives] == [
        ("candidate", "phi4-handwriting")
    ]
    assert region.structure["handwriting_review"]["reason"] == ("specialist_candidate")
    assert result.pages[0].route == "review"
    assert result.pages[0].text.value == "incumbent"


@pytest.mark.parametrize(
    ("candidate", "reason"),
    [
        ("<NO_HANDWRITING>", "abstention_candidate"),
        ("", "empty_candidate"),
        ("x" * 40, "candidate_expanded_context"),
        ("bad\ntext", "candidate_control_characters"),
    ],
)
def test_literal_rejections_never_replace_or_enter_metadata(
    tmp_path: Path,
    candidate: str,
    reason: str,
) -> None:
    source = tmp_path / "page.png"
    Image.new("RGB", (60, 40), "white").save(source)
    stage = HandwritingStage(
        CropReader([candidate, candidate]),
        text_provider="incumbent",
    )

    result = process_document(
        source,
        FixedReader(
            [
                _region(
                    "field",
                    "weak",
                    0.1,
                    BoundingBox(5, 5, 30, 20),
                    1,
                    kind="handwriting",
                )
            ]
        ),
        stages=[stage],
    )

    region = result.pages[0].regions[0]
    assert region.text == "weak"
    assert region.provider == "incumbent"
    assert region.alternatives == []
    assert region.resolution == "resolved"
    assert region.structure["handwriting_review"]["reason"] == reason
    assert result.pages[0].text.value == "weak"
    if candidate:
        assert candidate not in json.dumps(region.structure)


def test_no_eligible_region_does_not_initialize_specialist(tmp_path: Path) -> None:
    source = tmp_path / "page.png"
    Image.new("RGB", (60, 40), "white").save(source)
    crop_reader = CropReader([])
    stage = HandwritingStage(crop_reader, text_provider="incumbent")

    result = process_document(
        source,
        FixedReader([_region("strong", "clear", 0.9, BoundingBox(5, 5, 30, 20), 1)]),
        stages=[stage],
    )

    assert crop_reader.calls == 0
    assert result.pages[0].route == "accept_local"


def test_tall_low_confidence_printed_text_is_not_routed(tmp_path: Path) -> None:
    source = tmp_path / "page.png"
    Image.new("RGB", (100, 100), "white").save(source)
    crop_reader = CropReader([])
    stage = HandwritingStage(crop_reader, text_provider="incumbent")

    result = process_document(
        source,
        FixedReader(
            [
                _region(
                    "printed",
                    "PRINTED TITLE",
                    0.1,
                    BoundingBox(5, 5, 95, 70),
                    1,
                    kind="text",
                )
            ]
        ),
        stages=[stage],
    )

    assert crop_reader.calls == 0
    assert result.pages[0].regions[0].text == "PRINTED TITLE"


def test_explicit_handwriting_candidate_is_routed_and_preserves_incumbent(
    tmp_path: Path,
) -> None:
    source = tmp_path / "page.png"
    Image.new("RGB", (100, 80), "white").save(source)
    crop_reader = CropReader(["specialist", "specialist"])
    stage = HandwritingStage(crop_reader, text_provider="incumbent")

    result = process_document(
        source,
        FixedReader(
            [
                _region(
                    "marked",
                    "incumbent",
                    0.2,
                    BoundingBox(10, 10, 60, 30),
                    1,
                    kind="text",
                    structure={"handwriting_candidate": True},
                )
            ]
        ),
        stages=[stage],
    )

    region = result.pages[0].regions[0]
    assert crop_reader.calls == 1
    assert region.text == "incumbent"
    assert region.provider == "incumbent"
    assert [(item.text, item.provider) for item in region.alternatives] == [
        ("specialist", "phi4-handwriting")
    ]
    assert region.structure["handwriting_review"]["reason"] == ("specialist_candidate")


def test_anchored_residual_proposal_is_reread_as_unresolved_evidence(
    tmp_path: Path,
) -> None:
    source = tmp_path / "page.png"
    Image.new("RGB", (100, 80), (240, 230, 220)).save(source)
    crop_reader = CropReader(["Take 5 mg every morning", "Take 5 mg every morning"])
    stage = HandwritingStage(crop_reader, text_provider="incumbent")
    proposal = _anchored_proposal(BoundingBox(20, 20, 60, 36))
    unrelated_unreadable = TextRegion(
        id="unrelated-unreadable",
        kind="handwriting",
        text="",
        confidence=None,
        bounding_box=BoundingBox(70, 20, 90, 36),
        reading_order=3,
        provider="other-proposal",
        resolution="unreadable",
        structure={"handwriting_candidate": True},
    )

    result = process_document(
        source,
        FixedReader(
            [
                _region("label", "Dose:", 0.98, BoundingBox(2, 20, 16, 36), 1),
                proposal,
                unrelated_unreadable,
            ]
        ),
        stages=[stage],
    )

    page = result.pages[0]
    assert crop_reader.calls == 1
    assert crop_reader.crop_sizes == [(40, 16), (64, 40)]
    assert crop_reader.crop_modes == ["RGB", "RGB"]
    assert page.route == "review"
    assert page.text.value == "Dose:"
    assert page.text.evidence_ids == ["label"]
    processed = next(region for region in page.regions if region.id == proposal.id)
    assert processed.text == ""
    assert processed.resolution == "unreadable"
    assert processed.provider == "opencv-anchored-ink"
    assert [(item.text, item.provider) for item in processed.alternatives] == [
        ("Take 5 mg every morning", "phi4-handwriting")
    ]
    alternative = processed.alternatives[0]
    assert alternative.text_provenance == {
        "method": "dual_crop_exact_agreement",
        "page_number": 1,
        "crops": {
            "tight": {"bounding_box": [20, 20, 60, 36]},
            "context": {"bounding_box": [8, 8, 72, 48]},
        },
        "model": crop_reader.provenance,
        "view": "agreed",
    }
    assert processed.structure["handwriting_review"] == {
        "required": True,
        "reason": "specialist_candidate",
        "provenance": {
            key: value
            for key, value in alternative.text_provenance.items()
            if key != "view"
        },
    }
    assert unrelated_unreadable.alternatives == []
    assert "handwriting_review" not in unrelated_unreadable.structure


@pytest.mark.parametrize(
    ("candidate", "reason"),
    [
        ("", "empty_candidate"),
        ("<UNREADABLE>", "abstention_candidate"),
        ("<|x|>", "candidate_control_tokens"),
    ],
)
def test_anchored_residual_rejects_nonliteral_specialist_outputs(
    tmp_path: Path,
    candidate: str,
    reason: str,
) -> None:
    source = tmp_path / "page.png"
    Image.new("RGB", (80, 60), "white").save(source)
    crop_reader = CropReader([candidate, candidate])
    proposal = _anchored_proposal(BoundingBox(20, 20, 50, 35))

    result = process_document(
        source,
        FixedReader([proposal]),
        stages=[HandwritingStage(crop_reader, text_provider="incumbent")],
    )

    assert crop_reader.calls == 1
    assert len(crop_reader.crop_sizes) == 2
    page = result.pages[0]
    processed = page.regions[0]
    assert processed.text == ""
    assert processed.resolution == "unreadable"
    assert processed.alternatives == []
    assert processed.structure["handwriting_review"]["reason"] == reason
    assert page.route == "review"
    assert page.text.value == ""


def test_anchored_residual_crop_disagreement_keeps_only_valid_alternatives(
    tmp_path: Path,
) -> None:
    source = tmp_path / "page.png"
    Image.new("RGB", (80, 60), "white").save(source)
    crop_reader = CropReader(["5 mg", "<|x|>"])

    result = process_document(
        source,
        FixedReader([_anchored_proposal(BoundingBox(20, 20, 50, 35))]),
        stages=[HandwritingStage(crop_reader, text_provider="incumbent")],
    )

    page = result.pages[0]
    proposal = page.regions[0]
    assert crop_reader.calls == 1
    assert len(crop_reader.crop_sizes) == 2
    assert proposal.text == ""
    assert proposal.resolution == "unreadable"
    assert [(item.text, item.provider) for item in proposal.alternatives] == [
        ("5 mg", "phi4-handwriting:tight")
    ]
    assert proposal.alternatives[0].text_provenance["view"] == "tight"
    assert proposal.structure["handwriting_review"]["reason"] == ("crop_disagreement")
    assert page.route == "review"
    assert page.text.value == ""


def test_manual_review_routes_high_confidence_text_inside_a_form_grid(
    tmp_path: Path,
) -> None:
    source = tmp_path / "page.png"
    Image.new("RGB", (120, 80), "white").save(source)
    crop_reader = CropReader(["handwritten", "handwritten"])
    stage = HandwritingStage(crop_reader, text_provider="incumbent")
    region = _region(
        "form-value",
        "incumbent",
        0.98,
        BoundingBox(20, 20, 90, 42),
        1,
        kind="text",
        structure={"role": "table_source", "parent_id": "form-grid"},
    )

    reviewed = stage.review_region(source, 1, region)

    assert crop_reader.calls == 1
    assert reviewed.text == "incumbent"
    assert [(item.text, item.provider) for item in reviewed.alternatives] == [
        ("handwritten", "phi4-handwriting")
    ]
    assert reviewed.structure["role"] == "table_source"
    assert reviewed.structure["handwriting_candidate_source"] == "manual"
    assert reviewed.structure["handwriting_review"]["reason"] == (
        "specialist_candidate"
    )


def test_demo_rereads_an_explicit_form_region_end_to_end() -> None:
    class FormReader:
        name = "incumbent"
        version = "test-1"

        def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
            return [
                _region(
                    "form-value",
                    "incumbent",
                    0.98,
                    BoundingBox(20, 20, 90, 42),
                    1,
                    kind="text",
                    structure={"role": "table_source", "parent_id": "form-grid"},
                )
            ]

    image = io.BytesIO()
    Image.new("RGB", (120, 80), "white").save(image, format="PNG")
    crop_reader = CropReader(["handwritten", "handwritten"])
    app = create_app(
        FormReader(),
        handwriting_stage=HandwritingStage(crop_reader, text_provider="incumbent"),
    )

    with TestClient(app) as client:
        processed = client.post(
            "/api/process",
            files={"file": ("form.png", image.getvalue(), "image/png")},
        )
        assert processed.status_code == 200
        assert crop_reader.calls == 0
        session_id = processed.json()["session_id"]

        reread = client.post(
            f"/api/sessions/{session_id}/handwriting",
            json={"page_number": 1, "region_id": "form-value"},
        )

        assert reread.status_code == 200
        assert crop_reader.calls == 1
        payload = reread.json()
        assert payload["pipeline_stages"] == []
        page = payload["result"]["pages"][0]
        region = page["regions"][0]
        assert page["route"] == "review"
        assert region["text"] == "incumbent"
        assert region["structure"]["role"] == "table_source"
        assert region["structure"]["handwriting_candidate_source"] == "manual"
        assert [
            (item["text"], item["provider"]) for item in region["alternatives"]
        ] == [("handwritten", "phi4-handwriting")]
        exported = client.get(f"/api/sessions/{session_id}/result.json")
        assert exported.json() == payload["result"]


def test_reader_loads_only_a_provenanced_matching_adapter(tmp_path: Path) -> None:
    adapter = tmp_path / "vision_decoder_lora.pt"
    adapter.write_bytes(b"fixture")
    parameter_name = "model.layers.0.qkv_proj.lora_A.vision.weight"
    payload = {
        "format": PHI4_ADAPTER_FORMAT,
        "state": {parameter_name: object()},
        "provenance": {
            "training_family_count": 1,
            "training_family_ids": ["private-train-family"],
        },
    }
    model = FakePhiModel(parameter_name)
    torch_module = SimpleNamespace(
        load=lambda *args, **kwargs: payload,
        inference_mode=nullcontext,
    )
    processor = FakePhiProcessor()
    reader = Phi4HandwritingReader(
        adapter,
        processor=processor,
        model=model,
        generation_config=object(),
        torch_module=torch_module,
    )

    images = [Image.new("RGB", (20, 10), "white") for _ in range(3)]
    output = reader.transcribe_batch(images)

    assert output == ["literal", "literal", "literal"]
    assert processor.batch_sizes == [2, 1]
    assert model.adapter == "vision"
    assert model.loaded_state == payload["state"]
    assert reader.provenance["adapter"] == {
        "format": PHI4_ADAPTER_FORMAT,
        "source": adapter.name,
        "validated": True,
        "training_family_count": 1,
    }
    assert str(tmp_path) not in json.dumps(reader.provenance)


def test_reader_rejects_invalid_adapter_without_exposing_payload(
    tmp_path: Path,
) -> None:
    adapter = tmp_path / "vision_decoder_lora.pt"
    adapter.write_bytes(b"PRIVATE_LITERAL")
    parameter_name = "model.layers.0.qkv_proj.lora_A.vision.weight"
    private_value = "PRIVATE_LITERAL"
    payload = {
        "format": PHI4_ADAPTER_FORMAT,
        "state": {parameter_name: object()},
        "provenance": {"private": private_value},
    }
    reader = Phi4HandwritingReader(
        adapter,
        processor=FakePhiProcessor(),
        model=FakePhiModel(parameter_name),
        torch_module=SimpleNamespace(
            load=lambda *args, **kwargs: payload,
            inference_mode=nullcontext,
        ),
    )

    with pytest.raises(ReaderError) as raised:
        reader.transcribe_batch([Image.new("RGB", (20, 10), "white")])

    assert raised.value.code == "phi4_handwriting_adapter_failed"
    assert private_value not in str(raised.value)


def test_local_model_directory_is_not_reported_as_official(tmp_path: Path) -> None:
    adapter = tmp_path / "vision_decoder_lora.pt"
    adapter.write_bytes(b"fixture")
    model_dir = tmp_path / "local-model"
    model_dir.mkdir()

    reader = Phi4HandwritingReader(adapter, model_name_or_path=model_dir)

    assert reader.provenance["id"] == "local-model"
    assert reader.provenance["source"] == "local_directory"
    assert reader.provenance["revision"] == "unverified"
    assert reader.provenance["origin"] == "unverified"
    assert reader.provenance["license"] == "unverified"
    assert reader.provenance["identity_verified"] is False
    assert reader.provenance["adapter"] == {
        "format": PHI4_ADAPTER_FORMAT,
        "source": adapter.name,
        "validated": False,
    }
    assert str(tmp_path) not in json.dumps(reader.provenance)


@pytest.mark.parametrize("raise_inside", [False, True])
def test_phi4_sdpa_override_is_scoped(raise_inside: bool) -> None:
    def first() -> bool:
        return True

    def second() -> bool:
        return True

    transformers_utils = SimpleNamespace(
        is_flash_attn_2_available=first,
        is_flash_attn_greater_or_equal_2_10=second,
    )
    import_utils = SimpleNamespace(
        is_flash_attn_2_available=first,
        is_flash_attn_greater_or_equal_2_10=second,
    )

    def run() -> None:
        with _force_phi4_sdpa(transformers_utils, import_utils):
            assert transformers_utils.is_flash_attn_2_available() is False
            assert import_utils.is_flash_attn_greater_or_equal_2_10() is False
            if raise_inside:
                raise RuntimeError("fixture")

    if raise_inside:
        with pytest.raises(RuntimeError, match="fixture"):
            run()
    else:
        run()

    assert transformers_utils.is_flash_attn_2_available is first
    assert transformers_utils.is_flash_attn_greater_or_equal_2_10 is second
    assert import_utils.is_flash_attn_2_available is first
    assert import_utils.is_flash_attn_greater_or_equal_2_10 is second


@pytest.mark.parametrize("first_raises", [False, True])
def test_phi4_sdpa_override_serializes_concurrent_contexts(
    first_raises: bool,
) -> None:
    def first() -> bool:
        return True

    def second() -> bool:
        return True

    transformers_utils = SimpleNamespace(
        is_flash_attn_2_available=first,
        is_flash_attn_greater_or_equal_2_10=second,
    )
    import_utils = SimpleNamespace(
        is_flash_attn_2_available=first,
        is_flash_attn_greater_or_equal_2_10=second,
    )
    first_entered = threading.Event()
    release_first = threading.Event()
    second_attempted = threading.Event()
    second_entered = threading.Event()
    release_second = threading.Event()

    def run_first() -> None:
        try:
            with _force_phi4_sdpa(transformers_utils, import_utils):
                first_entered.set()
                assert release_first.wait(timeout=1)
                if first_raises:
                    raise RuntimeError("fixture")
        except RuntimeError:
            assert first_raises

    def run_second() -> None:
        assert first_entered.wait(timeout=1)
        second_attempted.set()
        with _force_phi4_sdpa(transformers_utils, import_utils):
            second_entered.set()
            assert release_second.wait(timeout=1)

    first_thread = threading.Thread(target=run_first)
    second_thread = threading.Thread(target=run_second)
    first_thread.start()
    assert first_entered.wait(timeout=1)
    second_thread.start()
    assert second_attempted.wait(timeout=1)
    assert not second_entered.wait(timeout=0.05)

    release_first.set()
    assert second_entered.wait(timeout=1)
    assert transformers_utils.is_flash_attn_2_available() is False
    assert import_utils.is_flash_attn_greater_or_equal_2_10() is False
    release_second.set()
    first_thread.join(timeout=1)
    second_thread.join(timeout=1)

    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert transformers_utils.is_flash_attn_2_available is first
    assert transformers_utils.is_flash_attn_greater_or_equal_2_10 is second
    assert import_utils.is_flash_attn_2_available is first
    assert import_utils.is_flash_attn_greater_or_equal_2_10 is second


class FakePhiModel:
    def __init__(self, parameter_name: str) -> None:
        self.parameter_name = parameter_name
        self.adapter: str | None = None
        self.loaded_state: object | None = None

    def set_lora_adapter(self, adapter: str) -> None:
        self.adapter = adapter

    def named_parameters(self) -> list[tuple[str, object]]:
        return [(self.parameter_name, object())]

    def load_state_dict(self, state: object, *, strict: bool) -> object:
        assert strict is False
        self.loaded_state = state
        return SimpleNamespace(unexpected_keys=[])

    def eval(self) -> None:
        return None

    def generate(self, **kwargs: object) -> list[list[int]]:
        return [[10, 11, 12] for _ in kwargs["input_ids"]]


class FakePhiProcessor:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def __call__(self, **kwargs: object) -> FakeInputs:
        assert len(kwargs["text"]) == len(kwargs["images"])
        self.batch_sizes.append(len(kwargs["images"]))
        assert kwargs["padding"] is True
        assert kwargs["return_tensors"] == "pt"
        return FakeInputs(input_ids=FakeBatchIds([0] * len(kwargs["images"])))

    def decode(self, tokens: list[int], **kwargs: object) -> str:
        assert tokens == [12]
        return "literal"


class FakeInputs(dict):
    def to(self, device: str) -> FakeInputs:
        assert device == "cuda:0"
        return self


class FakeBatchIds(list[int]):
    @property
    def shape(self) -> tuple[int, int]:
        return len(self), 2


def _region(
    id: str,
    text: str,
    confidence: float,
    box: BoundingBox,
    order: int,
    *,
    kind: str = "word",
    structure: dict[str, object] | None = None,
) -> TextRegion:
    return TextRegion(
        id=id,
        kind=kind,
        text=text,
        confidence=confidence,
        bounding_box=box,
        reading_order=order,
        provider="incumbent",
        text_provenance={"method": "fixture"},
        structure=structure,
    )


def _anchored_proposal(box: BoundingBox) -> TextRegion:
    return TextRegion(
        id="anchored-proposal",
        kind="handwriting",
        text="",
        confidence=None,
        bounding_box=box,
        reading_order=2,
        provider="opencv-anchored-ink",
        text_provenance={"method": "fixture"},
        resolution="unreadable",
        structure={
            "role": "handwriting_candidate",
            "handwriting_candidate": True,
            "handwriting_candidate_source": "anchored_residual",
        },
    )
