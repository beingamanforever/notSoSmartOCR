from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from experiments.serve_gpu_demo import create_verified_app
from ocr_pipeline.falcon_presentation import FalconFormulaStage
from ocr_pipeline.handwriting import (
    TROCR_MODEL_ID,
    TROCR_MODEL_LICENSE,
    TROCR_MODEL_ORIGIN,
    TROCR_MODEL_REVISION,
    HandwritingStage,
)
from ocr_pipeline.orientation import DocTROrientationDetector, OrientationReader
from ocr_pipeline.preprocessing import (
    PageFrameReader,
    TiledReader,
    WideBandFallbackReader,
)
from ocr_pipeline.providers import ReaderError


def test_verified_gpu_app_wires_word_reader_and_table_specialists() -> None:
    native_calls: list[dict[str, object]] = []
    app_calls: list[tuple[object, tuple[object, ...], object | None, object]] = []

    def native_factory(**options: object) -> object:
        native_calls.append(options)
        return object()

    def app_factory(
        reader: object,
        *,
        stages: tuple[object, ...],
        handwriting_stage: object | None,
        composition: object,
    ) -> object:
        app_calls.append((reader, stages, handwriting_stage, composition))
        return object()

    result = create_verified_app(
        Namespace(
            nemotron_model_dir=Path("/models/nemotron"),
            tatr_source=Path("/models/tatr"),
            tatr_detection_model=Path("/models/detection.pth"),
            tatr_structure_model=Path("/models/structure.pth"),
            tesseract_executable=Path("/tools/tesseract"),
            device="cuda",
        ),
        native_factory=native_factory,
        app_factory=app_factory,
    )

    assert result is not None
    assert native_calls == [{"model_dir": "/models/nemotron", "lang": "multi"}]
    reader, stages, handwriting_stage, composition = app_calls[0]
    assert handwriting_stage is None
    assert isinstance(reader, PageFrameReader)
    orientation = reader.reader
    assert isinstance(orientation, OrientationReader)
    assert orientation.defer_restore is True
    assert isinstance(orientation.orientation_detector, DocTROrientationDetector)
    assert orientation.margin_reader.page_segmentation_mode == 3
    assert orientation.margin_reader.thresholding_method is None
    assert isinstance(orientation.reader, WideBandFallbackReader)
    assert isinstance(orientation.reader.reader, TiledReader)
    assert orientation.reader.reader.reader.merge_level == "word"
    assert orientation.reader.reader.reader.batch_size == 1
    assert orientation.reader.fallback_reader.page_segmentation_mode == 6
    assert orientation.reader.fallback_reader.thresholding_method is None
    assert orientation.reader.confirmation_reader.page_segmentation_mode == 6
    assert orientation.reader.confirmation_reader.thresholding_method == 2
    assert [stage.name for stage in stages] == [
        "tables",
        "controls",
        "faint-tiny-text",
        "evidence-risk",
        "evidence-layout",
        "digit-verification",
        "anchored-ink",
    ]
    stage = stages[0]
    assert stage.name == "tables"
    assert stage.extractor.enable_ruled_table_proposals is True
    assert [challenger.name for challenger in stage.challengers] == [
        "tesseract_raw",
        "sauvola",
        "tesseract_cell",
    ]
    assert stage.parallel_challengers is True
    assert stage.low_primary_confidence == 0.88
    assert [challenger.scope for challenger in stage.challengers] == [
        "table",
        "table",
        "cell",
    ]
    assert stage.challengers[0].reader is not orientation.reader.fallback_reader
    assert stage.challengers[0].reader.page_segmentation_mode == 4
    assert stage.challengers[0].reader.thresholding_method is None
    assert stage.challengers[1].reader.page_segmentation_mode == 4
    assert stage.challengers[1].reader.thresholding_method == 2
    assert stage.challengers[2].reader.page_segmentation_mode == 7
    assert stage.challengers[2].reader.thresholding_method is None
    assert stages[1].label_provider == orientation.reader.reader.reader.name
    assert stages[2].reader.name == "nemotron-ocr-v2"
    assert stages[2].reader.merge_level == "word"
    assert stages[2].reader.batch_size == 8
    assert composition.label == "Full GPU pipeline"
    assert composition.primary_ocr == (
        "NVIDIA Nemotron OCR v2 (word merge, batch size 1)"
    )
    assert composition.orientation == (
        "Mindee docTR proposals with Tesseract OSD evidence check"
    )
    assert composition.tesseract_roles == (
        "orientation OSD",
        "side-margin text",
        "selective wide-band fallback and confirmation",
        "table challengers (raw, Sauvola, and selective cell reread)",
    )
    assert composition.stages == (
        "tables",
        "controls",
        "faint-tiny-text",
        "evidence-risk",
        "evidence-layout",
        "digit-verification",
        "anchored-ink",
    )
    assert composition.handwriting == "not configured"
    assert "ruled-form proposals" in composition.note


def test_verified_gpu_app_warms_the_complete_pipeline_before_serving() -> None:
    warmup_image = Path("/examples/academic.png")
    warmup_calls: list[tuple[Path, object, tuple[object, ...]]] = []
    app_calls: list[dict[str, object]] = []

    def handle_warmup(
        image_path: Path,
        reader: object,
        *,
        stages: tuple[object, ...],
    ) -> object:
        warmup_calls.append((image_path, reader, stages))
        return Namespace(status="success", pages=[object()])

    create_verified_app(
        Namespace(
            nemotron_model_dir=Path("/models/nemotron"),
            tatr_source=Path("/models/tatr"),
            tatr_detection_model=Path("/models/detection.pth"),
            tatr_structure_model=Path("/models/structure.pth"),
            tesseract_executable=Path("/tools/tesseract"),
            warmup_image=warmup_image,
            device="cuda",
        ),
        native_factory=lambda **options: object(),
        warmup_runner=handle_warmup,
        app_factory=lambda reader, **options: app_calls.append(options),
    )

    assert len(warmup_calls) == 2
    assert warmup_calls[0] == warmup_calls[1]
    image_path, reader, stages = warmup_calls[0]
    assert image_path == warmup_image
    assert isinstance(reader, PageFrameReader)
    assert [stage.name for stage in stages] == [
        "tables",
        "controls",
        "faint-tiny-text",
        "evidence-risk",
        "evidence-layout",
        "digit-verification",
        "anchored-ink",
    ]
    assert app_calls[0]["warmup_completed"] is True


def test_verified_gpu_app_rejects_failed_warmup() -> None:
    with pytest.raises(RuntimeError, match="OCR warmup"):
        create_verified_app(
            Namespace(
                nemotron_model_dir=Path("/models/nemotron"),
                tatr_source=Path("/models/tatr"),
                tatr_detection_model=Path("/models/detection.pth"),
                tatr_structure_model=Path("/models/structure.pth"),
                tesseract_executable=Path("/tools/tesseract"),
                warmup_image=Path("/examples/failed.png"),
                device="cuda",
            ),
            native_factory=lambda **options: object(),
            warmup_runner=lambda *args, **options: Namespace(
                status="failed",
                pages=[],
            ),
            app_factory=lambda reader, **options: object(),
        )


def test_verified_gpu_app_rejects_failed_presentation_warmup() -> None:
    class FakePresentationReader:
        name = "ministral-ocr"

    with pytest.raises(RuntimeError, match="Presentation warmup"):
        create_verified_app(
            Namespace(
                nemotron_model_dir=Path("/models/nemotron"),
                tatr_source=Path("/models/tatr"),
                tatr_detection_model=Path("/models/detection.pth"),
                tatr_structure_model=Path("/models/structure.pth"),
                tesseract_executable=Path("/tools/tesseract"),
                ministral_presentation_model=Path("/models/ministral"),
                warmup_image=Path("/examples/review.png"),
                device="cuda",
            ),
            native_factory=lambda **options: object(),
            presentation_reader_factory=lambda **options: FakePresentationReader(),
            warmup_runner=lambda *args, **options: Namespace(
                status="success",
                pages=[Namespace(page_number=1, route="review")],
            ),
            presentation_warmup_runner=lambda *args, **options: {
                1: {"status": "failed"}
            },
            app_factory=lambda reader, **options: object(),
        )


def test_verified_gpu_app_wires_local_ministral_as_review_only_presentation() -> None:
    presentation_calls: list[dict[str, object]] = []
    app_calls: list[tuple[object, dict[str, object]]] = []

    class FakePresentationReader:
        name = "ministral-ocr"

    def presentation_factory(**options: object) -> object:
        presentation_calls.append(options)
        return FakePresentationReader()

    def app_factory(reader: object, **options: object) -> object:
        app_calls.append((reader, options))
        return object()

    result = create_verified_app(
        Namespace(
            nemotron_model_dir=Path("/models/nemotron"),
            tatr_source=Path("/models/tatr"),
            tatr_detection_model=Path("/models/detection.pth"),
            tatr_structure_model=Path("/models/structure.pth"),
            tesseract_executable=Path("/tools/tesseract"),
            ministral_presentation_model=Path("/models/ministral"),
            ministral_model_revision="b" * 40,
            ministral_max_new_tokens=4096,
            ministral_max_pages=3,
            device="cuda",
        ),
        native_factory=lambda **options: object(),
        presentation_reader_factory=presentation_factory,
        app_factory=app_factory,
    )

    assert result is not None
    assert presentation_calls == [
        {
            "model_name": "/models/ministral",
            "model_revision": "b" * 40,
            "max_new_tokens": 4096,
        }
    ]
    options = app_calls[0][1]
    assert isinstance(options["presentation_reader"], FakePresentationReader)
    assert options["max_presentation_pages"] == 3
    assert options["composition"].stages == (
        "tables",
        "controls",
        "faint-tiny-text",
        "dispute_resolution",
        "evidence-risk",
        "evidence-layout",
        "digit-verification",
        "anchored-ink",
        "page-presentation",
    )
    assert "never mutates canonical evidence" in options["composition"].note


def test_verified_gpu_app_uses_warm_ministral_presentation_service() -> None:
    service_calls: list[tuple[str, dict[str, object]]] = []
    app_calls: list[dict[str, object]] = []

    class FakePresentationReader:
        name = "ministral-ocr"

    def service_factory(url: str, **options: object) -> object:
        service_calls.append((url, options))
        return FakePresentationReader()

    def app_factory(reader: object, **options: object) -> object:
        app_calls.append(options)
        return object()

    result = create_verified_app(
        Namespace(
            nemotron_model_dir=Path("/models/nemotron"),
            tatr_source=Path("/models/tatr"),
            tatr_detection_model=Path("/models/detection.pth"),
            tatr_structure_model=Path("/models/structure.pth"),
            tesseract_executable=Path("/tools/tesseract"),
            ministral_presentation_url="http://127.0.0.1:8084",
            ministral_timeout_seconds=75,
            ministral_max_pages=2,
            device="cuda",
        ),
        native_factory=lambda **options: object(),
        presentation_service_reader_factory=service_factory,
        app_factory=app_factory,
    )

    assert result is not None
    assert service_calls == [("http://127.0.0.1:8084", {"timeout_seconds": 75})]
    options = app_calls[0]
    assert isinstance(options["presentation_reader"], FakePresentationReader)
    assert options["max_presentation_pages"] == 2
    assert options["composition"].stages == (
        "tables",
        "controls",
        "faint-tiny-text",
        "dispute_resolution",
        "evidence-risk",
        "evidence-layout",
        "digit-verification",
        "anchored-ink",
        "page-presentation",
    )


def test_verified_gpu_app_uses_falcon_category_crops_without_dispute_stage() -> None:
    service_calls: list[tuple[str, dict[str, object]]] = []
    presentation_calls: list[tuple[object, dict[str, object]]] = []
    app_calls: list[dict[str, object]] = []

    class FakeFalconReader:
        name = "falcon-ocr"

        def check_health(self) -> None:
            service_calls.append(("health", {}))

    class FakeFalconPresentation:
        name = "falcon-presentation"

    def service_factory(url: str, **options: object) -> object:
        service_calls.append((url, options))
        return FakeFalconReader()

    def presentation_factory(reader: object, **options: object) -> object:
        presentation_calls.append((reader, options))
        return FakeFalconPresentation()

    result = create_verified_app(
        Namespace(
            nemotron_model_dir=Path("/models/nemotron"),
            tatr_source=Path("/models/tatr"),
            tatr_detection_model=Path("/models/detection.pth"),
            tatr_structure_model=Path("/models/structure.pth"),
            tesseract_executable=Path("/tools/tesseract"),
            falcon_presentation_url="http://127.0.0.1:8085",
            falcon_timeout_seconds=80,
            falcon_max_crops=12,
            falcon_max_pages=2,
            katex_asset_root=Path("/assets/katex"),
            device="cuda",
        ),
        native_factory=lambda **options: object(),
        falcon_service_reader_factory=service_factory,
        falcon_presentation_factory=presentation_factory,
        app_factory=lambda reader, **options: app_calls.append(options),
    )

    assert result is None
    assert service_calls == [
        (
            "http://127.0.0.1:8085",
            {"timeout_seconds": 80, "max_batch_items": 12},
        ),
        ("health", {}),
    ]
    assert len(presentation_calls) == 1
    assert presentation_calls[0][1] == {"max_crops": 12, "formula_padding": 12}
    assert isinstance(presentation_calls[0][0], FakeFalconReader)
    options = app_calls[0]
    assert isinstance(options["presentation_reader"], FakeFalconPresentation)
    assert options["max_presentation_pages"] == 2
    assert options["katex_asset_root"] == Path("/assets/katex")
    assert options["composition"].stages == (
        "tables",
        "controls",
        "faint-tiny-text",
        "evidence-risk",
        "evidence-layout",
        "falcon-formula",
        "digit-verification",
        "anchored-ink",
        "page-presentation",
    )
    formula_stage = next(
        stage for stage in options["stages"] if stage.name == "falcon-formula"
    )
    assert isinstance(formula_stage, FalconFormulaStage)
    assert formula_stage.reader is options["presentation_reader"]
    assert "source-ink formula crops" in options["composition"].note
    assert "human acceptance" in options["composition"].note
    assert options["composition"].build_label == "2026-09-06-form-evidence-v4"
    assert "review evidence" in options["composition"].note


def test_verified_gpu_app_configures_phi4_field_candidate_rereading() -> None:
    app_calls: list[tuple[object, tuple[object, ...], object | None, object]] = []
    handwriting_calls: list[tuple[Path, dict[str, object]]] = []

    def app_factory(
        reader: object,
        *,
        stages: tuple[object, ...],
        handwriting_stage: object | None,
        composition: object,
    ) -> object:
        app_calls.append((reader, stages, handwriting_stage, composition))
        return object()

    class FakeHandwritingReader:
        name = "phi4-handwriting"
        provenance: dict[str, object] = {}

        def __init__(self, max_batch_items: int) -> None:
            self.max_batch_items = max_batch_items

    def handwriting_factory(
        adapter_path: Path,
        **options: object,
    ) -> FakeHandwritingReader:
        handwriting_calls.append((adapter_path, options))
        return FakeHandwritingReader(int(options["max_batch_items"]))

    create_verified_app(
        Namespace(
            nemotron_model_dir=Path("/models/nemotron"),
            tatr_source=Path("/models/tatr"),
            tatr_detection_model=Path("/models/detection.pth"),
            tatr_structure_model=Path("/models/structure.pth"),
            tesseract_executable=Path("/tools/tesseract"),
            phi4_handwriting_adapter=Path("/models/handwriting.pt"),
            phi4_model="/models/phi4",
            phi4_model_revision="local-revision",
            phi4_max_regions=3,
            phi4_max_new_tokens=64,
            phi4_context_padding=18,
            device="cuda",
        ),
        native_factory=lambda **options: object(),
        handwriting_reader_factory=handwriting_factory,
        app_factory=app_factory,
    )

    assert handwriting_calls == [
        (
            Path("/models/handwriting.pt"),
            {
                "model_name_or_path": "/models/phi4",
                "model_revision": "local-revision",
                "device": "cuda",
                "max_new_tokens": 64,
                "max_batch_items": 6,
                "batch_size": 2,
                "local_files_only": True,
            },
        )
    ]
    _, stages, handwriting, composition = app_calls[0]
    assert [stage.name for stage in stages] == [
        "tables",
        "controls",
        "faint-tiny-text",
        "evidence-risk",
        "evidence-layout",
        "digit-verification",
        "anchored-ink",
        "handwriting-lines",
        "handwriting",
    ]
    assert isinstance(handwriting, HandwritingStage)
    assert handwriting.max_regions == 3
    assert handwriting.context_padding == 18
    assert composition.stages == (
        "tables",
        "controls",
        "faint-tiny-text",
        "evidence-risk",
        "evidence-layout",
        "digit-verification",
        "anchored-ink",
        "handwriting-lines",
        "handwriting",
    )
    assert composition.handwriting == "configured"


def test_verified_gpu_app_uses_warm_phi4_service_for_candidate_rereading() -> None:
    app_calls: list[tuple[object, tuple[object, ...], object | None, object]] = []
    service_calls: list[tuple[str, dict[str, object]]] = []

    class FakeHandwritingReader:
        name = "phi4-handwriting"
        provenance: dict[str, object] = {}

        def __init__(self, max_batch_items: int) -> None:
            self.max_batch_items = max_batch_items

    def service_factory(
        service_url: str,
        **options: object,
    ) -> FakeHandwritingReader:
        service_calls.append((service_url, options))
        return FakeHandwritingReader(int(options["max_batch_items"]))

    create_verified_app(
        Namespace(
            nemotron_model_dir=Path("/models/nemotron"),
            tatr_source=Path("/models/tatr"),
            tatr_detection_model=Path("/models/detection.pth"),
            tatr_structure_model=Path("/models/structure.pth"),
            tesseract_executable=Path("/tools/tesseract"),
            phi4_handwriting_url="http://127.0.0.1:8083",
            phi4_max_regions=3,
            phi4_context_padding=18,
            phi4_timeout_seconds=90,
            device="cuda",
        ),
        native_factory=lambda **options: object(),
        handwriting_service_reader_factory=service_factory,
        app_factory=lambda reader, *, stages, handwriting_stage, composition: (
            app_calls.append((reader, stages, handwriting_stage, composition))
        ),
    )

    assert service_calls == [
        (
            "http://127.0.0.1:8083",
            {"max_batch_items": 6, "timeout_seconds": 90},
        )
    ]
    stages = app_calls[0][1]
    assert [stage.name for stage in stages] == [
        "tables",
        "controls",
        "faint-tiny-text",
        "evidence-risk",
        "evidence-layout",
        "digit-verification",
        "anchored-ink",
        "handwriting-lines",
        "handwriting",
    ]
    handwriting = app_calls[0][2]
    assert isinstance(handwriting, HandwritingStage)
    assert handwriting.context_padding == 18
    assert app_calls[0][3].handwriting == "configured"


def test_verified_gpu_app_wires_cached_trocr_for_field_candidate_rereading() -> None:
    app_calls: list[dict[str, object]] = []
    reader_calls: list[dict[str, object]] = []

    class FakeHandwritingReader:
        name = "trocr-handwriting"
        provenance: dict[str, object] = {
            "id": TROCR_MODEL_ID,
            "source": TROCR_MODEL_ID,
            "revision": TROCR_MODEL_REVISION,
            "origin": TROCR_MODEL_ORIGIN,
            "license": TROCR_MODEL_LICENSE,
            "identity_verified": True,
            "local_files_only": True,
        }

        def __init__(self, max_batch_items: int) -> None:
            self.max_batch_items = max_batch_items

        def check_health(self) -> None:
            reader_calls.append({"health": True})

    def reader_factory(**options: object) -> FakeHandwritingReader:
        reader_calls.append(options)
        return FakeHandwritingReader(int(options["max_batch_items"]))

    create_verified_app(
        Namespace(
            nemotron_model_dir=Path("/models/nemotron"),
            tatr_source=Path("/models/tatr"),
            tatr_detection_model=Path("/models/detection.pth"),
            tatr_structure_model=Path("/models/structure.pth"),
            tesseract_executable=Path("/tools/tesseract"),
            trocr_handwriting_model=TROCR_MODEL_ID,
            trocr_model_revision=TROCR_MODEL_REVISION,
            trocr_max_regions=3,
            trocr_max_new_tokens=64,
            trocr_batch_size=4,
            trocr_context_padding=18,
            device="cuda",
        ),
        native_factory=lambda **options: object(),
        trocr_reader_factory=reader_factory,
        app_factory=lambda reader, **options: app_calls.append(options),
    )

    assert reader_calls == [
        {
            "model_name_or_path": TROCR_MODEL_ID,
            "model_revision": TROCR_MODEL_REVISION,
            "device": "cuda",
            "max_new_tokens": 64,
            "max_batch_items": 6,
            "batch_size": 4,
        },
        {"health": True},
    ]
    options = app_calls[0]
    assert isinstance(options["handwriting_stage"], HandwritingStage)
    assert options["handwriting_stage"] is options["stages"][-1]
    assert [stage.name for stage in options["stages"][-4:]] == [
        "digit-verification",
        "anchored-ink",
        "handwriting-lines",
        "handwriting",
    ]
    assert options["handwriting_stage"].context_padding == 18
    assert options["composition"].handwriting == "configured"


@pytest.mark.parametrize(
    ("model", "provenance"),
    [
        (
            TROCR_MODEL_ID,
            {
                "id": TROCR_MODEL_ID,
                "source": TROCR_MODEL_ID,
                "revision": "main",
                "origin": TROCR_MODEL_ORIGIN,
                "license": TROCR_MODEL_LICENSE,
                "identity_verified": False,
                "local_files_only": True,
            },
        ),
        (
            Path("/models/local-trocr"),
            {
                "id": "local-trocr",
                "source": "local_directory",
                "revision": "unverified",
                "origin": "unverified",
                "license": "unverified",
                "identity_verified": False,
                "local_files_only": True,
            },
        ),
        (
            TROCR_MODEL_ID,
            {
                "id": TROCR_MODEL_ID,
                "source": TROCR_MODEL_ID,
                "revision": TROCR_MODEL_REVISION,
                "origin": "unverified",
                "license": TROCR_MODEL_LICENSE,
                "identity_verified": True,
                "local_files_only": True,
            },
        ),
    ],
)
def test_verified_gpu_app_rejects_unverified_trocr_provenance(
    model: str | Path,
    provenance: dict[str, object],
) -> None:
    health_calls: list[bool] = []

    class UnverifiedReader:
        name = "trocr-handwriting"
        max_batch_items = 2

        def __init__(self) -> None:
            self.provenance = provenance

        def check_health(self) -> None:
            health_calls.append(True)

    with pytest.raises(ReaderError) as raised:
        create_verified_app(
            Namespace(
                nemotron_model_dir=Path("/models/nemotron"),
                tatr_source=Path("/models/tatr"),
                tatr_detection_model=Path("/models/detection.pth"),
                tatr_structure_model=Path("/models/structure.pth"),
                tesseract_executable=Path("/tools/tesseract"),
                trocr_handwriting_model=model,
                trocr_model_revision=provenance["revision"],
                trocr_max_regions=1,
                device="cuda",
            ),
            native_factory=lambda **options: object(),
            trocr_reader_factory=lambda **options: UnverifiedReader(),
        )

    assert raised.value.code == "trocr_handwriting_unverified"
    assert health_calls == []


def test_verified_gpu_app_rejects_unhealthy_trocr_backend() -> None:
    class UnhealthyReader:
        name = "trocr-handwriting"
        max_batch_items = 2
        provenance = {
            "id": TROCR_MODEL_ID,
            "source": TROCR_MODEL_ID,
            "revision": TROCR_MODEL_REVISION,
            "origin": TROCR_MODEL_ORIGIN,
            "license": TROCR_MODEL_LICENSE,
            "identity_verified": True,
            "local_files_only": True,
        }

        def check_health(self) -> None:
            raise ReaderError(
                "trocr_handwriting_unavailable",
                "TrOCR handwriting model files are not available locally",
            )

    with pytest.raises(ReaderError) as raised:
        create_verified_app(
            Namespace(
                nemotron_model_dir=Path("/models/nemotron"),
                tatr_source=Path("/models/tatr"),
                tatr_detection_model=Path("/models/detection.pth"),
                tatr_structure_model=Path("/models/structure.pth"),
                tesseract_executable=Path("/tools/tesseract"),
                trocr_handwriting_model=TROCR_MODEL_ID,
                trocr_max_regions=1,
                device="cuda",
            ),
            native_factory=lambda **options: object(),
            trocr_reader_factory=lambda **options: UnhealthyReader(),
        )

    assert raised.value.code == "trocr_handwriting_unavailable"


def test_verified_gpu_app_rejects_unadopted_classifier_routing() -> None:
    with pytest.raises(ValueError, match="did not meet the adoption threshold"):
        create_verified_app(
            Namespace(
                nemotron_model_dir=Path("/models/nemotron"),
                tatr_source=Path("/models/tatr"),
                tatr_detection_model=Path("/models/detection.pth"),
                tatr_structure_model=Path("/models/structure.pth"),
                tesseract_executable=Path("/tools/tesseract"),
                phi4_handwriting_adapter=Path("/models/handwriting.pt"),
                handwriting_classifier=Path("/models/classifier.pt"),
                device="cuda",
            ),
            native_factory=lambda **options: object(),
        )


def test_verified_gpu_app_rejects_classifier_without_handwriting_reader() -> None:
    with pytest.raises(
        ValueError,
        match="handwriting-classifier requires --phi4-handwriting-adapter",
    ):
        create_verified_app(
            Namespace(
                nemotron_model_dir=Path("/models/nemotron"),
                tatr_source=Path("/models/tatr"),
                tatr_detection_model=Path("/models/detection.pth"),
                tatr_structure_model=Path("/models/structure.pth"),
                tesseract_executable=Path("/tools/tesseract"),
                handwriting_classifier=Path("/models/classifier.pt"),
                device="cuda",
            ),
            native_factory=lambda **options: object(),
        )


def test_verified_gpu_app_validates_production_paths_before_import(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError) as error:
        create_verified_app(
            Namespace(
                nemotron_model_dir=missing / "nemotron",
                tatr_source=missing / "tatr",
                tatr_detection_model=missing / "detection.pth",
                tatr_structure_model=missing / "structure.pth",
                tesseract_executable=missing / "tesseract",
                device="cuda",
            )
        )

    message = str(error.value)
    assert "Full GPU pipeline cannot start" in message
    assert "Nemotron model directory" in message
    assert "Table Transformer source" in message
    assert "TATR detection checkpoint" in message
    assert "TATR structure checkpoint" in message
    assert "Tesseract executable" in message
