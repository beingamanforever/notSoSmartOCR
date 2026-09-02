from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from experiments.serve_gpu_demo import create_verified_app
from ocr_pipeline.handwriting import HandwritingStage
from ocr_pipeline.handwriting_classifier import HandwritingClassifierStage
from ocr_pipeline.orientation import DocTROrientationDetector, OrientationReader
from ocr_pipeline.preprocessing import TiledReader, WideBandFallbackReader


def test_verified_gpu_app_wires_word_reader_and_table_specialists() -> None:
    native_calls: list[dict[str, object]] = []
    app_calls: list[tuple[object, tuple[object, ...]]] = []

    def native_factory(**options: object) -> object:
        native_calls.append(options)
        return object()

    def app_factory(reader: object, *, stages: tuple[object, ...]) -> object:
        app_calls.append((reader, stages))
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
    reader, stages = app_calls[0]
    assert isinstance(reader, OrientationReader)
    assert reader.defer_restore is True
    assert isinstance(reader.orientation_detector, DocTROrientationDetector)
    assert isinstance(reader.reader, WideBandFallbackReader)
    assert isinstance(reader.reader.reader, TiledReader)
    assert reader.reader.reader.reader.merge_level == "word"
    assert reader.reader.reader.reader.batch_size == 3
    assert reader.reader.fallback_reader.page_segmentation_mode == 6
    assert reader.reader.fallback_reader.thresholding_method is None
    assert reader.reader.confirmation_reader.page_segmentation_mode == 6
    assert reader.reader.confirmation_reader.thresholding_method == 2
    assert [stage.name for stage in stages] == [
        "tables",
        "controls",
        "evidence-risk",
    ]
    stage = stages[0]
    assert stage.name == "tables"
    assert [challenger.name for challenger in stage.challengers] == [
        "tesseract_raw",
        "sauvola",
    ]
    assert stage.parallel_challengers is True
    assert stage.challengers[0].reader is not reader.reader.fallback_reader
    assert stage.challengers[0].reader.page_segmentation_mode == 3
    assert stage.challengers[0].reader.thresholding_method is None
    assert stage.challengers[1].reader.page_segmentation_mode == 3
    assert stage.challengers[1].reader.thresholding_method == 2
    assert stages[1].label_provider == reader.reader.reader.reader.name


def test_verified_gpu_app_adds_phi4_only_when_adapter_is_configured() -> None:
    app_calls: list[tuple[object, tuple[object, ...]]] = []
    handwriting_calls: list[tuple[Path, dict[str, object]]] = []

    def app_factory(reader: object, *, stages: tuple[object, ...]) -> object:
        app_calls.append((reader, stages))
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
    stages = app_calls[0][1]
    assert [stage.name for stage in stages] == [
        "tables",
        "controls",
        "handwriting",
        "evidence-risk",
    ]
    handwriting = stages[-2]
    assert isinstance(handwriting, HandwritingStage)
    assert handwriting.max_regions == 3
    assert handwriting.context_padding == 18


def test_verified_gpu_app_uses_warm_phi4_service_when_configured() -> None:
    app_calls: list[tuple[object, tuple[object, ...]]] = []
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
        app_factory=lambda reader, *, stages: app_calls.append((reader, stages)),
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
        "handwriting",
        "evidence-risk",
    ]
    handwriting = stages[-2]
    assert isinstance(handwriting, HandwritingStage)
    assert handwriting.context_padding == 18


def test_verified_gpu_app_routes_classifier_before_phi4() -> None:
    app_calls: list[tuple[object, tuple[object, ...]]] = []
    classifier_calls: list[tuple[Path, dict[str, object]]] = []

    def classifier_factory(checkpoint: Path, **options: object) -> object:
        classifier_calls.append((checkpoint, options))
        return type(
            "Classifier",
            (),
            {"name": "classifier", "provenance": {}, "score_batch": lambda *_: []},
        )()

    class FakeHandwritingReader:
        name = "phi4-handwriting"
        provenance: dict[str, object] = {}
        max_batch_items = 4

    create_verified_app(
        Namespace(
            nemotron_model_dir=Path("/models/nemotron"),
            tatr_source=Path("/models/tatr"),
            tatr_detection_model=Path("/models/detection.pth"),
            tatr_structure_model=Path("/models/structure.pth"),
            tesseract_executable=Path("/tools/tesseract"),
            phi4_handwriting_adapter=Path("/models/handwriting.pt"),
            handwriting_classifier=Path("/models/classifier.pt"),
            handwriting_classifier_threshold=0.99,
            handwriting_classifier_max_regions=12,
            handwriting_classifier_batch_size=24,
            handwriting_classifier_device="cpu",
            phi4_max_regions=2,
            device="cuda",
        ),
        native_factory=lambda **options: object(),
        handwriting_reader_factory=lambda *args, **options: FakeHandwritingReader(),
        handwriting_classifier_factory=classifier_factory,
        app_factory=lambda reader, *, stages: app_calls.append((reader, stages)),
    )

    assert classifier_calls == [
        (
            Path("/models/classifier.pt"),
            {"device": "cpu", "batch_size": 24},
        )
    ]
    stages = app_calls[0][1]
    assert [stage.name for stage in stages] == [
        "tables",
        "controls",
        "handwriting-classifier",
        "handwriting",
        "evidence-risk",
    ]
    classifier = stages[-3]
    assert isinstance(classifier, HandwritingClassifierStage)
    assert classifier.score_threshold == 0.99
    assert classifier.max_regions == 12
