from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from experiments.serve_gpu_demo import create_verified_app
from ocr_pipeline.orientation import DocTROrientationDetector, OrientationReader
from ocr_pipeline.preprocessing import TiledReader


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
    assert isinstance(reader.reader, TiledReader)
    assert reader.reader.reader.merge_level == "word"
    assert reader.reader.reader.batch_size == 1
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
    assert stage.challengers[1].reader.thresholding_method == 2
    assert stages[1].label_provider == reader.reader.reader.name
