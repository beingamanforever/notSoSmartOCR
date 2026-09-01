"""Serve the verified local GPU OCR cascade."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from ocr_pipeline.controls import GeometricControlStage
from ocr_pipeline.demo import create_app
from ocr_pipeline.orientation import DocTROrientationDetector, OrientationReader
from ocr_pipeline.preprocessing import TiledReader
from ocr_pipeline.providers import NemotronOCRV2Reader, TesseractReader
from ocr_pipeline.risk import EvidenceRiskStage
from ocr_pipeline.tables import TableChallenger, TatrTableExtractor, TatrTableStage


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    app = create_verified_app(args)
    try:
        import uvicorn
    except ImportError as error:
        raise RuntimeError("Install uvicorn to run the OCR demo") from error
    uvicorn.run(app, host=args.host, port=args.port, workers=1)
    return 0


def create_verified_app(
    args: argparse.Namespace,
    *,
    native_factory: Callable[..., object] | None = None,
    app_factory: Callable[..., Any] = create_app,
) -> Any:
    if native_factory is None:
        try:
            from nemotron_ocr.inference.pipeline_v2 import NemotronOCRV2
        except (ImportError, OSError) as error:
            raise RuntimeError("Nemotron OCR v2 is unavailable") from error
        native_factory = NemotronOCRV2

    native = native_factory(model_dir=str(args.nemotron_model_dir), lang="multi")
    base_reader = NemotronOCRV2Reader(
        language="multi",
        merge_level="word",
        batch_size=1,
        pipeline=native,
    )
    tiny_text_reader = TiledReader(base_reader)
    reader = OrientationReader(
        tiny_text_reader,
        osd_executable=str(args.tesseract_executable),
        orientation_detector=DocTROrientationDetector(device=args.device),
        defer_restore=True,
    )
    extractor = TatrTableExtractor(
        args.tatr_source,
        args.tatr_detection_model,
        args.tatr_structure_model,
        device=args.device,
        crop_padding=5,
        minimum_detection_score=0.5,
    )
    raw = TesseractReader(
        executable=str(args.tesseract_executable),
        language="eng",
        timeout_seconds=120,
        page_segmentation_mode=3,
    )
    enhanced = TesseractReader(
        executable=str(args.tesseract_executable),
        language="eng",
        timeout_seconds=120,
        page_segmentation_mode=3,
        thresholding_method=2,
    )
    stage = TatrTableStage(
        extractor,
        challengers=(
            TableChallenger("tesseract_raw", raw),
            TableChallenger("sauvola", enhanced),
        ),
        low_primary_confidence=0.9,
    )
    controls = GeometricControlStage(label_provider=base_reader.name)
    risk = EvidenceRiskStage(text_provider=base_reader.name)
    return app_factory(reader, stages=(stage, controls, risk))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nemotron-model-dir", required=True, type=Path)
    parser.add_argument("--tatr-source", required=True, type=Path)
    parser.add_argument("--tatr-detection-model", required=True, type=Path)
    parser.add_argument("--tatr-structure-model", required=True, type=Path)
    parser.add_argument("--tesseract-executable", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
