"""Serve the verified local GPU OCR cascade."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from ocr_pipeline.controls import GeometricControlStage
from ocr_pipeline.demo import create_app
from ocr_pipeline.handwriting import HandwritingStage
from ocr_pipeline.handwriting_classifier import (
    HandwritingClassifierStage,
    MobileNetHandwritingClassifier,
)
from ocr_pipeline.orientation import DocTROrientationDetector, OrientationReader
from ocr_pipeline.preprocessing import TiledReader, WideBandFallbackReader
from ocr_pipeline.providers import (
    PHI4_MODEL_ID,
    PHI4_MODEL_REVISION,
    NemotronOCRV2Reader,
    Phi4HandwritingReader,
    Phi4HandwritingServiceReader,
    TesseractReader,
)
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
    handwriting_reader_factory: Callable[..., object] = Phi4HandwritingReader,
    handwriting_service_reader_factory: Callable[..., object] = (
        Phi4HandwritingServiceReader
    ),
    handwriting_classifier_factory: Callable[..., object] = (
        MobileNetHandwritingClassifier
    ),
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
        batch_size=3,
        pipeline=native,
    )
    wide_band = TesseractReader(
        executable=str(args.tesseract_executable),
        language="eng",
        timeout_seconds=120,
        page_segmentation_mode=6,
    )
    wide_band_confirmation = TesseractReader(
        executable=str(args.tesseract_executable),
        language="eng",
        timeout_seconds=120,
        page_segmentation_mode=6,
        thresholding_method=2,
    )
    tiny_text_reader = TiledReader(base_reader)
    wide_band_reader = WideBandFallbackReader(
        tiny_text_reader,
        wide_band,
        confirmation_reader=wide_band_confirmation,
    )
    reader = OrientationReader(
        wide_band_reader,
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
        parallel_challengers=True,
    )
    controls = GeometricControlStage(label_provider=base_reader.name)
    risk = EvidenceRiskStage(text_provider=base_reader.name)
    stages: list[object] = [stage, controls]
    adapter_path = getattr(args, "phi4_handwriting_adapter", None)
    service_url = getattr(args, "phi4_handwriting_url", None)
    if adapter_path is not None or service_url is not None:
        max_regions = getattr(args, "phi4_max_regions", 8)
        classifier_path = getattr(args, "handwriting_classifier", None)
        if classifier_path is not None:
            classifier = handwriting_classifier_factory(
                classifier_path,
                device=getattr(args, "handwriting_classifier_device", "cpu"),
                batch_size=getattr(args, "handwriting_classifier_batch_size", 32),
            )
            stages.append(
                HandwritingClassifierStage(
                    classifier,
                    text_provider=base_reader.name,
                    score_threshold=getattr(
                        args,
                        "handwriting_classifier_threshold",
                        0.99,
                    ),
                    max_regions=getattr(
                        args,
                        "handwriting_classifier_max_regions",
                        16,
                    ),
                    context_padding=getattr(
                        args,
                        "phi4_context_padding",
                        12,
                    ),
                )
            )
        if service_url is not None:
            handwriting_reader = handwriting_service_reader_factory(
                service_url,
                max_batch_items=max_regions * 2,
                timeout_seconds=getattr(args, "phi4_timeout_seconds", 120),
            )
        else:
            handwriting_reader = handwriting_reader_factory(
                adapter_path,
                model_name_or_path=getattr(args, "phi4_model", PHI4_MODEL_ID),
                model_revision=getattr(
                    args,
                    "phi4_model_revision",
                    PHI4_MODEL_REVISION,
                ),
                device=args.device,
                max_new_tokens=getattr(args, "phi4_max_new_tokens", 128),
                max_batch_items=max_regions * 2,
                batch_size=getattr(args, "phi4_batch_size", 2),
                local_files_only=True,
            )
        stages.append(
            HandwritingStage(
                handwriting_reader,
                text_provider=base_reader.name,
                max_regions=max_regions,
                context_padding=getattr(args, "phi4_context_padding", 12),
            )
        )
    stages.append(risk)
    return app_factory(reader, stages=tuple(stages))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nemotron-model-dir", required=True, type=Path)
    parser.add_argument("--tatr-source", required=True, type=Path)
    parser.add_argument("--tatr-detection-model", required=True, type=Path)
    parser.add_argument("--tatr-structure-model", required=True, type=Path)
    parser.add_argument("--tesseract-executable", required=True, type=Path)
    handwriting = parser.add_mutually_exclusive_group()
    handwriting.add_argument(
        "--phi4-handwriting-adapter",
        type=Path,
        help="Enable local Phi-4 crop rereading with this trained adapter",
    )
    handwriting.add_argument(
        "--phi4-handwriting-url",
        help="Use a warm Phi-4 handwriting service on loopback HTTP",
    )
    parser.add_argument(
        "--handwriting-classifier",
        type=Path,
        help="Propose handwriting crops for Phi-4 with this local classifier",
    )
    parser.add_argument(
        "--handwriting-classifier-threshold",
        type=_probability,
        default=0.99,
    )
    parser.add_argument(
        "--handwriting-classifier-max-regions",
        type=_positive_int,
        default=16,
    )
    parser.add_argument(
        "--handwriting-classifier-batch-size",
        type=_positive_int,
        default=32,
    )
    parser.add_argument("--handwriting-classifier-device", default="cpu")
    parser.add_argument(
        "--phi4-model",
        default=PHI4_MODEL_ID,
        help="Pinned cached model ID or a local model directory",
    )
    parser.add_argument("--phi4-model-revision", default=PHI4_MODEL_REVISION)
    parser.add_argument("--phi4-max-regions", type=_positive_int, default=8)
    parser.add_argument("--phi4-max-new-tokens", type=_positive_int, default=128)
    parser.add_argument("--phi4-batch-size", type=_positive_int, default=2)
    parser.add_argument("--phi4-context-padding", type=_positive_int, default=12)
    parser.add_argument("--phi4-timeout-seconds", type=float, default=120)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    return parser


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _probability(value: str) -> float:
    parsed = float(value)
    if not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError("value must be from 0 to 1")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
