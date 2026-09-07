"""Serve the verified local GPU OCR cascade."""

from __future__ import annotations

import argparse
import os
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from ocr_pipeline.anchored_ink import AnchoredInkProposalStage
from ocr_pipeline.digit_verification import DigitVerificationStage
from ocr_pipeline.handwriting_lines import DocTRLineDetector, HandwritingLineStage
from ocr_pipeline.comprehension import HttpTiltEngine, TiltFieldStage, VllmTiltEngine
from ocr_pipeline.controls import GeometricControlStage
from ocr_pipeline.correction import LexiconCorrectionStage
from ocr_pipeline.demo import CompositionDescriptor, _read_presentations, create_app
from ocr_pipeline.dispute_resolution import DisputeResolutionStage
from ocr_pipeline.evidence_layout import EvidenceLayoutStage
from ocr_pipeline.falcon import FalconOCRServiceReader
from ocr_pipeline.falcon_layout import FalconLayoutReader
from ocr_pipeline.fused_reader import FusedReader
from ocr_pipeline.falcon_presentation import (
    FalconFormulaStage,
    FalconPresentationReader,
)
from ocr_pipeline.faint_text import FaintTinyTextStage
from ocr_pipeline.handwriting import (
    TROCR_MODEL_ID,
    TROCR_MODEL_REVISION,
    HandwritingStage,
    TrOCRHandwritingReader,
    is_verified_trocr_model_provenance,
)
from ocr_pipeline.orientation import (
    DocTROrientationDetector,
    OrientationReader,
    PaddleDocOrientationDetector,
)
from ocr_pipeline.pipeline import process_document
from ocr_pipeline.preprocessing import (
    PageFrameReader,
    RestoredViewReader,
    TiledReader,
    WideBandFallbackReader,
)
from ocr_pipeline.providers import (
    MINISTRAL_MODEL_REVISION,
    PHI4_MODEL_ID,
    PHI4_MODEL_REVISION,
    MinistralOCRReader,
    MinistralOCRServiceReader,
    MinistralStructuredImageCall,
    NemotronOCRV2Reader,
    Phi4HandwritingReader,
    Phi4HandwritingServiceReader,
    ReaderError,
    TesseractReader,
)
from ocr_pipeline.restoration import DocResRestorer
from ocr_pipeline.risk import EvidenceRiskStage
from ocr_pipeline.heron_layout import HeronTableDetector
from ocr_pipeline.tables import TableChallenger, TatrTableExtractor, TatrTableStage
from ocr_pipeline.tableformer_structure import TableFormerStructure


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
    line_detector_factory: Callable[..., Any] = DocTRLineDetector,
    handwriting_reader_factory: Callable[..., object] = Phi4HandwritingReader,
    handwriting_service_reader_factory: Callable[..., object] = (
        Phi4HandwritingServiceReader
    ),
    trocr_reader_factory: Callable[..., object] = TrOCRHandwritingReader,
    presentation_reader_factory: Callable[..., object] = MinistralOCRReader,
    presentation_service_reader_factory: Callable[..., object] = (
        MinistralOCRServiceReader
    ),
    falcon_service_reader_factory: Callable[..., object] = FalconOCRServiceReader,
    falcon_presentation_factory: Callable[..., object] = FalconPresentationReader,
    app_factory: Callable[..., Any] = create_app,
    warmup_runner: Callable[..., object] = process_document,
    presentation_warmup_runner: Callable[..., object] = _read_presentations,
) -> Any:
    adapter_path = getattr(args, "phi4_handwriting_adapter", None)
    service_url = getattr(args, "phi4_handwriting_url", None)
    trocr_model = getattr(args, "trocr_handwriting_model", None)
    trocr_revision = getattr(args, "trocr_model_revision", TROCR_MODEL_REVISION)
    classifier_path = getattr(args, "handwriting_classifier", None)
    if trocr_model is not None and (
        str(trocr_model) != TROCR_MODEL_ID or trocr_revision != TROCR_MODEL_REVISION
    ):
        raise ReaderError(
            "trocr_handwriting_unverified",
            "TrOCR handwriting requires the verified official pinned model",
        )
    if (
        classifier_path is not None
        and adapter_path is None
        and service_url is None
        and trocr_model is None
    ):
        raise ValueError(
            "--handwriting-classifier requires --phi4-handwriting-adapter "
            "or --phi4-handwriting-url"
        )
    if classifier_path is not None:
        raise ValueError(
            "--handwriting-classifier is not enabled because measured automatic "
            "handwriting localization did not meet the adoption threshold"
        )

    falcon_layout_url = getattr(args, "falcon_layout_url", None)
    if falcon_layout_url is not None:
        # Falcon-Perception reads checkbox glyphs that Nemotron drops entirely, so it is
        # the reader when its service is configured. Regions are page areas, not words.
        layout_reader = FalconLayoutReader(falcon_layout_url)
        layout_reader.check_health()
        base_reader = layout_reader
        # With Nemotron as well, the two are fused: Falcon's text keeps its parse while
        # the word reader supplies pixel boxes, a real recognition confidence, and the
        # words no layout region covered.
        if getattr(args, "nemotron_model_dir", None) is not None:
            if native_factory is None:
                try:
                    from nemotron_ocr.inference.pipeline_v2 import NemotronOCRV2
                except (ImportError, OSError) as error:
                    raise RuntimeError("Nemotron OCR v2 is unavailable") from error
                native_factory = NemotronOCRV2
            base_reader = FusedReader(
                layout_reader,
                NemotronOCRV2Reader(
                    language="multi",
                    merge_level="word",
                    batch_size=1,
                    pipeline=native_factory(
                        model_dir=str(args.nemotron_model_dir), lang="multi"
                    ),
                    execution_lock=threading.Lock(),
                ),
            )
        faint_text_reader = base_reader
        native_factory = None
    else:
        if native_factory is None:
            _validate_local_configuration(
                args,
                adapter_path=adapter_path,
                classifier_path=classifier_path,
            )
            try:
                from nemotron_ocr.inference.pipeline_v2 import NemotronOCRV2
            except (ImportError, OSError) as error:
                raise RuntimeError("Nemotron OCR v2 is unavailable") from error
            native_factory = NemotronOCRV2

        native = native_factory(model_dir=str(args.nemotron_model_dir), lang="multi")
        nemotron_execution_lock = threading.Lock()
        base_reader = NemotronOCRV2Reader(
            language="multi",
            merge_level="word",
            batch_size=1,
            pipeline=native,
            execution_lock=nemotron_execution_lock,
        )
        faint_text_reader = NemotronOCRV2Reader(
            language="multi",
            merge_level="word",
            batch_size=8,
            pipeline=native,
            execution_lock=nemotron_execution_lock,
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
    # Restoration sits innermost so it sees the deskewed, frame-cropped page, and it is a
    # no-op unless a DocRes checkout and checkpoint are configured.
    docres_checkpoint = getattr(args, "docres_checkpoint", None)
    docres_source = getattr(args, "docres_source", None)
    page_reader = base_reader
    if docres_checkpoint and docres_source:
        page_reader = RestoredViewReader(
            base_reader,
            DocResRestorer(docres_checkpoint, docres_source),
            maximum_mean_confidence=getattr(args, "docres_max_mean_confidence", 0.75),
        )
    if falcon_layout_url is not None:
        # Tiling and the wide-band Tesseract fallback exist to compensate for a word-level
        # detector missing small or shallow text. Falcon reads whole regions, so each
        # wrapper only buys a second full pass: they cost 32s of a 41s page here.
        wide_band_reader = page_reader
    else:
        tiny_text_reader = TiledReader(page_reader)
        wide_band_reader = WideBandFallbackReader(
            tiny_text_reader,
            wide_band,
            confirmation_reader=wide_band_confirmation,
        )
    oriented = OrientationReader(
        wide_band_reader,
        osd_executable=str(args.tesseract_executable),
        orientation_detector=_orientation_detector(args),
        margin_reader=TesseractReader(
            executable=str(args.tesseract_executable),
            language="eng",
            timeout_seconds=120,
            page_segmentation_mode=3,
        ),
        defer_restore=True,
    )
    # The frame crop focuses a word-level detector and then re-reads the whole page to
    # check nothing fell outside it. Falcon's layout model already ignores scan borders,
    # so that second full pass is 6.5s of pure duplication.
    reader = oriented if falcon_layout_url is not None else PageFrameReader(oriented)
    # Both run on CPU: the GPU is within 1.5 GB of full with the readers loaded, and
    # heron is 28 ms on an A100 class card, so CPU latency is affordable per page.
    proposal_detector = (
        HeronTableDetector(device=getattr(args, "heron_device", "cpu"))
        if getattr(args, "heron_proposals", True)
        else None
    )
    structure_fallback = (
        TableFormerStructure(device=getattr(args, "tableformer_device", "cpu"))
        if getattr(args, "tableformer_fallback", True)
        else None
    )
    extractor = TatrTableExtractor(
        args.tatr_source,
        args.tatr_detection_model,
        args.tatr_structure_model,
        device=args.device,
        crop_padding=5,
        minimum_detection_score=0.5,
        enable_ruled_table_proposals=True,
        proposal_detector=proposal_detector,
        structure_fallback=structure_fallback,
    )
    raw = TesseractReader(
        executable=str(args.tesseract_executable),
        language="eng",
        timeout_seconds=120,
        page_segmentation_mode=4,
    )
    enhanced = TesseractReader(
        executable=str(args.tesseract_executable),
        language="eng",
        timeout_seconds=120,
        page_segmentation_mode=4,
        thresholding_method=2,
    )
    cell = TesseractReader(
        executable=str(args.tesseract_executable),
        language="eng",
        timeout_seconds=120,
        page_segmentation_mode=7,
    )
    stage = TatrTableStage(
        extractor,
        challengers=(
            TableChallenger("tesseract_raw", raw),
            TableChallenger("sauvola", enhanced),
            TableChallenger("tesseract_cell", cell, scope="cell"),
        ),
        low_primary_confidence=0.88,
        parallel_challengers=True,
    )
    controls = GeometricControlStage(
        label_provider=(
            (base_reader.name, "falcon-perception")
            if falcon_layout_url is not None
            else base_reader.name
        )
    )
    risk = EvidenceRiskStage(text_provider=base_reader.name)
    stages: list[object] = [stage, controls]
    if falcon_layout_url is None:
        # The faint-text stage retiles the page and rereads it. With Falcon that is a
        # second full pass per tile for text it already read in the first one.
        stages.append(FaintTinyTextStage(faint_text_reader))
    handwriting_stage = None
    if adapter_path is not None or service_url is not None or trocr_model is not None:
        using_trocr = trocr_model is not None
        max_regions = getattr(
            args,
            "trocr_max_regions" if using_trocr else "phi4_max_regions",
            8,
        )
        if using_trocr:
            handwriting_reader = trocr_reader_factory(
                model_name_or_path=trocr_model,
                model_revision=trocr_revision,
                device=args.device,
                max_new_tokens=getattr(args, "trocr_max_new_tokens", 128),
                max_batch_items=max_regions * 2,
                batch_size=min(
                    getattr(args, "trocr_batch_size", 4),
                    max_regions * 2,
                ),
            )
            if not is_verified_trocr_model_provenance(
                getattr(handwriting_reader, "provenance", None)
            ):
                raise ReaderError(
                    "trocr_handwriting_unverified",
                    "TrOCR handwriting requires the verified official pinned model",
                )
            handwriting_reader.check_health()
        elif service_url is not None:
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
        handwriting_stage = HandwritingStage(
            handwriting_reader,
            text_provider=base_reader.name,
            max_regions=max_regions,
            context_padding=getattr(
                args,
                "trocr_context_padding" if using_trocr else "phi4_context_padding",
                12,
            ),
        )
    presentation_reader = None
    dispute_reader = None
    formula_stage = None
    presentation_model = getattr(args, "ministral_presentation_model", None)
    presentation_url = getattr(args, "ministral_presentation_url", None)
    falcon_presentation_url = getattr(args, "falcon_presentation_url", None)
    if presentation_url is not None:
        presentation_reader = presentation_service_reader_factory(
            presentation_url,
            timeout_seconds=getattr(args, "ministral_timeout_seconds", 180),
        )
        dispute_reader = presentation_reader
    elif presentation_model is not None:
        presentation_reader = presentation_reader_factory(
            model_name=str(presentation_model),
            model_revision=getattr(
                args,
                "ministral_model_revision",
                MINISTRAL_MODEL_REVISION,
            ),
            max_new_tokens=getattr(args, "ministral_max_new_tokens", 8192),
        )
        dispute_reader = presentation_reader
    elif falcon_presentation_url is not None:
        falcon_reader = falcon_service_reader_factory(
            falcon_presentation_url,
            timeout_seconds=getattr(args, "falcon_timeout_seconds", 180),
            max_batch_items=getattr(args, "falcon_max_crops", 24),
        )
        falcon_reader.check_health()
        presentation_reader = falcon_presentation_factory(
            falcon_reader,
            max_crops=getattr(args, "falcon_max_crops", 24),
            formula_padding=getattr(args, "falcon_formula_padding", 12),
        )
        formula_stage = FalconFormulaStage(presentation_reader)
    if dispute_reader is not None:
        stages.append(
            DisputeResolutionStage(
                MinistralStructuredImageCall(dispute_reader),
            )
        )
    stages.extend((risk, EvidenceLayoutStage()))
    if formula_stage is not None:
        stages.append(formula_stage)
    stages.append(DigitVerificationStage(cell, text_provider=base_reader.name))
    stages.append(AnchoredInkProposalStage(label_provider=base_reader.name))
    lexicon_path = getattr(args, "lexicon", None)
    if lexicon_path:
        # Runs before layout so downstream stages see the alternatives. Proposes only
        # words already in the supplied list, so it cannot invent a drug name.
        stages.insert(
            0,
            LexiconCorrectionStage(
                set(Path(lexicon_path).read_text(encoding="utf-8").split())
            ),
        )
    if handwriting_stage is not None:
        line_detector = line_detector_factory(device=args.device)
        # loaded now so the first real request does not pay for the model
        line_detector.check_health()
        stages.append(
            HandwritingLineStage(
                line_detector,
                text_provider=base_reader.name,
                max_lines=getattr(args, "handwriting_max_lines", 24),
            )
        )
        stages.append(handwriting_stage)
    tilt_url = getattr(args, "tilt_url", None)
    if tilt_url is not None or getattr(args, "arctic_tilt", False):
        # Last, so the fields it is asked about carry their final state: anything still
        # not_located or illegible here is what geometry genuinely could not settle.
        engine = (
            HttpTiltEngine(tilt_url)
            if tilt_url is not None
            else VllmTiltEngine(gpu_memory_utilization=args.tilt_gpu_memory_utilization)
        )
        stages.append(TiltFieldStage(engine, max_questions=args.tilt_max_questions))
    handwriting = "configured" if handwriting_stage is not None else "not configured"
    configured_stages = [item.name for item in stages]
    if presentation_reader is not None:
        configured_stages.append("page-presentation")
    presentation_note = ""
    if falcon_presentation_url is not None:
        presentation_note = (
            " Falcon-OCR reads source-ink formula crops. Its raw output remains "
            "review evidence until human acceptance makes it canonical."
        )
    composition = CompositionDescriptor(
        id="full-gpu-pipeline",
        label="Full GPU pipeline",
        scope="full",
        primary_ocr=_primary_ocr_label(args, falcon_layout_url),
        orientation=(
            f"{_orientation_detector(args).name} proposals "
            "with Tesseract OSD evidence check"
        ),
        tesseract_roles=(
            "orientation OSD",
            "side-margin text",
            "selective wide-band fallback and confirmation",
            "table challengers (raw, Sauvola, and selective cell reread)",
        ),
        stages=tuple(configured_stages),
        handwriting=handwriting,
        build_label="2026-09-06-form-evidence-v4",
        note=(
            "Configuration only. Table Transformer is augmented by conservative "
            "ruled-form proposals and count-gated grid repair. Models are loaded "
            "lazily where supported. Label-anchored form ink is proposed "
            "automatically, but specialist text remains pending until independently "
            "validated. Optional page presentation never mutates canonical evidence. "
            "Readiness and execution are "
            f"reported only after processing.{presentation_note}"
        ),
    )
    app_options = {
        "stages": tuple(stages),
        "handwriting_stage": handwriting_stage,
        "composition": composition,
    }
    katex_asset_root = getattr(args, "katex_asset_root", None)
    if katex_asset_root is not None:
        app_options["katex_asset_root"] = katex_asset_root
    feedback_root = getattr(args, "feedback_root", None)
    if feedback_root is not None:
        app_options["feedback_root"] = feedback_root
    if presentation_reader is not None:
        app_options["presentation_reader"] = presentation_reader
        default_pages = getattr(args, "falcon_max_pages", 4)
        if dispute_reader is not None:
            default_pages = getattr(args, "ministral_max_pages", 4)
        app_options["max_presentation_pages"] = default_pages
    warmup_image = getattr(args, "warmup_image", None)
    if warmup_image is not None:
        for _ in range(2):
            warmup = warmup_runner(warmup_image, reader, stages=tuple(stages))
            if getattr(warmup, "status", None) != "success" or not getattr(
                warmup,
                "pages",
                None,
            ):
                raise RuntimeError("OCR warmup did not produce a successful document")
            if presentation_reader is not None:
                presentations = presentation_warmup_runner(
                    warmup.pages,
                    (warmup_image,),
                    presentation_reader,
                    max_pages=app_options["max_presentation_pages"],
                )
                if any(
                    presentation.get("status") == "failed"
                    for presentation in presentations.values()
                ):
                    raise RuntimeError("Presentation warmup failed")
        app_options["warmup_completed"] = True
    return app_factory(reader, **app_options)


def _primary_ocr_label(args: argparse.Namespace, falcon_layout_url: str | None) -> str:
    if falcon_layout_url is None:
        return "NVIDIA Nemotron OCR v2 (word merge, batch size 1)"
    falcon = "Falcon-Perception layout OCR (PP-DocLayoutV3 regions, per-region read)"
    if getattr(args, "nemotron_model_dir", None) is None:
        return falcon
    return (
        f"{falcon}, fused with NVIDIA Nemotron OCR v2 word boxes "
        "for pixel geometry and recognition confidence"
    )


def _orientation_detector(args: argparse.Namespace) -> Any:
    """PP-LCNet by default: docTR's classifier returns near-chance confidence on
    pages that are not document-shaped, and the pipeline then rotates them."""
    if getattr(args, "orientation_model", "paddle") == "doctr":
        return DocTROrientationDetector(device=args.device)
    return PaddleDocOrientationDetector()


def _validate_local_configuration(
    args: argparse.Namespace,
    *,
    adapter_path: Path | None,
    classifier_path: Path | None,
) -> None:
    required = (
        ("Nemotron model directory", Path(args.nemotron_model_dir), "directory"),
        ("Table Transformer source", Path(args.tatr_source), "directory"),
        ("TATR detection checkpoint", Path(args.tatr_detection_model), "file"),
        ("TATR structure checkpoint", Path(args.tatr_structure_model), "file"),
        ("Tesseract executable", Path(args.tesseract_executable), "executable"),
    )
    if adapter_path is not None:
        required += (("Phi-4 handwriting adapter", Path(adapter_path), "file"),)
    if classifier_path is not None:
        required += (("Handwriting classifier", Path(classifier_path), "file"),)
    presentation_model = getattr(args, "ministral_presentation_model", None)
    if presentation_model is not None:
        required += (
            (
                "Ministral presentation model",
                Path(presentation_model),
                "directory",
            ),
        )
    katex_asset_root = getattr(args, "katex_asset_root", None)
    if katex_asset_root is not None:
        required += (("KaTeX asset root", Path(katex_asset_root), "directory"),)
    warmup_image = getattr(args, "warmup_image", None)
    if warmup_image is not None:
        required += (("Warmup image", Path(warmup_image), "file"),)

    invalid: list[str] = []
    for label, path, kind in required:
        valid = path.is_dir() if kind == "directory" else path.is_file()
        if kind == "executable":
            valid = valid and os.access(path, os.X_OK)
        if not valid:
            invalid.append(f"{label}: {path} ({kind} required)")
    if invalid:
        raise FileNotFoundError(
            "Full GPU pipeline cannot start:\n" + "\n".join(invalid)
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nemotron-model-dir", type=Path)
    parser.add_argument(
        "--falcon-layout-url",
        help="Read pages with the Falcon-Perception layout OCR service instead of Nemotron",
    )
    parser.add_argument("--tatr-source", required=True, type=Path)
    parser.add_argument("--tatr-detection-model", required=True, type=Path)
    parser.add_argument("--tatr-structure-model", required=True, type=Path)
    parser.add_argument(
        "--heron-proposals", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--heron-device", default="cpu")
    parser.add_argument(
        "--tableformer-fallback", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--tableformer-device", default="cpu")
    parser.add_argument("--tesseract-executable", required=True, type=Path)
    presentation = parser.add_mutually_exclusive_group()
    presentation.add_argument(
        "--ministral-presentation-model",
        type=Path,
        help="Enable review-only page presentation with this local model directory",
    )
    presentation.add_argument(
        "--ministral-presentation-url",
        help="Use a warm Ministral page service on loopback HTTP",
    )
    presentation.add_argument(
        "--falcon-presentation-url",
        help="Use Falcon category crops from a warm loopback service",
    )
    parser.add_argument(
        "--ministral-model-revision",
        default=MINISTRAL_MODEL_REVISION,
    )
    parser.add_argument(
        "--ministral-max-new-tokens",
        type=_positive_int,
        default=8192,
    )
    parser.add_argument("--ministral-max-pages", type=_positive_int, default=4)
    parser.add_argument(
        "--ministral-timeout-seconds",
        type=_positive_int,
        default=180,
    )
    parser.add_argument("--falcon-max-crops", type=_positive_int, default=24)
    parser.add_argument("--falcon-formula-padding", type=_nonnegative_int, default=12)
    parser.add_argument("--falcon-max-pages", type=_positive_int, default=4)
    parser.add_argument(
        "--falcon-timeout-seconds",
        type=_positive_int,
        default=180,
    )
    parser.add_argument(
        "--feedback-root",
        type=Path,
        help="Directory that keeps reviewer feedback across restarts",
    )
    parser.add_argument(
        "--katex-asset-root",
        type=Path,
        help="Serve local KaTeX assets for formula presentation",
    )
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
    handwriting.add_argument(
        "--trocr-handwriting-model",
        nargs="?",
        const=TROCR_MODEL_ID,
        help=(
            "Enable local-only TrOCR line rereading from the cached, pinned official "
            "model"
        ),
    )
    parser.add_argument(
        "--handwriting-classifier",
        type=Path,
        help="Rejected automatic routing checkpoint; passing it fails clearly",
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
    parser.add_argument("--trocr-model-revision", default=TROCR_MODEL_REVISION)
    parser.add_argument("--trocr-max-regions", type=_positive_int, default=24)
    parser.add_argument("--handwriting-max-lines", type=_positive_int, default=24)
    parser.add_argument("--trocr-max-new-tokens", type=_positive_int, default=128)
    parser.add_argument("--trocr-batch-size", type=_positive_int, default=4)
    parser.add_argument("--trocr-context-padding", type=_positive_int, default=12)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--orientation-model",
        choices=("paddle", "doctr"),
        default="paddle",
    )
    parser.add_argument(
        "--warmup-image",
        type=Path,
        help="Run the full pipeline twice before the server becomes ready",
    )
    parser.add_argument("--lexicon", type=Path, default=None)
    parser.add_argument("--arctic-tilt", action="store_true")
    parser.add_argument(
        "--tilt-url",
        help="Use a warm Arctic-TILT service on loopback HTTP instead of in-process vLLM",
    )
    parser.add_argument("--tilt-max-questions", type=_positive_int, default=24)
    parser.add_argument("--tilt-gpu-memory-utilization", type=float, default=0.2)
    parser.add_argument("--docres-checkpoint", type=Path, default=None)
    parser.add_argument("--docres-source", type=Path, default=None)
    parser.add_argument("--docres-max-mean-confidence", type=float, default=0.75)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    return parser


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value cannot be negative")
    return parsed


def _probability(value: str) -> float:
    parsed = float(value)
    if not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError("value must be from 0 to 1")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
