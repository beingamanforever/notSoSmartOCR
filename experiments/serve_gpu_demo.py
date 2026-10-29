"""Serve the OCR pipeline: docTR orientation, Heron layout, and Falcon-OCR regions."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import tempfile
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Any

from ocr_pipeline.demo import (
    MAX_DECODED_PIXELS,
    MAX_DOCUMENT_PAGES,
    MAX_PAGE_PIXELS,
    CompositionDescriptor,
    create_app,
)
from ocr_pipeline.markdown_refinement import refine_document
from ocr_pipeline.cache import BoundedCache
from ocr_pipeline.usage import UsageLog
from ocr_pipeline.openrouter import ReplyCache, budgeted_transport
from ocr_pipeline.orientation import (
    OFFICIAL_TEMPERATURE,
    OFFICIAL_UPRIGHT_PRIOR,
    DocTROrientationDetector,
)
from ocr_pipeline.pipeline import (
    DEFAULT_PDF_DPI,
    _prepare_pages,
    _source_kind,
    falcon_pipeline,
    process_document,
)

LOGGER = logging.getLogger(__name__)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    app = create_verified_app(args)
    try:
        import uvicorn
    except ImportError as error:
        raise RuntimeError("Install uvicorn to run the OCR demo") from error
    uvicorn.run(app, host=args.host, port=args.port, workers=1)
    return 0


def _warm_pdf_text_layer_import() -> None:
    """Pay the text-layer backend's one-time import cost at startup, not on the
    first born-digital PDF: on the A10G host it took a page-rendering request from
    ~1s to ~8s (torch and transformers come in transitively), measured with
    experiments/profile_ocr_stages.py. Output is unaffected either way, since
    nothing here touches a real upload; a host without pdftoppm just skips it.
    """
    if shutil.which("pdftoppm") is None:
        return
    try:
        import fitz

        with tempfile.TemporaryDirectory(prefix="ocr-warmup-") as directory:
            source = Path(directory) / "warmup.pdf"
            with fitz.open() as pdf:
                pdf.new_page(width=72, height=72)
                pdf.save(source)
            _prepare_pages(
                source, _source_kind(source), Path(directory), DEFAULT_PDF_DPI, "pdftoppm"
            )
    except Exception:
        LOGGER.exception("PDF text-layer warmup failed; the first real PDF pays the cost")


def create_verified_app(
    args: argparse.Namespace,
    *,
    layout_reader_factory: Callable[[str], Any] | None = None,
    orientation_detector_factory: Callable[..., Any] = DocTROrientationDetector,
    app_factory: Callable[..., Any] = create_app,
    warmup_runner: Callable[..., object] = process_document,
) -> Any:
    _warm_pdf_text_layer_import()
    # Opt-in, in-memory caches keyed by content; both are off unless sized.
    ttl = args.cache_ttl_hours * 3600
    caches = {
        name: BoundedCache(size << 20, ttl)
        for name, size in (("pages", args.page_cache_mb), ("llm_replies", args.qwen_reply_cache_mb))
        if size > 0
    }
    reader, stages = falcon_pipeline(
        args.falcon_layout_url,
        device=args.device,
        page_workers=args.page_workers,
        layout_reader_factory=layout_reader_factory,
        detector_factory=partial(
            orientation_detector_factory,
            weights=args.orientation_weights,
            temperature=args.orientation_temperature,
            upright_prior=args.orientation_upright_prior,
        ),
        page_cache=caches.get("pages"),
    )
    configured_stages = [stage.name for stage in stages]
    if args.qwen_crops:
        if not os.environ.get("OPENROUTER_API_KEY"):
            raise ValueError("Qwen crop refinement requires OPENROUTER_API_KEY")
        configured_stages.append("qwen-crop-refinement")
    composition = CompositionDescriptor(
        id="full-gpu-pipeline",
        label="Full GPU pipeline",
        scope="full",
        primary_ocr="Falcon OCR with learned layout and native decoder scores",
        orientation=(
            "docTR page classifier averaged over four rotations, with docTR skew "
            "estimation"
        ),
        tesseract_roles=(),
        stages=tuple(configured_stages),
        handwriting="read by Falcon OCR",
        build_label="source-linked OCR",
        note=(
            "Configuration only. Layout regions and checkbox states come from the "
            "learned detector; text and transcribed marks come from Falcon. "
            "Readiness and execution are reported only after processing."
        ),
    )
    app_options: dict[str, Any] = {
        "stages": stages,
        "composition": composition,
        "max_pages": args.max_document_pages,
        "max_decoded_pixels": args.max_decoded_pixels,
        "max_page_pixels": args.max_page_pixels,
        "pdf_dpi": args.pdf_dpi,
    }
    if args.qwen_crops:
        app_options["crop_refiner"] = partial(
            refine_document,
            workers=args.qwen_workers,
            zero_data_retention=not args.qwen_allow_non_zdr,
            transport=_with_reply_cache(
                partial(
                    budgeted_transport,
                    database=args.qwen_spend_database,
                    budget_usd=args.qwen_budget_usd,
                    # Worst case per Qwen 3.7 Flash request at 16k output tokens is about
                    # $0.003; the default $0.25 hold left by a killed request locked $4.50
                    # of a $5 budget while real spend was $0.08.
                    reserve_usd=0.01,
                ),
                caches.get("llm_replies"),
            ),
        )
    if caches:
        app_options["caches"] = caches
    app_options["max_active_parses"] = args.max_active_parses
    if args.usage_root is not None:
        app_options["usage_log"] = UsageLog(args.usage_root, keep=args.usage_retention)
        # The dashboard shows what people send, so it stays off without a token.
        app_options["admin_token"] = os.environ.get(args.admin_token_env) or None
    if args.katex_asset_root is not None:
        app_options["katex_asset_root"] = args.katex_asset_root
    if args.feedback_root is not None:
        app_options["feedback_root"] = args.feedback_root
    if args.warmup_image is not None:
        for _ in range(2):
            warmup = warmup_runner(args.warmup_image, reader, stages=stages)
            if getattr(warmup, "status", None) != "success" or not getattr(
                warmup, "pages", None
            ):
                raise RuntimeError("OCR warmup did not produce a successful document")
        app_options["warmup_completed"] = True
    app = app_factory(reader, **app_options)
    if args.enrollment_root is not None:
        from ocr_pipeline.enrollment.app import build_app

        # One interface: enrollment review lives under /enrollment of the OCR service and
        # reads documents through this same service.
        app.mount("/enrollment", build_app(
            args.enrollment_root, f"http://127.0.0.1:{args.port}",
            llm_budget_usd=args.enrollment_llm_budget_usd, allow_jev=args.enrollment_allow_jev,
        ))
    return app


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--falcon-layout-url",
        required=True,
        help="Loopback URL of the Heron and Falcon-OCR layout service",
    )
    parser.add_argument(
        "--page-workers",
        type=_positive_int,
        default=4,
        help="Concurrent page requests; the GPU service controls model execution",
    )
    parser.add_argument(
        "--max-active-parses",
        type=_positive_int,
        default=32,
        help="Parses running or waiting before new uploads get a 503 busy reply",
    )
    parser.add_argument(
        "--usage-root",
        type=Path,
        default=None,
        help="Directory for per-parse usage records and first-page images (off by default)",
    )
    parser.add_argument(
        "--usage-retention",
        type=_positive_int,
        default=2000,
        help="Usage records kept; older ones are deleted with their images",
    )
    parser.add_argument(
        "--admin-token-env",
        default="OCR_ADMIN_TOKEN",
        help="Environment variable holding the token that opens /admin",
    )
    parser.add_argument(
        "--page-cache-mb",
        type=int,
        default=0,
        help=(
            "Memory for readings of byte-identical pages, compressed (0, the default, "
            "disables it). A page result is 140 KB median as JSON before compression, so "
            "1024 holds thousands of pages"
        ),
    )
    parser.add_argument(
        "--qwen-reply-cache-mb",
        type=int,
        default=0,
        help=(
            "Memory for LLM replies to byte-identical requests (0, the default, disables "
            "it). Replies are 0.6 KB median and 3.4 KB at most, so 64 holds tens of thousands"
        ),
    )
    parser.add_argument(
        "--cache-ttl-hours",
        type=float,
        default=24.0,
        help="Age after which a cached page reading or reply is no longer used",
    )
    parser.add_argument(
        "--qwen-workers",
        type=_positive_int,
        default=16,
        help=(
            "Pages refined concurrently. Each page is its own independent request, so "
            "this changes only latency; at the page-worker count a 20-page PDF waited "
            "through five sequential rounds of LLM calls"
        ),
    )
    parser.add_argument(
        "--max-document-pages",
        type=_positive_int,
        default=MAX_DOCUMENT_PAGES,
        help="Maximum pages accepted per uploaded document",
    )
    parser.add_argument(
        "--max-decoded-pixels",
        type=_positive_int,
        default=MAX_DECODED_PIXELS,
        help="Maximum total decoded pixels per uploaded document",
    )
    parser.add_argument(
        "--max-page-pixels",
        type=_positive_int,
        default=MAX_PAGE_PIXELS,
        help="Pages above this many pixels are downscaled to it",
    )
    parser.add_argument(
        "--pdf-dpi", type=_positive_int, default=DEFAULT_PDF_DPI, help="PDF DPI"
    )
    parser.add_argument(
        "--qwen-crops",
        action="store_true",
        help="Refine selected Falcon regions with the benchmark Qwen3.7 Flash crop reader",
    )
    parser.add_argument(
        "--qwen-allow-non-zdr",
        action="store_true",
        help="Allow the explicitly authorized non-ZDR Qwen provider route",
    )
    parser.add_argument("--qwen-budget-usd", type=float, default=1.0)
    parser.add_argument(
        "--qwen-spend-database",
        type=Path,
        default=Path.home() / ".local/state/notsosmartocr/hosted-cost.sqlite",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--orientation-weights",
        type=Path,
        help="docTR page-orientation checkpoint from the official training script",
    )
    parser.add_argument(
        "--orientation-temperature",
        type=float,
        default=OFFICIAL_TEMPERATURE,
        help="Calibration temperature fitted for the orientation weights",
    )
    parser.add_argument(
        "--orientation-upright-prior",
        type=float,
        default=OFFICIAL_UPRIGHT_PRIOR,
        help="Deployment prior that a page is upright, chosen for the weights",
    )
    parser.add_argument(
        "--warmup-image",
        type=Path,
        help="Run the full pipeline twice before the server becomes ready",
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
    parser.add_argument(
        "--enrollment-root",
        type=Path,
        help="Serve enrollment review at /enrollment, storing cases here (outside the repo)",
    )
    parser.add_argument("--enrollment-llm-budget-usd", type=float, default=5.0)
    parser.add_argument(
        "--enrollment-allow-jev",
        action="store_true",
        help="Let synthetic enrollment cases use the verifier service",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    return parser


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _with_reply_cache(transport: Any, cache: BoundedCache | None) -> Any:
    return ReplyCache(transport, cache) if cache is not None else transport


if __name__ == "__main__":
    raise SystemExit(main())
