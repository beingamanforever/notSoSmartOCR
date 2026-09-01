"""Ordered PDF and image ingestion."""

from __future__ import annotations

import copy
import re
import subprocess
import tempfile
import time
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from .contracts import (
    BoundingBox,
    DocumentResult,
    Failure,
    PageResult,
    RegionStage,
    TextRegion,
)
from .providers import LocalReader, ReaderError
from .rendering import render_evidence

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def process_document(
    source: str | Path,
    reader: LocalReader,
    *,
    pdf_dpi: int = 300,
    pdftoppm_executable: str = "pdftoppm",
    stages: Sequence[RegionStage] = (),
    timings: dict[str, float] | None = None,
) -> DocumentResult:
    if timings is not None:
        timings.clear()
    source_path = Path(source)
    source_kind = _source_kind(source_path)
    result = DocumentResult(
        document_id=source_path.stem or source_path.name,
        source={"name": source_path.name, "kind": source_kind},
        status="failed",
    )

    if not source_path.is_file():
        result.failures.append(
            _failure("ingest", "source_not_found", f"File not found: {source_path}")
        )
        return result
    if source_kind == "unsupported":
        result.failures.append(
            _failure(
                "ingest",
                "unsupported_source",
                f"Unsupported file type: {source_path.suffix or '<none>'}",
            )
        )
        return result

    try:
        with tempfile.TemporaryDirectory(prefix="ocr-pipeline-") as temporary_dir:
            prepare_started = time.perf_counter()
            try:
                pages = _prepare_pages(
                    source_path,
                    source_kind,
                    Path(temporary_dir),
                    pdf_dpi,
                    pdftoppm_executable,
                )
            finally:
                _add_timing(timings, "prepare", prepare_started)
            batch_results: list[list[TextRegion] | ReaderError] | None = None
            read_batch = getattr(reader, "read_batch", None)
            if (
                len(pages) > 1
                and callable(read_batch)
                and getattr(reader, "batch_size", 1) > 1
            ):
                reader_started = time.perf_counter()
                try:
                    batch_results = read_batch(
                        pages,
                        list(range(1, len(pages) + 1)),
                    )
                except ReaderError as error:
                    batch_results = [ReaderError(error.code, str(error)) for _ in pages]
                finally:
                    _add_timing(timings, "reader", reader_started)
                if not isinstance(batch_results, list) or len(batch_results) != len(
                    pages
                ):
                    batch_results = [
                        ReaderError(
                            "invalid_batch_output",
                            "Batch reader returned the wrong number of page results",
                        )
                        for _ in pages
                    ]
                else:
                    batch_results = [
                        item
                        if isinstance(item, ReaderError)
                        or (
                            isinstance(item, list)
                            and all(isinstance(region, TextRegion) for region in item)
                        )
                        else ReaderError(
                            "invalid_batch_output",
                            "Batch reader returned an invalid page result",
                        )
                        for item in batch_results
                    ]
            for page_number, page_path in enumerate(pages, start=1):
                reader_result = (
                    batch_results[page_number - 1]
                    if batch_results is not None
                    else None
                )
                result.pages.append(
                    _read_page(
                        page_path,
                        page_number,
                        reader,
                        result.failures,
                        reader_result,
                        stages,
                        timings,
                    )
                )
    except PipelineError as error:
        result.failures.append(_failure(error.stage, error.code, str(error)))

    result.status = _document_status(result)
    return result


class PipelineError(RuntimeError):
    def __init__(self, stage: str, code: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage
        self.code = code


def _read_page(
    image_path: Path,
    page_number: int,
    reader: LocalReader,
    failures: list[Failure],
    reader_result: list[TextRegion] | ReaderError | None = None,
    stages: Sequence[RegionStage] = (),
    timings: dict[str, float] | None = None,
) -> PageResult:
    page_failure_ids: list[str] = []
    try:
        with Image.open(image_path) as image:
            width, height = image.size
    except (OSError, UnidentifiedImageError) as error:
        failure = _failure(
            "image",
            "invalid_image",
            str(error),
            page_number,
            len(failures) + 1,
        )
        failures.append(failure)
        return PageResult(
            page_number=page_number,
            width=0,
            height=0,
            reader=reader.name,
            route="review",
            text=render_evidence([]),
            failure_ids=[failure.id],
        )

    reader_started = None
    try:
        if isinstance(reader_result, ReaderError):
            raise reader_result
        if reader_result is None:
            reader_started = time.perf_counter()
        regions = (
            reader.read(image_path, page_number)
            if reader_result is None
            else reader_result
        )
    except ReaderError as error:
        failure = _failure(
            "ocr", error.code, str(error), page_number, len(failures) + 1
        )
        failures.append(failure)
        page_failure_ids.append(failure.id)
        regions = []
    finally:
        if reader_started is not None:
            _add_timing(timings, "reader", reader_started)

    if not regions and not page_failure_ids:
        failure = _failure(
            "ocr",
            "no_text_detected",
            "The reader returned no text regions",
            page_number,
            len(failures) + 1,
        )
        failures.append(failure)
        page_failure_ids.append(failure.id)
        regions = [
            TextRegion(
                id=f"p{page_number}-empty-page-1",
                kind="page_text",
                text="",
                confidence=None,
                bounding_box=BoundingBox(0, 0, width, height),
                reading_order=1,
                provider=reader.name,
            )
        ]

    stage_view = getattr(reader, "stage_view", None)
    stage_view_started = time.perf_counter()
    stage_view_recorded = False
    try:
        stage_context = (
            stage_view(image_path, page_number)
            if callable(stage_view)
            else nullcontext(image_path)
        )
        with stage_context as stage_image_path:
            _add_timing(timings, "stage_view", stage_view_started)
            stage_view_recorded = True
            regions = _apply_stages(
                stage_image_path,
                page_number,
                regions,
                stages,
                failures,
                page_failure_ids,
                timings,
            )
    except ReaderError as error:
        if not stage_view_recorded:
            _add_timing(timings, "stage_view", stage_view_started)
        failure = _failure(
            "orientation",
            error.code,
            str(error),
            page_number,
            len(failures) + 1,
        )
        failures.append(failure)
        page_failure_ids.append(failure.id)

    restore_regions = getattr(reader, "restore_regions", None)
    if callable(restore_regions):
        try:
            restore_started = time.perf_counter()
            regions = restore_regions(regions, page_number)
            _add_timing(timings, "restore", restore_started)
        except ReaderError as error:
            _add_timing(timings, "restore", restore_started)
            failure = _failure(
                "orientation",
                error.code,
                str(error),
                page_number,
                len(failures) + 1,
            )
            failures.append(failure)
            page_failure_ids.append(failure.id)
            regions = []
    page_needs_review = getattr(reader, "page_needs_review", None)
    reader_review = bool(callable(page_needs_review) and page_needs_review(page_number))
    needs_review = (
        bool(page_failure_ids)
        or reader_review
        or any(_region_needs_review(region) for region in regions)
    )
    return PageResult(
        page_number=page_number,
        width=width,
        height=height,
        reader=reader.name,
        route="review" if needs_review else "accept_local",
        text=render_evidence(regions),
        regions=regions,
        failure_ids=page_failure_ids,
    )


def _apply_stages(
    image_path: Path,
    page_number: int,
    regions: list[TextRegion],
    stages: Sequence[RegionStage],
    failures: list[Failure],
    page_failure_ids: list[str],
    timings: dict[str, float] | None = None,
) -> list[TextRegion]:
    for stage in stages:
        stage_started = time.perf_counter()
        try:
            candidate = stage.apply(image_path, page_number, copy.deepcopy(regions))
        except ReaderError as error:
            failure = _failure(
                stage.name,
                error.code,
                str(error),
                page_number,
                len(failures) + 1,
            )
            failures.append(failure)
            page_failure_ids.append(failure.id)
            continue
        finally:
            _add_timing(timings, f"stage.{stage.name}", stage_started)
        regions = candidate
    return regions


def _add_timing(
    timings: dict[str, float] | None,
    name: str,
    started: float,
) -> None:
    if timings is None:
        return
    timings[name] = timings.get(name, 0.0) + time.perf_counter() - started


def _region_needs_review(region: TextRegion) -> bool:
    if region.resolution != "resolved":
        return True
    cells = (region.structure or {}).get("cells", [])
    if isinstance(cells, list) and any(
        isinstance(cell, dict) and cell.get("resolution") != "resolved"
        for cell in cells
    ):
        return True
    text = " ".join(region.text.casefold().split())
    return any(
        " ".join(alternative.text.casefold().split()) != text
        for alternative in region.alternatives
    )


def _prepare_pages(
    source: Path,
    source_kind: str,
    temporary_dir: Path,
    pdf_dpi: int,
    pdftoppm_executable: str,
) -> list[Path]:
    if source_kind == "image":
        if source.suffix.lower() in {".tif", ".tiff"}:
            return _prepare_tiff_pages(source, temporary_dir)
        return [_normalize_exif_orientation(source, temporary_dir)]
    if pdf_dpi <= 0:
        raise PipelineError("render", "invalid_dpi", "PDF DPI must be positive")

    output_prefix = temporary_dir / "page"
    try:
        completed = subprocess.run(
            [
                pdftoppm_executable,
                "-r",
                str(pdf_dpi),
                "-png",
                str(source),
                str(output_prefix),
            ],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
    except FileNotFoundError as error:
        raise PipelineError("render", "renderer_unavailable", str(error)) from error
    except subprocess.TimeoutExpired as error:
        raise PipelineError(
            "render", "renderer_timeout", "pdftoppm exceeded 180 seconds"
        ) from error

    if completed.returncode != 0:
        message = completed.stderr.strip() or "pdftoppm returned no error message"
        raise PipelineError("render", "renderer_failed", message)

    numbered_pages = []
    for page_path in temporary_dir.glob("page-*.png"):
        match = re.fullmatch(r"page-(\d+)\.png", page_path.name)
        if match:
            numbered_pages.append((int(match.group(1)), page_path))
    if not numbered_pages:
        raise PipelineError("render", "no_pages", "pdftoppm produced no pages")

    numbered_pages.sort(key=lambda item: item[0])
    return [page_path for _, page_path in numbered_pages]


def _prepare_tiff_pages(source: Path, temporary_dir: Path) -> list[Path]:
    pages: list[Path] = []
    try:
        with Image.open(source) as image:
            for index in range(image.n_frames):
                image.seek(index)
                frame = ImageOps.exif_transpose(image.copy())
                if frame.mode not in {
                    "1",
                    "L",
                    "LA",
                    "P",
                    "RGB",
                    "RGBA",
                    "I",
                    "I;16",
                }:
                    frame = frame.convert("RGB")
                output = temporary_dir / f"page-{index + 1}.png"
                frame.save(output, format="PNG")
                pages.append(output)
    except (EOFError, OSError, UnidentifiedImageError, ValueError) as error:
        raise PipelineError("image", "invalid_image", str(error)) from error
    if not pages:
        raise PipelineError("image", "no_pages", "TIFF contained no frames")
    return pages


def _normalize_exif_orientation(source: Path, temporary_dir: Path) -> Path:
    try:
        with Image.open(source) as image:
            orientation = image.getexif().get(274, 1)
            if orientation not in range(2, 9):
                return source

            normalized = ImageOps.exif_transpose(image)
            if normalized.mode not in {"1", "L", "LA", "P", "RGB", "RGBA", "I", "I;16"}:
                normalized = normalized.convert("RGB")
            output = temporary_dir / "page-1.png"
            normalized.save(output, format="PNG")
            return output
    except (OSError, UnidentifiedImageError):
        return source


def _source_kind(source: Path) -> str:
    suffix = source.suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix in IMAGE_SUFFIXES:
        return "image"
    return "unsupported"


def _failure(
    stage: str,
    code: str,
    message: str,
    page_number: int | None = None,
    sequence: int = 1,
) -> Failure:
    return Failure(
        id=f"failure-{sequence}",
        stage=stage,
        code=code,
        message=message,
        page_number=page_number,
    )


def _document_status(result: DocumentResult) -> str:
    if not result.failures:
        return "success"
    successful_pages = sum(not page.failure_ids for page in result.pages)
    return "partial" if successful_pages else "failed"
