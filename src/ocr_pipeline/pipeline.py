"""Ordered PDF and image ingestion."""

from __future__ import annotations

import copy
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError

from .cascade import add_runtime_risk_evidence
from .contracts import (
    ALTERNATIVE_DECISION_STATES,
    BoundingBox,
    DocumentResult,
    Failure,
    PageResult,
    RegionStage,
    TextRegion,
)
from .cross_page_tables import CrossPageTableStage
from .providers import LocalReader, ReaderError
from .rendering import render_evidence

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
DEFAULT_PDF_DPI = 300

RENDER_WORKERS = min(8, os.cpu_count() or 1)


@dataclass(frozen=True)
class _PreparedPages:
    source: Path
    pages: tuple[Path, ...]


def _layout_reader(url: str, device: str = "cuda") -> Any:
    """Falcon regions plus docTR words, so a scan's text no region holds is still read."""
    from functools import partial

    from .falcon_layout import FalconLayoutReader
    from .text_detection import detect_words

    return FalconLayoutReader(url, word_detector=partial(detect_words, device=device))


def falcon_pipeline(
    layout_url: str,
    *,
    device: str = "cuda",
    page_workers: int = 4,
    layout_reader_factory: Callable[[str], Any] | None = None,
    detector_factory: Callable[..., Any] | None = None,
) -> tuple[LocalReader, tuple[RegionStage, ...]]:
    """The served composition: docTR orientation, Heron and Falcon regions, assembly."""
    from .evidence_layout import EvidenceLayoutStage
    from .orientation import DocTROrientationDetector, OrientationReader
    from .risk import EvidenceRiskStage

    layout_reader = (
        layout_reader_factory(layout_url)
        if layout_reader_factory
        else _layout_reader(layout_url, device)
    )
    layout_reader.check_health()
    reader = OrientationReader(
        layout_reader,
        detector=(detector_factory or DocTROrientationDetector)(device=device),
        defer_restore=True,
        batch_size=page_workers,
    )
    stages = (
        EvidenceRiskStage(),
        EvidenceLayoutStage(),
    )
    return reader, stages


def process_document(
    source: str | Path,
    reader: LocalReader,
    *,
    pdf_dpi: int = DEFAULT_PDF_DPI,
    pdftoppm_executable: str = "pdftoppm",
    stages: Sequence[RegionStage] = (),
    cross_page_table_stage: CrossPageTableStage | None = None,
    timings: dict[str, float] | None = None,
    stage_execution: list[dict[str, object]] | None = None,
    page_execution: list[dict[str, object]] | None = None,
    prepared_pages: _PreparedPages | None = None,
    max_pages: int | None = None,
    max_page_pixels: int | None = None,
    max_pixels: int | None = None,
) -> DocumentResult:
    if timings is not None:
        timings.clear()
    if stage_execution is not None:
        stage_execution.clear()
    if page_execution is not None:
        page_execution.clear()
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
        with ExitStack() as stack:
            if prepared_pages is None:
                temporary_dir = stack.enter_context(
                    tempfile.TemporaryDirectory(prefix="ocr-pipeline-")
                )
                prepare_started = time.perf_counter()
                try:
                    prepared = _prepare_pages(
                        source_path,
                        source_kind,
                        Path(temporary_dir),
                        pdf_dpi,
                        pdftoppm_executable,
                        max_pages=max_pages,
                        max_page_pixels=max_page_pixels,
                        max_pixels=max_pixels,
                    )
                finally:
                    _add_timing(timings, "prepare", prepare_started)
            else:
                prepared = prepared_pages
            if prepared.source.resolve() != source_path.resolve():
                raise PipelineError(
                    "render",
                    "prepared_source_mismatch",
                    "Prepared pages belong to a different source",
                )
            pages = list(prepared.pages)
            if not pages:
                raise PipelineError(
                    "render", "no_pages", "Prepared input contained no pages"
                )
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
            pages_queued = time.perf_counter()
            for page_number, page_path in enumerate(pages, start=1):
                page_started = time.perf_counter()
                page_timings: dict[str, float] = {}
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
                        page_timings,
                        stage_execution,
                    )
                )
                _merge_timings(timings, page_timings)
                if page_execution is not None:
                    page_execution.append(
                        {
                            "page_number": page_number,
                            "queue_seconds": page_started - pages_queued,
                            "execution_seconds": time.perf_counter() - page_started,
                            "steps": page_timings,
                            "batched_reader": batch_results is not None,
                        }
                    )
            if cross_page_table_stage is not None and len(pages) > 1:
                continuation_started = time.perf_counter()
                execution: dict[str, object] = {
                    "page_number": None,
                    "stage": cross_page_table_stage.name,
                    "status": "fired",
                    "input_regions": sum(len(page.regions) for page in result.pages),
                    "output_regions": sum(len(page.regions) for page in result.pages),
                    "added_regions": 0,
                    "removed_regions": 0,
                    "modified_regions": 0,
                    "added_artifacts": 0,
                }
                try:
                    result.table_continuations = cross_page_table_stage.apply(
                        pages,
                        result.pages,
                    )
                    execution["added_artifacts"] = len(result.table_continuations)
                    if result.table_continuations:
                        execution["status"] = "productive"
                except ReaderError as error:
                    execution["status"] = "failed"
                    execution["failure_code"] = error.code
                    result.failures.append(
                        _failure(
                            cross_page_table_stage.name,
                            error.code,
                            str(error),
                            sequence=len(result.failures) + 1,
                        )
                    )
                    for page in result.pages:
                        page.route = "review"
                finally:
                    _add_timing(
                        timings,
                        f"stage.{cross_page_table_stage.name}",
                        continuation_started,
                    )
                    execution["elapsed_seconds"] = (
                        time.perf_counter() - continuation_started
                    )
                    if stage_execution is not None:
                        stage_execution.append(execution)
    except PipelineError as error:
        result.failures.append(_failure(error.stage, error.code, str(error)))

    review_pages = add_runtime_risk_evidence(result)
    for page in result.pages:
        if page.page_number in review_pages:
            page.route = "review"

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
    stage_execution: list[dict[str, object]] | None = None,
) -> PageResult:
    page_failure_ids: list[str] = []
    try:
        with Image.open(image_path) as image:
            width, height = image.size
    except (OSError, UnidentifiedImageError) as error:
        _record_skipped_stages(
            stage_execution,
            stages,
            page_number,
            0,
            "invalid_image",
        )
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
                stage_execution,
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
        _record_skipped_stages(
            stage_execution,
            stages,
            page_number,
            len(regions),
            error.code,
        )

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
    recover_regions = getattr(reader, "recover_regions", None)
    if callable(recover_regions):
        recovery_started = time.perf_counter()
        preserved_regions = regions
        try:
            regions = recover_regions(
                image_path,
                page_number,
                copy.deepcopy(regions),
            )
        except ReaderError as error:
            failure = _failure(
                "ocr",
                error.code,
                str(error),
                page_number,
                len(failures) + 1,
            )
            failures.append(failure)
            page_failure_ids.append(failure.id)
            regions = preserved_regions
            record_failure = getattr(reader, "record_recovery_failure", None)
            if callable(record_failure):
                record_failure(page_number, error)
        finally:
            _add_timing(timings, "recover", recovery_started)
    page_text = render_evidence(regions)
    readable_ids = {
        region.id
        for region in regions
        if region.resolution == "resolved"
        and (region.text.strip() or region.kind == "checkbox")
    }
    if not page_failure_ids and not readable_ids.intersection(page_text.evidence_ids):
        failure = _failure(
            "ocr",
            "no_text_detected",
            "No readable text was available after processing",
            page_number,
            len(failures) + 1,
        )
        failures.append(failure)
        page_failure_ids.append(failure.id)
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
        text=page_text,
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
    stage_execution: list[dict[str, object]] | None = None,
) -> list[TextRegion]:
    for stage in stages:
        stage_started = time.perf_counter()
        run: dict[str, object] = {
            "page_number": page_number,
            "stage": stage.name,
            "status": "fired",
            "input_regions": len(regions),
            "output_regions": len(regions),
            "added_regions": 0,
            "removed_regions": 0,
            "modified_regions": 0,
        }
        try:
            candidate = stage.apply(image_path, page_number, copy.deepcopy(regions))
            before = {region.id: region for region in regions}
            after = {region.id: region for region in candidate}
            added = after.keys() - before.keys()
            removed = before.keys() - after.keys()
            modified = sum(
                before[region_id] != after[region_id]
                for region_id in before.keys() & after.keys()
            )
            run.update(
                {
                    "status": "productive" if added or removed or modified else "fired",
                    "output_regions": len(candidate),
                    "added_regions": len(added),
                    "removed_regions": len(removed),
                    "modified_regions": modified,
                }
            )
        except ReaderError as error:
            run.update({"status": "failed", "failure_code": error.code})
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
            run["elapsed_seconds"] = _add_timing(
                timings,
                f"stage.{stage.name}",
                stage_started,
            )
            if stage_execution is not None:
                stage_execution.append(run)
        regions = candidate
    return regions


def _record_skipped_stages(
    stage_execution: list[dict[str, object]] | None,
    stages: Sequence[RegionStage],
    page_number: int,
    input_regions: int,
    reason: str,
) -> None:
    if stage_execution is None:
        return
    recorded = {(run["page_number"], run["stage"]) for run in stage_execution}
    for stage in stages:
        if (page_number, stage.name) in recorded:
            continue
        stage_execution.append(
            {
                "page_number": page_number,
                "stage": stage.name,
                "status": "skipped",
                "input_regions": input_regions,
                "output_regions": input_regions,
                "added_regions": 0,
                "removed_regions": 0,
                "modified_regions": 0,
                "elapsed_seconds": 0.0,
                "skip_reason": reason,
            }
        )


def _add_timing(
    timings: dict[str, float] | None,
    name: str,
    started: float,
) -> float:
    elapsed = time.perf_counter() - started
    if timings is not None:
        timings[name] = timings.get(name, 0.0) + elapsed
    return elapsed


def _merge_timings(
    timings: dict[str, float] | None,
    additions: dict[str, float],
) -> None:
    if timings is None:
        return
    for name, seconds in additions.items():
        timings[name] = timings.get(name, 0.0) + seconds


def _region_needs_review(region: TextRegion) -> bool:
    if region.resolution != "resolved":
        return True
    provenance = region.text_provenance or {}
    if (
        provenance.get("recognition_agreement") is False
        or provenance.get("recognition_status") == "unavailable"
    ):
        return True
    structure = region.structure or {}
    if structure.get("grid_disagreement"):
        return True
    if structure.get("coverage_status") == "insufficient_control_group":
        return True
    if structure.get("block_type") == "formula" and structure.get(
        "formula_recognition"
    ) not in {"specialist_supported", "human_accepted"}:
        return True
    handwriting_review = structure.get("handwriting_review")
    if isinstance(handwriting_review, dict) and handwriting_review.get("required"):
        return True
    cells = structure.get("cells", [])
    if isinstance(cells, list) and any(
        not isinstance(cell, dict) or _cell_needs_review(cell) for cell in cells
    ):
        return True
    text = " ".join(region.text.casefold().split())
    return any(
        alternative.decision_state not in ALTERNATIVE_DECISION_STATES
        or (
            alternative.decision_state == "pending"
            and " ".join(alternative.text.casefold().split()) != text
        )
        for alternative in region.alternatives
    )


def _cell_needs_review(cell: dict[str, object]) -> bool:
    if cell.get("resolution") != "resolved":
        return True
    text = " ".join(str(cell.get("text", "")).casefold().split())
    alternatives = cell.get("alternatives", [])
    if not isinstance(alternatives, list):
        return True
    for alternative in alternatives:
        if not isinstance(alternative, dict):
            return True
        decision_state = alternative.get("decision_state", "pending")
        if decision_state not in ALTERNATIVE_DECISION_STATES:
            return True
        alternative_text = alternative.get("text")
        if decision_state == "pending" and (
            not isinstance(alternative_text, str)
            or " ".join(alternative_text.casefold().split()) != text
        ):
            return True
    return False


def _prepare_pages(
    source: Path,
    source_kind: str,
    temporary_dir: Path,
    pdf_dpi: int,
    pdftoppm_executable: str,
    *,
    max_pages: int | None = None,
    max_page_pixels: int | None = None,
    max_pixels: int | None = None,
) -> _PreparedPages:
    _validate_resource_limits(max_pages, max_page_pixels, max_pixels)
    if source_kind == "image":
        if source.suffix.lower() in {".tif", ".tiff"}:
            pages = _prepare_tiff_pages(
                source,
                temporary_dir,
                max_pages=max_pages,
                max_page_pixels=max_page_pixels,
                max_pixels=max_pixels,
            )
        else:
            _validate_image_limits(source, max_pages, max_pixels)
            pages = [_canonical_image_page(source, temporary_dir, max_page_pixels)]
        return _PreparedPages(source.resolve(), tuple(pages))
    if pdf_dpi <= 0:
        raise PipelineError("render", "invalid_dpi", "PDF DPI must be positive")
    if shutil.which(pdftoppm_executable) is None:
        raise PipelineError(
            "render", "renderer_unavailable", f"{pdftoppm_executable} not found"
        )

    page_count = _validate_pdf_limits(source, pdf_dpi, max_pages, max_pixels)
    if page_count is None:
        page_count = _pdf_page_count(source)
    # pdftoppm renders one page at a time, so contiguous page ranges render in
    # parallel processes: 44 contract pages took 60 s serially and 13 s eight-way.
    workers = max(1, min(RENDER_WORKERS, page_count))
    bounds = [page_count * worker // workers for worker in range(workers + 1)]
    ranges = [(start + 1, end) for start, end in pairwise(bounds) if end > start]
    with ThreadPoolExecutor(len(ranges)) as pool:
        list(
            pool.map(
                lambda pages: _render_pdf_pages(
                    pdftoppm_executable, pdf_dpi, source, temporary_dir / "page", pages
                ),
                ranges,
            )
        )

    numbered_pages = []
    for page_path in temporary_dir.glob("page-*.png"):
        match = re.fullmatch(r"page-(\d+)\.png", page_path.name)
        if match:
            numbered_pages.append((int(match.group(1)), page_path))
    if not numbered_pages:
        raise PipelineError("render", "no_pages", "pdftoppm produced no pages")

    numbered_pages.sort(key=lambda item: item[0])
    for _, page_path in numbered_pages:
        with Image.open(page_path) as image:
            canonical = (
                None
                if _is_canonical(image, max_page_pixels)
                else _canonical_page(image, max_page_pixels)
            )
        if canonical is not None:
            canonical.save(page_path, format="PNG")
    pages = [page_path for _, page_path in numbered_pages]
    from .text_layer import write_cell_sidecars

    try:
        write_cell_sidecars(source, pages)
    except Exception as error:
        raise PipelineError("render", "text_layer_failed", str(error)) from error
    return _PreparedPages(source.resolve(), tuple(pages))


def _render_pdf_pages(
    executable: str, dpi: int, source: Path, prefix: Path, pages: tuple[int, int]
) -> None:
    command = [
        executable,
        "-cropbox",
        "-r",
        str(dpi),
        "-png",
        "-f",
        str(pages[0]),
        "-l",
        str(pages[1]),
    ]
    try:
        completed = subprocess.run(
            [*command, str(source), str(prefix)],
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


def _pdf_page_count(source: Path) -> int:
    match = re.search(r"^Pages:\s+(\d+)\s*$", _pdfinfo(source), re.MULTILINE)
    if match is None or int(match.group(1)) < 1:
        raise PipelineError(
            "render", "metadata_failed", "pdfinfo did not report a page count"
        )
    return int(match.group(1))


def _validate_resource_limits(
    max_pages: int | None,
    max_page_pixels: int | None,
    max_pixels: int | None,
) -> None:
    if max_pages is not None and max_pages <= 0:
        raise ValueError("max_pages must be positive")
    if max_page_pixels is not None and max_page_pixels <= 0:
        raise ValueError("max_page_pixels must be positive")
    if max_pixels is not None and max_pixels <= 0:
        raise ValueError("max_pixels must be positive")


def _validate_image_limits(
    source: Path,
    max_pages: int | None,
    max_pixels: int | None,
) -> None:
    try:
        with Image.open(source) as image:
            _check_page_limit(1, max_pages)
            _check_pixel_limit(image.width * image.height, max_pixels, "Document")
    except Image.DecompressionBombError as error:
        raise PipelineError(
            "ingest",
            "pixel_limit_exceeded",
            "Image exceeds the decoded pixel safety limit",
        ) from error
    except (OSError, UnidentifiedImageError) as error:
        raise PipelineError("image", "invalid_image", str(error)) from error


def _validate_pdf_limits(
    source: Path,
    pdf_dpi: int,
    max_pages: int | None,
    max_pixels: int | None,
) -> int | None:
    if max_pages is None and max_pixels is None:
        return None

    summary = _pdfinfo(source)
    match = re.search(r"^Pages:\s+(\d+)\s*$", summary, re.MULTILINE)
    if match is None:
        raise PipelineError(
            "render",
            "metadata_failed",
            "pdfinfo did not report a page count",
        )
    page_count = int(match.group(1))
    if page_count < 1:
        raise PipelineError("render", "no_pages", "PDF contained no pages")
    _check_page_limit(page_count, max_pages)
    if max_pixels is None:
        return page_count

    details = _pdfinfo(source, page_count)
    sizes = re.findall(
        r"^Page\s+\d+\s+size:\s+([0-9.]+)\s+x\s+([0-9.]+)\s+pts",
        details,
        re.MULTILINE,
    )
    if len(sizes) != page_count:
        raise PipelineError(
            "render",
            "metadata_failed",
            "pdfinfo did not report every page size",
        )
    scale = pdf_dpi / 72
    page_pixel_counts = [
        math.ceil(float(width) * scale) * math.ceil(float(height) * scale)
        for width, height in sizes
    ]
    _check_pixel_limit(sum(page_pixel_counts), max_pixels, "Document")
    return page_count


def _pdfinfo(source: Path, page_count: int | None = None) -> str:
    command = ["pdfinfo"]
    if page_count is not None:
        command.extend(["-f", "1", "-l", str(page_count)])
    command.append(str(source))
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except FileNotFoundError as error:
        raise PipelineError("render", "renderer_unavailable", str(error)) from error
    except subprocess.TimeoutExpired as error:
        raise PipelineError(
            "render", "renderer_timeout", "pdfinfo exceeded 30 seconds"
        ) from error
    if completed.returncode != 0:
        message = completed.stderr.strip() or "pdfinfo returned no error message"
        raise PipelineError("render", "renderer_failed", message)
    return completed.stdout


def _check_page_limit(page_count: int, max_pages: int | None) -> None:
    if max_pages is not None and page_count > max_pages:
        raise PipelineError(
            "ingest",
            "page_limit_exceeded",
            f"Document has {page_count} pages; the limit is {max_pages}",
        )


def _check_pixel_limit(
    pixel_count: int,
    max_pixels: int | None,
    scope: str,
) -> None:
    if max_pixels is not None and pixel_count > max_pixels:
        raise PipelineError(
            "ingest",
            "pixel_limit_exceeded",
            f"{scope} decodes to {pixel_count} pixels; the limit is {max_pixels}",
        )


def _prepare_tiff_pages(
    source: Path,
    temporary_dir: Path,
    *,
    max_pages: int | None = None,
    max_page_pixels: int | None = None,
    max_pixels: int | None = None,
) -> list[Path]:
    pages: list[Path] = []
    try:
        with Image.open(source) as image:
            _check_page_limit(image.n_frames, max_pages)
            decoded_pixels = 0
            for index in range(image.n_frames):
                image.seek(index)
                decoded_pixels += image.width * image.height
                _check_pixel_limit(decoded_pixels, max_pixels, "Document")
                frame = _canonical_page(image.copy(), max_page_pixels)
                output = temporary_dir / f"page-{index + 1}.png"
                frame.save(output, format="PNG")
                pages.append(output)
    except Image.DecompressionBombError as error:
        raise PipelineError(
            "ingest",
            "pixel_limit_exceeded",
            "Image exceeds the decoded pixel safety limit",
        ) from error
    except (EOFError, OSError, UnidentifiedImageError, ValueError) as error:
        raise PipelineError("image", "invalid_image", str(error)) from error
    if not pages:
        raise PipelineError("image", "no_pages", "TIFF contained no frames")
    return pages


def _canonical_image_page(
    source: Path, temporary_dir: Path, max_page_pixels: int | None
) -> Path:
    try:
        with Image.open(source) as image:
            if _is_canonical(image, max_page_pixels):
                return source
            page = _canonical_page(image, max_page_pixels)
    except (OSError, UnidentifiedImageError) as error:
        raise PipelineError("image", "invalid_image", str(error)) from error
    output = temporary_dir / "page-1.png"
    page.save(output, format="PNG")
    return output


def _is_canonical(image: Image.Image, max_page_pixels: int | None) -> bool:
    return (
        image.mode in {"RGB", "L"}
        and image.getexif().get(274, 1) not in range(2, 9)
        and (max_page_pixels is None or image.width * image.height <= max_page_pixels)
    )


def _canonical_page(image: Image.Image, max_page_pixels: int | None) -> Image.Image:
    """One 8-bit RGB page raster, so every stage reads the same upright pixels."""
    page = ImageOps.exif_transpose(image)
    if page.mode in {"I;16", "I;16B", "I;16L", "I", "F"}:
        page = page.convert("F") if page.mode == "F" else page.convert("I")
        top = page.getextrema()[1] or 1
        # Rescale deep grayscale by its bit depth instead of clipping it to white.
        scale = 255 / (65535 if top <= 65535 else top) if top > 255 else 1
        page = page.point(lambda value: value * scale).convert("L")
    if page.mode in {"RGBA", "LA", "PA"} or (
        page.mode == "P" and "transparency" in page.info
    ):
        rgba = page.convert("RGBA")
        page = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba)
    page = page.convert("RGB")
    pixels = page.width * page.height
    if max_page_pixels is not None and pixels > max_page_pixels:
        scale = math.sqrt(max_page_pixels / pixels)
        size = (max(1, int(page.width * scale)), max(1, int(page.height * scale)))
        page = page.resize(size, Image.Resampling.LANCZOS)
    return page


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
