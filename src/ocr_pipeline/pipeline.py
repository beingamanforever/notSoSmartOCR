"""Ordered PDF and image ingestion."""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from .contracts import DocumentResult, EvidenceText, Failure, PageResult
from .providers import LocalReader, ReaderError

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def process_document(
    source: str | Path,
    reader: LocalReader,
    *,
    pdf_dpi: int = 300,
    pdftoppm_executable: str = "pdftoppm",
) -> DocumentResult:
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
            pages = _prepare_pages(
                source_path,
                source_kind,
                Path(temporary_dir),
                pdf_dpi,
                pdftoppm_executable,
            )
            for page_number, page_path in enumerate(pages, start=1):
                result.pages.append(
                    _read_page(page_path, page_number, reader, result.failures)
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
            text=EvidenceText(value="", evidence_ids=[]),
            failure_ids=[failure.id],
        )

    try:
        regions = reader.read(image_path, page_number)
    except ReaderError as error:
        failure = _failure(
            "ocr", error.code, str(error), page_number, len(failures) + 1
        )
        failures.append(failure)
        page_failure_ids.append(failure.id)
        regions = []
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

    evidence_ids = [region.id for region in regions]
    return PageResult(
        page_number=page_number,
        width=width,
        height=height,
        reader=reader.name,
        route="review" if page_failure_ids else "accept_local",
        text=EvidenceText(
            value=" ".join(region.text for region in regions),
            evidence_ids=evidence_ids,
        ),
        regions=regions,
        failure_ids=page_failure_ids,
    )


def _prepare_pages(
    source: Path,
    source_kind: str,
    temporary_dir: Path,
    pdf_dpi: int,
    pdftoppm_executable: str,
) -> list[Path]:
    if source_kind == "image":
        return [source]
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
