"""Export OCR predictions in OmniDocBench end-to-end Markdown format."""

from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from ocr_pipeline.contracts import DocumentResult, PageResult, TextRegion
from ocr_pipeline.pipeline import process_document
from ocr_pipeline.providers import (
    GLMOCRDirectReader,
    GLMOCRReader,
    GraniteDoclingReader,
    LocalReader,
    NemotronOCRV2Reader,
    PaddleOCRVLReader,
    TesseractReader,
)

if __package__:
    from experiments.public_benchmark import reader_config
else:
    from public_benchmark import reader_config

REPORT_NAME = "run_report.json"
STRUCTURED_DIR_NAME = "structured"
READER_CHOICES = (
    "tesseract",
    "paddleocr-vl",
    "glm-ocr",
    "glm-ocr-direct",
    "granite-docling",
    "nemotron-ocr-v2",
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export OmniDocBench page predictions as Markdown"
    )
    parser.add_argument("annotations", type=Path, help="OmniDocBench JSON file")
    parser.add_argument("image_root", type=Path, help="Root containing page images")
    parser.add_argument("output_dir", type=Path, help="Prediction output directory")
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument(
        "--limit", type=_positive_int, help="Run the first N selected pages"
    )
    parser.add_argument(
        "--attribute",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Select a page_attribute value; repeat for multiple attributes",
    )
    parser.add_argument("--reader", choices=READER_CHOICES, default="tesseract")
    parser.add_argument(
        "--workers",
        type=_positive_int,
        default=1,
        help="Concurrent page workers; values above one are supported by Tesseract only",
    )
    parser.add_argument("--language", default="eng")
    parser.add_argument("--tesseract", default="tesseract")
    parser.add_argument("--backend", default="native")
    parser.add_argument("--device")
    parser.add_argument("--ocr-api-host")
    parser.add_argument("--ocr-api-port", type=_positive_int)
    parser.add_argument("--layout-device")
    parser.add_argument("--max-new-tokens", type=_positive_int, default=8192)
    parser.add_argument(
        "--granite-output-format",
        choices=("text", "markdown"),
        default="markdown",
    )
    parser.add_argument(
        "--nemotron-language",
        choices=("multi", "en"),
        default="multi",
    )
    parser.add_argument(
        "--nemotron-merge-level",
        choices=("word", "sentence", "paragraph"),
        default="paragraph",
    )
    parser.add_argument("--use-doc-orientation-classify", action="store_true")
    parser.add_argument("--use-doc-unwarping", action="store_true")
    args = parser.parse_args(argv)

    try:
        report = export_predictions(
            args.annotations,
            args.image_root,
            args.output_dir,
            _reader_from_args(args),
            dataset_revision=args.dataset_revision,
            limit=args.limit,
            attributes=_parse_attributes(args.attribute),
            workers=args.workers,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    return 0 if report["failed"] == 0 and report["partial"] == 0 else 1


def export_predictions(
    annotations: Path,
    image_root: Path,
    output_dir: Path,
    reader: LocalReader,
    *,
    dataset_revision: str,
    limit: int | None = None,
    attributes: dict[str, str] | None = None,
    workers: int = 1,
) -> dict[str, object]:
    if not dataset_revision.strip():
        raise ValueError("OmniDocBench dataset revision must not be empty")
    if workers < 1:
        raise ValueError("Workers must be positive")
    if workers != 1 and not isinstance(reader, TesseractReader):
        raise ValueError("Concurrent OmniDocBench export supports Tesseract only")

    records = json.loads(annotations.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("OmniDocBench annotations must be a JSON list")

    selected = [record for record in records if _matches(record, attributes or {})]
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise ValueError("No OmniDocBench pages matched the selection")

    image_paths = [_image_path(record, index) for index, record in enumerate(selected)]
    prediction_names = [Path(image_path).stem + ".md" for image_path in image_paths]
    if len(set(prediction_names)) != len(prediction_names):
        raise ValueError("Selected OmniDocBench pages have duplicate prediction names")

    output_dir.mkdir(parents=True)
    structured_dir = output_dir / STRUCTURED_DIR_NAME
    structured_dir.mkdir()
    pages = []
    counts = {"success": 0, "partial": 0, "failed": 0}
    covered = 0
    abstained = 0
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        outcomes = executor.map(
            lambda path: _process_page(image_root / path, reader), image_paths
        )
        evaluated = list(outcomes)
    for index, (annotated_path, outcome) in enumerate(zip(image_paths, evaluated)):
        result, status, failures, markdown, latency_ms = outcome
        prediction_name = prediction_names[index]
        structured_name = Path(prediction_name).with_suffix(".json").name
        (output_dir / prediction_name).write_text(markdown, encoding="utf-8")
        (structured_dir / structured_name).write_text(
            json.dumps(
                _structured_result(
                    result,
                    case_id=annotated_path,
                    image_path=annotated_path,
                    status=status,
                    failures=failures,
                ),
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        counts[status] += 1
        is_covered = bool(markdown.strip())
        covered += is_covered
        abstained += not is_covered
        pages.append(
            {
                "index": index,
                "case_id": annotated_path,
                "image_path": annotated_path,
                "prediction": prediction_name,
                "structured_prediction": str(
                    Path(STRUCTURED_DIR_NAME) / structured_name
                ),
                "status": status,
                "covered": is_covered,
                "abstained": not is_covered,
                "failures": failures,
                "latency_ms": latency_ms,
            }
        )
    wall_latency_ms = (time.perf_counter() - started) * 1000

    latencies = [float(page["latency_ms"]) for page in pages]
    report: dict[str, object] = {
        "dataset": "OmniDocBench",
        "dataset_revision": dataset_revision,
        "reader": reader.name,
        "markdown_output_dir": str(output_dir),
        "structured_output_dir": str(structured_dir),
        "run_config": {
            **reader_config(reader),
            "markdown_assembly": (
                "tesseract_tsv_paragraphs"
                if isinstance(reader, TesseractReader)
                else "reader_regions"
            ),
            "workers": workers,
            "limit": limit,
            "attributes": dict(sorted((attributes or {}).items())),
        },
        "case_ids": image_paths,
        "attempted": len(pages),
        "covered": covered,
        "abstained": abstained,
        **counts,
        "latency_ms": {
            "p50": round(_percentile(latencies, 0.50), 3),
            "p95": round(_percentile(latencies, 0.95), 3),
        },
        "wall_latency_ms": round(wall_latency_ms, 3),
        "pages_per_second": round(len(pages) / (wall_latency_ms / 1000), 6)
        if wall_latency_ms > 0
        else 0.0,
        "pages": pages,
    }
    (output_dir / REPORT_NAME).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _process_page(
    source: Path, reader: LocalReader
) -> tuple[
    DocumentResult | None,
    str,
    list[dict[str, object]],
    str,
    float,
]:
    started = time.perf_counter()
    try:
        result = process_document(source, reader)
        status = result.status
        failures = [asdict(failure) for failure in result.failures]
        markdown = _markdown(result)
    except Exception as error:
        result = None
        status = "failed"
        failures = [
            {
                "id": "failure-1",
                "stage": "export",
                "code": "unhandled_error",
                "message": str(error),
                "page_number": None,
            }
        ]
        markdown = ""
    latency_ms = round((time.perf_counter() - started) * 1000, 3)
    return result, status, failures, markdown, latency_ms


def _structured_result(
    result: DocumentResult | None,
    *,
    case_id: str,
    image_path: str,
    status: str,
    failures: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "schema_version": result.schema_version if result is not None else 1,
        "case_id": case_id,
        "image_path": image_path,
        "document_id": result.document_id if result is not None else None,
        "source": result.source if result is not None else None,
        "status": status,
        "geometry_availability": _document_geometry(result),
        "pages": [_structured_page(page) for page in result.pages]
        if result is not None
        else [],
        "failures": failures,
    }


def _structured_page(page: PageResult) -> dict[str, object]:
    regions = sorted(page.regions, key=lambda region: region.reading_order)
    return {
        "page_number": page.page_number,
        "width": page.width,
        "height": page.height,
        "reader": page.reader,
        "route": page.route,
        "text": asdict(page.text),
        "regions": [_structured_region(region, page) for region in regions],
        "failure_ids": page.failure_ids,
        "geometry_availability": _page_geometry(page),
    }


def _structured_region(region: TextRegion, page: PageResult) -> dict[str, object]:
    return {
        **asdict(region),
        "geometry_availability": _region_geometry(region, page),
    }


def _document_geometry(result: DocumentResult | None) -> str:
    if result is None or not result.pages:
        return "unavailable"
    geometry = {_page_geometry(page) for page in result.pages}
    if geometry == {"available"}:
        return "available"
    if geometry == {"page_only"}:
        return "page_only"
    if geometry == {"unavailable"}:
        return "unavailable"
    return "partial"


def _page_geometry(page: PageResult) -> str:
    if page.width <= 0 or page.height <= 0 or not page.regions:
        return "unavailable"
    geometry = {_region_geometry(region, page) for region in page.regions}
    if geometry == {"available"}:
        return "available"
    if geometry == {"page_only"}:
        return "page_only"
    return "partial"


def _region_geometry(region: TextRegion, page: PageResult) -> str:
    box = region.bounding_box
    is_full_page = (
        box.left == 0
        and box.top == 0
        and box.right == page.width
        and box.bottom == page.height
    )
    return "page_only" if region.kind == "page_text" and is_full_page else "available"


def _markdown(result: DocumentResult) -> str:
    pages = [_page_markdown(page) for page in result.pages]
    pages = [page for page in pages if page]
    if not pages:
        return ""
    return "\n\n".join(pages) + "\n"


def _page_markdown(page: PageResult) -> str:
    regions = [
        region
        for region in sorted(page.regions, key=lambda item: item.reading_order)
        if region.text.strip()
    ]
    if not regions:
        return ""
    if all(_tesseract_line_key(region) is not None for region in regions):
        return _tesseract_markdown(regions)
    return "\n\n".join(region.text.strip() for region in regions)


def _tesseract_markdown(regions: list[TextRegion]) -> str:
    paragraphs: list[list[list[str]]] = []
    active_paragraph: tuple[int, int] | None = None
    active_key: tuple[int, int, int] | None = None
    for region in regions:
        key = _tesseract_line_key(region)
        if key is None:
            raise ValueError("Tesseract word is missing its TSV line identifier")
        paragraph = key[:2]
        if paragraph != active_paragraph:
            paragraphs.append([])
            active_paragraph = paragraph
            active_key = None
        if key != active_key:
            paragraphs[-1].append([])
            active_key = key
        paragraphs[-1][-1].append(region.text.strip())
    return "\n\n".join(
        " ".join(word for line in lines for word in line) for lines in paragraphs
    )


def _tesseract_line_key(region: TextRegion) -> tuple[int, int, int] | None:
    provenance = region.text_provenance
    if region.kind != "word" or not isinstance(provenance, dict):
        return None
    if provenance.get("method") != "tesseract_tsv":
        return None
    try:
        return (
            int(provenance["block_num"]),
            int(provenance["paragraph_num"]),
            int(provenance["line_num"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _image_path(record: object, index: int) -> str:
    if not isinstance(record, dict) or not isinstance(record.get("page_info"), dict):
        raise ValueError(f"Invalid page_info in annotation record {index}")
    image_path = record["page_info"].get("image_path")
    if not isinstance(image_path, str) or not image_path.strip():
        raise ValueError(f"Invalid image_path in annotation record {index}")
    return image_path


def _matches(record: object, attributes: dict[str, str]) -> bool:
    if not attributes:
        return True
    if not isinstance(record, dict) or not isinstance(record.get("page_info"), dict):
        return False
    page_attributes = record["page_info"].get("page_attribute")
    if not isinstance(page_attributes, dict):
        return False
    for key, expected in attributes.items():
        actual = page_attributes.get(key)
        if isinstance(actual, list):
            if expected not in {str(value) for value in actual}:
                return False
        elif str(actual) != expected:
            return False
    return True


def _parse_attributes(values: list[str]) -> dict[str, str]:
    attributes = {}
    for value in values:
        key, separator, expected = value.partition("=")
        if not separator or not key or not expected:
            raise ValueError(f"Invalid page attribute {value!r}; expected KEY=VALUE")
        attributes[key] = expected
    return attributes


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _reader_from_args(args: argparse.Namespace) -> LocalReader:
    if args.reader == "tesseract":
        return TesseractReader(language=args.language, executable=args.tesseract)
    if args.reader == "paddleocr-vl":
        return PaddleOCRVLReader(
            backend=args.backend,
            device=args.device,
            use_doc_orientation_classify=args.use_doc_orientation_classify or None,
            use_doc_unwarping=args.use_doc_unwarping or None,
        )
    if args.reader == "glm-ocr":
        return GLMOCRReader(
            ocr_api_host=args.ocr_api_host,
            ocr_api_port=args.ocr_api_port,
            layout_device=args.layout_device,
        )
    if args.reader == "glm-ocr-direct":
        return GLMOCRDirectReader(max_new_tokens=args.max_new_tokens)
    if args.reader == "granite-docling":
        return GraniteDoclingReader(
            max_new_tokens=args.max_new_tokens,
            output_format=args.granite_output_format,
        )
    return NemotronOCRV2Reader(
        language=args.nemotron_language,
        merge_level=args.nemotron_merge_level,
    )


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return number


if __name__ == "__main__":
    raise SystemExit(main())
