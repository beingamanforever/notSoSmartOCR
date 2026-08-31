"""Export OCR predictions in OmniDocBench end-to-end Markdown format."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

from ocr_pipeline.contracts import DocumentResult
from ocr_pipeline.pipeline import process_document
from ocr_pipeline.providers import (
    GLMOCRDirectReader,
    GLMOCRReader,
    LocalReader,
    PaddleOCRVLReader,
    TesseractReader,
)

if __package__:
    from experiments.public_benchmark import reader_config
else:
    from public_benchmark import reader_config

REPORT_NAME = "run_report.json"
READER_CHOICES = ("tesseract", "paddleocr-vl", "glm-ocr", "glm-ocr-direct")


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
    parser.add_argument("--language", default="eng")
    parser.add_argument("--tesseract", default="tesseract")
    parser.add_argument("--backend", default="native")
    parser.add_argument("--device")
    parser.add_argument("--ocr-api-host")
    parser.add_argument("--ocr-api-port", type=_positive_int)
    parser.add_argument("--layout-device")
    parser.add_argument("--max-new-tokens", type=_positive_int, default=8192)
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
) -> dict[str, object]:
    records = json.loads(annotations.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("OmniDocBench annotations must be a JSON list")

    selected = [record for record in records if _matches(record, attributes or {})]
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise ValueError("No OmniDocBench pages matched the selection")

    output_dir.mkdir(parents=True, exist_ok=True)
    pages = []
    counts = {"success": 0, "partial": 0, "failed": 0}
    for index, record in enumerate(selected):
        annotated_path = _image_path(record, index)
        source = image_root / annotated_path
        started = time.perf_counter()
        try:
            result = process_document(source, reader)
            status = result.status
            failures = [failure.__dict__ for failure in result.failures]
            markdown = _markdown(result)
        except Exception as error:
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

        prediction_name = Path(annotated_path).stem + ".md"
        (output_dir / prediction_name).write_text(markdown, encoding="utf-8")
        counts[status] += 1
        pages.append(
            {
                "index": index,
                "image_path": annotated_path,
                "prediction": prediction_name,
                "status": status,
                "failures": failures,
                "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            }
        )

    report: dict[str, object] = {
        "dataset": "OmniDocBench",
        "dataset_revision": dataset_revision,
        "reader": reader.name,
        "run_config": {
            **reader_config(reader),
            "limit": limit,
            "attributes": dict(sorted((attributes or {}).items())),
        },
        "attempted": len(pages),
        **counts,
        "pages": pages,
    }
    (output_dir / REPORT_NAME).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _markdown(result: DocumentResult) -> str:
    regions = [
        region
        for page in result.pages
        for region in sorted(page.regions, key=lambda item: item.reading_order)
        if region.text.strip()
    ]
    if not regions:
        return ""
    return "\n\n".join(region.text.strip() for region in regions) + "\n"


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
    return GLMOCRDirectReader(max_new_tokens=args.max_new_tokens)


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return number


if __name__ == "__main__":
    raise SystemExit(main())
