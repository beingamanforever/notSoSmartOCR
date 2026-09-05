"""Export document-level OCR predictions for MPDocBench-Parse."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from collections import defaultdict
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
        description="Export MPDocBench document predictions as Markdown"
    )
    parser.add_argument("annotations", type=Path, help="Official MPDocBench JSON file")
    parser.add_argument("image_root", type=Path, help="Root containing page images")
    parser.add_argument("output_dir", type=Path, help="New prediction run directory")
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--evaluator-revision", required=True)
    parser.add_argument(
        "--limit", type=_positive_int, help="Run the first N documents for development"
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
    parser.add_argument(
        "--granite-output-format",
        choices=("text", "markdown"),
        default="markdown",
    )
    parser.add_argument("--nemotron-language", choices=("multi", "en"), default="multi")
    parser.add_argument(
        "--nemotron-merge-level",
        choices=("word", "sentence", "paragraph"),
        default="paragraph",
    )
    parser.add_argument("--use-doc-orientation-classify", action="store_true")
    parser.add_argument("--use-doc-unwarping", action="store_true")
    parser.add_argument("--control", choices=("oracle", "empty"))
    parser.add_argument("--evaluator-root", type=Path)
    args = parser.parse_args(argv)

    try:
        if args.control:
            if args.evaluator_root is None:
                raise ValueError("--evaluator-root is required with --control")
            report = export_control_predictions(
                args.annotations,
                args.output_dir,
                evaluator_root=args.evaluator_root,
                control=args.control,
                dataset_revision=args.dataset_revision,
                evaluator_revision=args.evaluator_revision,
                limit=args.limit,
            )
        else:
            report = export_predictions(
                args.annotations,
                args.image_root,
                args.output_dir,
                _reader_from_args(args),
                dataset_revision=args.dataset_revision,
                evaluator_revision=args.evaluator_revision,
                limit=args.limit,
            )
    except (
        ImportError,
        OSError,
        RuntimeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        parser.error(str(error))
    return (
        0 if args.control or (report["failed"] == 0 and report["partial"] == 0) else 1
    )


def export_control_predictions(
    annotations: Path,
    output_dir: Path,
    *,
    evaluator_root: Path,
    control: str,
    dataset_revision: str,
    evaluator_revision: str,
    limit: int | None = None,
    official_metrics: object | None = None,
) -> dict[str, object]:
    if control not in {"oracle", "empty"}:
        raise ValueError(f"Unsupported MPDocBench control: {control}")
    if not dataset_revision.strip():
        raise ValueError("MPDocBench dataset revision must not be empty")
    if not evaluator_revision.strip():
        raise ValueError("MPDocBench evaluator revision must not be empty")

    records = json.loads(annotations.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("MPDocBench annotations must be a JSON list")
    selected = records[:limit] if limit is not None else records
    if not selected:
        raise ValueError("No MPDocBench documents were selected")
    documents = [_document_info(record, index) for index, record in enumerate(selected)]
    prediction_names = [Path(item["image_path"]).stem + ".md" for item in documents]
    if len(prediction_names) != len(set(prediction_names)):
        raise ValueError(
            "Selected MPDocBench documents have duplicate prediction names"
        )

    metrics = official_metrics or _OfficialMetrics(evaluator_root)
    markdown_predictions = []
    document_records = []
    relation_pairs = {
        "text": [set(), set()],
        "table": [set(), set()],
    }
    text_continuation_scores = []
    table_continuation_scores = []
    text_continuation_exact = []
    table_continuation_exact = []
    text_cross_page_exact = []
    table_cross_page_exact = []
    text_cross_page_continuations = 0
    table_cross_page_continuations = 0
    heading_scores = []

    for index, (record, document, prediction_name) in enumerate(
        zip(selected, documents, prediction_names, strict=True)
    ):
        if not isinstance(record, dict):
            raise ValueError(f"Invalid MPDocBench document {index}")
        table_gt, table_groups, table_map = metrics.extract_table_relations(record)
        text_gt, text_groups, text_map = metrics.extract_text_relations(record)
        markdown = (
            _oracle_markdown(record, table_groups, text_groups)
            if control == "oracle"
            else ""
        )
        markdown_predictions.append((prediction_name, markdown))

        heading_gt = _heading_ground_truth(record)
        heading_pred = _heading_blocks(markdown)
        heading_score, _ = metrics.head_teds.evaluate(heading_pred, heading_gt)
        heading_scores.append(float(heading_score))

        per_document_relations = {}
        for category, gt_pairs, groups, annotation_map in (
            ("text", text_gt, text_groups, text_map),
            ("table", table_gt, table_groups, table_map),
        ):
            predicted_values = (
                _oracle_category_values(record, groups, category)
                if control == "oracle"
                else []
            )
            predicted_pairs = metrics.detect_relations(
                record,
                category,
                predicted_values,
            )
            _, gt_cross = metrics.split_pairs_by_page(gt_pairs, annotation_map)
            _, pred_cross = metrics.split_pairs_by_page(predicted_pairs, annotation_map)
            relation_pairs[category][0].update(_namespace_pairs(gt_cross, index))
            relation_pairs[category][1].update(_namespace_pairs(pred_cross, index))
            per_document_relations[category] = {
                "ground_truth_cross_page_pairs": len(gt_cross),
                "predicted_cross_page_pairs": len(pred_cross),
            }

        for group in text_groups:
            items = _group_items(text_map, group)
            ground_truth = "".join(str(item.get("text", "")) for item in items)
            prediction = ground_truth if control == "oracle" else ""
            text_continuation_scores.append(
                float(metrics.text_edit_distance(ground_truth, prediction))
            )
            exact_match = float(prediction == ground_truth)
            text_continuation_exact.append(exact_match)
            crosses_pages = _crosses_pages(items)
            text_cross_page_continuations += crosses_pages
            if crosses_pages:
                text_cross_page_exact.append(exact_match)
        for group in table_groups:
            items = _group_items(table_map, group)
            ground_truth = str(items[0].get("merged_html", items[0].get("html", "")))
            prediction = ground_truth if control == "oracle" else ""
            normalized_prediction = metrics.normalize_table_for_teds(prediction)
            normalized_ground_truth = metrics.normalize_table_for_teds(ground_truth)
            table_continuation_scores.append(
                float(
                    metrics.table_teds.evaluate(
                        normalized_prediction,
                        normalized_ground_truth,
                    )
                )
            )
            exact_match = float(normalized_prediction == normalized_ground_truth)
            table_continuation_exact.append(exact_match)
            crosses_pages = _crosses_pages(items)
            table_cross_page_continuations += crosses_pages
            if crosses_pages:
                table_cross_page_exact.append(exact_match)

        document_records.append(
            {
                "document_id": document["image_path"],
                "prediction": prediction_name,
                "page_count": len(document["images_list"]),
                "covered": bool(markdown),
                "abstained": not bool(markdown),
                "HeadTEDS": round(float(heading_score), 6),
                "cross_page_relations": per_document_relations,
            }
        )

    relation_results = {
        category: metrics.relation_f1(*pairs)
        for category, pairs in relation_pairs.items()
    }
    covered = sum(bool(markdown) for _, markdown in markdown_predictions)
    page_count = sum(len(document["images_list"]) for document in documents)
    report: dict[str, object] = {
        "dataset": "MPDocBench-Parse",
        "dataset_revision": dataset_revision,
        "evaluator_revision": evaluator_revision,
        "evaluator_root": str(evaluator_root),
        "control": control,
        "reader": f"control:{control}",
        "markdown_output_dir": str(output_dir),
        "run_config": {"control": control, "limit": limit},
        "document_ids": [str(document["image_path"]) for document in documents],
        "page_ids": [
            str(page_id)
            for document in documents
            for page_id in document["images_list"]
        ],
        "attempted": len(documents),
        "covered": covered,
        "abstained": len(documents) - covered,
        "success": covered,
        "partial": 0,
        "failed": len(documents) - covered,
        "page_counts": {
            "attempted": page_count,
            "covered": page_count if control == "oracle" else 0,
            "abstained": 0 if control == "oracle" else page_count,
        },
        "failure_policy": (
            "all selected documents and empty or missing-equivalent predictions remain "
            "in document and ground-truth metric denominators"
        ),
        "metrics": {
            "HeadTEDS": {
                "all": round(sum(heading_scores) / len(heading_scores), 6),
                "documents": len(heading_scores),
            },
            "text_relation": {
                "Relation_F1_cross_page": _round_metric(relation_results["text"])
            },
            "table_relation": {
                "Relation_F1_cross_page": _round_metric(relation_results["table"])
            },
            "merged_text_block": {
                "Edit_dist": {
                    "edit_sample_avg": _mean(text_continuation_scores),
                    "instances": len(text_continuation_scores),
                    "cross_page_instances": text_cross_page_continuations,
                },
                "continuation_accuracy": {
                    "all": _mean(text_continuation_exact),
                    "cross_page": _mean(text_cross_page_exact),
                    "definition": "project-defined exact text match",
                },
            },
            "merged_table": {
                "TEDS": {
                    "all": _mean(table_continuation_scores),
                    "instances": len(table_continuation_scores),
                    "cross_page_instances": table_cross_page_continuations,
                },
                "TEDS-100": {
                    "all": _mean(
                        [float(score == 1.0) for score in table_continuation_scores]
                    ),
                    "perfect": sum(score == 1.0 for score in table_continuation_scores),
                    "instances": len(table_continuation_scores),
                    "definition": "fraction of table instances with TEDS equal to 1",
                },
                "continuation_accuracy": {
                    "all": _mean(table_continuation_exact),
                    "cross_page": _mean(table_cross_page_exact),
                    "definition": "project-defined exact normalized HTML match",
                },
            },
        },
        "documents": document_records,
    }

    output_dir.mkdir(parents=True)
    for prediction_name, markdown in markdown_predictions:
        (output_dir / prediction_name).write_text(markdown, encoding="utf-8")
    (output_dir / REPORT_NAME).write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


class _OfficialMetrics:
    def __init__(self, evaluator_root: Path) -> None:
        required = (
            evaluator_root / "metrics" / "head_metric.py",
            evaluator_root / "metrics" / "table_metric.py",
            evaluator_root / "utils" / "relation_utils.py",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise ValueError(
                "MPDocBench evaluator root is missing official metric files: "
                + ", ".join(missing)
            )
        root = str(evaluator_root.resolve())
        sys.path.insert(0, root)
        try:
            head_metric = _load_module("mpdoc_head_metric", required[0])
            table_metric = _load_module("mpdoc_table_metric", required[1])
            relation_utils = _load_module("mpdoc_relation_utils", required[2])
        except ImportError as error:
            dependency = error.name or str(error)
            raise ImportError(
                "Official MPDocBench metrics could not load dependency "
                f"{dependency!r}; install the evaluator requirements"
            ) from error
        finally:
            sys.path.remove(root)

        self.head_teds = head_metric.HeadTEDS()
        self.table_teds = table_metric.TEDS(structure_only=False)
        self.extract_table_relations = relation_utils.extract_table_truncated_info
        self.extract_text_relations = relation_utils.extract_text_truncated_info
        self.split_pairs_by_page = relation_utils.split_pairs_by_page
        self.relation_f1 = relation_utils.compute_relation_f1
        self.text_edit_distance = relation_utils._edit_dist
        self._detect_table = relation_utils.detect_pred_merges
        self._detect_text = relation_utils.detect_pred_text_merges
        self._normalize_table = relation_utils._norm_table
        self._normalize_text = relation_utils._norm_text

    def normalize_table_for_teds(self, value: str) -> str:
        normalized = self._normalize_table(value)
        return f"<html><body>{normalized}</body></html>" if normalized else ""

    def detect_relations(
        self,
        record: dict[str, object],
        category: str,
        predicted_values: list[str],
    ) -> set[tuple[object, object]]:
        items = [
            item
            for item in record["layout_dets"]
            if item.get("category_type")
            == ("table" if category == "table" else "text_block")
        ]
        items.sort(key=_element_order)
        if category == "table":
            normalized = [self._normalize_table(value) for value in predicted_values]
            pairs, _ = self._detect_table(items, normalized)
        else:
            normalized = [self._normalize_text(value) for value in predicted_values]
            pairs, _ = self._detect_text(items, normalized)
        return pairs


def _load_module(name: str, path: Path) -> object:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load official MPDocBench module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _oracle_markdown(
    record: dict[str, object],
    table_groups: list[set[object]],
    text_groups: list[set[object]],
) -> str:
    headings = _oracle_headings(record)
    group_by_id = {
        annotation_id: group
        for group in [*table_groups, *text_groups]
        for annotation_id in group
    }
    emitted_groups = set()
    parts = list(headings.values())
    for item in sorted(record["layout_dets"], key=_element_order):
        annotation_id = item.get("anno_id")
        if item.get("category_type") == "title":
            continue
        group = group_by_id.get(annotation_id)
        if group is not None:
            group_key = frozenset(group)
            if group_key in emitted_groups:
                continue
            emitted_groups.add(group_key)
            category = str(item.get("category_type"))
            annotation_map = {
                element.get("anno_id"): element for element in record["layout_dets"]
            }
            grouped = _group_items(annotation_map, group)
            if category == "table":
                value = str(grouped[0].get("merged_html", grouped[0].get("html", "")))
            else:
                value = "".join(str(element.get("text", "")) for element in grouped)
        elif item.get("category_type") == "table":
            value = str(item.get("html", ""))
        else:
            value = str(item.get("text", item.get("latex", "")))
        if value.strip():
            parts.append(value.strip())
    return "\n\n".join(parts) + ("\n" if parts else "")


def _oracle_category_values(
    record: dict[str, object],
    groups: list[set[object]],
    category: str,
) -> list[str]:
    category_type = "table" if category == "table" else "text_block"
    items = [
        item
        for item in record["layout_dets"]
        if item.get("category_type") == category_type
    ]
    annotation_map = {item.get("anno_id"): item for item in items}
    group_by_id = {annotation_id: group for group in groups for annotation_id in group}
    emitted_groups = set()
    values = []
    for item in sorted(items, key=_element_order):
        group = group_by_id.get(item.get("anno_id"))
        if group is None:
            values.append(str(item.get("html" if category == "table" else "text", "")))
            continue
        group_key = frozenset(group)
        if group_key in emitted_groups:
            continue
        emitted_groups.add(group_key)
        grouped = _group_items(annotation_map, group)
        if category == "table":
            values.append(
                str(grouped[0].get("merged_html", grouped[0].get("html", "")))
            )
        else:
            values.append("".join(str(element.get("text", "")) for element in grouped))
    return values


def _heading_ground_truth(record: dict[str, object]) -> list[object]:
    headings = {
        item["anno_id"]: item
        for item in record["layout_dets"]
        if item.get("category_type") == "title"
    }
    relations = [
        relation
        for relation in record.get("extra", {}).get("relation", [])
        if relation.get("relation_type") == "parent_son"
        and relation.get("target_anno_id") in headings
        and (
            relation.get("source_anno_id") == "root"
            or relation.get("source_anno_id") in headings
        )
    ]
    return [relations, headings]


def _oracle_headings(record: dict[str, object]) -> dict[object, str]:
    relations, headings = _heading_ground_truth(record)
    children: dict[object, list[object]] = defaultdict(list)
    child_ids = set()
    for relation in relations:
        source = relation["source_anno_id"]
        target = relation["target_anno_id"]
        children[source].append(target)
        child_ids.add(target)
    for annotation_id in headings:
        if annotation_id not in child_ids and annotation_id in children:
            children["root"].append(annotation_id)
    for values in children.values():
        values.sort(key=lambda annotation_id: headings[annotation_id]["anno_id"])

    rendered = {}

    def visit(annotation_id: object, depth: int) -> None:
        item = headings[annotation_id]
        rendered[annotation_id] = f"{'#' * depth} {str(item.get('text', '')).strip()}"
        for child_id in children.get(annotation_id, []):
            visit(child_id, depth + 1)

    for root_id in children.get("root", []):
        visit(root_id, 1)
    return rendered


def _heading_blocks(markdown: str) -> list[str]:
    return [
        block.strip()
        for block in markdown.split("\n\n")
        if block.strip().startswith("#")
    ]


def _group_items(
    annotation_map: dict[object, dict[str, object]], group: set[object]
) -> list[dict[str, object]]:
    return sorted(
        [annotation_map[annotation_id] for annotation_id in group],
        key=_element_order,
    )


def _element_order(item: dict[str, object]) -> tuple[int, int]:
    return int(item.get("page_id") or 0), int(item.get("order") or 0)


def _crosses_pages(items: list[dict[str, object]]) -> int:
    return int(len({item.get("page_id") for item in items}) > 1)


def _namespace_pairs(
    pairs: set[tuple[object, object]], document_index: int
) -> set[tuple[tuple[int, object], tuple[int, object]]]:
    return {
        ((document_index, source), (document_index, target)) for source, target in pairs
    }


def _round_metric(metric: dict[str, object]) -> dict[str, object]:
    return {
        key: round(value, 6) if isinstance(value, float) else value
        for key, value in metric.items()
    }


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def export_predictions(
    annotations: Path,
    image_root: Path,
    output_dir: Path,
    reader: LocalReader,
    *,
    dataset_revision: str,
    evaluator_revision: str,
    limit: int | None = None,
) -> dict[str, object]:
    if not dataset_revision.strip():
        raise ValueError("MPDocBench dataset revision must not be empty")
    if not evaluator_revision.strip():
        raise ValueError("MPDocBench evaluator revision must not be empty")

    records = json.loads(annotations.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("MPDocBench annotations must be a JSON list")
    selected = records[:limit] if limit is not None else records
    if not selected:
        raise ValueError("No MPDocBench documents were selected")

    documents = [_document_info(record, index) for index, record in enumerate(selected)]
    prediction_names = [
        Path(document["image_path"]).stem + ".md" for document in documents
    ]
    if len(set(prediction_names)) != len(prediction_names):
        raise ValueError(
            "Selected MPDocBench documents have duplicate prediction names"
        )

    output_dir.mkdir(parents=True)
    structured_dir = output_dir / STRUCTURED_DIR_NAME
    structured_dir.mkdir()

    document_records = []
    page_records = []
    document_counts = {"success": 0, "partial": 0, "failed": 0}
    for index, (document, prediction_name) in enumerate(
        zip(documents, prediction_names, strict=True)
    ):
        document_record, structured = _process_document_pages(
            index,
            document,
            image_root,
            reader,
            prediction_name,
        )
        markdown = document_record.pop("markdown")
        structured_name = Path(prediction_name).with_suffix(".json").name
        document_record["structured_prediction"] = str(
            Path(STRUCTURED_DIR_NAME) / structured_name
        )
        (output_dir / prediction_name).write_text(markdown, encoding="utf-8")
        (structured_dir / structured_name).write_text(
            json.dumps(structured, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        document_counts[str(document_record["status"])] += 1
        page_records.extend(document_record["pages"])
        document_records.append(document_record)

    document_latencies = [float(item["latency_ms"]) for item in document_records]
    page_latencies = [float(item["latency_ms"]) for item in page_records]
    covered = sum(bool(item["covered"]) for item in document_records)
    page_covered = sum(bool(item["covered"]) for item in page_records)
    page_counts = {
        "attempted": len(page_records),
        "covered": page_covered,
        "abstained": len(page_records) - page_covered,
        "success": sum(item["status"] == "success" for item in page_records),
        "partial": sum(item["status"] == "partial" for item in page_records),
        "failed": sum(item["status"] == "failed" for item in page_records),
    }
    report: dict[str, object] = {
        "dataset": "MPDocBench-Parse",
        "dataset_revision": dataset_revision,
        "evaluator_revision": evaluator_revision,
        "reader": reader.name,
        "markdown_output_dir": str(output_dir),
        "structured_output_dir": str(structured_dir),
        "run_config": {**_reader_config(reader), "limit": limit},
        "document_ids": [str(document["image_path"]) for document in documents],
        "page_ids": [str(page["page_id"]) for page in page_records],
        "attempted": len(document_records),
        "covered": covered,
        "abstained": len(document_records) - covered,
        **document_counts,
        "page_counts": page_counts,
        "latency_ms": {
            "document": _latency_percentiles(document_latencies),
            "page": _latency_percentiles(page_latencies),
        },
        "documents": document_records,
    }
    (output_dir / REPORT_NAME).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _process_document_pages(
    index: int,
    document: dict[str, object],
    image_root: Path,
    reader: LocalReader,
    prediction_name: str,
) -> tuple[dict[str, object], dict[str, object]]:
    started = time.perf_counter()
    page_records = []
    markdown_parts = []
    for page_number, page_id in enumerate(document["images_list"], start=1):
        page_started = time.perf_counter()
        result = None
        try:
            result = process_document(
                image_root / str(page_id),
                _PageNumberReader(reader, page_number),
            )
            status = result.status
        except Exception as error:
            status = "failed"
            failures = [
                {
                    "id": f"p{page_number}-failure-1",
                    "stage": "export",
                    "code": "unhandled_error",
                    "message": str(error),
                    "page_number": page_number,
                }
            ]
        else:
            failures = _structured_failures(result, page_number)

        markdown = _markdown(result)
        if markdown:
            markdown_parts.append(markdown)
        page_records.append(
            _structured_page_record(
                result,
                page_id=str(page_id),
                page_number=page_number,
                status=status,
                failures=failures,
                latency_ms=round((time.perf_counter() - page_started) * 1000, 3),
            )
        )

    markdown = "\n\n".join(markdown_parts)
    if markdown:
        markdown += "\n"
    covered = bool(markdown)
    status = _document_status(page_records)
    failures = [failure for page in page_records for failure in page["failures"]]
    document_record = {
        "index": index,
        "document_id": document["image_path"],
        "image_path": document["image_path"],
        "prediction": prediction_name,
        "status": status,
        "covered": covered,
        "abstained": not covered,
        "page_ids": list(document["images_list"]),
        "page_count": len(page_records),
        "failures": failures,
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        "pages": [_report_page(page) for page in page_records],
        "markdown": markdown,
    }
    structured = {
        "schema_version": 2,
        "document_id": document["image_path"],
        "image_path": document["image_path"],
        "status": status,
        "covered": covered,
        "abstained": not covered,
        "geometry_availability": _document_geometry(page_records),
        "pages": page_records,
        "failures": failures,
    }
    return document_record, structured


def _report_page(page: dict[str, object]) -> dict[str, object]:
    return {
        key: page[key]
        for key in (
            "page_id",
            "image_path",
            "page_number",
            "status",
            "covered",
            "abstained",
            "route",
            "failures",
            "latency_ms",
        )
    }


class _PageNumberReader:
    def __init__(self, reader: LocalReader, page_number: int) -> None:
        self.reader = reader
        self.page_number = page_number
        self.name = reader.name

    def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
        return self.reader.read(image_path, self.page_number)


def _structured_page_record(
    result: DocumentResult | None,
    *,
    page_id: str,
    page_number: int,
    status: str,
    failures: list[dict[str, object]],
    latency_ms: float,
) -> dict[str, object]:
    page = result.pages[0] if result is not None and result.pages else None
    failure_ids = [str(failure["id"]) for failure in failures]
    if page is None:
        return {
            "page_id": page_id,
            "image_path": page_id,
            "source": result.source if result is not None else None,
            "page_number": page_number,
            "status": status,
            "covered": False,
            "abstained": True,
            "width": 0,
            "height": 0,
            "reader": None,
            "route": "review",
            "text": {"value": "", "evidence_ids": []},
            "regions": [],
            "failure_ids": failure_ids,
            "failures": failures,
            "geometry_availability": "unavailable",
            "latency_ms": latency_ms,
        }

    regions = sorted(page.regions, key=lambda region: region.reading_order)
    covered = any(region.text.strip() for region in regions)
    return {
        "page_id": page_id,
        "image_path": page_id,
        "source": result.source,
        "page_number": page_number,
        "status": status,
        "covered": covered,
        "abstained": not covered,
        "width": page.width,
        "height": page.height,
        "reader": page.reader,
        "route": page.route,
        "text": asdict(page.text),
        "regions": [_structured_region(region, page) for region in regions],
        "failure_ids": failure_ids,
        "failures": failures,
        "geometry_availability": _page_geometry(page),
        "latency_ms": latency_ms,
    }


def _structured_failures(
    result: DocumentResult, page_number: int
) -> list[dict[str, object]]:
    failures = []
    for failure in result.failures:
        item = asdict(failure)
        item["id"] = f"p{page_number}-{failure.id}"
        item["page_number"] = page_number
        failures.append(item)
    return failures


def _structured_region(region: TextRegion, page: PageResult) -> dict[str, object]:
    return {**asdict(region), "geometry_availability": _region_geometry(region, page)}


def _document_geometry(pages: list[dict[str, object]]) -> str:
    geometry = {str(page["geometry_availability"]) for page in pages}
    if len(geometry) == 1:
        return geometry.pop()
    return "partial"


def _page_geometry(page: PageResult) -> str:
    if page.width <= 0 or page.height <= 0 or not page.regions:
        return "unavailable"
    geometry = {_region_geometry(region, page) for region in page.regions}
    if len(geometry) == 1:
        return geometry.pop()
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


def _document_status(pages: list[dict[str, object]]) -> str:
    statuses = {str(page["status"]) for page in pages}
    if statuses == {"success"}:
        return "success"
    if statuses == {"failed"}:
        return "failed"
    return "partial"


def _markdown(result: DocumentResult | None) -> str:
    if result is None:
        return ""
    regions = [
        region
        for page in result.pages
        for region in sorted(page.regions, key=lambda item: item.reading_order)
        if region.text.strip()
    ]
    return "\n\n".join(region.text.strip() for region in regions)


def _document_info(record: object, index: int) -> dict[str, object]:
    if not isinstance(record, dict) or not isinstance(record.get("page_info"), dict):
        raise ValueError(f"Invalid page_info in MPDocBench document {index}")
    page_info = record["page_info"]
    image_path = page_info.get("image_path")
    images_list = page_info.get("images_list")
    if not isinstance(image_path, str) or not image_path.strip():
        raise ValueError(f"Invalid image_path in MPDocBench document {index}")
    if (
        not isinstance(images_list, list)
        or not images_list
        or any(not isinstance(path, str) or not path.strip() for path in images_list)
    ):
        raise ValueError(f"Invalid images_list in MPDocBench document {index}")
    return {"image_path": image_path, "images_list": list(images_list)}


def _reader_config(reader: LocalReader) -> dict[str, object]:
    config = reader_config(reader)
    if isinstance(reader, GraniteDoclingReader):
        config["reader_options"]["output_format"] = reader.output_format
    return config


def _latency_percentiles(values: list[float]) -> dict[str, float]:
    return {
        "p50": round(_percentile(values, 0.50), 3),
        "p95": round(_percentile(values, 0.95), 3),
    }


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
