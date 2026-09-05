"""Evaluate layout predictions with OmniDocBench's official COCO metric."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Callable, Sequence

EVAL_CATEGORIES = (
    "title",
    "text",
    "abandon",
    "figure",
    "figure_caption",
    "table",
    "table_caption",
    "table_footnote",
    "isolate_formula",
    "formula_caption",
)
PREDICTION_CATEGORIES = (
    "title",
    "plain text",
    "abandon",
    "figure",
    "figure_caption",
    "table",
    "table_caption",
    "table_footnote",
    "isolate_formula",
    "formula_caption",
)
GT_CATEGORY_MAP = {
    "figure_footnote": "figure_footnote",
    "figure_caption": "figure_caption",
    "page_number": "abandon",
    "header": "abandon",
    "page_footnote": "abandon",
    "table_footnote": "table_footnote",
    "code_txt": "figure",
    "equation_caption": "formula_caption",
    "equation_isolated": "isolate_formula",
    "table": "table",
    "refernece": "text",
    "table_caption": "table_caption",
    "figure": "figure",
    "title": "title",
    "text_block": "text",
    "footer": "abandon",
}
PRED_CATEGORY_MAP = dict(zip(PREDICTION_CATEGORIES, EVAL_CATEGORIES, strict=True))
PRED_CATEGORY_IDS = {
    category: index for index, category in enumerate(PREDICTION_CATEGORIES)
}
PANEL_LANGUAGES = ("english", "simplified_chinese")
MIN_CASES = 30

OfficialScore = Callable[
    [list[dict[str, object]], dict[str, object], Path], dict[str, object]
]
COTE_COMPONENTS = ("coverage", "overlap", "trespass", "excess", "cote")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score OmniDocBench layout predictions without dropping pages"
    )
    parser.add_argument("annotations", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--evaluator-root", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--predictions", type=Path)
    source.add_argument("--control", choices=("oracle", "empty"))
    parser.add_argument(
        "--base-per-language",
        type=_positive_int,
        help="Select up to N English and Chinese v1.5 pages per document source",
    )
    args = parser.parse_args(argv)

    try:
        if args.output.exists():
            raise FileExistsError(f"Output already exists: {args.output}")
        report = run_benchmark(
            args.annotations,
            evaluator_root=args.evaluator_root,
            predictions_path=args.predictions,
            control=args.control,
            base_per_language=args.base_per_language,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    return 0


def run_benchmark(
    annotations_path: Path,
    *,
    evaluator_root: Path,
    predictions_path: Path | None = None,
    control: str | None = None,
    base_per_language: int | None = None,
    score_official: OfficialScore | None = None,
) -> dict[str, object]:
    records = _load_records(annotations_path)
    selected = select_panel(records, base_per_language)
    if len(selected) < MIN_CASES:
        raise ValueError(f"Layout benchmark requires at least {MIN_CASES} pages")
    case_ids = [_case_id(record, index) for index, record in enumerate(selected)]
    stems = [Path(case_id).stem for case_id in case_ids]
    if len(stems) != len(set(stems)):
        raise ValueError("Selected pages have duplicate prediction names")

    if control:
        predictions = build_control_predictions(selected, control)
        prediction_source = f"control:{control}"
    elif predictions_path:
        predictions = _load_predictions(predictions_path)
        prediction_source = str(predictions_path)
    else:
        raise ValueError("A prediction file or control is required")

    normalized, prediction_info = _normalize_predictions(predictions, set(stems))
    score = score_official or _score_official
    official = score(selected, normalized, evaluator_root)
    classwise = score_classwise(selected, normalized)
    union_area_iou = score_union_area_iou(selected, normalized)
    cote = score_cote(selected, normalized)
    covered = set(prediction_info["covered_case_ids"])
    source_counts = Counter(
        str(record["page_info"]["page_attribute"]["data_source"]) for record in selected
    )
    language_counts = Counter(
        str(record["page_info"]["page_attribute"]["language"]) for record in selected
    )
    layout_counts = Counter(
        str(record["page_info"]["page_attribute"]["layout"]) for record in selected
    )
    unsupported_gt = Counter(
        str(item.get("category_type"))
        for record in selected
        for item in record["layout_dets"]
        if _mapped_gt_category(item) is None
    )

    return {
        "benchmark": "OmniDocBench layout detection",
        "status": "complete",
        "dataset": {
            "annotations": str(annotations_path),
            "pages": len(selected),
            "panel": (
                "all provided pages"
                if base_per_language is None
                else (
                    "v1.5 base pages, up to "
                    f"{base_per_language} per document source and language"
                )
            ),
            "case_ids": case_ids,
            "document_sources": dict(sorted(source_counts.items())),
            "languages": dict(sorted(language_counts.items())),
            "layouts": dict(sorted(layout_counts.items())),
        },
        "evaluator": {
            "root": str(evaluator_root),
            "interface": "detection_dataset_simple_format",
            "metric": "mmeval COCODetection bbox with classwise=True",
            "classes": list(EVAL_CATEGORIES),
            "known_protocol_limits": [
                "The official v1.5 config spells reference as refernece, so reference boxes are excluded.",
                "The official adapter does not apply each annotation's ignore flag.",
                "mmeval keys ending in _precision are class AP, not threshold precision.",
            ],
        },
        "predictions": {
            "source": prediction_source,
            **prediction_info,
        },
        "coverage": {
            "attempted_pages": len(selected),
            "covered_pages": len(covered),
            "abstained_pages": len(selected) - len(covered),
            "coverage_rate": round(len(covered) / len(selected), 6),
            "failure_policy": "missing or empty page predictions remain in the denominator",
        },
        "ground_truth": {
            "evaluated_boxes": classwise["micro"]["ground_truth_boxes"],
            "unsupported_boxes": sum(unsupported_gt.values()),
            "unsupported_categories": dict(sorted(unsupported_gt.items())),
        },
        "metrics": {
            "official_coco": official,
            "iou_0_5_detection": classwise,
            "union_area_iou": union_area_iou,
            "cote": cote,
        },
    }


def build_control_predictions(
    records: list[dict[str, object]], control: str
) -> dict[str, object]:
    if control not in {"oracle", "empty"}:
        raise ValueError(f"Unsupported control: {control}")
    results = []
    if control == "oracle":
        for index, record in enumerate(records):
            image_name = Path(_case_id(record, index)).stem
            for item in record["layout_dets"]:
                category = _mapped_gt_category(item)
                if category is None:
                    continue
                pred_category = PREDICTION_CATEGORIES[EVAL_CATEGORIES.index(category)]
                results.append(
                    {
                        "image_name": image_name,
                        "bbox": _official_bbox(item.get("poly")),
                        "category_id": PRED_CATEGORY_IDS[pred_category],
                        "score": 1.0,
                    }
                )
    return {
        "results": results,
        "categories": {
            str(index): category for index, category in enumerate(PREDICTION_CATEGORIES)
        },
    }


def score_classwise(
    records: list[dict[str, object]],
    predictions: dict[str, object],
    *,
    iou_threshold: float = 0.5,
) -> dict[str, object]:
    ground_truth = _ground_truth_boxes(records)
    predicted = _prediction_boxes(predictions)
    counts = {
        category: {"true_positives": 0, "false_positives": 0, "false_negatives": 0}
        for category in EVAL_CATEGORIES
    }
    case_ids = [
        Path(_case_id(record, index)).stem for index, record in enumerate(records)
    ]
    for case_id in case_ids:
        for category in EVAL_CATEGORIES:
            gt_boxes = ground_truth[case_id][category]
            pred_boxes = sorted(
                predicted[case_id][category], key=lambda item: item[0], reverse=True
            )
            unmatched = set(range(len(gt_boxes)))
            for _, pred_box in pred_boxes:
                best_index = None
                best_iou = 0.0
                for gt_index in unmatched:
                    overlap = _iou(pred_box, gt_boxes[gt_index])
                    if overlap > best_iou:
                        best_iou = overlap
                        best_index = gt_index
                if best_index is not None and best_iou >= iou_threshold:
                    counts[category]["true_positives"] += 1
                    unmatched.remove(best_index)
                else:
                    counts[category]["false_positives"] += 1
            counts[category]["false_negatives"] += len(unmatched)

    by_class = {
        category: _detection_metrics(**category_counts)
        for category, category_counts in counts.items()
    }
    totals = {
        key: sum(class_counts[key] for class_counts in counts.values())
        for key in ("true_positives", "false_positives", "false_negatives")
    }
    macro_classes = [item for item in by_class.values() if item["ground_truth_boxes"]]
    if not macro_classes:
        raise ValueError("No supported OmniDocBench layout boxes were found")
    return {
        "definition": (
            "score-descending greedy one-to-one matching within page and class at IoU >= 0.5; "
            "all supplied predictions are scored"
        ),
        "by_class": by_class,
        "micro": _detection_metrics(**totals),
        "macro": {
            metric: round(
                sum(float(item[metric]) for item in macro_classes) / len(macro_classes),
                6,
            )
            for metric in ("precision", "recall", "f1")
        },
    }


def score_union_area_iou(
    records: list[dict[str, object]], predictions: dict[str, object]
) -> dict[str, object]:
    ground_truth = _ground_truth_boxes(records)
    predicted = _prediction_boxes(predictions)
    areas = {
        category: {"intersection_area": 0.0, "union_area": 0.0}
        for category in EVAL_CATEGORIES
    }
    per_page_iou = {}
    page_scores = []
    case_ids = [
        Path(_case_id(record, index)).stem for index, record in enumerate(records)
    ]
    for case_id in case_ids:
        category_scores = []
        for category in EVAL_CATEGORIES:
            gt_boxes = ground_truth[case_id][category]
            pred_boxes = [box for _, box in predicted[case_id][category]]
            gt_area = _rectangle_union_area(gt_boxes)
            pred_area = _rectangle_union_area(pred_boxes)
            union_area = _rectangle_union_area([*gt_boxes, *pred_boxes])
            intersection_area = max(
                0.0,
                min(gt_area, pred_area, gt_area + pred_area - union_area),
            )
            areas[category]["intersection_area"] += intersection_area
            areas[category]["union_area"] += union_area
            if union_area:
                category_scores.append(intersection_area / union_area)
        page_iou = (
            sum(category_scores) / len(category_scores) if category_scores else 0.0
        )
        page_scores.append(page_iou)
        per_page_iou[case_id] = round(page_iou, 6)

    by_class = {
        category: _area_iou_metrics(**category_areas)
        for category, category_areas in areas.items()
    }
    totals = {
        key: sum(category_areas[key] for category_areas in areas.values())
        for key in ("intersection_area", "union_area")
    }
    return {
        "definition": (
            "primary assignment-free geometric IoU: within each page, compute exact "
            "rectangle-union IoU for every category active in ground truth or predictions, "
            "average those categories equally, then take the median over every selected "
            "page; pages with no active category score zero"
        ),
        "dataset_median_iou": round(median(page_scores), 6) if page_scores else 0.0,
        "scored_pages": len(page_scores),
        "per_page_iou": per_page_iou,
        "by_class": by_class,
        "area_weighted_micro_iou": _area_iou_metrics(**totals),
        "area_weighted_micro_iou_note": (
            "secondary diagnostic pooling intersection and union area across all pages "
            "and categories"
        ),
    }


def score_cote(
    records: list[dict[str, object]], predictions: dict[str, object]
) -> dict[str, object]:
    """Compute exact box-based COTe diagnostics for every selected page."""
    predicted = _cote_prediction_boxes(predictions)
    page_scores: dict[str, dict[str, float | int]] = {}
    totals = {component: 0.0 for component in COTE_COMPONENTS}

    for index, record in enumerate(records):
        case_id = Path(_case_id(record, index)).stem
        width, height = _page_extent(record, index)
        ground_truth = _cote_ground_truth_boxes(record)
        page = _score_cote_page(
            ground_truth,
            predicted[case_id],
            width=width,
            height=height,
        )
        page_scores[case_id] = {
            component: round(float(page[component]), 6) for component in COTE_COMPONENTS
        } | {
            "ground_truth_regions": len(ground_truth),
            "predicted_regions": len(predicted[case_id]),
        }
        for component in COTE_COMPONENTS:
            totals[component] += float(page[component])

    page_count = len(records)
    return {
        "definition": "COTe = Coverage - Overlap - Trespass; Excess is a support metric",
        "ground_truth_unit": (
            "Each supported OmniDocBench ground-truth region is treated as one "
            "physical-unit fallback; no semantic SSUs are inferred."
        ),
        "assignment": (
            "Each prediction is assigned class-agnostically to the physical unit with "
            "the greatest intersection area; ties use the lowest source-order region "
            "index. Predictions with no ground-truth intersection remain unassigned."
        ),
        "class_policy": (
            "Canonical class labels are preserved on ground truth and predictions; COTe "
            "geometry is class-agnostic, while the accompanying COCO, IoU@0.5, and union "
            "IoU diagnostics retain class-sensitive scoring."
        ),
        "aggregation": (
            "Arithmetic mean of per-page components over every selected page, including "
            "missing and empty predictions."
        ),
        "paper": "https://arxiv.org/abs/2603.12718v2",
        "implementation_reference": "https://github.com/THE-3TC/cotescore",
        "classes": list(EVAL_CATEGORIES),
        "scored_pages": page_count,
        "dataset_mean": {
            component: round(totals[component] / page_count, 6) if page_count else 0.0
            for component in COTE_COMPONENTS
        },
        "per_page": page_scores,
    }


def _score_official(
    records: list[dict[str, object]],
    predictions: dict[str, object],
    evaluator_root: Path,
) -> dict[str, object]:
    module_path = evaluator_root / "dataset" / "detection_dataset.py"
    if not module_path.is_file():
        raise ValueError(f"OmniDocBench evaluator is missing {module_path}")
    module = _load_evaluator_module(evaluator_root, module_path)
    with tempfile.TemporaryDirectory(prefix="omnidoc-layout-") as temp_dir:
        temp_root = Path(temp_dir)
        annotations_path = temp_root / "annotations.json"
        predictions_path = temp_root / "predictions.json"
        virtual_records = json.loads(json.dumps(records))
        for index, record in enumerate(virtual_records):
            record["page_info"]["image_path"] = (
                Path(_case_id(record, index)).stem + ".jpg"
            )
        annotations_path.write_text(json.dumps(virtual_records), encoding="utf-8")
        predictions_path.write_text(json.dumps(predictions), encoding="utf-8")
        dataset = module.DetectionDatasetSimpleFormat(
            _official_config(annotations_path, predictions_path)
        )
        raw = dataset.coco_det_metric(
            predictions=dataset.samples["preds"],
            groundtruths=dataset.samples["gts"],
        )

    empty_predictions = not predictions["results"]
    ground_truth_counts = Counter(
        category
        for record in records
        for item in record["layout_dets"]
        if (category := _mapped_gt_category(item)) is not None
    )
    summary = {
        key: _official_value(
            raw.get(key), empty_predictions and bool(ground_truth_counts)
        )
        for key in (
            "bbox_mAP",
            "bbox_mAP_50",
            "bbox_mAP_75",
            "bbox_mAP_s",
            "bbox_mAP_m",
            "bbox_mAP_l",
        )
    }
    class_ap = {
        category: _official_value(
            raw.get(f"bbox_{category}_precision"),
            empty_predictions and ground_truth_counts[category] > 0,
        )
        for category in EVAL_CATEGORIES
    }
    return {
        "summary": summary,
        "class_ap": class_ap,
        "class_ap_note": "COCO AP averaged over IoU 0.50:0.95",
        "raw_result_empty": not bool(raw),
        "empty_prediction_policy": (
            "mmeval returns no metric keys for a fully empty result; scores with ground truth are reported as zero"
            if empty_predictions and not raw
            else None
        ),
    }


def _load_evaluator_module(evaluator_root: Path, module_path: Path):
    module_name = "_omnidocbench_layout_detection_dataset"
    if module_name in sys.modules:
        return sys.modules[module_name]
    sys.path.insert(0, str(evaluator_root))
    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load OmniDocBench evaluator from {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    finally:
        sys.path.remove(str(evaluator_root))


def _official_config(
    annotations_path: Path, predictions_path: Path
) -> dict[str, object]:
    return {
        "dataset": {
            "ground_truth": {"data_path": str(annotations_path)},
            "prediction": {"data_path": str(predictions_path)},
        },
        "categories": {
            "eval_cat": {"block_level": list(EVAL_CATEGORIES)},
            "gt_cat_mapping": GT_CATEGORY_MAP,
            "pred_cat_mapping": PRED_CATEGORY_MAP,
        },
    }


def select_panel(
    records: list[dict[str, object]], per_language: int | None
) -> list[dict[str, object]]:
    if per_language is None:
        return records
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        attributes = record["page_info"]["page_attribute"]
        if attributes.get("subset") != "v1.5":
            continue
        source = str(attributes.get("data_source"))
        language = str(attributes.get("language"))
        if language in PANEL_LANGUAGES:
            groups[(source, language)].append(record)
    selected = []
    sources = sorted({source for source, _ in groups})
    for source in sources:
        for language in PANEL_LANGUAGES:
            selected.extend(groups[(source, language)][:per_language])
    return selected


def _normalize_predictions(
    payload: dict[str, object], selected_stems: set[str]
) -> tuple[dict[str, object], dict[str, object]]:
    raw_categories = payload.get("categories")
    raw_results = payload.get("results")
    if not isinstance(raw_categories, dict) or not isinstance(raw_results, list):
        raise ValueError("Predictions must contain categories and results")
    category_names = {str(key): str(value) for key, value in raw_categories.items()}
    normalized_results = []
    covered = set()
    unsupported = Counter()
    ignored_results = 0
    for index, item in enumerate(raw_results):
        if not isinstance(item, dict):
            raise ValueError(f"Prediction {index} must be an object")
        image_name = item.get("image_name")
        category_id = item.get("category_id")
        if not isinstance(image_name, str):
            raise ValueError(f"Prediction {index} has an invalid image_name")
        stem = Path(image_name).stem
        if stem not in selected_stems:
            ignored_results += 1
            continue
        pred_name = category_names.get(str(category_id))
        canonical = PRED_CATEGORY_MAP.get(str(pred_name))
        if canonical is None:
            unsupported[str(pred_name)] += 1
            continue
        bbox = _prediction_bbox(item.get("bbox"), index)
        score = _score(item.get("score"), index)
        output_name = PREDICTION_CATEGORIES[EVAL_CATEGORIES.index(canonical)]
        normalized_results.append(
            {
                "image_name": stem,
                "bbox": bbox,
                "category_id": PRED_CATEGORY_IDS[output_name],
                "score": score,
            }
        )
        covered.add(stem)
    return (
        {
            "results": normalized_results,
            "categories": {
                str(index): category
                for index, category in enumerate(PREDICTION_CATEGORIES)
            },
        },
        {
            "boxes": len(normalized_results),
            "covered_case_ids": sorted(covered),
            "ignored_outside_panel": ignored_results,
            "unsupported_boxes": sum(unsupported.values()),
            "unsupported_categories": dict(sorted(unsupported.items())),
        },
    )


def _ground_truth_boxes(
    records: list[dict[str, object]],
) -> defaultdict[str, defaultdict[str, list[list[float]]]]:
    boxes: defaultdict[str, defaultdict[str, list[list[float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index, record in enumerate(records):
        case_id = Path(_case_id(record, index)).stem
        for item in record["layout_dets"]:
            category = _mapped_gt_category(item)
            if category is not None:
                boxes[case_id][category].append(_official_bbox(item.get("poly")))
    return boxes


def _prediction_boxes(
    predictions: dict[str, object],
) -> defaultdict[str, defaultdict[str, list[tuple[float, list[float]]]]]:
    boxes: defaultdict[str, defaultdict[str, list[tuple[float, list[float]]]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    categories = {
        str(key): str(value) for key, value in predictions["categories"].items()
    }
    for index, item in enumerate(predictions["results"]):
        pred_name = categories[str(item["category_id"])]
        category = PRED_CATEGORY_MAP[pred_name]
        boxes[str(item["image_name"])][category].append(
            (_score(item["score"], index), _prediction_bbox(item["bbox"], index))
        )
    return boxes


def _cote_ground_truth_boxes(
    record: dict[str, object],
) -> list[tuple[int, str, list[float]]]:
    boxes = []
    for index, item in enumerate(record["layout_dets"]):
        category = _mapped_gt_category(item)
        if category is not None:
            boxes.append((index, category, _official_bbox(item.get("poly"))))
    return boxes


def _cote_prediction_boxes(
    predictions: dict[str, object],
) -> defaultdict[str, list[tuple[str, list[float]]]]:
    boxes: defaultdict[str, list[tuple[str, list[float]]]] = defaultdict(list)
    categories = {
        str(key): str(value) for key, value in predictions["categories"].items()
    }
    for index, item in enumerate(predictions["results"]):
        pred_name = categories[str(item["category_id"])]
        boxes[str(item["image_name"])].append(
            (PRED_CATEGORY_MAP[pred_name], _prediction_bbox(item["bbox"], index))
        )
    return boxes


def _score_cote_page(
    ground_truth: list[tuple[int, str, list[float]]],
    predictions: list[tuple[str, list[float]]],
    *,
    width: float,
    height: float,
) -> dict[str, float]:
    clipped_ground_truth = [
        (region_index, category, clipped)
        for region_index, category, box in ground_truth
        if (clipped := _clip_box(box, width, height)) is not None
    ]
    clipped_predictions = [
        (category, _clip_box(box, width, height)) for category, box in predictions
    ]
    boxes = [box for _, _, box in clipped_ground_truth]
    boxes.extend(box for _, box in clipped_predictions if box is not None)
    x_edges = sorted(
        {0.0, width, *(edge for box in boxes for edge in (box[0], box[2]))}
    )
    y_edges = sorted(
        {0.0, height, *(edge for box in boxes for edge in (box[1], box[3]))}
    )
    x_indexes = {edge: index for index, edge in enumerate(x_edges)}
    y_indexes = {edge: index for index, edge in enumerate(y_edges)}
    cell_owners: list[list[int | None]] = [
        [None] * (len(x_edges) - 1) for _ in range(len(y_edges) - 1)
    ]
    prediction_counts = [[0] * (len(x_edges) - 1) for _ in range(len(y_edges) - 1)]

    for region_index, _, box in clipped_ground_truth:
        left, top, right, bottom = _box_cell_range(box, x_indexes, y_indexes)
        for y_index in range(top, bottom):
            owners = cell_owners[y_index]
            for x_index in range(left, right):
                if owners[x_index] is None:
                    owners[x_index] = region_index

    trespass_area = 0.0
    for _, box in clipped_predictions:
        if box is None:
            continue
        left, top, right, bottom = _box_cell_range(box, x_indexes, y_indexes)
        area_by_owner: defaultdict[int, float] = defaultdict(float)
        for y_index in range(top, bottom):
            owners = cell_owners[y_index]
            counts = prediction_counts[y_index]
            cell_height = y_edges[y_index + 1] - y_edges[y_index]
            for x_index in range(left, right):
                counts[x_index] += 1
                owner = owners[x_index]
                if owner is not None:
                    area_by_owner[owner] += (
                        x_edges[x_index + 1] - x_edges[x_index]
                    ) * cell_height
        if area_by_owner:
            assigned = min(
                area_by_owner,
                key=lambda owner: (-area_by_owner[owner], owner),
            )
            trespass_area += sum(area_by_owner.values()) - area_by_owner[assigned]

    ground_truth_area = 0.0
    covered_area = 0.0
    overlap_area = 0.0
    excess_area = 0.0
    for y_index, owners in enumerate(cell_owners):
        cell_height = y_edges[y_index + 1] - y_edges[y_index]
        for x_index, owner in enumerate(owners):
            cell_area = (x_edges[x_index + 1] - x_edges[x_index]) * cell_height
            prediction_count = prediction_counts[y_index][x_index]
            if owner is None:
                if prediction_count:
                    excess_area += cell_area
                continue
            ground_truth_area += cell_area
            if prediction_count:
                covered_area += cell_area
                overlap_area += cell_area * (prediction_count - 1)

    if ground_truth_area:
        coverage = covered_area / ground_truth_area
        overlap = overlap_area / ground_truth_area
        trespass = trespass_area / ground_truth_area
    else:
        coverage = 0.0 if predictions else 1.0
        overlap = 0.0
        trespass = 0.0
    background_area = width * height - ground_truth_area
    excess = excess_area / background_area if background_area else 0.0
    return {
        "coverage": coverage,
        "overlap": overlap,
        "trespass": trespass,
        "excess": excess,
        "cote": coverage - overlap - trespass,
    }


def _page_extent(record: dict[str, object], index: int) -> tuple[float, float]:
    page_info = record["page_info"]
    try:
        width = float(page_info["width"])
        height = float(page_info["height"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Annotation {index} has invalid page dimensions") from error
    if not all(math.isfinite(value) and value > 0 for value in (width, height)):
        raise ValueError(f"Annotation {index} has invalid page dimensions")
    return width, height


def _clip_box(box: list[float], width: float, height: float) -> list[float] | None:
    left = min(width, max(0.0, box[0]))
    top = min(height, max(0.0, box[1]))
    right = min(width, max(0.0, box[2]))
    bottom = min(height, max(0.0, box[3]))
    return [left, top, right, bottom] if right > left and bottom > top else None


def _box_cell_range(
    box: list[float],
    x_indexes: dict[float, int],
    y_indexes: dict[float, int],
) -> tuple[int, int, int, int]:
    return (
        x_indexes[box[0]],
        y_indexes[box[1]],
        x_indexes[box[2]],
        y_indexes[box[3]],
    )


def _detection_metrics(
    true_positives: int, false_positives: int, false_negatives: int
) -> dict[str, object]:
    predicted_boxes = true_positives + false_positives
    ground_truth_boxes = true_positives + false_negatives
    precision = true_positives / predicted_boxes if predicted_boxes else 0.0
    recall = true_positives / ground_truth_boxes if ground_truth_boxes else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "predicted_boxes": predicted_boxes,
        "ground_truth_boxes": ground_truth_boxes,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


def _area_iou_metrics(intersection_area: float, union_area: float) -> dict[str, float]:
    iou = intersection_area / union_area if union_area else 0.0
    return {
        "intersection_area": round(intersection_area, 6),
        "union_area": round(union_area, 6),
        "iou": round(iou, 6),
    }


def _mapped_gt_category(item: object) -> str | None:
    if not isinstance(item, dict):
        raise ValueError("Each layout annotation must be an object")
    raw_category = str(item.get("category_type"))
    category = GT_CATEGORY_MAP.get(raw_category, raw_category)
    return category if category in EVAL_CATEGORIES else None


def _official_bbox(poly: object) -> list[float]:
    if not isinstance(poly, list) or len(poly) < 6:
        raise ValueError("Layout polygon must contain at least six coordinates")
    try:
        values = [float(value) for value in poly]
    except (TypeError, ValueError) as error:
        raise ValueError("Layout polygon coordinates must be numeric") from error
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Layout polygon coordinates must be finite")
    left, right = sorted((values[0], values[2]))
    top, bottom = sorted((values[1], values[5]))
    if right <= left or bottom <= top:
        raise ValueError("Layout polygon must have positive area")
    return [left, top, right, bottom]


def _prediction_bbox(value: object, index: int) -> list[float]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"Prediction {index} bbox must contain four coordinates")
    try:
        bbox = [float(coordinate) for coordinate in value]
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Prediction {index} bbox coordinates must be numeric"
        ) from error
    if not all(math.isfinite(coordinate) for coordinate in bbox):
        raise ValueError(f"Prediction {index} bbox coordinates must be finite")
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        raise ValueError(f"Prediction {index} bbox must have positive area")
    return bbox


def _score(value: object, index: int) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Prediction {index} score must be numeric") from error
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError(f"Prediction {index} score must be between zero and one")
    return score


def _iou(first: list[float], second: list[float]) -> float:
    width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = width * height
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def _rectangle_union_area(boxes: Sequence[list[float]]) -> float:
    events = sorted(
        event
        for left, top, right, bottom in boxes
        for event in ((left, 1, top, bottom), (right, -1, top, bottom))
    )
    if not events:
        return 0.0

    active: Counter[tuple[float, float]] = Counter()
    area = 0.0
    previous_x = events[0][0]
    index = 0
    while index < len(events):
        x = events[index][0]
        area += (x - previous_x) * _interval_union_length(active)
        while index < len(events) and events[index][0] == x:
            _, change, top, bottom = events[index]
            interval = (top, bottom)
            active[interval] += change
            if not active[interval]:
                del active[interval]
            index += 1
        previous_x = x
    return area


def _interval_union_length(intervals: Counter[tuple[float, float]]) -> float:
    length = 0.0
    current_bottom: float | None = None
    for top, bottom in sorted(intervals):
        if current_bottom is None or top > current_bottom:
            length += bottom - top
            current_bottom = bottom
        elif bottom > current_bottom:
            length += bottom - current_bottom
            current_bottom = bottom
    return length


def _case_id(record: object, index: int) -> str:
    if not isinstance(record, dict) or not isinstance(record.get("page_info"), dict):
        raise ValueError(f"Annotation {index} has invalid page_info")
    image_path = record["page_info"].get("image_path")
    if not isinstance(image_path, str) or not image_path.strip():
        raise ValueError(f"Annotation {index} has invalid image_path")
    layout_dets = record.get("layout_dets")
    if not isinstance(layout_dets, list):
        raise ValueError(f"Annotation {index} has invalid layout_dets")
    return image_path


def _load_records(path: Path) -> list[dict[str, object]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("OmniDocBench annotations must be a JSON list")
    return value


def _load_predictions(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Layout predictions must be a JSON object")
    return value


def _finite_or_none(value: object) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _official_value(value: object, zero_when_missing: bool) -> float | None:
    if value is None and zero_when_missing:
        return 0.0
    return _finite_or_none(value)


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return number


if __name__ == "__main__":
    raise SystemExit(main())
