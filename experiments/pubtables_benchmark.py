"""Score PubTables-1M cell predictions with Microsoft's pinned GriTS code."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence

DATASET_ID = "bsmock/pubtables-1m"
DATASET_REVISION = "35b1c097807e0b07ec5313879b85956b7b3890db"
DATASET_LICENSE = "CDLA-Permissive-2.0"
SCORER_REVISION = "16d124f616109746b7785f03085100f1f6247575"
MODEL_ID = "bsmock/TATR-v1.1-Pub"
MODEL_REVISION = "865c1f9c024e038952ee4c8156b82a606384b33d"
MODEL_LICENSE = "MIT"
MODEL_FILE = "TATR-v1.1-Pub-msft.pth"
MODEL_SIZE_BYTES = 115_508_750
ALL_MODEL_ID = "bsmock/TATR-v1.1-All"
ALL_MODEL_REVISION = "30e6112ea20d72cc9fb8f443e34ca119d497cfe6"
ALL_MODEL_FILE = "TATR-v1.1-All-msft.pth"
TATR_CROP_MARGIN = 5
DEFAULT_CASES = 60
MIN_CASES = 30
MAX_CELLS = 4096
MAX_INDEX = 256
MAX_GRID_AREA = 4096
CLASS_NAMES = (
    "table",
    "table column",
    "table row",
    "table column header",
    "table projected row header",
    "table spanning cell",
    "no object",
)
CLASS_THRESHOLDS = {name: 0.5 for name in CLASS_NAMES[:-1]} | {"no object": 10}
METRICS = ("grits_top", "grits_con", "grits_loc", "cell_exact_match")


@dataclass(frozen=True)
class Case:
    case_id: str
    target: dict[str, Any]
    image_path: Path | None


class OfficialScorer:
    """Load the trusted upstream modules once and expose their cell interface."""

    def __init__(self, root: Path) -> None:
        checkout = root.resolve()
        source = checkout if (checkout / "grits.py").is_file() else checkout / "src"
        if (
            not (source / "grits.py").is_file()
            or not (source / "postprocess.py").is_file()
        ):
            raise ValueError("Supplied scorer checkout lacks required source files")
        self.root = checkout
        self.grits = _load_module("pubtables_official_grits", source / "grits.py")
        self.postprocess = _load_module(
            "pubtables_official_postprocess", source / "postprocess.py"
        )
        for name in (
            "cells_to_grid",
            "cells_to_relspan_grid",
            "grits_top",
            "grits_con",
            "grits_loc",
        ):
            if not callable(getattr(self.grits, name, None)):
                raise ValueError(f"Supplied GriTS scorer lacks {name}")

    def gold_cells(self, target: dict[str, Any]) -> list[dict[str, Any]]:
        structure = _mapping(target.get("structure"), "structure")
        words = _mapping(target.get("words"), "words")
        if structure.get("representation") != "pascal_voc_xml":
            raise ValueError("PubTables structure must use Pascal VOC XML")
        if words.get("representation") != "word_boxes_json":
            raise ValueError("PubTables words must use word-box JSON")

        boxes, labels = _xml_objects(structure.get("xml"))
        tokens = _word_tokens(words.get("data"))
        boxes, scores, labels = self.postprocess.apply_class_thresholds(
            boxes,
            labels,
            [1.0] * len(labels),
            list(CLASS_NAMES),
            CLASS_THRESHOLDS,
        )
        objects = [
            {"bbox": box, "score": score, "label": label}
            for box, score, label in zip(boxes, scores, labels, strict=True)
        ]
        tables = [item for item in objects if item["label"] == 0]
        if not tables:
            raise ValueError("PubTables annotation contains no table")
        table_box = max(tables, key=lambda item: item["score"])["bbox"]
        table_tokens = [
            token
            for token in tokens
            if self.postprocess.iob(token["bbox"], table_box) >= 0.5
        ]
        try:
            _, cells, _ = self.postprocess.objects_to_cells(
                {"objects": objects, "page_num": 0},
                objects,
                table_tokens,
                list(CLASS_NAMES),
                CLASS_THRESHOLDS,
            )
        except Exception as error:
            raise ValueError(
                f"Official PubTables postprocessing failed: {error}"
            ) from error
        return _normalize_cells(cells)

    def score(
        self,
        gold: list[dict[str, Any]],
        predicted: Any,
    ) -> dict[str, float]:
        import numpy as np

        prediction = _normalize_cells(predicted)
        true_top = _object_grid(self.grits.cells_to_relspan_grid(gold), np)
        pred_top = _object_grid(self.grits.cells_to_relspan_grid(prediction), np)
        true_con = _object_grid(self.grits.cells_to_grid(gold, key="cell_text"), np)
        pred_con = _object_grid(
            self.grits.cells_to_grid(prediction, key="cell_text"), np
        )
        true_loc = _object_grid(self.grits.cells_to_grid(gold, key="bbox"), np)
        pred_loc = _object_grid(self.grits.cells_to_grid(prediction, key="bbox"), np)
        result = {
            "grits_top": float(self.grits.grits_top(true_top, pred_top)[0]),
            "grits_con": float(self.grits.grits_con(true_con, pred_con)[0]),
            "grits_loc": float(self.grits.grits_loc(true_loc, pred_loc)[0]),
            "cell_exact_match": _cell_exact_match(gold, prediction),
        }
        if not all(
            math.isfinite(value) and 0 <= value <= 1 for value in result.values()
        ):
            raise ValueError("Official GriTS returned a score outside [0, 1]")
        return result


class TatrPredictor:
    """Run the supplied structure inference on prepared table crops."""

    def __init__(
        self,
        scorer_root: Path,
        model_root: Path,
        checkpoint_file: str,
        device: str,
    ) -> None:
        checkpoint = _checkpoint_path(model_root, checkpoint_file)
        if not checkpoint.is_file():
            raise ValueError(f"TATR checkpoint is missing: {checkpoint}")
        if device.startswith("cuda"):
            import torch

            if not torch.cuda.is_available():
                raise ValueError("CUDA was requested but is unavailable")

        started = time.perf_counter()
        inference = _load_tatr_inference(scorer_root)
        source = scorer_root / "src" if (scorer_root / "src").is_dir() else scorer_root
        self.pipeline = inference.TableExtractionPipeline(
            str_device=device,
            str_config_path=source / "structure_config.json",
            str_model_path=checkpoint,
        )
        self.device = device
        self.load_ms = round((time.perf_counter() - started) * 1000, 3)

    def predict(self, case: Case) -> dict[str, Any]:
        if case.image_path is None or not case.image_path.is_file():
            return _failed_prediction("missing_table_image")
        from PIL import Image

        started = time.perf_counter()
        try:
            with Image.open(case.image_path) as source_image:
                image = source_image.convert("RGB")
            crop, offset = _tight_crop(image, case.target)
            tokens = _crop_tokens(case.target, offset, crop.size)
            self._start_measurement()
            result = self.pipeline.recognize(crop, tokens, out_cells=True)
            self._finish_measurement()
            tables = result.get("cells")
            if not isinstance(tables, list) or len(tables) != 1:
                raise ValueError("TATR returned an unexpected table count")
            cells = _translate_cells(tables[0], offset)
            return _prediction_record(
                "success",
                cells,
                {
                    "latency_ms": (time.perf_counter() - started) * 1000,
                    "peak_gpu_memory_mb": self._peak_memory_mb(),
                    "cost_usd": 0,
                },
            )
        except Exception as error:
            prediction = _failed_prediction(
                f"tatr_inference_error: {type(error).__name__}: {error}"
            )
            prediction["latency_ms"] = (time.perf_counter() - started) * 1000
            prediction["peak_gpu_memory_mb"] = self._peak_memory_mb()
            return prediction

    def _start_measurement(self) -> None:
        if not self.device.startswith("cuda"):
            return
        import torch

        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)

    def _finish_measurement(self) -> None:
        if not self.device.startswith("cuda"):
            return
        import torch

        torch.cuda.synchronize(self.device)

    def _peak_memory_mb(self) -> float | None:
        if not self.device.startswith("cuda"):
            return None
        import torch

        return torch.cuda.max_memory_allocated(self.device) / 2**20


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score a failure-inclusive PubTables-1M test panel"
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--scorer-root", type=Path, required=True)
    parser.add_argument(
        "--scorer-revision",
        help="scorer revision independently established by the caller",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--predictions", type=Path)
    source.add_argument("--control", choices=("oracle", "empty"))
    source.add_argument("--tatr-model-root", type=Path)
    parser.add_argument(
        "--checkpoint-revision",
        help="checkpoint revision independently established by the caller",
    )
    parser.add_argument("--checkpoint-file", default=MODEL_FILE)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=_positive_int, default=DEFAULT_CASES)
    args = parser.parse_args(argv)

    try:
        if args.output.exists():
            raise FileExistsError(f"Output already exists: {args.output}")
        report = run_benchmark(
            args.dataset_root,
            scorer_root=args.scorer_root,
            scorer_revision=args.scorer_revision,
            predictions_root=args.predictions,
            control=args.control,
            tatr_model_root=args.tatr_model_root,
            checkpoint_revision=args.checkpoint_revision,
            checkpoint_file=args.checkpoint_file,
            device=args.device,
            limit=args.limit,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    return 0


def run_benchmark(
    dataset_root: Path,
    *,
    scorer_root: Path,
    scorer_revision: str | None = None,
    predictions_root: Path | None = None,
    control: str | None = None,
    tatr_model_root: Path | None = None,
    checkpoint_revision: str | None = None,
    checkpoint_file: str = MODEL_FILE,
    device: str = "cuda",
    limit: int = DEFAULT_CASES,
) -> dict[str, object]:
    sources = sum(
        value is not None for value in (predictions_root, control, tatr_model_root)
    )
    if sources != 1:
        raise ValueError(
            "Provide exactly one prediction directory, control, or TATR model"
        )
    if control not in {None, "oracle", "empty"}:
        raise ValueError(f"Unsupported control: {control}")
    if checkpoint_revision is not None and tatr_model_root is None:
        raise ValueError("Checkpoint revision requires a TATR model root")

    scorer = OfficialScorer(scorer_root)
    cases = _load_cases(dataset_root, limit)
    if len(cases) < MIN_CASES:
        raise ValueError(f"PubTables benchmark requires at least {MIN_CASES} tables")

    predictor = (
        TatrPredictor(scorer_root, tatr_model_root, checkpoint_file, device)
        if tatr_model_root is not None
        else None
    )
    records: list[dict[str, Any]] = []
    scores: list[dict[str, float] | None] = []
    for case in cases:
        gold = scorer.gold_cells(case.target)
        if control is not None:
            prediction = {
                "status": "success",
                "cells": gold if control == "oracle" else [],
                "latency_ms": None,
                "peak_gpu_memory_mb": None,
                "cost_usd": None,
                "failure": None,
            }
        elif predictions_root is not None:
            assert predictions_root is not None
            prediction = _load_prediction(predictions_root, case.case_id)
        else:
            assert predictor is not None
            prediction = predictor.predict(case)

        score = None
        if prediction["status"] == "success":
            try:
                score = scorer.score(gold, prediction["cells"])
            except (IndexError, KeyError, TypeError, ValueError) as error:
                prediction = {**prediction, "status": "failed", "failure": str(error)}
        scores.append(score)
        records.append(
            {
                "case_id": case.case_id,
                "status": prediction["status"],
                "latency_ms": prediction["latency_ms"],
                "peak_gpu_memory_mb": prediction["peak_gpu_memory_mb"],
                "cost_usd": prediction["cost_usd"],
                "failure": prediction["failure"],
                "metrics": score,
            }
        )

    valid_scores = [score for score in scores if score is not None]
    status_counts = Counter(record["status"] for record in records)
    return {
        "benchmark": "PubTables-1M table structure recognition",
        "status": "complete",
        "dataset": {
            "id": DATASET_ID,
            "revision": DATASET_REVISION,
            "revision_evidence": (
                "public repository metadata verified; prepared cases do not embed "
                "their source revision"
            ),
            "license": DATASET_LICENSE,
            "split": "test",
            "root": str(dataset_root),
            "attempted_tables": len(cases),
            "case_ids": [case.case_id for case in cases],
            "panel": (
                f"{limit} deterministic evenly spaced tables over the supplied "
                "test panel's lexical order"
            ),
        },
        "scorer": _scorer_metadata(scorer_root, scorer_revision),
        "prediction_source": (
            f"control:{control}"
            if control is not None
            else (
                f"checkpoint:{_checkpoint_path(tatr_model_root, checkpoint_file)}"
                if tatr_model_root is not None
                else str(predictions_root)
            )
        ),
        "metric_definitions": {
            "grits_top": "GriTS topology F-score from the reported scorer",
            "grits_con": "GriTS cell-content F-score from the reported scorer",
            "grits_loc": "GriTS cell-location F-score from the reported scorer",
            "cell_exact_match": (
                "multiset F1 over exact row span, column span, and cell text; "
                "location is measured separately"
            ),
        },
        "coverage": {
            "attempted": len(cases),
            "valid": len(valid_scores),
            "failed": status_counts["failed"],
            "abstained": status_counts["abstained"],
            "coverage_rate": round(len(valid_scores) / len(cases), 6),
            "failure_policy": "failed, abstained, missing, and invalid cases score zero",
        },
        "metrics": {
            "all_cases": _mean_scores(valid_scores, len(cases)),
            "served_only": _mean_scores(valid_scores, len(valid_scores)),
        },
        "operations": {
            **_operation_metrics(records),
            "model_load_ms": None if predictor is None else predictor.load_ms,
        },
        "cases": records,
        **(
            {
                "reference_model": _model_metadata(
                    tatr_model_root,
                    checkpoint_revision,
                    checkpoint_file,
                )
            }
            if tatr_model_root is not None
            else {}
        ),
    }


def _scorer_metadata(root: Path, revision: str | None) -> dict[str, object]:
    metadata: dict[str, object] = {
        "path": str(root),
        "revision": revision,
        "interface": "src/grits.py cell-grid functions",
        "provenance": (
            "caller-supplied scorer path; revision not supplied"
            if revision is None
            else "caller-supplied scorer path and independently established revision"
        ),
    }
    if revision == SCORER_REVISION:
        metadata.update(
            {
                "repository": "https://github.com/microsoft/table-transformer",
                "license": "MIT",
            }
        )
    return metadata


def _model_metadata(
    root: Path,
    revision: str | None,
    checkpoint_file: str,
) -> dict[str, object]:
    checkpoint = _checkpoint_path(root, checkpoint_file)
    metadata: dict[str, object] = {
        "checkpoint": str(checkpoint),
        "revision": revision,
        "provenance": (
            "caller-supplied checkpoint path; revision not supplied"
            if revision is None
            else "caller-supplied checkpoint path and independently established revision"
        ),
        "preprocessing": (
            "supplied structure transform after a fixed 5 px tight crop; "
            "predicted boxes are translated to source-image coordinates"
        ),
    }
    if revision == MODEL_REVISION and checkpoint_file == MODEL_FILE:
        metadata.update(
            {
                "id": MODEL_ID,
                "license": MODEL_LICENSE,
                "checkpoint_size_bytes": MODEL_SIZE_BYTES,
                "architecture": "Table Transformer with ResNet-18 backbone",
                "origin": "Developed at Microsoft from Facebook DETR and ResNet-18",
                "preprocessing": (
                    "Microsoft structure transform after a fixed 5 px tight crop; "
                    "predicted boxes are translated to source-image coordinates"
                ),
            }
        )
    elif revision == ALL_MODEL_REVISION and checkpoint_file == ALL_MODEL_FILE:
        metadata.update(
            {
                "id": ALL_MODEL_ID,
                "license": MODEL_LICENSE,
                "checkpoint_size_bytes": MODEL_SIZE_BYTES,
                "architecture": "Table Transformer with ResNet-18 backbone",
                "origin": "Developed at Microsoft from Facebook DETR and ResNet-18",
                "training_data": "PubTables-1M and FinTabNet.c",
                "preprocessing": (
                    "Microsoft structure transform after a fixed 5 px tight crop; "
                    "predicted boxes are translated to source-image coordinates"
                ),
            }
        )
    return metadata


def _checkpoint_path(root: Path, checkpoint_file: str) -> Path:
    if not checkpoint_file or Path(checkpoint_file).name != checkpoint_file:
        raise ValueError("Checkpoint file must be a filename")
    return root / checkpoint_file


def _load_cases(root: Path, limit: int) -> list[Case]:
    if not root.is_dir():
        raise ValueError(f"PubTables dataset root is missing: {root}")
    prepared = sorted(root.glob("*/ground_truth.json"))
    if prepared:
        cases = [_prepared_case(path) for path in _even_panel(prepared, limit)]
    else:
        test_root = root / "test"
        words_root = root / "words"
        xml_paths = sorted(test_root.glob("*.xml"))
        if not xml_paths or not words_root.is_dir():
            raise ValueError(
                "Expected prepared ground_truth.json cases or extracted test/ and words/"
            )
        cases = [
            _raw_case(path, words_root, root / "images")
            for path in _even_panel(xml_paths, limit)
        ]
    ids = [case.case_id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("PubTables panel contains duplicate case IDs")
    return cases


def _prepared_case(path: Path) -> Case:
    payload = _mapping(_read_json(path), "ground truth")
    case_id = payload.get("case_id")
    if not isinstance(case_id, str) or not case_id:
        raise ValueError(f"Ground truth lacks case_id: {path}")
    target = _mapping(payload.get("target"), "target")
    if target.get("representation") != "pubtables_structure_and_words":
        raise ValueError(f"Unsupported PubTables target: {path}")
    input_files = payload.get("input_files")
    image_path = None
    if isinstance(input_files, list) and len(input_files) == 1:
        image_path = path.parent / str(input_files[0])
    return Case(case_id, target, image_path)


def _raw_case(xml_path: Path, words_root: Path, image_root: Path) -> Case:
    stem = xml_path.stem
    candidates = [words_root / f"{stem}_words.json", words_root / f"{stem}.json"]
    matches = [path for path in candidates if path.is_file()]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one word annotation for {stem}, found {len(matches)}"
        )
    image_matches = [
        path
        for suffix in (".jpg", ".jpeg", ".png")
        if (path := image_root / f"{stem}{suffix}").is_file()
    ]
    return Case(
        stem,
        {
            "representation": "pubtables_structure_and_words",
            "structure": {
                "representation": "pascal_voc_xml",
                "xml": xml_path.read_text(encoding="utf-8-sig"),
            },
            "words": {
                "representation": "word_boxes_json",
                "data": _read_json(matches[0]),
            },
        },
        image_matches[0] if len(image_matches) == 1 else None,
    )


def _even_panel(paths: list[Path], limit: int) -> list[Path]:
    if len(paths) <= limit:
        return paths
    size = len(paths)
    return [paths[((2 * index + 1) * size) // (2 * limit)] for index in range(limit)]


def _load_prediction(root: Path, case_id: str) -> dict[str, Any]:
    if not root.is_dir():
        raise ValueError(f"Prediction directory is missing: {root}")
    candidates = (
        root / f"{case_id}.json",
        root / f"{case_id}_cells.json",
        root / f"{case_id}_0_objects.json",
    )
    matches = [path for path in candidates if path.is_file()]
    if not matches:
        return _failed_prediction("missing_prediction")
    if len(matches) > 1:
        return _failed_prediction("ambiguous_prediction_files")
    try:
        payload = _read_json(matches[0])
        if isinstance(payload, list):
            return _prediction_record("success", payload, {})
        value = _mapping(payload, "prediction")
        operations = value.get("operations")
        operations = operations if isinstance(operations, dict) else {}
        body = value.get("prediction", value)
        body = _mapping(body, "prediction body")
        status = value.get("status", body.get("status", "success"))
        if operations.get("failed"):
            status = "failed"
        elif operations.get("abstained"):
            status = "abstained"
        if status not in {"success", "failed", "abstained"}:
            raise ValueError(f"invalid status: {status!r}")
        return _prediction_record(
            str(status), body.get("cells", []), operations | value
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        return _failed_prediction(f"invalid_prediction: {error}")


def _prediction_record(
    status: str, cells: Any, metadata: dict[str, Any]
) -> dict[str, Any]:
    return {
        "status": status,
        "cells": cells,
        "latency_ms": _optional_number(metadata.get("latency_ms")),
        "peak_gpu_memory_mb": _optional_number(
            metadata.get("peak_gpu_memory_mb", metadata.get("gpu_memory_mb"))
        ),
        "cost_usd": _optional_number(metadata.get("cost_usd")),
        "failure": None if status == "success" else status,
    }


def _failed_prediction(reason: str) -> dict[str, Any]:
    return {
        "status": "failed",
        "cells": [],
        "latency_ms": None,
        "peak_gpu_memory_mb": None,
        "cost_usd": None,
        "failure": reason,
    }


def _xml_objects(value: Any) -> tuple[list[list[float]], list[int]]:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("PubTables XML must be non-empty text")
    try:
        root = ET.fromstring(value)
    except ET.ParseError as error:
        raise ValueError(f"Invalid PubTables XML: {error}") from error
    label_ids = {name: index for index, name in enumerate(CLASS_NAMES)}
    boxes: list[list[float]] = []
    labels: list[int] = []
    for item in root.findall("object"):
        label = item.findtext("name")
        box = item.find("bndbox")
        if label not in label_ids or label == "no object" or box is None:
            raise ValueError(f"Invalid PubTables object label or box: {label!r}")
        try:
            bbox = [
                float(box.findtext(name, ""))
                for name in ("xmin", "ymin", "xmax", "ymax")
            ]
        except ValueError as error:
            raise ValueError("PubTables object has a non-numeric box") from error
        if _bbox(bbox) is None:
            raise ValueError("PubTables object has an invalid box")
        boxes.append(bbox)
        labels.append(label_ids[label])
    if not boxes or 1 not in labels or 2 not in labels:
        raise ValueError("PubTables XML lacks table rows or columns")
    return boxes, labels


def _word_tokens(value: Any) -> list[dict[str, Any]]:
    tokens = value.get("words") if isinstance(value, dict) else value
    if not isinstance(tokens, list):
        raise ValueError("PubTables words must be a list or contain a words list")
    result = []
    for index, token in enumerate(tokens):
        if not isinstance(token, dict) or not isinstance(token.get("text"), str):
            raise ValueError("PubTables word is missing text")
        bbox = _bbox(token.get("bbox"))
        if bbox is None:
            raise ValueError("PubTables word has an invalid box")
        result.append(
            {
                **token,
                "bbox": bbox,
                "span_num": token.get("span_num", index),
                "line_num": token.get("line_num", 0),
                "block_num": token.get("block_num", 0),
            }
        )
    return result


def _normalize_cells(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) > MAX_CELLS:
        raise ValueError("Prediction cells must be a bounded list")
    normalized = []
    max_row = -1
    max_column = -1
    for cell in value:
        if not isinstance(cell, dict):
            raise ValueError("Each prediction cell must be an object")
        rows = _indexes(cell.get("row_nums"))
        columns = _indexes(cell.get("column_nums"))
        bbox = _bbox(cell.get("bbox"))
        text = cell.get("cell_text", cell.get("cell text"))
        if rows is None or columns is None or bbox is None or not isinstance(text, str):
            raise ValueError("Prediction cell lacks a valid span, box, or text")
        max_row = max(max_row, rows[-1])
        max_column = max(max_column, columns[-1])
        if (max_row + 1) * (max_column + 1) > MAX_GRID_AREA:
            raise ValueError("Prediction grid is too large")
        normalized.append(
            {
                "row_nums": rows,
                "column_nums": columns,
                "bbox": bbox,
                "cell_text": text,
            }
        )
    return normalized


def _indexes(value: Any) -> list[int] | None:
    if (
        not isinstance(value, list)
        or not value
        or len(value) > MAX_INDEX
        or any(
            isinstance(item, bool)
            or not isinstance(item, int)
            or not 0 <= item < MAX_INDEX
            for item in value
        )
    ):
        return None
    result = sorted(set(value))
    if len(result) != len(value) or result != list(range(result[0], result[-1] + 1)):
        return None
    return result


def _bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    if any(
        isinstance(item, bool) or not isinstance(item, int | float) for item in value
    ):
        return None
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        return None
    if result[0] > result[2] or result[1] > result[3]:
        return None
    return result


def _cell_exact_match(
    gold: list[dict[str, Any]], predicted: list[dict[str, Any]]
) -> float:
    def handle_key(
        cell: dict[str, Any],
    ) -> tuple[tuple[int, ...], tuple[int, ...], str]:
        return tuple(cell["row_nums"]), tuple(cell["column_nums"]), cell["cell_text"]

    true_cells = Counter(handle_key(cell) for cell in gold)
    pred_cells = Counter(handle_key(cell) for cell in predicted)
    matches = sum((true_cells & pred_cells).values())
    denominator = sum(true_cells.values()) + sum(pred_cells.values())
    return 1.0 if denominator == 0 else 2 * matches / denominator


def _object_grid(rows: Any, np: ModuleType) -> Any:
    if not isinstance(rows, list) or not rows:
        return np.empty((0, 0), dtype=object)
    if not all(isinstance(row, list) for row in rows):
        raise ValueError("Official GriTS returned an invalid grid")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("Official GriTS returned a non-rectangular grid")
    grid = np.empty((len(rows), width), dtype=object)
    for row_index, row in enumerate(rows):
        for column_index, value in enumerate(row):
            grid[row_index, column_index] = value
    return grid


def _mean_scores(scores: list[dict[str, float]], denominator: int) -> dict[str, float]:
    return {
        name: round(sum(score[name] for score in scores) / denominator, 6)
        if denominator
        else 0.0
        for name in METRICS
    }


def _operation_metrics(records: list[dict[str, Any]]) -> dict[str, object]:
    latencies = [
        float(item["latency_ms"]) for item in records if item["latency_ms"] is not None
    ]
    memories = [
        float(item["peak_gpu_memory_mb"])
        for item in records
        if item["peak_gpu_memory_mb"] is not None
    ]
    costs = [
        float(item["cost_usd"]) for item in records if item["cost_usd"] is not None
    ]
    total_latency_seconds = sum(latencies) / 1000
    attempted = len(records)
    failed = sum(item["status"] == "failed" for item in records)
    abstained = sum(item["status"] == "abstained" for item in records)
    p50 = _percentile(latencies, 0.50)
    p95 = _percentile(latencies, 0.95)
    complete_cost = len(costs) == attempted
    return {
        "latency_ms": {
            "observed": len(latencies),
            "missing": attempted - len(latencies),
            "p50": None if p50 is None else round(p50, 3),
            "p95": None if p95 is None else round(p95, 3),
        },
        "peak_gpu_memory_mb": max(memories, default=None),
        "throughput_tables_per_second": round(len(latencies) / total_latency_seconds, 6)
        if total_latency_seconds
        else None,
        "cost_observed": len(costs),
        "cost_missing": attempted - len(costs),
        "total_cost_usd": round(sum(costs), 8) if complete_cost else None,
        "cost_per_attempted_table_usd": round(sum(costs) / attempted, 8)
        if complete_cost
        else None,
        "failure_rate": round(failed / attempted, 6),
        "abstention_rate": round(abstained / attempted, 6),
    }


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def _tight_crop(image: Any, target: dict[str, Any]) -> tuple[Any, tuple[int, int]]:
    structure = _mapping(target.get("structure"), "structure")
    boxes, labels = _xml_objects(structure.get("xml"))
    table_boxes = [box for box, label in zip(boxes, labels, strict=True) if label == 0]
    if not table_boxes:
        raise ValueError("PubTables annotation contains no table box")
    box = table_boxes[0]
    left = max(0, math.floor(box[0]) - TATR_CROP_MARGIN)
    top = max(0, math.floor(box[1]) - TATR_CROP_MARGIN)
    right = min(image.width, math.ceil(box[2]) + TATR_CROP_MARGIN)
    bottom = min(image.height, math.ceil(box[3]) + TATR_CROP_MARGIN)
    if right <= left or bottom <= top:
        raise ValueError("PubTables table crop is empty")
    return image.crop((left, top, right, bottom)), (left, top)


def _crop_tokens(
    target: dict[str, Any], offset: tuple[int, int], size: tuple[int, int]
) -> list[dict[str, Any]]:
    words = _mapping(target.get("words"), "words")
    left, top = offset
    width, height = size
    result = []
    for token in _word_tokens(words.get("data")):
        box = token["bbox"]
        shifted = [
            max(0.0, box[0] - left),
            max(0.0, box[1] - top),
            min(float(width), box[2] - left),
            min(float(height), box[3] - top),
        ]
        if shifted[0] >= shifted[2] or shifted[1] >= shifted[3]:
            continue
        result.append({**token, "bbox": shifted})
    return result


def _translate_cells(value: Any, offset: tuple[int, int]) -> Any:
    if not isinstance(value, list):
        return value
    left, top = offset
    result = []
    for cell in value:
        if not isinstance(cell, dict) or _bbox(cell.get("bbox")) is None:
            result.append(cell)
            continue
        box = cell["bbox"]
        result.append(
            {
                **cell,
                "bbox": [box[0] + left, box[1] + top, box[2] + left, box[3] + top],
            }
        )
    return result


def _load_tatr_inference(root: Path) -> ModuleType:
    checkout = root if (root / "src" / "inference.py").is_file() else root.parent
    source = checkout / "src"
    detr = checkout / "detr"
    if not (source / "inference.py").is_file() or not detr.is_dir():
        raise ValueError("Supplied scorer checkout lacks inference sources")
    added = [str(source), str(detr)]
    sys.path[:0] = added
    try:
        return _load_module("pubtables_official_inference", source / "inference.py")
    finally:
        for path in added:
            sys.path.remove(path)


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import supplied scorer module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"PubTables {name} must be an object")
    return value


def _optional_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) and number >= 0 else None


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return number


if __name__ == "__main__":
    raise SystemExit(main())
