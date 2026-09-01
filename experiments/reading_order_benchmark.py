"""Compare lossless reading-order policies on one paragraph OCR result per page."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Sequence

from PIL import Image, ImageOps

from ocr_pipeline.contracts import TextRegion
from ocr_pipeline.providers import LocalReader, NemotronOCRV2Reader, ReaderError

if __package__:
    from .orientation_benchmark import (
        DEFAULT_RECTIFICATION_PADDING,
        ROTATIONS,
        detect_tesseract_orientation,
        rectify_document,
    )
    from .public_benchmark import (
        NORMALIZATION,
        BenchmarkCase,
        _score,
        discover_cases,
    )
else:
    from orientation_benchmark import (  # type: ignore[no-redef]
        DEFAULT_RECTIFICATION_PADDING,
        ROTATIONS,
        detect_tesseract_orientation,
        rectify_document,
    )
    from public_benchmark import (  # type: ignore[no-redef]
        NORMALIZATION,
        BenchmarkCase,
        _score,
        discover_cases,
    )

DEVELOPMENT_GAP_THRESHOLDS = (0.5, 1.0, 1.5)
MIN_SELECTION_CASES = 30
OSD_FAILURE_MODES = ("fail", "zero")
BASE_ARMS = ("baseline", "top_left")
EXPERIMENT_ID = "clinical_reading_order_policy"
SELECTION_RULE = (
    "minimum failure-inclusive micro WER, then micro CER, then lower threshold; "
    "eligible only on at least 30 cases when micro WER and CER both improve "
    "without lower coverage"
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare reading order with one paragraph OCR call per page"
    )
    parser.add_argument("root", type=Path, help="Extracted ClinOCR dataset root")
    parser.add_argument("output", type=Path, help="JSON results path")
    parser.add_argument("--mode", choices=("dev", "eval"), default="dev")
    parser.add_argument(
        "--gap-threshold",
        action="append",
        type=float,
        help="Development-only gap as a multiple of median region height",
    )
    parser.add_argument(
        "--dev-result",
        type=Path,
        help="Completed development result that selected the eval threshold",
    )
    parser.add_argument("--subset", action="append")
    parser.add_argument("--language", choices=("multi", "en"), default="en")
    parser.add_argument("--tesseract-osd", default="tesseract")
    parser.add_argument(
        "--osd-failure-mode",
        choices=OSD_FAILURE_MODES,
        default="fail",
        help="Development ablation used when Tesseract OSD returns no angle",
    )
    parser.add_argument(
        "--rectify-padding",
        type=float,
        default=DEFAULT_RECTIFICATION_PADDING,
    )
    args = parser.parse_args(argv)

    try:
        thresholds, selection_source = _cli_thresholds(
            args.mode,
            args.gap_threshold,
            args.dev_result,
            args.output,
        )
        if not 0 <= args.rectify_padding <= 0.25:
            raise ValueError("--rectify-padding must be between 0 and 0.25")
        role = "exemplar" if args.mode == "dev" else "eval"
        cases = discover_cases("clinocr", args.root, clinocr_role=role)
        if args.subset:
            selected = set(args.subset)
            unknown = selected - {case.subset for case in cases}
            if unknown:
                raise ValueError(f"Unknown subsets: {sorted(unknown)}")
            cases = [case for case in cases if case.subset in selected]
        if not cases:
            raise ValueError(f"No {role} cases found in {args.root}")

        reader = NemotronOCRV2Reader(
            language=args.language,
            merge_level="paragraph",
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)

        def handle_record(records: list[dict[str, object]], total_cases: int) -> None:
            record = records[-1]
            print(
                f"reading-order {len(records)}/{total_cases} "
                f"{record['id']} {record['status']}",
                file=sys.stderr,
                flush=True,
            )
            _write_json_atomic(
                args.output,
                {
                    "experiment": EXPERIMENT_ID,
                    "status": "running",
                    "mode": args.mode,
                    "completed_cases": len(records),
                    "total_cases": total_cases,
                    "cases": records,
                },
            )

        payload = run_experiment(
            cases,
            args.root,
            reader,
            mode=args.mode,
            gap_thresholds=thresholds,
            osd_executable=args.tesseract_osd,
            osd_failure_mode=args.osd_failure_mode,
            rectify_padding=args.rectify_padding,
            selection_source=selection_source,
            handle_record=handle_record,
        )
        _write_json_atomic(args.output, payload)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0


def run_experiment(
    cases: list[BenchmarkCase],
    root: Path,
    reader: LocalReader,
    *,
    mode: str,
    gap_thresholds: Sequence[float],
    osd_executable: str = "tesseract",
    osd_failure_mode: str = "fail",
    rectify_padding: float = DEFAULT_RECTIFICATION_PADDING,
    selection_source: str | None = None,
    handle_record: Callable[[list[dict[str, object]], int], None] | None = None,
) -> dict[str, object]:
    if not cases:
        raise ValueError("At least one benchmark case is required")
    thresholds = _thresholds(mode, gap_thresholds)
    if mode == "eval" and not selection_source:
        raise ValueError("Eval execution requires a development result source")
    if osd_failure_mode not in OSD_FAILURE_MODES:
        raise ValueError(f"Unsupported OSD failure mode: {osd_failure_mode}")
    records = []
    for case in cases:
        records.append(
            _evaluate_case(
                case,
                root,
                reader,
                thresholds,
                osd_executable,
                osd_failure_mode,
                rectify_padding,
            )
        )
        if handle_record:
            handle_record(records, len(cases))
    arm_names = [*BASE_ARMS, *(_adaptive_arm_name(value) for value in thresholds)]
    summary = _summarize_records(records, arm_names)
    selection = (
        _select_threshold(summary["arms"], thresholds)
        if mode == "dev"
        else {
            "arm": _adaptive_arm_name(thresholds[0]),
            "gap_threshold": thresholds[0],
            "source": "development_result",
            "development_result": selection_source,
            "eligible_for_eval": True,
            "statistical_evidence": "not_assessed_by_selection_rule",
        }
    )
    return {
        "experiment": EXPERIMENT_ID,
        "status": "complete",
        "dataset": "clinocr",
        "normalization": NORMALIZATION,
        "mode": mode,
        "case_ids": [case.id for case in cases],
        "config": {
            "reader": reader.name,
            "language": getattr(reader, "language", None),
            "merge_level": getattr(reader, "merge_level", None),
            "batch_size": getattr(reader, "batch_size", None),
            "paragraph_reader_method": _paragraph_reader_method(reader),
            "paragraph_reader_call_limit_per_page": 1,
            "osd_prepass_call_limit_per_page": 1,
            "rectification": True,
            "rectify_padding": rectify_padding,
            "orientation_selector": "tesseract_osd_direct",
            "osd_executable": osd_executable,
            "osd_failure_mode": osd_failure_mode,
            "gap_threshold_unit": "median_region_height_in_image_pixels",
            "gap_thresholds": list(thresholds),
            "threshold_source": selection_source if mode == "eval" else "development",
            "semantic_or_table_guards": False,
        },
        "selection": selection,
        "summary": summary,
        "subsets": {
            subset: _summarize_records(
                [record for record in records if record["subset"] == subset],
                arm_names,
            )
            for subset in sorted({str(record["subset"]) for record in records})
        },
        "cases": records,
    }


def baseline_order(regions: list[TextRegion]) -> list[TextRegion]:
    ordered = sorted(
        enumerate(regions), key=lambda item: (item[1].reading_order, item[0])
    )
    return [region for _, region in ordered]


def stable_top_left_order(regions: list[TextRegion]) -> list[TextRegion]:
    ordered = sorted(
        enumerate(regions),
        key=lambda item: (
            item[1].bounding_box.top,
            item[1].bounding_box.left,
            item[1].bounding_box.bottom,
            item[1].bounding_box.right,
            item[0],
        ),
    )
    return [region for _, region in ordered]


def adaptive_xy_cut_order(
    regions: list[TextRegion], gap_threshold: float
) -> list[TextRegion]:
    if not math.isfinite(gap_threshold) or gap_threshold <= 0:
        raise ValueError("gap_threshold must be finite and positive")
    if len(regions) < 2:
        return list(regions)
    median_height = statistics.median(
        region.bounding_box.bottom - region.bounding_box.top for region in regions
    )
    if median_height <= 0:
        raise ValueError("Region bounding boxes must have positive height")
    return _cut(regions, float(median_height), gap_threshold)


def validate_ordering(
    source: list[TextRegion],
    ordered: list[TextRegion],
    repeated: list[TextRegion],
    arm_name: str,
) -> dict[str, object]:
    same_objects = Counter(map(id, source)) == Counter(map(id, ordered))
    deterministic = [id(region) for region in ordered] == [
        id(region) for region in repeated
    ]
    coverage = len(ordered) / len(source) if source else 1.0
    report = {
        "source_regions": len(source),
        "ordered_regions": len(ordered),
        "coverage": round(coverage, 6),
        "same_object_multiset": same_objects,
        "deterministic": deterministic,
        "passed": same_objects and deterministic and coverage == 1.0,
    }
    if not report["passed"]:
        raise ValueError(f"Ordering invariant failed for {arm_name}: {report}")
    return report


def _cut(
    regions: list[TextRegion], median_height: float, gap_threshold: float
) -> list[TextRegion]:
    partition = _best_cut(regions, median_height * gap_threshold)
    if partition is None:
        return stable_top_left_order(regions)
    first, second = partition
    if not first or not second or max(len(first), len(second)) >= len(regions):
        return stable_top_left_order(regions)
    return [
        *_cut(first, median_height, gap_threshold),
        *_cut(second, median_height, gap_threshold),
    ]


def _best_cut(
    regions: list[TextRegion], minimum_gap: float
) -> tuple[list[TextRegion], list[TextRegion]] | None:
    candidates = []
    for axis in ("x", "y"):
        ordered = sorted(regions, key=lambda region: _axis_interval(region, axis))
        running_end = _axis_interval(ordered[0], axis)[1]
        for index, region in enumerate(ordered[1:], start=1):
            start, end = _axis_interval(region, axis)
            gap = start - running_end
            if gap >= minimum_gap:
                candidates.append((gap, axis == "x", ordered[:index], ordered[index:]))
            running_end = max(running_end, end)
    if not candidates:
        return None
    _, _, first, second = max(candidates, key=lambda candidate: candidate[:2])
    return first, second


def _axis_interval(region: TextRegion, axis: str) -> tuple[int, int]:
    box = region.bounding_box
    return (box.left, box.right) if axis == "x" else (box.top, box.bottom)


def _evaluate_case(
    case: BenchmarkCase,
    root: Path,
    reader: LocalReader,
    thresholds: tuple[float, ...],
    osd_executable: str,
    osd_failure_mode: str,
    rectify_padding: float,
) -> dict[str, object]:
    preprocessing: dict[str, object] = {
        "rectification_status": "not_run",
        "osd_prepass_calls": 0,
        "osd_prepass_status": "not_run",
    }
    regions: list[TextRegion] = []
    failure: dict[str, str] | None = None
    ocr_latency_ms = 0.0
    paragraph_reader_calls = 0
    with tempfile.TemporaryDirectory(prefix="ocr-reading-order-") as directory:
        try:
            prepared_path = _prepare_page(
                case.image_path,
                Path(directory),
                osd_executable,
                osd_failure_mode,
                rectify_padding,
                preprocessing,
            )
        except ReaderError as error:
            failure = {"code": error.code, "message": str(error)}
        except (OSError, ValueError) as error:
            failure = {"code": "page_preparation_failed", "message": str(error)}

        if failure is None:
            started = time.perf_counter()
            paragraph_reader_calls = 1
            try:
                regions = _read_paragraphs(reader, prepared_path)
            except ReaderError as error:
                failure = {"code": error.code, "message": str(error)}
            except (OSError, ValueError) as error:
                failure = {"code": "paragraph_reader_failed", "message": str(error)}
            finally:
                ocr_latency_ms = (time.perf_counter() - started) * 1000
            if failure is None and not regions:
                failure = {
                    "code": "no_text_regions",
                    "message": "The paragraph reader returned no text regions",
                }
            elif failure is None and not any(region.text.strip() for region in regions):
                failure = {
                    "code": "no_text_content",
                    "message": "The paragraph reader returned only blank text",
                }

    status = "success" if failure is None else "failed"
    source_regions = [
        _serialize_region(region, index) for index, region in enumerate(regions)
    ]

    orderers: dict[str, Callable[[list[TextRegion]], list[TextRegion]]] = {
        "baseline": baseline_order,
        "top_left": stable_top_left_order,
        **{
            _adaptive_arm_name(value): lambda items, value=value: adaptive_xy_cut_order(
                items, value
            )
            for value in thresholds
        },
    }
    arms = {}
    for arm_name, orderer in orderers.items():
        started = time.perf_counter()
        ordered = orderer(regions)
        ordering_latency_ms = (time.perf_counter() - started) * 1000
        repeated = orderer(regions)
        invariants = validate_ordering(regions, ordered, repeated, arm_name)
        prediction = "\n\n".join(region.text for region in ordered)
        arms[arm_name] = {
            "status": status,
            "failure_code": failure["code"] if failure else None,
            "prediction": prediction,
            "region_ids": [region.id for region in ordered],
            "source_indices": _source_indices(regions, ordered),
            "metrics": _score(prediction, case.reference),
            "ordering_latency_ms": round(ordering_latency_ms, 3),
            "invariants": invariants,
        }

    return {
        "id": case.id,
        "cluster_id": case.cluster_id,
        "subset": case.subset,
        "image": _relative_path(case.image_path, root),
        "reference": case.reference,
        "status": status,
        "failure": failure,
        "source_region_ids": [region.id for region in regions],
        "source_regions": source_regions,
        "median_region_height_px": (
            round(
                statistics.median(
                    region.bounding_box.bottom - region.bounding_box.top
                    for region in regions
                ),
                3,
            )
            if regions
            else None
        ),
        "preprocessing": preprocessing,
        "paragraph_reader_method": _paragraph_reader_method(reader),
        "paragraph_reader_calls": paragraph_reader_calls,
        "osd_prepass_calls": preprocessing["osd_prepass_calls"],
        "ocr_latency_ms": round(ocr_latency_ms, 3),
        "arms": arms,
    }


def _prepare_page(
    image_path: Path,
    directory: Path,
    osd_executable: str,
    osd_failure_mode: str,
    rectify_padding: float,
    details: dict[str, object],
) -> Path:
    started = time.perf_counter()
    with Image.open(image_path) as source:
        source_size = source.size
        exif_orientation = int(source.getexif().get(274, 1))
        normalized = ImageOps.exif_transpose(source)
        normalized_size = normalized.size
        try:
            rectified = rectify_document(
                normalized,
                padding_fraction=rectify_padding,
            )
        except ValueError as error:
            details["rectification_status"] = "failed"
            raise ReaderError("rectification_failed", str(error)) from error
    rectification_latency_ms = (time.perf_counter() - started) * 1000
    details.update(
        {
            "source_size": list(source_size),
            "exif_orientation": exif_orientation,
            "exif_transposed": exif_orientation in range(2, 9),
            "exif_normalized_size": list(normalized_size),
            "rectification_status": "success",
            "rectification_latency_ms": round(rectification_latency_ms, 3),
            "rectified_size": list(rectified.size),
        }
    )
    rectified_path = directory / "rectified.png"
    rectified.save(rectified_path, format="PNG")

    started = time.perf_counter()
    details["osd_prepass_calls"] = 1
    details["osd_prepass_status"] = "running"
    try:
        osd = detect_tesseract_orientation(rectified_path, executable=osd_executable)
    except ReaderError as error:
        osd_latency_ms = (time.perf_counter() - started) * 1000
        details["osd_prepass_failure"] = {
            "code": error.code,
            "message": str(error),
        }
        if osd_failure_mode == "fail":
            details["osd_prepass_status"] = "failed"
            raise
        details.update(
            {
                "osd_prepass_status": "fallback_zero",
                "osd_latency_ms": round(osd_latency_ms, 3),
                "osd_fallback_angle": 0,
                "prepared_width": rectified.width,
                "prepared_height": rectified.height,
            }
        )
        prepared_path = directory / "prepared.png"
        rectified.save(prepared_path, format="PNG")
        return prepared_path
    osd_latency_ms = (time.perf_counter() - started) * 1000
    angle = int(osd["angle"])
    if angle not in ROTATIONS:
        details["osd_prepass_status"] = "failed"
        raise ReaderError(
            "invalid_orientation_angle",
            f"Unsupported orientation angle: {angle}",
        )
    details["osd_prepass_status"] = "success"
    transform = ROTATIONS[angle]
    prepared = rectified.transpose(transform) if transform is not None else rectified
    prepared_path = directory / "prepared.png"
    prepared.save(prepared_path, format="PNG")
    details.update(
        {
            "osd_latency_ms": round(osd_latency_ms, 3),
            "osd": osd,
            "prepared_width": prepared.width,
            "prepared_height": prepared.height,
        }
    )
    return prepared_path


def _summarize_records(
    records: list[dict[str, object]], arm_names: list[str]
) -> dict[str, object]:
    successful = sum(record["status"] == "success" for record in records)
    failure_codes = Counter(
        str(record["failure"]["code"])
        for record in records
        if isinstance(record["failure"], dict)
    )
    return {
        "cases": len(records),
        "successful_ocr_cases": successful,
        "failed_cases": len(records) - successful,
        "coverage": round(successful / len(records), 6) if records else 0.0,
        "failure_codes": dict(sorted(failure_codes.items())),
        "all_ordering_invariants_passed": all(
            arm["invariants"]["passed"]
            for record in records
            for arm in record["arms"].values()
        ),
        "arms": {arm_name: _summarize_arm(records, arm_name) for arm_name in arm_names},
    }


def _summarize_arm(
    records: list[dict[str, object]], arm_name: str
) -> dict[str, object]:
    arm_records = [record["arms"][arm_name] for record in records]
    metrics = {}
    for metric_name in ("cer", "wer"):
        values = [arm["metrics"][metric_name] for arm in arm_records]
        edits = sum(int(value["edits"]) for value in values)
        units = sum(int(value["reference_units"]) for value in values)
        metrics[metric_name] = {
            "case_mean": round(
                sum(float(value["rate"]) for value in values) / len(values), 6
            ),
            "micro": round(edits / units, 6) if units else 0.0,
            "edits": edits,
            "reference_units": units,
        }
    covered = sum(arm["status"] == "success" for arm in arm_records)
    failure_codes = Counter(
        str(arm["failure_code"])
        for arm in arm_records
        if arm["failure_code"] is not None
    )
    return {
        "cases": len(records),
        "covered_cases": covered,
        "coverage": round(covered / len(records), 6) if records else 0.0,
        "failure_codes": dict(sorted(failure_codes.items())),
        "ordering_latency_ms": round(
            sum(float(arm["ordering_latency_ms"]) for arm in arm_records), 3
        ),
        **metrics,
    }


def _select_threshold(
    arms: object,
    thresholds: tuple[float, ...],
) -> dict[str, object]:
    if not isinstance(arms, dict):
        raise ValueError("Development result has no arm summaries")

    def rank(threshold: float) -> tuple[float, float, float]:
        arm = arms.get(_adaptive_arm_name(threshold))
        if not isinstance(arm, dict):
            raise ValueError(f"Development result is missing threshold {threshold:g}")
        try:
            return (
                float(arm["wer"]["micro"]),
                float(arm["cer"]["micro"]),
                threshold,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Development arm metrics are invalid") from error

    selected = min(thresholds, key=rank)
    selected_arm = arms[_adaptive_arm_name(selected)]
    baseline_arm = arms.get("baseline")
    if not isinstance(baseline_arm, dict):
        raise ValueError("Development result is missing the baseline arm")
    try:
        selection_cases = int(baseline_arm["cases"])
        selected_cases = int(selected_arm["cases"])
        cer_delta = round(
            float(selected_arm["cer"]["micro"]) - float(baseline_arm["cer"]["micro"]),
            6,
        )
        wer_delta = round(
            float(selected_arm["wer"]["micro"]) - float(baseline_arm["wer"]["micro"]),
            6,
        )
        coverage_delta = round(
            float(selected_arm["coverage"]) - float(baseline_arm["coverage"]),
            6,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Development baseline metrics are invalid") from error
    if (
        selection_cases < 1
        or selected_cases != selection_cases
        or isinstance(baseline_arm["cases"], bool)
        or isinstance(selected_arm["cases"], bool)
    ):
        raise ValueError("Development arm case counts are invalid")
    return {
        "arm": _adaptive_arm_name(selected),
        "gap_threshold": selected,
        "rule": SELECTION_RULE,
        "evidence": "development_point_metrics_only",
        "micro_cer_delta": cer_delta,
        "micro_wer_delta": wer_delta,
        "coverage_delta": coverage_delta,
        "selection_cases": selection_cases,
        "minimum_selection_cases": MIN_SELECTION_CASES,
        "eligible_for_eval": (
            selection_cases >= MIN_SELECTION_CASES
            and cer_delta < 0
            and wer_delta < 0
            and coverage_delta >= 0
        ),
        "statistical_evidence": "not_assessed_by_selection_rule",
    }


def _cli_thresholds(
    mode: str,
    values: Sequence[float] | None,
    dev_result: Path | None,
    output: Path,
) -> tuple[tuple[float, ...], str | None]:
    if mode == "dev":
        if dev_result is not None:
            raise ValueError("--dev-result is only valid in eval mode")
        return _thresholds(mode, values), None
    if values is not None:
        raise ValueError(
            "Eval mode rejects --gap-threshold; use --dev-result selection"
        )
    if dev_result is None:
        raise ValueError("Eval mode requires --dev-result")
    if output.resolve() == dev_result.resolve():
        raise ValueError("Eval output must not overwrite the development result")
    return (_load_dev_threshold(dev_result),), str(dev_result)


def _load_dev_threshold(path: Path) -> float:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid development result JSON: {error}") from error
    if not isinstance(payload, dict) or payload.get("mode") != "dev":
        raise ValueError("Development result must be a dev-mode artifact")
    if (
        payload.get("experiment") != EXPERIMENT_ID
        or payload.get("dataset") != "clinocr"
    ):
        raise ValueError("Development result is not a ClinOCR reading-order artifact")
    if payload.get("status") != "complete":
        raise ValueError("Development result must be complete")
    try:
        configured = payload["config"]["gap_thresholds"]
        summary_arms = payload["summary"]["arms"]
        recorded = payload["selection"]
    except (KeyError, TypeError) as error:
        raise ValueError("Development result is missing selection evidence") from error
    if not isinstance(configured, list):
        raise ValueError("Development result has invalid gap thresholds")
    try:
        thresholds = _thresholds("dev", [float(value) for value in configured])
    except (TypeError, ValueError) as error:
        raise ValueError("Development result has invalid gap thresholds") from error
    expected = _select_threshold(summary_arms, thresholds)
    if not isinstance(recorded, dict) or recorded.get("rule") != SELECTION_RULE:
        raise ValueError("Development result uses an unknown selection rule")
    if recorded != expected:
        raise ValueError("Development selection does not match its point metrics")
    if not expected["eligible_for_eval"]:
        raise ValueError("Development candidate is not eligible for frozen evaluation")
    return float(expected["gap_threshold"])


def _thresholds(mode: str, values: Sequence[float] | None) -> tuple[float, ...]:
    if mode not in {"dev", "eval"}:
        raise ValueError(f"Unsupported mode: {mode}")
    thresholds = tuple(DEVELOPMENT_GAP_THRESHOLDS if values is None else values)
    if not thresholds or any(
        not math.isfinite(value) or value <= 0 for value in thresholds
    ):
        raise ValueError("Gap thresholds must be finite and positive")
    if mode == "eval" and (values is None or len(thresholds) != 1):
        raise ValueError("Eval execution requires exactly one selected gap threshold")
    if len(set(thresholds)) != len(thresholds):
        raise ValueError("Gap thresholds must be unique")
    return thresholds


def _adaptive_arm_name(gap_threshold: float) -> str:
    return f"adaptive_xy_cut_{gap_threshold:g}"


def _serialize_region(region: TextRegion, source_index: int) -> dict[str, object]:
    box = region.bounding_box
    return {
        "source_index": source_index,
        "id": region.id,
        "kind": region.kind,
        "text": region.text,
        "confidence": region.confidence,
        "bounding_box": {
            "left": box.left,
            "top": box.top,
            "right": box.right,
            "bottom": box.bottom,
        },
        "reading_order": region.reading_order,
        "provider": region.provider,
        "text_provenance": region.text_provenance,
    }


def _source_indices(source: list[TextRegion], ordered: list[TextRegion]) -> list[int]:
    positions: dict[int, list[int]] = {}
    for index, region in enumerate(source):
        positions.setdefault(id(region), []).append(index)
    for indexes in positions.values():
        indexes.reverse()
    return [positions[id(region)].pop() for region in ordered]


def _paragraph_reader_method(reader: LocalReader) -> str:
    return (
        "read_with_merge_level"
        if callable(getattr(reader, "read_with_merge_level", None))
        else "read"
    )


def _read_paragraphs(reader: LocalReader, image_path: Path) -> list[TextRegion]:
    read_with_merge_level = getattr(reader, "read_with_merge_level", None)
    if callable(read_with_merge_level):
        return read_with_merge_level(image_path, 1, "paragraph")
    return reader.read(image_path, 1)


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            temporary_path = Path(stream.name)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
