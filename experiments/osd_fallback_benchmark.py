"""Measure a fixed, selective fallback when Tesseract OSD fails."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Sequence

from ocr_pipeline.contracts import BoundingBox, TextRegion
from ocr_pipeline.providers import LocalReader, NemotronOCRV2Reader, ReaderError

if __package__:
    from .orientation_benchmark import DEFAULT_RECTIFICATION_PADDING
    from .public_benchmark import (
        NORMALIZATION,
        BenchmarkCase,
        _score,
        _summarize,
        discover_cases,
    )
    from .reading_order_benchmark import (
        _paragraph_reader_method,
        _prepare_page,
        _read_paragraphs,
        _relative_path,
        _write_json_atomic,
        baseline_order,
    )
else:
    from orientation_benchmark import DEFAULT_RECTIFICATION_PADDING  # type: ignore[no-redef]
    from public_benchmark import (  # type: ignore[no-redef]
        NORMALIZATION,
        BenchmarkCase,
        _score,
        _summarize,
        discover_cases,
    )
    from reading_order_benchmark import (  # type: ignore[no-redef]
        _paragraph_reader_method,
        _prepare_page,
        _read_paragraphs,
        _relative_path,
        _write_json_atomic,
        baseline_order,
    )

EXPERIMENT_ID = "clinical_osd_selective_fallback"
DEV_CASES = 56
EVAL_CASES = 328
MIN_ATTEMPTED_CASES = 30
MIN_PARAGRAPH_REGIONS = 10
MIN_MEAN_CONFIDENCE = 0.92
ARM_NAMES = ("strict_osd", "selective_fallback")
POLICY = {
    "preparation": "exif_transpose_then_rectify_then_tesseract_osd",
    "osd_failure_angle_degrees": 0,
    "fallback_min_paragraph_regions": MIN_PARAGRAPH_REGIONS,
    "fallback_min_mean_available_confidence": MIN_MEAN_CONFIDENCE,
    "fallback_confidence_requirement": "every_paragraph_region",
    "strict_arm": "empty_on_every_osd_fallback",
    "selective_arm": "accept_only_when_both_fallback_thresholds_pass",
    "merge_level": "paragraph",
    "ocr_runs_per_attempted_page": 1,
}
PROMOTION_RULE = (
    "at least 30 attempted development cases; selective failure-inclusive micro "
    "CER and WER both improve over strict OSD; coverage does not fall; every "
    "accepted fallback case has CER below 0.2"
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark a fixed selective fallback for Tesseract OSD failure"
    )
    parser.add_argument("root", type=Path, help="Extracted ClinOCR dataset root")
    parser.add_argument("output", type=Path, help="JSON results path")
    parser.add_argument("--mode", choices=("dev", "eval"), default="dev")
    parser.add_argument("--dev-result", type=Path)
    parser.add_argument("--language", choices=("multi", "en"), default="en")
    parser.add_argument("--tesseract", default="tesseract")
    parser.add_argument(
        "--rectify-padding",
        type=float,
        default=DEFAULT_RECTIFICATION_PADDING,
    )
    args = parser.parse_args(argv)

    output_claim_inode: int | None = None
    try:
        if not 0 <= args.rectify_padding <= 0.25:
            raise ValueError("--rectify-padding must be between 0 and 0.25")
        if args.mode == "dev" and args.dev_result is not None:
            raise ValueError("--dev-result is only valid in eval mode")
        if args.mode == "eval" and args.dev_result is None:
            raise ValueError("Eval mode requires --dev-result")
        if args.dev_result and args.output.resolve() == args.dev_result.resolve():
            raise ValueError("Eval output must not overwrite the development result")

        output_claim_inode = _claim_output(args.output)

        role = "exemplar" if args.mode == "dev" else "eval"
        cases = discover_cases("clinocr", args.root, clinocr_role=role)
        dev_cases = (
            discover_cases("clinocr", args.root, clinocr_role="exemplar")
            if args.mode == "eval"
            else None
        )
        reader = NemotronOCRV2Reader(language=args.language, merge_level="paragraph")

        def handle_record(records: list[dict[str, object]], total: int) -> None:
            record = records[-1]
            print(
                f"osd-fallback {len(records)}/{total} {record['id']} {record['status']}",
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
                    "total_cases": total,
                    "cases": records,
                },
            )

        payload = run_experiment(
            cases,
            args.root,
            reader,
            mode=args.mode,
            dev_result=args.dev_result,
            dev_cases=dev_cases,
            osd_executable=args.tesseract,
            rectify_padding=args.rectify_padding,
            handle_record=handle_record,
        )
        _write_json_atomic(args.output, payload)
    except (OSError, ValueError) as error:
        if output_claim_inode is not None:
            _release_empty_claim(args.output, output_claim_inode)
        parser.error(str(error))
    return 0


def run_experiment(
    cases: list[BenchmarkCase],
    root: Path,
    reader: LocalReader,
    *,
    mode: str,
    dev_result: Path | None = None,
    dev_cases: list[BenchmarkCase] | None = None,
    osd_executable: str = "tesseract",
    rectify_padding: float = DEFAULT_RECTIFICATION_PADDING,
    handle_record: Callable[[list[dict[str, object]], int], None] | None = None,
) -> dict[str, object]:
    _validate_panel(cases, mode)
    if not 0 <= rectify_padding <= 0.25:
        raise ValueError("Rectification padding must be between 0 and 0.25")
    if mode == "dev":
        if dev_result is not None or dev_cases is not None:
            raise ValueError("Development execution cannot use a development result")
        development_source = None
    else:
        if dev_result is None or dev_cases is None:
            raise ValueError("Eval execution requires its development result and panel")
        _validate_panel(dev_cases, "dev")
        dev_payload = _validate_dev_artifact(dev_result, dev_cases, root)
        dev_config = dev_payload["config"]
        current_config = {
            "reader": reader.name,
            "language": getattr(reader, "language", None),
            "rectify_padding": rectify_padding,
            "osd_executable": osd_executable,
        }
        if any(dev_config.get(key) != value for key, value in current_config.items()):
            raise ValueError(
                "Eval execution config must match the development artifact"
            )
        development_source = str(dev_result)

    records: list[dict[str, object]] = []
    for case in cases:
        records.append(
            _evaluate_case(
                case,
                root,
                reader,
                osd_executable,
                rectify_padding,
            )
        )
        if handle_record:
            handle_record(records, len(cases))

    summary = _summarize_records(records)
    selection = (
        _select_promotion(records, summary)
        if mode == "dev"
        else {
            "promotable": True,
            "rule": PROMOTION_RULE,
            "source": "validated_development_result",
            "development_result": development_source,
        }
    )
    return {
        "experiment": EXPERIMENT_ID,
        "status": "complete",
        "dataset": "clinocr",
        "dataset_root": str(root.resolve()),
        "mode": mode,
        "case_ids": [case.id for case in cases],
        "normalization": NORMALIZATION,
        "policy": dict(POLICY),
        "config": {
            "reader": reader.name,
            "language": getattr(reader, "language", None),
            "merge_level": "paragraph",
            "paragraph_reader_method": _paragraph_reader_method(reader),
            "rectification": True,
            "rectify_padding": rectify_padding,
            "orientation_selector": "tesseract_osd_then_fixed_zero_fallback",
            "osd_executable": osd_executable,
            "osd_failure_mode": "zero",
            "development_result": development_source,
        },
        "selection": selection,
        "summary": summary,
        "subsets": {
            subset: _summarize_records(
                [record for record in records if record["subset"] == subset]
            )
            for subset in sorted({str(record["subset"]) for record in records})
        },
        "cases": records,
    }


def _evaluate_case(
    case: BenchmarkCase,
    root: Path,
    reader: LocalReader,
    osd_executable: str,
    rectify_padding: float,
) -> dict[str, object]:
    preprocessing: dict[str, object] = {
        "rectification_status": "not_run",
        "osd_prepass_calls": 0,
        "osd_prepass_status": "not_run",
    }
    regions = []
    page_failure: dict[str, str] | None = None
    ocr_calls = 0
    ocr_latency_ms = 0.0
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="ocr-osd-fallback-") as directory:
        try:
            prepared_path = _prepare_page(
                case.image_path,
                Path(directory),
                osd_executable,
                "zero",
                rectify_padding,
                preprocessing,
            )
        except ReaderError as error:
            page_failure = _failure(error.code, str(error), "preparation")
        except (OSError, ValueError) as error:
            page_failure = _failure(
                "page_preparation_failed", str(error), "preparation"
            )

        if page_failure is None:
            ocr_started = time.perf_counter()
            ocr_calls = 1
            try:
                regions = _read_paragraphs(reader, prepared_path)
                _validate_regions(regions)
            except ReaderError as error:
                page_failure = _failure(error.code, str(error), "paragraph_reader")
            except (AttributeError, OSError, TypeError, ValueError) as error:
                page_failure = _failure(
                    "paragraph_reader_failed", str(error), "paragraph_reader"
                )
            finally:
                ocr_latency_ms = (time.perf_counter() - ocr_started) * 1000
    total_latency_ms = (time.perf_counter() - started) * 1000

    osd_fallback = preprocessing.get("osd_prepass_status") == "fallback_zero"
    confidences = [
        float(region.confidence) for region in regions if region.confidence is not None
    ]
    mean_confidence = sum(confidences) / len(confidences) if confidences else None
    fallback_accepted = bool(
        osd_fallback
        and page_failure is None
        and len(regions) >= MIN_PARAGRAPH_REGIONS
        and len(confidences) == len(regions)
        and mean_confidence is not None
        and mean_confidence >= MIN_MEAN_CONFIDENCE
    )
    evidence = {
        "osd_fallback": osd_fallback,
        "fallback_angle_degrees": 0 if osd_fallback else None,
        "paragraph_regions": len(regions),
        "available_confidences": len(confidences),
        "all_regions_have_confidence": bool(regions)
        and len(confidences) == len(regions),
        "mean_available_confidence": (
            round(mean_confidence, 6) if mean_confidence is not None else None
        ),
        "minimum_paragraph_regions": MIN_PARAGRAPH_REGIONS,
        "minimum_mean_available_confidence": MIN_MEAN_CONFIDENCE,
        "fallback_accepted": fallback_accepted,
    }
    ordered_text = "\n\n".join(region.text for region in baseline_order(regions))
    if not ordered_text.strip() and page_failure is None:
        page_failure = _failure(
            "no_text_content",
            "The paragraph reader returned no nonblank text",
            "paragraph_reader",
        )
        fallback_accepted = False
        evidence["fallback_accepted"] = False

    strict_failure = page_failure
    strict_prediction = ordered_text
    if osd_fallback:
        strict_prediction = ""
        strict_failure = _failure(
            "osd_fallback_disallowed",
            "Strict OSD emits no text when Tesseract OSD fails",
            "orientation",
        )
    selective_failure = page_failure
    selective_prediction = ordered_text
    if osd_fallback and not fallback_accepted:
        selective_prediction = ""
        selective_failure = _failure(
            "osd_fallback_abstained",
            "Zero-degree fallback did not satisfy the fixed acceptance policy",
            "orientation",
        )

    arms = {
        "strict_osd": _arm_record(
            strict_prediction, case.reference, strict_failure, total_latency_ms
        ),
        "selective_fallback": _arm_record(
            selective_prediction, case.reference, selective_failure, total_latency_ms
        ),
    }
    return {
        "id": case.id,
        "cluster_id": case.cluster_id,
        "subset": case.subset,
        "image": _relative_path(case.image_path, root),
        "reference": case.reference,
        "status": arms["selective_fallback"]["status"],
        "page_failure": page_failure,
        "preprocessing": preprocessing,
        "fallback_evidence": evidence,
        "region_ids": [region.id for region in regions],
        "region_confidences": [region.confidence for region in regions],
        "regions": [_serialize_region(region) for region in regions],
        "ocr_calls": ocr_calls,
        "ocr_latency_ms": round(ocr_latency_ms, 3),
        "total_latency_ms": round(total_latency_ms, 3),
        "arms": arms,
    }


def _arm_record(
    prediction: str,
    reference: str,
    failure: dict[str, str] | None,
    latency_ms: float,
) -> dict[str, object]:
    return {
        "prediction": prediction,
        "status": "success" if failure is None else "failed",
        "metrics": _score(prediction, reference),
        "failures": [failure] if failure else [],
        "latency_ms": round(latency_ms, 3),
    }


def _summarize_records(records: list[dict[str, object]]) -> dict[str, object]:
    return {
        "cases": len(records),
        "attempted_cases": sum(int(record["ocr_calls"]) == 1 for record in records),
        "osd_fallback_cases": sum(
            bool(record["fallback_evidence"]["osd_fallback"])
            for record in records  # type: ignore[index]
        ),
        "accepted_fallback_cases": sum(
            bool(record["fallback_evidence"]["fallback_accepted"])  # type: ignore[index]
            for record in records
        ),
        "arms": {
            arm_name: _summarize(
                [record["arms"][arm_name] for record in records]  # type: ignore[index]
            )
            for arm_name in ARM_NAMES
        },
    }


def _select_promotion(
    records: list[dict[str, object]], summary: dict[str, object]
) -> dict[str, object]:
    arms = summary["arms"]
    strict = arms["strict_osd"]  # type: ignore[index]
    selective = arms["selective_fallback"]  # type: ignore[index]
    attempted = int(summary["attempted_cases"])
    cer_delta = round(
        float(selective["cer"]["micro"]) - float(strict["cer"]["micro"]), 6
    )
    wer_delta = round(
        float(selective["wer"]["micro"]) - float(strict["wer"]["micro"]), 6
    )
    coverage_delta = round(float(selective["coverage"]) - float(strict["coverage"]), 6)
    accepted_cer = [
        float(record["arms"]["selective_fallback"]["metrics"]["cer"]["rate"])  # type: ignore[index]
        for record in records
        if record["fallback_evidence"]["fallback_accepted"]  # type: ignore[index]
    ]
    accepted_below_limit = all(value < 0.2 for value in accepted_cer)
    promotable = (
        attempted >= MIN_ATTEMPTED_CASES
        and cer_delta < 0
        and wer_delta < 0
        and coverage_delta >= 0
        and accepted_below_limit
    )
    return {
        "rule": PROMOTION_RULE,
        "attempted_cases": attempted,
        "minimum_attempted_cases": MIN_ATTEMPTED_CASES,
        "micro_cer_delta": cer_delta,
        "micro_wer_delta": wer_delta,
        "coverage_delta": coverage_delta,
        "accepted_fallback_cases": len(accepted_cer),
        "maximum_accepted_fallback_cer": (
            round(max(accepted_cer), 6) if accepted_cer else None
        ),
        "all_accepted_fallback_cer_below_0.2": accepted_below_limit,
        "promotable": promotable,
        "statistical_evidence": "not_assessed_by_fixed_promotion_rule",
    }


def _validate_dev_artifact(
    path: Path, expected_cases: list[BenchmarkCase], root: Path
) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid development result JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("Development result must be a JSON object")
    if (
        payload.get("experiment") != EXPERIMENT_ID
        or payload.get("dataset") != "clinocr"
        or payload.get("dataset_root") != str(root.resolve())
        or payload.get("mode") != "dev"
        or payload.get("status") != "complete"
    ):
        raise ValueError("Development result is not a complete dev artifact")
    if payload.get("policy") != POLICY:
        raise ValueError("Development result does not use the exact fixed policy")
    config = payload.get("config")
    if not isinstance(config, dict) or (
        config.get("reader") != "nemotron-ocr-v2"
        or config.get("language") not in {"en", "multi"}
        or config.get("merge_level") != "paragraph"
        or config.get("paragraph_reader_method") != "read_with_merge_level"
        or config.get("rectification") is not True
        or not isinstance(config.get("rectify_padding"), (int, float))
        or not 0 <= float(config["rectify_padding"]) <= 0.25
        or not isinstance(config.get("osd_executable"), str)
        or not config["osd_executable"]
        or config.get("osd_failure_mode") != "zero"
        or config.get("orientation_selector")
        != "tesseract_osd_then_fixed_zero_fallback"
        or config.get("development_result") is not None
    ):
        raise ValueError("Development result has an incompatible execution config")
    records = payload.get("cases")
    if not isinstance(records, list) or len(records) != DEV_CASES:
        raise ValueError("Development result must contain all 56 exemplar cases")
    if payload.get("case_ids") != [case.id for case in expected_cases]:
        raise ValueError(
            "Development result case panel does not match ClinOCR exemplars"
        )
    for record, case in zip(records, expected_cases, strict=True):
        _validate_dev_record(record, case, root)
    expected_summary = _summarize_records(records)
    if payload.get("summary") != expected_summary:
        raise ValueError("Development summary does not match its case records")
    expected_selection = _select_promotion(records, expected_summary)
    if payload.get("selection") != expected_selection:
        raise ValueError("Development selection does not match its evidence")
    if not expected_selection["promotable"]:
        raise ValueError("Development policy is not eligible for frozen evaluation")
    return payload


def _validate_dev_record(record: object, case: BenchmarkCase, root: Path) -> None:
    if not isinstance(record, dict):
        raise ValueError("Development case record is invalid")
    expected_link = {
        "id": case.id,
        "cluster_id": case.cluster_id,
        "subset": case.subset,
        "image": _relative_path(case.image_path, root),
        "reference": case.reference,
    }
    if any(record.get(key) != value for key, value in expected_link.items()):
        raise ValueError(f"Development source linkage changed for {case.id}")
    arms = record.get("arms")
    if not isinstance(arms, dict) or set(arms) != set(ARM_NAMES):
        raise ValueError(f"Development arms are invalid for {case.id}")
    for arm_name in ARM_NAMES:
        arm = arms[arm_name]
        if not isinstance(arm, dict):
            raise ValueError(f"Development arm is invalid for {case.id}")
        prediction = arm.get("prediction")
        if not isinstance(prediction, str):
            raise ValueError(f"Development prediction is invalid for {case.id}")
        if arm.get("metrics") != _score(prediction, case.reference):
            raise ValueError(f"Development metrics changed for {case.id}/{arm_name}")
    evidence = record.get("fallback_evidence")
    if not isinstance(evidence, dict):
        raise ValueError(f"Development fallback evidence is invalid for {case.id}")
    preprocessing = record.get("preprocessing")
    confidences = record.get("region_confidences")
    region_ids = record.get("region_ids")
    regions = _deserialize_regions(record.get("regions"), case.id)
    expected_region_ids = [region.id for region in regions]
    expected_confidences = [region.confidence for region in regions]
    if (
        not isinstance(preprocessing, dict)
        or not isinstance(confidences, list)
        or not isinstance(region_ids, list)
        or region_ids != expected_region_ids
        or confidences != expected_confidences
        or len(confidences) != len(region_ids)
        or evidence.get("paragraph_regions") != len(region_ids)
        or isinstance(record.get("ocr_calls"), bool)
        or record.get("ocr_calls") not in {0, 1}
    ):
        raise ValueError(f"Development observable evidence is invalid for {case.id}")
    page_failure = record.get("page_failure")
    if page_failure is not None and not isinstance(page_failure, dict):
        raise ValueError(f"Development page failure is invalid for {case.id}")
    total_latency = record.get("total_latency_ms")
    ocr_latency = record.get("ocr_latency_ms")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
        for value in (total_latency, ocr_latency)
    ):
        raise ValueError(f"Development latency is invalid for {case.id}")
    osd_status = preprocessing.get("osd_prepass_status")
    if osd_status in {"success", "fallback_zero"}:
        if preprocessing.get("osd_prepass_calls") != 1 or record["ocr_calls"] != 1:
            raise ValueError(f"Development call counts are invalid for {case.id}")
    elif record["ocr_calls"] != 0:
        raise ValueError(f"Development preparation state is invalid for {case.id}")
    available = []
    for confidence in confidences:
        if confidence is None:
            continue
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(float(confidence))
            or not 0 <= float(confidence) <= 1
        ):
            raise ValueError(f"Development confidence is invalid for {case.id}")
        available.append(float(confidence))
    mean_confidence = sum(available) / len(available) if available else None
    expected_mean = round(mean_confidence, 6) if mean_confidence is not None else None
    all_regions_have_confidence = bool(regions) and len(available) == len(regions)
    if (
        evidence.get("available_confidences") != len(available)
        or evidence.get("all_regions_have_confidence") != all_regions_have_confidence
        or evidence.get("mean_available_confidence") != expected_mean
        or evidence.get("minimum_paragraph_regions") != MIN_PARAGRAPH_REGIONS
        or evidence.get("minimum_mean_available_confidence") != MIN_MEAN_CONFIDENCE
        or evidence.get("osd_fallback")
        != (preprocessing.get("osd_prepass_status") == "fallback_zero")
        or evidence.get("fallback_angle_degrees")
        != (0 if evidence.get("osd_fallback") else None)
    ):
        raise ValueError(f"Development fallback evidence changed for {case.id}")
    if evidence.get("osd_fallback"):
        if not isinstance(preprocessing.get("osd_prepass_failure"), dict):
            raise ValueError(f"OSD failure evidence is missing for {case.id}")
        if arms["strict_osd"].get("prediction") != "":
            raise ValueError(f"Strict OSD retained fallback text for {case.id}")
        _validate_arm_state(
            arms["strict_osd"],
            _failure(
                "osd_fallback_disallowed",
                "Strict OSD emits no text when Tesseract OSD fails",
                "orientation",
            ),
            case.id,
            "strict_osd",
        )
        accepted = bool(evidence.get("fallback_accepted"))
        expected_accepted = bool(
            page_failure is None
            and len(region_ids) >= MIN_PARAGRAPH_REGIONS
            and all_regions_have_confidence
            and mean_confidence is not None
            and mean_confidence >= MIN_MEAN_CONFIDENCE
            and baseline_order(regions)
        )
        if accepted != expected_accepted:
            raise ValueError(f"Fallback acceptance changed for {case.id}")
        expected_prediction = (
            "\n\n".join(region.text for region in baseline_order(regions))
            if accepted
            else ""
        )
        if arms["selective_fallback"].get("prediction") != expected_prediction:
            raise ValueError(f"Selective fallback prediction changed for {case.id}")
        selective_failure = (
            None
            if accepted
            else _failure(
                "osd_fallback_abstained",
                "Zero-degree fallback did not satisfy the fixed acceptance policy",
                "orientation",
            )
        )
        _validate_arm_state(
            arms["selective_fallback"],
            selective_failure,
            case.id,
            "selective_fallback",
        )
    elif any(
        arms[arm_name].get("prediction")
        != "\n\n".join(region.text for region in baseline_order(regions))
        for arm_name in ARM_NAMES
    ):
        raise ValueError(f"Non-fallback prediction changed for {case.id}")
    else:
        for arm_name in ARM_NAMES:
            _validate_arm_state(arms[arm_name], page_failure, case.id, arm_name)
    if record.get("status") != arms["selective_fallback"].get("status") or any(
        arm.get("latency_ms") != total_latency for arm in arms.values()
    ):
        raise ValueError(f"Development derived-arm state changed for {case.id}")


def _validate_arm_state(
    arm: dict[str, object],
    failure: object,
    case_id: str,
    arm_name: str,
) -> None:
    expected_status = "success" if failure is None else "failed"
    expected_failures = [] if failure is None else [failure]
    latency = arm.get("latency_ms")
    if (
        arm.get("status") != expected_status
        or arm.get("failures") != expected_failures
        or isinstance(latency, bool)
        or not isinstance(latency, (int, float))
        or not math.isfinite(float(latency))
        or float(latency) < 0
    ):
        raise ValueError(f"Development arm state changed for {case_id}/{arm_name}")


def _validate_panel(cases: list[BenchmarkCase], mode: str) -> None:
    if mode not in {"dev", "eval"}:
        raise ValueError(f"Unsupported mode: {mode}")
    expected = DEV_CASES if mode == "dev" else EVAL_CASES
    if len(cases) != expected:
        raise ValueError(f"{mode.title()} mode requires all {expected} ClinOCR cases")
    case_ids = [case.id for case in cases]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("ClinOCR case IDs must be unique")


def _validate_regions(regions: object) -> None:
    if not isinstance(regions, list):
        raise ValueError("The paragraph reader did not return a list")
    region_ids = set()
    reading_orders = set()
    for region in regions:
        if not isinstance(region, TextRegion):
            raise ValueError("The paragraph reader returned a non-TextRegion value")
        if not region.id or region.id in region_ids:
            raise ValueError("Paragraph region IDs must be nonempty and unique")
        if region.reading_order in reading_orders:
            raise ValueError("Paragraph reading orders must be unique")
        if not region.text.strip():
            raise ValueError("Paragraph text must be nonblank")
        region_ids.add(region.id)
        reading_orders.add(region.reading_order)
        confidence = region.confidence
        if confidence is not None and (
            not math.isfinite(float(confidence)) or not 0 <= float(confidence) <= 1
        ):
            raise ValueError("Paragraph confidence must be between zero and one")


def _serialize_region(region: TextRegion) -> dict[str, object]:
    box = region.bounding_box
    return {
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
    }


def _deserialize_regions(value: object, case_id: str) -> list[TextRegion]:
    if not isinstance(value, list):
        raise ValueError(f"Development source regions are missing for {case_id}")
    regions = []
    region_keys = {
        "id",
        "kind",
        "text",
        "confidence",
        "bounding_box",
        "reading_order",
        "provider",
    }
    box_keys = {"left", "top", "right", "bottom"}
    for item in value:
        if not isinstance(item, dict) or set(item) != region_keys:
            raise ValueError(f"Development source region is invalid for {case_id}")
        box = item["bounding_box"]
        if not isinstance(box, dict) or set(box) != box_keys:
            raise ValueError(f"Development source box is invalid for {case_id}")
        coordinates = [box[key] for key in ("left", "top", "right", "bottom")]
        if any(
            isinstance(number, bool) or not isinstance(number, int)
            for number in coordinates
        ):
            raise ValueError(f"Development source box is invalid for {case_id}")
        reading_order = item["reading_order"]
        if isinstance(reading_order, bool) or not isinstance(reading_order, int):
            raise ValueError(f"Development reading order is invalid for {case_id}")
        confidence = item["confidence"]
        if confidence is not None and (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(float(confidence))
            or not 0 <= float(confidence) <= 1
        ):
            raise ValueError(f"Development confidence is invalid for {case_id}")
        if any(
            not isinstance(item[key], str) for key in ("id", "kind", "text", "provider")
        ):
            raise ValueError(f"Development source region is invalid for {case_id}")
        regions.append(
            TextRegion(
                id=item["id"],
                kind=item["kind"],
                text=item["text"],
                confidence=float(confidence) if confidence is not None else None,
                bounding_box=BoundingBox(*coordinates),
                reading_order=reading_order,
                provider=item["provider"],
            )
        )
    _validate_regions(regions)
    return regions


def _claim_output(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise ValueError(f"Output path already exists: {path}") from error
    try:
        return os.fstat(descriptor).st_ino
    finally:
        os.close(descriptor)


def _release_empty_claim(path: Path, inode: int) -> None:
    try:
        details = path.stat()
    except FileNotFoundError:
        return
    if details.st_ino == inode and details.st_size == 0:
        path.unlink()


def _failure(code: str, message: str, stage: str) -> dict[str, str]:
    return {"code": code, "message": message, "stage": stage}


if __name__ == "__main__":
    raise SystemExit(main())
