"""Measure Nemotron OCR v2 native batching on a frozen ClinOCR slice."""

from __future__ import annotations

import argparse
import json
import random
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Callable, Sequence

from PIL import Image, ImageOps, UnidentifiedImageError

from ocr_pipeline.contracts import TextRegion
from ocr_pipeline.operations import CudaMonitor
from ocr_pipeline.providers import NemotronOCRV2Reader, ReaderError

if __package__:
    from experiments.orientation_benchmark import (
        DEFAULT_RECTIFICATION_PADDING,
        ROTATIONS,
        detect_tesseract_orientation,
        rectify_document,
    )
    from experiments.public_benchmark import (
        NORMALIZATION,
        BenchmarkCase,
        _score,
        _summarize,
        _transcription_text,
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
        _summarize,
        _transcription_text,
        discover_cases,
    )

BATCH_SIZES = (1, 2, 4, 8)
DEFAULT_SEED = 20260901


@dataclass(frozen=True)
class PreparationInput:
    id: str
    image_path: Path
    page_number: int


@dataclass(frozen=True)
class PreparedCase:
    id: str
    image_path: Path | None
    page_number: int
    details: dict[str, object]
    error: ReaderError | None = None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark one resident Nemotron reader across native batch sizes"
    )
    parser.add_argument("root", type=Path, help="Extracted ClinOCR dataset root")
    parser.add_argument("output", type=Path, help="JSON results path")
    parser.add_argument(
        "--clinocr-role",
        choices=("exemplar", "eval"),
        default="exemplar",
        help="Use exemplars while developing and eval only after freezing policy",
    )
    parser.add_argument("--subset", default="rotated")
    parser.add_argument("--language", choices=("multi", "en"), default="en")
    parser.add_argument(
        "--merge-level",
        choices=("word", "sentence", "paragraph"),
        default="paragraph",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--permutations", type=_positive_int, default=4)
    parser.add_argument(
        "--rectify",
        action="store_true",
        help="Apply the current scored page-quadrilateral rectification once",
    )
    parser.add_argument(
        "--rectify-padding",
        type=float,
        default=DEFAULT_RECTIFICATION_PADDING,
    )
    parser.add_argument(
        "--tesseract-osd",
        nargs="?",
        const="tesseract",
        help="Apply accepted Tesseract OSD rotation once, optionally with this binary",
    )
    parser.add_argument("--osd-min-confidence", type=float, default=15.0)
    args = parser.parse_args(argv)

    try:
        _validate_options(args.rectify_padding, args.osd_min_confidence)
        payload = run_benchmark(
            args.root,
            clinocr_role=args.clinocr_role,
            subset=args.subset,
            language=args.language,
            merge_level=args.merge_level,
            seed=args.seed,
            permutations=args.permutations,
            rectify=args.rectify,
            rectify_padding=args.rectify_padding,
            osd_executable=args.tesseract_osd,
            osd_min_confidence=args.osd_min_confidence,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    return 0


def run_benchmark(
    root: Path,
    *,
    clinocr_role: str,
    subset: str,
    language: str = "en",
    merge_level: str = "paragraph",
    seed: int = DEFAULT_SEED,
    permutations: int = 3,
    rectify: bool = False,
    rectify_padding: float = DEFAULT_RECTIFICATION_PADDING,
    osd_executable: str | None = None,
    osd_min_confidence: float = 15.0,
    reader: object | None = None,
    osd_detector: Callable[..., dict[str, object]] = detect_tesseract_orientation,
    timer: Callable[[], float] = time.perf_counter,
    cuda_monitor: object | None = None,
) -> dict[str, object]:
    _validate_options(rectify_padding, osd_min_confidence)
    if permutations < 1:
        raise ValueError("permutations must be at least 1")

    cases = [
        case
        for case in discover_cases("clinocr", root, clinocr_role=clinocr_role)
        if case.subset == subset
    ]
    if not cases:
        raise ValueError(f"No ClinOCR {clinocr_role} cases found for subset {subset}")

    resident_reader = reader or NemotronOCRV2Reader(
        language=language,
        merge_level=merge_level,
    )
    if not callable(getattr(resident_reader, "read_batch", None)):
        raise ValueError("Reader must provide read_batch")
    monitor = cuda_monitor or CudaMonitor()
    schedules = _request_schedules(cases, seed, permutations)
    batch_size_orders = _batch_size_orders(seed, permutations)

    with tempfile.TemporaryDirectory(prefix="nemotron-batch-benchmark-") as directory:
        preparation_inputs = [
            PreparationInput(case.id, case.image_path, index)
            for index, case in enumerate(cases, start=1)
        ]
        prepared = prepare_cases(
            preparation_inputs,
            Path(directory),
            rectify=rectify,
            rectify_padding=rectify_padding,
            osd_executable=osd_executable,
            osd_min_confidence=osd_min_confidence,
            osd_detector=osd_detector,
            timer=timer,
        )
        warmup = _warmup(resident_reader, prepared, monitor, timer)
        runs = _run_sweep(
            resident_reader,
            cases,
            prepared,
            schedules,
            batch_size_orders,
            root,
            monitor,
            timer,
        )

    _add_exact_equality(runs)
    return {
        "experiment": "nemotron_native_batch_sweep",
        "dataset": "clinocr",
        "dataset_root": str(root.resolve()),
        "normalization": NORMALIZATION,
        "selection": {
            "clinocr_role": clinocr_role,
            "subset": subset,
            "case_ids": [case.id for case in cases],
        },
        "reader": {
            "name": str(getattr(resident_reader, "name", "unknown")),
            "language": language,
            "merge_level": merge_level,
            "resident_instances": 1,
        },
        "preparation_policy": {
            "gold_used": False,
            "exif_normalization": "PIL.ImageOps.exif_transpose, matching production",
            "rectify": rectify,
            "rectification": "orientation_benchmark.rectify_document"
            if rectify
            else None,
            "rectify_padding": rectify_padding if rectify else None,
            "tesseract_osd": osd_executable,
            "osd_min_confidence": osd_min_confidence if osd_executable else None,
            "prepared_once_and_reused": True,
        },
        "preparation": [item.details for item in prepared],
        "execution": {
            "seed": seed,
            "permutations": permutations,
            "batch_sizes": list(BATCH_SIZES),
            "batch_size_orders": {
                str(index): order for index, order in enumerate(batch_size_orders)
            },
            "run_order": [run["run_id"] for run in runs],
            "request_orders": {
                str(index): [case.id for case in schedule]
                for index, schedule in enumerate(schedules)
            },
            "cuda_peak_allocated_memory_available": bool(
                getattr(monitor, "available", False)
            ),
        },
        "warmup": warmup,
        "exact_equality": _overall_exact_equality(runs),
        "batch_size_aggregates": _batch_size_aggregates(runs),
        "runs": runs,
    }


def prepare_cases(
    inputs: list[PreparationInput],
    directory: Path,
    *,
    rectify: bool,
    rectify_padding: float,
    osd_executable: str | None,
    osd_min_confidence: float,
    osd_detector: Callable[..., dict[str, object]],
    timer: Callable[[], float] = time.perf_counter,
) -> list[PreparedCase]:
    """Prepare images without accepting references or other gold information."""
    prepared = []
    for item in inputs:
        prepared.append(
            _prepare_case(
                item,
                directory,
                rectify,
                rectify_padding,
                osd_executable,
                osd_min_confidence,
                osd_detector,
                timer,
            )
        )
    return prepared


def _prepare_case(
    item: PreparationInput,
    directory: Path,
    rectify: bool,
    rectify_padding: float,
    osd_executable: str | None,
    osd_min_confidence: float,
    osd_detector: Callable[..., dict[str, object]],
    timer: Callable[[], float],
) -> PreparedCase:
    started = timer()
    details: dict[str, object] = {
        "id": item.id,
        "page_number": item.page_number,
        "source_image": str(item.image_path),
        "status": "failed",
        "error": None,
    }
    try:
        with Image.open(item.image_path) as source:
            source_size = source.size
            exif_orientation = int(source.getexif().get(274, 1))
            image = ImageOps.exif_transpose(source)
            if image.mode not in {
                "1",
                "L",
                "LA",
                "P",
                "RGB",
                "RGBA",
                "I",
                "I;16",
            }:
                image = image.convert("RGB")
        details.update(
            {
                "source_size": list(source_size),
                "exif_orientation": exif_orientation,
                "exif_transposed": exif_orientation in range(2, 9),
                "exif_normalized_size": list(image.size),
            }
        )
        if rectify:
            try:
                image = rectify_document(image, padding_fraction=rectify_padding)
            except ValueError as error:
                raise ReaderError("rectification_failed", str(error)) from error
            details["rectified_size"] = list(image.size)

        if osd_executable:
            osd_path = directory / f"osd-{item.page_number}.png"
            image.save(osd_path, format="PNG")
            try:
                osd = osd_detector(osd_path, executable=osd_executable)
            except ReaderError as error:
                details["osd_error"] = _error_dict(error)
            else:
                try:
                    angle = int(osd["angle"])
                    confidence = float(osd["confidence"])
                except (KeyError, TypeError, ValueError) as error:
                    raise ReaderError(
                        "invalid_osd_output", "OSD returned invalid angle or confidence"
                    ) from error
                if angle not in ROTATIONS:
                    raise ReaderError(
                        "invalid_osd_output", f"Unsupported OSD angle: {angle}"
                    )
                accepted = confidence >= osd_min_confidence
                details["osd"] = {**osd, "accepted": accepted}
                transform = ROTATIONS[angle] if accepted else None
                if transform is not None:
                    image = image.transpose(transform)

        output = directory / f"page-{item.page_number}.png"
        image.save(output, format="PNG")
        details.update(
            {
                "status": "success",
                "prepared_image": output.name,
                "prepared_size": list(image.size),
            }
        )
        result = PreparedCase(item.id, output, item.page_number, details)
    except (OSError, UnidentifiedImageError, ReaderError, ValueError) as error:
        reader_error = (
            error
            if isinstance(error, ReaderError)
            else ReaderError("image_preparation_failed", str(error))
        )
        details["error"] = _error_dict(reader_error)
        result = PreparedCase(
            item.id,
            None,
            item.page_number,
            details,
            reader_error,
        )
    details["latency_ms"] = round((timer() - started) * 1000, 3)
    return result


def _request_schedules(
    cases: list[BenchmarkCase], seed: int, permutations: int
) -> list[list[BenchmarkCase]]:
    schedules = []
    for permutation in range(permutations):
        schedule = list(cases)
        random.Random(seed + permutation).shuffle(schedule)
        schedules.append(schedule)
    return schedules


def _batch_size_orders(seed: int, repetitions: int) -> list[list[int]]:
    base_order = list(BATCH_SIZES)
    random.Random(seed).shuffle(base_order)
    orders = []
    for repetition in range(repetitions):
        offset = repetition % len(base_order)
        orders.append(base_order[offset:] + base_order[:offset])
    return orders


def _warmup(
    reader: object,
    prepared: list[PreparedCase],
    monitor: object,
    timer: Callable[[], float],
) -> dict[str, object]:
    candidate = next((case for case in prepared if case.image_path is not None), None)
    if candidate is None:
        return {
            "recorded_separately": True,
            "status": "skipped",
            "reason": "No image was prepared successfully",
        }
    setattr(reader, "batch_size", 1)
    batch = _timed_batch(reader, [candidate], monitor, timer)
    outcome = batch.pop("outcomes")[0]
    return {
        "recorded_separately": True,
        "excluded_from_scored_runs": True,
        "status": "failed" if isinstance(outcome, ReaderError) else "success",
        "case_id": candidate.id,
        **batch,
        "returned_error": (
            _error_dict(outcome) if isinstance(outcome, ReaderError) else None
        ),
    }


def _run_sweep(
    reader: object,
    cases: list[BenchmarkCase],
    prepared: list[PreparedCase],
    schedules: list[list[BenchmarkCase]],
    batch_size_orders: list[list[int]],
    root: Path,
    monitor: object,
    timer: Callable[[], float],
) -> list[dict[str, object]]:
    prepared_by_id = {item.id: item for item in prepared}
    original_ids = [case.id for case in cases]
    references = {case.id: case for case in cases}
    runs = []
    run_index = 0
    for permutation, (schedule, batch_size_order) in enumerate(
        zip(schedules, batch_size_orders, strict=True)
    ):
        for treatment_position, batch_size in enumerate(batch_size_order):
            setattr(reader, "batch_size", batch_size)
            outcomes: dict[str, list[TextRegion] | ReaderError] = {
                item.id: item.error for item in prepared if item.error is not None
            }
            batches = []
            runnable = [prepared_by_id[case.id] for case in schedule]
            runnable = [item for item in runnable if item.image_path is not None]
            for batch_index, batch_cases in enumerate(_chunks(runnable, batch_size)):
                timed = _timed_batch(reader, batch_cases, monitor, timer)
                batch_outcomes = timed.pop("outcomes")
                for prepared_case, outcome in zip(
                    batch_cases, batch_outcomes, strict=True
                ):
                    outcomes[prepared_case.id] = outcome
                batches.append(
                    {
                        "batch_index": batch_index,
                        "case_ids": [item.id for item in batch_cases],
                        **timed,
                    }
                )

            records = []
            latency_by_id = {
                case_id: float(batch["latency_ms"])
                for batch in batches
                for case_id in batch["case_ids"]
            }
            for case_id in original_ids:
                records.append(
                    _case_record(
                        references[case_id],
                        prepared_by_id[case_id],
                        outcomes[case_id],
                        root,
                        latency_by_id.get(case_id, 0.0),
                    )
                )
            total_latency_ms = sum(float(batch["latency_ms"]) for batch in batches)
            summary = _summarize(records)
            summary["latency_ms"] = _latency_summary(list(latency_by_id.values()))
            summary["batch_latency_ms"] = _latency_summary(
                [float(batch["latency_ms"]) for batch in batches]
            )
            summary["inference_latency_ms"] = round(total_latency_ms, 3)
            summary["timed_pages"] = len(runnable)
            summary["preparation_failed_pages"] = len(cases) - len(runnable)
            summary["pages_per_second"] = (
                round(len(runnable) / (total_latency_ms / 1000), 6)
                if total_latency_ms
                else 0.0
            )
            summary["cuda_peak_allocated_bytes"] = max(
                (
                    int(batch["cuda_peak_allocated_bytes"])
                    for batch in batches
                    if batch["cuda_peak_allocated_bytes"] is not None
                ),
                default=None,
            )
            runs.append(
                {
                    "run_id": f"run-{run_index:02d}-p{permutation}-b{batch_size}",
                    "run_index": run_index,
                    "permutation": permutation,
                    "batch_size": batch_size,
                    "treatment_position": treatment_position,
                    "request_case_ids": [case.id for case in schedule],
                    "returned_case_ids": original_ids,
                    "summary": summary,
                    "batches": batches,
                    "cases": records,
                }
            )
            run_index += 1
    return runs


def _timed_batch(
    reader: object,
    batch: list[PreparedCase],
    monitor: object,
    timer: Callable[[], float],
) -> dict[str, object]:
    paths = [item.image_path for item in batch]
    if any(path is None for path in paths):
        raise ValueError("Timed batches require prepared image paths")
    getattr(monitor, "begin")()
    started = timer()
    try:
        outcomes = reader.read_batch(  # type: ignore[attr-defined]
            paths,
            [item.page_number for item in batch],
        )
    except ReaderError as error:
        outcomes = [ReaderError(error.code, str(error)) for _ in batch]
    except Exception as error:
        outcomes = [ReaderError("batch_inference_failed", str(error)) for _ in batch]
    peak_memory = getattr(monitor, "finish")()
    latency_ms = (timer() - started) * 1000
    if not isinstance(outcomes, list) or len(outcomes) != len(batch):
        error = ReaderError(
            "invalid_batch_output",
            f"Batch reader returned {len(outcomes) if isinstance(outcomes, list) else 'non-list'} results for {len(batch)} pages",
        )
        outcomes = [ReaderError(error.code, str(error)) for _ in batch]
    else:
        outcomes = [_validated_outcome(outcome) for outcome in outcomes]
    return {
        "pages": len(batch),
        "latency_ms": round(latency_ms, 3),
        "pages_per_second": round(len(batch) / (latency_ms / 1000), 6)
        if latency_ms
        else 0.0,
        "cuda_peak_allocated_bytes": peak_memory,
        "outcomes": outcomes,
    }


def _case_record(
    case: BenchmarkCase,
    prepared: PreparedCase,
    outcome: list[TextRegion] | ReaderError,
    root: Path,
    latency_ms: float,
) -> dict[str, object]:
    error = outcome if isinstance(outcome, ReaderError) else None
    regions = [] if error else outcome
    if not error and not regions:
        error = ReaderError("no_text_detected", "The reader returned no text regions")
    canonical_regions = [asdict(region) for region in regions]
    structured_prediction = " ".join(
        region.text for region in sorted(regions, key=lambda item: item.reading_order)
    )
    prediction = _transcription_text(structured_prediction)
    returned_error = _error_dict(error) if error else None
    return {
        "id": case.id,
        "cluster_id": case.cluster_id,
        "subset": case.subset,
        "image": str(case.image_path.relative_to(root)),
        "page_number": prepared.page_number,
        "prediction": prediction,
        "structured_prediction": structured_prediction,
        "reference": case.reference,
        "status": "failed" if error else "success",
        "metrics": _score(prediction, case.reference),
        "failures": [returned_error] if returned_error else [],
        "returned_error": returned_error,
        "canonical_regions": canonical_regions,
        "latency_ms": round(latency_ms, 3),
    }


def _add_exact_equality(runs: list[dict[str, object]]) -> None:
    baseline = runs[0]
    baseline_cases = {
        case["id"]: _outcome_signature(case) for case in baseline["cases"]
    }
    for run in runs:
        matching = sum(
            _outcome_signature(case) == baseline_cases[case["id"]]
            for case in run["cases"]
        )
        run["exact_equality"] = {
            "baseline_run_id": baseline["run_id"],
            "matching_cases": matching,
            "cases": len(baseline_cases),
            "rate": round(matching / len(baseline_cases), 6),
        }


def _overall_exact_equality(runs: list[dict[str, object]]) -> dict[str, object]:
    baseline_cases = {case["id"]: _outcome_signature(case) for case in runs[0]["cases"]}
    matching = sum(
        all(
            _outcome_signature(
                next(case for case in run["cases"] if case["id"] == case_id)
            )
            == signature
            for run in runs[1:]
        )
        for case_id, signature in baseline_cases.items()
    )
    return {
        "baseline_run_id": runs[0]["run_id"],
        "runs": len(runs),
        "matching_cases_across_all_runs": matching,
        "cases": len(baseline_cases),
        "rate": round(matching / len(baseline_cases), 6),
    }


def _batch_size_aggregates(
    runs: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    aggregates = {}
    for batch_size in BATCH_SIZES:
        batch_runs = [run for run in runs if run["batch_size"] == batch_size]
        baseline_cases = {
            case["id"]: _outcome_signature(case) for case in batch_runs[0]["cases"]
        }
        matching = sum(
            _outcome_signature(case) == baseline_cases[case["id"]]
            for run in batch_runs
            for case in run["cases"]
        )
        case_runs = len(baseline_cases) * len(batch_runs)
        cuda_peaks = [
            int(run["summary"]["cuda_peak_allocated_bytes"])
            for run in batch_runs
            if run["summary"]["cuda_peak_allocated_bytes"] is not None
        ]
        aggregates[str(batch_size)] = {
            "runs": len(batch_runs),
            "median_pages_per_second": round(
                median(float(run["summary"]["pages_per_second"]) for run in batch_runs),
                6,
            ),
            "median_full_page_latency_ms": {
                "p50": round(
                    median(
                        float(run["summary"]["latency_ms"]["p50"]) for run in batch_runs
                    ),
                    3,
                ),
                "p95": round(
                    median(
                        float(run["summary"]["latency_ms"]["p95"]) for run in batch_runs
                    ),
                    3,
                ),
            },
            "median_cuda_peak_allocated_bytes": (
                median(cuda_peaks) if cuda_peaks else None
            ),
            "exact_output_equality": {
                "baseline_run_id": batch_runs[0]["run_id"],
                "matching_case_runs": matching,
                "case_runs": case_runs,
                "rate": round(matching / case_runs, 6),
            },
        }
    return aggregates


def _outcome_signature(case: dict[str, object]) -> tuple[object, object]:
    return case["canonical_regions"], case["returned_error"]


def _chunks(values: list[PreparedCase], size: int) -> list[list[PreparedCase]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def _latency_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0}
    ordered = sorted(values)
    return {
        "mean": round(sum(values) / len(values), 3),
        "p50": round(_percentile(ordered, 0.5), 3),
        "p95": round(_percentile(ordered, 0.95), 3),
    }


def _percentile(ordered: list[float], quantile: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _error_dict(error: ReaderError) -> dict[str, str]:
    return {"code": error.code, "message": str(error)}


def _validated_outcome(outcome: object) -> list[TextRegion] | ReaderError:
    if isinstance(outcome, ReaderError):
        return outcome
    if isinstance(outcome, list) and all(
        isinstance(region, TextRegion) for region in outcome
    ):
        return outcome
    return ReaderError(
        "invalid_batch_output",
        "Batch reader returned a page result that was not TextRegion list or ReaderError",
    )


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def _validate_options(rectify_padding: float, osd_min_confidence: float) -> None:
    if not 0 <= rectify_padding <= 0.25:
        raise ValueError("--rectify-padding must be between 0 and 0.25")
    if osd_min_confidence < 0:
        raise ValueError("--osd-min-confidence cannot be negative")


if __name__ == "__main__":
    raise SystemExit(main())
