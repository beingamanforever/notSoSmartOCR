"""Benchmark Kraken PP-OCRv6-medium on fixed public handwriting lines."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any
import unicodedata

from ocr_pipeline.verification import edit_counts


MODEL_ID = "small-models-for-glam/kraken-ppocrv6-medium"
MODEL_CARD = "https://huggingface.co/small-models-for-glam/kraken-ppocrv6-medium"
PROMOTION_EXACT = 0.80
PROMOTION_CER = 0.10


@dataclass(frozen=True)
class LineCase:
    page: str
    line_id: str
    bbox: tuple[int, int, int, int]
    reference: str


# Manual reference boxes isolate recognition from page-line localization. The
# boxes were checked against the original 3072 x 4080 user-supplied images.
PUBLIC_LINES = (
    LineCase(
        "example_6.png",
        "e6-title",
        (760, 1100, 2550, 1220),
        "p-forms, exterior product, interior product",
    ),
    LineCase(
        "example_6.png",
        "e6-section",
        (750, 1220, 2850, 1325),
        "Covariant antisymmetric tensors, exterior and interior products",
    ),
    LineCase(
        "example_6.png",
        "e6-definition",
        (760, 1320, 3020, 1430),
        "Rank-p covariant antisymmetric tensors are called p-forms and are written",
    ),
    LineCase(
        "example_6.png",
        "e6-exterior-intro",
        (350, 1710, 3000, 1830),
        "Let α, β be a p-form, a q-form, respectively. We may define the exterior",
    ),
    LineCase(
        "example_6.png",
        "e6-exterior-continuation",
        (350, 1820, 1900, 1940),
        "product of α and β the following way:",
    ),
    LineCase(
        "example_7.png",
        "e7-shell",
        (520, 965, 2120, 1075),
        "Again, a spherical shell of light is a v = v₀.",
    ),
    LineCase(
        "example_7.png",
        "e7-geometry",
        (450, 1070, 2750, 1180),
        "The full geometry of a spherical shell of light is",
    ),
    LineCase(
        "example_7.png",
        "e7-merging",
        (400, 1175, 2050, 1290),
        "obtained by merging these two metrics.",
    ),
    LineCase(
        "example_7.png",
        "e7-time",
        (250, 1850, 1980, 1965),
        "This solution is time-dependent thus v contains t. At some",
    ),
    LineCase(
        "example_7.png",
        "e7-horizon",
        (250, 1960, 1950, 2075),
        "point, the shell of light will cross the horizon and become",
    ),
    LineCase(
        "example_8.png",
        "e8-title",
        (100, 980, 2000, 1110),
        "Scattering and cross section",
    ),
    LineCase(
        "example_8.png",
        "e8-definitions",
        (0, 1090, 1300, 1220),
        "1.1 Definitions",
    ),
    LineCase(
        "example_8.png",
        "e8-definition-1",
        (150, 1200, 3020, 1320),
        "Let's consider the number of particles per unit time N dΩ, scattered into an",
    ),
    LineCase(
        "example_8.png",
        "e8-definition-2",
        (150, 1310, 3020, 1430),
        "element of solid angle dΩ in the direction (θ, φ): it is proportional to the",
    ),
    LineCase(
        "example_8.png",
        "e8-definition-3",
        (150, 1420, 3020, 1540),
        "incident flux of particles j_I, number of particles per unit time crossing",
    ),
    LineCase(
        "example_8.png",
        "e8-definition-4",
        (150, 1530, 3020, 1650),
        "a unit area normal to direction of incidence. Collisions are characterized by",
    ),
)


@dataclass
class KrakenRuntime:
    model: Any
    config: Any
    box_line: Callable[..., Any]
    segmentation: Callable[..., Any]
    open_image: Callable[[Path], Any]
    torch: Any | None = None


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run_benchmark(
            args.image_dir,
            args.model,
            args.output,
            device=args.device,
            batch_size=args.batch_size,
            precision=args.precision,
            num_line_workers=args.num_line_workers,
            warmup=not args.skip_warmup,
        )
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        write_json(
            args.output,
            {
                "status": "rejected_incomplete",
                "failure": f"{type(error).__name__}: {error}",
            },
        )
        parser.error(str(error))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=positive_int, default=16)
    parser.add_argument("--precision", default="32-true")
    parser.add_argument("--num-line-workers", type=nonnegative_int, default=2)
    parser.add_argument("--skip-warmup", action="store_true")
    return parser


def run_benchmark(
    image_dir: Path,
    model_path: Path,
    output: Path,
    *,
    device: str = "cuda:0",
    batch_size: int = 16,
    precision: str = "32-true",
    num_line_workers: int = 2,
    warmup: bool = True,
    runtime: KrakenRuntime | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    pages = load_pages(image_dir)
    load_started = clock()
    runtime = runtime or load_runtime(
        model_path,
        device=device,
        batch_size=batch_size,
        precision=precision,
        num_line_workers=num_line_workers,
    )
    model_load_ms = round((clock() - load_started) * 1000, 3)

    if warmup:
        first_page = PUBLIC_LINES[0].page
        recognize_page(runtime, pages[first_page], [PUBLIC_LINES[0]])
    reset_peak_memory(runtime.torch)

    rows: list[dict[str, Any]] = []
    page_latencies: list[float] = []
    benchmark_started = clock()
    for page_name in sorted(pages):
        cases = [case for case in PUBLIC_LINES if case.page == page_name]
        page_started = clock()
        records = recognize_page(runtime, pages[page_name], cases)
        page_ms = round((clock() - page_started) * 1000, 3)
        page_latencies.append(page_ms)
        if len(records) != len(cases):
            raise RuntimeError(
                f"{page_name}: expected {len(cases)} records, received {len(records)}"
            )
        rows.extend(
            row_from_record(case, record, page_ms / len(cases))
            for case, record in zip(cases, records, strict=True)
        )

    benchmark_ms = round((clock() - benchmark_started) * 1000, 3)
    metrics = score_rows(rows)
    threshold_pass = (
        metrics["strict_exact_rate"] >= PROMOTION_EXACT
        and metrics["cer"] <= PROMOTION_CER
        and metrics["empty_output_count"] == 0
    )
    decision = "review_only" if threshold_pass else "reject"
    reason = (
        "The small public notebook slice clears the numeric screen, but it is "
        "not clinical promotion evidence."
        if threshold_pass
        else "The recognizer misses the fixed literal accuracy promotion screen."
    )
    payload = {
        "status": "complete",
        "model": {
            "id": MODEL_ID,
            "card": MODEL_CARD,
            "checkpoint_file": model_path.name,
            "api": "kraken.tasks.RecognitionTaskModel",
        },
        "localization": {
            "source": "manual_public_reference_boxes",
            "coordinate_space": "source_pixels",
            "page_size": [3072, 4080],
            "purpose": "recognition-only evaluation",
        },
        "configuration": {
            "device": device,
            "batch_size": batch_size,
            "precision": precision,
            "num_line_workers": num_line_workers,
            "warmup": warmup,
        },
        "latency": {
            "model_load_ms": model_load_ms,
            "benchmark_wall_ms": benchmark_ms,
            "page_ms": page_latencies,
            "page_p50_ms": percentile(page_latencies, 0.50),
            "page_p95_ms": percentile(page_latencies, 0.95),
            "amortized_line_p50_ms": percentile(
                [row["amortized_line_ms"] for row in rows], 0.50
            ),
            "amortized_line_p95_ms": percentile(
                [row["amortized_line_ms"] for row in rows], 0.95
            ),
            "lines_per_second": round(len(rows) / (benchmark_ms / 1000), 3),
            **peak_memory(runtime.torch),
        },
        "metrics": metrics,
        "confidence_calibration": confidence_calibration(rows),
        "decision": {
            "verdict": decision,
            "reason": reason,
            "production_wiring_allowed": False,
            "numeric_screen": {
                "strict_exact_rate_at_least": PROMOTION_EXACT,
                "cer_at_most": PROMOTION_CER,
                "no_empty_outputs": True,
                "passed": threshold_pass,
            },
        },
        "rows": rows,
    }
    write_json(output, payload)
    return payload


def load_runtime(
    model_path: Path,
    *,
    device: str,
    batch_size: int,
    precision: str,
    num_line_workers: int,
) -> KrakenRuntime:
    from PIL import Image
    import torch
    from kraken.configs import RecognitionInferenceConfig
    from kraken.containers import BBoxLine, Segmentation
    from kraken.ketos.util import to_ptl_device
    from kraken.tasks import RecognitionTaskModel

    if not model_path.is_file():
        raise ValueError(f"missing Kraken checkpoint: {model_path}")
    accelerator, devices = to_ptl_device(device)
    config = RecognitionInferenceConfig(
        accelerator=accelerator,
        device=devices,
        batch_size=batch_size,
        precision=precision,
        num_line_workers=num_line_workers,
        bidi_reordering=False,
    )
    return KrakenRuntime(
        model=RecognitionTaskModel.load_model(model_path),
        config=config,
        box_line=BBoxLine,
        segmentation=Segmentation,
        open_image=Image.open,
        torch=torch,
    )


def load_pages(image_dir: Path) -> dict[str, Path]:
    pages = {name: image_dir / name for name in sorted({c.page for c in PUBLIC_LINES})}
    missing = [str(path) for path in pages.values() if not path.is_file()]
    if missing:
        raise ValueError(f"missing public handwriting pages: {', '.join(missing)}")
    return pages


def recognize_page(
    runtime: KrakenRuntime, page_path: Path, cases: Sequence[LineCase]
) -> list[Any]:
    with runtime.open_image(page_path) as source:
        image = source.convert("RGB")
        width, height = image.size
        for case in cases:
            x0, y0, x1, y1 = case.bbox
            if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
                raise ValueError(f"{case.line_id}: bbox is outside {width} x {height}")
        lines = [
            runtime.box_line(
                id=case.line_id,
                bbox=case.bbox,
                text_direction="horizontal-lr",
            )
            for case in cases
        ]
        segmentation = runtime.segmentation(
            type="bbox",
            imagename=str(page_path),
            text_direction="horizontal-lr",
            script_detection=False,
            lines=lines,
        )
        if str(getattr(runtime.model, "seg_type", "bbox")).startswith("baseline"):
            segmentation = segmentation.to_baselines()
        return list(runtime.model.predict(image, segmentation, runtime.config))


def row_from_record(
    case: LineCase, record: Any, amortized_line_ms: float
) -> dict[str, Any]:
    prediction = str(record.prediction)
    confidences = [float(value) for value in record.confidences]
    normalized_prediction = normalize(prediction)
    normalized_reference = normalize(case.reference)
    char_counts = edit_counts(normalized_prediction, normalized_reference)
    word_counts = edit_counts(
        normalized_prediction.split(), normalized_reference.split()
    )
    case_values = asdict(case)
    case_values["bbox"] = list(case.bbox)
    return {
        **case_values,
        "prediction": prediction,
        "normalized_prediction": normalized_prediction,
        "strict_exact": normalized_prediction == normalized_reference,
        "empty_output": not normalized_prediction,
        "character_edits": asdict(char_counts),
        "word_edits": asdict(word_counts),
        "character_confidences": confidences,
        "mean_character_confidence": (
            round(statistics.fmean(confidences), 6) if confidences else None
        ),
        "confidence_length_matches_prediction": len(confidences) == len(prediction),
        "amortized_line_ms": round(amortized_line_ms, 3),
    }


def score_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    char_edits = sum(sum(row["character_edits"].values()) for row in rows)
    word_edits = sum(sum(row["word_edits"].values()) for row in rows)
    reference_chars = sum(len(normalize(row["reference"])) for row in rows)
    reference_words = sum(len(normalize(row["reference"]).split()) for row in rows)
    exact_count = sum(row["strict_exact"] for row in rows)
    return {
        "line_count": len(rows),
        "strict_exact_count": exact_count,
        "strict_exact_rate": round(exact_count / len(rows), 6) if rows else 0.0,
        "cer": round(char_edits / reference_chars, 6) if reference_chars else 0.0,
        "wer": round(word_edits / reference_words, 6) if reference_words else 0.0,
        "character_edits": char_edits,
        "reference_characters": reference_chars,
        "word_edits": word_edits,
        "reference_words": reference_words,
        "empty_output_count": sum(row["empty_output"] for row in rows),
        "confidence_length_mismatch_count": sum(
            not row["confidence_length_matches_prediction"] for row in rows
        ),
    }


def confidence_calibration(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    pairs: list[tuple[float, bool]] = []
    deletion_count = 0
    mismatched_rows: list[str] = []
    for row in rows:
        prediction = row["prediction"]
        confidences = row["character_confidences"]
        if len(prediction) != len(confidences):
            mismatched_rows.append(row["line_id"])
            continue
        correctness, deletions = prediction_correctness(prediction, row["reference"])
        deletion_count += deletions
        pairs.extend(zip(confidences, correctness, strict=True))

    bins = []
    expected_calibration_error = 0.0
    for index in range(10):
        lower = index / 10
        upper = (index + 1) / 10
        values = [
            pair
            for pair in pairs
            if lower <= pair[0] < upper or (index == 9 and pair[0] == 1.0)
        ]
        if not values:
            continue
        mean_confidence = statistics.fmean(value[0] for value in values)
        accuracy = statistics.fmean(value[1] for value in values)
        expected_calibration_error += (
            len(values) / len(pairs) * abs(mean_confidence - accuracy)
        )
        bins.append(
            {
                "lower": lower,
                "upper": upper,
                "count": len(values),
                "mean_confidence": round(mean_confidence, 6),
                "accuracy": round(accuracy, 6),
            }
        )

    return {
        "scope": "emitted_codepoints_only",
        "aligned_codepoint_count": len(pairs),
        "reference_deletions_excluded": deletion_count,
        "rows_excluded_for_length_mismatch": mismatched_rows,
        "mean_confidence": (
            round(statistics.fmean(value[0] for value in pairs), 6) if pairs else None
        ),
        "emitted_codepoint_accuracy": (
            round(statistics.fmean(value[1] for value in pairs), 6) if pairs else None
        ),
        "brier_score": (
            round(statistics.fmean((value[0] - value[1]) ** 2 for value in pairs), 6)
            if pairs
            else None
        ),
        "ece_10_bin": round(expected_calibration_error, 6) if pairs else None,
        "bins": bins,
        "selective_line_exact": selective_exact(rows),
    }


def prediction_correctness(prediction: str, reference: str) -> tuple[list[bool], int]:
    width = len(prediction)
    height = len(reference)
    costs = [[0] * (width + 1) for _ in range(height + 1)]
    steps: list[list[str | None]] = [[None] * (width + 1) for _ in range(height + 1)]
    for column in range(1, width + 1):
        costs[0][column] = column
        steps[0][column] = "insert"
    for row in range(1, height + 1):
        costs[row][0] = row
        steps[row][0] = "delete"
    for row in range(1, height + 1):
        for column in range(1, width + 1):
            if prediction[column - 1] == reference[row - 1]:
                costs[row][column] = costs[row - 1][column - 1]
                steps[row][column] = "match"
                continue
            choices = (
                (costs[row - 1][column - 1] + 1, "substitute"),
                (costs[row - 1][column] + 1, "delete"),
                (costs[row][column - 1] + 1, "insert"),
            )
            costs[row][column], steps[row][column] = min(
                choices, key=lambda choice: choice[0]
            )

    correctness = [False] * width
    deletions = 0
    row, column = height, width
    while row or column:
        step = steps[row][column]
        if step == "match":
            correctness[column - 1] = True
            row -= 1
            column -= 1
        elif step == "substitute":
            row -= 1
            column -= 1
        elif step == "delete":
            deletions += 1
            row -= 1
        elif step == "insert":
            column -= 1
        else:
            raise RuntimeError("invalid alignment state")
    return correctness, deletions


def selective_exact(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(
        (row for row in rows if row["mean_character_confidence"] is not None),
        key=lambda row: row["mean_character_confidence"],
        reverse=True,
    )
    output = []
    for coverage in (0.25, 0.50, 0.75, 1.0):
        count = math.ceil(len(ranked) * coverage)
        selected = ranked[:count]
        output.append(
            {
                "coverage": coverage,
                "line_count": count,
                "strict_exact_rate": (
                    round(sum(row["strict_exact"] for row in selected) / count, 6)
                    if count
                    else None
                ),
            }
        )
    return output


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def reset_peak_memory(torch_module: Any | None) -> None:
    if torch_module is not None and torch_module.cuda.is_available():
        torch_module.cuda.reset_peak_memory_stats()


def peak_memory(torch_module: Any | None) -> dict[str, float | None]:
    if torch_module is None or not torch_module.cuda.is_available():
        return {"peak_allocated_mib": None, "peak_reserved_mib": None}
    scale = 1024 * 1024
    return {
        "peak_allocated_mib": round(
            torch_module.cuda.max_memory_allocated() / scale, 3
        ),
        "peak_reserved_mib": round(torch_module.cuda.max_memory_reserved() / scale, 3),
    }


def percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * quantile) - 1)
    return round(ordered[index], 3)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be at least zero")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
