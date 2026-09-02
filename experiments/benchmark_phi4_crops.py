"""Evaluate Phi-4 Multimodal on the frozen C14 handwriting crops."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any

if __package__:
    from experiments import benchmark_phi4 as phi4
else:
    import benchmark_phi4 as phi4  # type: ignore[no-redef]


EXPECTED_CROPS = 44
VARIANTS = ("native", "scaled")
MODEL_REVISION = "93f923e1a7727d1c4f446756212d9d3e8fcc5d81"
SYMBOLIC_NULL_MARKS = frozenset("Øø∅⌀")


@dataclass(frozen=True)
class CropCase:
    field_id: str
    case_id: str
    bbox: tuple[int, int, int, int]
    reference: str
    native: Path
    native_file: str
    scaled: Path
    scaled_file: str


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run_benchmark(
            args.run_root,
            args.output,
            model_name=args.model,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
            dtype=args.dtype,
            attention=args.attention,
            warmup=not args.skip_warmup,
            seed=args.seed,
        )
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    return 0


def build_parser() -> argparse.ArgumentParser:
    repository = Path(__file__).resolve().parents[1]
    default_root = (
        repository
        / "internal-clinical-ocr-benchmark"
        / "challenging-formats-20260902"
        / "runs"
        / "ministral-c14-crops-source-only"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--run-root", type=Path, default=default_root)
    parser.add_argument("--model", default=phi4.MODEL_ID)
    parser.add_argument("--max-new-tokens", type=_positive_int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument(
        "--attention",
        choices=("sdpa", "flash_attention_2"),
        default="sdpa",
    )
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def run_benchmark(
    run_root: Path,
    output: Path,
    *,
    model_name: str = phi4.MODEL_ID,
    max_new_tokens: int = 128,
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    attention: str = "sdpa",
    warmup: bool = True,
    seed: int = 0,
    processor: Any | None = None,
    model: Any | None = None,
    generation_config: Any | None = None,
    torch_module: Any | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    _validate_options(run_root, output, max_new_tokens, dtype)
    cases = load_crop_cases(run_root)

    load_started = clock()
    if processor is None or model is None or torch_module is None:
        processor, model, generation_config, torch_module = phi4.load_model(
            model_name,
            MODEL_REVISION,
            device=device,
            dtype=dtype,
            attention=attention,
            local_files_only=True,
            seed=seed,
        )
    model_load_ms = round((clock() - load_started) * 1000, 3)
    payload = _initial_payload(
        run_root=run_root,
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        device=device,
        dtype=dtype,
        attention=attention,
        seed=seed,
        model=model,
        processor=processor,
        torch_module=torch_module,
        model_load_ms=model_load_ms,
    )
    phi4._write_payload(output, payload)

    if warmup:
        warmup_record = _run_case(
            cases[0],
            "native",
            processor=processor,
            model=model,
            generation_config=generation_config,
            torch_module=torch_module,
            device=device,
            max_new_tokens=max_new_tokens,
            clock=clock,
        )
        payload["operations"]["warmup"] = {
            "attempted": True,
            "status": warmup_record["status"],
            "latency_ms": warmup_record["latency_ms"],
            "failures": warmup_record["failures"],
        }
    else:
        payload["operations"]["warmup"] = {"attempted": False}
    phi4._reset_peak_memory(torch_module, device)

    benchmark_started = clock()
    for variant in VARIANTS:
        rows: list[dict[str, Any]] = []
        payload["runs"][variant] = {"status": "running", "rows": rows}
        phi4._write_payload(output, payload)
        for case in cases:
            rows.append(
                _run_case(
                    case,
                    variant,
                    processor=processor,
                    model=model,
                    generation_config=generation_config,
                    torch_module=torch_module,
                    device=device,
                    max_new_tokens=max_new_tokens,
                    clock=clock,
                )
            )
            phi4._write_payload(output, payload)
        payload["runs"][variant] = {
            "status": "complete",
            "summary": summarize_rows(rows),
            "rows": rows,
        }
        phi4._write_payload(output, payload)

    payload["operations"]["benchmark_wall_ms"] = round(
        (clock() - benchmark_started) * 1000, 3
    )
    payload["operations"]["cuda_memory_mib"] = phi4._cuda_memory(torch_module, device)
    payload["status"] = "complete"
    payload["evidence"] = {
        "paired_variants_complete": all(
            len(payload["runs"][variant]["rows"]) == EXPECTED_CROPS
            for variant in VARIANTS
        ),
        "failures_remain_in_denominators": True,
        "predictions_persisted_locally": True,
    }
    phi4._write_payload(output, payload)
    return payload


def load_crop_cases(run_root: Path) -> list[CropCase]:
    ground_truth_path = run_root / "ground_truth.json"
    try:
        records = json.loads(ground_truth_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"invalid C14 crop annotations: {ground_truth_path}"
        ) from error
    if not isinstance(records, list) or len(records) != EXPECTED_CROPS:
        count = len(records) if isinstance(records, list) else "invalid"
        raise ValueError(f"expected {EXPECTED_CROPS} frozen C14 crops, found {count}")

    cases = []
    seen_ids = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("invalid C14 crop annotation")
        required = {"case_id", "field_id", "bbox", "reference", *VARIANTS}
        if not required <= record.keys():
            raise ValueError("invalid C14 crop annotation")
        field_id = record["field_id"]
        case_id = record["case_id"]
        reference = record["reference"]
        bbox = record["bbox"]
        if (
            not isinstance(field_id, str)
            or not isinstance(case_id, str)
            or not field_id.startswith(f"{case_id}-H")
            or field_id in seen_ids
            or not isinstance(reference, str)
            or not reference.strip()
            or not isinstance(bbox, list)
            or len(bbox) != 4
            or not all(isinstance(value, int) for value in bbox)
        ):
            raise ValueError("invalid C14 crop annotation")
        seen_ids.add(field_id)

        paths = {
            variant: run_root / "crops" / str(record[variant]) for variant in VARIANTS
        }
        if any(not path.is_file() for path in paths.values()):
            raise FileNotFoundError(f"frozen crop was not found for {field_id}")
        cases.append(
            CropCase(
                field_id=field_id,
                case_id=case_id,
                bbox=tuple(bbox),
                reference=reference,
                native=paths["native"],
                native_file=paths["native"].relative_to(run_root).as_posix(),
                scaled=paths["scaled"],
                scaled_file=paths["scaled"].relative_to(run_root).as_posix(),
            )
        )
    return cases


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reference_characters = sum(
        row["metrics"]["hallucinated_text_rate"]["normalized_reference_characters"]
        for row in rows
    )
    insertions = sum(
        row["metrics"]["character_edit_counts"]["insertions"] for row in rows
    )
    failure_codes = Counter(
        failure["code"] for row in rows for failure in row["failures"]
    )
    latencies = sorted(float(row["latency_ms"]) for row in rows)
    strict_exact = sum(row["strict_exact"] for row in rows)
    normalized_exact = sum(row["normalized_exact"] for row in rows)
    critical_fields = sum(row["critical_field"] for row in rows)
    return {
        "fields": len(rows),
        "failed_fields": sum(row["status"] == "failed" for row in rows),
        "failure_codes": dict(sorted(failure_codes.items())),
        "strict_exact": strict_exact,
        "strict_exact_rate": _ratio(strict_exact, len(rows)),
        "normalized_exact": normalized_exact,
        "normalized_exact_rate": _ratio(normalized_exact, len(rows)),
        "character_insertions": insertions,
        "hallucinated_character_rate": _ratio(insertions, reference_characters),
        "critical_fields": critical_fields,
        "critical_substitutions": sum(row["critical_substitution"] for row in rows),
        "critical_misses": sum(row["critical_miss"] for row in rows),
        "latency_ms": {
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "max": round(max(latencies), 3),
        },
    }


def _run_case(
    case: CropCase,
    variant: str,
    *,
    processor: Any,
    model: Any,
    generation_config: Any,
    torch_module: Any,
    device: str,
    max_new_tokens: int,
    clock: Callable[[], float],
) -> dict[str, Any]:
    image_path = getattr(case, variant)
    record = phi4._run_input(
        id=case.field_id,
        image_path=image_path,
        source_file=getattr(case, f"{variant}_file"),
        prompt=phi4.CROP_PROMPT,
        reference=case.reference,
        processor=processor,
        model=model,
        generation_config=generation_config,
        torch_module=torch_module,
        device=device,
        max_new_tokens=max_new_tokens,
        clock=clock,
    )
    if _is_repetitive(record["prediction"]):
        record["status"] = "failed"
        record["failures"].append(
            {
                "code": "phi4_repetition",
                "type": "RepetitionLoop",
                "message": "generation contains a repeated suffix loop",
            }
        )
    critical_field = _is_critical_reference(case.reference)
    record.update(
        {
            "field_id": record.pop("id"),
            "case_id": case.case_id,
            "bbox": list(case.bbox),
            "variant": variant,
            "repetition_detected": any(
                failure["code"] == "phi4_repetition" for failure in record["failures"]
            ),
            "critical_field": critical_field,
            "critical_substitution": bool(
                critical_field
                and record["prediction"].strip()
                and not record["normalized_exact"]
            ),
            "critical_miss": bool(critical_field and not record["prediction"].strip()),
        }
    )
    return record


def _is_critical_reference(reference: str) -> bool:
    return any(
        character.isalnum() and character not in SYMBOLIC_NULL_MARKS
        for character in reference
    )


def _is_repetitive(text: str) -> bool:
    normalized = " ".join(text.split())
    if len(normalized) < 24:
        return False

    words = normalized.split()
    for width in range(1, min(8, len(words) // 3) + 1):
        repeated_words = _suffix_repetitions(words, width)
        if repeated_words >= 3 and repeated_words * width >= max(6, len(words) // 2):
            return True

    compact = normalized.replace(" ", "")
    for width in range(1, min(32, len(compact) // 4) + 1):
        repeated_characters = _suffix_repetitions(compact, width)
        if repeated_characters >= 4 and repeated_characters * width >= max(
            24, len(compact) // 2
        ):
            return True
    return False


def _suffix_repetitions(values: Sequence[Any], width: int) -> int:
    suffix = values[-width:]
    repetitions = 1
    cursor = len(values) - 2 * width
    while cursor >= 0 and values[cursor : cursor + width] == suffix:
        repetitions += 1
        cursor -= width
    return repetitions


def _initial_payload(
    *,
    run_root: Path,
    model_name: str,
    max_new_tokens: int,
    device: str,
    dtype: str,
    attention: str,
    seed: int,
    model: Any,
    processor: Any,
    torch_module: Any,
    model_load_ms: float,
) -> dict[str, Any]:
    config = getattr(model, "config", None)
    return {
        "benchmark": "Phi-4 Multimodal frozen C14 handwriting crop ablation",
        "status": "running",
        "privacy": {
            "execution": "local_only",
            "private_uploads": False,
            "ground_truth_in_prompt": False,
            "model_downloads": False,
        },
        "panel": {
            "run_root": str(run_root.resolve()),
            "ground_truth": str((run_root / "ground_truth.json").resolve()),
            "fields": EXPECTED_CROPS,
            "variants": list(VARIANTS),
            "inferences": EXPECTED_CROPS * len(VARIANTS),
            "evidence_fields": ["case_id", "field_id", "bbox", "source_file"],
        },
        "model": {
            "requested": model_name,
            "requested_revision": MODEL_REVISION,
            "loaded_name_or_path": getattr(config, "_name_or_path", None),
            "loaded_commit_hash": getattr(config, "_commit_hash", None),
            "class": type(model).__name__,
            "processor_class": type(processor).__name__,
        },
        "generation": {
            "prompt": phi4.CROP_PROMPT,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "num_beams": 1,
            "seed": seed,
            "repetition_policy": "repeated suffix loop is a failed field",
            "token_limit_policy": "generation at the cap is a failed field",
        },
        "runtime": {
            "device": device,
            "dtype": dtype,
            "attention": attention,
            "local_files_only": True,
            "torch": getattr(torch_module, "__version__", None),
            "cuda": getattr(getattr(torch_module, "version", None), "cuda", None),
        },
        "operations": {
            "model_load_ms": model_load_ms,
            "warmup": None,
            "benchmark_wall_ms": None,
            "cuda_memory_mib": None,
        },
        "runs": {},
        "evidence": None,
    }


def _validate_options(
    run_root: Path, output: Path, max_new_tokens: int, dtype: str
) -> None:
    if not run_root.is_dir():
        raise FileNotFoundError(f"C14 crop run root was not found: {run_root}")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if max_new_tokens <= 0:
        raise ValueError("max-new-tokens must be positive")
    if dtype not in {"bfloat16", "float16"}:
        raise ValueError(f"unsupported dtype: {dtype}")


def _percentile(values: list[float], fraction: float) -> float:
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return round(values[lower] * (1 - weight) + values[upper] * weight, 3)


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
