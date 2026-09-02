"""Run local Phi-4 OCR on unmatched handwriting review crops."""

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


MODEL_REVISION = phi4.MODEL_REVISION
REVIEW_REASON = "no_conservative_match"
RESULT_FILE = "phi4-review-queue.json"


@dataclass(frozen=True)
class ReviewCase:
    field_id: str
    case_id: str
    family_id: str
    category_id: str
    crop_path: Path
    crop_file: str


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run_benchmark(
            args.review_queue,
            args.output_dir,
            model_name=args.model,
            revision=args.revision,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
            dtype=args.dtype,
            attention=args.attention,
            seed=args.seed,
            limit=args.limit,
        )
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review_queue", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--model", default=phi4.MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--max-new-tokens", type=_positive_int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument(
        "--attention",
        choices=("sdpa", "flash_attention_2"),
        default="sdpa",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=_positive_int)
    return parser


def run_benchmark(
    review_queue: Path,
    output_dir: Path,
    *,
    model_name: str = phi4.MODEL_ID,
    revision: str = MODEL_REVISION,
    max_new_tokens: int = 128,
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    attention: str = "sdpa",
    seed: int = 0,
    limit: int | None = None,
    processor: Any | None = None,
    model: Any | None = None,
    generation_config: Any | None = None,
    torch_module: Any | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    output = output_dir / RESULT_FILE
    _validate_options(
        review_queue,
        output_dir,
        output,
        revision,
        max_new_tokens,
        dtype,
        limit,
    )
    available = load_review_cases(review_queue)
    cases = available[:limit] if limit is not None else available

    load_started = clock()
    if processor is None or model is None or torch_module is None:
        processor, model, generation_config, torch_module = phi4.load_model(
            model_name,
            revision,
            device=device,
            dtype=dtype,
            attention=attention,
            local_files_only=True,
            seed=seed,
        )
    model_load_ms = round((clock() - load_started) * 1000, 3)

    payload = _initial_payload(
        review_queue=review_queue,
        model_name=model_name,
        revision=revision,
        max_new_tokens=max_new_tokens,
        device=device,
        dtype=dtype,
        attention=attention,
        seed=seed,
        limit=limit,
        available=len(available),
        selected=len(cases),
        processor=processor,
        model=model,
        torch_module=torch_module,
        model_load_ms=model_load_ms,
    )
    phi4._write_payload(output, payload)
    phi4._reset_peak_memory(torch_module, device)

    benchmark_started = clock()
    rows = payload["rows"]
    for case in cases:
        raw = phi4._run_input(
            id=case.field_id,
            image_path=case.crop_path,
            source_file=case.crop_file,
            prompt=phi4.CROP_PROMPT,
            reference=None,
            processor=processor,
            model=model,
            generation_config=generation_config,
            torch_module=torch_module,
            device=device,
            max_new_tokens=max_new_tokens,
            clock=clock,
        )
        rows.append(_result_row(case, raw))
        phi4._write_payload(output, payload)

    payload["summary"] = summarize_rows(rows)
    payload["operations"]["benchmark_wall_ms"] = round(
        (clock() - benchmark_started) * 1000, 3
    )
    payload["operations"]["cuda_memory_mib"] = phi4._cuda_memory(torch_module, device)
    payload["status"] = "complete"
    phi4._write_payload(output, payload)
    return payload


def load_review_cases(review_queue: Path) -> list[ReviewCase]:
    root = review_queue.parent.resolve()
    cases = []
    seen_ids = set()
    try:
        lines = review_queue.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ValueError(f"invalid review queue: {review_queue}") from error

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid review queue JSON on line {line_number}"
            ) from error
        if not isinstance(record, dict):
            raise ValueError(f"invalid review queue row on line {line_number}")
        if record.get("review_reason") != REVIEW_REASON:
            continue
        case = _load_case(record, root, line_number)
        if case.field_id in seen_ids:
            raise ValueError(f"duplicate field_id: {case.field_id}")
        seen_ids.add(case.field_id)
        cases.append(case)

    if not cases:
        raise ValueError(f"review queue has no {REVIEW_REASON} rows")
    return sorted(cases, key=lambda case: case.field_id)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = sorted(float(row["elapsed_ms"]) for row in rows)
    failures = Counter(row["error_type"] for row in rows if row["error_type"])
    failed = sum(row["status"] != "success" for row in rows)
    return {
        "attempted": len(rows),
        "succeeded": len(rows) - failed,
        "failed": failed,
        "failure_rate": round(failed / len(rows), 6) if rows else None,
        "failure_types": dict(sorted(failures.items())),
        "latency_ms": {
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "max": round(max(latencies), 3) if latencies else None,
        },
        "failures_remain_in_denominator": True,
    }


def _load_case(record: dict[str, Any], root: Path, line_number: int) -> ReviewCase:
    required = ("field_id", "case_id", "family_id", "category_id", "review_crop_path")
    values = {name: record.get(name) for name in required}
    if any(
        not isinstance(value, str) or not value.strip() for value in values.values()
    ):
        raise ValueError(f"invalid review queue identifiers on line {line_number}")

    crop_file = values["review_crop_path"]
    relative = Path(crop_file)
    if relative.is_absolute():
        raise ValueError(f"review crop path must be relative on line {line_number}")
    crop_path = (root / relative).resolve()
    if not crop_path.is_relative_to(root) or not crop_path.is_file():
        raise FileNotFoundError(f"review crop was not found on line {line_number}")
    return ReviewCase(
        field_id=values["field_id"],
        case_id=values["case_id"],
        family_id=values["family_id"],
        category_id=values["category_id"],
        crop_path=crop_path,
        crop_file=relative.as_posix(),
    )


def _result_row(case: ReviewCase, raw: dict[str, Any]) -> dict[str, Any]:
    failures = raw.get("failures") or []
    first_failure = failures[0] if failures else {}
    return {
        "field_id": case.field_id,
        "case_id": case.case_id,
        "family_id": case.family_id,
        "category_id": case.category_id,
        "review_crop_path": case.crop_file,
        "prediction": str(raw.get("prediction", "")),
        "elapsed_ms": float(raw.get("latency_ms", 0.0)),
        "status": str(raw.get("status", "failed")),
        "error_type": first_failure.get("type"),
        "error": first_failure.get("message"),
        "generated_tokens": int(raw.get("generated_tokens", 0)),
        "hit_token_limit": bool(raw.get("hit_token_limit", False)),
    }


def _initial_payload(
    *,
    review_queue: Path,
    model_name: str,
    revision: str,
    max_new_tokens: int,
    device: str,
    dtype: str,
    attention: str,
    seed: int,
    limit: int | None,
    available: int,
    selected: int,
    processor: Any,
    model: Any,
    torch_module: Any,
    model_load_ms: float,
) -> dict[str, Any]:
    config = getattr(model, "config", None)
    return {
        "benchmark": "Phi-4 unmatched handwriting review queue",
        "status": "running",
        "privacy": {
            "execution": "local_only",
            "private_uploads": False,
            "ground_truth_in_prompt": False,
            "references_serialized": False,
            "model_downloads": False,
        },
        "panel": {
            "review_queue": str(review_queue.resolve()),
            "reason": REVIEW_REASON,
            "available": available,
            "selected": selected,
            "limit": limit,
            "order": "field_id ascending",
        },
        "model": {
            "requested": model_name,
            "requested_revision": revision,
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
            "benchmark_wall_ms": None,
            "cuda_memory_mib": None,
        },
        "summary": None,
        "rows": [],
    }


def _validate_options(
    review_queue: Path,
    output_dir: Path,
    output: Path,
    revision: str,
    max_new_tokens: int,
    dtype: str,
    limit: int | None,
) -> None:
    if not review_queue.is_file():
        raise FileNotFoundError(f"review queue was not found: {review_queue}")
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"output directory is not a directory: {output_dir}")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if not revision.strip():
        raise ValueError("revision must be an exact non-empty revision")
    if max_new_tokens <= 0:
        raise ValueError("max-new-tokens must be positive")
    if dtype not in {"bfloat16", "float16"}:
        raise ValueError(f"unsupported dtype: {dtype}")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return round(values[lower] * (1 - weight) + values[upper] * weight, 3)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
