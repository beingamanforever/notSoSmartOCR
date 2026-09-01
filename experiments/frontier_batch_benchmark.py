"""Benchmark Gemini batch vision on public OCR transcription data."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

from ocr_pipeline.openrouter import (
    DEFAULT_MAX_TOKENS,
    GEMINI_37_FLASH_BATCH_MODEL,
    GEMINI_37_FLASH_MODEL,
    MAX_TOKENS,
    OpenRouterError,
)
from ocr_pipeline.openrouter_batch import BatchItemResult, repair_images_batch

if __package__:
    from experiments.frontier_benchmark import PROMPT, PROMPT_VERSION, TEXT_SCHEMA
    from experiments.public_benchmark import (
        NORMALIZATION,
        BenchmarkCase,
        _limit_cases,
        _relative_path,
        _score,
        _summarize,
        _transcription_text,
        discover_cases,
    )
else:
    from frontier_benchmark import (  # type: ignore[no-redef]
        PROMPT,
        PROMPT_VERSION,
        TEXT_SCHEMA,
    )
    from public_benchmark import (  # type: ignore[no-redef]
        NORMALIZATION,
        BenchmarkCase,
        _limit_cases,
        _relative_path,
        _score,
        _summarize,
        _transcription_text,
        discover_cases,
    )

BATCH_RETENTION_DAYS = 30


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark OpenRouter Gemini batch vision on public OCR data"
    )
    parser.add_argument("dataset", choices=("clinocr", "funsd"))
    parser.add_argument("root", type=Path, help="Extracted dataset root")
    parser.add_argument("output", type=Path, help="JSON results path")
    parser.add_argument(
        "--model",
        choices=(GEMINI_37_FLASH_BATCH_MODEL,),
        default=GEMINI_37_FLASH_BATCH_MODEL,
    )
    parser.add_argument("--provider", required=True, help="Exact provider slug")
    parser.add_argument(
        "--batch-id",
        help="Resume polling an existing batch instead of submitting a new one",
    )
    parser.add_argument(
        "--clinocr-role",
        choices=("exemplar", "eval"),
        default="exemplar",
        help="Use exemplars for development; select eval only after freezing",
    )
    parser.add_argument("--subset", action="append")
    parser.add_argument("--limit-per-subset", type=_positive_int)
    parser.add_argument(
        "--max-tokens",
        type=_max_tokens,
        default=DEFAULT_MAX_TOKENS,
        help=f"Maximum output tokens, from 1 to {MAX_TOKENS}",
    )
    args = parser.parse_args(argv)

    try:
        payload = run_benchmark(
            args.dataset,
            args.root,
            args.model,
            provider_slug=args.provider,
            batch_id=args.batch_id,
            clinocr_role=args.clinocr_role,
            selected_subsets=set(args.subset) if args.subset else None,
            limit_per_subset=args.limit_per_subset,
            max_tokens=args.max_tokens,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (OSError, OpenRouterError, ValueError) as error:
        parser.error(str(error))
    return 0


def run_benchmark(
    dataset: str,
    root: Path,
    model: str = GEMINI_37_FLASH_BATCH_MODEL,
    *,
    provider_slug: str | None = None,
    batch_id: str | None = None,
    clinocr_role: str = "exemplar",
    selected_subsets: set[str] | None = None,
    limit_per_subset: int | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict[str, object]:
    _validate_options(model, provider_slug, batch_id, limit_per_subset, max_tokens)
    cases = _select_cases(
        dataset,
        root,
        clinocr_role,
        selected_subsets,
        limit_per_subset,
    )
    assert provider_slug is not None
    started = time.perf_counter()
    batch = repair_images_batch(
        [(case.id, case.image_path) for case in cases],
        PROMPT,
        TEXT_SCHEMA,
        model=model,
        max_tokens=max_tokens,
        provider_slug=provider_slug,
        batch_id=batch_id,
    )
    batch_wall_latency_ms = (time.perf_counter() - started) * 1000
    by_id = {item.custom_id: item for item in batch.items}
    records = [
        _record(case, root, model, by_id[case.id], batch.latency_ms) for case in cases
    ]
    subsets = {
        subset: _summarize(
            [record for record in records if record["subset"] == subset],
            latency_field="api_latency_ms",
            latency_basis="provider_api_latency_ms",
        )
        for subset in sorted({case.subset for case in cases})
    }
    selected = sorted(selected_subsets) if selected_subsets else None
    run_config = {
        "reader": "openrouter-batch",
        "requested_model": model,
        "submitted_model": GEMINI_37_FLASH_MODEL,
        "provider_slug": provider_slug,
        "batch_id": batch_id,
        "clinocr_role": clinocr_role if dataset == "clinocr" else None,
        "selected_subsets": selected,
        "limit_per_subset": limit_per_subset,
        "prompt_version": PROMPT_VERSION,
        "max_tokens": max_tokens,
    }
    summary = _summarize(
        records,
        wall_latency_ms=batch_wall_latency_ms if batch_id is None else None,
        latency_field="api_latency_ms",
        latency_basis="provider_api_latency_ms",
    )
    summary["throughput_basis"] = (
        "submission_to_completed_wall"
        if batch_id is None
        else "unavailable_for_resumed_batch"
    )
    if batch_id is not None:
        summary["wall_latency_ms"] = None
        summary["pages_per_second"] = None
    return {
        "dataset": dataset,
        "dataset_root": str(root.resolve()),
        "reader": "openrouter-batch",
        "run_config": run_config,
        "requested_model": model,
        "submitted_model": GEMINI_37_FLASH_MODEL,
        "provider_slug": provider_slug,
        "clinocr_role": clinocr_role if dataset == "clinocr" else None,
        "selected_subsets": selected,
        "prompt_version": PROMPT_VERSION,
        "max_tokens": max_tokens,
        "normalization": NORMALIZATION,
        "batch": {
            "id": batch.batch_id,
            "resumed": batch_id is not None,
            "submission_config_verified": batch_id is None,
            "status": batch.status,
            "polls": batch.polls,
            "latency_ms": batch.latency_ms,
            "wall_latency_ms": round(batch_wall_latency_ms, 3),
            "retention": {
                "days": BATCH_RETENTION_DAYS,
                "scope": "inputs_and_results",
                "caveat": (
                    "OpenRouter retains batch inputs and results for 30 days; "
                    "submit public benchmark data only."
                ),
            },
        },
        "summary": summary,
        "subsets": subsets,
        "cases": records,
    }


def _validate_options(
    model: str,
    provider_slug: str | None,
    batch_id: str | None,
    limit_per_subset: int | None,
    max_tokens: int,
) -> None:
    if model != GEMINI_37_FLASH_BATCH_MODEL:
        raise ValueError(f"Unsupported frontier batch model: {model}")
    if provider_slug is None:
        raise ValueError("Provider slug is required for controlled benchmarks")
    if not provider_slug or provider_slug != provider_slug.strip():
        raise ValueError("Provider slug must be a non-empty trimmed string")
    if batch_id is not None and (not batch_id or batch_id != batch_id.strip()):
        raise ValueError("Batch ID must be a non-empty trimmed string")
    if limit_per_subset is not None and limit_per_subset < 1:
        raise ValueError("Limit per subset must be positive")
    if isinstance(max_tokens, bool) or not 1 <= max_tokens <= MAX_TOKENS:
        raise ValueError(f"Max tokens must be between 1 and {MAX_TOKENS}")


def _select_cases(
    dataset: str,
    root: Path,
    clinocr_role: str,
    selected_subsets: set[str] | None,
    limit_per_subset: int | None,
) -> list[BenchmarkCase]:
    cases = discover_cases(dataset, root, clinocr_role=clinocr_role)
    if selected_subsets:
        unknown = selected_subsets - {case.subset for case in cases}
        if unknown:
            raise ValueError(f"Unknown subsets: {sorted(unknown)}")
        cases = [case for case in cases if case.subset in selected_subsets]
    if limit_per_subset is not None:
        cases = _limit_cases(cases, limit_per_subset)
    if not cases:
        raise ValueError(f"No evaluation cases found in {root}")
    return cases


def _record(
    case: BenchmarkCase,
    root: Path,
    requested_model: str,
    item: BatchItemResult,
    batch_latency_ms: float,
) -> dict[str, object]:
    error = item.error
    structured_prediction = "" if error else str(item.content["text"])
    prediction = _transcription_text(structured_prediction)
    if error:
        failures = [{"stage": "provider", "code": error.code, "message": str(error)}]
    else:
        failures = []
    status = "success" if not failures else "failed"
    return {
        "id": case.id,
        "cluster_id": case.cluster_id,
        "subset": case.subset,
        "image": _relative_path(case.image_path, root),
        "prediction": prediction,
        "structured_prediction": structured_prediction,
        "reference": case.reference,
        "status": status,
        "metrics": _score(prediction, case.reference),
        "failures": failures,
        "requested_model": requested_model,
        "model": item.model,
        "provider": item.provider,
        "usage": item.usage or {},
        "cost": item.cost,
        "attempts": 1,
        "finish_reason": item.finish_reason,
        "native_finish_reason": item.native_finish_reason,
        "provider_error": (
            {
                "code": error.code,
                "status_code": error.status_code,
                "attempts": error.attempts,
                "latency_ms": error.latency_ms,
            }
            if error
            else None
        ),
        "api_latency_ms": item.latency_ms,
        "batch_latency_ms": batch_latency_ms,
        "latency_ms": item.latency_ms
        if item.latency_ms is not None
        else batch_latency_ms,
    }


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return number


def _max_tokens(value: str) -> int:
    number = int(value)
    if not 1 <= number <= MAX_TOKENS:
        raise argparse.ArgumentTypeError(
            f"max tokens must be between 1 and {MAX_TOKENS}"
        )
    return number


if __name__ == "__main__":
    raise SystemExit(main())
