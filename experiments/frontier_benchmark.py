"""Benchmark direct hosted vision models on public OCR transcription data."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

from ocr_pipeline.openrouter import (
    DEFAULT_MAX_TOKENS,
    IMAGE_REPAIR_MODELS,
    MAX_TOKENS,
    OpenRouterError,
    repair_image,
)

if __package__:
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

TEXT_SCHEMA = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
    "additionalProperties": False,
}
PROMPT = (
    "Transcribe every visible character exactly in reading order. Preserve line "
    "breaks, punctuation, and table cell text. Do not explain, correct, or infer."
)
PROMPT_VERSION = "literal-transcription-v1"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark a direct OpenRouter vision model on public OCR data"
    )
    parser.add_argument("dataset", choices=("clinocr", "funsd"))
    parser.add_argument("root", type=Path, help="Extracted dataset root")
    parser.add_argument("output", type=Path, help="JSON results path")
    parser.add_argument(
        "--model",
        choices=sorted(IMAGE_REPAIR_MODELS),
        default="qwen/qwen3.8-flash",
    )
    parser.add_argument("--provider", help="Exact OpenRouter provider slug")
    parser.add_argument(
        "--subset",
        action="append",
        help="Run only this subset; repeat to select more than one",
    )
    parser.add_argument(
        "--limit-per-subset",
        type=_positive_int,
        help="Run the first N cases in each subset",
    )
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
            selected_subsets=set(args.subset) if args.subset else None,
            limit_per_subset=args.limit_per_subset,
            max_tokens=args.max_tokens,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0


def run_benchmark(
    dataset: str,
    root: Path,
    model: str,
    *,
    provider_slug: str | None = None,
    selected_subsets: set[str] | None = None,
    limit_per_subset: int | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict[str, object]:
    if model not in IMAGE_REPAIR_MODELS:
        raise ValueError(f"Unsupported frontier model: {model}")
    if provider_slug is not None and (
        not provider_slug or provider_slug != provider_slug.strip()
    ):
        raise ValueError("Provider slug must be a non-empty trimmed string")
    if limit_per_subset is not None and limit_per_subset < 1:
        raise ValueError("Limit per subset must be positive")
    if isinstance(max_tokens, bool) or not 1 <= max_tokens <= MAX_TOKENS:
        raise ValueError(f"Max tokens must be between 1 and {MAX_TOKENS}")

    cases = discover_cases(dataset, root)
    if selected_subsets:
        available_subsets = {case.subset for case in cases}
        unknown_subsets = selected_subsets - available_subsets
        if unknown_subsets:
            raise ValueError(f"Unknown subsets: {sorted(unknown_subsets)}")
        cases = [case for case in cases if case.subset in selected_subsets]
    if limit_per_subset is not None:
        cases = _limit_cases(cases, limit_per_subset)
    if not cases:
        raise ValueError(f"No evaluation cases found in {root}")

    records = [
        _evaluate_case(case, root, model, provider_slug, max_tokens) for case in cases
    ]
    subsets = {
        subset: _summarize([record for record in records if record["subset"] == subset])
        for subset in sorted({case.subset for case in cases})
    }
    return {
        "dataset": dataset,
        "dataset_root": str(root.resolve()),
        "reader": "openrouter-direct",
        "run_config": {
            "reader": "openrouter-direct",
            "requested_model": model,
            "provider_slug": provider_slug,
            "selected_subsets": (
                sorted(selected_subsets) if selected_subsets else None
            ),
            "limit_per_subset": limit_per_subset,
            "prompt_version": PROMPT_VERSION,
            "max_tokens": max_tokens,
        },
        "requested_model": model,
        "provider_slug": provider_slug,
        "selected_subsets": sorted(selected_subsets) if selected_subsets else None,
        "prompt_version": PROMPT_VERSION,
        "max_tokens": max_tokens,
        "normalization": NORMALIZATION,
        "summary": _summarize(records),
        "subsets": subsets,
        "cases": records,
    }


def _evaluate_case(
    case: BenchmarkCase,
    root: Path,
    model: str,
    provider_slug: str | None,
    max_tokens: int,
) -> dict[str, object]:
    started = time.perf_counter()
    try:
        result = repair_image(
            case.image_path,
            PROMPT,
            TEXT_SCHEMA,
            model=model,
            max_tokens=max_tokens,
            provider_slug=provider_slug,
        )
        structured_prediction = str(result.content["text"])
        prediction = _transcription_text(structured_prediction)
        failures = []
        status = "success"
        if not prediction.strip():
            failures = [
                {
                    "stage": "provider",
                    "code": "empty_prediction",
                    "message": "The provider returned no visible text",
                }
            ]
            status = "failed"
        actual_model = result.model
        provider = result.provider
        usage = result.usage
        cost = result.cost
        api_latency_ms = result.latency_ms
        attempts = result.attempts
        finish_reason = result.finish_reason
        native_finish_reason = result.native_finish_reason
        provider_error = None
    except OpenRouterError as error:
        structured_prediction = ""
        prediction = ""
        failures = [
            {
                "stage": "provider",
                "code": error.code,
                "message": str(error),
            }
        ]
        status = "failed"
        actual_model = None
        provider = None
        usage = {}
        cost = None
        api_latency_ms = error.latency_ms
        attempts = error.attempts
        finish_reason = None
        native_finish_reason = None
        provider_error = {
            "code": error.code,
            "status_code": error.status_code,
            "attempts": error.attempts,
            "latency_ms": error.latency_ms,
        }

    wall_latency_ms = round((time.perf_counter() - started) * 1000, 3)
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
        "requested_model": model,
        "model": actual_model,
        "provider": provider,
        "usage": usage,
        "cost": cost,
        "attempts": attempts,
        "finish_reason": finish_reason,
        "native_finish_reason": native_finish_reason,
        "provider_error": provider_error,
        "api_latency_ms": api_latency_ms,
        "wall_latency_ms": wall_latency_ms,
        "latency_ms": wall_latency_ms,
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
