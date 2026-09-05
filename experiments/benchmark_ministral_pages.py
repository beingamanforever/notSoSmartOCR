"""Benchmark the local Ministral page OCR challenger on the frozen panel."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any

if __package__:
    from experiments.benchmark_phi4 import PageCase, load_page_cases
    from experiments.evaluate_challenge_set import evaluate_challenge_set
else:
    from benchmark_phi4 import PageCase, load_page_cases  # type: ignore[no-redef]
    from evaluate_challenge_set import (  # type: ignore[no-redef]
        evaluate_challenge_set,
    )

from ocr_pipeline.pipeline import process_document
from ocr_pipeline.providers import (
    MINISTRAL_MODEL_ID,
    MINISTRAL_MODEL_REVISION,
    MINISTRAL_OCR_PROMPT,
    LocalReader,
    MinistralOCRReader,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = run_benchmark(
            args.challenge_root,
            args.output,
            case_ids=tuple(args.case or ()),
            model_name=args.model,
            revision=args.revision,
            max_new_tokens=args.max_new_tokens,
        )
    except (
        FileExistsError,
        FileNotFoundError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "challenge_root",
        type=Path,
        nargs="?",
        default=(
            repository
            / "internal-clinical-ocr-benchmark"
            / "challenging-formats-20260902"
        ),
    )
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--case", action="append", help="Case ID to run; repeat as needed"
    )
    parser.add_argument("--model", default=MINISTRAL_MODEL_ID)
    parser.add_argument("--revision", default=MINISTRAL_MODEL_REVISION)
    parser.add_argument("--max-new-tokens", type=_positive_int, default=8192)
    return parser


def run_benchmark(
    challenge_root: Path,
    output: Path,
    *,
    case_ids: tuple[str, ...] = (),
    model_name: str = MINISTRAL_MODEL_ID,
    revision: str = MINISTRAL_MODEL_REVISION,
    max_new_tokens: int = 8192,
    reader: LocalReader | None = None,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Benchmark output already exists: {output}")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")

    cases = _select_cases(load_page_cases(challenge_root, "full", ()), case_ids)
    page_reader = reader or MinistralOCRReader(
        model_name=model_name,
        model_revision=revision,
        prompt=MINISTRAL_OCR_PROMPT,
        max_new_tokens=max_new_tokens,
    )
    output.mkdir(parents=True)
    model_output = output / "model-output"

    completed = failed = 0
    for case in cases:
        record = _run_case(case, page_reader)
        if record["result"]["status"] == "success":
            completed += 1
        else:
            failed += 1
        _write_json(
            model_output / case.id.split("-", 1)[0] / f"{case.id}.json",
            record,
        )

    with tempfile.TemporaryDirectory(prefix="ministral-page-benchmark-") as temporary:
        selected_annotations = Path(temporary) / "annotations"
        _copy_annotations(challenge_root, selected_annotations, cases)
        evaluation = evaluate_challenge_set(selected_annotations, model_output)

    report = {
        "benchmark": "Ministral frozen clinical page OCR challenger",
        "status": "complete",
        "cases": {
            "selected": [case.id for case in cases],
            "attempted": len(cases),
            "completed": completed,
            "failed": failed,
            "failures_remain_in_denominator": True,
        },
        "model": getattr(page_reader, "provenance", {"requested": model_name}),
        "prompt": MINISTRAL_OCR_PROMPT,
        "generation": {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
        },
        "evaluation": evaluation,
    }
    _write_json(output / "report.json", report)
    return report


def _select_cases(cases: list[PageCase], case_ids: tuple[str, ...]) -> list[PageCase]:
    if not case_ids:
        return cases
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Selected case IDs must be unique")
    by_id = {case.id: case for case in cases}
    unknown = [case_id for case_id in case_ids if case_id not in by_id]
    if unknown:
        raise ValueError(f"Unknown frozen case ID: {unknown[0]}")
    return [by_id[case_id] for case_id in case_ids]


def _run_case(case: PageCase, reader: LocalReader) -> dict[str, Any]:
    timings: dict[str, float] = {}
    started = time.perf_counter()
    result = process_document(case.source, reader, timings=timings)
    elapsed = time.perf_counter() - started
    return {
        "case_id": case.id,
        "category_id": case.id.split("-", 1)[0],
        "filename": case.source.name,
        "source_file": case.source_file,
        "request_status": "complete",
        "elapsed_seconds": round(elapsed, 3),
        "timing": {
            "total_seconds": round(elapsed, 3),
            "pipeline_steps": {
                name: round(value, 3) for name, value in timings.items()
            },
        },
        "result": result.to_dict(),
    }


def _copy_annotations(
    challenge_root: Path,
    destination: Path,
    cases: list[PageCase],
) -> None:
    for case in cases:
        category = case.id.split("-", 1)[0]
        source = (
            challenge_root / "annotations" / "primary" / category / f"{case.id}.json"
        )
        target = destination / category / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
