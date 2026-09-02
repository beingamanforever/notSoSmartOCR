"""Compare stock Phi-4 and a handwriting LoRA on the fixed C14 crop panel."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
import json
from pathlib import Path
import time
from typing import Any

if __package__:
    from experiments import benchmark_phi4_crops as c14
    from experiments import evaluate_phi4_handwriting as evaluate
else:
    import benchmark_phi4_crops as c14  # type: ignore[no-redef]
    import evaluate_phi4_handwriting as evaluate  # type: ignore[no-redef]


MAX_NEW_TOKENS = 128


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run_evaluation(
            args.run_root,
            args.adapter,
            args.output,
            model_name=args.model,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
            local_files_only=not args.allow_download,
        )
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("adapter", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", default=evaluate.finetune.MODEL_ID)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=_positive_int, default=MAX_NEW_TOKENS)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--allow-download", action="store_true")
    return parser


def run_evaluation(
    run_root: Path,
    adapter: Path,
    output: Path,
    *,
    model_name: str = evaluate.finetune.MODEL_ID,
    device: str = "cuda:0",
    max_new_tokens: int = MAX_NEW_TOKENS,
    seed: int = 17,
    local_files_only: bool = True,
    processor: Any | None = None,
    model: Any | None = None,
    torch_module: Any | None = None,
    clock: Callable[[], float] = time.perf_counter,
    warmup: bool = True,
) -> dict[str, Any]:
    evaluate.finetune.validate_runtime_choice(model_name, device)
    if not adapter.is_file():
        raise FileNotFoundError(f"adapter was not found: {adapter}")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if max_new_tokens <= 0:
        raise ValueError("max-new-tokens must be positive")

    cases = c14.load_crop_cases(run_root)
    records, metadata = _evaluation_records(cases)
    load_started = clock()
    if processor is None or model is None or torch_module is None:
        torch_module, processor, model, _ = evaluate.finetune.load_runtime(
            model_name,
            device=device,
            seed=seed,
            local_files_only=local_files_only,
        )
    model_load_ms = round((clock() - load_started) * 1000, 3)
    evaluate.finetune.enable_vision_decoder_lora(model)
    config = getattr(model, "config", None)
    if config is not None:
        config.use_cache = True

    stock = _run_arm(
        records,
        metadata,
        processor=processor,
        model=model,
        torch_module=torch_module,
        device=device,
        max_new_tokens=max_new_tokens,
        clock=clock,
        warmup=warmup,
    )
    evaluate.finetune.load_trainable_state(
        model,
        adapter,
        torch_module,
        evaluation_family_ids={record.family_id for record in records},
    )
    adapted = _run_arm(
        records,
        metadata,
        processor=processor,
        model=model,
        torch_module=torch_module,
        device=device,
        max_new_tokens=max_new_tokens,
        clock=clock,
        warmup=warmup,
    )

    payload = {
        "benchmark": "Phi-4 fixed C14 handwriting crop comparison",
        "status": "complete",
        "privacy": {
            "execution": "local_only",
            "private_text_persisted": False,
            "ground_truth_in_prompt": False,
            "model_downloads": not local_files_only,
        },
        "model": {
            "id": evaluate.finetune.MODEL_ID,
            "revision": evaluate.finetune.MODEL_REVISION,
            "adapter": str(adapter.resolve()),
        },
        "dataset": {
            "root": str(run_root.resolve()),
            "fields": len(cases),
            "views": list(c14.VARIANTS),
            "paired_inferences": len(records),
            "private_real_only": True,
            "fixed_panel": True,
        },
        "generation": {
            "prompt": evaluate.finetune.PROMPT,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "num_beams": 1,
            "normalization": "NFKC, casefold, whitespace collapse",
            "critical_substitution_policy": (
                "non-empty incorrect output for a resolved clinical field"
            ),
            "seed": seed,
        },
        "runtime": {
            "device": device,
            "dtype": "bfloat16",
            "attention": "sdpa",
            "use_cache": True,
            "model_load_ms": model_load_ms,
            "warmup_per_arm": warmup,
        },
        "arms": {"stock": stock, "adapter": adapted},
        "evidence": {
            "paired_fields_complete": (
                len(stock["rows"]) == len(adapted["rows"]) == len(records)
            ),
            "identical_pairs": [row["pair_id"] for row in stock["rows"]]
            == [row["pair_id"] for row in adapted["rows"]],
            "failures_remain_in_denominators": True,
            "private_text_persisted": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def _evaluation_records(
    cases: list[c14.CropCase],
) -> tuple[list[evaluate.finetune.CropRecord], dict[str, dict[str, Any]]]:
    records = []
    metadata = {}
    for case in cases:
        for view in c14.VARIANTS:
            pair_id = f"{case.field_id}:{view}"
            records.append(
                evaluate.finetune.CropRecord(
                    field_id=pair_id,
                    case_id=case.case_id,
                    family_id=case.case_id,
                    reference=case.reference,
                    crop_path=getattr(case, view),
                    split="test",
                    data_origin="private",
                )
            )
            metadata[pair_id] = {
                "pair_id": pair_id,
                "field_id": case.field_id,
                "case_id": case.case_id,
                "view": view,
                "bbox": list(case.bbox),
            }
    return records, metadata


def _run_arm(
    records: list[evaluate.finetune.CropRecord],
    metadata: dict[str, dict[str, Any]],
    *,
    warmup: bool,
    **runtime: Any,
) -> dict[str, Any]:
    if warmup:
        evaluate._predict_record(records[0], **runtime)
    rows = evaluate._predict_arm(records, **runtime)
    return _summarize(records, rows, metadata)


def _summarize(
    records: list[evaluate.finetune.CropRecord],
    rows: list[dict[str, Any]],
    metadata: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    summary = evaluate._summarize(records, rows)
    summary["metrics"] = _metrics(records, rows)
    summary["views"] = {
        view: _metrics(
            [record for record in records if metadata[record.field_id]["view"] == view],
            [row for row in rows if metadata[row["field_id"]]["view"] == view],
        )
        for view in c14.VARIANTS
    }
    summary["rows"] = [
        {
            **metadata[row["field_id"]],
            "status": row["status"],
            "failures": row["failures"],
            "latency_ms": row["latency_ms"],
            "generated_tokens": row["generated_tokens"],
        }
        for row in rows
    ]
    return summary


def _metrics(
    records: list[evaluate.finetune.CropRecord], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    metrics = evaluate._summarize(records, rows)["metrics"]
    record_by_id = {record.field_id: record for record in records}
    row_by_id = {row["field_id"]: row for row in rows}
    critical_fields = {
        record.field_id
        for record in records
        if c14._is_critical_reference(record.reference)
    }
    metrics["critical_fields"] = len(critical_fields)
    metrics["critical_substitutions"] = sum(
        field_id in critical_fields
        and bool(evaluate.finetune.normalize_text(row_by_id[field_id]["prediction"]))
        and evaluate.finetune.normalize_text(row_by_id[field_id]["prediction"])
        != evaluate.finetune.normalize_text(record_by_id[field_id].reference)
        for field_id in record_by_id
    )
    return metrics


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
