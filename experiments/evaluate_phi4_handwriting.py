"""Compare stock Phi-4 and a handwriting LoRA on a family-held-out crop set."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Sequence
import json
from pathlib import Path
import time
from typing import Any

from PIL import Image

if __package__:
    from experiments import finetune_phi4_handwriting as finetune
else:
    import finetune_phi4_handwriting as finetune  # type: ignore[no-redef]


MAX_NEW_TOKENS = 128


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run_evaluation(
            args.dataset,
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
    parser.add_argument("dataset", type=Path)
    parser.add_argument("adapter", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", default=finetune.MODEL_ID)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=_positive_int, default=MAX_NEW_TOKENS)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--allow-download", action="store_true")
    return parser


def run_evaluation(
    dataset: Path,
    adapter: Path,
    output: Path,
    *,
    model_name: str = finetune.MODEL_ID,
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
    finetune.validate_runtime_choice(model_name, device)
    if not adapter.is_file():
        raise FileNotFoundError(f"adapter was not found: {adapter}")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if max_new_tokens <= 0:
        raise ValueError("max-new-tokens must be positive")

    train, dev = finetune.load_splits(dataset)
    if not all(record.data_origin == "private" for record in dev):
        raise ValueError("held-out evaluation rows must all be private real crops")

    load_started = clock()
    if processor is None or model is None or torch_module is None:
        torch_module, processor, model, _ = finetune.load_runtime(
            model_name,
            device=device,
            seed=seed,
            local_files_only=local_files_only,
        )
    model_load_ms = round((clock() - load_started) * 1000, 3)
    finetune.enable_vision_decoder_lora(model)
    config = getattr(model, "config", None)
    if config is not None:
        config.use_cache = True

    if warmup:
        _predict_record(
            dev[0],
            processor=processor,
            model=model,
            torch_module=torch_module,
            device=device,
            max_new_tokens=max_new_tokens,
            clock=clock,
        )
    stock_rows = _predict_arm(
        dev,
        processor=processor,
        model=model,
        torch_module=torch_module,
        device=device,
        max_new_tokens=max_new_tokens,
        clock=clock,
    )

    finetune.load_trainable_state(
        model,
        adapter,
        torch_module,
        evaluation_family_ids={record.family_id for record in dev},
    )
    if warmup:
        _predict_record(
            dev[0],
            processor=processor,
            model=model,
            torch_module=torch_module,
            device=device,
            max_new_tokens=max_new_tokens,
            clock=clock,
        )
    adapted_rows = _predict_arm(
        dev,
        processor=processor,
        model=model,
        torch_module=torch_module,
        device=device,
        max_new_tokens=max_new_tokens,
        clock=clock,
    )

    payload = {
        "benchmark": "Phi-4 handwriting family-held-out crop comparison",
        "status": "complete",
        "privacy": {
            "execution": "local_only",
            "private_identifiers_persisted": False,
            "private_text_persisted": False,
            "ground_truth_in_prompt": False,
            "model_downloads": not local_files_only,
        },
        "model": {
            "id": finetune.MODEL_ID,
            "revision": finetune.MODEL_REVISION,
        },
        "dataset": {
            **finetune.dataset_summary(train, dev, "canary"),
            "split": "dev",
            "family_disjoint": True,
            "real_only": True,
        },
        "generation": {
            "prompt": finetune.PROMPT,
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
        "arms": {
            "stock": _summarize(dev, stock_rows),
            "adapter": _summarize(dev, adapted_rows),
        },
        "evidence": {
            "paired_fields_complete": len(stock_rows) == len(adapted_rows) == len(dev),
            "failures_remain_in_denominators": True,
            "private_text_persisted": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def _predict_arm(
    records: list[finetune.CropRecord],
    **runtime: Any,
) -> list[dict[str, Any]]:
    model = runtime["model"]
    model.eval()
    return [_predict_record(record, **runtime) for record in records]


def _predict_record(
    record: finetune.CropRecord,
    *,
    processor: Any,
    model: Any,
    torch_module: Any,
    device: str,
    max_new_tokens: int,
    clock: Callable[[], float],
) -> dict[str, Any]:
    started = clock()
    prediction = ""
    generated_tokens = 0
    failures = []
    try:
        _synchronize(torch_module, device)
        started = clock()
        prompt = finetune.inference_prompt(processor)
        with Image.open(record.crop_path) as opened:
            inputs = processor(
                prompt, images=[opened.convert("RGB")], return_tensors="pt"
            ).to(device)
        prompt_tokens = inputs.input_ids.size(1)
        generation_inputs = dict(inputs)
        generation_inputs.setdefault("input_mode", 1)
        with torch_module.inference_mode():
            output = model.generate(
                **generation_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
            )
        _synchronize(torch_module, device)
        sequences = getattr(output, "sequences", output)
        generated = sequences[0][prompt_tokens:]
        generated_tokens = int(
            generated.shape[-1] if hasattr(generated, "shape") else len(generated)
        )
        prediction = processor.decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        if generated_tokens >= max_new_tokens:
            failures.append({"code": "phi4_token_limit", "type": "GenerationLimit"})
    except Exception as error:
        failures.append({"code": "phi4_inference_failed", "type": type(error).__name__})
    latency_ms = round((clock() - started) * 1000, 3)
    return {
        "field_id": record.field_id,
        "family_id": record.family_id,
        "prediction": prediction,
        "status": "failed" if failures else "success",
        "failures": failures,
        "latency_ms": latency_ms,
        "generated_tokens": generated_tokens,
    }


def _summarize(
    records: list[finetune.CropRecord], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    predictions = [
        {"field_id": row["field_id"], "prediction": row["prediction"]} for row in rows
    ]
    validation = finetune.validation_metrics(records, predictions)
    latencies = sorted(float(row["latency_ms"]) for row in rows)
    failures = Counter(failure["code"] for row in rows for failure in row["failures"])
    by_id = {record.field_id: record for record in records}
    critical_substitutions = sum(
        by_id[row["field_id"]].target_state == "resolved"
        and bool(finetune.normalize_text(row["prediction"]))
        and finetune.normalize_text(row["prediction"])
        != finetune.normalize_text(by_id[row["field_id"]].reference)
        for row in rows
    )
    safe_rows = [
        {
            "index": index,
            "status": row["status"],
            "failures": row["failures"],
            "latency_ms": row["latency_ms"],
            "generated_tokens": row["generated_tokens"],
        }
        for index, row in enumerate(rows)
    ]
    return {
        "metrics": {
            "fields": validation["fields"],
            "failed_fields": sum(row["status"] == "failed" for row in rows),
            "failure_codes": dict(sorted(failures.items())),
            "exact": validation["normalized_exact"],
            "exact_rate": validation["normalized_exact_rate"],
            "cer": validation["cer"],
            "missed_character_rate": validation["missed_character_rate"],
            "hallucinated_character_rate": validation["hallucinated_character_rate"],
            "abstention_fields": validation["abstention_fields"],
            "abstention_accuracy": validation["abstention_accuracy"],
            "critical_substitutions": critical_substitutions,
            "latency_ms": {
                "p50": _percentile(latencies, 0.50),
                "p95": _percentile(latencies, 0.95),
            },
        },
        "rows": safe_rows,
    }


def _synchronize(torch_module: Any, device: str) -> None:
    if device.startswith("cuda") and torch_module.cuda.is_available():
        torch_module.cuda.synchronize()


def _percentile(values: list[float], fraction: float) -> float:
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
