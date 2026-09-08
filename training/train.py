"""Experimental Falcon-OCR LoRA training, with before/after held-out evaluation.

Run in the official Falcon PyTorch CUDA environment with PYTHONPATH=src.
JSONL rows require image, text, kind, and image_id (original source identity).
Image paths may be relative to the JSONL file. Optional category is text, table,
formula, or code; code uses the official text prompt. Targets must follow
that task's output format.
Outputs are unmerged adapters and evaluation records, never serving defaults.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import json
import math
from pathlib import Path
import random
import time
from unittest.mock import patch

from PIL import Image

from ocr_pipeline.verification import edit_distance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    if (
        any(
            value <= 0
            for value in (
                args.epochs,
                args.rank,
                args.accumulation,
                args.max_new_tokens,
            )
        )
        or not math.isfinite(args.learning_rate)
        or args.learning_rate <= 0
    ):
        parser.error("Training counts and learning rate must be positive and finite")
    if args.output.exists():
        parser.error("Use a new output directory to preserve previous experiments")
    train, validation = load_rows(args.train), load_rows(args.validation)
    if {row["image_id"] for row in train} & {row["image_id"] for row in validation}:
        parser.error("Training and validation source images overlap")
    if {row["image"] for row in train} & {row["image"] for row in validation}:
        parser.error("Training and validation image paths overlap")
    run(args, train, validation)


def run(args, train, validation) -> None:
    import torch
    from torch.nn import functional as F
    from falcon_perception import load_and_prepare_model, setup_torch_config
    from model import add_lora, forward, load_adapter, prepare_batch

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    setup_torch_config()
    model, tokenizer, config = load_and_prepare_model(
        hf_local_dir=str(args.model_dir), device="cuda", dtype="bfloat16", compile=False
    )
    if model.args.perception_heads:
        raise ValueError("This experiment requires Falcon-OCR, not Falcon-Perception")
    for row in train:
        ids = tokenizer._tok.encode(row["text"], add_special_tokens=False).ids
        if not ids or tokenizer.decode(ids) != row["text"]:
            raise ValueError(
                "Training target must round-trip through the native tokenizer"
            )
        row["target_ids"] = ids + [tokenizer.end_of_query_token_id]
    args.output.mkdir(parents=True)
    settings = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    settings.update(
        train_samples=len(train),
        validation_samples=len(validation),
        train_source_images=len({row["image_id"] for row in train}),
        validation_source_images=len({row["image_id"] for row in validation}),
        termination_id=tokenizer.end_of_query_token_id,
        dtype=str(model.dtype),
        torch_version=torch.__version__,
        checkpoint_format="unmerged LoRA A/B tensors; alpha equals rank",
    )
    (args.output / "settings.json").write_text(json.dumps(settings, indent=2))
    before = evaluate(
        model,
        tokenizer,
        config,
        validation,
        args.output / "before.jsonl",
        args.max_new_tokens,
    )
    torch.compiler.reset()
    parameters = add_lora(model, rank=args.rank)
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate)
    model.train()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    step = 0
    with (args.output / "training.jsonl").open("w") as sink:
        for epoch in range(args.epochs):
            random.shuffle(train)
            for start in range(0, len(train), args.accumulation):
                group = train[start : start + args.accumulation]
                optimizer.zero_grad(set_to_none=True)
                losses = []
                for row in group:
                    prompt = prepare_prompt(model, tokenizer, config, row)
                    batch = prepare_batch(model, tokenizer, prompt, [row["target_ids"]])
                    logits = forward(model, batch)
                    loss = F.cross_entropy(
                        logits.float().flatten(0, 1),
                        batch["targets"].flatten(),
                        ignore_index=-100,
                    )
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite training loss")
                    # Each example has equal weight, including the final partial group.
                    (loss / len(group)).backward()
                    losses.append(float(loss.detach()))
                    del logits, loss, batch, prompt
                if any(
                    parameter.grad is None or not torch.isfinite(parameter.grad).all()
                    for parameter in parameters
                ):
                    raise FloatingPointError("Missing or nonfinite adapter gradient")
                optimizer.step()
                if any(not torch.isfinite(parameter).all() for parameter in parameters):
                    raise FloatingPointError("Nonfinite adapter parameter")
                step += 1
                record = dict(
                    step=step,
                    epoch=epoch + 1,
                    samples=len(group),
                    mean_sample_nll=sum(losses) / len(losses),
                    elapsed_seconds=time.perf_counter() - started,
                )
                sink.write(json.dumps(record) + "\n")
                sink.flush()
                print("training", record, flush=True)
    torch.save(
        {
            name: parameter.detach().cpu()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        },
        args.output / "adapter.pt",
    )
    del model, optimizer, parameters
    gc.collect()
    torch.cuda.empty_cache()
    torch.compiler.reset()
    model, tokenizer, config = load_and_prepare_model(
        hf_local_dir=str(args.model_dir), device="cuda", dtype="bfloat16", compile=False
    )
    model = load_adapter(model, args.output / "adapter.pt", rank=args.rank)
    after = evaluate(
        model,
        tokenizer,
        config,
        validation,
        args.output / "after.jsonl",
        args.max_new_tokens,
    )
    report = dict(
        settings=settings,
        before=before,
        after=after,
        steps=step,
        peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
    )
    (args.output / "results.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


def load_rows(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or any(
            not isinstance(row.get(key), str) or not row[key].strip()
            for key in ("image", "text", "kind", "image_id")
        ):
            raise ValueError(
                f"{path}:{line_number}: require nonempty image/text/kind/image_id"
            )
        category = row.get("category", "text")
        if not isinstance(category, str) or category not in {
            "text",
            "table",
            "formula",
            "code",
        }:
            raise ValueError(f"{path}:{line_number}: unsupported OCR category")
        image = Path(row["image"])
        if not image.is_absolute():
            image = path.parent / image
        rows.append({**row, "image": str(image.resolve()), "category": category})
    if not rows:
        raise ValueError(f"{path}: dataset is empty")
    return rows


def prepare_prompt(model, tokenizer, config, row):
    from falcon_perception.batch_inference import process_batch_and_generate
    from falcon_perception.paged_ocr_inference import OCRInferenceEngine
    from ocr_pipeline.falcon import _prepare_falcon_crop

    with Image.open(row["image"]) as source:
        image = source.convert("RGB")
        try:
            prepared = _prepare_falcon_crop(image, 1024)
            try:
                category = "text" if row["category"] == "code" else row["category"]
                prompt = process_batch_and_generate(
                    tokenizer,
                    [(prepared.image, OCRInferenceEngine._make_ocr_prompt(category))],
                    max_length=config.max_seq_len,
                    min_dimension=64,
                    max_dimension=1024,
                )
            finally:
                prepared.image.close()
        finally:
            image.close()
    return {key: value.to(model.device) for key, value in prompt.items()}


def evaluate(model, tokenizer, config, rows, output, max_new_tokens):
    from falcon_perception.batch_inference import BatchInferenceEngine
    from falcon_perception.sampling import sample_token

    model.eval()
    engine = BatchInferenceEngine(
        model, tokenizer, kernel_options={"BLOCK_M": 64, "BLOCK_N": 64, "num_stages": 1}
    )
    stops = [tokenizer.eos_token_id, tokenizer.end_of_query_token_id]
    summaries = defaultdict(
        lambda: dict(samples=0, characters=0, edits=0, exact=0, failures=0)
    )
    with output.open("w") as sink:
        for original in rows:
            row = dict(original, prediction="")
            ids = []

            def sample(logits, rng=None, temperature=0.0, top_k=None):
                # The native engine rounds its cache budget up to a 128-token boundary.
                if len(ids) >= max_new_tokens:
                    raise RuntimeError(
                        "generation did not terminate within token limit"
                    )
                result = sample_token(
                    logits, rng=rng, temperature=temperature, top_k=top_k
                )
                ids.append(int(result[0].item()))
                return result

            try:
                prompt = prepare_prompt(model, tokenizer, config, row)
                with patch("falcon_perception.batch_inference.sample_token", sample):
                    engine.generate(
                        **prompt,
                        max_new_tokens=max_new_tokens,
                        temperature=0.0,
                        stop_token_ids=stops,
                        task="ocr_plain",
                        coord_dedup_threshold=0.0,
                    )
                terminated = bool(ids and ids[-1] in stops)
                if not terminated:
                    row["failure"] = "generation did not terminate"
            except Exception as error:
                row["failure"] = f"{type(error).__name__}: {error}"
            row["prediction"] = tokenizer.decode(
                ids[:-1] if ids and ids[-1] in stops else ids
            )
            row["edits"] = edit_distance(row["text"], row["prediction"])
            summary = summaries[row["category"] + "/" + row["kind"]]
            summary["samples"] += 1
            summary["characters"] += len(row["text"])
            summary["edits"] += row["edits"]
            summary["exact"] += (
                row["text"] == row["prediction"] and "failure" not in row
            )
            summary["failures"] += "failure" in row
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            sink.flush()
    for summary in summaries.values():
        summary["cer"] = summary["edits"] / summary["characters"]
        summary["exact_rate"] = summary["exact"] / summary["samples"]
    return dict(summaries)


if __name__ == "__main__":
    main()
