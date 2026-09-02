"""Evaluate Granite Docling on the frozen C14 handwritten-field panel."""

from __future__ import annotations

import argparse
import html
import json
import random
import re
import time
import unicodedata
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

MODEL_ID = "ibm-granite/granite-docling-258M"
CROP_PROMPT = "Convert this page to docling."
BBOX_PROMPT_PREFIX = "OCR the text in a specific location: "
MODES = ("crop", "bbox")
NONCRITICAL_FIELDS = {f"C14-D001-P001-H{index:03d}" for index in range(9, 14)}
SPECIAL_TOKEN = re.compile(r"<\|[^<>]*\|>")
DOCTAG = re.compile(r"</?[^<>]+>")


def build_parser() -> argparse.ArgumentParser:
    repository = Path(__file__).resolve().parents[1]
    run_root = (
        repository
        / "internal-clinical-ocr-benchmark"
        / "challenging-formats-20260902"
        / "runs"
        / "ministral-c14-crops-source-only"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=run_root)
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument("--annotation-root", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--inference-only", action="store_true")
    parser.add_argument("--score-input", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("max-new-tokens must be positive")
    if args.inference_only and args.score_input:
        raise ValueError("inference-only and score-input are mutually exclusive")

    ground_truth_path = args.ground_truth or args.run_root / "ground_truth.json"
    annotation_root = args.annotation_root or (
        args.run_root.parents[1] / "annotations" / "primary" / "C14"
    )
    source_root = args.source_root or (
        args.run_root.parents[1] / "sources" / "C14-user-reported"
    )
    output_path = args.output or args.run_root / "granite_docling_258m_results.json"
    if args.score_input:
        records = load_ground_truth(ground_truth_path)
        score_inference(args.score_input, output_path, records, args.modes)
        return

    if args.inference_only:
        records = discover_inputs(args.run_root, source_root)
    else:
        records = (
            load_ground_truth(ground_truth_path)
            if ground_truth_path.is_file()
            else load_annotations(annotation_root)
        )

    torch, transformers, processor, model = load_model(
        args.model,
        args.model_revision,
        args.seed,
    )
    payload: dict[str, Any] = {
        "configuration": {
            "model": MODEL_ID,
            "model_input": args.model,
            "revision": args.model_revision,
            "decoder": {
                "do_sample": False,
                "num_beams": 1,
                "max_new_tokens": args.max_new_tokens,
            },
            "dtype": str(model.dtype),
            "device": str(model.device),
            "seed": args.seed,
            "crop_prompt": CROP_PROMPT,
            "bbox_prompt_template": BBOX_PROMPT_PREFIX
            + "<loc_x1><loc_y1><loc_x2><loc_y2>",
            "bbox_grid": "Docling Core DocumentToken.get_location default 500 by 500",
            "text_extraction": "remove generated special tokens and DocTags, HTML-unescape, collapse whitespace",
            "normalization": "Unicode NFKC, casefold, collapsed whitespace",
            "inference_only": args.inference_only,
            "coordinate_reconstruction": (
                "exact native-crop template match in source page, then remove the fixed 4-pixel crop padding"
                if args.inference_only
                else None
            ),
        },
        "environment": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(model.device),
        },
        "panel": {
            "ground_truth": (
                str(ground_truth_path)
                if ground_truth_path.is_file()
                else str(annotation_root)
            ),
            "source_root": str(source_root),
            "fields": len(records),
        },
        "runs": {},
    }
    print("GRANITE_C14_CONFIG " + json.dumps(payload["configuration"], sort_keys=True))
    print("GRANITE_C14_ENV " + json.dumps(payload["environment"], sort_keys=True))
    print(f"GRANITE_C14_PANEL fields={len(records)} modes={','.join(args.modes)}")

    for mode in args.modes:
        warmup(
            record=records[0],
            mode=mode,
            run_root=args.run_root,
            source_root=source_root,
            processor=processor,
            model=model,
            torch=torch,
            max_new_tokens=args.max_new_tokens,
        )
        rows = run_mode(
            records=records,
            mode=mode,
            run_root=args.run_root,
            source_root=source_root,
            processor=processor,
            model=model,
            torch=torch,
            max_new_tokens=args.max_new_tokens,
            payload=payload,
            output_path=output_path,
        )
        payload["runs"][mode] = {"rows": rows}
        if not args.inference_only:
            payload["runs"][mode].update(
                {
                    "summary": score_rows(rows),
                    "review_sets": select_review_sets(rows, args.seed),
                }
            )
        write_json(output_path, payload)

    torch.cuda.synchronize(model.device)
    payload["memory_mib"] = {
        "peak_allocated": torch.cuda.max_memory_allocated(model.device) / 1024**2,
        "peak_reserved": torch.cuda.max_memory_reserved(model.device) / 1024**2,
        "resident_allocated": torch.cuda.memory_allocated(model.device) / 1024**2,
        "resident_reserved": torch.cuda.memory_reserved(model.device) / 1024**2,
    }
    write_json(output_path, payload)
    for mode, result in payload["runs"].items():
        if "summary" in result:
            print(
                "GRANITE_C14_SUMMARY "
                + json.dumps({"mode": mode, **result["summary"]}, sort_keys=True)
            )
    print("GRANITE_C14_MEMORY " + json.dumps(payload["memory_mib"], sort_keys=True))
    print(f"GRANITE_C14_OUTPUT {output_path}")


def load_model(
    model_name: str, revision: str | None, seed: int
) -> tuple[Any, Any, Any, Any]:
    import torch
    import transformers
    from transformers import AutoProcessor

    try:
        from transformers import AutoModelForVision2Seq as AutoModel
    except ImportError:
        from transformers import AutoModelForMultimodalLM as AutoModel

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.cuda.reset_peak_memory_stats()

    load_options: dict[str, Any] = {"local_files_only": True}
    if revision:
        load_options["revision"] = revision
    processor = AutoProcessor.from_pretrained(model_name, **load_options)
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        _attn_implementation="sdpa",
        **load_options,
    ).to("cuda")
    model.eval()
    return torch, transformers, processor, model


def load_ground_truth(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or len(raw) != 44:
        raise ValueError(
            f"expected 44 frozen fields, found {len(raw) if isinstance(raw, list) else 'invalid'}"
        )
    return load_ground_truth_records(raw)


def load_annotations(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(root.glob("*.json")):
        annotation = json.loads(path.read_text(encoding="utf-8"))
        case_id = annotation.get("case_id")
        handwriting = annotation.get("handwriting")
        if not isinstance(case_id, str) or not isinstance(handwriting, list):
            raise ValueError(f"invalid C14 annotation: {path}")
        for index, span in enumerate(handwriting, start=1):
            if not isinstance(span, dict):
                raise ValueError(f"invalid handwriting span: {path}")
            field_id = f"{case_id}-H{index:03d}"
            records.append(
                {
                    "case_id": case_id,
                    "field_id": field_id,
                    "reference": span.get("text"),
                    "bbox": span.get("bbox"),
                    "native": f"{field_id}-native.png",
                    "scaled": f"{field_id}-scaled.png",
                }
            )
    if len(records) != 44:
        raise ValueError(f"expected 44 frozen fields, found {len(records)}")
    return load_ground_truth_records(records)


def discover_inputs(run_root: Path, source_root: Path) -> list[dict[str, Any]]:
    import cv2

    records = []
    pages: dict[str, Any] = {}
    for crop_path in sorted((run_root / "crops").glob("*-native.png")):
        field_id = crop_path.name.removesuffix("-native.png")
        case_id = field_id.rsplit("-H", 1)[0]
        page = pages.setdefault(
            case_id,
            cv2.imread(str(source_root / f"{case_id}.png")),
        )
        crop = cv2.imread(str(crop_path))
        if page is None or crop is None:
            raise ValueError(f"cannot read images for {field_id}")
        minimum, _, location, _ = cv2.minMaxLoc(
            cv2.matchTemplate(page, crop, cv2.TM_SQDIFF_NORMED)
        )
        if minimum > 1e-5:
            raise ValueError(
                f"native crop does not exactly match source for {field_id}"
            )
        left, top = location
        height, width = crop.shape[:2]
        records.append(
            {
                "case_id": case_id,
                "field_id": field_id,
                "bbox": [left + 4, top + 4, left + width - 4, top + height - 4],
                "native": crop_path.name,
            }
        )
    if len(records) != 44:
        raise ValueError(f"expected 44 frozen crops, found {len(records)}")
    return records


def load_ground_truth_records(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    required = {"case_id", "field_id", "reference", "bbox", "native"}
    for record in raw:
        if not isinstance(record, dict) or not required <= record.keys():
            raise ValueError("invalid ground-truth record")
        if not isinstance(record["reference"], str):
            raise ValueError(f"invalid reference for {record['field_id']}")
        bbox = record["bbox"]
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"invalid bbox for {record['field_id']}")
    return raw


def warmup(
    *,
    record: dict[str, Any],
    mode: str,
    run_root: Path,
    source_root: Path,
    processor: Any,
    model: Any,
    torch: Any,
    max_new_tokens: int,
) -> None:
    image, prompt = prepare_input(record, mode, run_root, source_root)
    try:
        generate(image, prompt, processor, model, torch, max_new_tokens)
    finally:
        image.close()
    print(f"GRANITE_C14_WARM mode={mode}")


def run_mode(
    *,
    records: list[dict[str, Any]],
    mode: str,
    run_root: Path,
    source_root: Path,
    processor: Any,
    model: Any,
    torch: Any,
    max_new_tokens: int,
    payload: dict[str, Any],
    output_path: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        image, prompt = prepare_input(record, mode, run_root, source_root)
        started = time.perf_counter()
        error = None
        raw_output = ""
        generated_tokens = 0
        hit_token_limit = False
        try:
            raw_output, generated_tokens = generate(
                image,
                prompt,
                processor,
                model,
                torch,
                max_new_tokens,
            )
            hit_token_limit = generated_tokens >= max_new_tokens
            prediction = extract_text(raw_output)
        except Exception as exception:
            torch.cuda.synchronize(model.device)
            prediction = ""
            error = f"{type(exception).__name__}: {exception}"
        elapsed = time.perf_counter() - started
        image.close()
        inference = {
            "case_id": record["case_id"],
            "field_id": record["field_id"],
            "bbox": record["bbox"],
            "prediction": prediction,
            "raw_output": raw_output,
            "prompt": prompt,
            "error": error,
            "latency_seconds": elapsed,
            "generated_tokens": generated_tokens,
            "hit_token_limit": hit_token_limit,
        }
        row = (
            score_prediction(
                record=record,
                prediction=prediction,
                raw_output=raw_output,
                error=error,
                elapsed=elapsed,
                prompt=prompt,
                generated_tokens=generated_tokens,
                hit_token_limit=hit_token_limit,
            )
            if "reference" in record
            else inference
        )
        rows.append(row)
        payload["runs"][mode] = {"rows": rows, "status": "running"}
        write_json(output_path, payload)
        metrics = (
            f"exact={int(row['strict_exact'])} normalized_exact={int(row['normalized_exact'])} "
            f"char_edits={row['character_edits']['total']}"
            if "strict_exact" in row
            else f"output_chars={len(prediction)}"
        )
        print(
            "GRANITE_C14_CASE "
            f"mode={mode} index={index}/{len(records)} field={record['field_id']} "
            f"{metrics} latency={elapsed:.6f} failure={int(error is not None)}"
        )
    return rows


def prepare_input(
    record: dict[str, Any], mode: str, run_root: Path, source_root: Path
) -> tuple[Image.Image, str]:
    if mode == "crop":
        path = run_root / "crops" / record["native"]
        return Image.open(path).convert("RGB"), CROP_PROMPT
    if mode != "bbox":
        raise ValueError(f"unsupported mode: {mode}")

    path = source_root / f"{record['case_id']}.png"
    image = Image.open(path).convert("RGB")
    from docling_core.types.doc.tokens import DocumentToken

    location = DocumentToken.get_location(
        tuple(record["bbox"]),
        page_w=image.width,
        page_h=image.height,
    )
    return image, BBOX_PROMPT_PREFIX + location


def generate(
    image: Image.Image,
    prompt_text: str,
    processor: Any,
    model: Any,
    torch: Any,
    max_new_tokens: int,
) -> tuple[str, int]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(
        model.device
    )
    torch.cuda.synchronize(model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
        )
    torch.cuda.synchronize(model.device)
    prompt_tokens = inputs["input_ids"].shape[1]
    generated = output[0][prompt_tokens:]
    raw_output = processor.decode(generated, skip_special_tokens=False).lstrip()
    return raw_output, len(generated)


def extract_text(raw_output: str) -> str:
    without_special = SPECIAL_TOKEN.sub(" ", raw_output)
    without_tags = DOCTAG.sub(" ", without_special)
    return " ".join(html.unescape(without_tags).split())


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def edit_operations(
    reference: Sequence[Any], prediction: Sequence[Any]
) -> dict[str, int]:
    previous = [(index, 0, index, 0) for index in range(len(prediction) + 1)]
    for reference_index, reference_item in enumerate(reference, start=1):
        current = [(reference_index, 0, 0, reference_index)]
        for prediction_index, prediction_item in enumerate(prediction, start=1):
            if reference_item == prediction_item:
                current.append(previous[prediction_index - 1])
                continue
            substitution = add_operation(previous[prediction_index - 1], "substitution")
            insertion = add_operation(current[prediction_index - 1], "insertion")
            deletion = add_operation(previous[prediction_index], "deletion")
            current.append(
                min(insertion, substitution, deletion, key=lambda value: value[0])
            )
        previous = current
    total, substitutions, insertions, deletions = previous[-1]
    return {
        "total": total,
        "substitutions": substitutions,
        "insertions": insertions,
        "deletions": deletions,
    }


def add_operation(
    state: tuple[int, int, int, int], operation: str
) -> tuple[int, int, int, int]:
    total, substitutions, insertions, deletions = state
    return (
        total + 1,
        substitutions + int(operation == "substitution"),
        insertions + int(operation == "insertion"),
        deletions + int(operation == "deletion"),
    )


def score_prediction(
    *,
    record: dict[str, Any],
    prediction: str,
    raw_output: str,
    error: str | None,
    elapsed: float,
    prompt: str,
    generated_tokens: int,
    hit_token_limit: bool,
) -> dict[str, Any]:
    reference = record["reference"]
    normalized_reference = normalize(reference)
    normalized_prediction = normalize(prediction)
    character_edits = edit_operations(normalized_reference, normalized_prediction)
    word_edits = edit_operations(
        normalized_reference.split(), normalized_prediction.split()
    )
    is_critical = record["field_id"] not in NONCRITICAL_FIELDS
    normalized_exact = normalized_reference == normalized_prediction
    return {
        "case_id": record["case_id"],
        "field_id": record["field_id"],
        "bbox": record["bbox"],
        "reference": reference,
        "prediction": prediction,
        "raw_output": raw_output,
        "prompt": prompt,
        "strict_exact": reference == prediction,
        "normalized_exact": normalized_exact,
        "character_edits": character_edits,
        "word_edits": word_edits,
        "critical_field": is_critical,
        "critical_substitution": bool(
            is_critical and normalized_prediction and not normalized_exact
        ),
        "critical_miss": bool(is_critical and not normalized_prediction),
        "error": error,
        "latency_seconds": elapsed,
        "generated_tokens": generated_tokens,
        "hit_token_limit": hit_token_limit,
    }


def score_inference(
    input_path: Path,
    output_path: Path,
    records: list[dict[str, Any]],
    modes: list[str],
) -> None:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    references = {record["field_id"]: record for record in records}
    for mode, result in payload.get("runs", {}).items():
        if mode not in modes:
            result["status"] = "partial_not_scored"
            continue
        scored_rows = []
        for inference in result.get("rows", []):
            field_id = inference.get("field_id")
            if field_id not in references:
                raise ValueError(f"unexpected inference field: {field_id}")
            scored_rows.append(
                score_prediction(
                    record=references[field_id],
                    prediction=inference["prediction"],
                    raw_output=inference["raw_output"],
                    error=inference["error"],
                    elapsed=inference["latency_seconds"],
                    prompt=inference["prompt"],
                    generated_tokens=inference["generated_tokens"],
                    hit_token_limit=inference["hit_token_limit"],
                )
            )
        if len(scored_rows) != 44:
            raise ValueError(f"expected 44 {mode} rows, found {len(scored_rows)}")
        scored_ids = {row["field_id"] for row in scored_rows}
        if scored_ids != references.keys():
            raise ValueError(f"{mode} inference fields do not match ground truth")
        result["rows"] = scored_rows
        result["summary"] = score_rows(scored_rows)
        result["review_sets"] = select_review_sets(
            scored_rows, payload["configuration"]["seed"]
        )
    payload["configuration"]["inference_only"] = False
    payload["configuration"]["scored_modes"] = modes
    payload["panel"]["ground_truth"] = "local frozen ground_truth.json"
    write_json(output_path, payload)
    for mode, result in payload["runs"].items():
        if "summary" not in result:
            continue
        print(
            "GRANITE_C14_SUMMARY "
            + json.dumps({"mode": mode, **result["summary"]}, sort_keys=True)
        )
    print(f"GRANITE_C14_OUTPUT {output_path}")


def score_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reference_characters = sum(len(normalize(row["reference"])) for row in rows)
    reference_words = sum(len(normalize(row["reference"]).split()) for row in rows)
    character_totals = sum_edits(rows, "character_edits")
    word_totals = sum_edits(rows, "word_edits")
    latencies = [row["latency_seconds"] for row in rows]
    strict_exact = sum(row["strict_exact"] for row in rows)
    normalized_exact = sum(row["normalized_exact"] for row in rows)
    critical_fields = sum(row["critical_field"] for row in rows)
    return {
        "fields": len(rows),
        "failures": sum(row["error"] is not None for row in rows),
        "empty_predictions": sum(not normalize(row["prediction"]) for row in rows),
        "strict_exact": strict_exact,
        "strict_exact_rate": strict_exact / len(rows),
        "normalized_exact": normalized_exact,
        "normalized_exact_rate": normalized_exact / len(rows),
        "reference_characters": reference_characters,
        "reference_words": reference_words,
        "character_edits": character_totals,
        "word_edits": word_totals,
        "cer": character_totals["total"] / reference_characters,
        "wer": word_totals["total"] / reference_words,
        "missed_character_rate": character_totals["deletions"] / reference_characters,
        "hallucinated_character_rate": character_totals["insertions"]
        / reference_characters,
        "critical_fields": critical_fields,
        "critical_substitutions": sum(row["critical_substitution"] for row in rows),
        "critical_misses": sum(row["critical_miss"] for row in rows),
        "hit_token_limit": sum(row["hit_token_limit"] for row in rows),
        "warm_latency_seconds": {
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "max": max(latencies),
        },
    }


def sum_edits(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    names = ("total", "substitutions", "insertions", "deletions")
    return {name: sum(row[key][name] for row in rows) for name in names}


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("percentile needs at least one value")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def select_review_sets(rows: list[dict[str, Any]], seed: int) -> dict[str, list[str]]:
    worst = sorted(
        rows,
        key=lambda row: (
            row["character_edits"]["total"],
            row["word_edits"]["total"],
            row["field_id"],
        ),
        reverse=True,
    )[:10]
    worst_ids = {row["field_id"] for row in worst}
    remaining = [row for row in rows if row["field_id"] not in worst_ids]
    random_rows = random.Random(seed).sample(remaining, 10)
    return {
        "worst_10": [row["field_id"] for row in worst],
        "random_10_nonoverlapping": [row["field_id"] for row in random_rows],
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


if __name__ == "__main__":
    main()
