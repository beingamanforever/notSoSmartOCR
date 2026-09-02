"""Benchmark frozen Phi-4 Multimodal OCR on the local C08/C14 panel."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import random
import time
from typing import Any

from PIL import Image

if __package__:
    from experiments.evaluate_challenge_set import (
        _normalize_text,
        _unused_phrase_span,
    )
    from experiments.public_benchmark import NORMALIZATION, _score, _summarize
else:
    from evaluate_challenge_set import (  # type: ignore[no-redef]
        _normalize_text,
        _unused_phrase_span,
    )
    from public_benchmark import NORMALIZATION, _score, _summarize  # type: ignore[no-redef]


MODEL_ID = "microsoft/Phi-4-multimodal-instruct"
MODEL_REVISION = "93f923e1a7727d1c4f446756212d9d3e8fcc5d81"
PAGE_PROMPT = (
    "Transcribe every visible character exactly in reading order. Preserve line "
    "breaks and punctuation, including handwriting, faint text, checkbox marks, "
    "table cells, headers, and footers. Do not explain, correct, summarize, or infer."
)
CROP_PROMPT = (
    "Transcribe every visible character exactly. Return only the literal text. "
    "Do not explain, correct, or infer."
)
PROMPT_VERSION = "literal-ocr-v1"
TRACKS = ("triage", "full")
DEFAULT_TRIAGE_IDS = (
    "C14-D001-P001",
    "C14-D002-P001",
    "C08-D003-P001",
    "C08-D007-P007",
    "C08-D017-P003",
)
TRIAGE_REASONS = {
    "C14-D001-P001": "faint handwriting, controls, and small tables",
    "C14-D002-P001": "handwriting, naked marks, and a rotated tiny footer",
    "C08-D003-P001": "dense form with 21 handwritten spans and tables",
    "C08-D007-P007": "tiny dense control and table grid",
    "C08-D017-P003": "180-degree rotated page with dense controls",
}
CATEGORY_SOURCES = {
    "C08": "C08-handwritten",
    "C14": "C14-user-reported",
}
EXPECTED_PAGE_COUNTS = {"C08": 20, "C14": 2}
EXPECTED_CROPS = 44
PROCESSOR_ATTRIBUTES = (
    "dynamic_hd",
    "hd_transform_order",
    "image_size",
    "max_num_crops",
    "min_num_crops",
    "num_crops",
)


@dataclass(frozen=True)
class PageCase:
    id: str
    source: Path
    source_file: str
    reference: str
    reference_scope: str
    challenges: tuple[str, ...]
    unresolved_spans: tuple[str, ...]
    handwriting: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class CropCase:
    id: str
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
            args.challenge_root,
            args.output,
            model_name=args.model,
            revision=args.revision,
            track=args.track,
            triage_ids=tuple(args.triage_case or DEFAULT_TRIAGE_IDS),
            max_new_tokens=args.max_new_tokens,
            device=args.device,
            dtype=args.dtype,
            attention=args.attention,
            local_files_only=not args.allow_download,
            warmup=not args.skip_warmup,
            seed=args.seed,
        )
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
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
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument(
        "--revision",
        default=MODEL_REVISION,
        help="Exact model commit or immutable local revision",
    )
    parser.add_argument("--track", choices=TRACKS, default="triage")
    parser.add_argument(
        "--triage-case",
        action="append",
        help="Exactly five case IDs; repeat once per case",
    )
    parser.add_argument("--max-new-tokens", type=_positive_int, default=4096)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument(
        "--attention",
        choices=("sdpa", "flash_attention_2"),
        default="sdpa",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow downloading the pinned public model revision",
    )
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def run_benchmark(
    challenge_root: Path,
    output: Path,
    *,
    model_name: str,
    revision: str,
    track: str,
    triage_ids: tuple[str, ...] = DEFAULT_TRIAGE_IDS,
    max_new_tokens: int = 4096,
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    attention: str = "sdpa",
    local_files_only: bool = True,
    warmup: bool = True,
    seed: int = 0,
    processor: Any | None = None,
    model: Any | None = None,
    generation_config: Any | None = None,
    torch_module: Any | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    _validate_options(
        challenge_root,
        output,
        revision,
        track,
        triage_ids,
        max_new_tokens,
        dtype,
    )
    pages = load_page_cases(challenge_root, track, triage_ids)
    crops = load_crop_cases(challenge_root) if track == "full" else []

    load_started = clock()
    if processor is None or model is None or torch_module is None:
        processor, model, generation_config, torch_module = load_model(
            model_name,
            revision,
            device=device,
            dtype=dtype,
            attention=attention,
            local_files_only=local_files_only,
            seed=seed,
        )
    model_load_ms = (clock() - load_started) * 1000
    payload = _initial_payload(
        challenge_root=challenge_root,
        model_name=model_name,
        revision=revision,
        track=track,
        triage_ids=triage_ids,
        max_new_tokens=max_new_tokens,
        device=device,
        dtype=dtype,
        attention=attention,
        local_files_only=local_files_only,
        seed=seed,
        model=model,
        processor=processor,
        torch_module=torch_module,
        model_load_ms=model_load_ms,
    )
    _write_payload(output, payload)

    warmup_record = None
    if warmup:
        warmup_record = _run_input(
            id=pages[0].id,
            image_path=pages[0].source,
            source_file=pages[0].source_file,
            prompt=PAGE_PROMPT,
            reference=None,
            processor=processor,
            model=model,
            generation_config=generation_config,
            torch_module=torch_module,
            device=device,
            max_new_tokens=max_new_tokens,
            clock=clock,
        )
    payload["operations"]["warmup"] = _warmup_summary(warmup_record)
    _reset_peak_memory(torch_module, device)

    benchmark_started = clock()
    page_records = _run_pages(
        pages,
        payload,
        output,
        processor=processor,
        model=model,
        generation_config=generation_config,
        torch_module=torch_module,
        device=device,
        max_new_tokens=max_new_tokens,
        clock=clock,
    )
    _finish_track(payload, "full_pages", page_records, seed, all_review=True)
    _write_payload(output, payload)

    if track == "full":
        for variant in ("native", "scaled"):
            crop_records = _run_crops(
                crops,
                variant,
                payload,
                output,
                processor=processor,
                model=model,
                generation_config=generation_config,
                torch_module=torch_module,
                device=device,
                max_new_tokens=max_new_tokens,
                clock=clock,
            )
            _finish_track(
                payload,
                f"c14_crops_{variant}",
                crop_records,
                seed,
                all_review=False,
            )
            _write_payload(output, payload)

    payload["operations"]["benchmark_wall_ms"] = round(
        (clock() - benchmark_started) * 1000, 3
    )
    payload["operations"]["cuda_memory_mib"] = _cuda_memory(torch_module, device)
    payload["status"] = "complete"
    payload["evidence"] = {
        "scope": "reject_only" if track == "triage" else "promotion_evaluation",
        "promotion_evidence_complete": track == "full",
        "promotion_requirement": (
            "Run the full frozen 22-page panel and both 44-crop arms."
        ),
        "failures_remain_in_denominators": True,
    }
    _write_payload(output, payload)
    return payload


def load_page_cases(
    challenge_root: Path,
    track: str,
    triage_ids: tuple[str, ...],
) -> list[PageCase]:
    if track == "triage":
        return [_load_page_case(challenge_root, case_id) for case_id in triage_ids]

    pages: list[PageCase] = []
    for category in ("C14", "C08"):
        annotation_root = challenge_root / "annotations" / "primary" / category
        paths = sorted(annotation_root.glob("*.json"))
        expected = EXPECTED_PAGE_COUNTS[category]
        if len(paths) != expected:
            raise ValueError(
                f"expected {expected} frozen {category} pages, found {len(paths)}"
            )
        pages.extend(
            _load_page_case(challenge_root, path.stem, annotation_path=path)
            for path in paths
        )
    return pages


def load_crop_cases(challenge_root: Path) -> list[CropCase]:
    run_root = challenge_root / "runs" / "ministral-c14-crops-source-only"
    ground_truth_path = run_root / "ground_truth.json"
    records = _read_json(ground_truth_path)
    if not isinstance(records, list) or len(records) != EXPECTED_CROPS:
        count = len(records) if isinstance(records, list) else "invalid"
        raise ValueError(f"expected {EXPECTED_CROPS} frozen C14 crops, found {count}")

    crops = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("invalid C14 crop record")
        required = {"case_id", "field_id", "bbox", "reference", "native", "scaled"}
        if not required <= record.keys():
            raise ValueError("invalid C14 crop record")
        bbox = record["bbox"]
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"invalid crop bbox for {record['field_id']}")
        native = run_root / "crops" / str(record["native"])
        scaled = run_root / "crops" / str(record["scaled"])
        for image_path in (native, scaled):
            if not image_path.is_file():
                raise FileNotFoundError(f"frozen crop was not found: {image_path}")
        crops.append(
            CropCase(
                id=str(record["field_id"]),
                case_id=str(record["case_id"]),
                bbox=tuple(int(value) for value in bbox),
                reference=str(record["reference"]),
                native=native,
                native_file=native.relative_to(challenge_root).as_posix(),
                scaled=scaled,
                scaled_file=scaled.relative_to(challenge_root).as_posix(),
            )
        )
    return crops


def load_model(
    model_name: str,
    revision: str,
    *,
    device: str,
    dtype: str,
    attention: str,
    local_files_only: bool,
    seed: int,
) -> tuple[Any, Any, Any, Any]:
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoProcessor,
        GenerationConfig,
    )

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    options = {
        "revision": revision,
        "trust_remote_code": True,
        "local_files_only": local_files_only,
    }
    processor = AutoProcessor.from_pretrained(model_name, **options)
    generation_config = GenerationConfig.from_pretrained(model_name, **options)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=getattr(torch, dtype),
        _attn_implementation=attention,
        **options,
    ).to(device)
    model.eval()
    return processor, model, generation_config, torch


def _run_pages(
    cases: list[PageCase],
    payload: dict[str, Any],
    output: Path,
    **runtime: Any,
) -> list[dict[str, Any]]:
    records = []
    payload["tracks"]["full_pages"] = {"status": "running", "cases": records}
    for case in cases:
        record = _run_input(
            id=case.id,
            image_path=case.source,
            source_file=case.source_file,
            prompt=PAGE_PROMPT,
            reference=case.reference,
            **runtime,
        )
        record.update(
            {
                "reference_scope": case.reference_scope,
                "unresolved_span_count": len(case.unresolved_spans),
                "challenges": list(case.challenges),
                "selection_reason": TRIAGE_REASONS.get(case.id),
                "handwriting": _handwriting_score(
                    record["prediction"], case.handwriting
                ),
            }
        )
        records.append(record)
        _write_payload(output, payload)
    return records


def _run_crops(
    cases: list[CropCase],
    variant: str,
    payload: dict[str, Any],
    output: Path,
    **runtime: Any,
) -> list[dict[str, Any]]:
    track_name = f"c14_crops_{variant}"
    records = []
    payload["tracks"][track_name] = {"status": "running", "cases": records}
    for case in cases:
        image_path = getattr(case, variant)
        source_file = getattr(case, f"{variant}_file")
        record = _run_input(
            id=case.id,
            image_path=image_path,
            source_file=source_file,
            prompt=CROP_PROMPT,
            reference=case.reference,
            **runtime,
        )
        record.update(
            {
                "case_id": case.case_id,
                "bbox": list(case.bbox),
                "variant": variant,
                "reference_scope": "complete",
            }
        )
        records.append(record)
        _write_payload(output, payload)
    return records


def _run_input(
    *,
    id: str,
    image_path: Path,
    source_file: str,
    prompt: str,
    reference: str | None,
    processor: Any,
    model: Any,
    generation_config: Any,
    torch_module: Any,
    device: str,
    max_new_tokens: int,
    clock: Callable[[], float],
) -> dict[str, Any]:
    started = clock()
    prediction = ""
    generated_tokens = 0
    hit_token_limit = False
    metadata: dict[str, Any] = {}
    failures = []
    try:
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        try:
            prediction, generated_tokens, metadata = generate_text(
                image,
                prompt,
                processor=processor,
                model=model,
                generation_config=generation_config,
                torch_module=torch_module,
                device=device,
                max_new_tokens=max_new_tokens,
            )
        finally:
            image.close()
        hit_token_limit = generated_tokens >= max_new_tokens
        if hit_token_limit:
            status = "failed"
            failures.append(
                {
                    "code": "phi4_token_limit",
                    "type": "GenerationLimit",
                    "message": (f"generation reached the {max_new_tokens}-token limit"),
                }
            )
        else:
            status = "success"
    except Exception as error:
        _synchronize(torch_module, device)
        status = "failed"
        failures.append(
            {
                "code": "phi4_inference_failed",
                "type": type(error).__name__,
                "message": str(error)[:500],
            }
        )
    latency_ms = round((clock() - started) * 1000, 3)
    record: dict[str, Any] = {
        "id": id,
        "source_file": source_file,
        "status": status,
        "prediction": prediction,
        "failures": failures,
        "latency_ms": latency_ms,
        "generated_tokens": generated_tokens,
        "hit_token_limit": hit_token_limit,
        "processor_metadata": metadata,
    }
    if reference is not None:
        record["reference"] = reference
        record["metrics"] = _score(prediction, reference)
        record["strict_exact"] = prediction == reference
        record["normalized_exact"] = record["metrics"]["cer"]["edits"] == 0
    return record


def generate_text(
    image: Image.Image,
    prompt_text: str,
    *,
    processor: Any,
    model: Any,
    generation_config: Any,
    torch_module: Any,
    device: str,
    max_new_tokens: int,
) -> tuple[str, int, dict[str, Any]]:
    prompt = f"<|user|><|image_1|>{prompt_text}<|end|><|assistant|>"
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    metadata = _processor_metadata(processor, inputs)
    inputs = inputs.to(device)
    input_ids = inputs.get("input_ids")
    if input_ids is None or not hasattr(input_ids, "shape"):
        raise ValueError("Phi-4 processor did not return input_ids")
    prompt_tokens = int(input_ids.shape[-1])
    _synchronize(torch_module, device)
    with torch_module.inference_mode():
        output = model.generate(
            **inputs,
            generation_config=generation_config,
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
    return prediction, generated_tokens, metadata


def _load_page_case(
    challenge_root: Path,
    case_id: str,
    *,
    annotation_path: Path | None = None,
) -> PageCase:
    category = case_id.split("-", 1)[0]
    source_dir = CATEGORY_SOURCES.get(category)
    if source_dir is None:
        raise ValueError(f"unsupported frozen category: {category}")
    annotation_path = annotation_path or (
        challenge_root / "annotations" / "primary" / category / f"{case_id}.json"
    )
    annotation = _read_json(annotation_path)
    if not isinstance(annotation, dict) or annotation.get("case_id") != case_id:
        raise ValueError(f"invalid annotation for {case_id}")
    if annotation.get("source_only") is not True:
        raise ValueError(f"annotation is not source-only: {case_id}")
    transcription = annotation.get("transcription")
    if not isinstance(transcription, dict):
        raise ValueError(f"missing transcription for {case_id}")
    reference = transcription.get("reading_order_text")
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError(f"missing reading-order text for {case_id}")
    unresolved = transcription.get("unresolved_spans") or []
    challenges = annotation.get("challenges") or []
    handwriting = annotation.get("handwriting") or []
    if not all(
        isinstance(value, list) for value in (unresolved, challenges, handwriting)
    ):
        raise ValueError(f"invalid annotation lists for {case_id}")
    source = challenge_root / "sources" / source_dir / f"{case_id}.png"
    if not source.is_file():
        raise FileNotFoundError(f"frozen source was not found: {source}")
    scope = (
        "complete"
        if annotation.get("page_legibility") == "complete" and not unresolved
        else "partial"
    )
    return PageCase(
        id=case_id,
        source=source,
        source_file=source.relative_to(challenge_root).as_posix(),
        reference=reference,
        reference_scope=scope,
        challenges=tuple(str(value) for value in challenges),
        unresolved_spans=tuple(str(value) for value in unresolved),
        handwriting=tuple(
            (str(item.get("text", "")), str(item.get("legibility", "unknown")))
            for item in handwriting
            if isinstance(item, dict)
        ),
    )


def _initial_payload(
    *,
    challenge_root: Path,
    model_name: str,
    revision: str,
    track: str,
    triage_ids: tuple[str, ...],
    max_new_tokens: int,
    device: str,
    dtype: str,
    attention: str,
    local_files_only: bool,
    seed: int,
    model: Any,
    processor: Any,
    torch_module: Any,
    model_load_ms: float,
) -> dict[str, Any]:
    config = getattr(model, "config", None)
    model_path = Path(model_name)
    return {
        "benchmark": "Phi-4 Multimodal frozen clinical OCR candidate",
        "status": "running",
        "privacy": {
            "execution": "local_only",
            "private_uploads": False,
            "ground_truth_in_prompt": False,
        },
        "protocol": {
            "track": track,
            "triage_case_ids": list(triage_ids) if track == "triage" else None,
            "page_panel": "C14 2 pages plus C08 20 pages"
            if track == "full"
            else "five-page hard triage",
            "crop_panel": "C14 44 native plus 44 scaled crops"
            if track == "full"
            else None,
            "normalization": NORMALIZATION,
            "source_only": True,
        },
        "model": {
            "requested": model_name,
            "requested_revision": revision,
            "local_path": str(model_path.resolve()) if model_path.exists() else None,
            "loaded_name_or_path": getattr(config, "_name_or_path", None),
            "loaded_commit_hash": getattr(config, "_commit_hash", None),
            "model_type": getattr(config, "model_type", None),
            "architectures": getattr(config, "architectures", None),
            "class": type(model).__name__,
            "processor_class": type(processor).__name__,
        },
        "generation": {
            "prompt_version": PROMPT_VERSION,
            "page_prompt": PAGE_PROMPT,
            "crop_prompt": CROP_PROMPT,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "num_beams": 1,
            "seed": seed,
        },
        "runtime": {
            "device": device,
            "dtype": dtype,
            "attention": attention,
            "local_files_only": local_files_only,
            "torch": getattr(torch_module, "__version__", None),
            "cuda": getattr(getattr(torch_module, "version", None), "cuda", None),
            "gpu": _gpu_name(torch_module, device),
        },
        "operations": {
            "model_load_ms": round(model_load_ms, 3),
            "warmup": None,
            "benchmark_wall_ms": None,
            "cuda_memory_mib": None,
        },
        "tracks": {},
        "evidence": None,
        "challenge_root": str(challenge_root.resolve()),
    }


def _finish_track(
    payload: dict[str, Any],
    name: str,
    records: list[dict[str, Any]],
    seed: int,
    *,
    all_review: bool,
) -> None:
    wall_ms = sum(float(record["latency_ms"]) for record in records)
    complete = [
        record for record in records if record.get("reference_scope") == "complete"
    ]
    payload["tracks"][name] = {
        "status": "complete",
        "summary": _summarize(records, wall_latency_ms=wall_ms),
        "complete_reference_summary": (_summarize(complete) if complete else None),
        "reference_scope_counts": dict(
            sorted(Counter(record["reference_scope"] for record in records).items())
        ),
        "exact_match": {
            "strict": sum(record["strict_exact"] for record in records),
            "normalized": sum(record["normalized_exact"] for record in records),
            "attempted": len(records),
        },
        "handwriting_exact_recovery": _aggregate_handwriting(records),
        "manual_review": _manual_review(records, seed, all_rows=all_review),
        "cases": records,
    }


def _manual_review(
    records: list[dict[str, Any]], seed: int, *, all_rows: bool
) -> list[dict[str, Any]]:
    if all_rows or len(records) <= 20:
        selected = [("all", record) for record in records]
    else:
        worst = sorted(
            records,
            key=lambda record: (
                float(record["metrics"]["cer"]["rate"]),
                record["id"],
            ),
            reverse=True,
        )[:10]
        worst_ids = {record["id"] for record in worst}
        remaining = [record for record in records if record["id"] not in worst_ids]
        random_rows = random.Random(seed).sample(remaining, min(10, len(remaining)))
        selected = [
            *(("worst", record) for record in worst),
            *(("random", record) for record in random_rows),
        ]
    return [
        {
            "selection": selection,
            "id": record["id"],
            "source_file": record["source_file"],
            "status": record["status"],
            "reference": record["reference"],
            "prediction": record["prediction"],
            "cer": record["metrics"]["cer"]["rate"],
            "latency_ms": record["latency_ms"],
            "failures": record["failures"],
        }
        for selection, record in selected
    ]


def _handwriting_score(
    prediction: str, handwriting: tuple[tuple[str, str], ...]
) -> dict[str, Any]:
    normalized = _normalize_text(prediction)
    used: list[tuple[int, int]] = []
    rows = []
    for text, legibility in handwriting:
        if legibility not in {"legible", "partial"} or not text.strip():
            continue
        span = _unused_phrase_span(normalized, _normalize_text(text), used)
        recovered = span is not None
        if span is not None:
            used.append(span)
        rows.append(
            {
                "text": text,
                "legibility": legibility,
                "recovered": recovered,
            }
        )
    return {
        "eligible": len(rows),
        "recovered": sum(row["recovered"] for row in rows),
        "spans": rows,
    }


def _aggregate_handwriting(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    scored = [record["handwriting"] for record in records if "handwriting" in record]
    if not scored:
        return None
    eligible = sum(item["eligible"] for item in scored)
    recovered = sum(item["recovered"] for item in scored)
    return {
        "eligible": eligible,
        "recovered": recovered,
        "rate": round(recovered / eligible, 6) if eligible else None,
        "matching": "casefolded whitespace-normalized nonoverlapping exact spans",
    }


def _processor_metadata(processor: Any, inputs: Any) -> dict[str, Any]:
    input_shapes = {}
    crop_tile_fields = {}
    for key, value in inputs.items():
        shape = _shape(value)
        if shape is not None:
            input_shapes[str(key)] = {
                "shape": shape,
                "dtype": str(getattr(value, "dtype", "unknown")),
            }
        if any(token in str(key).casefold() for token in ("crop", "tile", "image")):
            crop_tile_fields[str(key)] = {
                "shape": shape,
                "values": _small_values(value),
            }
    image_processor = getattr(processor, "image_processor", None)
    attributes = {}
    for name in PROCESSOR_ATTRIBUTES:
        value = getattr(image_processor, name, None)
        safe = _json_scalar(value)
        if safe is not None:
            attributes[name] = safe
    return {
        "input_shapes": input_shapes,
        "crop_tile_fields": crop_tile_fields,
        "image_processor": attributes,
    }


def _small_values(value: Any) -> list[Any] | None:
    try:
        values = value.detach().cpu().reshape(-1).tolist()
    except AttributeError:
        if not isinstance(value, (list, tuple)):
            return None
        values = list(value)
    if len(values) > 32:
        return None
    safe = [_json_scalar(item) for item in values]
    return safe if all(item is not None for item in safe) else None


def _json_scalar(value: Any) -> Any | None:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)) and len(value) <= 8:
        converted = [_json_scalar(item) for item in value]
        return converted if all(item is not None for item in converted) else None
    return None


def _shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return [int(size) for size in shape]


def _warmup_summary(record: dict[str, Any] | None) -> dict[str, Any]:
    if record is None:
        return {"attempted": False}
    return {
        "attempted": True,
        "status": record["status"],
        "latency_ms": record["latency_ms"],
        "failures": record["failures"],
    }


def _reset_peak_memory(torch_module: Any, device: str) -> None:
    if _cuda_available(torch_module, device):
        _synchronize(torch_module, device)
        torch_module.cuda.reset_peak_memory_stats(device)


def _cuda_memory(torch_module: Any, device: str) -> dict[str, Any]:
    if not _cuda_available(torch_module, device):
        return {
            "available": False,
            "peak_allocated": None,
            "peak_reserved": None,
            "resident_allocated": None,
            "resident_reserved": None,
        }
    _synchronize(torch_module, device)
    return {
        "available": True,
        "peak_allocated": round(
            torch_module.cuda.max_memory_allocated(device) / 1024**2, 3
        ),
        "peak_reserved": round(
            torch_module.cuda.max_memory_reserved(device) / 1024**2, 3
        ),
        "resident_allocated": round(
            torch_module.cuda.memory_allocated(device) / 1024**2, 3
        ),
        "resident_reserved": round(
            torch_module.cuda.memory_reserved(device) / 1024**2, 3
        ),
        "unit": "MiB",
        "scope": "post-warmup benchmark process",
    }


def _gpu_name(torch_module: Any, device: str) -> str | None:
    if not _cuda_available(torch_module, device):
        return None
    return str(torch_module.cuda.get_device_name(device))


def _synchronize(torch_module: Any, device: str) -> None:
    if _cuda_available(torch_module, device):
        torch_module.cuda.synchronize(device)


def _cuda_available(torch_module: Any, device: str) -> bool:
    cuda = getattr(torch_module, "cuda", None)
    return bool(
        device.startswith("cuda")
        and cuda is not None
        and callable(getattr(cuda, "is_available", None))
        and cuda.is_available()
    )


def _validate_options(
    challenge_root: Path,
    output: Path,
    revision: str,
    track: str,
    triage_ids: tuple[str, ...],
    max_new_tokens: int,
    dtype: str,
) -> None:
    if not challenge_root.is_dir():
        raise FileNotFoundError(f"challenge root was not found: {challenge_root}")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if not revision.strip():
        raise ValueError("revision must be an exact non-empty revision")
    if track not in TRACKS:
        raise ValueError(f"unsupported track: {track}")
    if track == "triage" and (len(triage_ids) != 5 or len(set(triage_ids)) != 5):
        raise ValueError("triage requires exactly five unique case IDs")
    if max_new_tokens <= 0:
        raise ValueError("max-new-tokens must be positive")
    if dtype not in {"bfloat16", "float16"}:
        raise ValueError(f"unsupported dtype: {dtype}")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON input: {path}") from error


def _write_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
