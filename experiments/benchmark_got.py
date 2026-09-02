"""Run a reject-only GOT-OCR2.0 triage on five frozen clinical pages."""

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
    from public_benchmark import (  # type: ignore[no-redef]
        NORMALIZATION,
        _score,
        _summarize,
    )


MODEL_ID = "stepfun-ai/GOT-OCR2_0"
MODEL_REVISION = "979938bf89ccdc949c0131ddd3841e24578a4742"
OFFICIAL_MAX_NEW_TOKENS = 4096
OFFICIAL_IMAGE_SIZE = 1024
OFFICIAL_MAX_CROPS = 6
METHODS = ("single", "multi_crop")
ORIENTATION_POLICIES = ("source", "sweep")
ROTATIONS = (0, 90, 180, 270)
TRIAGE_IDS = (
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run_benchmark(
            args.challenge_root,
            args.output,
            model_name=args.model,
            revision=args.revision,
            method=args.method,
            orientation_policy=args.orientation_policy,
            device=args.device,
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
        help="Exact checkpoint commit or an immutable local revision",
    )
    parser.add_argument("--method", choices=METHODS, default="multi_crop")
    parser.add_argument(
        "--orientation-policy",
        choices=ORIENTATION_POLICIES,
        default="sweep",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow downloading the pinned public checkpoint",
    )
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def run_benchmark(
    challenge_root: Path,
    output: Path,
    *,
    model_name: str = MODEL_ID,
    revision: str = MODEL_REVISION,
    method: str = "multi_crop",
    orientation_policy: str = "sweep",
    device: str = "cuda:0",
    local_files_only: bool = True,
    warmup: bool = True,
    seed: int = 0,
    tokenizer: Any | None = None,
    model: Any | None = None,
    torch_module: Any | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    _validate_options(
        challenge_root,
        output,
        model_name,
        revision,
        method,
        orientation_policy,
        tokenizer,
        model,
        torch_module,
    )
    cases = [_load_page_case(challenge_root, case_id) for case_id in TRIAGE_IDS]

    load_started = clock()
    if tokenizer is None:
        tokenizer, model, torch_module = load_model(
            model_name,
            revision,
            device=device,
            local_files_only=local_files_only,
            seed=seed,
        )
    model_load_ms = (clock() - load_started) * 1000
    _verify_loaded_revision(model_name, revision, model)
    payload = _initial_payload(
        challenge_root=challenge_root,
        model_name=model_name,
        revision=revision,
        method=method,
        orientation_policy=orientation_policy,
        device=device,
        local_files_only=local_files_only,
        seed=seed,
        tokenizer=tokenizer,
        model=model,
        torch_module=torch_module,
        model_load_ms=model_load_ms,
    )
    _write_payload(output, payload)

    payload["operations"]["warmup"] = (
        _warmup(
            cases[0],
            tokenizer=tokenizer,
            model=model,
            torch_module=torch_module,
            method=method,
            device=device,
            clock=clock,
        )
        if warmup
        else {"attempted": False}
    )
    _reset_peak_memory(torch_module, device)

    benchmark_started = clock()
    records = []
    for case in cases:
        record = _run_page(
            case,
            tokenizer=tokenizer,
            model=model,
            torch_module=torch_module,
            method=method,
            orientation_policy=orientation_policy,
            device=device,
            clock=clock,
        )
        records.append(record)
        payload["cases"] = records
        _write_payload(output, payload)

    payload["summary"] = _summarize(
        records,
        wall_latency_ms=sum(float(record["latency_ms"]) for record in records),
    )
    complete = [record for record in records if record["reference_scope"] == "complete"]
    payload["complete_reference_summary"] = _summarize(complete) if complete else None
    payload["reference_scope_counts"] = dict(
        sorted(Counter(record["reference_scope"] for record in records).items())
    )
    payload["handwriting_unlocalized_phrase_recovery"] = _aggregate_handwriting(records)
    payload["manual_review"] = [_manual_review_row(record) for record in records]
    payload["operations"]["benchmark_wall_ms"] = round(
        (clock() - benchmark_started) * 1000,
        3,
    )
    payload["operations"]["cuda_memory_mib"] = _aggregate_memory(
        records,
        torch_module,
        device,
    )
    payload["status"] = "complete"
    payload["evidence"] = {
        "scope": "reject_only",
        "promotion_evidence_complete": False,
        "failures_remain_in_denominators": True,
        "ground_truth_used_for_selection": False,
        "promotion_requirement": (
            "A non-rejected candidate still requires the frozen 22-page panel, "
            "44 native and 44 scaled C14 crops, public benchmarks, and manual review."
        ),
        "default_backend_eligible": False,
        "default_backend_exclusion": (
            "Research comparator only because the official architecture derives "
            "from Qwen2 and Vary."
        ),
    }
    _write_payload(output, payload)
    return payload


def load_model(
    model_name: str,
    revision: str,
    *,
    device: str,
    local_files_only: bool,
    seed: int,
) -> tuple[Any, Any, Any]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("the official GOT chat API requires CUDA")
    torch.cuda.set_device(device)
    options = {
        "revision": revision,
        "trust_remote_code": True,
        "local_files_only": local_files_only,
    }
    tokenizer = AutoTokenizer.from_pretrained(model_name, **options)
    model = AutoModel.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        pad_token_id=tokenizer.eos_token_id,
        **options,
    )
    model = model.eval().cuda()
    return tokenizer, model, torch


def _run_page(
    case: PageCase,
    *,
    tokenizer: Any,
    model: Any,
    torch_module: Any,
    method: str,
    orientation_policy: str,
    device: str,
    clock: Callable[[], float],
) -> dict[str, Any]:
    _reset_peak_memory(torch_module, device)
    started = clock()
    candidates = []
    failures = []
    try:
        with Image.open(case.source) as opened:
            source = opened.convert("RGB")
        try:
            rotations = ROTATIONS if orientation_policy == "sweep" else (0,)
            for rotation in rotations:
                candidate = _run_candidate(
                    source,
                    rotation,
                    tokenizer=tokenizer,
                    model=model,
                    torch_module=torch_module,
                    method=method,
                    device=device,
                    clock=clock,
                )
                candidates.append(candidate)
                if candidate["failure"]:
                    failures.append(candidate["failure"])
        finally:
            source.close()
    except Exception as error:
        failures.append(_failure("got_image_failed", error))

    selected = _select_candidate(candidates)
    prediction = str(selected["prediction"]) if selected else ""
    status = "success" if selected else "failed"
    record = {
        "id": case.id,
        "source_file": case.source_file,
        "status": status,
        "prediction": prediction,
        "reference": case.reference,
        "reference_scope": case.reference_scope,
        "unresolved_span_count": len(case.unresolved_spans),
        "challenges": list(case.challenges),
        "selection_reason": TRIAGE_REASONS[case.id],
        "method": method,
        "orientation_policy": orientation_policy,
        "selected_rotation_degrees_ccw": (
            selected["rotation_degrees_ccw"] if selected else None
        ),
        "orientation_candidates": candidates,
        "failures": failures,
        "latency_ms": round((clock() - started) * 1000, 3),
        "metrics": _score(prediction, case.reference),
        "strict_exact": prediction == case.reference,
        "handwriting_unlocalized_phrase_recovery": _handwriting_score(
            prediction,
            case.handwriting,
        ),
        "cuda_memory_mib": _cuda_memory(torch_module, device),
    }
    record["normalized_exact"] = record["metrics"]["cer"]["edits"] == 0
    return record


def _run_candidate(
    source: Image.Image,
    rotation: int,
    *,
    tokenizer: Any,
    model: Any,
    torch_module: Any,
    method: str,
    device: str,
    clock: Callable[[], float],
) -> dict[str, Any]:
    image = source.copy() if rotation == 0 else source.rotate(rotation, expand=True)
    started = clock()
    prediction = ""
    failure = None
    try:
        _synchronize(torch_module, device)
        with torch_module.inference_mode():
            chat = model.chat_crop if method == "multi_crop" else model.chat
            prediction = chat(
                tokenizer,
                image,
                ocr_type="ocr",
                render=False,
                save_render_file=None,
                print_prompt=False,
                gradio_input=True,
                stream_flag=False,
            )
        _synchronize(torch_module, device)
        prediction = str(prediction).strip()
        if not prediction:
            raise RuntimeError("GOT returned an empty OCR response")
    except Exception as error:
        _synchronize(torch_module, device)
        failure = _failure("got_inference_failed", error, rotation=rotation)
    finally:
        image.close()
    generated_tokens = _token_count(tokenizer, prediction)
    possible_token_limit = generated_tokens >= OFFICIAL_MAX_NEW_TOKENS - 1
    if failure is None and possible_token_limit:
        failure = _failure(
            "got_token_limit",
            RuntimeError("GOT reached the official generation limit"),
            rotation=rotation,
            generated_tokens=generated_tokens,
        )
    signal = _selection_signal(prediction)
    return {
        "rotation_degrees_ccw": rotation,
        "status": "failed" if failure else "success",
        "prediction": prediction,
        "generated_tokens_after_decode": generated_tokens,
        "possible_token_limit": possible_token_limit,
        "latency_ms": round((clock() - started) * 1000, 3),
        "selection_signal": signal,
        "image_input": _image_input(source.size, rotation, method),
        "failure": failure,
    }


def _warmup(
    case: PageCase,
    **runtime: Any,
) -> dict[str, Any]:
    try:
        with Image.open(case.source) as opened:
            source = opened.convert("RGB")
        try:
            record = _run_candidate(source, 0, **runtime)
        finally:
            source.close()
    except Exception as error:
        return {
            "attempted": True,
            "status": "failed",
            "failure": _failure("got_warmup_failed", error),
        }
    return {
        "attempted": True,
        "status": record["status"],
        "latency_ms": record["latency_ms"],
        "failure": record["failure"],
    }


def _select_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    successful = [
        candidate for candidate in candidates if candidate["status"] == "success"
    ]
    if not successful:
        return None
    return max(
        successful,
        key=lambda candidate: (
            candidate["selection_signal"]["score"],
            candidate["selection_signal"]["alphanumeric_characters"],
            -ROTATIONS.index(candidate["rotation_degrees_ccw"]),
        ),
    )


def _selection_signal(text: str) -> dict[str, int]:
    lines = [" ".join(line.casefold().split()) for line in text.splitlines()]
    lines = [line for line in lines if line]
    alphanumeric = sum(character.isalnum() for character in text)
    seen = set()
    repeated = 0
    for line in lines:
        count = sum(character.isalnum() for character in line)
        if line in seen:
            repeated += count
        else:
            seen.add(line)
    return {
        "score": alphanumeric - repeated,
        "alphanumeric_characters": alphanumeric,
        "repeated_line_alphanumeric_characters": repeated,
    }


def _image_input(
    source_size: tuple[int, int],
    rotation: int,
    method: str,
) -> dict[str, Any]:
    width, height = source_size
    if rotation in {90, 270}:
        width, height = height, width
    if method == "single":
        return {
            "rotated_size": [width, height],
            "processor_size": [OFFICIAL_IMAGE_SIZE, OFFICIAL_IMAGE_SIZE],
            "patches": 1,
            "thumbnail": False,
        }
    columns, rows = _multi_crop_grid(width, height)
    patch_count = columns * rows
    return {
        "rotated_size": [width, height],
        "processor_size": [OFFICIAL_IMAGE_SIZE, OFFICIAL_IMAGE_SIZE],
        "grid": [columns, rows],
        "patches": patch_count + (1 if patch_count != 1 else 0),
        "thumbnail": patch_count != 1,
    }


def _multi_crop_grid(width: int, height: int) -> tuple[int, int]:
    ratios = sorted(
        {
            (columns, rows)
            for count in range(1, OFFICIAL_MAX_CROPS + 1)
            for columns in range(1, count + 1)
            for rows in range(1, count + 1)
            if 1 <= columns * rows <= OFFICIAL_MAX_CROPS
        },
        key=lambda ratio: ratio[0] * ratio[1],
    )
    aspect_ratio = width / height
    best = ratios[0]
    best_difference = float("inf")
    area = width * height
    for ratio in ratios:
        difference = abs(aspect_ratio - ratio[0] / ratio[1])
        if difference < best_difference:
            best_difference = difference
            best = ratio
        elif difference == best_difference:
            target_area = OFFICIAL_IMAGE_SIZE**2 * ratio[0] * ratio[1]
            if area > 0.5 * target_area:
                best = ratio
    return best


def _load_page_case(challenge_root: Path, case_id: str) -> PageCase:
    category = case_id.split("-", 1)[0]
    source_dir = CATEGORY_SOURCES[category]
    annotation_path = (
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


def _handwriting_score(
    prediction: str,
    handwriting: tuple[tuple[str, str], ...],
) -> dict[str, Any]:
    normalized = _normalize_text(prediction)
    used: list[tuple[int, int]] = []
    rows = []
    excluded = 0
    for text, legibility in handwriting:
        eligible = legibility == "legible" and text.strip() and "[[" not in text
        if not eligible:
            excluded += 1
            continue
        span = _unused_phrase_span(normalized, _normalize_text(text), used)
        recovered = span is not None
        if span is not None:
            used.append(span)
        rows.append({"text": text, "recovered": recovered})
    return {
        "eligible": len(rows),
        "recovered": sum(row["recovered"] for row in rows),
        "excluded_partial_or_illegible": excluded,
        "spans": rows,
    }


def _aggregate_handwriting(records: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [record["handwriting_unlocalized_phrase_recovery"] for record in records]
    eligible = sum(row["eligible"] for row in rows)
    recovered = sum(row["recovered"] for row in rows)
    excluded = sum(row["excluded_partial_or_illegible"] for row in rows)
    return {
        "eligible": eligible,
        "recovered": recovered,
        "rate": round(recovered / eligible, 6) if eligible else None,
        "excluded_partial_or_illegible": excluded,
        "matching": "casefolded whitespace-normalized nonoverlapping exact spans",
        "interpretation": (
            "Upper-bound unlocalized phrase presence. GOT emits no geometry, so a "
            "phrase duplicated in printed text can count as present."
        ),
        "handwriting_specific": False,
    }


def _manual_review_row(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": record["id"],
        "source_file": record["source_file"],
        "selection_reason": record["selection_reason"],
        "challenges": record["challenges"],
        "status": record["status"],
        "reference_scope": record["reference_scope"],
        "selected_rotation_degrees_ccw": record["selected_rotation_degrees_ccw"],
        "reference": record["reference"],
        "prediction": record["prediction"],
        "cer": record["metrics"]["cer"]["rate"],
        "wer": record["metrics"]["wer"]["rate"],
        "missed_text_rate": record["metrics"]["missed_text_rate"]["rate"],
        "hallucinated_text_rate": record["metrics"]["hallucinated_text_rate"]["rate"],
        "handwriting_unlocalized_phrase_recovery": record[
            "handwriting_unlocalized_phrase_recovery"
        ],
        "orientation_candidates": record["orientation_candidates"],
        "failures": record["failures"],
    }


def _initial_payload(
    *,
    challenge_root: Path,
    model_name: str,
    revision: str,
    method: str,
    orientation_policy: str,
    device: str,
    local_files_only: bool,
    seed: int,
    tokenizer: Any,
    model: Any,
    torch_module: Any,
    model_load_ms: float,
) -> dict[str, Any]:
    config = getattr(model, "config", None)
    model_path = Path(model_name)
    return {
        "benchmark": "GOT-OCR2.0 frozen clinical OCR hard triage",
        "status": "running",
        "privacy": {
            "execution": "local_only",
            "private_uploads": False,
            "ground_truth_in_prompt": False,
        },
        "protocol": {
            "scope": "reject_only",
            "case_ids": list(TRIAGE_IDS),
            "cases": len(TRIAGE_IDS),
            "source_only": True,
            "normalization": NORMALIZATION,
            "failure_policy": "failed pages receive empty predictions",
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
            "tokenizer_class": type(tokenizer).__name__,
            "trust_remote_code": True,
            "research_only": True,
        },
        "generation": {
            "method": method,
            "ocr_type": "ocr",
            "prompt": "official checkpoint literal OCR prompt",
            "do_sample": False,
            "num_beams": 1,
            "stream": False,
            "official_max_new_tokens": OFFICIAL_MAX_NEW_TOKENS,
            "token_limit_source": "hard-coded by the official chat methods",
            "single_no_repeat_ngram_size": 20 if method == "single" else None,
            "multi_crop_max_patches_before_thumbnail": (
                OFFICIAL_MAX_CROPS if method == "multi_crop" else None
            ),
            "multi_crop_thumbnail": method == "multi_crop",
            "image_size": OFFICIAL_IMAGE_SIZE,
            "seed": seed,
        },
        "orientation": {
            "policy": orientation_policy,
            "rotations_degrees_ccw": (
                list(ROTATIONS) if orientation_policy == "sweep" else [0]
            ),
            "selection": (
                "maximum alphanumeric characters after repeated-line penalty; "
                "ties prefer 0, 90, 180, then 270 degrees"
                if orientation_policy == "sweep"
                else "source orientation"
            ),
            "all_candidates_retained": True,
            "ground_truth_used": False,
        },
        "runtime": {
            "device": device,
            "local_files_only": local_files_only,
            "torch": getattr(torch_module, "__version__", None),
            "cuda": getattr(getattr(torch_module, "version", None), "cuda", None),
            "gpu": _gpu_name(torch_module, device),
            "official_tested_stack": {
                "python": "3.10",
                "torch": "2.0.1",
                "torchvision": "0.15.2",
                "transformers": "4.37.2",
                "tiktoken": "0.6.0",
                "accelerate": "0.28.0",
            },
        },
        "sources": {
            "accessed": "2026-09-02",
            "repository": "https://github.com/Ucas-HaoranWei/GOT-OCR2.0",
            "model": "https://huggingface.co/stepfun-ai/GOT-OCR2_0",
            "paper": "https://arxiv.org/abs/2409.01704",
        },
        "operations": {
            "model_load_ms": round(model_load_ms, 3),
            "warmup": None,
            "benchmark_wall_ms": None,
            "cuda_memory_mib": None,
        },
        "challenge_root": str(challenge_root.resolve()),
        "summary": None,
        "complete_reference_summary": None,
        "reference_scope_counts": None,
        "handwriting_unlocalized_phrase_recovery": None,
        "manual_review": [],
        "cases": [],
        "evidence": None,
    }


def _token_count(tokenizer: Any, text: str) -> int:
    if not text:
        return 0
    encode = getattr(tokenizer, "encode", None)
    if callable(encode):
        return len(encode(text, add_special_tokens=False))
    encoded = tokenizer([text])
    input_ids = getattr(encoded, "input_ids", None)
    if not input_ids:
        raise ValueError("GOT tokenizer did not return input IDs")
    return len(input_ids[0])


def _failure(code: str, error: Exception, **details: Any) -> dict[str, Any]:
    return {
        "code": code,
        "type": type(error).__name__,
        "message": str(error)[:500],
        **details,
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
            torch_module.cuda.max_memory_allocated(device) / 1024**2,
            3,
        ),
        "peak_reserved": round(
            torch_module.cuda.max_memory_reserved(device) / 1024**2,
            3,
        ),
        "resident_allocated": round(
            torch_module.cuda.memory_allocated(device) / 1024**2,
            3,
        ),
        "resident_reserved": round(
            torch_module.cuda.memory_reserved(device) / 1024**2,
            3,
        ),
        "unit": "MiB",
    }


def _aggregate_memory(
    records: list[dict[str, Any]],
    torch_module: Any,
    device: str,
) -> dict[str, Any]:
    current = _cuda_memory(torch_module, device)
    available = [
        record["cuda_memory_mib"]
        for record in records
        if record["cuda_memory_mib"]["available"]
    ]
    if not available:
        return current
    return {
        **current,
        "peak_allocated": max(item["peak_allocated"] for item in available),
        "peak_reserved": max(item["peak_reserved"] for item in available),
        "scope": "maximum post-warmup per-page process peak",
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
    model_name: str,
    revision: str,
    method: str,
    orientation_policy: str,
    tokenizer: Any | None,
    model: Any | None,
    torch_module: Any | None,
) -> None:
    if not challenge_root.is_dir():
        raise FileNotFoundError(f"challenge root was not found: {challenge_root}")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if not revision.strip():
        raise ValueError("revision must be an exact non-empty revision")
    if not Path(model_name).exists() and not _is_commit_revision(revision):
        raise ValueError("remote model revision must be a full commit SHA")
    if method not in METHODS:
        raise ValueError(f"unsupported method: {method}")
    if orientation_policy not in ORIENTATION_POLICIES:
        raise ValueError(f"unsupported orientation policy: {orientation_policy}")
    supplied = (tokenizer is not None, model is not None, torch_module is not None)
    if any(supplied) and not all(supplied):
        raise ValueError("tokenizer, model, and torch_module must be supplied together")


def _verify_loaded_revision(model_name: str, revision: str, model: Any) -> None:
    if Path(model_name).exists():
        return
    loaded = getattr(getattr(model, "config", None), "_commit_hash", None)
    if not isinstance(loaded, str) or loaded.casefold() != revision.casefold():
        raise RuntimeError(
            f"loaded model commit {loaded!r} does not match requested revision"
        )


def _is_commit_revision(revision: str) -> bool:
    return len(revision) == 40 and all(
        character in "0123456789abcdefABCDEF" for character in revision
    )


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON input: {path}") from error


def _write_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
