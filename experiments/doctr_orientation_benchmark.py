"""Benchmark docTR page orientation on a frozen ClinOCR split."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageOps

if __package__:
    from experiments.orientation_benchmark import (
        DEFAULT_RECTIFICATION_PADDING,
        rectify_document,
    )
    from experiments.public_benchmark import BenchmarkCase, discover_cases
else:
    from orientation_benchmark import DEFAULT_RECTIFICATION_PADDING, rectify_document
    from public_benchmark import BenchmarkCase, discover_cases

DEFAULT_ARCH = "mobilenet_v3_small_page_orientation"
SUPPORTED_ROTATIONS = {-90, 0, 90, 180}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark a dedicated docTR page-orientation classifier"
    )
    parser.add_argument("root", type=Path, help="Extracted ClinOCR dataset root")
    parser.add_argument("output", type=Path, help="JSON results path")
    parser.add_argument(
        "--clinocr-role",
        choices=("exemplar", "eval"),
        default="exemplar",
        help="Use exemplars while fixing policy; evaluate only after freezing it",
    )
    parser.add_argument("--subset", default="rotated")
    parser.add_argument("--batch-size", type=_positive_int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--arch", default=DEFAULT_ARCH)
    parser.add_argument(
        "--no-rectify",
        action="store_false",
        dest="rectify",
        help="Classify raw scans instead of the shared document rectification",
    )
    parser.add_argument(
        "--rectify-padding",
        type=float,
        default=DEFAULT_RECTIFICATION_PADDING,
    )
    parser.set_defaults(rectify=True)
    args = parser.parse_args(argv)

    try:
        if not 0 <= args.rectify_padding <= 0.25:
            raise ValueError("--rectify-padding must be between 0 and 0.25")
        payload = run_benchmark(
            args.root,
            clinocr_role=args.clinocr_role,
            subset=args.subset,
            batch_size=args.batch_size,
            device=args.device,
            arch=args.arch,
            rectify=args.rectify,
            rectify_padding=args.rectify_padding,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    return 0


def run_benchmark(
    root: Path,
    *,
    clinocr_role: str,
    subset: str,
    batch_size: int,
    device: str,
    arch: str = DEFAULT_ARCH,
    rectify: bool = True,
    rectify_padding: float = DEFAULT_RECTIFICATION_PADDING,
) -> dict[str, object]:
    cases = [
        case
        for case in discover_cases("clinocr", root, clinocr_role=clinocr_role)
        if case.subset == subset
    ]
    if not cases:
        raise ValueError(f"No ClinOCR {clinocr_role} cases found for subset {subset}")

    predictor, torch_module, doctr_version = _load_predictor(
        arch,
        batch_size,
        device,
    )
    warmup_image, _ = _prepare_image(cases[0], rectify, rectify_padding)
    predictor([warmup_image])
    _synchronize(torch_module, device)

    records = []
    batch_latencies = []
    for batch_index, batch in enumerate(_chunks(cases, batch_size)):
        prepared = [_prepare_image(case, rectify, rectify_padding) for case in batch]
        images = [image for image, _ in prepared]
        _synchronize(torch_module, device)
        started = time.perf_counter()
        prediction = predictor(images)
        _synchronize(torch_module, device)
        batch_latency_ms = (time.perf_counter() - started) * 1000
        batch_latencies.append(batch_latency_ms)
        records.extend(
            _prediction_records(
                root,
                batch,
                prepared,
                prediction,
                batch_index,
                batch_latency_ms,
            )
        )

    return {
        "experiment": "doctr_page_orientation",
        "dataset": "clinocr",
        "dataset_root": str(root.resolve()),
        "clinocr_role": clinocr_role,
        "subset": subset,
        "model": {
            "library": "python-doctr",
            "library_version": doctr_version,
            "architecture": arch,
            "pretrained": True,
            "device": device,
            "dtype": "float32",
        },
        "policy": {
            "rectify": rectify,
            "rectify_padding": rectify_padding if rectify else None,
            "batch_size": batch_size,
            "warmup_pages": 1,
            "angle_mapping": "docTR -90 maps to lossless rotation 270",
        },
        "summary": {
            "pages": len(records),
            "mean_inference_ms_per_page": round(sum(batch_latencies) / len(records), 3),
            "batch_latency_ms": _latency_summary(batch_latencies),
            "rectification_latency_ms": _latency_summary(
                [float(record["rectification_latency_ms"]) for record in records]
            ),
        },
        "cases": records,
    }


def _load_predictor(arch: str, batch_size: int, device: str):
    try:
        import doctr
        import torch
        from doctr.models import page_orientation_predictor
    except ImportError as error:
        raise RuntimeError(
            "python-doctr and a compatible PyTorch build are required"
        ) from error

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is unavailable for requested device {device}")
    predictor = page_orientation_predictor(
        arch=arch,
        pretrained=True,
        batch_size=batch_size,
    ).to(device)
    return predictor, torch, doctr.__version__


def _synchronize(torch_module, device: str) -> None:
    if device.startswith("cuda"):
        torch_module.cuda.synchronize()


def _prepare_image(
    case: BenchmarkCase,
    rectify: bool,
    rectify_padding: float,
) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    with Image.open(case.image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        if rectify:
            image = rectify_document(image, padding_fraction=rectify_padding)
        array = np.asarray(image)
    latency_ms = (time.perf_counter() - started) * 1000
    return array, latency_ms


def _prediction_records(
    root: Path,
    cases: list[BenchmarkCase],
    prepared: list[tuple[np.ndarray, float]],
    prediction: object,
    batch_index: int,
    batch_latency_ms: float,
) -> list[dict[str, object]]:
    if not isinstance(prediction, (list, tuple)) or len(prediction) != 3:
        raise ValueError("docTR returned an invalid orientation prediction")
    class_indices, rotations, confidences = prediction
    if not all(len(values) == len(cases) for values in prediction):
        raise ValueError("docTR returned a mismatched orientation batch")

    records = []
    per_page_latency = batch_latency_ms / len(cases)
    for case, (_, rectify_ms), class_index, rotation, confidence in zip(
        cases,
        prepared,
        class_indices,
        rotations,
        confidences,
        strict=True,
    ):
        rotation = int(rotation)
        confidence = float(confidence)
        if rotation not in SUPPORTED_ROTATIONS:
            raise ValueError(f"docTR returned unsupported rotation {rotation}")
        if not 0 <= confidence <= 1:
            raise ValueError(f"docTR returned invalid confidence {confidence}")
        records.append(
            {
                "id": case.id,
                "cluster_id": case.cluster_id,
                "subset": case.subset,
                "image": str(case.image_path.relative_to(root)),
                "class_index": int(class_index),
                "predicted_rotation": rotation,
                "lossless_rotation": rotation % 360,
                "confidence": confidence,
                "batch_index": batch_index,
                "allocated_inference_latency_ms": round(per_page_latency, 3),
                "rectification_latency_ms": round(rectify_ms, 3),
            }
        )
    return records


def _chunks(cases: list[BenchmarkCase], size: int) -> list[list[BenchmarkCase]]:
    return [cases[index : index + size] for index in range(0, len(cases), size)]


def _latency_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0}
    return {
        "mean": round(sum(values) / len(values), 3),
        "p50": round(_percentile(values, 0.5), 3),
        "p95": round(_percentile(values, 0.95), 3),
    }


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
