"""Measure whether DocRes appearance normalisation lifts reader confidence on degraded forms.

`controls.py` refuses to scan for marks around any label read below 0.9 confidence, so on
fax-generation scans a legible printed label silently disables mark detection. This measures
the gate directly: word confidence before and after DocRes appearance normalisation.

Two variants are compared against the original scan. `appearance_prompt` alone is the
training-free OpenCV step from DocRes and needs no checkpoint. `--docres-root` additionally
runs the real Restormer, which is the model the paper actually evaluates.

Usage:
    PYTHONPATH=src python experiments/benchmark_docres_appearance.py <image-dir> [--limit N]
    PYTHONPATH=src python experiments/benchmark_docres_appearance.py <image-dir> \
        --docres-root /path/to/DocRes --checkpoint /path/to/docres.pkl
"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
from pathlib import Path

import cv2
import numpy as np

from ocr_pipeline.providers import TesseractReader

# The confidence floor in controls.py:286 below which no mark slot is scanned.
LABEL_GATE = 0.9


def appearance_prompt(img):
    """Verbatim from DocRes (MIT), ZZZHANG-jx/DocRes, inference.py.

    Estimates a background plane by dilation plus median blur, then normalises the
    residual. Training-free, so it needs no checkpoint.
    """
    h, w = img.shape[:2]
    img = cv2.resize(img, (1024, 1024))
    rgb_planes = cv2.split(img)
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(
            diff_img,
            None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_8UC1,
        )
        result_norm_planes.append(norm_img)
    result_norm = cv2.merge(result_norm_planes)
    return cv2.resize(result_norm, (w, h))


class DocResAppearance:
    """The real Restormer appearance pass, using DocRes's own architecture and weights.

    Mirrors `appearance()` in DocRes inference.py, except it stays in float32 so it runs
    on CPU, and it takes an array rather than a path.
    """

    MAX_SIZE = 1600

    def __init__(self, docres_root: Path, checkpoint: Path) -> None:
        import sys

        import torch

        sys.path.insert(0, str(docres_root))
        from data.preprocess.crop_merge_image import stride_integral
        from models import restormer_arch
        from utils import convert_state_dict

        self._torch = torch
        self._stride_integral = stride_integral
        self._model = restormer_arch.Restormer(
            inp_channels=6,
            out_channels=3,
            dim=48,
            num_blocks=[2, 3, 3, 4],
            num_refinement_blocks=4,
            heads=[1, 2, 4, 8],
            ffn_expansion_factor=2.66,
            bias=False,
            LayerNorm_type="WithBias",
            dual_pixel_task=True,
        )
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self._model.load_state_dict(convert_state_dict(state["model_state"]))
        self._model.eval()

    def restore(self, image):
        height, width = image.shape[:2]
        prompt = appearance_prompt(image)
        combined = np.concatenate((image, prompt), -1)

        padding_h = padding_w = 0
        if max(width, height) < self.MAX_SIZE:
            combined, padding_h, padding_w = self._stride_integral(combined, 8)
        else:
            combined = cv2.resize(combined, (self.MAX_SIZE, self.MAX_SIZE))

        tensor = self._torch.from_numpy(
            (combined / 255.0).transpose(2, 0, 1)
        ).unsqueeze(0)
        with self._torch.no_grad():
            predicted = self._torch.clamp(self._model(tensor.float()), 0, 1)
        predicted = (predicted[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        if max(width, height) < self.MAX_SIZE:
            return predicted[padding_h:, padding_w:]
        # Divide out the estimated shadow map so the result keeps the source resolution.
        predicted[predicted == 0] = 1
        shadow = (
            cv2.resize(image, (self.MAX_SIZE, self.MAX_SIZE)).astype(float) / predicted
        )
        shadow = cv2.resize(shadow, (width, height))
        shadow[shadow == 0] = 0.00001
        return np.clip(image.astype(float) / shadow, 0, 255).astype(np.uint8)


def score(reader: TesseractReader, image_path: Path) -> dict[str, float]:
    regions = reader.read(image_path, 1)
    words = [region for region in regions if region.text.strip()]
    confidences = [
        region.confidence for region in words if region.confidence is not None
    ]
    labels = [
        region
        for region in words
        if region.text.rstrip().endswith(":") and region.confidence is not None
    ]
    return {
        "words": len(words),
        "mean_confidence": round(statistics.fmean(confidences), 4)
        if confidences
        else 0.0,
        "above_gate": sum(value >= LABEL_GATE for value in confidences),
        "labels": len(labels),
        "labels_above_gate": sum(region.confidence >= LABEL_GATE for region in labels),
    }


def compare(
    reader: TesseractReader,
    image_path: Path,
    work_dir: Path,
    restorer: DocResAppearance | None,
) -> dict[str, object]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"unreadable image: {image_path}")
    row: dict[str, object] = {
        "image": image_path.name,
        "original": score(reader, image_path),
    }
    variants = {"appearance": appearance_prompt(image)}
    if restorer is not None:
        variants["docres"] = restorer.restore(image)
    for name, restored in variants.items():
        restored_path = work_dir / f"{image_path.stem}-{name}.png"
        cv2.imwrite(str(restored_path), restored)
        row[name] = score(reader, restored_path)
    return row


def handle_report(results: list[dict[str, object]], variants: list[str]) -> None:
    for variant in variants:
        print(f"\n=== {variant} vs original ===")
        print(f"{'image':<38} {'words':>14} {'mean conf':>20} {'labels>=0.9':>16}")
        for row in results:
            before, after = row["original"], row[variant]
            print(
                f"{str(row['image'])[:37]:<38} "
                f"{before['words']:>5} -> {after['words']:<5} "
                f"{before['mean_confidence']:>9.3f} -> {after['mean_confidence']:<9.3f} "
                f"{before['labels_above_gate']:>6} -> {after['labels_above_gate']:<6}"
            )
        improved = sum(
            row[variant]["labels_above_gate"] > row["original"]["labels_above_gate"]
            for row in results
        )
        regressed = sum(
            row[variant]["labels_above_gate"] < row["original"]["labels_above_gate"]
            for row in results
        )
        gained = statistics.fmean(
            row[variant]["mean_confidence"] - row["original"]["mean_confidence"]
            for row in results
        )
        print(
            f"anchor labels above the gate: {improved} improved, {regressed} regressed; "
            f"mean confidence change {gained:+.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--docres-root", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()
    if bool(args.docres_root) != bool(args.checkpoint):
        raise SystemExit("--docres-root and --checkpoint must be given together")

    images = sorted(
        path
        for path in args.image_dir.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )[: args.limit]
    if not images:
        raise SystemExit(f"no images in {args.image_dir}")

    restorer = (
        DocResAppearance(args.docres_root, args.checkpoint)
        if args.docres_root
        else None
    )
    reader = TesseractReader()
    with tempfile.TemporaryDirectory(prefix="docres-appearance-") as work:
        results = [compare(reader, path, Path(work), restorer) for path in images]

    handle_report(results, ["appearance"] + (["docres"] if restorer else []))
    if args.output:
        args.output.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
