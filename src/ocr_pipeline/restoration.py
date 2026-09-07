"""DocRes appearance restoration for degraded scans.

Fax-generation scans lose printed labels below the confidence floor that mark detection
depends on, and lose whole regions outright. DocRes restores the page; `RestoredViewReader`
in preprocessing.py reads that view alongside the original and fuses the two.

The Restormer architecture and the appearance prompt come from the DocRes repository
(MIT, https://github.com/ZZZHANG-jx/DocRes) rather than being reimplemented, so a checkout
plus `docres.pkl` is required. Restoration is skipped when the checkout is absent.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any

from .providers import ReaderError

PROVIDER = "docres-appearance"
MODEL = {
    "id": "docres-restormer-appearance-v1",
    "origin": "ZZZHANG-jx, DocRes (CVPR 2024)",
    "license": "MIT",
}
# DocRes resizes above this edge length, then divides the shadow map back out. A clinical
# scan is around 3400x4400, so that resize squashes the aspect ratio and throws away the
# stroke detail the recognizers need, which is why large pages are tiled instead.
MAX_SIZE = 1600
# PreP-OCR (arXiv:2505.20429) restores large pages in overlapping patches and discards the
# outer border of each, so no output pixel comes from a patch edge and no seam appears.
TILE_SIZE = 1024
TILE_BORDER = 128
RESTORMER_SETTINGS = {
    "inp_channels": 6,
    "out_channels": 3,
    "dim": 48,
    "num_blocks": [2, 3, 3, 4],
    "num_refinement_blocks": 4,
    "heads": [1, 2, 4, 8],
    "ffn_expansion_factor": 2.66,
    "bias": False,
    "LayerNorm_type": "WithBias",
    "dual_pixel_task": True,
}


class DocResRestorer:
    """Restore a page image with the DocRes appearance task.

    The model is loaded on first use so that constructing the stage stays cheap, and the
    checkpoint is read with `weights_only` so loading cannot execute arbitrary code.
    """

    name = PROVIDER

    def __init__(self, checkpoint: Path, source_root: Path) -> None:
        self.checkpoint = Path(checkpoint)
        self.source_root = Path(source_root)
        self._model: Any = None
        self._lock = threading.Lock()

    def available(self) -> bool:
        return self.checkpoint.is_file() and (self.source_root / "models").is_dir()

    def restore(self, image_path: Path, destination: Path) -> Path:
        cv2, np = _load_vision()
        image = cv2.imread(str(image_path))
        if image is None:
            raise ReaderError("restoration_image_failed", f"unreadable: {image_path}")
        restored = self._restore_array(image, cv2, np)
        if not cv2.imwrite(str(destination), restored):
            raise ReaderError("restoration_write_failed", f"unwritable: {destination}")
        return destination

    def _restore_array(self, image: Any, cv2: Any, np: Any) -> Any:
        height, width = image.shape[:2]
        # The background estimate stays global: computing it per tile would make each tile
        # normalise against its own local background and reintroduce seams.
        combined = np.concatenate((image, appearance_prompt(image, cv2, np)), -1)
        if max(width, height) >= MAX_SIZE:
            return self._restore_tiles(combined, np)
        return self._restore_whole(combined, np)

    def _restore_whole(self, combined: Any, np: Any) -> Any:
        torch, model, stride_integral = self._load()
        padded, padding_h, padding_w = stride_integral(combined, 8)
        return self._predict(padded, torch, model, np)[padding_h:, padding_w:]

    def _restore_tiles(self, combined: Any, np: Any) -> Any:
        # ponytail: one scan direction. PreP-OCR scans four ways and takes the pixel-wise
        # median to suppress per-patch artefacts; add that if artefacts show up in review.
        torch, model, _ = self._load()
        height, width = combined.shape[:2]
        step = TILE_SIZE - 2 * TILE_BORDER
        # Reflect a border so edge pixels get the same context as interior ones.
        padded = np.pad(
            combined,
            ((TILE_BORDER, TILE_SIZE), (TILE_BORDER, TILE_SIZE), (0, 0)),
            mode="reflect",
        )
        restored = np.zeros((height, width, 3), dtype=np.uint8)
        for top in range(0, height, step):
            for left in range(0, width, step):
                tile = padded[top : top + TILE_SIZE, left : left + TILE_SIZE]
                predicted = self._predict(tile, torch, model, np)
                centre = predicted[
                    TILE_BORDER : TILE_BORDER + step, TILE_BORDER : TILE_BORDER + step
                ]
                rows = min(step, height - top)
                columns = min(step, width - left)
                restored[top : top + rows, left : left + columns] = centre[
                    :rows, :columns
                ]
        return restored

    def _predict(self, patch: Any, torch: Any, model: Any, np: Any) -> Any:
        tensor = torch.from_numpy((patch / 255.0).transpose(2, 0, 1)).unsqueeze(0)
        with torch.no_grad():
            predicted = torch.clamp(model(tensor.float()), 0, 1)
        return (predicted[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    def _load(self) -> tuple[Any, Any, Any]:
        with self._lock:
            if self._model is None:
                self._model = _build_model(self.source_root, self.checkpoint)
            return self._model


def appearance_prompt(image: Any, cv2: Any, np: Any) -> Any:
    """Background-plane estimate DocRes concatenates onto the image (DocRes, MIT).

    Dilation plus median blur approximates the page background; the normalised residual
    is what the Restormer consumes as its second three channels.
    """
    height, width = image.shape[:2]
    resized = cv2.resize(image, (1024, 1024))
    normalized_planes = []
    for plane in cv2.split(resized):
        dilated = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        background = cv2.medianBlur(dilated, 21)
        difference = 255 - cv2.absdiff(plane, background)
        normalized_planes.append(
            cv2.normalize(
                difference,
                None,
                alpha=0,
                beta=255,
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_8UC1,
            )
        )
    return cv2.resize(cv2.merge(normalized_planes), (width, height))


def _build_model(source_root: Path, checkpoint: Path) -> tuple[Any, Any, Any]:
    try:
        import torch
    except ImportError as error:
        raise ReaderError(
            "restoration_dependency_unavailable",
            "DocRes restoration requires PyTorch",
        ) from error

    root = str(source_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from data.preprocess.crop_merge_image import stride_integral
        from models import restormer_arch
        from utils import convert_state_dict
    except ImportError as error:
        raise ReaderError(
            "restoration_source_unavailable",
            f"DocRes checkout is incomplete at {source_root}",
        ) from error

    model = restormer_arch.Restormer(**RESTORMER_SETTINGS)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(convert_state_dict(state["model_state"]))
    model.eval()
    return torch, model, stride_integral


def _load_vision() -> tuple[Any, Any]:
    try:
        import cv2
        import numpy as np
    except ImportError as error:
        raise ReaderError(
            "restoration_dependency_unavailable",
            "DocRes restoration requires OpenCV and NumPy",
        ) from error
    return cv2, np
