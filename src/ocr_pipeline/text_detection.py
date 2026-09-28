"""Word boxes from docTR's text detector, the cells a scan lacks for coverage checks."""

from __future__ import annotations

import threading
from pathlib import Path

from .providers import ReaderError

# Mindee docTR (Apache-2.0); the same library already orients pages.
DETECTION_ARCH = "db_resnet50"
_lock = threading.Lock()
_predictors: dict[str, object] = {}


def detect_words(image_path: Path, device: str = "cuda") -> list[list[float]]:
    """[left, top, right, bottom, score] per detected word, in page pixels."""
    import numpy as np
    from PIL import Image

    with Image.open(image_path) as opened:
        page = np.asarray(opened.convert("RGB"))
    height, width = page.shape[:2]
    with _lock:
        try:
            import torch
            from doctr.models import detection_predictor

            if device.startswith("cuda") and not torch.cuda.is_available():
                raise ReaderError("text_detection_unavailable", f"{device} is unavailable")
            if device not in _predictors:
                _predictors[device] = (
                    detection_predictor(DETECTION_ARCH, pretrained=True).eval().to(device)
                )
            with torch.inference_mode():
                words = _predictors[device]([page])[0]["words"]
        except ReaderError:
            raise
        except Exception as error:
            raise ReaderError("text_detection_failed", str(error)) from error
    return [
        [float(x0 * width), float(y0 * height), float(x1 * width), float(y1 * height), float(score)]
        for x0, y0, x1, y1, score in words
        if x1 > x0 and y1 > y0
    ]
