"""Words from docTR's OCR predictor: the cells a scan lacks, and a second reader's doubt.

The boxes stand in for text cells in coverage checks. The recognition confidence comes
from a reader trained on printed text, so it drops on handwriting, where Falcon's own
likelihood stays high on confident misreads (the input-type signal production APIs
expose as a handwriting style: Azure `isHandwritten`, Textract `TextType`).
"""

from __future__ import annotations

import threading
from pathlib import Path

from .providers import ReaderError

# Mindee docTR (Apache-2.0); the same library already orients pages.
DETECTION_ARCH = "db_resnet50"
RECOGNITION_ARCH = "crnn_vgg16_bn"
_lock = threading.Lock()
_predictors: dict[str, object] = {}


def detect_words(image_path: Path, device: str = "cuda") -> list[list[float]]:
    """[left, top, right, bottom, detection score, recognition confidence] per word, in page pixels."""
    import numpy as np
    from PIL import Image

    with Image.open(image_path) as opened:
        page = np.asarray(opened.convert("RGB"))
    height, width = page.shape[:2]
    with _lock:
        try:
            import torch
            from doctr.models import ocr_predictor

            if device.startswith("cuda") and not torch.cuda.is_available():
                raise ReaderError("text_detection_unavailable", f"{device} is unavailable")
            if device not in _predictors:
                _predictors[device] = (
                    ocr_predictor(DETECTION_ARCH, RECOGNITION_ARCH, pretrained=True)
                    .eval()
                    .to(device)
                )
                # Retain thin handwritten strokes before DB contour extraction.
                # The default 0.3 erases the isolated totals in "Page 3 of 4".
                _predictors[device].det_predictor.model.postprocessor.bin_thresh = 0.2
            with torch.inference_mode():
                result = _predictors[device]([page]).pages[0]
        except ReaderError:
            raise
        except Exception as error:
            raise ReaderError("text_detection_failed", str(error)) from error
    words = [
        (*word.geometry[0], *word.geometry[1], word.objectness_score, word.confidence)
        for block in result.blocks
        for line in block.lines
        for word in line.words
    ]
    return [
        [x0 * width, y0 * height, x1 * width, y1 * height, float(score), float(confidence)]
        for x0, y0, x1, y1, score, confidence in words
        if x1 > x0 and y1 > y0
    ]
