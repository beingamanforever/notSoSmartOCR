"""NVIDIA Nemotron Parse 2.0 reader for its published output contract."""

from __future__ import annotations

import math
import re
import threading
from collections.abc import Callable
from pathlib import Path

from PIL import Image

from .contracts import BoundingBox, TextRegion
from .providers import ReaderError

MODEL_ID = "nvidia/NVIDIA-Nemotron-Parse-2.0"
MODEL_REVISION = "b6742064f4a8cf22a10383ece5e7fbead355ac04"
MODEL_LICENSE = "OpenMDW-1.1"
TOKENIZER_LICENSE = "CC-BY-4.0"
VISION_ID = "nvidia/C-RADIOv2-H"
VISION_REVISION = "0d8f4c18c877166eda07ddae1386bcad256b7a6a"
VISION_LICENSE = "NVIDIA Open Model License Agreement"
DECODER_ORIGIN = "Facebook AI Research (Meta)"
DECODER_UPSTREAM_LICENSE = "MIT"
DEFAULT_PROMPT = (
    "</s><s><predict_bbox><predict_classes><output_markdown><predict_no_text_in_pic>"
)
TARGET_WIDTH = 1664
TARGET_HEIGHT = 2048

_REGION_PATTERN = re.compile(
    r"<x_(\d+(?:\.\d+)?)><y_(\d+(?:\.\d+)?)>"
    r"(.*?)"
    r"<x_(\d+(?:\.\d+)?)><y_(\d+(?:\.\d+)?)>"
    r"<class_([^>]+)>",
    re.DOTALL,
)

OutputGenerator = Callable[[Path, str], str]


class NemotronParseReader:
    """Read document regions without merging them into production output."""

    name = "nemotron-parse-2.0"

    def __init__(
        self,
        *,
        generator: OutputGenerator | None = None,
        device: str = "cuda:0",
        local_files_only: bool = True,
    ) -> None:
        self.device = device
        self.local_files_only = local_files_only
        self._generator = generator
        self._lock = threading.Lock()

    def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
        try:
            with Image.open(image_path) as image:
                width, height = image.size
        except (OSError, ValueError) as error:
            raise ReaderError("nemotron_parse_image_failed", str(error)) from error

        with self._lock:
            generator = self._generator or self._initialize_generator()
            try:
                output = generator(image_path, DEFAULT_PROMPT)
            except ReaderError:
                raise
            except Exception as error:
                raise ReaderError(
                    "nemotron_parse_predict_failed", str(error)
                ) from error

        try:
            return _parse_output(
                output,
                page_number=page_number,
                provider=self.name,
                image_size=(width, height),
            )
        except (TypeError, ValueError) as error:
            raise ReaderError(
                "invalid_reader_output", f"Invalid Nemotron Parse 2.0 output: {error}"
            ) from error

    def _initialize_generator(self) -> OutputGenerator:
        try:
            self._generator = _TransformersGenerator(
                device=self.device,
                local_files_only=self.local_files_only,
            )
        except ImportError as error:
            raise ReaderError("nemotron_parse_import_failed", str(error)) from error
        except OSError as error:
            raise ReaderError("nemotron_parse_model_unavailable", str(error)) from error
        except Exception as error:
            raise ReaderError("nemotron_parse_init_failed", str(error)) from error
        return self._generator


class _TransformersGenerator:
    def __init__(
        self,
        *,
        device: str,
        local_files_only: bool,
    ) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoProcessor, GenerationConfig
        except (ImportError, OSError) as error:
            raise ImportError(str(error)) from error

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA is unavailable for requested device {device}")

        load_options: dict[str, object] = {
            "trust_remote_code": True,
            "local_files_only": local_files_only,
            "revision": MODEL_REVISION,
        }

        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.processor = AutoProcessor.from_pretrained(MODEL_ID, **load_options)
        self.model = (
            AutoModel.from_pretrained(
                MODEL_ID,
                torch_dtype=dtype,
                **load_options,
            )
            .to(device)
            .eval()
        )
        self.generation_config = GenerationConfig.from_pretrained(
            MODEL_ID,
            **load_options,
        )
        self.device = device

    def __call__(self, image_path: Path, prompt: str) -> str:
        try:
            with Image.open(image_path) as source:
                image = source.convert("RGB")
            inputs = self.processor(
                images=[image],
                text=prompt,
                return_tensors="pt",
                add_special_tokens=False,
            ).to(self.device)
            outputs = self.model.generate(
                **inputs,
                generation_config=self.generation_config,
            )
            output = self.processor.batch_decode(
                outputs,
                skip_special_tokens=True,
            )[0]
        except ReaderError:
            raise
        except Exception as error:
            raise ReaderError("nemotron_parse_predict_failed", str(error)) from error
        if not isinstance(output, str):
            raise ReaderError(
                "invalid_reader_output", "Decoded output must be a string"
            )
        return output


def _parse_output(
    output: object,
    *,
    page_number: int,
    provider: str,
    image_size: tuple[int, int],
) -> list[TextRegion]:
    if not isinstance(output, str):
        raise TypeError("output must be a string")

    matches = list(_REGION_PATTERN.finditer(output))
    residual = _REGION_PATTERN.sub("", output)
    residual = residual.replace("</s>", "").replace("<s>", "").strip()
    if residual:
        raise ValueError("output contains text outside complete regions")
    if not matches:
        return []

    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")

    regions = []
    for index, match in enumerate(matches, start=1):
        x1, y1, text, x2, y2, semantic_class = match.groups()
        normalized_box = tuple(float(value) for value in (x1, y1, x2, y2))
        _validate_normalized_box(normalized_box)
        float_box = _transform_box(normalized_box, width, height)
        left, top, right, bottom = _integer_box(float_box, width, height)
        provenance = {
            "method": "nemotron_parse_2_output_contract",
            "model": {
                "id": MODEL_ID,
                "revision": MODEL_REVISION,
                "license": MODEL_LICENSE,
            },
            "tokenizer_license": TOKENIZER_LICENSE,
            "vision_encoder": {
                "id": VISION_ID,
                "reference_revision": VISION_REVISION,
                "reference_model_license": VISION_LICENSE,
                "embedded_artifact_license": MODEL_LICENSE,
                "origin": "NVIDIA",
            },
            "decoder": {
                "architecture": "mBART",
                "origin": DECODER_ORIGIN,
                "upstream_license": DECODER_UPSTREAM_LICENSE,
                "embedded_artifact_license": MODEL_LICENSE,
            },
            "prompt": DEFAULT_PROMPT,
            "semantic_class": semantic_class,
            "normalized_bbox": list(normalized_box),
            "transformed_bbox": list(float_box),
        }
        regions.append(
            TextRegion(
                id=f"p{page_number}-block-{index}",
                kind=semantic_class,
                text=text,
                confidence=None,
                bounding_box=BoundingBox(left, top, right, bottom),
                reading_order=index,
                provider=provider,
                text_provenance=provenance,
                structure={
                    "semantic_class": semantic_class,
                    "normalized_bbox": list(normalized_box),
                },
            )
        )
    return regions


def _validate_normalized_box(box: tuple[float, float, float, float]) -> None:
    if not all(math.isfinite(value) for value in box):
        raise ValueError("bounding boxes must be finite")
    x1, y1, x2, y2 = box
    if min(box) < 0 or max(box) > 1:
        raise ValueError("bounding boxes must be normalized from 0 to 1")
    if x1 >= x2 or y1 >= y2:
        raise ValueError("bounding boxes must have positive area")


def _transform_box(
    box: tuple[float, float, float, float],
    original_width: int,
    original_height: int,
) -> tuple[float, float, float, float]:
    aspect_ratio = original_width / original_height
    resized_width = original_width
    resized_height = original_height

    if original_height > TARGET_HEIGHT:
        resized_height = TARGET_HEIGHT
        resized_width = max(1, int(resized_height * aspect_ratio))
    if resized_width > TARGET_WIDTH:
        resized_width = TARGET_WIDTH
        resized_height = max(1, int(resized_width / aspect_ratio))

    pad_left = max(0, TARGET_WIDTH - resized_width) // 2
    pad_top = max(0, TARGET_HEIGHT - resized_height) // 2
    x1, y1, x2, y2 = box
    left = ((x1 * TARGET_WIDTH) - pad_left) * original_width / resized_width
    right = ((x2 * TARGET_WIDTH) - pad_left) * original_width / resized_width
    top = ((y1 * TARGET_HEIGHT) - pad_top) * original_height / resized_height
    bottom = ((y2 * TARGET_HEIGHT) - pad_top) * original_height / resized_height
    return (
        min(original_width, max(0.0, left)),
        min(original_height, max(0.0, top)),
        min(original_width, max(0.0, right)),
        min(original_height, max(0.0, bottom)),
    )


def _integer_box(
    box: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    integer_box = (
        max(0, min(width, math.floor(left))),
        max(0, min(height, math.floor(top))),
        max(0, min(width, math.ceil(right))),
        max(0, min(height, math.ceil(bottom))),
    )
    if integer_box[0] >= integer_box[2] or integer_box[1] >= integer_box[3]:
        raise ValueError("transformed bounding boxes must have positive area")
    return integer_box
