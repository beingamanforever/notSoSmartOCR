"""Local OCR readers."""

from __future__ import annotations

import csv
import io
import math
import subprocess
import threading
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Protocol

from PIL import Image

from .contracts import BoundingBox, TextRegion


class LocalReader(Protocol):
    name: str

    def read(self, image_path: Path, page_number: int) -> list[TextRegion]: ...


class ReaderError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class TesseractReader:
    name = "tesseract"

    def __init__(
        self,
        language: str = "eng",
        executable: str = "tesseract",
        timeout_seconds: int = 120,
    ) -> None:
        self.language = language
        self.executable = executable
        self.timeout_seconds = timeout_seconds

    def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
        try:
            completed = subprocess.run(
                [
                    self.executable,
                    str(image_path),
                    "stdout",
                    "-l",
                    self.language,
                    "tsv",
                ],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except FileNotFoundError as error:
            raise ReaderError("reader_unavailable", str(error)) from error
        except subprocess.TimeoutExpired as error:
            raise ReaderError(
                "reader_timeout",
                f"Tesseract exceeded {self.timeout_seconds} seconds",
            ) from error

        if completed.returncode != 0:
            message = completed.stderr.strip() or "Tesseract returned no error message"
            raise ReaderError("reader_failed", message)

        return _parse_tesseract_tsv(completed.stdout, page_number, self.name)


class PaddleOCRVLReader:
    name = "paddleocr-vl-1.6"
    pipeline_version = "v1.6"

    def __init__(
        self,
        *,
        backend: str = "native",
        device: str | None = None,
        use_doc_orientation_classify: bool | None = None,
        use_doc_unwarping: bool | None = None,
        pipeline: object | None = None,
    ) -> None:
        self.backend = backend
        self.device = device
        self.use_doc_orientation_classify = use_doc_orientation_classify
        self.use_doc_unwarping = use_doc_unwarping
        self._pipeline = pipeline
        self._lock = threading.Lock()

    def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
        # ponytail: one lock protects the native pipeline; use worker processes if
        # measured throughput later requires concurrent model instances.
        with self._lock:
            pipeline = (
                self._pipeline
                if self._pipeline is not None
                else self._initialize_pipeline()
            )
            try:
                results = list(pipeline.predict(str(image_path)))
            except Exception as error:
                raise ReaderError("paddle_predict_failed", str(error)) from error

        if len(results) != 1:
            raise ReaderError(
                "invalid_reader_output",
                f"PaddleOCR-VL returned {len(results)} results for one image",
            )

        try:
            return _parse_paddle_result(results[0], page_number, self.name)
        except (KeyError, TypeError, ValueError) as error:
            raise ReaderError(
                "invalid_reader_output", f"Invalid PaddleOCR-VL output: {error}"
            ) from error

    def _initialize_pipeline(self) -> object:
        try:
            from paddleocr import PaddleOCRVL
        except (ImportError, OSError) as error:
            raise ReaderError("paddle_import_failed", str(error)) from error

        options: dict[str, object] = {
            "pipeline_version": self.pipeline_version,
            "vl_rec_backend": self.backend,
        }
        if self.device is not None:
            options["device"] = self.device
        if self.use_doc_orientation_classify is not None:
            options["use_doc_orientation_classify"] = self.use_doc_orientation_classify
        if self.use_doc_unwarping is not None:
            options["use_doc_unwarping"] = self.use_doc_unwarping
        try:
            self._pipeline = PaddleOCRVL(**options)
        except Exception as error:
            raise ReaderError("paddle_init_failed", str(error)) from error
        return self._pipeline


class GLMOCRReader:
    name = "glm-ocr"

    def __init__(
        self,
        *,
        ocr_api_host: str | None = None,
        ocr_api_port: int | None = None,
        layout_device: str | None = None,
        pipeline: object | None = None,
    ) -> None:
        self.ocr_api_host = ocr_api_host
        self.ocr_api_port = ocr_api_port
        self.layout_device = layout_device
        self._pipeline = pipeline
        self._lock = threading.Lock()

    def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
        with Image.open(image_path) as image:
            width, height = image.size

        with self._lock:
            pipeline = (
                self._pipeline
                if self._pipeline is not None
                else self._initialize_pipeline()
            )
            try:
                result = pipeline.parse(
                    str(image_path),
                    preserve_order=True,
                    save_layout_visualization=False,
                )
                output = result.to_dict()
            except Exception as error:
                raise ReaderError("glm_predict_failed", str(error)) from error

        reported_error = _optional_value(output, "error")
        if reported_error:
            raise ReaderError("glm_predict_failed", str(reported_error))
        try:
            return _parse_glm_result(output, page_number, self.name, width, height)
        except (KeyError, TypeError, ValueError) as error:
            raise ReaderError(
                "invalid_reader_output", f"Invalid GLM-OCR output: {error}"
            ) from error

    def _initialize_pipeline(self) -> object:
        try:
            from glmocr import GlmOcr
        except (ImportError, OSError) as error:
            raise ReaderError("glm_import_failed", str(error)) from error

        options: dict[str, object] = {"mode": "selfhosted"}
        if self.ocr_api_host is not None:
            options["ocr_api_host"] = self.ocr_api_host
        if self.ocr_api_port is not None:
            options["ocr_api_port"] = self.ocr_api_port
        if self.layout_device is not None:
            options["layout_device"] = self.layout_device
        try:
            self._pipeline = GlmOcr(**options)
        except Exception as error:
            raise ReaderError("glm_init_failed", str(error)) from error
        return self._pipeline


class GLMOCRDirectReader:
    """Model-only GLM-OCR ablation without layout analysis."""

    name = "glm-ocr-direct"

    def __init__(
        self,
        *,
        model_name: str = "zai-org/GLM-OCR",
        max_new_tokens: int = 8192,
        processor: object | None = None,
        model: object | None = None,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self._processor = processor
        self._model = model
        self._lock = threading.Lock()

    def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
        with Image.open(image_path) as image:
            width, height = image.size

        with self._lock:
            processor, model = self._initialize_components()
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "url": str(image_path)},
                        {"type": "text", "text": "Text Recognition:"},
                    ],
                }
            ]
            try:
                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(model.device)
                inputs.pop("token_type_ids", None)
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )
            except Exception as error:
                raise ReaderError("glm_direct_predict_failed", str(error)) from error

            try:
                input_length = inputs["input_ids"].shape[1]
                text = processor.decode(
                    generated_ids[0][input_length:], skip_special_tokens=True
                )
            except Exception as error:
                raise ReaderError("glm_direct_output_failed", str(error)) from error

        if not isinstance(text, str) or not text.strip():
            raise ReaderError(
                "glm_direct_output_failed", "GLM-OCR returned no decoded text"
            )
        return [
            TextRegion(
                id=f"p{page_number}-page-1",
                kind="page_text",
                text=text.strip(),
                confidence=None,
                bounding_box=BoundingBox(0, 0, width, height),
                reading_order=1,
                provider=self.name,
            )
        ]

    def _initialize_components(self) -> tuple[object, object]:
        if self._processor is not None and self._model is not None:
            return self._processor, self._model

        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except (ImportError, OSError) as error:
            raise ReaderError("glm_direct_import_failed", str(error)) from error

        try:
            if self._processor is None:
                self._processor = AutoProcessor.from_pretrained(self.model_name)
            if self._model is None:
                self._model = AutoModelForImageTextToText.from_pretrained(
                    pretrained_model_name_or_path=self.model_name,
                    torch_dtype="auto",
                    device_map="auto",
                )
        except Exception as error:
            raise ReaderError("glm_direct_init_failed", str(error)) from error
        return self._processor, self._model


def _parse_tesseract_tsv(
    output: str, page_number: int, provider: str
) -> list[TextRegion]:
    regions: list[TextRegion] = []
    rows = csv.DictReader(io.StringIO(output), delimiter="\t", quoting=csv.QUOTE_NONE)

    try:
        for row in rows:
            text = (row.get("text") or "").strip()
            if row.get("level") != "5" or not text:
                continue

            left = int(row["left"])
            top = int(row["top"])
            width = int(row["width"])
            height = int(row["height"])
            confidence_value = float(row["conf"])
            confidence = (
                None if confidence_value < 0 else round(confidence_value / 100, 4)
            )
            order = len(regions) + 1
            regions.append(
                TextRegion(
                    id=f"p{page_number}-word-{order}",
                    kind="word",
                    text=text,
                    confidence=confidence,
                    bounding_box=BoundingBox(
                        left=left,
                        top=top,
                        right=left + width,
                        bottom=top + height,
                    ),
                    reading_order=order,
                    provider=provider,
                )
            )
    except (KeyError, TypeError, ValueError) as error:
        raise ReaderError(
            "invalid_reader_output", f"Invalid Tesseract TSV: {error}"
        ) from error

    return regions


def _parse_paddle_result(
    result: object, page_number: int, provider: str
) -> list[TextRegion]:
    parsing_results = _required_value(result, "parsing_res_list")
    if not isinstance(parsing_results, list):
        raise TypeError("parsing_res_list must be a list")

    layout_result = _optional_value(result, "layout_det_res")
    layout_boxes = (
        _optional_value(layout_result, "boxes", []) if layout_result is not None else []
    )
    if not isinstance(layout_boxes, list):
        raise TypeError("layout_det_res.boxes must be a list")

    layout_metadata = {}
    for box in layout_boxes:
        score = _optional_value(box, "score")
        layout_metadata[
            (
                str(_required_value(box, "label")),
                _box_coordinates(_required_value(box, "coordinate")),
            )
        ] = _confidence(score) if score is not None else None

    regions = []
    for position, parsed in enumerate(parsing_results, start=1):
        label = str(_required_value(parsed, "label"))
        coordinates = _box_coordinates(_required_value(parsed, "bbox"))
        content = _required_value(parsed, "content")
        if not isinstance(content, str):
            raise TypeError("content must be a string")
        confidence = layout_metadata.get((label, coordinates))
        left, top, right, bottom = coordinates
        regions.append(
            TextRegion(
                id=f"p{page_number}-block-{position}",
                kind=label,
                text=content,
                confidence=confidence,
                bounding_box=BoundingBox(left, top, right, bottom),
                reading_order=position,
                provider=provider,
            )
        )
    return regions


def _parse_glm_result(
    result: object,
    page_number: int,
    provider: str,
    width: int,
    height: int,
) -> list[TextRegion]:
    pages = _required_value(result, "json_result")
    if not isinstance(pages, list) or len(pages) != 1:
        count = len(pages) if isinstance(pages, list) else "invalid"
        raise ValueError(f"json_result must contain one page, got {count}")
    if not isinstance(pages[0], list):
        raise TypeError("json_result page must be a list")

    regions = []
    for position, block in enumerate(pages[0], start=1):
        label = _required_value(block, "label")
        content = _required_value(block, "content")
        if not isinstance(label, str) or not isinstance(content, str):
            raise TypeError("region label and content must be strings")
        left, top, right, bottom = _normalized_box(
            _required_value(block, "bbox_2d"), width, height
        )
        regions.append(
            TextRegion(
                id=f"p{page_number}-block-{position}",
                kind=label,
                text=content,
                confidence=None,
                bounding_box=BoundingBox(left, top, right, bottom),
                reading_order=position,
                provider=provider,
            )
        )
    return regions


def _required_value(value: object, key: str) -> object:
    missing = object()
    result = _optional_value(value, key, missing)
    if result is missing:
        raise KeyError(key)
    return result


def _optional_value(value: object, key: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _box_coordinates(value: object) -> tuple[int, int, int, int]:
    coordinates = _coordinate_values(value)
    left, top, right, bottom = (int(round(number)) for number in coordinates)
    if right <= left or bottom <= top:
        raise ValueError("bounding box must have positive area")
    return left, top, right, bottom


def _normalized_box(
    value: object, width: int, height: int
) -> tuple[int, int, int, int]:
    left, top, right, bottom = _coordinate_values(value)
    if min(left, top, right, bottom) < 0 or max(left, top, right, bottom) > 1000:
        raise ValueError("normalized bounding box must be within 0 and 1000")
    return _box_coordinates(
        [
            left * width / 1000,
            top * height / 1000,
            right * width / 1000,
            bottom * height / 1000,
        ]
    )


def _coordinate_values(value: object) -> list[float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise TypeError("bounding box must contain four coordinates")
    try:
        coordinates = [float(number) for number in value]
    except (TypeError, ValueError) as error:
        raise TypeError("bounding box must contain four numbers") from error
    if len(coordinates) != 4 or not all(
        math.isfinite(number) for number in coordinates
    ):
        raise ValueError("bounding box must contain four finite numbers")
    return coordinates


def _confidence(value: object) -> float:
    confidence = float(value)
    if not math.isfinite(confidence):
        raise ValueError("confidence must be finite")
    return confidence
