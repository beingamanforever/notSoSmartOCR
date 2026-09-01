"""Local OCR readers."""

from __future__ import annotations

import csv
import io
import math
import subprocess
import threading
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Protocol

from PIL import Image

from .contracts import BoundingBox, TextRegion

NEMOTRON_BOX_TOLERANCE = 0.02


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
        page_segmentation_mode: int | None = None,
        thresholding_method: int | None = None,
    ) -> None:
        if page_segmentation_mode is not None and not 0 <= page_segmentation_mode <= 13:
            raise ValueError("Tesseract page segmentation mode must be from 0 to 13")
        if thresholding_method is not None and thresholding_method not in {0, 1, 2}:
            raise ValueError("Tesseract thresholding method must be 0, 1, or 2")
        self.language = language
        self.executable = executable
        self.timeout_seconds = timeout_seconds
        self.page_segmentation_mode = page_segmentation_mode
        self.thresholding_method = thresholding_method

    def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
        command = [
            self.executable,
            str(image_path),
            "stdout",
            "-l",
            self.language,
        ]
        if self.page_segmentation_mode is not None:
            command.extend(["--psm", str(self.page_segmentation_mode)])
        if self.thresholding_method is not None:
            command.extend(["-c", f"thresholding_method={self.thresholding_method}"])
        command.append("tsv")
        try:
            completed = subprocess.run(
                command,
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

        regions = _parse_tesseract_tsv(completed.stdout, page_number, self.name)
        configured = {
            key: value
            for key, value in (
                ("page_segmentation_mode", self.page_segmentation_mode),
                ("thresholding_method", self.thresholding_method),
            )
            if value is not None
        }
        if configured:
            for region in regions:
                assert region.text_provenance is not None
                region.text_provenance["tesseract_config"] = configured
        return regions


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


class GraniteDoclingReader:
    """Compact IBM page parser that serializes DocTags as text or Markdown."""

    name = "granite-docling"

    def __init__(
        self,
        *,
        model_name: str = "ibm-granite/granite-docling-258M",
        max_new_tokens: int = 8192,
        output_format: str = "text",
        processor: object | None = None,
        model: object | None = None,
        converter: Callable[[str, Image.Image], str] | None = None,
    ) -> None:
        if output_format not in {"text", "markdown"}:
            raise ValueError("Granite output format must be text or markdown")
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.output_format = output_format
        self._processor = processor
        self._model = model
        self._converter = converter
        self._lock = threading.Lock()

    def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB")
            width, height = image.size

        with self._lock:
            converter = (
                self._converter
                if self._converter is not None
                else self._initialize_converter()
            )
            processor, model = self._initialize_components()
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "Convert this page to docling."},
                    ],
                }
            ]
            try:
                prompt = processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                )
                inputs = processor(
                    text=prompt,
                    images=[image],
                    return_tensors="pt",
                ).to(model.device)
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )
            except Exception as error:
                raise ReaderError("granite_predict_failed", str(error)) from error

            try:
                prompt_length = inputs["input_ids"].shape[1]
                doctags = processor.decode(
                    generated_ids[0][prompt_length:],
                    skip_special_tokens=False,
                ).lstrip()
                text = converter(doctags, image)
            except Exception as error:
                raise ReaderError("granite_output_failed", str(error)) from error

        if not text.strip():
            raise ReaderError(
                "granite_output_failed", "Granite Docling returned no document text"
            )
        return [
            TextRegion(
                id=f"p{page_number}-page-1",
                kind=f"page_{self.output_format}",
                text=text.strip(),
                confidence=None,
                bounding_box=BoundingBox(0, 0, width, height),
                reading_order=1,
                provider=self.name,
            )
        ]

    def _initialize_converter(self) -> Callable[[str, Image.Image], str]:
        self._converter = _load_doctags_converter(self.output_format)
        return self._converter

    def _initialize_components(self) -> tuple[object, object]:
        if self._processor is not None and self._model is not None:
            return self._processor, self._model

        try:
            from transformers import AutoModelForMultimodalLM, AutoProcessor
        except (ImportError, OSError) as error:
            raise ReaderError("granite_import_failed", str(error)) from error

        try:
            if self._processor is None:
                self._processor = AutoProcessor.from_pretrained(self.model_name)
            if self._model is None:
                self._model = AutoModelForMultimodalLM.from_pretrained(
                    self.model_name,
                    torch_dtype="auto",
                    device_map="auto",
                )
        except Exception as error:
            raise ReaderError("granite_init_failed", str(error)) from error
        return self._processor, self._model


def _load_doctags_converter(
    output_format: str = "text",
) -> Callable[[str, Image.Image], str]:
    if output_format not in {"text", "markdown"}:
        raise ValueError("Granite output format must be text or markdown")
    try:
        from docling_core.types.doc import DoclingDocument
        from docling_core.types.doc.document import DocTagsDocument
    except (ImportError, OSError) as error:
        raise ReaderError("granite_import_failed", str(error)) from error

    def convert(doctags: str, image: Image.Image) -> str:
        tagged_document = DocTagsDocument.from_doctags_and_image_pairs(
            [doctags],
            [image],
        )
        document = DoclingDocument(name="Document")
        document.load_from_doctags(tagged_document)
        if output_format == "markdown":
            return document.export_to_markdown()
        return document.export_to_text()

    return convert


class NemotronOCRV2Reader:
    """NVIDIA detector, recognizer, and reading-order pipeline."""

    name = "nemotron-ocr-v2"

    def __init__(
        self,
        *,
        language: str = "multi",
        merge_level: str = "paragraph",
        batch_size: int = 1,
        pipeline: object | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.language = language
        self.merge_level = merge_level
        self.batch_size = batch_size
        self._pipeline = pipeline
        self._lock = threading.Lock()

    def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
        return self.read_with_merge_level(
            image_path,
            page_number,
            self.merge_level,
        )

    def read_with_merge_level(
        self,
        image_path: Path,
        page_number: int,
        merge_level: str,
    ) -> list[TextRegion]:
        with Image.open(image_path) as image:
            image_size = image.size

        with self._lock:
            pipeline = (
                self._pipeline
                if self._pipeline is not None
                else self._initialize_pipeline()
            )
            try:
                predictions = pipeline(
                    str(image_path),
                    merge_level=merge_level,
                )
            except Exception as error:
                raise ReaderError("nemotron_predict_failed", str(error)) from error

        try:
            return _parse_nemotron_predictions(
                predictions,
                page_number,
                self.name,
                image_size,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ReaderError(
                "invalid_reader_output",
                f"Invalid Nemotron OCR v2 output: {error}",
            ) from error

    def read_batch(
        self,
        image_paths: list[Path],
        page_numbers: list[int],
    ) -> list[list[TextRegion] | ReaderError]:
        if len(image_paths) != len(page_numbers):
            raise ValueError("image_paths and page_numbers must have equal lengths")

        image_sizes = []
        for image_path in image_paths:
            with Image.open(image_path) as image:
                image_sizes.append(image.size)

        results: list[list[TextRegion] | ReaderError] = []
        with self._lock:
            try:
                pipeline = (
                    self._pipeline
                    if self._pipeline is not None
                    else self._initialize_pipeline()
                )
            except ReaderError as error:
                return [ReaderError(error.code, str(error)) for _ in image_paths]

            for start in range(0, len(image_paths), self.batch_size):
                paths = image_paths[start : start + self.batch_size]
                numbers = page_numbers[start : start + self.batch_size]
                sizes = image_sizes[start : start + self.batch_size]
                try:
                    batch_predictions = pipeline(
                        [str(path) for path in paths],
                        merge_level=self.merge_level,
                    )
                except Exception as error:
                    results.extend(
                        ReaderError("nemotron_predict_failed", str(error))
                        for _ in paths
                    )
                    continue

                if not isinstance(batch_predictions, list) or len(
                    batch_predictions
                ) != len(paths):
                    message = (
                        "Invalid Nemotron OCR v2 batch output: expected "
                        f"{len(paths)} page results"
                    )
                    results.extend(
                        ReaderError("invalid_reader_output", message) for _ in paths
                    )
                    continue

                for predictions, page_number, image_size in zip(
                    batch_predictions,
                    numbers,
                    sizes,
                    strict=True,
                ):
                    try:
                        regions = _parse_nemotron_predictions(
                            predictions,
                            page_number,
                            self.name,
                            image_size,
                        )
                    except (KeyError, TypeError, ValueError) as error:
                        results.append(
                            ReaderError(
                                "invalid_reader_output",
                                f"Invalid Nemotron OCR v2 output: {error}",
                            )
                        )
                    else:
                        results.append(regions)
        return results

    def _initialize_pipeline(self) -> object:
        try:
            from nemotron_ocr.inference.pipeline_v2 import NemotronOCRV2
        except (ImportError, OSError) as error:
            raise ReaderError("nemotron_import_failed", str(error)) from error

        try:
            self._pipeline = NemotronOCRV2(lang=self.language)
        except Exception as error:
            raise ReaderError("nemotron_init_failed", str(error)) from error
        return self._pipeline


def _parse_nemotron_predictions(
    predictions: object,
    page_number: int,
    provider: str,
    image_size: tuple[int, int],
) -> list[TextRegion]:
    if isinstance(predictions, (str, bytes)) or not isinstance(predictions, Iterable):
        raise TypeError("predictions must be an iterable")

    regions = []
    for prediction in predictions:
        text = _required_value(prediction, "text")
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        stripped_text = text.strip()
        if not stripped_text:
            continue

        confidence = _confidence(_required_value(prediction, "confidence"))
        left, top, right, bottom = _nemotron_box(
            prediction,
            *image_size,
        )
        regions.append(
            TextRegion(
                id=f"p{page_number}-block-{len(regions) + 1}",
                kind="text",
                text=stripped_text,
                confidence=confidence,
                bounding_box=BoundingBox(left, top, right, bottom),
                reading_order=len(regions) + 1,
                provider=provider,
            )
        )
    return regions


def _nemotron_box(
    prediction: object,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    left, lower, right, upper = _coordinate_values(
        [
            _required_value(prediction, "left"),
            _required_value(prediction, "lower"),
            _required_value(prediction, "right"),
            _required_value(prediction, "upper"),
        ]
    )
    if (
        min(left, lower, right, upper) < -NEMOTRON_BOX_TOLERANCE
        or max(left, lower, right, upper) > 1 + NEMOTRON_BOX_TOLERANCE
    ):
        raise ValueError("Nemotron bounding box must be normalized from 0 to 1")
    left, lower, right, upper = (
        min(1.0, max(0.0, coordinate)) for coordinate in (left, lower, right, upper)
    )
    return _box_coordinates(
        [
            math.floor(left * width),
            math.floor(lower * height),
            math.ceil(right * width),
            math.ceil(upper * height),
        ]
    )


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
            provenance = {
                "method": "tesseract_tsv",
                "block_num": int(row["block_num"]),
                "paragraph_num": int(row["par_num"]),
                "line_num": int(row["line_num"]),
                "word_num": int(row["word_num"]),
            }
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
                    text_provenance=provenance,
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
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("confidence must be between 0 and 1")
    return confidence
