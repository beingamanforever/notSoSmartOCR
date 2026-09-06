"""Local OCR readers."""

from __future__ import annotations

import base64
import csv
import io
import json
import math
import os
import subprocess
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from PIL import Image

from .contracts import BoundingBox, TextRegion
from .openrouter import OpenRouterError, OpenRouterResult

MINISTRAL_OCR_PROMPT = (
    "Return only a literal Markdown transcription of the visible document, with no "
    "preamble, commentary, or code fence. Preserve reading order, line breaks, "
    "headings, tables, form labels, checkbox states, handwriting, mathematical "
    "symbols, and original spelling. Never repair grammar, complete a phrase, "
    "explain a diagram, or add text that is not visible. Write [illegible] when a "
    "character sequence cannot be read literally."
)
MINISTRAL_MODEL_ID = "mistralai/Ministral-3-3B-Instruct-2512"
MINISTRAL_MODEL_REVISION = "b35d4dfe56c142746f54dbd64f579faab2744308"
MINISTRAL_MODEL_ORIGIN = "Mistral AI, France"
MINISTRAL_MODEL_LICENSE = "Apache-2.0"
PHI4_HANDWRITING_PROMPT = (
    "Transcribe only the handwritten text exactly. Do not correct or explain it."
)
PHI4_MODEL_ID = "microsoft/Phi-4-multimodal-instruct"
PHI4_MODEL_REVISION = "93f923e1a7727d1c4f446756212d9d3e8fcc5d81"
PHI4_MODEL_ORIGIN = "Microsoft, United States"
PHI4_MODEL_LICENSE = "MIT"
PHI4_ADAPTER_FORMAT = "phi4_vision_decoder_lora_v2"
PHI4_LORA_PARTS = ("lora_A.vision", "lora_B.vision")
PHI4_LORA_TARGETS = ("qkv_proj", "o_proj", "gate_up_proj", "down_proj")
_PHI4_SDPA_LOCK = threading.RLock()
MAX_HANDWRITING_SERVICE_RESPONSE_BYTES = 1_000_000
MAX_MINISTRAL_SERVICE_RESPONSE_BYTES = 5_000_000


class LocalReader(Protocol):
    name: str

    def read(self, image_path: Path, page_number: int) -> list[TextRegion]: ...


class ReaderError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class Phi4HandwritingReader:
    """Local Phi-4 crop recognizer with a validated vision-decoder adapter."""

    name = "phi4-handwriting"

    def __init__(
        self,
        adapter_path: Path,
        *,
        model_name_or_path: str | Path = PHI4_MODEL_ID,
        model_revision: str = PHI4_MODEL_REVISION,
        device: str = "cuda:0",
        max_new_tokens: int = 128,
        max_batch_items: int = 16,
        batch_size: int = 2,
        local_files_only: bool = True,
        processor: object | None = None,
        model: object | None = None,
        generation_config: object | None = None,
        torch_module: object | None = None,
    ) -> None:
        adapter_path = Path(adapter_path)
        model_source = Path(model_name_or_path)
        if not adapter_path.is_file():
            raise FileNotFoundError(f"Phi-4 adapter was not found: {adapter_path}")
        if max_new_tokens <= 0:
            raise ValueError("Phi-4 max_new_tokens must be positive")
        if max_batch_items <= 0:
            raise ValueError("Phi-4 max_batch_items must be positive")
        if batch_size <= 0 or batch_size > max_batch_items:
            raise ValueError("Phi-4 batch_size must fit within max_batch_items")
        if str(model_name_or_path) == PHI4_MODEL_ID:
            if model_revision != PHI4_MODEL_REVISION:
                raise ValueError("Phi-4 must use the pinned model revision")
        elif not model_source.is_dir():
            raise FileNotFoundError(
                f"Local Phi-4 model directory was not found: {model_source}"
            )
        self.adapter_path = adapter_path
        self.model_name_or_path = str(model_name_or_path)
        self.model_revision = model_revision
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.max_batch_items = max_batch_items
        self.batch_size = batch_size
        self.local_files_only = local_files_only
        self._processor = processor
        self._model = model
        self._generation_config = generation_config
        self._torch = torch_module
        self._adapter_loaded = False
        self._adapter_provenance: dict[str, Any] | None = None
        self._lock = threading.Lock()

    @property
    def provenance(self) -> dict[str, Any]:
        adapter = self._adapter_provenance or {
            "format": PHI4_ADAPTER_FORMAT,
            "source": self.adapter_path.name,
            "validated": False,
        }
        official_model = self.model_name_or_path == PHI4_MODEL_ID
        return {
            "id": PHI4_MODEL_ID
            if official_model
            else Path(self.model_name_or_path).name,
            "source": self.model_name_or_path if official_model else "local_directory",
            "revision": self.model_revision if official_model else "unverified",
            "origin": PHI4_MODEL_ORIGIN if official_model else "unverified",
            "license": PHI4_MODEL_LICENSE if official_model else "unverified",
            "identity_verified": official_model,
            "local_files_only": self.local_files_only,
            "adapter": dict(adapter),
        }

    def transcribe_batch(self, images: Sequence[Image.Image]) -> list[str]:
        if not images:
            return []
        if len(images) > self.max_batch_items:
            raise ReaderError(
                "phi4_handwriting_batch_too_large",
                "Phi-4 handwriting crop batch exceeds its configured limit",
            )
        with self._lock:
            processor, model, generation_config, torch_module = (
                self._initialize_components()
            )
            output = []
            for start in range(0, len(images), self.batch_size):
                output.extend(
                    self._transcribe_batch(
                        images[start : start + self.batch_size],
                        processor,
                        model,
                        generation_config,
                        torch_module,
                    )
                )
            return output

    def _initialize_components(self) -> tuple[object, object, object | None, object]:
        if self._torch is None or self._processor is None or self._model is None:
            try:
                import torch
                import transformers.utils as transformers_utils
                from transformers.utils import import_utils
            except (ImportError, OSError) as error:
                raise ReaderError(
                    "phi4_handwriting_import_failed",
                    "Phi-4 handwriting dependencies are unavailable",
                ) from error

            try:
                with _force_phi4_sdpa(transformers_utils, import_utils):
                    from transformers import (
                        AutoModelForCausalLM,
                        AutoProcessor,
                        GenerationConfig,
                    )

                    options: dict[str, object] = {
                        "trust_remote_code": True,
                        "local_files_only": self.local_files_only,
                    }
                    if self.model_name_or_path == PHI4_MODEL_ID:
                        options["revision"] = self.model_revision
                    self._torch = torch
                    self._processor = AutoProcessor.from_pretrained(
                        self.model_name_or_path,
                        dynamic_hd=1,
                        **options,
                    )
                    self._generation_config = GenerationConfig.from_pretrained(
                        self.model_name_or_path,
                        **options,
                    )
                    self._model = AutoModelForCausalLM.from_pretrained(
                        self.model_name_or_path,
                        torch_dtype=torch.bfloat16,
                        _attn_implementation="sdpa",
                        **options,
                    ).to(self.device)
            except Exception as error:
                raise ReaderError(
                    "phi4_handwriting_init_failed",
                    "Phi-4 handwriting model could not be initialized",
                ) from error

        if not self._adapter_loaded:
            self._load_adapter(self._model, self._torch)
        return (
            self._processor,
            self._model,
            self._generation_config,
            self._torch,
        )

    def _load_adapter(self, model: object, torch_module: object) -> None:
        try:
            model.set_lora_adapter("vision")
            parameters = list(model.named_parameters())
            expected = {
                name
                for name, _ in parameters
                if any(part in name for part in PHI4_LORA_PARTS)
                and any(target in name for target in PHI4_LORA_TARGETS)
            }
            if not expected:
                raise ValueError("model exposes no vision-decoder LoRA")
            payload = torch_module.load(
                self.adapter_path,
                map_location="cpu",
                weights_only=True,
            )
            state, provenance = _phi4_adapter_payload(payload, expected)
            load_result = model.load_state_dict(state, strict=False)
            if getattr(load_result, "unexpected_keys", []):
                raise ValueError("adapter contains unexpected parameters")
            model.eval()
        except Exception as error:
            raise ReaderError(
                "phi4_handwriting_adapter_failed",
                "Phi-4 handwriting adapter provenance or parameters are invalid",
            ) from error
        self._adapter_provenance = {
            "format": PHI4_ADAPTER_FORMAT,
            "source": self.adapter_path.name,
            "validated": True,
            **provenance,
        }
        self._adapter_loaded = True

    def _transcribe_batch(
        self,
        images: Sequence[Image.Image],
        processor: object,
        model: object,
        generation_config: object | None,
        torch_module: object,
    ) -> list[str]:
        prompt = f"<|user|><|image_1|>{PHI4_HANDWRITING_PROMPT}<|end|><|assistant|>"
        try:
            inputs = processor(
                text=[prompt] * len(images),
                images=list(images),
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            input_ids = inputs.get("input_ids")
            if input_ids is None or not hasattr(input_ids, "shape"):
                raise ValueError("processor returned no token ids")
            prompt_tokens = int(input_ids.shape[-1])
            generation_options: dict[str, object] = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": False,
                "num_beams": 1,
            }
            if generation_config is not None:
                generation_options["generation_config"] = generation_config
            with torch_module.inference_mode():
                output = model.generate(**inputs, **generation_options)
            sequences = getattr(output, "sequences", output)
            if len(sequences) != len(images):
                raise ValueError("generation returned the wrong batch size")
            generated = [sequence[prompt_tokens:] for sequence in sequences]
            if any(
                int(tokens.shape[-1] if hasattr(tokens, "shape") else len(tokens))
                >= self.max_new_tokens
                for tokens in generated
            ):
                raise ValueError("generation reached its token limit")
            texts = [
                processor.decode(
                    tokens,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                for tokens in generated
            ]
        except Exception as error:
            raise ReaderError(
                "phi4_handwriting_inference_failed",
                "Phi-4 handwriting inference did not produce a bounded result",
            ) from error
        if any(not isinstance(text, str) for text in texts):
            raise ReaderError(
                "phi4_handwriting_output_failed",
                "Phi-4 handwriting output was not text",
            )
        return [text.strip() for text in texts]


class Phi4HandwritingServiceReader:
    """Call a local, warm Phi-4 handwriting process over loopback HTTP."""

    name = "phi4-handwriting"

    def __init__(
        self,
        service_url: str,
        *,
        max_batch_items: int = 16,
        timeout_seconds: float = 120,
    ) -> None:
        parsed = urlsplit(service_url)
        if parsed.scheme != "http" or parsed.hostname not in {
            "127.0.0.1",
            "localhost",
            "::1",
        }:
            raise ValueError("Phi-4 handwriting service must use loopback HTTP")
        if max_batch_items <= 0:
            raise ValueError("Phi-4 max_batch_items must be positive")
        if timeout_seconds <= 0:
            raise ValueError("Phi-4 service timeout must be positive")
        self.service_url = service_url.rstrip("/")
        self.max_batch_items = max_batch_items
        self.timeout_seconds = timeout_seconds
        self._provenance: dict[str, Any] = {
            "id": PHI4_MODEL_ID,
            "source": "loopback_service",
            "revision": PHI4_MODEL_REVISION,
            "origin": PHI4_MODEL_ORIGIN,
            "license": PHI4_MODEL_LICENSE,
            "identity_verified": True,
            "local_files_only": True,
            "adapter": {
                "format": PHI4_ADAPTER_FORMAT,
                "source": "sidecar",
                "validated": False,
            },
        }

    @property
    def provenance(self) -> dict[str, Any]:
        return dict(self._provenance)

    def transcribe_batch(self, images: Sequence[Image.Image]) -> list[str]:
        if not images:
            return []
        if len(images) > self.max_batch_items:
            raise ReaderError(
                "phi4_handwriting_batch_too_large",
                "Phi-4 handwriting crop batch exceeds its configured limit",
            )
        payload = json.dumps(
            {"images": [_encode_png(image) for image in images]},
            separators=(",", ":"),
        ).encode("utf-8")
        request = Request(
            f"{self.service_url}/transcribe",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read(MAX_HANDWRITING_SERVICE_RESPONSE_BYTES + 1)
            if len(body) > MAX_HANDWRITING_SERVICE_RESPONSE_BYTES:
                raise ValueError("service response exceeded its limit")
            result = json.loads(body)
            if not isinstance(result, dict):
                raise ValueError("service response was not an object")
            texts = result.get("texts")
            provenance = result.get("provenance")
            if (
                not isinstance(texts, list)
                or len(texts) != len(images)
                or any(not isinstance(text, str) for text in texts)
                or not isinstance(provenance, dict)
            ):
                raise ValueError("service response did not match its contract")
        except (HTTPError, URLError, OSError, ValueError) as error:
            raise ReaderError(
                "phi4_handwriting_service_failed",
                "The local Phi-4 handwriting service did not return a valid result",
            ) from error
        self._provenance = dict(provenance)
        return [text.strip() for text in texts]


def _encode_png(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


@contextmanager
def _force_phi4_sdpa(
    transformers_utils: object,
    import_utils: object,
) -> Iterator[None]:
    """Keep Phi-4 remote imports off an incompatible FlashAttention binary."""
    with _PHI4_SDPA_LOCK:
        names = (
            "is_flash_attn_2_available",
            "is_flash_attn_greater_or_equal_2_10",
        )
        targets = tuple(
            (module, name, getattr(module, name))
            for module in (transformers_utils, import_utils)
            for name in names
        )

        def unavailable(*args: object, **kwargs: object) -> bool:
            return False

        for module, name, _ in targets:
            setattr(module, name, unavailable)
        try:
            yield
        finally:
            for module, name, original in targets:
                setattr(module, name, original)


def _phi4_adapter_payload(
    payload: object,
    expected_names: set[str],
) -> tuple[dict[str, object], dict[str, object]]:
    if not isinstance(payload, dict) or payload.get("format") != PHI4_ADAPTER_FORMAT:
        raise ValueError("unsupported adapter format")
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("adapter lacks training provenance")
    family_ids = provenance.get("training_family_ids")
    family_count = provenance.get("training_family_count")
    if (
        not isinstance(family_ids, list)
        or not family_ids
        or not all(
            isinstance(value, str) and bool(value) and value == value.strip().casefold()
            for value in family_ids
        )
        or len(set(family_ids)) != len(family_ids)
        or family_count != len(family_ids)
    ):
        raise ValueError("adapter has invalid training provenance")
    state = payload.get("state")
    if not isinstance(state, dict) or set(state) != expected_names:
        raise ValueError("adapter does not match the vision-decoder LoRA")
    return state, {
        "training_family_count": family_count,
    }


# Tesseract's OpenMP path costs more than it returns: on an 8-core host a single crop
# reads 2.5x faster with one thread, and concurrent crops otherwise oversubscribe the
# machine badly. One process per crop, one thread per process.
_SINGLE_THREADED_ENV = {**os.environ, "OMP_THREAD_LIMIT": "1"}


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
                env=_SINGLE_THREADED_ENV,
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


class MinistralOCRReader:
    """Unmeasured Ministral full-page OCR challenger."""

    name = "ministral-ocr"

    def __init__(
        self,
        *,
        model_name: str = MINISTRAL_MODEL_ID,
        model_revision: str = MINISTRAL_MODEL_REVISION,
        device_map: str = "cuda:0",
        prompt: str = MINISTRAL_OCR_PROMPT,
        max_new_tokens: int = 8192,
        processor: object | None = None,
        model: object | None = None,
    ) -> None:
        if not prompt.strip():
            raise ValueError("Ministral OCR prompt must not be empty")
        if max_new_tokens <= 0:
            raise ValueError("Ministral OCR max_new_tokens must be positive")
        if not device_map.strip():
            raise ValueError("Ministral OCR device_map must not be empty")
        if len(model_revision) != 40 or any(
            char not in "0123456789abcdef" for char in model_revision.lower()
        ):
            raise ValueError("Ministral OCR model revision must be an immutable commit")
        if (
            model_name == MINISTRAL_MODEL_ID
            and model_revision != MINISTRAL_MODEL_REVISION
        ):
            raise ValueError("Ministral OCR must use the pinned model revision")
        self.model_name = model_name
        self.model_revision = model_revision
        self.device_map = device_map
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self.local_files_only = True
        self._processor = processor
        self._model = model
        self._lock = threading.Lock()

    @property
    def provenance(self) -> dict[str, Any]:
        official_model = self.model_name == MINISTRAL_MODEL_ID
        return {
            "id": MINISTRAL_MODEL_ID if official_model else None,
            "loaded_from": self.model_name,
            "revision": self.model_revision,
            "origin": MINISTRAL_MODEL_ORIGIN if official_model else None,
            "license": MINISTRAL_MODEL_LICENSE if official_model else None,
            "identity_verified": official_model,
            "local_files_only": self.local_files_only,
        }

    def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
        try:
            with Image.open(image_path) as image:
                width, height = image.size
        except (OSError, ValueError) as error:
            raise ReaderError("ministral_image_failed", str(error)) from error

        text, input_format = self.generate_text(image_path, self.prompt)
        provenance = {
            "method": "ministral_full_page_generation",
            "provider": self.name,
            "model": self.provenance,
            "prompt": self.prompt,
            "generation": {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": False,
            },
            "processor_input": input_format,
        }
        return [
            TextRegion(
                id=f"p{page_number}-page-1",
                kind="page_text",
                text=text,
                confidence=None,
                bounding_box=BoundingBox(0, 0, width, height),
                reading_order=1,
                provider=self.name,
                text_provenance=provenance,
            )
        ]

    def generate_text(self, image_path: Path, prompt: str) -> tuple[str, str]:
        if not prompt.strip():
            raise ReaderError("ministral_prompt_failed", "Prompt must not be empty")
        with self._lock:
            processor, model = self._initialize_components()
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "url": str(image_path)},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            try:
                input_format = "chat_template"
                try:
                    inputs = processor.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                    )
                except ValueError as error:
                    if "does not have a chat template" not in str(error):
                        raise
                    input_format = "base_image_text"
                    with Image.open(image_path) as source:
                        inputs = processor(
                            images=source.convert("RGB"),
                            text=f"<s>[INST][IMG]{prompt}[/INST]",
                            return_tensors="pt",
                        )
                inputs = inputs.to(model.device)
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )
            except Exception as error:
                raise ReaderError("ministral_predict_failed", str(error)) from error

            try:
                prompt_length = inputs["input_ids"].shape[1]
                text = processor.decode(
                    generated_ids[0][prompt_length:],
                    skip_special_tokens=True,
                )
            except Exception as error:
                raise ReaderError("ministral_output_failed", str(error)) from error

        if not isinstance(text, str) or not text.strip():
            raise ReaderError(
                "ministral_output_failed", "Ministral OCR returned no decoded text"
            )
        return text.strip(), input_format

    def _initialize_components(self) -> tuple[object, object]:
        if self._processor is not None and self._model is not None:
            return self._processor, self._model

        try:
            from transformers import (
                AutoProcessor,
                FineGrainedFP8Config,
                Mistral3ForConditionalGeneration,
            )
        except (ImportError, OSError) as error:
            raise ReaderError("ministral_import_failed", str(error)) from error

        try:
            if self._processor is None:
                self._processor = AutoProcessor.from_pretrained(
                    self.model_name,
                    revision=self.model_revision,
                    fix_mistral_regex=True,
                    local_files_only=self.local_files_only,
                )
            if self._model is None:
                self._model = Mistral3ForConditionalGeneration.from_pretrained(
                    self.model_name,
                    revision=self.model_revision,
                    device_map=self.device_map,
                    local_files_only=self.local_files_only,
                    quantization_config=FineGrainedFP8Config(dequantize=True),
                )
        except Exception as error:
            raise ReaderError("ministral_init_failed", str(error)) from error
        return self._processor, self._model


class MinistralOCRServiceReader:
    """Call a warm local Ministral page reader over loopback HTTP."""

    name = "ministral-ocr"

    def __init__(
        self,
        service_url: str,
        *,
        prompt: str = MINISTRAL_OCR_PROMPT,
        timeout_seconds: float = 180,
    ) -> None:
        parsed = urlsplit(service_url)
        if parsed.scheme != "http" or parsed.hostname not in {
            "127.0.0.1",
            "localhost",
            "::1",
        }:
            raise ValueError("Ministral OCR service must use loopback HTTP")
        if not prompt.strip():
            raise ValueError("Ministral OCR prompt must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("Ministral OCR service timeout must be positive")
        self.service_url = service_url.rstrip("/")
        self.prompt = prompt
        self.timeout_seconds = timeout_seconds
        self.model_name = MINISTRAL_MODEL_ID
        self.model_revision = MINISTRAL_MODEL_REVISION
        self.local_files_only = True
        self._provenance = {
            "id": None,
            "loaded_from": self.service_url,
            "revision": None,
            "origin": None,
            "license": None,
            "identity_verified": False,
            "local_files_only": True,
            "source": "loopback_service",
        }

    @property
    def provenance(self) -> dict[str, Any]:
        return dict(self._provenance)

    def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
        try:
            with Image.open(image_path) as image:
                width, height = image.size
        except (OSError, ValueError) as error:
            raise ReaderError("ministral_image_failed", str(error)) from error
        text, input_format = self.generate_text(image_path, self.prompt)
        return [
            TextRegion(
                id=f"p{page_number}-page-1",
                kind="page_text",
                text=text,
                confidence=None,
                bounding_box=BoundingBox(0, 0, width, height),
                reading_order=1,
                provider=self.name,
                text_provenance={
                    "method": "ministral_full_page_generation",
                    "provider": self.name,
                    "model": self.provenance,
                    "prompt": self.prompt,
                    "processor_input": input_format,
                },
            )
        ]

    def generate_text(self, image_path: Path, prompt: str) -> tuple[str, str]:
        if not prompt.strip():
            raise ReaderError("ministral_prompt_failed", "Prompt must not be empty")
        try:
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
            try:
                encoded = _encode_png(image)
            finally:
                image.close()
        except (OSError, ValueError) as error:
            raise ReaderError("ministral_image_failed", str(error)) from error
        payload = json.dumps(
            {"image": encoded, "prompt": prompt},
            separators=(",", ":"),
        ).encode("utf-8")
        request = Request(
            f"{self.service_url}/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read(MAX_MINISTRAL_SERVICE_RESPONSE_BYTES + 1)
            if len(body) > MAX_MINISTRAL_SERVICE_RESPONSE_BYTES:
                raise ValueError("service response exceeded its limit")
            result = json.loads(body)
            if not isinstance(result, dict) or set(result) != {"text", "provenance"}:
                raise ValueError("service response did not match its contract")
            text = result["text"]
            provenance = result["provenance"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError("service returned no text")
            if not _valid_ministral_provenance(provenance):
                raise ValueError("service model provenance was invalid")
        except (HTTPError, URLError, OSError, ValueError) as error:
            raise ReaderError(
                "ministral_service_failed",
                "The local Ministral service did not return a valid result",
            ) from error
        self._provenance = {**provenance, "source": "loopback_service"}
        return text.strip(), "loopback_service"


def _valid_ministral_provenance(value: Any) -> bool:
    if not isinstance(value, dict) or value.get("local_files_only") is not True:
        return False
    if not isinstance(value.get("loaded_from"), str) or not value["loaded_from"]:
        return False
    verified = value.get("identity_verified") is True
    if not verified:
        return value.get("id") is None
    return all(
        (
            value.get("id") == MINISTRAL_MODEL_ID,
            value.get("revision") == MINISTRAL_MODEL_REVISION,
            value.get("origin") == MINISTRAL_MODEL_ORIGIN,
            value.get("license") == MINISTRAL_MODEL_LICENSE,
        )
    )


class MinistralStructuredImageCall:
    """Use the same local Ministral instance for strict JSON image decisions."""

    def __init__(
        self,
        reader: MinistralOCRReader | MinistralOCRServiceReader,
    ) -> None:
        self.reader = reader

    def __call__(
        self,
        image_path: Path,
        prompt: str,
        response_schema: Mapping[str, Any],
    ) -> OpenRouterResult:
        schema = json.dumps(response_schema, ensure_ascii=False, separators=(",", ":"))
        instruction = (
            f"{prompt}\nReturn only one JSON object matching this schema exactly: "
            f"{schema}"
        )
        started = time.perf_counter()
        try:
            text, _ = self.reader.generate_text(image_path, instruction)
            content = json.loads(text)
        except ReaderError as error:
            raise OpenRouterError(
                str(error),
                code=error.code,
                latency_ms=(time.perf_counter() - started) * 1000,
            ) from error
        except json.JSONDecodeError as error:
            raise OpenRouterError(
                "Local Ministral returned invalid structured JSON",
                code="ministral_structured_output_failed",
                latency_ms=(time.perf_counter() - started) * 1000,
            ) from error
        provenance = self.reader.provenance
        model = provenance.get("id") or provenance.get("loaded_from")
        return OpenRouterResult(
            content=content,
            model=str(model) if model else self.reader.name,
            provider="local",
            usage={},
            cost=None,
            latency_ms=(time.perf_counter() - started) * 1000,
            attempts=1,
        )


class GraniteDoclingReader:
    """Compact IBM page parser that serializes DocTags as text or Markdown."""

    name = "granite-docling"

    def __init__(
        self,
        *,
        model_name: str = "ibm-granite/granite-docling-258M",
        max_new_tokens: int = 8192,
        output_format: str = "text",
        local_files_only: bool = True,
        processor: object | None = None,
        model: object | None = None,
        converter: Callable[[str, Image.Image], str] | None = None,
    ) -> None:
        if output_format not in {"text", "markdown"}:
            raise ValueError("Granite output format must be text or markdown")
        if max_new_tokens <= 0:
            raise ValueError("Granite max_new_tokens must be positive")
        if not local_files_only:
            raise ValueError("Granite Docling requires local_files_only=True")
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.output_format = output_format
        self.local_files_only = True
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
                output_ids = generated_ids[0][prompt_length:]
                generated_token_count = len(output_ids)
                if generated_token_count >= self.max_new_tokens:
                    raise ReaderError(
                        "granite_output_truncated",
                        "Granite Docling reached max_new_tokens before completing DocTags",
                    )
                doctags = processor.decode(
                    output_ids,
                    skip_special_tokens=False,
                ).lstrip()
                text = converter(doctags, image)
            except ReaderError:
                raise
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
                text_provenance={
                    "method": "granite_docling_full_page_generation",
                    "provider": self.name,
                    "model": {
                        "id": (
                            self.model_name
                            if self.model_name == "ibm-granite/granite-docling-258M"
                            else None
                        ),
                        "loaded_from": self.model_name,
                        "origin": (
                            "IBM Research, United States"
                            if self.model_name == "ibm-granite/granite-docling-258M"
                            else None
                        ),
                        "license": (
                            "Apache-2.0"
                            if self.model_name == "ibm-granite/granite-docling-258M"
                            else None
                        ),
                        "identity_verified": (
                            self.model_name == "ibm-granite/granite-docling-258M"
                        ),
                        "local_files_only": True,
                    },
                    "generation": {
                        "max_new_tokens": self.max_new_tokens,
                        "generated_tokens": generated_token_count,
                        "finish_reason": "before_token_limit",
                        "raw_doctags": doctags,
                    },
                    "text_authority": "structure_challenger_only",
                },
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
                self._processor = AutoProcessor.from_pretrained(
                    self.model_name,
                    local_files_only=self.local_files_only,
                )
            if self._model is None:
                self._model = AutoModelForMultimodalLM.from_pretrained(
                    self.model_name,
                    torch_dtype="auto",
                    device_map="auto",
                    local_files_only=self.local_files_only,
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
        execution_lock: Any | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.language = language
        self.merge_level = merge_level
        self.batch_size = batch_size
        self._pipeline = pipeline
        self._lock = execution_lock if execution_lock is not None else threading.Lock()

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
                merge_level,
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
                            self.merge_level,
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
    merge_level: str,
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
        (left, top, right, bottom), box_adjustment = _nemotron_box(
            prediction,
            *image_size,
        )
        provenance: dict[str, object] = {"merge_level": merge_level}
        if box_adjustment is not None:
            provenance["bounding_box_adjustment"] = box_adjustment
        regions.append(
            TextRegion(
                id=f"p{page_number}-block-{len(regions) + 1}",
                kind="text",
                text=stripped_text,
                confidence=confidence,
                bounding_box=BoundingBox(left, top, right, bottom),
                reading_order=len(regions) + 1,
                provider=provider,
                text_provenance=provenance,
            )
        )
    return regions


def _nemotron_box(
    prediction: object,
    width: int,
    height: int,
) -> tuple[tuple[int, int, int, int], dict[str, object] | None]:
    left, lower, right, upper = _coordinate_values(
        [
            _required_value(prediction, "left"),
            _required_value(prediction, "lower"),
            _required_value(prediction, "right"),
            _required_value(prediction, "upper"),
        ]
    )
    if right <= left or upper <= lower:
        raise ValueError("Nemotron bounding box must have positive area")
    center_x = (left + right) / 2
    center_y = (lower + upper) / 2
    if not 0 <= center_x <= 1 or not 0 <= center_y <= 1:
        raise ValueError(
            "Nemotron bounding box must be anchored in the image; "
            f"received {(left, lower, right, upper)} for image {(width, height)}"
        )
    normalized_box = (left, lower, right, upper)
    clipped_box = tuple(
        min(1.0, max(0.0, coordinate)) for coordinate in (left, lower, right, upper)
    )
    left, lower, right, upper = clipped_box
    coordinates = _box_coordinates(
        (
            math.floor(left * width),
            math.floor(lower * height),
            math.ceil(right * width),
            math.ceil(upper * height),
        )
    )
    adjustment = None
    if clipped_box != normalized_box:
        adjustment = {
            "method": "clip_to_image",
            "normalized_box": list(normalized_box),
        }
    return coordinates, adjustment


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
