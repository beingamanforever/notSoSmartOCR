"""Pinned local Falcon-OCR reader for benchmark and review use."""

from __future__ import annotations

import base64
import io
import json
import math
import threading
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from PIL import Image

from .contracts import BoundingBox, TextRegion
from .providers import ReaderError

FALCON_MODEL_ID = "tiiuae/Falcon-OCR"
FALCON_MODEL_REVISION = "42ec56b72a23984ac059e7c8a6d397a8529423fe"
FALCON_MODEL_ORIGIN = "TII, UAE"
FALCON_MODEL_LICENSE = "Apache-2.0"
FALCON_INFERENCE_REPOSITORY = "https://github.com/tiiuae/Falcon-Perception"
FALCON_INFERENCE_REPOSITORY_REVISION = "59a845adfac23c684bc4beafd26b380cde5ddfc1"
FALCON_OCR_CATEGORIES = (
    "plain",
    "text",
    "table",
    "formula",
    "caption",
    "footnote",
    "list-item",
    "page-footer",
    "page-header",
    "section-header",
    "title",
)
FALCON_SERVICE_MAX_BATCH_ITEMS = 24
FALCON_SERVICE_MAX_RESPONSE_BYTES = 8_000_000
FALCON_SPATIAL_PATCH_SIZE = 16
FALCON_CATEGORY_TOKEN_LIMITS = {
    "plain": 1536,
    "text": 256,
    "table": 1024,
    "formula": 512,
    "caption": 192,
    "footnote": 192,
    "list-item": 192,
    "page-footer": 128,
    "page-header": 128,
    "section-header": 128,
    "title": 128,
}


@dataclass(frozen=True)
class FalconGenerationConfig:
    max_new_tokens: int = 3072
    temperature: float = 0.0
    max_dimension: int = 1024
    compile: bool = False


class FalconOCRReader:
    """Use Falcon-OCR core generation without its optional layout pipeline."""

    name = "falcon-ocr"

    def __init__(
        self,
        *,
        model_name_or_path: str | Path = FALCON_MODEL_ID,
        local_model_path: str | Path | None = None,
        model_revision: str = FALCON_MODEL_REVISION,
        category: str = "plain",
        device_map: str = "cuda:0",
        local_files_only: bool = True,
        max_new_tokens: int = 3072,
        temperature: float = 0.0,
        max_dimension: int = 1024,
        compile: bool = False,
        model: object | None = None,
    ) -> None:
        model_source = str(model_name_or_path)
        official_model = model_source == FALCON_MODEL_ID
        local_source = str(local_model_path) if local_model_path is not None else None
        if not local_files_only:
            raise ValueError("Falcon-OCR requires local_files_only=True")
        if official_model and model_revision != FALCON_MODEL_REVISION:
            raise ValueError("Falcon-OCR must use the pinned model revision")
        if local_source is not None and not Path(local_source).is_dir():
            raise FileNotFoundError(
                f"Local Falcon-OCR model directory was not found: {local_source}"
            )
        if not official_model and not Path(model_source).is_dir():
            raise FileNotFoundError(
                f"Local Falcon-OCR model directory was not found: {model_source}"
            )
        if category not in FALCON_OCR_CATEGORIES:
            raise ValueError(f"Unsupported Falcon-OCR category: {category}")
        if not device_map.strip() or not device_map.startswith("cuda"):
            raise ValueError("Falcon-OCR device_map must select CUDA")
        if max_new_tokens <= 0 or max_new_tokens > 3072:
            raise ValueError("Falcon-OCR max_new_tokens must be from 1 to 3072")
        if temperature < 0:
            raise ValueError("Falcon-OCR temperature must not be negative")
        if max_dimension <= 0 or max_dimension % FALCON_SPATIAL_PATCH_SIZE != 0:
            raise ValueError(
                "Falcon-OCR max_dimension must be a positive multiple of 16"
            )

        self.model_name_or_path = model_source
        self.local_model_path = local_source
        self.model_revision = model_revision
        self.category = category
        self.device_map = device_map
        self.local_files_only = True
        self.generation_config = FalconGenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            max_dimension=max_dimension,
            compile=compile,
        )
        self._model = model
        self._lock = threading.Lock()

    @property
    def provenance(self) -> dict[str, Any]:
        official_model = self.model_name_or_path == FALCON_MODEL_ID
        return {
            "id": FALCON_MODEL_ID if official_model else None,
            "loaded_from": self.local_model_path or self.model_name_or_path,
            "revision": self.model_revision if official_model else "unverified",
            "origin": FALCON_MODEL_ORIGIN if official_model else "unverified",
            "license": FALCON_MODEL_LICENSE if official_model else "unverified",
            "identity_verified": official_model,
            "local_files_only": True,
            "inference_repository_audit_reference": {
                "url": FALCON_INFERENCE_REPOSITORY,
                "revision": FALCON_INFERENCE_REPOSITORY_REVISION,
                "loaded_code_match": "unverified",
            },
        }

    @property
    def generation(self) -> dict[str, object]:
        return {"category": self.category, **asdict(self.generation_config)}

    def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
        try:
            with Image.open(image_path) as source:
                width, height = source.size
                image = source.convert("RGB")
        except (OSError, ValueError) as error:
            raise ReaderError("falcon_image_failed", str(error)) from error

        try:
            text = self.transcribe_crops([image], [self.category])[0]
        finally:
            image.close()
        generation = {
            **self.generation,
            "effective_max_new_tokens": falcon_token_limit(
                self.category, self.generation_config.max_new_tokens
            ),
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
                text_provenance={
                    "method": "falcon_core_full_page_generation",
                    "provider": self.name,
                    "model": self.provenance,
                    "category": self.category,
                    "generation": generation,
                    "output_validation": {
                        "raw_response_preserved": True,
                        "nonempty": True,
                        "exact_repetition_loop": False,
                        "termination_observable": False,
                        "truncation_observable": False,
                    },
                },
            )
        ]

    def transcribe_crops(
        self,
        images: Sequence[Image.Image],
        categories: Sequence[str],
    ) -> list[str]:
        """Transcribe caller-owned crops without creating or changing geometry."""
        if not images:
            if categories:
                raise ValueError(
                    "Falcon-OCR images and categories must have equal lengths"
                )
            return []
        if len(images) != len(categories):
            raise ValueError("Falcon-OCR images and categories must have equal lengths")
        invalid_category = next(
            (item for item in categories if item not in FALCON_OCR_CATEGORIES),
            None,
        )
        if invalid_category is not None:
            raise ValueError(f"Unsupported Falcon-OCR category: {invalid_category}")
        if not all(isinstance(image, Image.Image) for image in images):
            raise TypeError("Falcon-OCR crops must be PIL images")

        options = asdict(self.generation_config)
        options["max_new_tokens"] = max(
            falcon_token_limit(category, int(options["max_new_tokens"]))
            for category in categories
        )
        prepared = [
            _prepare_falcon_crop(image, self.generation_config.max_dimension)
            for image in images
        ]
        with self._lock:
            model = self._model or self._initialize_model()
            try:
                outputs = model.generate(
                    [item.image for item in prepared],
                    category=list(categories),
                    **options,
                )
            except Exception as error:
                raise ReaderError("falcon_predict_failed", str(error)) from error
            finally:
                for item in prepared:
                    if item.owned:
                        item.image.close()
        if not isinstance(outputs, list) or len(outputs) != len(images):
            raise ReaderError(
                "falcon_output_failed",
                "Falcon-OCR returned the wrong number of crop transcriptions",
            )

        texts = []
        for text in outputs:
            if not isinstance(text, str) or not text.strip():
                raise ReaderError(
                    "falcon_output_failed", "Falcon-OCR returned empty text"
                )
            if _is_exact_repetition_loop(text):
                raise ReaderError(
                    "falcon_repetition_loop",
                    "Falcon-OCR returned an exact repetition loop",
                )
            texts.append(text)
        return texts

    def _initialize_model(self) -> object:
        try:
            import torch
            from transformers import AutoModelForCausalLM
        except (ImportError, OSError) as error:
            raise ReaderError("falcon_import_failed", str(error)) from error

        options: dict[str, object] = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
            "device_map": self.device_map,
            "local_files_only": True,
        }
        load_source = self.local_model_path or self.model_name_or_path
        if self.model_name_or_path == FALCON_MODEL_ID and self.local_model_path is None:
            options["revision"] = FALCON_MODEL_REVISION
        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                load_source,
                **options,
            )
        except Exception as error:
            raise ReaderError("falcon_init_failed", str(error)) from error
        return self._model


class FalconOCRServiceReader:
    """Call a warm Falcon-OCR core model over loopback HTTP."""

    name = "falcon-ocr"

    def __init__(
        self,
        service_url: str,
        *,
        category: str = "plain",
        timeout_seconds: float = 180,
        max_batch_items: int = FALCON_SERVICE_MAX_BATCH_ITEMS,
    ) -> None:
        parsed = urlsplit(service_url)
        if parsed.scheme != "http" or parsed.hostname not in {
            "127.0.0.1",
            "localhost",
            "::1",
        }:
            raise ValueError("Falcon-OCR service must use loopback HTTP")
        if category not in FALCON_OCR_CATEGORIES:
            raise ValueError(f"Unsupported Falcon-OCR category: {category}")
        if timeout_seconds <= 0:
            raise ValueError("Falcon-OCR service timeout must be positive")
        if max_batch_items <= 0:
            raise ValueError("Falcon-OCR service batch limit must be positive")
        self.service_url = service_url.rstrip("/")
        self.category = category
        self.timeout_seconds = timeout_seconds
        self.max_batch_items = max_batch_items
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
        self._generation_config: dict[str, object] = {}

    @property
    def provenance(self) -> dict[str, Any]:
        return dict(self._provenance)

    @property
    def generation(self) -> dict[str, object]:
        return {"category": self.category, **self._generation_config}

    def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
        try:
            with Image.open(image_path) as source:
                width, height = source.size
                image = source.convert("RGB")
        except (OSError, ValueError) as error:
            raise ReaderError("falcon_image_failed", str(error)) from error
        try:
            text = self.transcribe_crops([image], [self.category])[0]
        finally:
            image.close()
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
                    "method": "falcon_core_full_page_generation",
                    "provider": self.name,
                    "model": self.provenance,
                    "category": self.category,
                    "generation": self.generation,
                    "output_validation": _output_validation(text),
                },
            )
        ]

    def transcribe_crops(
        self,
        images: Sequence[Image.Image],
        categories: Sequence[str],
    ) -> list[str]:
        if not images or len(images) != len(categories):
            raise ValueError("Falcon-OCR images and categories must have equal lengths")
        if len(images) > self.max_batch_items:
            raise ValueError("Falcon-OCR crop batch exceeded its configured limit")
        invalid_category = next(
            (item for item in categories if item not in FALCON_OCR_CATEGORIES),
            None,
        )
        if invalid_category is not None:
            raise ValueError(f"Unsupported Falcon-OCR category: {invalid_category}")
        if not all(isinstance(image, Image.Image) for image in images):
            raise TypeError("Falcon-OCR crops must be PIL images")

        payload = json.dumps(
            {
                "images": [_encode_png(image) for image in images],
                "categories": list(categories),
            },
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
                body = response.read(FALCON_SERVICE_MAX_RESPONSE_BYTES + 1)
            if len(body) > FALCON_SERVICE_MAX_RESPONSE_BYTES:
                raise ValueError("service response exceeded its limit")
            result = json.loads(body)
            if not isinstance(result, dict) or set(result) != {
                "generation_config",
                "provenance",
                "texts",
            }:
                raise ValueError("service response did not match its contract")
            texts = result["texts"]
            provenance = result["provenance"]
            generation_config = result["generation_config"]
            if (
                not isinstance(texts, list)
                or len(texts) != len(images)
                or any(not isinstance(text, str) or not text.strip() for text in texts)
                or any(_is_exact_repetition_loop(text) for text in texts)
            ):
                raise ValueError("service returned invalid text")
            if not _valid_service_provenance(provenance):
                raise ValueError("service model provenance was invalid")
            if not isinstance(generation_config, dict):
                raise ValueError("service generation config was invalid")
        except (HTTPError, URLError, OSError, ValueError) as error:
            raise ReaderError(
                "falcon_service_failed",
                "The local Falcon-OCR service did not return a valid result",
            ) from error
        self._provenance = {**provenance, "source": "loopback_service"}
        self._generation_config = dict(generation_config)
        return texts


def _is_exact_repetition_loop(text: str) -> bool:
    """Detect a whole output made from at least four exact repeated units."""
    period = (text + text).find(text, 1)
    return (
        period < len(text)
        and len(text) % period == 0
        and len(text) // period >= 4
        and bool(text[:period].strip())
    )


@dataclass(frozen=True)
class _PreparedCrop:
    image: Image.Image
    owned: bool


def _prepare_falcon_crop(image: Image.Image, max_dimension: int) -> _PreparedCrop:
    """Pad thin crops so Falcon's aspect resize retains one spatial patch."""
    width, height = image.size
    longest = max(width, height)
    minimum_short_side = max(
        FALCON_SPATIAL_PATCH_SIZE,
        math.ceil(longest * FALCON_SPATIAL_PATCH_SIZE / max_dimension),
    )
    target_width = max(width, minimum_short_side)
    target_height = max(height, minimum_short_side)
    if (target_width, target_height) == image.size:
        return _PreparedCrop(image, False)

    padded = Image.new("RGB", (target_width, target_height), "white")
    converted = image.convert("RGB")
    try:
        padded.paste(
            converted,
            ((target_width - width) // 2, (target_height - height) // 2),
        )
    finally:
        converted.close()
    return _PreparedCrop(padded, True)


def falcon_token_limit(category: str, configured_limit: int) -> int:
    """Bound generation by the routed crop task, never above the configured cap."""
    if category not in FALCON_OCR_CATEGORIES:
        raise ValueError(f"Unsupported Falcon-OCR category: {category}")
    if configured_limit <= 0:
        raise ValueError("Falcon-OCR token limit must be positive")
    return min(configured_limit, FALCON_CATEGORY_TOKEN_LIMITS[category])


def _output_validation(text: str) -> dict[str, bool]:
    return {
        "raw_response_preserved": True,
        "nonempty": bool(text.strip()),
        "exact_repetition_loop": _is_exact_repetition_loop(text),
        "termination_observable": False,
        "truncation_observable": False,
    }


def _encode_png(image: Image.Image) -> str:
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return base64.b64encode(stream.getvalue()).decode("ascii")


def _valid_service_provenance(value: Any) -> bool:
    if not isinstance(value, dict) or value.get("local_files_only") is not True:
        return False
    loaded_from = value.get("loaded_from")
    if not isinstance(loaded_from, str) or not loaded_from:
        return False
    if value.get("identity_verified") is not True:
        return value.get("id") is None
    return is_verified_falcon_model_provenance(value)


def is_verified_falcon_model_provenance(value: Any) -> bool:
    """Return whether provenance identifies the pinned official Falcon model."""
    return isinstance(value, dict) and all(
        (
            value.get("id") == FALCON_MODEL_ID,
            value.get("revision") == FALCON_MODEL_REVISION,
            value.get("origin") == FALCON_MODEL_ORIGIN,
            value.get("license") == FALCON_MODEL_LICENSE,
            value.get("identity_verified") is True,
            value.get("local_files_only") is True,
            isinstance(value.get("loaded_from"), str),
            bool(value.get("loaded_from")),
        )
    )
