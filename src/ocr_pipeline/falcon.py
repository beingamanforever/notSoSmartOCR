"""Pinned Falcon-OCR identity and crop preparation shared by service and client."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from PIL import Image


FALCON_MODEL_ID = "tiiuae/Falcon-OCR"
FALCON_MODEL_REVISION = "fe757d59ecd79d4d68760162306a70a015761ad9"
FALCON_MODEL_ORIGIN = "TII, UAE"
FALCON_MODEL_LICENSE = "Apache-2.0"
FALCON_INFERENCE_REPOSITORY = "https://github.com/tiiuae/Falcon-Perception"
FALCON_INFERENCE_REPOSITORY_REVISION = "c457916c9974efbacfa91f0f6ecc2c49c6543e56"
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
FALCON_SERVICE_MAX_RESPONSE_BYTES = 8_000_000
FALCON_SPATIAL_PATCH_SIZE = 16
FALCON_MAX_IMAGE_SIZE = 1536
FALCON_MAX_NEW_TOKENS = 8192
FALCON_MAX_CONTEXT_TOKENS = 16384


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
