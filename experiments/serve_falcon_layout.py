"""Serve the official Falcon-Perception layout-aware OCR engine over loopback HTTP.

The engine detects regions with Heron, crops each one, and reads it with a
category-specific prompt, batching the crops continuously. It needs its own torch pin, so
it runs in a separate process like the Falcon crop reader and the Arctic-TILT service.

`score` is Heron's detection confidence. `generation` contains Falcon's native
token probabilities and termination evidence, kept separate from detection scores.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import io
import json
import logging
import math
import os
import re
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import wait
from contextlib import contextmanager, nullcontext
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Protocol

from PIL import Image, UnidentifiedImageError

from ocr_pipeline.falcon import (
    FALCON_INFERENCE_REPOSITORY_REVISION,
    FALCON_MAX_CONTEXT_TOKENS,
    FALCON_MAX_IMAGE_SIZE,
    FALCON_MAX_NEW_TOKENS,
    FALCON_MODEL_REVISION,
    FALCON_OCR_CATEGORIES,
)
from ocr_pipeline.heron_layout import DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION

# Detection and crop options the caller may override per request, so a coverage sweep
# does not need a restart per combination.
LAYOUT_OPTIONS = {
    "layout_threshold": float,
    "max_image_size": int,
    "min_image_size": int,
    # A dense table stops mid-markup at the official 4096, losing its remaining rows.
    "max_new_tokens": int,
    # Not a detection option: asks for the full-page read alongside the region reads.
    "page_text": bool,
}
ROUTE_TRAINING_ACTIONS = {
    "layout": "layout_selective_qwen",
    "plain": "plain_falcon",
}
ROUTE_EXTRACTOR_VERSION = 1
HERON_ROUTE_CLASSES = (
    "caption",
    "footnote",
    "formula",
    "list_item",
    "page_footer",
    "page_header",
    "picture",
    "section_header",
    "table",
    "text",
    "title",
    "document_index",
    "code",
    "checkbox_selected",
    "checkbox_unselected",
    "form",
    "key_value_region",
)
HERON_ROUTE_FEATURE_NAMES = (
    "image_log1p_width",
    "image_log1p_height",
    "image_aspect_ratio",
    "detections_per_megapixel",
    *(
        f"{label}.{suffix}"
        for label in HERON_ROUTE_CLASSES
        for suffix in (
            "count_fraction",
            "max_score",
            "area_ratio_sum",
            "area_ratio_max",
        )
    ),
    *(
        f"{label}.{suffix}"
        for label in HERON_ROUTE_CLASSES
        for suffix in ("class_score_mean", "class_score_max")
    ),
)
MAX_REQUEST_BYTES = 64_000_000
MAX_IMAGE_PIXELS = 32_000_000
LOGGER = logging.getLogger(__name__)
# olmocr's production fix for VLM decode repetition: escalate temperature and retry.
RETRY_TEMPERATURES = (0.2, 0.5, 0.8)
MIN_DECODER_CYCLE_TOKENS = 256


def _cycle_step(
    state: tuple[int | None, int | None, int, int], token: int
) -> tuple[int, int | None, int, int]:
    last, previous, one, two = state
    return (
        token,
        last,
        one + 1 if token == last else 1,
        two + 1 if token == previous else 2,
    )


def _decoder_cycle(token_ids: list[int]) -> bool:
    """Detect a sustained trailing or near-complete decoder cycle."""
    if len(token_ids) < MIN_DECODER_CYCLE_TOKENS:
        return False
    state: tuple[int | None, int | None, int, int] = (None, None, 0, 0)
    longest = 0
    for token in token_ids:
        state = _cycle_step(state, token)
        longest = max(longest, state[2], state[3])
    return _cycle_detected(state, longest, len(token_ids))


def _cycle_detected(
    state: tuple[int | None, int | None, int, int], longest: int, length: int
) -> bool:
    return length >= MIN_DECODER_CYCLE_TOKENS and (
        max(state[2:]) >= MIN_DECODER_CYCLE_TOKENS or longest * 10 >= length * 9
    )


class DecodedText(str):
    """Keep per-sequence evidence through the official engine's string interface."""

    def __new__(
        cls, text: str, generation: dict[str, Any], request_idx: int | None = None
    ):
        value = super().__new__(cls, text)
        value.generation = generation
        value.request_idx = request_idx
        return value


class LayoutEngine(Protocol):
    def generate_with_layout(
        self, images: list[Any], **options: Any
    ) -> list[list[dict]]: ...

    def generate_plain(self, images: list[Any], **options: Any) -> list[str]: ...


def load_route_model(path: os.PathLike[str] | str) -> dict[str, Any]:
    """Load the dependency-free binary router exported by the offline trainer."""
    try:
        with open(path, encoding="utf-8") as source:
            model = json.load(source)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid Falcon route model: {error}") from error
    if not isinstance(model, dict):
        raise ValueError("Falcon route model must be a JSON object")
    feature_names = model.get("feature_names")
    coefficients = model.get("coefficients")
    means = model.get("means")
    scales = model.get("scales")
    cost_head = model.get("cost_head")
    extractor = model.get("extractor")
    if (
        model.get("version") != 1
        or not isinstance(model.get("id"), str)
        or not model["id"].strip()
        or model.get("status") != "calibrated"
        or model.get("training_actions") != ROUTE_TRAINING_ACTIONS
        or not isinstance(feature_names, list)
        or feature_names != list(HERON_ROUTE_FEATURE_NAMES)
        or not all(isinstance(values, list) for values in (coefficients, means, scales))
        or any(
            len(values) != len(feature_names)
            for values in (coefficients, means, scales)
        )
        or not isinstance(cost_head, dict)
        or not isinstance(cost_head.get("coefficients"), list)
        or len(cost_head["coefficients"]) != len(feature_names)
        or not _valid_extractor(extractor)
    ):
        raise ValueError("Invalid Falcon route model schema")
    numbers = [
        *coefficients,
        *means,
        *scales,
        model.get("intercept"),
        model.get("threshold"),
        model.get("cost_tradeoff"),
        *cost_head["coefficients"],
        cost_head.get("intercept"),
    ]
    if any(
        type(value) not in (int, float) or not math.isfinite(value) for value in numbers
    ):
        raise ValueError("Falcon route model values must be finite numbers")
    if any(scale <= 0 for scale in scales):
        raise ValueError("Falcon route model scales must be positive")
    if model["cost_tradeoff"] < 0:
        raise ValueError("Falcon route model cost tradeoff must be nonnegative")
    return model


def _valid_extractor(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and value.get("version") == ROUTE_EXTRACTOR_VERSION
        and isinstance(value.get("heron_model_id"), str)
        and bool(value["heron_model_id"])
        and isinstance(value.get("heron_model_revision"), str)
        and bool(re.fullmatch(r"[0-9a-f]{40}", value["heron_model_revision"]))
        and type(value.get("layout_score_threshold")) in (int, float)
        and math.isfinite(value["layout_score_threshold"])
        and 0 <= value["layout_score_threshold"] <= 1
    )


def _route_page(
    model: dict[str, Any], image: Image.Image, detections: list[dict[str, Any]]
) -> dict[str, Any]:
    features = route_features(*image.size, detections, model["feature_names"])
    quality_delta = float(model["intercept"]) + sum(
        float(coefficient) * (feature - float(mean)) / float(scale)
        for coefficient, feature, mean, scale in zip(
            model["coefficients"],
            features,
            model["means"],
            model["scales"],
            strict=True,
        )
    )
    cost_saving = float(model["cost_head"]["intercept"]) + sum(
        float(coefficient) * (feature - float(mean)) / float(scale)
        for coefficient, feature, mean, scale in zip(
            model["cost_head"]["coefficients"],
            features,
            model["means"],
            model["scales"],
            strict=True,
        )
    )
    value = quality_delta + float(model["cost_tradeoff"]) * cost_saving
    action = "plain" if value > model["threshold"] else "layout"
    return {
        "action": action,
        "training_action": model["training_actions"][action],
        "plain_quality_delta": quality_delta,
        "plain_cost_saving": cost_saving,
        "plain_value": value,
        "cost_tradeoff": float(model["cost_tradeoff"]),
        "threshold": float(model["threshold"]),
        "model": {"id": model["id"], "version": model["version"]},
    }


def route_feature_values(
    width: int,
    height: int,
    detections: list[dict[str, Any]],
) -> dict[str, float]:
    if width <= 0 or height <= 0:
        raise ValueError("Route image dimensions must be positive")
    page_area = width * height
    normalized = []
    for detection in detections:
        category = str(detection.get("label") or detection.get("category") or "")
        category = "code" if category == "algorithm" else category.replace("-", "_")
        bbox = detection.get("box") or detection.get("bbox")
        score = detection.get("score")
        if (
            not category
            or not isinstance(bbox, (list, tuple))
            or len(bbox) != 4
            or type(score) not in (int, float)
            or not math.isfinite(score)
        ):
            raise ValueError("Route proposals must contain category, bbox, and score")
        try:
            left, top, right, bottom = (float(value) for value in bbox)
        except (TypeError, ValueError) as error:
            raise ValueError("Route proposal bbox must be numeric") from error
        if not all(math.isfinite(value) for value in (left, top, right, bottom)):
            raise ValueError("Route proposal bbox must be finite")
        raw_class_scores = detection.get("class_scores") or {}
        if not isinstance(raw_class_scores, dict):
            raise ValueError("Route proposal class_scores must be an object")
        clean_class_scores = {}
        for label, value in raw_class_scores.items():
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("Route proposal class scores must be finite numbers")
            clean_class_scores[str(label).replace("-", "_")] = float(value)
        normalized.append(
            {
                "label": category,
                "score": float(score),
                "area_ratio": max(0.0, right - left)
                * max(0.0, bottom - top)
                / page_area,
                "class_scores": clean_class_scores,
            }
        )
    count = max(1, len(normalized))
    values = {
        "image_log1p_width": math.log1p(width),
        "image_log1p_height": math.log1p(height),
        "image_aspect_ratio": width / height,
        "detections_per_megapixel": len(normalized) / max(1.0, page_area / 1_000_000),
    }
    for label in HERON_ROUTE_CLASSES:
        selected = [item for item in normalized if item["label"] == label]
        scores = [item["score"] for item in selected]
        areas = [item["area_ratio"] for item in selected]
        values[f"{label}.count_fraction"] = len(selected) / count
        values[f"{label}.max_score"] = max(scores, default=0.0)
        values[f"{label}.area_ratio_sum"] = sum(areas)
        values[f"{label}.area_ratio_max"] = max(areas, default=0.0)
    for label in HERON_ROUTE_CLASSES:
        scores = [item["class_scores"].get(label, 0.0) for item in normalized]
        values[f"{label}.class_score_mean"] = (
            sum(scores) / len(scores) if scores else 0.0
        )
        values[f"{label}.class_score_max"] = max(scores, default=0.0)
    return values


def route_features(
    width: int,
    height: int,
    detections: list[dict[str, Any]],
    names: list[str],
) -> list[float]:
    if names != list(HERON_ROUTE_FEATURE_NAMES):
        raise ValueError("Route feature names do not match the measured schema")
    values = route_feature_values(width, height, detections)
    return [values[name] for name in names]


def _clamp_box(
    bbox: Sequence[float], width: int, height: int
) -> tuple[int, int, int, int] | None:
    left = max(0, min(width, int(bbox[0])))
    top = max(0, min(height, int(bbox[1])))
    right = max(0, min(width, int(bbox[2])))
    bottom = max(0, min(height, int(bbox[3])))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _exclusive_form_indices(items: list[dict[str, Any]]) -> set[int]:
    """Keep learned form crops as source evidence for structured refinement."""
    return {
        index
        for index, item in enumerate(items)
        if item["label"] not in {"key_value_region", "document_index"}
    }


def _route_detection(proposals: list[dict[str, Any]]) -> dict[str, Any]:
    kept = _exclusive_form_indices(proposals)
    raw = [
        {
            "label": item["label"],
            "box": list(item["box"]),
            "score": item["score"],
            **{key: item[key] for key in ("query_id", "class_scores") if key in item},
        }
        for item in proposals
    ]
    layout = [
        {
            "category": "algorithm"
            if item["label"] == "code"
            else item["label"].replace("_", "-"),
            "bbox": list(item["box"]),
            "score": item["score"],
            **{key: item[key] for key in ("query_id", "class_scores") if key in item},
        }
        for index, item in enumerate(proposals)
        if index in kept
    ]
    return {"raw": raw, "layout": layout}


def create_server(
    engine: LayoutEngine,
    host: str,
    port: int,
    *,
    model: dict[str, Any],
    tokenizer: Any = None,
    detect_layout: Callable[[list[Image.Image]], list[list[dict[str, Any]]]]
    | None = None,
    detect_route: Callable[[list[Image.Image]], list[dict[str, Any]]] | None = None,
    layout: dict[str, Any] | None = None,
    category_by_layout: dict[str, str] | None = None,
    max_request_bytes: int = MAX_REQUEST_BYTES,
    max_image_pixels: int = MAX_IMAGE_PIXELS,
    feature_config: dict[str, Any] | None = None,
    route_model: dict[str, Any] | None = None,
    extractor_metadata: dict[str, Any] | None = None,
) -> ThreadingHTTPServer:
    layout = dict(layout or {})
    category_by_layout = dict(category_by_layout or {})
    execution_lock = threading.Lock()
    unavailable = threading.Event()
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Falcon layout service must bind to loopback")
    if route_model is not None and detect_route is None:
        raise ValueError("A Falcon route model requires one-pass Heron route detection")
    if route_model is not None and route_model["extractor"] != extractor_metadata:
        raise ValueError("Falcon route model extractor does not match the service")

    @contextmanager
    def execute_engine():
        import torch

        with execution_lock, torch.inference_mode():
            if unavailable.is_set():
                raise RuntimeError("Falcon engine recovery failed; restart required")
            try:
                yield
            except Exception:
                try:
                    _release_failed_generation(engine)
                except Exception:
                    unavailable.set()
                    LOGGER.exception("Falcon engine recovery failed")
                raise

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/health":
                self._write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return
            self._write_json(
                HTTPStatus.SERVICE_UNAVAILABLE
                if unavailable.is_set()
                else HTTPStatus.OK,
                {
                    "status": "unavailable" if unavailable.is_set() else "ready",
                    "pid": os.getpid(),
                    "model": model,
                    **(
                        {
                            "routing_model": {
                                "id": route_model["id"],
                                "version": route_model["version"],
                            }
                        }
                        if route_model is not None
                        else {}
                    ),
                },
            )

        def do_POST(self) -> None:  # noqa: N802
            if self.path == "/generate":
                self._generate_crops()
                return
            if self.path != "/read":
                self._write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return
            image = None
            try:
                image, overrides = self._read_input()
            except (ValueError, OSError, UnidentifiedImageError, binascii.Error):
                self._write_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_request"})
                return
            want_page = bool(overrides.pop("page_text", False))
            if route_model is not None and want_page:
                image.close()
                self._write_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "page_text_not_available_with_learned_routing"},
                )
                return
            cap = int(
                {**layout, **overrides}.get("max_new_tokens") or FALCON_MAX_NEW_TOKENS
            )
            try:
                options = {**layout, **overrides}
                routing = None
                if route_model is not None:
                    with execute_engine():
                        detections = detect_route([image])
                        if not isinstance(detections, list) or len(detections) != 1:
                            raise ValueError(
                                "Heron must return one proposal list per page"
                            )
                        detection = detections[0]
                        if (
                            not isinstance(detection, dict)
                            or not isinstance(detection.get("raw"), list)
                            or not isinstance(detection.get("layout"), list)
                        ):
                            raise ValueError("Heron route detection is malformed")
                        routing = _route_page(route_model, image, detection["raw"])
                        if routing["action"] == "plain":
                            reading = engine.generate_plain(
                                images=[image], use_tqdm=False
                            )[0]
                            if not isinstance(reading, str) or not reading.strip():
                                raise ValueError(
                                    "Falcon plain decoder returned no text"
                                )
                            page_text = ""
                            built = [
                                {
                                    "category": "page_text",
                                    "bbox": [
                                        0.0,
                                        0.0,
                                        float(image.width),
                                        float(image.height),
                                    ],
                                    "text": reading,
                                    "truncated": self._truncated(reading),
                                    "looped": self._looped(reading, "text"),
                                    "generation": getattr(reading, "generation", {}),
                                }
                            ]
                        else:
                            options["detections"] = [detection["layout"]]
                            elements = engine.generate_with_layout(
                                images=[image], use_tqdm=False, **options
                            )[0]
                            page_text = ""
                            built = [
                                self._build_element(element, image, cap)
                                for element in elements
                                if element.get("bbox")
                            ]
                else:
                    if detect_layout is not None:
                        options["detections"] = detect_layout([image])
                    with execute_engine():
                        elements = engine.generate_with_layout(
                            images=[image], use_tqdm=False, **options
                        )[0]
                        # Optional comparison path retained for existing callers.
                        page_text = (
                            engine.generate_plain(images=[image], use_tqdm=False)[0]
                            if want_page
                            else ""
                        )
                        built = [
                            self._build_element(element, image, cap)
                            for element in elements
                            if element.get("bbox")
                        ]
            except Exception:
                LOGGER.exception("Falcon layout read failed")
                self._write_json(
                    HTTPStatus.SERVICE_UNAVAILABLE
                    if unavailable.is_set()
                    else HTTPStatus.UNPROCESSABLE_ENTITY,
                    {
                        "error": "falcon_engine_unavailable"
                        if unavailable.is_set()
                        else "falcon_layout_failed"
                    },
                )
                return
            finally:
                image.close()
            self._write_json(
                HTTPStatus.OK,
                {
                    "elements": built,
                    "page_text": page_text,
                    "model": model,
                    **({"routing": routing} if routing is not None else {}),
                },
            )

        def _generate_crops(self) -> None:
            from ocr_pipeline.falcon import _prepare_falcon_crop

            images = []
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= max_request_bytes:
                    raise ValueError("invalid content length")
                payload = json.loads(self.rfile.read(length))
                encoded, categories = payload.get("images"), payload.get("categories")
                return_generation = payload.get("return_generation", False)
                return_features = payload.get("return_features", False)
                return_post_features = payload.get("return_post_features", False)
                if (
                    not isinstance(return_generation, bool)
                    or not isinstance(return_features, bool)
                    or not isinstance(return_post_features, bool)
                    or return_post_features
                    and not return_features
                    or return_features
                    and not return_generation
                    or not isinstance(encoded, list)
                    or not 0 < len(encoded) <= 24
                    or not isinstance(categories, list)
                    or len(encoded) != len(categories)
                    or any(
                        category not in FALCON_OCR_CATEGORIES for category in categories
                    )
                ):
                    raise ValueError("invalid crop batch")
                for item in encoded:
                    with Image.open(
                        io.BytesIO(base64.b64decode(item, validate=True))
                    ) as source:
                        if source.width * source.height > max_image_pixels:
                            raise ValueError("image exceeded its pixel limit")
                        rgb = source.convert("RGB")
                        prepared = _prepare_falcon_crop(
                            rgb, layout.get("max_image_size", FALCON_MAX_IMAGE_SIZE)
                        )
                        if prepared.owned:
                            rgb.close()
                        images.append(prepared.image)
            except (ValueError, TypeError, AttributeError, OSError, binascii.Error):
                for image in images:
                    image.close()
                self._write_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_request"})
                return
            captured = None
            generation_seconds = 0.0
            cap = None
            try:
                cap = int(layout.get("max_new_tokens", FALCON_MAX_NEW_TOKENS))
                with execute_engine():
                    scope = nullcontext()
                    if return_features:
                        from ocr_pipeline.falcon_features import capture_prefill

                        scope = capture_prefill(engine, post=return_post_features)
                    with scope as captured:
                        started = time.perf_counter()
                        try:
                            texts = engine.generate_plain(
                                images=images,
                                category=categories,
                                max_new_tokens=cap,
                                use_tqdm=False,
                            )
                        finally:
                            generation_seconds = time.perf_counter() - started
                    if len(texts) != len(images):
                        raise ValueError(
                            "crop generation returned the wrong batch size"
                        )
                    if return_generation:
                        indices = [getattr(text, "request_idx", None) for text in texts]
                        if any(type(index) is not int for index in indices) or sorted(
                            indices
                        ) != list(range(len(images))):
                            raise ValueError(
                                "crop generation returned ambiguous identities"
                            )
                        texts = [text for _, text in sorted(zip(indices, texts))]
                    elif any(
                        not text.strip()
                        or self._truncated(text)
                        or self._looped(text, category)
                        for text, category in zip(texts, categories, strict=True)
                    ):
                        raise ValueError("crop generation did not finish cleanly")
                response = {"texts": texts, "model": model}
                if return_generation:
                    response["results"] = [
                        self._crop_result(text, index, categories[index])
                        for index, text in enumerate(texts)
                    ]
                if return_features:
                    self._attach_features(response, captured, generation_seconds, cap)
                    if return_post_features:
                        self._attach_post_features(response, captured)
                self._write_json(HTTPStatus.OK, response)
            except Exception:
                LOGGER.exception("Falcon crop read failed")
                error = (
                    "falcon_engine_unavailable"
                    if unavailable.is_set()
                    else "falcon_crop_failed"
                )
                response = {"error": error}
                if return_generation:
                    response["model"] = model
                    response["results"] = [
                        {
                            "index": index,
                            "text": None,
                            "generation": None,
                            "status": "failed",
                            "error": error,
                        }
                        for index in range(len(images))
                    ]
                if return_features:
                    self._attach_features(response, captured, generation_seconds, cap)
                    if return_post_features:
                        self._attach_post_features(response, captured)
                self._write_json(
                    HTTPStatus.SERVICE_UNAVAILABLE
                    if unavailable.is_set()
                    else HTTPStatus.UNPROCESSABLE_ENTITY,
                    response,
                )
            finally:
                for image in images:
                    image.close()

        def _attach_features(self, response, captured, generation_seconds, cap):
            records = captured.records if captured is not None else {}
            complete = len(records) == len(response["results"])
            for result in response["results"]:
                index = result["index"]
                record = records.get(index)
                complete = complete and record is not None
                result["features"] = {
                    "request_idx": index,
                    "status": "success" if record is not None else "failed",
                    "error": None
                    if record is not None
                    else "initial_prefill_not_captured",
                    "values": record["features"].tolist()
                    if record is not None
                    else None,
                    "association": {
                        key: value for key, value in record.items() if key != "features"
                    }
                    if record is not None
                    else None,
                }
            response["feature_capture"] = {
                "status": "complete"
                if complete
                else "partial"
                if records
                else "missing",
                "unmatched_request_indices": sorted(
                    set(records) - {row["index"] for row in response["results"]}
                ),
                "config": {
                    "source": "live_paged_initial_prefill",
                    "representation": "last_prompt_normalized_hidden_then_mean_image_tokens",
                    "dimension": 1536,
                    "dtype": "float32",
                    "model": model,
                    "runtime": feature_config,
                    "max_image_size": layout.get(
                        "max_image_size", FALCON_MAX_IMAGE_SIZE
                    ),
                    "max_new_tokens": cap,
                },
                "timings": {
                    "meaning": "inclusive_host_seconds_not_gpu_kernel_timings",
                    "generation_seconds": generation_seconds,
                    "validation_seconds": captured.validation_seconds
                    if captured is not None
                    else None,
                    "pooling_seconds": captured.pooling_seconds
                    if captured is not None
                    else None,
                },
                "prefill_calls": captured.prefill_calls if captured is not None else 0,
                "forward_calls": captured.forward_calls if captured is not None else 0,
                "skipped_continuations": captured.skipped_continuations
                if captured is not None
                else 0,
            }

        def _attach_post_features(self, response, captured):
            records = captured.post_records if captured is not None else {}
            statuses = []
            for result in response["results"]:
                index = result["index"]
                record = records.get(index)
                status = record["status"] if record else "failed"
                statuses.append(status)
                result["post_features"] = {
                    "request_idx": index,
                    "status": status,
                    "error": record["error"]
                    if record
                    else "emission_decisions_not_captured",
                    "values": record["features"].tolist()
                    if record and record["features"] is not None
                    else None,
                    "association": {
                        key: value
                        for key, value in record.items()
                        if key not in {"features", "status", "error"}
                    }
                    if record
                    else None,
                }
            response["post_feature_capture"] = {
                "status": "complete"
                if all(status == "success" for status in statuses)
                else "partial"
                if records
                else "missing",
                "unmatched_request_indices": sorted(
                    set(records) - {result["index"] for result in response["results"]}
                ),
                "config": {
                    **response["feature_capture"]["config"],
                    "source": "live_paged_emission_decisions",
                    "representation": "last_emission_decision_hidden_then_mean_emission_decision_hidden",
                },
                "timings": {
                    "meaning": "inclusive_host_seconds_not_gpu_kernel_timings",
                    "pooling_seconds": captured.post_pooling_seconds
                    if captured is not None
                    else None,
                    "completion_seconds": captured.post_completion_seconds
                    if captured is not None
                    else None,
                },
                "decode_calls": captured.decode_calls if captured is not None else 0,
            }

        def _crop_result(self, text: str, index: int, category: str) -> dict[str, Any]:
            generation = getattr(text, "generation", None)
            error = None
            try:
                if not isinstance(text, str) or not isinstance(generation, dict):
                    raise ValueError("missing native generation evidence")
                if not isinstance(generation.get("stop_token_seen"), bool):
                    raise ValueError("invalid decoder termination evidence")
                count, temperature = (
                    generation.get("generated_tokens"),
                    generation.get("temperature"),
                )
                if (
                    type(count) is not int
                    or count < 0
                    or type(temperature) not in (int, float)
                    or temperature < 0
                ):
                    raise ValueError("invalid native generation settings")
                json.dumps(generation, allow_nan=False)
                scores = [
                    generation[key]
                    for key in ("token_score", "token_score_min")
                    if key in generation
                ]
                tokens = generation.get("tokens", [])
                if not isinstance(tokens, list) or len(tokens) > count:
                    raise ValueError("invalid token evidence")
                for token in tokens:
                    if not isinstance(token, dict):
                        raise ValueError("invalid token evidence")
                    start, end = token["start"], token["end"]
                    if (
                        type(start) is not int
                        or type(end) is not int
                        or not 0 <= start < end <= len(text)
                    ):
                        raise ValueError("invalid token offsets")
                    scores.append(token["score"])
                if any(
                    type(score) not in (int, float) or not 0 <= score <= 1
                    for score in scores
                ):
                    raise ValueError("invalid native token probability")
                if not text.strip():
                    error = "empty_text"
                elif self._truncated(text):
                    error = "truncated"
                elif self._looped(text, category):
                    error = "decoder_cycle"
            except (KeyError, TypeError, ValueError):
                generation = None
                error = "invalid_generation"
            return {
                "index": index,
                "text": text,
                "generation": generation,
                "status": "failed" if error else "success",
                "error": error,
            }

        def _build_element(
            self, element: dict[str, Any], image: Image.Image, cap: int
        ) -> dict[str, Any]:
            category = str(element.get("category") or "text")
            bbox = [float(value) for value in element["bbox"]]
            text = element.get("text") or ""
            truncated = self._truncated(text)
            looped = self._looped(text, category)
            built = {
                "category": category,
                "bbox": bbox,
                "score": float(element.get("score") or 0.0),
                "text": text,
                "truncated": truncated,
                "generation": getattr(text, "generation", {}),
                **{
                    key: element[key]
                    for key in ("query_id", "class_scores")
                    if key in element
                },
            }
            if not truncated and not looped:
                return built
            attempts = self._retry_read(image, bbox, category, cap)
            if not attempts:
                if looped:
                    built["looped"] = True
                return built
            built["read_attempts"] = [
                {"text": reading, "generation": getattr(reading, "generation", {})}
                for reading in [text, *attempts]
            ]
            built["retries"] = len(attempts)
            last = attempts[-1]
            if (
                last.strip()
                and not self._truncated(last)
                and not self._looped(last, category)
            ):
                built["text"] = last
                built["generation"] = getattr(last, "generation", {})
                built["truncated"] = False
            elif looped:
                built["looped"] = True
            return built

        def _retry_read(
            self,
            image: Image.Image,
            bbox: list[float],
            category: str,
            cap: int,
        ) -> list[str]:
            """Retry the source crop at the official loader's minimum resolution."""
            ocr_category = category_by_layout.get(category)
            box = _clamp_box(bbox, image.width, image.height)
            if ocr_category is None or box is None:
                return []
            crop = image.crop(box)
            attempts = []
            try:
                budget = FALCON_MAX_NEW_TOKENS
                temperatures = (0.0, *RETRY_TEMPERATURES)
                for temperature in temperatures:
                    text = engine.generate_plain(
                        images=[crop],
                        category=[ocr_category],
                        temperature=temperature,
                        max_new_tokens=budget,
                        min_image_size=256,
                        max_image_size=int(
                            layout.get("max_image_size", FALCON_MAX_IMAGE_SIZE)
                        ),
                        use_tqdm=False,
                    )[0]
                    attempts.append(text)
                    if (
                        text.strip()
                        and not self._truncated(text)
                        and not self._looped(text, category)
                    ):
                        break
                return attempts
            finally:
                _drain_tokenization(engine)
                crop.close()

        @staticmethod
        def _truncated(text: str) -> bool:
            """Use actual decoder termination, never the apparent repetition of text."""
            detail = getattr(text, "generation", None)
            return detail is not None and not detail["stop_token_seen"]

        @staticmethod
        def _looped(text: str, category: str) -> bool:
            detail = getattr(text, "generation", None)
            return category not in {
                "table",
                "formula",
                "checkbox-selected",
                "checkbox-unselected",
            } and bool(detail and detail.get("decoder_cycle"))

        def _read_input(self) -> tuple[Image.Image, dict[str, Any]]:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > max_request_bytes:
                raise ValueError("invalid content length")
            payload = json.loads(self.rfile.read(content_length))
            encoded = payload.get("image") if isinstance(payload, dict) else None
            if not isinstance(encoded, str):
                raise ValueError("invalid request")
            overrides = {
                name: LAYOUT_OPTIONS[name](payload[name])
                for name in LAYOUT_OPTIONS
                if name in payload
            }
            if (
                not 0
                < overrides.get("max_new_tokens", FALCON_MAX_NEW_TOKENS)
                <= FALCON_MAX_NEW_TOKENS
            ):
                raise ValueError("invalid max_new_tokens")
            max_image_size = overrides.get("max_image_size", FALCON_MAX_IMAGE_SIZE)
            if (
                not 0 < max_image_size <= FALCON_MAX_IMAGE_SIZE
                or max_image_size % 16 != 0
            ):
                raise ValueError("invalid max_image_size")
            data = base64.b64decode(encoded, validate=True)
            with Image.open(io.BytesIO(data)) as opened:
                if opened.width * opened.height > max_image_pixels:
                    raise ValueError("image exceeded its pixel limit")
                return opened.convert("RGB"), overrides

        def _write_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def _drain_tokenization(engine: LayoutEngine) -> None:
    futures = getattr(engine, "_prefetch_futures", {})
    for future in futures.values():
        future.cancel()
    wait([future for future in futures.values() if not future.cancelled()])
    futures.clear()


def _release_failed_generation(engine: LayoutEngine) -> None:
    cache = getattr(engine, "paged_kv_cache", None)
    if cache is None:
        return
    _drain_tokenization(engine)
    # A failure can strand a slot before it is associated with any queued sequence.
    free = set(cache.free_batch_idx)
    for slot in range(1, cache.max_batch_size):
        if slot not in free:
            cache.erase(slot)
    engine._input_pos.zero_()
    engine._rope_pos_t.zero_()
    engine._token_ids_ready = None
    engine._token_ids_count = 0
    engine.waiting.clear()
    engine.running.clear()
    engine.done.clear()


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.max_new_tokens > FALCON_MAX_NEW_TOKENS:
        parser.error(f"--max-new-tokens cannot exceed {FALCON_MAX_NEW_TOKENS}")
    if args.max_image_size > FALCON_MAX_IMAGE_SIZE or args.max_image_size % 16 != 0:
        parser.error(
            f"--max-image-size must be a multiple of 16 up to {FALCON_MAX_IMAGE_SIZE}"
        )
    route_model = load_route_model(args.route_model) if args.route_model else None

    from falcon_perception import (
        load_and_prepare_model,
        setup_torch_config,
    )
    from falcon_perception.data import ImageProcessor
    from falcon_perception.paged_inference import PagedInferenceEngine, SamplingParams
    from falcon_perception.paged_ocr_inference import (
        LAYOUT_TO_OCR_CATEGORY,
        OCRInferenceEngine,
    )
    from ocr_pipeline.heron_layout import HeronTableDetector

    layout_categories = {
        **LAYOUT_TO_OCR_CATEGORY,
        "checkbox-selected": LAYOUT_TO_OCR_CATEGORY["text"],
        "checkbox-unselected": LAYOUT_TO_OCR_CATEGORY["text"],
    }

    class DocumentLayoutEngine(OCRInferenceEngine):
        """Keep the official decoder; use one learned layout pass for every modality."""

        def _check_done(self, sequences):
            # The official engine already copies one token ID per active sequence here.
            assert self._token_ids_ready is not None
            self._token_ids_ready.synchronize()
            token_ids = self._token_ids_pinned[: self._token_ids_count].tolist()
            self._token_ids_ready = None
            done = []
            for sequence, token_id in zip(sequences, token_ids, strict=True):
                assert sequence.sampling_params is not None
                assert sequence.batch_idx is not None
                monitor = getattr(
                    sequence,
                    "loop_monitor",
                    False,
                )
                if monitor:
                    state = _cycle_step(
                        getattr(sequence, "_cycle_state", (None, None, 0, 0)),
                        token_id,
                    )
                    sequence._cycle_state = state
                    sequence._cycle_longest = max(
                        getattr(sequence, "_cycle_longest", 0), state[2], state[3]
                    )
                    if _cycle_detected(
                        state, sequence._cycle_longest, sequence.output_length
                    ):
                        sequence._decoder_cycle_stopped = True
                if (
                    getattr(sequence, "_decoder_cycle_stopped", False)
                    or token_id in sequence.sampling_params.stop_token_ids
                    or sequence.input_length + sequence.output_length
                    >= self.max_seq_length
                    or sequence.output_length == sequence.sampling_params.max_new_tokens
                ):
                    done.append(sequence)
            if done:
                self._finalize_done(done)
            return [sequence for sequence in sequences if not sequence.finished]

        def load_layout_model(self, layout_model=None):
            # The caller supplies the IBM detector. Never lazily load Paddle weights.
            return

        def run_layout_detection(self, images, threshold=0.5):
            return [item["layout"] for item in self.run_route_detection(images)]

        def run_route_detection(self, images):
            return [
                _route_detection(self.layout_detector.detect_elements(image))
                for image in images
            ]

        @staticmethod
        def build_crop_sequences(
            pil_img,
            detections,
            *,
            min_image_size=64,
            max_image_size=FALCON_MAX_IMAGE_SIZE,
            min_crop_dim=16,
        ):
            from falcon_perception.paged_inference import Sequence
            from ocr_pipeline.falcon import _prepare_falcon_crop

            sequences = []
            for index, detection in enumerate(detections):
                category = layout_categories.get(detection["category"])
                box = _clamp_box(detection["bbox"], *pil_img.size)
                if category is None or box is None:
                    continue
                crop = pil_img.crop(box)
                prepared = _prepare_falcon_crop(crop, max_image_size)
                if prepared.owned:
                    crop.close()
                sequence = Sequence(
                    text=OCRInferenceEngine._make_ocr_prompt(category),
                    image=prepared.image,
                    min_image_size=min_image_size,
                    max_image_size=max_image_size,
                    request_idx=0,
                    task="ocr",
                )
                sequence.loop_monitor = detection["category"] not in {
                    "table",
                    "formula",
                    "checkbox-selected",
                    "checkbox-unselected",
                }
                sequences.append((sequence, index))
            return sequences

        def generate_with_layout(self, images, **options):
            # Different learned classes may share pixels. Upstream overlap suppression
            # randomly drops one of equal-size text/picture proposals, losing the text.
            detections = options.pop("detections", None)
            if detections is None:
                detections = self.run_layout_detection(images)
            results = [
                [
                    {
                        **detection,
                        "text": "",
                    }
                    for detection in page
                ]
                for page in detections
            ]
            sequences, origins = [], []
            try:
                for page_index, (image, page) in enumerate(zip(images, detections)):
                    for sequence, detection_index in self.build_crop_sequences(
                        image,
                        page,
                        min_image_size=options.get("min_image_size", 64),
                        max_image_size=options.get(
                            "max_image_size", FALCON_MAX_IMAGE_SIZE
                        ),
                    ):
                        sequence.request_idx = len(sequences)
                        sequences.append(sequence)
                        origins.append((page_index, detection_index))
                if sequences:
                    completed = PagedInferenceEngine.generate(
                        self,
                        sequences,
                        sampling_params=SamplingParams(
                            max_new_tokens=options.get(
                                "max_new_tokens", FALCON_MAX_NEW_TOKENS
                            ),
                            stop_token_ids=self._stop_token_ids(),
                        ),
                        temperature=options.get("temperature", 0.0),
                        top_k=options.get("top_k"),
                        use_tqdm=options.get("use_tqdm", True),
                        print_stats=options.get("print_stats", False),
                        profiler=options.get("profiler"),
                    )
                    if sorted(seq.request_idx for seq in completed) != list(
                        range(len(sequences))
                    ):
                        raise ValueError("decoder did not return every requested crop")
                    for sequence in completed:
                        page_index, detection_index = origins[sequence.request_idx]
                        results[page_index][detection_index]["text"] = (
                            self._decode_seq_text(sequence)
                        )
            finally:
                _drain_tokenization(self)
                for sequence in sequences:
                    sequence._image_raw.close()
            return results

        def _decode_seq_text(self, sequence):
            import torch

            ids = torch.stack(sequence._output_ids).cpu().tolist()
            probabilities = torch.stack(sequence._output_probs).float().cpu().tolist()
            stopped = bool(ids and ids[-1] in self._stop_token_ids())
            content_ids = ids[:-1] if stopped else ids
            raw = self.tokenizer.decode(content_ids)
            text = raw.strip()
            detail = {
                "generated_tokens": len(ids),
                "stop_token_seen": stopped,
                "temperature": self.temperature,
            }
            if getattr(sequence, "_decoder_cycle_stopped", False) or _decoder_cycle(
                content_ids
            ):
                detail["decoder_cycle"] = True
            # Sampling changes the probability distribution. Report recognition scores
            # only for the original softmax used by the official greedy path.
            if self.temperature == 0.0 and content_ids:
                scores = probabilities[: len(content_ids)]
                detail["token_score"] = math.exp(
                    sum(math.log(max(p, 1e-30)) for p in scores) / len(scores)
                )
                detail["token_score_min"] = min(scores)
                encoded = self.tokenizer._tok.encode(raw, add_special_tokens=False)
                # Offsets are trustworthy only when re-encoding reproduces the IDs.
                if encoded.ids == content_ids:
                    start = len(raw) - len(raw.lstrip())
                    detail["tokens"] = [
                        {
                            "start": max(0, left - start),
                            "end": min(len(text), right - start),
                            "score": score,
                        }
                        for (left, right), score in zip(encoded.offsets, scores)
                        if right > start and left < start + len(text)
                    ]
            return DecodedText(text, detail, sequence.request_idx)

    setup_torch_config()
    model, tokenizer, _ = load_and_prepare_model(
        hf_model_id=args.model,
        hf_revision=args.model_revision,
        device=args.device,
        # The demo defaults to float32, which asks for a 33 GiB KV cache on this GPU.
        dtype=args.dtype,
        compile=args.compile,
    )
    engine = DocumentLayoutEngine(
        model,
        tokenizer,
        ImageProcessor(patch_size=16, merge_size=1),
        max_batch_size=args.max_batch_size,
        max_seq_length=model.args.max_seq_len,
        n_pages=args.n_pages,
        capture_cudagraph=args.cudagraph,
    )
    if model.args.max_seq_len != FALCON_MAX_CONTEXT_TOKENS:
        raise ValueError(
            "Falcon context length does not match the pinned v1.5 configuration"
        )
    engine.layout_detector = HeronTableDetector(
        model_id=args.layout_model,
        revision=args.layout_model_revision,
        device=args.layout_device,
        score_threshold=args.layout_threshold,
    )
    server = create_server(
        engine,
        args.host,
        args.port,
        tokenizer=tokenizer,
        detect_layout=engine.run_layout_detection
        if args.layout_device == "cpu"
        else None,
        detect_route=engine.run_route_detection if route_model is not None else None,
        layout={
            "layout_threshold": args.layout_threshold,
            "max_new_tokens": args.max_new_tokens,
            "max_image_size": args.max_image_size,
        },
        category_by_layout=layout_categories,
        max_image_pixels=args.max_image_pixels,
        feature_config={
            "compile_model": args.compile,
            "capture_cudagraph": args.cudagraph,
            "max_batch_size": args.max_batch_size,
            "n_pages": args.n_pages,
            "max_seq_length": model.args.max_seq_len,
        },
        model={
            "id": args.model,
            "revision": args.model_revision,
            "inference_repository_revision": FALCON_INFERENCE_REPOSITORY_REVISION,
            "engine": "falcon-perception ocr_layout",
            "layout_model": args.layout_model,
            "layout_model_revision": args.layout_model_revision,
            "origin": "TII, UAE",
            "license": "Apache-2.0",
            "dtype": args.dtype,
            "score_meaning": "layout detection confidence; decoder probabilities in generation",
        },
        route_model=route_model,
        extractor_metadata={
            "version": ROUTE_EXTRACTOR_VERSION,
            "heron_model_id": args.layout_model,
            "heron_model_revision": args.layout_model_revision,
            "layout_score_threshold": args.layout_threshold,
        }
        if route_model is not None
        else None,
    )
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="tiiuae/Falcon-OCR")
    parser.add_argument(
        "--model-revision", type=_model_revision, default=FALCON_MODEL_REVISION
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument("--max-batch-size", type=int, default=16)
    parser.add_argument("--n-pages", type=int, default=256)
    # Both on by default, as in the official demo: they take a dense table page from
    # 27.4s to 9.2s. Use the --no- forms to isolate a compile or graph-capture fault.
    parser.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--cudagraph", action=argparse.BooleanOptionalAction, default=True
    )
    # Retained for existing launch commands; learned proposal ownership is preserved.
    parser.add_argument("--layout-threshold", type=float, default=0.3)
    parser.add_argument(
        "--max-new-tokens", type=_positive_int, default=FALCON_MAX_NEW_TOKENS
    )
    parser.add_argument("--layout-model", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--layout-model-revision",
        type=_model_revision,
        default=DEFAULT_MODEL_REVISION,
    )
    parser.add_argument("--layout-device", default="cpu")
    parser.add_argument(
        "--route-model",
        type=os.fspath,
        help="Validated linear JSON router for layout versus native full-page Falcon",
    )
    parser.add_argument(
        "--max-image-size", type=_positive_int, default=FALCON_MAX_IMAGE_SIZE
    )
    parser.add_argument(
        "--max-image-pixels", type=_positive_int, default=MAX_IMAGE_PIXELS
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8087)
    return parser


def _model_revision(value: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise argparse.ArgumentTypeError("Use a complete immutable model revision")
    return value


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return number


if __name__ == "__main__":
    raise SystemExit(main())
