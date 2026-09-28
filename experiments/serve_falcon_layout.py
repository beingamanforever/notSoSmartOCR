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
from pathlib import Path
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
from ocr_pipeline.falcon_layout import EMPTY_CATEGORIES

# Detection and crop options the caller may override per request, so a coverage sweep
# does not need a restart per combination.
LAYOUT_OPTIONS = {
    "max_image_size": int,
    "min_image_size": int,
    # A dense table stops mid-markup at the official 4096, losing its remaining rows.
    "max_new_tokens": int,
}
MAX_REQUEST_BYTES = 64_000_000
MAX_IMAGE_PIXELS = 32_000_000
LOGGER = logging.getLogger(__name__)
# olmocr's production fix for VLM decode repetition: escalate temperature and retry.
RETRY_TEMPERATURES = (0.2, 0.5, 0.8)
# olmocr keeps the first retry that stops; on faint crops that kept sampled inventions
# ("1 2 3 ... 1000" for a grey address line). A sampled read is kept only when another
# successful read agrees with it (self-consistency); a hallucination rarely repeats.
RETRY_AGREEMENT = 0.9
# On a born-digital page these still decode: structure and LaTeX are not in the cells.
DECODED_WITH_CELLS = frozenset({"table", "formula"})
MAX_CELLS = 20_000
RETRY_MIN_IMAGE_SIZE = 256
MIN_DECODER_CYCLE_TOKENS = 256
MAX_DECODER_CYCLE_PERIOD = 64
DECODER_CYCLE_CHECK_INTERVAL = 32


def _decoder_cycle(token_ids: Sequence[int]) -> bool:
    """Detect a recent repeat of any short period spanning the minimum cycle length."""
    import numpy as np

    tail = np.asarray(
        token_ids[-(MIN_DECODER_CYCLE_TOKENS + MAX_DECODER_CYCLE_PERIOD) :]
    )
    if len(tail) < MIN_DECODER_CYCLE_TOKENS:
        return False
    # A short output that is almost entirely one cycle also counts.
    needed = min(MIN_DECODER_CYCLE_TOKENS, 0.9 * len(token_ids))
    for period in range(1, MAX_DECODER_CYCLE_PERIOD + 1):
        same = np.concatenate(([0], tail[period:] == tail[:-period], [0]))
        edges = np.flatnonzero(np.diff(same.astype(np.int8)))
        if (edges[1::2] - edges[::2]).max(initial=0) + period >= needed:
            return True
    return False


def _agreed_reading(readings: Sequence[str]) -> str | None:
    """The latest reading another reading agrees with, if any."""
    from difflib import SequenceMatcher

    normalized = [" ".join(reading.split()).casefold() for reading in readings]
    for index in reversed(range(len(readings))):
        if any(
            SequenceMatcher(None, normalized[index], other).ratio() >= RETRY_AGREEMENT
            for position, other in enumerate(normalized)
            if position != index
        ):
            return readings[index]
    return None


def _without_reserved(
    ids: Sequence[int], scores: Sequence[float], reserved: frozenset[int]
) -> tuple[list[int], list[float]]:
    """Drop reserved vocabulary slots (">>UNUSED_261<<"): they carry no text, and the
    official decode keeps them, so an empty crop reads as that literal."""
    kept = [(i, p) for i, p in zip(ids, scores) if i not in reserved]
    return [i for i, _ in kept], [p for _, p in kept]


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


def _clamp_box(
    bbox: Sequence[float], width: int, height: int
) -> tuple[int, int, int, int] | None:
    if len(bbox) != 4 or not all(math.isfinite(value) for value in bbox):
        return None
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None
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
        if item["label"] != "key_value_region"
        or not any(
            parent["label"] == "form"
            and parent["box"][0] <= item["box"][0]
            and parent["box"][1] <= item["box"][1]
            and parent["box"][2] >= item["box"][2]
            and parent["box"][3] >= item["box"][3]
            for parent in items
        )
    }


# Falcon Perception's official layout path: PP-DocLayoutV3 at its default threshold, then
# its nested-box and duplicate filters. Labels map to the Heron vocabulary so every later
# stage (routing, regions, rendering) is shared by both detectors.
PP_LAYOUT_THRESHOLD = 0.3
PP_CONTAINMENT = 0.8
PP_TO_HERON = {
    "abstract": "text",
    "algorithm": "code",
    "aside_text": "text",
    "chart": "picture",
    "content": "document_index",
    "doc_title": "title",
    "figure_title": "caption",
    "footer": "page_footer",
    "footnote": "footnote",
    "formula": "formula",
    "formula_number": "text",
    "header": "page_header",
    "image": "picture",
    "number": "page_footer",
    "paragraph_title": "section_header",
    "reference": "text",
    "reference_content": "text",
    "seal": "picture",
    "table": "table",
    "text": "text",
    "vision_footnote": "footnote",
}


def _official_proposals(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Official PP-DocLayoutV3 output (reading-order sorted) as Heron-style proposals."""
    from falcon_perception.paged_ocr_inference import (
        dedup_overlapping_detections,
        filter_nested_detections,
    )

    kept = dedup_overlapping_detections(
        filter_nested_detections(detections, PP_CONTAINMENT)
    )
    return [
        {"label": PP_TO_HERON[item["category"]], "score": item["score"], "box": tuple(item["bbox"])}
        for item in kept
        if item["category"] in PP_TO_HERON
    ]


def _route_detection(
    proposals: list[dict[str, Any]],
    resolved: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    resolved = proposals if resolved is None else resolved
    kept = _exclusive_form_indices(resolved)
    return [
        {
            "category": "algorithm"
            if item["label"] == "code"
            else item["label"].replace("_", "-"),
            "bbox": list(item["box"]),
            "score": item["score"],
            **{
                key: item[key]
                for key in ("query_id", "class_scores", "cell_text", "orphan_text")
                if key in item
            },
        }
        for index, item in enumerate(resolved)
        if index in kept
    ]


def _read_digital_tables(
    table_reader: Callable[..., str],
    image: Image.Image,
    detections: list[dict[str, Any]],
    cells: list[dict[str, Any]],
) -> None:
    """Give each born-digital table TableFormer's grid over the page's own cells."""
    for detection in detections:
        if detection["category"] != "table" or not detection.get("cell_text"):
            continue
        try:
            detection["cell_text"] = table_reader(image, detection["bbox"], cells)
        except Exception:
            # The table still decodes from pixels; the failure stays in the log.
            LOGGER.exception("TableFormer failed on a born-digital table")
            continue
        detection["table_source"] = getattr(table_reader, "name", "tableformer")


def _valid_cells(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) > MAX_CELLS:
        raise ValueError("invalid cells")
    cells = []
    for cell in value:
        box = cell.get("bbox") if isinstance(cell, dict) else None
        if (
            not isinstance(cell, dict)
            or not isinstance(cell.get("text"), str)
            or not _finite_numbers(box, 4)
        ):
            raise ValueError("invalid cell")
        cells.append({"text": cell["text"], "bbox": [float(v) for v in box]})
    return cells


def _finite_numbers(values: Any, count: int) -> bool:
    return (
        isinstance(values, list)
        and len(values) == count
        and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
            for v in values
        )
    )


def _valid_words(value: Any) -> list[list[float]]:
    """Detected word boxes, [left, top, right, bottom, score] in page pixels."""
    if not isinstance(value, list) or len(value) > MAX_CELLS:
        raise ValueError("invalid words")
    if not all(
        _finite_numbers(word, 5) and word[0] < word[2] and word[1] < word[3]
        for word in value
    ):
        raise ValueError("invalid word")
    return [[float(v) for v in word] for word in value]


def create_server(
    engine: LayoutEngine,
    host: str,
    port: int,
    *,
    model: dict[str, Any],
    tokenizer: Any = None,
    detect_layout: Callable[[list[Image.Image]], list[list[dict[str, Any]]]]
    | None = None,
    layout: dict[str, Any] | None = None,
    category_by_layout: dict[str, str] | None = None,
    max_request_bytes: int = MAX_REQUEST_BYTES,
    max_image_pixels: int = MAX_IMAGE_PIXELS,
    feature_config: dict[str, Any] | None = None,
    table_reader: Callable[..., str] | None = None,
    detect_on_engine: bool = False,
) -> ThreadingHTTPServer:
    layout = dict(layout or {})
    category_by_layout = dict(category_by_layout or {})
    execution_lock = threading.Lock()
    unavailable = threading.Event()
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Falcon layout service must bind to loopback")

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
            cap = int(
                {**layout, **overrides}.get("max_new_tokens") or FALCON_MAX_NEW_TOKENS
            )
            try:
                options = {**layout, **overrides}
                cells = options.pop("cells", None)
                words = options.pop("words", None)
                timings = {"meaning": "inclusive_host_seconds_not_gpu_kernel_timings"}
                started = time.perf_counter()
                if detect_layout is not None:
                    # A GPU detector shares the device with decoding, so it takes the engine lock.
                    with execute_engine() if detect_on_engine else nullcontext():
                        options["detections"] = (
                            detect_layout([image], cells=[cells])
                            if cells
                            else detect_layout([image], words=[words])
                            if words
                            else detect_layout([image])
                        )
                timings["layout_seconds"] = time.perf_counter() - started
                if cells and table_reader is not None and "detections" in options:
                    started = time.perf_counter()
                    _read_digital_tables(
                        table_reader, image, options["detections"][0], cells
                    )
                    timings["digital_table_seconds"] = time.perf_counter() - started
                with execute_engine():
                    built = self._read_layout(image, options, cap, timings)
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
                    "model": model,
                    "timings": timings,
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

        def _read_layout(
            self,
            image: Image.Image,
            options: dict[str, Any],
            cap: int,
            timings: dict[str, Any] | None = None,
        ) -> list[dict[str, Any]]:
            timings = {} if timings is None else timings
            started = time.perf_counter()
            elements = engine.generate_with_layout(
                images=[image], use_tqdm=False, **options
            )[0]
            timings["decode_seconds"] = time.perf_counter() - started
            timings["crops"] = len(elements)
            started = time.perf_counter()
            # Failed crops are retried together through the same crop path.
            attempts = {
                index: []
                for index, element in enumerate(elements)
                if self._failed(element)
            }
            for temperature in RETRY_TEMPERATURES:
                pending = [
                    index
                    for index, tries in attempts.items()
                    if _agreed_reading(self._successes(elements[index], tries)) is None
                ]
                if not pending:
                    break
                retried = engine.generate_with_layout(
                    images=[image],
                    use_tqdm=False,
                    **{
                        **options,
                        "detections": [[elements[index] for index in pending]],
                        "temperature": temperature,
                        "min_image_size": RETRY_MIN_IMAGE_SIZE,
                        "max_new_tokens": cap,
                    },
                )[0]
                for index, element in zip(pending, retried, strict=True):
                    attempts[index].append(element["text"])
            timings["retry_seconds"] = time.perf_counter() - started
            timings["retried_crops"] = len(attempts)
            return [
                self._build_element(element, attempts.get(index, []))
                for index, element in enumerate(elements)
                if element.get("bbox")
            ]

        def _successes(self, element: dict[str, Any], tries: list[str]) -> list[str]:
            return [
                reading
                for reading in tries
                if reading.strip() and not self._failed({**element, "text": reading})
            ]

        def _failed(self, element: dict[str, Any]) -> bool:
            text = element.get("text", "")
            category = str(element.get("category") or "text")
            empty = not text.strip() and category not in EMPTY_CATEGORIES | {
                "checkbox-selected",
                "checkbox-unselected",
            }
            return empty or self._truncated(text) or self._looped(text, category)

        def _build_element(
            self, element: dict[str, Any], attempts: list[str]
        ) -> dict[str, Any]:
            category = str(element.get("category") or "text")
            text = element.get("text", "")
            looped = self._looped(text, category)
            built = {
                "category": category,
                "bbox": [float(value) for value in element["bbox"]],
                "score": float(element.get("score") or 0.0),
                "text": text,
                "truncated": self._truncated(text),
                "generation": getattr(text, "generation", {}),
                **{
                    key: element[key]
                    for key in ("query_id", "class_scores")
                    if key in element
                },
                **(
                    {"text_source": "pdf_text_layer"}
                    if element.get("cell_text")
                    and (
                        "table_source" in element or category not in DECODED_WITH_CELLS
                    )
                    else {}
                ),
                **(
                    {"table_source": element["table_source"]}
                    if "table_source" in element
                    else {}
                ),
            }
            if not attempts:
                if looped:
                    built["looped"] = True
                return built
            built["read_attempts"] = [
                {"text": reading, "generation": getattr(reading, "generation", {})}
                for reading in [text, *attempts]
            ]
            built["retries"] = len(attempts)
            agreed = _agreed_reading(self._successes(element, attempts))
            if agreed is not None:
                built["text"] = agreed
                built["generation"] = getattr(agreed, "generation", {})
                built["truncated"] = False
            elif looped:
                built["looped"] = True
            return built

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
            if "cells" in payload:
                overrides["cells"] = _valid_cells(payload["cells"])
            if "words" in payload:
                overrides["words"] = _valid_words(payload["words"])
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
            min_image_size = overrides.get(
                "min_image_size", layout.get("min_image_size", 64)
            )
            if not 0 < min_image_size <= max_image_size:
                raise ValueError("invalid min_image_size")
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
    if args.control_slices == 1 or args.control_slices < 0:
        parser.error("--control-slices must be 0 (off) or at least 2")
    if args.min_image_size > args.max_image_size:
        parser.error("--min-image-size must not exceed --max-image-size")
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
    from ocr_pipeline.digital_tables import DigitalTableReader
    from ocr_pipeline.coverage import orphan_text_detections
    from ocr_pipeline.heron_layout import (
        CONTROL_LABELS,
        CONTROL_REGION_LABELS,
        HeronTableDetector,
        resolve_layout,
    )

    layout_categories = {
        **LAYOUT_TO_OCR_CATEGORY,
        "checkbox-selected": LAYOUT_TO_OCR_CATEGORY["text"],
        "checkbox-unselected": LAYOUT_TO_OCR_CATEGORY["text"],
        "document-index": LAYOUT_TO_OCR_CATEGORY["text"],
        # Heron labels most handwriting `picture` (54% of NAF handwritten fields), so
        # pictures are read too: +10 exact handwriting snippets, table TEDS 0.749 -> 0.768.
        "picture": LAYOUT_TO_OCR_CATEGORY["text"],
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
                    history = getattr(sequence, "_cycle_history", None)
                    if history is None:
                        history = sequence._cycle_history = []
                    history.append(token_id)
                    if len(history) % DECODER_CYCLE_CHECK_INTERVAL == 0 and (
                        _decoder_cycle(history)
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

        layout_backend = "heron"
        control_slices = 0
        orphan_text = True

        def load_layout_model(self, layout_model=None):
            # Paddle weights load only when explicitly chosen as the layout backend.
            if self.layout_backend == "pp-doclayout-v3":
                OCRInferenceEngine.load_layout_model(self)

        def run_layout_detection(self, images, threshold=0.5, cells=None, words=None):
            detections = []
            for index, image in enumerate(images):
                page_cells = cells[index] if cells else None
                page_words = words[index] if words else None
                if self.layout_backend == "pp-doclayout-v3":
                    proposals = _official_proposals(
                        OCRInferenceEngine.run_layout_detection(
                            self, [image], threshold=PP_LAYOUT_THRESHOLD
                        )[0]
                    )
                    # A born-digital page still takes Docling's cell assignment.
                    resolved = (
                        resolve_layout(proposals, *image.size, page_cells)
                        if page_cells
                        else proposals
                    )
                else:
                    proposals = self.layout_detector.detect_elements(image)
                    # Only tiles over control, form, key-value or table proposals are sliced.
                    regions = [
                        item["box"]
                        for item in proposals
                        if item["label"] in CONTROL_REGION_LABELS
                    ]
                    if self.control_slices and regions:
                        proposals += self.layout_detector.detect_sliced_controls(
                            image,
                            self.control_slices,
                            regions,
                            [item for item in proposals if item["label"] in CONTROL_LABELS],
                        )
                    resolved = resolve_layout(proposals, *image.size, page_cells)
                if page_words and not page_cells and self.orphan_text:
                    resolved = [*resolved, *orphan_text_detections(resolved, page_words, image)]
                detections.append(_route_detection(proposals, resolved))
            return detections

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
                if "table_source" in detection or (
                    detection.get("cell_text")
                    and detection["category"] not in DECODED_WITH_CELLS
                ):
                    continue  # the page's own text layer already holds this text
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
            # Overlaps are resolved in run_layout_detection. Upstream suppression is tuned
            # for PP-DocLayout and randomly drops equal-size text/picture proposals.
            detections = options.pop("detections", None)
            if detections is None:
                detections = self.run_layout_detection(images)
            results = [
                [
                    {
                        **detection,
                        "text": detection.get("cell_text", ""),
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

        def _reserved_ids(self):
            if not hasattr(self, "_reserved"):
                added = self.tokenizer._tok.get_added_tokens_decoder()
                self._reserved = frozenset(
                    i for i, token in added.items() if token.special and token.content.startswith(">>UNUSED_")
                )
            return self._reserved

        def _decode_seq_text(self, sequence):
            import torch

            ids = torch.stack(sequence._output_ids).cpu().tolist()
            probabilities = torch.stack(sequence._output_probs).float().cpu().tolist()
            stopped = bool(ids and ids[-1] in self._stop_token_ids())
            content_ids = ids[:-1] if stopped else ids
            content_ids, probabilities = _without_reserved(
                content_ids, probabilities, self._reserved_ids()
            )
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
    # A local fine-tuned checkpoint directory is pinned by its path, not a hub revision.
    layout_revision = (
        None
        if Path(args.layout_model).is_dir() or args.layout_backend != "heron"
        else args.layout_model_revision
    )
    engine.layout_backend = args.layout_backend
    engine.control_slices = args.control_slices
    engine.orphan_text = not args.no_orphan_text
    engine.load_layout_model()
    engine.layout_detector = HeronTableDetector(
        model_id=args.layout_model,
        revision=layout_revision,
        device=args.layout_device,
        score_threshold=args.layout_threshold,
    )
    server = create_server(
        engine,
        args.host,
        args.port,
        tokenizer=tokenizer,
        detect_layout=engine.run_layout_detection,
        detect_on_engine=args.layout_device != "cpu",
        layout={
            "layout_threshold": args.layout_threshold,
            "max_new_tokens": args.max_new_tokens,
            "max_image_size": args.max_image_size,
            "min_image_size": args.min_image_size,
        },
        category_by_layout=layout_categories,
        max_image_pixels=args.max_image_pixels,
        table_reader=DigitalTableReader(
            device=args.device, artifacts_path=args.tableformer_artifacts
        ),
        feature_config={
            "compile_model": args.compile,
            "capture_cudagraph": args.cudagraph,
            "max_batch_size": args.max_batch_size,
            "n_pages": args.n_pages,
            "max_seq_length": model.args.max_seq_len,
            "control_slices": args.control_slices,
            "orphan_text": {"enabled": not args.no_orphan_text, "word_detector": "docTR db_resnet50"},
        },
        model={
            "id": args.model,
            "revision": args.model_revision,
            "inference_repository_revision": FALCON_INFERENCE_REPOSITORY_REVISION,
            "engine": "falcon-perception ocr_layout",
            "layout_model": args.layout_model
            if args.layout_backend == "heron"
            else "PaddlePaddle/PP-DocLayoutV3_safetensors",
            "layout_model_revision": layout_revision,
            "origin": "TII, UAE",
            "license": "Apache-2.0",
            "dtype": args.dtype,
            "score_meaning": "layout detection confidence; decoder probabilities in generation",
        },
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
        "--no-orphan-text",
        action="store_true",
        help="do not read detected words that no layout region holds (A/B only)",
    )
    parser.add_argument(
        "--control-slices",
        type=int,
        # Handwritten forms 40.5% -> 49.6%, checkbox coverage 56.5% -> 60.0% (3x3).
        default=3,
        help="tiles per side for the sliced checkbox pass (0: off)",
    )
    parser.add_argument(
        "--max-new-tokens", type=_positive_int, default=FALCON_MAX_NEW_TOKENS
    )
    parser.add_argument("--layout-model", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--layout-backend", choices=("heron", "pp-doclayout-v3"), default="heron"
    )
    parser.add_argument(
        "--layout-model-revision",
        type=_model_revision,
        default=DEFAULT_MODEL_REVISION,
    )
    parser.add_argument("--layout-device", default="cpu")
    parser.add_argument(
        "--max-image-size", type=_positive_int, default=FALCON_MAX_IMAGE_SIZE
    )
    # Official Falcon resize floor: shorter crops are upscaled to it (64 in generate_plain).
    parser.add_argument("--min-image-size", type=_positive_int, default=64)
    parser.add_argument(
        "--max-image-pixels", type=_positive_int, default=MAX_IMAGE_PIXELS
    )
    parser.add_argument(
        "--tableformer-artifacts",
        type=Path,
        help="Parent of docling-project--docling-models; downloaded when omitted",
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
