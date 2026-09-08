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
import threading
from collections.abc import Callable, Sequence
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Protocol

from PIL import Image, UnidentifiedImageError

from ocr_pipeline.falcon import FALCON_OCR_CATEGORIES

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
MAX_REQUEST_BYTES = 64_000_000
MAX_IMAGE_PIXELS = 32_000_000
LOGGER = logging.getLogger(__name__)
# olmocr's production fix for VLM decode repetition: escalate temperature and retry.
RETRY_TEMPERATURES = (0.2, 0.5, 0.8)


class DecodedText(str):
    """Keep per-sequence evidence through the official engine's string interface."""

    def __new__(cls, text: str, generation: dict[str, Any]):
        value = super().__new__(cls, text)
        value.generation = generation
        return value


class LayoutEngine(Protocol):
    def generate_with_layout(
        self, images: list[Any], **options: Any
    ) -> list[list[dict]]: ...

    def generate_plain(self, images: list[Any], **options: Any) -> list[str]: ...


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
) -> ThreadingHTTPServer:
    layout = dict(layout or {})
    category_by_layout = dict(category_by_layout or {})
    execution_lock = threading.Lock()
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Falcon layout service must bind to loopback")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/health":
                self._write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return
            self._write_json(HTTPStatus.OK, {"status": "ready", "model": model})

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
            cap = int({**layout, **overrides}.get("max_new_tokens") or 4096)
            try:
                import torch

                options = {**layout, **overrides}
                if detect_layout is not None:
                    options["detections"] = detect_layout([image])
                with execution_lock, torch.inference_mode():
                    elements = engine.generate_with_layout(
                        images=[image], use_tqdm=False, **options
                    )[0]
                    # The same official model reading the whole page, used as the
                    # completeness oracle: whatever the layout detector never proposed
                    # as a region still shows up here.
                    page_text = (
                        engine.generate_plain(images=[image], use_tqdm=False)[0]
                        if want_page
                        else ""
                    )
                    # Retries re-read a crop of the still-open request image, so they
                    # must run inside this block, ahead of the `finally` that closes it.
                    built = [
                        self._build_element(element, image, cap)
                        for element in elements
                        if element.get("bbox")
                    ]
            except Exception:
                LOGGER.exception("Falcon layout read failed")
                self._write_json(
                    HTTPStatus.UNPROCESSABLE_ENTITY, {"error": "falcon_layout_failed"}
                )
                return
            finally:
                image.close()
            self._write_json(
                HTTPStatus.OK,
                {"elements": built, "page_text": page_text, "model": model},
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
                if (
                    not isinstance(encoded, list)
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
                            rgb, layout.get("max_image_size", 1024)
                        )
                        if prepared.owned:
                            rgb.close()
                        images.append(prepared.image)
            except (ValueError, TypeError, AttributeError, OSError, binascii.Error):
                for image in images:
                    image.close()
                self._write_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_request"})
                return
            try:
                import torch

                cap = int(layout.get("max_new_tokens", 16384))
                with execution_lock, torch.inference_mode():
                    texts = engine.generate_plain(
                        images=images,
                        category=categories,
                        max_new_tokens=cap,
                        use_tqdm=False,
                    )
                    if len(texts) != len(images) or any(
                        not text.strip() or self._truncated(text) for text in texts
                    ):
                        raise ValueError("crop generation did not finish cleanly")
                self._write_json(HTTPStatus.OK, {"texts": texts, "model": model})
            except Exception:
                LOGGER.exception("Falcon crop read failed")
                self._write_json(
                    HTTPStatus.UNPROCESSABLE_ENTITY, {"error": "falcon_crop_failed"}
                )
            finally:
                for image in images:
                    image.close()

        def _build_element(
            self, element: dict[str, Any], image: Image.Image, cap: int
        ) -> dict[str, Any]:
            category = str(element.get("category") or "text")
            bbox = [float(value) for value in element["bbox"]]
            text = element.get("text") or ""
            truncated = self._truncated(text)
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
            if not truncated:
                return built
            attempts = self._retry_read(image, bbox, category, cap)
            if not attempts:
                return built
            built["read_attempts"] = [
                {"text": reading, "generation": getattr(reading, "generation", {})}
                for reading in [text, *attempts]
            ]
            built["retries"] = len(attempts)
            last = attempts[-1]
            if last.strip() and not self._truncated(last):
                built["text"] = last
                built["generation"] = getattr(last, "generation", {})
                built["truncated"] = False
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
                budget = min(32768, max(8192, cap * 2))
                temperatures = (0.0, *RETRY_TEMPERATURES)
                for temperature in temperatures:
                    text = engine.generate_plain(
                        images=[crop],
                        category=[ocr_category],
                        temperature=temperature,
                        max_new_tokens=budget,
                        min_image_size=256,
                        max_image_size=1024,
                        use_tqdm=False,
                    )[0]
                    attempts.append(text)
                    if text.strip() and not self._truncated(text):
                        break
                return attempts
            finally:
                crop.close()

        @staticmethod
        def _truncated(text: str) -> bool:
            """Use actual decoder termination, never the apparent repetition of text."""
            detail = getattr(text, "generation", None)
            return detail is not None and not detail["stop_token_seen"]

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


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)

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

        def load_layout_model(self, layout_model=None):
            # The caller supplies the IBM detector. Never lazily load Paddle weights.
            return

        def run_layout_detection(self, images, threshold=0.5):
            batches = []
            for image in images:
                detections = [
                    {
                        "category": "algorithm"
                        if item["label"] == "code"
                        else item["label"].replace("_", "-"),
                        "bbox": list(item["box"]),
                        "score": item["score"],
                        **{
                            key: item[key]
                            for key in ("query_id", "class_scores")
                            if key in item
                        },
                    }
                    for item in self.layout_detector.detect_elements(image)
                    # Wrapping groups are not additional text reads.
                    if item["label"]
                    not in {"form", "key_value_region", "document_index"}
                ]
                batches.append(detections)
            return batches

        @staticmethod
        def build_crop_sequences(
            pil_img,
            detections,
            *,
            min_image_size=64,
            max_image_size=1024,
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
                        max_image_size=options.get("max_image_size", 1024),
                    ):
                        sequence.request_idx = len(sequences)
                        sequences.append(sequence)
                        origins.append((page_index, detection_index))
                if sequences:
                    completed = PagedInferenceEngine.generate(
                        self,
                        sequences,
                        sampling_params=SamplingParams(
                            max_new_tokens=options.get("max_new_tokens", 16384),
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
            return DecodedText(text, detail)

    setup_torch_config()
    model, tokenizer, _ = load_and_prepare_model(
        hf_model_id=args.model,
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
    engine.layout_detector = HeronTableDetector(
        model_id=args.layout_model,
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
        layout={
            "layout_threshold": args.layout_threshold,
            "max_new_tokens": args.max_new_tokens,
            "max_image_size": args.max_image_size,
        },
        category_by_layout=layout_categories,
        model={
            "id": args.model,
            "engine": "falcon-perception ocr_layout",
            "layout_model": args.layout_model,
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
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--layout-model", default="ds4sd/docling-layout-heron-101")
    parser.add_argument("--layout-device", default="cpu")
    # Officially 1024: a near-full-page crop is downscaled to that, which erases the
    # smallest text on a dense page.
    parser.add_argument("--max-image-size", type=int, default=1024)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8087)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
