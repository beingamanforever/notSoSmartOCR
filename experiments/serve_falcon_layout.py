"""Serve the official Falcon-Perception layout-aware OCR engine over loopback HTTP.

The engine detects regions with PP-DocLayoutV3, crops each one, and reads it with a
category-specific prompt, batching the crops continuously. It needs its own torch pin, so
it runs in a separate process like the Falcon crop reader and the Arctic-TILT service.

Note on `score`: it is PP-DocLayoutV3's detection confidence, not a recognition
confidence. Falcon-Perception exposes no logprobs, so nothing here reports how sure the
reader is that the characters are right.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import io
import json
import logging
import re
from collections.abc import Sequence
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Protocol

from PIL import Image, UnidentifiedImageError

# Detection and crop options the caller may override per request, so a coverage sweep
# does not need a restart per combination.
LAYOUT_OPTIONS = {
    "containment_threshold": float,
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
_MIN_CYCLE_TOKENS, _MAX_CYCLE_TOKENS = 1, 10


class LayoutEngine(Protocol):
    def generate_with_layout(
        self, images: list[Any], **options: Any
    ) -> list[list[dict]]: ...

    def generate_plain(self, images: list[Any], **options: Any) -> list[str]: ...


def _looped(text: str) -> bool:
    """Whether `text` degenerated into a short cycle repeating past a normal run.

    A dense table's decode loop repeats whole cell values ("$1,659" "$546" ...) and
    self-terminates inside the token budget, so `truncated` alone never catches it.
    A cycle of 2-10 tokens repeating 5+ times, or a single token repeating 10+ times,
    is a loop; shorter runs like four consecutive "NA" cells or a dot-leader line are not.
    Markup tags become separators first: a loop glued by "<br>" inside one table cell
    is otherwise a single giant whitespace token and invisible to the cycle scan.
    """
    tokens = re.sub(r"<[^>]+>", " ", text).split()
    for cycle_len in range(_MIN_CYCLE_TOKENS, _MAX_CYCLE_TOKENS + 1):
        threshold = 10 if cycle_len == 1 else 5
        run = best_run = 0
        for i in range(cycle_len, len(tokens)):
            if tokens[i] == tokens[i - cycle_len]:
                run += 1
                best_run = max(best_run, run)
            else:
                run = 0
        if best_run // cycle_len + 1 >= threshold:
            return True
    return False


def _retry_budget(width: int, height: int) -> int:
    """Marker's area-scaled floor, so a dense table crop is not budget-starved."""
    return max(2048, min(8192, width * height // 750))


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
    layout: dict[str, Any] | None = None,
    category_by_layout: dict[str, str] | None = None,
    max_request_bytes: int = MAX_REQUEST_BYTES,
    max_image_pixels: int = MAX_IMAGE_PIXELS,
) -> ThreadingHTTPServer:
    layout = dict(layout or {})
    category_by_layout = dict(category_by_layout or {})
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Falcon layout service must bind to loopback")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/health":
                self._write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return
            self._write_json(HTTPStatus.OK, {"status": "ready", "model": model})

        def do_POST(self) -> None:  # noqa: N802
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

                with torch.inference_mode():
                    elements = engine.generate_with_layout(
                        images=[image], use_tqdm=False, **{**layout, **overrides}
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

        def _build_element(
            self, element: dict[str, Any], image: Image.Image, cap: int
        ) -> dict[str, Any]:
            category = str(element.get("category") or "text")
            bbox = [float(value) for value in element["bbox"]]
            text = str(element.get("text") or "")
            truncated = self._truncated(text, cap)
            built = {
                "category": category,
                "bbox": bbox,
                "score": float(element.get("score") or 0.0),
                "text": text,
                "truncated": truncated,
            }
            if not (_looped(text) or truncated):
                return built
            fixed = self._retry_read(image, bbox, category)
            if fixed is None:
                if _looped(text):
                    built["looped"] = True
                return built
            fixed_text, retries = fixed
            built["text"] = fixed_text
            built["truncated"] = False
            built["retries"] = retries
            return built

        def _retry_read(
            self, image: Image.Image, bbox: list[float], category: str
        ) -> tuple[str, int] | None:
            """Escalate temperature against the element's own crop (olmocr's fix)."""
            ocr_category = category_by_layout.get(category)
            box = _clamp_box(bbox, image.width, image.height)
            if ocr_category is None or box is None:
                return None
            crop = image.crop(box)
            try:
                budget = _retry_budget(crop.width, crop.height)
                for attempt, temperature in enumerate(RETRY_TEMPERATURES, start=1):
                    text = engine.generate_plain(
                        images=[crop],
                        category=[ocr_category],
                        temperature=temperature,
                        max_new_tokens=budget,
                        use_tqdm=False,
                    )[0]
                    if not _looped(text) and not self._truncated(text, budget):
                        return text, attempt
                return None
            finally:
                crop.close()

        def _truncated(self, text: str, cap: int) -> bool:
            """Whether the read consumed its whole token budget, so it never stopped.

            A healthy read ends on a stop token well inside the budget. One that spends
            every token has not finished - in practice it has fallen into a repetition
            loop, which is how a barcode strip became 3898 characters of invented text.
            """
            if tokenizer is None or not text:
                return False
            try:
                return len(tokenizer.encode(text)) >= cap
            except Exception:
                return False

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
    from falcon_perception.paged_ocr_inference import (
        LAYOUT_TO_OCR_CATEGORY,
        OCRInferenceEngine,
    )

    setup_torch_config()
    model, tokenizer, _ = load_and_prepare_model(
        hf_model_id=args.model,
        device=args.device,
        # The demo defaults to float32, which asks for a 33 GiB KV cache on this GPU.
        dtype=args.dtype,
        compile=args.compile,
    )
    engine = OCRInferenceEngine(
        model,
        tokenizer,
        ImageProcessor(patch_size=16, merge_size=1),
        max_batch_size=args.max_batch_size,
        n_pages=args.n_pages,
        capture_cudagraph=args.cudagraph,
    )
    server = create_server(
        engine,
        args.host,
        args.port,
        tokenizer=tokenizer,
        layout={
            "containment_threshold": args.containment_threshold,
            "layout_threshold": args.layout_threshold,
            "max_new_tokens": args.max_new_tokens,
            "max_image_size": args.max_image_size,
        },
        category_by_layout=LAYOUT_TO_OCR_CATEGORY,
        model={
            "id": args.model,
            "engine": "falcon-perception ocr_layout",
            "layout_model": "PaddlePaddle/PP-DocLayoutV3",
            "origin": "TII, UAE",
            "license": "Apache-2.0",
            "dtype": args.dtype,
            "score_meaning": "layout detection confidence, not recognition confidence",
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
    # The engine drops a detection that sits inside a larger one, so a page whose whole
    # body is detected as one table loses every region within it. Officially 0.8.
    parser.add_argument("--containment-threshold", type=float, default=0.8)
    parser.add_argument("--layout-threshold", type=float, default=0.3)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    # Officially 1024: a near-full-page crop is downscaled to that, which erases the
    # smallest text on a dense page.
    parser.add_argument("--max-image-size", type=int, default=1024)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8087)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
