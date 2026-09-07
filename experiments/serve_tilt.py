"""Serve Arctic-TILT field comprehension over loopback HTTP.

Arctic-TILT ships as a fork of vLLM 0.8.3, which pins an older torch than the reader
stack needs. Running it in its own process keeps that dependency out of the pipeline
venv, the same arrangement the Falcon-OCR crop reader uses.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import io
import json
import logging
from collections.abc import Sequence
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Protocol

from PIL import Image, UnidentifiedImageError

MAX_REQUEST_BYTES = 64_000_000
MAX_IMAGE_PIXELS = 32_000_000
MAX_QUESTIONS = 64
LOGGER = logging.getLogger(__name__)


class FieldEngine(Protocol):
    def answer_pages(self, pages: list[Any], questions: list[str]) -> list[Any]: ...


def create_server(
    engine: FieldEngine,
    host: str,
    port: int,
    *,
    max_request_bytes: int = MAX_REQUEST_BYTES,
    max_image_pixels: int = MAX_IMAGE_PIXELS,
    max_questions: int = MAX_QUESTIONS,
) -> ThreadingHTTPServer:
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Arctic-TILT service must bind to loopback")

    from ocr_pipeline.comprehension import MODEL, PageInput

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/health":
                self._write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return
            self._write_json(HTTPStatus.OK, {"status": "ready", "model": MODEL})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/answer":
                self._write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return
            try:
                pages, questions = self._read_input()
            except (ValueError, OSError, UnidentifiedImageError, binascii.Error):
                self._write_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_request"})
                return
            try:
                with TemporaryDirectory(prefix="tilt-page-") as directory:
                    inputs = []
                    for index, (image, words, boxes, size) in enumerate(pages):
                        path = Path(directory) / f"page-{index}.png"
                        image.save(path, format="PNG")
                        inputs.append(PageInput(path, words, boxes, size))
                    answers = engine.answer_pages(inputs, questions)
            except Exception:
                LOGGER.exception("Arctic-TILT answering failed")
                self._write_json(
                    HTTPStatus.UNPROCESSABLE_ENTITY, {"error": "tilt_engine_failed"}
                )
                return
            finally:
                for image, *_ in pages:
                    image.close()
            self._write_json(
                HTTPStatus.OK,
                {"answers": [asdict(answer) for answer in answers], "model": MODEL},
            )

        def _read_input(self) -> tuple[list[tuple[Any, ...]], list[str]]:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > max_request_bytes:
                raise ValueError("invalid content length")
            payload = json.loads(self.rfile.read(content_length))
            if not isinstance(payload, dict):
                raise ValueError("invalid request")
            raw_pages = payload.get("pages")
            questions = payload.get("questions")
            if (
                not isinstance(raw_pages, list)
                or not raw_pages
                or not isinstance(questions, list)
                or not questions
                or len(questions) > max_questions
                or any(not isinstance(item, str) for item in questions)
            ):
                raise ValueError("invalid request")
            pages = []
            for raw in raw_pages:
                if not isinstance(raw, dict):
                    raise ValueError("invalid page")
                words = raw.get("words")
                boxes = raw.get("boxes")
                size = raw.get("size")
                encoded = raw.get("image")
                if (
                    not isinstance(encoded, str)
                    or not isinstance(words, list)
                    or not isinstance(boxes, list)
                    or len(words) != len(boxes)
                    or not isinstance(size, list)
                    or len(size) != 2
                ):
                    raise ValueError("invalid page")
                data = base64.b64decode(encoded, validate=True)
                with Image.open(io.BytesIO(data)) as opened:
                    if opened.width * opened.height > max_image_pixels:
                        raise ValueError("image exceeded its pixel limit")
                    image = opened.convert("RGB")
                pages.append(
                    (
                        image,
                        [str(word) for word in words],
                        [tuple(int(value) for value in box) for box in boxes],
                        (int(size[0]), int(size[1])),
                    )
                )
            return pages, questions

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
    from ocr_pipeline.comprehension import VllmTiltEngine

    engine = VllmTiltEngine(
        args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
    )
    server = create_server(engine, args.host, args.port)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def _parser() -> argparse.ArgumentParser:
    from ocr_pipeline.comprehension import MODEL

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL["id"])
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.2)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8086)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
