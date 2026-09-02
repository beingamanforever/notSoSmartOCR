"""Serve the local Phi-4 handwriting adapter over loopback HTTP."""

from __future__ import annotations

import argparse
import base64
import binascii
import io
import json
import sys
from collections.abc import Sequence
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Protocol

from PIL import Image, UnidentifiedImageError

MAX_REQUEST_BYTES = 32_000_000
MAX_IMAGE_PIXELS = 20_000_000


class CropReader(Protocol):
    max_batch_items: int

    @property
    def provenance(self) -> dict[str, Any]: ...

    def transcribe_batch(self, images: Sequence[Image.Image]) -> list[str]: ...


def create_server(
    reader: CropReader,
    host: str,
    port: int,
    *,
    max_request_bytes: int = MAX_REQUEST_BYTES,
    max_image_pixels: int = MAX_IMAGE_PIXELS,
) -> ThreadingHTTPServer:
    """Create a memory-only loopback service around a warm crop reader."""
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("handwriting service must bind to loopback")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/health":
                self._write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return
            self._write_json(
                HTTPStatus.OK,
                {"status": "ready", "provenance": reader.provenance},
            )

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/transcribe":
                self._write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return
            images: list[Image.Image] = []
            try:
                images = self._read_images()
                texts = reader.transcribe_batch(images)
                if len(texts) != len(images) or any(
                    not isinstance(text, str) for text in texts
                ):
                    raise ValueError("reader returned the wrong batch size")
            except (ValueError, OSError, UnidentifiedImageError, binascii.Error):
                self._write_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "invalid_request"},
                )
                return
            except Exception:
                self._write_json(
                    HTTPStatus.UNPROCESSABLE_ENTITY,
                    {"error": "handwriting_reader_failed"},
                )
                return
            finally:
                for image in images:
                    image.close()
            self._write_json(
                HTTPStatus.OK,
                {"texts": texts, "provenance": reader.provenance},
            )

        def _read_images(self) -> list[Image.Image]:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > max_request_bytes:
                raise ValueError("invalid content length")
            payload = json.loads(self.rfile.read(content_length))
            encoded = payload.get("images") if isinstance(payload, dict) else None
            if (
                not isinstance(encoded, list)
                or not encoded
                or len(encoded) > reader.max_batch_items
                or any(not isinstance(item, str) for item in encoded)
            ):
                raise ValueError("invalid image batch")
            images = []
            try:
                for item in encoded:
                    data = base64.b64decode(item, validate=True)
                    with Image.open(io.BytesIO(data)) as opened:
                        if opened.width * opened.height > max_image_pixels:
                            raise ValueError("image exceeded its pixel limit")
                        images.append(opened.convert("RGB"))
            except Exception:
                for image in images:
                    image.close()
                raise
            return images

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
    if args.torch_site is not None:
        sys.path.append(str(args.torch_site))

    from ocr_pipeline.providers import (
        PHI4_MODEL_ID,
        PHI4_MODEL_REVISION,
        Phi4HandwritingReader,
    )

    reader = Phi4HandwritingReader(
        args.adapter,
        model_name_or_path=args.model or PHI4_MODEL_ID,
        model_revision=args.model_revision or PHI4_MODEL_REVISION,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        max_batch_items=args.max_batch_items,
        batch_size=args.batch_size,
        local_files_only=True,
    )
    if args.warmup:
        image = Image.new("RGB", (320, 96), "white")
        try:
            reader.transcribe_batch([image])
        finally:
            image.close()

    server = create_server(reader, args.host, args.port)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--model")
    parser.add_argument("--model-revision")
    parser.add_argument("--torch-site", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-batch-items", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8083)
    parser.add_argument("--warmup", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
