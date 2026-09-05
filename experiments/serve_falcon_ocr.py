"""Serve the pinned local Falcon-OCR core reader over loopback HTTP."""

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
from typing import Any, Protocol

from PIL import Image, UnidentifiedImageError

from ocr_pipeline.falcon import FALCON_OCR_CATEGORIES

MAX_REQUEST_BYTES = 64_000_000
MAX_IMAGE_PIXELS = 32_000_000
MAX_BATCH_ITEMS = 24
LOGGER = logging.getLogger(__name__)


class CropReader(Protocol):
    @property
    def provenance(self) -> dict[str, Any]: ...

    @property
    def generation_config(self) -> object: ...

    def transcribe_crops(
        self,
        images: Sequence[Image.Image],
        categories: Sequence[str],
    ) -> list[str]: ...


def create_server(
    reader: CropReader,
    host: str,
    port: int,
    *,
    max_request_bytes: int = MAX_REQUEST_BYTES,
    max_image_pixels: int = MAX_IMAGE_PIXELS,
    max_batch_items: int = MAX_BATCH_ITEMS,
) -> ThreadingHTTPServer:
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Falcon-OCR service must bind to loopback")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/health":
                self._write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return
            self._write_json(
                HTTPStatus.OK,
                {
                    "status": "ready",
                    "provenance": reader.provenance,
                    "generation_config": asdict(reader.generation_config),
                },
            )

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/generate":
                self._write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return
            images: list[Image.Image] = []
            try:
                images, categories = self._read_input()
                texts = reader.transcribe_crops(images, categories)
            except (ValueError, OSError, UnidentifiedImageError, binascii.Error):
                self._write_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_request"})
                return
            except Exception:
                LOGGER.exception("Falcon-OCR crop generation failed")
                self._write_json(
                    HTTPStatus.UNPROCESSABLE_ENTITY,
                    {"error": "falcon_reader_failed"},
                )
                return
            finally:
                for image in images:
                    image.close()
            self._write_json(
                HTTPStatus.OK,
                {
                    "texts": texts,
                    "provenance": reader.provenance,
                    "generation_config": asdict(reader.generation_config),
                },
            )

        def _read_input(self) -> tuple[list[Image.Image], list[str]]:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > max_request_bytes:
                raise ValueError("invalid content length")
            payload = json.loads(self.rfile.read(content_length))
            encoded = payload.get("images") if isinstance(payload, dict) else None
            categories = (
                payload.get("categories") if isinstance(payload, dict) else None
            )
            if (
                not isinstance(encoded, list)
                or not encoded
                or len(encoded) > max_batch_items
                or not isinstance(categories, list)
                or len(categories) != len(encoded)
                or any(item not in FALCON_OCR_CATEGORIES for item in categories)
            ):
                raise ValueError("invalid request")
            images = []
            for item in encoded:
                if not isinstance(item, str):
                    raise ValueError("invalid image")
                data = base64.b64decode(item, validate=True)
                with Image.open(io.BytesIO(data)) as opened:
                    if opened.width * opened.height > max_image_pixels:
                        raise ValueError("image exceeded its pixel limit")
                    images.append(opened.convert("RGB"))
            return images, categories

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
    from ocr_pipeline.falcon import FalconOCRReader

    reader = FalconOCRReader(
        model_name_or_path=args.model,
        local_model_path=args.local_model_path,
        model_revision=args.model_revision,
        device_map=args.device_map,
        max_new_tokens=args.max_new_tokens,
        max_dimension=args.max_dimension,
    )
    server = create_server(reader, args.host, args.port)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def _parser() -> argparse.ArgumentParser:
    from ocr_pipeline.falcon import FALCON_MODEL_REVISION

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--local-model-path", type=Path)
    parser.add_argument("--model-revision", default=FALCON_MODEL_REVISION)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--max-dimension", type=int, default=1536)
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8085)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
