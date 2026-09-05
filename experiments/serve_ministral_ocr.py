"""Serve the pinned local Ministral page reader over loopback HTTP."""

from __future__ import annotations

import argparse
import base64
import binascii
import io
import json
import tempfile
from collections.abc import Sequence
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Protocol

from PIL import Image, UnidentifiedImageError

MAX_REQUEST_BYTES = 32_000_000
MAX_IMAGE_PIXELS = 32_000_000
MAX_PROMPT_CHARS = 100_000


class PageReader(Protocol):
    @property
    def provenance(self) -> dict[str, Any]: ...

    def generate_text(self, image_path: Path, prompt: str) -> tuple[str, str]: ...


def create_server(
    reader: PageReader,
    host: str,
    port: int,
    *,
    max_request_bytes: int = MAX_REQUEST_BYTES,
    max_image_pixels: int = MAX_IMAGE_PIXELS,
) -> ThreadingHTTPServer:
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Ministral OCR service must bind to loopback")

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
            if self.path != "/generate":
                self._write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return
            try:
                image, prompt = self._read_input()
            except (ValueError, OSError, UnidentifiedImageError, binascii.Error):
                self._write_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_request"})
                return
            try:
                with tempfile.TemporaryDirectory(prefix="ministral-page-") as temporary:
                    image_path = Path(temporary) / "page.png"
                    image.save(image_path, format="PNG")
                    text, _ = reader.generate_text(image_path, prompt)
            except Exception:
                self._write_json(
                    HTTPStatus.UNPROCESSABLE_ENTITY,
                    {"error": "ministral_reader_failed"},
                )
                return
            finally:
                image.close()
            self._write_json(
                HTTPStatus.OK,
                {"text": text, "provenance": reader.provenance},
            )

        def _read_input(self) -> tuple[Image.Image, str]:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > max_request_bytes:
                raise ValueError("invalid content length")
            payload = json.loads(self.rfile.read(content_length))
            encoded = payload.get("image") if isinstance(payload, dict) else None
            prompt = payload.get("prompt") if isinstance(payload, dict) else None
            if (
                not isinstance(encoded, str)
                or not isinstance(prompt, str)
                or not prompt.strip()
                or len(prompt) > MAX_PROMPT_CHARS
            ):
                raise ValueError("invalid request")
            data = base64.b64decode(encoded, validate=True)
            with Image.open(io.BytesIO(data)) as opened:
                if opened.width * opened.height > max_image_pixels:
                    raise ValueError("image exceeded its pixel limit")
                return opened.convert("RGB"), prompt

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
    from ocr_pipeline.providers import MinistralOCRReader

    reader = MinistralOCRReader(
        model_name=str(args.model),
        model_revision=args.model_revision,
        device_map=args.device_map,
        max_new_tokens=args.max_new_tokens,
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
    from ocr_pipeline.providers import MINISTRAL_MODEL_REVISION

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--model-revision", default=MINISTRAL_MODEL_REVISION)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8084)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
