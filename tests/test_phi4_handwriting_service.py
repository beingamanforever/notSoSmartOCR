from __future__ import annotations

import threading

from PIL import Image
import pytest

from experiments.serve_phi4_handwriting import create_server
from ocr_pipeline.providers import (
    PHI4_ADAPTER_FORMAT,
    Phi4HandwritingServiceReader,
    ReaderError,
)


class EchoReader:
    max_batch_items = 4
    provenance = {
        "id": "microsoft/Phi-4-multimodal-instruct",
        "adapter": {
            "format": PHI4_ADAPTER_FORMAT,
            "source": "adapter.pt",
            "validated": True,
        },
    }

    def __init__(self) -> None:
        self.sizes: list[tuple[int, int]] = []

    def transcribe_batch(self, images: list[Image.Image]) -> list[str]:
        self.sizes = [image.size for image in images]
        return [f"crop {width}x{height}" for width, height in self.sizes]


def test_loopback_service_transcribes_images_end_to_end() -> None:
    source = EchoReader()
    server = create_server(source, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    reader = Phi4HandwritingServiceReader(
        f"http://127.0.0.1:{port}",
        max_batch_items=4,
        timeout_seconds=5,
    )
    images = [
        Image.new("RGB", (40, 20), "white"),
        Image.new("RGB", (64, 24), "white"),
    ]
    try:
        output = reader.transcribe_batch(images)
    finally:
        for image in images:
            image.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert output == ["crop 40x20", "crop 64x24"]
    assert source.sizes == [(40, 20), (64, 24)]
    assert reader.provenance == source.provenance


def test_service_reader_rejects_non_loopback_urls() -> None:
    with pytest.raises(ValueError, match="loopback"):
        Phi4HandwritingServiceReader("https://example.com")


def test_service_rejects_oversized_image_without_calling_reader() -> None:
    source = EchoReader()
    server = create_server(source, "127.0.0.1", 0, max_image_pixels=100)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    reader = Phi4HandwritingServiceReader(
        f"http://127.0.0.1:{port}",
        timeout_seconds=5,
    )
    image = Image.new("RGB", (20, 20), "white")
    try:
        with pytest.raises(ReaderError, match="service did not return"):
            reader.transcribe_batch([image])
    finally:
        image.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert source.sizes == []
