from __future__ import annotations

import threading
from pathlib import Path

from PIL import Image
import pytest

from experiments.serve_ministral_ocr import create_server
from ocr_pipeline.providers import (
    MINISTRAL_MODEL_REVISION,
    MinistralOCRServiceReader,
    ReaderError,
)


class EchoReader:
    provenance = {
        "id": None,
        "loaded_from": "/models/ministral",
        "revision": MINISTRAL_MODEL_REVISION,
        "origin": None,
        "license": None,
        "identity_verified": False,
        "local_files_only": True,
    }

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate_text(self, image_path: Path, prompt: str) -> tuple[str, str]:
        with Image.open(image_path) as image:
            size = image.size
        self.prompts.append(prompt)
        return f"# Draft\n\nPage {size[0]}x{size[1]}", "chat_template"


def test_loopback_ministral_service_reads_page_end_to_end(tmp_path: Path) -> None:
    source = EchoReader()
    server = create_server(source, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    page = tmp_path / "page.png"
    Image.new("RGB", (40, 20), "white").save(page)
    reader = MinistralOCRServiceReader(
        f"http://127.0.0.1:{server.server_address[1]}",
        prompt="Literal page prompt",
        timeout_seconds=5,
    )
    try:
        regions = reader.read(page, 2)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert source.prompts == ["Literal page prompt"]
    assert len(regions) == 1
    assert regions[0].id == "p2-page-1"
    assert regions[0].text == "# Draft\n\nPage 40x20"
    assert regions[0].bounding_box.right == 40
    assert regions[0].bounding_box.bottom == 20
    assert regions[0].text_provenance["model"]["source"] == "loopback_service"


def test_ministral_service_reader_rejects_non_loopback_url() -> None:
    with pytest.raises(ValueError, match="loopback"):
        MinistralOCRServiceReader("https://example.com")


def test_ministral_service_rejects_oversized_image(tmp_path: Path) -> None:
    source = EchoReader()
    server = create_server(source, "127.0.0.1", 0, max_image_pixels=100)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    page = tmp_path / "page.png"
    Image.new("RGB", (20, 20), "white").save(page)
    reader = MinistralOCRServiceReader(
        f"http://127.0.0.1:{server.server_address[1]}",
        timeout_seconds=5,
    )
    try:
        with pytest.raises(ReaderError) as error:
            reader.read(page, 1)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert error.value.code == "ministral_service_failed"
    assert source.prompts == []
