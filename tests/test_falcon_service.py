from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
import pytest

from ocr_pipeline import falcon as falcon_module
from ocr_pipeline.falcon import FalconOCRServiceReader
from ocr_pipeline.providers import ReaderError


class FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.body = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, limit: int) -> bytes:
        assert len(self.body) < limit
        return self.body


def _provenance() -> dict[str, object]:
    return {
        "id": None,
        "loaded_from": "/models/falcon-ocr",
        "revision": "unverified",
        "origin": "unverified",
        "license": "unverified",
        "identity_verified": False,
        "local_files_only": True,
        "inference_repository_audit_reference": {
            "url": "https://github.com/tiiuae/Falcon-Perception",
            "revision": "59a845adfac23c684bc4beafd26b380cde5ddfc1",
            "loaded_code_match": "unverified",
        },
    }


def test_service_reader_batches_categories_and_preserves_raw_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = []

    def fake_urlopen(request: object, timeout: float) -> FakeResponse:
        requests.append((request, timeout))
        return FakeResponse(
            {
                "texts": ["  Heading\n", "<table><tr><td>1</td></tr></table>"],
                "provenance": _provenance(),
                "generation_config": {
                    "max_new_tokens": 3072,
                    "temperature": 0.0,
                    "max_dimension": 1540,
                    "compile": False,
                },
            }
        )

    monkeypatch.setattr(falcon_module, "urlopen", fake_urlopen)
    reader = FalconOCRServiceReader("http://127.0.0.1:8085", timeout_seconds=19)
    images = [Image.new("RGB", (2, 3)), Image.new("RGB", (4, 5))]

    texts = reader.transcribe_crops(images, ["section-header", "table"])

    assert texts == ["  Heading\n", "<table><tr><td>1</td></tr></table>"]
    request, timeout = requests[0]
    assert timeout == 19
    payload = json.loads(request.data)
    assert payload["categories"] == ["section-header", "table"]
    assert len(payload["images"]) == 2
    assert request.full_url == "http://127.0.0.1:8085/generate"
    assert reader.provenance == {**_provenance(), "source": "loopback_service"}
    assert reader.generation == {
        "category": "plain",
        "max_new_tokens": 3072,
        "temperature": 0.0,
        "max_dimension": 1540,
        "compile": False,
    }


def test_service_reader_full_page_region_keeps_service_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (20, 10)).save(image_path)
    monkeypatch.setattr(
        falcon_module,
        "urlopen",
        lambda request, timeout: FakeResponse(
            {
                "texts": ["raw page"],
                "provenance": _provenance(),
                "generation_config": {"max_new_tokens": 100},
            }
        ),
    )

    region = FalconOCRServiceReader("http://localhost:8085", category="text").read(
        image_path, 4
    )[0]

    assert region.text == "raw page"
    assert region.bounding_box.__dict__ == {
        "left": 0,
        "top": 0,
        "right": 20,
        "bottom": 10,
    }
    assert region.text_provenance["model"]["source"] == "loopback_service"
    assert region.text_provenance["output_validation"] == {
        "raw_response_preserved": True,
        "nonempty": True,
        "exact_repetition_loop": False,
        "termination_observable": False,
        "truncation_observable": False,
    }


@pytest.mark.parametrize(
    "url",
    ["https://127.0.0.1:8085", "http://gpu.internal:8085", "file:///tmp/service"],
)
def test_service_reader_rejects_nonloopback_url(url: str) -> None:
    with pytest.raises(ValueError, match="loopback"):
        FalconOCRServiceReader(url)


def test_service_reader_rejects_invalid_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        falcon_module,
        "urlopen",
        lambda request, timeout: FakeResponse(
            {
                "texts": ["value"],
                "provenance": {"local_files_only": False},
                "generation_config": {},
            }
        ),
    )

    with pytest.raises(ReaderError, match="did not return a valid result"):
        FalconOCRServiceReader("http://127.0.0.1:8085").transcribe_crops(
            [Image.new("RGB", (1, 1))],
            ["text"],
        )


def test_service_reader_enforces_batch_limit() -> None:
    reader = FalconOCRServiceReader(
        "http://127.0.0.1:8085",
        max_batch_items=1,
    )

    with pytest.raises(ValueError, match="batch exceeded"):
        reader.transcribe_crops(
            [Image.new("RGB", (1, 1)), Image.new("RGB", (1, 1))],
            ["text", "text"],
        )
