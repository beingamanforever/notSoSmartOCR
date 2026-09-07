from __future__ import annotations

import base64
import io
import json
import threading
import urllib.request
from pathlib import Path

from PIL import Image
import pytest

from experiments import serve_falcon_layout, serve_falcon_ocr
from ocr_pipeline import falcon as falcon_module
from ocr_pipeline.falcon import (
    FALCON_MODEL_ID,
    FALCON_MODEL_LICENSE,
    FALCON_MODEL_ORIGIN,
    FALCON_MODEL_REVISION,
    FalconOCRReader,
    FalconOCRServiceReader,
)
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
        "id": FALCON_MODEL_ID,
        "loaded_from": "/models/falcon-ocr",
        "revision": FALCON_MODEL_REVISION,
        "origin": FALCON_MODEL_ORIGIN,
        "license": FALCON_MODEL_LICENSE,
        "identity_verified": True,
        "local_files_only": True,
        "inference_repository_audit_reference": {
            "url": "https://github.com/tiiuae/Falcon-Perception",
            "revision": "59a845adfac23c684bc4beafd26b380cde5ddfc1",
            "loaded_code_match": "unverified",
        },
    }


def _generation_config(max_new_tokens: int = 1536) -> dict[str, object]:
    return {
        "max_new_tokens": max_new_tokens,
        "temperature": 0.0,
        "max_dimension": 1536,
        "compile": False,
    }


def test_falcon_service_initializes_model_before_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = FalconOCRReader()
    calls: list[str] = []
    monkeypatch.setattr(
        reader,
        "_initialize_model",
        lambda: calls.append("initialize") or object(),
    )

    class FakeServer:
        daemon_threads = False

        def __init__(self, address: object, handler: object) -> None:
            del address, handler
            calls.append("bind")

    monkeypatch.setattr(serve_falcon_ocr, "ThreadingHTTPServer", FakeServer)

    serve_falcon_ocr.create_server(reader, "127.0.0.1", 8085)

    assert calls == ["initialize", "bind"]


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
                "generation_config": _generation_config(3072),
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
        "max_dimension": 1536,
        "compile": False,
    }


def test_service_reader_health_requires_ready_initialized_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = []

    def fake_urlopen(request: object, timeout: float) -> FakeResponse:
        requests.append((request, timeout))
        return FakeResponse(
            {
                "status": "ready",
                "provenance": _provenance(),
                "generation_config": _generation_config(),
            }
        )

    monkeypatch.setattr(falcon_module, "urlopen", fake_urlopen)
    reader = FalconOCRServiceReader("http://127.0.0.1:8085", timeout_seconds=19)

    reader.check_health()

    request, timeout = requests[0]
    assert request.full_url == "http://127.0.0.1:8085/health"
    assert timeout == 19
    assert reader.provenance == {**_provenance(), "source": "loopback_service"}
    assert reader.generation["max_new_tokens"] == 1536


def test_service_reader_rejects_false_ready_health(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        falcon_module,
        "urlopen",
        lambda request, timeout: FakeResponse(
            {
                "status": "starting",
                "provenance": _provenance(),
                "generation_config": {},
            }
        ),
    )

    with pytest.raises(ReaderError, match="not ready"):
        FalconOCRServiceReader("http://127.0.0.1:8085").check_health()


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
                "generation_config": _generation_config(100),
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
                "provenance": {
                    **_provenance(),
                    "id": None,
                    "identity_verified": False,
                },
                "generation_config": _generation_config(),
            }
        ),
    )

    with pytest.raises(ReaderError, match="did not return a valid result"):
        FalconOCRServiceReader("http://127.0.0.1:8085").transcribe_crops(
            [Image.new("RGB", (1, 1))],
            ["text"],
        )


@pytest.mark.parametrize(
    "generation_config",
    ({}, {**_generation_config(), "temperature": float("inf")}),
)
def test_service_reader_rejects_invalid_generation_contract(
    monkeypatch: pytest.MonkeyPatch,
    generation_config: dict[str, object],
) -> None:
    monkeypatch.setattr(
        falcon_module,
        "urlopen",
        lambda request, timeout: FakeResponse(
            {
                "texts": ["value"],
                "provenance": _provenance(),
                "generation_config": generation_config,
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


# --- Falcon-Perception layout service: repetition-loop retry ladder ---


class FakeTokenizer:
    def encode(self, text: str) -> list[str]:
        return text.split()


class FakeLayoutEngine:
    """Fakes the `LayoutEngine` protocol `serve_falcon_layout.create_server` expects."""

    def __init__(
        self,
        elements: list[dict[str, object]],
        plain_by_temperature: dict | None = None,
    ) -> None:
        self.elements = elements
        self.plain_by_temperature = plain_by_temperature or {}
        self.retry_calls: list[dict[str, object]] = []

    def generate_with_layout(
        self, images: list[Image.Image], **options: object
    ) -> list[list[dict[str, object]]]:
        return [self.elements]

    def generate_plain(self, images: list[Image.Image], **options: object) -> list[str]:
        if "category" not in options:
            return ["page text"]
        self.retry_calls.append(options)
        return [self.plain_by_temperature[round(float(options["temperature"]), 2)]]


def _start_layout_server(engine: FakeLayoutEngine, **options: object):
    server = serve_falcon_layout.create_server(
        engine, "127.0.0.1", 0, model={"id": "test-falcon-layout"}, **options
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _post_layout_read(server, image: Image.Image) -> dict[str, object]:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    payload = json.dumps(
        {"image": base64.b64encode(buffer.getvalue()).decode("ascii")}
    ).encode("utf-8")
    request = urllib.request.Request(
        f"http://127.0.0.1:{server.server_address[1]}/read",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=5) as reply:
        return json.loads(reply.read())


def test_looped_table_cell_is_retried_and_replaced_by_a_clean_attempt() -> None:
    looped_text = "$1,659 $546 " * 24
    clean_text = "$1,659\n$546\nTotal $2,205"
    engine = FakeLayoutEngine(
        elements=[
            {
                "category": "table",
                "bbox": [0.0, 0.0, 40.0, 40.0],
                "score": 0.9,
                "text": looped_text,
            }
        ],
        plain_by_temperature={0.2: looped_text, 0.5: clean_text, 0.8: clean_text},
    )
    server = _start_layout_server(
        engine, tokenizer=FakeTokenizer(), category_by_layout={"table": "table"}
    )
    try:
        body = _post_layout_read(server, Image.new("RGB", (40, 40), "white"))
    finally:
        server.shutdown()
        server.server_close()

    element = body["elements"][0]
    assert element["text"] == clean_text
    assert element["truncated"] is False
    assert element["retries"] == 2
    assert "looped" not in element


def test_looped_table_cell_keeps_original_text_when_every_retry_still_loops() -> None:
    looped_text = "$1,659 $546 " * 24
    engine = FakeLayoutEngine(
        elements=[
            {
                "category": "table",
                "bbox": [0.0, 0.0, 40.0, 40.0],
                "score": 0.9,
                "text": looped_text,
            }
        ],
        plain_by_temperature={0.2: looped_text, 0.5: looped_text, 0.8: looped_text},
    )
    server = _start_layout_server(
        engine, tokenizer=FakeTokenizer(), category_by_layout={"table": "table"}
    )
    try:
        body = _post_layout_read(server, Image.new("RGB", (40, 40), "white"))
    finally:
        server.shutdown()
        server.server_close()

    element = body["elements"][0]
    assert element["text"] == looped_text
    assert element["looped"] is True
    assert "retries" not in element


def test_looped_detects_table_repetition_but_not_na_runs_or_dot_leaders() -> None:
    tail = "Prior Year $187 " + "$1,659 $546 " * 24
    assert serve_falcon_layout._looped(tail) is True

    financial_row = "Prior authorization: NA NA NA NA Copay: 20 Deductible: met"
    assert serve_falcon_layout._looped(financial_row) is False

    dot_leader = "Section 4 ..... 12"
    assert serve_falcon_layout._looped(dot_leader) is False


def test_a_loop_glued_by_markup_tags_is_still_detected() -> None:
    """The financial cell loop arrives as one <br>-separated blob with no whitespace,
    which a plain whitespace tokenizer sees as a single giant token."""
    from serve_falcon_layout import _looped

    glued = "$1,127<br>11.4%<br>" + "$1,659<br>$546<br>" * 24
    assert _looped(glued)
    assert not _looped("$1,127<br>11.4%<br>$1,659<br>$546")
