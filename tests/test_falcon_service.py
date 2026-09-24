from __future__ import annotations

import base64
from dataclasses import asdict
import io
import json
import os
import threading
import urllib.error
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
from ocr_pipeline.contracts import BoundingBox, TextRegion
from ocr_pipeline.fused_reader import FusedReader


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
            "revision": "c457916c9974efbacfa91f0f6ecc2c49c6543e56",
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


def test_decoder_cycle_requires_a_long_near_complete_token_run() -> None:
    detect = serve_falcon_layout._decoder_cycle
    assert detect([2006, 638] * 125 + [2006] * 6)
    assert not detect([2006, 638] * 10)
    assert not detect(list(range(300)))
    assert not detect([2006, 638] * 100 + list(range(100)))
    assert detect(list(range(100)) + [2006, 638] * 200)
    state = (None, None, 0, 0)
    for token in [2006, 638] * 128:
        state = serve_falcon_layout._cycle_step(state, token)
    assert max(state[2:]) == 256


def test_form_proposal_needs_unowned_source_pixels() -> None:
    proposals = [
        {"label": "form", "box": [0, 0, 100, 100]},
        {"label": "form", "box": [0, 0, 50, 50]},
        {"label": "text", "box": [60, 60, 90, 90]},
        {"label": "form", "box": [100, 0, 150, 50]},
    ]

    assert serve_falcon_layout._exclusive_form_indices(proposals) == {0, 1, 2, 3}


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


def _route_model(action: str) -> dict[str, object]:
    names = list(serve_falcon_layout.HERON_ROUTE_FEATURE_NAMES)
    return {
        "version": 1,
        "id": "test-falcon-mode-router",
        "status": "calibrated",
        "training_actions": {
            "layout": "layout_selective_qwen",
            "plain": "plain_falcon",
        },
        "feature_names": names,
        "coefficients": [0.0] * len(names),
        "means": [0.0] * len(names),
        "scales": [1.0] * len(names),
        "intercept": 10.0 if action == "plain" else -10.0,
        "threshold": 0.5,
        "cost_tradeoff": 0.0,
        "cost_head": {"coefficients": [0.0] * len(names), "intercept": 0.0},
        "extractor": _route_extractor(),
    }


def _route_extractor() -> dict[str, object]:
    return {
        "version": serve_falcon_layout.ROUTE_EXTRACTOR_VERSION,
        "heron_model_id": "ds4sd/docling-layout-heron-101",
        "heron_model_revision": "2e4993cf6bb211112084a2f80938f26138008917",
        "layout_score_threshold": 0.3,
    }


def _region_role_model(*, selected: bool = True) -> dict[str, object]:
    names = list(serve_falcon_layout.HERON_REGION_ROLE_FEATURE_NAMES)
    return {
        "version": 1,
        "id": "test-heron-region-role",
        "status": "calibrated",
        "feature_names": names,
        "coefficients": [0.0] * len(names),
        "means": [0.0] * len(names),
        "scales": [1.0] * len(names),
        "intercept": 10.0 if selected else -10.0,
        "threshold": 0.5,
        "extractor": _route_extractor(),
    }


def _full_class_scores() -> dict[str, float]:
    return {label: 0.0 for label in serve_falcon_layout.HERON_ROUTE_CLASSES}


def test_route_model_loader_rejects_non_visual_features(tmp_path: Path) -> None:
    path = tmp_path / "router.json"
    path.write_text(json.dumps(_route_model("layout")))
    assert serve_falcon_layout.load_route_model(path)["id"] == (
        "test-falcon-mode-router"
    )

    model = _route_model("layout")
    model["feature_names"] = ["document_id"]
    path.write_text(json.dumps(model))
    with pytest.raises(ValueError, match="schema"):
        serve_falcon_layout.load_route_model(path)

    model = _route_model("layout")
    model["threshold"] = -0.25
    path.write_text(json.dumps(model))
    assert serve_falcon_layout.load_route_model(path)["threshold"] == -0.25


def test_route_model_loader_rejects_development_artifact(tmp_path: Path) -> None:
    path = tmp_path / "router.json"
    model = _route_model("layout")
    model["status"] = "development_only_optional_separate_calibration_required"
    path.write_text(json.dumps(model))

    with pytest.raises(ValueError, match="schema"):
        serve_falcon_layout.load_route_model(path)


def test_learned_models_require_matching_extractor_metadata() -> None:
    model = _route_model("layout")
    engine = FakeLayoutEngine([])

    def detect(images):
        return [{"raw": [], "layout": []}]

    server = serve_falcon_layout.create_server(
        engine,
        "127.0.0.1",
        0,
        model={"id": "test"},
        detect_route=detect,
        route_model=model,
        extractor_metadata=_route_extractor(),
    )
    server.server_close()

    mismatched = {**_route_extractor(), "layout_score_threshold": 0.4}
    with pytest.raises(ValueError, match="extractor does not match"):
        serve_falcon_layout.create_server(
            engine,
            "127.0.0.1",
            0,
            model={"id": "test"},
            detect_route=detect,
            route_model=model,
            extractor_metadata=mismatched,
        )

    with pytest.raises(ValueError, match="region role model extractor"):
        serve_falcon_layout.create_server(
            engine,
            "127.0.0.1",
            0,
            model={"id": "test"},
            heron_region_role_model=_region_role_model(),
            extractor_metadata=mismatched,
        )


def test_route_features_match_measured_heron_schema() -> None:
    import math

    detections = [
        {
            "label": "text",
            "box": [0, 0, 50, 20],
            "score": 0.8,
            "class_scores": {"text": 0.6, "table": 0.2},
        },
        {
            "category": "table",
            "bbox": [0, 0, 100, 100],
            "score": 0.9,
            "class_scores": {"text": 0.1, "table": 0.7},
        },
    ]
    values = serve_falcon_layout.route_feature_values(100, 200, detections)
    expected = [
        math.log1p(100),
        math.log1p(200),
        0.5,
        2.0,
    ]
    for label in serve_falcon_layout.HERON_ROUTE_CLASSES:
        expected.extend(
            {
                "text": [0.5, 0.8, 0.05, 0.05],
                "table": [0.5, 0.9, 0.5, 0.5],
            }.get(label, [0.0, 0.0, 0.0, 0.0])
        )
    for label in serve_falcon_layout.HERON_ROUTE_CLASSES:
        expected.extend(
            {
                "text": [0.35, 0.6],
                "table": [0.45, 0.7],
            }.get(label, [0.0, 0.0])
        )

    assert len(expected) == 106
    assert list(values) == list(serve_falcon_layout.HERON_ROUTE_FEATURE_NAMES)
    assert list(values.values()) == pytest.approx(expected)


def test_route_detection_preserves_raw_form_and_key_value_proposals() -> None:
    proposals = [
        {"label": "key_value_region", "box": (0, 0, 10, 10), "score": 0.8},
        {"label": "form", "box": (0, 0, 20, 20), "score": 0.9},
    ]

    detection = serve_falcon_layout._route_detection(proposals)

    assert [item["label"] for item in detection["raw"]] == [
        "key_value_region",
        "form",
    ]
    assert [item["category"] for item in detection["layout"]] == ["form"]


def test_region_role_appends_form_companion_without_changing_table() -> None:
    scores = _full_class_scores()
    table = {
        "label": "table",
        "box": (1, 2, 30, 40),
        "score": 0.9,
        "query_id": 7,
        "class_scores": scores,
    }

    detection = serve_falcon_layout._route_detection([table], _region_role_model())

    assert detection["layout"][0] == {
        "category": "table",
        "bbox": [1, 2, 30, 40],
        "score": 0.9,
        "query_id": 7,
        "class_scores": scores,
    }
    assert detection["layout"][1] == {
        **detection["layout"][0],
        "category": "form",
        "region_role": {
            "action": "add_form_source",
            "model": {"id": "test-heron-region-role", "version": 1},
            "score": pytest.approx(0.99995, abs=1e-5),
            "threshold": 0.5,
            "source_query_id": 7,
        },
    }
    assert detection["raw"] == [
        {
            "label": "table",
            "box": [1, 2, 30, 40],
            "score": 0.9,
            "query_id": 7,
            "class_scores": scores,
        }
    ]
    engine = FakeLayoutEngine([{**detection["layout"][1], "text": ""}])
    server = _start_layout_server(engine)
    try:
        element = _post_layout_read(server, Image.new("RGB", (40, 40)))["elements"][0]
    finally:
        server.shutdown()
        server.server_close()
    assert element["region_role"] == detection["layout"][1]["region_role"]


def test_region_role_companion_keeps_canonical_table_ownership(tmp_path: Path) -> None:
    from ocr_pipeline.falcon_layout import FalconLayoutReader

    role = {
        "action": "add_form_source",
        "model": {"id": "test-heron-region-role", "version": 1},
        "score": 0.9,
        "threshold": 0.5,
        "source_query_id": 7,
    }
    engine = FakeLayoutEngine(
        [
            {
                "category": "table",
                "bbox": [1, 2, 30, 40],
                "score": 0.9,
                "query_id": 7,
                "class_scores": _full_class_scores(),
                "text": "<table><tr><td>Dose</td></tr></table>",
            },
            {
                "category": "form",
                "bbox": [1, 2, 30, 40],
                "score": 0.9,
                "query_id": 7,
                "class_scores": _full_class_scores(),
                "region_role": role,
                "text": "",
            },
        ]
    )
    server = _start_layout_server(engine)
    image_path = tmp_path / "page.png"
    Image.new("RGB", (40, 40), "white").save(image_path)
    try:
        regions = FalconLayoutReader(f"http://127.0.0.1:{server.server_port}").read(
            image_path, 1
        )
    finally:
        server.shutdown()
        server.server_close()

    table = next(region for region in regions if region.kind == "table")
    form = next(
        region
        for region in regions
        if (region.structure or {}).get("role") == "form_source"
    )
    assert table.text == "Dose"
    assert form.structure["owner_region_id"] == table.id
    assert form.text_provenance["layout_detection"]["region_role"] == role


def test_region_role_below_threshold_keeps_current_output() -> None:
    proposals = [
        {
            "label": "table",
            "box": (0, 0, 20, 20),
            "score": 0.8,
            "query_id": 1,
            "class_scores": _full_class_scores(),
        }
    ]

    assert serve_falcon_layout._route_detection(
        proposals, _region_role_model(selected=False)
    ) == serve_falcon_layout._route_detection(proposals)


@pytest.mark.parametrize(
    "class_scores",
    [
        {"table": 0.9},
        {**_full_class_scores(), "table": float("nan")},
    ],
)
def test_region_role_malformed_features_abstain(
    class_scores: dict[str, float],
) -> None:
    proposals = [
        {
            "label": "table",
            "box": (0, 0, 20, 20),
            "score": 0.8,
            "query_id": 1,
            "class_scores": class_scores,
        }
    ]

    assert serve_falcon_layout._route_detection(
        proposals, _region_role_model()
    ) == serve_falcon_layout._route_detection(proposals)


def test_region_role_suppresses_only_contained_companions() -> None:
    proposals = [
        {
            "label": "table",
            "box": box,
            "score": 0.8,
            "query_id": query_id,
            "class_scores": _full_class_scores(),
        }
        for query_id, box in enumerate(
            ((0, 0, 100, 100), (1, 1, 99, 99), (0, 0, 100, 100))
        )
    ]

    layout = serve_falcon_layout._route_detection(proposals, _region_role_model())[
        "layout"
    ]

    assert [item["category"] for item in layout] == [
        "table",
        "table",
        "table",
        "form",
    ]
    assert [item["query_id"] for item in layout[:3]] == [0, 1, 2]
    assert layout[3]["region_role"]["source_query_id"] == 0


def test_region_role_leaves_native_form_unchanged() -> None:
    form = {
        "label": "form",
        "box": (0, 0, 20, 20),
        "score": 0.8,
        "query_id": 1,
        "class_scores": _full_class_scores(),
    }

    detection = serve_falcon_layout._route_detection([form], _region_role_model())

    assert detection["layout"] == [
        {
            "category": "form",
            "bbox": [0, 0, 20, 20],
            "score": 0.8,
            "query_id": 1,
            "class_scores": form["class_scores"],
        }
    ]


def test_region_role_reuses_matching_native_form() -> None:
    scores = _full_class_scores()
    proposals = [
        {
            "label": "table",
            "box": (0, 0, 20, 20),
            "score": 0.9,
            "query_id": 1,
            "class_scores": scores,
        },
        {
            "label": "form",
            "box": (0, 0, 20, 20),
            "score": 0.8,
            "query_id": 2,
            "class_scores": scores,
        },
    ]

    layout = serve_falcon_layout._route_detection(proposals, _region_role_model())[
        "layout"
    ]

    assert [item["category"] for item in layout] == ["table", "form"]
    assert layout[1]["query_id"] == 2
    assert layout[1]["region_role"]["source_query_id"] == 1


def test_region_role_ownership_is_independent_of_element_order(tmp_path: Path) -> None:
    from ocr_pipeline.falcon_layout import FalconLayoutReader

    scores = _full_class_scores()
    proposals = [
        {
            "label": "form",
            "box": (0, 0, 20, 20),
            "score": 0.9,
            "query_id": 2,
            "class_scores": scores,
        },
        {
            "label": "table",
            "box": (0, 0, 20, 20),
            "score": 0.8,
            "query_id": 1,
            "class_scores": scores,
        },
    ]
    layout = serve_falcon_layout._route_detection(proposals, _region_role_model())[
        "layout"
    ]
    engine = FakeLayoutEngine(
        [
            {**layout[0], "text": ""},
            {
                **layout[1],
                "text": "<table><tr><td>Dose</td></tr></table>",
            },
        ]
    )
    server = _start_layout_server(engine)
    image_path = tmp_path / "page.png"
    Image.new("RGB", (40, 40), "white").save(image_path)
    try:
        regions = FalconLayoutReader(f"http://127.0.0.1:{server.server_port}").read(
            image_path, 1
        )
    finally:
        server.shutdown()
        server.server_close()

    table = next(region for region in regions if region.kind == "table")
    form = next(
        region
        for region in regions
        if (region.structure or {}).get("role") == "form_source"
    )
    assert form.structure["owner_region_id"] == table.id


def test_region_role_omits_owner_for_dropped_empty_table(tmp_path: Path) -> None:
    from ocr_pipeline.falcon_layout import FalconLayoutReader

    role = {
        "action": "add_form_source",
        "model": {"id": "test-heron-region-role", "version": 1},
        "score": 0.9,
        "threshold": 0.5,
        "source_query_id": 7,
    }
    engine = FakeLayoutEngine(
        [
            {
                "category": "table",
                "bbox": [1, 2, 30, 40],
                "score": 0.9,
                "query_id": 7,
                "class_scores": _full_class_scores(),
                "text": "",
            },
            {
                "category": "form",
                "bbox": [1, 2, 30, 40],
                "score": 0.9,
                "query_id": 8,
                "class_scores": _full_class_scores(),
                "region_role": role,
                "text": "",
            },
        ]
    )
    server = _start_layout_server(engine)
    image_path = tmp_path / "page.png"
    Image.new("RGB", (40, 40), "white").save(image_path)
    try:
        regions = FalconLayoutReader(f"http://127.0.0.1:{server.server_port}").read(
            image_path, 1
        )
    finally:
        server.shutdown()
        server.server_close()

    assert not any(region.kind == "table" for region in regions)
    form = next(
        region
        for region in regions
        if (region.structure or {}).get("role") == "form_source"
    )
    assert "owner_region_id" not in form.structure


def test_region_role_loader_rejects_development_artifact(tmp_path: Path) -> None:
    path = tmp_path / "region-role.json"
    model = _region_role_model()
    model["development"] = {"grouped_validation": {"sources": 12}}
    path.write_text(json.dumps(model))

    assert serve_falcon_layout.load_heron_region_role_model(path)["id"] == (
        "test-heron-region-role"
    )

    model["unknown"] = True
    path.write_text(json.dumps(model))
    with pytest.raises(ValueError, match="schema"):
        serve_falcon_layout.load_heron_region_role_model(path)

    model.pop("unknown")
    model["status"] = "development"
    path.write_text(json.dumps(model))

    with pytest.raises(ValueError, match="schema"):
        serve_falcon_layout.load_heron_region_role_model(path)


def test_router_threshold_uses_total_quality_before_page_win_count() -> None:
    import numpy as np

    from experiments.train_falcon_mode_router import _threshold, _totals

    values = np.asarray([0.9, 0.8, 0.7, 0.6])
    qualities = np.asarray(
        [
            [100.0, 0.0],
            [1.0, 1.1],
            [1.0, 1.1],
            [1.0, 1.1],
        ]
    )
    costs = np.zeros((4, 2))
    threshold = _threshold(values, qualities, costs)
    selected = values > threshold

    assert selected.tolist() == [False, False, False, False]
    assert _totals(selected, qualities, costs)["quality"] == 103.0


def test_router_trainer_replays_named_cached_features() -> None:
    import warnings

    import numpy as np

    from experiments.train_falcon_mode_router import _fit, _predict, _training_arrays

    names = list(serve_falcon_layout.HERON_ROUTE_FEATURE_NAMES)
    rows = []
    for group, layout_quality, plain_quality in (
        ("a", 2.0, 1.0),
        ("b", 1.0, 2.0),
    ):
        rows.append(
            {
                "split": "development",
                "source_group": group,
                "feature_values": {name: 0.0 for name in names},
                "layout_quality": layout_quality,
                "plain_quality": plain_quality,
                "layout_expert_cost": 0.1,
                "plain_expert_cost": 0.0,
            }
        )
    matrix, targets, groups, qualities, costs = _training_arrays(rows, names)

    assert matrix.shape == (2, 106)
    assert targets.tolist() == [-1.0, 1.0]
    assert groups.tolist() == ["a", "b"]
    assert np.array_equal(qualities, [[2.0, 1.0], [1.0, 2.0]])
    assert np.array_equal(costs, [[0.1, 0.0], [0.1, 0.0]])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scaler, estimator = _fit(matrix, targets)
        replay = _predict(scaler.transform(matrix), estimator)
    assert not caught
    assert estimator.solver == "lsqr"
    assert estimator.alpha == 100.0
    assert np.isfinite(replay).all()

    rows[0]["feature_values"].pop(names[-1])
    with pytest.raises(ValueError, match="exact named"):
        _training_arrays(rows, names)


def test_cost_head_can_choose_lower_cost_quality_preserving_nonprefix_policy() -> None:
    import numpy as np

    from experiments.train_falcon_mode_router import _select_policy

    quality_predictions = np.asarray([0.2, 0.1])
    cost_predictions = np.asarray([0.0, 1.0])
    qualities = np.asarray([[10.0, 0.0], [5.0, 5.0]])
    costs = np.asarray([[0.0, 0.0], [1.0, 0.0]])

    tradeoff, threshold = _select_policy(
        quality_predictions, cost_predictions, qualities, costs
    )

    selected = quality_predictions + tradeoff * cost_predictions > threshold
    assert selected.tolist() == [False, True]


def test_nested_development_policy_does_not_tune_on_outer_group_labels() -> None:
    import numpy as np

    from experiments.train_falcon_mode_router import _nested_grouped_policy

    matrix = np.zeros((6, len(serve_falcon_layout.HERON_ROUTE_FEATURE_NAMES)))
    matrix[:, 0] = np.arange(6)
    groups = np.asarray(["a", "a", "b", "b", "c", "c"])
    qualities = np.asarray(
        [[2.0, 1.0], [1.0, 2.0], [3.0, 1.0], [1.0, 3.0], [4.0, 1.0], [1.0, 4.0]]
    )
    costs = np.asarray([[1.0, 0.0]] * 6)
    quality_targets = qualities[:, 1] - qualities[:, 0]
    cost_targets = costs[:, 0] - costs[:, 1]

    original = _nested_grouped_policy(
        matrix, quality_targets, cost_targets, groups, qualities, costs
    )
    changed_qualities = qualities.copy()
    changed_qualities[:2] = [[0.0, 1000.0], [1000.0, 0.0]]
    changed_targets = changed_qualities[:, 1] - changed_qualities[:, 0]
    changed = _nested_grouped_policy(
        matrix, changed_targets, cost_targets, groups, changed_qualities, costs
    )

    assert changed[:2].tolist() == original[:2].tolist()


def test_global_development_selection_can_be_optimistic() -> None:
    import numpy as np

    from experiments.train_falcon_mode_router import (
        _grouped_predictions,
        _nested_grouped_policy,
        _select_policy,
        _totals,
    )

    matrix = np.asarray([[-0.33], [-0.88], [-0.66], [-0.67], [0.38], [-0.11]])
    groups = np.asarray(["a", "a", "b", "b", "c", "c"])
    qualities = np.asarray(
        [[5.0, 4.0], [4.0, 4.0], [8.0, 5.0], [3.0, 9.0], [8.0, 3.0], [2.0, 2.0]]
    )
    costs = np.asarray(
        [[0.0, 2.0], [2.0, 2.0], [3.0, 3.0], [2.0, 2.0], [0.0, 1.0], [3.0, 3.0]]
    )
    quality_targets = qualities[:, 1] - qualities[:, 0]
    cost_targets = costs[:, 0] - costs[:, 1]
    quality_oof, cost_oof = _grouped_predictions(
        matrix, quality_targets, cost_targets, groups
    )
    tradeoff, threshold = _select_policy(quality_oof, cost_oof, qualities, costs)
    globally_selected = quality_oof + tradeoff * cost_oof > threshold
    nested_selected = _nested_grouped_policy(
        matrix, quality_targets, cost_targets, groups, qualities, costs
    )

    assert _totals(globally_selected, qualities, costs)["quality"] == 30.0
    assert _totals(nested_selected, qualities, costs)["quality"] == 24.0


@pytest.mark.parametrize("action", ["layout", "plain"])
def test_learned_mode_router_runs_heron_once_and_exactly_one_decoder(
    action: str,
) -> None:
    calls = {"detect": 0, "layout": 0, "plain": 0}

    class Engine(FakeLayoutEngine):
        def generate_with_layout(self, images, **options):
            calls["layout"] += 1
            assert len(options["detections"]) == 1
            return super().generate_with_layout(images, **options)

        def generate_plain(self, images, **options):
            calls["plain"] += 1
            return [
                serve_falcon_layout.DecodedText(
                    "full page", {"stop_token_seen": True, "temperature": 0.0}
                )
            ]

    def detect(images):
        calls["detect"] += 1
        return [
            {
                "raw": [
                    {
                        "label": "text",
                        "box": [1, 1, 39, 39],
                        "score": 0.9,
                        "query_id": 7,
                        "class_scores": {"text": 0.9, "table": 0.1},
                    }
                ],
                "layout": [
                    {
                        "category": "text",
                        "bbox": [1, 1, 39, 39],
                        "score": 0.9,
                        "query_id": 7,
                        "class_scores": {"text": 0.9, "table": 0.1},
                    }
                ],
            }
        ]

    engine = Engine(
        [{"category": "text", "bbox": [1, 1, 39, 39], "score": 0.9, "text": "crop"}]
    )
    server = _start_layout_server(
        engine,
        detect_route=detect,
        route_model=_route_model(action),
        extractor_metadata=_route_extractor(),
    )
    try:
        body = _post_layout_read(server, Image.new("RGB", (40, 40), "white"))
    finally:
        server.shutdown()
        server.server_close()

    assert calls == {
        "detect": 1,
        "layout": int(action == "layout"),
        "plain": int(action == "plain"),
    }
    assert body["routing"]["action"] == action
    assert (
        body["routing"]["training_action"]
        == {
            "layout": "layout_selective_qwen",
            "plain": "plain_falcon",
        }[action]
    )
    assert body["routing"]["model"] == {
        "id": "test-falcon-mode-router",
        "version": 1,
    }
    assert [element["category"] for element in body["elements"]] == [
        "page_text" if action == "plain" else "text"
    ]


def test_plain_route_failure_is_visible_without_layout_fallback() -> None:
    calls = {"detect": 0, "layout": 0, "plain": 0}

    class Engine(FakeLayoutEngine):
        def generate_with_layout(self, images, **options):
            calls["layout"] += 1
            return super().generate_with_layout(images, **options)

        def generate_plain(self, images, **options):
            calls["plain"] += 1
            raise RuntimeError("plain decoder failed")

    def detect(images):
        calls["detect"] += 1
        return [
            {
                "raw": [{"label": "text", "box": [0, 0, 40, 40], "score": 0.9}],
                "layout": [{"category": "text", "bbox": [0, 0, 40, 40], "score": 0.9}],
            }
        ]

    server = _start_layout_server(
        Engine([]),
        detect_route=detect,
        route_model=_route_model("plain"),
        extractor_metadata=_route_extractor(),
    )
    try:
        with pytest.raises(urllib.error.HTTPError) as failure:
            _post_layout_read(server, Image.new("RGB", (40, 40), "white"))
        assert failure.value.code == 422
        assert json.loads(failure.value.read()) == {"error": "falcon_layout_failed"}
    finally:
        server.shutdown()
        server.server_close()
    assert calls == {"detect": 1, "layout": 0, "plain": 1}


def test_plain_route_reader_returns_page_region_with_router_provenance(
    tmp_path: Path,
) -> None:
    from ocr_pipeline.falcon_layout import FalconLayoutReader

    engine = FakeLayoutEngine([])
    engine.generate_plain = lambda **options: [
        serve_falcon_layout.DecodedText(
            "# Full page", {"stop_token_seen": True, "temperature": 0.0}
        )
    ]
    server = _start_layout_server(
        engine,
        detect_route=lambda images: [{"raw": [], "layout": []}],
        route_model=_route_model("plain"),
        extractor_metadata=_route_extractor(),
    )
    image_path = tmp_path / "page.png"
    Image.new("RGB", (40, 60), "white").save(image_path)
    try:
        regions = FalconLayoutReader(f"http://127.0.0.1:{server.server_port}").read(
            image_path, 3
        )
    finally:
        server.shutdown()
        server.server_close()

    assert len(regions) == 1
    assert regions[0].kind == "page_text"
    assert regions[0].text == "Full page"
    assert regions[0].bounding_box.right == 40
    assert regions[0].bounding_box.bottom == 60
    assert regions[0].text_provenance["method"] == "falcon_perception_plain_ocr"
    assert regions[0].text_provenance["routing"]["model"]["id"] == (
        "test-falcon-mode-router"
    )


def test_nonterminated_table_read_retries_and_retains_original_evidence() -> None:
    looped_text = serve_falcon_layout.DecodedText(
        "$1,659 $546 " * 24, {"stop_token_seen": False, "temperature": 0.0}
    )
    clean_text = serve_falcon_layout.DecodedText(
        "$1,659\n$546\nTotal $2,205", {"stop_token_seen": True, "temperature": 0.5}
    )
    engine = FakeLayoutEngine(
        elements=[
            {
                "category": "table",
                "bbox": [0.0, 0.0, 40.0, 40.0],
                "score": 0.9,
                "text": looped_text,
            }
        ],
        plain_by_temperature={
            0.0: looped_text,
            0.2: looped_text,
            0.5: clean_text,
            0.8: clean_text,
        },
    )
    server = _start_layout_server(engine, category_by_layout={"table": "table"})
    try:
        body = _post_layout_read(server, Image.new("RGB", (40, 40), "white"))
    finally:
        server.shutdown()
        server.server_close()

    element = body["elements"][0]
    assert element["text"] == clean_text
    assert element["truncated"] is False
    assert element["retries"] == 3
    assert engine.retry_calls[0]["min_image_size"] == 256
    assert "looped" not in element
    assert element["read_attempts"][0] == {
        "text": looped_text,
        "generation": looped_text.generation,
    }
    assert element["read_attempts"][-1] == {
        "text": clean_text,
        "generation": clean_text.generation,
    }


def test_nonterminated_table_retains_failed_attempt_evidence() -> None:
    looped_text = serve_falcon_layout.DecodedText(
        "$1,659 $546 " * 24, {"stop_token_seen": False, "temperature": 0.0}
    )
    engine = FakeLayoutEngine(
        elements=[
            {
                "category": "table",
                "bbox": [0.0, 0.0, 40.0, 40.0],
                "score": 0.9,
                "text": looped_text,
            }
        ],
        plain_by_temperature={
            0.0: looped_text,
            0.2: looped_text,
            0.5: looped_text,
            0.8: looped_text,
        },
    )
    server = _start_layout_server(engine, category_by_layout={"table": "table"})
    try:
        body = _post_layout_read(server, Image.new("RGB", (40, 40), "white"))
    finally:
        server.shutdown()
        server.server_close()

    element = body["elements"][0]
    assert element["text"] == looped_text
    assert element["truncated"] is True
    assert element["retries"] == 4
    assert len(element["read_attempts"]) == 5
    assert all(
        not attempt["generation"]["stop_token_seen"]
        for attempt in element["read_attempts"]
    )


def test_eos_decoder_cycle_rereads_and_quarantines_unrecovered_text() -> None:
    from fastapi.testclient import TestClient

    from ocr_pipeline.demo import create_app
    from ocr_pipeline.falcon_layout import FalconLayoutReader

    original = serve_falcon_layout.DecodedText(
        "ζ " * 1250,
        {"stop_token_seen": True, "decoder_cycle": True, "generated_tokens": 2500},
    )
    engine = FakeLayoutEngine(
        [{"category": "text", "bbox": [0, 0, 40, 40], "score": 0.9, "text": original}],
        {temperature: original for temperature in (0.0, 0.2, 0.5, 0.8)},
    )
    server = _start_layout_server(engine, category_by_layout={"text": "text"})
    buffer = io.BytesIO()
    Image.new("RGB", (40, 40), "white").save(buffer, format="PNG")
    try:
        reader = FalconLayoutReader(f"http://127.0.0.1:{server.server_port}")
        with TestClient(create_app(reader)) as client:
            response = client.post(
                "/api/process",
                files={"file": ("page.png", buffer.getvalue(), "image/png")},
            )
            assert response.status_code == 200
            payload = response.json()
            page = payload["result"]["pages"][0]
            region = page["regions"][0]
            assert region["resolution"] == "unreadable"
            assert region["text"] == original.strip()
            assert region["text_provenance"]["read_retries"] == 4
            assert len(region["text_provenance"]["read_attempts"]) == 5
            assert page["route"] == "review"
            markdown = client.get(
                f"/api/sessions/{payload['session_id']}/result.md"
            ).text
            assert "ζ" not in markdown
            assert "ζ" not in page["text"]["value"]
    finally:
        server.shutdown()
        server.server_close()


def test_exclusive_form_is_visible_as_unresolved_source_crop(tmp_path: Path) -> None:
    from ocr_pipeline.falcon_layout import FalconLayoutReader
    from ocr_pipeline.markdown_refinement import _crop_groups

    engine = FakeLayoutEngine(
        [{"category": "form", "bbox": [2, 2, 38, 38], "score": 0.8, "text": ""}]
    )
    server = _start_layout_server(
        engine,
        detect_route=lambda images: [
            {
                "raw": [{"label": "form", "box": [2, 2, 38, 38], "score": 0.8}],
                "layout": [{"category": "form", "bbox": [2, 2, 38, 38], "score": 0.8}],
            }
        ],
        route_model=_route_model("layout"),
        extractor_metadata=_route_extractor(),
    )
    image = tmp_path / "form.png"
    Image.new("RGB", (40, 40), "white").save(image)
    try:
        region = FalconLayoutReader(f"http://127.0.0.1:{server.server_port}").read(
            image, 1
        )[0]
        assert region.kind == "figure"
        assert region.resolution == "unreadable"
        assert region.text_provenance["layout_category"] == "form"
        assert region.text_provenance["routing"]["action"] == "layout"
        assert _crop_groups([asdict(region)], 40, 40)[0]["box"] == {
            "left": 2,
            "top": 2,
            "right": 38,
            "bottom": 38,
        }
    finally:
        server.shutdown()
        server.server_close()


def test_checkbox_region_keeps_router_provenance(tmp_path: Path) -> None:
    from ocr_pipeline.falcon_layout import FalconLayoutReader

    engine = FakeLayoutEngine(
        [
            {
                "category": "checkbox-selected",
                "bbox": [2, 2, 20, 20],
                "score": 0.9,
                "text": "Consent",
            }
        ]
    )
    server = _start_layout_server(
        engine,
        detect_route=lambda images: [
            {
                "raw": [
                    {
                        "label": "checkbox_selected",
                        "box": [2, 2, 20, 20],
                        "score": 0.9,
                    }
                ],
                "layout": [
                    {
                        "category": "checkbox-selected",
                        "bbox": [2, 2, 20, 20],
                        "score": 0.9,
                    }
                ],
            }
        ],
        route_model=_route_model("layout"),
        extractor_metadata=_route_extractor(),
    )
    image = tmp_path / "checkbox.png"
    Image.new("RGB", (40, 40), "white").save(image)
    try:
        region = FalconLayoutReader(f"http://127.0.0.1:{server.server_port}").read(
            image, 1
        )[0]
    finally:
        server.shutdown()
        server.server_close()

    assert region.kind == "checkbox"
    assert region.text_provenance["routing"]["action"] == "layout"


def test_form_only_upload_preserves_crop_and_reports_unresolved_page() -> None:
    from fastapi.testclient import TestClient

    from ocr_pipeline.contracts import BoundingBox, TextRegion
    from ocr_pipeline.demo import create_app
    from ocr_pipeline.falcon_layout import FalconLayoutReader

    engine = FakeLayoutEngine(
        [{"category": "form", "bbox": [2, 2, 38, 38], "score": 0.8, "text": ""}]
    )
    server = _start_layout_server(engine)
    buffer = io.BytesIO()
    Image.new("RGB", (40, 40), "white").save(buffer, format="PNG")
    try:
        reader = FalconLayoutReader(f"http://127.0.0.1:{server.server_port}")
        with TestClient(create_app(reader)) as client:
            response = client.post(
                "/api/process",
                files={"file": ("form.png", buffer.getvalue(), "image/png")},
            )
            assert response.status_code == 200
            payload = response.json()
            result = payload["result"]
            assert result["status"] == "failed"
            assert [failure["code"] for failure in result["failures"]] == [
                "no_text_detected"
            ]
            page = result["pages"][0]
            assert page["failure_ids"] == [result["failures"][0]["id"]]
            assert page["route"] == "review"
            assert page["text"]["value"] == ""
            assert len(page["regions"]) == 1
            assert page["regions"][0]["resolution"] == "unreadable"
            assert page["regions"][0]["text_provenance"]["layout_category"] == "form"
            assert page["regions"][0]["structure"]["role"] == "form_source"
            assert page["regions"][0]["structure"]["image"]["data"]
            markdown = client.get(
                f"/api/sessions/{payload['session_id']}/result.md"
            ).text
            assert "Status: failed" in markdown
            assert "ocr/no_text_detected" in markdown
            assert "## Page 1" in markdown

        class AcceptedRead:
            name = "accepted-read"

            def apply(self, image_path, page_number, regions):
                regions.append(
                    TextRegion(
                        id=f"p{page_number}-accepted-1",
                        kind="text",
                        text="Dose: 5 mg",
                        confidence=None,
                        bounding_box=BoundingBox(2, 2, 38, 38),
                        reading_order=2,
                        provider="test-correction",
                    )
                )
                return regions

        with TestClient(create_app(reader, stages=[AcceptedRead()])) as client:
            response = client.post(
                "/api/process",
                files={"file": ("form.png", buffer.getvalue(), "image/png")},
            )
            assert response.status_code == 200
            result = response.json()["result"]
            assert result["status"] == "success"
            assert result["failures"] == []
            assert result["pages"][0]["text"]["value"] == "Dose: 5 mg"
            assert len(result["pages"][0]["regions"]) == 2
    finally:
        server.shutdown()
        server.server_close()


def test_eos_decoder_cycle_keeps_successful_reread_as_alternative() -> None:
    original = serve_falcon_layout.DecodedText(
        "glyph " * 300, {"stop_token_seen": True, "decoder_cycle": True}
    )
    recovered = serve_falcon_layout.DecodedText(
        "Emergency contact", {"stop_token_seen": True, "temperature": 0.2}
    )
    engine = FakeLayoutEngine(
        [{"category": "text", "bbox": [0, 0, 40, 40], "score": 0.9, "text": original}],
        {0.0: original, 0.2: recovered},
    )
    server = _start_layout_server(engine, category_by_layout={"text": "text"})
    try:
        element = _post_layout_read(server, Image.new("RGB", (40, 40)))["elements"][0]
    finally:
        server.shutdown()
        server.server_close()
    assert element["text"] == recovered
    assert element["retries"] == 2
    assert "looped" not in element
    assert [item["text"] for item in element["read_attempts"]] == [
        original,
        original,
        recovered,
    ]


def test_crop_endpoint_reports_decoder_cycle_without_claiming_success() -> None:
    reading = serve_falcon_layout.DecodedText(
        "glyph " * 300,
        {
            "stop_token_seen": True,
            "decoder_cycle": True,
            "generated_tokens": 600,
            "temperature": 0.0,
        },
        0,
    )
    engine = FakeLayoutEngine([])
    engine.generate_plain = lambda **options: [reading]
    server = _start_layout_server(engine)
    buffer = io.BytesIO()
    Image.new("RGB", (40, 40), "white").save(buffer, format="PNG")
    image = base64.b64encode(buffer.getvalue()).decode()

    def request(return_generation):
        payload = {
            "images": [image],
            "categories": ["text"],
            "return_generation": return_generation,
        }
        return urllib.request.Request(
            f"http://127.0.0.1:{server.server_port}/generate",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )

    try:
        with pytest.raises(urllib.error.HTTPError) as failure:
            urllib.request.urlopen(request(False), timeout=5)
        assert failure.value.code == 422
        with urllib.request.urlopen(request(True), timeout=5) as reply:
            result = json.loads(reply.read())["results"][0]
        assert result["status"] == "failed"
        assert result["error"] == "decoder_cycle"
        assert result["text"] == reading
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize("category", ["text", "checkbox-selected"])
def test_upload_keeps_completed_retries_under_review(category):
    from fastapi.testclient import TestClient
    from ocr_pipeline.demo import create_app
    from ocr_pipeline.falcon_layout import FalconLayoutReader

    original = serve_falcon_layout.DecodedText(
        "unfinished read", {"stop_token_seen": False, "temperature": 0.0}
    )
    retry = serve_falcon_layout.DecodedText(
        "completed alternative", {"stop_token_seen": True, "temperature": 0.0}
    )
    engine = FakeLayoutEngine(
        [
            {
                "category": category,
                "bbox": [0, 0, 40, 40],
                "score": 0.9,
                "text": original,
            }
        ],
        {0.0: retry},
    )
    server = _start_layout_server(engine, category_by_layout={category: "text"})
    buffer = io.BytesIO()
    Image.new("RGB", (40, 40), "white").save(buffer, format="PNG")
    try:
        reader = FalconLayoutReader(f"http://127.0.0.1:{server.server_port}")
        with TestClient(create_app(reader)) as client:
            response = client.post(
                "/api/process",
                files={"file": ("page.png", buffer.getvalue(), "image/png")},
            )
            assert response.status_code == 200
            payload = response.json()
            page = payload["result"]["pages"][0]
            region = page["regions"][0]
            assert region["text"] == retry
            assert region["resolution"] == "conflicting"
            assert page["route"] == "review"
            attempts = region["text_provenance"]["read_attempts"]
            assert [item["text"] for item in attempts] == [original, retry]
            exported = client.get(
                f"/api/sessions/{payload['session_id']}/result.json"
            ).json()
            assert exported["pages"][0]["regions"][0] == region
            markdown = client.get(
                f"/api/sessions/{payload['session_id']}/result.md"
            ).text
            assert retry not in markdown
            assert retry not in page["text"]["value"]
            if category == "checkbox-selected":
                assert "[x] [unresolved text]" in markdown
                assert "[x] [unresolved text]" in page["text"]["value"]
    finally:
        server.shutdown()
        server.server_close()


def test_upload_promotes_only_one_leading_decoded_ballot_mark() -> None:
    from fastapi.testclient import TestClient
    from ocr_pipeline.demo import create_app
    from ocr_pipeline.falcon_layout import FalconLayoutReader

    generation = {
        "stop_token_seen": True,
        "temperature": 0.0,
        "token_score": 0.9,
    }
    readings = [
        serve_falcon_layout.DecodedText("- □ Wheezes", generation),
        serve_falcon_layout.DecodedText("Cough □", generation),
        serve_falcon_layout.DecodedText("□ □ Fever", generation),
    ]
    engine = FakeLayoutEngine(
        [
            {
                "category": "list-item",
                "bbox": [0, index * 10, 40, index * 10 + 10],
                "score": 0.8,
                "text": reading,
            }
            for index, reading in enumerate(readings)
        ]
    )
    server = _start_layout_server(engine)
    buffer = io.BytesIO()
    Image.new("RGB", (40, 40), "white").save(buffer, format="PNG")
    try:

        class Geometry:
            def read_with_merge_level(
                self, image_path: Path, page_number: int, merge_level: str
            ) -> list[TextRegion]:
                del image_path, merge_level
                return [
                    TextRegion(
                        id="word-wheezes",
                        kind="word",
                        text="Wheezes",
                        confidence=0.95,
                        bounding_box=BoundingBox(4, 1, 36, 9),
                        reading_order=0,
                        provider="geometry-reader",
                    )
                ]

        reader = FusedReader(
            FalconLayoutReader(f"http://127.0.0.1:{server.server_port}"),
            Geometry(),
        )
        with TestClient(create_app(reader)) as client:
            response = client.post(
                "/api/process",
                files={"file": ("page.png", buffer.getvalue(), "image/png")},
            )
            assert response.status_code == 200
            payload = response.json()
            page = payload["result"]["pages"][0]
            checkbox, internal, multiple = page["regions"]
            assert checkbox["kind"] == "checkbox"
            assert checkbox["text"] == "Wheezes"
            assert checkbox["resolution"] == "resolved"
            assert checkbox["bounding_box"] == {
                "left": 0,
                "top": 0,
                "right": 40,
                "bottom": 10,
            }
            assert checkbox["structure"]["state"] == "unselected"
            assert checkbox["structure"]["label"] == "Wheezes"
            assert checkbox["structure"]["label_evidence_ids"] == []
            assert [
                word["text"] for word in checkbox["structure"]["word_evidence"]
            ] == ["Wheezes"]
            assert checkbox["text_provenance"]["raw_text"] == readings[0]
            assert checkbox["text_provenance"]["generation"] == generation
            assert checkbox["confidence"] == 0.9
            assert checkbox["text_provenance"]["confidence_meaning"] == (
                "decoder token score, from the text reader"
            )
            assert checkbox["text_provenance"]["recognition_confidence_mean"] == 0.95
            assert (
                checkbox["text_provenance"]["method"]
                == "decoded_ballot_glyph_promotion"
            )
            assert checkbox["text_provenance"]["state_source"] == "decoded_ballot_glyph"
            assert checkbox["text_provenance"]["state_glyph"] == "□"
            assert internal["kind"] == multiple["kind"] == "list-item"
            assert page["text"]["value"].count("Wheezes") == 1
            markdown = client.get(
                f"/api/sessions/{payload['session_id']}/result.md"
            ).text
            assert markdown.count("[ ] Wheezes") == 1
            assert markdown.count("Wheezes") == 1
            assert "Geometry recognition score" in client.get("/").text
    finally:
        server.shutdown()
        server.server_close()


def test_upload_renders_checkbox_label_instead_of_decoder_table_markup():
    from fastapi.testclient import TestClient
    from ocr_pipeline.demo import create_app
    from ocr_pipeline.falcon_layout import FalconLayoutReader

    reading = serve_falcon_layout.DecodedText(
        "<table><tr><td>☐</td><td>Please Comment</td></tr></table>",
        {"stop_token_seen": True, "temperature": 0.0},
    )
    engine = FakeLayoutEngine(
        [
            {
                "category": "checkbox-unselected",
                "bbox": [0, 0, 40, 40],
                "score": 0.9,
                "text": reading,
            }
        ]
    )
    server = _start_layout_server(
        engine, category_by_layout={"checkbox-unselected": "text"}
    )
    buffer = io.BytesIO()
    Image.new("RGB", (40, 40), "white").save(buffer, format="PNG")
    try:
        reader = FalconLayoutReader(f"http://127.0.0.1:{server.server_port}")
        with TestClient(create_app(reader)) as client:
            response = client.post(
                "/api/process",
                files={"file": ("page.png", buffer.getvalue(), "image/png")},
            )
            assert response.status_code == 200
            payload = response.json()
            region = payload["result"]["pages"][0]["regions"][0]
            assert region["text"] == "Please Comment"
            assert region["text_provenance"]["raw_text"] == reading
            markdown = client.get(
                f"/api/sessions/{payload['session_id']}/result.md"
            ).text
            assert "[ ] Please Comment" in markdown
            assert "<table" not in markdown
            demo = client.get("/").text
            assert "Actual pixels" in demo
            assert "Qwen reread accepted · Calibrated confidence unavailable" in demo
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("☐ Please Comment", "Please Comment"),
        ("Pale □", "Pale"),
        ("■ Glucose Test Strips", "Glucose Test Strips"),
    ],
)
def test_checkbox_label_removes_presentation_marks(value: str, expected: str) -> None:
    from ocr_pipeline.falcon_layout import _checkbox_label

    assert _checkbox_label(value) == expected


@pytest.mark.parametrize(
    "text",
    [
        "this " * 12,
        "<table>" + "<tr><td>NA</td></tr>" * 12 + "</table>",
        "$1,127<br>11.4%<br>" + "$1,659<br>$546<br>" * 24,
    ],
)
def test_http_preserves_eos_terminated_repeated_words_and_table_values(text):
    reading = serve_falcon_layout.DecodedText(
        text, {"stop_token_seen": True, "temperature": 0.0}
    )
    engine = FakeLayoutEngine(
        [{"category": "table", "bbox": [0, 0, 40, 40], "score": 0.9, "text": reading}]
    )
    engine.generate_plain = lambda **options: [reading]
    server = _start_layout_server(engine, category_by_layout={"table": "table"})
    try:
        with Image.new("RGB", (40, 40)) as image:
            body = _post_layout_read(server, image)
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
        request = urllib.request.Request(
            f"http://127.0.0.1:{server.server_address[1]}/generate",
            data=json.dumps(
                {
                    "images": [base64.b64encode(buffer.getvalue()).decode()],
                    "categories": ["table"],
                }
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=5) as reply:
            assert json.loads(reply.read())["texts"] == [text]
    finally:
        server.shutdown()
        server.server_close()
    assert body["elements"][0]["text"] == text
    assert body["elements"][0]["generation"] == reading.generation
    assert "retries" not in body["elements"][0]
    assert "looped" not in body["elements"][0]
    assert engine.retry_calls == []


def test_layout_http_preserves_overlapping_modalities_and_tiny_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the real service engine with only model execution substituted."""
    import sys
    from types import SimpleNamespace

    import torch

    probabilities = torch.tensor(
        [[0.1, 0.95, 0.1], [0.9, 0.85, 0.1], [0.7, 0.1, 0.1], [0.1, 0.1, 0.6]]
    )
    boxes = torch.tensor(
        [[0, 0, 80, 40], [0, 0, 80, 40], [1, 1, 5, 9], [70, 1, 79, 10]],
        dtype=torch.float32,
    )
    normalized = boxes / torch.tensor([80, 40, 80, 40])
    model_output = SimpleNamespace(
        logits=torch.logit(probabilities)[None],
        pred_boxes=torch.cat(
            [
                (normalized[:, :2] + normalized[:, 2:]) / 2,
                normalized[:, 2:] - normalized[:, :2],
            ],
            dim=-1,
        )[None],
    )

    class LayoutModel:
        config = SimpleNamespace(
            id2label={0: "text", 1: "picture", 2: "checkbox_selected"}
        )

        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, **inputs):
            return model_output

    class LayoutProcessor:
        def __call__(self, **inputs):
            return SimpleNamespace(to=lambda device: {})

        def post_process_object_detection(self, outputs, **options):
            # The stock focal postprocessor selects class/query pairs, so query 1
            # appears twice and consumes the slot that should hold query 3.
            scores, indices = outputs.logits[0].sigmoid().flatten().topk(4)
            return [
                {"scores": scores, "labels": indices % 3, "boxes": boxes[indices // 3]}
            ]

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoImageProcessor=SimpleNamespace(
                from_pretrained=lambda name, **options: LayoutProcessor()
            ),
            AutoModelForObjectDetection=SimpleNamespace(
                from_pretrained=lambda name, **options: LayoutModel()
            ),
        ),
    )
    tokenizer = SimpleNamespace(
        eos_token_id=0,
        end_of_query_token_id=2,
        decode=lambda ids: "I:",
        _tok=SimpleNamespace(
            encode=lambda text, **kwargs: SimpleNamespace(ids=[1], offsets=[(0, 2)])
        ),
    )
    crop_sizes = []

    class CropSequence(SimpleNamespace):
        def __init__(self, *, image, **kwargs):
            super().__init__(_image_raw=image, **kwargs)

    class Decoder:
        def __init__(self, model, tokenizer, processor, **options):
            assert options["max_seq_length"] == model.args.max_seq_len == 16384
            self.tokenizer = tokenizer

        def generate(self, sequences, *, temperature, **options):
            self.temperature = temperature
            for sequence in sequences:
                crop_sizes.append(sequence._image_raw.size)
                sequence._output_ids = [torch.tensor(1), torch.tensor(0)]
                sequence._output_probs = [torch.tensor(0.75), torch.tensor(0.9)]
            # Completion may differ from source order; identity must be retained.
            return list(reversed(sequences))

    class OfficialOCR(Decoder):
        _make_ocr_prompt = staticmethod(lambda category: category)

        def _stop_token_ids(self):
            return [0, 2]

        def generate_with_layout(self, *args, **kwargs):
            raise AssertionError("must bypass upstream cross-class overlap suppression")

    loaded_revisions = []

    def load_native_model(**kwargs):
        loaded_revisions.append(kwargs["hf_revision"])
        return SimpleNamespace(args=SimpleNamespace(max_seq_len=16384)), tokenizer, None

    monkeypatch.setitem(
        sys.modules,
        "falcon_perception",
        SimpleNamespace(
            load_and_prepare_model=load_native_model,
            setup_torch_config=lambda: None,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "falcon_perception.data",
        SimpleNamespace(ImageProcessor=lambda **kwargs: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "falcon_perception.paged_inference",
        SimpleNamespace(
            PagedInferenceEngine=Decoder,
            Sequence=CropSequence,
            SamplingParams=SimpleNamespace,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "falcon_perception.paged_ocr_inference",
        SimpleNamespace(
            OCRInferenceEngine=OfficialOCR, LAYOUT_TO_OCR_CATEGORY={"text": "text"}
        ),
    )
    engines = []

    def capture_engine(engine, *args, **kwargs):
        engines.append(engine)
        assert kwargs["model"]["revision"] == loaded_revisions[-1]
        return SimpleNamespace(
            serve_forever=lambda **kwargs: None, server_close=lambda: None
        )

    create_server = serve_falcon_layout.create_server
    monkeypatch.setattr(serve_falcon_layout, "create_server", capture_engine)
    serve_falcon_layout.main([])
    assert loaded_revisions == [FALCON_MODEL_REVISION]
    for revision in ("main", "v1.5", "42ec56b"):
        with pytest.raises(SystemExit):
            serve_falcon_layout.main(["--model-revision", revision])
    for option, value in (
        ("--max-new-tokens", "-1"),
        ("--max-image-size", "0"),
    ):
        with pytest.raises(SystemExit):
            serve_falcon_layout.main([option, value])
    assert loaded_revisions == [FALCON_MODEL_REVISION]
    monkeypatch.setattr(serve_falcon_layout, "create_server", create_server)
    server = _start_layout_server(engines[0])
    try:
        with Image.new("RGB", (80, 40), "white") as image:
            outputs = [_post_layout_read(server, image) for _ in range(3)]
    finally:
        server.shutdown()
        server.server_close()
    assert outputs[0] == outputs[1] == outputs[2]
    elements = outputs[0]["elements"]
    assert [item["category"] for item in elements] == [
        "picture",
        "text",
        "text",
        "checkbox-selected",
    ]
    assert [item["text"] for item in elements] == ["", "I:", "I:", "I:"]
    assert elements[2]["bbox"] == [1, 1, 5, 9]
    assert elements[2]["generation"]["token_score"] == pytest.approx(0.75)
    assert elements[2]["generation"]["tokens"] == [
        {"start": 0, "end": 2, "score": 0.75}
    ]
    assert elements[0]["generation"] == {}
    assert [element["query_id"] for element in elements] == [0, 1, 2, 3]
    assert elements[1]["class_scores"]["picture"] == pytest.approx(0.85)
    assert elements[1]["class_scores"]["text"] == pytest.approx(0.9)
    assert all(min(size) >= 16 for size in crop_sizes)

    engines[0].layout_detector._detector = lambda image: [
        {"label": "picture", "box": [0, 0, 80, 40], "score": 0.8}
    ]
    with Image.new("RGB", (80, 40)) as image:
        result = engines[0].generate_with_layout([image])
    assert result == [
        [{"category": "picture", "bbox": [0, 0, 80, 40], "score": 0.8, "text": ""}]
    ]
    assert len(crop_sizes) == 9


@pytest.mark.parametrize("route", ["generate", "read", "retry"])
@pytest.mark.parametrize("reset_fails", [False, True])
def test_layout_http_recovers_failed_generation_before_closing_images(
    route: str, reset_fails: bool
) -> None:
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace
    from urllib.error import HTTPError

    import torch

    drain_started = threading.Event()
    allow_tokenization = threading.Event()
    pixels_read = threading.Event()
    pool = ThreadPoolExecutor(max_workers=1)

    class Engine:
        def __init__(self):
            self.paged_kv_cache = SimpleNamespace(
                max_batch_size=3,
                free_batch_idx=[2, 1],
                capacity=[0, 0, 0],
                erase=self.erase,
            )
            self._input_pos = torch.ones(3, dtype=torch.int32)
            self._rope_pos_t = torch.ones(3, dtype=torch.int64)
            self._token_ids_ready, self._token_ids_count = object(), 2
            self._prefetch_futures = {}
            self.waiting, self.running, self.done = deque(), deque(), deque()
            self.failed_once = False
            self.calls = 0
            self.erased = []

        def erase(self, slot):
            assert pixels_read.is_set()
            assert slot != 0 and slot not in self.paged_kv_cache.free_batch_idx
            if reset_fails:
                raise RuntimeError("Injected allocator recovery failure")
            self.erased.append(slot)
            self.paged_kv_cache.free_batch_idx.append(slot)
            self.paged_kv_cache.capacity[slot] = 0

        def read(self, image):
            self.calls += 1
            if self.failed_once:
                assert set(self.paged_kv_cache.free_batch_idx) == {1, 2}
                return serve_falcon_layout.DecodedText(
                    "literal",
                    {"generated_tokens": 1, "stop_token_seen": True, "temperature": 0},
                    0,
                )
            self.failed_once = True
            self.paged_kv_cache.free_batch_idx.clear()
            # Slot two failed immediately after allocation, before reserving pages.
            self.paged_kv_cache.capacity[1] = 128
            started = threading.Event()

            def tokenize():
                started.set()
                assert allow_tokenization.wait(5)
                assert image.getpixel((0, 0)) == (255, 255, 255)
                pixels_read.set()

            future = pool.submit(tokenize)
            assert started.wait(5)
            cancel = future.cancel

            def notify_cancel():
                drain_started.set()
                return cancel()

            future.cancel = notify_cancel
            self._prefetch_futures[1] = future
            raise RuntimeError("Injected generation failure after allocation")

        def generate_plain(self, images, **options):
            return [self.read(images[0])]

        def generate_with_layout(self, images, **options):
            text = (
                serve_falcon_layout.DecodedText(
                    "unfinished", {"stop_token_seen": False}
                )
                if route == "retry"
                else self.read(images[0])
            )
            return [
                [{"category": "text", "bbox": [0, 0, 32, 32], "score": 1, "text": text}]
            ]

    engine = Engine()
    pointers = (engine._input_pos.data_ptr(), engine._rope_pos_t.data_ptr())
    server = _start_layout_server(engine, category_by_layout={"text": "text"})
    image = Image.new("RGB", (32, 32), "white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image.close()
    encoded = base64.b64encode(buffer.getvalue()).decode()
    payload = (
        {"images": [encoded], "categories": ["text"], "return_generation": True}
        if route == "generate"
        else {"image": encoded}
    )

    def request(path, body=None):
        req = urllib.request.Request(
            f"http://127.0.0.1:{server.server_port}/{path}",
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Content-Type": "application/json"},
        )
        try:
            reply = urllib.request.urlopen(req, timeout=5)
        except HTTPError as error:
            reply = error
        with reply:
            return reply.status, json.loads(reply.read())

    path = "generate" if route == "generate" else "read"
    try:
        with ThreadPoolExecutor(max_workers=1) as requests_pool:
            first = requests_pool.submit(request, path, payload)
            assert drain_started.wait(5)
            assert not first.done()
            allow_tokenization.set()
            first_status, first_body = first.result(timeout=5)
        second_status, _ = request(path, payload)
        health_status, health = request("health")
        assert health["pid"] == os.getpid()
        if route == "generate":
            assert first_body["model"] == health["model"]
        assert pixels_read.is_set() and not engine._prefetch_futures
        assert pointers == (engine._input_pos.data_ptr(), engine._rope_pos_t.data_ptr())
        if reset_fails:
            assert first_status == second_status == health_status == 503
            assert health["status"] == "unavailable" and engine.calls == 1
            assert first_body["error"] == "falcon_engine_unavailable"
        else:
            assert first_status == 422 and second_status == health_status == 200
            assert engine.erased == [1, 2] and engine.calls == 2
            assert engine.paged_kv_cache.capacity == [0, 0, 0]
            assert not engine._input_pos.any() and not engine._rope_pos_t.any()
            assert engine._token_ids_ready is None and engine._token_ids_count == 0
    finally:
        allow_tokenization.set()
        pool.shutdown(wait=True, cancel_futures=True)
        server.shutdown()
        server.server_close()
