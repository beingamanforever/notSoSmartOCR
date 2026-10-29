"""Upload-to-export coverage of the same crop reader used by the benchmark."""

import io
import json
from functools import partial

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from ocr_pipeline.contracts import BoundingBox, TextRegion
from ocr_pipeline.demo import create_app
from ocr_pipeline.markdown_refinement import refine_document
from ocr_pipeline.openrouter import QWEN_37_FLASH_MODEL, budgeted_transport


@pytest.mark.parametrize("mode", ["correct", "unavailable", "not_needed", "invalid"])
def test_crop_result_reaches_session_and_exports(monkeypatch, tmp_path, mode):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture")
    calls = []

    class Reader:
        name = "falcon-fixture"

        def read(self, path, page_number):
            return [
                TextRegion(
                    "printed",
                    "paragraph",
                    "Printed heading",
                    None,
                    BoundingBox(0, 0, 120, 30),
                    1,
                    self.name,
                ),
                TextRegion(
                    "note",
                    "paragraph" if mode == "not_needed" else "handwriting",
                    "Initial reading",
                    None,
                    BoundingBox(0, 40, 120, 80),
                    2,
                    self.name,
                ),
            ]

    def transport(request, timeout):
        body = json.loads(request.data)
        calls.append(body)
        assert body["model"] == QWEN_37_FLASH_MODEL
        assert body["reasoning"] == {"enabled": False}
        assert "Initial reading" not in json.dumps(body)
        if mode == "unavailable":
            return 503, {}, b'{"error":{"message":"Unavailable"}}'
        answer = {"crops": [{"region_id": "crop-1", "markdown": "Corrected note"}]}
        if mode == "invalid":
            answer["crops"] = []
        return (
            200,
            {},
            json.dumps(
                {
                    "model": body["model"],
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": json.dumps(answer)},
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 10,
                        "cost": 0.001,
                    },
                }
            ).encode(),
        )

    image = io.BytesIO()
    Image.new("RGB", (120, 80), "white").save(image, format="PNG")
    transport_with_budget = partial(
        budgeted_transport,
        database=tmp_path / "cost.sqlite",
        budget_usd=0.25,
        transport=transport,
    )
    app = create_app(
        Reader(), crop_refiner=partial(refine_document, transport=transport_with_budget)
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/process", files={"file": ("page.png", image.getvalue(), "image/png")}
        )
        assert response.status_code == 200
        payload = response.json()
        session = payload["session_id"]
        saved = client.get(f"/api/sessions/{session}").json()
        exported = client.get(f"/api/sessions/{session}/result.json").json()
        markdown = client.get(f"/api/sessions/{session}/result.md").text
        assert exported == saved["result"] == payload["result"]
        page = exported["pages"][0]
        assert "Printed heading" in markdown
        assert "qwen-crop-refinement" in payload["pipeline_stages"]
        assert len(calls) == (0 if mode == "not_needed" else 1)
        assert client.get(f"/api/sessions/{session}/pages/1").status_code == 200
        if mode == "correct":
            assert "Corrected note" in markdown and "Initial reading" not in markdown
            assert "Corrected note" in page["text"]["value"]
            note = next(r for r in page["regions"] if r["id"] == "note")
            assert note["provider"] == QWEN_37_FLASH_MODEL
            assert note["alternatives"][-1]["text"] == "Initial reading"
            assert note["alternatives"][-1]["decision_state"] == "superseded"
            assert page["refinement"]["cost"] == 0.001
        else:
            assert "Initial reading" in markdown
            if mode != "not_needed":
                assert exported["status"] == "partial"
                assert page["route"] == "review"
                assert exported["failures"][-1]["stage"] == "qwen-crop-refinement"
                assert page["refinement"]["canonical_unchanged"]
        if mode == "unavailable":
            # An unpriced failure retains its reservation across a second upload.
            second = client.post(
                "/api/process",
                files={"file": ("page.png", image.getvalue(), "image/png")},
            ).json()
            assert len(calls) == 1
            assert second["result"]["pages"][0]["refinement"]["attempts"] == 0
            assert "budget exhausted" in second["result"]["failures"][-1]["message"]


@pytest.mark.parametrize(
    "angle, expected_box",
    [
        (90, [70, 36, 90, 100]),
        (180, [60, 116, 100, 148]),
        (270, [30, 100, 50, 164]),
    ],
)
def test_rotated_form_crops_and_returned_controls_share_source_coordinates(
    monkeypatch, angle, expected_box
):
    import base64
    from PIL import ImageDraw

    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture")
    source = Image.new("RGB", (120, 200), "white")
    ImageDraw.Draw(source).rectangle((10, 20, 35, 50), fill="black")
    expected = source.crop((10, 20, 110, 180)).rotate(angle, expand=True)

    class Reader:
        name = "rotated-form-fixture"

        def read(self, path, page_number):
            return [
                TextRegion(
                    "form",
                    "figure",
                    "",
                    None,
                    BoundingBox(10, 20, 110, 180),
                    1,
                    self.name,
                    text_provenance={"orientation": {"angle": angle}},
                    resolution="unreadable",
                    structure={"role": "form_source"},
                ),
                TextRegion(
                    "known",
                    "checkbox",
                    "Known option",
                    None,
                    BoundingBox(15, 25, 25, 35),
                    2,
                    self.name,
                    structure={"state": "unselected"},
                ),
            ]

    def transport(request, timeout):
        body = json.loads(request.data)
        images = [
            part["image_url"]["url"]
            for message in body["messages"]
            if isinstance(message["content"], list)
            for part in message["content"]
            if part["type"] == "image_url"
        ]
        assert len(images) == 2  # the page overview, then the form crop
        with Image.open(
            io.BytesIO(base64.b64decode(images[-1].split(",", 1)[1]))
        ) as crop:
            assert crop.size == expected.size
            assert crop.convert("RGB").tobytes() == expected.tobytes()
        answer = {
            "crops": [
                {
                    "region_id": "crop-1",
                    "markdown": "[form options extracted]",
                    "controls": [
                        {
                            "label": "New option",
                            "group_label": "",
                            "state": "selected",
                            "bbox_2d": [100, 200, 500, 400],
                        }
                    ],
                }
            ]
        }
        return (
            200,
            {},
            json.dumps(
                {
                    "model": QWEN_37_FLASH_MODEL,
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": json.dumps(answer)},
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 30,
                        "cost": 0.001,
                    },
                }
            ).encode(),
        )

    image = io.BytesIO()
    source.save(image, format="PNG")
    app = create_app(
        Reader(), crop_refiner=partial(refine_document, transport=transport)
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/process",
            files={"file": ("rotated.png", image.getvalue(), "image/png")},
        )
        assert response.status_code == 200
        page = response.json()["result"]["pages"][0]
        assert page["refinement"]["rotation_degrees"] == angle
        control = next(r for r in page["regions"] if r["text"] == "New option")
        assert list(control["bounding_box"].values()) == expected_box


@pytest.mark.parametrize("failed_page", [None, 2])
def test_parallel_refinement_preserves_requests_outputs_and_failure_order(
    monkeypatch, tmp_path, failed_page
):
    import copy
    import threading

    from ocr_pipeline.contracts import DocumentResult, PageResult
    from ocr_pipeline.openrouter import OpenRouterError
    from ocr_pipeline.rendering import render_evidence

    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture")
    document = DocumentResult("three-pages", {"kind": "image"}, "success")
    paths = []
    for number, color in enumerate(("red", "green", "blue"), 1):
        path = tmp_path / f"page-{number}.png"
        Image.new("RGB", (120, 80), color).save(path)
        paths.append(path)
        regions = [
            TextRegion(
                f"note-{number}",
                "handwriting",
                f"Initial {number}",
                None,
                BoundingBox(0, 0, 120, 80),
                1,
                "fixture",
            )
        ]
        document.pages.append(
            PageResult(
                number,
                120,
                80,
                "fixture",
                "read",
                render_evidence(regions),
                regions,
            )
        )
    requests = []
    barrier = threading.Barrier(3)
    concurrent = False
    seen = []

    def transport(request, timeout):
        if concurrent:
            number = requests.index(request.data) + 1
            seen.append(number)
            barrier.wait(timeout=3)
        else:
            requests.append(request.data)
            number = len(requests)
        if number == failed_page:
            raise OpenRouterError("Unavailable", attempts=1)
        answer = {"crops": [{"region_id": "crop-1", "markdown": "Corrected note"}]}
        return (
            200,
            {},
            json.dumps(
                {
                    "model": QWEN_37_FLASH_MODEL,
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": json.dumps(answer)},
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 10,
                        "cost": 0.001,
                    },
                }
            ).encode(),
        )

    serial = copy.deepcopy(document)
    refine_document(serial, paths, transport=transport, workers=1)
    concurrent = True
    parallel = copy.deepcopy(document)
    refine_document(parallel, paths, transport=transport, workers=3)
    assert sorted(seen) == [1, 2, 3]
    for result in (serial, parallel):
        for page in result.pages:
            page.refinement.pop("elapsed_seconds")
            page.refinement.pop("latency_ms", None)
    assert serial.to_dict() == parallel.to_dict()
    if failed_page:
        assert parallel.failures[0].page_number == failed_page
        assert parallel.pages[failed_page - 1].text.value == f"Initial {failed_page}"


@pytest.mark.parametrize("unknown_charge", [False, True])
def test_parallel_budget_waits_for_reservations_and_retains_unknown_costs(
    monkeypatch, tmp_path, unknown_charge
):
    import sqlite3
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from urllib.request import Request

    from ocr_pipeline import openrouter

    database = tmp_path / "parallel.sqlite"
    entered = threading.Event()
    release = threading.Event()
    waiting = threading.Event()
    original_wait = openrouter._budget_condition.wait

    def observe_wait(*args, **kwargs):
        waiting.set()
        return original_wait(*args, **kwargs)

    monkeypatch.setattr(openrouter._budget_condition, "wait", observe_wait)
    calls = []

    def transport(request, timeout):
        calls.append(request.full_url)
        if len(calls) == 1:
            entered.set()
            assert release.wait(3)
            if unknown_charge:
                raise OSError("connection lost after send")
        return 200, {}, b'{"usage":{"cost":0.001}}'

    call = partial(
        budgeted_transport,
        Request("https://example.invalid"),
        2,
        database=database,
        budget_usd=0.26,
        transport=transport,
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(call)
        assert entered.wait(3)
        second = pool.submit(call)
        try:
            assert waiting.wait(3)
            assert not second.done()
        finally:
            release.set()
        if unknown_charge:
            with pytest.raises(OSError):
                first.result()
            with pytest.raises(openrouter.OpenRouterError, match="budget exhausted"):
                second.result()
        else:
            assert first.result()[0] == second.result()[0] == 200
    with sqlite3.connect(database) as ledger:
        rows = ledger.execute(
            "SELECT amount, actual FROM charges ORDER BY id"
        ).fetchall()
    assert rows == (
        [(0.25, None)] if unknown_charge else [(0.001, 0.001), (0.001, 0.001)]
    )
