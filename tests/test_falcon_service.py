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

from experiments import serve_falcon_layout
from ocr_pipeline.falcon import (
    FALCON_MODEL_ID,
    FALCON_MODEL_LICENSE,
    FALCON_MODEL_ORIGIN,
    FALCON_MODEL_REVISION,
)
from ocr_pipeline.contracts import BoundingBox, TextRegion


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


# --- Falcon-Perception layout service: repetition-loop retry ladder ---


def test_reserved_vocabulary_slots_are_not_text() -> None:
    ids, scores = serve_falcon_layout._without_reserved(
        [273, 11, 273, 12], [0.9, 0.8, 0.7, 0.6], frozenset({273})
    )
    assert (ids, scores) == ([11, 12], [0.8, 0.6])


def test_decoder_cycle_requires_a_long_trailing_repeat_of_any_short_period() -> None:
    detect = serve_falcon_layout._decoder_cycle
    assert detect([2006, 638] * 125 + [2006] * 6)
    assert not detect([2006, 638] * 10)
    assert not detect(list(range(300)))
    assert not detect([2006, 638] * 100 + list(range(100)))
    assert detect(list(range(100)) + [2006, 638] * 200)
    row = [11, 12, 13, 14, 15, 16, 17]
    assert detect(list(range(40)) + row * 40)
    assert not detect(row * 30)
    assert detect(list(range(500)) + list(range(1000, 1060)) * 6)


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
        if "temperature" not in options:
            return [self.elements]
        self.retry_calls.append(options)
        text = self.plain_by_temperature[round(float(options["temperature"]), 2)]
        return [[{**item, "text": text} for item in options["detections"][0]]]

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


def test_route_detection_excludes_key_value_region_when_form_covers_it() -> None:
    proposals = [
        {"label": "key_value_region", "box": (0, 0, 10, 10), "score": 0.8},
        {"label": "form", "box": (0, 0, 20, 20), "score": 0.9},
    ]

    detection = serve_falcon_layout._route_detection(proposals)

    assert [item["category"] for item in detection] == ["form"]


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
    # The 0.5 reading is kept only once the 0.8 reading agrees with it.
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
    assert element["retries"] == 3
    assert len(element["read_attempts"]) == 4
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
            assert region["text_provenance"]["read_retries"] == 3
            assert len(region["text_provenance"]["read_attempts"]) == 4
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
        detect_layout=lambda images: [
            [{"category": "form", "bbox": [2, 2, 38, 38], "score": 0.8}]
        ],
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
        # A form holding no box or ballot mark is not read for controls.
        assert _crop_groups([asdict(region)], 40, 40) == []
    finally:
        server.shutdown()
        server.server_close()


def test_checkbox_region_reads_via_detected_layout(tmp_path: Path) -> None:
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
        detect_layout=lambda images: [
            [{"category": "checkbox-selected", "bbox": [2, 2, 20, 20], "score": 0.9}]
        ],
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


def test_form_only_upload_preserves_crop_and_reports_unresolved_page() -> None:
    from fastapi.testclient import TestClient

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
        {0.0: original, 0.2: recovered, 0.5: recovered},
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
        recovered,
        recovered,
    ]


def test_disagreeing_sampled_reads_are_evidence_not_the_reading() -> None:
    # A faint line read as a loop, then three different sampled inventions.
    original = serve_falcon_layout.DecodedText(
        "<th></th>" * 300, {"stop_token_seen": True, "decoder_cycle": True}
    )
    inventions = {
        temperature: serve_falcon_layout.DecodedText(
            text, {"stop_token_seen": True, "temperature": temperature}
        )
        for temperature, text in (
            (0.2, " ".join(str(n) for n in range(1, 400))),
            (0.5, "SOLD\tSOLD\tSOLD"),
            (0.8, "South Windsor 2701 Tamarack Ave"),
        )
    }
    engine = FakeLayoutEngine(
        [{"category": "text", "bbox": [0, 0, 40, 40], "score": 0.9, "text": original}],
        {0.0: original, **inventions},
    )
    server = _start_layout_server(engine, category_by_layout={"text": "text"})
    try:
        element = _post_layout_read(server, Image.new("RGB", (40, 40)))["elements"][0]
    finally:
        server.shutdown()
        server.server_close()

    assert element["text"] == original
    assert element["looped"] is True
    assert element["retries"] == 3
    assert [item["text"] for item in element["read_attempts"]] == [
        original,
        *inventions.values(),
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
def test_upload_shows_completed_retries_and_sends_them_to_review(category):
    from fastapi.testclient import TestClient
    from ocr_pipeline.demo import create_app
    from ocr_pipeline.falcon_layout import FalconLayoutReader
    from ocr_pipeline.risk import EvidenceRiskStage

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
        {0.2: retry, 0.5: retry},
    )
    server = _start_layout_server(engine, category_by_layout={category: "text"})
    buffer = io.BytesIO()
    Image.new("RGB", (40, 40), "white").save(buffer, format="PNG")
    try:
        reader = FalconLayoutReader(f"http://127.0.0.1:{server.server_port}")
        with TestClient(create_app(reader, stages=(EvidenceRiskStage(),))) as client:
            response = client.post(
                "/api/process",
                files={"file": ("page.png", buffer.getvalue(), "image/png")},
            )
            assert response.status_code == 200
            payload = response.json()
            page = payload["result"]["pages"][0]
            region = page["regions"][0]
            assert region["text"] == retry
            assert region["resolution"] == "resolved"
            assert page["route"] == "review"
            attempts = region["text_provenance"]["read_attempts"]
            assert [item["text"] for item in attempts] == [original, retry, retry]
            exported = client.get(
                f"/api/sessions/{payload['session_id']}/result.json"
            ).json()
            assert exported["pages"][0]["regions"][0] == region
            markdown = client.get(
                f"/api/sessions/{payload['session_id']}/result.md"
            ).text
            assert retry in markdown and retry in page["text"]["value"]
            assert "[unresolved text]" not in markdown
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
        reader = FalconLayoutReader(f"http://127.0.0.1:{server.server_port}")
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
            assert checkbox["text_provenance"]["raw_text"] == readings[0]
            assert checkbox["text_provenance"]["generation"] == generation
            assert checkbox["confidence"] == 0.9
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
        ("O Yes", "Yes"),  # a radio circle read as a letter
        ("O2 Precautions", "O2 Precautions"),
        ("0 - Able to independently feed self", "0 - Able to independently feed self"),
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
            get_added_tokens_decoder=dict,
            encode=lambda text, **kwargs: SimpleNamespace(ids=[1], offsets=[(0, 2)]),
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
    # Host timings differ per call; everything else must be identical.
    assert all(output.pop("timings")["crops"] == 2 for output in outputs)
    assert outputs[0] == outputs[1] == outputs[2]
    elements = outputs[0]["elements"]
    # The page-sized picture is dropped and the nested text is read by its container,
    # while the checkbox inside the text keeps its own control reading.
    assert [item["category"] for item in elements] == ["text", "checkbox-selected"]
    assert [item["text"] for item in elements] == ["I:", "I:"]
    assert elements[0]["bbox"] == [0, 0, 80, 40]
    assert elements[1]["bbox"] == [70, 1, 79, 10]
    assert elements[0]["generation"]["token_score"] == pytest.approx(0.75)
    assert elements[0]["generation"]["tokens"] == [
        {"start": 0, "end": 2, "score": 0.75}
    ]
    assert [element["query_id"] for element in elements] == [1, 3]
    assert elements[0]["class_scores"]["picture"] == pytest.approx(0.85)
    assert elements[0]["class_scores"]["text"] == pytest.approx(0.9)
    assert all(min(size) >= 16 for size in crop_sizes)

    engines[0].layout_detector._detector = lambda image: [
        {"label": "picture", "box": [0, 0, 80, 40], "score": 0.8}
    ]
    with Image.new("RGB", (80, 40)) as image:
        result = engines[0].generate_with_layout([image])
    # Docling drops a picture covering the whole page.
    assert result == [[]]
    assert len(crop_sizes) == 6


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
                if route == "retry" and "temperature" not in options
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
            # The retry route reads twice more: a sampled read is kept once another agrees.
            assert engine.erased == [1, 2]
            assert engine.calls == (3 if route == "retry" else 2)
            assert engine.paged_kv_cache.capacity == [0, 0, 0]
            assert not engine._input_pos.any() and not engine._rope_pos_t.any()
            assert engine._token_ids_ready is None and engine._token_ids_count == 0
    finally:
        allow_tokenization.set()
        pool.shutdown(wait=True, cancel_futures=True)
        server.shutdown()
        server.server_close()


def test_born_digital_cells_reach_layout_detection_and_bad_cells_are_rejected() -> None:
    seen = []

    def detect_layout(images, cells=None):
        seen.append(cells)
        return [[]]

    server = _start_layout_server(FakeLayoutEngine([]), detect_layout=detect_layout)
    buffer = io.BytesIO()
    Image.new("RGB", (40, 30), "white").save(buffer, format="PNG")
    image = base64.b64encode(buffer.getvalue()).decode()

    def post(cells):
        request = urllib.request.Request(
            f"http://127.0.0.1:{server.server_address[1]}/read",
            data=json.dumps({"image": image, "cells": cells}).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=5) as reply:
                return reply.status
        except urllib.error.HTTPError as error:
            return error.code

    try:
        assert post([{"text": "Yes", "bbox": [0, 0, 10, 10]}]) == 200
        assert post([{"text": "Yes", "bbox": [0, 0, 10]}]) == 400
    finally:
        server.shutdown()
        server.server_close()

    assert seen == [[[{"text": "Yes", "bbox": [0.0, 0.0, 10.0, 10.0]}]]]


@pytest.mark.parametrize("on_engine", [False, True])
def test_a_gpu_layout_detector_runs_inside_the_engine_context(on_engine) -> None:
    import torch

    seen = []

    def detect_layout(images, cells=None):
        seen.append(torch.is_inference_mode_enabled())
        return [[]]

    server = _start_layout_server(
        FakeLayoutEngine([]), detect_layout=detect_layout, detect_on_engine=on_engine
    )
    try:
        _post_layout_read(server, Image.new("RGB", (40, 30), "white"))
    finally:
        server.shutdown()
        server.server_close()

    assert seen == [on_engine]


def test_a_born_digital_table_takes_tableformer_structure_not_a_pixel_read() -> None:
    calls = []

    def table_reader(image, box, cells):
        calls.append((box, cells))
        return "<table><tr><td>Difficulty with ADLs</td><td>☑</td></tr></table>"

    engine = FakeLayoutEngine([])
    detections = [
        {"category": "table", "bbox": [0, 0, 30, 20], "score": 0.9, "cell_text": "x"}
    ]
    engine.generate_with_layout = lambda images, **options: [
        [
            {**item, "text": item.get("cell_text", "")}
            for item in options["detections"][0]
        ]
    ]
    server = _start_layout_server(
        engine,
        detect_layout=lambda images, cells=None: [detections],
        table_reader=table_reader,
    )
    buffer = io.BytesIO()
    Image.new("RGB", (40, 30), "white").save(buffer, format="PNG")
    cells = [{"text": "☑", "bbox": [20, 5, 25, 10]}]
    request = urllib.request.Request(
        f"http://127.0.0.1:{server.server_address[1]}/read",
        data=json.dumps(
            {"image": base64.b64encode(buffer.getvalue()).decode(), "cells": cells}
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as reply:
            element = json.loads(reply.read())["elements"][0]
    finally:
        server.shutdown()
        server.server_close()

    assert calls == [([0, 0, 30, 20], [{"text": "☑", "bbox": [20.0, 5.0, 25.0, 10.0]}])]
    assert element["text"].startswith("<table>")
    assert element["text_source"] == "pdf_text_layer"
    assert element["table_source"] == "tableformer"


def test_a_crop_floor_above_the_crop_ceiling_is_rejected() -> None:
    server = _start_layout_server(FakeLayoutEngine([]))
    buffer = io.BytesIO()
    Image.new("RGB", (20, 20), "white").save(buffer, format="PNG")
    image = base64.b64encode(buffer.getvalue()).decode()

    def post(options):
        request = urllib.request.Request(
            f"http://127.0.0.1:{server.server_address[1]}/read",
            data=json.dumps({"image": image, **options}).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=5) as reply:
                return reply.status
        except urllib.error.HTTPError as error:
            return error.code

    try:
        assert post({"min_image_size": 128}) == 200
        assert post({"min_image_size": 2048}) == 400
        assert post({"min_image_size": 0}) == 400
    finally:
        server.shutdown()
        server.server_close()


def test_retry_crop_floor_respects_request_ceiling() -> None:
    engine = FakeLayoutEngine(
        [{'category': 'text', 'bbox': [0, 0, 40, 40], 'score': .9, 'text': ''}],
        {0.2: 'recovered', 0.5: 'recovered', 0.8: 'recovered'},
    )
    server = _start_layout_server(engine, layout={'max_image_size': 128})
    try:
        body = _post_layout_read(server, Image.new('RGB', (40, 40), 'white'))
    finally:
        server.shutdown()
        server.server_close()
    assert body['elements'][0]['text'] == 'recovered'
    assert engine.retry_calls
    assert all(call['min_image_size'] == 128 for call in engine.retry_calls)
