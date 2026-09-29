"""Offline checks for canonical result and source-image preservation."""

import base64
import io
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from ocr_pipeline.contracts import BoundingBox, TextRegion
from ocr_pipeline.demo import create_app
from ocr_pipeline.falcon_layout import FalconLayoutReader


@pytest.mark.parametrize("state,marker", [("selected", "[x]"), ("unselected", "[ ]")])
def test_manual_checkbox_edits_preserve_state_in_text_markdown_and_exports(
    state, marker
):
    class CheckboxReader(ControlledReader):
        def read(self, image_path, page_number):
            return [
                replace(
                    super().read(image_path, page_number)[0],
                    kind="checkbox",
                    text="Original label",
                    structure={"state": state, "label": "Original label"},
                    text_provenance={"method": "learned_checkbox_detection"},
                )
            ]

    with TestClient(create_app(CheckboxReader())) as client:
        initial = client.post(
            "/api/process",
            files={"file": ("page.png", _page_png(), "image/png")},
        ).json()
        session_url = f"/api/sessions/{initial['session_id']}"
        original = initial["result"]["pages"][0]["regions"][0]
        assert (
            initial["result"]["pages"][0]["text"]["value"] == f"{marker} Original label"
        )
        for revision, label in enumerate(("Corrected label", "Final label"), start=1):
            response = client.post(
                session_url + "/corrections",
                json={
                    "page_number": 1,
                    "region_id": original["id"],
                    "revision": revision,
                    "request_id": f"checkbox-edit-{revision}",
                    "action": "edit",
                    "text": label,
                },
            )
            assert response.status_code == 200
            saved = response.json()
            page = saved["result"]["pages"][0]
            assert page["text"]["value"] == f"{marker} {label}"
            region = page["regions"][0]
            assert region["text"] == region["structure"]["label"] == label
            assert region["structure"]["state"] == state
            assert region["provider"] == "human-review"
            assert region["confidence"] is None
            assert region["text_provenance"]["method"] == "learned_checkbox_detection"
            assert region["text_provenance"]["human_review"]["action"] == "edit"
            assert client.get(session_url + "/result.json").json() == saved["result"]
            assert f"{marker} {label}" in client.get(session_url + "/result.md").text
        assert any(
            alternative["text"] == "Original label"
            and alternative["decision_state"] == "superseded"
            for alternative in region["alternatives"]
        )


def test_manual_edits_refresh_text_markdown_and_exports_at_each_revision():
    with TestClient(create_app(ControlledReader())) as client:
        initial = client.post(
            "/api/process",
            files={"file": ("page.png", _page_png(), "image/png")},
        ).json()
        session_url = f"/api/sessions/{initial['session_id']}"
        original = initial["result"]["pages"][0]["regions"][0]
        assert original["text"] in client.get(session_url + "/result.md").text
        for revision, edited_text in enumerate(
            ("Corrected first line.\nSecond line.", "Final corrected text."), start=1
        ):
            response = client.post(
                session_url + "/corrections",
                json={
                    "page_number": 1,
                    "region_id": original["id"],
                    "revision": revision,
                    "request_id": f"edit-{revision}",
                    "action": "edit",
                    "text": edited_text,
                },
            )
            assert response.status_code == 200
            saved = response.json()
            assert saved["revision"] == revision + 1
            page = saved["result"]["pages"][0]
            assert page["text"]["value"] == edited_text
            assert page["regions"][0]["text"] == edited_text
            assert page["regions"][0]["provider"] == "human-review"
            assert client.get(session_url + "/result.json").json() == saved["result"]
            assert client.get(session_url).json()["result"] == saved["result"]
            markdown = client.get(session_url + f"/result.md?revision={revision + 1}")
            assert markdown.status_code == 200
            assert edited_text in markdown.text
            assert original["text"] not in markdown.text
            assert (
                client.get(session_url + f"/result.md?revision={revision}").status_code
                == 409
            )
        assert any(
            alternative["text"] == original["text"]
            for alternative in page["regions"][0]["alternatives"]
        )


def test_reused_decode_preserves_distinct_source_crops(tmp_path, monkeypatch):
    source = tmp_path / "page.png"
    boxes = [(0, 0, 100, 60), (100, 0, 200, 60)]
    with Image.new("RGB", (200, 60), "white") as image:
        image.paste("navy", (10, 10, 40, 40))
        image.paste("red", (120, 20, 180, 50))
        image.save(source)
    expected = []
    for box in boxes:
        with Image.open(source) as image, image.crop(box) as crop:
            buffer = io.BytesIO()
            crop.save(buffer, format="PNG")
            expected.append(buffer.getvalue())
    assert expected[0] != expected[1]
    reader = FalconLayoutReader("http://127.0.0.1:1")
    monkeypatch.setattr(
        reader,
        "_request",
        lambda *args: {
            "elements": [
                {"category": "picture", "text": "", "bbox": list(box), "score": 0.9}
                for box in boxes
            ]
        },
    )
    opened = Mock(wraps=Image.open)
    monkeypatch.setattr(Image, "open", opened)
    regions = reader.read(source, 1)
    opened.assert_called_once_with(source)
    assert [
        base64.b64decode(region.structure["image"]["data"]) for region in regions
    ] == expected


class ControlledReader:
    name = "controlled-reader"
    version = "test-1"

    def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
        with Image.open(image_path) as image:
            assert image.size == (120, 80)
        return [
            TextRegion(
                id=f"page-{page_number}-region-1",
                kind="paragraph",
                text=f"Controlled page {page_number}",
                confidence=0.91,
                bounding_box=BoundingBox(10, 12, 90, 36),
                reading_order=1,
                provider=self.name,
                text_provenance={"source": "controlled-test"},
            )
        ]


def _page_png() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (120, 80), "white").save(output, format="PNG")
    return output.getvalue()


def test_display_crops_preserve_canonical_exports_and_expire_with_session() -> None:
    import base64

    image = _page_png()

    class FigureReader(ControlledReader):
        def read(self, image_path, page_number):
            return [
                TextRegion(
                    id="figure-1",
                    kind="figure",
                    text="",
                    confidence=None,
                    bounding_box=BoundingBox(0, 0, 120, 80),
                    reading_order=1,
                    provider=self.name,
                    structure={
                        "image": {
                            "media_type": "image/png",
                            "width": 120,
                            "height": 80,
                            "data": base64.b64encode(image).decode(),
                        }
                    },
                )
            ]

    with TestClient(create_app(FigureReader())) as client:
        response = client.post(
            "/api/process?compact=1", files={"file": ("page.png", image, "image/png")}
        )
        assert response.status_code == 200
        display = response.json()
        session_url = f"/api/sessions/{display['session_id']}"
        asset = display["result"]["pages"][0]["regions"][0]["structure"]["image"]
        assert "data" not in asset
        crop = client.get(asset["url"])
        assert crop.content == image
        assert crop.headers["Cache-Control"] == "no-store"
        canonical = client.get(session_url).json()
        original = canonical["result"]["pages"][0]["regions"][0]["structure"]["image"]
        assert base64.b64decode(original["data"]) == image
        assert "url" not in original
        restored = json.loads(json.dumps(display))
        restored_asset = restored["result"]["pages"][0]["regions"][0]["structure"][
            "image"
        ]
        restored_asset["data"] = base64.b64encode(
            client.get(restored_asset.pop("url")).content
        ).decode()
        assert restored == canonical
        exported = client.get(session_url + "/result.json").json()
        assert exported["pages"] == canonical["result"]["pages"]
        assert client.get(session_url + "?compact=1").json() == display
        assert client.get(session_url + "/regions/missing.png").status_code == 404
        client.delete(session_url)
        assert client.get(asset["url"]).status_code == 404


def test_example_thumbnails_leave_original_input_intact() -> None:
    with TestClient(create_app(ControlledReader())) as client:
        original = client.get("/api/examples/handwriting")
        thumbnail = client.get("/api/examples/handwriting?thumb=1")
        assert thumbnail.status_code == 200
        assert len(thumbnail.content) < len(original.content) / 10
        with Image.open(io.BytesIO(thumbnail.content)) as image:
            assert max(image.size) <= 480
        assert client.get("/api/examples/handwriting").content == original.content


def test_next_document_uses_gpu_while_refinement_waits_and_keeps_reader_state() -> None:
    refining = threading.Event()
    second_read = threading.Event()
    finish_second_read = threading.Event()
    release = threading.Event()

    class StatefulReader(ControlledReader):
        calls = 0

        def read(self, image_path, page_number):
            self.calls += 1
            if self.calls == 2:
                second_read.set()
                assert finish_second_read.wait(10)
            return [
                replace(region, text=f"Document {self.calls}")
                for region in super().read(image_path, page_number)
            ]

        def coverage_assessment(self, page_count):
            return {"status": "assessed", "call": self.calls, "pages": []}

        def page_needs_review(self, page_number):
            return self.calls == 1

    def refine(document, paths):
        refining.set()
        assert release.wait(5)
        document.pages[0].refinement = {
            "status": "generated",
            "accepted_crop_ids": ["crop-1"],
        }

    app = create_app(StatefulReader(), crop_refiner=refine)
    with TestClient(app) as client:

        def submit():
            return client.post(
                "/api/process", files={"file": ("page.png", _page_png(), "image/png")}
            )

        with ThreadPoolExecutor(2) as pool:
            first = pool.submit(submit)
            try:
                assert refining.wait(5)
                second = pool.submit(submit)
                assert second_read.wait(5), (
                    "Hosted refinement still blocks the next GPU read"
                )
                release.set()
                first_response = first.result(timeout=5)
            finally:
                release.set()
                finish_second_read.set()
            results = [first_response, second.result()]
        assert all(response.status_code == 200 for response in results)
        assert [
            response.json()["coverage_assessment"]["call"] for response in results
        ] == [1, 2]
        concurrent = [response.json() for response in results]
        assert [
            app.state.sessions.get(response["session_id"])["reader_review_pages"]
            for response in concurrent
        ] == [[1], []]

    # The same deterministic readers must produce identical evidence with either schedule.
    with TestClient(create_app(StatefulReader(), crop_refiner=refine)) as client:
        serial = [submit().json(), submit().json()]
    for overlapped, sequential in zip(concurrent, serial, strict=True):
        for key in ("result", "coverage_assessment", "uncertainty", "presentation"):
            assert overlapped[key] == sequential[key]


def test_stacked_table_token_offsets_follow_trimmed_text() -> None:
    from ocr_pipeline.falcon_layout import _table_parts

    first = "<table><tr><td>A</td></tr></table>"
    second = "<table><tr><td>B</td></tr></table>"
    text = f"  Caption\n{first} \n {second}  "
    split = text.index(second)
    parts = _table_parts(
        {
            "category": "table",
            "text": text,
            "generation": {
                "tokens": [
                    {"start": 2, "end": 9, "score": 0.8},
                    {"start": split - 5, "end": split + 5, "score": 0.6},
                ]
            },
        }
    )
    assert parts[0]["text"] == f"Caption\n{first}"
    assert parts[1]["text"] == second
    assert parts[0]["generation"]["tokens"][0]["start"] == 0
    assert parts[0]["generation"]["tokens"][-1]["end"] == len(parts[0]["text"])
    assert parts[1]["generation"]["tokens"] == [{"start": 0, "end": 5, "score": 0.6}]


def test_feedback_keeps_original_document_and_each_judged_revision(tmp_path):
    """Both votes survive session cleanup with the complete input and edited output."""
    image = Image.new("RGB", (120, 80), "white")
    data = io.BytesIO()
    image.save(data, format="TIFF", save_all=True, append_images=[image])
    source = data.getvalue()
    feedback = tmp_path / "feedback"
    app = create_app(ControlledReader(), feedback_root=feedback)
    with TestClient(app) as client:
        initial = client.post(
            "/api/process", files={"file": ("two-pages.tiff", source, "image/tiff")}
        ).json()
        url = f"/api/sessions/{initial['session_id']}"
        edited = client.post(
            url + "/corrections",
            json={
                "page_number": 1,
                "region_id": initial["result"]["pages"][0]["regions"][0]["id"],
                "revision": 1,
                "request_id": "feedback-edit",
                "action": "edit",
                "text": "Corrected output",
            },
        ).json()
        markdown = client.get(url + "/result.md").text
        records = []
        for verdict in ("good", "problem"):
            response = client.post(
                url + "/feedback",
                json={
                    "verdict": verdict,
                    "reason": "other",
                    "revision": edited["revision"],
                    "page_number": 1,
                    "filename": "untrusted-name.png",
                },
            )
            assert response.status_code == 200
            record = response.json()
            assert record["filename"] == "two-pages.tiff"
            records.append(record)
        assert (
            client.post(
                url + "/feedback",
                json={"verdict": "good", "revision": 1, "page_number": 1},
            ).status_code
            == 409
        )
        session = app.state.sessions.get(initial["session_id"])
        import shutil

        shutil.rmtree(session["directory"])
        for record in records:
            folder = feedback / record["id"]
            assert (folder / "input.tiff").read_bytes() == source
            assert (folder / "result.md").read_text() == markdown
            result = json.loads((folder / "result.json").read_text())["result"]
            assert result["pages"][0]["text"]["value"] == "Corrected output"
            assert len(result["pages"]) == 2
            assert (
                json.loads((folder / "record.json").read_text())["verdict"]
                == record["verdict"]
            )
            assert Image.open(folder / "page.png").size == image.size


def test_parallel_layout_tiles_preserve_scores_boxes_and_order():
    from ocr_pipeline.heron_layout import HeronTableDetector

    barrier = threading.Barrier(4)
    image = Image.new("RGB", (120, 80), "white")

    def detect(crop):
        return [
            {
                "label": "checkbox_selected",
                "score": 0.875,
                "box": (1, 2, 9, 10),
                "query_id": 7,
            }
        ]

    def concurrent(crop):
        barrier.wait(timeout=5)
        return detect(crop)

    baseline = HeronTableDetector(detector=detect).detect_sliced_controls(image, 2)
    actual = HeronTableDetector(
        detector=concurrent, slice_workers=4
    ).detect_sliced_controls(image, 2)
    assert actual == baseline
    assert len(actual) == 4
