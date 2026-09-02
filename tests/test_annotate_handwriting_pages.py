from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from threading import Thread
from typing import Iterator
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from PIL import Image
import pytest

from experiments.annotate_handwriting_pages import AnnotationStore, build_server


def test_http_save_load_and_validation_boundary(tmp_path: Path) -> None:
    queue, output = _queue(tmp_path)

    with _running_server(queue, output) as base_url:
        page = _get_json(f"{base_url}/api/page?index=0")
        assert page["page_id"] == "page-001"
        assert (page["width"], page["height"]) == (40, 30)
        assert page["completed"] == 0

        missing_reviewer = _post(base_url, {"regions": []})
        assert missing_reviewer.code == 400
        assert "reviewer_id is required" in missing_reviewer.read().decode()

        invalid_box = _post(
            base_url,
            {
                "reviewer_id": "reviewer-1",
                "regions": [
                    {
                        "bbox": [5, 5, 41, 20],
                        "text": "outside",
                        "legibility": "legible",
                        "region_type": "field",
                    }
                ],
            },
        )
        assert invalid_box.code == 400
        assert "bbox is outside" in invalid_box.read().decode()

        saved = _post(
            base_url,
            {
                "page_id": "page-001",
                "reviewer_id": " reviewer-1 ",
                "regions": [
                    {
                        "bbox": [2, 3, 21, 18],
                        "text": "  β blocker 10 mg\nnightly ",
                        "legibility": "ambiguous",
                        "region_type": "line",
                    },
                    {
                        "bbox": [22, 4, 35, 16],
                        "text": "",
                        "legibility": "unreadable",
                        "region_type": "signature",
                    },
                ],
            },
        )
        assert saved.code == 200
        assert json.loads(saved.read())["completed"] == 1

        loaded = _get_json(f"{base_url}/api/page?index=0")["annotation"]
        assert loaded["reviewer_id"] == "reviewer-1"
        assert loaded["regions"][0]["bbox"] == [2, 3, 21, 18]
        assert loaded["regions"][0]["text"] == "  β blocker 10 mg\nnightly "
        assert loaded["regions"][0]["region_type"] == "line"

    assert not list(tmp_path.glob(".annotations.jsonl.*.tmp"))
    reopened = AnnotationStore(queue, output).page_payload(0)["annotation"]
    assert reopened == loaded


def test_rejects_non_loopback_bind(tmp_path: Path) -> None:
    queue, output = _queue(tmp_path)

    with pytest.raises(ValueError, match="loopback"):
        build_server(queue, output, "0.0.0.0", 0)


def test_accepts_confident_negatives_and_unreadable_text_regions(
    tmp_path: Path,
) -> None:
    queue, output = _queue(tmp_path)
    store = AnnotationStore(queue, output)
    saved = store.save(
        0,
        {
            "reviewer_id": "reviewer-1",
            "regions": [
                {
                    "bbox": [1, 1, 10, 10],
                    "text": "",
                    "legibility": "legible",
                    "region_type": "blank",
                },
                {
                    "bbox": [11, 1, 20, 10],
                    "text": "",
                    "legibility": "legible",
                    "region_type": "printed_only",
                },
                {
                    "bbox": [21, 1, 30, 10],
                    "text": "",
                    "legibility": "legible",
                    "region_type": "stray_mark",
                },
                {
                    "bbox": [1, 11, 20, 20],
                    "text": "",
                    "legibility": "unreadable",
                    "region_type": "field",
                },
            ],
        },
    )

    assert [row["region_type"] for row in saved["annotation"]["regions"]] == [
        "blank",
        "printed_only",
        "stray_mark",
        "field",
    ]

    with pytest.raises(ValueError, match="negative transcription must be empty"):
        store.save(
            0,
            {
                "reviewer_id": "reviewer-1",
                "regions": [
                    {
                        "bbox": [1, 1, 10, 10],
                        "text": "copied context",
                        "legibility": "legible",
                        "region_type": "blank",
                    }
                ],
            },
        )
    with pytest.raises(ValueError, match="unreadable transcription must be empty"):
        store.save(
            0,
            {
                "reviewer_id": "reviewer-1",
                "regions": [
                    {
                        "bbox": [1, 1, 10, 10],
                        "text": "guess",
                        "legibility": "unreadable",
                        "region_type": "line",
                    }
                ],
            },
        )


def _queue(root: Path) -> tuple[Path, Path]:
    pages = root / "pages"
    pages.mkdir()
    Image.new("RGB", (40, 30), "white").save(pages / "page-001.png")
    queue = root / "queue.jsonl"
    queue.write_text(
        json.dumps({"page_id": "page-001", "image_path": "pages/page-001.png"}) + "\n",
        encoding="utf-8",
    )
    return queue, root / "annotations.jsonl"


@contextmanager
def _running_server(queue: Path, output: Path) -> Iterator[str]:
    server = build_server(queue, output, port=0)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _get_json(url: str) -> dict[str, object]:
    with urlopen(url, timeout=2) as response:
        return json.load(response)


def _post(base_url: str, payload: object):
    request = Request(
        f"{base_url}/api/page?index=0",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        return urlopen(request, timeout=2)
    except HTTPError as error:
        return error
