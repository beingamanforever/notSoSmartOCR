from __future__ import annotations

from pathlib import Path

from PIL import Image

from ocr_pipeline.comprehension import FieldAnswer, TiltFieldStage
from ocr_pipeline.contracts import BoundingBox, TextRegion


class ControlledTiltEngine:
    """Stands in for Arctic-TILT so the stage is testable without CUDA."""

    def __init__(self, replies: dict[str, tuple[str, ...]] | None = None) -> None:
        self.replies = replies or {"Physician Progress Note": ("History and Physical",)}
        self.questions: list[str] = []
        self.words: list[str] = []

    def answer(self, image_path, words, boxes, size, questions):
        self.questions = list(questions)
        self.words = list(words)
        self.size = size
        return [
            FieldAnswer(
                field_id=question,
                label=question,
                values=self.replies[question],
                score=0.82,
            )
            for question in questions
            if question in self.replies
        ]


def word(identifier: str, text: str, box: tuple[int, int, int, int], order: int):
    return TextRegion(
        id=identifier,
        kind="word",
        text=text,
        confidence=0.9,
        bounding_box=BoundingBox(*box),
        reading_order=order,
        provider="controlled",
    )


def form_page(state: str = "not_located") -> list[TextRegion]:
    block = TextRegion(
        id="p1-block-1",
        kind="layout_block",
        text="",
        confidence=None,
        bounding_box=BoundingBox(0, 0, 600, 400),
        reading_order=99,
        provider="layout",
        structure={
            "role": "layout_block",
            "fields": [
                {
                    "id": "p1-field-1",
                    "label": "Physician Progress Note",
                    "state": state,
                },
                {"id": "p1-field-2", "label": "Patient Name", "state": "present"},
            ],
        },
    )
    return [
        word("p1-word-1", "Physician Progress Note:", (10, 10, 200, 30), 1),
        word("p1-word-2", "History and Physical", (220, 10, 400, 30), 2),
        block,
    ]


def page_image(tmp_path: Path) -> Path:
    path = tmp_path / "page.png"
    Image.new("L", (600, 400), 255).save(path)
    return path


def test_unsettled_fields_get_a_proposal_that_never_becomes_canonical(tmp_path) -> None:
    regions = form_page()
    engine = ControlledTiltEngine()

    TiltFieldStage(engine).apply(page_image(tmp_path), 1, regions)

    field = regions[-1].structure["fields"][0]
    assert field["comprehension"]["values"] == ["History and Physical"]
    assert field["comprehension"]["score"] == 0.82
    assert field["comprehension"]["decision_state"] == "pending"
    assert field["state"] == "not_located", "a proposal must not settle the field state"
    assert "comprehension" not in regions[-1].structure["fields"][1], (
        "a field geometry already settled must not be re-asked"
    )


def test_only_unsettled_labels_are_asked(tmp_path) -> None:
    engine = ControlledTiltEngine()

    TiltFieldStage(engine).apply(page_image(tmp_path), 1, form_page())

    assert engine.questions == ["Physician Progress Note"]
    assert engine.words == ["Physician Progress Note:", "History and Physical"], (
        "TILT reads positioned words in reading order, layout blocks excluded"
    )
    assert engine.size == (600, 400)


def test_settled_pages_never_reach_the_model(tmp_path) -> None:
    engine = ControlledTiltEngine()

    TiltFieldStage(engine).apply(page_image(tmp_path), 1, form_page(state="present"))

    assert engine.questions == []


def test_the_question_budget_is_reported_rather_than_silently_truncating(
    tmp_path,
) -> None:
    regions = form_page()
    regions[-1].structure["fields"].append(
        {"id": "p1-field-3", "label": "Allergies", "state": "illegible"}
    )

    TiltFieldStage(ControlledTiltEngine(), max_questions=1).apply(
        page_image(tmp_path), 1, regions
    )

    assert regions[-1].structure["comprehension_budget"] == {"asked": 1, "not_asked": 1}


def test_the_http_engine_round_trips_answers_through_the_service(tmp_path) -> None:
    """The model runs in its own process, so the stage must survive the wire format."""
    import json
    import sys
    from http.server import ThreadingHTTPServer
    from threading import Thread

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    from serve_tilt import create_server

    from ocr_pipeline.comprehension import HttpTiltEngine

    class Recorder:
        def __init__(self) -> None:
            self.words: list[str] = []
            self.questions: list[str] = []

        def answer_pages(self, pages, questions):
            self.words = list(pages[0].words)
            self.questions = list(questions)
            self.size = pages[0].size
            self.page_count = len(pages)
            assert all(page.image_path.is_file() for page in pages)
            return [
                FieldAnswer(
                    field_id=question, label=question, values=("F",), score=0.91
                )
                for question in questions
            ]

    recorder = Recorder()
    server: ThreadingHTTPServer = create_server(recorder, "127.0.0.1", 0)
    Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}"
        engine = HttpTiltEngine(url)
        engine.check_health()
        answers = engine.answer(
            page_image(tmp_path),
            ["Gender:"],
            [(507, 123, 557, 133)],
            (823, 593),
            ["Gender:"],
        )
    finally:
        server.shutdown()
        server.server_close()

    assert recorder.words == ["Gender:"]
    assert recorder.questions == ["Gender:"]
    assert recorder.size == (823, 593)
    assert recorder.page_count == 1
    assert [(answer.label, answer.values, answer.score) for answer in answers] == [
        ("Gender:", ("F",), 0.91)
    ]
    assert json.dumps([answer.values for answer in answers])


def test_the_http_engine_refuses_a_non_loopback_service() -> None:
    """Page pixels leave the process, so the service must stay on the host."""
    import pytest

    from ocr_pipeline.comprehension import HttpTiltEngine

    with pytest.raises(ValueError):
        HttpTiltEngine("http://10.20.5.179:8086")
