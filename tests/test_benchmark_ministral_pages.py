from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
import pytest

from experiments.benchmark_ministral_pages import build_parser, run_benchmark
from ocr_pipeline.contracts import BoundingBox, TextRegion
from ocr_pipeline.providers import ReaderError


class PageReader:
    name = "test-ministral"
    provenance = {"id": "test/ministral", "identity_verified": True}

    def __init__(self, *, fail_on_call: int | None = None) -> None:
        self.calls = 0
        self.fail_on_call = fail_on_call

    def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise ReaderError("test_reader_failed", "expected benchmark failure")
        return [
            TextRegion(
                id="page-text",
                kind="page_text",
                text=f"prediction {self.calls}",
                confidence=None,
                bounding_box=BoundingBox(0, 0, 20, 30),
                reading_order=1,
                provider=self.name,
            )
        ]


def test_benchmark_reuses_reader_and_keeps_failures_in_selected_denominator(
    tmp_path: Path,
) -> None:
    challenge = _challenge(tmp_path)
    output = tmp_path / "run"
    reader = PageReader(fail_on_call=2)

    report = run_benchmark(
        challenge,
        output,
        case_ids=("C14-D001-P001", "C08-D001-P001"),
        max_new_tokens=512,
        reader=reader,
    )

    assert reader.calls == 2
    assert report["cases"] == {
        "selected": ["C14-D001-P001", "C08-D001-P001"],
        "attempted": 2,
        "completed": 1,
        "failed": 1,
        "failures_remain_in_denominator": True,
    }
    assert report["model"] == PageReader.provenance
    assert report["generation"] == {"max_new_tokens": 512, "do_sample": False}
    assert report["evaluation"]["cases"] == {
        "total": 2,
        "annotated": 2,
        "model_outputs": 2,
        "paired": 2,
        "missing_model_output": 0,
        "unannotated_model_output": 0,
    }
    assert report["evaluation"]["coverage"]["failed"] == 1
    outputs = sorted((output / "model-output").rglob("*.json"))
    assert [
        path.relative_to(output / "model-output").as_posix() for path in outputs
    ] == [
        "C08/C08-D001-P001.json",
        "C14/C14-D001-P001.json",
    ]
    failure = json.loads(outputs[0].read_text(encoding="utf-8"))
    assert failure["result"]["failures"][0]["code"] == "test_reader_failed"
    assert json.loads((output / "report.json").read_text(encoding="utf-8")) == report


def test_benchmark_rejects_unknown_duplicate_cases_and_existing_output(
    tmp_path: Path,
) -> None:
    challenge = _challenge(tmp_path)

    with pytest.raises(ValueError, match="Unknown frozen case ID"):
        run_benchmark(
            challenge,
            tmp_path / "unknown",
            case_ids=("C08-D999-P001",),
            reader=PageReader(),
        )
    with pytest.raises(ValueError, match="must be unique"):
        run_benchmark(
            challenge,
            tmp_path / "duplicate",
            case_ids=("C08-D001-P001", "C08-D001-P001"),
            reader=PageReader(),
        )

    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        run_benchmark(challenge, existing, reader=PageReader())


def test_parser_accepts_repeated_cases_and_positive_token_limit() -> None:
    args = build_parser().parse_args(
        [
            "run",
            "--case",
            "C14-D001-P001",
            "--case",
            "C08-D001-P001",
            "--max-new-tokens",
            "256",
        ]
    )

    assert args.case == ["C14-D001-P001", "C08-D001-P001"]
    assert args.max_new_tokens == 256


def _challenge(tmp_path: Path) -> Path:
    challenge = tmp_path / "challenge"
    for category, count, source_dir in (
        ("C08", 20, "C08-handwritten"),
        ("C14", 2, "C14-user-reported"),
    ):
        for index in range(1, count + 1):
            case_id = f"{category}-D{index:03d}-P001"
            source = challenge / "sources" / source_dir / f"{case_id}.png"
            source.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (20, 30), "white").save(source)
            annotation = {
                "case_id": case_id,
                "category_id": category,
                "source_only": True,
                "page_legibility": "complete",
                "challenges": [],
                "transcription": {
                    "reading_order_text": f"reference {case_id}",
                    "unresolved_spans": [],
                },
                "tables": [],
                "controls": [],
                "handwriting": [],
            }
            annotation_path = (
                challenge / "annotations" / "primary" / category / f"{case_id}.json"
            )
            annotation_path.parent.mkdir(parents=True, exist_ok=True)
            annotation_path.write_text(json.dumps(annotation), encoding="utf-8")
    return challenge
