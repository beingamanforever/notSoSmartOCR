from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from ocr_pipeline.layout import HeronLayoutDetector, LayoutDetection, _normalize_label

from experiments.heron_layout_export import export_predictions, main


class FakeDetector:
    name = "fake-heron"
    model_name = "model"
    model_revision = "revision"
    device = "cpu"
    threshold = 0.6

    def detect(self, image_path: Path) -> list[LayoutDetection]:
        if image_path.stem == "failed":
            raise RuntimeError("detector unavailable")
        if image_path.stem == "empty":
            return []
        return [
            LayoutDetection("section_header", 0.9, (1.0, 2.0, 30.0, 12.0), self.name),
            LayoutDetection("text", 0.8, (1.0, 15.0, 45.0, 30.0), self.name),
            LayoutDetection(
                "checkbox_selected", 0.7, (2.0, 32.0, 8.0, 38.0), self.name
            ),
        ]


def test_export_is_failure_inclusive_and_uses_only_page_paths(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    records = []
    for name in ("page.png", "empty.png", "failed.png"):
        Image.new("RGB", (60, 40), "white").save(image_root / name)
        records.append(
            {
                "page_info": {"image_path": name},
                "layout_dets": [{"category_type": "must_not_control_prediction"}],
            }
        )
    annotations = tmp_path / "annotations.json"
    annotations.write_text(json.dumps(records), encoding="utf-8")

    output = tmp_path / "run"
    report = export_predictions(
        annotations,
        image_root,
        output,
        FakeDetector(),
        dataset_revision="official-revision",
    )

    predictions = json.loads((output / "predictions.json").read_text())
    assert predictions["categories"]["0"] == "title"
    assert predictions["categories"]["1"] == "plain text"
    assert predictions["results"] == [
        {
            "image_name": "page",
            "bbox": [1.0, 2.0, 30.0, 12.0],
            "category_id": 0,
            "score": 0.9,
        },
        {
            "image_name": "page",
            "bbox": [1.0, 15.0, 45.0, 30.0],
            "category_id": 1,
            "score": 0.8,
        },
    ]
    assert report["case_ids"] == ["page.png", "empty.png", "failed.png"]
    assert report["attempted"] == 3
    assert report["covered"] == 1
    assert report["abstained"] == 1
    assert report["failed"] == 1
    assert report["unsupported_detections"] == {"checkbox_selected": 1}
    assert report["selection"] == {"base_per_language": None, "limit": None}
    assert report["pages"][2]["failure"]["code"] == "layout_failed"
    assert set(report["latency_ms"]) == {"p50", "p95"}


def test_export_refuses_existing_output_and_duplicate_stems(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    annotations = tmp_path / "annotations.json"
    annotations.write_text(
        json.dumps([{"page_info": {"image_path": "page.png"}}]),
        encoding="utf-8",
    )
    with pytest.raises(FileExistsError):
        export_predictions(
            annotations,
            tmp_path,
            output,
            FakeDetector(),
            dataset_revision="revision",
        )

    records = [
        {"page_info": {"image_path": "a/page.png"}},
        {"page_info": {"image_path": "b/page.jpg"}},
    ]
    annotations.write_text(json.dumps(records), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate prediction names"):
        export_predictions(
            annotations,
            tmp_path,
            tmp_path / "new-run",
            FakeDetector(),
            dataset_revision="revision",
        )


def test_heron_configuration_and_label_normalization() -> None:
    detector = HeronLayoutDetector(
        model_revision="exact",
        device="cpu",
        threshold=0.6,
    )
    assert detector.model_revision == "exact"
    assert detector.device == "cpu"
    assert _normalize_label("Checkbox-Selected") == "checkbox_selected"
    assert _normalize_label(" Key-Value Region ") == "key_value_region"

    with pytest.raises(ValueError, match="revision"):
        HeronLayoutDetector(model_revision="")
    with pytest.raises(ValueError, match="threshold"):
        HeronLayoutDetector(model_revision="exact", threshold=1.1)


def test_cli_writes_predictions_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    Image.new("RGB", (60, 40), "white").save(image_root / "page.png")
    annotations = tmp_path / "annotations.json"
    annotations.write_text(
        json.dumps([{"page_info": {"image_path": "page.png"}}]),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "experiments.heron_layout_export.HeronLayoutDetector",
        lambda **options: FakeDetector(),
    )

    output = tmp_path / "run"
    assert (
        main(
            [
                str(annotations),
                str(image_root),
                str(output),
                "--dataset-revision",
                "official-revision",
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    assert (output / "predictions.json").is_file()
    assert json.loads((output / "run_report.json").read_text())["attempted"] == 1
