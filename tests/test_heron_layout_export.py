from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from PIL import Image

from ocr_pipeline.layout import (
    HeronLayoutDetector,
    LayoutDetection,
    NemotronPageElementsV3Detector,
    _normalize_label,
)

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


class FakeNemotronModel:
    thresholds_per_class = {"Table": 0.1, "Header-Footer": 0.1}
    labels = ["Table", "Header-Footer"]

    def __init__(self) -> None:
        self.devices: list[str] = []
        self.eval_calls = 0
        self.images: list[np.ndarray] = []
        self.image_shapes: list[tuple[int, ...]] = []

    def to(self, device: str) -> FakeNemotronModel:
        self.devices.append(device)
        return self

    def eval(self) -> None:
        self.eval_calls += 1

    def preprocess(self, image: np.ndarray) -> str:
        self.images.append(image)
        return "prepared"

    def __call__(self, inputs: str, image_shape: tuple[int, ...]) -> list[str]:
        assert inputs == "prepared"
        self.image_shapes.append(image_shape)
        return ["raw-predictions"]


class FakeNemotronDetector:
    name = "nemotron-page-elements-v3"
    model_name = "nvidia/nemotron-page-elements-v3"
    model_revision = None
    model_revision_enforced = False
    device = "cpu"
    threshold = {"header_footer": 0.1, "chart": 0.1, "infographic": 0.1}

    def detect(self, image_path: Path) -> list[LayoutDetection]:
        assert image_path.name == "page.png"
        return [
            LayoutDetection("header_footer", 0.9, (1.0, 2.0, 30.0, 12.0), self.name),
            LayoutDetection("chart", 0.8, (1.0, 15.0, 45.0, 30.0), self.name),
            LayoutDetection("infographic", 0.7, (2.0, 32.0, 8.0, 38.0), self.name),
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
    assert detector.model_revision_enforced is True
    assert detector.device == "cpu"
    assert _normalize_label("Checkbox-Selected") == "checkbox_selected"
    assert _normalize_label(" Key-Value Region ") == "key_value_region"

    with pytest.raises(ValueError, match="revision"):
        HeronLayoutDetector(model_revision="")
    with pytest.raises(ValueError, match="threshold"):
        HeronLayoutDetector(model_revision="exact", threshold=1.1)


def test_nemotron_page_elements_detector_converts_normalized_boxes(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (200, 100), "white").save(image_path)
    model = FakeNemotronModel()

    def postprocess(
        predictions: str,
        thresholds: dict[str, float],
        labels: list[str],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert predictions == "raw-predictions"
        assert thresholds is model.thresholds_per_class
        assert labels is model.labels
        return (
            np.array([[0.1, 0.2, 0.6, 0.8], [0.0, 0.0, 1.0, 1.0]]),
            np.array([0, 1]),
            np.array([0.9, 0.7]),
        )

    detector = NemotronPageElementsV3Detector(
        device="cpu",
        model=model,
        postprocessor=postprocess,
    )

    assert detector.detect(image_path) == [
        LayoutDetection("table", 0.9, (20.0, 20.0, 120.0, 80.0), detector.name),
        LayoutDetection(
            "header_footer",
            0.7,
            (0.0, 0.0, 200.0, 100.0),
            detector.name,
        ),
    ]
    assert detector.model_name == "nvidia/nemotron-page-elements-v3"
    assert detector.model_revision is None
    assert detector.model_revision_enforced is False
    assert detector.device == "cpu"
    assert detector.threshold == model.thresholds_per_class
    assert model.images[0].shape == (100, 200, 3)
    assert model.images[0].dtype == np.uint8
    assert model.image_shapes == [(100, 200, 3)]


def test_nemotron_page_elements_detector_rejects_invalid_results(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (20, 10), "white").save(image_path)

    detector = NemotronPageElementsV3Detector(
        model=FakeNemotronModel(),
        postprocessor=lambda *_: ([[0.0, 0.0, 1.1, 1.0]], [0], [0.9]),
    )

    with pytest.raises(ValueError, match="out-of-bounds bounding box"):
        detector.detect(image_path)


def test_nemotron_page_elements_detector_lazy_loads_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (20, 10), "white").save(image_path)
    model = FakeNemotronModel()
    model_names = []
    model_module = ModuleType("nemotron_page_elements_v3.model")
    utils_module = ModuleType("nemotron_page_elements_v3.utils")
    setattr(
        model_module,
        "define_model",
        lambda model_name: model_names.append(model_name) or model,
    )
    setattr(
        utils_module,
        "postprocess_preds_page_element",
        lambda *_: ([[0.0, 0.0, 1.0, 1.0]], [0], [0.9]),
    )
    monkeypatch.setitem(
        sys.modules,
        "nemotron_page_elements_v3",
        ModuleType("nemotron_page_elements_v3"),
    )
    monkeypatch.setitem(sys.modules, "nemotron_page_elements_v3.model", model_module)
    monkeypatch.setitem(sys.modules, "nemotron_page_elements_v3.utils", utils_module)

    detector = NemotronPageElementsV3Detector(
        device="cpu",
    )
    assert model_names == []

    detector.detect(image_path)
    detector.detect(image_path)

    assert model_names == ["page_element_v3"]
    assert model.devices == ["cpu"]
    assert model.eval_calls == 1


def test_nemotron_page_elements_detector_rejects_unenforced_revision() -> None:
    with pytest.raises(ValueError, match="cannot enforce model revisions"):
        NemotronPageElementsV3Detector(model_revision="exact")

    with pytest.raises(ValueError, match="supports only its official model"):
        NemotronPageElementsV3Detector(model_name="other/model")


def test_export_maps_nemotron_labels_and_reports_unpinned_revision(
    tmp_path: Path,
) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    Image.new("RGB", (60, 40), "white").save(image_root / "page.png")
    annotations = tmp_path / "annotations.json"
    annotations.write_text(
        json.dumps([{"page_info": {"image_path": "page.png"}}]),
        encoding="utf-8",
    )

    report = export_predictions(
        annotations,
        image_root,
        tmp_path / "run",
        FakeNemotronDetector(),
        dataset_revision="official-revision",
    )
    results = json.loads(
        (tmp_path / "run" / "predictions.json").read_text(encoding="utf-8")
    )["results"]

    assert [result["category_id"] for result in results] == [2, 3, 3]
    assert report["model_revision"] is None
    assert report["model_revision_enforced"] is False
    assert report["mapping"]["header_footer"] == "abandon"
    assert report["mapping"]["chart"] == "figure"
    assert report["mapping"]["infographic"] == "figure"


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
