from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from experiments import doctr_orientation_benchmark
from experiments.public_benchmark import BenchmarkCase


class FakePredictor:
    def __call__(self, images: list[np.ndarray]) -> list[list[int | float]]:
        rotations = [0, -90, 90, 180][: len(images)]
        return [
            list(range(len(images))),
            rotations,
            [0.9] * len(images),
        ]


def test_prediction_records_map_negative_rotation(tmp_path: Path) -> None:
    cases = [
        BenchmarkCase(
            id=f"rotated/case-{index}",
            cluster_id=str(index),
            subset="rotated",
            image_path=tmp_path / f"case-{index}.png",
            reference="",
        )
        for index in range(2)
    ]
    records = doctr_orientation_benchmark._prediction_records(
        tmp_path,
        cases,
        [(np.zeros((2, 2, 3)), 1.0), (np.zeros((2, 2, 3)), 2.0)],
        [[0, 1], [0, -90], [0.8, 0.9]],
        0,
        4.0,
    )

    assert [record["lossless_rotation"] for record in records] == [0, 270]
    assert records[1]["allocated_inference_latency_ms"] == 2.0


def test_run_benchmark_uses_only_requested_split(
    tmp_path: Path,
    monkeypatch,
) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (8, 8), "white").save(image_path)
    cases = [
        BenchmarkCase(
            id="rotated/page",
            cluster_id="1",
            subset="rotated",
            image_path=image_path,
            reference="",
        ),
        BenchmarkCase(
            id="normal/page",
            cluster_id="1",
            subset="normal",
            image_path=image_path,
            reference="",
        ),
    ]
    observed = {}

    def fake_discover(dataset, root, *, clinocr_role):
        observed.update(dataset=dataset, root=root, role=clinocr_role)
        return cases

    monkeypatch.setattr(
        doctr_orientation_benchmark,
        "discover_cases",
        fake_discover,
    )
    monkeypatch.setattr(
        doctr_orientation_benchmark,
        "_load_predictor",
        lambda arch, batch_size, device: (FakePredictor(), object(), "1.0.1"),
    )
    monkeypatch.setattr(doctr_orientation_benchmark, "_synchronize", lambda *_: None)

    payload = doctr_orientation_benchmark.run_benchmark(
        tmp_path,
        clinocr_role="exemplar",
        subset="rotated",
        batch_size=4,
        device="cpu",
        rectify=False,
    )

    assert observed == {"dataset": "clinocr", "root": tmp_path, "role": "exemplar"}
    assert payload["clinocr_role"] == "exemplar"
    assert payload["summary"]["pages"] == 1
    assert payload["cases"][0]["id"] == "rotated/page"


def test_prediction_records_reject_invalid_confidence(tmp_path: Path) -> None:
    case = BenchmarkCase(
        id="rotated/page",
        cluster_id="1",
        subset="rotated",
        image_path=tmp_path / "page.png",
        reference="",
    )

    try:
        doctr_orientation_benchmark._prediction_records(
            tmp_path,
            [case],
            [(np.zeros((2, 2, 3)), 1.0)],
            [[0], [0], [1.2]],
            0,
            1.0,
        )
    except ValueError as error:
        assert "invalid confidence" in str(error)
    else:
        raise AssertionError("invalid confidence was accepted")


def test_prepare_image_applies_exif_orientation(tmp_path: Path) -> None:
    image_path = tmp_path / "oriented.jpg"
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (10, 20), "white").save(image_path, exif=exif)
    case = BenchmarkCase(
        id="rotated/page",
        cluster_id="1",
        subset="rotated",
        image_path=image_path,
        reference="",
    )

    prepared, _ = doctr_orientation_benchmark._prepare_image(
        case,
        rectify=False,
        rectify_padding=0.05,
    )

    assert prepared.shape == (10, 20, 3)
