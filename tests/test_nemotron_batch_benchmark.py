from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from ocr_pipeline.contracts import BoundingBox, TextRegion
from ocr_pipeline.providers import ReaderError

from experiments import nemotron_batch_benchmark


class FakeCudaMonitor:
    available = True

    def __init__(self) -> None:
        self.calls = 0

    def begin(self) -> None:
        pass

    def finish(self) -> int:
        self.calls += 1
        return 1_000 + self.calls


class FakeReader:
    name = "fake-nemotron"

    def __init__(self, failed_pages: set[int] | None = None) -> None:
        self.batch_size = 1
        self.failed_pages = failed_pages or set()
        self.calls: list[tuple[int, list[int], list[tuple[int, int]]]] = []

    def read_batch(
        self,
        image_paths: list[Path],
        page_numbers: list[int],
    ) -> list[list[TextRegion] | ReaderError]:
        sizes = []
        for image_path in image_paths:
            with Image.open(image_path) as image:
                sizes.append(image.size)
        self.calls.append((self.batch_size, list(page_numbers), sizes))
        results = []
        for page_number, image_path in zip(page_numbers, image_paths, strict=True):
            if page_number in self.failed_pages:
                results.append(
                    ReaderError("fake_failure", f"failed page {page_number}")
                )
                continue
            results.append(
                [
                    TextRegion(
                        id=f"p{page_number}-block-1",
                        kind="paragraph",
                        text=f"page {page_number}",
                        confidence=0.875,
                        bounding_box=BoundingBox(1, 2, 8, 9),
                        reading_order=1,
                        provider="fake-nemotron",
                        text_provenance={"prepared": image_path.name},
                    )
                ]
            )
        return results


def test_sweep_reorders_outputs_and_keeps_failures_in_exact_summaries(
    tmp_path: Path,
) -> None:
    root = _clinocr_root(tmp_path, cases=3)
    reader = FakeReader(failed_pages={2})

    payload = nemotron_batch_benchmark.run_benchmark(
        root,
        clinocr_role="exemplar",
        subset="rotated",
        seed=17,
        permutations=4,
        reader=reader,
        timer=_step_timer(),
        cuda_monitor=FakeCudaMonitor(),
    )

    expected_ids = [
        "rotated/template_1_sample_1_rotated",
        "rotated/template_2_sample_1_rotated",
        "rotated/template_3_sample_1_rotated",
    ]
    assert payload["execution"]["seed"] == 17
    batch_size_orders = payload["execution"]["batch_size_orders"]
    assert batch_size_orders == {
        "0": [1, 4, 2, 8],
        "1": [4, 2, 8, 1],
        "2": [2, 8, 1, 4],
        "3": [8, 1, 4, 2],
    }
    assert payload["execution"]["run_order"] == [
        f"run-{index:02d}-p{permutation}-b{batch_size}"
        for index, (permutation, batch_size) in enumerate(
            (permutation, batch_size)
            for permutation in range(4)
            for batch_size in batch_size_orders[str(permutation)]
        )
    ]
    assert len(payload["runs"]) == 16
    assert all(
        {batch_size_orders[str(repetition)][position] for repetition in range(4)}
        == {1, 2, 4, 8}
        for position in range(4)
    )
    assert any(run["request_case_ids"] != expected_ids for run in payload["runs"])
    for run in payload["runs"]:
        assert run["returned_case_ids"] == expected_ids
        assert [case["id"] for case in run["cases"]] == expected_ids
        assert [case["page_number"] for case in run["cases"]] == [1, 2, 3]
        assert run["summary"]["cases"] == 3
        assert run["summary"]["covered_cases"] == 2
        assert run["summary"]["coverage"] == pytest.approx(2 / 3)
        assert run["summary"]["failure_codes"] == {"fake_failure": 1}
        assert run["summary"]["cer"]["reference_units"] == 18
        assert run["summary"]["wer"]["reference_units"] == 6
        assert run["exact_equality"]["rate"] == 1.0
        assert run["summary"]["cuda_peak_allocated_bytes"] is not None
        for batch in run["batches"]:
            records = {case["id"]: case for case in run["cases"]}
            assert all(
                records[case_id]["latency_ms"] == batch["latency_ms"]
                for case_id in batch["case_ids"]
            )
    failed = payload["runs"][0]["cases"][1]
    assert failed["prediction"] == ""
    assert failed["returned_error"] == {
        "code": "fake_failure",
        "message": "failed page 2",
    }
    assert failed["metrics"]["cer"]["rate"] == 1.0
    assert payload["exact_equality"]["rate"] == 1.0
    batch_two = payload["batch_size_aggregates"]["2"]
    assert batch_two["runs"] == 4
    assert batch_two["median_pages_per_second"] == 1500.0
    assert batch_two["median_full_page_latency_ms"] == {"p50": 1.0, "p95": 1.0}
    assert batch_two["median_cuda_peak_allocated_bytes"] is not None
    assert batch_two["exact_output_equality"] == {
        "baseline_run_id": next(
            run["run_id"] for run in payload["runs"] if run["batch_size"] == 2
        ),
        "matching_case_runs": 12,
        "case_runs": 12,
        "rate": 1.0,
    }
    canonical = payload["runs"][0]["cases"][0]["canonical_regions"][0]
    assert canonical == {
        "id": "p1-block-1",
        "kind": "paragraph",
        "text": "page 1",
        "confidence": 0.875,
        "bounding_box": {"left": 1, "top": 2, "right": 8, "bottom": 9},
        "reading_order": 1,
        "provider": "fake-nemotron",
        "text_provenance": {"prepared": "page-1.png"},
        "resolution": "resolved",
        "alternatives": [],
        "structure": None,
    }


def test_exif_and_osd_are_prepared_once_then_reused(tmp_path: Path) -> None:
    root = _clinocr_root(tmp_path, cases=1, image_size=(20, 10), orientation=6)
    reader = FakeReader()
    osd_calls: list[tuple[Path, str, tuple[int, int]]] = []

    def fake_osd(image_path: Path, *, executable: str) -> dict[str, object]:
        with Image.open(image_path) as image:
            size = image.size
        osd_calls.append((image_path, executable, size))
        return {"angle": 0, "confidence": 99.0, "script": "Latin"}

    payload = nemotron_batch_benchmark.run_benchmark(
        root,
        clinocr_role="exemplar",
        subset="rotated",
        permutations=1,
        osd_executable="fake-tesseract",
        osd_detector=fake_osd,
        reader=reader,
        timer=_step_timer(),
        cuda_monitor=FakeCudaMonitor(),
    )

    assert len(osd_calls) == 1
    assert osd_calls[0][1:] == ("fake-tesseract", (10, 20))
    assert payload["preparation"][0]["exif_orientation"] == 6
    assert payload["preparation"][0]["exif_normalized_size"] == [10, 20]
    assert payload["preparation"][0]["prepared_size"] == [10, 20]
    assert payload["preparation"][0]["osd"]["accepted"] is True
    assert all(size == (10, 20) for _, _, sizes in reader.calls for size in sizes)
    assert len({path.name for path, _, _ in osd_calls}) == 1
    assert payload["preparation_policy"]["gold_used"] is False
    assert payload["warmup"]["excluded_from_scored_runs"] is True
    assert len(payload["runs"]) == 4


def test_osd_failure_is_returned_but_unrotated_page_is_still_scored(
    tmp_path: Path,
) -> None:
    root = _clinocr_root(tmp_path, cases=1)

    def failed_osd(image_path: Path, *, executable: str) -> dict[str, object]:
        raise ReaderError("osd_failed", f"{executable} failed for {image_path.name}")

    payload = nemotron_batch_benchmark.run_benchmark(
        root,
        clinocr_role="exemplar",
        subset="rotated",
        permutations=1,
        osd_executable="fake-osd",
        osd_detector=failed_osd,
        reader=FakeReader(),
        timer=_step_timer(),
        cuda_monitor=FakeCudaMonitor(),
    )

    assert payload["preparation"][0]["osd_error"] == {
        "code": "osd_failed",
        "message": "fake-osd failed for osd-1.png",
    }
    assert payload["preparation"][0]["status"] == "success"
    assert payload["runs"][0]["summary"]["coverage"] == 1.0


def test_exact_equality_detects_batch_dependent_regions(tmp_path: Path) -> None:
    root = _clinocr_root(tmp_path, cases=2)

    class BatchDependentReader(FakeReader):
        def read_batch(
            self,
            image_paths: list[Path],
            page_numbers: list[int],
        ) -> list[list[TextRegion] | ReaderError]:
            results = super().read_batch(image_paths, page_numbers)
            for result in results:
                if not isinstance(result, ReaderError):
                    result[0].text_provenance = {"batch_size": self.batch_size}
            return results

    payload = nemotron_batch_benchmark.run_benchmark(
        root,
        clinocr_role="exemplar",
        subset="rotated",
        permutations=1,
        reader=BatchDependentReader(),
        timer=_step_timer(),
        cuda_monitor=FakeCudaMonitor(),
    )

    assert payload["runs"][0]["exact_equality"]["rate"] == 1.0
    assert payload["runs"][1]["exact_equality"]["rate"] == 0.0
    assert payload["exact_equality"]["rate"] == 0.0
    assert all(
        aggregate["exact_output_equality"]["rate"] == 1.0
        for aggregate in payload["batch_size_aggregates"].values()
    )


@pytest.mark.parametrize(
    "arguments, message",
    [
        (["--permutations", "0"], "must be at least 1"),
        (["--rectify-padding", "0.3"], "between 0 and 0.25"),
        (["--osd-min-confidence", "-1"], "cannot be negative"),
    ],
)
def test_cli_rejects_invalid_options(
    tmp_path: Path,
    arguments: list[str],
    message: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as raised:
        nemotron_batch_benchmark.main(
            [str(tmp_path / "root"), str(tmp_path / "out.json"), *arguments]
        )

    assert raised.value.code == 2
    assert message in capsys.readouterr().err


def test_cli_writes_selected_frozen_role_and_subset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "result.json"
    captured: dict[str, object] = {}

    def fake_run(root: Path, **options: object) -> dict[str, object]:
        captured.update(options)
        return {"ok": True}

    monkeypatch.setattr(nemotron_batch_benchmark, "run_benchmark", fake_run)

    assert (
        nemotron_batch_benchmark.main(
            [
                str(tmp_path),
                str(output),
                "--clinocr-role",
                "eval",
                "--subset",
                "forms",
            ]
        )
        == 0
    )
    assert captured["clinocr_role"] == "eval"
    assert captured["subset"] == "forms"
    assert json.loads(output.read_text(encoding="utf-8")) == {"ok": True}


def _clinocr_root(
    tmp_path: Path,
    *,
    cases: int,
    image_size: tuple[int, int] = (10, 10),
    orientation: int | None = None,
) -> Path:
    root = tmp_path / "clinocr"
    scans = root / "scans" / "rotated"
    references = root / "ground_truth" / "rotated"
    scans.mkdir(parents=True)
    references.mkdir(parents=True)
    rows = ["subset,template,sample,role"]
    for index in range(1, cases + 1):
        stem = f"template_{index}_sample_1_rotated"
        rows.append(f"rotated,{index},1,exemplar")
        image = Image.new("RGB", image_size, "white")
        image_path = scans / f"{stem}.jpg"
        if orientation is None:
            image.save(image_path)
        else:
            exif = Image.Exif()
            exif[274] = orientation
            image.save(image_path, exif=exif)
        (references / f"{stem}.txt").write_text(f"page {index}", encoding="utf-8")
    (root / "oneshot_lookup.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return root


def _step_timer() -> object:
    value = 0.0

    def timer() -> float:
        nonlocal value
        value += 0.001
        return value

    return timer
