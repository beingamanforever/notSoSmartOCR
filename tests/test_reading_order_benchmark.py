from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from ocr_pipeline.contracts import BoundingBox, TextRegion
from ocr_pipeline.providers import ReaderError

from experiments import reading_order_benchmark
from experiments.public_benchmark import BenchmarkCase


def test_adaptive_xy_cut_orders_vertical_columns_before_rows() -> None:
    regions = [
        _region("left-top", 0, 0, 40, 10, 4),
        _region("right-top", 100, 0, 140, 10, 1),
        _region("left-bottom", 0, 30, 40, 40, 3),
        _region("right-bottom", 100, 30, 140, 40, 2),
    ]

    ordered = reading_order_benchmark.adaptive_xy_cut_order(regions, 1.0)

    assert [region.id for region in ordered] == [
        "left-top",
        "left-bottom",
        "right-top",
        "right-bottom",
    ]
    assert all(
        output is source
        for output, source in zip(
            ordered, [regions[0], regions[2], regions[1], regions[3]], strict=True
        )
    )


def test_adaptive_xy_cut_orders_horizontal_bands_before_columns() -> None:
    regions = [
        _region("bottom-right", 35, 100, 65, 110, 1),
        _region("top-left", 0, 0, 30, 10, 4),
        _region("bottom-left", 0, 100, 30, 110, 2),
        _region("top-right", 35, 0, 65, 10, 3),
    ]

    ordered = reading_order_benchmark.adaptive_xy_cut_order(regions, 1.0)

    assert [region.id for region in ordered] == [
        "top-left",
        "top-right",
        "bottom-left",
        "bottom-right",
    ]


def test_overlapping_regions_fall_back_to_stable_top_left() -> None:
    regions = [
        _region("second", 5, 5, 25, 25, 1),
        _region("first", 0, 0, 20, 20, 2),
        _region("third", 10, 10, 30, 30, 3),
    ]

    ordered = reading_order_benchmark.adaptive_xy_cut_order(regions, 0.5)

    assert ordered == reading_order_benchmark.stable_top_left_order(regions)
    assert [region.id for region in ordered] == ["first", "second", "third"]


def test_no_progress_cut_falls_back_instead_of_recursing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    regions = [
        _region("second", 20, 20, 30, 30, 1),
        _region("first", 0, 0, 10, 10, 2),
    ]
    monkeypatch.setattr(
        reading_order_benchmark,
        "_best_cut",
        lambda items, minimum_gap: (list(items), []),
    )

    ordered = reading_order_benchmark.adaptive_xy_cut_order(regions, 1.0)

    assert [region.id for region in ordered] == ["first", "second"]


def test_orders_are_deterministic_with_equal_geometry() -> None:
    regions = [
        _region("b", 0, 0, 10, 10, 2),
        _region("a", 0, 0, 10, 10, 1),
        _region("c", 20, 0, 30, 10, 3),
    ]

    first = reading_order_benchmark.adaptive_xy_cut_order(regions, 0.5)
    second = reading_order_benchmark.adaptive_xy_cut_order(regions, 0.5)

    assert [id(region) for region in first] == [id(region) for region in second]
    assert [region.id for region in first] == ["b", "a", "c"]


def test_invariant_failure_rejects_a_dropped_region() -> None:
    regions = [
        _region("first", 0, 0, 10, 10, 1),
        _region("second", 20, 0, 30, 10, 2),
    ]

    with pytest.raises(ValueError, match="Ordering invariant failed"):
        reading_order_benchmark.validate_ordering(
            regions,
            regions[:1],
            regions[:1],
            "broken",
        )


def test_fake_reader_end_to_end_uses_one_paragraph_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (160, 80), "white").save(image_path)
    monkeypatch.setattr(
        reading_order_benchmark,
        "rectify_document",
        lambda image, padding_fraction: image.copy(),
    )
    monkeypatch.setattr(
        reading_order_benchmark,
        "detect_tesseract_orientation",
        lambda path, executable: {
            "angle": 0,
            "rotate_clockwise": 0,
            "confidence": 12.0,
            "script": "Latin",
            "script_confidence": 8.0,
        },
    )

    class FakeReader:
        name = "fake-paragraph-reader"
        language = "en"
        merge_level = "paragraph"
        batch_size = 1

        def __init__(self) -> None:
            self.calls = 0

        def read_with_merge_level(
            self,
            prepared_path: Path,
            page_number: int,
            merge_level: str,
        ) -> list[TextRegion]:
            self.calls += 1
            assert prepared_path.name == "prepared.png"
            assert page_number == 1
            assert merge_level == "paragraph"
            left = _region("left", 0, 0, 40, 10, 2)
            left.text_provenance = {"method": "recognizer"}
            return [
                _region("right", 100, 0, 140, 10, 1),
                left,
            ]

    reader = FakeReader()
    case = BenchmarkCase(
        id="normal/case-1",
        cluster_id="template-1",
        subset="normal",
        image_path=image_path,
        reference="left right",
    )

    result = reading_order_benchmark.run_experiment(
        [case],
        tmp_path,
        reader,
        mode="eval",
        gap_thresholds=[1.0],
        osd_executable="fake-tesseract",
        selection_source="dev-result.json",
    )

    assert reader.calls == 1
    assert result["case_ids"] == ["normal/case-1"]
    assert result["config"]["merge_level"] == "paragraph"
    assert result["config"]["language"] == "en"
    assert result["config"]["batch_size"] == 1
    assert result["config"]["paragraph_reader_method"] == "read_with_merge_level"
    assert result["config"]["gap_thresholds"] == [1.0]
    assert result["summary"]["all_ordering_invariants_passed"] is True
    record = result["cases"][0]
    assert record["paragraph_reader_calls"] == 1
    assert record["osd_prepass_calls"] == 1
    assert record["source_region_ids"] == ["right", "left"]
    assert record["arms"]["baseline"]["region_ids"] == ["right", "left"]
    assert record["arms"]["top_left"]["region_ids"] == ["left", "right"]
    adaptive = record["arms"]["adaptive_xy_cut_1"]
    assert adaptive["region_ids"] == ["left", "right"]
    assert adaptive["prediction"] == "left\n\nright"
    assert adaptive["metrics"]["wer"]["rate"] == 0.0
    assert adaptive["invariants"]["coverage"] == 1.0
    assert record["preprocessing"]["osd"]["confidence"] == 12.0
    assert record["median_region_height_px"] == 10
    assert [
        record["source_regions"][index]["text"] for index in adaptive["source_indices"]
    ] == [
        "left",
        "right",
    ]
    left_source = record["source_regions"][1]
    assert left_source == {
        "source_index": 1,
        "id": "left",
        "kind": "paragraph",
        "text": "left",
        "confidence": 0.9,
        "bounding_box": {"left": 0, "top": 0, "right": 40, "bottom": 10},
        "reading_order": 2,
        "provider": "fake",
        "text_provenance": {"method": "recognizer"},
    }


def test_exif_transpose_precedes_rectification_and_nonzero_osd_rotation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = tmp_path / "exif.jpg"
    image = Image.new("RGB", (40, 20), "white")
    exif = image.getexif()
    exif[274] = 6
    image.save(image_path, exif=exif)

    def fake_rectify(image: Image.Image, padding_fraction: float) -> Image.Image:
        assert image.size == (20, 40)
        assert padding_fraction == 0.05
        return image.copy()

    monkeypatch.setattr(reading_order_benchmark, "rectify_document", fake_rectify)
    monkeypatch.setattr(
        reading_order_benchmark,
        "detect_tesseract_orientation",
        lambda path, executable: {
            "angle": 90,
            "rotate_clockwise": 90,
            "confidence": 20.0,
            "script": "Latin",
            "script_confidence": 10.0,
        },
    )

    class FakeReader:
        name = "fake"

        def read(self, prepared_path: Path, page_number: int) -> list[TextRegion]:
            with Image.open(prepared_path) as prepared:
                assert prepared.size == (40, 20)
            return [_region("text", 0, 0, 20, 10, 1)]

    result = reading_order_benchmark.run_experiment(
        [_case("normal/case-exif", "normal", image_path, "text")],
        tmp_path,
        FakeReader(),
        mode="dev",
        gap_thresholds=[1.0],
    )

    preprocessing = result["cases"][0]["preprocessing"]
    assert preprocessing["source_size"] == [40, 20]
    assert preprocessing["exif_orientation"] == 6
    assert preprocessing["exif_transposed"] is True
    assert preprocessing["exif_normalized_size"] == [20, 40]
    assert preprocessing["osd"]["angle"] == 90
    assert preprocessing["prepared_width"] == 40
    assert preprocessing["prepared_height"] == 20


@pytest.mark.parametrize(
    ("reader_output", "failure_code"),
    [
        (ReaderError("reader_failed", "reader stopped"), "reader_failed"),
        ([], "no_text_regions"),
    ],
)
def test_reader_failures_remain_in_coverage_and_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reader_output: ReaderError | list[TextRegion],
    failure_code: str,
) -> None:
    image_path = tmp_path / f"{failure_code}.png"
    Image.new("RGB", (80, 40), "white").save(image_path)
    _patch_preparation(monkeypatch)

    class FakeReader:
        name = "fake"

        def read(self, prepared_path: Path, page_number: int) -> list[TextRegion]:
            if isinstance(reader_output, ReaderError):
                raise reader_output
            return reader_output

    result = reading_order_benchmark.run_experiment(
        [_case(f"normal/{failure_code}", "normal", image_path, "expected text")],
        tmp_path,
        FakeReader(),
        mode="dev",
        gap_thresholds=[1.0],
    )

    record = result["cases"][0]
    assert record["status"] == "failed"
    assert record["failure"]["code"] == failure_code
    assert record["paragraph_reader_calls"] == 1
    assert record["osd_prepass_calls"] == 1
    assert result["summary"]["coverage"] == 0.0
    assert result["summary"]["failure_codes"] == {failure_code: 1}
    for arm in result["summary"]["arms"].values():
        assert arm["coverage"] == 0.0
        assert arm["failure_codes"] == {failure_code: 1}
        assert arm["wer"]["reference_units"] == 2


def test_osd_failure_does_not_count_a_paragraph_reader_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = tmp_path / "osd-failure.png"
    Image.new("RGB", (80, 40), "white").save(image_path)
    monkeypatch.setattr(
        reading_order_benchmark,
        "rectify_document",
        lambda image, padding_fraction: image.copy(),
    )

    def fail_osd(path: Path, executable: str) -> dict[str, object]:
        raise ReaderError("osd_failed", "no orientation")

    monkeypatch.setattr(
        reading_order_benchmark,
        "detect_tesseract_orientation",
        fail_osd,
    )

    class UncalledReader:
        name = "uncalled"

        def read(self, prepared_path: Path, page_number: int) -> list[TextRegion]:
            raise AssertionError("paragraph reader must not run after OSD failure")

    result = reading_order_benchmark.run_experiment(
        [_case("normal/osd-failure", "normal", image_path, "expected")],
        tmp_path,
        UncalledReader(),
        mode="dev",
        gap_thresholds=[1.0],
    )

    record = result["cases"][0]
    assert record["failure"]["code"] == "osd_failed"
    assert record["osd_prepass_calls"] == 1
    assert record["paragraph_reader_calls"] == 0
    assert record["preprocessing"]["osd_prepass_status"] == "failed"


def test_zero_degree_fallback_preserves_osd_failure_and_runs_reader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = tmp_path / "osd-fallback.png"
    Image.new("RGB", (80, 40), "white").save(image_path)
    monkeypatch.setattr(
        reading_order_benchmark,
        "rectify_document",
        lambda image, padding_fraction: image.copy(),
    )

    def fail_osd(path: Path, executable: str) -> dict[str, object]:
        raise ReaderError("osd_failed", "no orientation")

    monkeypatch.setattr(
        reading_order_benchmark,
        "detect_tesseract_orientation",
        fail_osd,
    )

    class FakeReader:
        name = "fake"

        def read(self, prepared_path: Path, page_number: int) -> list[TextRegion]:
            with Image.open(prepared_path) as prepared:
                assert prepared.size == (80, 40)
            return [_region("expected", 0, 0, 40, 10, 1)]

    result = reading_order_benchmark.run_experiment(
        [_case("normal/osd-fallback", "normal", image_path, "expected")],
        tmp_path,
        FakeReader(),
        mode="dev",
        gap_thresholds=[1.0],
        osd_failure_mode="zero",
    )

    record = result["cases"][0]
    assert record["status"] == "success"
    assert record["failure"] is None
    assert record["osd_prepass_calls"] == 1
    assert record["paragraph_reader_calls"] == 1
    assert record["preprocessing"]["osd_prepass_status"] == "fallback_zero"
    assert record["preprocessing"]["osd_fallback_angle"] == 0
    assert record["preprocessing"]["osd_prepass_failure"] == {
        "code": "osd_failed",
        "message": "no orientation",
    }
    assert result["config"]["osd_failure_mode"] == "zero"
    assert result["summary"]["coverage"] == 1.0


def test_invalid_osd_failure_mode_is_rejected(tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (80, 40), "white").save(image_path)

    with pytest.raises(ValueError, match="Unsupported OSD failure mode"):
        reading_order_benchmark.run_experiment(
            [_case("normal/page", "normal", image_path, "expected")],
            tmp_path,
            object(),
            mode="dev",
            gap_thresholds=[1.0],
            osd_failure_mode="guess",
        )


def test_subset_summaries_and_dev_selection_use_failure_inclusive_points(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_path = tmp_path / "first.png"
    second_path = tmp_path / "second.png"
    Image.new("RGB", (80, 40), "white").save(first_path)
    Image.new("RGB", (80, 40), "white").save(second_path)
    _patch_preparation(monkeypatch)

    class FakeReader:
        name = "fake"

        def read(self, prepared_path: Path, page_number: int) -> list[TextRegion]:
            return [_region("text", 0, 0, 20, 10, 1)]

    result = reading_order_benchmark.run_experiment(
        [
            _case("normal/one", "normal", first_path, "text"),
            _case("rotated/two", "rotated", second_path, "text"),
        ],
        tmp_path,
        FakeReader(),
        mode="dev",
        gap_thresholds=[1.0, 0.5],
    )

    assert set(result["subsets"]) == {"normal", "rotated"}
    assert result["subsets"]["normal"]["cases"] == 1
    assert result["subsets"]["rotated"]["coverage"] == 1.0
    assert result["selection"] == {
        "arm": "adaptive_xy_cut_0.5",
        "gap_threshold": 0.5,
        "rule": reading_order_benchmark.SELECTION_RULE,
        "evidence": "development_point_metrics_only",
        "micro_cer_delta": 0.0,
        "micro_wer_delta": 0.0,
        "coverage_delta": 0.0,
        "selection_cases": 2,
        "minimum_selection_cases": 30,
        "eligible_for_eval": False,
        "statistical_evidence": "not_assessed_by_selection_rule",
    }


def test_eval_cli_uses_untampered_dev_selection_only(
    tmp_path: Path,
) -> None:
    dev_path = tmp_path / "dev.json"
    output_path = tmp_path / "eval.json"
    arms = {
        "baseline": {
            "cases": 30,
            "wer": {"micro": 0.15},
            "cer": {"micro": 0.25},
            "coverage": 1.0,
        },
        "adaptive_xy_cut_0.5": {
            "cases": 30,
            "wer": {"micro": 0.1},
            "cer": {"micro": 0.2},
            "coverage": 1.0,
        },
        "adaptive_xy_cut_1": {
            "cases": 30,
            "wer": {"micro": 0.2},
            "cer": {"micro": 0.1},
            "coverage": 1.0,
        },
    }
    selection = reading_order_benchmark._select_threshold(arms, (0.5, 1.0))
    dev_path.write_text(
        json.dumps(
            {
                "experiment": reading_order_benchmark.EXPERIMENT_ID,
                "status": "complete",
                "dataset": "clinocr",
                "mode": "dev",
                "config": {"gap_thresholds": [0.5, 1.0]},
                "summary": {"arms": arms},
                "selection": selection,
            }
        ),
        encoding="utf-8",
    )

    thresholds, source = reading_order_benchmark._cli_thresholds(
        "eval", None, dev_path, output_path
    )
    assert thresholds == (0.5,)
    assert source == str(dev_path)
    with pytest.raises(ValueError, match="rejects --gap-threshold"):
        reading_order_benchmark._cli_thresholds("eval", [1.0], dev_path, output_path)
    with pytest.raises(ValueError, match="must not overwrite"):
        reading_order_benchmark._cli_thresholds("eval", None, dev_path, dev_path)

    payload = json.loads(dev_path.read_text(encoding="utf-8"))
    payload["selection"]["gap_threshold"] = 1.0
    dev_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        reading_order_benchmark._cli_thresholds("eval", None, dev_path, output_path)


def test_eval_rejects_a_development_candidate_that_worsens_baseline(
    tmp_path: Path,
) -> None:
    dev_path = tmp_path / "dev.json"
    arms = {
        "baseline": {
            "cases": 30,
            "wer": {"micro": 0.1},
            "cer": {"micro": 0.1},
            "coverage": 1.0,
        },
        "adaptive_xy_cut_0.5": {
            "cases": 30,
            "wer": {"micro": 0.2},
            "cer": {"micro": 0.2},
            "coverage": 1.0,
        },
    }
    selection = reading_order_benchmark._select_threshold(arms, (0.5,))
    dev_path.write_text(
        json.dumps(
            {
                "experiment": reading_order_benchmark.EXPERIMENT_ID,
                "status": "complete",
                "dataset": "clinocr",
                "mode": "dev",
                "config": {"gap_thresholds": [0.5]},
                "summary": {"arms": arms},
                "selection": selection,
            }
        ),
        encoding="utf-8",
    )

    assert selection["eligible_for_eval"] is False
    assert selection["micro_cer_delta"] == 0.1
    assert selection["micro_wer_delta"] == 0.1
    with pytest.raises(ValueError, match="not eligible"):
        reading_order_benchmark._cli_thresholds(
            "eval", None, dev_path, tmp_path / "eval.json"
        )


def test_eval_cli_rejects_an_inline_threshold(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        reading_order_benchmark.main(
            [
                str(tmp_path),
                str(tmp_path / "eval.json"),
                "--mode",
                "eval",
                "--gap-threshold",
                "1.0",
            ]
        )

    assert "use --dev-result selection" in capsys.readouterr().err


def test_eval_execution_requires_one_selected_threshold() -> None:
    with pytest.raises(ValueError, match="exactly one selected"):
        reading_order_benchmark._thresholds("eval", None)
    with pytest.raises(ValueError, match="exactly one selected"):
        reading_order_benchmark._thresholds("eval", [0.5, 1.0])
    assert reading_order_benchmark._thresholds("eval", [1.0]) == (1.0,)


def _case(
    case_id: str,
    subset: str,
    image_path: Path,
    reference: str,
) -> BenchmarkCase:
    return BenchmarkCase(
        id=case_id,
        cluster_id="template-1",
        subset=subset,
        image_path=image_path,
        reference=reference,
    )


def _patch_preparation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        reading_order_benchmark,
        "rectify_document",
        lambda image, padding_fraction: image.copy(),
    )
    monkeypatch.setattr(
        reading_order_benchmark,
        "detect_tesseract_orientation",
        lambda path, executable: {
            "angle": 0,
            "rotate_clockwise": 0,
            "confidence": 12.0,
            "script": "Latin",
            "script_confidence": 8.0,
        },
    )


def _region(
    region_id: str,
    left: int,
    top: int,
    right: int,
    bottom: int,
    reading_order: int,
) -> TextRegion:
    return TextRegion(
        id=region_id,
        kind="paragraph",
        text=region_id.replace("-", " "),
        confidence=0.9,
        bounding_box=BoundingBox(left, top, right, bottom),
        reading_order=reading_order,
        provider="fake",
    )
