from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import pytest
from PIL import Image

from experiments import osd_fallback_benchmark, reading_order_benchmark
from experiments.public_benchmark import BenchmarkCase, _score
from ocr_pipeline.contracts import BoundingBox, TextRegion
from ocr_pipeline.providers import ReaderError


class FakeReader:
    name = "nemotron-ocr-v2"
    language = "en"
    merge_level = "paragraph"

    def __init__(
        self,
        *,
        count: int = 10,
        confidence: float | None = 0.92,
    ) -> None:
        self.count = count
        self.confidence = confidence
        self.calls = 0

    def read_with_merge_level(
        self, image_path: Path, page_number: int, merge_level: str
    ) -> list[TextRegion]:
        self.calls += 1
        assert image_path.name == "prepared.png"
        assert page_number == 1
        assert merge_level == "paragraph"
        return [
            TextRegion(
                id=f"region-{index}",
                kind="paragraph",
                text=f"word{index}",
                confidence=self.confidence,
                bounding_box=BoundingBox(0, index * 10, 100, index * 10 + 8),
                reading_order=index,
                provider=self.name,
            )
            for index in range(self.count)
        ]


class PartialConfidenceReader(FakeReader):
    def read_with_merge_level(
        self, image_path: Path, page_number: int, merge_level: str
    ) -> list[TextRegion]:
        regions = super().read_with_merge_level(image_path, page_number, merge_level)
        for region in regions[1:]:
            region.confidence = None
        return regions


class BadReader(FakeReader):
    def read_with_merge_level(
        self, image_path: Path, page_number: int, merge_level: str
    ) -> list[TextRegion]:
        regions = super().read_with_merge_level(image_path, page_number, merge_level)
        for region in regions:
            region.text = "xxxxxxxxxxxxxxxx"
        return regions


def test_osd_success_gives_both_arms_one_ocr_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(tmp_path, 0)
    _patch_preparation(monkeypatch, osd_fails=False)
    reader = FakeReader()

    record = osd_fallback_benchmark._evaluate_case(
        case,
        tmp_path,
        reader,
        "fake-tesseract",
        0.05,
    )

    assert reader.calls == 1
    assert record["ocr_calls"] == 1
    assert record["fallback_evidence"]["osd_fallback"] is False
    strict = record["arms"]["strict_osd"]
    selective = record["arms"]["selective_fallback"]
    assert strict["prediction"] == selective["prediction"]
    assert strict["metrics"] == selective["metrics"]
    assert strict["status"] == selective["status"] == "success"


def test_osd_failure_accepts_fixed_zero_fallback_at_inclusive_thresholds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(tmp_path, 0)
    _patch_preparation(monkeypatch, osd_fails=True)
    reader = FakeReader(count=10, confidence=0.92)

    record = osd_fallback_benchmark._evaluate_case(
        case,
        tmp_path,
        reader,
        "fake-tesseract",
        0.05,
    )

    assert reader.calls == 1
    evidence = record["fallback_evidence"]
    assert evidence == {
        "osd_fallback": True,
        "fallback_angle_degrees": 0,
        "paragraph_regions": 10,
        "available_confidences": 10,
        "all_regions_have_confidence": True,
        "mean_available_confidence": 0.92,
        "minimum_paragraph_regions": 10,
        "minimum_mean_available_confidence": 0.92,
        "fallback_accepted": True,
    }
    assert record["arms"]["strict_osd"]["prediction"] == ""
    assert record["arms"]["strict_osd"]["failures"][0]["code"] == (
        "osd_fallback_disallowed"
    )
    assert record["arms"]["selective_fallback"]["prediction"] == _reference()
    assert record["arms"]["selective_fallback"]["metrics"]["cer"]["rate"] == 0


@pytest.mark.parametrize(
    ("count", "confidence"),
    [(9, 0.99), (10, 0.919), (10, None)],
)
def test_fallback_abstains_when_fixed_evidence_is_insufficient(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    count: int,
    confidence: float | None,
) -> None:
    case = _case(tmp_path, 0)
    _patch_preparation(monkeypatch, osd_fails=True)

    record = osd_fallback_benchmark._evaluate_case(
        case,
        tmp_path,
        FakeReader(count=count, confidence=confidence),
        "fake-tesseract",
        0.05,
    )

    assert record["fallback_evidence"]["fallback_accepted"] is False
    selective = record["arms"]["selective_fallback"]
    assert selective["prediction"] == ""
    assert selective["status"] == "failed"
    assert selective["failures"] == [
        {
            "code": "osd_fallback_abstained",
            "message": "Zero-degree fallback did not satisfy the fixed acceptance policy",
            "stage": "orientation",
        }
    ]


def test_fallback_abstains_when_region_confidence_is_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(tmp_path, 0)
    _patch_preparation(monkeypatch, osd_fails=True)

    record = osd_fallback_benchmark._evaluate_case(
        case,
        tmp_path,
        PartialConfidenceReader(),
        "fake-tesseract",
        0.05,
    )

    evidence = record["fallback_evidence"]
    assert evidence["paragraph_regions"] == 10
    assert evidence["available_confidences"] == 1
    assert evidence["all_regions_have_confidence"] is False
    assert evidence["fallback_accepted"] is False


def test_dev_run_uses_all_56_cases_and_is_promotable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cases = _cases(tmp_path)
    _patch_preparation(monkeypatch, osd_fails=True)
    reader = FakeReader()

    payload = osd_fallback_benchmark.run_experiment(
        cases,
        tmp_path,
        reader,
        mode="dev",
        osd_executable="fake-tesseract",
    )

    assert reader.calls == 56
    assert payload["case_ids"] == [case.id for case in cases]
    assert payload["summary"]["cases"] == 56
    assert payload["summary"]["attempted_cases"] == 56
    assert payload["summary"]["arms"]["strict_osd"]["cer"]["micro"] == 1
    assert payload["summary"]["arms"]["selective_fallback"]["cer"]["micro"] == 0
    assert payload["selection"]["promotable"] is True
    assert payload["selection"]["all_accepted_fallback_cer_below_0.2"] is True


def test_promotion_rejects_an_accepted_fallback_at_cer_point_two() -> None:
    records = [
        _selection_record("a", "abcde", "abxde", accepted=True),
        *[
            _selection_record(str(index), "abcde", "abcde", accepted=True)
            for index in range(1, 30)
        ],
    ]
    summary = osd_fallback_benchmark._summarize_records(records)

    selection = osd_fallback_benchmark._select_promotion(records, summary)

    assert records[0]["arms"]["selective_fallback"]["metrics"]["cer"]["rate"] == 0.2
    assert selection["all_accepted_fallback_cer_below_0.2"] is False
    assert selection["promotable"] is False


def test_dev_artifact_validation_recomputes_policy_panel_metrics_and_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cases = _cases(tmp_path)
    _patch_preparation(monkeypatch, osd_fails=True)
    payload = osd_fallback_benchmark.run_experiment(
        cases,
        tmp_path,
        FakeReader(),
        mode="dev",
    )
    result_path = tmp_path / "dev.json"
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    validated = osd_fallback_benchmark._validate_dev_artifact(
        result_path, cases, tmp_path
    )

    assert validated["selection"]["promotable"] is True

    changed_policy = json.loads(json.dumps(payload))
    changed_policy["policy"]["fallback_min_paragraph_regions"] = 9
    result_path.write_text(json.dumps(changed_policy), encoding="utf-8")
    with pytest.raises(ValueError, match="exact fixed policy"):
        osd_fallback_benchmark._validate_dev_artifact(result_path, cases, tmp_path)

    changed_evidence = json.loads(json.dumps(payload))
    changed_evidence["cases"][0]["fallback_evidence"]["paragraph_regions"] = 9
    result_path.write_text(json.dumps(changed_evidence), encoding="utf-8")
    with pytest.raises(ValueError, match="observable evidence"):
        osd_fallback_benchmark._validate_dev_artifact(result_path, cases, tmp_path)

    changed_selection = json.loads(json.dumps(payload))
    changed_selection["selection"]["promotable"] = False
    result_path.write_text(json.dumps(changed_selection), encoding="utf-8")
    with pytest.raises(ValueError, match="selection does not match"):
        osd_fallback_benchmark._validate_dev_artifact(result_path, cases, tmp_path)


def test_dev_artifact_rejects_coordinated_prediction_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cases = _cases(tmp_path)
    _patch_preparation(monkeypatch, osd_fails=True)
    payload = osd_fallback_benchmark.run_experiment(
        cases,
        tmp_path,
        BadReader(),
        mode="dev",
    )
    assert payload["selection"]["promotable"] is False
    for record in payload["cases"]:
        selective = record["arms"]["selective_fallback"]
        selective["prediction"] = _reference()
        selective["metrics"] = _score(_reference(), _reference())
    payload["summary"] = osd_fallback_benchmark._summarize_records(payload["cases"])
    payload["selection"] = osd_fallback_benchmark._select_promotion(
        payload["cases"], payload["summary"]
    )
    assert payload["selection"]["promotable"] is True
    result_path = tmp_path / "tampered.json"
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="prediction changed"):
        osd_fallback_benchmark._validate_dev_artifact(result_path, cases, tmp_path)


def test_dev_artifact_rejects_a_different_dataset_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cases = _cases(tmp_path)
    _patch_preparation(monkeypatch, osd_fails=True)
    payload = osd_fallback_benchmark.run_experiment(
        cases,
        tmp_path,
        FakeReader(),
        mode="dev",
    )
    payload["dataset_root"] = str((tmp_path / "other-root").resolve())
    result_path = tmp_path / "wrong-root.json"
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="complete dev artifact"):
        osd_fallback_benchmark._validate_dev_artifact(result_path, cases, tmp_path)


def test_output_claim_preserves_existing_file(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    output.write_bytes(b"existing result")

    with pytest.raises(ValueError, match="already exists"):
        osd_fallback_benchmark._claim_output(output)

    assert output.read_bytes() == b"existing result"


def test_only_one_concurrent_output_claim_succeeds(tmp_path: Path) -> None:
    output = tmp_path / "result.json"

    def claim() -> int | None:
        try:
            return osd_fallback_benchmark._claim_output(output)
        except ValueError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: claim(), range(2)))

    assert sum(result is not None for result in results) == 1


def test_panel_rejects_partial_dev_and_eval_runs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="all 56"):
        osd_fallback_benchmark._validate_panel(_cases(tmp_path)[:30], "dev")
    with pytest.raises(ValueError, match="all 328"):
        osd_fallback_benchmark._validate_panel(_cases(tmp_path), "eval")


def _patch_preparation(monkeypatch: pytest.MonkeyPatch, *, osd_fails: bool) -> None:
    monkeypatch.setattr(
        reading_order_benchmark,
        "rectify_document",
        lambda image, padding_fraction: image.copy(),
    )
    if osd_fails:

        def fail_osd(path: Path, executable: str) -> dict[str, object]:
            raise ReaderError("osd_failed", "not enough characters")

        monkeypatch.setattr(
            reading_order_benchmark,
            "detect_tesseract_orientation",
            fail_osd,
        )
        return
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


def _cases(root: Path) -> list[BenchmarkCase]:
    return [_case(root, index) for index in range(56)]


def _case(root: Path, index: int) -> BenchmarkCase:
    image_path = root / f"case-{index}.png"
    if not image_path.exists():
        Image.new("RGB", (160, 120), "white").save(image_path)
    return BenchmarkCase(
        id=f"normal/case-{index}",
        cluster_id=f"template-{index % 8}",
        subset="normal",
        image_path=image_path,
        reference=_reference(),
    )


def _reference() -> str:
    return "\n\n".join(f"word{index}" for index in range(10))


def _selection_record(
    case_id: str, reference: str, prediction: str, *, accepted: bool
) -> dict[str, object]:
    strict = {
        "prediction": "",
        "status": "failed",
        "metrics": _score("", reference),
        "failures": [{"code": "osd_fallback_disallowed"}],
        "latency_ms": 1.0,
    }
    selective = {
        "prediction": prediction,
        "status": "success",
        "metrics": _score(prediction, reference),
        "failures": [],
        "latency_ms": 1.0,
    }
    return {
        "id": case_id,
        "subset": "normal",
        "ocr_calls": 1,
        "fallback_evidence": {
            "osd_fallback": True,
            "fallback_accepted": accepted,
        },
        "arms": {"strict_osd": strict, "selective_fallback": selective},
    }
