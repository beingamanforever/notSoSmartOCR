from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from experiments import paired_comparison  # noqa: E402


def test_cli_compares_every_case_and_is_deterministic(tmp_path: Path) -> None:
    baseline = _payload(
        [
            ("failed", "alpha", ""),
            ("success", "gamma", "gamma"),
        ]
    )
    candidate = _payload(
        [
            ("success", "alpha", "alpha"),
            ("failed", "gamma", ""),
        ]
    )
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    output_path = tmp_path / "comparison.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

    project_root = Path(__file__).parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(project_root / "experiments" / "paired_comparison.py"),
            str(baseline_path),
            str(candidate_path),
            str(output_path),
            "--resamples",
            "200",
            "--seed",
            "17",
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["cases"] == 2
    assert result["resamples"] == 200
    assert result["seed"] == 17
    assert result["metrics"]["cer"]["reference_units"] == 10
    assert result["metrics"]["cer"]["baseline_micro"] == 0.5
    assert result["metrics"]["cer"]["candidate_micro"] == 0.5
    assert len(result["metrics"]["cer"]["micro_delta_bootstrap_ci_95"]) == 2
    assert result["metrics"]["cer"]["paired_cases"] == {
        "candidate_wins": 1,
        "ties": 0,
        "candidate_losses": 1,
    }
    assert result["metrics"]["wer"]["reference_units"] == 2
    assert result["metrics"]["wer"]["baseline_micro"] == 0.5
    assert result["metrics"]["wer"]["candidate_micro"] == 0.5
    assert result["sample_counts"] == {"pages": 2, "clusters": 2}
    assert result == paired_comparison.compare_files(
        baseline_path, candidate_path, resamples=200, seed=17
    )


def test_same_cascade_file_selects_local_and_repaired_metrics() -> None:
    payload = {
        "cases": [
            {
                "id": "case-b",
                "reference": "hello world",
                "local": {
                    "status": "failed",
                    "prediction": "hello",
                    "metrics": _metrics("hello", "hello world"),
                },
                "repaired": {
                    "status": "success",
                    "prediction": "hello world",
                    "metrics": _metrics("hello world", "hello world"),
                },
            },
            {
                "id": "case-a",
                "reference": "gamma",
                "local": {
                    "status": "success",
                    "prediction": "gamma",
                    "metrics": _metrics("gamma", "gamma"),
                },
                "repaired": {
                    "status": "success",
                    "prediction": "gamma",
                    "metrics": _metrics("gamma", "gamma"),
                },
            },
        ],
        "dataset": "funsd",
        "normalization": paired_comparison.NORMALIZATION,
    }

    result = paired_comparison.compare_payloads(
        payload,
        payload,
        baseline_variant="local",
        candidate_variant="repaired",
        resamples=100,
        seed=3,
    )

    assert result["cases"] == 2
    assert result["metrics"]["cer"]["baseline_micro"] == round(6 / 16, 6)
    assert result["metrics"]["cer"]["candidate_micro"] == 0.0
    assert result["metrics"]["cer"]["micro_delta"] == -0.375
    assert result["metrics"]["cer"]["paired_cases"] == {
        "candidate_wins": 1,
        "ties": 1,
        "candidate_losses": 0,
    }
    assert (
        result["metrics"]["cer"]["holm_adjusted_p_value"]
        >= result["metrics"]["cer"]["two_sided_sign_flip_p_value"]
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload["cases"].append(deepcopy(payload["cases"][0])),
            "Duplicate candidate case id",
        ),
        (
            lambda payload: payload["cases"].pop(),
            "Case IDs differ",
        ),
        (
            lambda payload: payload["cases"][0].__setitem__("reference", "other"),
            "References differ",
        ),
        (
            lambda payload: payload["cases"][0]["metrics"]["cer"].__setitem__(
                "reference_units", 99
            ),
            "incompatible CER reference denominator",
        ),
    ],
)
def test_rejects_unpaired_inputs(mutate, message: str) -> None:
    baseline = _payload(
        [
            ("success", "alpha", "alpha"),
            ("failed", "beta", ""),
        ]
    )
    candidate = deepcopy(baseline)
    mutate(candidate)

    with pytest.raises(ValueError, match=message):
        paired_comparison.compare_payloads(baseline, candidate, resamples=10, seed=0)


def test_rejects_missing_metrics_and_nonpositive_resamples() -> None:
    payload = _payload([("success", "alpha", "alpha")])
    missing_metrics = deepcopy(payload)
    del missing_metrics["cases"][0]["metrics"]["wer"]

    with pytest.raises(ValueError, match="has no WER metrics"):
        paired_comparison.compare_payloads(payload, missing_metrics, resamples=10)
    with pytest.raises(ValueError, match="resamples must be positive"):
        paired_comparison.compare_payloads(payload, payload, resamples=0)


def test_recomputes_edits_instead_of_trusting_stored_metrics() -> None:
    baseline = _payload([("success", "alpha beta", "alpha x")])
    candidate = _payload([("success", "alpha beta", "ＡＬＰＨＡ　ＢＥＴＡ")])
    baseline["cases"][0]["metrics"]["cer"]["edits"] = 0
    baseline["cases"][0]["metrics"]["wer"]["edits"] = 0
    candidate["cases"][0]["metrics"]["cer"]["edits"] = 999
    candidate["cases"][0]["metrics"]["wer"]["edits"] = 999

    result = paired_comparison.compare_payloads(
        baseline, candidate, resamples=20, seed=4
    )

    assert result["metrics"]["cer"]["candidate_micro"] == 0.0
    assert result["metrics"]["cer"]["micro_delta"] < 0
    assert result["metrics"]["wer"]["candidate_micro"] == 0.0
    assert result["metrics"]["wer"]["micro_delta"] < 0


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.__setitem__("dataset", "other"),
        lambda payload: payload.__setitem__("normalization", "lowercase only"),
    ],
)
def test_rejects_incompatible_benchmark_metadata(mutation) -> None:
    baseline = _payload([("success", "alpha", "alpha")])
    candidate = deepcopy(baseline)
    mutation(candidate)

    with pytest.raises(ValueError, match="metadata differs"):
        paired_comparison.compare_payloads(baseline, candidate, resamples=10)


def test_rejects_unknown_normalization_even_when_files_agree() -> None:
    payload = _payload([("success", "alpha", "alpha")])
    payload["normalization"] = "lowercase only"

    with pytest.raises(ValueError, match="Unsupported benchmark normalization"):
        paired_comparison.compare_payloads(payload, payload, resamples=10)


def test_cluster_resampling_is_deterministic_for_clinocr_templates() -> None:
    baseline = _payload(
        [
            ("success", "aaaa", "bbbb"),
            ("success", "cccc", "dddd"),
            ("success", "eeee", "eeee"),
        ],
        dataset="clinocr",
        case_ids=[
            "normal/template_1_sample_2_normal",
            "rotated/template_1_sample_2_rotated",
            "normal/template_2_sample_2_normal",
        ],
    )
    candidate = _payload(
        [
            ("success", "aaaa", "aaaa"),
            ("success", "cccc", "cccc"),
            ("success", "eeee", "ffff"),
        ],
        dataset="clinocr",
        case_ids=[
            "normal/template_1_sample_2_normal",
            "rotated/template_1_sample_2_rotated",
            "normal/template_2_sample_2_normal",
        ],
    )

    first = paired_comparison.compare_payloads(
        baseline, candidate, resamples=200, seed=9
    )
    second = paired_comparison.compare_payloads(
        baseline, candidate, resamples=200, seed=9
    )

    assert first == second
    assert first["sample_counts"] == {"pages": 3, "clusters": 2}
    assert first["primary_resampling_unit"] == "cluster"
    assert first["exploratory_resampling_unit"] == "page"
    assert "micro_delta_page_bootstrap_ci_95_exploratory" in first["metrics"]["cer"]


def test_explicit_cluster_id_overrides_derived_clinocr_template() -> None:
    payload = _payload(
        [
            ("success", "aaaa", "aaaa"),
            ("success", "bbbb", "bbbb"),
        ],
        dataset="clinocr",
        case_ids=[
            "normal/template_1_sample_2_normal",
            "normal/template_2_sample_2_normal",
        ],
    )
    for case in payload["cases"]:
        case["cluster_id"] = "shared-source"

    result = paired_comparison.compare_payloads(payload, payload, resamples=20, seed=2)

    assert result["sample_counts"] == {"pages": 2, "clusters": 1}


def test_requires_prediction_and_recomputed_reference_denominator() -> None:
    payload = _payload([("success", "alpha", "alpha")])
    missing_prediction = deepcopy(payload)
    del missing_prediction["cases"][0]["prediction"]
    bad_denominator = deepcopy(payload)
    bad_denominator["cases"][0]["metrics"]["cer"]["reference_units"] = 4

    with pytest.raises(ValueError, match="no valid prediction"):
        paired_comparison.compare_payloads(payload, missing_prediction, resamples=10)
    with pytest.raises(ValueError, match="incompatible CER reference denominator"):
        paired_comparison.compare_payloads(payload, bad_denominator, resamples=10)


def _payload(
    rows: list[tuple[str, str, str]],
    *,
    dataset: str = "funsd",
    case_ids: list[str] | None = None,
) -> dict[str, object]:
    cases = []
    for index, (status, reference, prediction) in enumerate(rows):
        cases.append(
            {
                "id": case_ids[index] if case_ids else f"case-{index}",
                "reference": reference,
                "prediction": prediction,
                "status": status,
                "metrics": _metrics(prediction, reference),
            }
        )
    return {
        "dataset": dataset,
        "normalization": paired_comparison.NORMALIZATION,
        "cases": cases,
    }


def _metrics(prediction: str, reference: str) -> dict[str, object]:
    return paired_comparison._score(prediction, reference)
