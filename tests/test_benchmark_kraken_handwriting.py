from __future__ import annotations

import json
from types import SimpleNamespace

from experiments import benchmark_kraken_handwriting as benchmark


class FakeImage:
    size = (3072, 4080)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def convert(self, _mode):
        return self


class FakeSegmentation:
    def __init__(self, **values):
        self.__dict__.update(values)

    def to_baselines(self):
        self.type = "baselines"
        return self


class FakeModel:
    seg_type = "bbox"

    def __init__(self, references):
        self.references = references

    def predict(self, _image, segmentation, _config):
        for line in segmentation.lines:
            prediction = self.references[line.id]
            yield SimpleNamespace(
                prediction=prediction,
                confidences=[0.95] * len(prediction),
            )


def test_public_benchmark_runs_end_to_end_with_current_task_contract(tmp_path):
    for page in {case.page for case in benchmark.PUBLIC_LINES}:
        (tmp_path / page).touch()
    output = tmp_path / "result.json"
    references = {case.line_id: case.reference for case in benchmark.PUBLIC_LINES}
    runtime = benchmark.KrakenRuntime(
        model=FakeModel(references),
        config=object(),
        box_line=lambda **values: SimpleNamespace(**values),
        segmentation=FakeSegmentation,
        open_image=lambda _path: FakeImage(),
    )
    elapsed = 0.0

    def clock():
        nonlocal elapsed
        elapsed += 0.01
        return elapsed

    result = benchmark.run_benchmark(
        tmp_path,
        tmp_path / "medium.safetensors",
        output,
        runtime=runtime,
        clock=clock,
    )

    assert json.loads(output.read_text(encoding="utf-8")) == result
    assert result["status"] == "complete"
    assert result["metrics"]["line_count"] == 16
    assert result["metrics"]["strict_exact_rate"] == 1.0
    assert result["metrics"]["cer"] == 0.0
    assert result["confidence_calibration"]["ece_10_bin"] == 0.05
    assert result["localization"]["source"] == "manual_public_reference_boxes"
    assert result["decision"]["verdict"] == "review_only"
    assert result["decision"]["production_wiring_allowed"] is False


def test_character_confidence_alignment_counts_substitution_and_deletion():
    correctness, deletions = benchmark.prediction_correctness("cat", "cut!")

    assert correctness == [True, False, True]
    assert deletions == 1


def test_score_rows_keeps_empty_and_wrong_lines_in_denominator():
    rows = [
        {
            "reference": "abc",
            "strict_exact": True,
            "empty_output": False,
            "character_edits": {"insertions": 0, "deletions": 0, "substitutions": 0},
            "word_edits": {"insertions": 0, "deletions": 0, "substitutions": 0},
            "confidence_length_matches_prediction": True,
        },
        {
            "reference": "def",
            "strict_exact": False,
            "empty_output": True,
            "character_edits": {"insertions": 0, "deletions": 3, "substitutions": 0},
            "word_edits": {"insertions": 0, "deletions": 1, "substitutions": 0},
            "confidence_length_matches_prediction": True,
        },
    ]

    result = benchmark.score_rows(rows)

    assert result["strict_exact_rate"] == 0.5
    assert result["cer"] == 0.5
    assert result["wer"] == 0.5
    assert result["empty_output_count"] == 1
