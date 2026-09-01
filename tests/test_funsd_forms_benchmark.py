from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments import funsd_forms_benchmark as forms


def test_oracle_and_empty_controls_score_all_50_pages_end_to_end(
    tmp_path: Path,
) -> None:
    dataset_root = _dataset(tmp_path)
    oracle_path = _write_json(tmp_path / "oracle.json", _predictions(_oracle_cases(50)))
    oracle_output = tmp_path / "oracle-result.json"

    assert forms.main([str(dataset_root), str(oracle_path), str(oracle_output)]) == 0

    oracle = json.loads(oracle_output.read_text(encoding="utf-8"))
    assert oracle["dataset"] == {
        "id": "funsd",
        "revision": "original",
        "role": "test",
        "cases": 50,
        "source": "official original FUNSD testing_data/annotations",
        "evaluation_use": "public research benchmark",
    }
    assert oracle["summary"]["cases"] == 50
    assert oracle["summary"]["supplied_cases"] == 50
    assert oracle["summary"]["status_counts"] == {"success": 50}
    assert oracle["metrics"]["field_exact_match"] == {
        "matched_fields": 50,
        "reference_fields": 50,
        "predicted_fields": 50,
        "accuracy": 1.0,
    }
    assert oracle["metrics"]["normalized_value_accuracy"]["accuracy"] == 1.0
    relation = oracle["metrics"]["key_value_relation"]
    assert relation["precision"] == relation["recall"] == relation["f1"] == 1.0
    assert relation["precision_denominator"] == 50
    assert relation["recall_denominator"] == 50

    empty_path = _write_json(tmp_path / "empty.json", _predictions([]))
    empty_output = tmp_path / "empty-result.json"
    assert forms.main([str(dataset_root), str(empty_path), str(empty_output)]) == 0

    empty = json.loads(empty_output.read_text(encoding="utf-8"))
    assert empty["summary"]["missing_cases"] == 50
    assert empty["summary"]["status_counts"] == {"failed": 50}
    assert empty["summary"]["failure_codes"] == {"missing_prediction": 50}
    assert empty["metrics"]["field_exact_match"]["accuracy"] == 0.0
    assert empty["metrics"]["normalized_value_accuracy"]["accuracy"] == 0.0
    assert empty["metrics"]["key_value_relation"] == {
        "reference_relations": 50,
        "predicted_relations": 0,
        "true_positives": 0,
        "false_positives": 0,
        "false_negatives": 50,
        "precision_denominator": 0,
        "recall_denominator": 50,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }

    with pytest.raises(SystemExit):
        forms.main([str(dataset_root), str(oracle_path), str(oracle_output)])


def test_metrics_separate_exact_text_value_normalization_and_relation(
    tmp_path: Path,
) -> None:
    dataset_root = _dataset(tmp_path)
    predictions = _oracle_cases(50)
    predictions[0]["entities"][0]["text"] = " Field 0 "
    predictions[0]["entities"][1]["text"] = "  VALUE   0 "
    predictions[1]["entities"][1]["text"] = "wrong"
    predictions[2]["relations"] = []

    result = forms.run_benchmark(
        dataset_root,
        _write_json(tmp_path / "predictions.json", _predictions(predictions)),
    )

    assert result["metrics"]["field_exact_match"]["matched_fields"] == 47
    assert result["metrics"]["normalized_value_accuracy"]["matched_values"] == 48
    relation = result["metrics"]["key_value_relation"]
    assert relation["true_positives"] == 49
    assert relation["false_negatives"] == 1
    assert relation["false_positives"] == 0
    assert relation["f1"] == 0.989899


def test_rejects_unknown_and_duplicate_case_or_entity_ids(tmp_path: Path) -> None:
    dataset_root = _dataset(tmp_path)
    unknown = _oracle_cases(1)
    unknown[0]["id"] = "test/not-in-panel"
    with pytest.raises(ValueError, match="outside the FUNSD test panel"):
        forms.run_benchmark(
            dataset_root,
            _write_json(tmp_path / "unknown.json", _predictions(unknown)),
        )

    duplicate_cases = _oracle_cases(1) * 2
    with pytest.raises(ValueError, match="Duplicate prediction case ID"):
        forms.run_benchmark(
            dataset_root,
            _write_json(
                tmp_path / "duplicate-cases.json", _predictions(duplicate_cases)
            ),
        )

    duplicate_entity = _oracle_cases(1)
    duplicate_entity[0]["entities"][1]["id"] = "q"
    with pytest.raises(ValueError, match="Duplicate entity ID"):
        forms.run_benchmark(
            dataset_root,
            _write_json(
                tmp_path / "duplicate-entity.json", _predictions(duplicate_entity)
            ),
        )


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("dataset", "other", "Prediction dataset must be funsd"),
        ("dataset_revision", "new", "dataset revision must be original"),
        ("dataset_role", "train", "dataset role must be test"),
    ],
)
def test_prediction_metadata_is_pinned(
    tmp_path: Path, key: str, value: str, message: str
) -> None:
    payload = _predictions([])
    payload[key] = value

    with pytest.raises(ValueError, match=message):
        forms.run_benchmark(
            _dataset(tmp_path),
            _write_json(tmp_path / f"{key}.json", payload),
        )


def _dataset(root: Path) -> Path:
    annotations = root / "testing_data" / "annotations"
    annotations.mkdir(parents=True)
    for index in range(50):
        question_id = index * 4
        answer_id = question_id + 1
        annotation = {
            "form": [
                {
                    "id": question_id,
                    "text": f"Field {index}",
                    "label": "question",
                    "box": [0, 0, 40, 10],
                    "words": [],
                    "linking": [[question_id, answer_id]],
                },
                {
                    "id": answer_id,
                    "text": f"Value {index}",
                    "label": "answer",
                    "box": [50, 0, 100, 10],
                    "words": [],
                    "linking": [[answer_id, question_id]],
                },
                {
                    "id": question_id + 2,
                    "text": "Section",
                    "label": "header",
                    "box": [0, 20, 40, 30],
                    "words": [],
                    "linking": [[question_id + 2, question_id]],
                },
                {
                    "id": question_id + 3,
                    "text": "Noise",
                    "label": "other",
                    "box": [50, 20, 100, 30],
                    "words": [],
                    "linking": [],
                },
            ]
        }
        _write_json(annotations / f"case-{index:02d}.json", annotation)
    return root


def _oracle_cases(count: int) -> list[dict[str, object]]:
    cases = []
    for index in range(count):
        cases.append(
            {
                "id": f"test/case-{index:02d}",
                "status": "success",
                "latency_ms": index + 0.5,
                "entities": [
                    {
                        "id": "q",
                        "text": f"Field {index}",
                        "label": "question",
                        "box": [0, 0, 40, 10],
                    },
                    {
                        "id": "a",
                        "text": f"Value {index}",
                        "label": "answer",
                        "box": [50, 0, 100, 10],
                    },
                ],
                "relations": [["q", "a"]],
                "failures": [],
            }
        )
    return cases


def _predictions(cases: list[dict[str, object]]) -> dict[str, object]:
    return {
        "dataset": "funsd",
        "dataset_revision": "original",
        "dataset_role": "test",
        "cases": cases,
    }


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path
