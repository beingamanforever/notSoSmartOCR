"""Score precomputed structured predictions on the official FUNSD test split."""

from __future__ import annotations

import argparse
import json
import math
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

DATASET_ID = "funsd"
DATASET_REVISION = "original"
DATASET_ROLE = "test"
EXPECTED_CASES = 50
ENTITY_LABELS = {"question", "answer", "header", "other"}
STATUS = {"success", "partial", "failed", "abstained"}
CASE_KEYS = {"id", "status", "latency_ms", "entities", "relations", "failures"}
ENTITY_KEYS = {"id", "text", "label", "box"}
IOU_THRESHOLD = 0.5


@dataclass(frozen=True)
class Entity:
    id: str
    text: str
    label: str
    box: tuple[float, float, float, float]


@dataclass(frozen=True)
class CaseData:
    id: str
    entities: tuple[Entity, ...]
    relations: tuple[tuple[str, str], ...]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score structured form predictions on all 50 FUNSD test pages"
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)

    try:
        if args.output.exists():
            raise ValueError(f"Output already exists: {args.output}")
        result = run_benchmark(args.dataset_root, args.predictions)
        _write_new_json(args.output, result)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    return 0


def run_benchmark(dataset_root: Path, predictions_path: Path) -> dict[str, object]:
    references = _load_references(dataset_root)
    metadata, prediction_rows = _load_predictions_file(predictions_path)
    _validate_metadata(metadata)
    supplied = _load_prediction_rows(prediction_rows)
    unknown_ids = set(supplied) - set(references)
    if unknown_ids:
        raise ValueError(
            f"Predictions contain IDs outside the FUNSD test panel: "
            f"{sorted(unknown_ids)[:3]}"
        )

    case_results = []
    status_counts: Counter[str] = Counter()
    failure_codes: Counter[str] = Counter()
    for case_id, reference in references.items():
        prediction = supplied.get(case_id) or _missing_prediction(case_id)
        result = _score_case(reference, prediction)
        case_results.append(result)
        status_counts[str(result["status"])] += 1
        failure_codes.update(
            str(item["code"])
            for item in result["failures"]  # type: ignore[index]
        )

    totals = _sum_counts(case_results)
    return {
        "benchmark": "FUNSD structured forms",
        "status": "complete",
        "dataset": {
            "id": DATASET_ID,
            "revision": DATASET_REVISION,
            "role": DATASET_ROLE,
            "cases": EXPECTED_CASES,
            "source": "official original FUNSD testing_data/annotations",
            "evaluation_use": "public research benchmark",
        },
        "prediction_protocol": "precomputed_only_no_inference_in_evaluator",
        "matching_policy": {
            "relation_scope": (
                "Only annotated question-answer links are key-value relations; "
                "header and other relation types are outside these metrics"
            ),
            "entity_matching": (
                "Same explicit entity label and box IoU >= 0.5, matched one-to-one "
                "by globally descending IoU with stable ID tie breaks"
            ),
            "field_exact_match": (
                "A reference key-value link counts only when the linked entities "
                "match and both predicted texts equal reference texts after Unicode "
                "NFC; the denominator is every reference key-value link"
            ),
            "normalized_value_accuracy": (
                "A reference link counts when the relation matches and the answer "
                "texts match after Unicode NFKC, case folding, and whitespace "
                "collapse; the denominator is every reference key-value link"
            ),
            "key_value_relation": (
                "A predicted question-answer link is correct when both endpoints "
                "map to the endpoints of one reference link; unmatched and extra "
                "links are false positives and missing links are false negatives"
            ),
        },
        "case_ids": list(references),
        "summary": {
            "cases": EXPECTED_CASES,
            "supplied_cases": len(supplied),
            "missing_cases": EXPECTED_CASES - len(supplied),
            "status_counts": dict(sorted(status_counts.items())),
            "failure_codes": dict(sorted(failure_codes.items())),
            "reference_entities": totals["reference_entities"],
            "predicted_entities": totals["predicted_entities"],
            "matched_entities": totals["matched_entities"],
            "ignored_reference_non_key_value_relations": totals[
                "ignored_reference_relations"
            ],
            "ignored_predicted_non_key_value_relations": totals[
                "ignored_predicted_relations"
            ],
        },
        "metrics": _metrics(totals),
        "cases": case_results,
    }


def _load_references(root: Path) -> dict[str, CaseData]:
    annotations = root / "testing_data" / "annotations"
    if not annotations.is_dir():
        raise ValueError(f"FUNSD test annotations not found: {annotations}")
    paths = sorted(annotations.glob("*.json"))
    if len(paths) != EXPECTED_CASES:
        raise ValueError(
            f"FUNSD original test role must contain all {EXPECTED_CASES} annotations"
        )

    references: dict[str, CaseData] = {}
    for path in paths:
        case_id = f"test/{path.stem}"
        if case_id in references:
            raise ValueError(f"Duplicate FUNSD case ID: {case_id}")
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        references[case_id] = _reference_case(case_id, payload)
    return references


def _reference_case(case_id: str, payload: object) -> CaseData:
    if not isinstance(payload, dict) or not isinstance(payload.get("form"), list):
        raise ValueError(f"Invalid FUNSD annotation: {case_id}")
    entities = tuple(
        _entity(item, f"{case_id}/form/{index}", allow_reference_fields=True)
        for index, item in enumerate(payload["form"])
    )
    by_id = _entities_by_id(entities, case_id)
    relation_pairs: set[tuple[str, str]] = set()
    for index, item in enumerate(payload["form"]):
        assert isinstance(item, dict)
        links = item.get("linking")
        if not isinstance(links, list):
            raise ValueError(f"Invalid FUNSD links: {case_id}/form/{index}")
        for link in links:
            left, right = _relation(link, by_id, f"{case_id}/form/{index}")
            relation_pairs.add(tuple(sorted((left, right))))
    return CaseData(case_id, entities, tuple(sorted(relation_pairs)))


def _load_predictions_file(
    path: Path,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    if not path.is_file():
        raise ValueError(f"Predictions not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("cases"), list):
        raise ValueError("Prediction JSON must contain a cases list")
    rows = payload["cases"]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("Prediction cases must be objects")
    return (
        {key: value for key, value in payload.items() if key != "cases"},
        rows,
    )


def _validate_metadata(metadata: dict[str, object]) -> None:
    if metadata.get("dataset") != DATASET_ID:
        raise ValueError(f"Prediction dataset must be {DATASET_ID}")
    if metadata.get("dataset_revision") != DATASET_REVISION:
        raise ValueError(f"Prediction dataset revision must be {DATASET_REVISION}")
    if metadata.get("dataset_role") != DATASET_ROLE:
        raise ValueError(f"Prediction dataset role must be {DATASET_ROLE}")


def _load_prediction_rows(
    rows: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    predictions = {}
    for row in rows:
        case_id = _nonempty_text(row.get("id"), "Prediction case id")
        if case_id in predictions:
            raise ValueError(f"Duplicate prediction case ID: {case_id}")
        unknown_keys = set(row) - CASE_KEYS
        if unknown_keys:
            raise ValueError(
                f"Prediction {case_id} has unsupported fields: {sorted(unknown_keys)}"
            )
        status = row.get("status")
        if status not in STATUS:
            raise ValueError(f"Prediction {case_id} has invalid status")
        entities_value = row.get("entities")
        relations_value = row.get("relations")
        failures = row.get("failures")
        if not isinstance(entities_value, list) or not isinstance(
            relations_value, list
        ):
            raise ValueError(f"Prediction {case_id} has invalid structured output")
        if not isinstance(failures, list) or any(
            not isinstance(item, dict) or not isinstance(item.get("code"), str)
            for item in failures
        ):
            raise ValueError(f"Prediction {case_id} has invalid failures")
        if status in {"partial", "failed", "abstained"} and not failures:
            raise ValueError(f"Prediction {case_id} must explain its {status} status")
        if status in {"failed", "abstained"} and (entities_value or relations_value):
            raise ValueError(f"Prediction {case_id} with {status} status must be empty")
        latency = _latency(row.get("latency_ms"), case_id)
        entities = tuple(
            _entity(item, f"{case_id}/entities/{index}")
            for index, item in enumerate(entities_value)
        )
        by_id = _entities_by_id(entities, case_id)
        relations = []
        seen_relations = set()
        for index, item in enumerate(relations_value):
            relation = _relation(item, by_id, f"{case_id}/relations/{index}")
            canonical = tuple(sorted(relation))
            if canonical in seen_relations:
                raise ValueError(f"Duplicate relation in prediction {case_id}")
            seen_relations.add(canonical)
            relations.append(relation)
        predictions[case_id] = {
            "id": case_id,
            "status": status,
            "latency_ms": latency,
            "entities": entities,
            "relations": tuple(relations),
            "failures": failures,
        }
    return predictions


def _score_case(
    reference: CaseData, prediction: dict[str, object]
) -> dict[str, object]:
    predicted_entities = prediction["entities"]
    predicted_relations = prediction["relations"]
    assert isinstance(predicted_entities, tuple)
    assert isinstance(predicted_relations, tuple)
    matches = _match_entities(predicted_entities, reference.entities)
    predicted_by_id = {entity.id: entity for entity in predicted_entities}
    reference_by_id = {entity.id: entity for entity in reference.entities}
    reference_fields, ignored_reference = _key_value_relations(
        reference.relations, reference_by_id
    )
    predicted_fields, ignored_predicted = _key_value_relations(
        predicted_relations, predicted_by_id
    )

    reference_set = set(reference_fields)
    relation_tp = 0
    field_exact = 0
    normalized_values = 0
    for question_id, answer_id in predicted_fields:
        mapped = (matches.get(question_id), matches.get(answer_id))
        if None in mapped or mapped not in reference_set:
            continue
        relation_tp += 1
        reference_question = reference_by_id[mapped[0]]  # type: ignore[index]
        reference_answer = reference_by_id[mapped[1]]  # type: ignore[index]
        predicted_question = predicted_by_id[question_id]
        predicted_answer = predicted_by_id[answer_id]
        if _exact_text(predicted_question.text) == _exact_text(
            reference_question.text
        ) and _exact_text(predicted_answer.text) == _exact_text(reference_answer.text):
            field_exact += 1
        if _normalize_value(predicted_answer.text) == _normalize_value(
            reference_answer.text
        ):
            normalized_values += 1

    counts = {
        "reference_entities": len(reference.entities),
        "predicted_entities": len(predicted_entities),
        "matched_entities": len(matches),
        "reference_fields": len(reference_fields),
        "predicted_fields": len(predicted_fields),
        "field_exact": field_exact,
        "normalized_values": normalized_values,
        "relation_tp": relation_tp,
        "relation_fp": len(predicted_fields) - relation_tp,
        "relation_fn": len(reference_fields) - relation_tp,
        "ignored_reference_relations": ignored_reference,
        "ignored_predicted_relations": ignored_predicted,
    }
    return {
        "id": reference.id,
        "status": prediction["status"],
        "latency_ms": prediction["latency_ms"],
        "failures": prediction["failures"],
        "counts": counts,
        "metrics": _metrics(counts),
    }


def _match_entities(
    predictions: tuple[Entity, ...], references: tuple[Entity, ...]
) -> dict[str, str]:
    candidates = []
    for prediction in predictions:
        for reference in references:
            if prediction.label != reference.label:
                continue
            overlap = _box_iou(prediction.box, reference.box)
            if overlap >= IOU_THRESHOLD:
                candidates.append((-overlap, prediction.id, reference.id))
    matches = {}
    used_references = set()
    for _, prediction_id, reference_id in sorted(candidates):
        if prediction_id in matches or reference_id in used_references:
            continue
        matches[prediction_id] = reference_id
        used_references.add(reference_id)
    return matches


def _key_value_relations(
    relations: tuple[tuple[str, str], ...], entities: dict[str, Entity]
) -> tuple[tuple[tuple[str, str], ...], int]:
    fields = set()
    ignored = 0
    for left_id, right_id in relations:
        left = entities[left_id]
        right = entities[right_id]
        if left.label == "question" and right.label == "answer":
            fields.add((left_id, right_id))
        elif left.label == "answer" and right.label == "question":
            fields.add((right_id, left_id))
        else:
            ignored += 1
    return tuple(sorted(fields)), ignored


def _metrics(counts: dict[str, int]) -> dict[str, object]:
    reference_fields = counts["reference_fields"]
    relation = _prf(counts["relation_tp"], counts["relation_fp"], counts["relation_fn"])
    return {
        "field_exact_match": {
            "matched_fields": counts["field_exact"],
            "reference_fields": reference_fields,
            "predicted_fields": counts["predicted_fields"],
            "accuracy": _ratio(counts["field_exact"], reference_fields),
        },
        "normalized_value_accuracy": {
            "matched_values": counts["normalized_values"],
            "reference_values": reference_fields,
            "accuracy": _ratio(counts["normalized_values"], reference_fields),
        },
        "key_value_relation": {
            "reference_relations": reference_fields,
            "predicted_relations": counts["predicted_fields"],
            **relation,
        },
    }


def _sum_counts(cases: list[dict[str, object]]) -> dict[str, int]:
    names = (
        "reference_entities",
        "predicted_entities",
        "matched_entities",
        "reference_fields",
        "predicted_fields",
        "field_exact",
        "normalized_values",
        "relation_tp",
        "relation_fp",
        "relation_fn",
        "ignored_reference_relations",
        "ignored_predicted_relations",
    )
    return {
        name: sum(int(case["counts"][name]) for case in cases)  # type: ignore[index]
        for name in names
    }


def _prf(
    true_positive: int, false_positive: int, false_negative: int
) -> dict[str, object]:
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = true_positive / precision_denominator if precision_denominator else 0.0
    recall = true_positive / recall_denominator if recall_denominator else 0.0
    f1 = (
        0.0
        if precision + recall == 0
        else 2 * precision * recall / (precision + recall)
    )
    return {
        "true_positives": true_positive,
        "false_positives": false_positive,
        "false_negatives": false_negative,
        "precision_denominator": precision_denominator,
        "recall_denominator": recall_denominator,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 6)


def _entity(
    value: object, context: str, *, allow_reference_fields: bool = False
) -> Entity:
    if not isinstance(value, dict):
        raise ValueError(f"Entity must be an object: {context}")
    allowed_keys = ENTITY_KEYS | (
        {"words", "linking"} if allow_reference_fields else set()
    )
    if set(value) - allowed_keys:
        raise ValueError(f"Entity has unsupported fields: {context}")
    entity_id = _nonempty_text(value.get("id"), f"Entity id: {context}")
    text = value.get("text")
    label = value.get("label")
    if not isinstance(text, str):
        raise ValueError(f"Entity text must be a string: {context}")
    if label not in ENTITY_LABELS:
        raise ValueError(f"Entity label is invalid: {context}")
    return Entity(entity_id, text, label, _box(value.get("box"), context))


def _entities_by_id(entities: tuple[Entity, ...], context: str) -> dict[str, Entity]:
    result = {}
    for entity in entities:
        if entity.id in result:
            raise ValueError(f"Duplicate entity ID in {context}: {entity.id}")
        result[entity.id] = entity
    return result


def _relation(
    value: object, entities: dict[str, Entity], context: str
) -> tuple[str, str]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"Relation must contain two entity IDs: {context}")
    left = _nonempty_text(value[0], f"Relation endpoint: {context}")
    right = _nonempty_text(value[1], f"Relation endpoint: {context}")
    if left == right or left not in entities or right not in entities:
        raise ValueError(
            f"Relation references unknown or identical entities: {context}"
        )
    return left, right


def _box(value: object, context: str) -> tuple[float, float, float, float]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"Entity box must contain four numbers: {context}")
    if any(
        isinstance(item, bool)
        or not isinstance(item, (int, float))
        or not math.isfinite(float(item))
        for item in value
    ):
        raise ValueError(f"Entity box contains invalid coordinates: {context}")
    left, top, right, bottom = (float(item) for item in value)
    if right <= left or bottom <= top:
        raise ValueError(f"Entity box must have positive area: {context}")
    return left, top, right, bottom


def _box_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = width * height
    union = (
        (left[2] - left[0]) * (left[3] - left[1])
        + (right[2] - right[0]) * (right[3] - right[1])
        - intersection
    )
    return 0.0 if union <= 0 else intersection / union


def _exact_text(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def _normalize_value(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _nonempty_text(value: object, context: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(f"{context} must be a string or integer")
    text = str(value)
    if not text:
        raise ValueError(f"{context} must not be empty")
    return text


def _latency(value: object, case_id: str) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise ValueError(f"Prediction {case_id} has invalid latency_ms")
    return round(float(value), 3)


def _missing_prediction(case_id: str) -> dict[str, object]:
    return {
        "id": case_id,
        "status": "failed",
        "latency_ms": None,
        "entities": (),
        "relations": (),
        "failures": [
            {
                "code": "missing_prediction",
                "stage": "structured_extraction",
                "message": "No prediction was supplied for this official test page",
            }
        ],
    }


def _write_new_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        json.dump(payload, output, ensure_ascii=False, indent=2)
        output.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
