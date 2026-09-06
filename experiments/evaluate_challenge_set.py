"""Evaluate a private OCR challenge set without exporting document content."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
import json
import math
from pathlib import Path
import re
from typing import Any
import unicodedata

from ocr_pipeline.verification import edit_counts, normalized_edit_distance


LATENCY_STEPS = (
    "prepare",
    "reader",
    "stage_view",
    "stage.tables",
    "stage.controls",
    "stage.handwriting",
    "stage.evidence-risk",
    "restore",
)
LEGIBILITY_LEVELS = ("complete", "partial", "illegible", "unknown")
HANDWRITING_LEVELS = ("legible", "partial", "illegible", "unknown")
CASE_CATEGORY = re.compile(r"^(C\d+)", re.IGNORECASE)
AGGREGATE_ROUTES = frozenset({"accept_local", "review"})
AGGREGATE_REGION_KINDS = frozenset(
    {
        "caption",
        "checkbox",
        "coverage_risk",
        "figure",
        "footnote",
        "formula",
        "heading",
        "list",
        "list_item",
        "page_footer",
        "page_header",
        "page_html",
        "page_markdown",
        "page_text",
        "paragraph",
        "picture",
        "radio",
        "section_header",
        "table",
        "table_candidate",
        "text",
        "title",
        "word",
    }
)
AGGREGATE_FAILURE_CODES = frozenset(
    {
        "control_dependency_unavailable",
        "control_image_failed",
        "document_failed",
        "invalid_batch_output",
        "invalid_dpi",
        "invalid_image",
        "invalid_orientation_angle",
        "invalid_orientation_box",
        "invalid_orientation_classifier_output",
        "invalid_osd_output",
        "invalid_reader_output",
        "invalid_table_output",
        "ministral_image_failed",
        "ministral_import_failed",
        "ministral_init_failed",
        "ministral_output_failed",
        "ministral_predict_failed",
        "missing_model_output",
        "no_pages",
        "no_text_detected",
        "orientation_classifier_failed",
        "orientation_classifier_image_failed",
        "orientation_classifier_unavailable",
        "orientation_image_failed",
        "orientation_stage_image_failed",
        "orientation_views_failed",
        "osd_failed",
        "osd_timeout",
        "osd_unavailable",
        "preprocess_failed",
        "reader_failed",
        "reader_timeout",
        "reader_unavailable",
        "renderer_failed",
        "renderer_timeout",
        "renderer_unavailable",
        "request_failed",
        "risk_image_failed",
        "source_not_found",
        "table_challenger_image_failed",
        "table_detection_failed",
        "table_image_failed",
        "table_model_unavailable",
        "table_preprocess_failed",
        "table_structure_failed",
        "unsupported_source",
    }
)
CONTROL_STATES = {
    "ambiguous": "ambiguous",
    "checked": "selected",
    "selected": "selected",
    "uncertain": "ambiguous",
    "unchecked": "unselected",
    "unselected": "unselected",
}
CONTROL_ANNOTATION_SCOPES = frozenset({"exhaustive", "selected_only"})
BOX_IOU_THRESHOLDS = (0.5, 0.7)
CONTROL_IOU_THRESHOLD = BOX_IOU_THRESHOLDS[0]
HANDWRITING_IOU_THRESHOLD = BOX_IOU_THRESHOLDS[0]


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = evaluate_challenge_set(args.annotations, args.model_output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return 0


def evaluate_challenge_set(
    annotation_root: Path,
    model_output_root: Path,
) -> dict[str, Any]:
    annotations = _load_records(annotation_root, annotations=True)
    outputs = _load_records(model_output_root, annotations=False)
    identities = sorted(set(annotations) | set(outputs))
    case_pairs = [(annotations.get(key), outputs.get(key)) for key in identities]
    annotation_pairs = [
        (annotations[key], outputs.get(key)) for key in sorted(annotations)
    ]
    annotated_outputs = [output for _, output in annotation_pairs if output is not None]

    return {
        "cases": _case_counts(case_pairs),
        "annotations": _annotation_counts(annotation_pairs),
        "transcription": _transcription_metrics(annotation_pairs),
        "coverage": _coverage_metrics(annotation_pairs),
        "regions": _region_counts(annotated_outputs),
        "tables": _table_metrics(annotation_pairs),
        "controls": _control_metrics(annotation_pairs),
        "handwriting": _handwriting_metrics(annotation_pairs),
        "latency_seconds": _latency_metrics(annotated_outputs),
    }


def _load_records(
    root: Path,
    *,
    annotations: bool,
) -> dict[tuple[str, str], dict[str, Any]]:
    if not root.is_dir():
        raise FileNotFoundError("Evaluation input root was not found")
    records: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted(root.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(
                "Evaluation input contains an invalid JSON record"
            ) from error
        if not isinstance(payload, dict):
            raise ValueError("Evaluation input contains a non-object JSON record")
        if annotations and payload.get("source_only") is not True:
            continue
        identity = _identity(payload, path)
        if identity in records:
            raise ValueError("Evaluation input contains a duplicate case identity")
        records[identity] = payload
    return records


def _identity(payload: Mapping[str, Any], path: Path) -> tuple[str, str]:
    case_id = payload.get("case_id")
    if not isinstance(case_id, str) or not case_id.strip():
        filename = payload.get("filename")
        case_id = Path(filename).stem if isinstance(filename, str) else path.stem
    case_id = case_id.strip()
    category = payload.get("category_id")
    if not isinstance(category, str) or not category.strip():
        match = CASE_CATEGORY.match(case_id)
        category = match.group(1) if match else ""
    return category.strip().upper(), case_id.casefold()


def _case_counts(
    pairs: Sequence[tuple[dict[str, Any] | None, dict[str, Any] | None]],
) -> dict[str, int]:
    return {
        "total": len(pairs),
        "annotated": sum(annotation is not None for annotation, _ in pairs),
        "model_outputs": sum(output is not None for _, output in pairs),
        "paired": sum(
            annotation is not None and output is not None
            for annotation, output in pairs
        ),
        "missing_model_output": sum(
            annotation is not None and output is None for annotation, output in pairs
        ),
        "unannotated_model_output": sum(
            annotation is None and output is not None for annotation, output in pairs
        ),
    }


def _annotation_counts(
    pairs: Sequence[tuple[dict[str, Any] | None, dict[str, Any] | None]],
) -> dict[str, Any]:
    annotations = [annotation for annotation, _ in pairs if annotation is not None]
    completeness = Counter(_page_legibility(annotation) for annotation in annotations)
    unresolved = [
        len(_as_list(_as_dict(annotation.get("transcription")).get("unresolved_spans")))
        for annotation in annotations
    ]
    return {
        "completeness": {
            level: completeness.get(level, 0) for level in LEGIBILITY_LEVELS
        },
        "unresolved_spans": {
            "pages": sum(count > 0 for count in unresolved),
            "total": sum(unresolved),
        },
    }


def _transcription_metrics(
    pairs: Sequence[tuple[dict[str, Any] | None, dict[str, Any] | None]],
) -> dict[str, Any]:
    character_insertions = 0
    character_deletions = 0
    character_substitutions = 0
    word_edits = 0
    reference_characters = 0
    reference_words = 0
    reference_tokens = 0
    prediction_tokens = 0
    found_tokens = 0
    added_tokens = 0
    distances: list[float] = []
    excluded = Counter()
    scored = 0

    for annotation, output in pairs:
        if annotation is None:
            continue
        legibility = _page_legibility(annotation)
        transcription = _as_dict(annotation.get("transcription"))
        unresolved = _as_list(transcription.get("unresolved_spans"))
        reference_value = transcription.get("reading_order_text")
        if legibility != "complete":
            excluded[legibility] += 1
            continue
        if unresolved:
            excluded["unresolved"] += 1
            continue
        if not isinstance(reference_value, str) or not reference_value.strip():
            excluded["missing_reference"] += 1
            continue

        reference = _normalize_text(reference_value)
        prediction = _normalize_text(_prediction_text(output))
        reference_counts = Counter(reference.split())
        prediction_counts = Counter(prediction.split())
        character_counts = edit_counts(prediction, reference)
        word_counts = edit_counts(prediction.split(), reference.split())
        character_insertions += character_counts.insertions
        character_deletions += character_counts.deletions
        character_substitutions += character_counts.substitutions
        word_edits += word_counts.edits
        reference_characters += len(reference)
        reference_words += len(reference.split())
        reference_tokens += reference_counts.total()
        prediction_tokens += prediction_counts.total()
        found_tokens += (reference_counts & prediction_counts).total()
        added_tokens += (prediction_counts - reference_counts).total()
        distances.append(normalized_edit_distance(prediction, reference))
        scored += 1

    character_edits = (
        character_insertions + character_deletions + character_substitutions
    )
    return {
        "prediction_policy": "evidence_text_with_table_cells_once",
        "scored_cases": scored,
        "excluded": {
            "partial": excluded.get("partial", 0),
            "illegible": excluded.get("illegible", 0),
            "unknown": excluded.get("unknown", 0),
            "unresolved": excluded.get("unresolved", 0),
            "missing_reference": excluded.get("missing_reference", 0),
        },
        "reference_characters": reference_characters,
        "reference_words": reference_words,
        "character_edits": {
            "insertions": character_insertions,
            "deletions": character_deletions,
            "substitutions": character_substitutions,
        },
        "cer": _rate(character_edits, reference_characters),
        "wer": _rate(word_edits, reference_words),
        "normalized_edit_distance_mean": _mean(distances),
        "missed_text_rate": _rate(character_deletions, reference_characters),
        "hallucinated_text_rate": _rate(character_insertions, reference_characters),
        "token_multiset": {
            "policy": "normalized_whitespace_token_multiset",
            "reference_tokens": reference_tokens,
            "prediction_tokens": prediction_tokens,
            "found_tokens": found_tokens,
            "added_tokens": added_tokens,
            "tokens_found": _rate(found_tokens, reference_tokens),
            "tokens_added": _rate(added_tokens, prediction_tokens),
        },
    }


def _coverage_metrics(
    pairs: Sequence[tuple[dict[str, Any] | None, dict[str, Any] | None]],
) -> dict[str, Any]:
    total = len(pairs)
    covered = 0
    abstained = 0
    failed = 0
    failure_codes: Counter[str] = Counter()
    for _, output in pairs:
        if not _output_succeeded(output):
            failed += 1
            for code in _failure_codes(output):
                failure_codes[code] += 1
            continue
        if _prediction_text(output).strip():
            covered += 1
        else:
            abstained += 1
    return {
        "covered": covered,
        "abstained": abstained,
        "failed": failed,
        "coverage_rate": _rate(covered, total),
        "abstention_rate": _rate(abstained, total),
        "failure_rate": _rate(failed, total),
        "failure_codes": dict(sorted(failure_codes.items())),
    }


def _region_counts(outputs: Iterable[dict[str, Any]]) -> dict[str, Any]:
    kinds: Counter[str] = Counter()
    routes: Counter[str] = Counter()
    for output in outputs:
        for page in _pages(output):
            route = page.get("route")
            if isinstance(route, str) and route:
                routes[_aggregate_label(route, AGGREGATE_ROUTES)] += 1
            for region in _as_list(page.get("regions")):
                kind = _as_dict(region).get("kind")
                if isinstance(kind, str) and kind:
                    kinds[_aggregate_label(kind, AGGREGATE_REGION_KINDS)] += 1
    return {
        "kind_counts": dict(sorted(kinds.items())),
        "route_counts": dict(sorted(routes.items())),
    }


def _table_metrics(
    pairs: Sequence[tuple[dict[str, Any] | None, dict[str, Any] | None]],
) -> dict[str, Any]:
    true_positive = false_positive = false_negative = true_negative = 0
    annotated_tables = predicted_tables = 0
    descriptor_pairs = missed_tables = extra_tables = 0
    row_eligible = row_pair_eligible = row_correct = 0
    column_eligible = column_pair_eligible = column_correct = 0
    row_errors: list[float] = []
    column_errors: list[float] = []

    for annotation, output in pairs:
        if annotation is None:
            continue
        reference = [_as_dict(item) for item in _as_list(annotation.get("tables"))]
        prediction = _predicted_tables(output)
        annotated_tables += len(reference)
        predicted_tables += len(prediction)
        if reference and prediction:
            true_positive += 1
        elif reference:
            false_negative += 1
        elif prediction:
            false_positive += 1
        else:
            true_negative += 1

        for expected in reference:
            row_eligible += isinstance(expected.get("row_count"), int)
            column_eligible += isinstance(expected.get("column_count"), int)

        descriptor_pairs += min(len(reference), len(prediction))
        missed_tables += max(0, len(reference) - len(prediction))
        extra_tables += max(0, len(prediction) - len(reference))
        for expected, actual in zip(reference, prediction):
            row = expected.get("row_count")
            predicted_row = actual.get("row_count")
            if isinstance(row, int):
                row_pair_eligible += 1
                row_correct += predicted_row == row
                if isinstance(predicted_row, int):
                    row_errors.append(abs(predicted_row - row))
            column = expected.get("column_count")
            predicted_column = actual.get("column_count")
            if isinstance(column, int):
                column_pair_eligible += 1
                column_correct += predicted_column == column
                if isinstance(predicted_column, int):
                    column_errors.append(abs(predicted_column - column))

    return {
        "annotated_count": annotated_tables,
        "predicted_count_on_annotated_pages": predicted_tables,
        "presence": {
            "true_positive_pages": true_positive,
            "false_positive_pages": false_positive,
            "false_negative_pages": false_negative,
            "true_negative_pages": true_negative,
            "precision": _rate(true_positive, true_positive + false_positive),
            "recall": _rate(true_positive, true_positive + false_negative),
            "f1": _f1(true_positive, false_positive, false_negative),
        },
        "descriptors": {
            "pairing": "annotation-and-prediction-reading-order",
            "spatial_matching_supported": False,
            "pairing_limitation": (
                "reference table boxes are unavailable; declared annotation and "
                "prediction reading order is used"
            ),
            "count_semantics": "annotation-declared-not-geometric",
            "paired_tables": descriptor_pairs,
            "missed_reference_tables": missed_tables,
            "extra_predicted_tables": extra_tables,
            "row_count_eligible": row_eligible,
            "row_count_accuracy": _rate(row_correct, row_eligible),
            "row_count_eligible_on_pairs": row_pair_eligible,
            "row_count_accuracy_on_pairs": _rate(row_correct, row_pair_eligible),
            "row_count_numeric_pairs": len(row_errors),
            "row_count_mae_on_numeric_pairs": _mean(row_errors),
            "column_count_eligible": column_eligible,
            "column_count_accuracy": _rate(column_correct, column_eligible),
            "column_count_eligible_on_pairs": column_pair_eligible,
            "column_count_accuracy_on_pairs": _rate(
                column_correct, column_pair_eligible
            ),
            "column_count_numeric_pairs": len(column_errors),
            "column_count_mae_on_numeric_pairs": _mean(column_errors),
        },
    }


def _control_metrics(
    pairs: Sequence[tuple[dict[str, Any] | None, dict[str, Any] | None]],
) -> dict[str, Any]:
    annotated_count = predicted_count = matched = state_correct = 0
    reference_states: list[str] = []
    predicted_states: list[str] = []
    for annotation, output in pairs:
        if annotation is None:
            continue
        reference = [_as_dict(item) for item in _as_list(annotation.get("controls"))]
        prediction = _predicted_controls(output)
        annotated_count += len(reference)
        predicted_count += len(prediction)
        reference_by_key = _unique_controls(reference)
        prediction_by_key = _unique_controls(prediction)
        for key in reference_by_key.keys() & prediction_by_key.keys():
            expected = reference_by_key[key]
            actual = prediction_by_key[key]
            expected_state = _control_state(expected.get("state"))
            actual_state = _control_state(actual.get("state"))
            if expected_state is None or actual_state is None:
                continue
            matched += 1
            state_correct += expected_state == actual_state
            reference_states.append(expected_state)
            predicted_states.append(actual_state)

    return {
        "annotated_count": annotated_count,
        "predicted_count_on_annotated_pages": predicted_count,
        "safely_matched": matched,
        "safe_match_coverage": _rate(matched, annotated_count),
        "state_accuracy_on_safe_matches": _rate(state_correct, matched),
        "state_macro_f1_on_safe_matches": _macro_f1(reference_states, predicted_states),
        "bbox": _control_bbox_metrics(pairs),
    }


def _control_bbox_metrics(
    pairs: Sequence[tuple[dict[str, Any] | None, dict[str, Any] | None]],
) -> dict[str, Any]:
    primary = _control_bbox_metrics_at_threshold(pairs, CONTROL_IOU_THRESHOLD)
    strict = _control_bbox_metrics_at_threshold(pairs, BOX_IOU_THRESHOLDS[1])
    return {
        "matching_policy": "hungarian_one_to_one_max_cardinality_then_iou",
        "iou_threshold": CONTROL_IOU_THRESHOLD,
        "iou_thresholds": list(BOX_IOU_THRESHOLDS),
        **primary,
        "iou_0_7": strict,
    }


def _control_bbox_metrics_at_threshold(
    pairs: Sequence[tuple[dict[str, Any] | None, dict[str, Any] | None]],
    iou_threshold: float,
) -> dict[str, Any]:
    exhaustive_pages = exhaustive_reference = exhaustive_prediction = 0
    bbox_matches = checked_matches = unchecked_matches = 0
    checked_reference = checked_prediction = unchecked_reference = 0
    state_eligible = state_correct = 0
    selected_only_pages = selected_only_reference = selected_only_matches = 0
    missing_reference_bbox = missing_prediction_bbox = 0

    for annotation, output in pairs:
        if annotation is None:
            continue
        scope = annotation.get("control_annotation_scope")
        if scope not in CONTROL_ANNOTATION_SCOPES:
            continue

        reference = [_as_dict(item) for item in _as_list(annotation.get("controls"))]
        prediction = _predicted_controls(output)
        localized_reference = [item for item in reference if _control_box(item)]
        localized_prediction = [item for item in prediction if _control_box(item)]
        missing_reference_bbox += len(reference) - len(localized_reference)
        missing_prediction_bbox += len(prediction) - len(localized_prediction)

        checked_refs = [
            item
            for item in localized_reference
            if _control_state(item.get("state")) == "selected"
        ]
        checked_preds = [
            item
            for item in localized_prediction
            if _control_state(item.get("state")) == "selected"
        ]
        checked_pairs = _match_controls(checked_refs, checked_preds, iou_threshold)

        if scope == "selected_only":
            selected_only_pages += 1
            selected_only_reference += len(checked_refs)
            selected_only_matches += len(checked_pairs)
            continue

        exhaustive_pages += 1
        exhaustive_reference += len(localized_reference)
        exhaustive_prediction += len(prediction)
        checked_reference += len(checked_refs)
        checked_prediction += sum(
            _control_state(item.get("state")) == "selected" for item in prediction
        )
        checked_matches += len(checked_pairs)

        unchecked_refs = [
            item
            for item in localized_reference
            if _control_state(item.get("state")) == "unselected"
        ]
        unchecked_preds = [
            item
            for item in localized_prediction
            if _control_state(item.get("state")) == "unselected"
        ]
        unchecked_reference += len(unchecked_refs)
        unchecked_matches += len(
            _match_controls(unchecked_refs, unchecked_preds, iou_threshold)
        )

        matches = _match_controls(
            localized_reference,
            localized_prediction,
            iou_threshold,
        )
        bbox_matches += len(matches)
        for reference_index, prediction_index in matches:
            expected = _control_state(localized_reference[reference_index].get("state"))
            actual = _control_state(localized_prediction[prediction_index].get("state"))
            if expected is None or actual is None:
                continue
            state_eligible += 1
            state_correct += expected == actual

    bbox_false_positive = exhaustive_prediction - bbox_matches
    bbox_false_negative = exhaustive_reference - bbox_matches
    checked_false_positive = checked_prediction - checked_matches
    checked_false_negative = checked_reference - checked_matches
    return {
        "missing_bounding_box": {
            "reference": missing_reference_bbox,
            "prediction": missing_prediction_bbox,
        },
        "length_penalty": {
            "policy": "negative_absolute_count_difference_over_reference",
            "value": _length_penalty(
                exhaustive_reference,
                exhaustive_prediction,
            ),
        },
        "exhaustive": {
            "pages": exhaustive_pages,
            "reference_count": exhaustive_reference,
            "predicted_count": exhaustive_prediction,
            "matched": bbox_matches,
            "precision": _rate(bbox_matches, bbox_matches + bbox_false_positive),
            "recall": _rate(bbox_matches, bbox_matches + bbox_false_negative),
            "f1": _f1(bbox_matches, bbox_false_positive, bbox_false_negative),
            "checked": {
                "reference_count": checked_reference,
                "predicted_count": checked_prediction,
                "matched": checked_matches,
                "precision": _rate(
                    checked_matches, checked_matches + checked_false_positive
                ),
                "recall": _rate(
                    checked_matches, checked_matches + checked_false_negative
                ),
                "f1": _f1(
                    checked_matches,
                    checked_false_positive,
                    checked_false_negative,
                ),
            },
            "unchecked_recall": {
                "reference_count": unchecked_reference,
                "matched": unchecked_matches,
                "recall": _rate(unchecked_matches, unchecked_reference),
            },
            "state_accuracy": {
                "eligible": state_eligible,
                "correct": state_correct,
                "accuracy": _rate(state_correct, state_eligible),
            },
        },
        "selected_only": {
            "pages": selected_only_pages,
            "checked_reference_count": selected_only_reference,
            "checked_matched": selected_only_matches,
            "checked_recall": _rate(selected_only_matches, selected_only_reference),
        },
    }


def _handwriting_metrics(
    pairs: Sequence[tuple[dict[str, Any] | None, dict[str, Any] | None]],
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    localized_eligible: Counter[str] = Counter()
    localized_recovered = {threshold: Counter() for threshold in BOX_IOU_THRESHOLDS}
    presence_eligible: Counter[str] = Counter()
    presence_recovered: Counter[str] = Counter()
    for annotation, output in pairs:
        if annotation is None:
            continue
        prediction = _normalize_text(_prediction_text(output))
        used_spans: list[tuple[int, int]] = []
        regions = _handwriting_regions(output)
        localized_reference = []
        for raw_item in _as_list(annotation.get("handwriting")):
            item = _as_dict(raw_item)
            level = item.get("legibility")
            level = level if level in HANDWRITING_LEVELS else "unknown"
            counts[level] += 1
            text = item.get("text")
            if level not in {"legible", "partial"}:
                continue
            if not isinstance(text, str) or not text.strip():
                continue
            phrase = _normalize_text(text)
            presence_eligible[level] += 1
            span = _unused_phrase_span(prediction, phrase, used_spans)
            if span is not None:
                used_spans.append(span)
                presence_recovered[level] += 1

            expected_box = _generic_box(item.get("bbox"))
            if expected_box is None:
                continue
            localized_eligible[level] += 1
            localized_reference.append((level, phrase, expected_box))

        for threshold in BOX_IOU_THRESHOLDS:
            for reference_index, _ in _match_handwriting(
                localized_reference,
                regions,
                threshold,
            ):
                level = localized_reference[reference_index][0]
                localized_recovered[threshold][level] += 1

    primary_recovery = localized_recovered[HANDWRITING_IOU_THRESHOLD]
    strict_recovery = localized_recovered[BOX_IOU_THRESHOLDS[1]]
    return {
        "annotation_counts": {
            level: counts.get(level, 0) for level in HANDWRITING_LEVELS
        },
        "legible_exact_recovery": {
            "eligible": localized_eligible.get("legible", 0),
            "recovered": primary_recovery.get("legible", 0),
            "rate": _rate(
                primary_recovery.get("legible", 0),
                localized_eligible.get("legible", 0),
            ),
        },
        "partial_exact_recovery": {
            "eligible": localized_eligible.get("partial", 0),
            "recovered": primary_recovery.get("partial", 0),
            "rate": _rate(
                primary_recovery.get("partial", 0),
                localized_eligible.get("partial", 0),
            ),
        },
        "localized_exact_recovery": {
            "iou_0_5": {
                "legible": _recovery_counts(
                    localized_eligible,
                    primary_recovery,
                    "legible",
                ),
                "partial": _recovery_counts(
                    localized_eligible,
                    primary_recovery,
                    "partial",
                ),
            },
            "iou_0_7": {
                "legible": _recovery_counts(
                    localized_eligible,
                    strict_recovery,
                    "legible",
                ),
                "partial": _recovery_counts(
                    localized_eligible,
                    strict_recovery,
                    "partial",
                ),
            },
        },
        "page_presence": {
            "legible": {
                "eligible": presence_eligible.get("legible", 0),
                "recovered": presence_recovered.get("legible", 0),
                "rate": _rate(
                    presence_recovered.get("legible", 0),
                    presence_eligible.get("legible", 0),
                ),
            },
            "partial": {
                "eligible": presence_eligible.get("partial", 0),
                "recovered": presence_recovered.get("partial", 0),
                "rate": _rate(
                    presence_recovered.get("partial", 0),
                    presence_eligible.get("partial", 0),
                ),
            },
        },
        "unlocalized_scorable": sum(presence_eligible.values())
        - sum(localized_eligible.values()),
        "matching_policy": ("hungarian_one_to_one_exact_text_max_cardinality_then_iou"),
        "iou_thresholds": list(BOX_IOU_THRESHOLDS),
        "length_penalty_supported": False,
        "length_penalty_limitation": (
            "the output schema does not identify every predicted handwriting region"
        ),
        "localized_edit_metrics_supported": False,
    }


def _match_handwriting(
    reference: Sequence[tuple[str, str, tuple[float, float, float, float]]],
    regions: Sequence[tuple[str, str, tuple[float, float, float, float]]],
    iou_threshold: float,
) -> list[tuple[int, int]]:
    def score(reference_index: int, prediction_index: int) -> float:
        _, phrase, expected_box = reference[reference_index]
        _, text, actual_box = regions[prediction_index]
        if _unused_phrase_span(text, phrase, []) is None:
            return 0.0
        return _box_iou(expected_box, actual_box)

    return _optimal_matches(
        len(reference),
        len(regions),
        score,
        iou_threshold,
    )


def _handwriting_regions(
    output: dict[str, Any] | None,
) -> list[tuple[str, str, tuple[float, float, float, float]]]:
    candidates = []
    for page_index, page in enumerate(_pages(output), start=1):
        page_regions = [_as_dict(region) for region in _as_list(page.get("regions"))]
        regions_by_id = {
            region["id"]: region
            for region in page_regions
            if isinstance(region.get("id"), str)
        }
        page_text = _as_dict(page.get("text"))
        evidence = page_text.get("evidence_ids")
        evidence_ids = (
            {value for value in evidence if isinstance(value, str)}
            if isinstance(evidence, list) and evidence
            else None
        )
        if evidence_ids is not None:
            expanded_ids = set()
            for region_id in evidence_ids:
                region = regions_by_id.get(region_id)
                structure = _as_dict(region.get("structure")) if region else {}
                children = structure.get("child_evidence_ids")
                if structure.get("role") == "layout_block" and isinstance(
                    children, list
                ):
                    child_ids = {
                        child_id
                        for child_id in children
                        if isinstance(child_id, str) and child_id in regions_by_id
                    }
                    expanded_ids.update(child_ids or {region_id})
                else:
                    expanded_ids.add(region_id)
            evidence_ids = expanded_ids
        for region_index, region in enumerate(page_regions):
            region_id = region.get("id")
            if evidence_ids is not None and region_id not in evidence_ids:
                continue
            if region.get("kind") == "table":
                for cell_index, raw_cell in enumerate(
                    _as_list(_as_dict(region.get("structure")).get("cells"))
                ):
                    cell = _as_dict(raw_cell)
                    _append_handwriting_region(
                        candidates,
                        cell.get("id")
                        or f"p{page_index}-r{region_index}-c{cell_index}",
                        cell.get("text"),
                        cell.get("bbox"),
                    )
                continue
            _append_handwriting_region(
                candidates,
                region_id or f"p{page_index}-r{region_index}",
                region.get("text"),
                region.get("bounding_box"),
            )
    return candidates


def _append_handwriting_region(
    candidates: list[tuple[str, str, tuple[float, float, float, float]]],
    region_id: object,
    text: object,
    box: object,
) -> None:
    actual_box = _generic_box(box)
    if not isinstance(region_id, str) or not isinstance(text, str):
        return
    normalized = _normalize_text(text)
    if not normalized or actual_box is None:
        return
    candidates.append((region_id, normalized, actual_box))


def _latency_metrics(outputs: Iterable[dict[str, Any]]) -> dict[str, Any]:
    totals: list[float] = []
    steps: defaultdict[str, list[float]] = defaultdict(list)
    invalid_total = 0
    invalid_steps: Counter[str] = Counter()
    for output in outputs:
        timing = _as_dict(output.get("timing"))
        total = timing.get("total_seconds", output.get("elapsed_seconds"))
        if _is_latency(total):
            totals.append(float(total))
        elif total is not None:
            invalid_total += 1
        pipeline_steps = _as_dict(timing.get("pipeline_steps"))
        for step in LATENCY_STEPS:
            if step not in pipeline_steps:
                continue
            value = pipeline_steps.get(step)
            if _is_latency(value):
                steps[step].append(float(value))
            else:
                invalid_steps[step] += 1
    return {
        "total": _distribution(totals),
        "pipeline_steps": {
            step: _distribution(steps[step]) for step in LATENCY_STEPS if steps[step]
        },
        "invalid": {
            "total": invalid_total,
            "pipeline_steps": dict(sorted(invalid_steps.items())),
        },
    }


def _prediction_text(output: dict[str, Any] | None) -> str:
    return "\n".join(_page_prediction_text(page) for page in _pages(output))


def _page_prediction_text(page: dict[str, Any]) -> str:
    page_text = _as_dict(page.get("text"))
    fallback = page_text.get("value")
    fallback = fallback if isinstance(fallback, str) else ""
    evidence_ids = page_text.get("evidence_ids")
    if not isinstance(evidence_ids, list) or not evidence_ids:
        return fallback

    regions = {
        region.get("id"): region
        for raw_region in _as_list(page.get("regions"))
        if isinstance((region := _as_dict(raw_region)).get("id"), str)
    }
    if any(region_id not in regions for region_id in evidence_ids):
        return fallback

    values = []
    for region_id in evidence_ids:
        region = regions[region_id]
        value = (
            _table_text(region) if region.get("kind") == "table" else region.get("text")
        )
        if isinstance(value, str) and value.strip():
            values.append(value)
    return "\n".join(values)


def _table_text(region: dict[str, Any]) -> str:
    cells = [
        _as_dict(cell)
        for cell in _as_list(_as_dict(region.get("structure")).get("cells"))
    ]
    cells.sort(
        key=lambda cell: (
            min(_numeric(value) for value in _as_list(cell.get("row_nums")))
            if _as_list(cell.get("row_nums"))
            else float("inf"),
            min(_numeric(value) for value in _as_list(cell.get("column_nums")))
            if _as_list(cell.get("column_nums"))
            else float("inf"),
        )
    )
    values = [
        value.strip()
        for cell in cells
        if isinstance((value := cell.get("text")), str) and value.strip()
    ]
    if values:
        return "\n".join(values)
    value = region.get("text")
    return value if isinstance(value, str) else ""


def _pages(output: dict[str, Any] | None) -> list[dict[str, Any]]:
    if output is None:
        return []
    result = _as_dict(output.get("result"))
    return [_as_dict(page) for page in _as_list(result.get("pages"))]


def _output_succeeded(output: dict[str, Any] | None) -> bool:
    if output is None or output.get("request_status") == "failed":
        return False
    return _as_dict(output.get("result")).get("status") == "success"


def _failure_codes(output: dict[str, Any] | None) -> list[str]:
    if output is None:
        return ["missing_model_output"]
    if output.get("request_status") == "failed":
        return ["request_failed"]
    failures = _as_list(_as_dict(output.get("result")).get("failures"))
    codes = [
        _aggregate_label(code, AGGREGATE_FAILURE_CODES)
        for failure in failures
        if isinstance((code := _as_dict(failure).get("code")), str) and code
    ]
    return codes or ["document_failed"]


def _predicted_tables(output: dict[str, Any] | None) -> list[dict[str, Any]]:
    regions = [
        _as_dict(region)
        for page in _pages(output)
        for region in _as_list(page.get("regions"))
        if _as_dict(region).get("kind") == "table"
    ]
    regions.sort(key=lambda region: _numeric(region.get("reading_order")))
    return [_as_dict(region.get("structure")) for region in regions]


def _predicted_controls(output: dict[str, Any] | None) -> list[dict[str, Any]]:
    controls = []
    for page in _pages(output):
        for raw_region in _as_list(page.get("regions")):
            region = _as_dict(raw_region)
            structure = _as_dict(region.get("structure"))
            control_type = structure.get("control_type", region.get("kind"))
            if control_type not in {"checkbox", "radio"}:
                continue
            controls.append(
                {
                    "kind": control_type,
                    "label": structure.get("label"),
                    "state": structure.get("state"),
                    "bounding_box": region.get("bounding_box"),
                }
            )
    return controls


def _match_controls(
    reference: Sequence[dict[str, Any]],
    prediction: Sequence[dict[str, Any]],
    iou_threshold: float,
) -> list[tuple[int, int]]:
    def score(reference_index: int, prediction_index: int) -> float:
        expected = reference[reference_index]
        actual = prediction[prediction_index]
        expected_kind = expected.get("kind")
        if expected_kind != actual.get("kind"):
            return 0.0
        return _control_iou(expected, actual)

    return _optimal_matches(
        len(reference),
        len(prediction),
        score,
        iou_threshold,
    )


def _optimal_matches(
    reference_count: int,
    prediction_count: int,
    score: Callable[[int, int], float],
    threshold: float,
) -> list[tuple[int, int]]:
    if not reference_count or not prediction_count:
        return []

    raw_scores = [
        [
            score(reference_index, prediction_index)
            for prediction_index in range(prediction_count)
        ]
        for reference_index in range(reference_count)
    ]
    match_bonus = min(reference_count, prediction_count) + 1.0
    assignment_scores = [
        [value + match_bonus if value >= threshold else 0.0 for value in row]
        for row in raw_scores
    ]
    assignment = _maximum_weight_assignment(assignment_scores)
    return [
        (reference_index, prediction_index)
        for reference_index, prediction_index in assignment
        if raw_scores[reference_index][prediction_index] >= threshold
    ]


def _maximum_weight_assignment(
    scores: Sequence[Sequence[float]],
) -> list[tuple[int, int]]:
    row_count = len(scores)
    column_count = len(scores[0])
    transposed = row_count > column_count
    costs = (
        [
            [-scores[row][column] for row in range(row_count)]
            for column in range(column_count)
        ]
        if transposed
        else [[-value for value in row] for row in scores]
    )

    matched_column = [0] * (len(costs[0]) + 1)
    previous_column = [0] * (len(costs[0]) + 1)
    row_potential = [0.0] * (len(costs) + 1)
    column_potential = [0.0] * (len(costs[0]) + 1)

    for row in range(1, len(costs) + 1):
        matched_column[0] = row
        minimum_cost = [math.inf] * (len(costs[0]) + 1)
        used_column = [False] * (len(costs[0]) + 1)
        column = 0
        while True:
            used_column[column] = True
            matched_row = matched_column[column]
            next_column = 0
            delta = math.inf
            for candidate in range(1, len(costs[0]) + 1):
                if used_column[candidate]:
                    continue
                reduced_cost = (
                    costs[matched_row - 1][candidate - 1]
                    - row_potential[matched_row]
                    - column_potential[candidate]
                )
                if reduced_cost < minimum_cost[candidate]:
                    minimum_cost[candidate] = reduced_cost
                    previous_column[candidate] = column
                if minimum_cost[candidate] < delta:
                    delta = minimum_cost[candidate]
                    next_column = candidate
            for candidate in range(len(costs[0]) + 1):
                if used_column[candidate]:
                    row_potential[matched_column[candidate]] += delta
                    column_potential[candidate] -= delta
                else:
                    minimum_cost[candidate] -= delta
            column = next_column
            if matched_column[column] == 0:
                break

        while True:
            previous = previous_column[column]
            matched_column[column] = matched_column[previous]
            column = previous
            if column == 0:
                break

    assignment = [
        (matched_row - 1, column - 1)
        for column, matched_row in enumerate(matched_column[1:], start=1)
        if matched_row
    ]
    if transposed:
        assignment = [(column, row) for row, column in assignment]
    return sorted(assignment)


def _recovery_counts(
    eligible: Counter[str],
    recovered: Counter[str],
    level: str,
) -> dict[str, int | float | None]:
    eligible_count = eligible.get(level, 0)
    recovered_count = recovered.get(level, 0)
    return {
        "eligible": eligible_count,
        "recovered": recovered_count,
        "rate": _rate(recovered_count, eligible_count),
    }


def _length_penalty(reference_count: int, prediction_count: int) -> float | None:
    if reference_count:
        return round(-abs(reference_count - prediction_count) / reference_count, 6)
    if prediction_count:
        return -1.0
    return None


def _control_iou(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> float:
    expected_box = _control_box(expected)
    actual_box = _control_box(actual)
    if expected_box is None or actual_box is None:
        return 0.0
    return _box_iou(expected_box, actual_box)


def _control_box(
    control: Mapping[str, Any],
) -> tuple[float, float, float, float] | None:
    return _generic_box(control.get("bounding_box"))


def _generic_box(value: object) -> tuple[float, float, float, float] | None:
    if isinstance(value, Mapping):
        values = tuple(value.get(key) for key in ("left", "top", "right", "bottom"))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = tuple(value)
    else:
        return None
    if len(values) != 4:
        return None
    if not all(_is_number(value) and math.isfinite(value) for value in values):
        return None
    left, top, right, bottom = values
    if right <= left or bottom <= top:
        return None
    return float(left), float(top), float(right), float(bottom)


def _box_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / (first_area + second_area - intersection)


def _unique_controls(
    controls: Sequence[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    grouped: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for control in controls:
        kind = control.get("kind")
        label = control.get("label")
        if not isinstance(kind, str) or not isinstance(label, str):
            continue
        normalized_label = _normalize_text(label)
        if normalized_label:
            grouped[kind.casefold(), normalized_label].append(control)
    return {key: items[0] for key, items in grouped.items() if len(items) == 1}


def _page_legibility(annotation: Mapping[str, Any]) -> str:
    value = annotation.get("page_legibility")
    return value if value in LEGIBILITY_LEVELS else "unknown"


def _aggregate_label(value: str, allowlist: frozenset[str]) -> str:
    return value if value in allowlist else "other"


def _control_state(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    return CONTROL_STATES.get(value.strip().casefold())


def _unused_phrase_span(
    prediction: str,
    phrase: str,
    used_spans: Sequence[tuple[int, int]],
) -> tuple[int, int] | None:
    pattern = re.compile(rf"(?<!\w){re.escape(phrase)}(?!\w)")
    for match in pattern.finditer(prediction):
        span = match.span()
        if all(span[1] <= used[0] or span[0] >= used[1] for used in used_spans):
            return span
    return None


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(normalized.split())


def _macro_f1(reference: Sequence[str], prediction: Sequence[str]) -> float | None:
    if not reference:
        return None
    labels = set(reference) | set(prediction)
    scores = []
    for label in labels:
        true_positive = sum(
            expected == label and actual == label
            for expected, actual in zip(reference, prediction, strict=True)
        )
        false_positive = sum(
            expected != label and actual == label
            for expected, actual in zip(reference, prediction, strict=True)
        )
        false_negative = sum(
            expected == label and actual != label
            for expected, actual in zip(reference, prediction, strict=True)
        )
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(2 * true_positive / denominator)
    return _mean(scores)


def _f1(true_positive: int, false_positive: int, false_negative: int) -> float | None:
    denominator = 2 * true_positive + false_positive + false_negative
    return _rate(2 * true_positive, denominator)


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "p50": _percentile(values, 0.5),
        "p95": _percentile(values, 0.95),
        "max": round(max(values), 6) if values else None,
    }


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction, 6)


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _mean(values: Sequence[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def _numeric(value: object) -> float:
    return float(value) if _is_number(value) else float("inf")


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _is_latency(value: object) -> bool:
    return _is_number(value) and math.isfinite(value) and value >= 0


def _as_dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: object) -> list[Any]:
    return value if isinstance(value, list) else []


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("model_output", type=Path)
    parser.add_argument("output", type=Path)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
