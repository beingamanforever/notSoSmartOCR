from __future__ import annotations

from itertools import product

import pytest

from ocr_pipeline.verification import (
    EditCounts,
    consensus_scores,
    edit_counts,
    edit_distance,
    literal_text_risks,
    normalized_edit_distance,
)


def test_normalized_edit_distance_handles_empty_and_substitution() -> None:
    assert normalized_edit_distance("", "") == 0.0
    assert normalized_edit_distance("dose", "d0se") == 0.25
    assert normalized_edit_distance("abc", "") == 1.0


def test_consensus_scores_match_mean_pairwise_distance() -> None:
    scores = consensus_scores(("Hello World", "Hello Wrld", "Hallo World"))

    assert scores == pytest.approx((1 / 11, 1.5 / 11, 1.5 / 11))


@pytest.mark.parametrize("texts", [(), ("only",)])
def test_consensus_scores_require_multiple_candidates(texts: tuple[str, ...]) -> None:
    with pytest.raises(ValueError, match="At least two"):
        consensus_scores(texts)


def test_consensus_scores_reject_non_string_candidates() -> None:
    with pytest.raises(TypeError, match="strings"):
        consensus_scores(("text", 3))  # type: ignore[arg-type]


def test_edit_counts_separates_unsupported_and_missed_words() -> None:
    counts = edit_counts(
        "patient has severe pain today".split(),
        "patient has pain yesterday".split(),
    )

    assert counts.insertions == 1
    assert counts.deletions == 0
    assert counts.substitutions == 1
    assert counts.edits == 2


def test_edit_counts_matches_reference_exhaustively() -> None:
    sequences = [
        values for length in range(5) for values in product(("a", "b"), repeat=length)
    ]

    for prediction in sequences:
        for reference in sequences:
            assert edit_counts(prediction, reference) == _reference_edit_counts(
                prediction, reference
            )


def test_bit_parallel_distance_matches_alignment_exhaustively() -> None:
    sequences = [
        values for length in range(6) for values in product(("a", "b"), repeat=length)
    ]

    for prediction in sequences:
        for reference in sequences:
            assert (
                edit_distance(prediction, reference)
                == edit_counts(prediction, reference).edits
            )


def test_edit_counts_handles_realistic_long_text() -> None:
    prediction = "patient takes aspirin 10 mg daily ".split() * 400
    reference = "patient takes aspirin 10 mcg nightly ".split() * 400

    assert edit_counts(prediction, reference) == EditCounts(0, 0, 800)


def test_literal_text_risks_accept_common_valid_date_orders() -> None:
    text = "DOB: 05/14/1985 Visit Date: 2024-05-18 Reported Date: 14.05.1985"

    assert literal_text_risks(text) == ()


def test_literal_text_risks_are_review_only_signals() -> None:
    text = (
        "DOB: 05/ Service Date: 02/31/2024 Reported Date: 06/06/99999 "
        "Phone: 217-555-1212"
    )

    assert literal_text_risks(text) == (
        "invalid_date_shape",
        "invalid_calendar_date",
        "truncated_labeled_date",
    )


def _reference_edit_counts(
    prediction: tuple[str, ...], reference: tuple[str, ...]
) -> EditCounts:
    previous = [EditCounts(index, 0, 0) for index in range(len(prediction) + 1)]
    for reference_index, reference_value in enumerate(reference, start=1):
        current = [EditCounts(0, reference_index, 0)]
        for prediction_index, prediction_value in enumerate(prediction, start=1):
            if reference_value == prediction_value:
                current.append(previous[prediction_index - 1])
                continue
            substitution = previous[prediction_index - 1]
            deletion = previous[prediction_index]
            insertion = current[-1]
            candidates = (
                EditCounts(
                    substitution.insertions,
                    substitution.deletions,
                    substitution.substitutions + 1,
                ),
                EditCounts(
                    deletion.insertions,
                    deletion.deletions + 1,
                    deletion.substitutions,
                ),
                EditCounts(
                    insertion.insertions + 1,
                    insertion.deletions,
                    insertion.substitutions,
                ),
            )
            current.append(min(candidates, key=lambda counts: counts.edits))
        previous = current
    return previous[-1]
