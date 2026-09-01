"""Training-free OCR disagreement and literal-text risk signals."""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Hashable, Sequence
from dataclasses import dataclass

DATE_TOKEN = re.compile(r"(?<!\d)(\d{1,6})([./-])(\d{1,6})([./-])(\d{1,6})(?!\d)")
LABELED_DATE = re.compile(
    r"\b(?:dob|date of birth|birth date|visit date|service date|reported date)"
    r"\s*[:#]?\s*(\d[\d./-]*)",
    re.IGNORECASE,
)
TEXT_RISK_ORDER = (
    "invalid_date_shape",
    "invalid_calendar_date",
    "truncated_labeled_date",
)


@dataclass(frozen=True)
class EditCounts:
    """Counts from one deterministic minimum-edit alignment."""

    insertions: int
    deletions: int
    substitutions: int

    @property
    def edits(self) -> int:
        return self.insertions + self.deletions + self.substitutions


def normalized_edit_distance(left: str, right: str) -> float:
    """Return character edit distance divided by the longer input length."""
    if not isinstance(left, str) or not isinstance(right, str):
        raise TypeError("OCR candidates must be strings")
    denominator = max(len(left), len(right))
    return edit_distance(left, right) / denominator if denominator else 0.0


def edit_distance(left: Sequence[Hashable], right: Sequence[Hashable]) -> int:
    """Return exact Levenshtein distance with a bit-parallel row update."""
    if len(left) > len(right):
        left, right = right, left
    if not left:
        return len(right)

    positions: dict[Hashable, int] = {}
    for index, value in enumerate(left):
        positions[value] = positions.get(value, 0) | (1 << index)

    positive = ~0
    negative = 0
    distance = len(left)
    final_bit = 1 << (len(left) - 1)
    for value in right:
        matches = positions.get(value, 0)
        vertical = matches | negative
        horizontal = (((matches & positive) + positive) ^ positive) | matches
        positive_shift = negative | ~(horizontal | positive)
        negative_shift = positive & horizontal
        if positive_shift & final_bit:
            distance += 1
        elif negative_shift & final_bit:
            distance -= 1
        positive_shift = (positive_shift << 1) | 1
        negative_shift <<= 1
        positive = negative_shift | ~(vertical | positive_shift)
        negative = positive_shift & vertical
    return distance


def consensus_scores(texts: Sequence[str]) -> tuple[float, ...]:
    """Return each candidate's mean disagreement with all other candidates."""
    if len(texts) < 2:
        raise ValueError("At least two OCR candidates are required")
    if any(not isinstance(text, str) for text in texts):
        raise TypeError("OCR candidates must be strings")

    count = len(texts) - 1
    return tuple(
        sum(
            normalized_edit_distance(text, other)
            for other_index, other in enumerate(texts)
            if other_index != index
        )
        / count
        for index, text in enumerate(texts)
    )


def edit_counts(
    prediction: Sequence[object], reference: Sequence[object]
) -> EditCounts:
    """Align a prediction to a reference and count edit operation types."""
    start = 0
    prediction_end = len(prediction)
    reference_end = len(reference)
    while (
        start < prediction_end
        and start < reference_end
        and prediction[start] == reference[start]
    ):
        start += 1
    while (
        prediction_end > start
        and reference_end > start
        and prediction[prediction_end - 1] == reference[reference_end - 1]
    ):
        prediction_end -= 1
        reference_end -= 1

    width = prediction_end - start
    previous_edits = list(range(width + 1))
    previous_insertions = list(range(width + 1))
    previous_deletions = [0] * (width + 1)
    previous_substitutions = [0] * (width + 1)
    current_edits = [0] * (width + 1)
    current_insertions = [0] * (width + 1)
    current_deletions = [0] * (width + 1)
    current_substitutions = [0] * (width + 1)

    for reference_index in range(start, reference_end):
        row = reference_index - start + 1
        current_edits[0] = row
        current_insertions[0] = 0
        current_deletions[0] = row
        current_substitutions[0] = 0
        reference_value = reference[reference_index]
        for column, prediction_index in enumerate(range(start, prediction_end), 1):
            if reference_value == prediction[prediction_index]:
                current_edits[column] = previous_edits[column - 1]
                current_insertions[column] = previous_insertions[column - 1]
                current_deletions[column] = previous_deletions[column - 1]
                current_substitutions[column] = previous_substitutions[column - 1]
                continue

            substitution_edits = previous_edits[column - 1] + 1
            deletion_edits = previous_edits[column] + 1
            insertion_edits = current_edits[column - 1] + 1
            if (
                substitution_edits <= deletion_edits
                and substitution_edits <= insertion_edits
            ):
                current_edits[column] = substitution_edits
                current_insertions[column] = previous_insertions[column - 1]
                current_deletions[column] = previous_deletions[column - 1]
                current_substitutions[column] = previous_substitutions[column - 1] + 1
            elif deletion_edits <= insertion_edits:
                current_edits[column] = deletion_edits
                current_insertions[column] = previous_insertions[column]
                current_deletions[column] = previous_deletions[column] + 1
                current_substitutions[column] = previous_substitutions[column]
            else:
                current_edits[column] = insertion_edits
                current_insertions[column] = current_insertions[column - 1] + 1
                current_deletions[column] = current_deletions[column - 1]
                current_substitutions[column] = current_substitutions[column - 1]

        previous_edits, current_edits = current_edits, previous_edits
        previous_insertions, current_insertions = (
            current_insertions,
            previous_insertions,
        )
        previous_deletions, current_deletions = (
            current_deletions,
            previous_deletions,
        )
        previous_substitutions, current_substitutions = (
            current_substitutions,
            previous_substitutions,
        )

    return EditCounts(
        previous_insertions[-1],
        previous_deletions[-1],
        previous_substitutions[-1],
    )


def literal_text_risks(text: str) -> tuple[str, ...]:
    """Find conservative date anomalies without changing the source text."""
    if not isinstance(text, str):
        raise TypeError("OCR text must be a string")

    found: set[str] = set()
    if any(
        _date_risk(match) == "invalid_calendar_date"
        for match in DATE_TOKEN.finditer(text)
    ):
        found.add("invalid_calendar_date")
    for match in LABELED_DATE.finditer(text):
        value = match.group(1)
        token = DATE_TOKEN.fullmatch(value)
        risk = _date_risk(token) if token else None
        if risk:
            found.add(risk)
        if token is None and _is_truncated_date(value):
            found.add("truncated_labeled_date")
    return tuple(risk for risk in TEXT_RISK_ORDER if risk in found)


def _date_risk(match: re.Match[str]) -> str | None:
    first, _, second, _, third = match.groups()
    if len(first) == 4:
        if len(second) > 2 or len(third) > 2:
            return "invalid_date_shape"
        choices = [(int(first), int(second), int(third))]
    elif len(third) in {2, 4} and len(first) <= 2 and len(second) <= 2:
        year = int(third) if len(third) == 4 else 2000 + int(third)
        choices = [
            (year, int(first), int(second)),
            (year, int(second), int(first)),
        ]
    else:
        return "invalid_date_shape"
    return (
        None
        if any(_is_date(*choice) for choice in choices)
        else "invalid_calendar_date"
    )


def _is_date(year: int, month: int, day: int) -> bool:
    try:
        dt.date(year, month, day)
    except ValueError:
        return False
    return True


def _is_truncated_date(value: str) -> bool:
    if DATE_TOKEN.fullmatch(value):
        return False
    return (
        value.endswith(("/", ".", "-"))
        or sum(value.count(separator) for separator in "/.-") < 2
    )
