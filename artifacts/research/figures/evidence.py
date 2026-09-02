"""Read the measured Markdown tables used by the report figures."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def table_rows(
    path: Path,
    first_column: str,
    *,
    occurrence: int = 1,
) -> dict[str, list[str]]:
    """Return rows from the table whose first header cell matches."""
    lines = path.read_text(encoding="utf-8").splitlines()
    header = f"| {first_column} |"
    starts = [index for index, line in enumerate(lines) if line.startswith(header)]
    if occurrence <= 0 or len(starts) < occurrence:
        raise ValueError(
            f"Table {first_column!r} occurrence {occurrence} not found in {path}"
        )
    start = starts[occurrence - 1]

    rows: dict[str, list[str]] = {}
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        rows[cells[0]] = cells[1:]
    return rows


def decimal(value: str) -> float:
    """Parse the leading decimal from a Markdown table cell."""
    token = value.replace("**", "").split(",", 1)[0].strip()
    return float(token)


def fraction(value: str) -> float:
    """Parse the leading numerator and denominator from a Markdown cell."""
    token = value.replace("**", "").split(",", 1)[0].strip()
    numerator, denominator = token.split("/", 1)
    return float(numerator.strip()) / float(denominator.strip())
