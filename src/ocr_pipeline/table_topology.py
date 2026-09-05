"""Validate canonical table cells before rendering them."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


class TableTopologyError(ValueError):
    """Raised when canonical table cells do not describe one complete grid."""


@dataclass(frozen=True)
class TopologyCell:
    value: Mapping[str, Any]
    rows: tuple[int, ...]
    columns: tuple[int, ...]

    @property
    def row(self) -> int:
        return self.rows[0]

    @property
    def column(self) -> int:
        return self.columns[0]


@dataclass(frozen=True)
class TableTopology:
    row_count: int
    column_count: int
    cells: tuple[TopologyCell, ...]

    @property
    def has_spans(self) -> bool:
        return any(len(cell.rows) > 1 or len(cell.columns) > 1 for cell in self.cells)


def validate_table_topology(structure: Mapping[str, Any]) -> TableTopology:
    """Return a complete, non-overlapping rectangular table or fail closed."""
    row_count = _positive_count(structure.get("row_count"), "row_count")
    column_count = _positive_count(structure.get("column_count"), "column_count")
    values = structure.get("cells")
    if not isinstance(values, list):
        raise TableTopologyError("cells must be a list")

    cells: list[TopologyCell] = []
    occupied: dict[tuple[int, int], int] = {}
    for index, value in enumerate(values):
        if not isinstance(value, Mapping):
            raise TableTopologyError(f"cell {index + 1} must be an object")
        rows = _span(value.get("row_nums"), row_count, "row", index)
        columns = _span(value.get("column_nums"), column_count, "column", index)
        cell = TopologyCell(value=value, rows=rows, columns=columns)
        for row in rows:
            for column in columns:
                position = (row, column)
                if position in occupied:
                    raise TableTopologyError(
                        f"cells {occupied[position] + 1} and {index + 1} overlap "
                        f"at row {row + 1}, column {column + 1}"
                    )
                occupied[position] = index
        cells.append(cell)

    expected = row_count * column_count
    if len(occupied) != expected:
        missing = next(
            (row, column)
            for row in range(row_count)
            for column in range(column_count)
            if (row, column) not in occupied
        )
        raise TableTopologyError(
            f"table has a gap at row {missing[0] + 1}, column {missing[1] + 1}"
        )

    return TableTopology(
        row_count=row_count,
        column_count=column_count,
        cells=tuple(sorted(cells, key=lambda cell: (cell.row, cell.column))),
    )


def _positive_count(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TableTopologyError(f"{name} must be a positive integer")
    return value


def _span(value: Any, limit: int, axis: str, cell_index: int) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise TableTopologyError(f"cell {cell_index + 1} has no {axis}_nums")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise TableTopologyError(
            f"cell {cell_index + 1} {axis}_nums must contain integers"
        )
    indexes = tuple(sorted(value))
    if len(set(indexes)) != len(indexes):
        raise TableTopologyError(f"cell {cell_index + 1} has duplicate {axis} indexes")
    if indexes[0] < 0 or indexes[-1] >= limit:
        raise TableTopologyError(f"cell {cell_index + 1} {axis} span is out of bounds")
    if indexes != tuple(range(indexes[0], indexes[-1] + 1)):
        raise TableTopologyError(
            f"cell {cell_index + 1} has a non-contiguous {axis} span"
        )
    return indexes
