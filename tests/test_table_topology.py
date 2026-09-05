from __future__ import annotations

import pytest

from ocr_pipeline.table_topology import TableTopologyError, validate_table_topology


def test_accepts_complete_grid_and_preserves_blank_cell() -> None:
    blank = {"row_nums": [1], "column_nums": [1], "text": ""}
    structure = {
        "row_count": 2,
        "column_count": 2,
        "cells": [
            {"row_nums": [0], "column_nums": [0], "text": "Name"},
            {"row_nums": [0], "column_nums": [1], "text": "Value"},
            {"row_nums": [1], "column_nums": [0], "text": "Dose"},
            blank,
        ],
    }

    topology = validate_table_topology(structure)

    assert topology.row_count == 2
    assert topology.column_count == 2
    assert topology.cells[-1].value is blank
    assert topology.cells[-1].value["text"] == ""
    assert not topology.has_spans


def test_accepts_complete_rectangular_span() -> None:
    topology = validate_table_topology(
        {
            "row_count": 2,
            "column_count": 2,
            "cells": [
                {"row_nums": [0, 1], "column_nums": [0], "text": "Group"},
                {"row_nums": [0], "column_nums": [1], "text": "A"},
                {"row_nums": [1], "column_nums": [1], "text": "B"},
            ],
        }
    )

    assert topology.has_spans


@pytest.mark.parametrize("name,value", [("row_count", 0), ("column_count", True)])
def test_rejects_invalid_dimensions(name: str, value: object) -> None:
    structure = {"row_count": 1, "column_count": 1, "cells": []}
    structure[name] = value

    with pytest.raises(TableTopologyError, match=f"{name} must be"):
        validate_table_topology(structure)


def test_rejects_out_of_bounds_cell() -> None:
    with pytest.raises(TableTopologyError, match="column span is out of bounds"):
        validate_table_topology(
            {
                "row_count": 1,
                "column_count": 1,
                "cells": [{"row_nums": [0], "column_nums": [1], "text": "A"}],
            }
        )


def test_rejects_non_contiguous_span() -> None:
    with pytest.raises(TableTopologyError, match="non-contiguous row span"):
        validate_table_topology(
            {
                "row_count": 3,
                "column_count": 1,
                "cells": [
                    {"row_nums": [0, 2], "column_nums": [0], "text": "A"},
                    {"row_nums": [1], "column_nums": [0], "text": "B"},
                ],
            }
        )


def test_rejects_overlapping_cells() -> None:
    with pytest.raises(TableTopologyError, match="overlap at row 1, column 2"):
        validate_table_topology(
            {
                "row_count": 1,
                "column_count": 2,
                "cells": [
                    {"row_nums": [0], "column_nums": [0, 1], "text": "A"},
                    {"row_nums": [0], "column_nums": [1], "text": "B"},
                ],
            }
        )


def test_rejects_gap_in_declared_grid() -> None:
    with pytest.raises(TableTopologyError, match="gap at row 2, column 2"):
        validate_table_topology(
            {
                "row_count": 2,
                "column_count": 2,
                "cells": [
                    {"row_nums": [0], "column_nums": [0, 1], "text": "Header"},
                    {"row_nums": [1], "column_nums": [0], "text": "A"},
                ],
            }
        )
