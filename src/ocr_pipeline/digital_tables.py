"""Structure of born-digital tables: Docling's TableFormer over the page's own text cells.

A table decoder reading pixels drops glyph-only cells such as checkbox marks, so a
"Label [box]" grid loses the column that says which label is ticked. On a page with a
text layer, Docling's standard pipeline predicts the grid with TableFormer and fills
it by matching the exact PDF cells; the same official predictor is used here. Pages
without a text layer keep the Falcon table read.
"""

from __future__ import annotations

import html
import threading
from collections.abc import Sequence
from typing import Any


class DigitalTableReader:
    name = "docling-tableformer"

    def __init__(self, device: str = "cuda", artifacts_path=None) -> None:
        self.device = device
        self.artifacts_path = artifacts_path
        self._predictor = None
        self._lock = threading.Lock()

    def __call__(
        self,
        image: Any,
        box: Sequence[float],
        cells: Sequence[dict[str, Any]],
    ) -> str:
        """HTML for the table in `box`, its cells holding the page's own text."""
        import numpy

        left, top, right, bottom = box
        tokens = [
            {
                "id": index,
                "text": cell["text"],
                "bbox": {"l": x0, "t": y0, "r": x1, "b": y1, "coord_origin": "TOPLEFT"},
            }
            for index, cell in enumerate(cells)
            for x0, y0, x1, y1 in [cell["bbox"]]
            if cell["text"].strip()
            and left <= (x0 + x1) / 2 <= right
            and top <= (y0 + y1) / 2 <= bottom
        ]
        page = {
            "width": image.width,
            "height": image.height,
            "image": numpy.asarray(image),
            "tokens": tokens,
        }
        with self._lock:
            output = self._model().multi_table_predict(
                page, [list(box)], do_matching=True
            )[0]
        details = output["predict_details"]
        return table_html(
            output["tf_responses"],
            details.get("num_rows", 0),
            details.get("num_cols", 0),
        )

    def _model(self):
        if self._predictor is None:
            from docling.datamodel.accelerator_options import AcceleratorOptions
            from docling.datamodel.pipeline_options import (
                TableFormerMode,
                TableStructureOptions,
            )
            from docling.models.stages.table_structure.table_structure_model import (
                TableStructureModel,
            )

            model = TableStructureModel(
                enabled=True,
                artifacts_path=self.artifacts_path,
                options=TableStructureOptions(
                    do_cell_matching=True, mode=TableFormerMode.ACCURATE
                ),
                accelerator_options=AcceleratorOptions(device=self.device),
            )
            self._predictor = model.tf_predictor
        return self._predictor


def table_html(responses: Sequence[dict[str, Any]], rows: int, columns: int) -> str:
    """Render TableFormer's cells as the HTML table the reader already parses."""
    from docling_core.types.doc import TableCell

    placed: dict[tuple[int, int], TableCell] = {}
    for response in responses:
        cell = TableCell.model_validate(response)
        placed[(cell.start_row_offset_idx, cell.start_col_offset_idx)] = cell
    covered = {
        (row, column)
        for cell in placed.values()
        for row in range(cell.start_row_offset_idx, cell.end_row_offset_idx)
        for column in range(cell.start_col_offset_idx, cell.end_col_offset_idx)
    }
    lines = ["<table>"]
    for row in range(rows):
        lines.append("<tr>")
        for column in range(columns):
            cell = placed.get((row, column))
            if cell is None:
                if (row, column) not in covered:
                    lines.append("<td></td>")
                continue
            tag = "th" if cell.column_header else "td"
            spans = "".join(
                f' {name}="{value}"'
                for name, value in (
                    ("rowspan", cell.row_span),
                    ("colspan", cell.col_span),
                )
                if value > 1
            )
            lines.append(f"<{tag}{spans}>{html.escape(cell.text.strip())}</{tag}>")
        lines.append("</tr>")
    lines.append("</table>")
    return "".join(lines)
