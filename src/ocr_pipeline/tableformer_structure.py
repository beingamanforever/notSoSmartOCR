"""Docling TableFormer table structure predictor (geometry only).

We consume TableFormer's image-only structure path (``do_matching=False``,
which routes to ``predict_dummy`` in ``tf_predictor.py``) and drop its own
``MatchingPostProcessor`` token matcher entirely: that matcher fabricates cell
text via an unbounded nearest-column snap (docling-ibm-models issue 188). Our
own pipeline assigns text to cells by geometry afterwards.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from .contracts import BoundingBox
from .providers import ReaderError
from .tables import TableCell

MODEL_ID = "ds4sd/docling-models"
MODEL_REVISION = "v2.3.0"
ARTIFACT_SUBTREE = "model_artifacts/tableformer/accurate"


class TableFormerStructure:
    """Lazy adapter around Docling's TableFormer structure predictor."""

    name = "docling-tableformer"

    def __init__(
        self,
        *,
        device: str = "cuda",
        artifacts_path: Path | None = None,
        predictor: Callable[[Image.Image], Sequence[Mapping[str, Any]]] | None = None,
    ) -> None:
        self.device = device
        self.artifacts_path = (
            Path(artifacts_path) if artifacts_path is not None else None
        )
        self._predictor = predictor
        self._tf_predictor: object | None = None
        self._lock = threading.Lock()

    def predict(self, crop: Image.Image) -> tuple[TableCell, ...]:
        with self._lock:
            call = (
                self._predictor
                if self._predictor is not None
                else self._call_tf_predictor
            )
            try:
                responses = call(crop)
            except ReaderError:
                raise
            except Exception as error:
                raise ReaderError("tableformer_predict_failed", str(error)) from error
        return _parse_cells(responses)

    def model_provenance(self) -> dict[str, Any]:
        return {
            "id": MODEL_ID,
            "model": "TableFormer (TableModel04rs, OTSL)",
            "code_license": "MIT",
            "weights_license": "CDLA-Permissive-2.0",
            "origin": "IBM Research",
            "note": (
                "Geometry only: docling's MatchingPostProcessor token matching is "
                "not used because it fabricates cell text via an unbounded "
                "nearest-column snap (docling-ibm-models issue 188); our own "
                "pipeline assigns text to cells by geometry."
            ),
        }

    def _call_tf_predictor(self, crop: Image.Image) -> Sequence[Mapping[str, Any]]:
        import numpy

        tf_predictor = self._get_tf_predictor()
        image = crop.convert("RGB")
        width, height = image.size
        page_input = {
            "width": float(width),
            "height": float(height),
            "image": numpy.asarray(image),
            "tokens": [],
        }
        table_bbox = [0.0, 0.0, float(width), float(height)]
        output = tf_predictor.multi_table_predict(
            page_input, [table_bbox], do_matching=False
        )
        return output[0]["tf_responses"]

    def _get_tf_predictor(self) -> object:
        if self._tf_predictor is not None:
            return self._tf_predictor
        try:
            from huggingface_hub import snapshot_download

            import docling_ibm_models.tableformer.common as tableformer_common
            from docling_ibm_models.tableformer.data_management.tf_predictor import (
                TFPredictor,
            )
        except ImportError as error:
            raise ReaderError("tableformer_load_failed", str(error)) from error

        try:
            download_root = Path(
                snapshot_download(
                    repo_id=MODEL_ID,
                    revision=MODEL_REVISION,
                    allow_patterns=[f"{ARTIFACT_SUBTREE}/*"],
                    local_dir=(
                        str(self.artifacts_path)
                        if self.artifacts_path is not None
                        else None
                    ),
                )
            )
            save_dir = download_root / ARTIFACT_SUBTREE
            config = tableformer_common.read_config(str(save_dir / "tm_config.json"))
            config["model"]["save_dir"] = str(save_dir)
            self._tf_predictor = TFPredictor(config, device=self.device)
        except Exception as error:
            raise ReaderError("tableformer_load_failed", str(error)) from error
        return self._tf_predictor


def _parse_cells(responses: Sequence[Mapping[str, Any]]) -> tuple[TableCell, ...]:
    cells = []
    for response in responses:
        box = _cell_bounding_box(response.get("bbox"))
        if box is None:
            continue
        row_start = int(response.get("start_row_offset_idx", 0))
        row_end = int(response.get("end_row_offset_idx", row_start + 1))
        column_start = int(response.get("start_col_offset_idx", 0))
        column_end = int(response.get("end_col_offset_idx", column_start + 1))
        cells.append(
            TableCell(
                bounding_box=box,
                row_nums=tuple(range(row_start, max(row_start + 1, row_end))),
                column_nums=tuple(
                    range(column_start, max(column_start + 1, column_end))
                ),
                column_header=bool(response.get("column_header", False)),
                # TableFormer's "row_section" (OTSL "srow") is a label row
                # spanning every column, the same role TATR calls a
                # "projected row header".
                projected_row_header=bool(response.get("row_section", False)),
            )
        )
    return tuple(cells)


def _cell_bounding_box(bbox: object) -> BoundingBox | None:
    if not isinstance(bbox, Mapping):
        return None
    try:
        left = math.floor(float(bbox["l"]))
        top = math.floor(float(bbox["t"]))
        right = math.ceil(float(bbox["r"]))
        bottom = math.ceil(float(bbox["b"]))
    except (KeyError, TypeError, ValueError):
        return None
    if right <= left or bottom <= top:
        return None
    return BoundingBox(left, top, right, bottom)
