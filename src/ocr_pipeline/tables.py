"""Evidence-linked table extraction with Microsoft Table Transformer."""

from __future__ import annotations

import copy
import importlib.util
import inspect
import math
import re
import sys
import tempfile
import threading
import unicodedata
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from difflib import SequenceMatcher
from pathlib import Path
from statistics import median
from types import ModuleType
from typing import Any, Callable, Protocol

from PIL import Image, UnidentifiedImageError

from .contracts import BoundingBox, TextAlternative, TextRegion
from .providers import LocalReader, ReaderError

DEFAULT_DETECTION_ID = "microsoft/table-transformer-detection"
DEFAULT_DETECTION_REVISION = "2357cbe2b5a5d1c03e54f32764f06058933b65ab"
DEFAULT_STRUCTURE_ID = "microsoft/table-transformer-structure-recognition-v1.1-all"
DEFAULT_STRUCTURE_REVISION = "7587a7ef111d9dcbf8ac695f1376ab7014340a0c"
DEFAULT_SOURCE_REVISION = "16d124f616109746b7785f03085100f1f6247575"
MODEL_LICENSE = "MIT"
MODEL_ORIGIN = "Microsoft"
TABLE_DUPLICATE_CONTAINMENT = 0.98
SUBGRID_CONTAINMENT = 0.85
NEAR_PAGE_TABLE_AREA = 0.65
# Share of cells that may be set aside before a grid is treated as unreliable.
MAX_RECOVERABLE_CELL_COLLISIONS = 0.1
MIN_NEAR_PAGE_ROWS = 3
MIN_NEAR_PAGE_COLUMNS = 3
MIN_NEAR_PAGE_CELLS = 9
MAX_PARALLEL_CHALLENGERS = 4
MAX_CELL_REREADS_PER_PAGE = 64
CELL_REREAD_PADDING = 0
CELL_REREAD_SCALE = 3
DECIMAL_CELL_REREAD_SCALE = 6
GROUPED_CELL_REREAD_PADDING = 2
VALUE_PATTERN = re.compile(r"[+-]?\(?\d[\d,]*(?:\.\d+)?%?\)?")
OBSERVED_VALUE_PATTERN = re.compile(
    r"(?:[$#]\s*)?[+-]?(?:\(\s*)?\d[\d,]*(?:\.\d+)?%?(?:\s*\))?"
)
LABEL_UNIT_PATTERN = re.compile(r"\(\s*(\$[A-Za-z])(?:[,\s)]|$)")
LABEL_UNIT_SLOT_PATTERN = re.compile(r"\((?:\$[^)]{1,12}|[A-Z]{1,3})\)")
LABEL_UNIT_MARKER_PATTERN = re.compile(r"(\(\s*)(\$[A-Za-z0-9]|[A-Z]{1,3})(?=[,\s)])")
CELL_CROSSING_OVERLAP = 0.2
CELL_CROSSING_SCALE = 1.25
RULED_PROPOSAL_SOURCE = "opencv_ruled_table"
LAYOUT_PROPOSAL_SOURCE = "reader_layout_region"
RULED_DUPLICATE_OVERLAP = 0.9
HORIZONTAL_PANEL_PROPOSAL_SOURCE = "opencv_horizontal_panel_decomposition"
NUMERIC_COLUMN_CORE_SOURCE = "numeric_column_core"
ALIGNED_NUMERIC_ROWS_SOURCE = "ocr_aligned_numeric_bands"
MIN_NUMERIC_CORE_COLUMNS = 3
MIN_NUMERIC_CORE_FILLED_CELLS = 2
MIN_ALIGNED_NUMERIC_COLUMNS = 2


@dataclass(frozen=True)
class TableCell:
    bounding_box: BoundingBox
    row_nums: tuple[int, ...]
    column_nums: tuple[int, ...]
    column_header: bool = False
    projected_row_header: bool = False
    span_boxes: tuple[BoundingBox, ...] = ()
    span_texts: tuple[str, ...] = ()


@dataclass(frozen=True)
class TablePrediction:
    bounding_box: BoundingBox
    cells: tuple[TableCell, ...]
    confidence: float | None
    model: dict[str, Any]


class TableExtractor(Protocol):
    name: str

    def extract(
        self,
        image_path: Path,
        tokens: list[dict[str, Any]],
    ) -> list[TablePrediction]: ...


@dataclass(frozen=True)
class TableChallenger:
    name: str
    reader: LocalReader
    scope: str = "table"
    prepare: Callable[[Image.Image], Image.Image] | None = None


@dataclass(frozen=True)
class _Candidate:
    text: str
    confidence: float | None
    source: str
    evidence_ids: tuple[str, ...]
    regions: tuple[TextRegion, ...]


@dataclass(frozen=True)
class _TableCrop:
    path: Path
    offset: tuple[int, int]


@dataclass(frozen=True)
class _RuledTableProposal:
    bounding_box: BoundingBox
    row_edges: tuple[int, ...]
    column_edges: tuple[int, ...]


class TatrTableStage:
    """Build structured tables while retaining every OCR evidence region."""

    name = "tables"

    def __init__(
        self,
        extractor: TableExtractor,
        *,
        challengers: Sequence[TableChallenger] = (),
        low_primary_confidence: float = 0.9,
        challenger_padding: int | tuple[int, int] | None = None,
        parallel_challengers: bool = False,
    ) -> None:
        if not 0 <= low_primary_confidence <= 1:
            raise ValueError("low_primary_confidence must be from 0 to 1")
        names = [challenger.name for challenger in challengers]
        if any(not name.strip() for name in names) or len(names) != len(set(names)):
            raise ValueError("Table challenger names must be non-empty and unique")
        if any(
            challenger.scope not in {"table", "page", "cell"}
            for challenger in challengers
        ):
            raise ValueError("Table challenger scope must be table, page, or cell")
        if challenger_padding is not None:
            padding = (
                (challenger_padding, challenger_padding)
                if isinstance(challenger_padding, int)
                else challenger_padding
            )
            if len(padding) != 2 or min(padding) < 0:
                raise ValueError("challenger_padding must contain non-negative values")
        self.extractor = extractor
        self.challengers = tuple(challengers)
        self.low_primary_confidence = low_primary_confidence
        self.challenger_padding = challenger_padding
        self.parallel_challengers = parallel_challengers

    def apply(
        self,
        image_path: Path,
        page_number: int,
        regions: list[TextRegion],
    ) -> list[TextRegion]:
        tokens = _to_tokens(regions)
        layout_boxes = [
            region.bounding_box for region in regions if region.kind == "table"
        ]
        parameters = inspect.signature(self.extractor.extract).parameters
        keywords: dict[str, Any] = {}
        if layout_boxes and "layout_boxes" in parameters:
            keywords["layout_boxes"] = layout_boxes
        if "word_boxes" in parameters:
            keywords["word_boxes"] = _word_tokens(regions)
        predictions = self.extractor.extract(image_path, tokens, **keywords)
        if not predictions:
            return regions

        try:
            with Image.open(image_path) as source:
                page_size = source.size
        except (OSError, UnidentifiedImageError) as error:
            raise ReaderError("table_image_failed", str(error)) from error

        predictions = sorted(
            predictions,
            key=lambda item: (
                item.bounding_box.top,
                item.bounding_box.left,
                item.bounding_box.bottom,
                item.bounding_box.right,
            ),
        )
        rejected = []
        accepted = []
        dropped_cells_by_table: dict[int, tuple[TableCell, ...]] = {}
        for table_index, prediction in enumerate(predictions, start=1):
            primary = _assign_regions([prediction], regions)[0]
            resolved, dropped = _resolve_cell_collisions(prediction)
            diagnostic = _rejected_table_candidate(
                resolved,
                primary + _word_evidence_regions(regions, resolved.cells),
                page_size,
                page_number,
                table_index,
                self.extractor.name,
            )
            if diagnostic is None:
                accepted.append((table_index, resolved))
                if dropped:
                    dropped_cells_by_table[id(resolved)] = dropped
            else:
                rejected.append(diagnostic)
        predictions, subgrid_rejected = _reject_subgrid_duplicates(
            accepted,
            regions,
            page_number,
            self.extractor.name,
        )
        rejected.extend(subgrid_rejected)
        if not predictions:
            return sorted(
                regions + rejected,
                key=lambda region: (
                    region.reading_order,
                    0 if region.kind == "table_candidate" else 1,
                    region.id,
                ),
            )

        primary_by_table = _assign_regions(predictions, regions)
        # A region that crosses cell boundaries is excluded from cell assignment, so a
        # table read as one blob leaves every cell blank. Its recognised words carry
        # boxes and confidences, so they slot into cells the way the official TATR
        # pipeline slots page tokens.
        assignable_by_table = [
            primary + _word_evidence_regions(regions, prediction.cells)
            for prediction, primary in zip(predictions, primary_by_table, strict=True)
        ]
        challengers_by_table = self._read_challengers(
            image_path,
            page_number,
            predictions,
            assignable_by_table,
        )

        tables: list[TextRegion] = []
        retained_challengers: list[TextRegion] = []
        for table_index, prediction in enumerate(predictions, start=1):
            table_id = f"p{page_number}-tables-table-{table_index}"
            primary = assignable_by_table[table_index - 1]
            challenger_groups = {
                name: assigned[table_index - 1]
                for name, assigned in challengers_by_table.items()
            }
            table, primary_sources, challenger_sources = self._table_region(
                table_id,
                prediction,
                primary,
                challenger_groups,
                table_index,
                dropped_cells_by_table.get(id(prediction), ()),
                page_regions=regions,
            )
            _mark_sources(primary_sources, table_id)
            for items in challenger_sources.values():
                _mark_sources(items, table_id)
                retained_challengers.extend(items)
            tables.append(table)

        output = regions + retained_challengers + tables + rejected
        return sorted(
            output,
            key=lambda region: (
                region.reading_order,
                0 if region.kind == "table" else 1,
                region.id,
            ),
        )

    def _read_challengers(
        self,
        image_path: Path,
        page_number: int,
        predictions: Sequence[TablePrediction],
        primary_by_table: Sequence[list[TextRegion]],
    ) -> dict[str, list[list[TextRegion]]]:
        result: dict[str, list[list[TextRegion]]] = {}
        if not self.challengers:
            return result
        try:
            with Image.open(image_path) as source:
                image = source.convert("RGB")
        except (OSError, UnidentifiedImageError) as error:
            raise ReaderError("table_challenger_image_failed", str(error)) from error

        configured_padding = self.challenger_padding
        if configured_padding is None:
            configured_padding = int(getattr(self.extractor, "crop_padding", 5))
        padding = (
            (configured_padding, configured_padding)
            if isinstance(configured_padding, int)
            else configured_padding
        )
        standard = tuple(
            challenger for challenger in self.challengers if challenger.scope != "cell"
        )
        cells = tuple(
            challenger for challenger in self.challengers if challenger.scope == "cell"
        )
        with tempfile.TemporaryDirectory(prefix="ocr-table-") as directory:
            root = Path(directory)
            table_crops = (
                _write_table_crops(image, predictions, padding, root)
                if any(challenger.scope == "table" for challenger in standard)
                else ()
            )
            if self.parallel_challengers and len(standard) > 1:
                with ThreadPoolExecutor(
                    max_workers=min(len(standard), MAX_PARALLEL_CHALLENGERS),
                    thread_name_prefix="table-ocr",
                ) as executor:
                    futures = [
                        executor.submit(
                            self._read_challenger,
                            challenger,
                            image,
                            image_path,
                            root,
                            page_number,
                            predictions,
                            table_crops,
                        )
                        for challenger in standard
                    ]
                    for challenger, future in zip(
                        standard,
                        futures,
                        strict=True,
                    ):
                        result[challenger.name] = future.result()
            else:
                for challenger in standard:
                    result[challenger.name] = self._read_challenger(
                        challenger,
                        image,
                        image_path,
                        root,
                        page_number,
                        predictions,
                        table_crops,
                    )
            for challenger in cells:
                result[challenger.name] = self._read_cell_challenger(
                    challenger,
                    image,
                    root,
                    page_number,
                    predictions,
                    primary_by_table,
                    result,
                )
        return result

    def _read_cell_challenger(
        self,
        challenger: TableChallenger,
        image: Image.Image,
        root: Path,
        page_number: int,
        predictions: Sequence[TablePrediction],
        primary_by_table: Sequence[list[TextRegion]],
        prior: dict[str, list[list[TextRegion]]],
    ) -> list[list[TextRegion]]:
        output: list[list[TextRegion]] = []
        remaining_rereads = MAX_CELL_REREADS_PER_PAGE
        for table_index, prediction in enumerate(predictions, start=1):
            primary_cells = _assign_cells(
                prediction.cells,
                primary_by_table[table_index - 1],
            )
            prior_cells = {
                name: _assign_cells(prediction.cells, tables[table_index - 1])
                for name, tables in prior.items()
            }
            jobs = []
            for cell_index, cell in enumerate(prediction.cells, start=1):
                if remaining_rereads <= 0:
                    break
                primary = _candidate(primary_cells[cell_index - 1], "primary")
                existing = [
                    _candidate(items[cell_index - 1], name)
                    for name, items in prior_cells.items()
                ]
                row_peers = _row_peer_candidates(
                    cell_index - 1,
                    prediction.cells,
                    primary_cells,
                )
                label_unit = _expected_label_unit(
                    _column_peer_candidates(
                        cell_index - 1,
                        prediction.cells,
                        primary_cells,
                    )
                )
                if not _needs_cell_reread(
                    primary,
                    existing,
                    self.low_primary_confidence,
                    row_peers,
                    min(cell.column_nums, default=0) > 0,
                    label_unit,
                ):
                    continue
                padding, scale = _cell_reread_view(row_peers)
                crop, offset = _table_crop(
                    image,
                    cell.bounding_box,
                    (padding, padding),
                )
                if not _normalize(primary.text) and not _has_visible_ink(crop):
                    crop.close()
                    continue
                path = root / f"{_slug(challenger.name)}-{table_index}-{cell_index}.png"
                prepared = None
                scaled = None
                try:
                    prepared = (
                        challenger.prepare(crop.copy())
                        if challenger.prepare is not None
                        else crop
                    )
                    if not isinstance(prepared, Image.Image):
                        raise TypeError("Table preprocessing must return a PIL image")
                    scaled = prepared.resize(
                        (
                            prepared.width * scale,
                            prepared.height * scale,
                        ),
                        Image.Resampling.LANCZOS,
                    )
                    scaled.save(path, format="PNG")
                except Exception as error:
                    raise ReaderError("table_preprocess_failed", str(error)) from error
                finally:
                    if scaled is not None:
                        scaled.close()
                    if prepared is not None and prepared is not crop:
                        prepared.close()
                    crop.close()
                jobs.append((cell_index, cell, path, offset, scale))
                remaining_rereads -= 1
            if self.parallel_challengers and len(jobs) > 1:
                with ThreadPoolExecutor(
                    max_workers=min(len(jobs), MAX_PARALLEL_CHALLENGERS),
                    thread_name_prefix="cell-ocr",
                ) as executor:
                    readings = list(
                        executor.map(
                            lambda job: challenger.reader.read(job[2], page_number),
                            jobs,
                        )
                    )
            else:
                readings = [
                    challenger.reader.read(path, page_number)
                    for _, _, path, _, _ in jobs
                ]
            table_regions = []
            for job, regions in zip(jobs, readings, strict=True):
                cell_index, cell, _, offset, scale = job
                for source_index, region in enumerate(regions, start=1):
                    provenance = dict(region.text_provenance or {})
                    provenance.update(
                        {
                            "challenger": challenger.name,
                            "challenger_scope": "cell",
                            "source_id": region.id,
                            "source_provider": region.provider,
                            "source_cell_bbox": asdict(cell.bounding_box),
                            "scale": scale,
                        }
                    )
                    table_regions.append(
                        replace(
                            region,
                            id=(
                                f"p{page_number}-tables-{_slug(challenger.name)}-"
                                f"t{table_index}-c{cell_index}-source-{source_index}"
                            ),
                            provider=challenger.name,
                            bounding_box=BoundingBox(
                                left=offset[0]
                                + math.floor(region.bounding_box.left / scale),
                                top=offset[1]
                                + math.floor(region.bounding_box.top / scale),
                                right=offset[0]
                                + math.ceil(region.bounding_box.right / scale),
                                bottom=offset[1]
                                + math.ceil(region.bounding_box.bottom / scale),
                            ),
                            text_provenance=provenance,
                            alternatives=list(region.alternatives),
                            structure=copy.deepcopy(region.structure),
                        )
                    )
            output.append(table_regions)
        return output

    def _read_challenger(
        self,
        challenger: TableChallenger,
        image: Image.Image,
        image_path: Path,
        root: Path,
        page_number: int,
        predictions: Sequence[TablePrediction],
        table_crops: Sequence[_TableCrop],
    ) -> list[list[TextRegion]]:
        if challenger.scope == "page":
            prepared_path = image_path
            if challenger.prepare is not None:
                prepared_path = root / f"{_slug(challenger.name)}-page.png"
                _prepare_view(image, challenger, prepared_path)
            regions = challenger.reader.read(prepared_path, page_number)
            namespaced = [
                _challenger_region(
                    region,
                    challenger.name,
                    page_number,
                    0,
                    index,
                )
                for index, region in enumerate(regions, start=1)
            ]
            return _assign_regions(predictions, namespaced)

        tables = []
        for table_index, crop in enumerate(table_crops, start=1):
            prepared_path = crop.path
            if challenger.prepare is not None:
                prepared_path = root / f"{_slug(challenger.name)}-{table_index}.png"
                try:
                    with Image.open(crop.path) as image:
                        _prepare_view(image, challenger, prepared_path)
                except (OSError, UnidentifiedImageError) as error:
                    raise ReaderError(
                        "table_challenger_image_failed",
                        str(error),
                    ) from error
            regions = challenger.reader.read(prepared_path, page_number)
            tables.append(
                [
                    _challenger_region(
                        region,
                        challenger.name,
                        page_number,
                        table_index,
                        index,
                        crop.offset,
                    )
                    for index, region in enumerate(regions, start=1)
                ]
            )
        return tables

    def _table_region(
        self,
        table_id: str,
        prediction: TablePrediction,
        primary: list[TextRegion],
        challenger_groups: dict[str, list[TextRegion]],
        table_index: int,
        dropped_cells: tuple[TableCell, ...] = (),
        page_regions: Sequence[TextRegion] = (),
    ) -> tuple[TextRegion, list[TextRegion], dict[str, list[TextRegion]]]:
        primary_cells = _assign_cells(prediction.cells, primary)
        challenger_cells = {
            name: _assign_cells(prediction.cells, regions)
            for name, regions in challenger_groups.items()
        }
        challenger_sources = {
            name: _assigned_regions(items) for name, items in challenger_cells.items()
        }
        primary_candidates = [
            _candidate(assigned, "primary") for assigned in primary_cells
        ]
        row_count = (
            max(
                (max(cell.row_nums, default=-1) for cell in prediction.cells),
                default=-1,
            )
            + 1
        )
        column_count = (
            max(
                (max(cell.column_nums, default=-1) for cell in prediction.cells),
                default=-1,
            )
            + 1
        )
        overlay = _text_grid_overlay(
            page_regions or primary, prediction, row_count, column_count
        )
        # A word token's parent is consumed by the table only when the grids agree.
        # On disagreement the words still give cells text and confidence, but the
        # parent stays a region of its own so a reviewer sees both readings.
        primary_sources = _collapse_word_sources(
            _assigned_regions(primary_cells),
            primary,
            consumed_ids=(
                {overlay.region.id}
                if overlay is not None and overlay.agreement
                else frozenset()
            ),
        )
        cells = []
        table_alternatives = []
        for cell_index, cell in enumerate(prediction.cells, start=1):
            primary_candidate = primary_candidates[cell_index - 1]
            challenger_candidates = [
                _candidate(items[cell_index - 1], name)
                for name, items in challenger_cells.items()
            ]
            resolved = _resolve_cell(
                primary_candidate,
                challenger_candidates,
                self.low_primary_confidence,
                _row_peer_candidates(
                    cell_index - 1,
                    prediction.cells,
                    primary_cells,
                ),
                _expected_label_unit(
                    _column_peer_candidates(
                        cell_index - 1,
                        prediction.cells,
                        primary_cells,
                    )
                ),
            )
            if overlay is not None and overlay.agreement:
                resolved = _apply_grid_text(
                    resolved,
                    overlay,
                    cell,
                    primary_candidate,
                    challenger_candidates,
                )
            cell_id = f"{table_id}-cell-{cell_index}"
            cell_alternatives = [
                TextAlternative(
                    text=item.text,
                    confidence=item.confidence,
                    provider=item.source,
                    text_provenance={"evidence_ids": list(item.evidence_ids)},
                )
                for item in resolved["alternatives"]
            ]
            if resolved["resolution"] == "conflicting":
                table_alternatives.extend(
                    TextAlternative(
                        text=f"{cell_id}: {item.text}",
                        confidence=item.confidence,
                        provider=item.source,
                        text_provenance={
                            "cell_id": cell_id,
                            "evidence_ids": list(item.evidence_ids),
                        },
                    )
                    for item in resolved["alternatives"]
                )
            cells.append(
                {
                    "id": cell_id,
                    "bbox": asdict(cell.bounding_box),
                    "row_nums": list(cell.row_nums),
                    "column_nums": list(cell.column_nums),
                    "text": resolved["selected"].text,
                    "source": resolved["selected"].source,
                    "confidence": resolved["selected"].confidence,
                    "resolution": resolved["resolution"],
                    "alternatives": [asdict(item) for item in cell_alternatives],
                    "evidence_ids": list(resolved["selected"].evidence_ids),
                    "supporters": [
                        {
                            "provider": item.source,
                            "evidence_ids": list(item.evidence_ids),
                        }
                        for item in resolved["supporters"]
                    ],
                    "column_header": cell.column_header,
                    "projected_row_header": cell.projected_row_header,
                    "span_bboxes": [asdict(box) for box in cell.span_boxes],
                    "decision": resolved["decision"],
                }
            )

        _resolve_structural_blank_corner(cells, self.extractor.name)

        reading_order = min(
            (region.reading_order for region in primary_sources), default=table_index
        )
        confidence_values = [
            cell["confidence"] for cell in cells if cell["confidence"] is not None
        ]
        confidence = (
            sum(confidence_values) / len(confidence_values)
            if confidence_values
            else prediction.confidence
        )
        table = TextRegion(
            id=table_id,
            kind="table",
            text=_markdown(cells, row_count, column_count),
            confidence=confidence,
            bounding_box=prediction.bounding_box,
            reading_order=reading_order,
            provider=self.extractor.name,
            text_provenance={
                "method": "geometry_first_table_fusion",
                "source_region_ids": [region.id for region in primary_sources],
            },
            resolution="resolved",
            alternatives=table_alternatives,
            structure={
                "role": "table",
                "row_count": row_count,
                "column_count": column_count,
                "cells": cells,
                "model": copy.deepcopy(prediction.model),
                "detection_confidence": prediction.confidence,
                **(
                    {
                        "text_grid": {
                            "region_id": overlay.region.id,
                            "row_count": overlay.shape[0],
                            "column_count": overlay.shape[1],
                            "agreement": overlay.agreement,
                        }
                    }
                    if overlay is not None
                    else {}
                ),
                **(
                    {
                        "dropped_cells": [
                            {
                                "bounding_box": asdict(cell.bounding_box),
                                "row_nums": list(cell.row_nums),
                                "column_nums": list(cell.column_nums),
                                "reason": "cell_position_collision",
                            }
                            for cell in dropped_cells
                        ]
                    }
                    if dropped_cells
                    else {}
                ),
            },
        )
        if overlay is not None:
            if overlay.agreement:
                # The grid's text now lives in the table's cells; the source region is
                # marked so the same table does not render twice.
                if all(region.id != overlay.region.id for region in primary_sources):
                    primary_sources.append(overlay.region)
            else:
                # Two models disagree on the grid's shape. Force-fitting one onto the
                # other invents structure, so both regions stay, flagged for review.
                structure = dict(overlay.region.structure or {})
                structure["grid_disagreement"] = {
                    "table_id": table_id,
                    "table_shape": [row_count, column_count],
                    "own_shape": list(overlay.shape),
                }
                overlay.region.structure = structure
        return table, primary_sources, challenger_sources


class TatrTableExtractor:
    """Lazy adapter around Microsoft's official Table Transformer pipeline."""

    name = "table-transformer"

    def __init__(
        self,
        source_root: Path,
        detection_model_path: Path,
        structure_model_path: Path,
        *,
        device: str = "cuda",
        crop_padding: int = 5,
        minimum_detection_score: float = 0.5,
        detection_id: str = DEFAULT_DETECTION_ID,
        detection_revision: str = DEFAULT_DETECTION_REVISION,
        structure_id: str = DEFAULT_STRUCTURE_ID,
        structure_revision: str = DEFAULT_STRUCTURE_REVISION,
        source_revision: str = DEFAULT_SOURCE_REVISION,
        enable_ruled_table_proposals: bool = False,
        pipeline: object | None = None,
        proposal_detector: object | None = None,
        structure_fallback: object | None = None,
    ) -> None:
        if crop_padding < 0:
            raise ValueError("crop_padding must be non-negative")
        if not 0 <= minimum_detection_score <= 1:
            raise ValueError("minimum_detection_score must be from 0 to 1")
        provenance = (
            detection_id,
            detection_revision,
            structure_id,
            structure_revision,
            source_revision,
        )
        if any(not value.strip() for value in provenance):
            raise ValueError("TATR provenance fields must be non-empty")
        self.source_root = Path(source_root)
        self.detection_model_path = Path(detection_model_path)
        self.structure_model_path = Path(structure_model_path)
        self.device = device
        self.crop_padding = crop_padding
        self.minimum_detection_score = minimum_detection_score
        self.detection_id = detection_id
        self.detection_revision = detection_revision
        self.structure_id = structure_id
        self.structure_revision = structure_revision
        self.source_revision = source_revision
        self.enable_ruled_table_proposals = enable_ruled_table_proposals
        # A general layout detector proposes crops TATR's training distribution lacks
        # (full-page financial layouts); the structure fallback replaces a TATR grid
        # that came back empty or with invalid cell topology.
        self.proposal_detector = proposal_detector
        self.structure_fallback = structure_fallback
        self._pipeline = pipeline
        self._lock = threading.Lock()

    def extract(
        self,
        image_path: Path,
        tokens: list[dict[str, Any]],
        layout_boxes: Sequence[BoundingBox] = (),
        word_boxes: Sequence[dict[str, Any]] = (),
    ) -> list[TablePrediction]:
        try:
            with Image.open(image_path) as source:
                image = source.convert("RGB")
        except (OSError, UnidentifiedImageError) as error:
            raise ReaderError("table_image_failed", str(error)) from error

        with self._lock:
            pipeline = self._get_pipeline()
            try:
                detection = pipeline.detect(
                    image,
                    tokens=copy.deepcopy(tokens),
                    out_objects=True,
                    out_crops=True,
                    crop_padding=self.crop_padding,
                )
            except Exception as error:
                raise ReaderError("table_detection_failed", str(error)) from error
            objects, crops = self._detection_outputs(detection, pipeline)
            objects, crops = _suppress_contained_detections(
                objects,
                crops,
                image.size,
            )
            recognition_inputs = [
                (obj, crop, None, False)
                for obj, crop in zip(objects, crops, strict=True)
            ]
            if self.enable_ruled_table_proposals:
                proposals = _ruled_table_proposals(image)
                recognition_inputs = _merge_ruled_proposals(
                    image,
                    tokens,
                    recognition_inputs,
                    proposals,
                )
            # Layout-first fallback: the page reader's layout model fires on tables the
            # table detector's training distribution lacks (a full-page financial
            # layout), so its table regions become crops when no detection covers them.
            for box in _novel_layout_boxes(layout_boxes, recognition_inputs, image.size):
                recognition_inputs.append(
                    _layout_recognition_input(image, tokens, box)
                )
            detector_boxes: list[BoundingBox] = []
            if self.proposal_detector is not None:
                try:
                    detector_boxes = self.proposal_detector.detect(image)
                except ReaderError:
                    detector_boxes = []
                for box in _novel_layout_boxes(
                    detector_boxes, recognition_inputs, image.size
                ):
                    recognition_inputs.append(
                        _layout_recognition_input(
                            image, tokens, box, source=self.proposal_detector.name
                        )
                    )
            predictions = []
            for obj, crop, proposal, proposal_is_detection in recognition_inputs:
                try:
                    recognized = pipeline.recognize(
                        crop["image"],
                        crop["tokens"],
                        out_objects=True,
                        out_cells=True,
                    )
                except Exception as error:
                    raise ReaderError("table_structure_failed", str(error)) from error
                fallback_padding = 0 if proposal_is_detection else self.crop_padding
                cells = self._recognized_cells(
                    recognized,
                    obj,
                    image.size,
                    crop["image"].size,
                    crop_padding=fallback_padding,
                )
                structure_fallback_trigger = None
                fallback_attempted = False
                if self.structure_fallback is not None and not cells:
                    fallback_attempted = True
                    fallback = self._fallback_cells(
                        crop, obj, image.size, crop_padding=fallback_padding
                    )
                    if fallback:
                        cells = fallback
                        structure_fallback_trigger = "tatr structure returned no cells"
                if proposal_is_detection and not cells:
                    continue
                model = self.model_provenance()
                confidence = _optional_confidence(obj.get("score"))
                if obj.get("layout_proposal"):
                    model["proposal"] = {
                        "source": obj.get("proposal_source", LAYOUT_PROPOSAL_SOURCE),
                        "detection_confidence_calibrated": False,
                        "used_for_detection": True,
                    }
                if proposal is not None:
                    cells, proposal_metadata = _repair_ruled_grid(
                        cells,
                        proposal,
                        proposal_is_detection=proposal_is_detection,
                    )
                    if proposal_is_detection and not cells:
                        continue
                    if proposal_is_detection:
                        confidence = None
                    model["proposal"] = proposal_metadata
                parent_conflicts = _table_topology_conflicts(cells)
                if (
                    self.structure_fallback is not None
                    and parent_conflicts
                    and structure_fallback_trigger is None
                ):
                    fallback_attempted = True
                    fallback = self._fallback_cells(
                        crop, obj, image.size, crop_padding=fallback_padding
                    )
                    if fallback:
                        cells = fallback
                        structure_fallback_trigger = (
                            f"tatr structure had {parent_conflicts} topology conflicts"
                        )
                        parent_conflicts = 0
                if structure_fallback_trigger is not None:
                    model["structure"] = {
                        **self.structure_fallback.model_provenance(),
                        "replaced": model["structure"],
                        "fallback_reason": structure_fallback_trigger,
                    }
                if (
                    (parent_conflicts or _near_page_trivial_grid(cells, obj, image))
                    and proposal is None
                    and detector_boxes
                ):
                    # A conflicted detection is often several stacked tables read as
                    # one - and so is a near-page detection whose grid came back
                    # trivially small (falcon token variance flips TATR between the
                    # two across processes). The layout detector's boxes inside it
                    # are the panels; each gets its own structure pass. Novelty
                    # filtering skipped these very boxes because the dead detection
                    # covered them.
                    recovered = self._recover_detector_panels(
                        pipeline,
                        image,
                        tokens,
                        obj,
                        parent_conflicts,
                        detector_boxes,
                        word_boxes,
                    )
                    if recovered:
                        predictions.extend(recovered)
                        continue
                if (
                    self.enable_ruled_table_proposals
                    and proposal is None
                    and parent_conflicts
                ):
                    recovered = self._recover_horizontal_panels(
                        pipeline,
                        image,
                        tokens,
                        obj,
                        parent_conflicts,
                    )
                    if recovered:
                        predictions.extend(recovered)
                        continue
                if not fallback_attempted:
                    cells, model = self._arbitrate_structure(
                        cells,
                        model,
                        crop,
                        obj,
                        image.size,
                        fallback_padding,
                        list(word_boxes) or tokens,
                    )
                predictions.append(
                    TablePrediction(
                        bounding_box=_bounded_box(obj["bbox"], image.size),
                        cells=tuple(cells),
                        confidence=confidence,
                        model=model,
                    )
                )
        return predictions

    def _arbitrate_structure(
        self,
        cells: list[TableCell],
        model: dict[str, Any],
        crop: dict[str, Any],
        obj: dict[str, Any],
        page_size: tuple[int, int],
        crop_padding: int,
        scoring_tokens: Sequence[dict[str, Any]],
    ) -> tuple[list[TableCell], dict[str, Any]]:
        """A valid but coarse TATR grid loses to a fallback grid its tokens fill better.

        The academic-paper author block: TATR's grid is topologically valid, but its
        merged cells run whole author columns together. Arbitration is gated to the
        suspect grids because the fallback costs seconds per crop on CPU.
        """
        if self.structure_fallback is None:
            return cells, model
        table_box = _bounded_box(obj["bbox"], page_size)
        table_tokens = [
            token
            for token in scoring_tokens
            if _valid_box(token.get("bbox"))
            and _centre_in_box(token["bbox"], table_box)
        ]
        if not _needs_structure_arbitration(cells, table_tokens):
            return cells, model
        challenger = self._fallback_cells(
            crop, obj, page_size, crop_padding=crop_padding
        )
        # TATR wins ties: strict tuple comparison on (filled cells, covered tokens).
        if not challenger or _grid_fill(challenger, table_tokens) <= _grid_fill(
            cells, table_tokens
        ):
            return cells, model
        model = {
            **model,
            "structure": {
                **self.structure_fallback.model_provenance(),
                "replaced": model["structure"],
                "fallback_reason": "table tokens fill this grid better than tatr's",
            },
        }
        return challenger, model

    def _recover_detector_panels(
        self,
        pipeline: object,
        image: Image.Image,
        tokens: list[dict[str, Any]],
        parent: dict[str, Any],
        parent_conflicts: int,
        detector_boxes: Sequence[BoundingBox],
        word_boxes: Sequence[dict[str, Any]] = (),
    ) -> list[TablePrediction]:
        parent_box = _bounded_box(parent["bbox"], image.size)
        contained = [
            box
            for box in detector_boxes
            if _box_area(box) > 0
            and _intersection_area(box, parent_box) / _box_area(box) >= 0.7
        ]
        # The detector can propose two near-identical boxes for one table; the
        # larger one stands for both. Selection greedily favours the larger box,
        # the surviving panels keep the detector's order.
        kept: list[BoundingBox] = []
        for box in sorted(contained, key=_box_area, reverse=True):
            if any(
                _intersection_area(box, other) / _box_area(box) >= 0.6
                for other in kept
            ):
                continue
            kept.append(box)
        panels = [box for box in contained if box in kept]
        if len(panels) < 2:
            return []
        # Word geometry scores the candidate grids; region-level tokens are far too
        # coarse to tell a 4x1 band grid from the real one.
        scoring_tokens = list(word_boxes) or tokens
        recovered = []
        for panel_index, panel in enumerate(panels, start=1):
            detection = {
                "label": parent["label"],
                "score": None,
                "bbox": [panel.left, panel.top, panel.right, panel.bottom],
            }
            crop = {
                "image": image.crop((panel.left, panel.top, panel.right, panel.bottom)),
                "tokens": _tokens_for_proposal(tokens, panel),
            }
            try:
                recognized = pipeline.recognize(
                    crop["image"],
                    crop["tokens"],
                    out_objects=True,
                    out_cells=True,
                )
                tatr_cells = self._recognized_cells(
                    recognized,
                    detection,
                    image.size,
                    crop["image"].size,
                    crop_padding=0,
                )
            except Exception:
                tatr_cells = []
            # Both structure models parse the panel; the grid the panel's own tokens
            # fill best wins. TATR read the financial panel as a valid but useless
            # 4x1 band grid, which token fill scores far below the real 17x5.
            panel_tokens = [
                token
                for token in scoring_tokens
                if _valid_box(token.get("bbox"))
                and _centre_in_box(token["bbox"], panel)
            ]
            candidates: list[tuple[list[TableCell], dict[str, Any] | None]] = []
            tatr_cells = _conflict_free_cells(tatr_cells, panel)
            if tatr_cells:
                candidates.append((tatr_cells, None))
            if self.structure_fallback is not None:
                fallback = self._fallback_cells(
                    crop, detection, image.size, crop_padding=0
                )
                if fallback:
                    candidates.append(
                        (fallback, self.structure_fallback.model_provenance())
                    )
            if not candidates:
                # An invalid panel is dropped, not the whole decomposition: the
                # panels are independent tables, and the reader's own table region
                # still carries this panel's content as evidence.
                continue
            cells, structure_note = max(
                candidates,
                key=lambda item: _grid_fill(item[0], panel_tokens),
            )
            model = self.model_provenance()
            if structure_note is not None:
                model["structure"] = {
                    **structure_note,
                    "replaced": model["structure"],
                    "fallback_reason": (
                        "panel tokens fill this grid better than tatr's"
                    ),
                }
            model["proposal"] = {
                "source": f"{self.proposal_detector.name}_panel_decomposition",
                "detection_confidence_calibrated": False,
                "used_for_detection": True,
                "parent_bbox": asdict(parent_box),
                "parent_detection_confidence": _optional_confidence(
                    parent.get("score")
                ),
                "parent_topology_conflicts": parent_conflicts,
                "panel_index": panel_index,
                "panel_count": len(panels),
            }
            recovered.append(
                TablePrediction(
                    bounding_box=panel,
                    cells=tuple(cells),
                    confidence=None,
                    model=model,
                )
            )
        return recovered

    def _recover_horizontal_panels(
        self,
        pipeline: object,
        image: Image.Image,
        tokens: list[dict[str, Any]],
        parent: dict[str, Any],
        parent_conflicts: int,
    ) -> list[TablePrediction]:
        parent_box = _bounded_box(parent["bbox"], image.size)
        panels = _horizontal_table_panels(image, parent_box)
        if len(panels) < 2:
            return []

        recovered = []
        for panel_index, panel in enumerate(panels, start=1):
            detection = {
                "label": parent["label"],
                "score": None,
                "bbox": [panel.left, panel.top, panel.right, panel.bottom],
            }
            crop = image.crop((panel.left, panel.top, panel.right, panel.bottom))
            panel_tokens = _tokens_for_proposal(tokens, panel)
            try:
                recognized = pipeline.recognize(
                    crop,
                    panel_tokens,
                    out_objects=True,
                    out_cells=True,
                )
                cells = self._recognized_cells(
                    recognized,
                    detection,
                    image.size,
                    crop.size,
                    crop_padding=0,
                )
            except Exception:
                return []
            if not _usable_panel_grid(cells):
                return []
            core = self._reparse_numeric_core(
                pipeline,
                image,
                tokens,
                panel,
                cells,
            )
            core_metadata = None
            if core is not None:
                panel, cells, core_metadata = core
            model = self.model_provenance()
            model["proposal"] = {
                "source": HORIZONTAL_PANEL_PROPOSAL_SOURCE,
                "detection_confidence_calibrated": False,
                "used_for_detection": True,
                "parent_bbox": asdict(parent_box),
                "parent_detection_confidence": _optional_confidence(
                    parent.get("score")
                ),
                "parent_topology_conflicts": parent_conflicts,
                "panel_index": panel_index,
                "panel_count": len(panels),
            }
            if core_metadata is not None:
                model["proposal"]["column_core"] = core_metadata
            recovered.append(
                TablePrediction(
                    bounding_box=panel,
                    cells=tuple(cells),
                    confidence=None,
                    model=model,
                )
            )
        return recovered

    def _reparse_numeric_core(
        self,
        pipeline: object,
        image: Image.Image,
        tokens: list[dict[str, Any]],
        panel: BoundingBox,
        cells: Sequence[TableCell],
    ) -> tuple[BoundingBox, list[TableCell], dict[str, Any]] | None:
        core = _numeric_column_core(cells, panel)
        if core is None:
            return None
        core_box, expected_columns = core
        crop = image.crop(
            (core_box.left, core_box.top, core_box.right, core_box.bottom)
        )
        detection = {
            "label": "table",
            "score": None,
            "bbox": [core_box.left, core_box.top, core_box.right, core_box.bottom],
        }
        core_tokens = _tokens_for_proposal(tokens, core_box)
        try:
            recognized = pipeline.recognize(
                crop,
                core_tokens,
                out_objects=True,
                out_cells=True,
            )
            reparsed = self._recognized_cells(
                recognized,
                detection,
                image.size,
                crop.size,
                crop_padding=0,
            )
        except Exception:
            return None
        if not _usable_panel_grid(reparsed):
            return None
        row_count, column_count = _table_shape(reparsed)
        parent_rows, parent_columns = _table_shape(cells)
        if column_count != expected_columns or row_count < parent_rows:
            return None
        aligned = _repair_dense_numeric_rows(reparsed, core_tokens, core_box)
        row_alignment = None
        if aligned is not None:
            reparsed, row_alignment = aligned
        return (
            core_box,
            reparsed,
            {
                "source": NUMERIC_COLUMN_CORE_SOURCE,
                "parent_column_count": parent_columns,
                "column_count": column_count,
                **({"row_alignment": row_alignment} if row_alignment else {}),
            },
        )

    def _fallback_cells(
        self,
        crop: dict[str, Any],
        obj: dict[str, Any],
        page_size: tuple[int, int],
        *,
        crop_padding: int,
    ) -> list[TableCell]:
        """Structure from the fallback model on the same crop, tokens slotted by us.

        The fallback predicts geometry only; its own token matcher is never used
        (docling-ibm-models issue 188: it fabricates cell text). A crop token belongs
        to the cell containing its centre, the same containment rule the official
        TATR postprocess applies, and a token inside no cell is left for the stage's
        word-evidence recovery rather than snapped to a nearest column.
        """
        try:
            predicted = self.structure_fallback.predict(crop["image"])
        except ReaderError:
            return []
        rotated = obj["label"] == "table rotated"
        cells = []
        for cell in predicted:
            box = cell.bounding_box
            spans = [
                token
                for token in crop["tokens"]
                if _valid_box(token.get("bbox"))
                and _centre_in_box(token["bbox"], box)
                and str(token.get("text", "")).strip()
            ]
            try:
                translated = _translate_cell_box(
                    [box.left, box.top, box.right, box.bottom],
                    obj["bbox"],
                    crop_padding,
                    rotated,
                    page_size,
                    crop["image"].size,
                )
                span_boxes = tuple(
                    _translate_cell_box(
                        token["bbox"],
                        obj["bbox"],
                        crop_padding,
                        rotated,
                        page_size,
                        crop["image"].size,
                    )
                    for token in spans
                )
            except ReaderError:
                continue
            cells.append(
                TableCell(
                    bounding_box=translated,
                    row_nums=tuple(cell.row_nums),
                    column_nums=tuple(cell.column_nums),
                    column_header=cell.column_header,
                    projected_row_header=cell.projected_row_header,
                    span_boxes=span_boxes,
                    span_texts=tuple(
                        str(token.get("text", "")).strip() for token in spans
                    ),
                )
            )
        return _conflict_free_cells(cells, _bounded_box(obj["bbox"], page_size))

    def model_provenance(self) -> dict[str, Any]:
        return {
            "origin": MODEL_ORIGIN,
            "license": MODEL_LICENSE,
            "source_revision": self.source_revision,
            "detection": {
                "id": self.detection_id,
                "revision": self.detection_revision,
                "path": str(self.detection_model_path),
            },
            "structure": {
                "id": self.structure_id,
                "revision": self.structure_revision,
                "path": str(self.structure_model_path),
            },
        }

    def _get_pipeline(self) -> object:
        if self._pipeline is not None:
            return self._pipeline
        source = _source_dir(self.source_root)
        required = (
            source / "inference.py",
            source / "detection_config.json",
            source / "structure_config.json",
            self.detection_model_path,
            self.structure_model_path,
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise ReaderError(
                "table_model_unavailable",
                f"Missing Table Transformer files: {', '.join(missing)}",
            )
        try:
            inference = _load_inference(self.source_root)
            self._pipeline = inference.TableExtractionPipeline(
                det_device=self.device,
                str_device=self.device,
                det_config_path=source / "detection_config.json",
                det_model_path=self.detection_model_path,
                str_config_path=source / "structure_config.json",
                str_model_path=self.structure_model_path,
            )
        except ReaderError:
            raise
        except Exception as error:
            raise ReaderError("table_model_unavailable", str(error)) from error
        return self._pipeline

    def _detection_outputs(
        self, value: object, pipeline: object
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if not isinstance(value, dict):
            raise ReaderError(
                "invalid_table_output", "Detection output is not an object"
            )
        raw_objects = value.get("objects")
        crops = value.get("crops")
        if not isinstance(raw_objects, list) or not isinstance(crops, list):
            raise ReaderError(
                "invalid_table_output", "Detection output lacks objects or crops"
            )
        thresholds = getattr(pipeline, "det_class_thresholds", {})
        upstream_objects = []
        for obj in raw_objects:
            if not isinstance(obj, dict) or obj.get("label") not in {
                "table",
                "table rotated",
            }:
                continue
            score = _optional_confidence(obj.get("score"))
            if score is None:
                raise ReaderError(
                    "invalid_table_output", "A table detection has no confidence"
                )
            if score >= float(thresholds.get(obj["label"], 0)):
                upstream_objects.append(obj)
        if len(upstream_objects) != len(crops):
            raise ReaderError(
                "invalid_table_output",
                "Table Transformer returned mismatched detections and crops",
            )
        if any(
            not isinstance(crop, dict)
            or "image" not in crop
            or not isinstance(crop.get("tokens"), list)
            for crop in crops
        ):
            raise ReaderError("invalid_table_output", "A table crop is invalid")
        paired = [
            (obj, crop)
            for obj, crop in zip(upstream_objects, crops, strict=True)
            if float(obj["score"]) >= self.minimum_detection_score
        ]
        if any(not _valid_box(obj.get("bbox")) for obj, _ in paired):
            raise ReaderError("invalid_table_output", "A table detection has no box")
        return [obj for obj, _ in paired], [crop for _, crop in paired]

    def _recognized_cells(
        self,
        value: object,
        detection: dict[str, Any],
        page_size: tuple[int, int],
        crop_size: tuple[int, int],
        *,
        crop_padding: int | None = None,
    ) -> list[TableCell]:
        if not isinstance(value, dict) or not isinstance(value.get("cells"), list):
            raise ReaderError("invalid_table_output", "Structure output lacks cells")
        tables = value["cells"]
        if len(tables) != 1 or not isinstance(tables[0], list):
            raise ReaderError(
                "invalid_table_output", "Structure output must contain one table"
            )
        cells = []
        for cell in tables[0]:
            if not isinstance(cell, dict) or not _valid_box(cell.get("bbox")):
                raise ReaderError("invalid_table_output", "A table cell has no box")
            rows = _indexes(cell.get("row_nums"))
            columns = _indexes(cell.get("column_nums"))
            if not rows or not columns:
                raise ReaderError(
                    "invalid_table_output", "A table cell has no grid span"
                )
            box = _translate_cell_box(
                cell["bbox"],
                detection["bbox"],
                self.crop_padding if crop_padding is None else crop_padding,
                detection["label"] == "table rotated",
                page_size,
                crop_size,
            )
            spans = cell.get("spans")
            spans = spans if isinstance(spans, list) else []
            cells.append(
                TableCell(
                    bounding_box=box,
                    row_nums=rows,
                    column_nums=columns,
                    column_header=bool(cell.get("column header", False)),
                    projected_row_header=bool(cell.get("projected row header", False)),
                    span_boxes=tuple(
                        _translate_cell_box(
                            span["bbox"],
                            detection["bbox"],
                            self.crop_padding if crop_padding is None else crop_padding,
                            detection["label"] == "table rotated",
                            page_size,
                            crop_size,
                        )
                        for span in spans
                        if isinstance(span, dict) and _valid_box(span.get("bbox"))
                    ),
                    span_texts=tuple(
                        str(span.get("text", "")).strip()
                        for span in spans
                        if isinstance(span, dict) and str(span.get("text", "")).strip()
                    ),
                )
            )
        return cells


def _ruled_table_proposals(image: Image.Image) -> list[_RuledTableProposal]:
    import cv2
    import numpy as np

    gray = np.asarray(image.convert("L"), dtype=np.uint8).copy()
    height, width = gray.shape
    horizontal_length = max(12, round(width * 0.03))
    vertical_length = max(12, round(height * 0.03))
    masks = [
        cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )[1],
        cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY_INV)[1],
    ]
    proposals = []
    for mask in masks:
        horizontal = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_length, 1)),
        )
        vertical = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_length)),
        )
        combined = cv2.bitwise_or(horizontal, vertical)
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            combined,
            connectivity=8,
        )
        for label in range(1, component_count):
            left, top, box_width, box_height, _ = stats[label]
            if box_width < 20 or box_height < 20:
                continue
            right = left + box_width
            bottom = top + box_height
            component_horizontal = horizontal[top:bottom, left:right]
            component_vertical = vertical[top:bottom, left:right]
            column_edges = _line_centers(
                np.count_nonzero(component_vertical, axis=0),
                box_height * 0.55,
                left,
            )
            if len(column_edges) < 3:
                continue
            row_edges = tuple(
                edge
                for edge in _line_centers(
                    np.count_nonzero(component_horizontal, axis=1),
                    box_width * 0.35,
                    top,
                    gap=max(3, round(height * 0.004)),
                )
                if _crosses_separators(
                    component_vertical,
                    edge - top,
                    tuple(column - left for column in column_edges),
                    margin=max(2, round(height * 0.002)),
                )
            )
            if len(row_edges) < 3:
                continue
            box = BoundingBox(
                column_edges[0],
                row_edges[0],
                column_edges[-1],
                row_edges[-1],
            )
            if _reject_ruled_box(box, image.size):
                continue
            proposals.append(_RuledTableProposal(box, row_edges, column_edges))
    return _dedupe_ruled_proposals(proposals)


def _horizontal_table_panels(
    image: Image.Image,
    parent: BoundingBox,
) -> list[BoundingBox]:
    import cv2
    import numpy as np

    crop = np.asarray(
        image.crop((parent.left, parent.top, parent.right, parent.bottom)).convert("L"),
        dtype=np.uint8,
    ).copy()
    height, width = crop.shape
    if height < 3 or width < 3:
        return []

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(20, round(width * 0.15)), 1),
    )
    rows = []
    for mask in (
        cv2.threshold(
            crop,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )[1],
        cv2.threshold(crop, 235, 255, cv2.THRESH_BINARY_INV)[1],
    ):
        horizontal = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        rows.extend(
            _line_centers(
                np.count_nonzero(horizontal, axis=1),
                width * 0.75,
                parent.top,
                gap=max(2, round(height * 0.002)),
            )
        )
    boundaries = _merge_positions(rows, max(2, round(height * 0.003)))
    if len(boundaries) < 3:
        return []

    minimum_panel_height = max(20, round(height * 0.08))
    if any(
        lower - upper < minimum_panel_height
        for upper, lower in zip(boundaries, boundaries[1:])
    ):
        return []

    edge_margin = max(4, round(height * 0.03))
    first_is_edge = boundaries[0] - parent.top <= edge_margin
    top = parent.top if first_is_edge else boundaries[0]
    bottom = (
        boundaries[-1]
        if parent.bottom - boundaries[-1] <= edge_margin
        else parent.bottom
    )
    separator_candidates = boundaries[1:] if first_is_edge else boundaries
    internal = tuple(
        boundary for boundary in separator_candidates if top < boundary < bottom
    )
    edges = (top, *internal, bottom)
    panels = [
        BoundingBox(parent.left, panel_top, parent.right, panel_bottom)
        for panel_top, panel_bottom in zip(edges, edges[1:])
        if panel_bottom - panel_top >= minimum_panel_height
    ]
    return panels if len(panels) == len(edges) - 1 else []


def _merge_positions(values: Sequence[int], tolerance: int) -> tuple[int, ...]:
    if not values:
        return ()
    groups = [[value] for value in sorted(set(values))]
    merged = [groups[0]]
    for group in groups[1:]:
        if group[0] <= merged[-1][-1] + tolerance:
            merged[-1].extend(group)
        else:
            merged.append(group)
    return tuple(round(sum(group) / len(group)) for group in merged)


def _usable_panel_grid(cells: Sequence[TableCell]) -> bool:
    if _table_topology_conflicts(cells):
        return False
    rows = {row for cell in cells for row in cell.row_nums}
    columns = {column for cell in cells for column in cell.column_nums}
    return len(rows) >= 2 and len(columns) >= 2


def _numeric_column_core(
    cells: Sequence[TableCell],
    panel: BoundingBox,
) -> tuple[BoundingBox, int] | None:
    _, column_count = _table_shape(cells)
    texts_by_column: dict[int, list[str]] = {
        column: [] for column in range(column_count)
    }
    for cell in cells:
        if len(cell.column_nums) != 1:
            continue
        text = " ".join(cell.span_texts).strip()
        if text:
            texts_by_column[cell.column_nums[0]].append(text)

    numeric_columns = []
    for column in range(column_count):
        texts = texts_by_column[column]
        numeric = sum(_is_numeric_cell_text(text) for text in texts)
        numeric_columns.append(
            len(texts) >= MIN_NUMERIC_CORE_FILLED_CELLS and numeric / len(texts) >= 0.6
        )

    runs: list[tuple[int, int]] = []
    start = None
    for column, numeric in enumerate((*numeric_columns, False)):
        if numeric and start is None:
            start = column
        elif not numeric and start is not None:
            runs.append((start, column))
            start = None
    if not runs:
        return None
    first_numeric, after_numeric = max(
        runs,
        key=lambda run: (run[1] - run[0], -run[0]),
    )
    if after_numeric - first_numeric < MIN_NUMERIC_CORE_COLUMNS:
        return None

    first_column = max(0, first_numeric - 1)
    if first_column == 0 and after_numeric == column_count:
        return None
    selected = [
        cell
        for cell in cells
        if min(cell.column_nums) >= first_column
        and max(cell.column_nums) < after_numeric
    ]
    if not selected:
        return None
    left = min(cell.bounding_box.left for cell in selected)
    right = max(cell.bounding_box.right for cell in selected)
    if right <= left or left <= panel.left and right >= panel.right:
        return None
    return (
        BoundingBox(left, panel.top, right, panel.bottom),
        after_numeric - first_column,
    )


def _is_numeric_cell_text(text: str) -> bool:
    normalized = unicodedata.normalize("NFKC", text).strip().casefold()
    if normalized in {"na", "n/a", "nm"}:
        return True
    compact = re.sub(r"[\s$€£¥₹§,+().%#:/-]", "", normalized)
    return bool(compact) and compact.isdigit()


def _table_shape(cells: Sequence[TableCell]) -> tuple[int, int]:
    return (
        max((max(cell.row_nums, default=-1) for cell in cells), default=-1) + 1,
        max((max(cell.column_nums, default=-1) for cell in cells), default=-1) + 1,
    )


def _repair_dense_numeric_rows(
    cells: Sequence[TableCell],
    tokens: Sequence[dict[str, Any]],
    table_box: BoundingBox,
) -> tuple[list[TableCell], dict[str, Any]] | None:
    row_count, column_count = _table_shape(cells)
    if column_count < MIN_NUMERIC_CORE_COLUMNS + 1:
        return None
    column_boxes = _single_column_boxes(cells, column_count)
    if column_boxes is None:
        return None

    numeric_tokens = []
    for token in tokens:
        box = _page_token_box(token, table_box)
        if box is None or not _is_numeric_cell_text(str(token.get("text", ""))):
            continue
        center_x = (box.left + box.right) / 2
        column = next(
            (
                index
                for index, column_box in enumerate(column_boxes[1:], start=1)
                if column_box.left <= center_x <= column_box.right
            ),
            None,
        )
        if column is not None:
            numeric_tokens.append((column, box))
    bands = _aligned_numeric_bands(numeric_tokens)
    if len(bands) < 2:
        return None
    aligned_band_count = len(bands)

    for row in range(row_count):
        row_cells = [
            cell
            for cell in cells
            if cell.row_nums == (row,)
            and len(cell.column_nums) == 1
            and cell.column_nums[0] > 0
        ]
        if len(row_cells) < MIN_ALIGNED_NUMERIC_COLUMNS:
            continue
        if any(
            any(
                cell.bounding_box.top <= center <= cell.bounding_box.bottom
                for cell in row_cells
            )
            for _, _, center in bands
        ):
            continue
        bands.append(
            (
                min(cell.bounding_box.top for cell in row_cells),
                max(cell.bounding_box.bottom for cell in row_cells),
                median(
                    (cell.bounding_box.top + cell.bounding_box.bottom) / 2
                    for cell in row_cells
                ),
            )
        )
    bands.sort(key=lambda band: band[2])

    rows_by_band = []
    for _, _, center in bands:
        matching_rows = {
            min(cell.row_nums)
            for cell in cells
            if len(cell.row_nums) == 1
            and len(cell.column_nums) == 1
            and cell.column_nums[0] > 0
            and cell.bounding_box.top <= center <= cell.bounding_box.bottom
        }
        rows_by_band.append(min(matching_rows) if matching_rows else None)
    merged_numeric_bands = len([row for row in rows_by_band if row is not None]) != len(
        {row for row in rows_by_band if row is not None}
    )
    label_only_projection = any(
        cell.projected_row_header
        and len(cell.column_nums) == column_count
        and not any(
            cell.bounding_box.top <= center <= cell.bounding_box.bottom
            for _, _, center in bands
        )
        for cell in cells
    )
    if not merged_numeric_bands and not label_only_projection:
        return None

    table_top = min(cell.bounding_box.top for cell in cells)
    table_bottom = max(cell.bounding_box.bottom for cell in cells)
    centers = [center for _, _, center in bands]
    edges = [table_top]
    edges.extend(
        round((upper + lower) / 2) for upper, lower in zip(centers, centers[1:])
    )
    edges.append(table_bottom)

    for cell in cells:
        matching = [
            index
            for index, center in enumerate(centers)
            if cell.bounding_box.top <= center <= cell.bounding_box.bottom
        ]
        is_label_cell = cell.column_nums == (0,)
        is_label_projection = (
            cell.projected_row_header and len(cell.column_nums) == column_count
        )
        if is_label_cell and len(matching) == 1:
            index = matching[0]
            if index > 0:
                edges[index] = min(edges[index], cell.bounding_box.top)
            if index + 1 < len(edges) - 1:
                edges[index + 1] = max(edges[index + 1], cell.bounding_box.bottom)
        elif is_label_projection and not matching:
            following = next(
                (
                    index
                    for index, center in enumerate(centers)
                    if center >= cell.bounding_box.bottom
                ),
                None,
            )
            if following is not None:
                edges[following] = min(edges[following], cell.bounding_box.top)

    if any(lower <= upper for upper, lower in zip(edges, edges[1:])):
        return None
    header_rows = {
        index
        for index, center in enumerate(centers)
        if any(
            cell.column_header
            and cell.bounding_box.top <= center <= cell.bounding_box.bottom
            for cell in cells
        )
    }
    repaired = [
        TableCell(
            bounding_box=BoundingBox(
                column_box.left,
                edges[row],
                column_box.right,
                edges[row + 1],
            ),
            row_nums=(row,),
            column_nums=(column,),
            column_header=row in header_rows,
        )
        for row in range(len(bands))
        for column, column_box in enumerate(column_boxes)
    ]
    if not _usable_panel_grid(repaired):
        return None
    return repaired, {
        "source": ALIGNED_NUMERIC_ROWS_SOURCE,
        "model_row_count": row_count,
        "aligned_band_count": aligned_band_count,
        "row_count": len(bands),
    }


def _single_column_boxes(
    cells: Sequence[TableCell],
    column_count: int,
) -> list[BoundingBox] | None:
    boxes = []
    for column in range(column_count):
        matches = [cell.bounding_box for cell in cells if cell.column_nums == (column,)]
        if not matches:
            return None
        boxes.append(
            BoundingBox(
                round(median(box.left for box in matches)),
                min(box.top for box in matches),
                round(median(box.right for box in matches)),
                max(box.bottom for box in matches),
            )
        )
    return boxes


def _page_token_box(
    token: dict[str, Any],
    table_box: BoundingBox,
) -> BoundingBox | None:
    if not _valid_box(token.get("bbox")):
        return None
    left, top, right, bottom = _float_box(token["bbox"])
    return BoundingBox(
        math.floor(left + table_box.left),
        math.floor(top + table_box.top),
        math.ceil(right + table_box.left),
        math.ceil(bottom + table_box.top),
    )


def _aligned_numeric_bands(
    tokens: Sequence[tuple[int, BoundingBox]],
) -> list[tuple[int, int, float]]:
    if not tokens:
        return []
    tolerance = max(
        2,
        round(median(box.bottom - box.top for _, box in tokens) * 0.6),
    )
    groups: list[list[tuple[int, BoundingBox]]] = []
    for item in sorted(tokens, key=lambda value: (value[1].top + value[1].bottom) / 2):
        center = (item[1].top + item[1].bottom) / 2
        if not groups:
            groups.append([item])
            continue
        group_centers = [(box.top + box.bottom) / 2 for _, box in groups[-1]]
        if abs(center - median(group_centers)) <= tolerance:
            groups[-1].append(item)
        else:
            groups.append([item])

    bands = []
    for group in groups:
        if len({column for column, _ in group}) < MIN_ALIGNED_NUMERIC_COLUMNS:
            continue
        bands.append(
            (
                min(box.top for _, box in group),
                max(box.bottom for _, box in group),
                median((box.top + box.bottom) / 2 for _, box in group),
            )
        )
    return bands


def _line_centers(
    values: Any,
    minimum: float,
    offset: int,
    *,
    gap: int = 3,
) -> tuple[int, ...]:
    positions = [
        int(position) for position, value in enumerate(values) if value >= minimum
    ]
    if not positions:
        return ()
    groups = [[positions[0]]]
    for position in positions[1:]:
        if position <= groups[-1][-1] + gap:
            groups[-1].append(position)
        else:
            groups.append([position])
    return tuple(int(offset + round(sum(group) / len(group))) for group in groups)


def _crosses_separators(
    vertical: Any,
    row: int,
    columns: tuple[int, ...],
    *,
    margin: int,
) -> bool:
    top = max(0, row - margin)
    bottom = min(vertical.shape[0], row + margin + 1)
    crossings = sum(
        bool(
            vertical[
                top:bottom,
                max(0, column - margin) : column + margin + 1,
            ].any()
        )
        for column in columns
    )
    return crossings / len(columns) >= 0.8


def _reject_ruled_box(
    box: BoundingBox,
    page_size: tuple[int, int],
) -> bool:
    width, height = page_size
    page_area = width * height
    if _box_area(box) < page_area * 0.002:
        return True
    near_page_border = (
        box.left <= width * 0.01
        and box.top <= height * 0.01
        and box.right >= width * 0.99
        and box.bottom >= height * 0.99
    )
    return near_page_border


def _dedupe_ruled_proposals(
    proposals: Sequence[_RuledTableProposal],
) -> list[_RuledTableProposal]:
    ranked = sorted(
        proposals,
        key=lambda proposal: (
            -(len(proposal.row_edges) * len(proposal.column_edges)),
            -_box_area(proposal.bounding_box),
            proposal.bounding_box.top,
            proposal.bounding_box.left,
        ),
    )
    kept = []
    for proposal in ranked:
        if any(
            _boxes_duplicate(proposal.bounding_box, other.bounding_box)
            for other in kept
        ):
            continue
        kept.append(proposal)
    return sorted(
        kept,
        key=lambda proposal: (
            proposal.bounding_box.top,
            proposal.bounding_box.left,
            proposal.bounding_box.bottom,
            proposal.bounding_box.right,
        ),
    )


def _merge_ruled_proposals(
    image: Image.Image,
    tokens: list[dict[str, Any]],
    detections: list[
        tuple[dict[str, Any], dict[str, Any], _RuledTableProposal | None, bool]
    ],
    proposals: Sequence[_RuledTableProposal],
) -> list[tuple[dict[str, Any], dict[str, Any], _RuledTableProposal | None, bool]]:
    kept_detections = []
    selected_proposals: list[_RuledTableProposal] = []
    for item in detections:
        detection_box = _bounded_box(item[0]["bbox"], image.size)
        contained = [
            proposal
            for proposal in proposals
            if _contains_box(detection_box, proposal.bounding_box)
        ]
        if _should_split_detection(detection_box, contained):
            selected_proposals.extend(contained)
        else:
            matched = next(
                (
                    proposal
                    for proposal in contained
                    if _boxes_match(detection_box, proposal.bounding_box)
                ),
                None,
            )
            kept_detections.append((item[0], item[1], matched, False))

    for proposal in proposals:
        if proposal in selected_proposals:
            continue
        if any(
            _boxes_duplicate(
                proposal.bounding_box,
                _bounded_box(item[0]["bbox"], image.size),
            )
            for item in kept_detections
        ):
            continue
        selected_proposals.append(proposal)

    selected_proposals = _dedupe_ruled_proposals(selected_proposals)
    proposal_inputs = [
        _proposal_recognition_input(image, tokens, proposal)
        for proposal in selected_proposals
    ]
    combined = [*kept_detections, *proposal_inputs]
    return sorted(
        combined,
        key=lambda item: (
            _bounded_box(item[0]["bbox"], image.size).top,
            _bounded_box(item[0]["bbox"], image.size).left,
        ),
    )


def _contains_box(outer: BoundingBox, inner: BoundingBox) -> bool:
    area = _box_area(inner)
    return area > 0 and _intersection_area(outer, inner) / area >= 0.95


def _should_split_detection(
    detection_box: BoundingBox,
    proposals: Sequence[_RuledTableProposal],
) -> bool:
    if len(proposals) < 2:
        return False
    ordered = sorted(proposals, key=lambda proposal: proposal.bounding_box.top)
    vertically_separated = any(
        upper.bounding_box.bottom <= lower.bounding_box.top
        for upper, lower in zip(ordered, ordered[1:])
    )
    covered_area = sum(_box_area(proposal.bounding_box) for proposal in proposals)
    meaningful_area = covered_area / _box_area(detection_box) >= 0.15
    return vertically_separated and meaningful_area


def _boxes_duplicate(left: BoundingBox, right: BoundingBox) -> bool:
    smaller = min(_box_area(left), _box_area(right))
    return (
        smaller > 0
        and _intersection_area(left, right) / smaller >= RULED_DUPLICATE_OVERLAP
    )


def _boxes_match(left: BoundingBox, right: BoundingBox) -> bool:
    intersection = _intersection_area(left, right)
    return (
        intersection / _box_area(left) >= 0.8 and intersection / _box_area(right) >= 0.8
    )


def _novel_layout_boxes(
    layout_boxes: Sequence[BoundingBox],
    inputs: Sequence[tuple[dict[str, Any], dict[str, Any], Any, bool]],
    page_size: tuple[int, int],
) -> list[BoundingBox]:
    """Layout table regions no planned table crop substantially covers.

    Majority coverage by an existing crop means the structure model will already read
    that area; below it, the table detector genuinely missed most of the table.
    """
    existing = [_bounded_box(item[0]["bbox"], page_size) for item in inputs]
    novel = []
    for box in layout_boxes:
        area = _box_area(box)
        if area <= 0:
            continue
        if any(
            _intersection_area(box, other) / area >= 0.5 for other in existing
        ):
            continue
        novel.append(box)
        existing.append(box)
    return novel


def _layout_recognition_input(
    image: Image.Image,
    tokens: list[dict[str, Any]],
    box: BoundingBox,
    source: str = LAYOUT_PROPOSAL_SOURCE,
) -> tuple[dict[str, Any], dict[str, Any], None, bool]:
    bounded = _bounded_box([box.left, box.top, box.right, box.bottom], image.size)
    return (
        {
            "label": "table",
            "score": None,
            "bbox": [bounded.left, bounded.top, bounded.right, bounded.bottom],
            "layout_proposal": True,
            "proposal_source": source,
        },
        {
            "image": image.crop(
                (bounded.left, bounded.top, bounded.right, bounded.bottom)
            ),
            "tokens": _tokens_for_proposal(tokens, bounded),
        },
        None,
        True,
    )


def _proposal_recognition_input(
    image: Image.Image,
    tokens: list[dict[str, Any]],
    proposal: _RuledTableProposal,
) -> tuple[dict[str, Any], dict[str, Any], _RuledTableProposal, bool]:
    box = proposal.bounding_box
    crop_tokens = _tokens_for_proposal(tokens, box)
    return (
        {
            "label": "table",
            "score": None,
            "bbox": [box.left, box.top, box.right, box.bottom],
        },
        {
            "image": image.crop((box.left, box.top, box.right, box.bottom)),
            "tokens": crop_tokens,
        },
        proposal,
        True,
    )


def _tokens_for_proposal(
    tokens: list[dict[str, Any]],
    box: BoundingBox,
) -> list[dict[str, Any]]:
    crop_tokens = []
    for source in copy.deepcopy(tokens):
        if not _valid_box(source.get("bbox")):
            continue
        left, top, right, bottom = _float_box(source["bbox"])
        token_box = BoundingBox(
            math.floor(left),
            math.floor(top),
            math.ceil(right),
            math.ceil(bottom),
        )
        token_area = _box_area(token_box)
        if token_area == 0 or _intersection_area(token_box, box) / token_area < 0.5:
            continue
        source["bbox"] = [
            left - box.left,
            top - box.top,
            right - box.left,
            bottom - box.top,
        ]
        crop_tokens.append(source)
    return crop_tokens


def _repair_ruled_grid(
    cells: list[TableCell],
    proposal: _RuledTableProposal,
    *,
    proposal_is_detection: bool,
) -> tuple[list[TableCell], dict[str, Any]]:
    tatr_rows = max((max(cell.row_nums) for cell in cells), default=-1) + 1
    tatr_columns = max((max(cell.column_nums) for cell in cells), default=-1) + 1
    metadata: dict[str, Any] = {
        "source": RULED_PROPOSAL_SOURCE,
        "detection_confidence_calibrated": False,
        "used_for_detection": proposal_is_detection,
        "tatr_row_count": tatr_rows,
        "tatr_column_count": tatr_columns,
    }
    ruled_rows = len(proposal.row_edges) - 1
    ruled_columns = len(proposal.column_edges) - 1
    if (
        tatr_rows <= 0
        or tatr_columns <= 0
        or not _rectangular_tatr_grid(cells, tatr_rows, tatr_columns)
        or (ruled_rows <= tatr_rows and ruled_columns <= tatr_columns)
        or _count_agreement(ruled_rows, tatr_rows) <= 0.7
        or _count_agreement(ruled_columns, tatr_columns) <= 0.7
    ):
        return cells, metadata

    header_row = any(cell.column_header and 0 in cell.row_nums for cell in cells)
    repaired = [
        TableCell(
            BoundingBox(left, top, right, bottom),
            (row,),
            (column,),
            column_header=header_row and row == 0,
        )
        for row, (top, bottom) in enumerate(
            zip(proposal.row_edges, proposal.row_edges[1:])
        )
        for column, (left, right) in enumerate(
            zip(proposal.column_edges, proposal.column_edges[1:])
        )
    ]
    metadata["grid_repair"] = "full_rule_cartesian"
    return repaired, metadata


def _rectangular_tatr_grid(
    cells: Sequence[TableCell],
    rows: int,
    columns: int,
) -> bool:
    positions = []
    for cell in cells:
        if len(cell.row_nums) != 1 or len(cell.column_nums) != 1:
            return False
        positions.append((cell.row_nums[0], cell.column_nums[0]))
    return len(positions) == rows * columns and len(set(positions)) == len(positions)


def _count_agreement(left: int, right: int) -> float:
    return min(left, right) / max(left, right) if left > 0 and right > 0 else 0.0


def _rejected_table_candidate(
    prediction: TablePrediction,
    primary: list[TextRegion],
    page_size: tuple[int, int],
    page_number: int,
    table_index: int,
    provider: str,
) -> TextRegion | None:
    page_area = page_size[0] * page_size[1]
    table_area_ratio = _box_area(prediction.bounding_box) / page_area
    row_count = (
        max((max(cell.row_nums, default=-1) for cell in prediction.cells), default=-1)
        + 1
    )
    column_count = (
        max(
            (max(cell.column_nums, default=-1) for cell in prediction.cells),
            default=-1,
        )
        + 1
    )
    grid_cells = row_count * column_count
    assigned = _assign_cells(prediction.cells, primary)
    occupied_cells = [
        cell
        for cell, evidence in zip(prediction.cells, assigned, strict=True)
        if evidence
    ]
    occupied = {
        (row, column)
        for cell in occupied_cells
        for row in cell.row_nums
        for column in cell.column_nums
    }
    coverage = len(occupied) / grid_cells if grid_cells else 0.0
    columns = set(range(column_count))
    full_width_rows = {
        row
        for cell in prediction.cells
        if set(cell.column_nums) == columns
        for row in cell.row_nums
    }
    independent_columns_by_row: dict[int, set[int]] = {}
    for cell in occupied_cells:
        if len(cell.row_nums) != 1 or len(cell.column_nums) != 1:
            continue
        independent_columns_by_row.setdefault(cell.row_nums[0], set()).add(
            cell.column_nums[0]
        )
    complete_rows = sum(
        row_columns == columns for row_columns in independent_columns_by_row.values()
    )
    low_complexity = (
        row_count < MIN_NEAR_PAGE_ROWS
        or column_count < MIN_NEAR_PAGE_COLUMNS
        or grid_cells < MIN_NEAR_PAGE_CELLS
        or len(prediction.cells) < MIN_NEAR_PAGE_CELLS
    )
    near_page_low_complexity = (
        table_area_ratio >= NEAR_PAGE_TABLE_AREA and low_complexity
    )
    has_explicit_spans = any(cell.span_boxes for cell in prediction.cells)
    single_axis_list = (
        min(row_count, column_count) == 1 and grid_cells >= 3 and not has_explicit_spans
    )
    unsupported_spanning_layout = len(full_width_rows) >= 2 and complete_rows == 0
    topology_conflicts = _table_topology_conflicts(prediction.cells)
    if not prediction.cells or row_count <= 0 or column_count <= 0:
        method = "table_structure_rejection"
        reason = "empty_table_structure"
    elif topology_conflicts:
        method = "table_structure_rejection"
        reason = "invalid_cell_topology"
    elif near_page_low_complexity:
        method = "near_page_low_complexity_rejection"
        reason = "near_page_low_complexity_grid"
    elif single_axis_list:
        method = "table_semantics_rejection"
        reason = "single_axis_list"
    elif unsupported_spanning_layout:
        method = "table_semantics_rejection"
        reason = "unsupported_spanning_layout"
    else:
        return None

    structure = {
        "role": "table_candidate",
        "status": "rejected",
        "reason": reason,
        "row_count": row_count,
        "column_count": column_count,
        "grid_cells": grid_cells,
        "predicted_cells": len(prediction.cells),
        "occupied_cells": len(occupied),
        "cell_coverage": round(coverage, 6),
        "table_area_ratio": round(table_area_ratio, 6),
        "model": copy.deepcopy(prediction.model),
        "detection_confidence": prediction.confidence,
    }
    if method == "table_semantics_rejection":
        structure.update(
            {
                "occupied_predicted_cells": len(occupied_cells),
                "complete_independent_rows": complete_rows,
                "full_width_rows": len(full_width_rows),
            }
        )
    if topology_conflicts:
        structure["topology_conflicts"] = topology_conflicts

    return TextRegion(
        id=f"p{page_number}-tables-candidate-{table_index}",
        kind="table_candidate",
        text="",
        confidence=prediction.confidence,
        bounding_box=prediction.bounding_box,
        reading_order=min(
            (region.reading_order for region in primary), default=table_index
        ),
        provider=provider,
        text_provenance={
            "method": method,
            "source_region_ids": _source_ids(primary),
        },
        resolution="unreadable",
        structure=structure,
    )


def _reject_subgrid_duplicates(
    accepted: list[tuple[int, TablePrediction]],
    regions: list[TextRegion],
    page_number: int,
    provider: str,
) -> tuple[list[TablePrediction], list[TextRegion]]:
    """Reject an accepted grid nested inside another that resolves more word cells.

    TATR can return both a table and a column-truncated sub-grid of it; each passes
    acceptance on its own, so the sub-grid's rows would render twice.
    """
    if len(accepted) < 2:
        return [prediction for _, prediction in accepted], []
    tokens = _word_tokens(regions)
    fills = [_grid_fill(prediction.cells, tokens)[0] for _, prediction in accepted]
    kept: list[TablePrediction] = []
    rejections: list[TextRegion] = []
    for position, (table_index, prediction) in enumerate(accepted):
        area = _box_area(prediction.bounding_box)
        parent = next(
            (
                other_position
                for other_position, (_, other) in enumerate(accepted)
                if other_position != position
                and area > 0
                and _intersection_area(prediction.bounding_box, other.bounding_box)
                / area
                >= SUBGRID_CONTAINMENT
                and fills[other_position] > fills[position]
            ),
            None,
        )
        if parent is None:
            kept.append(prediction)
            continue
        primary = _assign_regions([prediction], regions)[0] + _word_evidence_regions(
            regions, prediction.cells
        )
        row_count, column_count = _table_shape(prediction.cells)
        rejections.append(
            TextRegion(
                id=f"p{page_number}-tables-candidate-{table_index}",
                kind="table_candidate",
                text="",
                confidence=prediction.confidence,
                bounding_box=prediction.bounding_box,
                reading_order=min(
                    (region.reading_order for region in primary),
                    default=table_index,
                ),
                provider=provider,
                text_provenance={
                    "method": "table_subgrid_rejection",
                    "source_region_ids": _source_ids(primary),
                },
                resolution="unreadable",
                structure={
                    "role": "table_candidate",
                    "status": "rejected",
                    "reason": "subgrid_of_accepted_table",
                    "row_count": row_count,
                    "column_count": column_count,
                    "filled_cells": fills[position],
                    "superset_bbox": asdict(accepted[parent][1].bounding_box),
                    "superset_filled_cells": fills[parent],
                    "model": copy.deepcopy(prediction.model),
                    "detection_confidence": prediction.confidence,
                },
            )
        )
    return kept, rejections


def _resolve_cell_collisions(
    prediction: TablePrediction,
) -> tuple[TablePrediction, tuple[TableCell, ...]]:
    """Drop the minimum set of colliding cells instead of discarding the whole table.

    A detector that misses or doubles a few cells in a large grid still recovered most of
    it. Rejecting on any single collision does not scale with table size, so the smaller
    of each colliding pair is set aside and reported rather than the table being lost.
    """
    kept: list[TableCell] = []
    dropped: list[TableCell] = []
    # Larger cells win: a genuine merged cell outranks a spurious fragment.
    for cell in sorted(
        prediction.cells, key=lambda item: -_box_area(item.bounding_box)
    ):
        if _collides_with_any(cell, kept):
            dropped.append(cell)
        else:
            kept.append(cell)
    if not dropped:
        return prediction, ()
    # Scale-invariant: a few bad cells in a large grid is recoverable, but a grid where
    # a large share of cells collide is structurally unreliable and stays rejected.
    if len(dropped) / len(prediction.cells) > MAX_RECOVERABLE_CELL_COLLISIONS:
        return prediction, ()
    ordered = tuple(cell for cell in prediction.cells if cell not in dropped)
    resolved = replace(prediction, cells=ordered)
    return resolved, tuple(dropped)


def _collides_with_any(cell: TableCell, others: Sequence[TableCell]) -> bool:
    return any(_table_topology_conflicts((cell, other)) for other in others)


def _table_topology_conflicts(cells: Sequence[TableCell]) -> int:
    conflicts = 0
    for index, left in enumerate(cells):
        left_positions = {
            (row, column) for row in left.row_nums for column in left.column_nums
        }
        left_boxes = left.span_boxes or (left.bounding_box,)
        for right in cells[index + 1 :]:
            right_positions = {
                (row, column) for row in right.row_nums for column in right.column_nums
            }
            if left_positions & right_positions:
                conflicts += 1
                continue
            right_boxes = right.span_boxes or (right.bounding_box,)
            if any(
                min(_box_area(left_box), _box_area(right_box)) > 0
                and _intersection_area(left_box, right_box)
                / min(_box_area(left_box), _box_area(right_box))
                >= CELL_CROSSING_OVERLAP
                for left_box in left_boxes
                for right_box in right_boxes
            ):
                conflicts += 1
    return conflicts


def _resolve_cell(
    primary: _Candidate,
    challengers: list[_Candidate],
    low_primary_confidence: float,
    row_peers: Sequence[_Candidate] = (),
    label_unit: str | None = None,
) -> dict[str, Any]:
    cell_reread = next(
        (
            candidate
            for candidate in challengers
            if any(
                region.text_provenance
                and region.text_provenance.get("challenger_scope") == "cell"
                for region in candidate.regions
            )
            and _normalize(candidate.text)
        ),
        None,
    )
    standard_challengers = [
        candidate for candidate in challengers if candidate is not cell_reread
    ]
    challenger = _best_challenger(standard_challengers)
    text_repair = (
        _supported_text_repair(primary, challenger[0], challenger[2])
        if challenger is not None and challenger[1]
        else None
    )
    selected = primary
    supporters = [primary]
    decision = "primary"
    if cell_reread is not None:
        confirmations = [
            candidate
            for candidate in standard_challengers
            if _same_tokens(candidate.text, cell_reread.text)
        ]
        primary_present = bool(_normalize(primary.text))
        blank_recovery_supported = bool(
            _confidence(cell_reread.confidence) >= low_primary_confidence
            and any(
                _confidence(candidate.confidence) >= low_primary_confidence
                and _independent_candidates(cell_reread, candidate)
                for candidate in confirmations
            )
        )
        repaired_unit = _splice_label_unit(primary.text, cell_reread.text, label_unit)
        if repaired_unit is not None:
            supporters = [cell_reread, *confirmations]
            supporters.insert(0, primary)
            selected = _with_support(
                replace(primary, text=repaired_unit),
                supporters,
            )
            decision = "supported_unit_repair"
        else:
            cell_reread = _numeric_fragment(cell_reread)
            confirmations = [
                candidate
                for candidate in standard_challengers
                if _same_numeric_core(candidate.text, cell_reread.text)
                or _same_tokens(candidate.text, cell_reread.text)
            ]
            blank_recovery_supported = bool(
                _confidence(cell_reread.confidence) >= low_primary_confidence
                and any(
                    _confidence(candidate.confidence) >= low_primary_confidence
                    and _independent_candidates(cell_reread, candidate)
                    for candidate in confirmations
                )
            )
            primary_score = _row_format_score(primary.text, row_peers)
            reread_score = _row_format_score(cell_reread.text, row_peers)
            primary_matches = _same_numeric_core(primary.text, cell_reread.text) or (
                _same_tokens(primary.text, cell_reread.text)
            )
            if reread_score > primary_score and (
                blank_recovery_supported
                or primary_matches
                or primary_present
                and confirmations
            ):
                supporters = [cell_reread, *confirmations]
                selected = _with_support(cell_reread, supporters)
                decision = "cell_reread"
            elif primary_matches:
                supporters = [primary, cell_reread]
                selected = _with_support(primary, supporters)
                decision = "cell_confirmation"
            elif (
                not primary_present
                and blank_recovery_supported
                and _is_numeric_cell_text(cell_reread.text)
            ):
                supporters = [cell_reread, *confirmations]
                selected = _with_support(cell_reread, supporters)
                decision = "cell_reread"
            elif (
                confirmations
                and primary_score == reread_score
                and _confidence(primary.confidence) < low_primary_confidence
            ):
                supporters = [cell_reread, *confirmations]
                selected = _with_support(cell_reread, supporters)
                decision = "cell_reread"
            else:
                decision = "cell_disagreement"
    elif not _normalize(primary.text) and challenger is not None and challenger[1]:
        selected = _with_support(challenger[0], challenger[2])
        supporters = challenger[2]
        decision = "primary_missing"
    elif text_repair is not None:
        selected = text_repair
        supporters = challenger[2]
        decision = "supported_text_repair"
    elif (
        challenger is not None
        and challenger[1]
        and _overlap_conflict(primary.regions)
        and _same_value(primary.text, challenger[0].text)
    ):
        selected = _with_support(challenger[0], challenger[2])
        supporters = challenger[2]
        decision = "overlap_conflict"
    if selected is primary and challenger is not None and challenger[1]:
        recovered = _recover_supported_wrapper(primary, challenger[2])
        if recovered is not None:
            selected = recovered
            supporters = [primary, *challenger[2]]
            decision = "supported_wrapper"

    alternatives = _unique_candidates(
        [primary, *challengers], selected_text=selected.text
    )
    strong_conflict = bool(
        selected is primary
        and challenger is not None
        and challenger[1]
        and _normalize(challenger[0].text)
        and _normalize(challenger[0].text) != _normalize(primary.text)
        and not _same_tokens(challenger[0].text, primary.text)
        and _confidence(challenger[0].confidence) >= _confidence(primary.confidence)
    )
    cell_conflict = bool(
        decision == "cell_disagreement"
        and cell_reread is not None
        and _is_numeric_cell_text(primary.text)
        and _is_numeric_cell_text(cell_reread.text)
        and not _same_numeric_core(primary.text, cell_reread.text)
    )
    no_evidence = not _normalize(selected.text) and not selected.evidence_ids
    return {
        "selected": selected,
        "supporters": [] if no_evidence else supporters,
        "alternatives": alternatives,
        "resolution": (
            "unreadable"
            if no_evidence
            else "conflicting"
            if strong_conflict or cell_conflict
            else "resolved"
        ),
        "decision": (
            "no_cell_evidence"
            if no_evidence
            else "strong_disagreement"
            if strong_conflict
            else decision
        ),
    }


def _resolve_structural_blank_corner(
    cells: list[dict[str, Any]],
    provider: str,
) -> None:
    positioned: dict[tuple[int, int], dict[str, Any]] = {}
    rows: set[int] = set()
    columns: set[int] = set()
    for cell in cells:
        cell_rows = cell["row_nums"]
        cell_columns = cell["column_nums"]
        rows.update(cell_rows)
        columns.update(cell_columns)
        if len(cell_rows) != 1 or len(cell_columns) != 1:
            continue
        position = (cell_rows[0], cell_columns[0])
        if position in positioned:
            return
        positioned[position] = cell

    corner = positioned.get((0, 0))
    if (
        corner is None
        or len(rows) < 2
        or len(columns) < 2
        or not corner["column_header"]
        or corner["decision"] != "no_cell_evidence"
    ):
        return

    header_cells = [positioned.get((0, column)) for column in sorted(columns - {0})]
    row_label_cells = [positioned.get((row, 0)) for row in sorted(rows - {0})]
    if not all(
        cell is not None and cell["column_header"] and _cell_has_text_evidence(cell)
        for cell in header_cells
    ):
        return
    if not all(
        cell is not None and not cell["column_header"] and _cell_has_text_evidence(cell)
        for cell in row_label_cells
    ):
        return

    corner["source"] = provider
    corner["resolution"] = "resolved"
    corner["decision"] = "structural_blank_corner"


def _cell_has_text_evidence(cell: dict[str, Any]) -> bool:
    return bool(cell["evidence_ids"] and _normalize(cell["text"]))


def _best_challenger(
    candidates: list[_Candidate],
) -> tuple[_Candidate, bool, list[_Candidate]] | None:
    nonempty = [candidate for candidate in candidates if _normalize(candidate.text)]
    if not nonempty:
        return None
    if len(nonempty) == 1:
        return nonempty[0], True, [nonempty[0]]
    normalized = [_normalize(item.text) for item in nonempty]
    values = [_value(item.text) for item in nonempty]
    agreed = len(set(normalized)) == 1 or (all(values) and len(set(values)) == 1)
    selected = (
        min(
            nonempty,
            key=lambda item: (
                _value_noise(item.text),
                -_confidence(item.confidence),
            ),
        )
        if all(values) and len(set(values)) == 1
        else max(nonempty, key=lambda item: _confidence(item.confidence))
    )
    supporters = [
        candidate
        for candidate in nonempty
        if _normalize(candidate.text) == _normalize(selected.text)
        or (_value(candidate.text) and _value(candidate.text) == _value(selected.text))
    ]
    return selected, agreed, supporters


def _needs_cell_reread(
    primary: _Candidate,
    challengers: Sequence[_Candidate],
    low_primary_confidence: float,
    row_peers: Sequence[_Candidate],
    numeric_column: bool = True,
    label_unit: str | None = None,
) -> bool:
    numeric_challengers = [
        candidate for candidate in challengers if _is_numeric_cell_text(candidate.text)
    ]
    if _adequate_numeric_consensus(numeric_challengers, low_primary_confidence):
        resolution = _resolve_cell(
            primary,
            list(challengers),
            low_primary_confidence,
            row_peers,
            label_unit,
        )["resolution"]
        if resolution == "resolved":
            return False
    if not _normalize(primary.text):
        return numeric_column or bool(numeric_challengers)
    if not numeric_column and not _is_numeric_cell_text(primary.text):
        return bool(
            label_unit
            and _label_unit(primary.text) is None
            and LABEL_UNIT_SLOT_PATTERN.search(primary.text)
        )
    expected = _expected_numeric_format(row_peers)
    if expected and not _matches_numeric_format(primary.text, expected):
        return True
    if not _is_numeric_cell_text(primary.text):
        return bool(expected or numeric_challengers)
    return _confidence(primary.confidence) < low_primary_confidence and (
        any(
            not _same_numeric_core(primary.text, candidate.text)
            for candidate in numeric_challengers
        )
        or expected.get("decimal_places", 0) > 0
        and not _has_decimal_value(primary.text)
    )


def _adequate_numeric_consensus(
    challengers: Sequence[_Candidate],
    minimum_confidence: float,
) -> bool:
    adequate = [
        candidate
        for candidate in challengers
        if _confidence(candidate.confidence) >= minimum_confidence
    ]
    consensus = _best_challenger(adequate)
    return bool(
        consensus is not None
        and consensus[1]
        and len({candidate.source for candidate in consensus[2]}) >= 2
    )


def _row_peer_candidates(
    index: int,
    cells: Sequence[TableCell],
    assigned: Sequence[list[TextRegion]],
) -> list[_Candidate]:
    rows = set(cells[index].row_nums)
    return [
        _candidate(assigned[peer_index], "primary")
        for peer_index, cell in enumerate(cells)
        if peer_index != index and rows.intersection(cell.row_nums)
    ]


def _column_peer_candidates(
    index: int,
    cells: Sequence[TableCell],
    assigned: Sequence[list[TextRegion]],
) -> list[_Candidate]:
    columns = set(cells[index].column_nums)
    return [
        _candidate(assigned[peer_index], "primary")
        for peer_index, cell in enumerate(cells)
        if peer_index != index and columns.intersection(cell.column_nums)
    ]


def _label_unit(text: str) -> str | None:
    match = LABEL_UNIT_PATTERN.search(text)
    return _normalize(match.group(1)) if match is not None else None


def _expected_label_unit(column_peers: Sequence[_Candidate]) -> str | None:
    units = [
        unit for candidate in column_peers if (unit := _label_unit(candidate.text))
    ]
    if not units:
        return None
    unit = max(units, key=units.count)
    return unit if units.count(unit) >= 2 else None


def _splice_label_unit(
    primary: str,
    reread: str,
    expected: str | None,
) -> str | None:
    reread_unit = LABEL_UNIT_PATTERN.search(reread)
    primary_marker = LABEL_UNIT_MARKER_PATTERN.search(primary)
    if (
        expected is None
        or reread_unit is None
        or primary_marker is None
        or _normalize(reread_unit.group(1)) != expected
    ):
        return None
    start, end = primary_marker.span(2)
    return f"{primary[:start]}{reread_unit.group(1)}{primary[end:]}"


def _cell_reread_view(row_peers: Sequence[_Candidate]) -> tuple[int, int]:
    expected = _expected_numeric_format(row_peers)
    if expected.get("decimal_places", 0) > 0:
        padding = (
            GROUPED_CELL_REREAD_PADDING
            if not expected.get("currency") and not expected.get("percent")
            else CELL_REREAD_PADDING
        )
        return padding, DECIMAL_CELL_REREAD_SCALE
    if expected.get("grouped_thousands"):
        return GROUPED_CELL_REREAD_PADDING, CELL_REREAD_SCALE
    return CELL_REREAD_PADDING, CELL_REREAD_SCALE


def _has_visible_ink(image: Image.Image) -> bool:
    grayscale = image.convert("L")
    try:
        pixels = grayscale.load()
        dark_rows = {
            y
            for y in range(grayscale.height)
            if sum(pixels[x, y] < 200 for x in range(grayscale.width))
            >= max(1, math.ceil(grayscale.width * 0.8))
        }
        dark_columns = {
            x
            for x in range(grayscale.width)
            if sum(pixels[x, y] < 200 for y in range(grayscale.height))
            >= max(1, math.ceil(grayscale.height * 0.8))
        }
        dark_pixels = sum(
            pixels[x, y] < 200
            for y in range(grayscale.height)
            if y not in dark_rows
            for x in range(grayscale.width)
            if x not in dark_columns
        )
        return dark_pixels >= 3
    finally:
        grayscale.close()


def _independent_candidates(left: _Candidate, right: _Candidate) -> bool:
    return bool(_candidate_providers(left).isdisjoint(_candidate_providers(right)))


def _candidate_providers(candidate: _Candidate) -> set[str]:
    return {
        str((region.text_provenance or {}).get("source_provider") or region.provider)
        for region in candidate.regions
    }


def _numeric_fragment(candidate: _Candidate) -> _Candidate:
    matches = list(OBSERVED_VALUE_PATTERN.finditer(candidate.text))
    if not matches:
        return candidate
    match = max(
        matches,
        key=lambda item: (
            sum(character.isdigit() for character in item.group(0)),
            len(item.group(0)),
        ),
    )
    observed = match.group(0).strip()
    fragment = observed.rstrip(",")
    remainder = f"{candidate.text[: match.start()]}{candidate.text[match.end() :]}"
    if (
        _is_numeric_cell_text(candidate.text)
        and observed == fragment
        and remainder.strip() not in {",", "."}
    ):
        return candidate
    if not _is_numeric_cell_text(fragment):
        return candidate
    return replace(candidate, text=fragment)


def _supported_text_repair(
    primary: _Candidate,
    challenger: _Candidate,
    supporters: Sequence[_Candidate],
) -> _Candidate | None:
    if (
        len(supporters) < 2
        or _is_numeric_cell_text(primary.text)
        or _is_numeric_cell_text(challenger.text)
    ):
        return None
    similarity = SequenceMatcher(
        None,
        _normalize(primary.text),
        _normalize(challenger.text),
    ).ratio()
    if similarity < 0.8:
        return None
    return _with_support(challenger, list(supporters))


def _numeric_format(text: str) -> dict[str, object] | None:
    if not _is_numeric_cell_text(text):
        return None
    matches = list(VALUE_PATTERN.finditer(text))
    if not matches:
        return None
    match = max(
        matches,
        key=lambda item: (
            sum(character.isdigit() for character in item.group(0)),
            len(item.group(0)),
        ),
    )
    token = match.group(0)
    core = re.sub(r"[^\d.]", "", token)
    if not core:
        return None
    integer = core.split(".", 1)[0]
    prefix = text[: match.start()]
    return {
        "currency": "$" in prefix,
        "rank": "#" in prefix,
        "percent": "%" in token,
        "decimal_places": len(core.partition(".")[2]) if "." in core else 0,
        "grouped_thousands": "," in token,
        "integer_digits": len(integer),
    }


def _expected_numeric_format(
    row_peers: Sequence[_Candidate],
) -> dict[str, object]:
    formats = [
        current
        for candidate in row_peers
        if (current := _numeric_format(candidate.text)) is not None
    ]
    if len(formats) < 2:
        return {}
    expected: dict[str, object] = {}
    for feature in ("currency", "rank", "percent", "decimal_places"):
        values = [current[feature] for current in formats]
        value = max(values, key=values.count)
        if values.count(value) >= 2 and values.count(value) > len(values) / 2:
            expected[feature] = value
    if any(current["grouped_thousands"] for current in formats):
        expected["grouped_thousands"] = True
    return expected


def _matches_numeric_format(text: str, expected: dict[str, object]) -> bool:
    current = _numeric_format(text)
    if current is None:
        return False
    for feature in ("currency", "rank", "percent", "decimal_places"):
        if feature in expected and current[feature] != expected[feature]:
            return False
    return not (
        expected.get("grouped_thousands")
        and current["integer_digits"] >= 4
        and not current["grouped_thousands"]
    )


def _row_format_score(text: str, row_peers: Sequence[_Candidate]) -> int:
    expected = _expected_numeric_format(row_peers)
    current = _numeric_format(text)
    if not expected or current is None:
        return 0
    score = sum(
        current[feature] == value
        for feature, value in expected.items()
        if feature != "grouped_thousands"
    )
    if expected.get("grouped_thousands"):
        score += bool(current["integer_digits"] < 4 or current["grouped_thousands"])
    return score


def _has_decimal_value(text: str) -> bool:
    parts = _value_parts(text)
    return parts is not None and "." in parts[0]


def _with_support(selected: _Candidate, supporters: list[_Candidate]) -> _Candidate:
    evidence_ids = tuple(
        dict.fromkeys(
            evidence_id
            for supporter in supporters
            for evidence_id in supporter.evidence_ids
        )
    )
    regions = tuple(
        dict.fromkeys(
            region.id for supporter in supporters for region in supporter.regions
        )
    )
    regions_by_id = {
        region.id: region for supporter in supporters for region in supporter.regions
    }
    return replace(
        selected,
        evidence_ids=evidence_ids,
        regions=tuple(regions_by_id[region_id] for region_id in regions),
    )


def _recover_supported_wrapper(
    primary: _Candidate,
    supporters: list[_Candidate],
) -> _Candidate | None:
    primary_parts = _value_parts(primary.text)
    if primary_parts is None or len(supporters) < 2:
        return None
    parts = [_value_parts(candidate.text) for candidate in supporters]
    if any(item is None or item[0] != primary_parts[0] for item in parts):
        return None
    wrappers = [item[1] for item in parts if item is not None]
    if len(set(wrappers)) != 1 or not any(wrappers[0]):
        return None
    primary_wrapper = primary_parts[1]
    if any(
        current and current != proposed
        for current, proposed in zip(primary_wrapper, wrappers[0], strict=True)
    ):
        return None
    wrapper = tuple(
        current or proposed
        for current, proposed in zip(primary_wrapper, wrappers[0], strict=True)
    )
    if wrapper == primary_wrapper:
        return None
    text = _apply_value_wrapper(primary.text, wrapper)
    supported = _with_support(replace(primary, text=text), [primary, *supporters])
    return supported


def _value_parts(value: str) -> tuple[str, tuple[bool, str, bool, bool]] | None:
    matches = list(VALUE_PATTERN.finditer(value))
    if not matches:
        return None
    match = max(
        matches,
        key=lambda item: (
            sum(char.isdigit() for char in item.group(0)),
            len(item.group(0)),
        ),
    )
    token = match.group(0)
    core = re.sub(r"[^\d.]", "", token)
    if not core:
        return None
    prefix = value[: match.start()]
    currency = "$" in prefix
    sign = token[0] if token[0] in "+-" else ""
    parenthesized = token.lstrip("+-").startswith("(") and token.endswith(")")
    percent = token.rstrip(")").endswith("%")
    return core, (currency, sign, parenthesized, percent)


def _apply_value_wrapper(
    value: str,
    wrapper: tuple[bool, str, bool, bool],
) -> str:
    match = max(
        VALUE_PATTERN.finditer(value),
        key=lambda item: (
            sum(char.isdigit() for char in item.group(0)),
            len(item.group(0)),
        ),
    )
    token = match.group(0).lstrip("+-")
    token = token[1:-1] if token.startswith("(") and token.endswith(")") else token
    token = token.removesuffix("%")
    currency, sign, parenthesized, percent = wrapper
    rendered = f"{sign}{token}"
    if parenthesized:
        rendered = f"({rendered})"
    if percent:
        rendered = f"{rendered}%"
    if currency:
        rendered = f"$ {rendered}"
    return f"{value[: match.start()]}{rendered}{value[match.end() :]}".strip()


def _candidate(regions: list[TextRegion], source: str) -> _Candidate:
    ordered = sorted(
        regions,
        key=lambda item: (
            item.reading_order,
            item.bounding_box.top,
            item.bounding_box.left,
        ),
    )
    confidences = [item.confidence for item in ordered if item.confidence is not None]
    return _Candidate(
        text=" ".join(item.text.strip() for item in ordered if item.text.strip()),
        confidence=sum(confidences) / len(confidences) if confidences else None,
        source=(
            source
            if source != "primary"
            else "+".join(sorted({item.provider for item in ordered})) or "primary"
        ),
        evidence_ids=tuple(item.id for item in ordered),
        regions=tuple(ordered),
    )


def _assign_regions(
    tables: Sequence[TablePrediction], regions: Sequence[TextRegion]
) -> list[list[TextRegion]]:
    return _assign_to_boxes(
        [table.bounding_box for table in tables],
        regions,
    )


def _assign_cells(
    cells: Sequence[TableCell], regions: Sequence[TextRegion]
) -> list[list[TextRegion]]:
    grid = _span_grid(cells)
    if grid is None:
        return _assign_to_boxes(
            [cell.bounding_box for cell in cells],
            [
                region
                for region in regions
                if not _crosses_cell_boundaries(cells, region.bounding_box)
            ],
        )

    row_centers, column_centers, table_box = grid
    assigned = [[] for _ in cells]
    for region in regions:
        box = region.bounding_box
        if not region.text.strip():
            continue
        if _crosses_cell_boundaries(cells, box):
            continue
        center_x = (box.left + box.right) / 2
        center_y = (box.top + box.bottom) / 2
        if not (
            table_box.left <= center_x <= table_box.right
            and table_box.top <= center_y <= table_box.bottom
        ):
            continue
        row = min(row_centers, key=lambda key: abs(row_centers[key] - center_y))
        column = min(
            column_centers,
            key=lambda key: abs(column_centers[key] - center_x),
        )
        choices = [
            (index, cell)
            for index, cell in enumerate(cells)
            if row in cell.row_nums and column in cell.column_nums
        ]
        if not choices:
            continue
        index, _ = min(choices, key=lambda item: _box_area(item[1].bounding_box))
        assigned[index].append(region)
    return assigned


def _crosses_cell_boundaries(
    cells: Sequence[TableCell],
    region_box: BoundingBox,
) -> bool:
    area = _box_area(region_box)
    if area <= 0:
        return False
    # A region holding cell centres across two rows and two columns cannot belong to
    # any one cell. The overlap-ratio test below misses this for fine grids: a
    # whole-table region over a 5x4 grid intersects every cell at a twentieth of its
    # own area. One axis alone is not enough - text straddling two row bands of its
    # own column is normal and belongs to the nearest cell.
    rows_spanned: set[int] = set()
    columns_spanned: set[int] = set()
    for cell in cells:
        centre_x = (cell.bounding_box.left + cell.bounding_box.right) / 2
        centre_y = (cell.bounding_box.top + cell.bounding_box.bottom) / 2
        if (
            region_box.left < centre_x < region_box.right
            and region_box.top < centre_y < region_box.bottom
        ):
            rows_spanned.update(cell.row_nums)
            columns_spanned.update(cell.column_nums)
    if len(rows_spanned) >= 2 and len(columns_spanned) >= 2:
        return True
    overlaps = [
        cell
        for cell in cells
        if _intersection_area(region_box, cell.bounding_box) / area
        >= CELL_CROSSING_OVERLAP
    ]
    rows = {row for cell in overlaps for row in cell.row_nums}
    columns = {column for cell in overlaps for column in cell.column_nums}
    width = region_box.right - region_box.left
    height = region_box.bottom - region_box.top
    widths = [cell.bounding_box.right - cell.bounding_box.left for cell in overlaps]
    heights = [cell.bounding_box.bottom - cell.bounding_box.top for cell in overlaps]
    crosses_columns = (
        len(columns) > 1 and widths and width > median(widths) * CELL_CROSSING_SCALE
    )
    crosses_rows = (
        len(rows) > 1 and heights and height > median(heights) * CELL_CROSSING_SCALE
    )
    return crosses_columns or crosses_rows


def _assigned_regions(groups: Sequence[Sequence[TextRegion]]) -> list[TextRegion]:
    return sorted(
        (region for group in groups for region in group),
        key=lambda region: (region.reading_order, region.id),
    )


def _word_evidence_regions(
    regions: Sequence[TextRegion], cells: Sequence[TableCell]
) -> list[TextRegion]:
    """Word tokens for cell assignment, the way official TATR slots page tokens.

    Only from table regions whose own box cannot land whole in a single cell: those are
    excluded from direct assignment, so without their words every cell under them
    renders blank. Words the region's own text lost are skipped - the under-read repair
    already emitted them as regions of their own, and they assign themselves.
    """
    if not cells:
        return []
    union = BoundingBox(
        min(cell.bounding_box.left for cell in cells),
        min(cell.bounding_box.top for cell in cells),
        max(cell.bounding_box.right for cell in cells),
        max(cell.bounding_box.bottom for cell in cells),
    )
    words = []
    for region in regions:
        structure = region.structure if isinstance(region.structure, dict) else {}
        evidence = structure.get("word_evidence")
        if region.kind != "table" or not isinstance(evidence, list):
            continue
        box = region.bounding_box
        centre_in_grid = (
            union.left <= (box.left + box.right) / 2 <= union.right
            and union.top <= (box.top + box.bottom) / 2 <= union.bottom
        )
        if centre_in_grid and not _crosses_cell_boundaries(cells, box):
            # The region lands whole in one cell; its words would double that text.
            continue
        provider = str(
            (region.text_provenance or {}).get("geometry_provider") or region.provider
        )
        for index, entry in enumerate(evidence, start=1):
            if not isinstance(entry, dict) or not entry.get("in_region_text", True):
                continue
            text = str(entry.get("text") or "").strip()
            box = entry.get("bbox")
            if not text or not isinstance(box, dict):
                continue
            try:
                bounding_box = BoundingBox(
                    int(box["left"]),
                    int(box["top"]),
                    int(box["right"]),
                    int(box["bottom"]),
                )
            except (KeyError, TypeError, ValueError):
                continue
            if (
                bounding_box.right <= bounding_box.left
                or bounding_box.bottom <= bounding_box.top
            ):
                continue
            confidence = entry.get("confidence")
            words.append(
                TextRegion(
                    id=f"{region.id}-word-{index}",
                    kind="text",
                    text=text,
                    confidence=(
                        float(confidence)
                        if isinstance(confidence, (int, float))
                        else None
                    ),
                    bounding_box=bounding_box,
                    reading_order=region.reading_order,
                    provider=provider,
                    text_provenance={
                        "method": "table_word_evidence",
                        "source_region_id": region.id,
                    },
                )
            )
    return words


def _collapse_word_sources(
    assigned: Sequence[TextRegion],
    pool: Sequence[TextRegion],
    *,
    consumed_ids: frozenset[str] | set[str] = frozenset(),
) -> list[TextRegion]:
    """Replace expanded evidence words with the regions they came from.

    A word's parent joins the sources - and is later marked consumed - only when listed
    in consumed_ids; otherwise the token is dropped and the parent region stands alone.
    """
    by_id = {region.id: region for region in pool}
    collapsed: list[TextRegion] = []
    seen: set[str] = set()
    for region in assigned:
        provenance = region.text_provenance or {}
        if provenance.get("method") == "table_word_evidence":
            parent_id = str(provenance.get("source_region_id"))
            if parent_id not in consumed_ids:
                continue
            region = by_id.get(parent_id, region)
        if region.id not in seen:
            seen.add(region.id)
            collapsed.append(region)
    return collapsed


def _source_ids(regions: Sequence[TextRegion]) -> list[str]:
    ids: list[str] = []
    for region in regions:
        provenance = region.text_provenance or {}
        id_ = (
            str(provenance["source_region_id"])
            if provenance.get("method") == "table_word_evidence"
            else region.id
        )
        if id_ not in ids:
            ids.append(id_)
    return ids


@dataclass(frozen=True)
class _GridOverlay:
    region: TextRegion
    texts: dict[tuple[int, int], str]
    shape: tuple[int, int]
    agreement: bool


def _text_grid_overlay(
    regions: Sequence[TextRegion],
    prediction: TablePrediction,
    row_count: int,
    column_count: int,
) -> _GridOverlay | None:
    """The text-only grid a layout reader parsed over this table, if any.

    The grid's cells carry text but no geometry, so the only lossless join is by row
    and column index, and only when both models agree on the grid's shape. Anything
    else force-fits one model's structure onto the other's and invents cells.
    """
    candidates = []
    for region in regions:
        structure = region.structure if isinstance(region.structure, dict) else {}
        cells = structure.get("cells")
        if region.kind != "table" or not isinstance(cells, list) or not cells:
            continue
        if any(isinstance(cell, dict) and cell.get("bbox") for cell in cells):
            continue
        overlap = _intersection_area(region.bounding_box, prediction.bounding_box)
        if overlap > 0:
            candidates.append((overlap, region))
    if not candidates:
        return None
    _, region = max(candidates, key=lambda item: item[0])
    structure = region.structure or {}
    shape = (
        int(structure.get("row_count") or 0),
        int(structure.get("column_count") or 0),
    )
    texts: dict[tuple[int, int], str] = {}
    for cell in structure["cells"]:
        if not isinstance(cell, dict):
            continue
        rows = cell.get("row_nums")
        columns = cell.get("column_nums")
        text = str(cell.get("text") or "").strip()
        if rows and columns and text:
            texts[(min(rows), min(columns))] = text
    return _GridOverlay(region, texts, shape, shape == (row_count, column_count))


def _apply_grid_text(
    resolved: dict[str, Any],
    overlay: _GridOverlay,
    cell: TableCell,
    primary_candidate: _Candidate,
    challenger_candidates: list[_Candidate],
) -> dict[str, Any]:
    """Land the text grid's cell text in the geometry model's boxed cell.

    The grid's reader parses characters better but carries no confidence of its own, so
    the cell keeps the recognition confidence of the words that fell inside its box -
    the same fusion the page-level fused reader performs.
    """
    text = overlay.texts.get((min(cell.row_nums), min(cell.column_nums)), "")
    if not _normalize(text):
        return resolved
    selected = resolved["selected"]
    if _normalize(selected.text) == _normalize(text):
        return resolved
    grid_candidate = _Candidate(
        text=text,
        confidence=primary_candidate.confidence,
        source=overlay.region.provider,
        evidence_ids=(overlay.region.id,),
        regions=(overlay.region,),
    )
    supporters = [grid_candidate]
    if _normalize(primary_candidate.text):
        supporters.append(primary_candidate)
    return {
        "selected": grid_candidate,
        "supporters": supporters,
        "alternatives": _unique_candidates(
            [selected, primary_candidate, *challenger_candidates],
            selected_text=text,
        ),
        "resolution": "resolved",
        "decision": "text_grid_cell",
    }


def _span_grid(
    cells: Sequence[TableCell],
) -> tuple[dict[int, float], dict[int, float], BoundingBox] | None:
    if not cells:
        return None
    rows: dict[int, list[float]] = {}
    columns: dict[int, list[float]] = {}
    for cell in cells:
        row = min(cell.row_nums)
        column = min(cell.column_nums)
        for box in cell.span_boxes:
            rows.setdefault(row, []).append((box.top + box.bottom) / 2)
            columns.setdefault(column, []).append((box.left + box.right) / 2)
    expected_rows = {row for cell in cells for row in cell.row_nums}
    expected_columns = {column for cell in cells for column in cell.column_nums}
    if set(rows) != expected_rows or set(columns) != expected_columns:
        return None
    table_box = BoundingBox(
        left=min(cell.bounding_box.left for cell in cells),
        top=min(cell.bounding_box.top for cell in cells),
        right=max(cell.bounding_box.right for cell in cells),
        bottom=max(cell.bounding_box.bottom for cell in cells),
    )
    return (
        {key: median(values) for key, values in rows.items()},
        {key: median(values) for key, values in columns.items()},
        table_box,
    )


def _assign_to_boxes(
    boxes: Sequence[BoundingBox], regions: Sequence[TextRegion]
) -> list[list[TextRegion]]:
    assigned = [[] for _ in boxes]
    for region in regions:
        region_box = region.bounding_box
        area = _box_area(region_box)
        if area <= 0 or not region.text.strip():
            continue
        center_x = (region_box.left + region_box.right) / 2
        center_y = (region_box.top + region_box.bottom) / 2
        best: tuple[float, float, int] | None = None
        for index, box in enumerate(boxes):
            overlap = _intersection_area(region_box, box) / area
            contains = (
                box.left <= center_x <= box.right and box.top <= center_y <= box.bottom
            )
            if overlap < 0.5 and not contains:
                continue
            rank = (float(contains) + overlap, -float(_box_area(box)), index)
            if best is None or rank[:2] > best[:2]:
                best = rank
        if best is not None:
            assigned[best[2]].append(region)
    return assigned


def _mark_sources(regions: Sequence[TextRegion], table_id: str) -> None:
    for region in regions:
        structure = dict(region.structure or {})
        old_role = structure.get("role")
        if old_role and old_role != "table_source":
            structure["source_role"] = old_role
        structure.update({"role": "table_source", "parent_id": table_id})
        region.structure = structure


def attach_recovered_table_evidence(
    regions: Sequence[TextRegion],
    recovered: Sequence[TextRegion],
) -> None:
    """Fill unreadable table cells with later geometry-linked OCR evidence."""
    available = list(recovered)
    for table in (region for region in regions if region.kind == "table"):
        structure = table.structure
        if not isinstance(structure, dict) or not isinstance(
            structure.get("cells"), list
        ):
            continue
        cells = _table_cells_from_structure(structure["cells"])
        if cells is None:
            continue
        assigned = _assign_cells(cells, available)
        added = []
        for cell, evidence in zip(structure["cells"], assigned, strict=True):
            if (
                not evidence
                or cell.get("decision") != "no_cell_evidence"
                or _normalize(str(cell.get("text", "")))
            ):
                continue
            candidate = _candidate(evidence, "recovered")
            if not _normalize(candidate.text):
                continue
            cell.update(
                {
                    "text": candidate.text,
                    "source": candidate.source,
                    "confidence": candidate.confidence,
                    "resolution": "resolved",
                    "evidence_ids": list(candidate.evidence_ids),
                    "supporters": [
                        {
                            "provider": candidate.source,
                            "evidence_ids": list(candidate.evidence_ids),
                        }
                    ],
                    "decision": "recovered_missing_cell",
                }
            )
            added.extend(evidence)
        if not added:
            continue
        _mark_sources(added, table.id)
        provenance = dict(table.text_provenance or {})
        source_ids = provenance.setdefault("source_region_ids", [])
        source_ids.extend(region.id for region in added if region.id not in source_ids)
        table.text_provenance = provenance
        table.text = _markdown(
            structure["cells"],
            int(structure["row_count"]),
            int(structure["column_count"]),
        )
        confidences = [
            cell.get("confidence")
            for cell in structure["cells"]
            if cell.get("confidence") is not None
        ]
        table.confidence = (
            sum(float(confidence) for confidence in confidences) / len(confidences)
            if confidences
            else table.confidence
        )
        used = {region.id for region in added}
        available = [region for region in available if region.id not in used]


def _table_cells_from_structure(
    values: Sequence[object],
) -> list[TableCell] | None:
    cells = []
    for value in values:
        if not isinstance(value, dict):
            return None
        box = value.get("bbox")
        rows = _indexes(value.get("row_nums"))
        columns = _indexes(value.get("column_nums"))
        if not isinstance(box, dict) or not rows or not columns:
            return None
        try:
            bounding_box = BoundingBox(
                int(box["left"]),
                int(box["top"]),
                int(box["right"]),
                int(box["bottom"]),
            )
        except (KeyError, TypeError, ValueError):
            return None
        cells.append(TableCell(bounding_box, rows, columns))
    return cells


def _challenger_region(
    region: TextRegion,
    challenger: str,
    page_number: int,
    table_index: int,
    index: int,
    offset: tuple[int, int] = (0, 0),
) -> TextRegion:
    provenance = dict(region.text_provenance or {})
    provenance.update(
        {
            "challenger": challenger,
            "source_id": region.id,
            "source_provider": region.provider,
        }
    )
    return replace(
        region,
        id=(f"p{page_number}-tables-{_slug(challenger)}-t{table_index}-source-{index}"),
        provider=challenger,
        bounding_box=BoundingBox(
            left=region.bounding_box.left + offset[0],
            top=region.bounding_box.top + offset[1],
            right=region.bounding_box.right + offset[0],
            bottom=region.bounding_box.bottom + offset[1],
        ),
        text_provenance=provenance,
        alternatives=list(region.alternatives),
        structure=copy.deepcopy(region.structure),
    )


def _table_crop(
    image: Image.Image,
    box: BoundingBox,
    padding: tuple[int, int],
) -> tuple[Image.Image, tuple[int, int]]:
    horizontal, vertical = padding
    left = max(0, box.left - horizontal)
    top = max(0, box.top - vertical)
    right = min(image.width, box.right + horizontal)
    bottom = min(image.height, box.bottom + vertical)
    if right <= left or bottom <= top:
        raise ReaderError("invalid_table_output", "A table challenger crop is empty")
    return image.crop((left, top, right, bottom)), (left, top)


def _write_table_crops(
    image: Image.Image,
    predictions: Sequence[TablePrediction],
    padding: tuple[int, int],
    root: Path,
) -> tuple[_TableCrop, ...]:
    crops = []
    for table_index, prediction in enumerate(predictions, start=1):
        crop, offset = _table_crop(image, prediction.bounding_box, padding)
        path = root / f"table-{table_index}.png"
        try:
            crop.save(path, format="PNG")
        except Exception as error:
            raise ReaderError("table_preprocess_failed", str(error)) from error
        finally:
            crop.close()
        crops.append(_TableCrop(path, offset))
    return tuple(crops)


def _prepare_view(
    image: Image.Image,
    challenger: TableChallenger,
    path: Path,
) -> None:
    try:
        prepared = challenger.prepare(image.copy()) if challenger.prepare else image
        if not isinstance(prepared, Image.Image):
            raise TypeError("Table preprocessing must return a PIL image")
        prepared.save(path, format="PNG")
    except Exception as error:
        raise ReaderError("table_preprocess_failed", str(error)) from error


def sauvola_view(
    image: Image.Image,
    *,
    window_size: int = 31,
    k: float = 0.2,
    dynamic_range: float = 128,
) -> Image.Image:
    """Return a deterministic local Sauvola threshold view."""
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("window_size must be an odd integer of at least 3")
    if dynamic_range <= 0:
        raise ValueError("dynamic_range must be positive")
    import cv2
    import numpy as np

    gray = np.asarray(image.convert("L"), dtype=np.float32)
    size = (window_size, window_size)
    mean = cv2.boxFilter(gray, -1, size, normalize=True)
    square_mean = cv2.boxFilter(gray * gray, -1, size, normalize=True)
    deviation = np.sqrt(np.maximum(square_mean - mean * mean, 0))
    threshold = mean * (1 + k * (deviation / dynamic_range - 1))
    binary = np.where(gray > threshold, 255, 0).astype(np.uint8)
    return Image.fromarray(binary)


def _word_tokens(regions: Sequence[TextRegion]) -> list[dict[str, Any]]:
    """Every word box the fused reader recognised, page coordinates, deduplicated."""
    words = []
    seen: set[tuple[int, int, int, int, str]] = set()
    for region in regions:
        for entry in (region.structure or {}).get("word_evidence") or []:
            if not isinstance(entry, dict):
                continue
            box = entry.get("bbox")
            text = str(entry.get("text") or "").strip()
            if not isinstance(box, dict) or not text:
                continue
            try:
                key = (box["left"], box["top"], box["right"], box["bottom"], text)
            except (KeyError, TypeError):
                continue
            if key in seen:
                continue
            seen.add(key)
            words.append({"bbox": list(key[:4]), "text": text})
    return words


def _to_tokens(regions: Sequence[TextRegion]) -> list[dict[str, Any]]:
    return [
        {
            "bbox": [
                region.bounding_box.left,
                region.bounding_box.top,
                region.bounding_box.right,
                region.bounding_box.bottom,
            ],
            "text": region.text,
            "span_num": index,
            "line_num": region.reading_order,
            "block_num": 0,
            "confidence": region.confidence,
            "source_id": region.id,
            "provider": region.provider,
        }
        for index, region in enumerate(regions)
        if region.text.strip()
    ]


def _markdown(cells: list[dict[str, Any]], rows: int, columns: int) -> str:
    if rows <= 0 or columns <= 0:
        return ""
    grid = [["" for _ in range(columns)] for _ in range(rows)]
    for cell in cells:
        row = min(cell["row_nums"])
        column = min(cell["column_nums"])
        if row < rows and column < columns:
            grid[row][column] = _escape_markdown(cell["text"])
    lines = ["| " + " | ".join(row) + " |" for row in grid]
    lines.insert(1, "| " + " | ".join("---" for _ in range(columns)) + " |")
    return "\n".join(lines)


def _escape_markdown(value: str) -> str:
    return value.replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")


def _translate_cell_box(
    value: object,
    table_box: object,
    padding: int,
    rotated: bool,
    page_size: tuple[int, int],
    crop_size: tuple[int, int],
) -> BoundingBox:
    cell = _float_box(value)
    table = _float_box(table_box)
    left = table[0] - padding
    top = table[1] - padding
    if not rotated:
        translated = [cell[0] + left, cell[1] + top, cell[2] + left, cell[3] + top]
        return _bounded_box(translated, page_size)

    rotated_width = crop_size[0]
    translated = [
        cell[1] + left,
        rotated_width - cell[2] - 1 + top,
        cell[3] + left,
        rotated_width - cell[0] - 1 + top,
    ]
    return _bounded_box(translated, page_size)


def _bounded_box(value: object, size: tuple[int, int]) -> BoundingBox:
    left, top, right, bottom = _float_box(value)
    width, height = size
    box = BoundingBox(
        left=max(0, min(width, math.floor(left))),
        top=max(0, min(height, math.floor(top))),
        right=max(0, min(width, math.ceil(right))),
        bottom=max(0, min(height, math.ceil(bottom))),
    )
    if box.right <= box.left or box.bottom <= box.top:
        raise ReaderError("invalid_table_output", "A translated table box is empty")
    return box


def _source_dir(root: Path) -> Path:
    if (root / "inference.py").is_file():
        return root
    return root / "src"


def _load_inference(root: Path) -> ModuleType:
    source = _source_dir(root)
    checkout = source.parent
    detr = checkout / "detr"
    if not (source / "inference.py").is_file() or not detr.is_dir():
        raise ReaderError(
            "table_model_unavailable",
            "Table Transformer checkout lacks src/inference.py or detr",
        )
    paths = [str(source), str(detr)]
    sys.path[:0] = paths
    try:
        spec = importlib.util.spec_from_file_location(
            "ocr_pipeline_tatr_inference", source / "inference.py"
        )
        if spec is None or spec.loader is None:
            raise ImportError("Could not load Table Transformer inference module")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for path in paths:
            sys.path.remove(path)


def _indexes(value: object) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        return ()
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0
        for item in value
    ):
        return ()
    return tuple(sorted(set(value)))


def _near_page_trivial_grid(
    cells: Sequence[TableCell], obj: dict[str, Any], image: Image.Image
) -> bool:
    """A detection covering most of the page whose grid is too small to be real.

    Mirrors the stage's near_page_low_complexity rejection so the decomposition
    runs BEFORE that rejection throws the table away.
    """
    try:
        box = _bounded_box(obj["bbox"], image.size)
    except ReaderError:
        return False
    page_area = image.size[0] * image.size[1]
    if page_area <= 0 or _box_area(box) / page_area < NEAR_PAGE_TABLE_AREA:
        return False
    rows, columns = _table_shape(cells)
    return (
        rows < MIN_NEAR_PAGE_ROWS
        or columns < MIN_NEAR_PAGE_COLUMNS
        or rows * columns < MIN_NEAR_PAGE_CELLS
        or len(cells) < MIN_NEAR_PAGE_CELLS
    )


def _conflict_free_cells(
    cells: Sequence[TableCell], box: BoundingBox
) -> list[TableCell]:
    """A candidate grid with its recoverable collisions dropped, or nothing.

    The same tolerance the stage applies to accepted predictions: a few colliding
    cells in a large grid are set aside; a grid that stays conflicted after that is
    unusable as a candidate.
    """
    if not cells:
        return []
    resolved, _ = _resolve_cell_collisions(
        TablePrediction(
            bounding_box=box, cells=tuple(cells), confidence=None, model={}
        )
    )
    if _table_topology_conflicts(resolved.cells):
        return []
    return list(resolved.cells)


def _needs_structure_arbitration(
    cells: Sequence[TableCell], tokens: Sequence[dict[str, Any]]
) -> bool:
    """Whether an accepted TATR grid looks too coarse for the words printed on it.

    Fires on a merged cell spanning three or more rows or columns, or on tokens
    crowding into fewer than a fifth as many token-bearing cells.
    """
    if not cells or not tokens:
        return False
    if any(
        len(cell.row_nums) >= 3 or len(cell.column_nums) >= 3 for cell in cells
    ):
        return True
    filled, _ = _grid_fill(cells, tokens)
    return filled * 5 < len(tokens)


def _grid_fill(
    cells: Sequence[TableCell], tokens: Sequence[dict[str, Any]]
) -> tuple[int, int]:
    """How well a candidate grid resolves the tokens printed on it.

    Distinct token-bearing cells first (a coarse band grid puts every token into a
    handful of giant cells and scores low), covered tokens second as the tiebreak.
    """
    filled: set[int] = set()
    covered = 0
    for token in tokens:
        for index, cell in enumerate(cells):
            if _centre_in_box(token["bbox"], cell.bounding_box):
                filled.add(index)
                covered += 1
                break
    return len(filled), covered


def _centre_in_box(value: object, box: BoundingBox) -> bool:
    left, top, right, bottom = _float_box(value)
    centre_x = (left + right) / 2
    centre_y = (top + bottom) / 2
    return box.left <= centre_x <= box.right and box.top <= centre_y <= box.bottom


def _valid_box(value: object) -> bool:
    try:
        _float_box(value)
    except ReaderError:
        return False
    return True


def _float_box(value: object) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ReaderError("invalid_table_output", "A table box needs four coordinates")
    if any(
        isinstance(item, bool) or not isinstance(item, (int, float)) for item in value
    ):
        raise ReaderError("invalid_table_output", "A table box must be numeric")
    box = tuple(float(item) for item in value)
    if (
        not all(math.isfinite(item) for item in box)
        or box[2] <= box[0]
        or box[3] <= box[1]
    ):
        raise ReaderError("invalid_table_output", "A table box must have positive area")
    return box


def _optional_confidence(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    confidence = float(value)
    return confidence if math.isfinite(confidence) and 0 <= confidence <= 1 else None


def _unique_candidates(
    candidates: Sequence[_Candidate], *, selected_text: str
) -> list[_Candidate]:
    result = []
    seen = {_normalize(selected_text)}
    for candidate in candidates:
        normalized = _normalize(candidate.text)
        if normalized and normalized not in seen:
            result.append(candidate)
            seen.add(normalized)
    return result


def _normalize(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"\s", "", text.replace("−", "-").replace("–", "-"))


def _value(value: str) -> str:
    matches = VALUE_PATTERN.findall(value)
    if not matches:
        return ""
    return _normalize(
        max(matches, key=lambda item: (sum(char.isdigit() for char in item), len(item)))
    )


def _same_value(left: str, right: str) -> bool:
    return bool(_value(left) and _value(left) == _value(right))


def _same_numeric_core(left: str, right: str) -> bool:
    left_parts = _value_parts(left)
    right_parts = _value_parts(right)
    return bool(
        left_parts is not None
        and right_parts is not None
        and left_parts[0] == right_parts[0]
    )


def _value_noise(value: str) -> int:
    signature = _value(value)
    return (
        len(_normalize(value)) - len(signature) if signature else len(_normalize(value))
    )


def _same_tokens(left: str, right: str) -> bool:
    return bool(_normalize(left) and _normalize(left) == _normalize(right))


def _overlap_conflict(regions: Sequence[TextRegion]) -> bool:
    for index, first in enumerate(regions):
        first_text = _normalize(first.text)
        for second in regions[index + 1 :]:
            second_text = _normalize(second.text)
            smaller = min(_box_area(first.bounding_box), _box_area(second.bounding_box))
            if smaller <= 0:
                continue
            overlap = (
                _intersection_area(first.bounding_box, second.bounding_box) / smaller
            )
            nested = first_text in second_text or second_text in first_text
            if overlap >= 0.3 and nested:
                return True
    return False


def _intersection_area(left: BoundingBox, right: BoundingBox) -> int:
    width = max(0, min(left.right, right.right) - max(left.left, right.left))
    height = max(0, min(left.bottom, right.bottom) - max(left.top, right.top))
    return width * height


def _box_area(box: BoundingBox) -> int:
    return max(0, box.right - box.left) * max(0, box.bottom - box.top)


def _suppress_contained_detections(
    objects: list[dict[str, Any]],
    crops: list[dict[str, Any]],
    page_size: tuple[int, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ranked = sorted(
        enumerate(zip(objects, crops, strict=True)),
        key=lambda item: (-float(item[1][0]["score"]), item[0]),
    )
    kept: list[tuple[int, dict[str, Any], dict[str, Any], BoundingBox]] = []
    for index, (obj, crop) in ranked:
        box = _bounded_box(obj["bbox"], page_size)
        area = _box_area(box)
        duplicate = any(
            min(area, _box_area(other_box)) > 0
            and _intersection_area(box, other_box) / min(area, _box_area(other_box))
            >= TABLE_DUPLICATE_CONTAINMENT
            for _, _, _, other_box in kept
        )
        if not duplicate:
            kept.append((index, obj, crop, box))
    kept.sort(key=lambda item: item[0])
    return (
        [obj for _, obj, _, _ in kept],
        [crop for _, _, crop, _ in kept],
    )


def _confidence(value: float | None) -> float:
    return value if value is not None else -1.0


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-") or "reader"
