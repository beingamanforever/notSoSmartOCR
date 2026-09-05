"""Evidence-preserving cross-page table continuation."""

from __future__ import annotations

import copy
import math
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .contracts import PageResult, TableContinuation, TextRegion
from .providers import ReaderError
from .table_topology import TableTopology, TableTopologyError, validate_table_topology


@dataclass(frozen=True)
class TableContinuationPrediction:
    score: float
    provenance: Mapping[str, Any]


class TableContinuationClassifier(Protocol):
    name: str

    def classify(
        self,
        first_page_path: Path,
        second_page_path: Path,
        first_table: TextRegion,
        second_table: TextRegion,
    ) -> TableContinuationPrediction: ...


@dataclass(frozen=True)
class _TablePart:
    page_number: int
    table: TextRegion
    topology: TableTopology
    header_signature: tuple[tuple[tuple[int, ...], str], ...] | None


@dataclass(frozen=True)
class _ContinuationEdge:
    first: _TablePart
    second: _TablePart
    prediction: TableContinuationPrediction


class CrossPageTableStage:
    """Join classifier-approved tables without changing their source regions."""

    name = "cross-page-tables"

    def __init__(
        self,
        classifier: TableContinuationClassifier,
        *,
        minimum_score: float = 0.5,
    ) -> None:
        if not 0 <= minimum_score <= 1:
            raise ValueError("minimum_score must be from 0 to 1")
        if not classifier.name.strip():
            raise ValueError("classifier name must be non-empty")
        self.classifier = classifier
        self.minimum_score = minimum_score

    def apply(
        self,
        page_paths: Sequence[Path],
        pages: Sequence[PageResult],
    ) -> list[TableContinuation]:
        if len(page_paths) != len(pages):
            raise ReaderError(
                "table_continuation_page_mismatch",
                "Cross-page table stage received mismatched page images and results",
            )
        edges = [
            edge
            for index in range(len(pages) - 1)
            if (
                edge := self._classify_pair(
                    page_paths[index],
                    page_paths[index + 1],
                    pages[index],
                    pages[index + 1],
                )
            )
            is not None
        ]
        return [
            _merge_chain(chain, artifact_index, self.classifier.name)
            for artifact_index, chain in enumerate(_edge_chains(edges), start=1)
        ]

    def _classify_pair(
        self,
        first_page_path: Path,
        second_page_path: Path,
        first_page: PageResult,
        second_page: PageResult,
    ) -> _ContinuationEdge | None:
        if second_page.page_number != first_page.page_number + 1:
            return None
        first = _boundary_table(first_page, last=True)
        second = _boundary_table(second_page, last=False)
        if first is None or second is None or not _compatible(first, second):
            return None
        try:
            prediction = self.classifier.classify(
                first_page_path,
                second_page_path,
                copy.deepcopy(first.table),
                copy.deepcopy(second.table),
            )
        except ReaderError:
            raise
        except Exception as error:
            raise ReaderError(
                "table_continuation_classifier_failed",
                "Table continuation classifier failed during inference",
            ) from error
        if not isinstance(prediction, TableContinuationPrediction) or not isinstance(
            prediction.provenance, Mapping
        ):
            raise ReaderError(
                "invalid_table_continuation_prediction",
                "Table continuation classifier returned an invalid prediction",
            )
        if (
            isinstance(prediction.score, bool)
            or not isinstance(prediction.score, int | float)
            or not math.isfinite(prediction.score)
            or not 0 <= prediction.score <= 1
        ):
            raise ReaderError(
                "invalid_table_continuation_score",
                "Table continuation classifier returned a score outside 0 to 1",
            )
        if prediction.score < self.minimum_score:
            return None
        return _ContinuationEdge(first, second, prediction)


def _boundary_table(page: PageResult, *, last: bool) -> _TablePart | None:
    parts = [
        part
        for region in page.regions
        if region.kind == "table" and region.resolution == "resolved"
        if (part := _table_part(page.page_number, region)) is not None
    ]
    if not parts:
        return None
    parts.sort(
        key=lambda part: (
            part.table.bounding_box.top,
            part.table.bounding_box.left,
            part.table.bounding_box.bottom,
            part.table.bounding_box.right,
            part.table.id,
        )
    )
    return parts[-1] if last else parts[0]


def _table_part(page_number: int, table: TextRegion) -> _TablePart | None:
    structure = table.structure
    if not isinstance(structure, dict) or structure.get("role") != "table":
        return None
    try:
        topology = validate_table_topology(structure)
    except TableTopologyError:
        return None
    if topology.row_count < 2:
        return None
    header_is_valid, signature = _header_signature(topology)
    if not header_is_valid:
        return None
    return _TablePart(page_number, table, topology, signature)


def _header_signature(
    topology: TableTopology,
) -> tuple[bool, tuple[tuple[tuple[int, ...], str], ...] | None]:
    first_row = [cell for cell in topology.cells if 0 in cell.rows]
    marked_headers = [cell for cell in first_row if cell.value.get("column_header")]
    if not marked_headers:
        return True, None
    if any(
        cell.rows != (0,) or not cell.value.get("column_header") for cell in first_row
    ):
        return False, None
    signature = tuple(
        (cell.columns, _normalized_text(cell.value.get("text", "")))
        for cell in first_row
    )
    if not any(text for _, text in signature):
        return False, None
    return True, signature


def _normalized_text(value: Any) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value)).casefold().split())


def _compatible(first: _TablePart, second: _TablePart) -> bool:
    if first.topology.column_count != second.topology.column_count:
        return False
    if first.header_signature is None:
        return second.header_signature is None
    if second.header_signature is None:
        return True
    return first.header_signature == second.header_signature


def _edge_chains(
    edges: Sequence[_ContinuationEdge],
) -> list[list[_ContinuationEdge]]:
    chains: list[list[_ContinuationEdge]] = []
    for edge in edges:
        if chains:
            previous = chains[-1][-1].second
            if (
                previous.page_number == edge.first.page_number
                and previous.table is edge.first.table
            ):
                chains[-1].append(edge)
                continue
        chains.append([edge])
    return chains


def _merge_chain(
    edges: Sequence[_ContinuationEdge],
    artifact_index: int,
    classifier_name: str,
) -> TableContinuation:
    parts = [edges[0].first, *(edge.second for edge in edges)]
    explicit_headers = {
        part.header_signature for part in parts if part.header_signature is not None
    }
    if len(explicit_headers) > 1:
        raise ReaderError(
            "incompatible_table_continuation_headers",
            "Cross-page table chain contains incompatible explicit headers",
        )

    cells: list[dict[str, Any]] = []
    row_offset = 0
    retained_header = False
    omitted_header_count = 0
    for part in parts:
        omit_header = part.header_signature is not None and retained_header
        if part.header_signature is not None:
            retained_header = True
        if omit_header:
            omitted_header_count += 1
        for topology_cell in part.topology.cells:
            if omit_header and topology_cell.rows == (0,):
                continue
            cell = copy.deepcopy(dict(topology_cell.value))
            cell["row_nums"] = [
                row + row_offset - (1 if omit_header else 0)
                for row in topology_cell.rows
            ]
            cell["continuation_source"] = {
                "page_number": part.page_number,
                "table_id": part.table.id,
                "row_nums": list(topology_cell.rows),
            }
            cells.append(cell)
        row_offset += part.topology.row_count - (1 if omit_header else 0)

    try:
        validate_table_topology(
            {
                "row_count": row_offset,
                "column_count": parts[0].topology.column_count,
                "cells": cells,
            }
        )
    except TableTopologyError as error:
        raise ReaderError(
            "invalid_table_continuation_topology",
            f"Cross-page table concatenation produced invalid topology: {error}",
        ) from error

    return TableContinuation(
        id=f"cross-page-table-{artifact_index}",
        source_table_ids=[part.table.id for part in parts],
        source_page_numbers=[part.page_number for part in parts],
        score=min(float(edge.prediction.score) for edge in edges),
        header_row_count=1 if parts[0].header_signature is not None else 0,
        row_count=row_offset,
        column_count=parts[0].topology.column_count,
        cells=cells,
        provenance={
            "method": "pairwise_classifier_with_structural_guard",
            "classifier": classifier_name,
            "guards": {
                "adjacent_pages": True,
                "valid_source_topology": True,
                "matching_column_count": True,
                "compatible_explicit_headers": True,
                "repeated_headers_omitted": omitted_header_count,
                "valid_merged_topology": True,
            },
            "pair_predictions": [
                {
                    "first_page_number": edge.first.page_number,
                    "second_page_number": edge.second.page_number,
                    "first_table_id": edge.first.table.id,
                    "second_table_id": edge.second.table.id,
                    "score": float(edge.prediction.score),
                    "classifier": copy.deepcopy(dict(edge.prediction.provenance)),
                }
                for edge in edges
            ],
        },
    )
