"""Review-only Falcon transcriptions over canonical region geometry."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from PIL import Image, UnidentifiedImageError

from .contracts import (
    BoundingBox,
    EvidenceText,
    PageResult,
    TextAlternative,
    TextRegion,
)
from .evidence_layout import formula_ink_box
from .falcon import FalconOCRReader, falcon_token_limit
from .providers import ReaderError
from .table_topology import TableTopologyError, validate_table_topology

_CATEGORY_BY_LABEL = {
    "text": "text",
    "table": "table",
    "formula": "formula",
    "equation": "formula",
    "math": "formula",
    "title": "title",
    "section_header": "section-header",
    "heading": "section-header",
    "caption": "caption",
    "footnote": "footnote",
    "list_item": "list-item",
    "page_header": "page-header",
    "page_footer": "page-footer",
    "paragraph": "text",
    "line": "text",
    "form_row": "text",
    "text_block": "text",
    "list": "list-item",
    "header": "page-header",
    "footer": "page-footer",
}
_EXCLUDED_LABELS = frozenset(
    {
        "coverage_risk",
        "control",
        "checkbox",
        "table_cell",
        "table_source",
        "table_candidate",
        "candidate",
    }
)


@dataclass(frozen=True)
class _CropRequest:
    source: TextRegion
    category: str
    bounding_box: BoundingBox
    segment: dict[str, Any] | None = None


class FalconPresentationReader:
    """Batch category-routed crops without changing canonical page evidence."""

    name = "falcon-presentation"

    def __init__(
        self,
        reader: FalconOCRReader,
        *,
        max_crops: int = 32,
        max_table_segment_height: int = 1536,
        formula_padding: int = 12,
    ) -> None:
        if max_crops <= 0:
            raise ValueError("Falcon presentation max_crops must be positive")
        if max_table_segment_height <= 0:
            raise ValueError("Falcon table segment height must be positive")
        if formula_padding < 0:
            raise ValueError("Falcon formula padding cannot be negative")
        self.reader = reader
        self.max_crops = max_crops
        self.max_table_segment_height = max_table_segment_height
        self.formula_padding = formula_padding

    def read_page(self, image_path: Path, page: PageResult) -> list[TextRegion]:
        handwriting_ids = {
            region.id for region in page.regions if _is_handwriting_region(region)
        }
        if not any(_category(region, handwriting_ids) for region in page.regions):
            return []

        try:
            with Image.open(image_path) as source:
                image = source.convert("RGB")
        except (OSError, UnidentifiedImageError, ValueError) as error:
            raise ReaderError(
                "falcon_presentation_image_failed",
                "Falcon presentation image could not be prepared",
            ) from error

        selected = self._select(page.regions, image)
        if not selected:
            image.close()
            return []

        crops = [image.crop(_box_tuple(item.bounding_box)) for item in selected]
        image.close()
        try:
            categories = [item.category for item in selected]
            outputs = self.reader.transcribe_crops(crops, categories)
        except ReaderError:
            raise
        except Exception as error:
            raise ReaderError("falcon_presentation_failed", str(error)) from error
        finally:
            for crop in crops:
                crop.close()

        if not isinstance(outputs, list) or len(outputs) != len(selected):
            raise ReaderError(
                "falcon_presentation_output_failed",
                "Falcon presentation returned the wrong number of crop transcriptions",
            )

        regions = []
        for item, output in zip(selected, outputs, strict=True):
            source = item.source
            category = item.category
            if not isinstance(output, str) or not output.strip():
                raise ReaderError(
                    "falcon_presentation_output_failed",
                    "Falcon presentation returned invalid crop text",
                )
            validation = {
                "raw_response_preserved": True,
                "nonempty": True,
                "exact_repetition_loop": False,
                "termination_observable": False,
                "truncation_observable": False,
            }
            generation = {**copy.deepcopy(self.reader.generation), "category": category}
            configured_limit = generation.get("max_new_tokens")
            if isinstance(configured_limit, int):
                generation["effective_max_new_tokens"] = falcon_token_limit(
                    category, configured_limit
                )
            source_evidence_ids = _string_values(
                (source.structure or {}).get("child_evidence_ids", [])
            )
            provenance = {
                "method": "falcon_core_category_crop_generation",
                "review_only": True,
                "source_region_id": source.id,
                "source_kind": source.kind,
                "category": category,
                "model": copy.deepcopy(self.reader.provenance),
                "generation": generation,
                "raw_response": output,
                "output_validation": validation,
            }
            structure = {
                "role": "presentation_challenger",
                "review_only": True,
                "category": category,
                "source_region_id": source.id,
                "source_kind": source.kind,
                "control_glyphs_authoritative": False,
                "output_validation": copy.deepcopy(validation),
            }
            if source_evidence_ids:
                provenance["source_evidence_ids"] = source_evidence_ids
                structure["source_evidence_ids"] = copy.deepcopy(source_evidence_ids)
            if item.segment is not None:
                provenance["table_segment"] = copy.deepcopy(item.segment)
                structure["table_segment"] = copy.deepcopy(item.segment)
            regions.append(
                TextRegion(
                    id=_presentation_id(source.id, item.segment),
                    kind=source.kind,
                    text=output,
                    confidence=None,
                    bounding_box=copy.deepcopy(item.bounding_box),
                    reading_order=source.reading_order,
                    provider=self.name,
                    text_provenance=provenance,
                    structure=structure,
                )
            )
        return regions

    def _select(
        self,
        regions: list[TextRegion],
        image: Image.Image,
    ) -> list[_CropRequest]:
        selected: list[_CropRequest] = []
        seen: set[tuple[object, ...]] = set()
        image_size = image.size
        formula_mask: Image.Image | None = None
        handwriting_ids = {
            region.id for region in regions if _is_handwriting_region(region)
        }
        for region in regions:
            category = _category(region, handwriting_ids)
            if category is None or not _valid_box(region.bounding_box, image_size):
                continue
            if category == "formula" and formula_mask is None:
                formula_mask = image.convert("L").point(
                    lambda value: 255 if value < 220 else 0
                )
            requests = self._crop_requests(
                region,
                category,
                image_size,
                formula_mask,
            )
            if not requests:
                continue
            keys = [_request_key(item) for item in requests]
            if any(key in seen for key in keys):
                continue
            if len(selected) + len(requests) > self.max_crops:
                continue
            seen.update(keys)
            selected.extend(requests)
            if len(selected) == self.max_crops:
                break
        if formula_mask is not None:
            formula_mask.close()
        return selected

    def _crop_requests(
        self,
        region: TextRegion,
        category: str,
        image_size: tuple[int, int],
        formula_mask: Image.Image | None,
    ) -> list[_CropRequest]:
        if category == "formula":
            structure = region.structure if isinstance(region.structure, dict) else {}
            grouped_crop = _mapping_box(structure.get("formula_crop"))
            return [
                _CropRequest(
                    region,
                    category,
                    _padded_box(grouped_crop, image_size, self.formula_padding)
                    if grouped_crop is not None and _valid_box(grouped_crop, image_size)
                    else formula_ink_box(
                        formula_mask,
                        region.bounding_box,
                        image_size,
                        self.formula_padding,
                    )
                    if formula_mask is not None
                    else _padded_box(
                        region.bounding_box,
                        image_size,
                        self.formula_padding,
                    ),
                )
            ]
        if category != "table":
            return [_CropRequest(region, category, region.bounding_box)]
        structure = region.structure if isinstance(region.structure, dict) else {}
        try:
            topology = validate_table_topology(structure)
        except TableTopologyError:
            if (
                region.bounding_box.bottom - region.bounding_box.top
                <= self.max_table_segment_height
            ):
                return [_CropRequest(region, category, region.bounding_box)]
            return []
        if (
            region.bounding_box.bottom - region.bounding_box.top
            <= self.max_table_segment_height
        ):
            return [_CropRequest(region, category, region.bounding_box)]
        return _table_segment_requests(
            region,
            topology.cells,
            topology.row_count,
            image_size,
            self.max_table_segment_height,
        )


class FalconFormulaStage:
    """Record Falcon formula rereads as review evidence on canonical owners."""

    name = "falcon-formula"

    def __init__(self, reader: FalconPresentationReader) -> None:
        self.reader = reader

    def apply(
        self,
        image_path: Path,
        page_number: int,
        regions: list[TextRegion],
    ) -> list[TextRegion]:
        handwriting_ids = {
            region.id for region in regions if _is_handwriting_region(region)
        }
        owners = [
            region
            for region in regions
            if _category(region, handwriting_ids) == "formula"
        ]
        if not owners:
            return regions

        regions_by_id = {region.id: region for region in regions}
        page = PageResult(
            page_number=page_number,
            width=0,
            height=0,
            reader=self.reader.name,
            route="review",
            text=EvidenceText("", []),
            regions=owners,
        )
        owners_by_id = {owner.id: owner for owner in owners}
        for candidate in self.reader.read_page(image_path, page):
            candidate_provenance = candidate.text_provenance or {}
            candidate_text = _formula_candidate_text(candidate.text)
            source_id = candidate_provenance.get("source_region_id")
            owner = owners_by_id.get(source_id)
            if owner is None:
                continue
            backend = getattr(self.reader.reader, "name", candidate.provider)
            provenance = {
                **copy.deepcopy(candidate_provenance),
                "reader": backend,
                "presentation_reader": candidate.provider,
                "crop": {"bounding_box": asdict(candidate.bounding_box)},
                "page_number": page_number,
                "source_region_id": owner.id,
                "candidate_latex": candidate_text,
            }
            if not candidate_text:
                _record_formula_attempt(
                    owner,
                    "invalid_output",
                    "empty_normalized_formula",
                    provenance,
                )
                continue
            source_providers = sorted(
                {
                    child.provider
                    for child_id in (owner.structure or {}).get(
                        "child_evidence_ids", []
                    )
                    if isinstance(child_id, str)
                    and (child := regions_by_id.get(child_id)) is not None
                }
                or {owner.provider}
            )
            provenance["source_providers"] = source_providers
            if candidate_text == owner.text:
                same_reader = backend in source_providers
                _record_formula_attempt(
                    owner,
                    "same_reader_repeat" if same_reader else "supported",
                    (
                        "reader_not_independent"
                        if same_reader
                        else "independent_reader_matches_canonical"
                    ),
                    provenance,
                )
                if not same_reader:
                    structure = dict(owner.structure or {})
                    structure["formula_recognition"] = "specialist_supported"
                    owner.structure = structure
                continue

            if not any(
                alternative.decision_state == "pending"
                and alternative.text == candidate_text
                for alternative in owner.alternatives
            ):
                owner.alternatives.append(
                    TextAlternative(
                        text=candidate_text,
                        confidence=None,
                        provider=backend,
                        text_provenance=copy.deepcopy(provenance),
                    )
                )
            _record_formula_attempt(
                owner,
                "candidate_pending",
                "specialist_disagrees_with_canonical",
                provenance,
            )
            structure = dict(owner.structure or {})
            structure["formula_recognition"] = "specialist_pending"
            structure["formula_review"] = {
                "required": True,
                "reasons": ["structural_disagreement"],
                "provenance": copy.deepcopy(provenance),
            }
            owner.structure = structure
        return regions


def _category(region: TextRegion, handwriting_ids: set[str]) -> str | None:
    structure = region.structure if isinstance(region.structure, dict) else {}
    if structure.get("formula_recognition") == "model":
        return None
    provenance = (
        region.text_provenance if isinstance(region.text_provenance, dict) else {}
    )
    if structure.get("block_type") == "form_row" or _is_handwriting_region(
        region, handwriting_ids
    ):
        return None
    if structure.get("layout_owner_id"):
        return None
    if _label(region.kind) in {"word", "token"} or (
        _label(region.kind) == "text" and provenance.get("merge_level") == "word"
    ):
        return None
    labels = (
        _label(region.kind),
        _label(structure.get("role")),
        _label(structure.get("block_type")),
        _label(structure.get("semantic_class")),
    )
    if any(
        label in _EXCLUDED_LABELS or label.endswith("_candidate") for label in labels
    ):
        return None
    categories = [
        _CATEGORY_BY_LABEL[label] for label in labels if label in _CATEGORY_BY_LABEL
    ]
    category = next(
        (candidate for candidate in categories if candidate != "text"),
        "text" if "text" in categories else None,
    )
    if category == "formula" and "formula_attempt" in structure:
        return None
    if category == "text" and region.resolution == "resolved":
        return None
    if category == "table" and not _table_needs_presentation(region, structure):
        return None
    return category


def _table_needs_presentation(region: TextRegion, structure: dict[str, Any]) -> bool:
    try:
        topology = validate_table_topology(structure)
    except TableTopologyError:
        return region.resolution != "resolved"
    return region.resolution != "resolved" or any(
        cell.value.get("resolution", "resolved") != "resolved"
        for cell in topology.cells
    )


def _is_handwriting_region(
    region: TextRegion, handwriting_ids: set[str] | None = None
) -> bool:
    structure = region.structure if isinstance(region.structure, dict) else {}
    provenance = (
        region.text_provenance if isinstance(region.text_provenance, dict) else {}
    )
    labels = (
        _label(region.kind),
        _label(structure.get("role")),
        _label(structure.get("block_type")),
        _label(structure.get("semantic_class")),
        _label(provenance.get("style")),
    )
    if any("handwrit" in label for label in labels):
        return True
    if any(
        structure.get(name) is True
        for name in ("handwriting_candidate", "handwritten", "is_handwritten")
    ):
        return True
    if handwriting_ids:
        child_ids = structure.get("child_evidence_ids", [])
        return isinstance(child_ids, list) and any(
            child_id in handwriting_ids for child_id in child_ids
        )
    return False


def _table_segment_requests(
    region: TextRegion,
    topology_cells: tuple[Any, ...],
    row_count: int,
    image_size: tuple[int, int],
    max_height: int,
) -> list[_CropRequest]:
    cells: list[tuple[Any, BoundingBox]] = []
    for cell in topology_cells:
        box = _mapping_box(cell.value.get("bbox"))
        if box is None or not _inside(box, region.bounding_box):
            return []
        cells.append((cell, box))

    row_groups = _indivisible_row_groups(topology_cells, row_count)
    segments: list[tuple[int, int]] = []
    for start, end in row_groups:
        if not segments:
            segments.append((start, end))
            continue
        candidate = (segments[-1][0], end)
        candidate_box = _row_box(cells, *candidate)
        if candidate_box.bottom - candidate_box.top <= max_height:
            segments[-1] = candidate
        else:
            segments.append((start, end))

    requests: list[_CropRequest] = []
    for index, (start, end) in enumerate(segments):
        row_box = _row_box(cells, start, end)
        if row_box.bottom - row_box.top > max_height:
            return []
        crop_box = BoundingBox(
            region.bounding_box.left,
            row_box.top,
            region.bounding_box.right,
            row_box.bottom,
        )
        if not _valid_box(crop_box, image_size):
            return []
        segment_cells = [
            cell for cell, _ in cells if start <= cell.rows[0] and cell.rows[-1] < end
        ]
        segment = {
            "index": index,
            "row_start": start,
            "row_end": end,
            "source_region_id": region.id,
            "source_cell_ids": _string_values(
                cell.value.get("id") for cell in segment_cells
            ),
            "source_evidence_ids": _segment_evidence_ids(segment_cells),
        }
        requests.append(_CropRequest(region, "table", crop_box, segment))
    return requests


def _indivisible_row_groups(
    cells: tuple[Any, ...], row_count: int
) -> list[tuple[int, int]]:
    legal_boundaries = [
        row
        for row in range(1, row_count)
        if not any(cell.rows[0] < row <= cell.rows[-1] for cell in cells)
    ]
    boundaries = [0, *legal_boundaries, row_count]
    return list(zip(boundaries, boundaries[1:]))


def _row_box(cells: list[tuple[Any, BoundingBox]], start: int, end: int) -> BoundingBox:
    boxes = [
        box for cell, box in cells if start <= cell.rows[0] and cell.rows[-1] < end
    ]
    return BoundingBox(
        min(box.left for box in boxes),
        min(box.top for box in boxes),
        max(box.right for box in boxes),
        max(box.bottom for box in boxes),
    )


def _mapping_box(value: Any) -> BoundingBox | None:
    if not isinstance(value, Mapping):
        return None
    coordinates = tuple(value.get(name) for name in ("left", "top", "right", "bottom"))
    if not all(
        isinstance(item, int) and not isinstance(item, bool) for item in coordinates
    ):
        return None
    return BoundingBox(*coordinates)


def _inside(box: BoundingBox, outer: BoundingBox) -> bool:
    return (
        outer.left <= box.left < box.right <= outer.right
        and outer.top <= box.top < box.bottom <= outer.bottom
    )


def _segment_evidence_ids(cells: list[Any]) -> list[str]:
    values: list[Any] = []
    for cell in cells:
        evidence_ids = cell.value.get("evidence_ids")
        if isinstance(evidence_ids, list):
            values.extend(evidence_ids)
    return _string_values(values)


def _string_values(values: Any) -> list[str]:
    return list(dict.fromkeys(value for value in values if isinstance(value, str)))


def _request_key(request: _CropRequest) -> tuple[object, ...]:
    box = request.bounding_box
    return (
        request.source.id,
        box.left,
        box.top,
        box.right,
        box.bottom,
        request.category,
    )


def _presentation_id(source_id: str, segment: dict[str, Any] | None) -> str:
    suffix = "" if segment is None else f"-segment-{segment['index']}"
    return f"{source_id}-falcon-presentation{suffix}"


def _label(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return "_".join(value.strip().lower().replace("-", " ").split())


def _valid_box(box: BoundingBox, image_size: tuple[int, int]) -> bool:
    if not isinstance(box, BoundingBox):
        return False
    values = (box.left, box.top, box.right, box.bottom)
    if not all(
        isinstance(value, int) and not isinstance(value, bool) for value in values
    ):
        return False
    width, height = image_size
    return 0 <= box.left < box.right <= width and 0 <= box.top < box.bottom <= height


def _box_tuple(box: BoundingBox) -> tuple[int, int, int, int]:
    return box.left, box.top, box.right, box.bottom


def _padded_box(
    box: BoundingBox,
    image_size: tuple[int, int],
    padding: int,
) -> BoundingBox:
    width, height = image_size
    return BoundingBox(
        max(0, box.left - padding),
        max(0, box.top - padding),
        min(width, box.right + padding),
        min(height, box.bottom + padding),
    )


def _record_formula_attempt(
    region: TextRegion,
    outcome: str,
    reason: str,
    provenance: dict[str, Any],
) -> None:
    structure = dict(region.structure or {})
    structure["formula_attempt"] = {
        "outcome": outcome,
        "reason": reason,
        "provenance": copy.deepcopy(provenance),
    }
    region.structure = structure


def _formula_candidate_text(raw_response: str) -> str:
    text = raw_response.strip()
    if text.startswith("$$") and text.endswith("$$"):
        return text[2:-2].strip()
    if text.startswith(r"\[") and text.endswith(r"\]"):
        return text[2:-2].strip()
    return text
