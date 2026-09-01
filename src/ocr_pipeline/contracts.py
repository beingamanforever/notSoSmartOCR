"""JSON-serializable results shared by readers and callers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol


@dataclass
class BoundingBox:
    left: int
    top: int
    right: int
    bottom: int


@dataclass
class TextAlternative:
    text: str
    confidence: float | None
    provider: str
    text_provenance: dict[str, Any] | None = None


@dataclass
class TextRegion:
    id: str
    kind: str
    text: str
    confidence: float | None
    bounding_box: BoundingBox
    reading_order: int
    provider: str
    text_provenance: dict[str, Any] | None = None
    resolution: Literal["resolved", "unreadable", "conflicting"] = "resolved"
    alternatives: list[TextAlternative] = field(default_factory=list)
    structure: dict[str, Any] | None = None


class RegionStage(Protocol):
    name: str

    def apply(
        self,
        image_path: Path,
        page_number: int,
        regions: list[TextRegion],
    ) -> list[TextRegion]: ...


@dataclass
class EvidenceText:
    value: str
    evidence_ids: list[str]


@dataclass
class Failure:
    id: str
    stage: str
    code: str
    message: str
    page_number: int | None = None


@dataclass
class PageResult:
    page_number: int
    width: int
    height: int
    reader: str
    route: str
    text: EvidenceText
    regions: list[TextRegion] = field(default_factory=list)
    failure_ids: list[str] = field(default_factory=list)


@dataclass
class DocumentResult:
    document_id: str
    source: dict[str, str]
    status: str
    pages: list[PageResult] = field(default_factory=list)
    failures: list[Failure] = field(default_factory=list)
    schema_version: int = 2

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
