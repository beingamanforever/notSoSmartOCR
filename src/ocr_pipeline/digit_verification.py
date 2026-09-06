"""Confirm digit-bearing values with an independent reader.

Prose carries enough redundancy that a single wrong character is usually visible. A digit
string does not: one substitution silently changes an identifier, a dose, a date or a ZIP
and the primary reader can still report high confidence. This stage asks a second reader
about those values and surfaces disagreement as review evidence rather than guessing.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Protocol

from PIL import Image, UnidentifiedImageError

from .contracts import TextAlternative, TextRegion
from .providers import ReaderError

MEASURED_KINDS = frozenset({"text", "word"})
# Evidence other specialists already reread on disagreement.
EXCLUDED_OWNERS = frozenset(
    {
        "equation",
        "formula",
        "math",
        "table",
        "table_candidate",
        "table_cell",
        "table_source",
    }
)


class CropReader(Protocol):
    name: str

    def read(self, image_path: Path, page_number: int) -> list[TextRegion]: ...


class DigitVerificationStage:
    """Reread values containing digits and record independent agreement."""

    name = "digit-verification"

    def __init__(
        self,
        reader: CropReader,
        *,
        text_provider: str | None = None,
        max_regions: int = 24,
        padding: int = 4,
    ) -> None:
        if max_regions <= 0:
            raise ValueError("max_regions must be positive")
        if padding < 0:
            raise ValueError("padding must not be negative")
        self.reader = reader
        self.text_provider = text_provider
        self.max_regions = max_regions
        self.padding = padding

    def apply(
        self,
        image_path: Path,
        page_number: int,
        regions: list[TextRegion],
    ) -> list[TextRegion]:
        selected = self._selected(regions)
        if not selected:
            return regions

        try:
            with Image.open(image_path) as opened:
                page = opened.convert("RGB")
        except (OSError, UnidentifiedImageError) as error:
            raise ReaderError("digit_verification_failed", str(error)) from error

        try:
            with TemporaryDirectory() as root:
                for index, region in enumerate(selected, start=1):
                    candidate = self._read_region(page, region, Path(root), index)
                    if candidate is None:
                        continue
                    _record(region, candidate, self.reader.name)
        finally:
            page.close()
        return regions

    def _selected(self, regions: list[TextRegion]) -> list[TextRegion]:
        eligible = [
            region
            for region in regions
            if region.kind in MEASURED_KINDS
            and region.resolution == "resolved"
            and _digits(region.text)
            and not _owned_by_another_specialist(region)
            and (self.text_provider is None or region.provider == self.text_provider)
        ]
        # Longer digit runs carry the most silent risk, so verify those first.
        eligible.sort(key=lambda region: (-len(_digits(region.text)), region.id))
        return eligible[: self.max_regions]

    def _read_region(
        self,
        page: Image.Image,
        region: TextRegion,
        root: Path,
        index: int,
    ) -> str | None:
        box = region.bounding_box
        crop_box = (
            max(0, box.left - self.padding),
            max(0, box.top - self.padding),
            min(page.width, box.right + self.padding),
            min(page.height, box.bottom + self.padding),
        )
        if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
            return None
        path = root / f"digits-{index}.png"
        crop = page.crop(crop_box)
        try:
            crop.save(path, format="PNG")
        except Exception:
            return None
        finally:
            crop.close()
        try:
            read = self.reader.read(path, 1)
        except ReaderError:
            return None
        text = " ".join(item.text.strip() for item in read if item.text.strip())
        return text or None


def _record(region: TextRegion, candidate: str, provider: str) -> None:
    """Compare digits only: readers legitimately differ on spacing and punctuation."""
    incumbent_digits = _digits(region.text)
    candidate_digits = _digits(candidate)
    if not candidate_digits:
        return
    provenance = {
        "method": "independent_digit_reread",
        "provider": provider,
        "candidate": candidate,
    }
    structure = dict(region.structure or {})
    if candidate_digits == incumbent_digits:
        structure["digit_verification"] = {"outcome": "confirmed", **provenance}
        region.structure = structure
        return
    structure["digit_verification"] = {"outcome": "disagreed", **provenance}
    structure["review_required"] = True
    region.structure = structure
    region.resolution = "conflicting"
    region.alternatives.append(
        TextAlternative(
            text=candidate,
            confidence=None,
            provider=provider,
            text_provenance=provenance,
        )
    )


def _digits(text: str) -> str:
    return "".join(character for character in text if character.isdigit())


def _owned_by_another_specialist(region: TextRegion) -> bool:
    structure = region.structure if isinstance(region.structure, dict) else {}
    labels = {
        str(structure.get(name, "")).strip().casefold()
        for name in ("role", "layout_owner_type", "block_type", "semantic_class")
    }
    return bool(labels & EXCLUDED_OWNERS)
