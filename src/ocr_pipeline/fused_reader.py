"""Read a page with Falcon's text placed on Nemotron's geometry.

Each reader is strong where the other is weak. Falcon-Perception parses a page far
better - it keeps checkbox glyphs, tables and headings - but it returns one blob of text
per detected region, with no geometry inside that region and only the layout detector's
score, which says nothing about whether the characters are right. Nemotron OCR v2 returns
a box and a recognition confidence for every word, but reads the characters worse.

So the text comes from Falcon and the geometry and confidence come from Nemotron:

- a Falcon region takes the recognition confidence of the words that fall inside it, and
  keeps those word boxes as its evidence, so a highlight points at a real pixel box;
- words that fall inside no Falcon region are content Falcon's layout detector never
  proposed. They become regions of their own, grouped into lines, so a page whose body
  is detected as one table no longer loses everything around it.

Nothing here invents geometry. A Falcon region with no words under it keeps its own box
and says so in its provenance.

A region can also under-read: it claims words whose text never made it into the region's
own text (the JPMorgan table region claims 289 words against ~1000 characters of table
text). Those words carry real boxes and confidences, so the runs of claimed-but-absent
words are emitted as child regions of the under-read region - recovered from measured
geometry, never from an un-located transcript.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import replace
from difflib import SequenceMatcher
from pathlib import Path
from statistics import fmean
from typing import Protocol

from .contracts import BoundingBox, TextRegion
from .providers import ReaderError

PROVIDER = "falcon-on-nemotron"
# A word counts towards a region when its centre lies inside that region's box.
INSIDE = "centre"
# Comparison alphabet for "did the region's text include this word": case, whitespace
# and punctuation differences between the two readers are not missing content.
TOKEN = re.compile(r"[a-z0-9%$]+")
# Recovered content must come from a confident read. Measured across the 18-page set:
# garbled re-reads of text the region already carries score 0.34-0.70, while every
# genuine recovery (the financial page's bullet column, 63 lines) scores >= 0.77.
UNDERREAD_MIN_CONFIDENCE = 0.75


class GeometryReader(Protocol):
    def read_with_merge_level(
        self, image_path: Path, page_number: int, merge_level: str
    ) -> list[TextRegion]: ...


class TextReader(Protocol):
    def read(self, image_path: Path, page_number: int) -> list[TextRegion]: ...


class FusedReader:
    """Falcon's text, Nemotron's boxes and recognition confidence."""

    name = PROVIDER

    def __init__(
        self,
        text_reader: TextReader,
        geometry_reader: GeometryReader,
        *,
        merge_level: str = "word",
    ) -> None:
        self.text_reader = text_reader
        self.geometry_reader = geometry_reader
        self.merge_level = merge_level

    def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
        regions = self.text_reader.read(image_path, page_number)
        try:
            words = self.geometry_reader.read_with_merge_level(
                image_path, page_number, self.merge_level
            )
        except ReaderError as error:
            # Preserve the transcript while making unavailable recognition explicit.
            return [
                replace(
                    region,
                    confidence=region.confidence,
                    text_provenance={
                        **(region.text_provenance or {}),
                        "geometry_status": "unavailable",
                        "recognition_error": error.code,
                    },
                )
                for region in regions
            ]
        return fuse(regions, words, page_number)


def fuse(
    regions: list[TextRegion], words: list[TextRegion], page_number: int
) -> list[TextRegion]:
    """Attach word geometry to each region, then recover the words no region covers."""
    claimed: set[int] = set()
    fused = []
    for region in regions:
        if region.kind in {"figure", "checkbox"}:
            fused.append(region)
            continue
        inside = [
            index
            for index, word in enumerate(words)
            if index not in claimed
            and _centre_inside(word.bounding_box, region.bounding_box)
        ]
        claimed.update(inside)
        claimed_words = [words[index] for index in inside]
        enriched = _with_geometry(region, claimed_words)
        fused.append(enriched)
        fused.extend(_underread_children(enriched, claimed_words))
    fused.extend(
        _uncovered_regions(
            [word for index, word in enumerate(words) if index not in claimed],
            page_number,
            len(fused),
        )
    )
    return fused


def _with_geometry(
    region: TextRegion,
    words: list[TextRegion],
) -> TextRegion:
    provenance = dict(region.text_provenance or {})
    if not words:
        provenance["geometry"] = "no recognised word fell inside this region"
        provenance["geometry_status"] = "unavailable"
        native = (provenance.get("generation") or {}).get("token_score")
        if native is None:
            provenance["recognition_status"] = "unavailable"
        return replace(region, confidence=native, text_provenance=provenance)

    scores = [word.confidence for word in words if isinstance(word.confidence, float)]
    # A recognizer's score belongs to its own transcript. Geometry alone cannot
    # certify a different reader's characters, especially digits and punctuation.
    recognition_text = " ".join(word.text for word in words)
    agreement = " ".join(region.text.split()) == " ".join(recognition_text.split())
    provenance.update(
        {
            "geometry_provider": words[0].provider,
            "word_count": len(words),
            "confidence_meaning": "recognition, from the word reader",
            "recognition_confidence_min": min(scores) if scores else None,
            "recognition_text": recognition_text,
            "recognition_confidence_mean": fmean(scores) if scores else None,
            "recognition_agreement": agreement,
        }
    )
    structure = dict(region.structure or {})
    # in_region_text: whether the region's own text contains the word. A claimed word
    # that is absent was read off the page and then lost - the under-read repair and
    # the table stage's cell fill both key off this flag.
    structure["word_evidence"] = [
        _word_entry(word, present)
        for word, present in zip(words, _present_flags(region.text, words))
    ]
    return replace(
        region,
        confidence=(provenance.get("generation") or {}).get("token_score")
        if (provenance.get("generation") or {}).get("token_score") is not None
        else fmean(scores)
        if scores and agreement
        else None,
        text_provenance=provenance,
        structure=structure,
    )


def _uncovered_regions(
    words: list[TextRegion], page_number: int, offset: int
) -> list[TextRegion]:
    """Words no region claimed, grouped into the lines they were read on."""
    lines = _lines(words)
    return [
        TextRegion(
            id=f"p{page_number}-fused-{index}",
            kind="text",
            text=" ".join(word.text for word in line).strip(),
            confidence=fmean(
                [word.confidence for word in line if isinstance(word.confidence, float)]
                or [0.0]
            ),
            bounding_box=_union(line),
            reading_order=offset + index,
            provider=PROVIDER,
            text_provenance={
                "method": "word_reader_uncovered",
                "geometry_provider": line[0].provider,
                "confidence_meaning": "recognition, from the word reader",
                "recovered": "no layout region contained these words",
                "word_count": len(line),
            },
            resolution="resolved",
            structure={"word_evidence": [_word_entry(word, True) for word in line]},
        )
        for index, line in enumerate(lines, start=1)
        if any(word.text.strip() for word in line)
    ]


def _word_entry(word: TextRegion, present: bool) -> dict[str, object]:
    return {
        "text": word.text,
        "bbox": _box_dict(word.bounding_box),
        "confidence": word.confidence,
        "in_region_text": present,
    }


def _present_flags(text: str, words: list[TextRegion]) -> list[bool]:
    """Whether each claimed word's text survived into the region's own text.

    The geometry reader (Nemotron) sometimes returns a whole printed line as a single
    "word", re-read in its own pass rather than copied from Falcon's text - so it carries
    its own character noise ("foctnoted" for "footnoted", "Frie Browwn" for "Eric Brown").
    An exact token multiset test would call that whole line absent and re-emit it as a
    near-duplicate child region. A fuzzy match on the tokens that miss exactly tells a
    re-read with typos (almost every token close to one available in the text) apart from
    content the region genuinely never read (few or no tokens in common). A word counts
    as present once at least three quarters of its tokens are accounted for, exactly or
    fuzzily. A word with no comparable token (a bare glyph or punctuation mark) is treated
    as present: there is no text to recover for it, and emitting it would invent content.
    """
    available = Counter(TOKEN.findall(text.casefold()))
    flags = []
    for word in words:
        tokens = TOKEN.findall(word.text.casefold())
        if not tokens:
            flags.append(True)
            continue
        consumed: Counter = Counter()
        matched = 0
        for token in tokens:
            if available[token] - consumed[token] > 0:
                consumed[token] += 1
                matched += 1
                continue
            if len(token) < 4:
                continue
            fuzzy_key = _fuzzy_match(token, available, consumed)
            if fuzzy_key is not None:
                consumed[fuzzy_key] += 1
                matched += 1
        present = matched >= len(tokens) * 0.75
        if present:
            available.subtract(consumed)
        flags.append(present)
    return flags


def _fuzzy_match(token: str, available: Counter, consumed: Counter) -> str | None:
    """The best still-available token within edit distance of a token that missed exactly.

    A cheap prefilter (length within 2, sharing a first or last character) runs before any
    difflib comparison, since a table region can claim hundreds of words against a text of
    thousands of characters and most tokens either hit exactly or share nothing with any
    available token. quick_ratio further screens out weak candidates before the real
    (more expensive) ratio is computed. Keys are visited in sorted order so that ties
    resolve the same way on every run.
    """
    matcher = SequenceMatcher(a=token)
    best_key, best_ratio = None, 0.0
    for key in sorted(available):
        if available[key] - consumed[key] <= 0:
            continue
        if abs(len(key) - len(token)) > 2:
            continue
        if key[0] != token[0] and key[-1] != token[-1]:
            continue
        matcher.set_seq2(key)
        if matcher.quick_ratio() < 0.75:
            continue
        ratio = matcher.ratio()
        if ratio >= 0.75 and ratio > best_ratio:
            best_key, best_ratio = key, ratio
    return best_key


def _underread_children(
    region: TextRegion, words: list[TextRegion]
) -> list[TextRegion]:
    """Emit the region's claimed-but-absent words as child regions with real geometry.

    Safe precisely because these words carry their own boxes and confidences: an earlier
    repair that injected un-located full-page text produced 1565 spurious regions.
    """
    evidence = (region.structure or {}).get("word_evidence") or []
    absent = [
        word
        for word, entry in zip(words, evidence)
        if not entry["in_region_text"] and word.text.strip()
    ]
    if not absent:
        return []
    lines = [
        line
        for line in _lines(absent)
        if fmean(
            [word.confidence for word in line if isinstance(word.confidence, float)]
            or [0.0]
        )
        >= UNDERREAD_MIN_CONFIDENCE
    ]
    children = [
        TextRegion(
            id=f"{region.id}-underread-{index}",
            kind="text",
            text=" ".join(word.text for word in line).strip(),
            confidence=fmean(
                [word.confidence for word in line if isinstance(word.confidence, float)]
                or [0.0]
            ),
            bounding_box=_union(line),
            reading_order=region.reading_order,
            provider=PROVIDER,
            text_provenance={
                "method": "region_underread_repair",
                "parent_region_id": region.id,
                "geometry_provider": line[0].provider,
                "confidence_meaning": "recognition, from the word reader",
                "recovered": "claimed by the region but absent from its text",
                "word_count": len(line),
            },
            resolution="resolved",
            structure={"word_evidence": [_word_entry(word, True) for word in line]},
        )
        for index, line in enumerate(lines, start=1)
    ]
    provenance = dict(region.text_provenance or {})
    provenance["underread_recovered_words"] = len(absent)
    provenance["underread_child_ids"] = [child.id for child in children]
    region.text_provenance = provenance
    return children


def _lines(words: list[TextRegion]) -> list[list[TextRegion]]:
    """Group words that share a row, in reading order within the row."""
    lines: list[list[TextRegion]] = []
    for word in sorted(
        words, key=lambda item: (item.bounding_box.top, item.bounding_box.left)
    ):
        placed = next(
            (
                line
                for line in lines
                if _same_row(line[-1].bounding_box, word.bounding_box)
            ),
            None,
        )
        if placed is None:
            lines.append([word])
        else:
            placed.append(word)
    return [sorted(line, key=lambda item: item.bounding_box.left) for line in lines]


def _same_row(left: BoundingBox, right: BoundingBox) -> bool:
    """Whether two boxes overlap vertically by most of the shorter one's height."""
    overlap = min(left.bottom, right.bottom) - max(left.top, right.top)
    shorter = min(left.bottom - left.top, right.bottom - right.top)
    return shorter > 0 and overlap >= shorter / 2


def _centre_inside(word: BoundingBox, region: BoundingBox) -> bool:
    x = (word.left + word.right) / 2
    y = (word.top + word.bottom) / 2
    return region.left <= x <= region.right and region.top <= y <= region.bottom


def _union(words: list[TextRegion]) -> BoundingBox:
    boxes = [word.bounding_box for word in words]
    return BoundingBox(
        min(box.left for box in boxes),
        min(box.top for box in boxes),
        max(box.right for box in boxes),
        max(box.bottom for box in boxes),
    )


def _box_dict(box: BoundingBox) -> dict[str, int]:
    return {
        "left": box.left,
        "top": box.top,
        "right": box.right,
        "bottom": box.bottom,
    }
