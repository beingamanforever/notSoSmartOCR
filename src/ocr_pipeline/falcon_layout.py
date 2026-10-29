"""Read learned layout regions with the official Falcon OCR decoder.

Decoder scores and layout detection scores have separate provenance. A region box
does not imply word locations. Text region confidence is a calibrated probability;
table and checkbox confidence stays the uncalibrated decoder token likelihood.
"""

from __future__ import annotations

import base64
import html
import json
import io
import math
from dataclasses import replace
from contextlib import ExitStack
from collections import Counter
import re
from difflib import SequenceMatcher
import unicodedata
import urllib.error
import urllib.request
from urllib.parse import urlsplit
from html.parser import HTMLParser
from pathlib import Path
from collections.abc import Callable
from typing import Any

from PIL import Image

from .contracts import BoundingBox, TextRegion
from .providers import ReaderError
from .falcon import FALCON_SERVICE_MAX_RESPONSE_BYTES
from .text_layer import read_cells

PROVIDER = "falcon-perception"
# Match the presentation HTML span limit and bound expanded grids before allocation.
MAX_TABLE_SPAN = 100
MAX_TABLE_GRID_CELLS = 10_000
TABLE_TEXT_BLOCK_TAGS = frozenset(
    {
        "p",
        "div",
        "caption",
        "li",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "blockquote",
        "pre",
    }
)
HEADING = re.compile(r"^\s*(#{1,6})\s+(.*)$")
# Only paired asterisks: a lone one is a real footnote mark on many clinical forms.
EMPHASIS = re.compile(r"\*\*(.+?)\*\*", re.S)  # bold may span a heading's lines
# Falcon's single v1.5 prompt picks its own markup, so any region can carry these.
INLINE_TAG = re.compile(r"</?(?:sup|sub|strong|b|em|i|u|span|p|div|font)\b[^>]*>", re.I)
LINE_BREAK_TAG = re.compile(r"<br\s*/?>", re.I)
BALLOT_MARK = r"\[[xX ?]\]|[☐☑☒✓✔✗✘□■]"
DECODED_BALLOT_MARK = re.compile(BALLOT_MARK)
SELECTED_BALLOTS = frozenset({"[x]", "☑", "☒", "✓", "✔", "✗", "✘", "■"})
# Filled squares and check marks also serve as list bullets; empty boxes do not.
UNAMBIGUOUS_BALLOTS = frozenset({"[x]", "[ ]", "☐", "☑", "☒", "□"})
# Platt scaling (Platt 1999) of Falcon's mean token log-probability and Heron's
# detection score, fitted on 1,331 OmniDocBench text regions labelled with the official
# OmniDocBench text normalization. Document-grouped 5-fold CV: ECE 0.121 for the raw
# token score, 0.044 calibrated, AUROC 0.93.
CONFIDENCE_WEIGHTS = (7.930, 1.787)
CONFIDENCE_BIAS = 1.197
CALIBRATED_CONFIDENCE = (
    "calibrated probability that the text is within 10% normalized edit distance "
    "of the ground truth"
)
UNCALIBRATED_CONFIDENCE = "decoder token likelihood, uncalibrated"
TEXT_LAYER = "pdf_text_layer"
TYPED_MARK = re.compile(r"^[Xx]\s+|\s+[Xx]$")
PROGRAMMATIC_TEXT = "exact text from the PDF text layer, not recognized"
# Markup truncated mid-tag ("</sup" with no ">") is handed back as text, not a tag.
TRAILING_PARTIAL_TAG = re.compile(r"<[^<>]*$")
MARKUP = re.compile(r"<[^<>]*>")
# Layout categories that carry image pixels rather than text.
EMPTY_CATEGORIES = frozenset(
    {
        "image",
        "picture",
        "figure",
        "chart",
        "seal",
        "form",
        "key-value-region",
        "signature",
        "hand-mark",
    }
)
# Layout categories mapped onto the kinds the rest of the pipeline already understands.
KIND_BY_CATEGORY = {
    "page_text": "page_text",
    "table": "table",
    "formula": "formula",
    "doc_title": "title",
    "title": "title",
    "paragraph_title": "section-header",
    "section-header": "section-header",
    "page-header": "page-header",
    "page-footer": "page-footer",
    "header": "page-header",
    "footer": "page-footer",
    "footnote": "footnote",
    "vision_footnote": "footnote",
    "caption": "caption",
    "figure_title": "caption",
    "list-item": "list-item",
    "algorithm": "code",
    "code": "code",
}


class FalconLayoutReader:
    """Page reader backed by the Falcon-Perception `ocr_layout` engine."""

    name = "falcon-perception-layout"

    def __init__(
        self,
        url: str,
        *,
        timeout_seconds: float = 600,
        word_detector: Callable[[Path], list[list[float]]] | None = None,
    ) -> None:
        parsed = urlsplit(url)
        if parsed.scheme != "http" or parsed.hostname not in {
            "127.0.0.1",
            "localhost",
            "::1",
        }:
            raise ValueError("Falcon layout service must be reached on loopback")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.url = url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.word_detector = word_detector
        self._model: dict[str, Any] = {}

    def check_health(self) -> None:
        body = self._request(f"{self.url}/health", None)
        self._model = dict(body.get("model") or {})

    @property
    def provenance(self) -> dict[str, Any]:
        return dict(self._model)

    @property
    def generation(self) -> dict[str, Any]:
        return {"engine": "falcon-perception", "termination_observable": True}

    def transcribe_crops(self, images, categories) -> list[str]:
        body = self._generate(images, categories)
        texts = body.get("texts")
        if (
            not isinstance(texts, list)
            or len(texts) != len(images)
            or any(not isinstance(text, str) or not text.strip() for text in texts)
        ):
            raise ReaderError("falcon_crop_failed", "malformed crop payload")
        return texts

    def read_crops(self, images, category: str) -> list[dict[str, Any]]:
        """Each crop's {"text", "generation"} with Falcon's own token scores, read as one
        batch; a crop that did not finish cleanly comes back with empty text. A batch can
        read a crop differently than a request of its own (see panel_split)."""
        body = self._generate(images, [category] * len(images), return_generation=True)
        results = body.get("results")
        if not isinstance(results, list) or len(results) != len(images):
            raise ReaderError("falcon_crop_failed", "malformed crop payload")
        return [
            {"text": item["text"], "generation": item["generation"]}
            if item.get("status") == "success"
            else {"text": "", "generation": {}}
            for item in sorted(results, key=lambda item: item["index"])
        ]

    def _generate(self, images, categories, **options: bool) -> dict[str, Any]:
        from .falcon import FALCON_OCR_CATEGORIES

        if not images or len(images) != len(categories) or len(images) > 24:
            raise ValueError("Expected 1 to 24 crops with matching categories")
        if any(category not in FALCON_OCR_CATEGORIES for category in categories):
            raise ValueError("Unsupported crop category")
        encoded = []
        for image in images:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            encoded.append(base64.b64encode(buffer.getvalue()).decode("ascii"))
        return self._request(
            f"{self.url}/generate",
            {"images": encoded, "categories": list(categories), **options},
        )

    def read(
        self,
        image_path: Path,
        page_number: int,
        words: list[list[Any]] | None = None,
    ) -> list[TextRegion]:
        """`words` are this image's detected words when the caller already has them."""
        cells = read_cells(image_path)
        # A born-digital page has text cells; a scan sends detected words instead,
        # so text no layout region holds is still read (Docling's orphan clusters).
        if cells or not self.word_detector:
            words = []
        elif words is None:
            words = self.word_detector(image_path)
        payload = {
            "image": base64.b64encode(image_path.read_bytes()).decode("ascii"),
            **(
                {"cells": cells}
                if cells
                else {"words": [word[:5] for word in words]}
                if self.word_detector
                else {}
            ),
        }
        body = self._request(f"{self.url}/read", payload)
        elements = body.get("elements")
        if not isinstance(elements, list):
            raise ReaderError("falcon_layout_failed", "malformed element payload")
        model = dict(body.get("model") or self._model)

        regions = []
        # One detected box can hold stacked tables; each is its own table grid.
        elements = [part for element in elements for part in _table_parts(element)]
        with ExitStack() as images:
            source = None
            for index, element in enumerate(elements, start=1):
                category = str(element.get("category") or "text")
                text = str(element.get("text") or "").strip()
                box = _bounding_box(element.get("bbox"))
                if box is None:
                    continue
                detection = {
                    key: element[key]
                    for key in ("query_id", "class_scores", "orphan_text")
                    if key in element
                }
                terminated = _read_terminated(element)
                attempts = element.get("read_attempts") or []
                read_history = {
                    **({"read_attempts": attempts} if attempts else {}),
                    **({"read_terminated": terminated} if terminated else {}),
                    **(
                        {"read_retries": int(element["retries"])}
                        if element.get("retries") is not None
                        else {}
                    ),
                }
                # A clean retry is the read; its calibrated confidence carries the doubt.
                read_resolution = "unreadable" if terminated else "resolved"
                if category in {"checkbox-selected", "checkbox-unselected"}:
                    selected = category == "checkbox-selected"
                    raw_text = text
                    text = _checkbox_label(text)
                    if raw_text and not text:
                        read_resolution = "unreadable"
                    glyph = DECODED_BALLOT_MARK.match(raw_text)
                    glyph_selected = (
                        glyph and glyph.group().casefold() in SELECTED_BALLOTS
                    )
                    programmatic = element.get("text_source") == TEXT_LAYER
                    typed = programmatic and not glyph and TYPED_MARK.search(text)
                    if typed:
                        # A form filled in software marks its box with a typed "X".
                        text = TYPED_MARK.sub("", text).strip()
                        selected = True
                    if glyph and programmatic:
                        selected = bool(
                            glyph_selected
                        )  # the page's own glyph is the state
                    elif glyph and glyph_selected != selected:
                        read_resolution = "conflicting"
                    regions.append(
                        TextRegion(
                            id=f"p{page_number}-falcon-{index}",
                            kind="checkbox",
                            text=text,
                            confidence=(element.get("generation") or {}).get(
                                "token_score"
                            ),
                            bounding_box=box,
                            reading_order=index,
                            provider=PROVIDER,
                            text_provenance={
                                "method": TEXT_LAYER
                                if programmatic
                                else "learned_checkbox_detection",
                                "reading_order_method": "detector_sequence",
                                "model": model,
                                "layout_detection": {
                                    **detection,
                                    "category": category,
                                    "score": float(element["score"]),
                                },
                                "generation": element.get("generation") or {},
                                "raw_text": raw_text,
                                **read_history,
                                "confidence_meaning": PROGRAMMATIC_TEXT
                                if programmatic
                                else UNCALIBRATED_CONFIDENCE,
                            },
                            resolution=read_resolution,
                            structure={
                                "state": "selected" if selected else "unselected",
                                **(
                                    {
                                        "decoded_state": "selected"
                                        if glyph_selected
                                        else "unselected"
                                    }
                                    if glyph
                                    else {}
                                ),
                                "state_confidence": None
                                if programmatic and (glyph or typed)
                                else float(element["score"]),
                                "label_evidence_ids": [],
                                "label": text,
                            },
                        )
                    )
                    if not programmatic:
                        regions[-1:] = _split_merged_controls(regions[-1])
                    continue
                # A picture the reader transcribed (handwriting Heron calls `picture`) is text.
                if category in EMPTY_CATEGORIES and not (
                    category == "picture" and text
                ):
                    if source is None:
                        source = images.enter_context(Image.open(image_path))
                    box = BoundingBox(
                        max(0, box.left),
                        max(0, box.top),
                        min(source.width, box.right),
                        min(source.height, box.bottom),
                    )
                    if box.right <= box.left or box.bottom <= box.top:
                        continue
                    with source.crop(
                        (box.left, box.top, box.right, box.bottom)
                    ) as crop:
                        buffer = io.BytesIO()
                        crop.save(buffer, format="PNG")
                        asset = {
                            "media_type": "image/png",
                            "width": crop.width,
                            "height": crop.height,
                            "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
                        }
                    regions.append(
                        TextRegion(
                            id=f"p{page_number}-falcon-{index}",
                            kind="figure",
                            text="",
                            confidence=None,
                            bounding_box=box,
                            reading_order=index,
                            provider=PROVIDER,
                            text_provenance={
                                "method": "source_image_crop",
                                "reading_order_method": "detector_sequence",
                                "layout_detection": detection,
                                "layout_category": category,
                                "layout_detection_score": element.get("score"),
                                "model": model,
                            },
                            resolution="unreadable"
                            if category in {"form", "key-value-region"}
                            else read_resolution,
                            structure={
                                "image": asset,
                                **(
                                    {"role": "form_source"}
                                    if category in {"form", "key-value-region"}
                                    else {}
                                ),
                                # Located, never transcribed, as document APIs report it.
                                **(
                                    {"role": "signature"}
                                    if category == "signature"
                                    else {}
                                ),
                            },
                        )
                    )
                    continue
                text, kind = _plain_text(text, KIND_BY_CATEGORY.get(category, "text"))
                decoded_control = _decoded_ballot_control(
                    text, kind, element.get("class_scores")
                )
                if decoded_control is not None:
                    text, state, state_glyph = decoded_control
                    kind = "checkbox"
                structure = (
                    {
                        "state": state,
                        "label": text,
                        "label_evidence_ids": [],
                    }
                    if decoded_control is not None
                    else None
                )
                structure_error = None
                if terminated:
                    # Evidence, not content: the detector's kind, not the one its markup claims.
                    kind = KIND_BY_CATEGORY.get(category, "text")
                if kind == "table":
                    try:
                        structure = _table_structure(text, element.get("generation"))
                    except ValueError as error:
                        structure_error = str(error)
                if kind == "table" and structure:
                    text = _table_text(structure)
                programmatic = element.get("text_source") == TEXT_LAYER
                recognition_method = (
                    TEXT_LAYER
                    if programmatic
                    else "falcon_perception_plain_ocr"
                    if category == "page_text"
                    else "falcon_perception_layout_ocr"
                )
                token_score = (element.get("generation") or {}).get("token_score")
                # Fitted on greedy first reads; a sampled retry is outside that calibration.
                # The calibration was fitted on Heron detection scores; an orphan line's
                # score is docTR's, so it stays an uncalibrated decoder score.
                calibrated = (
                    token_score is not None
                    and element.get("score") is not None
                    and kind not in {"table", "checkbox"}
                    and not attempts
                    and not element.get("orphan_text")
                )
                region = TextRegion(
                    id=f"p{page_number}-falcon-{index}",
                    kind=kind,
                    text=text,
                    confidence=calibrated_confidence(token_score, element["score"])
                    if calibrated
                    else token_score,
                    bounding_box=box,
                    reading_order=index,
                    provider=PROVIDER,
                    text_provenance={
                        "method": "decoded_ballot_glyph_promotion"
                        if decoded_control is not None
                        else recognition_method,
                        **(
                            {
                                "recognition_method": recognition_method,
                                "state_source": "decoded_ballot_glyph",
                                "state_glyph": state_glyph,
                            }
                            if decoded_control is not None
                            else {}
                        ),
                        "reading_order_method": "page_decoder"
                        if category == "page_text"
                        else "detector_sequence",
                        "layout_detection": detection,
                        "generation": element.get("generation") or {},
                        "raw_text": element.get("text") or "",
                        **read_history,
                        **(
                            {"table_structure_error": structure_error}
                            if structure_error
                            else {}
                        ),
                        "layout_category": category,
                        **(
                            {"table_structure_model": element["table_source"]}
                            if "table_source" in element
                            else {}
                        ),
                        # Named so nobody reads this as a recognition confidence.
                        "layout_detection_score": element.get("score"),
                        "confidence_meaning": PROGRAMMATIC_TEXT
                        if programmatic
                        else CALIBRATED_CONFIDENCE
                        if calibrated
                        else UNCALIBRATED_CONFIDENCE
                        if token_score is not None
                        else "recognition unavailable",
                        "model": model,
                    },
                    resolution="unreadable" if structure_error else read_resolution,
                    structure=structure or {},
                )
                regions.append(region)
        if not regions:
            # A page with no readable region is a failed read, not an empty document.
            # Reporting it as success renders a handful of geometric controls and calls
            # the result "Complete", which is how a warming service looked like a page
            # containing eleven characters.
            raise ReaderError(
                "falcon_layout_empty",
                f"the layout reader returned no readable region for page {page_number}",
            )
        _assign_image_owners(regions)
        _flag_foreign_scripts(regions)
        _attach_second_reader(regions, words)
        return regions

    def _request(self, url: str, payload: dict[str, Any] | None) -> dict[str, Any]:
        data = (
            None
            if payload is None
            else json.dumps(payload, separators=(",", ":")).encode("utf-8")
        )
        request = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as reply:
                body = reply.read(FALCON_SERVICE_MAX_RESPONSE_BYTES + 1)
            if len(body) > FALCON_SERVICE_MAX_RESPONSE_BYTES:
                raise ValueError("Falcon response exceeded its size limit")
            result = json.loads(body)
            if not isinstance(result, dict):
                raise ValueError("Falcon response must be an object")
            return result
        except (urllib.error.URLError, TimeoutError, ValueError, OSError) as error:
            raise ReaderError("falcon_layout_failed", str(error)) from error


def calibrated_confidence(token_score: float, layout_score: float) -> float:
    logit = (
        CONFIDENCE_WEIGHTS[0] * math.log(max(token_score, 1e-12))
        + CONFIDENCE_WEIGHTS[1] * layout_score
        + CONFIDENCE_BIAS
    )
    return 1 / (1 + math.exp(-logit))


def _attach_second_reader(regions: list[TextRegion], words: list[list[Any]]) -> None:
    """Record, per region, how docTR's reading of the words it holds agrees with Falcon's.

    The lowest recognition confidence marks handwriting docTR cannot read. A word docTR
    reads surely that Falcon's text lacks marks an omission: Falcon returned "Page __ of
    __" where docTR read both handwritten 4s at 0.91 and 1.0.
    """
    for region in regions:
        if region.kind in {"table", "figure"} or region.text_provenance is None:
            continue
        box = region.bounding_box
        held = [
            word
            for word in words
            if len(word) > 6
            and box.left <= (word[0] + word[2]) / 2 <= box.right
            and box.top <= (word[1] + word[3]) / 2 <= box.bottom
        ]
        if held:
            region.text_provenance["second_reader"] = {
                "model": "docTR crnn_vgg16_bn",
                "min_word_confidence": min(word[5] for word in held),
                "words": len(held),
                "unread_words": [
                    word[6]
                    for word in held
                    if word[5] >= SURE_WORD and _unread(word[6], region.text)
                ],
            }


# docTR's recognition confidence for a word it read surely.
SURE_WORD = 0.9
# Below this similarity to every Falcon word, docTR's word is not a misspelling of one.
SAME_WORD = 0.75


def _unread(word: str, text: str) -> bool:
    """Whether Falcon's text lacks a word docTR read.

    Digits are exact (a dropped digit changes an ID), so the word's digits must appear in
    the text's. Letters tolerate docTR's own misreads of noisy print ("Missquri"): a word
    of four or more letters counts only when no Falcon word is close to it.
    """
    digits = "".join(character for character in word if character.isdigit())
    if digits:
        return digits not in "".join(character for character in text if character.isdigit())
    letters = "".join(character for character in word.casefold() if character.isalpha())
    if len(letters) < 4 or letters in "".join(filter(str.isalpha, text.casefold())):
        return False  # too short to judge, or Falcon split docTR's word ("for the")
    return all(
        SequenceMatcher(None, letters, other).ratio() < SAME_WORD
        for other in re.findall(r"[^\W\d_]+", text.casefold())
    )


def _flag_foreign_scripts(regions: list[TextRegion]) -> None:
    """A script other than the page's own is a misread, not a language, until reread.

    Small crops of ticks and boxes decode as CJK characters (a box reads as "电"), a
    handwritten "2" as Georgian "ა", and a whole handwritten roster once decoded as
    Georgian words, too many letters for a share threshold. Flagged regions are reread,
    never dropped, so a genuinely bilingual page loses nothing.
    """
    counts = [
        Counter() if region.kind == "formula" else _script_letters(region.text)
        for region in regions
    ]
    page = sum(counts, Counter())
    if len(page) < 2:
        return
    main = page.most_common(1)[0][0]
    minor = set(page) - {main}
    for region, letters in zip(regions, counts):
        # PDF characters were extracted, not generated by the crop decoder.
        if (region.text_provenance or {}).get(
            "confidence_meaning"
        ) == PROGRAMMATIC_TEXT:
            continue
        foreign = set(letters) & minor
        if not foreign or region.resolution == "unreadable":
            continue
        region.resolution = "conflicting"
        region.text_provenance["foreign_script"] = sorted(foreign)
        # A table renders cell by cell, so each misread cell carries the doubt itself:
        # a roster once showed "კვეთა" in four cells of a table marked conflicting.
        for cell in (region.structure or {}).get("cells", []):
            if set(_script_letters(str(cell.get("text") or ""))) & minor:
                cell["resolution"] = "conflicting"



def _script_letters(text: str) -> Counter:
    # NFKC folds symbol letters (µ, º) into their script. Greek letters are ordinary
    # symbols in Latin scientific text.
    letters = Counter(
        unicodedata.name(char, "?").split()[0]
        for char in unicodedata.normalize("NFKC", text)
        if char.isalpha()
    )
    letters.pop("GREEK", None)
    return letters


def _assign_image_owners(regions: list[TextRegion]) -> None:
    # These assets are crops of the same source page: exact containment preserves
    # every child pixel without merging boxes or suppressing text detections.
    images = [
        region
        for region in regions
        if region.kind == "figure"
        and not region.text
        and (region.text_provenance or {}).get("method") == "source_image_crop"
        and (region.structure or {}).get("image")
    ]
    images.sort(
        key=lambda region: (
            -(
                (region.bounding_box.right - region.bounding_box.left)
                * (region.bounding_box.bottom - region.bounding_box.top)
            )
        )
    )
    parents: list[TextRegion] = []
    for region in images:
        box = region.bounding_box
        parent = next(
            (
                item
                for item in parents
                if item.bounding_box.left <= box.left
                and item.bounding_box.top <= box.top
                and item.bounding_box.right >= box.right
                and item.bounding_box.bottom >= box.bottom
            ),
            None,
        )
        if parent is None:
            parents.append(region)
            continue
        role = (region.structure or {}).get("role")
        region.structure.update(
            # Hidden form containers do not replace their children's visible pixels.
            role=role
            if role in {"form_source", "signature"}
            or (parent.structure or {}).get("role") == "form_source"
            or parent.resolution != "resolved"
            else "figure_source",
            parent_region_id=parent.id,
        )
        children = parent.structure.setdefault("child_evidence_ids", [])
        if region.id not in children:
            children.append(region.id)
        if (
            region.structure.get("role") != "figure_source"
            and region.resolution == "resolved"
        ):
            parents.insert(0, region)


TABLE_START = re.compile(r"<table\b", re.IGNORECASE)


class _TableParser(HTMLParser):
    """Collect the model's HTML table as rows of (text, rowspan, colspan, header)."""

    def __init__(self, markup: str = "") -> None:
        super().__init__(convert_charrefs=True)
        self.line_offsets = [
            0,
            *(index + 1 for index, char in enumerate(markup) if char == "\n"),
        ]
        self.ranges: list[tuple[int, int]] = []
        self._cell_start = 0
        self.rows: list[list[tuple[str, int, int, bool]]] = []
        self._row: list[tuple[str, int, int, bool]] | None = None
        self._cell: list[str] | None = None
        self._span = (1, 1)
        self._header = False
        self._in_head = False
        self._table_seen = False
        self._outside: list[str] = []
        self.leading_text = ""
        self.trailing_text = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            if self._table_seen:
                raise ValueError("Multiple or nested tables cannot share one cell grid")
            self._table_seen = True
        elif tag == "thead":
            self._in_head = True
        elif tag in {"tbody", "tfoot"}:
            self._in_head = False
        elif tag == "tr":
            # A row still open when the next one starts was never closed. The model's
            # table markup is often truncated mid-row, so flushing here rather than
            # resetting is what keeps that row's cells.
            self._close_row()
            self._row = []
        elif tag in {"td", "th"}:
            self._close_cell()
            outside = "".join(self._outside).strip()
            if outside and self.ranges:
                raise ValueError(
                    "Text interleaved with table cells cannot share one grid"
                )
            if outside:
                self.leading_text = outside
            self._outside = []
            values = dict(attrs)
            self._cell = []
            self._cell_start = self._offset() + len(self.get_starttag_text())
            self._span = (_span(values.get("rowspan")), _span(values.get("colspan")))
            self._header = tag == "th" or self._in_head
        elif tag == "br" and self._cell is not None:
            self._cell.append("\n")
        if self._cell is None and (tag == "br" or tag in TABLE_TEXT_BLOCK_TAGS):
            self._outside.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"thead", "table"}:
            self._in_head = False
        elif tag == "tr":
            self._close_row()
        elif tag in {"td", "th"}:
            self._close_cell()
        elif tag == "table":
            self._close_row()
        if self._cell is None and tag in TABLE_TEXT_BLOCK_TAGS:
            self._outside.append("\n")

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)
        else:
            self._outside.append(data)

    def close(self) -> None:
        self.rawdata = TRAILING_PARTIAL_TAG.sub("", self.rawdata)
        super().close()
        # Truncated markup closes nothing: no </td>, no </tr>, no </table>.
        self._close_row()
        self.trailing_text = "".join(self._outside).strip()

    def _close_cell(self) -> None:
        if self._cell is None:
            return
        if self._row is None:
            self._row = []
        text = "".join(self._cell).strip()
        if len(self.ranges) >= MAX_TABLE_GRID_CELLS:
            raise ValueError("Table exceeds the 10000-cell resource limit")
        self._row.append((text, *self._span, self._header))
        self.ranges.append((self._cell_start, self._offset()))
        self._cell = None

    def _offset(self) -> int:
        line, column = self.getpos()
        return self.line_offsets[line - 1] + column

    def _close_row(self) -> None:
        self._close_cell()
        if self._row:
            self.rows.append(self._row)
        self._row = None


def _table_rows(text: str) -> list[list[str]] | None:
    """Cell texts of one HTML table, or None when the markup is not one grid."""
    parser = _TableParser(text)
    try:
        parser.feed(text)
        parser.close()
    except ValueError:
        return None
    if parser.leading_text or parser.trailing_text or not parser.rows:
        return None
    return [[cell for cell, *_ in row] for row in parser.rows]


def _unmarked_text(text: str) -> str:
    """Flatten Falcon's HTML markup to the plain text `TextRegion.text` carries."""
    if "<table" in text.casefold():
        rows = _table_rows(text)
        text = (
            "\n".join(" ".join(cell for cell in row if cell) for row in rows)
            if rows is not None
            else " ".join(MARKUP.sub(" ", text).split())
        )
    if not INLINE_TAG.search(text) and not LINE_BREAK_TAG.search(text):
        return text
    return html.unescape(INLINE_TAG.sub("", LINE_BREAK_TAG.sub("\n", text)))


def _split_merged_controls(region: TextRegion) -> list[TextRegion]:
    """One control per decoded box when Heron merged neighbours into one detection.

    Falcon's reading of the crop shows every printed box ("□ R □ L □ Heberden's node").
    Each box starts a control whose label runs to the next box; its state is the
    decoded glyph, and a leading label whose box the decoder dropped keeps the
    detector's state. Boxes are apportioned along the line by character position, an
    approximation recorded in the provenance. On the handwritten forms this found 14
    more controls and raised the forms macro accuracy from 0.683 to 0.691.
    """
    raw = (region.text_provenance or {}).get("raw_text") or ""
    marks = [m for m in DECODED_BALLOT_MARK.finditer(raw) if m.group() in UNAMBIGUOUS_BALLOTS]
    if len(marks) < 2:
        return [region]
    spans = [(None, 0, marks[0].start())] if raw[: marks[0].start()].strip() else []
    spans += [
        (m.group(), m.start(), marks[i + 1].start() if i + 1 < len(marks) else len(raw))
        for i, m in enumerate(marks)
    ]
    box, parts = region.bounding_box, []
    width = box.right - box.left
    for glyph, start, end in spans:
        label = _checkbox_label(raw[start:end])
        if not label:
            continue
        state = (
            ("selected" if glyph.casefold() in SELECTED_BALLOTS else "unselected")
            if glyph
            else region.structure["state"]
        )
        parts.append(
            replace(
                region,
                id=f"{region.id}-{len(parts) + 1}",
                text=label,
                bounding_box=BoundingBox(
                    box.left + width * start / len(raw), box.top,
                    box.left + width * end / len(raw), box.bottom,
                ),
                text_provenance={
                    **region.text_provenance,
                    "split_from": region.id,
                    "geometry": "apportioned by character position",
                },
                structure={**region.structure, "state": state, "label": label},
            )
        )
    return parts or [region]


def _checkbox_label(text: str) -> str:
    # Falcon may mark the label up as a heading; the region's kind is still checkbox.
    text = _plain_text(text, "text")[0]
    text = re.sub(rf"^(?:{BALLOT_MARK})\s*", "", text)
    # A control crop starts at its glyph; Falcon reads a radio circle as the letter O
    # ("O Yes": 72 of 4,187 labels in the domain and handwriting runs).
    text = re.sub(r"^[Oo]\s+(?=\S)", "", text)
    return re.sub(rf"\s*(?:{BALLOT_MARK})$", "", text).strip()


def _decoded_ballot_control(
    text: str, kind: str, class_scores: dict[str, float] | None = None
) -> tuple[str, str, str] | None:
    if kind not in {"text", "list-item"}:
        return None
    marks = list(DECODED_BALLOT_MARK.finditer(text))
    prefix_end = re.match(r"(?:[-*+]\s+)?", text).end()
    if len(marks) != 1 or marks[0].start() != prefix_end:
        return None
    glyph = marks[0].group()
    label = text[marks[0].end() :].strip()
    if glyph == "[?]" or not label:
        return None
    if glyph.casefold() not in UNAMBIGUOUS_BALLOTS and not _detector_sees_control(
        class_scores
    ):
        return None
    state = "selected" if glyph.casefold() in SELECTED_BALLOTS else "unselected"
    return label, state, glyph


def _detector_sees_control(class_scores: dict[str, float] | None) -> bool:
    """A bullet-like glyph is a control only if Heron scores it more as one than a list."""
    if not class_scores:
        return False
    control = max(
        class_scores.get("checkbox_selected", 0.0),
        class_scores.get("checkbox_unselected", 0.0),
    )
    return control > max(
        class_scores.get("list_item", 0.0), class_scores.get("text", 0.0)
    )


def _table_structure(
    markup: str, generation: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    """Place the model's HTML cells on the grid the rest of the pipeline renders.

    Without this the table's HTML source is the region's text, so the plain-text lane
    prints `<table><thead><tr><th>` at a reviewer instead of a table.
    """
    parser = _TableParser(markup)
    parser.feed(markup)
    parser.close()
    if not parser.rows:
        return None

    occupied: dict[tuple[int, int], bool] = {}
    placed: list[tuple[list[int], list[int], str, bool]] = []
    column_count = 0
    row_count = 0
    for row_index, row in enumerate(parser.rows):
        column = 0
        for text, rowspan, colspan, header in row:
            while (row_index, column) in occupied:
                column += 1
            row_count = max(row_count, row_index + rowspan)
            column_count = max(column_count, column + colspan)
            if row_count * column_count > MAX_TABLE_GRID_CELLS:
                raise ValueError("Table exceeds the 10000-cell resource limit")
            row_nums = list(range(row_index, row_index + rowspan))
            column_nums = list(range(column, column + colspan))
            for position in ((r, c) for r in row_nums for c in column_nums):
                occupied[position] = header
            placed.append((row_nums, column_nums, text, header))
            column += colspan
    if not occupied:
        return None

    cells: list[dict[str, Any]] = []
    taken: set[tuple[int, int]] = set()
    # Empty cells are positions in the source table, not permission to shift values.
    for rows, columns, text, header in placed:
        cell = _cell(len(cells), rows, columns, text, header)
        start, end = parser.ranges[len(cells)]
        scores = [
            token["score"]
            for token in (generation or {}).get("tokens", [])
            if token["start"] < end and token["end"] > start
        ]
        if scores and text:
            cell["confidence"] = math.exp(
                sum(math.log(max(score, 1e-30)) for score in scores) / len(scores)
            )
            cell["confidence_kind"] = "Decoder token score"
            cell["source"] = PROVIDER
        cells.append(cell)
        taken.update((r, c) for r in rows for c in columns)
    # A browser draws a short row as empty trailing cells, and the topology check needs
    # the grid to be a complete rectangle, so the same cells are made explicit here,
    # marked as filler: the model wrote no cell there, empty or not.
    for position in (
        (r, c)
        for r in range(row_count)
        for c in range(column_count)
        if (r, c) not in taken
    ):
        filler = _cell(len(cells), [position[0]], [position[1]], "", False)
        cells.append({**filler, "grid_fill": True})
    # Counted from the parsed rows, not the padded grid: a short header row gets filler
    # cells, and those are not headers.
    header_rows = 0
    for row in parser.rows:
        if not row or not all(cell[3] for cell in row):
            break
        header_rows += 1
    return {
        "role": "table",
        "header_row_count": header_rows,
        "row_count": row_count,
        "column_count": column_count,
        "cells": cells,
        **({"leading_text": parser.leading_text} if parser.leading_text else {}),
        **({"trailing_text": parser.trailing_text} if parser.trailing_text else {}),
    }


def _cell(
    index: int, row_nums: list[int], column_nums: list[int], text: str, header: bool
) -> dict[str, Any]:
    return {
        "id": f"falcon-cell-{index + 1}",
        "row_nums": row_nums,
        "column_nums": column_nums,
        "text": text,
        "resolution": "resolved",
        "column_header": header,
    }


def _table_parts(element: dict[str, Any]) -> list[dict[str, Any]]:
    """Split a table read holding several <table> elements into one read per table.

    The parts share the detected box (where one table ends is unknown) and each
    keeps its own slice of the decoder's token offsets.
    """
    text = str(element.get("text") or "")
    starts = [match.start() for match in TABLE_START.finditer(text)]
    # Text before the first table (a caption) stays with the first table's read.
    bounds = list(zip([0, *starts[1:]], [*starts[1:], len(text)]))
    if (
        element.get("category") != "table"
        or len(starts) < 2
        # A nested table opens before its parent closes; only stacked tables split.
        or any("</table>" not in text[start:end].lower() for start, end in bounds[:-1])
    ):
        return [element]
    generation = element.get("generation") or {}
    parts = []
    for start, end in bounds:
        start += len(text[start:end]) - len(text[start:end].lstrip())
        end = start + len(text[start:end].rstrip())
        tokens = [
            {
                **token,
                "start": max(token["start"], start) - start,
                "end": min(token["end"], end) - start,
            }
            for token in generation.get("tokens", [])
            if token["start"] < end and token["end"] > start
        ]
        parts.append(
            {
                **element,
                "text": text[start:end],
                "generation": {**generation, "tokens": tokens},
            }
        )
    return parts


def _table_text(structure: dict[str, Any]) -> str:
    grid = [
        ["" for _ in range(structure["column_count"])]
        for _ in range(structure["row_count"])
    ]
    for cell in structure["cells"]:
        grid[cell["row_nums"][0]][cell["column_nums"][0]] = cell["text"].replace(
            "\n", " "
        )
    table = "\n".join("\t".join(row).rstrip() for row in grid).strip()
    return "\n".join(
        text
        for text in (
            structure.get("leading_text"),
            table,
            structure.get("trailing_text"),
        )
        if text
    )


def _span(value: str | None) -> int:
    try:
        span = max(1, int(str(value)))
    except (TypeError, ValueError):
        if value is not None and value.strip().lstrip("+").isdecimal():
            raise ValueError("Table span exceeds the 100-span resource limit") from None
        return 1
    if span > MAX_TABLE_SPAN:
        raise ValueError("Table span exceeds the 100-span resource limit")
    return span


def _plain_text(text: str, kind: str) -> tuple[str, str]:
    """Strip the model's markdown markers, promoting a marked heading to its kind.

    Falcon writes markdown even though the official prompts only ask for content, so a
    title arrives as "# Title". `TextRegion.text` is plain text and `kind` carries the
    structure, so leaving the marker in place double-encodes it: the markdown lane adds
    its own prefix ("### # Title") and the text lane shows the bare marker to a reviewer.
    Table and formula markup is the region's content, so it passes through untouched.
    An HTML table in any other region becomes a table only when its content spans two
    rows and two columns without repeating whole rows; Falcon wraps a ruled handwritten
    line in an empty grid ("FORENSIC TESTING SERVICES" in one cell of a 2 x 2), and other
    markup flattens to plain text.
    """
    if re.fullmatch(r"(`{3,}|~{3,})[^\n]*\n[\s\S]*\n\1\s*", text):
        return text, "code"
    if kind in {"table", "formula", "code"}:
        return text, kind
    rows = _table_rows(text) if "<table" in text.casefold() else None
    filled = [(r, c) for r, row in enumerate(rows or []) for c, cell in enumerate(row) if cell.strip()]
    # Repeated whole rows are a decoder loop, not a table: an address line once came
    # back as four rows of "კვეთა | 2006". Blank rows of a ruled form do not count.
    lines = [tuple(cell.strip() for cell in row) for row in rows or [] if any(cell.strip() for cell in row)]
    looping = 2 * (len(lines) - len(set(lines))) >= len(lines) > 2
    if len({r for r, _ in filled}) > 1 and len({c for _, c in filled}) > 1 and not looping:
        return text, "table"
    text = _unmarked_text(text)
    level = 0
    lines = []
    for line in text.splitlines():
        heading = HEADING.match(line)
        if heading:
            level = level or len(heading.group(1))
            line = heading.group(2)
        lines.append(line.rstrip())
    plain = _boxed_date(EMPHASIS.sub(r"\1", "\n".join(lines)).strip())
    if level and kind == "text":
        kind = "title" if level == 1 else "section-header"
    return plain, kind


# A date written one part per printed box: "2 | 16 | 09" or "2 16 09".
BOXED_DATE = re.compile(r"(\d{1,2})[\s|]+(\d{1,2})[\s|]+(\d{2}|\d{4})")


def _boxed_date(text: str) -> str:
    """Join a whole-region month/day/year written in separate boxes as a US date."""
    match = BOXED_DATE.fullmatch(text)
    if not match or not (1 <= int(match[1]) <= 12 and 1 <= int(match[2]) <= 31):
        return text
    return "/".join(match.groups())


def _read_terminated(element: dict[str, Any]) -> str | None:
    """Why a read's text is evidence rather than a claim, if it is at all."""
    if element.get("truncated"):
        return "spent its whole token budget"
    if element.get("looped"):
        return "repetition loop the retry ladder did not recover"
    if element.get("text_source") != TEXT_LAYER and _exceeds_capacity(element):
        return "more characters than its pixels can hold"
    if not str(element.get("text") or "").strip() and element.get(
        "category", "text"
    ) not in EMPTY_CATEGORIES | {"checkbox-selected", "checkbox-unselected"}:
        return "returned no text"
    return None


# A legible glyph needs at least about 3x4 source pixels. Across 3568 Falcon reads the
# densest genuine text held one character per 38 px2; an 18x30 px margin mark read as a
# 1825-token table held one per 1.4 px2, with calibrated confidence 0.99.
MIN_GLYPH_AREA = 12


def _exceeds_capacity(element: dict[str, Any]) -> bool:
    box = element.get("bbox")
    if not isinstance(box, list) or len(box) != 4:
        return False
    area = max(0.0, float(box[2]) - float(box[0])) * max(
        0.0, float(box[3]) - float(box[1])
    )
    characters = len("".join(MARKUP.sub(" ", str(element.get("text") or "")).split()))
    return characters * MIN_GLYPH_AREA > area


def _bounding_box(value: Any) -> BoundingBox | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    values = [float(part) for part in value]
    if not all(math.isfinite(part) for part in values):
        return None
    if values[2] <= values[0] or values[3] <= values[1]:
        return None
    left, top, right, bottom = (int(round(part)) for part in values)
    if right <= left or bottom <= top:
        return None
    return BoundingBox(left, top, right, bottom)
