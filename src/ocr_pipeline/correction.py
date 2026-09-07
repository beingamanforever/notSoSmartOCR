"""Bounded, non-generative correction of printed text.

Kanerva et al. (arXiv:2502.01205) measured LLM OCR post-correction and found relative CER
changes of -14.9% for Mixtral on English and -76.5% on Finnish: the corrector made the text
worse. A generative rewriter is therefore not safe here, because the failure it produces on
clinical text is a plausible drug name that was never on the page.

This stage cannot invent. A candidate is only proposed when it already exists in a supplied
lexicon, sits within a small edit distance of what was read, and is the only such candidate.
Handwriting is never touched: a printed label can be checked against the form's own
vocabulary, but a handwritten drug name has no template to check against, so a correction
there would be a guess wearing a confidence score.

Corrections are alternatives. The source transcription is never replaced.
"""

from __future__ import annotations

from difflib import get_close_matches
from pathlib import Path

from .contracts import TextAlternative, TextRegion

PROVIDER = "lexicon-correction"
METHOD = "bounded_edit_lexicon_match"
# difflib similarity floor. The guarantee that nothing is invented comes from the lexicon
# being closed, not from this number, so it only controls how near a miss has to be.
SIMILARITY = 0.85
# Below this length a single edit changes too much of the word to be evidence of anything.
MIN_LENGTH = 4


class LexiconCorrectionStage:
    """Propose a lexicon word when a printed reading is one edit away from exactly one."""

    name = "lexicon-correction"

    def __init__(self, lexicon: set[str], *, similarity: float = SIMILARITY) -> None:
        if not 0 < similarity <= 1:
            raise ValueError("similarity must be from 0 to 1")
        self.lexicon = sorted(word for word in lexicon if len(word) >= MIN_LENGTH)
        self.similarity = similarity

    def apply(
        self,
        image_path: Path,
        page_number: int,
        regions: list[TextRegion],
    ) -> list[TextRegion]:
        for region in regions:
            if not _correctable(region):
                continue
            candidate = self._single_candidate(region.text.strip())
            if candidate is None:
                continue
            region.alternatives.append(
                TextAlternative(
                    text=candidate,
                    confidence=None,
                    provider=PROVIDER,
                    text_provenance={
                        "method": METHOD,
                        "source_text": region.text,
                        "similarity": self.similarity,
                        # No score: this says the word is reachable and unique, not that
                        # the ink says it. Only review or another reader settles that.
                    },
                )
            )
        return regions

    def _single_candidate(self, text: str) -> str | None:
        if len(text) < MIN_LENGTH or text in self.lexicon:
            return None
        # Two results asked for so ambiguity is visible: if a second lexicon word is just
        # as close, the evidence does not choose and nothing is proposed.
        matches = get_close_matches(text, self.lexicon, n=2, cutoff=self.similarity)
        return matches[0] if len(matches) == 1 else None


def _correctable(region: TextRegion) -> bool:
    """Printed, resolved, single-word text only."""
    structure = region.structure or {}
    return (
        region.resolution == "resolved"
        and region.kind not in {"handwriting", "checkbox", "table", "layout_block"}
        and structure.get("role") != "handwriting_candidate"
        and bool(region.text.strip())
        and " " not in region.text.strip()
        and not region.alternatives
    )
