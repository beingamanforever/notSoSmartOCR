from __future__ import annotations

from pathlib import Path

from ocr_pipeline.contracts import BoundingBox, TextAlternative, TextRegion
from ocr_pipeline.correction import LexiconCorrectionStage

FORM_WORDS = {"Non-Smoker", "Smoker", "Allergies", "Gender", "Spirometry"}


def region(
    text: str, *, kind: str = "word", resolution: str = "resolved"
) -> TextRegion:
    return TextRegion(
        id="p1-word-1",
        kind=kind,
        text=text,
        confidence=0.6,
        bounding_box=BoundingBox(0, 0, 40, 12),
        reading_order=1,
        provider="controlled",
        resolution=resolution,
    )


def run(regions: list[TextRegion], lexicon: set[str] | None = None) -> list[TextRegion]:
    stage = LexiconCorrectionStage(lexicon if lexicon is not None else FORM_WORDS)
    return stage.apply(Path("unused.png"), 1, regions)


def test_a_one_edit_printed_word_gets_the_lexicon_spelling_as_an_alternative() -> None:
    source = region("Non-Smok@r")

    run([source])

    assert source.text == "Non-Smok@r", "the source transcription is never replaced"
    assert [item.text for item in source.alternatives] == ["Non-Smoker"]
    assert source.alternatives[0].confidence is None, (
        "reachability is not evidence about the ink, so it carries no score"
    )
    assert source.alternatives[0].text_provenance["method"] == (
        "bounded_edit_lexicon_match"
    )


def test_a_word_absent_from_the_lexicon_is_never_invented() -> None:
    """The Olmesartin case: no lexicon entry, so nothing is proposed."""
    source = region("Olmesartin")

    run([source])

    assert source.alternatives == []


def test_an_ambiguous_word_is_left_alone() -> None:
    source = region("Smoked")

    run([source], lexicon={"Smoker", "Smokes"})

    assert source.alternatives == [], (
        "two candidates means the evidence does not choose"
    )


def test_handwriting_is_never_corrected() -> None:
    source = region("Non-Smok@r", kind="handwriting")

    run([source])

    assert source.alternatives == [], "handwriting has no template to check against"


def test_unresolved_and_already_disputed_regions_are_skipped() -> None:
    unresolved = region("Non-Smok@r", resolution="unreadable")
    disputed = region("Non-Smok@r")
    disputed.alternatives.append(
        TextAlternative(text="Non-Smoker", confidence=0.4, provider="other")
    )

    run([unresolved, disputed])

    assert unresolved.alternatives == []
    assert len(disputed.alternatives) == 1, "an existing dispute is not piled onto"


def test_exact_matches_and_short_words_produce_nothing() -> None:
    exact = region("Allergies")
    short = region("Gen")

    run([exact, short])

    assert exact.alternatives == []
    assert short.alternatives == []
