from __future__ import annotations

from ocr_pipeline.markdown_polish import POLISH_MODEL, polish_markdown

ORIGINAL = "# Report\n\n| A | B |\n|---|---|\n| 1 | Total: 500 |\n"


def test_polish_markdown_accepts_clean_reformat() -> None:
    cleaned = "# Report\n\n| A | B       |\n|---|---------|\n| 1 | Total: 500 |\n"

    def call(model: str, system: str, user: str) -> str:
        assert model == POLISH_MODEL
        assert "reformat" in system.lower()
        assert user == ORIGINAL
        return cleaned

    polished, provenance = polish_markdown(ORIGINAL, call=call)

    assert polished == cleaned
    assert provenance == {"polished": True, "model": POLISH_MODEL}


def test_polish_markdown_rejects_invented_number() -> None:
    def call(model: str, system: str, user: str) -> str:
        return ORIGINAL.replace("Total: 500", "Total: $9,999")

    polished, provenance = polish_markdown(ORIGINAL, call=call)

    assert polished == ORIGINAL
    assert provenance["polished"] is False
    assert "digit" in provenance["reason"]


def test_polish_markdown_falls_back_on_api_error() -> None:
    def call(model: str, system: str, user: str) -> str:
        raise RuntimeError("openrouter is down")

    polished, provenance = polish_markdown(ORIGINAL, call=call)

    assert polished == ORIGINAL
    assert provenance["polished"] is False
    assert "openrouter is down" in provenance["reason"]


def test_polish_markdown_rejects_length_blowup() -> None:
    def call(model: str, system: str, user: str) -> str:
        return ORIGINAL + ("padding line with no new digits\n" * 20)

    polished, provenance = polish_markdown(ORIGINAL, call=call)

    assert polished == ORIGINAL
    assert provenance["polished"] is False
    assert "length ratio" in provenance["reason"]


def test_polish_markdown_rejects_empty_response() -> None:
    def call(model: str, system: str, user: str) -> str:
        return "   "

    polished, provenance = polish_markdown(ORIGINAL, call=call)

    assert polished == ORIGINAL
    assert provenance == {"polished": False, "reason": "empty response"}


def test_polish_markdown_allows_dropped_numbers() -> None:
    def call(model: str, system: str, user: str) -> str:
        return ORIGINAL.replace("Total: 500", "Total")

    polished, provenance = polish_markdown(ORIGINAL, call=call)

    assert provenance["polished"] is True
    assert polished == ORIGINAL.replace("Total: 500", "Total")
