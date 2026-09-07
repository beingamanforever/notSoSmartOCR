"""LLM formatting pass over the final rendered Markdown.

This is a cosmetic reformatting step only. It never touches the plain-text
evidence lane, which carries the evidence links the review UI depends on.
A digit guard rejects any output that invents a number the model wasn't given.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .openrouter import OPENROUTER_URL

POLISH_MODEL = "qwen/qwen3.7-flash"
TIMEOUT_SECONDS = 120.0
MIN_LENGTH_RATIO = 0.5
MAX_LENGTH_RATIO = 1.3

SYSTEM_PROMPT = (
    "You are a document Markdown formatter. Reformat only: preserve every "
    "piece of content verbatim. Fix table column alignment and merge table "
    "rows that OCR split across lines. Keep GitHub-flavored Markdown tables. "
    "Keep checkbox and ring glyphs exactly as given. Never invent, correct, "
    "translate, or drop any value. Never add commentary, headers, or code "
    "fences around the document. Output only the reformatted document "
    "Markdown and nothing else."
)

CallFn = Callable[[str, str, str], str]


def polish_markdown(
    markdown: str,
    *,
    model: str = POLISH_MODEL,
    call: CallFn | None = None,
) -> tuple[str, dict]:
    """Reformat markdown with an LLM, falling back to the original on any failure."""
    call = call or _default_call
    try:
        polished = call(model, SYSTEM_PROMPT, markdown)
    except Exception as error:
        return markdown, {"polished": False, "reason": f"API error: {error}"}

    if not polished or not polished.strip():
        return markdown, {"polished": False, "reason": "empty response"}

    reason = _guard_failure(markdown, polished)
    if reason:
        return markdown, {"polished": False, "reason": reason}

    return polished, {"polished": True, "model": model}


def _guard_failure(original: str, polished: str) -> str | None:
    original_digits = Counter(re.findall(r"\d+", original))
    polished_digits = Counter(re.findall(r"\d+", polished))
    invented = polished_digits - original_digits
    if invented:
        return f"invented digits not present in input: {sorted(invented)}"

    length_ratio = len(polished) / len(original) if original else 1.0
    if not MIN_LENGTH_RATIO <= length_ratio <= MAX_LENGTH_RATIO:
        return f"length ratio {length_ratio:.2f} outside allowed {MIN_LENGTH_RATIO}-{MAX_LENGTH_RATIO} range"

    return None


def _default_call(model: str, system: str, user: str) -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
        "stream": False,
    }
    request = Request(
        OPENROUTER_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            body = json.loads(response.read())
    except (HTTPError, URLError, TimeoutError) as error:
        raise RuntimeError(f"OpenRouter call failed: {error}") from error
    return body["choices"][0]["message"]["content"]
