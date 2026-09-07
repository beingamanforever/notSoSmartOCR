"""Arctic-TILT field comprehension for fields our own geometry could not settle.

Geometric detection answers "is there ink in the slot beside this label". It cannot answer
"which option is selected", because that needs the label, the marks and the row read
together. Arctic-TILT is asked the field label as a question against the whole page, so a
tick that sits past a reach threshold, or beside a label with no colon, is still reachable.

Answers are proposals. They attach to the field as a separate object and never replace a
source transcription, because a plausible answer is not evidence of what the ink says.
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .contracts import TextRegion
from .providers import ReaderError

PROVIDER = "arctic-tilt"
MODEL = {
    "id": "Snowflake/snowflake-arctic-tilt-v1.3",
    "origin": "Snowflake Labs",
    "license": "Apache-2.0",
}
# Field states that geometry failed to settle and that are worth a second reader.
UNSETTLED_STATES = frozenset({"not_located", "illegible", "conflicting_readings"})


@dataclass(frozen=True)
class PageInput:
    """One page of a document as Arctic-TILT reads it: pixels plus positioned words."""

    image_path: Path
    words: list[str]
    boxes: list[tuple[int, int, int, int]]
    size: tuple[int, int]


@dataclass(frozen=True)
class FieldAnswer:
    """One field answer with the score the reference implementation reports."""

    field_id: str
    label: str
    values: tuple[str, ...]
    score: float | None


class TiltEngine(Protocol):
    """Answers field questions about a page. Implemented by vLLM on a CUDA host."""

    def answer(
        self,
        image_path: Path,
        words: list[str],
        boxes: list[tuple[int, int, int, int]],
        size: tuple[int, int],
        questions: list[str],
    ) -> list[FieldAnswer]: ...


class TiltFieldStage:
    """Ask a stronger reader about fields geometry left unsettled."""

    name = "tilt-fields"

    def __init__(self, engine: TiltEngine, *, max_questions: int = 24) -> None:
        if max_questions < 1:
            raise ValueError("max_questions must be positive")
        self.engine = engine
        self.max_questions = max_questions

    def apply(
        self,
        image_path: Path,
        page_number: int,
        regions: list[TextRegion],
    ) -> list[TextRegion]:
        unsettled = _unsettled_fields(regions)
        if not unsettled:
            return regions
        dropped = max(0, len(unsettled) - self.max_questions)
        asked = unsettled[: self.max_questions]

        words, boxes = _positioned_text(regions)
        if not words:
            return regions
        answers = {
            answer.label: answer
            for answer in self.engine.answer(
                image_path,
                words,
                boxes,
                _page_size(regions),
                [str(field["label"]) for _, field in asked],
            )
        }
        for _, field in asked:
            answer = answers.get(str(field["label"]))
            if answer is None:
                continue
            field["comprehension"] = {
                "provider": PROVIDER,
                "model": MODEL,
                "values": list(answer.values),
                "score": answer.score,
                "question": answer.label,
                # The field state is deliberately untouched: this is a proposal about what
                # the field says, not evidence that the ink was read.
                "decision_state": "pending",
            }
        if dropped:
            _record_budget(regions, len(asked), dropped)
        return regions


def _unsettled_fields(regions: list[TextRegion]) -> list[tuple[TextRegion, dict]]:
    unsettled = []
    for region in regions:
        structure = region.structure or {}
        if structure.get("role") != "layout_block":
            continue
        for field in structure.get("fields", []) or []:
            if not isinstance(field, dict):
                continue
            if field.get("state") in UNSETTLED_STATES and str(field.get("label", "")):
                unsettled.append((region, field))
    return unsettled


def _positioned_text(
    regions: list[TextRegion],
) -> tuple[list[str], list[tuple[int, int, int, int]]]:
    """Resolved words in reading order, with the pixel boxes TILT expects."""
    words: list[str] = []
    boxes: list[tuple[int, int, int, int]] = []
    for region in sorted(regions, key=lambda item: item.reading_order):
        if region.resolution != "resolved" or not region.text.strip():
            continue
        if (region.structure or {}).get("role") == "layout_block":
            continue
        box = region.bounding_box
        words.append(region.text)
        boxes.append((box.left, box.top, box.right, box.bottom))
    return words, boxes


def _page_size(regions: list[TextRegion]) -> tuple[int, int]:
    width = max((region.bounding_box.right for region in regions), default=1)
    height = max((region.bounding_box.bottom for region in regions), default=1)
    return width, height


def _record_budget(regions: list[TextRegion], asked: int, dropped: int) -> None:
    """Say out loud when the question budget truncated coverage."""
    for region in regions:
        structure = region.structure or {}
        if structure.get("role") != "layout_block":
            continue
        structure["comprehension_budget"] = {"asked": asked, "not_asked": dropped}
        region.structure = structure
        return


class HttpTiltEngine:
    """Arctic-TILT reached over loopback HTTP.

    The model runs behind its own vLLM fork, which pins an older torch than the reader
    stack, so it stays in a separate process rather than a separate import.
    """

    def __init__(self, url: str, *, timeout_seconds: float = 180) -> None:
        if not url.startswith(("http://127.0.0.1", "http://localhost")):
            raise ValueError("Arctic-TILT service must be reached on loopback")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.url = url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def check_health(self) -> None:
        self._request(f"{self.url}/health", None)

    def answer(
        self,
        image_path: Path,
        words: list[str],
        boxes: list[tuple[int, int, int, int]],
        size: tuple[int, int],
        questions: list[str],
    ) -> list[FieldAnswer]:
        return self.answer_pages([PageInput(image_path, words, boxes, size)], questions)

    def answer_pages(
        self, pages: list[PageInput], questions: list[str]
    ) -> list[FieldAnswer]:
        payload = {
            "pages": [
                {
                    "image": base64.b64encode(page.image_path.read_bytes()).decode(
                        "ascii"
                    ),
                    "words": page.words,
                    "boxes": [list(box) for box in page.boxes],
                    "size": list(page.size),
                }
                for page in pages
            ],
            "questions": questions,
        }
        body = self._request(f"{self.url}/answer", payload)
        answers = body.get("answers")
        if not isinstance(answers, list):
            raise ReaderError("tilt_service_failed", "malformed answer payload")
        return [
            FieldAnswer(
                field_id=str(item.get("field_id", "")),
                label=str(item.get("label", "")),
                values=tuple(str(value) for value in item.get("values") or ()),
                score=item.get("score"),
            )
            for item in answers
            if isinstance(item, dict)
        ]

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
                return json.loads(reply.read())
        except (urllib.error.URLError, TimeoutError, ValueError, OSError) as error:
            raise ReaderError("tilt_service_failed", str(error)) from error


class VllmTiltEngine:
    """Arctic-TILT served by its own vLLM fork. Requires CUDA, so it never imports here."""

    def __init__(
        self,
        model: str = MODEL["id"],
        *,
        gpu_memory_utilization: float = 0.8,
        max_num_seqs: int = 16,
        max_model_len: int = 32768,
    ) -> None:
        if max_model_len <= 0:
            raise ValueError("max_model_len must be positive")
        self.model = model
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_num_seqs = max_num_seqs
        self.max_model_len = max_model_len
        self._engine: Any = None
        self._preprocessor: Any = None

    def answer(
        self,
        image_path: Path,
        words: list[str],
        boxes: list[tuple[int, int, int, int]],
        size: tuple[int, int],
        questions: list[str],
    ) -> list[FieldAnswer]:
        return self.answer_pages([PageInput(image_path, words, boxes, size)], questions)

    def answer_pages(
        self, pages: list[PageInput], questions: list[str]
    ) -> list[FieldAnswer]:
        """Answer against a whole document. Arctic-TILT reads up to 400k tokens."""
        engine, preprocessor, parts = self._load()
        from PIL import Image

        built = []
        for page in pages:
            with Image.open(page.image_path) as image:
                built.append(
                    parts.Page(
                        words=page.words,
                        bboxes=[list(box) for box in page.boxes],
                        width=page.size[0],
                        height=page.size[1],
                        image=image.copy(),
                    )
                )
        document = parts.Document(ident=pages[0].image_path.stem, pages=built)
        asked = [parts.Question(feature_name=text, text=text) for text in questions]
        samples = preprocessor.preprocess(document, asked)

        from vllm import SamplingParams

        params = SamplingParams(
            temperature=0, logprobs=0, max_tokens=self._max_output()
        )
        for index, sample in enumerate(samples):
            engine.add_request(prompt=sample, request_id=str(index), params=params)

        collected: dict[str, FieldAnswer] = {}
        while engine.has_unfinished_requests():
            for output in engine.step():
                if not output.finished:
                    continue
                question = questions[int(output.request_id)]
                collected[question] = _as_answer(question, output)
        return [collected[question] for question in questions if question in collected]

    def _load(self) -> tuple[Any, Any, Any]:
        if self._engine is None:
            from vllm import LLMEngine
            from vllm.engine.arg_utils import EngineArgs
            from vllm.multimodal import tilt_processor

            self._engine = LLMEngine.from_engine_args(
                EngineArgs(
                    model=self.model,
                    task="tilt_generate",
                    scheduler_cls="vllm.tilt.scheduler.Scheduler",
                    gpu_memory_utilization=self.gpu_memory_utilization,
                    dtype="bfloat16",
                    max_num_seqs=self.max_num_seqs,
                    # The model advertises 125k tokens. Sizing the KV cache for that
                    # needs a GPU to itself; this one also holds the reader and the
                    # crop reader, so the context is bounded to what a form needs.
                    max_model_len=self.max_model_len,
                    enforce_eager=True,
                    disable_async_output_proc=True,
                )
            )
            self._preprocessor = tilt_processor.TiltPreprocessor.from_config(
                model_config=self._engine.model_config.hf_config,
                tokenizer=self._engine.get_tokenizer().backend_tokenizer,
            )
            self._parts = tilt_processor
        return self._engine, self._preprocessor, self._parts

    def _max_output(self) -> int:
        return self._engine.model_config.hf_config.max_output_length


def _as_answer(question: str, output: Any) -> FieldAnswer:
    """Split the pipe-separated values and score them the way the reference does."""
    generated = output.outputs[0]
    values = tuple(
        value.strip() for value in generated.text.split("|") if value.strip()
    )
    logprobs = generated.logprobs or []
    score = None
    if logprobs:
        import math

        # exp(min token logprob): the reference implementation's span score, which is a
        # weakest-link aggregate and not a calibrated probability of correctness.
        score = math.exp(min(next(iter(step.values())).logprob for step in logprobs))
    return FieldAnswer(field_id=question, label=question, values=values, score=score)
