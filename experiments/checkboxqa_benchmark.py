"""Answer CheckboxQA with the deployed pipeline plus Arctic-TILT, in their gold format.

Pages are read by the running demo over HTTP rather than by a second in-process cascade,
so this measures what is actually deployed and does not put a second copy of every model
on the GPU. Output is scored with the dataset's own evaluate.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ocr_pipeline.comprehension import HttpTiltEngine, PageInput  # noqa: E402

# Kinds that carry no transcription of their own and would only add noise as words.
SKIPPED_KINDS = frozenset({"coverage_risk", "table_candidate"})


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    gold = _read_gold(args.gold)
    if args.limit:
        gold = dict(list(gold.items())[: args.limit])
    engine = HttpTiltEngine(args.tilt_url, timeout_seconds=args.timeout_seconds)
    engine.check_health()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = _already_answered(args.output)
    with args.output.open("a", encoding="utf-8") as sink:
        for index, (name, questions) in enumerate(gold.items(), start=1):
            if name in done:
                continue
            document = args.documents / f"{name}.pdf"
            if not document.is_file():
                print(f"[{index}/{len(gold)}] {name}: missing pdf", flush=True)
                continue
            started = time.perf_counter()
            try:
                record = _answer_document(name, document, questions, engine, args)
            except Exception as error:  # noqa: BLE001 - one bad document must not stop the run
                print(f"[{index}/{len(gold)}] {name}: FAILED {error}", flush=True)
                continue
            sink.write(json.dumps(record) + "\n")
            sink.flush()
            print(
                f"[{index}/{len(gold)}] {name}: {len(questions)} questions "
                f"in {time.perf_counter() - started:.1f}s",
                flush=True,
            )
    return 0


def _answer_document(
    name: str,
    document: Path,
    questions: list[tuple[int, str]],
    engine: HttpTiltEngine,
    args: argparse.Namespace,
) -> dict[str, Any]:
    with TemporaryDirectory(prefix="checkboxqa-") as work:
        pages = _read_pages(document, Path(work), args)
        answers = {}
        for batch in _batched([text for _, text in questions], args.question_batch):
            for answer in engine.answer_pages(pages, batch):
                answers[answer.label] = answer
    return {
        "name": name,
        "extension": "pdf",
        "annotations": [
            {
                "id": identifier,
                "key": text,
                "values": [
                    {"value": value}
                    for value in (answers[text].values if text in answers else ())
                ]
                or [{"value": "None"}],
            }
            for identifier, text in questions
        ],
    }


def _read_pages(
    document: Path, work: Path, args: argparse.Namespace
) -> list[PageInput]:
    """Render each page and read it with the deployed pipeline."""
    import pymupdf

    pages = []
    with pymupdf.open(document) as pdf:
        for number, page in enumerate(pdf, start=1):
            if number > args.max_pages:
                break
            path = work / f"page-{number}.png"
            page.get_pixmap(dpi=args.dpi).save(path)
            words, boxes, size = _read_page(path, args)
            if words:
                pages.append(PageInput(path, words, boxes, size))
    if not pages:
        raise RuntimeError("no page produced readable text")
    return pages


def _read_page(
    path: Path, args: argparse.Namespace
) -> tuple[list[str], list[tuple[int, int, int, int]], tuple[int, int]]:
    payload = _post_file(f"{args.demo_url}/api/process", path, args.timeout_seconds)
    session = payload.get("session_id")
    try:
        page = payload["result"]["pages"][0]
        words: list[str] = []
        boxes: list[tuple[int, int, int, int]] = []
        for region in sorted(page["regions"], key=lambda item: item["reading_order"]):
            if region["kind"] in SKIPPED_KINDS:
                continue
            if (region.get("structure") or {}).get("role") == "layout_block":
                continue
            text = str(region.get("text") or "").strip()
            if region.get("resolution") != "resolved" or not text:
                continue
            box = region["bounding_box"]
            words.append(text)
            boxes.append((box["left"], box["top"], box["right"], box["bottom"]))
        size = (
            max((box[2] for box in boxes), default=1),
            max((box[3] for box in boxes), default=1),
        )
        return words, boxes, size
    finally:
        if session:
            _delete(f"{args.demo_url}/api/sessions/{session}", args.timeout_seconds)


def _post_file(url: str, path: Path, timeout: float) -> dict[str, Any]:
    boundary = "----checkboxqa"
    body = b"".join(
        [
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
            f'filename="{path.name}"\r\nContent-Type: image/png\r\n\r\n'.encode(),
            path.read_bytes(),
            f"\r\n--{boundary}--\r\n".encode(),
        ]
    )
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as reply:
        return json.loads(reply.read())


def _delete(url: str, timeout: float) -> None:
    request = urllib.request.Request(url, method="DELETE")
    try:
        with urllib.request.urlopen(request, timeout=timeout):
            return
    except OSError:
        return


def _read_gold(path: Path) -> dict[str, list[tuple[int, str]]]:
    gold: dict[str, list[tuple[int, str]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        gold[record["name"]] = [
            (annotation["id"], annotation["key"])
            for annotation in record["annotations"]
        ]
    return gold


def _already_answered(path: Path) -> set[str]:
    """Resuming matters: a full sweep is thousands of page reads."""
    if not path.is_file():
        return set()
    return {
        json.loads(line)["name"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def _batched(items: list[str], size: int) -> list[list[str]]:
    return [items[start : start + size] for start in range(0, len(items), size)]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--demo-url", default="http://127.0.0.1:8082")
    parser.add_argument("--tilt-url", default="http://127.0.0.1:8086")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--max-pages", type=int, default=32)
    parser.add_argument("--question-batch", type=int, default=16)
    parser.add_argument("--timeout-seconds", type=float, default=900)
    parser.add_argument("--limit", type=int, default=0)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
