"""Serve a local UI for pixel-aligned handwriting annotation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
import os
from pathlib import Path
import socket
import tempfile
from threading import Lock
from typing import Sequence
from urllib.parse import parse_qs, urlparse

from PIL import Image, UnidentifiedImageError


LEGIBILITY = {"legible", "ambiguous", "unreadable"}
NEGATIVE_TYPES = {"blank", "printed_only", "stray_mark"}
REGION_TYPES = {"field", "line", "signature"} | NEGATIVE_TYPES
MAX_REQUEST_BYTES = 1_000_000


@dataclass(frozen=True)
class Page:
    page_id: str
    image_path: Path
    width: int
    height: int
    media_type: str


class AnnotationStore:
    def __init__(self, queue_path: Path, output_path: Path) -> None:
        if queue_path.resolve() == output_path.resolve():
            raise ValueError("queue and output paths must be different")
        self.pages = _load_queue(queue_path)
        self.output_path = output_path
        self._page_by_id = {page.page_id: page for page in self.pages}
        self._records = self._load_output()
        self._lock = Lock()

    @property
    def completed(self) -> int:
        with self._lock:
            return len(self._records)

    def page_payload(self, index: int) -> dict[str, object]:
        try:
            page = self.pages[index]
        except IndexError as error:
            raise ValueError("page index is out of range") from error
        with self._lock:
            annotation = self._records.get(page.page_id)
            return {
                "index": index,
                "total": len(self.pages),
                "completed": len(self._records),
                "page_id": page.page_id,
                "width": page.width,
                "height": page.height,
                "image_url": f"/image?index={index}",
                "annotation": annotation,
            }

    def save(self, index: int, value: object) -> dict[str, object]:
        try:
            page = self.pages[index]
        except IndexError as error:
            raise ValueError("page index is out of range") from error
        record = _validate_annotation(value, page)
        with self._lock:
            self._records[page.page_id] = record
            self._write_output()
            return {
                "completed": len(self._records),
                "total": len(self.pages),
                "annotation": record,
            }

    def _load_output(self) -> dict[str, dict[str, object]]:
        if not self.output_path.exists():
            return {}
        records: dict[str, dict[str, object]] = {}
        for line_number, line in enumerate(
            self.output_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid output JSON on line {line_number}: {error.msg}"
                ) from error
            page_id = value.get("page_id") if isinstance(value, dict) else None
            page = self._page_by_id.get(page_id) if isinstance(page_id, str) else None
            if page is None:
                raise ValueError(f"unknown output page on line {line_number}")
            if page_id in records:
                raise ValueError(
                    f"duplicate output page on line {line_number}: {page_id}"
                )
            records[page_id] = _validate_annotation(value, page)
        return records

    def _write_output(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=self.output_path.parent,
                prefix=f".{self.output_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                for page in self.pages:
                    record = self._records.get(page.page_id)
                    if record is not None:
                        json.dump(record, stream, ensure_ascii=False, sort_keys=True)
                        stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
                temporary_path = Path(stream.name)
            temporary_path.replace(self.output_path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()


class AnnotationServer(ThreadingHTTPServer):
    daemon_threads = True


class IPv6AnnotationServer(AnnotationServer):
    address_family = socket.AF_INET6


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    server = build_server(args.queue, args.output, args.host, args.port)
    host, port = server.server_address[:2]
    display_host = f"[{host}]" if ":" in host else host
    print(f"Handwriting annotator: http://{display_host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("queue", type=Path, help="triage queue.jsonl")
    parser.add_argument("output", type=Path, help="review annotations JSONL")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser


def build_server(
    queue_path: Path,
    output_path: Path,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> ThreadingHTTPServer:
    address = _loopback_address(host)
    store = AnnotationStore(queue_path, output_path)

    class Handler(AnnotationHandler):
        annotation_store = store

    server_class = IPv6AnnotationServer if address.version == 6 else AnnotationServer
    return server_class((host, port), Handler)


def _loopback_address(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    if host == "localhost":
        return ipaddress.ip_address("127.0.0.1")
    try:
        address = ipaddress.ip_address(host)
    except ValueError as error:
        raise ValueError("host must be a loopback IP address or localhost") from error
    if not address.is_loopback:
        raise ValueError("host must be a loopback IP address or localhost")
    return address


class AnnotationHandler(BaseHTTPRequestHandler):
    annotation_store: AnnotationStore

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                self._send(HTTPStatus.OK, HTML.encode(), "text/html; charset=utf-8")
                return
            if parsed.path not in {"/api/page", "/image"}:
                self._send_error(HTTPStatus.NOT_FOUND, "not found")
                return
            index = _query_index(parsed.query)
            if parsed.path == "/api/page":
                self._send_json(
                    HTTPStatus.OK, self.annotation_store.page_payload(index)
                )
                return
            page = self.annotation_store.pages[index]
            self._send(HTTPStatus.OK, page.image_path.read_bytes(), page.media_type)
        except (IndexError, OSError, ValueError) as error:
            self._send_error(HTTPStatus.BAD_REQUEST, str(error))

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/page":
            self._send_error(HTTPStatus.NOT_FOUND, "not found")
            return
        if self.headers.get_content_type() != "application/json":
            self._send_error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "expected JSON")
            return
        try:
            length = int(self.headers.get("Content-Length", ""))
            if length < 1 or length > MAX_REQUEST_BYTES:
                raise ValueError("invalid request size")
            value = json.loads(self.rfile.read(length))
            result = self.annotation_store.save(_query_index(parsed.query), value)
            self._send_json(HTTPStatus.OK, result)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
            self._send_error(HTTPStatus.BAD_REQUEST, str(error))

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send_json(self, status: HTTPStatus, value: object) -> None:
        body = json.dumps(value, ensure_ascii=False).encode()
        self._send(status, body, "application/json; charset=utf-8")

    def _send_error(self, status: HTTPStatus, message: str) -> None:
        self._send_json(status, {"error": message})

    def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; style-src 'self' 'unsafe-inline'; "
            "script-src 'self' 'unsafe-inline'; img-src 'self'",
        )
        self.end_headers()
        self.wfile.write(body)


def _load_queue(path: Path) -> list[Page]:
    if not path.is_file():
        raise FileNotFoundError(f"queue was not found: {path}")
    pages: list[Page] = []
    seen: set[str] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid queue JSON on line {line_number}: {error.msg}"
            ) from error
        page_id = value.get("page_id") if isinstance(value, dict) else None
        image_value = value.get("image_path") if isinstance(value, dict) else None
        if not isinstance(page_id, str) or not page_id.strip():
            raise ValueError(f"queue line {line_number} lacks a page_id")
        if page_id in seen:
            raise ValueError(f"duplicate page_id on line {line_number}: {page_id}")
        if not isinstance(image_value, str) or not image_value:
            raise ValueError(f"queue line {line_number} lacks an image_path")
        image_path = Path(image_value)
        if not image_path.is_absolute():
            image_path = path.parent / image_path
        image_path = image_path.resolve()
        try:
            with Image.open(image_path) as image:
                width, height = image.size
                media_type = Image.MIME.get(
                    image.format or "", "application/octet-stream"
                )
        except (OSError, UnidentifiedImageError) as error:
            raise ValueError(f"could not read queue image: {image_path}") from error
        seen.add(page_id)
        pages.append(Page(page_id, image_path, width, height, media_type))
    if not pages:
        raise ValueError("queue has no pages")
    return pages


def _validate_annotation(value: object, page: Page) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("annotation must be an object")
    reviewer = value.get("reviewer_id")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("reviewer_id is required")
    supplied_page_id = value.get("page_id", page.page_id)
    if supplied_page_id != page.page_id:
        raise ValueError("page_id does not match the requested page")
    values = value.get("regions")
    if not isinstance(values, list):
        raise ValueError("regions must be a list")
    regions = [
        _validate_region(region, page, index)
        for index, region in enumerate(values, start=1)
    ]
    return {
        "page_id": page.page_id,
        "reviewer_id": reviewer.strip(),
        "regions": regions,
    }


def _validate_region(value: object, page: Page, index: int) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"region {index} must be an object")
    box = value.get("bbox")
    if (
        not isinstance(box, list)
        or len(box) != 4
        or any(not isinstance(item, int) or isinstance(item, bool) for item in box)
    ):
        raise ValueError(f"region {index} bbox must contain four integers")
    left, top, right, bottom = box
    if not (0 <= left < right <= page.width and 0 <= top < bottom <= page.height):
        raise ValueError(f"region {index} bbox is outside the image or empty")
    text = value.get("text")
    if not isinstance(text, str):
        raise ValueError(f"region {index} text must be a string")
    legibility = value.get("legibility")
    if legibility not in LEGIBILITY:
        raise ValueError(f"region {index} has invalid legibility")
    region_type = value.get("region_type")
    if region_type not in REGION_TYPES:
        raise ValueError(f"region {index} has invalid region_type")
    if region_type in NEGATIVE_TYPES:
        if text.strip():
            raise ValueError(f"region {index} negative transcription must be empty")
        if legibility != "legible":
            raise ValueError(f"region {index} negative must be confidently marked")
    elif legibility == "unreadable" and text.strip():
        raise ValueError(f"region {index} unreadable transcription must be empty")
    elif legibility != "unreadable" and not text.strip():
        raise ValueError(f"region {index} transcription is required")
    return {
        "bbox": box,
        "text": text,
        "legibility": legibility,
        "region_type": region_type,
    }


def _query_index(query: str) -> int:
    values = parse_qs(query).get("index")
    if values is None or len(values) != 1:
        raise ValueError("one page index is required")
    try:
        index = int(values[0])
    except ValueError as error:
        raise ValueError("page index must be an integer") from error
    if index < 0:
        raise ValueError("page index is out of range")
    return index


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Handwriting annotation</title>
  <style>
    :root { color-scheme: light; font-family: system-ui, sans-serif; }
    * { box-sizing: border-box; }
    body { margin: 0; background: #eef1f5; color: #172033; }
    header { align-items: center; background: #172033; color: white; display: flex;
      gap: 12px; padding: 10px 16px; position: sticky; top: 0; z-index: 2; }
    header strong { margin-right: auto; }
    button, input, select, textarea { font: inherit; }
    button { cursor: pointer; padding: 7px 11px; }
    button:disabled { cursor: default; opacity: .45; }
    main { display: grid; gap: 14px; grid-template-columns: minmax(0, 1fr) 340px;
      padding: 14px; }
    .page { background: #333b4b; border-radius: 6px; min-height: calc(100vh - 80px);
      overflow: auto; padding: 12px; text-align: center; }
    canvas { background: white; cursor: crosshair; height: auto; max-width: 100%; }
    aside { background: white; border-radius: 6px; padding: 14px; }
    label { display: block; font-size: 13px; font-weight: 650; margin: 12px 0 4px; }
    input, select, textarea { border: 1px solid #a9b1bf; border-radius: 4px; padding: 7px;
      width: 100%; }
    textarea { min-height: 90px; resize: vertical; }
    .actions { display: flex; flex-wrap: wrap; gap: 7px; margin-top: 10px; }
    #regions { border-top: 1px solid #d9dde5; margin-top: 14px; padding-top: 10px; }
    .region { background: #f3f5f8; border: 1px solid transparent; display: block;
      margin: 5px 0; overflow: hidden; text-align: left; text-overflow: ellipsis;
      white-space: nowrap; width: 100%; }
    .region.selected { border-color: #1467d8; background: #e6f0ff; }
    #status { color: #526078; font-size: 13px; min-height: 20px; margin-top: 8px; }
    @media (max-width: 850px) { main { grid-template-columns: 1fr; } .page { min-height: 55vh; } }
  </style>
</head>
<body>
  <header>
    <strong id="pageId">Loading...</strong>
    <span id="progress"></span>
    <button id="previous" type="button">Previous</button>
    <button id="next" type="button">Next</button>
  </header>
  <main>
    <section class="page"><canvas id="canvas"></canvas></section>
    <aside>
      <label for="reviewer">Reviewer ID</label>
      <input id="reviewer" autocomplete="off" required>
      <label for="transcription">Literal transcription</label>
      <textarea id="transcription" spellcheck="false"></textarea>
      <label for="legibility">Legibility</label>
      <select id="legibility">
        <option value="legible">Legible</option>
        <option value="ambiguous">Ambiguous</option>
        <option value="unreadable">Unreadable</option>
      </select>
      <label for="regionType">Region type</label>
      <select id="regionType">
        <option value="field">Field</option>
        <option value="line">Line</option>
        <option value="signature">Signature</option>
        <option value="blank">Blank field</option>
        <option value="printed_only">Printed only</option>
        <option value="stray_mark">Stray mark</option>
      </select>
      <div class="actions">
        <button id="add" type="button">Add region</button>
        <button id="update" type="button">Update region</button>
        <button id="delete" type="button">Delete region</button>
      </div>
      <div id="regions"></div>
      <div class="actions">
        <button id="save" type="button">Save page</button>
        <button id="saveNext" type="button">Save and next</button>
      </div>
      <div id="status" role="status"></div>
    </aside>
  </main>
  <script>
    const canvas = document.querySelector('#canvas');
    const context = canvas.getContext('2d');
    const reviewer = document.querySelector('#reviewer');
    const transcription = document.querySelector('#transcription');
    const legibility = document.querySelector('#legibility');
    const regionType = document.querySelector('#regionType');
    const status = document.querySelector('#status');
    const image = new Image();
    const negativeTypes = new Set(['blank', 'printed_only', 'stray_mark']);
    let pageIndex = Number(new URLSearchParams(location.search).get('page') || 0);
    let page = null, regions = [], selected = -1, draftBox = null, dragStart = null;
    let dirty = false;

    function point(event) {
      const rect = canvas.getBoundingClientRect();
      return [
        Math.max(0, Math.min(canvas.width, (event.clientX - rect.left) * canvas.width / rect.width)),
        Math.max(0, Math.min(canvas.height, (event.clientY - rect.top) * canvas.height / rect.height))
      ];
    }
    function normalizedBox(a, b) {
      return [Math.floor(Math.min(a[0], b[0])), Math.floor(Math.min(a[1], b[1])),
        Math.ceil(Math.max(a[0], b[0])), Math.ceil(Math.max(a[1], b[1]))];
    }
    function draw() {
      if (!image.complete) return;
      context.drawImage(image, 0, 0, canvas.width, canvas.height);
      context.lineWidth = Math.max(2, canvas.width / 700);
      regions.forEach((region, index) => drawBox(region.bbox, index === selected ? '#1269e8' : '#e33131'));
      if (draftBox) drawBox(draftBox, '#00a66a');
    }
    function drawBox(box, color) {
      context.strokeStyle = color;
      context.strokeRect(box[0], box[1], box[2] - box[0], box[3] - box[1]);
    }
    canvas.addEventListener('pointerdown', event => {
      dragStart = point(event); draftBox = null; canvas.setPointerCapture(event.pointerId);
    });
    canvas.addEventListener('pointermove', event => {
      if (!dragStart) return;
      draftBox = normalizedBox(dragStart, point(event)); draw();
    });
    canvas.addEventListener('pointerup', event => {
      if (!dragStart) return;
      draftBox = normalizedBox(dragStart, point(event)); dragStart = null;
      if (draftBox[2] <= draftBox[0] || draftBox[3] <= draftBox[1]) draftBox = null;
      dirty = true; draw();
    });

    async function load(index) {
      const response = await fetch(`/api/page?index=${index}`);
      if (!response.ok) { status.textContent = (await response.json()).error; return; }
      page = await response.json(); pageIndex = index;
      regions = page.annotation ? structuredClone(page.annotation.regions) : [];
      if (page.annotation) reviewer.value = page.annotation.reviewer_id;
      else reviewer.value = localStorage.getItem('handwriting-reviewer') || reviewer.value;
      selected = -1; draftBox = null; dirty = false; clearEditor();
      canvas.width = page.width; canvas.height = page.height;
      image.onload = draw; image.src = `${page.image_url}&v=${Date.now()}`;
      document.querySelector('#pageId').textContent = page.page_id;
      document.querySelector('#progress').textContent = `${page.completed}/${page.total} saved | page ${page.index + 1}/${page.total}`;
      document.querySelector('#previous').disabled = pageIndex === 0;
      document.querySelector('#next').disabled = pageIndex + 1 === page.total;
      renderRegions(); status.textContent = page.annotation ? 'Saved annotation loaded.' : 'Draw a box to begin.';
      history.replaceState(null, '', `?page=${pageIndex}`);
    }
    function clearEditor() {
      transcription.value = ''; legibility.value = 'legible'; regionType.value = 'field'; draftBox = null;
      syncEditor();
    }
    function syncEditor() {
      const negative = negativeTypes.has(regionType.value);
      if (negative) legibility.value = 'legible';
      if (negative || legibility.value === 'unreadable') transcription.value = '';
      transcription.disabled = negative || legibility.value === 'unreadable';
      legibility.disabled = negative;
    }
    function renderRegions() {
      const list = document.querySelector('#regions'); list.replaceChildren();
      regions.forEach((region, index) => {
        const button = document.createElement('button');
        button.type = 'button'; button.className = `region${index === selected ? ' selected' : ''}`;
        const label = region.text || (negativeTypes.has(region.region_type) ? region.region_type : '(unreadable)');
        button.textContent = `${index + 1}. ${label} [${region.bbox.join(', ')}]`;
        button.addEventListener('click', () => selectRegion(index)); list.append(button);
      });
    }
    function selectRegion(index) {
      selected = index; draftBox = [...regions[index].bbox];
      transcription.value = regions[index].text; legibility.value = regions[index].legibility;
      regionType.value = regions[index].region_type;
      syncEditor();
      renderRegions(); draw();
    }
    function editorRegion() {
      if (!draftBox) throw new Error('Draw a bounding box first.');
      if (!negativeTypes.has(regionType.value) && legibility.value !== 'unreadable' && !transcription.value.trim()) throw new Error('Enter a transcription.');
      return {bbox: [...draftBox], text: transcription.value,
        legibility: legibility.value, region_type: regionType.value};
    }
    function edit(action) {
      try { action(); dirty = true; status.textContent = 'Unsaved changes.'; renderRegions(); draw(); }
      catch (error) { status.textContent = error.message; }
    }
    document.querySelector('#add').addEventListener('click', () => edit(() => {
      regions.push(editorRegion()); selected = regions.length - 1; clearEditor(); selected = -1;
    }));
    document.querySelector('#update').addEventListener('click', () => edit(() => {
      if (selected < 0) throw new Error('Select a region to update.'); regions[selected] = editorRegion();
    }));
    document.querySelector('#delete').addEventListener('click', () => edit(() => {
      if (selected < 0) throw new Error('Select a region to delete.');
      regions.splice(selected, 1); selected = -1; clearEditor();
    }));
    reviewer.addEventListener('input', () => { dirty = true; });
    transcription.addEventListener('input', () => { dirty = true; });
    legibility.addEventListener('change', () => { syncEditor(); dirty = true; });
    regionType.addEventListener('change', () => { syncEditor(); dirty = true; });

    function hasPendingRegionEdit() {
      if (selected < 0) return draftBox !== null || transcription.value !== '';
      const current = regions[selected];
      return JSON.stringify(draftBox) !== JSON.stringify(current.bbox) ||
        transcription.value !== current.text || legibility.value !== current.legibility ||
        regionType.value !== current.region_type;
    }

    async function save() {
      if (hasPendingRegionEdit()) {
        status.textContent = selected < 0 ? 'Add the drafted region before saving.' : 'Update the edited region before saving.';
        return false;
      }
      const response = await fetch(`/api/page?index=${pageIndex}`, {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({page_id: page.page_id, reviewer_id: reviewer.value, regions})
      });
      const result = await response.json();
      if (!response.ok) { status.textContent = result.error; return false; }
      dirty = false; localStorage.setItem('handwriting-reviewer', reviewer.value);
      document.querySelector('#progress').textContent = `${result.completed}/${result.total} saved | page ${pageIndex + 1}/${result.total}`;
      status.textContent = 'Page saved atomically.'; return true;
    }
    async function navigate(delta) {
      if (dirty && !confirm('Discard unsaved changes?')) return;
      await load(pageIndex + delta);
    }
    document.querySelector('#save').addEventListener('click', save);
    document.querySelector('#saveNext').addEventListener('click', async () => {
      if (await save() && pageIndex + 1 < page.total) await load(pageIndex + 1);
    });
    document.querySelector('#previous').addEventListener('click', () => navigate(-1));
    document.querySelector('#next').addEventListener('click', () => navigate(1));
    addEventListener('beforeunload', event => { if (dirty) { event.preventDefault(); event.returnValue = ''; } });
    load(pageIndex);
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
