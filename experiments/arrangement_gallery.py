"""Regression gallery for OCR page arrangement (de-duplication, reading order, label/value
pairing, layout rendering).

Run mode processes pages through a running OCR service and checks the results::

    experiments/arrangement_gallery.py --url http://127.0.0.1:18194 --out DIR \\
        [--examples] [FILE ...]

Recheck mode re-runs the checks from previously saved ``process.json`` files using a
chosen ``ocr_pipeline`` source tree, without any new OCR (so page_arrangement.py can be
edited and re-checked for free). ``--label`` keeps multiple recheck reports side by side
(``report-<label>.md``) for before/after comparisons::

    experiments/arrangement_gallery.py --recheck DIR [--src PATH] [--label LABEL]

For each page this writes, under ``DIR/<name>/``: ``process.json`` (raw /api/process
response), ``result.json`` (with arrangement), ``result.md``, ``display.json`` (the UI's
compact display payload), ``page_<n>.png`` and ``order_p<n>.png`` (the page image with
each arrangement block boxed and numbered, suppressed regions dashed grey). ``DIR/report.md``
(or ``report-<label>.md`` in recheck mode) holds one row per page with any failing checks.

Checks per page:
  (a) token conservation -- the Markdown page body's word tokens vs. the arrangement's
      shown (non-suppressed, non-trimmed) region text.
  (b) section order -- the display response's reading sections (``text.evidence_ids``)
      vs. the arrangement's block order.
  (c) extractor rows -- ``page_arrangement.block_text(..., with_ids=True)`` cites region
      ids in arrangement order, consistent with the Markdown section order.
  (d) layout rules -- headings never come from a figure/stamp or a label/value pair,
      tables are HTML not pipe syntax, checkboxes render as "[x]"/"[ ]", list items as
      "- "/"1.", and no LLM-routing strings leak into the page.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

_WORD_RE = re.compile(r"\w+")
_ID_RE = re.compile(r"\[([^\[\]]+)\]")
# Tags/attributes the HTML table renderer emits that are not page content.
_MARKDOWN_SYNTAX_TOKENS = {"table", "tbody", "thead", "tr", "td", "th", "br", "rowspan", "colspan"}
_FORBIDDEN_STRINGS = ("Routed to LLM", "Corrected by LLM")
_LABELED_PATH_SEGMENTS = ("funsd", "omnidocbench", "synthetic", "packet")


# --- Loading the ocr_pipeline modules under test ------------------------------------


def _load_ocr_modules(src_root: Path) -> tuple[Any, Any]:
    src_str = str(src_root)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
    for module_name in ("ocr_pipeline", "ocr_pipeline.demo", "ocr_pipeline.page_arrangement"):
        sys.modules.pop(module_name, None)
    demo_module = importlib.import_module("ocr_pipeline.demo")
    page_arrangement_module = importlib.import_module("ocr_pipeline.page_arrangement")
    return demo_module, page_arrangement_module


# --- Pure check logic (used by both run and recheck modes) --------------------------


def _first_divergence_and_inversions(order_a: list[Any], order_b: list[Any]) -> tuple[int | None, int]:
    """``order_a``/``order_b`` must be permutations of the same ids. Returns the first
    differing position (``None`` if identical) and a Kendall-tau-like inversion count."""
    first_divergence = next((i for i, (a, b) in enumerate(zip(order_a, order_b)) if a != b), None)
    rank_in_b = {item: index for index, item in enumerate(order_b)}
    ranks = [rank_in_b[item] for item in order_a]
    # ponytail: O(n^2) inversion count; a page has at most a few hundred ids.
    inversions = sum(ranks[i] > ranks[j] for i in range(len(ranks)) for j in range(i + 1, len(ranks)))
    return first_divergence, inversions


def _dedupe_preserve_order(items: list[Any]) -> list[Any]:
    return list(dict.fromkeys(items))


def _page_body(result_md: str, page_number: int) -> str:
    match = re.search(rf"^## Page {page_number}\s*$", result_md, re.MULTILINE)
    if not match:
        return ""
    start = match.end()
    next_heading = re.search(r"^## (?:Page \d+|Failures)\s*$", result_md[start:], re.MULTILINE)
    end = start + next_heading.start() if next_heading else len(result_md)
    return result_md[start:end]


def _token_conservation_check(page: dict[str, Any], page_body: str) -> dict[str, Any]:
    arrangement = page["arrangement"]
    trimmed: dict[Any, set[int]] = {}
    for entry in arrangement.get("trimmed_lines", []):
        trimmed.setdefault(entry["region_id"], set()).add(entry["line"])
    index = {r["id"]: r for r in page.get("regions", []) if r.get("id") is not None}
    shown_ids = {i for block in arrangement.get("blocks", []) for i in block["label_ids"] + block["region_ids"]}
    region_tokens: Counter = Counter()
    for region_id in shown_ids:
        region = index.get(region_id)
        if region is None:
            continue
        skip = trimmed.get(region_id, set())
        lines = [line for i, line in enumerate((region.get("text") or "").split("\n")) if i not in skip]
        region_tokens.update(_WORD_RE.findall(" ".join(lines).casefold()))
    markdown_tokens = Counter(_WORD_RE.findall(page_body.casefold()))
    for token in _MARKDOWN_SYNTAX_TOKENS:
        markdown_tokens.pop(token, None)
    dropped = region_tokens - markdown_tokens
    duplicated = markdown_tokens - region_tokens
    return {"dropped": dict(dropped), "duplicated": dict(duplicated), "ok": not dropped and not duplicated}


def _section_order_check(display_page: dict[str, Any], arrangement: dict[str, Any]) -> dict[str, Any]:
    index = {r["id"]: r for r in display_page.get("regions", []) if r.get("id") is not None}
    blocks = arrangement.get("blocks", [])
    block_first_ids = [ids[0] for block in blocks if (ids := block["label_ids"] + block["region_ids"])]
    visible_ids = {i for block in blocks for i in block["label_ids"] + block["region_ids"]}
    display_order = []
    for evidence_id in display_page.get("text", {}).get("evidence_ids", []):
        region = index.get(evidence_id)
        children = (region.get("structure") or {}).get("child_evidence_ids") if region else None
        candidates = list(children) if children else [evidence_id]
        representative = next((c for c in candidates if c in visible_ids), None)
        if representative is not None:
            display_order.append(representative)
    common = set(display_order) & set(block_first_ids)
    display_restricted = _dedupe_preserve_order([i for i in display_order if i in common])
    arrangement_restricted = _dedupe_preserve_order([i for i in block_first_ids if i in common])
    first_divergence, inversions = _first_divergence_and_inversions(display_restricted, arrangement_restricted)
    return {
        "compared": len(display_restricted),
        "first_divergence": first_divergence,
        "inversions": inversions,
        "ok": first_divergence is None,
    }


def _extractor_rows_check(page: dict[str, Any], display_page: dict[str, Any], page_arrangement_module: Any) -> dict[str, Any]:
    arrangement = page["arrangement"]
    blocks = arrangement.get("blocks", [])
    known_ids = {r["id"] for r in page.get("regions", []) if r.get("id") is not None}
    arrangement_full_order = [i for block in blocks for i in block["label_ids"] + block["region_ids"]]
    cited_order: list[Any] = []
    block_order_evidence = None
    for block_index, block in enumerate(blocks):
        text = page_arrangement_module.block_text(page, block, with_ids=True, arrangement=arrangement)
        block_own_ids = block["label_ids"] + block["region_ids"]
        cited_in_block = [match for match in _ID_RE.findall(text) if match in known_ids]
        if block_order_evidence is None:
            expected = [i for i in block_own_ids if i in cited_in_block]
            if cited_in_block != expected:
                block_order_evidence = f"block {block_index} cited {cited_in_block}, expected {expected}"
        cited_order.extend(cited_in_block)
    expected_cited_order = [i for i in arrangement_full_order if i in cited_order]
    block_order_ok = block_order_evidence is None and cited_order == expected_cited_order
    if block_order_ok is False and block_order_evidence is None:
        block_order_evidence = f"cited order {cited_order} != arrangement order {expected_cited_order}"

    markdown_index = {r["id"]: r for r in display_page.get("regions", []) if r.get("id") is not None}
    markdown_region_order: list[Any] = []
    for evidence_id in display_page.get("text", {}).get("evidence_ids", []):
        region = markdown_index.get(evidence_id)
        children = (region.get("structure") or {}).get("child_evidence_ids") if region else None
        markdown_region_order.extend(children if children else [evidence_id])
    common = set(cited_order) & set(markdown_region_order)
    cited_restricted = _dedupe_preserve_order([i for i in cited_order if i in common])
    markdown_restricted = _dedupe_preserve_order([i for i in markdown_region_order if i in common])
    first_divergence, inversions = _first_divergence_and_inversions(cited_restricted, markdown_restricted)
    return {
        "block_order_ok": block_order_ok,
        "block_order_evidence": block_order_evidence,
        "compared_with_markdown": len(cited_restricted),
        "first_divergence": first_divergence,
        "inversions": inversions,
        "ok": block_order_ok and first_divergence is None,
    }


def _layout_rules_check(page: dict[str, Any], page_body: str, page_arrangement_module: Any) -> dict[str, Any]:
    arrangement = page["arrangement"]
    index = page_arrangement_module._region_index(page)
    violations: list[str] = []
    for block in arrangement.get("blocks", []):
        markdown = page_arrangement_module._block_markdown(page, block, index, arrangement)
        is_heading = markdown.lstrip().startswith("#")
        if is_heading and block["kind"] == "heading":
            source_region = index.get(block["region_ids"][0]) if block["region_ids"] else None
            if source_region is not None and source_region.get("kind") in {"figure", "stamp"}:
                violations.append(f"heading from a {source_region.get('kind')} region: {markdown[:60]!r}")
        if is_heading and block["kind"] == "key_value":
            violations.append(f"a label/value pair rendered as a heading: {markdown[:60]!r}")
        if block["kind"] == "table":
            pipe_rows = [line for line in markdown.splitlines() if re.match(r"^\s*\|.*\|\s*$", line)]
            if pipe_rows and "<table" not in markdown:
                violations.append(f"table rendered as a Markdown pipe table, not HTML: {pipe_rows[0]!r}")
        if block["kind"] == "controls":
            bad_lines = [
                line for line in markdown.splitlines() if line.startswith("- ") and not re.match(r"^- \[[x ?]\] ", line)
            ]
            if bad_lines:
                violations.append(f"checkbox not rendered as '[x]'/'[ ]': {bad_lines[0]!r}")
        if block["kind"] == "list" and markdown and not (markdown.startswith("- ") or re.match(r"^\d+\.", markdown)):
            violations.append(f"list item not rendered as '- ' or '1.': {markdown[:60]!r}")
    for phrase in _FORBIDDEN_STRINGS:
        if phrase in page_body:
            violations.append(f"found {phrase!r} in the rendered page")
    return {"violations": violations, "ok": not violations}


def _check_pages(
    name: str,
    pages_with_arrangement: list[dict[str, Any]],
    result_md: str,
    display_pages: dict[int, dict[str, Any]],
    page_arrangement_module: Any,
    *,
    out_dir: Path,
    label: str | None,
) -> list[dict[str, Any]]:
    rows = []
    for page in pages_with_arrangement:
        page_number = page["page_number"]
        arrangement = page.get("arrangement", {})
        row: dict[str, Any] = {"name": name, "page_number": page_number, "kind": _kind_of(name)}
        if "error" in arrangement:
            row["skipped"] = arrangement["error"]
            rows.append(row)
            continue
        body = _page_body(result_md, page_number)
        display_page = display_pages.get(page_number, {})
        row["token_conservation"] = _token_conservation_check(page, body)
        row["section_order"] = _section_order_check(display_page, arrangement)
        row["extractor_rows"] = _extractor_rows_check(page, display_page, page_arrangement_module)
        row["layout_rules"] = _layout_rules_check(page, body, page_arrangement_module)
        image_path = out_dir / name / f"page_{page_number}.png"
        if image_path.exists():
            suffix = f"-{label}" if label else ""
            _draw_order_image(image_path, page, arrangement, out_dir / name / f"order_p{page_number}{suffix}.png")
        rows.append(row)
    return rows


# --- Order image: boxed, numbered blocks; suppressed regions dashed grey ------------


def _scaled_box(box: dict[str, float], scale_x: float, scale_y: float) -> tuple[float, float, float, float]:
    return (box["left"] * scale_x, box["top"] * scale_y, box["right"] * scale_x, box["bottom"] * scale_y)


def _dashed_line(draw: ImageDraw.ImageDraw, start: tuple[float, float], end: tuple[float, float], *, color, dash=6.0, gap=4.0) -> None:
    length = math.hypot(end[0] - start[0], end[1] - start[1])
    if length == 0:
        return
    position = 0.0
    while position < length:
        segment_end = min(position + dash, length)
        f0, f1 = position / length, segment_end / length
        point_a = (start[0] + (end[0] - start[0]) * f0, start[1] + (end[1] - start[1]) * f0)
        point_b = (start[0] + (end[0] - start[0]) * f1, start[1] + (end[1] - start[1]) * f1)
        draw.line([point_a, point_b], fill=color, width=2)
        position += dash + gap


def _dashed_rectangle(draw: ImageDraw.ImageDraw, rect: tuple[float, float, float, float], *, color) -> None:
    x0, y0, x1, y1 = rect
    for start, end in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
        _dashed_line(draw, start, end, color=color)


def _draw_order_image(image_path: Path, page: dict[str, Any], arrangement: dict[str, Any], out_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    scale_x = image.width / (page.get("width") or image.width)
    scale_y = image.height / (page.get("height") or image.height)
    index = {r["id"]: r for r in page.get("regions", []) if r.get("id") is not None}
    for item in arrangement.get("suppressed", []):
        region = index.get(item["region_id"])
        box = region.get("bounding_box") if region else None
        if box:
            _dashed_rectangle(draw, _scaled_box(box, scale_x, scale_y), color=(160, 160, 160))
    for order, block in enumerate(arrangement.get("blocks", []), start=1):
        rect = _scaled_box(block["box"], scale_x, scale_y)
        draw.rectangle(rect, outline=(0, 140, 0), width=2)
        draw.text((rect[0] + 2, max(rect[1] - 12, 0)), str(order), fill=(0, 140, 0))
    image.save(out_path)


# --- Naming ---------------------------------------------------------------------------


def _input_name(path: Path, used: set[str]) -> str:
    parts = {part.lower() for part in path.parts}
    label = next((segment for segment in _LABELED_PATH_SEGMENTS if segment in parts), None)
    name = f"{label}_{path.stem}" if label else path.stem
    if name in used:
        name = f"{name}_{len(used)}"
    used.add(name)
    return name


def _kind_of(name: str) -> str:
    if name.startswith("funsd_"):
        return "scanned form (FUNSD)"
    if name.startswith("omnidocbench_"):
        return "document page (OmniDocBench)"
    if name.startswith("synthetic_"):
        return "fictional synthetic packet"
    if name.startswith("packet_"):
        return "real enrollment packet (private)"
    return name


# --- Run mode: talk to the OCR service -----------------------------------------------


def _post_process(url: str, *, example: str | None, file_path: Path | None) -> dict[str, Any]:
    import requests

    endpoint = f"{url}/api/process"
    if example is not None:
        response = requests.post(endpoint, params={"example": example}, timeout=600)
    else:
        with open(file_path, "rb") as handle:
            response = requests.post(endpoint, files={"file": (file_path.name, handle)}, timeout=600)
    response.raise_for_status()
    return response.json()


def _get_json(url: str) -> Any:
    import requests

    response = requests.get(url, timeout=300)
    response.raise_for_status()
    return response.json()


def _get_text(url: str) -> str:
    import requests

    response = requests.get(url, timeout=300)
    response.raise_for_status()
    return response.text


def _get_bytes(url: str) -> bytes:
    import requests

    response = requests.get(url, timeout=300)
    response.raise_for_status()
    return response.content


def _process_one(url: str, name: str, *, example: str | None, file_path: Path | None, out_dir: Path) -> dict[str, Any] | None:
    import requests

    target = out_dir / name
    target.mkdir(parents=True, exist_ok=True)
    try:
        process_response = _post_process(url, example=example, file_path=file_path)
    except requests.RequestException as error:
        return {"error": f"process failed: {error}"}
    (target / "process.json").write_text(json.dumps(process_response, indent=2, ensure_ascii=False))
    session_id = process_response["session_id"]
    try:
        result_with_arrangement = _get_json(f"{url}/api/sessions/{session_id}/result.json?arrangement=true")
        (target / "result.json").write_text(json.dumps(result_with_arrangement, indent=2, ensure_ascii=False))
        result_md = _get_text(f"{url}/api/sessions/{session_id}/result.md")
        (target / "result.md").write_text(result_md)
        display_response = _get_json(f"{url}/api/sessions/{session_id}?compact=1")
        (target / "display.json").write_text(json.dumps(display_response, indent=2, ensure_ascii=False))
        for page_number in range(1, len(result_with_arrangement["pages"]) + 1):
            image_bytes = _get_bytes(f"{url}/api/sessions/{session_id}/pages/{page_number}")
            (target / f"page_{page_number}.png").write_bytes(image_bytes)
    except requests.RequestException as error:
        return {"error": f"fetching results failed: {error}"}
    finally:
        try:
            requests.delete(f"{url}/api/sessions/{session_id}", timeout=30)
        except requests.RequestException:
            pass
    return {"result_json": result_with_arrangement, "result_md": result_md, "display_json": display_response}


# --- Recheck mode: recompute locally from saved process.json -------------------------


def _recheck_one(target: Path, demo_module: Any, page_arrangement_module: Any, *, out_dir: Path, label: str | None) -> list[dict[str, Any]] | None:
    process_path = target / "process.json"
    if not process_path.exists():
        return None
    process_response = json.loads(process_path.read_text())
    result_md = demo_module._result_markdown(process_response)
    display_response = demo_module._display_response(process_response)
    pages_with_arrangement = [
        {**page, "arrangement": demo_module._arrangement(page)} for page in process_response["result"]["pages"]
    ]
    display_pages = {p["page_number"]: p for p in display_response["result"]["pages"]}
    return _check_pages(target.name, pages_with_arrangement, result_md, display_pages, page_arrangement_module, out_dir=out_dir, label=label)


# --- Report -----------------------------------------------------------------------------

_CHECK_LABELS = [
    ("token_conservation", "token conservation"),
    ("section_order", "section order"),
    ("extractor_rows", "extractor rows"),
    ("layout_rules", "layout rules"),
]


def _one_line_evidence(check_name: str, check: dict[str, Any]) -> str:
    if check_name == "token_conservation":
        parts = []
        if check["dropped"]:
            parts.append(f"dropped {dict(list(check['dropped'].items())[:3])}")
        if check["duplicated"]:
            parts.append(f"duplicated {dict(list(check['duplicated'].items())[:3])}")
        return "; ".join(parts)
    if check_name == "section_order":
        return f"first divergence at position {check['first_divergence']}, {check['inversions']} inversions over {check['compared']} sections"
    if check_name == "extractor_rows":
        if not check["block_order_ok"]:
            return check["block_order_evidence"] or "block_text ids out of arrangement order"
        return f"first divergence at position {check['first_divergence']}, {check['inversions']} inversions over {check['compared_with_markdown']} ids"
    if check_name == "layout_rules":
        return check["violations"][0] if check["violations"] else ""
    return ""


def _write_report(rows: list[dict[str, Any]], out_path: Path) -> None:
    lines = ["# Arrangement gallery report", "", "| Name | Page | Kind | Failing checks | Evidence |", "| --- | --- | --- | --- | --- |"]
    for row in rows:
        if "skipped" in row:
            evidence = str(row["skipped"]).replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {row['name']} | {row['page_number']} | {row['kind']} | arrangement unavailable | {evidence} |")
            continue
        failing, evidence_parts = [], []
        for check_name, label in _CHECK_LABELS:
            check = row[check_name]
            if not check["ok"]:
                failing.append(label)
                evidence_parts.append(_one_line_evidence(check_name, check))
        status = ", ".join(failing) if failing else "OK"
        evidence_text = "; ".join(part for part in evidence_parts if part).replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {row['name']} | {row['page_number']} | {row['kind']} | {status} | {evidence_text} |")
    out_path.write_text("\n".join(lines) + "\n")


# --- Self-check (ponytail: the smallest thing that fails if the check logic breaks) --


def _selftest() -> None:
    assert _first_divergence_and_inversions(["a", "b", "c"], ["a", "c", "b"]) == (1, 1)
    assert _first_divergence_and_inversions(["a", "b"], ["a", "b"]) == (None, 0)

    page = {
        "page_number": 1,
        "regions": [{"id": "r1", "text": "Hello world"}, {"id": "r2", "text": "Hello world"}],
        "arrangement": {
            "version": 1,
            "blocks": [{"kind": "text", "label_ids": [], "region_ids": ["r1"], "box": {"left": 0, "top": 0, "right": 1, "bottom": 1}}],
            "suppressed": [],
            "trimmed_lines": [],
        },
    }
    assert _token_conservation_check(page, "Hello world")["ok"]
    conserved = _token_conservation_check(page, "Hello world world")
    assert conserved["duplicated"] == {"world": 1} and not conserved["ok"]
    dropped = _token_conservation_check(page, "Hello")
    assert dropped["dropped"] == {"world": 1} and not dropped["ok"]

    display_page = {
        "regions": [
            {"id": "s1", "structure": {"child_evidence_ids": ["r1"]}},
            {"id": "s2", "structure": {"child_evidence_ids": ["r2"]}},
        ],
        "text": {"evidence_ids": ["s1", "s2"]},
    }
    arrangement = {
        "blocks": [
            {"kind": "text", "label_ids": [], "region_ids": ["r2"], "box": {}},
            {"kind": "text", "label_ids": [], "region_ids": ["r1"], "box": {}},
        ]
    }
    order_check = _section_order_check(display_page, arrangement)
    assert order_check["first_divergence"] == 0 and order_check["inversions"] == 1, order_check
    print("selftest ok")


# --- CLI --------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", help="OCR service base URL, e.g. http://127.0.0.1:18194")
    parser.add_argument("--out", help="Output directory")
    parser.add_argument("--examples", action="store_true", help="Also process every built-in demo example")
    parser.add_argument("--recheck", metavar="DIR", help="Recompute checks from a previous run's saved process.json files")
    parser.add_argument("--src", default=str(Path(__file__).resolve().parents[1] / "src"), help="ocr_pipeline source root to import (default: this repo's src/)")
    parser.add_argument("--label", help="Suffix for report-<label>.md and order_p<n>-<label>.png in --recheck mode")
    parser.add_argument("--selftest", action="store_true", help="Run the offline logic self-check and exit")
    parser.add_argument("files", nargs="*", help="Files to upload (built-in examples are named via --examples)")
    args = parser.parse_args()

    if args.selftest:
        _selftest()
        return

    demo_module, page_arrangement_module = _load_ocr_modules(Path(args.src))

    if args.recheck:
        out_dir = Path(args.recheck)
        rows: list[dict[str, Any]] = []
        for target in sorted(p for p in out_dir.iterdir() if p.is_dir()):
            page_rows = _recheck_one(target, demo_module, page_arrangement_module, out_dir=out_dir, label=args.label)
            if page_rows:
                rows.extend(page_rows)
        report_name = f"report-{args.label}.md" if args.label else "report.md"
        checks_name = f"checks-{args.label}.json" if args.label else "checks.json"
        _write_report(rows, out_dir / report_name)
        (out_dir / checks_name).write_text(json.dumps(rows, indent=2, ensure_ascii=False))
        print(f"Rechecked {len(rows)} pages under {out_dir} with src={args.src}; wrote {report_name}")
        return

    if not args.url or not args.out:
        parser.error("--url and --out are required unless --recheck is given")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs: list[tuple[str, str | None, Path | None]] = []
    used_names: set[str] = set()
    if args.examples:
        for example_name in [*demo_module.EXAMPLE_FILES, demo_module.TABLE_EXAMPLE_NAME]:
            inputs.append((example_name, example_name, None))
            used_names.add(example_name)
    for raw_path in args.files:
        path = Path(raw_path)
        inputs.append((_input_name(path, used_names), None, path))

    rows = []
    started = time.monotonic()
    for name, example, file_path in inputs:
        print(f"Processing {name} ...", flush=True)
        processed = _process_one(args.url, name, example=example, file_path=file_path, out_dir=out_dir)
        if processed is None or "error" in processed:
            rows.append({"name": name, "page_number": 0, "kind": _kind_of(name), "skipped": processed["error"] if processed else "process failed"})
            continue
        display_pages = {p["page_number"]: p for p in processed["display_json"]["result"]["pages"]}
        rows.extend(
            _check_pages(
                name,
                processed["result_json"]["pages"],
                processed["result_md"],
                display_pages,
                page_arrangement_module,
                out_dir=out_dir,
                label=None,
            )
        )
    elapsed = time.monotonic() - started
    _write_report(rows, out_dir / "report.md")
    (out_dir / "checks.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"Processed {len(inputs)} inputs, {len(rows)} pages in {elapsed:.1f}s; wrote {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
