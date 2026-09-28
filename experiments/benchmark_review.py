"""Collect fresh demo responses and build a private, offline visual review.

Run with PYTHONPATH=src:. This collects evidence; it does not assign accuracy scores.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import shutil
import time
from urllib.parse import quote, urljoin, urlsplit

import bleach
from markdown_it import MarkdownIt
from PIL import Image
import requests

from ocr_pipeline.pipeline import IMAGE_SUFFIXES
from ocr_pipeline.rendering import render_page_markdown

MARKDOWN = MarkdownIt("commonmark", {"html": True}).enable("table")
TAGS = {
    "img",
    "p",
    "br",
    "pre",
    "code",
    "strong",
    "em",
    "ul",
    "ol",
    "li",
    "blockquote",
    "h1",
    "h2",
    "h3",
    "h4",
    "hr",
    "table",
    "thead",
    "tbody",
    "tr",
    "th",
    "td",
}
STYLE = """
body{margin:0;background:#f4f6f8;color:#17232e;font:16px/1.5 system-ui,sans-serif}
main{max-width:1500px;margin:auto;padding:24px}a{color:#145a91}h1{font-size:28px}
article,header{background:white;border:1px solid #d5dde4;border-radius:8px;padding:20px;margin:16px 0}
.columns{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:24px}
.source{position:relative;align-self:start}.source img{display:block;width:100%;height:auto}
.source svg{position:absolute;inset:0;width:100%;height:100%}.source rect{fill:#2375ae18;stroke:#2375ae;stroke-width:2;vector-effect:non-scaling-stroke}
.source rect.review{fill:#bb651c18;stroke:#bb651c}.page input:not(:checked)~.columns svg{display:none}
.output{overflow-wrap:anywhere;overflow:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}
.output img{max-width:100%;height:auto}
table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px;border:1px solid #d5dde4;text-align:left}
fieldset{margin:16px 0}fieldset label{display:inline-block;margin-right:16px}
main:has(#task-forms:checked) tr[data-tasks]:not([data-tasks~="forms"]),
main:has(#task-handwriting:checked) tr[data-tasks]:not([data-tasks~="handwriting"]),
main:has(#task-tables:checked) tr[data-tasks]:not([data-tasks~="tables"]){display:none}
summary{cursor:pointer;font-weight:600}details{margin:12px 0}.muted{color:#526372}.error{color:#a32d21}
@media(max-width:850px){.columns{grid-template-columns:1fr}main{padding:12px}}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("url")
    run.add_argument("inputs", type=Path)
    run.add_argument("output", type=Path)
    render = commands.add_parser("render")
    render.add_argument("output", type=Path)
    prepare = commands.add_parser(
        "prepare", help="Collect unique canonical benchmark pages"
    )
    prepare.add_argument("data", type=Path)
    prepare.add_argument("inputs", type=Path)
    score = commands.add_parser(
        "score", help="Score a completed canonical page collection"
    )
    score.add_argument("data", type=Path)
    score.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "run":
        collect(args.url, args.inputs, args.output)
    elif args.command == "prepare":
        prepare_inputs(args.data, args.inputs)
    elif args.command == "score":
        score_review(args.data, args.output)
    else:
        build_review(args.output)


def case_pages(data: Path):
    for path in sorted(data.rglob("ground_truth.json")):
        case = json.loads(path.read_text())
        if case.get("availability", "ready") == "ready":
            yield (
                case,
                {
                    image["page"]: path.parent / image["file"]
                    for image in case["images"]
                },
            )


def page_name(case: dict, number: int, path: Path) -> str:
    return f"{case['source_reference']}-p{number:04d}{path.suffix.lower()}"


def prepare_inputs(data: Path, inputs: Path) -> None:
    inputs.mkdir(parents=True, exist_ok=False)
    for case, pages in case_pages(data):
        for number, source in pages.items():
            destination = inputs / page_name(case, number, source)
            if destination.exists():
                if destination.read_bytes() != source.read_bytes():
                    raise ValueError("Conflicting copies of the same source page")
                continue
            shutil.copyfile(source, destination)


def score_review(data: Path, output: Path) -> dict:
    # Reuse the exact historical adapter; add the benchmark root to PYTHONPATH.
    from clinical_ocr_benchmark import run_benchmark
    from tools.import_baselines import adapt_case

    report = json.loads((output / "report.json").read_text())
    if report["status"] != "complete":
        raise ValueError("Only a completed collection can be scored")
    predictions = output / "predictions"
    documents = {item["source"]: item for item in report["documents"]}
    for case, sources in case_pages(data):
        for number, source in sources.items():
            document = documents.get(page_name(case, number, source))
            if document is None:
                continue
            captured = output / document["id"] / ("input" + source.suffix.lower())
            if captured.read_bytes() != source.read_bytes():
                raise ValueError("Captured input differs from current benchmark page")
            document["tasks"] = sorted(set(document.get("tasks", [])) | {case["task"]})
    predictions.mkdir(exist_ok=False)
    counts = Counter()
    for case, sources in case_pages(data):
        pages = {}
        for number, source in sources.items():
            document = documents.get(page_name(case, number, source))
            if document is None or document["status"] not in {"success", "partial"}:
                break
            pages[number] = (output / document["id"] / "result.md").read_text()
        if len(pages) != len(sources):
            counts["cases_missing_processed_pages"] += 1
            continue
        prediction, observations = adapt_case(case, pages)
        counts.update(observations)
        write_json(predictions / f"{case['case_id']}.json", prediction)
    summary = run_benchmark(data, predictions, output / "scores")
    report["scores"] = summary
    report["adapter_counts"] = dict(counts)
    write_json(output / "report.json", report)
    build_review(output)
    return summary


def collect(url: str, inputs: Path, output: Path) -> dict:
    sources = sorted(
        p
        for p in inputs.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES | {".pdf"}
    )
    if not sources:
        raise ValueError("No PDF or image inputs found")
    if output.exists():
        raise FileExistsError(f"Use a new run directory: {output}")
    url = url.rstrip("/")
    with requests.Session() as client:
        # Large evidence downloads can outlive the service's idle keep-alive timeout.
        client.headers["Connection"] = "close"
        # Fail before creating a run if the intended service is unavailable.
        reply = client.get(f"{url}/api/composition", timeout=30)
        reply.raise_for_status()
        composition = reply.json()
        output.mkdir(parents=True)
        write_json(output / "composition.json", composition)
        report = {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "status": "running",
            "scores": None,
            "documents": [],
        }
        # Reserve every source up front so an interrupted run keeps its denominator.
        for index, source in enumerate(sources, 1):
            report["documents"].append(
                {
                    "id": f"doc-{index:04d}",
                    "source": source.name,
                    "status": "pending",
                    "errors": [],
                }
            )
        write_json(output / "report.json", report)
        try:
            for source, document in zip(sources, report["documents"], strict=True):
                destination = output / document["id"]
                destination.mkdir()
                shutil.copyfile(source, destination / ("input" + source.suffix.lower()))
                started = time.perf_counter()
                try:
                    collect_document(client, url, source, destination, document)
                except (
                    requests.RequestException,
                    OSError,
                    ValueError,
                    KeyError,
                    TypeError,
                ) as error:
                    document["status"] = "failed"
                    document["errors"].append(f"{type(error).__name__}: {error}")
                document["wall_seconds"] = time.perf_counter() - started
                write_json(output / "report.json", report)
                print(document["id"], document["status"], flush=True)
            reply = client.get(f"{url}/api/composition", timeout=30)
            reply.raise_for_status()
            final_composition = reply.json()
            write_json(output / "composition-after.json", final_composition)
            if final_composition != composition:
                report["status"] = "configuration_changed"
                raise ValueError(
                    "Service composition changed during collection; do not compare this run"
                )
            report["status"] = "complete"
        finally:
            if report["status"] == "running":
                report["status"] = "interrupted"
            write_json(output / "report.json", report)
            build_review(output)
    return report


def collect_document(
    client, url: str, source: Path, destination: Path, document: dict
) -> None:
    with source.open("rb") as stream:
        reply = client.post(
            f"{url}/api/process", files={"file": (source.name, stream)}, timeout=7200
        )
    reply.raise_for_status()
    body = reply.json()
    write_json(destination / "response.json", body)
    result = body["result"]
    pages = result["pages"]
    document["status"] = result["status"]
    document["review_pages"] = sum(page.get("route") == "review" for page in pages)
    document["pages"] = len(pages)
    session = quote(str(body["session_id"]), safe="")
    reply = client.get(f"{url}/api/sessions/{session}/result.md", timeout=120)
    reply.raise_for_status()
    (destination / "result.md").write_text(reply.text, encoding="utf-8")
    images = body["page_images"]
    if len(images) != len(pages):
        raise ValueError("Preview count differs from result pages")
    for page, image_url in zip(pages, images, strict=True):
        asset_url = urljoin(url + "/", image_url)
        if (
            urlsplit(asset_url).netloc != urlsplit(url).netloc
            or urlsplit(asset_url).scheme != urlsplit(url).scheme
        ):
            raise ValueError("Preview points outside the OCR service")
        reply = client.get(asset_url, timeout=120)
        reply.raise_for_status()
        path = destination / f"page-{int(page['page_number']):04d}.png"
        path.write_bytes(reply.content)
        with Image.open(path) as image:
            if image.size != (page["width"], page["height"]):
                raise ValueError(
                    "Preview dimensions differ from region coordinate space"
                )
            image.verify()
    # Preserve the live session when evidence download failed, for recovery.
    reply = client.delete(f"{url}/api/sessions/{session}", timeout=60)
    if reply.status_code not in {200, 204}:
        document["errors"].append(f"Session cleanup returned HTTP {reply.status_code}")


def build_review(output: Path) -> None:
    report = json.loads((output / "report.json").read_text())
    rows = []
    for document in report["documents"]:
        identifier = document["id"]
        if Path(identifier).name != identifier or identifier in {".", ".."}:
            raise ValueError("Invalid document directory")
        directory = output / identifier
        errors = "".join(
            f'<p class="error">{escape(error)}</p>' for error in document["errors"]
        )
        sections = []
        response_path = directory / "response.json"
        if response_path.exists():
            body = json.loads(response_path.read_text())
            for page in body.get("result", {}).get("pages", []):
                sections.append(render_page(page, directory))
            sections.append(
                f"<details><summary>Stage timings and review evidence</summary><pre>{escape(json.dumps({k: body.get(k) for k in ('timing', 'stage_execution', 'page_execution', 'coverage_assessment', 'uncertainty')}, indent=2))}</pre></details>"
            )
        links = " ".join(
            f'<a href="{quote(p.name)}">{escape(p.name)}</a>'
            for p in sorted(directory.glob("*"))
            if p.is_file()
            and (p.name.startswith("input.") or p.suffix in {".json", ".md"})
        )
        if directory.exists():
            content = (
                f'<a href="../index.html">All documents</a><h1>{escape(document["source"])}</h1><p>Status: {escape(document["status"])} | Accuracy: not scored</p><p>{links}</p>{errors}'
                + "".join(sections)
            )
            (directory / "index.html").write_text(
                html_page(document["source"], content), encoding="utf-8"
            )
        title = (
            f'<a href="{identifier}/index.html">{escape(document["source"])}</a>'
            if directory.exists()
            else escape(document["source"])
        )
        rows.append(
            f'<tr data-tasks="{escape(" ".join(document.get("tasks", [])))}"><td>{title}</td><td>{escape(", ".join(document.get("tasks", [])))}</td><td>{escape(document["status"])}</td><td>{document.get("pages", "-")}</td><td>{document.get("review_pages", "-")}</td><td>{document.get("wall_seconds", 0):.1f}</td></tr>'
        )
    scores = report.get("scores")
    score_section = "<p>Accuracy scores have not been computed.</p>"
    if scores is not None:
        task_rows = "".join(
            f"<tr><th>{escape(task)}</th><td>{values['coverage_adjusted']:.1%}</td><td>{values['submitted']} / {values['eligible']}</td><td>{values['missing']}</td><td>{values['metrics'].get('missing_units', 0)}</td></tr>"
            for task, values in scores["tasks"].items()
        )
        score_section = (
            f"<h2>Benchmark scores</h2><p>{scores['data_readiness']['ready']} ready cases; {scores['data_readiness']['unavailable']} unavailable.</p>"
            f"<table><tr><th>Task</th><th>Coverage-adjusted score</th><th>Submitted / ready</th><th>Missing cases</th><th>Missing reference units</th></tr>{task_rows}</table>"
            "<p>Forms measures annotated checkbox states. Handwriting averages character and word accuracy on annotated sections. Tables combines cell content, position, row, column and title scores; it is not TEDS. Scores average cases equally, not pages.</p>"
            "<p>Scores use the existing target-aligned adapter. Handwriting measures annotated sections; tables require matching table counts and positional association. These are regression results, not an untouched final evaluation.</p>"
            '<a href="scores/report.md">Metric definitions and results</a> · <a href="scores/summary.json">Detailed scores</a>'
        )
    content = f"""<header><h1>OCR visual review</h1><p>Run: {escape(report["started_at"])} | {escape(report["status"])}</p>
<p>{len(report["documents"])} input documents, including failed and pending documents.</p>{score_section}
<p class="muted">Private document content. Open index.html after extracting the complete folder. No server or internet connection is needed. Blue boxes show extracted regions; orange boxes need review. Confidence values retain their recorded meaning.</p>
<a href="report.json">Run results</a> · <a href="composition.json">Service composition</a></header>
<fieldset {"hidden" if scores is None else ""}><legend>Show task</legend><label><input type="radio" name="task" checked> All documents</label>{"".join(f'<label><input type="radio" name="task" id="task-{task}"> {task}</label>' for task in ("forms", "handwriting", "tables"))}</fieldset>
<table><thead><tr><th>Document</th><th>Tasks</th><th>Status</th><th>Pages</th><th>Review pages</th><th>Seconds</th></tr></thead><tbody>{"".join(rows)}</tbody></table>"""
    (output / "index.html").write_text(
        html_page("OCR visual review", content), encoding="utf-8"
    )


def render_page(page: dict, directory: Path) -> str:
    number = int(page["page_number"])
    image = f"page-{number:04d}.png"
    regions = page.get("regions", [])
    evidence_ids = page.get("text", {}).get("evidence_ids", [])
    markdown = render_page_markdown(
        regions, evidence_ids, fallback=page.get("text", {}).get("value", "")
    )
    rendered = bleach.clean(
        MARKDOWN.render(markdown),
        tags=TAGS,
        attributes=review_attribute,
        protocols={"data"},
        strip=True,
    )
    boxes = []
    for region in regions:
        box = region.get("bounding_box")
        if not box:
            continue
        title = f"{region['id']}: {region['kind']}; {region.get('resolution')}; confidence={region.get('confidence')}; {(region.get('text_provenance') or {}).get('confidence_meaning', 'unspecified')}"
        state = "review" if region.get("resolution") != "resolved" else ""
        boxes.append(
            f'<rect class="{state}" x="{float(box["left"])}" y="{float(box["top"])}" width="{max(0, float(box["right"]) - float(box["left"]))}" height="{max(0, float(box["bottom"]) - float(box["top"]))}"><title>{escape(title)}</title></rect>'
        )
    source = (
        f'<div class="source"><a href="{image}"><img loading="lazy" src="{image}" alt="Source page {number}"></a><svg viewBox="0 0 {int(page["width"])} {int(page["height"])}" aria-label="Detected regions">{"".join(boxes)}</svg></div>'
        if (directory / image).exists()
        else '<p class="error">Source preview was not captured.</p>'
    )
    return f"""<article class="page"><h2>Page {number}</h2><p>Route: {escape(page.get("route", "unknown"))}</p>
<input type="checkbox" id="boxes-{number}" checked><label for="boxes-{number}"> Show region boxes (hover for confidence)</label>
<div class="columns">{source}<div class="output">{rendered or "<p>No canonical text.</p>"}<details><summary>Raw page evidence</summary><pre>{escape(json.dumps(page, ensure_ascii=False, indent=2))}</pre></details></div></div></article>"""


def review_attribute(tag: str, name: str, value: str) -> bool:
    if tag == "img":
        return name == "alt" or (
            name == "src" and value.startswith("data:image/png;base64,")
        )
    return tag in {"td", "th"} and name in {"rowspan", "colspan"}


def html_page(title: str, content: str) -> str:
    return f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="Content-Security-Policy" content="default-src \'none\'; img-src \'self\' data:; style-src \'unsafe-inline\'"><title>{escape(title)}</title><style>{STYLE}</style><main>{content}</main></html>'


def write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
