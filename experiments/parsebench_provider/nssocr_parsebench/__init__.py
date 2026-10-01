"""ParseBench provider for the Not So Smart OCR service (Falcon OCR with Qwen routing).

Each PDF goes to the service's /api/process unchanged; the page Markdown is the service's
own export (page bodies only), and every evidence region becomes a layout item in page
pixel coordinates, mapped from the Heron layout labels the service uses to ParseBench's
canonical labels (also Heron's 17-class schema). Modelled on ParseBench's Falcon-OCR
provider so the two are scored the same way.
"""

from __future__ import annotations

import html
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

from parse_bench.inference.pipelines import register_pipeline
from parse_bench.inference.providers.base import Provider, ProviderPermanentError, ProviderTransientError
from parse_bench.inference.providers.parse.falconocr import FalconOcrProvider
from parse_bench.inference.providers.registry import register_provider
from parse_bench.schemas.layout_ontology import CanonicalLabel
from parse_bench.schemas.parse_output import LayoutItemIR, LayoutSegmentIR, ParseLayoutPageIR, ParseOutput
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import InferenceRequest, InferenceResult, RawInferenceResult
from parse_bench.schemas.product import ProductType

LABELS = {
    "text": CanonicalLabel.TEXT,
    "handwriting": CanonicalLabel.TEXT,
    "paragraph": CanonicalLabel.TEXT,
    "title": CanonicalLabel.TITLE,
    "section-header": CanonicalLabel.SECTION_HEADER,
    "page-header": CanonicalLabel.PAGE_HEADER,
    "page-footer": CanonicalLabel.PAGE_FOOTER,
    "table": CanonicalLabel.TABLE,
    "figure": CanonicalLabel.PICTURE,
    "picture": CanonicalLabel.PICTURE,
    "caption": CanonicalLabel.CAPTION,
    "footnote": CanonicalLabel.FOOTNOTE,
    "formula": CanonicalLabel.FORMULA,
    "list-item": CanonicalLabel.LIST_ITEM,
    "code": CanonicalLabel.CODE,
    "form": CanonicalLabel.FORM,
    "key-value-region": CanonicalLabel.KEY_VALUE_REGION,
    "key_value_region": CanonicalLabel.KEY_VALUE_REGION,
    "document-index": CanonicalLabel.DOCUMENT_INDEX,
}
# The layout adapter reads segment labels in LlamaParse's v3 raw ontology: Heron's names, lower case.
RAW_LABEL = {
    label: {
        CanonicalLabel.KEY_VALUE_REGION: "key-value-region",
        CanonicalLabel.DOCUMENT_INDEX: "document-index",
    }.get(label, label.value.lower())
    for label in CanonicalLabel
}
# Evidence the service keeps for review, not page content.
AUXILIARY = {"layout_block", "coverage_risk", "word", "table_cell", "table_candidate"}
EMBEDDED_IMAGE = re.compile(r"!\[([^\]]*)\]\(data:image/[^)]*\)")


def scoring_markdown(pages: list[str]) -> str:
    """Pipe tables as HTML with quoted attributes: ParseBench's table metrics read HTML only.

    The same two steps ParseBench's own Falcon-OCR provider applies, reused from it.
    """
    markdown = FalconOcrProvider._convert_md_tables_to_html("\n\n".join(pages))
    return FalconOcrProvider._sanitize_html_attributes(markdown)


def page_bodies(markdown: str) -> list[str]:
    """The export's page sections without its metadata header, page headings or failure list."""
    pages, body = [], None
    for line in markdown.splitlines():
        if re.fullmatch(r"## Page \d+", line):
            body = []
            pages.append(body)
        elif line == "## Failures":
            body = None
        elif body is not None:
            body.append(line)
    return [EMBEDDED_IMAGE.sub(r"![\1]()", "\n".join(lines)).strip() for lines in pages]


def table_html(structure: dict[str, Any]) -> str:
    cells = structure.get("cells") or []
    if not cells:
        return ""
    rows: dict[int, list[dict[str, Any]]] = {}
    for cell in cells:
        if cell.get("row_nums") and cell.get("column_nums"):
            rows.setdefault(min(cell["row_nums"]), []).append(cell)
    out = ["<table>"]
    for row in sorted(rows):
        out.append("<tr>")
        for cell in sorted(rows[row], key=lambda c: min(c["column_nums"])):
            span = "".join(
                f' {name}="{len(cell[key])}"'
                for name, key in (("rowspan", "row_nums"), ("colspan", "column_nums"))
                if len(cell[key]) > 1
            )
            out.append(f"<td{span}>{html.escape(str(cell.get('text') or ''))}</td>")
        out.append("</tr>")
    return "".join(out + ["</table>"])


def layout_items(page: dict[str, Any]) -> list[LayoutItemIR]:
    items = []
    for region in page.get("regions", []):
        kind = region.get("kind")
        if kind in AUXILIARY:
            continue
        if kind == "checkbox":
            selected = (region.get("structure") or {}).get("state") == "selected"
            label = CanonicalLabel.CHECKBOX_SELECTED if selected else CanonicalLabel.CHECKBOX_UNSELECTED
        else:
            label = LABELS.get(kind, CanonicalLabel.TEXT)
        box = region.get("bounding_box") or {}
        try:
            left, top, right, bottom = (float(box[key]) for key in ("left", "top", "right", "bottom"))
        except (KeyError, TypeError, ValueError):
            continue
        segment = LayoutSegmentIR(
            x=left, y=top, w=max(0.0, right - left), h=max(0.0, bottom - top),
            confidence=1.0, label=RAW_LABEL[label],
        )
        text = str(region.get("text") or "") if region.get("resolution", "resolved") == "resolved" else ""
        if label == CanonicalLabel.TABLE:
            item = LayoutItemIR(type="table", html=table_html(region.get("structure") or {}), bbox=segment, layout_segments=[segment])
        elif label == CanonicalLabel.PICTURE:
            item = LayoutItemIR(type="image", bbox=segment, layout_segments=[segment])
        else:
            item = LayoutItemIR(type="text", md=text, value=text, bbox=segment, layout_segments=[segment])
        items.append(item)
    return items


@register_provider("nssocr")
class NotSoSmartOcrProvider(Provider):
    """Config: server_url (or NSSOCR_SERVER_URL), timeout seconds (default 1800)."""

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)
        self._url = str(self.base_config.get("server_url") or os.environ.get("NSSOCR_SERVER_URL") or "").rstrip("/")
        if not self._url:
            raise ProviderPermanentError("nssocr needs server_url or NSSOCR_SERVER_URL")
        self._timeout = int(self.base_config.get("timeout", 1800))

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.PARSE:
            raise ProviderPermanentError(f"nssocr supports PARSE only, got {request.product_type}")
        source = Path(request.source_file_path)
        started = datetime.now()
        try:
            with source.open("rb") as handle:
                reply = requests.post(f"{self._url}/api/process", files={"file": (source.name, handle)}, timeout=self._timeout)
            if reply.status_code in (429, 502, 503, 504):
                raise ProviderTransientError(f"HTTP {reply.status_code}: {reply.text[:200]}")
            if reply.status_code != 200:
                raise ProviderPermanentError(f"HTTP {reply.status_code}: {reply.text[:200]}")
            data = reply.json()
            session = data["session_id"]
            markdown = requests.get(f"{self._url}/api/sessions/{session}/result.md", timeout=300).text
            requests.delete(f"{self._url}/api/sessions/{session}", timeout=60)
        except requests.RequestException as error:
            raise ProviderTransientError(str(error)) from error
        completed = datetime.now()
        result = data["result"]
        raw = {
            "markdown_pages": page_bodies(markdown),
            # Not "pages": ParseBench's layout adapter reads a raw "pages" list as LlamaParse's own.
            "nssocr_pages": [
                {"page_number": p["page_number"], "width": p["width"], "height": p["height"],
                 "regions": [{k: r.get(k) for k in ("kind", "text", "bounding_box", "resolution", "structure")} for r in p["regions"]]}
                for p in result["pages"]
            ],
            "status": result.get("status"),
            "failures": result.get("failures"),
        }
        return RawInferenceResult(
            request=request, pipeline=pipeline, pipeline_name=pipeline.pipeline_name,
            product_type=request.product_type, raw_output=raw, started_at=started, completed_at=completed,
            latency_in_ms=int((completed - started).total_seconds() * 1000),
        )

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        raw = raw_result.raw_output
        layout_pages = [
            ParseLayoutPageIR(page_number=p["page_number"], width=float(p["width"]), height=float(p["height"]), items=items)
            for p in raw.get("nssocr_pages", [])
            if (items := layout_items(p))
        ]
        output = ParseOutput(
            task_type="parse", example_id=raw_result.request.example_id, pipeline_name=raw_result.pipeline_name,
            pages=[], markdown=scoring_markdown(raw.get("markdown_pages", [])), layout_pages=layout_pages,
        )
        return InferenceResult(
            request=raw_result.request, pipeline_name=raw_result.pipeline_name, product_type=raw_result.product_type,
            raw_output=raw, output=output, started_at=raw_result.started_at, completed_at=raw_result.completed_at,
            latency_in_ms=raw_result.latency_in_ms,
        )


register_pipeline(PipelineSpec(pipeline_name="nssocr", provider_name="nssocr", product_type=ProductType.PARSE, config={}))
