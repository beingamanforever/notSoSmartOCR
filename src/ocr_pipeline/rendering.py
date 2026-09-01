"""Render page text without losing its evidence links."""

from __future__ import annotations

from .contracts import EvidenceText, TextRegion


def render_evidence(regions: list[TextRegion]) -> EvidenceText:
    rendered = [
        region
        for region in regions
        if region.resolution == "resolved"
        and region.kind != "checkbox"
        and (region.structure or {}).get("role") != "table_source"
    ]
    return EvidenceText(
        value=" ".join(region.text for region in rendered),
        evidence_ids=[region.id for region in rendered],
    )
