"""Build development observations for the Falcon layout-versus-plain router."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from experiments.benchmark_clinical_handwriting import score_snippets
from experiments.serve_falcon_layout import HERON_ROUTE_FEATURE_NAMES


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--layout-package", type=Path, required=True)
    parser.add_argument("--plain-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--unknown-layout-cost-usd", type=float)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")

    rows = build_observations(
        args.features,
        args.layout_package,
        args.plain_run,
        unknown_layout_cost_usd=args.unknown_layout_cost_usd,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    print(
        json.dumps(
            {
                "pages": len(rows),
                "source_groups": len({row["source_group"] for row in rows}),
            }
        )
    )
    return 0


def build_observations(
    features_path: Path,
    layout_package: Path,
    plain_run: Path,
    *,
    unknown_layout_cost_usd: float | None = None,
) -> list[dict]:
    feature_rows = _index_rows(_read_json(features_path), "feature")
    references = _index_rows(
        _read_json(layout_package / "ground-truth" / "references.json"),
        "reference",
    )
    layout_outputs = _text_by_id(layout_package / "saved-responses", "notSoSmartOCR.md")
    plain_records = _records_by_id(plain_run)
    page_ids = set(references)
    _require_same_pages(
        page_ids,
        features=set(feature_rows),
        layout_outputs=set(layout_outputs),
        plain_outputs=set(plain_records),
    )

    layout_costs = _layout_costs(layout_package, references, unknown_layout_cost_usd)
    _validate_source_groups(feature_rows, references)
    feature_names = list(HERON_ROUTE_FEATURE_NAMES)
    rows = []
    for page_id in sorted(page_ids):
        reference = references[page_id]
        features = feature_rows[page_id]
        plain = plain_records[page_id]
        plain_text = plain.get("text")
        if not isinstance(plain_text, str):
            raise ValueError(
                f"Plain record {page_id} needs text, including empty text for failures"
            )
        rows.append(
            {
                "id": page_id,
                "split": "development",
                "source_group": features["group"],
                "feature_values": _feature_values(features, feature_names, page_id),
                "layout_quality": _exact_recovery(
                    layout_outputs[page_id], reference["references"]
                ),
                "plain_quality": _exact_recovery(plain_text, reference["references"]),
                "layout_expert_cost": layout_costs[page_id]["value"],
                "layout_expert_cost_observed": layout_costs[page_id]["observed"],
                "layout_expert_cost_source": layout_costs[page_id]["source"],
                "plain_expert_cost": 0.0,
            }
        )
    return rows


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as error:
        raise ValueError(f"Missing input: {path}") from error


def _index_rows(value, label: str) -> dict[str, dict]:
    if not isinstance(value, list):
        raise ValueError(f"{label.capitalize()} input must be a list")
    rows = {}
    for row in value:
        page_id = row.get("id") if isinstance(row, dict) else None
        if not isinstance(page_id, str) or not page_id:
            raise ValueError(f"Each {label} row needs an id")
        if page_id in rows:
            raise ValueError(f"Duplicate {label} page: {page_id}")
        rows[page_id] = row
    return rows


def _text_by_id(root: Path, filename: str) -> dict[str, str]:
    if not root.is_dir():
        raise ValueError(f"Missing input directory: {root}")
    return {path.parent.name: path.read_text() for path in root.glob(f"*/{filename}")}


def _records_by_id(root: Path) -> dict[str, dict]:
    records = {}
    for path in root.glob("*/record.json"):
        record = _read_json(path)
        page_id = record.get("id") if isinstance(record, dict) else None
        if page_id != path.parent.name:
            raise ValueError(f"Plain record id does not match directory: {path}")
        if page_id in records:
            raise ValueError(f"Duplicate plain page: {page_id}")
        records[page_id] = record
    return records


def _require_same_pages(expected: set[str], **inputs: set[str]) -> None:
    for label, pages in inputs.items():
        if pages != expected:
            missing = sorted(expected - pages)
            extra = sorted(pages - expected)
            raise ValueError(f"{label} page mismatch: missing={missing}, extra={extra}")


def _layout_costs(
    layout_package: Path,
    references: dict[str, dict],
    unknown_cost: float | None,
) -> dict[str, dict]:
    if unknown_cost is not None and (
        type(unknown_cost) not in (int, float)
        or not math.isfinite(unknown_cost)
        or unknown_cost < 0
    ):
        raise ValueError("Unknown layout cost must be finite and nonnegative")
    report = _read_json(layout_package / "evaluation" / "current-report.json")
    pages = report.get("pages") if isinstance(report, dict) else None
    if isinstance(pages, list):
        costs = {}
        for page in pages:
            page_id = page.get("id") if isinstance(page, dict) else None
            pipeline = (page.get("arms") or {}).get("pipeline", {})
            cost = pipeline.get("cost_usd")
            if not isinstance(page_id, str) or not page_id:
                raise ValueError("Each layout report page needs an id")
            if page_id in costs:
                raise ValueError(f"Duplicate layout report page: {page_id}")
            costs[page_id] = _cost_value(cost, unknown_cost, page_id)
        _require_same_pages(set(references), layout_report=set(costs))
        return costs

    records = report.get("records") if isinstance(report, dict) else None
    if not isinstance(records, list):
        raise ValueError("Layout report needs records")
    page_by_source = {}
    for page_id, page in references.items():
        source = (page.get("file"), page.get("page"))
        if not isinstance(source[0], str) or not isinstance(source[1], int):
            raise ValueError(f"Reference {page_id} needs file and integer page")
        if source in page_by_source:
            raise ValueError(f"Duplicate reference source page: {source}")
        page_by_source[source] = page_id

    costs = {}
    unknown_sources = []
    for record in records:
        source = (
            (record.get("file"), record.get("page"))
            if isinstance(record, dict)
            else (None, None)
        )
        page_id = page_by_source.get(source)
        if page_id is None:
            unknown_sources.append(source)
            continue
        cost = record.get("cost_usd")
        if page_id in costs:
            raise ValueError(f"Duplicate layout report page: {page_id}")
        costs[page_id] = _cost_value(cost, unknown_cost, page_id)
    if unknown_sources:
        raise ValueError(f"Layout report has unknown source pages: {unknown_sources}")
    _require_same_pages(set(references), layout_report=set(costs))
    return costs


def _cost_value(cost, unknown_cost: float | None, page_id: str) -> dict:
    if cost is None:
        if unknown_cost is None:
            raise ValueError(
                "Layout report has unknown costs; pass "
                "--unknown-layout-cost-usd explicitly"
            )
        return {
            "value": float(unknown_cost),
            "observed": False,
            "source": "explicit_unknown_cost_bound",
        }
    if type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0:
        raise ValueError(f"Layout record {page_id} needs finite nonnegative cost_usd")
    return {"value": float(cost), "observed": True, "source": "observed"}


def _validate_source_groups(
    features: dict[str, dict], references: dict[str, dict]
) -> None:
    group_to_file = {}
    file_to_group = {}
    for page_id, feature in features.items():
        group = feature.get("group")
        filename = references[page_id].get("file")
        if not isinstance(group, str) or not group:
            raise ValueError(f"Feature row {page_id} needs a source group")
        if not isinstance(filename, str) or not filename:
            raise ValueError(f"Reference {page_id} needs a source document")
        if group in group_to_file and group_to_file[group] != filename:
            raise ValueError(f"Source group {group} spans multiple documents")
        if filename in file_to_group and file_to_group[filename] != group:
            raise ValueError(f"Source document {filename} spans multiple groups")
        group_to_file[group] = filename
        file_to_group[filename] = group


def _feature_values(row: dict, names: list[str], page_id: str) -> dict[str, float]:
    raw_values = row.get("feature_values", row.get("features"))
    if isinstance(raw_values, dict):
        if set(raw_values) != set(names):
            raise ValueError(f"Feature row {page_id} does not match the named schema")
        values = [raw_values[name] for name in names]
    elif isinstance(raw_values, list) and len(raw_values) == len(names):
        values = raw_values
    else:
        raise ValueError(f"Feature row {page_id} does not match the named schema")
    if any(
        type(value) not in (int, float) or not math.isfinite(value) for value in values
    ):
        raise ValueError(f"Feature row {page_id} contains a non-finite value")
    return {name: float(value) for name, value in zip(names, values, strict=True)}


def _exact_recovery(text: str, references: list[dict]) -> int:
    if not isinstance(references, list):
        raise ValueError("Page references must be a list")
    return sum(score["recovered"] is True for score in score_snippets(text, references))


if __name__ == "__main__":
    raise SystemExit(main())
