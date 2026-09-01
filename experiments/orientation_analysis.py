"""Analyze cached automatic and four-angle OCR orientation results."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Mapping, Sequence, cast

ANGLES = ("0", "90", "180", "270")
METRICS = ("cer", "wer")
DEFAULT_CATASTROPHIC_REGRET_THRESHOLD = 0.2
DEFAULT_LOW_MARGIN_THRESHOLD = 0.01
ORACLE_LABEL = (
    "Non-deployable diagnostic oracle selected with reference CER. "
    "It is not a production orientation policy."
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze cached automatic and four-angle orientation results"
    )
    parser.add_argument("automatic", type=Path, help="Automatic orientation JSON")
    parser.add_argument("four_angle", type=Path, help="Four-angle orientation JSON")
    parser.add_argument("output", type=Path, help="Analysis JSON path")
    parser.add_argument(
        "--catastrophic-regret-threshold",
        type=float,
        default=DEFAULT_CATASTROPHIC_REGRET_THRESHOLD,
    )
    parser.add_argument(
        "--low-margin-threshold",
        type=float,
        default=DEFAULT_LOW_MARGIN_THRESHOLD,
    )
    args = parser.parse_args(argv)

    try:
        report = analyze_files(
            args.automatic,
            args.four_angle,
            catastrophic_regret_threshold=args.catastrophic_regret_threshold,
            low_margin_threshold=args.low_margin_threshold,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    return 0


def analyze_files(
    automatic_path: Path,
    four_angle_path: Path,
    *,
    catastrophic_regret_threshold: float = DEFAULT_CATASTROPHIC_REGRET_THRESHOLD,
    low_margin_threshold: float = DEFAULT_LOW_MARGIN_THRESHOLD,
) -> dict[str, object]:
    automatic_payload = json.loads(automatic_path.read_text(encoding="utf-8"))
    four_angle_payload = json.loads(four_angle_path.read_text(encoding="utf-8"))
    return {
        "sources": {
            "automatic": str(automatic_path),
            "four_angle": str(four_angle_path),
        },
        **analyze_payloads(
            automatic_payload,
            four_angle_payload,
            catastrophic_regret_threshold=catastrophic_regret_threshold,
            low_margin_threshold=low_margin_threshold,
        ),
    }


def analyze_payloads(
    automatic_payload: object,
    four_angle_payload: object,
    *,
    catastrophic_regret_threshold: float = DEFAULT_CATASTROPHIC_REGRET_THRESHOLD,
    low_margin_threshold: float = DEFAULT_LOW_MARGIN_THRESHOLD,
) -> dict[str, object]:
    _validate_threshold(catastrophic_regret_threshold, "catastrophic regret")
    _validate_threshold(low_margin_threshold, "low margin")
    automatic_run, angle_runs = _runs(automatic_payload, four_angle_payload)
    automatic_cases = _cases(automatic_run, "automatic")
    cases_by_angle = {
        angle: _cases(angle_runs[angle], f"angle {angle}") for angle in ANGLES
    }
    normalization = _validate_inputs(
        automatic_run,
        angle_runs,
        automatic_cases,
        cases_by_angle,
    )

    pages = []
    oracle_cases = []
    exact_count = 0
    oracle_angle_counts: Counter[str] = Counter()
    for case_id in sorted(automatic_cases):
        automatic_case = automatic_cases[case_id]
        angle_cases = {angle: cases_by_angle[angle][case_id] for angle in ANGLES}
        oracle_angles = _oracle_angles(angle_cases)
        selected_oracle_angle = oracle_angles[0]
        oracle_case = angle_cases[selected_oracle_angle]
        automatic_angle, score_margin = _selection(automatic_case, case_id)
        exact = automatic_angle in oracle_angles
        exact_count += int(exact)
        oracle_angle_counts.update(oracle_angles)
        automatic_cer = _metric_rate(automatic_case, "cer", "automatic", case_id)
        oracle_cer = _metric_rate(
            oracle_case,
            "cer",
            f"angle {selected_oracle_angle}",
            case_id,
        )
        regret = automatic_cer - oracle_cer
        cluster_id = str(automatic_case["cluster_id"])
        pages.append(
            {
                "id": case_id,
                "template": cluster_id,
                "automatic_angle": automatic_angle,
                "oracle_angles": list(oracle_angles),
                "selected_oracle_angle": selected_oracle_angle,
                "exact_oracle_angle_selection": exact,
                "automatic_cer": _round(automatic_cer),
                "oracle_cer": _round(oracle_cer),
                "cer_regret": _round(regret),
                "score_margin": _round(score_margin),
                "low_margin": score_margin <= low_margin_threshold,
                "catastrophic_regret": regret >= catastrophic_regret_threshold,
            }
        )
        oracle_cases.append(oracle_case)

    templates: dict[str, list[dict[str, object]]] = defaultdict(list)
    for page in pages:
        templates[str(page["template"])].append(page)

    return {
        "analysis": "cached_orientation_oracle_analysis",
        "normalization": normalization,
        "case_count": len(pages),
        "oracle": {
            "label": ORACLE_LABEL,
            "deployable": False,
            "selection": "lowest per-page CER",
            "tie_breaker_for_aggregate_metrics": "lowest numeric angle",
        },
        "aggregate": {
            "automatic": _aggregate(list(automatic_cases.values()), "automatic"),
            "oracle": _aggregate(oracle_cases, "oracle"),
        },
        "angle_selection": {
            "exact_oracle_angle_selection_count": exact_count,
            "exact_oracle_angle_selection_rate": _round(exact_count / len(pages)),
            "ties_accepted": True,
            "oracle_angle_counts": {
                angle: oracle_angle_counts[angle] for angle in ANGLES
            },
            "pages_with_oracle_ties": sum(
                len(page["oracle_angles"]) > 1 for page in pages
            ),
        },
        "cer_regret": {
            "definition": "automatic CER minus four-angle oracle CER",
            **_distribution([float(page["cer_regret"]) for page in pages]),
            "catastrophic_threshold": catastrophic_regret_threshold,
            "catastrophic_count": sum(
                bool(page["catastrophic_regret"]) for page in pages
            ),
        },
        "low_margin_association": {
            "definition": "automatic selector score_margin at or below threshold",
            "threshold": low_margin_threshold,
            "low_margin": _group_summary(
                [page for page in pages if bool(page["low_margin"])]
            ),
            "higher_margin": _group_summary(
                [page for page in pages if not bool(page["low_margin"])]
            ),
        },
        "templates": {
            template: _template_summary(group, automatic_cases, oracle_cases)
            for template, group in sorted(templates.items(), key=_template_sort_key)
        },
        "pages": pages,
    }


def _runs(
    automatic_payload: object,
    four_angle_payload: object,
) -> tuple[Mapping[str, object], Mapping[str, Mapping[str, object]]]:
    automatic_runs = _run_mapping(automatic_payload, "automatic")
    four_angle_runs = _run_mapping(four_angle_payload, "four-angle")
    if set(automatic_runs) != {"auto"}:
        raise ValueError("Automatic JSON must contain only the 'auto' run")
    if set(four_angle_runs) != set(ANGLES):
        raise ValueError(f"Four-angle JSON must contain runs {list(ANGLES)}")
    automatic_run = automatic_runs["auto"]
    if not isinstance(automatic_run, dict):
        raise ValueError("Automatic run must be an object")
    if not all(isinstance(four_angle_runs[angle], dict) for angle in ANGLES):
        raise ValueError("Every angle run must be an object")
    return automatic_run, cast(Mapping[str, Mapping[str, object]], four_angle_runs)


def _run_mapping(payload: object, label: str) -> Mapping[str, object]:
    if not isinstance(payload, dict) or not isinstance(payload.get("runs"), dict):
        raise ValueError(f"{label} JSON must contain a runs object")
    return payload["runs"]


def _cases(run: Mapping[str, object], label: str) -> dict[str, Mapping[str, object]]:
    raw_cases = run.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError(f"{label} run must contain a non-empty cases array")
    cases = {}
    for index, case in enumerate(raw_cases):
        if not isinstance(case, dict):
            raise ValueError(f"{label} case at index {index} must be an object")
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"{label} case at index {index} has no valid id")
        if case_id in cases:
            raise ValueError(f"Duplicate {label} case id: {case_id}")
        if not isinstance(case.get("reference"), str):
            raise ValueError(f"{label} case {case_id!r} has no valid reference")
        if not isinstance(case.get("cluster_id"), (str, int)):
            raise ValueError(f"{label} case {case_id!r} has no valid cluster_id")
        for metric in METRICS:
            _metric_rate(case, metric, label, case_id)
        cases[case_id] = case
    return cases


def _validate_inputs(
    automatic_run: Mapping[str, object],
    angle_runs: Mapping[str, Mapping[str, object]],
    automatic_cases: Mapping[str, Mapping[str, object]],
    cases_by_angle: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> str:
    normalization = automatic_run.get("normalization")
    if not isinstance(normalization, str) or not normalization:
        raise ValueError("Automatic run has no valid normalization")
    for angle in ANGLES:
        if angle_runs[angle].get("normalization") != normalization:
            raise ValueError(f"Normalization differs for angle {angle}")
        if set(cases_by_angle[angle]) != set(automatic_cases):
            raise ValueError(f"Case IDs differ for angle {angle}")

    for case_id, automatic_case in automatic_cases.items():
        for angle in ANGLES:
            angle_case = cases_by_angle[angle][case_id]
            if angle_case["reference"] != automatic_case["reference"]:
                raise ValueError(
                    f"References differ for case {case_id!r} at angle {angle}"
                )
            if str(angle_case["cluster_id"]) != str(automatic_case["cluster_id"]):
                raise ValueError(
                    f"Template IDs differ for case {case_id!r} at angle {angle}"
                )
            for metric in METRICS:
                automatic_units = _metric_counts(automatic_case, metric)[1]
                angle_units = _metric_counts(angle_case, metric)[1]
                if (
                    automatic_units is not None
                    and angle_units is not None
                    and automatic_units != angle_units
                ):
                    raise ValueError(
                        f"{metric.upper()} reference denominator differs for case "
                        f"{case_id!r} at angle {angle}"
                    )
    return normalization


def _selection(case: Mapping[str, object], case_id: str) -> tuple[str, float]:
    selection = case.get("orientation_selection")
    if not isinstance(selection, dict):
        raise ValueError(f"Automatic case {case_id!r} has no orientation selection")
    angle = str(selection.get("angle"))
    if angle not in ANGLES:
        raise ValueError(f"Automatic case {case_id!r} has invalid selected angle")
    margin = selection.get("score_margin")
    if not _is_number(margin) or not math.isfinite(float(margin)) or margin < 0:
        raise ValueError(f"Automatic case {case_id!r} has invalid score margin")
    return angle, float(margin)


def _oracle_angles(
    angle_cases: Mapping[str, Mapping[str, object]],
) -> tuple[str, ...]:
    scores = {
        angle: _metric_fraction_or_rate(case, "cer")
        for angle, case in angle_cases.items()
    }
    best = min(scores.values())
    return tuple(angle for angle in ANGLES if scores[angle] == best)


def _metric_fraction_or_rate(
    case: Mapping[str, object], metric: str
) -> Fraction | float:
    edits, units = _metric_counts(case, metric)
    if edits is not None and units is not None:
        return Fraction(edits, units)
    return _metric_rate(case, metric, "case", str(case.get("id")))


def _metric_rate(
    case: Mapping[str, object], metric: str, label: str, case_id: str
) -> float:
    metrics = case.get("metrics")
    value = metrics.get(metric) if isinstance(metrics, dict) else None
    if not isinstance(value, dict):
        raise ValueError(f"{label} case {case_id!r} has no {metric.upper()} metrics")
    edits, units = _metric_counts(case, metric)
    if edits is not None and units is not None:
        return edits / units
    rate = value.get("rate")
    if not _is_number(rate) or not math.isfinite(float(rate)) or rate < 0:
        raise ValueError(f"{label} case {case_id!r} has invalid {metric.upper()} rate")
    return float(rate)


def _metric_counts(
    case: Mapping[str, object], metric: str
) -> tuple[int | None, int | None]:
    metrics = case.get("metrics")
    value = metrics.get(metric) if isinstance(metrics, dict) else None
    if not isinstance(value, dict):
        return None, None
    edits = value.get("edits")
    units = value.get("reference_units")
    if edits is None and units is None:
        return None, None
    if (
        not isinstance(edits, int)
        or isinstance(edits, bool)
        or edits < 0
        or not isinstance(units, int)
        or isinstance(units, bool)
        or units <= 0
    ):
        raise ValueError(f"Invalid {metric.upper()} edit counts")
    return edits, units


def _aggregate(
    cases: list[Mapping[str, object]], label: str
) -> dict[str, dict[str, object]]:
    result = {}
    for metric in METRICS:
        counts = [_metric_counts(case, metric) for case in cases]
        if all(edits is not None and units is not None for edits, units in counts):
            total_edits = sum(int(edits) for edits, _ in counts)
            total_units = sum(int(units) for _, units in counts)
            result[metric] = {
                "edits": total_edits,
                "reference_units": total_units,
                "rate": _round(total_edits / total_units),
                "method": "summed_edits_over_reference_units",
            }
        else:
            rates = [
                _metric_rate(case, metric, label, str(case.get("id"))) for case in cases
            ]
            result[metric] = {
                "rate": _round(sum(rates) / len(rates)),
                "method": "mean_case_rate_counts_unavailable",
            }
    return result


def _distribution(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean": _round(sum(values) / len(values)),
        "minimum": _round(ordered[0]),
        "p25": _round(_percentile(ordered, 0.25)),
        "median": _round(_percentile(ordered, 0.5)),
        "p75": _round(_percentile(ordered, 0.75)),
        "p90": _round(_percentile(ordered, 0.9)),
        "p95": _round(_percentile(ordered, 0.95)),
        "maximum": _round(ordered[-1]),
    }


def _percentile(ordered: list[float], quantile: float) -> float:
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _group_summary(pages: list[dict[str, object]]) -> dict[str, int | float | None]:
    if not pages:
        return {
            "cases": 0,
            "exact_oracle_angle_selection_count": 0,
            "exact_oracle_angle_selection_rate": None,
            "mean_cer_regret": None,
            "catastrophic_regret_count": 0,
        }
    exact = sum(bool(page["exact_oracle_angle_selection"]) for page in pages)
    return {
        "cases": len(pages),
        "exact_oracle_angle_selection_count": exact,
        "exact_oracle_angle_selection_rate": _round(exact / len(pages)),
        "mean_cer_regret": _round(
            sum(float(page["cer_regret"]) for page in pages) / len(pages)
        ),
        "catastrophic_regret_count": sum(
            bool(page["catastrophic_regret"]) for page in pages
        ),
    }


def _template_summary(
    pages: list[dict[str, object]],
    automatic_cases: Mapping[str, Mapping[str, object]],
    oracle_cases: list[Mapping[str, object]],
) -> dict[str, object]:
    page_ids = {str(page["id"]) for page in pages}
    automatic = [automatic_cases[case_id] for case_id in sorted(page_ids)]
    oracle_by_id = {str(case["id"]): case for case in oracle_cases}
    oracle = [oracle_by_id[case_id] for case_id in sorted(page_ids)]
    exact = sum(bool(page["exact_oracle_angle_selection"]) for page in pages)
    regrets = [float(page["cer_regret"]) for page in pages]
    return {
        "cases": len(pages),
        "aggregate": {
            "automatic": _aggregate(automatic, "automatic"),
            "oracle": _aggregate(oracle, "oracle"),
        },
        "exact_oracle_angle_selection_count": exact,
        "exact_oracle_angle_selection_rate": _round(exact / len(pages)),
        "mean_cer_regret": _round(sum(regrets) / len(regrets)),
        "maximum_cer_regret": _round(max(regrets)),
        "catastrophic_regret_count": sum(
            bool(page["catastrophic_regret"]) for page in pages
        ),
        "low_margin_count": sum(bool(page["low_margin"]) for page in pages),
    }


def _template_sort_key(item: tuple[str, object]) -> tuple[int, int | str]:
    template = item[0]
    return (0, int(template)) if template.isdigit() else (1, template)


def _validate_threshold(value: float, label: str) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{label.capitalize()} threshold cannot be negative")


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _round(value: float) -> float:
    return round(value, 6)


if __name__ == "__main__":
    raise SystemExit(main())
