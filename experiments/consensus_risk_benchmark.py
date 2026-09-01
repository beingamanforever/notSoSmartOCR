"""Evaluate OCR disagreement as a calibration-free review signal."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Sequence

from ocr_pipeline.verification import (
    consensus_scores,
    edit_counts,
    literal_text_risks,
)

if __package__:
    from .public_benchmark import NORMALIZATION, _normalize, _score
else:
    from public_benchmark import NORMALIZATION, _normalize, _score

COVERAGE_POINTS = (1.0, 0.9, 0.8, 0.5)
DEFAULT_SEVERE_CER = 0.2


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure whether OCR disagreement predicts primary-reader error"
    )
    parser.add_argument("primary", type=Path, help="Primary OCR benchmark JSON")
    parser.add_argument("output", type=Path, help="Risk report JSON")
    parser.add_argument(
        "alternatives",
        type=Path,
        nargs="+",
        help="Independent OCR benchmark JSON files for the same cases",
    )
    parser.add_argument("--primary-run", help="Nested primary run, for example auto")
    parser.add_argument(
        "--severe-cer",
        type=_unit_interval,
        default=DEFAULT_SEVERE_CER,
        help="CER used only to evaluate severe-error ranking",
    )
    args = parser.parse_args(argv)

    try:
        primary = _read_run(args.primary, args.primary_run)
        alternatives = [_read_run(path, None) for path in args.alternatives]
        report = benchmark_risk(primary, alternatives, severe_cer=args.severe_cer)
        report["sources"] = {
            "primary": {"file": str(args.primary), "run": args.primary_run},
            "alternatives": [str(path) for path in args.alternatives],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    return 0


def benchmark_risk(
    primary: object,
    alternatives: Sequence[object],
    *,
    severe_cer: float = DEFAULT_SEVERE_CER,
) -> dict[str, object]:
    if not alternatives:
        raise ValueError("At least one alternative OCR run is required")
    if not 0 <= severe_cer <= 1:
        raise ValueError("severe_cer must be between zero and one")

    runs = [primary, *alternatives]
    metadata = [_metadata(run, index) for index, run in enumerate(runs)]
    if len(set(metadata)) != 1:
        raise ValueError(f"Benchmark metadata differs: {metadata}")
    dataset, normalization = metadata[0]
    if normalization != NORMALIZATION:
        raise ValueError(f"Unsupported normalization: {normalization!r}")

    case_maps = [_case_map(run, index) for index, run in enumerate(runs)]
    primary_ids = set(case_maps[0])
    for index, cases in enumerate(case_maps[1:], start=1):
        missing = primary_ids - set(cases)
        if missing:
            raise ValueError(
                f"Alternative {index} is missing {len(missing)} primary case IDs"
            )

    names = [_reader_name(run, index) for index, run in enumerate(runs)]
    records = [
        _case_record(case_id, names, case_maps, severe_cer)
        for case_id in sorted(primary_ids)
    ]
    literal_counts = Counter(
        reason for record in records for reason in record["literal_risks"]
    )
    return {
        "experiment": "consensus_disagreement_risk",
        "purpose": "calibration_free_research_diagnostic",
        "dataset": dataset,
        "normalization": normalization,
        "candidate_readers": names,
        "matched_panel": {
            "cases": len(primary_ids),
            "runs": [
                {
                    "reader": name,
                    "source_cases": len(cases),
                    "ignored_extra_cases": len(set(cases) - primary_ids),
                }
                for name, cases in zip(names, case_maps, strict=True)
            ],
        },
        "severe_cer": severe_cer,
        "selection_policy": "none",
        "warning": (
            "Agreement is a review signal, not proof of correctness; correlated "
            "readers can agree on the same error."
        ),
        "summary": {
            "cases": len(records),
            "primary": _primary_summary(records),
            "literal_risk_counts": dict(sorted(literal_counts.items())),
            "signals": {
                "consensus": _risk_summary(records, "consensus_risk", severe_cer),
                "literal": _risk_summary(records, "literal_risk_count", severe_cer),
            },
        },
        "cases": records,
    }


def _case_record(
    case_id: str,
    names: Sequence[str],
    case_maps: Sequence[dict[str, dict[str, object]]],
    severe_cer: float,
) -> dict[str, object]:
    cases = [case_map[case_id] for case_map in case_maps]
    reference = _required_text(cases[0], "reference", case_id)
    for index, case in enumerate(cases[1:], start=1):
        if _required_text(case, "reference", case_id) != reference:
            raise ValueError(f"Alternative {index} reference differs for {case_id}")

    predictions = [_required_text(case, "prediction", case_id) for case in cases]
    normalized = [_normalize(prediction) for prediction in predictions]
    scores = consensus_scores(normalized)
    metrics = _score(predictions[0], reference)
    normalized_reference = _normalize(reference)
    aligned = edit_counts(normalized[0].split(), normalized_reference.split())
    reference_words = len(normalized_reference.split())
    risks = literal_text_risks(predictions[0])
    cer = float(metrics["cer"]["rate"])
    return {
        "id": case_id,
        "cluster_id": str(cases[0].get("cluster_id", case_id)),
        "primary_cer": cer,
        "severe_error": cer >= severe_cer,
        "consensus_risk": round(scores[0], 6),
        "candidate_scores": {
            name: round(score, 6) for name, score in zip(names, scores, strict=True)
        },
        "literal_risks": list(risks),
        "literal_risk_count": len(risks),
        "aligned_word_insertion_rate": round(
            aligned.insertions / max(reference_words, 1), 6
        ),
        "aligned_word_deletion_rate": round(
            aligned.deletions / max(reference_words, 1), 6
        ),
        "edits": {
            "insertions": aligned.insertions,
            "deletions": aligned.deletions,
            "substitutions": aligned.substitutions,
            "reference_words": reference_words,
        },
    }


def _primary_summary(records: Sequence[dict[str, object]]) -> dict[str, object]:
    edits = [record["edits"] for record in records]
    total_reference = sum(int(item["reference_words"]) for item in edits)
    totals = {
        name: sum(int(item[name]) for item in edits)
        for name in ("insertions", "deletions", "substitutions")
    }
    return {
        "case_mean_cer": round(
            sum(float(record["primary_cer"]) for record in records) / len(records), 6
        ),
        "severe_errors": sum(bool(record["severe_error"]) for record in records),
        "word_error_rates": {
            name: round(value / max(total_reference, 1), 6)
            for name, value in totals.items()
        },
        "edit_counts": {**totals, "reference_words": total_reference},
    }


def _risk_summary(
    records: Sequence[dict[str, object]], field: str, severe_cer: float
) -> dict[str, object]:
    ordered = sorted(
        records, key=lambda record: (float(record[field]), str(record["id"]))
    )
    selective_risks = [
        sum(float(record["primary_cer"]) for record in ordered[:count]) / count
        for count in range(1, len(ordered) + 1)
    ]
    return {
        "aurc_case_mean_cer": round(sum(selective_risks) / len(selective_risks), 6),
        "augrc_severe_error": _augrc(ordered),
        "severe_error_auroc": _auroc(records, field, severe_cer),
        "cer_rank_concordance": _rank_concordance(records, field, "primary_cer"),
        "word_insertion_rank_concordance": _rank_concordance(
            records, field, "aligned_word_insertion_rate"
        ),
        "risk_at_coverage": {
            f"{coverage:.1f}": _coverage_record(ordered, field, coverage)
            for coverage in COVERAGE_POINTS
        },
    }


def _coverage_record(
    ordered: Sequence[dict[str, object]], field: str, coverage: float
) -> dict[str, float | int]:
    count = max(1, math.ceil(len(ordered) * coverage))
    selected = ordered[:count]
    return {
        "cases": count,
        "coverage": round(count / len(ordered), 6),
        "case_mean_cer": round(
            sum(float(record["primary_cer"]) for record in selected) / count, 6
        ),
        "severe_errors": sum(bool(record["severe_error"]) for record in selected),
        "max_accepted_risk": round(float(selected[-1][field]), 6),
    }


def _auroc(
    records: Sequence[dict[str, object]], field: str, severe_cer: float
) -> float | None:
    positives = [
        float(record[field])
        for record in records
        if float(record["primary_cer"]) >= severe_cer
    ]
    negatives = [
        float(record[field])
        for record in records
        if float(record["primary_cer"]) < severe_cer
    ]
    if not positives or not negatives:
        return None
    wins = sum(
        1.0 if positive > negative else 0.5 if positive == negative else 0.0
        for positive in positives
        for negative in negatives
    )
    return round(wins / (len(positives) * len(negatives)), 6)


def _augrc(ordered: Sequence[dict[str, object]]) -> float:
    failures = 0
    total = len(ordered)
    area = 0.0
    for record in ordered:
        failures += bool(record["severe_error"])
        area += failures / total
    return round(area / total, 6)


def _rank_concordance(
    records: Sequence[dict[str, object]], risk_field: str, outcome_field: str
) -> float | None:
    concordant = 0
    discordant = 0
    for left_index, left in enumerate(records):
        for right in records[left_index + 1 :]:
            risk_delta = float(left[risk_field]) - float(right[risk_field])
            outcome_delta = float(left[outcome_field]) - float(right[outcome_field])
            product = risk_delta * outcome_delta
            concordant += product > 0
            discordant += product < 0
    compared = concordant + discordant
    if not compared:
        return None
    return round((concordant - discordant) / compared, 6)


def _read_run(path: Path, run: str | None) -> object:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if run is None:
        return payload
    if not isinstance(payload, dict) or not isinstance(payload.get("runs"), dict):
        raise ValueError(f"Benchmark has no runs object: {path}")
    selected = payload["runs"].get(run)
    if not isinstance(selected, dict):
        raise ValueError(f"Benchmark has no run {run!r}: {path}")
    return selected


def _metadata(run: object, index: int) -> tuple[str, str]:
    if not isinstance(run, dict):
        raise ValueError(f"Run {index} must be an object")
    dataset = run.get("dataset")
    normalization = run.get("normalization")
    if not isinstance(dataset, str) or not isinstance(normalization, str):
        raise ValueError(f"Run {index} has invalid benchmark metadata")
    return dataset, normalization


def _case_map(run: object, index: int) -> dict[str, dict[str, object]]:
    if not isinstance(run, dict) or not isinstance(run.get("cases"), list):
        raise ValueError(f"Run {index} has no cases array")
    cases: dict[str, dict[str, object]] = {}
    for case in run["cases"]:
        if not isinstance(case, dict) or not isinstance(case.get("id"), str):
            raise ValueError(f"Run {index} contains an invalid case")
        case_id = case["id"]
        if case_id in cases:
            raise ValueError(f"Run {index} contains duplicate case {case_id}")
        cases[case_id] = case
    if not cases:
        raise ValueError(f"Run {index} contains no cases")
    return cases


def _reader_name(run: object, index: int) -> str:
    if isinstance(run, dict) and isinstance(run.get("reader"), str):
        return run["reader"]
    return f"reader-{index}"


def _required_text(case: dict[str, object], field: str, case_id: str) -> str:
    value = case.get(field)
    if not isinstance(value, str):
        raise ValueError(f"Case {case_id!r} has no valid {field}")
    return value


def _unit_interval(value: str) -> float:
    number = float(value)
    if not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("value must be between zero and one")
    return number


if __name__ == "__main__":
    raise SystemExit(main())
