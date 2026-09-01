"""Compute deterministic paired statistics for OCR benchmark results."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from ocr_pipeline.verification import edit_distance

if __package__:
    from .public_benchmark import NORMALIZATION, _normalize
else:
    from public_benchmark import NORMALIZATION, _normalize

METRICS = ("cer", "wer")
VARIANTS = ("top-level", "local", "repaired")
ARM_PREFIX = "arm:"
DEFAULT_RESAMPLES = 10_000
DEFAULT_SEED = 0


@dataclass(frozen=True)
class MetricScore:
    edits: int
    reference_units: int


@dataclass(frozen=True)
class CaseScore:
    reference: str
    cluster_id: str
    metrics: Mapping[str, MetricScore]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare two OCR benchmark results with paired statistics"
    )
    parser.add_argument("baseline", type=Path, help="Baseline benchmark JSON")
    parser.add_argument("candidate", type=Path, help="Candidate benchmark JSON")
    parser.add_argument("output", type=Path, help="Paired comparison JSON")
    parser.add_argument(
        "--baseline-variant",
        default="top-level",
        help="Case metrics to use, including arm:<name> for a nested benchmark arm",
    )
    parser.add_argument(
        "--candidate-variant",
        default="top-level",
        help="Case metrics to use, including arm:<name> for a nested benchmark arm",
    )
    parser.add_argument(
        "--baseline-run",
        help="Nested benchmark run to compare, for example auto",
    )
    parser.add_argument(
        "--candidate-run",
        help="Nested benchmark run to compare, for example auto",
    )
    parser.add_argument(
        "--resamples",
        type=_positive_int,
        default=DEFAULT_RESAMPLES,
        help="Document bootstrap and sign-flip resamples",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    try:
        report = compare_files(
            args.baseline,
            args.candidate,
            baseline_variant=args.baseline_variant,
            candidate_variant=args.candidate_variant,
            baseline_run=args.baseline_run,
            candidate_run=args.candidate_run,
            resamples=args.resamples,
            seed=args.seed,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    return 0


def compare_files(
    baseline_path: Path,
    candidate_path: Path,
    *,
    baseline_variant: str = "top-level",
    candidate_variant: str = "top-level",
    baseline_run: str | None = None,
    candidate_run: str | None = None,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, object]:
    baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    candidate_payload = json.loads(candidate_path.read_text(encoding="utf-8"))
    baseline_payload = _select_run(baseline_payload, baseline_run, "baseline")
    candidate_payload = _select_run(candidate_payload, candidate_run, "candidate")
    report = compare_payloads(
        baseline_payload,
        candidate_payload,
        baseline_variant=baseline_variant,
        candidate_variant=candidate_variant,
        resamples=resamples,
        seed=seed,
    )
    return {
        "baseline": {
            "file": str(baseline_path),
            "variant": baseline_variant,
            "run": baseline_run,
        },
        "candidate": {
            "file": str(candidate_path),
            "variant": candidate_variant,
            "run": candidate_run,
        },
        **report,
    }


def _select_run(payload: object, run: str | None, label: str) -> object:
    if run is None:
        return payload
    if not isinstance(payload, dict) or not isinstance(payload.get("runs"), dict):
        raise ValueError(f"{label} JSON has no runs object")
    selected = payload["runs"].get(run)
    if not isinstance(selected, dict):
        raise ValueError(f"{label} JSON has no run {run!r}")
    return selected


def compare_payloads(
    baseline_payload: object,
    candidate_payload: object,
    *,
    baseline_variant: str = "top-level",
    candidate_variant: str = "top-level",
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, object]:
    if resamples < 1:
        raise ValueError("resamples must be positive")
    dataset = _validate_metadata(baseline_payload, candidate_payload)
    baseline = _case_scores(baseline_payload, baseline_variant, "baseline", dataset)
    candidate = _case_scores(candidate_payload, candidate_variant, "candidate", dataset)
    _validate_pairs(baseline, candidate)

    case_ids = sorted(baseline)
    clusters = _cluster_case_ids(baseline, case_ids)
    cluster_bootstrap_deltas = _bootstrap_deltas(
        baseline, candidate, list(clusters.values()), resamples, seed
    )
    page_bootstrap_deltas = _bootstrap_deltas(
        baseline, candidate, [[case_id] for case_id in case_ids], resamples, seed
    )
    p_values = _sign_flip_p_values(
        baseline, candidate, list(clusters.values()), resamples, seed
    )
    adjusted_p_values = _holm_adjust(p_values)

    metrics = {}
    for metric in METRICS:
        baseline_scores = [baseline[case_id].metrics[metric] for case_id in case_ids]
        candidate_scores = [candidate[case_id].metrics[metric] for case_id in case_ids]
        reference_units = sum(score.reference_units for score in baseline_scores)
        baseline_micro = _micro_rate(baseline_scores)
        candidate_micro = _micro_rate(candidate_scores)
        metrics[metric] = {
            "reference_units": reference_units,
            "baseline_micro": round(baseline_micro, 6),
            "candidate_micro": round(candidate_micro, 6),
            "micro_delta": round(candidate_micro - baseline_micro, 6),
            "micro_delta_bootstrap_ci_95": [
                round(_percentile(cluster_bootstrap_deltas[metric], 0.025), 6),
                round(_percentile(cluster_bootstrap_deltas[metric], 0.975), 6),
            ],
            "micro_delta_page_bootstrap_ci_95_exploratory": [
                round(_percentile(page_bootstrap_deltas[metric], 0.025), 6),
                round(_percentile(page_bootstrap_deltas[metric], 0.975), 6),
            ],
            "paired_cases": _paired_outcomes(baseline_scores, candidate_scores),
            "two_sided_sign_flip_p_value": round(p_values[metric], 6),
            "holm_adjusted_p_value": round(adjusted_p_values[metric], 6),
        }

    return {
        "cases": len(case_ids),
        "sample_counts": {"pages": len(case_ids), "clusters": len(clusters)},
        "primary_resampling_unit": "cluster",
        "exploratory_resampling_unit": "page",
        "resamples": resamples,
        "seed": seed,
        "metrics": metrics,
    }


def _validate_metadata(baseline_payload: object, candidate_payload: object) -> str:
    baseline = _metadata(baseline_payload, "baseline")
    candidate = _metadata(candidate_payload, "candidate")
    if baseline != candidate:
        raise ValueError(
            f"Benchmark metadata differs: baseline={baseline}, candidate={candidate}"
        )
    dataset, normalization = baseline
    if normalization != NORMALIZATION:
        raise ValueError(
            "Unsupported benchmark normalization: "
            f"{normalization!r}; expected {NORMALIZATION!r}"
        )
    return dataset


def _metadata(payload: object, label: str) -> tuple[str, str]:
    if not isinstance(payload, dict):
        raise ValueError(f"{label} JSON must be an object")
    dataset = payload.get("dataset")
    normalization = payload.get("normalization")
    if not isinstance(dataset, str) or not dataset:
        raise ValueError(f"{label} JSON has no valid dataset")
    if not isinstance(normalization, str) or not normalization:
        raise ValueError(f"{label} JSON has no valid normalization")
    return dataset, normalization


def _case_scores(
    payload: object, variant: str, label: str, dataset: str
) -> dict[str, CaseScore]:
    if not isinstance(payload, dict) or not isinstance(payload.get("cases"), list):
        raise ValueError(f"{label} JSON must contain a cases array")

    scores = {}
    for index, case in enumerate(payload["cases"]):
        if not isinstance(case, dict):
            raise ValueError(f"{label} case at index {index} must be an object")
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"{label} case at index {index} has no valid id")
        if case_id in scores:
            raise ValueError(f"Duplicate {label} case id: {case_id}")
        reference = case.get("reference")
        if not isinstance(reference, str):
            raise ValueError(f"{label} case {case_id!r} has no valid reference")

        metric_container = _variant_container(case, variant, label, case_id)
        if not isinstance(metric_container, dict):
            raise ValueError(f"{label} case {case_id!r} has no {variant} variant")
        prediction = metric_container.get("prediction")
        if not isinstance(prediction, str):
            raise ValueError(f"{label} case {case_id!r} has no valid prediction")
        raw_metrics = metric_container.get("metrics")
        if not isinstance(raw_metrics, dict):
            raise ValueError(f"{label} case {case_id!r} has no metrics")
        recomputed = _recompute_metrics(prediction, reference)
        metrics = {
            metric: _metric_score(
                raw_metrics.get(metric), recomputed[metric], label, case_id, metric
            )
            for metric in METRICS
        }
        scores[case_id] = CaseScore(
            reference=reference,
            cluster_id=_cluster_id(case, case_id, dataset, label),
            metrics=metrics,
        )

    if not scores:
        raise ValueError(f"{label} JSON contains no cases")
    return scores


def _recompute_metrics(prediction: str, reference: str) -> dict[str, dict[str, int]]:
    normalized_prediction = _normalize(prediction)
    normalized_reference = _normalize(reference)
    return {
        "cer": {
            "edits": edit_distance(normalized_prediction, normalized_reference),
            "reference_units": len(normalized_reference),
        },
        "wer": {
            "edits": edit_distance(
                normalized_prediction.split(), normalized_reference.split()
            ),
            "reference_units": len(normalized_reference.split()),
        },
    }


def _variant_container(
    case: Mapping[str, object], variant: str, label: str, case_id: str
) -> object:
    if variant == "top-level":
        return case
    if variant in VARIANTS:
        return case.get(variant)
    if variant.startswith(ARM_PREFIX) and len(variant) > len(ARM_PREFIX):
        arms = case.get("arms")
        return arms.get(variant[len(ARM_PREFIX) :]) if isinstance(arms, dict) else None
    raise ValueError(f"Unsupported {label} variant: {variant}")


def _metric_score(
    value: object,
    recomputed: Mapping[str, float | int],
    label: str,
    case_id: str,
    metric: str,
) -> MetricScore:
    if not isinstance(value, dict):
        raise ValueError(f"{label} case {case_id!r} has no {metric.upper()} metrics")
    reference_units = value.get("reference_units")
    expected_units = recomputed["reference_units"]
    if not _is_nonnegative_int(reference_units) or reference_units != expected_units:
        raise ValueError(
            f"{label} case {case_id!r} has incompatible {metric.upper()} "
            f"reference denominator: {reference_units!r} != {expected_units}"
        )
    return MetricScore(edits=int(recomputed["edits"]), reference_units=reference_units)


def _cluster_id(
    case: Mapping[str, object], case_id: str, dataset: str, label: str
) -> str:
    explicit = case.get("cluster_id")
    if explicit is not None and (not isinstance(explicit, str) or not explicit):
        raise ValueError(f"{label} case {case_id!r} has invalid cluster_id")
    if dataset.casefold() == "clinocr":
        match = re.search(r"(?:^|/)template_(.+?)_sample_", case_id)
        if not match:
            raise ValueError(
                f"{label} ClinOCR case {case_id!r} has no derivable template cluster"
            )
        derived = f"template_{match.group(1)}"
        if isinstance(explicit, str):
            if explicit.isdigit():
                return f"template_{explicit}"
            if explicit.startswith("template_"):
                return explicit
            return explicit
        return derived
    return explicit if isinstance(explicit, str) else case_id


def _validate_pairs(
    baseline: Mapping[str, CaseScore], candidate: Mapping[str, CaseScore]
) -> None:
    baseline_ids = set(baseline)
    candidate_ids = set(candidate)
    if baseline_ids != candidate_ids:
        missing_candidate = sorted(baseline_ids - candidate_ids)
        missing_baseline = sorted(candidate_ids - baseline_ids)
        raise ValueError(
            "Case IDs differ: "
            f"missing from candidate={missing_candidate}, "
            f"missing from baseline={missing_baseline}"
        )

    for case_id in sorted(baseline):
        baseline_case = baseline[case_id]
        candidate_case = candidate[case_id]
        if baseline_case.reference != candidate_case.reference:
            raise ValueError(f"References differ for case {case_id!r}")
        if baseline_case.cluster_id != candidate_case.cluster_id:
            raise ValueError(f"Cluster IDs differ for case {case_id!r}")
        for metric in METRICS:
            baseline_units = baseline_case.metrics[metric].reference_units
            candidate_units = candidate_case.metrics[metric].reference_units
            if baseline_units != candidate_units:
                raise ValueError(
                    f"{metric.upper()} reference denominator differs for case "
                    f"{case_id!r}: {baseline_units} != {candidate_units}"
                )


def _bootstrap_deltas(
    baseline: Mapping[str, CaseScore],
    candidate: Mapping[str, CaseScore],
    clusters: list[list[str]],
    resamples: int,
    seed: int,
) -> dict[str, list[float]]:
    random_source = random.Random(seed)
    deltas = {metric: [] for metric in METRICS}
    for _ in range(resamples):
        sampled_ids = [
            case_id
            for _ in clusters
            for case_id in clusters[random_source.randrange(len(clusters))]
        ]
        for metric in METRICS:
            baseline_scores = [
                baseline[case_id].metrics[metric] for case_id in sampled_ids
            ]
            candidate_scores = [
                candidate[case_id].metrics[metric] for case_id in sampled_ids
            ]
            deltas[metric].append(
                _micro_rate(candidate_scores) - _micro_rate(baseline_scores)
            )
    return deltas


def _sign_flip_p_values(
    baseline: Mapping[str, CaseScore],
    candidate: Mapping[str, CaseScore],
    clusters: list[list[str]],
    resamples: int,
    seed: int,
) -> dict[str, float]:
    edit_differences = {
        metric: [
            sum(
                candidate[case_id].metrics[metric].edits
                - baseline[case_id].metrics[metric].edits
                for case_id in cluster
            )
            for cluster in clusters
        ]
        for metric in METRICS
    }
    observed = {metric: abs(sum(edit_differences[metric])) for metric in METRICS}
    extreme = {metric: 0 for metric in METRICS}
    random_source = random.Random(seed)
    for _ in range(resamples):
        signs = [1 if random_source.getrandbits(1) else -1 for _ in clusters]
        for metric in METRICS:
            flipped = sum(
                sign * difference
                for sign, difference in zip(signs, edit_differences[metric])
            )
            if abs(flipped) >= observed[metric]:
                extreme[metric] += 1
    return {metric: (extreme[metric] + 1) / (resamples + 1) for metric in METRICS}


def _cluster_case_ids(
    scores: Mapping[str, CaseScore], case_ids: Sequence[str]
) -> dict[str, list[str]]:
    clusters: dict[str, list[str]] = {}
    for case_id in case_ids:
        clusters.setdefault(scores[case_id].cluster_id, []).append(case_id)
    return dict(sorted(clusters.items()))


def _holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.__getitem__)
    adjusted = {}
    previous = 0.0
    count = len(ordered)
    for index, metric in enumerate(ordered):
        value = min(1.0, (count - index) * p_values[metric])
        previous = max(previous, value)
        adjusted[metric] = previous
    return adjusted


def _paired_outcomes(
    baseline: list[MetricScore], candidate: list[MetricScore]
) -> dict[str, int]:
    wins = sum(
        candidate_score.edits < baseline_score.edits
        for baseline_score, candidate_score in zip(baseline, candidate)
    )
    ties = sum(
        candidate_score.edits == baseline_score.edits
        for baseline_score, candidate_score in zip(baseline, candidate)
    )
    return {
        "candidate_wins": wins,
        "ties": ties,
        "candidate_losses": len(baseline) - wins - ties,
    }


def _micro_rate(scores: list[MetricScore]) -> float:
    edits = sum(score.edits for score in scores)
    reference_units = sum(score.reference_units for score in scores)
    if reference_units == 0:
        return 0.0 if edits == 0 else float(edits)
    return edits / reference_units


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    weight = position - lower_index
    return ordered[lower_index] * (1 - weight) + ordered[upper_index] * weight


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return number


def _is_nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


if __name__ == "__main__":
    raise SystemExit(main())
