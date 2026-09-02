"""Index private OCR transcripts as weak-supervision candidates."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import random
import re
import shutil
import tempfile
from typing import Any


ALIGNMENT_FILE = "alignments.jsonl"
SUMMARY_FILE = "summary.json"
SPLITS = ("train", "dev", "audit")
TEXTRACT_TIMESTAMP = re.compile(r"_20\d{6}_\d{6}$")
TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True)
class ProviderSpec:
    directory: str
    provider: str
    source_kind: str


PROVIDERS = (
    ProviderSpec("claude-haiku-4-5", "claude-haiku-4.5", "direct_model_ocr"),
    ProviderSpec("claude-sonnet-5", "claude-sonnet-5", "direct_model_ocr"),
    ProviderSpec("gpt-5.4-mini", "gpt-5.4-mini", "direct_model_ocr"),
    ProviderSpec("gpt-5.6-luna", "gpt-5.6-luna", "direct_model_ocr"),
    ProviderSpec("mistral_ocr_outputs", "mistral-ocr", "direct_model_ocr"),
    ProviderSpec("smartocr_gpt_5.4_mini", "smartocr-gpt-5.4-mini", "pipeline_ocr"),
    ProviderSpec("smartocr_luna", "smartocr-luna", "pipeline_ocr"),
    ProviderSpec("textract_gt", "aws-textract", "textract_ocr"),
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = prepare_index(
            args.source,
            args.output,
            seed=args.seed,
            dev_ratio=args.dev_ratio,
            audit_ratio=args.audit_ratio,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--dev-ratio", type=float, default=0.15)
    parser.add_argument("--audit-ratio", type=float, default=0.15)
    return parser


def prepare_index(
    source: Path,
    output: Path,
    *,
    seed: int = 17,
    dev_ratio: float = 0.15,
    audit_ratio: float = 0.15,
) -> dict[str, Any]:
    """Align transcripts and publish path-only, family-grouped candidates."""
    contracts_dir = source / "contracts"
    if not contracts_dir.is_dir():
        raise FileNotFoundError(f"contracts directory was not found: {contracts_dir}")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if min(dev_ratio, audit_ratio) < 0 or dev_ratio + audit_ratio >= 1:
        raise ValueError("dev and audit ratios must be non-negative and sum below 1")

    contracts = sorted(
        (path for path in contracts_dir.iterdir() if path.suffix.casefold() == ".pdf"),
        key=lambda path: path.name.casefold(),
    )
    if not contracts:
        raise ValueError("contracts directory contains no PDFs")
    documents: dict[str, Path] = {}
    for path in contracts:
        document_id = path.stem.casefold()
        if document_id in documents:
            raise ValueError(f"duplicate normalized contract id: {document_id}")
        documents[document_id] = path
    aliases = _unique_aliases(documents)
    family_by_document = {
        document_id: document_id.split(".", 1)[0] for document_id in documents
    }
    split_by_family = _assign_splits(
        family_by_document.values(), seed, dev_ratio, audit_ratio
    )

    provider_files: dict[str, dict[str, list[Path]]] = {}
    unmatched: dict[str, list[str]] = {}
    for spec in PROVIDERS:
        aligned, extra = _align_provider(source / spec.directory, documents, aliases)
        provider_files[spec.provider] = aligned
        unmatched[spec.provider] = [_relative(path, source) for path in extra]

    textract_runs = _load_textract_runs(source / "textract_gt", documents, aliases)
    alignments = []
    candidates = []
    status_counts: Counter[str] = Counter()
    missing_by_provider: Counter[str] = Counter()
    duplicate_by_provider: Counter[str] = Counter()
    indexed_transcripts = 0
    for document_id, contract_path in documents.items():
        family_id = family_by_document[document_id]
        split = split_by_family[family_id]
        outputs = []
        usable_texts: list[str] = []
        for spec in PROVIDERS:
            paths = provider_files[spec.provider].get(document_id, [])
            run = (
                textract_runs.get(document_id)
                if spec.provider == "aws-textract"
                else None
            )
            status = _output_status(
                paths, run, require_run=spec.provider == "aws-textract"
            )
            status_counts[status] += 1
            missing_by_provider[spec.provider] += status == "missing"
            duplicate_by_provider[spec.provider] += status == "duplicate"
            indexed_transcripts += len(paths)
            transcripts = [
                {
                    "path": _relative(path, source),
                    "provider": spec.provider,
                    "source_kind": spec.source_kind,
                    "size_bytes": path.stat().st_size,
                }
                for path in paths
            ]
            row = {
                "document_id": document_id,
                "family_id": family_id,
                "split": split,
                "provider": spec.provider,
                "source_kind": spec.source_kind,
                "status": status,
                "transcripts": transcripts,
                "run": run,
            }
            alignments.append(row)
            outputs.append(
                {
                    "provider": spec.provider,
                    "source_kind": spec.source_kind,
                    "status": status,
                    "transcript_paths": [item["path"] for item in transcripts],
                    "run": run,
                }
            )
            if status == "ok":
                usable_texts.append(paths[0].read_text(encoding="utf-8"))
        candidates.append(
            {
                "document_id": document_id,
                "family_id": family_id,
                "split": split,
                "source_pdf": _relative(contract_path, source),
                "provider_outputs": outputs,
                "agreement": _agreement(usable_texts),
                "supervision": "weak_only",
                "requires_human_review": True,
                "eligible_for_weak_supervision": split == "train"
                and len(usable_texts) >= 2,
            }
        )

    summary = {
        "status": "complete",
        "privacy": {
            "execution": "offline_private_only",
            "model_requests_made": False,
            "document_text_written": False,
        },
        "semantics": {
            "machine_outputs_are_ground_truth": False,
            "textract_is_ground_truth": False,
            "consensus_role": "weak_supervision_signal_only",
            "audit_requires_human_truth": True,
        },
        "documents": len(documents),
        "families": len(set(family_by_document.values())),
        "providers": len(PROVIDERS),
        "expected_alignments": len(documents) * len(PROVIDERS),
        "indexed_transcripts": indexed_transcripts,
        "alignment_status": dict(sorted(status_counts.items())),
        "missing_by_provider": dict(sorted(missing_by_provider.items())),
        "duplicate_by_provider": dict(sorted(duplicate_by_provider.items())),
        "unmatched_transcripts": {
            provider: paths for provider, paths in unmatched.items() if paths
        },
        "unmatched_transcript_count": sum(map(len, unmatched.values())),
        "split_documents": dict(Counter(row["split"] for row in candidates)),
        "failures_remain_in_denominator": True,
    }
    _publish(output, summary, alignments, candidates)
    return summary


def _unique_aliases(documents: dict[str, Path]) -> dict[str, str]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for document_id in documents:
        grouped[document_id.split(".", 1)[0]].append(document_id)
    return {
        alias: document_ids[0]
        for alias, document_ids in grouped.items()
        if len(document_ids) == 1
    }


def _assign_splits(
    families: Iterable[str],
    seed: int,
    dev_ratio: float,
    audit_ratio: float,
) -> dict[str, str]:
    unique = sorted(set(families))
    random.Random(seed).shuffle(unique)
    audit_count = round(len(unique) * audit_ratio)
    dev_count = round(len(unique) * dev_ratio)
    assignments = {}
    for index, family_id in enumerate(unique):
        if index < audit_count:
            split = "audit"
        elif index < audit_count + dev_count:
            split = "dev"
        else:
            split = "train"
        assignments[family_id] = split
    return assignments


def _align_provider(
    directory: Path,
    documents: dict[str, Path],
    aliases: dict[str, str],
) -> tuple[dict[str, list[Path]], list[Path]]:
    aligned: dict[str, list[Path]] = defaultdict(list)
    unmatched = []
    if not directory.is_dir():
        return aligned, unmatched
    for path in sorted(directory.glob("*.md"), key=lambda item: item.name.casefold()):
        document_id = _resolve_document(path.stem, documents, aliases)
        if document_id is None:
            unmatched.append(path)
            continue
        aligned[document_id].append(path)
    return aligned, unmatched


def _resolve_document(
    stem: str, documents: dict[str, Path], aliases: dict[str, str]
) -> str | None:
    key = TEXTRACT_TIMESTAMP.sub("", stem.casefold())
    if key in documents:
        return key
    return aliases.get(key)


def _load_textract_runs(
    directory: Path,
    documents: dict[str, Path],
    aliases: dict[str, str],
) -> dict[str, dict[str, Any]]:
    runs: dict[str, dict[str, Any]] = {}
    if not directory.is_dir():
        return runs
    for path in sorted(directory.glob("*_textract_meta.json")):
        stem = path.stem.removesuffix("_textract_meta")
        document_id = _resolve_document(stem, documents, aliases)
        if document_id is None:
            continue
        if document_id in runs:
            runs[document_id] = {"status": "duplicate_metadata"}
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            pages = payload.get("pages", [])
            if not isinstance(pages, list):
                raise ValueError("pages must be a list")
            failed_pages = sum(
                not isinstance(page, dict) or page.get("success") is not True
                for page in pages
            )
            runs[document_id] = {
                "status": "partial_failure" if failed_pages else "complete",
                "page_count": len(pages),
                "failed_pages": failed_pages,
            }
        except (OSError, ValueError, json.JSONDecodeError, TypeError):
            runs[document_id] = {"status": "invalid_metadata"}
    return runs


def _output_status(
    paths: list[Path], run: dict[str, Any] | None, *, require_run: bool
) -> str:
    if not paths:
        return "missing"
    if len(paths) > 1:
        return "duplicate"
    try:
        if not paths[0].read_text(encoding="utf-8").strip():
            return "empty"
    except (OSError, UnicodeError):
        return "unreadable"
    if require_run and run is None:
        return "missing_metadata"
    if run and run.get("status") != "complete":
        return "partial_failure"
    return "ok"


def _agreement(texts: list[str]) -> dict[str, Any]:
    normalized = [TOKEN_PATTERN.findall(text.casefold()) for text in texts]
    token_sets = [Counter(tokens) for tokens in normalized]
    pair_scores = [
        _token_f1(left, right)
        for index, left in enumerate(token_sets)
        for right in token_sets[index + 1 :]
    ]
    exact_counts = Counter(" ".join(tokens) for tokens in normalized)
    return {
        "basis": "mean_pairwise_token_f1",
        "provider_count": len(texts),
        "pair_count": len(pair_scores),
        "mean_pairwise_token_f1": (
            round(sum(pair_scores) / len(pair_scores), 6) if pair_scores else None
        ),
        "max_exact_provider_count": max(exact_counts.values(), default=0),
        "role": "weak_supervision_signal_only",
    }


def _token_f1(left: Counter[str], right: Counter[str]) -> float:
    left_count = sum(left.values())
    right_count = sum(right.values())
    if not left_count and not right_count:
        return 1.0
    if not left_count or not right_count:
        return 0.0
    overlap = sum((left & right).values())
    precision = overlap / left_count
    recall = overlap / right_count
    return 2 * precision * recall / (precision + recall) if overlap else 0.0


def _relative(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def _publish(
    output: Path,
    summary: dict[str, Any],
    alignments: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        (temporary / SUMMARY_FILE).write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _write_jsonl(temporary / ALIGNMENT_FILE, alignments)
        for split in SPLITS:
            _write_jsonl(
                temporary / f"{split}.jsonl",
                [row for row in candidates if row["split"] == split],
            )
        temporary.replace(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
