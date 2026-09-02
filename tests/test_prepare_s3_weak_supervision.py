from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.prepare_s3_weak_supervision import PROVIDERS, main, prepare_index


def test_indexes_filename_variants_without_promoting_machine_text(
    tmp_path: Path,
) -> None:
    source = _corpus(tmp_path)
    output = tmp_path / "index"

    summary = prepare_index(source, output, seed=3, dev_ratio=0.25, audit_ratio=0.25)

    alignments = _jsonl(output / "alignments.jsonl")
    candidates = sum(
        (_jsonl(output / f"{split}.jsonl") for split in ("train", "dev", "audit")), []
    )
    assert summary["documents"] == 4
    assert summary["expected_alignments"] == 4 * len(PROVIDERS)
    assert summary["alignment_status"] == {
        "duplicate": 1,
        "empty": 1,
        "missing": 1,
        "ok": 28,
        "partial_failure": 1,
    }
    assert summary["indexed_transcripts"] == 32
    assert summary["unmatched_transcript_count"] == 1
    assert summary["failures_remain_in_denominator"] is True
    assert len(alignments) == 4 * len(PROVIDERS)
    assert len(candidates) == 4
    assert all(
        set(transcript) == {"path", "provider", "source_kind", "size_bytes"}
        for row in alignments
        for transcript in row["transcripts"]
    )
    assert all(candidate["supervision"] == "weak_only" for candidate in candidates)
    assert all(candidate["requires_human_review"] is True for candidate in candidates)
    assert all(
        candidate["eligible_for_weak_supervision"] is False
        for candidate in candidates
        if candidate["split"] != "train"
    )
    assert all(
        candidate["agreement"]["role"] == "weak_supervision_signal_only"
        for candidate in candidates
    )
    assert {candidate["agreement"]["provider_count"] for candidate in candidates} == {7}
    assert summary["semantics"]["machine_outputs_are_ground_truth"] is False
    assert summary["semantics"]["textract_is_ground_truth"] is False

    serialized = "".join(path.read_text(encoding="utf-8") for path in output.iterdir())
    assert "PRIVATE TRANSCRIPT" not in serialized
    assert '"ground_truth":' not in serialized
    assert '"reference":' not in serialized


def test_keeps_ambiguous_short_names_unmatched_and_cli_is_deterministic(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    (source / "contracts").mkdir(parents=True)
    (source / "contracts" / "IF1.a.pdf").write_bytes(b"pdf")
    (source / "contracts" / "IF1.b.pdf").write_bytes(b"pdf")
    (source / "gpt-5.6-luna").mkdir()
    (source / "gpt-5.6-luna" / "IF1.md").write_text(
        "PRIVATE TRANSCRIPT", encoding="utf-8"
    )

    first = tmp_path / "first"
    second = tmp_path / "second"
    assert main([str(source), str(first), "--seed", "9"]) == 0
    assert main([str(source), str(second), "--seed", "9"]) == 0

    first_summary = json.loads((first / "summary.json").read_text(encoding="utf-8"))
    assert first_summary["unmatched_transcript_count"] == 1
    assert first_summary["alignment_status"] == {"missing": 16}
    for name in ("alignments.jsonl", "train.jsonl", "dev.jsonl", "audit.jsonl"):
        assert (first / name).read_text(encoding="utf-8") == (second / name).read_text(
            encoding="utf-8"
        )


def test_rejects_existing_output_and_invalid_ratios(tmp_path: Path) -> None:
    source = _corpus(tmp_path)
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError, match="output already exists"):
        prepare_index(source, output)
    with pytest.raises(ValueError, match="sum below 1"):
        prepare_index(source, tmp_path / "invalid", dev_ratio=0.5, audit_ratio=0.5)


def _corpus(root: Path) -> Path:
    source = root / "source"
    contracts = source / "contracts"
    contracts.mkdir(parents=True)
    document_ids = ["IF100.900", "IF200.800", "UUID.A.0", "IF300.700"]
    for document_id in document_ids:
        (contracts / f"{document_id}.pdf").write_bytes(b"pdf")
    for spec in PROVIDERS:
        directory = source / spec.directory
        directory.mkdir()
        for document_id in document_ids:
            stem = document_id
            if spec.directory in {
                "claude-haiku-4-5",
                "claude-sonnet-5",
                "gpt-5.4-mini",
                "gpt-5.6-luna",
            } and document_id.startswith("IF"):
                stem = document_id.split(".", 1)[0]
            if spec.directory == "textract_gt":
                stem += "_20260812_120000"
            text = "PRIVATE TRANSCRIPT same content"
            if spec.provider == "claude-haiku-4.5" and document_id == "IF100.900":
                text = ""
            if spec.provider == "mistral-ocr" and document_id == "IF200.800":
                continue
            (directory / f"{stem}.md").write_text(text, encoding="utf-8")
            if spec.provider == "aws-textract" and document_id == "IF300.700":
                duplicate = document_id + "_20260813_120000"
                (directory / f"{duplicate}.md").write_text(text, encoding="utf-8")
            if spec.provider == "aws-textract":
                failed = document_id == "UUID.A.0"
                metadata = {"pages": [{"success": not failed}, {"success": True}]}
                (directory / f"{stem}_textract_meta.json").write_text(
                    json.dumps(metadata), encoding="utf-8"
                )
        if spec.provider == "gpt-5.6-luna":
            (directory / "unmatched.md").write_text(
                "PRIVATE TRANSCRIPT", encoding="utf-8"
            )
    return source


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
