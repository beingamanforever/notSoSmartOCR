from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import pytest

from ocr_pipeline.openrouter import OpenRouterResult

sys.path.insert(0, str(Path(__file__).parents[1]))
from experiments import private_frontier_benchmark as benchmark  # noqa: E402

MODEL = "openai/gpt-5.6-sol"
PROVIDER = "openai"


def _write_case(
    root: Path,
    case_id: str,
    task: str,
    suffix: int,
    target: dict[str, Any],
    *,
    pages: int = 1,
) -> None:
    case_dir = root / case_id
    case_dir.mkdir(parents=True)
    images = []
    for page in range(1, pages + 1):
        name = f"image-{page:03d}.png"
        (case_dir / name).write_bytes(b"synthetic image")
        images.append({"image_id": f"image-{page:03d}", "file": name, "page": page})
    payload = {
        "schema_version": 1,
        "case_id": case_id,
        "task": task,
        "cohort": "synthetic",
        "availability": "ready",
        "unavailable_reason": None,
        "source_reference": f"source-{suffix:06d}",
        "images": images,
        "target": target,
    }
    (case_dir / "ground_truth.json").write_text(json.dumps(payload), encoding="utf-8")


def _approve(root: Path, provider: str = PROVIDER) -> None:
    approval = {
        "schema_version": 1,
        "data_root": str(root.resolve()),
        "data_classification": "synthetic",
        "provider_slug": provider,
    }
    (root / benchmark.APPROVAL_FILENAME).write_text(
        json.dumps(approval), encoding="utf-8"
    )


def _form_target(label: str = "Secret form label") -> dict[str, Any]:
    return {
        "controls": [
            {
                "control_id": "control-001",
                "label": label,
                "aliases": ["Visible alias"],
                "image_ids": ["image-001"],
                "state": "checked",
            }
        ]
    }


def _duplicate_form_target() -> dict[str, Any]:
    target = _form_target()
    target["controls"].append(
        {
            "control_id": "control-002",
            "label": "Secret second label",
            "aliases": ["Visible alias"],
            "image_ids": ["image-001"],
            "state": "unchecked",
        }
    )
    return target


def _handwriting_target() -> dict[str, Any]:
    return {
        "segments": [
            {
                "segment_id": "segment-001",
                "section": "Secret section",
                "text": "Secret handwritten target 4831",
                "image_ids": ["image-001", "image-002"],
            }
        ]
    }


def _table_target() -> dict[str, Any]:
    return {
        "tables": [
            {
                "table_id": "table-001",
                "label": "Secret table label",
                "aliases": ["Visible table"],
                "image_id": "image-001",
                "rows": [
                    {
                        "row_id": "row-001",
                        "label": "Secret row",
                        "aliases": ["Visible row"],
                    }
                ],
                "columns": [
                    {
                        "column_id": "column-001",
                        "label": "Secret column",
                        "aliases": ["Visible column"],
                    }
                ],
                "cells": [
                    {
                        "row_id": "row-001",
                        "column_id": "column-001",
                        "text": "Secret cell 9917",
                    }
                ],
                "merged_ranges": [],
            }
        ]
    }


def _result(content: dict[str, Any]) -> OpenRouterResult:
    return OpenRouterResult(
        content=content,
        model=MODEL,
        provider="OpenAI",
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        cost=0.001,
        latency_ms=12.0,
        attempts=1,
    )


def test_private_adapter_has_no_target_leakage_and_writes_evaluator_shapes(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    output = tmp_path / "run"
    _write_case(data, "form-000001", "forms", 1, _form_target())
    _write_case(
        data,
        "handwriting-000002",
        "handwriting",
        6,
        _handwriting_target(),
        pages=2,
    )
    _write_case(data, "table-000003", "tables", 11, _table_target())
    _approve(data)

    calls: list[dict[str, Any]] = []

    def fake_repair(image_path, prompt, schema, **options):
        serialized = json.dumps({"prompt": prompt, "schema": schema})
        for secret in (
            "source-000001",
            "Secret form label",
            "Secret section",
            "Secret handwritten target 4831",
            "Secret table label",
            "Secret row",
            "Secret column",
            "Secret cell 9917",
            "control-001",
            "table-001",
            "row-001",
            "column-001",
        ):
            assert secret not in serialized
        assert options == {
            "model": MODEL,
            "schema_name": options["schema_name"],
            "max_tokens": 777,
            "provider_slug": PROVIDER,
            "public_benchmark": True,
        }
        calls.append({"image": Path(image_path).name, "prompt": prompt})
        if schema is benchmark.FORM_SCHEMA:
            return _result(
                {
                    "controls": [
                        {
                            "label": "Visible alias",
                            "state": "checked",
                        }
                    ]
                }
            )
        if schema is benchmark.HANDWRITING_SCHEMA:
            page = int(Path(image_path).stem.rsplit("-", 1)[1])
            return _result(
                {
                    "text": f"Literal page {page}",
                    "legibility": "legible",
                    "uncertain_spans": [],
                }
            )
        return _result(
            {
                "tables": [
                    {
                        "label": "Visible table",
                        "rows": [{"label": "Visible row"}],
                        "columns": [{"label": "Visible column"}],
                        "cells": [
                            {
                                "row_index": 1,
                                "column_index": 1,
                                "text": "Predicted cell",
                            }
                        ],
                        "merged_ranges": [],
                    }
                ]
            }
        )

    summary = benchmark.run_benchmark(
        data,
        output,
        model=MODEL,
        provider=PROVIDER,
        split="all",
        review_passes=3,
        max_tokens=777,
        repair=fake_repair,
    )

    assert summary == {"eligible_cases": 3, "submitted_cases": 3}
    assert len(calls) == 12
    predictions = output / "predictions"
    audit = output / "audit"
    form = json.loads((predictions / "form-000001.json").read_text())
    handwriting = json.loads((predictions / "handwriting-000002.json").read_text())
    table = json.loads((predictions / "table-000003.json").read_text())
    assert form == {
        "case_id": "form-000001",
        "task": "forms",
        "prediction": {"controls": [{"control_id": "control-001", "state": "checked"}]},
    }
    assert handwriting == {
        "case_id": "handwriting-000002",
        "task": "handwriting",
        "prediction": {"text": "Literal page 1 Literal page 2"},
    }
    assert table["prediction"] == {
        "tables": [
            {
                "table_id": "table-001",
                "label": "Visible table",
                "image_id": "image-001",
                "rows": [{"row_id": "row-001", "label": "Visible row"}],
                "columns": [{"column_id": "column-001", "label": "Visible column"}],
                "cells": [
                    {
                        "row_id": "row-001",
                        "column_id": "column-001",
                        "text": "Predicted cell",
                    }
                ],
                "merged_ranges": [],
            }
        ]
    }
    serialized_outputs = "\n".join(
        path.read_text(encoding="utf-8")
        for directory in (predictions, audit)
        for path in sorted(directory.iterdir())
    )
    assert "source-" not in serialized_outputs
    assert "Secret cell 9917" not in serialized_outputs
    form_audit = json.loads((audit / "form-000001.json").read_text())
    assert form_audit["review_agreement"] == [{"page": 1, "agreed": True}]
    assert "not independent review" in form_audit["consistency_note"]
    assert form_audit["usage"] == {
        "prompt_tokens": 30,
        "completion_tokens": 15,
        "total_tokens": 45,
    }
    assert form_audit["cost"] == 0.003
    assert form_audit["latency_ms"] == 36.0


def test_group_split_is_source_disjoint_and_partitions_all_cases(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    for suffix in range(15):
        case_id = f"form-{suffix:06d}"
        _write_case(data, case_id, "forms", suffix, _form_target(case_id))
    _approve(data)

    def fake_repair(*args, **kwargs):
        return _result(
            {
                "controls": [
                    {
                        "label": "Visible alias",
                        "state": "checked",
                    }
                ]
            }
        )

    selected = {}
    for split in ("development", "calibration", "eval"):
        output = tmp_path / f"{split}-run"
        benchmark.run_benchmark(
            data,
            output,
            model=MODEL,
            provider=PROVIDER,
            split=split,
            repair=fake_repair,
        )
        predictions = output / "predictions"
        selected[split] = {path.stem for path in predictions.iterdir()}

    assert all(selected.values())
    assert not (selected["development"] & selected["calibration"])
    assert not (selected["development"] & selected["eval"])
    assert not (selected["calibration"] & selected["eval"])
    assert set.union(*selected.values()) == {f"form-{i:06d}" for i in range(15)}


def test_review_disagreement_abstains_and_keeps_audit(tmp_path: Path) -> None:
    data = tmp_path / "data"
    output = tmp_path / "run"
    _write_case(data, "form-000001", "forms", 1, _form_target())
    _approve(data)
    count = 0

    def fake_repair(*args, **kwargs):
        nonlocal count
        count += 1
        state = "unchecked" if count == 2 else "checked"
        return _result({"controls": [{"label": "Visible alias", "state": state}]})

    summary = benchmark.run_benchmark(
        data,
        output,
        model=MODEL,
        provider=PROVIDER,
        review_passes=3,
        split="all",
        repair=fake_repair,
    )

    assert count == 3
    assert summary == {"eligible_cases": 1, "submitted_cases": 0}
    predictions = output / "predictions"
    audit = output / "audit"
    assert list(predictions.iterdir()) == []
    record = json.loads((audit / "form-000001.json").read_text())
    assert record["status"] == "abstained"
    assert record["failure_reasons"] == ["review_disagreement"]
    assert record["review_agreement"] == [{"page": 1, "agreed": False}]
    assert len(record["calls"]) == 3


def test_refuses_existing_output_directory(tmp_path: Path) -> None:
    data = tmp_path / "data"
    output = tmp_path / "run"
    _write_case(data, "form-000001", "forms", 1, _form_target())
    _approve(data)
    output.mkdir()

    with pytest.raises(ValueError, match="Output directory must be new"):
        benchmark.run_benchmark(
            data,
            output,
            model=MODEL,
            provider=PROVIDER,
            split="all",
            repair=lambda *args, **kwargs: _result({"controls": []}),
        )


def test_rejects_unsupported_model_before_creating_outputs(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _write_case(data, "form-000001", "forms", 1, _form_target())
    predictions = tmp_path / "predictions"

    with pytest.raises(ValueError, match="Unsupported frontier model"):
        benchmark.run_benchmark(
            data,
            predictions,
            model="qwen/qwen3-32b",
            provider=PROVIDER,
        )
    assert not predictions.exists()


def test_requires_dataset_synthetic_external_processing_approval(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    _write_case(data, "form-000001", "forms", 1, _form_target())

    with pytest.raises(ValueError, match="Missing dataset approval record"):
        benchmark.run_benchmark(
            data,
            tmp_path / "run",
            model=MODEL,
            provider=PROVIDER,
            split="all",
        )


def test_real_private_dataset_is_rejected_without_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = Path(__file__).parents[1] / "internal-clinical-ocr-benchmark"
    assert data.is_dir()
    monkeypatch.setattr(
        benchmark, "APPROVAL_FILENAME", "missing-test-only-approval.json"
    )

    with pytest.raises(ValueError, match="Missing dataset approval record"):
        benchmark.run_benchmark(
            data,
            tmp_path / "run",
            model=MODEL,
            provider=PROVIDER,
            split="all",
            repair=lambda *args, **kwargs: pytest.fail("provider was called"),
        )


def test_approval_is_bound_to_exact_provider_and_rejects_extra_fields(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    _write_case(data, "form-000001", "forms", 1, _form_target())
    _approve(data, "another-provider")

    with pytest.raises(ValueError, match="does not match this root and provider"):
        benchmark.run_benchmark(
            data,
            tmp_path / "wrong-provider",
            model=MODEL,
            provider=PROVIDER,
            split="all",
        )

    approval_path = data / benchmark.APPROVAL_FILENAME
    approval = json.loads(approval_path.read_text())
    approval["api_key"] = "not-a-real-secret"
    approval_path.write_text(json.dumps(approval), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid fields"):
        benchmark.run_benchmark(
            data,
            tmp_path / "extra-field",
            model=MODEL,
            provider="another-provider",
            split="all",
        )

    del approval["api_key"]
    approval["data_root"] = str((tmp_path / "different-data").resolve())
    approval_path.write_text(json.dumps(approval), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match this root and provider"):
        benchmark.run_benchmark(
            data,
            tmp_path / "wrong-root",
            model=MODEL,
            provider="another-provider",
            split="all",
        )


def test_duplicate_form_labels_abstain_with_ambiguous_identity(tmp_path: Path) -> None:
    data = tmp_path / "data"
    output = tmp_path / "run"
    _write_case(data, "form-000001", "forms", 1, _duplicate_form_target())
    _approve(data)

    summary = benchmark.run_benchmark(
        data,
        output,
        model=MODEL,
        provider=PROVIDER,
        split="all",
        repair=lambda *args, **kwargs: _result(
            {
                "controls": [
                    {
                        "label": "Visible alias",
                        "state": "checked",
                    },
                    {
                        "label": "Visible alias",
                        "state": "unchecked",
                    },
                ]
            }
        ),
    )

    assert summary == {"eligible_cases": 1, "submitted_cases": 0}
    assert list((output / "predictions").iterdir()) == []
    audit = json.loads((output / "audit" / "form-000001.json").read_text())
    assert audit["failure_reasons"] == ["ambiguous_control_identity"]


def test_duplicate_form_identity_is_target_order_invariant(tmp_path: Path) -> None:
    controls = _duplicate_form_target()["controls"]
    raw = {
        "controls": [
            {"label": "Visible alias", "state": "checked"},
            {"label": "Visible alias", "state": "unchecked"},
        ]
    }
    results = []
    for index, ordered in enumerate((controls, list(reversed(controls)))):
        data = tmp_path / f"data-{index}"
        output = tmp_path / f"run-{index}"
        _write_case(data, "form-000001", "forms", index, {"controls": ordered})
        _approve(data)
        summary = benchmark.run_benchmark(
            data,
            output,
            model=MODEL,
            provider=PROVIDER,
            split="all",
            repair=lambda *args, **kwargs: _result(raw),
        )
        audit = json.loads((output / "audit" / "form-000001.json").read_text())
        results.append((summary, audit["failure_reasons"]))

    assert results == [
        ({"eligible_cases": 1, "submitted_cases": 0}, ["ambiguous_control_identity"]),
        ({"eligible_cases": 1, "submitted_cases": 0}, ["ambiguous_control_identity"]),
    ]


def test_reversed_unique_form_controls_map_by_visible_identity(tmp_path: Path) -> None:
    data = tmp_path / "data"
    output = tmp_path / "run"
    target = _form_target()
    target["controls"][0]["aliases"] = ["First visible label"]
    target["controls"].append(
        {
            "control_id": "control-002",
            "label": "Secret second label",
            "aliases": ["Second visible label"],
            "image_ids": ["image-001"],
            "state": "unchecked",
        }
    )
    _write_case(data, "form-000001", "forms", 1, target)
    _approve(data)
    raw = {
        "controls": [
            {
                "label": "Second visible label",
                "state": "unchecked",
            },
            {
                "label": "First visible label",
                "state": "checked",
            },
        ]
    }

    benchmark.run_benchmark(
        data,
        output,
        model=MODEL,
        provider=PROVIDER,
        split="all",
        repair=lambda *args, **kwargs: _result(raw),
    )

    prediction = json.loads((output / "predictions" / "form-000001.json").read_text())
    assert prediction["prediction"]["controls"] == [
        {"control_id": "control-001", "state": "checked"},
        {"control_id": "control-002", "state": "unchecked"},
    ]


def test_form_alias_matching_normalizes_unicode_case_and_whitespace(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    output = tmp_path / "run"
    target = _form_target()
    target["controls"][0]["aliases"] = ["Café   Control"]
    _write_case(data, "form-000001", "forms", 1, target)
    _approve(data)

    benchmark.run_benchmark(
        data,
        output,
        model=MODEL,
        provider=PROVIDER,
        split="all",
        repair=lambda *args, **kwargs: _result(
            {
                "controls": [
                    {
                        "label": "  CAFÉ\u00a0CONTROL  ",
                        "state": "checked",
                    }
                ]
            }
        ),
    )

    prediction = json.loads((output / "predictions" / "form-000001.json").read_text())
    assert prediction["prediction"]["controls"] == [
        {"control_id": "control-001", "state": "checked"}
    ]


def test_duplicate_table_axes_map_with_one_based_indices(tmp_path: Path) -> None:
    data = tmp_path / "data"
    output = tmp_path / "run"
    target = _table_target()
    table = target["tables"][0]
    table["rows"].append(
        {"row_id": "row-002", "label": "Secret row 2", "aliases": ["Repeat"]}
    )
    table["rows"][0]["aliases"] = ["Repeat"]
    table["columns"].append(
        {
            "column_id": "column-002",
            "label": "Secret column 2",
            "aliases": ["Duplicate"],
        }
    )
    table["columns"][0]["aliases"] = ["Duplicate"]
    table["cells"].append(
        {"row_id": "row-002", "column_id": "column-002", "text": "Secret"}
    )
    _write_case(data, "table-000001", "tables", 1, target)
    _approve(data)

    raw = {
        "tables": [
            {
                "label": "Visible table",
                "rows": [{"label": "Repeat"}, {"label": "Repeat"}],
                "columns": [{"label": "Duplicate"}, {"label": "Duplicate"}],
                "cells": [
                    {"row_index": 1, "column_index": 1, "text": "First"},
                    {"row_index": 2, "column_index": 2, "text": "Second"},
                ],
                "merged_ranges": [],
            }
        ]
    }
    benchmark.run_benchmark(
        data,
        output,
        model=MODEL,
        provider=PROVIDER,
        split="all",
        repair=lambda *args, **kwargs: _result(raw),
    )

    prediction = json.loads((output / "predictions" / "table-000001.json").read_text())[
        "prediction"
    ]["tables"][0]
    assert prediction["rows"] == [
        {"row_id": "row-001", "label": "Repeat"},
        {"row_id": "row-002", "label": "Repeat"},
    ]
    assert prediction["columns"] == [
        {"column_id": "column-001", "label": "Duplicate"},
        {"column_id": "column-002", "label": "Duplicate"},
    ]
    assert [(cell["row_id"], cell["column_id"]) for cell in prediction["cells"]] == [
        ("row-001", "column-001"),
        ("row-002", "column-002"),
    ]


def test_invented_control_abstains_instead_of_shifting(tmp_path: Path) -> None:
    data = tmp_path / "data"
    output = tmp_path / "run"
    _write_case(data, "form-000001", "forms", 1, _form_target())
    _approve(data)
    raw = {
        "controls": [
            {"label": "Visible alias", "state": "checked"},
            {"label": "Invented", "state": "unchecked"},
        ]
    }

    summary = benchmark.run_benchmark(
        data,
        output,
        model=MODEL,
        provider=PROVIDER,
        split="all",
        repair=lambda *args, **kwargs: _result(raw),
    )

    assert summary == {"eligible_cases": 1, "submitted_cases": 0}
    assert list((output / "predictions").iterdir()) == []
    audit = json.loads((output / "audit" / "form-000001.json").read_text())
    assert audit["failure_reasons"] == ["control_sequence_mismatch"]


def test_write_failure_removes_atomic_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "data"
    output = tmp_path / "run"
    _write_case(data, "form-000001", "forms", 1, _form_target())
    _approve(data)
    original = benchmark._write_json
    calls = 0

    def fail_second_write(path, value):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt("injected write failure")
        original(path, value)

    monkeypatch.setattr(benchmark, "_write_json", fail_second_write)
    with pytest.raises(KeyboardInterrupt, match="injected write failure"):
        benchmark.run_benchmark(
            data,
            output,
            model=MODEL,
            provider=PROVIDER,
            split="all",
            repair=lambda *args, **kwargs: _result(
                {
                    "controls": [
                        {
                            "label": "Visible alias",
                            "state": "checked",
                        }
                    ]
                }
            ),
        )

    assert not output.exists()
    assert list(tmp_path.glob(".run-*")) == []


def test_concurrent_empty_destination_is_not_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "data"
    output = tmp_path / "run"
    _write_case(data, "form-000001", "forms", 1, _form_target())
    _approve(data)
    publish = benchmark._publish_no_replace
    destination_inode = None

    def race(source: Path, destination: Path) -> None:
        nonlocal destination_inode
        destination.mkdir()
        destination_inode = destination.stat().st_ino
        publish(source, destination)

    monkeypatch.setattr(benchmark, "_publish_no_replace", race)
    with pytest.raises(FileExistsError):
        benchmark.run_benchmark(
            data,
            output,
            model=MODEL,
            provider=PROVIDER,
            split="all",
            repair=lambda *args, **kwargs: _result(
                {
                    "controls": [
                        {
                            "label": "Visible alias",
                            "state": "checked",
                        }
                    ]
                }
            ),
        )

    assert output.stat().st_ino == destination_inode
    assert list(output.iterdir()) == []
    assert list(tmp_path.glob(".run-*")) == []
