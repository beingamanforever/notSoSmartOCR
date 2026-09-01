from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from experiments import mpdocbench_export
from ocr_pipeline.contracts import BoundingBox, TextRegion


def test_exports_thirty_ordered_multi_page_documents(tmp_path: Path) -> None:
    image_root = tmp_path / "dataset"
    records = []
    for document_index in range(31):
        page_ids = [
            f"images/document-{document_index:02d}/page-2.png",
            f"images/document-{document_index:02d}/page-1.png",
        ]
        records.append(_record(f"pdfs/document-{document_index:02d}.pdf", page_ids))
        for page_id in page_ids:
            if document_index == 29 and page_id.endswith("page-1.png"):
                continue
            path = image_root / page_id
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (100, 60), "white").save(path)

    annotations = tmp_path / "MPDocBench.json"
    annotations.write_text(json.dumps(records), encoding="utf-8")
    reader = StructuredReader()
    output_dir = tmp_path / "predictions"

    report = mpdocbench_export.export_predictions(
        annotations,
        image_root,
        output_dir,
        reader,
        dataset_revision="dataset-commit-exact",
        evaluator_revision="evaluator-commit-exact",
        limit=30,
    )

    assert len(list(output_dir.glob("document-*.md"))) == 30
    assert not (output_dir / "document-30.md").exists()
    assert (output_dir / "document-00.md").read_text(encoding="utf-8") == (
        "# page 1\n\n"
        "<table><tr><td>page-2</td></tr></table>\n\n"
        "# page 2\n\n"
        "<table><tr><td>page-1</td></tr></table>\n"
    )
    assert reader.calls[:4] == [
        ("page-2.png", 1),
        ("page-1.png", 2),
        ("page-2.png", 1),
        ("page-1.png", 2),
    ]

    structured = json.loads(
        (output_dir / "structured" / "document-00.json").read_text(encoding="utf-8")
    )
    assert structured["schema_version"] == 2
    assert structured["document_id"] == "pdfs/document-00.pdf"
    assert [page["page_id"] for page in structured["pages"]] == records[0]["page_info"][
        "images_list"
    ]
    assert [page["page_number"] for page in structured["pages"]] == [1, 2]
    assert structured["pages"][0]["source"] == {
        "name": "page-2.png",
        "kind": "image",
    }
    assert [region["id"] for region in structured["pages"][0]["regions"]] == [
        "p1-heading",
        "p1-table",
    ]
    assert structured["pages"][0]["regions"][0]["text_provenance"] == {
        "method": "synthetic-reader",
        "source": "page-2.png",
    }
    assert structured["pages"][0]["regions"][0]["bounding_box"] == {
        "left": 5,
        "top": 5,
        "right": 95,
        "bottom": 25,
    }
    assert structured["pages"][0]["route"] == "accept_local"
    assert structured["geometry_availability"] == "available"

    partial = json.loads(
        (output_dir / "structured" / "document-29.json").read_text(encoding="utf-8")
    )
    assert partial["status"] == "partial"
    assert partial["covered"] is True
    assert partial["pages"][1]["page_id"].endswith("page-1.png")
    assert partial["pages"][1]["status"] == "failed"
    assert partial["pages"][1]["route"] == "review"
    assert partial["pages"][1]["failure_ids"] == ["p2-failure-1"]
    assert partial["pages"][1]["failures"][0]["code"] == "source_not_found"
    assert partial["pages"][1]["failures"][0]["page_number"] == 2

    assert report["dataset_revision"] == "dataset-commit-exact"
    assert report["evaluator_revision"] == "evaluator-commit-exact"
    assert report["run_config"] == {
        "reader": "structured",
        "reader_options": {},
        "limit": 30,
    }
    assert report["document_ids"] == [
        f"pdfs/document-{index:02d}.pdf" for index in range(30)
    ]
    assert report["page_ids"] == [
        page_id
        for record in records[:30]
        for page_id in record["page_info"]["images_list"]
    ]
    assert {
        key: report[key]
        for key in (
            "attempted",
            "covered",
            "abstained",
            "success",
            "partial",
            "failed",
        )
    } == {
        "attempted": 30,
        "covered": 30,
        "abstained": 0,
        "success": 29,
        "partial": 1,
        "failed": 0,
    }
    assert report["page_counts"] == {
        "attempted": 60,
        "covered": 59,
        "abstained": 1,
        "success": 59,
        "partial": 0,
        "failed": 1,
    }
    assert set(report["latency_ms"]) == {"document", "page"}
    assert set(report["latency_ms"]["document"]) == {"p50", "p95"}
    assert set(report["latency_ms"]["page"]) == {"p50", "p95"}
    assert all(
        set(document["pages"][0])
        == {
            "page_id",
            "image_path",
            "page_number",
            "status",
            "covered",
            "abstained",
            "route",
            "failures",
            "latency_ms",
        }
        for document in report["documents"]
    )


@pytest.mark.parametrize("existing_kind", ["directory", "file"])
def test_refuses_to_overwrite_existing_run(tmp_path: Path, existing_kind: str) -> None:
    annotations = tmp_path / "MPDocBench.json"
    annotations.write_text(
        json.dumps([_record("document.pdf", ["images/page.png"])]),
        encoding="utf-8",
    )
    output_dir = tmp_path / "predictions"
    if existing_kind == "directory":
        output_dir.mkdir()
        marker = output_dir / "keep.txt"
        marker.write_text("keep", encoding="utf-8")
    else:
        output_dir.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError):
        mpdocbench_export.export_predictions(
            annotations,
            tmp_path,
            output_dir,
            StructuredReader(),
            dataset_revision="dataset-revision",
            evaluator_revision="evaluator-revision",
        )

    if existing_kind == "directory":
        assert marker.read_text(encoding="utf-8") == "keep"
        assert list(output_dir.iterdir()) == [marker]
    else:
        assert output_dir.read_text(encoding="utf-8") == "keep"


def test_rejects_duplicate_prediction_names_before_writing(tmp_path: Path) -> None:
    annotations = tmp_path / "MPDocBench.json"
    annotations.write_text(
        json.dumps(
            [
                _record("first/same.pdf", ["images/first.png"]),
                _record("second/same.pdf", ["images/second.png"]),
            ]
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "predictions"

    with pytest.raises(ValueError, match="duplicate prediction names"):
        mpdocbench_export.export_predictions(
            annotations,
            tmp_path,
            output_dir,
            StructuredReader(),
            dataset_revision="dataset-revision",
            evaluator_revision="evaluator-revision",
        )

    assert not output_dir.exists()


def test_official_420_document_controls_keep_all_3135_pages(tmp_path: Path) -> None:
    records = [_control_record(index) for index in range(420)]
    annotations = tmp_path / "MPDocBench.json"
    annotations.write_text(json.dumps(records), encoding="utf-8")

    oracle = mpdocbench_export.export_control_predictions(
        annotations,
        tmp_path / "oracle",
        evaluator_root=tmp_path / "official",
        control="oracle",
        dataset_revision="dataset-revision",
        evaluator_revision="evaluator-revision",
        official_metrics=FakeOfficialMetrics(),
    )
    empty = mpdocbench_export.export_control_predictions(
        annotations,
        tmp_path / "empty",
        evaluator_root=tmp_path / "official",
        control="empty",
        dataset_revision="dataset-revision",
        evaluator_revision="evaluator-revision",
        official_metrics=FakeOfficialMetrics(),
    )

    assert oracle["attempted"] == empty["attempted"] == 420
    assert oracle["page_counts"]["attempted"] == 3135
    assert empty["page_counts"] == {
        "attempted": 3135,
        "covered": 0,
        "abstained": 3135,
    }
    assert len(oracle["documents"]) == len(empty["documents"]) == 420
    assert len(list((tmp_path / "oracle").glob("document-*.md"))) == 420
    assert len(list((tmp_path / "empty").glob("document-*.md"))) == 420
    assert empty["failed"] == 420
    assert "missing-equivalent predictions remain" in empty["failure_policy"]

    assert oracle["metrics"]["HeadTEDS"] == {"all": 1.0, "documents": 420}
    assert empty["metrics"]["HeadTEDS"] == {"all": 0.0, "documents": 420}
    assert oracle["metrics"]["text_relation"]["Relation_F1_cross_page"]["f1"] == 1.0
    assert empty["metrics"]["text_relation"]["Relation_F1_cross_page"]["fn"] == 1
    assert oracle["metrics"]["table_relation"]["Relation_F1_cross_page"]["f1"] == 1.0
    assert empty["metrics"]["table_relation"]["Relation_F1_cross_page"]["fn"] == 1
    assert oracle["metrics"]["merged_text_block"]["Edit_dist"] == {
        "edit_sample_avg": 0.0,
        "instances": 1,
        "cross_page_instances": 1,
    }
    assert empty["metrics"]["merged_text_block"]["Edit_dist"]["edit_sample_avg"] == 1.0
    assert oracle["metrics"]["merged_text_block"]["continuation_accuracy"] == {
        "all": 1.0,
        "cross_page": 1.0,
        "definition": "project-defined exact text match",
    }
    assert empty["metrics"]["merged_text_block"]["continuation_accuracy"]["all"] == 0.0
    assert oracle["metrics"]["merged_table"]["TEDS"]["all"] == 1.0
    assert empty["metrics"]["merged_table"]["TEDS"]["all"] == 0.0
    assert oracle["metrics"]["merged_table"]["continuation_accuracy"] == {
        "all": 1.0,
        "cross_page": 1.0,
        "definition": "project-defined exact normalized HTML match",
    }
    assert empty["metrics"]["merged_table"]["continuation_accuracy"]["all"] == 0.0


class FakeHeadTEDS:
    def evaluate(
        self, prediction: list[str], ground_truth: list[object]
    ) -> tuple[float, int]:
        relations = ground_truth[0]
        headings = ground_truth[1]
        children: dict[object, list[object]] = {}
        for relation in relations:
            children.setdefault(relation["source_anno_id"], []).append(
                relation["target_anno_id"]
            )
        for values in children.values():
            values.sort()
        expected = []

        def visit(annotation_id: object, depth: int) -> None:
            expected.append(f"{'#' * depth} {headings[annotation_id]['text']}")
            for child in children.get(annotation_id, []):
                visit(child, depth + 1)

        for root in children.get("root", []):
            visit(root, 1)
        return (float(prediction == expected), 1)


class FakeTableTEDS:
    def evaluate(self, prediction: str, ground_truth: str) -> float:
        return float(bool(prediction) and prediction == ground_truth)


class FakeOfficialMetrics:
    def __init__(self) -> None:
        self.head_teds = FakeHeadTEDS()
        self.table_teds = FakeTableTEDS()

    def extract_table_relations(
        self, record: dict[str, object]
    ) -> tuple[set[tuple[object, object]], list[set[object]], dict[object, object]]:
        return self._extract(record, "table")

    def extract_text_relations(
        self, record: dict[str, object]
    ) -> tuple[set[tuple[object, object]], list[set[object]], dict[object, object]]:
        return self._extract(record, "text_block")

    def detect_relations(
        self,
        record: dict[str, object],
        category: str,
        predicted_values: list[str],
    ) -> set[tuple[object, object]]:
        if not predicted_values:
            return set()
        relation_type = "table" if category == "table" else "text_block"
        return self._extract(record, relation_type)[0]

    @staticmethod
    def normalize_table_for_teds(value: str) -> str:
        return value

    @staticmethod
    def split_pairs_by_page(
        pairs: set[tuple[object, object]], annotation_map: dict[object, object]
    ) -> tuple[set[tuple[object, object]], set[tuple[object, object]]]:
        same = set()
        cross = set()
        for pair in pairs:
            source, target = (annotation_map[item] for item in pair)
            (same if source["page_id"] == target["page_id"] else cross).add(pair)
        return same, cross

    @staticmethod
    def relation_f1(
        ground_truth: set[tuple[object, object]],
        prediction: set[tuple[object, object]],
    ) -> dict[str, object]:
        true_positive = len(ground_truth & prediction)
        false_positive = len(prediction - ground_truth)
        false_negative = len(ground_truth - prediction)
        precision = true_positive / len(prediction) if prediction else 0.0
        recall = true_positive / len(ground_truth) if ground_truth else 0.0
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        return {
            "tp": true_positive,
            "fp": false_positive,
            "fn": false_negative,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    @staticmethod
    def text_edit_distance(ground_truth: str, prediction: str) -> float:
        return 0.0 if ground_truth == prediction else 1.0

    @staticmethod
    def _extract(
        record: dict[str, object], category: str
    ) -> tuple[set[tuple[object, object]], list[set[object]], dict[object, object]]:
        annotation_map = {item["anno_id"]: item for item in record["layout_dets"]}
        pairs = {
            (relation["source_anno_id"], relation["target_anno_id"])
            for relation in record["extra"]["relation"]
            if relation["relation_type"] == "truncated"
            and annotation_map[relation["source_anno_id"]]["category_type"] == category
        }
        groups = [set(pair) for pair in pairs]
        return pairs, groups, annotation_map


class StructuredReader:
    name = "structured"

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
        self.calls.append((image_path.name, page_number))
        provenance = {"method": "synthetic-reader", "source": image_path.name}
        return [
            TextRegion(
                id=f"p{page_number}-table",
                kind="table",
                text=f"<table><tr><td>{image_path.stem}</td></tr></table>",
                confidence=0.8,
                bounding_box=BoundingBox(5, 30, 95, 55),
                reading_order=2,
                provider=self.name,
                text_provenance=provenance,
            ),
            TextRegion(
                id=f"p{page_number}-heading",
                kind="title",
                text=f"# page {page_number}",
                confidence=0.9,
                bounding_box=BoundingBox(5, 5, 95, 25),
                reading_order=1,
                provider=self.name,
                text_provenance=provenance,
            ),
        ]


def _control_record(index: int) -> dict[str, object]:
    page_count = 8 if index < 195 else 7
    page_ids = [
        f"images/document-{index:03d}/page-{page:02d}.png"
        for page in range(1, page_count + 1)
    ]
    layout = [
        {
            "anno_id": 1,
            "category_type": "title",
            "page_id": 1,
            "order": 1,
            "text": f"Document {index}" + ("\nAnnual report" if index == 0 else ""),
        }
    ]
    relations = [
        {
            "source_anno_id": "root",
            "target_anno_id": 1,
            "relation_type": "parent_son",
        }
    ]
    if index == 0:
        layout.extend(
            [
                {
                    "anno_id": 2,
                    "category_type": "text_block",
                    "page_id": 1,
                    "order": 2,
                    "text": "continued ",
                },
                {
                    "anno_id": 3,
                    "category_type": "text_block",
                    "page_id": 2,
                    "order": 3,
                    "text": "text",
                },
                {
                    "anno_id": 4,
                    "category_type": "table",
                    "page_id": 1,
                    "order": 4,
                    "html": "<table><tr><td>A</td></tr></table>",
                    "merged_html": "<table><tr><td>A</td><td>B</td></tr></table>",
                },
                {
                    "anno_id": 5,
                    "category_type": "table",
                    "page_id": 2,
                    "order": 5,
                    "html": "<table><tr><td>B</td></tr></table>",
                    "merged_html": "<table><tr><td>A</td><td>B</td></tr></table>",
                },
            ]
        )
        relations.extend(
            [
                {
                    "source_anno_id": 2,
                    "target_anno_id": 3,
                    "relation_type": "truncated",
                },
                {
                    "source_anno_id": 4,
                    "target_anno_id": 5,
                    "relation_type": "truncated",
                },
            ]
        )
    return {
        "layout_dets": layout,
        "extra": {"relation": relations},
        "page_info": {
            "image_path": f"pdfs/document-{index:03d}.pdf",
            "images_list": page_ids,
        },
    }


def _record(image_path: str, page_ids: list[str]) -> dict[str, object]:
    return {
        "layout_dets": [],
        "page_info": {
            "image_path": image_path,
            "images_list": page_ids,
        },
    }
