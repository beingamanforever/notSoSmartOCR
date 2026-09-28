"""Exercise PDF ingestion, HTTP serving, two model calls and private review export."""

import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import tempfile
import threading
from unittest.mock import patch

import fitz
from openpyxl import Workbook
from PIL import Image

from experiments.benchmark_clinical_handwriting import (
    run,
    evaluate_saved_run,
    replay_saved_refinements,
    span_distance,
    score_snippets,
)


def test_structured_adapters_do_not_copy_target_answers(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).parents[1] / "internal-clinical-ocr-benchmark")
    )
    from experiments.evaluate_structured_handwritten import (
        adapt_form,
        adapt_tables,
    )

    form_page = {
        "regions": [
            {
                "kind": "checkbox",
                "text": "Consent",
                "structure": {"label": "Consent", "state": "selected"},
                "bounding_box": {"left": 0, "top": 0},
            }
        ]
    }
    form = {
        "target": {
            "controls": [
                {
                    "control_id": "control-1",
                    "label": "Consent",
                    "state": "checked",
                    "image_ids": ["image-1"],
                }
            ]
        }
    }
    changed_form = json.loads(json.dumps(form))
    changed_form["target"]["controls"][0]["state"] = "unchecked"
    pages = {"image-1": form_page}
    assert adapt_form(form, pages) == adapt_form(changed_form, pages)

    table_page = {
        "regions": [
            {
                "kind": "table",
                "bounding_box": {"left": 0, "top": 0, "right": 100, "bottom": 100},
                "structure": {
                    "cells": [
                        {"row_nums": [0], "column_nums": [0], "text": "Name"},
                        {"row_nums": [0], "column_nums": [1], "text": "Value"},
                        {"row_nums": [1], "column_nums": [0], "text": "Alice"},
                        {"row_nums": [1], "column_nums": [1], "text": "42"},
                    ]
                },
            },
            {
                "kind": "table",
                "bounding_box": {
                    "left": 120,
                    "top": 0,
                    "right": 220,
                    "bottom": 200,
                },
                "structure": {
                    "cells": [
                        {"row_nums": [0], "column_nums": [0], "text": "Name"},
                        {"row_nums": [0], "column_nums": [1], "text": "Value"},
                        {"row_nums": [1], "column_nums": [0], "text": "Bob"},
                        {"row_nums": [1], "column_nums": [1], "text": "7"},
                        {"row_nums": [2], "column_nums": [0], "text": "Extra"},
                    ]
                },
            },
        ]
    }
    table = {
        "target": {
            "tables": [
                {
                    "table_id": "table-1",
                    "image_id": "image-1",
                    "label": "Example",
                    "rows": [
                        {"row_id": "row-1", "label": "Name"},
                        {"row_id": "row-2", "label": "Alice"},
                        {"row_id": "row-3", "label": "Alice"},
                    ],
                    "columns": [
                        {"column_id": "column-1", "label": "Name"},
                        {"column_id": "column-2", "label": "Value"},
                    ],
                    "cells": [{"text": "target answer"}],
                }
            ]
        }
    }
    changed_table = json.loads(json.dumps(table))
    changed_table["target"]["tables"][0]["cells"][0]["text"] = "changed answer"
    changed_table["target"]["tables"][0]["rows"].append(
        {"row_id": "row-4", "label": "Not visible"}
    )
    changed_table["target"]["tables"][0]["columns"].append(
        {"column_id": "column-3", "label": "Also missing"}
    )
    prediction = adapt_tables(table, {"image-1": table_page})
    assert prediction == adapt_tables(changed_table, {"image-1": table_page})
    assert [row["row_id"] for row in prediction["tables"][0]["rows"]] == [
        "row-1",
        "row-2",
    ]


def test_structured_adapters_pair_repeated_labels_and_unlabelled_rows(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).parents[1] / "internal-clinical-ocr-benchmark")
    )
    from experiments.evaluate_structured_handwritten import adapt_form, adapt_tables

    def box(top):
        return {"left": 0, "top": top, "right": 10, "bottom": top + 5}

    page = {
        "regions": [
            {"kind": "checkbox", "bounding_box": box(top), "structure": {"label": label, "state": state}}
            for top, label, state in (
                (0, "Yes", "selected"),
                (10, "Yes", "unselected"),
                (20, "Diabetes:", "selected"),
            )
        ]
    }
    controls = [
        {"control_id": f"c{i}", "label": label, "image_ids": ["image-1"]}
        for i, label in enumerate(("Yes", "Yes", "Diabetes"))
    ]
    form = {"target": {"controls": controls}}
    assert adapt_form(form, {"image-1": page})["controls"] == [
        {"control_id": "c0", "state": "checked"},
        {"control_id": "c1", "state": "unchecked"},
        {"control_id": "c2", "state": "checked"},
    ]

    table_page = {
        "regions": [
            {"kind": "section-header", "text": "Vitals", "bounding_box": box(0)},
            {
                "kind": "table",
                "bounding_box": {"left": 0, "top": 10, "right": 100, "bottom": 100},
                "structure": {
                    "cells": [
                        {"row_nums": [r], "column_nums": [c], "text": text}
                        for r, c, text in (
                            (0, 0, "Item"), (0, 1, "Value"),
                            (1, 0, "BP"), (1, 1, "120/80"),
                            (2, 0, ""), (2, 1, "98.6"),
                        )
                    ]
                },
            },
        ]
    }
    table = {
        "target": {
            "tables": [
                {
                    "table_id": "t1",
                    "image_id": "image-1",
                    "label": "Vitals",
                    "rows": [
                        {"row_id": "row-1", "label": "Item"},
                        {"row_id": "row-2", "label": "BP"},
                        {"row_id": "row-3", "label": "row-3"},
                    ],
                    "columns": [
                        {"column_id": "column-1", "label": "Item"},
                        {"column_id": "column-2", "label": "Value"},
                    ],
                    "cells": [
                        {"row_id": "row-3", "column_id": "column-2", "text": "98.6"},
                    ],
                }
            ]
        }
    }
    table_page["regions"][1]["structure"]["cells"][2]["text"] = "B P"  # one-space misread
    table["target"]["tables"][0]["rows"][1]["label"] = "B.P"
    predicted = adapt_tables(table, {"image-1": table_page})["tables"][0]
    assert predicted["label"] == "Vitals"
    assert {"row_id": "row-3", "column_id": "column-2", "text": "98.6"} in predicted["cells"]


def test_comparison_end_to_end():
    assert span_distance("dose 5 mg", "header dose 5 mg footer") == 0
    assert span_distance("dose 5 mg", "header dose 4 mg footer") == 1
    assert span_distance("dose 5 mg".split(), "header dose 4 mg footer".split()) == 1
    assert span_distance("dose", "") == 4
    scores = score_snippets(
        "**HTN**, <b>dose</b> 5 mg", [{"row": 1, "text": "HTN", "scorable": True}]
    )
    assert scores[0]["recovered"] is True
    assert scores[0]["best_span_word_errors"] == 0
    assert scores[0]["best_word_span"] == "HTN"
    boxed = score_snippets(
        "ID [6][7][3][1][6][3]", [{"row": 2, "text": "673163", "scorable": True}]
    )[0]
    assert boxed["recovered"] is False
    assert boxed["format_insensitive_presence"] is True
    duplicate = score_snippets(
        "Cert",
        [
            {"row": 3, "text": "Cert", "scorable": True},
            {"row": 4, "text": "Cert", "scorable": True},
        ],
    )
    assert [score["recovered"] for score in duplicate] == [True, False]
    assert [score["format_insensitive_presence"] for score in duplicate] == [
        True,
        False,
    ]
    substring = score_snippets(
        "effectiveness", [{"row": 5, "text": "effective", "scorable": True}]
    )[0]
    assert substring["recovered"] is False
    assert substring["format_insensitive_presence"] is False
    overlapping = score_snippets(
        "cert period",
        [
            {"row": 6, "text": "cert", "scorable": True},
            {"row": 7, "text": "cert period", "scorable": True},
        ],
    )
    assert [score["recovered"] for score in overlapping] == [False, True]
    # Placing the most references beats greedy longest-first placement.
    shared = score_snippets(
        "John Smith Jr\n\nJohn Smith",
        [
            {"row": 8, "text": "John Smith", "scorable": True},
            {"row": 9, "text": "Smith Jr", "scorable": True},
        ],
    )
    assert [score["recovered"] for score in shared] == [True, True]
    # List markers are normalized identically in the reference and the prediction.
    listed = score_snippets(
        "- Aspirin 81 mg\n\n1. Walker daily",
        [
            {"row": 10, "text": "- Aspirin 81 mg", "scorable": True},
            {"row": 11, "text": "1. Walker daily", "scorable": True},
        ],
    )
    assert [score["recovered"] for score in listed] == [True, True]

    class Handler(BaseHTTPRequestHandler):
        falcon_model = "tiiuae/Falcon-OCR"
        result_status = "success"

        def log_message(self, *args):
            pass

        def do_GET(self):
            self.reply({"reader": "test-fixture"})

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            assert b"image/png" in body
            self.reply(
                {
                    "composition": {"primary_ocr": "Falcon OCR with Qwen crops"},
                    "stage_execution": [
                        {
                            "stage": "qwen-crop-refinement",
                            "status": "productive",
                            "attempts": 1,
                            "cost_usd": 0.001,
                            "accepted_crop_ids": ["crop-1"],
                            "rejected_crop_ids": [],
                        }
                    ],
                    "result": {
                        "status": self.result_status,
                        "pages": [
                            {
                                "page_number": 1,
                                "width": 600,
                                "height": 600,
                                "regions": [
                                    {
                                        "id": "p1-r1",
                                        "kind": "handwriting",
                                        "text": "dose 5 mg",
                                        "resolution": "resolved",
                                        "text_provenance": {
                                            "model": {"id": self.falcon_model},
                                            "generation": {"generated_tokens": 4},
                                        },
                                        "bounding_box": {
                                            "left": 0,
                                            "top": 0,
                                            "right": 600,
                                            "bottom": 300,
                                        },
                                    }
                                ],
                            }
                        ],
                    },
                }
            )

        def reply(self, value):
            body = json.dumps(value).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    requests = []

    def transport(request, timeout):
        payload = json.loads(request.data)
        assert payload["messages"][0]["role"] == "system"
        requests.append(payload)
        image_url = payload["messages"][-1]["content"][-1]["image_url"]["url"]
        with Image.open(io.BytesIO(base64.b64decode(image_url.split(",")[1]))) as image:
            assert image.width == 600
        assert "reference" not in json.dumps(payload).lower()
        content = {"markdown": "dose 5 mg"}
        return (
            200,
            {},
            json.dumps(
                {
                    "model": payload["model"],
                    "provider": "fixture",
                    "choices": [
                        {
                            "message": {"content": json.dumps(content)},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 300,
                        "completion_tokens": 12,
                        "cost": 0.001,
                    },
                }
            ).encode(),
        )

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.dict(os.environ, OPENROUTER_API_KEY="fixture-secret"),
        ):
            root = Path(temporary)
            sources = root / "sources"
            sources.mkdir()
            with fitz.open() as pdf:
                page = pdf.new_page(width=144, height=144)
                page.insert_text((10, 25), "dose 5 mg")
                pdf.save(sources / "example.pdf")
            workbook = Workbook()
            workbook.active.append(
                ["sr.no", "File Name", "Page No", "Screenshot Name", "Handwritten Text"]
            )
            workbook.active.append([1, "example", 1, "Dose", "dose 5 mg"])
            workbook.active.append([None, None, None, "Missing", "never present"])
            gt = root / "gt.xlsx"
            workbook.save(gt)
            output = root / "output"
            run(
                sources, gt, output, f"http://127.0.0.1:{server.server_port}", transport
            )
            report = json.loads((output / "report.json").read_text())
            assert report["status"] == "complete"
            evaluated = evaluate_saved_run(output)
            assert evaluated["status"] == "complete"
            assert evaluated["summary"]["pipeline"]["model_requests"] == 1
            assert evaluated["summary"]["full-page"]["model_requests"] == 1
            assert (output / "ground-truth-metrics.csv").exists()
            replay = replay_saved_refinements(output, root / "replay")
            assert replay["replayed_requests"] == 0
            assert replay["additional_api_cost_usd"] == 0
            assert len(requests) == 1
            assert all(p["provider"]["zdr"] is True for p in requests)
            for arm in ("pipeline", "full-page"):
                assert report["summary"][arm]["reference_snippets_recovered"] == 1
                assert report["summary"][arm]["scorable_snippets"] == 2
                assert report["summary"][arm]["known_api_cost_usd"] == 0.001
            case = output / "d001-p001"
            assert (case / "full-page-sent-1.png").read_bytes() == (
                case / "full-page.png"
            ).read_bytes()
            assert "dose 5 mg" in (output / "review.html").read_text()
            for path in output.rglob("*.json"):
                assert "fixture-secret" not in path.read_text()

            Handler.result_status = "partial"
            partial_output = root / "partial-output"
            run(
                sources,
                gt,
                partial_output,
                f"http://127.0.0.1:{server.server_port}",
                arms=("pipeline",),
            )
            partial = json.loads((partial_output / "report.json").read_text())
            assert partial["pages"][0]["arms"]["pipeline"]["status"] == "partial"
            assert partial["summary"]["pipeline"]["reference_snippets_recovered"] == 1
            Handler.result_status = "success"

            def failed_transport(request, timeout):
                status, headers, body = transport(request, timeout)
                payload = json.loads(body)
                payload["choices"][0]["finish_reason"] = "length"
                return status, headers, json.dumps(payload).encode()

            failed_output = root / "failed-output"
            run(
                sources,
                gt,
                failed_output,
                f"http://127.0.0.1:{server.server_port}",
                failed_transport,
                zero_data_retention=False,
            )
            failed = json.loads((failed_output / "report.json").read_text())
            assert failed["summary"]["pipeline"]["successful"] == 1
            assert failed["summary"]["full-page"]["successful"] == 0
            for arm in ("pipeline", "full-page"):
                assert failed["summary"][arm]["scorable_snippets"] == 2
                assert failed["summary"][arm]["known_api_cost_usd"] == 0.001
            assert len(requests) == 2
            assert requests[1]["provider"]["zdr"] is False
            assert all(p["provider"]["data_collection"] == "deny" for p in requests)
            Handler.falcon_model = "unexpected-reader"
            unverified_output = root / "unverified-output"
            run(
                sources,
                gt,
                unverified_output,
                f"http://127.0.0.1:{server.server_port}",
                transport,
            )
            unverified = json.loads((unverified_output / "report.json").read_text())
            assert unverified["pages"][0]["arms"]["pipeline"]["status"] == "failed"
            assert (
                "Falcon decoder execution evidence"
                in unverified["pages"][0]["arms"]["pipeline"]["error"]
            )
            assert not (unverified_output / "d001-p001/pipeline-request.json").exists()
            assert len(requests) == 3
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    test_comparison_end_to_end()
