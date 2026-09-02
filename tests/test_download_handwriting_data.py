from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path

import pytest

from experiments import download_handwriting_data as public_data


def test_cli_lists_catalog_and_plans_small_subset_without_writing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert public_data.main(["--list"]) == 0
    catalog = json.loads(capsys.readouterr().out)
    assert {item["id"] for item in catalog} >= {
        "read-2017-train-a",
        "nist-sd19-by-write",
        "nist-sd6",
    }

    output = tmp_path / "public"
    assert public_data.main([str(output), "--item", "read-2017-train-a"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["download"] is False
    assert plan["total_bytes"] == public_data.READ_TRAIN_A_BYTES
    assert plan["items"][0]["groups"]["document"].startswith("PAGE XML")
    assert not output.exists()


def test_manual_selection_keeps_writer_and_document_groups(tmp_path: Path) -> None:
    selection = tmp_path / "selection.jsonl"
    selection.write_text(
        json.dumps(
            {
                "id": "chosen-page",
                "source": "Example public source",
                "url": "https://example.org/chosen-page.png",
                "filename": "chosen-page.png",
                "declared_bytes": 4,
                "groups": {
                    "writer": "writer-7",
                    "document": "document-3",
                    "page": "page-1",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    item = public_data.load_selection(selection)[0]

    assert item.groups == {
        "writer": "writer-7",
        "document": "document-3",
        "page": "page-1",
    }


def test_default_limit_blocks_bulk_and_manual_private_s3(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exceeds --max-bytes"):
        public_data.validate_plan(
            [public_data.CATALOG["nist-sd19-by-write"]],
            public_data.DEFAULT_MAX_BYTES,
        )

    selection = tmp_path / "selection.jsonl"
    selection.write_text(
        json.dumps(
            {
                "id": "private-object",
                "source": "manual",
                "url": (
                    "https://private.s3.amazonaws.com/data.zip?X-Amz-Signature=secret"
                ),
                "filename": "data.zip",
                "declared_bytes": 10,
                "groups": {"writer": None, "document": "archive"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="private S3"):
        public_data.load_selection(selection)


def test_download_is_bounded_and_records_source_groups(tmp_path: Path) -> None:
    item = public_data.DownloadItem(
        id="chosen-page",
        source="Example public source",
        url="https://example.org/chosen-page.png",
        filename="chosen-page.png",
        declared_bytes=4,
        groups={"writer": "writer-7", "document": "document-3"},
    )

    records = public_data.download(
        [item],
        tmp_path,
        max_bytes=4,
        open_url=lambda request, timeout: _Response(
            b"data", request.full_url, {"Content-Length": "4"}
        ),
    )

    assert (tmp_path / "chosen-page.png").read_bytes() == b"data"
    assert records[0]["groups"] == {
        "writer": "writer-7",
        "document": "document-3",
    }
    saved = json.loads((tmp_path / "downloads.jsonl").read_text(encoding="utf-8"))
    assert saved == records[0]
    assert saved["actual_bytes"] == 4


def test_download_rejects_a_response_larger_than_declared(tmp_path: Path) -> None:
    item = public_data.DownloadItem(
        id="chosen-page",
        source="Example public source",
        url="https://example.org/chosen-page.png",
        filename="chosen-page.png",
        declared_bytes=4,
        groups={"writer": None, "document": "document-3"},
    )

    with pytest.raises(ValueError, match="Content-Length differs"):
        public_data.download(
            [item],
            tmp_path,
            max_bytes=5,
            open_url=lambda request, timeout: _Response(
                b"extra", request.full_url, {"Content-Length": "5"}
            ),
        )

    assert not (tmp_path / "chosen-page.png").exists()


class _Response(BytesIO):
    def __init__(self, data: bytes, url: str, headers: dict[str, str]) -> None:
        super().__init__(data)
        self.url = url
        self.headers = headers

    def geturl(self) -> str:
        return self.url
