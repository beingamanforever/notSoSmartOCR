from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from experiments.run_challenge_set import run_challenge_set


def test_runs_each_page_once_and_resumes_without_rewriting(tmp_path: Path) -> None:
    sources = tmp_path / "sources"
    category = sources / "C01-forms"
    category.mkdir(parents=True)
    for case in ("C01-D001-P001", "C01-D002-P001"):
        Image.new("RGB", (20, 30), "white").save(category / f"{case}.png")

    posted: list[Path] = []
    deleted: list[str] = []

    def post(base_url: str, image: Path, timeout: float) -> dict[str, Any]:
        assert base_url == "http://127.0.0.1:8080"
        assert timeout == 600
        assert image.read_bytes().startswith(b"\x89PNG")
        posted.append(image)
        return {
            "session_id": f"session-{len(posted)}",
            "result": {"status": "success", "schema_version": 2, "pages": []},
            "timing": {"total_seconds": 0.1},
        }

    def delete(base_url: str, session_id: str, timeout: float) -> None:
        assert base_url == "http://127.0.0.1:8080"
        assert timeout == 600
        deleted.append(session_id)

    output = tmp_path / "run" / "model-output"
    first = run_challenge_set(sources, output, post=post, delete=delete)
    second = run_challenge_set(sources, output, post=post, delete=delete)

    assert first == {"eligible": 2, "completed": 2, "failed": 0, "skipped": 0}
    assert second == {"eligible": 2, "completed": 0, "failed": 0, "skipped": 2}
    assert len(posted) == 2
    assert deleted == ["session-1", "session-2"]
    outputs = sorted(output.rglob("*.json"))
    assert [path.stem for path in outputs] == [
        "C01-D001-P001",
        "C01-D002-P001",
    ]
    assert json.loads(outputs[0].read_text())["result"]["schema_version"] == 2
