"""Run the local OCR workbench over an immutable challenge panel."""

from __future__ import annotations

import argparse
from collections.abc import Callable
import json
from pathlib import Path
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import uuid


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = run_challenge_set(
        args.sources,
        args.output,
        base_url=args.base_url,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["failed"] == 0 else 1


def run_challenge_set(
    source_root: Path,
    output_root: Path,
    *,
    base_url: str = "http://127.0.0.1:8080",
    timeout_seconds: float = 600,
    post: Callable[[str, Path, float], dict[str, Any]] | None = None,
    delete: Callable[[str, str, float], None] | None = None,
) -> dict[str, int]:
    if not source_root.is_dir():
        raise FileNotFoundError(f"Challenge sources not found: {source_root}")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if not _is_loopback(base_url):
        raise ValueError("Private challenge runs require a loopback endpoint")

    post = post or _post_file
    delete = delete or _delete_session
    images = sorted(
        (
            path
            for path in source_root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ),
        key=lambda path: _relative_key(path, source_root),
    )
    completed = 0
    failed = 0
    skipped = 0
    for index, image in enumerate(images, start=1):
        relative = image.relative_to(source_root)
        output = output_root / relative.with_suffix(".json")
        if output.exists():
            skipped += 1
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        session_id: str | None = None
        try:
            payload = post(base_url, image, timeout_seconds)
            session_id = payload.get("session_id")
            record = payload
            completed += 1
            status = "success"
        except (HTTPError, URLError, TimeoutError, ValueError) as error:
            record = {
                "case_id": image.stem,
                "request_status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
            failed += 1
            status = "failed"
        _write_once(output, record)
        if session_id:
            delete(base_url, session_id, timeout_seconds)
        print(f"[{index}/{len(images)}] {image.stem}: {status}", flush=True)

    return {
        "eligible": len(images),
        "completed": completed,
        "failed": failed,
        "skipped": skipped,
    }


def _post_file(base_url: str, image: Path, timeout_seconds: float) -> dict[str, Any]:
    boundary = uuid.uuid4().hex
    body = b"".join(
        (
            f"--{boundary}\r\n".encode(),
            (
                'Content-Disposition: form-data; name="file"; '
                f'filename="{image.name}"\r\n'
            ).encode(),
            b"Content-Type: application/octet-stream\r\n\r\n",
            image.read_bytes(),
            f"\r\n--{boundary}--\r\n".encode(),
        )
    )
    request = Request(
        f"{base_url.rstrip('/')}/api/process",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        payload = json.loads(response.read())
    if not isinstance(payload, dict) or "result" not in payload:
        raise ValueError("OCR workbench returned an invalid response")
    return payload


def _delete_session(base_url: str, session_id: str, timeout_seconds: float) -> None:
    request = Request(
        f"{base_url.rstrip('/')}/api/sessions/{session_id}",
        method="DELETE",
    )
    try:
        with urlopen(request, timeout=timeout_seconds):
            return
    except (HTTPError, URLError, TimeoutError):
        return


def _write_once(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")


def _is_loopback(base_url: str) -> bool:
    return base_url.startswith("http://127.0.0.1:") or base_url.startswith(
        "http://localhost:"
    )


def _relative_key(path: Path, root: Path) -> tuple[str, str]:
    relative = path.relative_to(root).as_posix()
    return relative.casefold(), relative


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--timeout-seconds", type=float, default=600)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
