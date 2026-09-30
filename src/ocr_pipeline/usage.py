"""Every parse with its first page and metrics, for the admin dashboard.

Records hold what people send (the first page image, the file name, the client address),
so the directory is private to the server user and bounded: past `keep` records the
oldest are deleted with their images.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import tempfile
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image

PAGE_SIDE = 1400  # the stored page image, long side in pixels
THUMB_SIDE = 360


class UsageLog:
    def __init__(self, root: Path, keep: int = 2000) -> None:
        self.root = Path(root)
        self.keep = keep
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._lock = threading.Lock()

    def record(
        self, entry: dict[str, Any], page_image: Path | None, details: dict[str, Any] | None = None
    ) -> str:
        """Store one parse; returns its id. `details` (regions, routing) is kept apart
        from the record so listing thousands of records stays light."""
        now = datetime.now(UTC)
        # Ids sort by time to the microsecond: order and retention follow arrival.
        entry_id = f"{now:%Y%m%dT%H%M%S%f}-{secrets.token_urlsafe(6)}"
        entry = {"id": entry_id, "created_at": now.isoformat(), **entry}
        with tempfile.TemporaryDirectory(prefix=".pending-", dir=self.root) as temporary:
            pending = Path(temporary)
            if page_image is not None:
                try:
                    with Image.open(page_image) as page:
                        page = page.convert("RGB")
                        for name, side in (("page.jpg", PAGE_SIDE), ("thumb.jpg", THUMB_SIDE)):
                            copy = page.copy()
                            copy.thumbnail((side, side))
                            copy.save(pending / name, format="JPEG", quality=85)
                except OSError:
                    entry["image_error"] = True
            (pending / "record.json").write_text(json.dumps(entry, default=str), encoding="utf-8")
            if details is not None:
                (pending / "details.json").write_text(json.dumps(details, default=str), encoding="utf-8")
            pending.rename(self.root / entry_id)
        self._prune()
        return entry_id

    def update(
        self, entry_id: str, fields: dict[str, Any], details: dict[str, Any] | None = None
    ) -> None:
        path = self._path(entry_id, "record.json")
        if path is None:
            return
        with self._lock:
            entry = {**json.loads(path.read_text(encoding="utf-8")), **fields}
            self._write(path, entry)
            if details is not None:
                self._write(path.with_name("details.json"), details)

    def details(self, entry_id: str) -> dict[str, Any] | None:
        path = self._path(entry_id, "details.json")
        return json.loads(path.read_text(encoding="utf-8")) if path else None

    @staticmethod
    def _write(path: Path, value: dict[str, Any]) -> None:
        spare = path.with_suffix(".tmp")
        spare.write_text(json.dumps(value, default=str), encoding="utf-8")
        os.replace(spare, path)

    def entries(self, limit: int = 500) -> list[dict[str, Any]]:
        """Newest first."""
        records = sorted(self.root.glob("*/record.json"), reverse=True)[:limit]
        out = []
        for path in records:
            try:
                out.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
        return out

    def image(self, entry_id: str, name: str) -> Path | None:
        return self._path(entry_id, name) if name in {"page.jpg", "thumb.jpg"} else None

    def _path(self, entry_id: str, name: str) -> Path | None:
        path = self.root / entry_id / name
        try:
            path.resolve().relative_to(self.root.resolve())
        except ValueError:
            return None
        return path if path.is_file() else None

    def _prune(self) -> None:
        with self._lock:
            records = sorted(p for p in self.root.iterdir() if p.is_dir() and not p.name.startswith("."))
            for stale in records[: max(0, len(records) - self.keep)]:
                shutil.rmtree(stale, ignore_errors=True)
