"""Local OCR readers."""

from __future__ import annotations

import csv
import io
import subprocess
from pathlib import Path
from typing import Protocol

from .contracts import BoundingBox, TextRegion


class LocalReader(Protocol):
    name: str

    def read(self, image_path: Path, page_number: int) -> list[TextRegion]: ...


class ReaderError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class TesseractReader:
    name = "tesseract"

    def __init__(
        self,
        language: str = "eng",
        executable: str = "tesseract",
        timeout_seconds: int = 120,
    ) -> None:
        self.language = language
        self.executable = executable
        self.timeout_seconds = timeout_seconds

    def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
        try:
            completed = subprocess.run(
                [
                    self.executable,
                    str(image_path),
                    "stdout",
                    "-l",
                    self.language,
                    "tsv",
                ],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except FileNotFoundError as error:
            raise ReaderError("reader_unavailable", str(error)) from error
        except subprocess.TimeoutExpired as error:
            raise ReaderError(
                "reader_timeout",
                f"Tesseract exceeded {self.timeout_seconds} seconds",
            ) from error

        if completed.returncode != 0:
            message = completed.stderr.strip() or "Tesseract returned no error message"
            raise ReaderError("reader_failed", message)

        return _parse_tesseract_tsv(completed.stdout, page_number, self.name)


def _parse_tesseract_tsv(
    output: str, page_number: int, provider: str
) -> list[TextRegion]:
    regions: list[TextRegion] = []
    rows = csv.DictReader(io.StringIO(output), delimiter="\t", quoting=csv.QUOTE_NONE)

    try:
        for row in rows:
            text = (row.get("text") or "").strip()
            if row.get("level") != "5" or not text:
                continue

            left = int(row["left"])
            top = int(row["top"])
            width = int(row["width"])
            height = int(row["height"])
            confidence_value = float(row["conf"])
            confidence = (
                None if confidence_value < 0 else round(confidence_value / 100, 4)
            )
            order = len(regions) + 1
            regions.append(
                TextRegion(
                    id=f"p{page_number}-word-{order}",
                    kind="word",
                    text=text,
                    confidence=confidence,
                    bounding_box=BoundingBox(
                        left=left,
                        top=top,
                        right=left + width,
                        bottom=top + height,
                    ),
                    reading_order=order,
                    provider=provider,
                )
            )
    except (KeyError, TypeError, ValueError) as error:
        raise ReaderError(
            "invalid_reader_output", f"Invalid Tesseract TSV: {error}"
        ) from error

    return regions
