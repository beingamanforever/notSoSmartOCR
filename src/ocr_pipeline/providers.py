"""Local OCR readers."""

from __future__ import annotations

import csv
import io
import math
import os
import subprocess
from collections.abc import Iterable, Mapping
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


# Tesseract's OpenMP path costs more than it returns: on an 8-core host a single crop
# reads 2.5x faster with one thread, and concurrent crops otherwise oversubscribe the
# machine badly. One process per crop, one thread per process.
_SINGLE_THREADED_ENV = {**os.environ, "OMP_THREAD_LIMIT": "1"}


class TesseractReader:
    name = "tesseract"

    def __init__(
        self,
        language: str = "eng",
        executable: str = "tesseract",
        timeout_seconds: int = 120,
        page_segmentation_mode: int | None = None,
        thresholding_method: int | None = None,
    ) -> None:
        if page_segmentation_mode is not None and not 0 <= page_segmentation_mode <= 13:
            raise ValueError("Tesseract page segmentation mode must be from 0 to 13")
        if thresholding_method is not None and thresholding_method not in {0, 1, 2}:
            raise ValueError("Tesseract thresholding method must be 0, 1, or 2")
        self.language = language
        self.executable = executable
        self.timeout_seconds = timeout_seconds
        self.page_segmentation_mode = page_segmentation_mode
        self.thresholding_method = thresholding_method

    def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
        command = [
            self.executable,
            str(image_path),
            "stdout",
            "-l",
            self.language,
        ]
        if self.page_segmentation_mode is not None:
            command.extend(["--psm", str(self.page_segmentation_mode)])
        if self.thresholding_method is not None:
            command.extend(["-c", f"thresholding_method={self.thresholding_method}"])
        command.append("tsv")
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
                env=_SINGLE_THREADED_ENV,
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

        regions = _parse_tesseract_tsv(completed.stdout, page_number, self.name)
        configured = {
            key: value
            for key, value in (
                ("page_segmentation_mode", self.page_segmentation_mode),
                ("thresholding_method", self.thresholding_method),
            )
            if value is not None
        }
        if configured:
            for region in regions:
                assert region.text_provenance is not None
                region.text_provenance["tesseract_config"] = configured
        return regions


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
            provenance = {
                "method": "tesseract_tsv",
                "block_num": int(row["block_num"]),
                "paragraph_num": int(row["par_num"]),
                "line_num": int(row["line_num"]),
                "word_num": int(row["word_num"]),
            }
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
                    text_provenance=provenance,
                )
            )
    except (KeyError, TypeError, ValueError) as error:
        raise ReaderError(
            "invalid_reader_output", f"Invalid Tesseract TSV: {error}"
        ) from error

    return regions


def _optional_value(value: object, key: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _box_coordinates(value: object) -> tuple[int, int, int, int]:
    coordinates = _coordinate_values(value)
    left, top, right, bottom = (int(round(number)) for number in coordinates)
    if right <= left or bottom <= top:
        raise ValueError("bounding box must have positive area")
    return left, top, right, bottom


def _normalized_box(
    value: object, width: int, height: int
) -> tuple[int, int, int, int]:
    left, top, right, bottom = _coordinate_values(value)
    if min(left, top, right, bottom) < 0 or max(left, top, right, bottom) > 1000:
        raise ValueError("normalized bounding box must be within 0 and 1000")
    return _box_coordinates(
        [
            left * width / 1000,
            top * height / 1000,
            right * width / 1000,
            bottom * height / 1000,
        ]
    )


def _coordinate_values(value: object) -> list[float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise TypeError("bounding box must contain four coordinates")
    try:
        coordinates = [float(number) for number in value]
    except (TypeError, ValueError) as error:
        raise TypeError("bounding box must contain four numbers") from error
    if len(coordinates) != 4 or not all(
        math.isfinite(number) for number in coordinates
    ):
        raise ValueError("bounding box must contain four finite numbers")
    return coordinates


def _confidence(value: object) -> float:
    confidence = float(value)
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("confidence must be between 0 and 1")
    return confidence
