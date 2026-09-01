"""Measure lossless integer-rotation views on a ClinOCR development split."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageOps

from ocr_pipeline.contracts import BoundingBox, TextRegion
from ocr_pipeline.providers import LocalReader, NemotronOCRV2Reader, ReaderError

if __package__:
    from experiments.public_benchmark import run_benchmark
else:
    from public_benchmark import run_benchmark

ROTATIONS = {
    0: None,
    90: Image.Transpose.ROTATE_90,
    180: Image.Transpose.ROTATE_180,
    270: Image.Transpose.ROTATE_270,
}
ORIENTATION_CONFIDENCE_FLOOR = 0.5
ORIENTATION_MAX_WORDS = 50
ORIENTATION_MIN_SUPPORTING_WORDS = 10
DEFAULT_RECTIFICATION_PADDING = 0.05


class RotatedViewReader:
    """Expose one lossless rotated transcription view through LocalReader."""

    def __init__(
        self,
        reader: LocalReader,
        angle: int,
        *,
        rectify: bool = False,
        rectify_padding: float = DEFAULT_RECTIFICATION_PADDING,
    ) -> None:
        if angle not in ROTATIONS:
            raise ValueError(f"Unsupported rotation: {angle}")
        self.reader = reader
        self.angle = angle
        self.rectify = rectify
        self.rectify_padding = rectify_padding
        prefix = f"{reader.name}-rectify" if rectify else reader.name
        self.name = f"{prefix}-rotate-{angle}"

    def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
        with Image.open(image_path) as source:
            exif_orientation = int(source.getexif().get(274, 1))
            normalized = ImageOps.exif_transpose(source)
            width, height = normalized.size
            try:
                prepared = (
                    rectify_document(
                        normalized,
                        padding_fraction=self.rectify_padding,
                    )
                    if self.rectify
                    else normalized.copy()
                )
            except ValueError as error:
                raise ReaderError("rectification_failed", str(error)) from error
        transform = ROTATIONS[self.angle]
        rotated = prepared.transpose(transform) if transform is not None else prepared

        with tempfile.TemporaryDirectory(prefix="ocr-rotation-") as directory:
            rotated_path = Path(directory) / "page.png"
            rotated.save(rotated_path, format="PNG")
            regions = self.reader.read(rotated_path, page_number)

        text = _region_text(regions)
        if not text:
            return []
        return [
            TextRegion(
                id=f"p{page_number}-rotation-{self.angle}",
                kind="page_text",
                text=text,
                confidence=None,
                bounding_box=BoundingBox(0, 0, width, height),
                reading_order=1,
                provider=self.name,
                text_provenance={
                    "exif_orientation": exif_orientation,
                    "exif_transposed": exif_orientation in range(2, 9),
                },
            )
        ]


class AutoOrientationReader:
    """Choose a page orientation from OCR confidence without reference text."""

    def __init__(
        self,
        reader: LocalReader,
        *,
        rectify: bool = False,
        osd_executable: str | None = None,
        osd_min_confidence: float = 15.0,
        final_merge_level: str | None = None,
        rectify_padding: float = DEFAULT_RECTIFICATION_PADDING,
        direct_osd: bool = False,
    ) -> None:
        if direct_osd and (not osd_executable or not final_merge_level):
            raise ValueError(
                "Direct OSD requires an OSD executable and a final merge level"
            )
        self.reader = reader
        self.rectify = rectify
        self.osd_executable = osd_executable
        self.osd_min_confidence = osd_min_confidence
        self.final_merge_level = final_merge_level
        self.rectify_padding = rectify_padding
        self.direct_osd = direct_osd
        prefix = f"{reader.name}-rectify" if rectify else reader.name
        self.name = f"{prefix}-auto-orientation"
        self.selections: list[dict[str, object]] = []

    def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
        selection: dict[str, object] = {
            "angle": None,
            "selected_score": None,
            "view_scores": {},
            "view_failures": {},
            "selector": "nemotron_word_evidence",
        }
        with Image.open(image_path) as source:
            exif_orientation = int(source.getexif().get(274, 1))
            normalized = ImageOps.exif_transpose(source)
            width, height = normalized.size
            selection["exif_orientation"] = exif_orientation
            selection["exif_transposed"] = exif_orientation in range(2, 9)
            try:
                prepared = (
                    rectify_document(
                        normalized,
                        padding_fraction=self.rectify_padding,
                    )
                    if self.rectify
                    else normalized.copy()
                )
            except ValueError as error:
                selection["failure"] = {
                    "code": "rectification_failed",
                    "message": str(error),
                }
                self.selections.append(selection)
                raise ReaderError("rectification_failed", str(error)) from error

        views: list[tuple[object, int, dict[str, object], list[TextRegion]]] = []
        direct_osd_angle: int | None = None
        with tempfile.TemporaryDirectory(prefix="ocr-orientation-") as directory:
            directory_path = Path(directory)
            candidate_angles = list(ROTATIONS)
            if self.osd_executable:
                prepared_path = directory_path / "page-osd.png"
                prepared.save(prepared_path, format="PNG")
                try:
                    osd = detect_tesseract_orientation(
                        prepared_path,
                        executable=self.osd_executable,
                    )
                except ReaderError as error:
                    selection["osd_failure"] = {
                        "code": error.code,
                        "message": str(error),
                    }
                else:
                    selection["osd"] = osd
                    if float(osd["confidence"]) >= self.osd_min_confidence:
                        candidate_angles = [int(osd["angle"])]
                        if self.direct_osd:
                            direct_osd_angle = candidate_angles[0]
                            selection["selector"] = "tesseract_osd_direct"
                        else:
                            selection["selector"] = "tesseract_osd"

            if direct_osd_angle is None:
                views.extend(
                    self._read_views(
                        prepared,
                        page_number,
                        directory_path,
                        candidate_angles,
                        selection,
                    )
                )
            if (
                direct_osd_angle is None
                and len(candidate_angles) == 1
                and (not views or float(views[0][2]["confidence_evidence"]) == 0.0)
            ):
                selection["selector"] = "nemotron_word_evidence_after_osd_failure"
                fallback_angles = [
                    angle for angle in ROTATIONS if angle not in candidate_angles
                ]
                views.extend(
                    self._read_views(
                        prepared,
                        page_number,
                        directory_path,
                        fallback_angles,
                        selection,
                    )
                )

        if direct_osd_angle is None and not views:
            selection["failure"] = {
                "code": "orientation_views_failed",
                "message": "All orientation views failed",
            }
            self.selections.append(selection)
            raise ReaderError(
                "orientation_views_failed",
                "All orientation views failed",
            )

        if direct_osd_angle is not None:
            angle = direct_osd_angle
            regions = []
            osd = selection.get("osd")
            if not isinstance(osd, dict):
                raise ValueError("Invalid direct OSD selection record")
            score = {
                "selection_metric": "tesseract_osd_confidence",
                "selection_value": float(osd["confidence"]),
            }
            selection["angle"] = angle
            selection["selected_score"] = score
            selection["score_margin_metric"] = score["selection_metric"]
            selection["score_margin"] = 1.0
        else:
            ranked_views = sorted(views, key=lambda item: item[0], reverse=True)
            _, angle, score, regions = ranked_views[0]
            best_value = float(score["selection_value"])
            second_value = (
                float(ranked_views[1][2]["selection_value"])
                if len(ranked_views) > 1
                and ranked_views[1][2]["selection_metric"] == score["selection_metric"]
                else 0.0
            )
            selection["angle"] = angle
            selection["selected_score"] = score
            selection["score_margin_metric"] = score["selection_metric"]
            selection["score_margin"] = round(
                (best_value - second_value) / max(best_value, 1e-9),
                6,
            )
        if self.final_merge_level:
            read_with_merge_level = getattr(self.reader, "read_with_merge_level", None)
            if not callable(read_with_merge_level):
                selection["failure"] = {
                    "code": "final_merge_level_unavailable",
                    "message": "The reader cannot rerun a selected orientation",
                }
                self.selections.append(selection)
                raise ReaderError(
                    "final_merge_level_unavailable",
                    "The reader cannot rerun a selected orientation",
                )
            transform = ROTATIONS[angle]
            final_view = (
                prepared.transpose(transform) if transform is not None else prepared
            )
            with tempfile.TemporaryDirectory(
                prefix="ocr-final-orientation-"
            ) as directory:
                final_path = Path(directory) / f"page-{angle}.png"
                final_view.save(final_path, format="PNG")
                try:
                    regions = read_with_merge_level(
                        final_path,
                        page_number,
                        self.final_merge_level,
                    )
                except ReaderError as error:
                    selection["failure"] = {
                        "code": error.code,
                        "message": str(error),
                    }
                    self.selections.append(selection)
                    raise
            selection["final_merge_level"] = self.final_merge_level
            if not regions:
                selection["failure"] = {
                    "code": "final_empty_output",
                    "message": "The selected orientation returned no final text regions",
                }
                self.selections.append(selection)
                raise ReaderError(
                    "final_empty_output",
                    "The selected orientation returned no final text regions",
                )
        self.selections.append(selection)
        text = _region_text(regions)
        if not text:
            return []
        return [
            TextRegion(
                id=f"p{page_number}-auto-orientation",
                kind="page_text",
                text=text,
                confidence=None,
                bounding_box=BoundingBox(0, 0, width, height),
                reading_order=1,
                provider=f"{self.name}-selected-{angle}",
            )
        ]

    def _read_views(
        self,
        prepared: Image.Image,
        page_number: int,
        directory: Path,
        angles: list[int],
        selection: dict[str, object],
    ) -> list[tuple[object, int, dict[str, object], list[TextRegion]]]:
        views = []
        view_scores = selection["view_scores"]
        view_failures = selection["view_failures"]
        if not isinstance(view_scores, dict) or not isinstance(view_failures, dict):
            raise ValueError("Invalid orientation selection record")
        for angle in angles:
            transform = ROTATIONS[angle]
            view = prepared.transpose(transform) if transform is not None else prepared
            view_path = directory / f"page-{angle}.png"
            view.save(view_path, format="PNG")
            try:
                regions = self.reader.read(view_path, page_number)
            except ReaderError as error:
                view_failures[str(angle)] = {
                    "code": error.code,
                    "message": str(error),
                }
                continue
            score = orientation_score(regions)
            view_scores[str(angle)] = score
            views.append((score["rank"], angle, score, regions))
        return views


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare Nemotron OCR across lossless integer rotations"
    )
    parser.add_argument("root", type=Path, help="Extracted ClinOCR dataset root")
    parser.add_argument("output", type=Path, help="JSON results path")
    parser.add_argument(
        "--language",
        choices=("multi", "en"),
        default="en",
        help="Nemotron OCR v2 weight variant",
    )
    parser.add_argument(
        "--clinocr-role",
        choices=("exemplar", "eval"),
        default="exemplar",
        help="Use exemplar while developing; eval only after the policy is frozen",
    )
    parser.add_argument("--subset", default="rotated")
    parser.add_argument(
        "--rectify",
        action="store_true",
        help="Rectify the largest document quadrilateral before rotation",
    )
    parser.add_argument(
        "--rectify-padding",
        type=float,
        default=DEFAULT_RECTIFICATION_PADDING,
        help="Outward page-corner padding used during rectification",
    )
    parser.add_argument(
        "--auto-select",
        action="store_true",
        help="Choose one orientation without using references",
    )
    parser.add_argument(
        "--orientation-selector",
        choices=("nemotron", "tesseract-osd"),
        default="nemotron",
        help="Gold-free selector used by --auto-select",
    )
    parser.add_argument(
        "--tesseract-osd",
        default="tesseract",
        help="Tesseract executable for the independent OSD prepass",
    )
    parser.add_argument(
        "--osd-min-confidence",
        type=float,
        default=15.0,
        help="OSD confidence required before skipping four-view selection",
    )
    parser.add_argument(
        "--osd-direct",
        action="store_true",
        help="Skip the redundant OCR evidence pass after accepted OSD",
    )
    args = parser.parse_args(argv)

    try:
        if args.orientation_selector != "nemotron" and not args.auto_select:
            raise ValueError("--orientation-selector requires --auto-select")
        if args.osd_direct and (
            not args.auto_select or args.orientation_selector != "tesseract-osd"
        ):
            raise ValueError("--osd-direct requires --auto-select and tesseract-osd")
        if args.osd_min_confidence < 0:
            raise ValueError("--osd-min-confidence cannot be negative")
        if not 0 <= args.rectify_padding <= 0.25:
            raise ValueError("--rectify-padding must be between 0 and 0.25")
        merge_level = "word" if args.auto_select else "paragraph"
        reader = NemotronOCRV2Reader(
            language=args.language,
            merge_level=merge_level,
        )
        if args.auto_select:
            automatic_reader = AutoOrientationReader(
                reader,
                rectify=args.rectify,
                osd_executable=(
                    args.tesseract_osd
                    if args.orientation_selector == "tesseract-osd"
                    else None
                ),
                osd_min_confidence=args.osd_min_confidence,
                final_merge_level="paragraph",
                rectify_padding=args.rectify_padding,
                direct_osd=args.osd_direct,
            )
            automatic_run = run_benchmark(
                "clinocr",
                args.root,
                1,
                automatic_reader,
                selected_subsets={args.subset},
                clinocr_role=args.clinocr_role,
            )
            for case, selection in zip(
                automatic_run["cases"],
                automatic_reader.selections,
                strict=True,
            ):
                case["orientation_selection"] = selection
            automatic_run["orientation_summary"] = orientation_attempt_summary(
                automatic_reader.selections,
                automatic_run["cases"],
            )
            runs = {"auto": automatic_run}
        else:
            runs = {
                str(angle): run_benchmark(
                    "clinocr",
                    args.root,
                    1,
                    RotatedViewReader(
                        reader,
                        angle,
                        rectify=args.rectify,
                        rectify_padding=args.rectify_padding,
                    ),
                    selected_subsets={args.subset},
                    clinocr_role=args.clinocr_role,
                )
                for angle in ROTATIONS
            }
        payload = {
            "experiment": "lossless_integer_rotation_views",
            "clinocr_role": args.clinocr_role,
            "subset": args.subset,
            "base_reader": reader.name,
            "language": args.language,
            "merge_level": merge_level,
            "final_merge_level": "paragraph" if args.auto_select else None,
            "rectify": args.rectify,
            "rectify_padding": args.rectify_padding if args.rectify else None,
            "auto_select": args.auto_select,
            "orientation_selector": args.orientation_selector,
            "osd_min_confidence": args.osd_min_confidence,
            "osd_direct": args.osd_direct,
            "cold_start_in_angle": None if args.auto_select else 0,
            "runs": runs,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0


def orientation_score(regions: list[TextRegion]) -> dict[str, object]:
    character_count = 0
    weighted_confidence = 0.0
    high_confidence_characters = 0
    repeated_characters = 0
    confidences = []
    for region in regions:
        characters = len("".join(region.text.split()))
        confidence = region.confidence if region.confidence is not None else 0.0
        repeated = sum(
            len(match.group(0)) - 3
            for match in re.finditer(r"(.)\1{3,}", region.text.casefold())
        )
        character_count += characters
        weighted_confidence += characters * confidence
        if confidence >= 0.8:
            high_confidence_characters += characters
        if characters:
            confidences.append(confidence)
        repeated_characters += repeated

    strongest_confidences = sorted(confidences, reverse=True)[:ORIENTATION_MAX_WORDS]
    confidence_evidence = sum(
        max(confidence - ORIENTATION_CONFIDENCE_FLOOR, 0.0)
        for confidence in strongest_confidences
    )
    supporting_words = sum(
        confidence > ORIENTATION_CONFIDENCE_FLOOR
        for confidence in strongest_confidences
    )

    mean_confidence = weighted_confidence / character_count if character_count else 0.0
    high_confidence_fraction = (
        high_confidence_characters / character_count if character_count else 0.0
    )
    repetition_fraction = (
        repeated_characters / character_count if character_count else 1.0
    )
    sufficient_support = supporting_words >= ORIENTATION_MIN_SUPPORTING_WORDS
    selection_value = mean_confidence if sufficient_support else confidence_evidence
    rank = (
        int(sufficient_support),
        round(selection_value, 6),
        round(confidence_evidence, 6),
        character_count,
    )
    return {
        "rank": rank,
        "confidence_evidence": round(confidence_evidence, 6),
        "mean_confidence": round(mean_confidence, 6),
        "high_confidence_fraction": round(high_confidence_fraction, 6),
        "repetition_fraction": round(repetition_fraction, 6),
        "characters": character_count,
        "words": len(confidences),
        "supporting_words": supporting_words,
        "sufficient_support": sufficient_support,
        "minimum_supporting_words": ORIENTATION_MIN_SUPPORTING_WORDS,
        "selection_metric": (
            "mean_confidence" if sufficient_support else "confidence_evidence"
        ),
        "selection_value": round(selection_value, 6),
        "confidence_floor": ORIENTATION_CONFIDENCE_FLOOR,
        "max_scored_words": ORIENTATION_MAX_WORDS,
    }


def detect_tesseract_orientation(
    image_path: Path,
    *,
    executable: str = "tesseract",
    timeout_seconds: int = 30,
) -> dict[str, object]:
    try:
        completed = subprocess.run(
            [
                executable,
                str(image_path),
                "stdout",
                "--psm",
                "0",
                "-l",
                "osd",
            ],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except FileNotFoundError as error:
        raise ReaderError("osd_unavailable", str(error)) from error
    except subprocess.TimeoutExpired as error:
        raise ReaderError(
            "osd_timeout",
            f"Tesseract OSD exceeded {timeout_seconds} seconds",
        ) from error

    output = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
    if completed.returncode != 0:
        raise ReaderError(
            "osd_failed",
            output.strip() or "Tesseract OSD returned no error message",
        )

    fields = {}
    for line in output.splitlines():
        key, separator, value = line.partition(":")
        if separator:
            fields[key.strip()] = value.strip()
    try:
        rotate_clockwise = int(fields["Rotate"])
        confidence = float(fields["Orientation confidence"])
    except (KeyError, ValueError) as error:
        raise ReaderError(
            "invalid_osd_output",
            "Tesseract OSD did not return rotation and confidence",
        ) from error
    angle = (-rotate_clockwise) % 360
    if angle not in ROTATIONS:
        raise ReaderError(
            "invalid_osd_output",
            f"Unsupported Tesseract OSD rotation: {rotate_clockwise}",
        )
    return {
        "angle": angle,
        "rotate_clockwise": rotate_clockwise,
        "confidence": confidence,
        "script": fields.get("Script"),
        "script_confidence": _optional_float(fields.get("Script confidence")),
    }


def _optional_float(value: str | None) -> float | None:
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def orientation_attempt_summary(
    selections: list[dict[str, object]],
    cases: list[dict[str, object]],
) -> dict[str, int]:
    view_attempts = 0
    view_failures = 0
    affected_pages = 0
    recovered_pages = 0
    selection_failures = 0
    osd_attempts = 0
    osd_failures = 0
    osd_selected_pages = 0
    osd_direct_pages = 0
    osd_fallback_pages = 0
    for selection, case in zip(selections, cases, strict=True):
        scores = selection["view_scores"]
        failures = selection["view_failures"]
        if not isinstance(scores, dict) or not isinstance(failures, dict):
            raise ValueError("Invalid orientation selection record")
        view_attempts += len(scores) + len(failures)
        view_failures += len(failures)
        affected_pages += bool(failures)
        recovered_pages += bool(failures) and case["status"] == "success"
        selection_failures += "failure" in selection
        osd_attempted = "osd" in selection or "osd_failure" in selection
        osd_attempts += osd_attempted
        osd_failures += "osd_failure" in selection
        osd_selected = selection.get("selector") in {
            "tesseract_osd",
            "tesseract_osd_direct",
        }
        osd_selected_pages += osd_selected
        osd_direct_pages += selection.get("selector") == "tesseract_osd_direct"
        osd_fallback_pages += osd_attempted and (not osd_selected)
    return {
        "pages": len(selections),
        "view_attempts": view_attempts,
        "view_failures": view_failures,
        "affected_pages": affected_pages,
        "recovered_pages": recovered_pages,
        "selection_failures": selection_failures,
        "osd_attempts": osd_attempts,
        "osd_failures": osd_failures,
        "osd_selected_pages": osd_selected_pages,
        "osd_direct_pages": osd_direct_pages,
        "osd_fallback_pages": osd_fallback_pages,
    }


def _region_text(regions: list[TextRegion]) -> str:
    return " ".join(
        region.text
        for region in sorted(regions, key=lambda region: region.reading_order)
        if region.text.strip()
    ).strip()


def rectify_document(
    source: Image.Image,
    *,
    padding_fraction: float = DEFAULT_RECTIFICATION_PADDING,
) -> Image.Image:
    """Rectify the most credible page quadrilateral without altering its content."""
    if not 0 <= padding_fraction <= 0.25:
        raise ValueError("Rectification padding must be between 0 and 0.25")
    try:
        import cv2
    except (ImportError, OSError) as error:
        raise ValueError(f"OpenCV is required for rectification: {error}") from error

    rgb = np.asarray(source.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        np.ones((7, 7), dtype=np.uint8),
        iterations=2,
    )
    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    minimum_area = rgb.shape[0] * rgb.shape[1] * 0.2
    corners = None
    contours = []
    for mask in (edges, threshold):
        found, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        contours.extend(found)
    candidates = sorted(contours, key=cv2.contourArea, reverse=True)
    if candidates and cv2.contourArea(candidates[0]) >= minimum_area:
        largest = candidates[0]
        hull = cv2.convexHull(largest)
        perimeter = cv2.arcLength(hull, True)
        for epsilon in (0.01, 0.02, 0.03, 0.05):
            polygon = cv2.approxPolyDP(hull, epsilon * perimeter, True)
            if len(polygon) == 4 and cv2.isContourConvex(polygon):
                corners = polygon.reshape(4, 2).astype(np.float32)
                break
        if corners is None:
            corners = cv2.boxPoints(cv2.minAreaRect(largest)).astype(np.float32)
    if corners is None:
        raise ValueError(
            "No page quadrilateral covered at least 20 percent of the image"
        )

    ordered = _order_corners(corners)
    center = ordered.mean(axis=0)
    ordered = center + (ordered - center) * (1 + padding_fraction)
    top_left, top_right, bottom_right, bottom_left = ordered
    target_width = int(
        round(
            max(
                np.linalg.norm(bottom_right - bottom_left),
                np.linalg.norm(top_right - top_left),
            )
        )
    )
    target_height = int(
        round(
            max(
                np.linalg.norm(top_right - bottom_right),
                np.linalg.norm(top_left - bottom_left),
            )
        )
    )
    if target_width < 2 or target_height < 2:
        raise ValueError("Rectified page has invalid dimensions")

    destination = np.array(
        [
            [0, 0],
            [target_width - 1, 0],
            [target_width - 1, target_height - 1],
            [0, target_height - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(ordered, destination)
    rectified = cv2.warpPerspective(
        rgb,
        matrix,
        (target_width, target_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return Image.fromarray(rectified, mode="RGB")


def _order_corners(corners: np.ndarray) -> np.ndarray:
    center = corners.mean(axis=0)
    angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
    return corners[np.argsort(angles)].astype(np.float32)


if __name__ == "__main__":
    raise SystemExit(main())
