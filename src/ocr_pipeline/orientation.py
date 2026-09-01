"""Lossless integer-rotation selection for local OCR readers."""

from __future__ import annotations

import re
import subprocess
import tempfile
import threading
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any, Iterator

from PIL import Image, ImageOps, UnidentifiedImageError

from .contracts import BoundingBox, TextRegion
from .providers import LocalReader, ReaderError

ROTATIONS = {
    0: None,
    90: Image.Transpose.ROTATE_90,
    180: Image.Transpose.ROTATE_180,
    270: Image.Transpose.ROTATE_270,
}
CONFIDENCE_FLOOR = 0.5
MAX_SCORED_WORDS = 50
MIN_SUPPORTING_WORDS = 10
COVERAGE_RECOVERY_RATIO = 2.0
COVERAGE_CONFIDENCE_TOLERANCE = 0.03
DOCTR_ARCH = "mobilenet_v3_small_page_orientation"
DOCTR_MODEL = {
    "library": "python-doctr",
    "architecture": DOCTR_ARCH,
    "publisher": "Mindee",
    "license": "Apache-2.0",
}


class DocTROrientationDetector:
    """Predict one lossless page rotation with docTR's small classifier."""

    name = "doctr-page-orientation"

    def __init__(
        self,
        *,
        device: str = "cuda",
        batch_size: int = 1,
        predictor: Callable[[list[Any]], object] | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.device = device
        self.batch_size = batch_size
        self.predictor = predictor
        self._lock = threading.Lock()

    def __call__(self, image_path: Path) -> dict[str, object]:
        try:
            import numpy as np

            with Image.open(image_path) as source:
                image = np.asarray(ImageOps.exif_transpose(source).convert("RGB"))
        except (ImportError, OSError, UnidentifiedImageError, ValueError) as error:
            raise ReaderError(
                "orientation_classifier_image_failed", str(error)
            ) from error

        with self._lock:
            predictor = self.predictor or self._load_predictor()
            try:
                prediction = predictor([image])
            except Exception as error:
                raise ReaderError(
                    "orientation_classifier_failed",
                    str(error),
                ) from error
        if not isinstance(prediction, (list, tuple)) or len(prediction) != 3:
            raise ReaderError(
                "invalid_orientation_classifier_output",
                "docTR returned an invalid orientation prediction",
            )
        _, rotations, confidences = prediction
        if len(rotations) != 1 or len(confidences) != 1:
            raise ReaderError(
                "invalid_orientation_classifier_output",
                "docTR returned a mismatched orientation batch",
            )
        angle = int(rotations[0]) % 360
        confidence = float(confidences[0])
        if angle not in ROTATIONS or not 0 <= confidence <= 1:
            raise ReaderError(
                "invalid_orientation_classifier_output",
                "docTR returned an unsupported angle or confidence",
            )
        return {
            "angle": angle,
            "confidence": confidence,
            "model": dict(DOCTR_MODEL),
        }

    def _load_predictor(self) -> Callable[[list[Any]], object]:
        try:
            import torch
            from doctr.models import page_orientation_predictor
        except (ImportError, OSError) as error:
            raise ReaderError(
                "orientation_classifier_unavailable",
                "python-doctr and a compatible PyTorch build are required",
            ) from error
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise ReaderError(
                "orientation_classifier_unavailable",
                f"CUDA is unavailable for requested device {self.device}",
            )
        try:
            predictor = page_orientation_predictor(
                arch=DOCTR_ARCH,
                pretrained=True,
                batch_size=self.batch_size,
            ).to(self.device)
        except Exception as error:
            raise ReaderError(
                "orientation_classifier_unavailable", str(error)
            ) from error
        self.predictor = predictor
        return predictor


class OrientationReader:
    """Prefer confident OSD and otherwise select the strongest OCR view."""

    def __init__(
        self,
        reader: LocalReader,
        *,
        osd_executable: str | None = "tesseract",
        osd_detector: Callable[[Path], dict[str, object]] | None = None,
        osd_min_confidence: float = 15.0,
        orientation_detector: Callable[[Path], dict[str, object]] | None = None,
        defer_restore: bool = False,
    ) -> None:
        if osd_min_confidence < 0:
            raise ValueError("osd_min_confidence cannot be negative")
        self.reader = reader
        self.name = f"{reader.name}-oriented"
        self.osd_min_confidence = osd_min_confidence
        self.orientation_detector = orientation_detector
        self.defer_restore = defer_restore
        self.osd_detector = osd_detector
        if self.osd_detector is None and osd_executable:
            self.osd_detector = partial(
                detect_tesseract_orientation,
                executable=osd_executable,
            )
        self.executable = getattr(reader, "executable", None)
        self._assessments: dict[int, dict[str, object]] = {}
        self._lock = threading.Lock()

    def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
        assessment: dict[str, Any] = {
            "page_number": page_number,
            "status": "running",
            "selector": "evidence_fallback",
            "view_scores": {},
            "view_failures": {},
        }
        try:
            with Image.open(image_path) as source:
                exif_orientation = int(source.getexif().get(274, 1))
                normalized = ImageOps.exif_transpose(source).convert("RGB")
        except (OSError, UnidentifiedImageError, ValueError) as error:
            self._fail(assessment, "orientation_image_failed", str(error))
            raise ReaderError("orientation_image_failed", str(error)) from error

        original_size = normalized.size
        assessment.update(
            {
                "original_size": list(original_size),
                "exif_orientation": exif_orientation,
                "exif_transposed": exif_orientation in range(2, 9),
            }
        )

        angles = list(ROTATIONS)
        with tempfile.TemporaryDirectory(prefix="ocr-orientation-") as directory:
            root = Path(directory)
            normalized_path = root / "page-osd.png"
            normalized.save(normalized_path, format="PNG")
            angles = self._candidate_angles(normalized_path, assessment)

            views = self._read_views(
                normalized,
                root,
                page_number,
                angles,
                assessment,
            )

        if not views:
            failures = assessment["view_failures"]
            detail = ", ".join(
                f"{angle}={failure['code']}"
                for angle, failure in sorted(failures.items())
            )
            message = "All orientation views failed"
            if detail:
                message = f"{message}: {detail}"
            self._fail(assessment, "orientation_views_failed", message)
            raise ReaderError("orientation_views_failed", message)

        ranked, selection_reason = _rank_views(views)
        angle, view_size, score, regions = ranked[0]
        assessment.update(
            {
                "angle": angle,
                "selected_score": score,
                "view_selection_reason": selection_reason,
                "score_margin_metric": (
                    "character_coverage"
                    if selection_reason == "coverage_recovery"
                    else str(score["selection_metric"])
                ),
                "score_margin": _score_margin(ranked, selection_reason),
                "rotated_size": list(view_size),
            }
        )
        needs_review = angle != 0 or assessment["selector"] in {
            "evidence_fallback",
            "orientation_evidence_fallback",
        }
        needs_review = needs_review or bool(assessment["view_failures"])
        assessment["status"] = (
            "review_recommended" if needs_review else "orientation_confirmed"
        )
        try:
            restored = [
                _annotate_region(region, assessment)
                if self.defer_restore
                else _restore_region(
                    region,
                    angle,
                    original_size,
                    view_size,
                    assessment,
                )
                for region in regions
            ]
        except ReaderError as error:
            self._fail(assessment, error.code, str(error))
            raise
        self._save_assessment(page_number, assessment)
        return restored

    def coverage_assessment(self, page_count: int) -> dict[str, object]:
        with self._lock:
            pages = [
                dict(
                    self._assessments.get(
                        page_number,
                        {"page_number": page_number, "status": "not_run"},
                    )
                )
                for page_number in range(1, page_count + 1)
            ]
        reviewed = [
            page for page in pages if page.get("status") == "review_recommended"
        ]
        failures = [page for page in pages if page.get("status") == "failed"]
        if failures or reviewed:
            return {
                "status": "review_recommended",
                "message": (
                    f"Orientation review is recommended on {len(reviewed)} page(s); "
                    f"{len(failures)} page(s) failed orientation OCR."
                ),
                "pages": pages,
            }
        if any(page.get("status") == "orientation_confirmed" for page in pages):
            return {
                "status": "assessed",
                "message": "The page orientation was confirmed by the configured selector.",
                "pages": pages,
            }
        return {
            "status": "not_assessed",
            "message": "No page orientation was assessed.",
            "pages": pages,
        }

    @contextmanager
    def stage_view(self, image_path: Path, page_number: int) -> Iterator[Path]:
        """Expose the selected upright view while downstream stages run."""
        if not self.defer_restore:
            yield image_path
            return
        with self._lock:
            assessment = dict(self._assessments.get(page_number, {}))
        angle = assessment.get("angle")
        if angle not in ROTATIONS or angle == 0:
            yield image_path
            return
        try:
            with Image.open(image_path) as source:
                normalized = ImageOps.exif_transpose(source).convert("RGB")
                transform = ROTATIONS[int(angle)]
                oriented = normalized.transpose(transform)
                with tempfile.TemporaryDirectory(
                    prefix="ocr-stage-orientation-"
                ) as directory:
                    path = Path(directory) / f"page-stage-{angle}.png"
                    oriented.save(path, format="PNG")
                    yield path
        except (OSError, UnidentifiedImageError, ValueError) as error:
            raise ReaderError("orientation_stage_image_failed", str(error)) from error

    def restore_regions(
        self,
        regions: list[TextRegion],
        page_number: int,
    ) -> list[TextRegion]:
        if not self.defer_restore:
            return regions
        with self._lock:
            assessment = dict(self._assessments.get(page_number, {}))
        angle = assessment.get("angle")
        original_size = assessment.get("original_size")
        view_size = assessment.get("rotated_size")
        if (
            angle not in ROTATIONS
            or not _valid_size(original_size)
            or not _valid_size(view_size)
        ):
            return regions
        return [
            _restore_region(
                region,
                int(angle),
                (int(original_size[0]), int(original_size[1])),
                (int(view_size[0]), int(view_size[1])),
                assessment,
            )
            for region in regions
        ]

    def page_needs_review(self, page_number: int) -> bool:
        with self._lock:
            assessment = self._assessments.get(page_number, {})
        return assessment.get("status") in {"review_recommended", "failed"}

    def _candidate_angles(
        self,
        image_path: Path,
        assessment: dict[str, Any],
    ) -> list[int]:
        prediction = self._detect_orientation(image_path, assessment)
        if prediction is None:
            osd = self._detect_osd(image_path, assessment)
            if osd and float(osd["confidence"]) >= self.osd_min_confidence:
                assessment["selector"] = "tesseract_osd"
                return [int(osd["angle"])]
            return list(ROTATIONS)

        angle = int(prediction["angle"])
        if angle == 0:
            assessment["selector"] = "orientation_classifier"
            assessment["osd_status"] = "skipped_zero_prediction"
            return [0]

        osd = self._detect_osd(image_path, assessment)
        if osd and int(osd["angle"]) == angle:
            assessment["selector"] = "orientation_agreement"
            return [angle]
        assessment["selector"] = "orientation_evidence_fallback"
        alternate = int(osd["angle"]) if osd else 0
        return list(dict.fromkeys((angle, alternate)))

    def _detect_orientation(
        self,
        image_path: Path,
        assessment: dict[str, Any],
    ) -> dict[str, object] | None:
        if self.orientation_detector is None:
            assessment["orientation_detector_status"] = "not_configured"
            return None
        try:
            prediction = _valid_orientation(self.orientation_detector(image_path))
        except ReaderError as error:
            assessment["orientation_detector_status"] = "failed"
            assessment["orientation_detector_failure"] = {
                "code": error.code,
                "message": str(error),
            }
            return None
        assessment["orientation_detector_status"] = "accepted"
        assessment["orientation_prediction"] = prediction
        return prediction

    def _detect_osd(
        self,
        image_path: Path,
        assessment: dict[str, Any],
    ) -> dict[str, object] | None:
        if self.osd_detector is None:
            assessment["osd_status"] = "not_configured"
            return None
        try:
            osd = _valid_osd(self.osd_detector(image_path))
        except ReaderError as error:
            assessment["osd_status"] = "failed"
            assessment["osd_failure"] = {
                "code": error.code,
                "message": str(error),
            }
            return None
        assessment["osd"] = osd
        assessment["osd_status"] = (
            "accepted"
            if float(osd["confidence"]) >= self.osd_min_confidence
            else "weak"
        )
        return osd

    def _read_views(
        self,
        source: Image.Image,
        root: Path,
        page_number: int,
        angles: list[int],
        assessment: dict[str, Any],
    ) -> list[tuple[int, tuple[int, int], dict[str, object], list[TextRegion]]]:
        views = []
        for angle in angles:
            transform = ROTATIONS[angle]
            image = source.transpose(transform) if transform is not None else source
            path = root / f"page-{angle}.png"
            image.save(path, format="PNG")
            try:
                regions = self.reader.read(path, page_number)
            except ReaderError as error:
                assessment["view_failures"][str(angle)] = {
                    "code": error.code,
                    "message": str(error),
                }
                continue
            if not regions:
                assessment["view_failures"][str(angle)] = {
                    "code": "empty_output",
                    "message": "The reader returned no text regions",
                }
                continue
            score = _orientation_score(regions)
            assessment["view_scores"][str(angle)] = score
            views.append((angle, image.size, score, regions))
        return views

    def _fail(
        self,
        assessment: dict[str, Any],
        code: str,
        message: str,
    ) -> None:
        assessment["status"] = "failed"
        assessment["failure"] = {"code": code, "message": message}
        self._save_assessment(int(assessment["page_number"]), assessment)

    def _save_assessment(
        self,
        page_number: int,
        assessment: dict[str, object],
    ) -> None:
        with self._lock:
            self._assessments[page_number] = assessment


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


def _orientation_score(regions: list[TextRegion]) -> dict[str, object]:
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

    strongest = sorted(confidences, reverse=True)[:MAX_SCORED_WORDS]
    confidence_evidence = sum(
        max(confidence - CONFIDENCE_FLOOR, 0.0) for confidence in strongest
    )
    supporting_words = sum(confidence > CONFIDENCE_FLOOR for confidence in strongest)
    mean_confidence = weighted_confidence / character_count if character_count else 0.0
    high_confidence_fraction = (
        high_confidence_characters / character_count if character_count else 0.0
    )
    repetition_fraction = (
        repeated_characters / character_count if character_count else 1.0
    )
    sufficient_support = supporting_words >= MIN_SUPPORTING_WORDS
    selection_value = mean_confidence if sufficient_support else confidence_evidence
    return {
        "rank": (
            int(sufficient_support),
            round(selection_value, 6),
            round(confidence_evidence, 6),
            character_count,
        ),
        "selection_metric": (
            "mean_confidence" if sufficient_support else "confidence_evidence"
        ),
        "selection_value": round(selection_value, 6),
        "confidence_evidence": round(confidence_evidence, 6),
        "mean_confidence": round(mean_confidence, 6),
        "high_confidence_fraction": round(high_confidence_fraction, 6),
        "repetition_fraction": round(repetition_fraction, 6),
        "characters": character_count,
        "words": len(confidences),
        "supporting_words": supporting_words,
    }


def _rank_views(
    views: list[tuple[int, tuple[int, int], dict[str, object], list[TextRegion]]],
) -> tuple[
    list[tuple[int, tuple[int, int], dict[str, object], list[TextRegion]]],
    str,
]:
    ranked = sorted(views, key=lambda item: item[2]["rank"], reverse=True)
    if len(ranked) < 2:
        return ranked, "single_view"

    strongest = ranked[0]
    broadest = max(ranked, key=lambda item: int(item[2]["characters"]))
    strongest_score = strongest[2]
    broadest_score = broadest[2]
    strongest_characters = int(strongest_score["characters"])
    broadest_characters = int(broadest_score["characters"])
    broadest_is_supported = (
        int(broadest_score["supporting_words"]) >= MIN_SUPPORTING_WORDS
    )
    confidence_is_close = float(
        broadest_score["mean_confidence"]
    ) + COVERAGE_CONFIDENCE_TOLERANCE >= float(strongest_score["mean_confidence"])
    coverage_is_material = broadest_characters >= max(
        strongest_characters * COVERAGE_RECOVERY_RATIO,
        1,
    )
    if (
        broadest != strongest
        and broadest_is_supported
        and confidence_is_close
        and coverage_is_material
    ):
        return [broadest, *(item for item in ranked if item != broadest)], (
            "coverage_recovery"
        )
    return ranked, "confidence_evidence"


def _score_margin(
    ranked: list[tuple[int, tuple[int, int], dict[str, object], list[TextRegion]]],
    selection_reason: str = "confidence_evidence",
) -> float:
    if len(ranked) < 2:
        return 1.0
    best = ranked[0][2]
    second = ranked[1][2]
    if selection_reason == "coverage_recovery":
        best_characters = float(best["characters"])
        second_characters = float(second["characters"])
        return round(
            (best_characters - second_characters) / max(best_characters, 1.0),
            6,
        )
    second_value = (
        float(second["selection_value"])
        if second["selection_metric"] == best["selection_metric"]
        else 0.0
    )
    best_value = float(best["selection_value"])
    return round((best_value - second_value) / max(best_value, 1e-9), 6)


def _restore_region(
    region: TextRegion,
    angle: int,
    original_size: tuple[int, int],
    view_size: tuple[int, int],
    assessment: dict[str, Any],
) -> TextRegion:
    annotated = _annotate_region(region, assessment)
    structure = _restore_structure_boxes(
        annotated.structure,
        angle,
        original_size,
        view_size,
    )
    return replace(
        annotated,
        bounding_box=_restore_box(
            annotated.bounding_box,
            angle,
            original_size,
            view_size,
        ),
        structure=structure,
    )


def _annotate_region(
    region: TextRegion,
    assessment: dict[str, Any],
) -> TextRegion:
    provenance = dict(region.text_provenance or {})
    orientation = {
        "angle": assessment["angle"],
        "selector": assessment["selector"],
        "score_margin": assessment["score_margin"],
        "original_size": list(assessment["original_size"]),
        "rotated_size": list(assessment["rotated_size"]),
        "osd": assessment.get("osd"),
        "osd_failure": assessment.get("osd_failure"),
        "view_failures": dict(assessment["view_failures"]),
    }
    if "orientation_prediction" in assessment:
        orientation["orientation_prediction"] = assessment["orientation_prediction"]
    if "orientation_detector_failure" in assessment:
        orientation["orientation_detector_failure"] = assessment[
            "orientation_detector_failure"
        ]
    provenance["orientation"] = orientation
    return replace(
        region,
        text_provenance=provenance,
        alternatives=list(region.alternatives),
    )


def _restore_structure_boxes(
    value: Any,
    angle: int,
    original_size: tuple[int, int],
    view_size: tuple[int, int],
) -> Any:
    if isinstance(value, list):
        return [
            _restore_structure_boxes(item, angle, original_size, view_size)
            for item in value
        ]
    if not isinstance(value, dict):
        return value
    if set(value) == {"left", "top", "right", "bottom"} and all(
        isinstance(value[key], int) and not isinstance(value[key], bool)
        for key in value
    ):
        restored = _restore_box(
            BoundingBox(**value),
            angle,
            original_size,
            view_size,
        )
        return {
            "left": restored.left,
            "top": restored.top,
            "right": restored.right,
            "bottom": restored.bottom,
        }
    return {
        key: _restore_structure_boxes(item, angle, original_size, view_size)
        for key, item in value.items()
    }


def _restore_box(
    box: BoundingBox,
    angle: int,
    original_size: tuple[int, int],
    view_size: tuple[int, int],
) -> BoundingBox:
    view_width, view_height = view_size
    if not (
        0 <= box.left < box.right <= view_width
        and 0 <= box.top < box.bottom <= view_height
    ):
        raise ReaderError(
            "invalid_orientation_box",
            f"Reader box falls outside the {view_width}x{view_height} rotated view",
        )
    width, height = original_size
    if angle == 0:
        restored = BoundingBox(box.left, box.top, box.right, box.bottom)
    elif angle == 90:
        restored = BoundingBox(width - box.bottom, box.left, width - box.top, box.right)
    elif angle == 180:
        restored = BoundingBox(
            width - box.right,
            height - box.bottom,
            width - box.left,
            height - box.top,
        )
    elif angle == 270:
        restored = BoundingBox(
            box.top, height - box.right, box.bottom, height - box.left
        )
    else:  # pragma: no cover - constructor and OSD validation constrain angles
        raise ReaderError("invalid_orientation_angle", f"Unsupported angle: {angle}")
    if not (
        0 <= restored.left < restored.right <= width
        and 0 <= restored.top < restored.bottom <= height
    ):
        raise ReaderError(
            "invalid_orientation_box",
            f"Restored box falls outside the {width}x{height} source page",
        )
    return restored


def _valid_osd(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ReaderError("invalid_osd_output", "OSD output is not an object")
    angle = value.get("angle")
    confidence = value.get("confidence")
    if angle not in ROTATIONS or isinstance(angle, bool):
        raise ReaderError("invalid_osd_output", "OSD returned an invalid angle")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ReaderError("invalid_osd_output", "OSD returned invalid confidence")
    return dict(value)


def _valid_orientation(value: object) -> dict[str, object]:
    prediction = _valid_osd(value)
    confidence = float(prediction["confidence"])
    if not 0 <= confidence <= 1:
        raise ReaderError(
            "invalid_orientation_classifier_output",
            "Orientation classifier confidence must be between zero and one",
        )
    return prediction


def _valid_size(value: object) -> bool:
    return bool(
        isinstance(value, list)
        and len(value) == 2
        and all(
            isinstance(item, int) and not isinstance(item, bool) and item > 0
            for item in value
        )
    )


def _optional_float(value: str | None) -> float | None:
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None
