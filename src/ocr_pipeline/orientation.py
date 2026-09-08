"""Lossless integer-rotation selection for local OCR readers."""

from __future__ import annotations

import re
import subprocess
import tempfile
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from functools import partial
from math import ceil, floor
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
DIRECT_ORIENTATION_CONFIDENCE = 0.9
MAX_SCORED_WORDS = 50
MIN_SUPPORTING_WORDS = 10
COVERAGE_RECOVERY_RATIO = 2.0
COVERAGE_CONFIDENCE_TOLERANCE = 0.03
GEOMETRY_CONFIDENCE_TOLERANCE = 0.03
MIN_GEOMETRY_WORDS = 10
HORIZONTAL_WORD_FRACTION = 0.65
VERTICAL_WORD_FRACTION = 0.35
ADAPTIVE_UPRIGHT_MEAN_CONFIDENCE = 0.8
ADAPTIVE_UPRIGHT_HIGH_CONFIDENCE_FRACTION = 0.65
ADAPTIVE_UPRIGHT_MIN_CHARACTERS = 40
ADAPTIVE_UPRIGHT_MAX_REPETITION = 0.02
VERTICAL_RESIDUAL_CONFIDENCE = 0.85
VERTICAL_RESIDUAL_ASPECT_RATIO = 2.0
VERTICAL_RESIDUAL_MARGIN_FRACTION = 0.2
VERTICAL_RESIDUAL_MAX_OVERLAP = 0.2
VERTICAL_RESIDUAL_MIN_CHARACTERS = 8
MARGIN_ROUTER_MAX_SIZE = 512
MARGIN_ROUTER_BAND_FRACTION = 0.2
MARGIN_ROUTER_EDGE_FRACTION = 0.01
MARGIN_ROUTER_DARK_PIXEL = 160
MARGIN_ROUTER_MIN_DARK_RATIO = 0.006
MARGIN_ROUTER_MIN_ROW_COVERAGE = 0.12
MARGIN_ROUTER_MIN_COLUMN_COVERAGE = 0.1
DOCTR_ARCH = "mobilenet_v3_small_page_orientation"
DOCTR_MODEL = {
    "library": "python-doctr",
    "architecture": DOCTR_ARCH,
    "publisher": "Mindee",
    "license": "Apache-2.0",
}
PADDLE_ORIENTATION_REPOSITORY = "PaddlePaddle/PP-LCNet_x1_0_doc_ori_onnx"
PADDLE_ORIENTATION_MODEL = {
    "library": "onnxruntime",
    "architecture": "PP-LCNet_x1_0_doc_ori",
    "publisher": "PaddlePaddle",
    "license": "Apache-2.0",
    "reported_accuracy": 0.9906,
}
# From the repository's own inference.yml, not inferred: resize short side to 256,
# centre crop 224, scale to [0,1], then ImageNet mean/std, CHW, top-1 over these labels.
PADDLE_ORIENTATION_LABELS = (0, 90, 180, 270)
PADDLE_RESIZE_SHORT = 256
PADDLE_CROP = 224
PADDLE_MEAN = (0.485, 0.456, 0.406)
PADDLE_STD = (0.229, 0.224, 0.225)


class PaddleDocOrientationDetector:
    """Predict one lossless page rotation with PaddleOCR's document-orientation model.

    docTR's small classifier returns near-chance confidence on pages that are not
    document-shaped, and the pipeline then rotates them on thin evidence. This model is
    purpose-built for the four-way document orientation task and reports 99.06% on
    PaddleOCR's own benchmark, at 7 MB.
    """

    name = "paddle-doc-orientation"
    # A dedicated four-class classifier reporting 99.06% is trusted at its own
    # argmax. Measured over the evaluation pages its correct calls run 0.73 to 0.93
    # with no errors, so docTR's 0.9 gate would discard sound predictions and hand
    # the page back to OCR-evidence voting, which is what rotated the posters.
    direct_confidence = 0.70

    def __init__(
        self,
        *,
        model_path: Path | None = None,
        repository: str = PADDLE_ORIENTATION_REPOSITORY,
        providers: tuple[str, ...] | None = None,
        session: Any | None = None,
    ) -> None:
        self.model_path = model_path
        self.repository = repository
        self.providers = providers
        self._session = session
        self._lock = threading.Lock()

    def __call__(self, image_path: Path) -> dict[str, object]:
        try:
            import numpy as np

            with Image.open(image_path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                tensor = _paddle_orientation_tensor(image, np)
        except (ImportError, OSError, UnidentifiedImageError, ValueError) as error:
            raise ReaderError(
                "orientation_classifier_image_failed", str(error)
            ) from error

        with self._lock:
            session = self._session or self._load_session()
            try:
                outputs = session.run(None, {session.get_inputs()[0].name: tensor})
            except Exception as error:
                raise ReaderError(
                    "orientation_classifier_failed", str(error)
                ) from error

        scores = outputs[0][0] if outputs else None
        if scores is None or len(scores) != len(PADDLE_ORIENTATION_LABELS):
            raise ReaderError(
                "invalid_orientation_classifier_output",
                "PP-LCNet returned an unexpected score vector",
            )
        index = int(max(range(len(scores)), key=lambda item: scores[item]))
        return {
            "angle": int(PADDLE_ORIENTATION_LABELS[index]),
            "confidence": float(scores[index]),
            "model": dict(PADDLE_ORIENTATION_MODEL),
        }

    def _load_session(self) -> Any:
        try:
            import onnxruntime
        except ImportError as error:
            raise ReaderError(
                "orientation_classifier_unavailable",
                "onnxruntime is required for the PaddleOCR orientation model",
            ) from error
        path = self.model_path
        if path is None:
            try:
                from huggingface_hub import hf_hub_download

                path = Path(hf_hub_download(self.repository, filename="inference.onnx"))
            except Exception as error:
                raise ReaderError(
                    "orientation_classifier_unavailable", str(error)
                ) from error
        providers = list(self.providers or onnxruntime.get_available_providers())
        try:
            self._session = onnxruntime.InferenceSession(str(path), providers=providers)
        except Exception as error:
            raise ReaderError(
                "orientation_classifier_unavailable", str(error)
            ) from error
        return self._session


def _paddle_orientation_tensor(image: Image.Image, np: Any) -> Any:
    width, height = image.size
    if min(width, height) <= 0:
        raise ValueError("page image has no area")
    scale = PADDLE_RESIZE_SHORT / min(width, height)
    resized = image.resize(
        (max(1, round(width * scale)), max(1, round(height * scale))),
        Image.BILINEAR,
    )
    left = max(0, (resized.width - PADDLE_CROP) // 2)
    top = max(0, (resized.height - PADDLE_CROP) // 2)
    cropped = resized.crop((left, top, left + PADDLE_CROP, top + PADDLE_CROP))
    array = np.asarray(cropped, dtype="float32") / 255.0
    array = (array - np.asarray(PADDLE_MEAN, dtype="float32")) / np.asarray(
        PADDLE_STD, dtype="float32"
    )
    return np.ascontiguousarray(array.transpose(2, 0, 1)[None], dtype="float32")


class DocTROrientationDetector:
    """Predict one lossless page rotation with docTR's small classifier."""

    name = "doctr-page-orientation"
    direct_confidence = DIRECT_ORIENTATION_CONFIDENCE

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
        margin_reader: LocalReader | None = None,
        defer_restore: bool = False,
        compare_ocr_views: bool = True,
        batch_size: int = 1,
    ) -> None:
        if osd_min_confidence < 0:
            raise ValueError("osd_min_confidence cannot be negative")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.reader = reader
        self.name = f"{reader.name}-oriented"
        self.osd_min_confidence = osd_min_confidence
        self.orientation_detector = orientation_detector
        self.margin_reader = margin_reader or reader
        self.defer_restore = defer_restore
        self.compare_ocr_views = compare_ocr_views
        self.batch_size = batch_size
        self.osd_detector = osd_detector
        if self.osd_detector is None and osd_executable:
            self.osd_detector = partial(
                detect_tesseract_orientation,
                executable=osd_executable,
            )
        self.executable = getattr(reader, "executable", None)
        self._assessments: dict[int, dict[str, object]] = {}
        self._lock = threading.Lock()

    def read_batch(
        self, image_paths: list[Path], page_numbers: list[int]
    ) -> list[list[TextRegion] | ReaderError]:
        """Overlap page preparation and I/O with an explicitly thread-safe reader."""
        if len(image_paths) != len(page_numbers):
            raise ValueError("image_paths and page_numbers must have equal lengths")
        if not image_paths:
            return []
        results: list[list[TextRegion] | ReaderError] = []
        with ThreadPoolExecutor(
            max_workers=min(self.batch_size, len(image_paths))
        ) as pool:
            futures = [
                pool.submit(self.read, path, number)
                for path, number in zip(image_paths, page_numbers, strict=True)
            ]
            for future in futures:
                try:
                    results.append(future.result())
                except ReaderError as error:
                    results.append(error)
        return results

    def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
        assessment: dict[str, Any] = {
            "page_number": page_number,
            "status": "running",
            "selector": "evidence_fallback",
            "view_scores": {},
            "view_failures": {},
            "nested_reader_reviews": {},
            "view_reader_execution": {},
            "margin_view_failures": {},
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
            angles, deferred_angles = self._candidate_angles(
                normalized_path,
                assessment,
            )

            views = self._read_views(
                normalized,
                root,
                page_number,
                angles,
                assessment,
            )
            rankable_views = list(views)
            if deferred_angles:
                initial_score = views[0][2] if len(views) == 1 else None
                accepted = _accept_upright_evidence(initial_score)
                assessment["adaptive_orientation"] = {
                    "status": "accepted_upright" if accepted else "expanded",
                    "initial_angles": angles,
                    "deferred_angles": deferred_angles,
                    "evidence": initial_score,
                }
                if not accepted:
                    expanded = self._read_views(
                        normalized,
                        root,
                        page_number,
                        deferred_angles,
                        assessment,
                    )
                    views.extend(expanded)
                    rankable_views.extend(expanded)

            margin_router = _vertical_margin_router(normalized, rankable_views)
            assessment["vertical_margin_router"] = margin_router
            margin_views = []
            if (
                margin_router["routed"]
                and len(rankable_views) == 1
                and rankable_views[0][0] == 0
            ):
                margin_views = self._read_margin_views(
                    normalized,
                    root,
                    page_number,
                    margin_router,
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

        ranked, selection_reason = _rank_views(rankable_views)
        angle, view_size, score, regions = ranked[0]
        assessment.update(
            {
                "angle": angle,
                "selected_score": score,
                "view_selection_reason": selection_reason,
                "score_margin_metric": (
                    "character_coverage"
                    if selection_reason == "coverage_recovery"
                    else (
                        "horizontal_word_fraction"
                        if selection_reason == "word_box_geometry"
                        else str(score["selection_metric"])
                    )
                ),
                "score_margin": _score_margin(ranked, selection_reason),
                "rotated_size": list(view_size),
            }
        )
        review_reasons = [
            reason
            for present, reason in (
                (angle != 0, "rotated_view_selected"),
                (
                    assessment["selector"]
                    in {
                        "evidence_fallback",
                        "orientation_evidence_fallback",
                        "orientation_uncertain",
                    },
                    str(assessment["selector"]),
                ),
                (bool(assessment["view_failures"]), "orientation_view_failed"),
                (
                    str(angle) in assessment["nested_reader_reviews"],
                    "selected_reader_requires_review",
                ),
            )
            if present
        ]
        assessment["review_reasons"] = review_reasons
        needs_review = bool(review_reasons)
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
            residuals = _recover_vertical_residuals(
                views,
                selected_angle=angle,
                selected_regions=restored,
                original_size=original_size,
                target_size=view_size if self.defer_restore else original_size,
                target_angle=angle if self.defer_restore else 0,
                page_number=page_number,
                assessment=assessment,
            )
            margin_residuals = _recover_margin_residuals(
                margin_views,
                selected_angle=angle,
                selected_regions=[*restored, *residuals],
                original_size=original_size,
                target_size=view_size if self.defer_restore else original_size,
                target_angle=angle if self.defer_restore else 0,
                page_number=page_number,
                assessment=assessment,
            )
        except ReaderError as error:
            self._fail(assessment, error.code, str(error))
            raise
        restored.extend(residuals)
        restored.extend(margin_residuals)
        assessment["vertical_residual_recovery"] = {
            "method": "orthogonal_margin_residual",
            "recovered_regions": len(residuals),
            "source_view_angles": sorted(
                {
                    int(
                        region.text_provenance["orientation_residual"][
                            "source_view_angle"
                        ]
                    )
                    for region in residuals
                    if region.text_provenance is not None
                }
            ),
        }
        assessment["side_margin_recovery"] = {
            "method": "side_margin_crop_rotation",
            "recovered_regions": len(margin_residuals),
            "source_margins": sorted(
                {
                    str(region.text_provenance["orientation_residual"]["source_margin"])
                    for region in margin_residuals
                    if region.text_provenance is not None
                }
            ),
        }
        if margin_router["routed"] and not margin_residuals:
            failures = assessment["margin_view_failures"]
            if margin_views:
                reason = "side_margin_recovery_unsupported"
            elif any(
                failure.get("code") != "empty_output" for failure in failures.values()
            ):
                reason = "side_margin_recovery_failed"
            else:
                reason = "side_margin_recovery_empty"
            review_reasons.append(reason)
            assessment["status"] = "review_recommended"
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
    ) -> tuple[list[int], list[int]]:
        prediction = self._detect_orientation(image_path, assessment)
        if not self.compare_ocr_views:
            # Generative token likelihood is not an orientation classifier: fluent
            # invented text can score higher than the correct source transcription.
            predicted = (
                int(prediction["angle"])
                if prediction
                and float(prediction["confidence"]) >= self._direct_confidence()
                else None
            )
            osd = self._detect_osd(image_path, assessment) if predicted != 0 else None
            detected = (
                int(osd["angle"])
                if osd and float(osd["confidence"]) >= self.osd_min_confidence
                else None
            )
            if predicted is not None and (detected is None or predicted == detected):
                assessment["selector"] = "orientation_classifier"
                return [predicted], []
            if predicted is None and detected is not None:
                assessment["selector"] = "tesseract_osd"
                return [detected], []
            assessment["selector"] = "orientation_uncertain"
            return [0], []
        if prediction is None:
            osd = self._detect_osd(image_path, assessment)
            if osd and float(osd["confidence"]) >= self.osd_min_confidence:
                angle = int(osd["angle"])
                assessment["selector"] = "tesseract_osd"
                if angle == 0:
                    return [0], []
                assessment["selector"] = "tesseract_osd_evidence_check"
                return [angle, 0], []
            return list(ROTATIONS), []

        angle = int(prediction["angle"])
        if angle == 0:
            if float(prediction["confidence"]) < self._direct_confidence():
                assessment["selector"] = "orientation_evidence_fallback"
                assessment["osd_status"] = "skipped_uncertain_zero_prediction"
                return [0], [90, 180, 270]
            assessment["selector"] = "orientation_classifier"
            assessment["osd_status"] = "skipped_zero_prediction"
            return [0], []

        osd = self._detect_osd(image_path, assessment)
        if osd and int(osd["angle"]) == angle:
            assessment["selector"] = "orientation_agreement"
            return [angle], []
        assessment["selector"] = "orientation_evidence_fallback"
        alternate = int(osd["angle"]) if osd else 0
        return list(dict.fromkeys((angle, alternate))), []

    def _direct_confidence(self) -> float:
        return float(
            getattr(
                self.orientation_detector,
                "direct_confidence",
                DIRECT_ORIENTATION_CONFIDENCE,
            )
        )

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
                nested_execution = self._nested_reader_execution(page_number)
                if nested_execution is not None:
                    assessment["view_reader_execution"][str(angle)] = nested_execution
                assessment["view_failures"][str(angle)] = {
                    "code": error.code,
                    "message": str(error),
                }
                continue
            nested_execution = self._nested_reader_execution(page_number)
            if nested_execution is not None:
                assessment["view_reader_execution"][str(angle)] = nested_execution
            nested_review = getattr(self.reader, "page_needs_review", None)
            if callable(nested_review) and nested_review(page_number):
                assessment["nested_reader_reviews"][str(angle)] = True
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

    def _read_margin_views(
        self,
        source: Image.Image,
        root: Path,
        page_number: int,
        router: dict[str, object],
        assessment: dict[str, Any],
    ) -> list[
        tuple[str, tuple[int, int, int, int], int, tuple[int, int], list[TextRegion]]
    ]:
        views = []
        margins = router.get("margins")
        routed_sides = router.get("routed_sides")
        if not isinstance(margins, dict) or not isinstance(routed_sides, list):
            return views
        for side in routed_sides:
            stats = margins.get(side)
            if not isinstance(side, str) or not isinstance(stats, dict):
                continue
            crop_values = stats.get("crop")
            if not (
                isinstance(crop_values, list)
                and len(crop_values) == 4
                and all(isinstance(value, int) for value in crop_values)
            ):
                continue
            crop_box = (
                crop_values[0],
                crop_values[1],
                crop_values[2],
                crop_values[3],
            )
            crop = source.crop(crop_box)
            for angle in (90, 270):
                image = crop.transpose(ROTATIONS[angle])
                path = root / f"page-margin-{side}-{angle}.png"
                image.save(path, format="PNG")
                key = f"{side}:{angle}"
                try:
                    regions = self.margin_reader.read(path, page_number)
                except ReaderError as error:
                    assessment["margin_view_failures"][key] = {
                        "code": error.code,
                        "message": str(error),
                    }
                    continue
                if not regions:
                    assessment["margin_view_failures"][key] = {
                        "code": "empty_output",
                        "message": "The reader returned no text regions",
                    }
                    continue
                views.append((side, crop_box, angle, image.size, regions))
        return views

    def _nested_reader_execution(
        self,
        page_number: int,
    ) -> dict[str, object] | None:
        coverage = getattr(self.reader, "coverage_assessment", None)
        if not callable(coverage):
            return None
        assessment = coverage(page_number)
        if not isinstance(assessment, dict):
            return None
        pages = assessment.get("pages")
        if not isinstance(pages, list) or len(pages) < page_number:
            return None
        page = pages[page_number - 1]
        if not isinstance(page, dict):
            return None
        keys = (
            "ran",
            "status",
            "selected_view",
            "fallback_reader",
            "fallback_reader_runs",
            "fallback_candidates",
            "confirmation_reader",
            "confirmation_reader_runs",
            "qualifying_regions",
            "band_count",
            "replaced_bands",
            "review_reasons",
            "nested_reader",
        )
        return {
            "reader": self.reader.name,
            **{key: deepcopy(page[key]) for key in keys if key in page},
        }

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
    horizontal_words = 0
    geometry_words = 0
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
        if characters:
            width = region.bounding_box.right - region.bounding_box.left
            height = region.bounding_box.bottom - region.bounding_box.top
            if width > 0 and height > 0:
                geometry_words += 1
                horizontal_words += int(width >= height)

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
    horizontal_word_fraction = (
        horizontal_words / geometry_words if geometry_words else 0.0
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
        "horizontal_word_fraction": round(horizontal_word_fraction, 6),
        "geometry_words": geometry_words,
        "characters": character_count,
        "words": len(confidences),
        "supporting_words": supporting_words,
    }


def _accept_upright_evidence(score: dict[str, object] | None) -> bool:
    if score is None:
        return False
    return (
        int(score["supporting_words"]) >= MIN_SUPPORTING_WORDS
        and int(score["geometry_words"]) >= MIN_GEOMETRY_WORDS
        and int(score["characters"]) >= ADAPTIVE_UPRIGHT_MIN_CHARACTERS
        and float(score["horizontal_word_fraction"]) >= HORIZONTAL_WORD_FRACTION
        and float(score["mean_confidence"]) >= ADAPTIVE_UPRIGHT_MEAN_CONFIDENCE
        and float(score["high_confidence_fraction"])
        >= ADAPTIVE_UPRIGHT_HIGH_CONFIDENCE_FRACTION
        and float(score["repetition_fraction"]) <= ADAPTIVE_UPRIGHT_MAX_REPETITION
    )


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

    horizontal_views = [
        view
        for view in ranked
        if int(view[2]["geometry_words"]) >= MIN_GEOMETRY_WORDS
        and float(view[2]["horizontal_word_fraction"]) >= HORIZONTAL_WORD_FRACTION
    ]
    if (
        int(strongest_score["geometry_words"]) >= MIN_GEOMETRY_WORDS
        and float(strongest_score["horizontal_word_fraction"]) <= VERTICAL_WORD_FRACTION
        and horizontal_views
    ):
        horizontal = horizontal_views[0]
        horizontal_score = horizontal[2]
        confidence_is_close = float(
            horizontal_score["mean_confidence"]
        ) + GEOMETRY_CONFIDENCE_TOLERANCE >= float(strongest_score["mean_confidence"])
        if horizontal != strongest and confidence_is_close:
            return [horizontal, *(item for item in ranked if item != horizontal)], (
                "word_box_geometry"
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
    if selection_reason == "word_box_geometry":
        return round(
            float(best["horizontal_word_fraction"])
            - float(second["horizontal_word_fraction"]),
            6,
        )
    second_value = (
        float(second["selection_value"])
        if second["selection_metric"] == best["selection_metric"]
        else 0.0
    )
    best_value = float(best["selection_value"])
    return round((best_value - second_value) / max(best_value, 1e-9), 6)


def _recover_vertical_residuals(
    views: list[tuple[int, tuple[int, int], dict[str, object], list[TextRegion]]],
    *,
    selected_angle: int,
    selected_regions: list[TextRegion],
    original_size: tuple[int, int],
    target_size: tuple[int, int],
    target_angle: int,
    page_number: int,
    assessment: dict[str, Any],
) -> list[TextRegion]:
    candidates: list[tuple[TextRegion, int, BoundingBox]] = []
    selected_text = {_normalized_text(region.text) for region in selected_regions}
    for angle, view_size, _, regions in views:
        if (angle - selected_angle) % 180 != 90:
            continue
        for region in regions:
            if not _is_vertical_residual_source(region):
                continue
            restored_box = _restore_box(
                region.bounding_box,
                angle,
                original_size,
                view_size,
            )
            if not _is_vertical_margin_box(restored_box, original_size):
                continue
            target_box = (
                restored_box
                if target_angle == 0
                else _rotate_box(
                    restored_box,
                    target_angle,
                    original_size,
                    target_size,
                )
            )
            if not _is_vertical_margin_box(target_box, target_size):
                continue
            if _normalized_text(region.text) in selected_text:
                continue
            if any(
                _box_overlap(target_box, selected.bounding_box)
                > VERTICAL_RESIDUAL_MAX_OVERLAP
                for selected in selected_regions
            ):
                continue
            candidates.append((region, angle, target_box))

    supported = [
        candidate
        for candidate in candidates
        if _has_vertical_support(candidate, candidates)
    ]
    accepted: list[tuple[TextRegion, int, BoundingBox]] = []
    for candidate in sorted(
        supported,
        key=lambda item: (
            -_box_area(item[2]),
            -(item[0].confidence or 0.0),
            item[1],
            item[0].reading_order,
        ),
    ):
        if any(_box_overlap(candidate[2], existing[2]) >= 0.5 for existing in accepted):
            continue
        accepted.append(candidate)

    existing_ids = {region.id for region in selected_regions}
    next_order = max((region.reading_order for region in selected_regions), default=0)
    recovered = []
    for source, angle, box in sorted(
        accepted,
        key=lambda item: (item[1], item[0].reading_order, item[2].top, item[2].left),
    ):
        identifier = len(recovered) + 1
        region_id = f"p{page_number}-orientation-residual-{identifier}"
        while region_id in existing_ids:
            identifier += 1
            region_id = f"p{page_number}-orientation-residual-{identifier}"
        existing_ids.add(region_id)
        provenance = dict(source.text_provenance or {})
        provenance["orientation_residual"] = {
            "method": "orthogonal_margin_residual",
            "source_view_angle": angle,
            "selected_view_angle": selected_angle,
            "original_provider": source.provider,
        }
        recovered.append(
            _annotate_region(
                replace(
                    source,
                    id=region_id,
                    bounding_box=box,
                    reading_order=next_order + len(recovered) + 1,
                    text_provenance=provenance,
                    alternatives=list(source.alternatives),
                ),
                assessment,
            )
        )
    return recovered


def _recover_margin_residuals(
    views: list[
        tuple[str, tuple[int, int, int, int], int, tuple[int, int], list[TextRegion]]
    ],
    *,
    selected_angle: int,
    selected_regions: list[TextRegion],
    original_size: tuple[int, int],
    target_size: tuple[int, int],
    target_angle: int,
    page_number: int,
    assessment: dict[str, Any],
) -> list[TextRegion]:
    candidates: list[
        tuple[TextRegion, str, tuple[int, int, int, int], int, BoundingBox]
    ] = []
    selected_text = {_normalized_text(region.text) for region in selected_regions}
    for side, crop_box, angle, view_size, regions in views:
        crop_left, crop_top, crop_right, crop_bottom = crop_box
        crop_size = (crop_right - crop_left, crop_bottom - crop_top)
        for region in regions:
            if not _is_margin_residual_source(region):
                continue
            try:
                crop_region = _restore_box(
                    region.bounding_box,
                    angle,
                    crop_size,
                    view_size,
                )
            except ReaderError:
                continue
            restored_box = BoundingBox(
                crop_region.left + crop_left,
                crop_region.top + crop_top,
                crop_region.right + crop_left,
                crop_region.bottom + crop_top,
            )
            if not _is_vertical_margin_box(
                restored_box,
                original_size,
                minimum_aspect_ratio=0.5,
            ):
                continue
            target_box = (
                restored_box
                if target_angle == 0
                else _rotate_box(
                    restored_box,
                    target_angle,
                    original_size,
                    target_size,
                )
            )
            if _normalized_text(region.text) in selected_text:
                continue
            if any(
                _box_overlap(target_box, selected.bounding_box)
                > VERTICAL_RESIDUAL_MAX_OVERLAP
                for selected in selected_regions
            ):
                continue
            candidates.append((region, side, crop_box, angle, target_box))

    supported = [
        candidate
        for candidate in candidates
        if _has_margin_support(candidate, candidates)
    ]
    accepted: list[
        tuple[TextRegion, str, tuple[int, int, int, int], int, BoundingBox]
    ] = []
    for candidate in sorted(
        supported,
        key=lambda item: (
            -_box_area(item[4]),
            -(item[0].confidence or 0.0),
            item[1],
            item[3],
            item[0].reading_order,
        ),
    ):
        if any(_box_overlap(candidate[4], existing[4]) >= 0.5 for existing in accepted):
            continue
        accepted.append(candidate)

    existing_ids = {region.id for region in selected_regions}
    next_order = max((region.reading_order for region in selected_regions), default=0)
    recovered = []
    for source, side, crop_box, angle, box in sorted(
        accepted,
        key=lambda item: (item[1], item[3], item[0].reading_order),
    ):
        identifier = len(recovered) + 1
        region_id = f"p{page_number}-orientation-residual-{identifier}"
        while region_id in existing_ids:
            identifier += 1
            region_id = f"p{page_number}-orientation-residual-{identifier}"
        existing_ids.add(region_id)
        provenance = dict(source.text_provenance or {})
        provenance["orientation_residual"] = {
            "method": "side_margin_crop_rotation",
            "source_view_angle": angle,
            "selected_view_angle": selected_angle,
            "source_margin": side,
            "source_crop": list(crop_box),
            "original_provider": source.provider,
        }
        recovered.append(
            _annotate_region(
                replace(
                    source,
                    id=region_id,
                    bounding_box=box,
                    reading_order=next_order + len(recovered) + 1,
                    text_provenance=provenance,
                    alternatives=list(source.alternatives),
                ),
                assessment,
            )
        )
    return recovered


def _vertical_margin_router(
    image: Image.Image,
    views: list[tuple[int, tuple[int, int], dict[str, object], list[TextRegion]]],
) -> dict[str, object]:
    upright = next((regions for angle, _, _, regions in views if angle == 0), None)
    if upright is None:
        return {
            "routed": False,
            "reason": "upright_view_unavailable",
            "margins": {},
            "routed_sides": [],
        }

    width, height = image.size
    scale = min(1.0, MARGIN_ROUTER_MAX_SIZE / max(width, height))
    reduced = image.convert("L").resize(
        (max(1, round(width * scale)), max(1, round(height * scale))),
        Image.Resampling.BOX,
    )
    reduced_width, reduced_height = reduced.size
    covered = bytearray(reduced_width * reduced_height)
    for region in upright:
        box = region.bounding_box
        left = max(0, min(reduced_width, floor(box.left * scale)))
        top = max(0, min(reduced_height, floor(box.top * scale)))
        right = max(0, min(reduced_width, ceil(box.right * scale)))
        bottom = max(0, min(reduced_height, ceil(box.bottom * scale)))
        if left >= right or top >= bottom:
            continue
        row = bytes([1]) * (right - left)
        for y in range(top, bottom):
            start = y * reduced_width + left
            covered[start : start + len(row)] = row

    pixels = list(reduced.getdata())
    edge = max(1, round(reduced_width * MARGIN_ROUTER_EDGE_FRACTION))
    band = max(edge + 1, round(reduced_width * MARGIN_ROUTER_BAND_FRACTION))
    ranges = {
        "left": (edge, band),
        "right": (reduced_width - band, reduced_width - edge),
    }
    crop_boxes = {
        "left": [0, 0, max(1, ceil(width * MARGIN_ROUTER_BAND_FRACTION)), height],
        "right": [
            min(width - 1, floor(width * (1 - MARGIN_ROUTER_BAND_FRACTION))),
            0,
            width,
            height,
        ],
    }
    margins = {
        side: {
            **_uncovered_margin_ink(
                pixels,
                covered,
                reduced_width,
                reduced_height,
                left,
                right,
            ),
            "crop": crop_boxes[side],
        }
        for side, (left, right) in ranges.items()
        if left < right
    }
    routed_sides = [
        side
        for side, stats in margins.items()
        if stats["dark_ratio"] >= MARGIN_ROUTER_MIN_DARK_RATIO
        and stats["row_coverage"] >= MARGIN_ROUTER_MIN_ROW_COVERAGE
        and stats["column_coverage"] >= MARGIN_ROUTER_MIN_COLUMN_COVERAGE
    ]
    return {
        "routed": bool(routed_sides),
        "reason": (
            "uncovered_side_margin_ink" if routed_sides else "no_margin_evidence"
        ),
        "margins": margins,
        "routed_sides": routed_sides,
    }


def _uncovered_margin_ink(
    pixels: list[int],
    covered: bytearray,
    width: int,
    height: int,
    left: int,
    right: int,
) -> dict[str, float]:
    dark_pixels = 0
    dark_rows = 0
    dark_columns: set[int] = set()
    minimum_row_pixels = max(2, (right - left) // 50)
    for y in range(height):
        row_dark = 0
        offset = y * width
        for x in range(left, right):
            index = offset + x
            if not covered[index] and pixels[index] < MARGIN_ROUTER_DARK_PIXEL:
                row_dark += 1
                dark_columns.add(x)
        dark_pixels += row_dark
        dark_rows += int(row_dark >= minimum_row_pixels)
    area = max((right - left) * height, 1)
    return {
        "dark_ratio": round(dark_pixels / area, 6),
        "row_coverage": round(dark_rows / max(height, 1), 6),
        "column_coverage": round(len(dark_columns) / max(right - left, 1), 6),
    }


def _is_vertical_residual_source(region: TextRegion) -> bool:
    if region.kind not in {"text", "word"} or region.resolution != "resolved":
        return False
    if (
        not region.text.strip()
        or (region.confidence or 0.0) < VERTICAL_RESIDUAL_CONFIDENCE
    ):
        return False
    width = region.bounding_box.right - region.bounding_box.left
    height = region.bounding_box.bottom - region.bounding_box.top
    return width >= height * VERTICAL_RESIDUAL_ASPECT_RATIO


def _is_margin_residual_source(region: TextRegion) -> bool:
    if region.kind not in {"text", "word"} or region.resolution != "resolved":
        return False
    if (
        not region.text.strip()
        or (region.confidence or 0.0) < VERTICAL_RESIDUAL_CONFIDENCE
    ):
        return False
    return True


def _is_vertical_margin_box(
    box: BoundingBox,
    page_size: tuple[int, int],
    *,
    minimum_aspect_ratio: float = VERTICAL_RESIDUAL_ASPECT_RATIO,
) -> bool:
    width = box.right - box.left
    height = box.bottom - box.top
    page_width, _ = page_size
    center = (box.left + box.right) / 2
    in_side_margin = center <= page_width * VERTICAL_RESIDUAL_MARGIN_FRACTION or (
        center >= page_width * (1 - VERTICAL_RESIDUAL_MARGIN_FRACTION)
    )
    return height >= width * minimum_aspect_ratio and in_side_margin


def _has_vertical_support(
    candidate: tuple[TextRegion, int, BoundingBox],
    candidates: list[tuple[TextRegion, int, BoundingBox]],
) -> bool:
    region, angle, box = candidate
    if len(_normalized_text(region.text)) >= VERTICAL_RESIDUAL_MIN_CHARACTERS:
        return True
    aligned_characters = sum(
        len(_normalized_text(other.text))
        for other, other_angle, other_box in candidates
        if other_angle == angle and _horizontal_overlap(box, other_box) >= 0.5
    )
    return aligned_characters >= VERTICAL_RESIDUAL_MIN_CHARACTERS


def _has_margin_support(
    candidate: tuple[
        TextRegion,
        str,
        tuple[int, int, int, int],
        int,
        BoundingBox,
    ],
    candidates: list[
        tuple[
            TextRegion,
            str,
            tuple[int, int, int, int],
            int,
            BoundingBox,
        ]
    ],
) -> bool:
    region, side, _, angle, box = candidate
    text = _normalized_text(region.text)
    aligned_characters = 0
    for other_candidate in candidates:
        if other_candidate is candidate:
            continue
        other, other_side, _, other_angle, other_box = other_candidate
        other_text = _normalized_text(other.text)
        if (
            other_side == side
            and _horizontal_overlap(box, other_box) >= 0.5
            and (other_angle == angle or other_text == text)
        ):
            aligned_characters += len(other_text)
    return aligned_characters >= VERTICAL_RESIDUAL_MIN_CHARACTERS


def _normalized_text(text: str) -> str:
    return "".join(character for character in text.casefold() if character.isalnum())


def _box_area(box: BoundingBox) -> int:
    return (box.right - box.left) * (box.bottom - box.top)


def _box_overlap(first: BoundingBox, second: BoundingBox) -> float:
    width = max(0, min(first.right, second.right) - max(first.left, second.left))
    height = max(0, min(first.bottom, second.bottom) - max(first.top, second.top))
    intersection = width * height
    return intersection / max(min(_box_area(first), _box_area(second)), 1)


def _horizontal_overlap(first: BoundingBox, second: BoundingBox) -> float:
    overlap = max(0, min(first.right, second.right) - max(first.left, second.left))
    narrower = min(first.right - first.left, second.right - second.left)
    return overlap / max(narrower, 1)


def _rotate_box(
    box: BoundingBox,
    angle: int,
    original_size: tuple[int, int],
    view_size: tuple[int, int],
) -> BoundingBox:
    width, height = original_size
    if angle == 0:
        rotated = BoundingBox(box.left, box.top, box.right, box.bottom)
    elif angle == 90:
        rotated = BoundingBox(box.top, width - box.right, box.bottom, width - box.left)
    elif angle == 180:
        rotated = BoundingBox(
            width - box.right,
            height - box.bottom,
            width - box.left,
            height - box.top,
        )
    elif angle == 270:
        rotated = BoundingBox(
            height - box.bottom,
            box.left,
            height - box.top,
            box.right,
        )
    else:  # pragma: no cover - caller supplies one of ROTATIONS
        raise ReaderError("invalid_orientation_angle", f"Unsupported angle: {angle}")
    view_width, view_height = view_size
    if not (
        0 <= rotated.left < rotated.right <= view_width
        and 0 <= rotated.top < rotated.bottom <= view_height
    ):
        raise ReaderError(
            "invalid_orientation_box",
            f"Rotated box falls outside the {view_width}x{view_height} target view",
        )
    return rotated


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
    if assessment["nested_reader_reviews"]:
        orientation["nested_reader_reviews"] = dict(assessment["nested_reader_reviews"])
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
