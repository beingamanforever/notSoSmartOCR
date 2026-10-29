"""Read each page upright using docTR's page classifier and skew estimate."""

from __future__ import annotations

import hashlib
import math
import tempfile
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError

from .cache import BoundedCache
from .contracts import BoundingBox, TextRegion
from .providers import LocalReader, ReaderError
from .text_layer import sidecar

DOCTR_ARCH = "mobilenet_v3_small_page_orientation"
DOCTR_MODEL = {
    "library": "python-doctr",
    "architecture": DOCTR_ARCH,
    "publisher": "Mindee",
    "license": "Apache-2.0",
}
# Falcon read straight contract pages losslessly up to 1 degree of skew and dropped
# lines from 1.5; jdeskew's worst error on real pages turned by known angles was 0.86.
MIN_SKEW_DEGREES = 1.0
# jdeskew's DISE-2021 operating point (Pham et al., ICIP 2022): mean error 0.08 degrees.
SKEW_ESTIMATE_HEIGHT = 2048
# Any angle is a tilt within 45 degrees plus a quarter turn. The page is levelled first
# (jdeskew searching +-45) and only the level page is classified, as the 4-class model
# was trained: on ORB this took arbitrary angles from 45.9% to 83.9% (Indic, 8 angles)
# and 94% to 100% (English, 12 angles); 4-class splits were unchanged within 0.8 points.
SKEW_SEARCH_DEGREES = 45.0
# The level page is classified in its four quarter turns and the votes, rolled into one
# frame, are averaged in log space; the model's bias towards one turn cancels. Tiles
# were dropped: on dense Latin scans they helped by 1 point, on ORB Indic they cost 6.
CLASSIFY_SIDE = 1650
# Temperature (Guo et al. 2017) fitted by NLL on 597 real pages in all four turns
# (document-grouped 5-fold CV 0.49-0.72; held out: 9 of 11 sideways pages fixed, none
# of 586 upright pages turned under the prior and review rule below).
OFFICIAL_TEMPERATURE = 0.6756
# Smoothed label counts of the same pages: sideways either way, upside down almost never.
OFFICIAL_UPRIGHT_PRIOR = 0.979
UPSIDE_DOWN_PRIOR = 0.0008
# Chow's reject rule: review when the chosen turn's posterior is below 1 - review cost.
REVIEW_CONFIDENCE = 0.9


@dataclass(frozen=True)
class PageOrientation:
    """A counterclockwise quarter-turn correction plus a small skew correction."""

    quarter_turns: int
    skew: float
    confidence: float
    probabilities: tuple[float, ...]

    @property
    def angle(self) -> int:
        return 90 * self.quarter_turns

    @property
    def rotation(self) -> float:
        return (self.angle + self.skew) % 360


class DocTROrientationDetector:
    """docTR's page classifier averaged over the four page rotations."""

    name = "doctr-page-orientation"

    def __init__(
        self,
        *,
        device: str = "cuda",
        weights: Path | None = None,
        temperature: float = OFFICIAL_TEMPERATURE,
        upright_prior: float = OFFICIAL_UPRIGHT_PRIOR,
        prior: tuple[float, float, float, float] | None = None,
        classifier: Callable[[list[Any]], Any] | None = None,
        skew_estimator: Callable[[Any], float] | None = None,
    ) -> None:
        """`prior` (turned 0, 1, 2, 3 quarter turns) replaces the intake prior that
        `upright_prior` builds: the class mix is a property of the intake, not the model."""
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not 0 < upright_prior < 1 - UPSIDE_DOWN_PRIOR:
            raise ValueError("upright_prior must be between 0 and 1")
        if prior is not None and (
            len(prior) != 4
            or not all(math.isfinite(p) and p > 0 for p in prior)
            or not math.isclose(sum(prior), 1.0, abs_tol=1e-6)
        ):
            raise ValueError("prior must be four positive probabilities summing to one")
        self.device = device
        self.weights = weights
        self.temperature = temperature
        sideways = (1 - upright_prior - UPSIDE_DOWN_PRIOR) / 2
        self.prior = (
            tuple(prior)
            if prior is not None
            else (
                upright_prior,
                sideways,
                UPSIDE_DOWN_PRIOR,
                sideways,
            )
        )
        self._classifier = classifier
        self._skew_estimator = skew_estimator
        self._lock = threading.Lock()

    def __call__(self, image: Image.Image) -> PageOrientation:
        import numpy as np

        page = image.convert("RGB")
        with self._lock:
            classify, estimate_skew = self._models()
        try:
            skew = float(estimate_skew(np.asarray(page)))
        except Exception as error:
            raise ReaderError("orientation_skew_failed", str(error)) from error
        if not math.isfinite(skew):
            raise ReaderError(
                "invalid_orientation_skew_output", "the skew estimate must be finite"
            )
        if abs(skew) < MIN_SKEW_DEGREES:
            skew = 0.0
        # Quarter turns and the skew rotation commute, so the quarter turn found on the
        # level page and the skew fold into one orientation of the source page.
        turned, posterior = self._quarter_turn(_level(page, skew), classify)
        return PageOrientation(
            quarter_turns=(-turned) % 4,
            skew=skew,
            confidence=float(posterior[turned]),
            probabilities=tuple(float(value) for value in posterior),
        )

    def _quarter_turn(
        self, image: Image.Image, classify: Callable[[list[Any]], Any]
    ) -> tuple[int, Any]:
        """The counterclockwise class the page is turned by, and its posterior."""
        import numpy as np

        page = image.copy()
        page.thumbnail((CLASSIFY_SIDE, CLASSIFY_SIDE))
        # docTR makes each view contiguous when preprocessing it; keep views here.
        views = [np.rot90(np.asarray(page), turn) for turn in range(4)]
        with self._lock:
            try:
                probabilities = np.asarray(classify(views), dtype=float)
            except Exception as error:
                raise ReaderError(
                    "orientation_classifier_failed", str(error)
                ) from error
        if (
            probabilities.shape != (len(views), 4)
            or not np.isfinite(probabilities).all()
            or (probabilities < 0).any()
            or (probabilities > 1).any()
            or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
        ):
            raise ReaderError(
                "invalid_orientation_classifier_output",
                "docTR must return four finite class probabilities summing to one "
                "for each view",
            )
        # Class k means the page was turned k quarter turns counterclockwise; view i
        # was turned i % 4 more, so its vote is rolled back into the page's frame.
        votes = [
            np.roll(vote, -(index % 4)) for index, vote in enumerate(probabilities)
        ]
        scores = np.log(np.clip(votes, 1e-12, 1)).mean(0) / self.temperature
        scores += np.log(self.prior)
        posterior = np.exp(scores - scores.max())
        posterior /= posterior.sum()
        turned = int(posterior.argmax())
        # Below the review threshold a turn is kept as supplied (and reviewed): on real
        # pages as supplied, 1 of 5 such turns was right (upright slides were turned).
        if turned and posterior[turned] < REVIEW_CONFIDENCE:
            turned = 0
        return turned, posterior

    def _models(self) -> tuple[Callable[[list[Any]], Any], Callable[[Any], float]]:
        if self._classifier is not None and self._skew_estimator is not None:
            return self._classifier, self._skew_estimator
        try:
            import torch
            from doctr.models import page_orientation_predictor
        except (ImportError, OSError) as error:
            raise ReaderError(
                "orientation_classifier_unavailable",
                "python-doctr and a compatible PyTorch build are required",
            ) from error
        if self._classifier is None:
            if self.device.startswith("cuda") and not torch.cuda.is_available():
                raise ReaderError(
                    "orientation_classifier_unavailable",
                    f"CUDA is unavailable for requested device {self.device}",
                )
            predictor = page_orientation_predictor(
                arch=DOCTR_ARCH, pretrained=True, batch_size=4
            )
            if self.weights is not None:
                # A checkpoint from docTR's official references/classification script.
                state = torch.load(self.weights, map_location="cpu")
                predictor.model.load_state_dict(state)
            predictor = predictor.to(self.device)
            predictor.model.eval()

            def classify(views: list[Any]) -> Any:
                # The official predictor returns only a rounded top score.
                with torch.inference_mode():
                    return torch.cat(
                        [
                            torch.softmax(
                                predictor.model(batch.to(self.device)).float(), 1
                            ).cpu()
                            for batch in predictor.pre_processor(views)
                        ]
                    )

            self._classifier = classify
        if self._skew_estimator is None:
            self._skew_estimator = _skew_angle
        return self._classifier, self._skew_estimator


def _skew_angle(page: Any) -> float:
    """Counterclockwise degrees that level the page's text (Pillow's rotate sign)."""
    from jdeskew.estimator import get_angle

    return get_angle(
        page, vertical_image_shape=SKEW_ESTIMATE_HEIGHT, angle_max=SKEW_SEARCH_DEGREES
    )


def _level(image: Image.Image, skew: float) -> Image.Image:
    """The page rotated level and cut to its ink, the corners filled with its own border."""
    import numpy as np

    if not skew:
        return image
    pixels = np.asarray(image)
    edge = np.concatenate([pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]])
    # White corners on a photographed or coloured page would add borders it never had.
    fill = tuple(int(value) for value in np.median(edge, 0))
    level = image.rotate(
        skew, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=fill
    )
    # The expanded canvas leaves the page small in its margins; classify the ink.
    ink = np.asarray(level.convert("L")) < 128
    rows, cols = np.flatnonzero(ink.any(1)), np.flatnonzero(ink.any(0))
    if not rows.size:
        return level
    return level.crop(
        (int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)
    )


TEXT_LAYER_UPRIGHT = PageOrientation(0, 0.0, 1.0, (1.0, 0.0, 0.0, 0.0))


@dataclass(frozen=True)
class _PageView:
    orientation: PageOrientation
    source_size: tuple[int, int]
    view_size: tuple[int, int]
    to_source: tuple[float, float, float, float, float, float]

    def box_to_source(self, box: BoundingBox) -> BoundingBox:
        a, b, c, d, e, f = self.to_source
        corners = [
            (a * x + b * y + c, d * x + e * y + f)
            for x in (box.left, box.right)
            for y in (box.top, box.bottom)
        ]
        width, height = self.source_size
        left = min(width, max(0, math.floor(min(x for x, _ in corners) + 1e-6)))
        top = min(height, max(0, math.floor(min(y for _, y in corners) + 1e-6)))
        right = max(0, min(width, math.ceil(max(x for x, _ in corners) - 1e-6)))
        bottom = max(0, min(height, math.ceil(max(y for _, y in corners) - 1e-6)))
        if right <= left or bottom <= top:
            raise ReaderError(
                "orientation_box_outside_source",
                "a detected region lies outside the source page after restoring orientation",
            )
        return BoundingBox(left, top, right, bottom)


def _page_view(size: tuple[int, int], orientation: PageOrientation) -> _PageView:
    """Compose Pillow's destination-to-source matrices for the two rotations.

    Both rotations expand the canvas to preserve edge content at the source scale.
    """
    turn, upright = _rotation_matrix(size, orientation.angle, expand=True)
    skew, view_size = _rotation_matrix(upright, orientation.skew, expand=True)
    a1, b1, c1, d1, e1, f1 = turn
    a2, b2, c2, d2, e2, f2 = skew
    composed = (
        a1 * a2 + b1 * d2,
        a1 * b2 + b1 * e2,
        a1 * c2 + b1 * f2 + c1,
        d1 * a2 + e1 * d2,
        d1 * b2 + e1 * e2,
        d1 * c2 + e1 * f2 + f1,
    )
    return _PageView(orientation, size, view_size, composed)


def _rotation_matrix(
    size: tuple[int, int], degrees: float, *, expand: bool
) -> tuple[tuple[float, ...], tuple[int, int]]:
    """Pillow's own `Image.rotate` destination-to-source matrix and output size."""
    width, height = size
    degrees %= 360
    # Pillow transposes exact quarter turns, including mixed odd/even dimensions.
    if expand and degrees == 90:
        return (0.0, -1.0, float(width), 1.0, 0.0, 0.0), (height, width)
    if expand and degrees == 270:
        return (0.0, 1.0, 0.0, -1.0, 0.0, float(height)), (height, width)
    angle = -math.radians(degrees)
    cos, sin = round(math.cos(angle), 15), round(math.sin(angle), 15)
    a, b, d, e = cos, sin, -sin, cos

    def apply(x: float, y: float, c: float, f: float) -> tuple[float, float]:
        return a * x + b * y + c, d * x + e * y + f

    c, f = apply(-width / 2, -height / 2, 0.0, 0.0)
    c, f = c + width / 2, f + height / 2
    if not expand:
        return (a, b, c, d, e, f), size
    corners = [
        apply(x, y, c, f) for x, y in ((0, 0), (width, 0), (width, height), (0, height))
    ]
    view_width = math.ceil(max(x for x, _ in corners)) - math.floor(
        min(x for x, _ in corners)
    )
    view_height = math.ceil(max(y for _, y in corners)) - math.floor(
        min(y for _, y in corners)
    )
    c, f = apply(-(view_width - width) / 2, -(view_height - height) / 2, c, f)
    return (a, b, c, d, e, f), (view_width, view_height)


def _render_view(image: Image.Image, view: _PageView) -> Image.Image:
    upright = image.rotate(view.orientation.angle, expand=True)
    if view.orientation.skew == 0:
        return upright
    return upright.rotate(
        view.orientation.skew,
        resample=Image.Resampling.BICUBIC,
        expand=True,
        fillcolor="white",
    )


class OrientationReader:
    """Read the upright view of each page and map its boxes back to the source."""

    def __init__(
        self,
        reader: LocalReader,
        *,
        detector: Callable[[Image.Image], PageOrientation] | None = None,
        defer_restore: bool = False,
        batch_size: int = 1,
        page_cache: BoundedCache | None = None,
    ) -> None:
        """`page_cache` returns the stored reading of a byte-identical page."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.reader = reader
        self.name = f"{reader.name}-oriented"
        self.detector = detector or DocTROrientationDetector()
        self.defer_restore = defer_restore
        self.batch_size = batch_size
        self._views: dict[int, _PageView] = {}
        self._assessments: dict[int, dict[str, object]] = {}
        self._lock = threading.Lock()
        self._speculation = ThreadPoolExecutor(max_workers=batch_size)
        self.page_cache = page_cache

    def read_batch(
        self, image_paths: list[Path], page_numbers: list[int]
    ) -> list[list[TextRegion] | ReaderError]:
        """Overlap page preparation and I/O with an explicitly thread-safe reader."""
        return list(self.read_iter(image_paths, page_numbers))

    def read_iter(
        self, image_paths: list[Path], page_numbers: list[int]
    ) -> Iterator[list[TextRegion] | ReaderError]:
        """Page results in page order, each as soon as it and those before it are read.

        Pages are still read `batch_size` at a time; a long PDF can show its first page
        while the rest are being read.
        """
        if len(image_paths) != len(page_numbers):
            raise ValueError("image_paths and page_numbers must have equal lengths")
        if not image_paths:
            return
        with ThreadPoolExecutor(
            max_workers=min(self.batch_size, len(image_paths))
        ) as pool:
            futures = [
                pool.submit(self.read, path, number)
                for path, number in zip(image_paths, page_numbers, strict=True)
            ]
            for future in futures:
                try:
                    yield future.result()
                except ReaderError as error:
                    yield error

    def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
        if self.page_cache is None:
            return self._read(image_path, page_number)
        # Region ids carry the page number, and a text layer changes the reading.
        digest = hashlib.sha256(image_path.read_bytes())
        if sidecar(image_path).is_file():
            digest.update(sidecar(image_path).read_bytes())
        key = f"{self.name}:{digest.hexdigest()}:{page_number}"

        def reading():
            regions = self._read(image_path, page_number)
            with self._lock:
                return regions, self._views[page_number], self._assessments[page_number]

        regions, view, assessment = self.page_cache.get_or_compute(key, reading)
        with self._lock:
            self._views[page_number] = view
            self._assessments[page_number] = assessment
        return regions

    def _read(self, image_path: Path, page_number: int) -> list[TextRegion]:
        born_digital = sidecar(image_path).is_file()
        # Scans are nearly always upright, so the reader's word detection starts beside
        # orientation on the source image and is kept only if no rotation follows.
        # Same pixels, same model: the words are identical to detecting them afterwards.
        word_detector = getattr(self.reader, "word_detector", None)
        words = (
            self._speculation.submit(word_detector, image_path)
            if word_detector and not born_digital
            else None
        )
        try:
            with Image.open(image_path) as source:
                page = ImageOps.exif_transpose(source).convert("RGB")
        except (OSError, UnidentifiedImageError, ValueError) as error:
            self._fail(page_number, "orientation_image_failed", str(error))
            raise ReaderError("orientation_image_failed", str(error)) from error
        try:
            # A born-digital page renders upright; its text layer fixes the orientation.
            orientation = TEXT_LAYER_UPRIGHT if born_digital else self.detector(page)
        except ReaderError as error:
            self._fail(page_number, error.code, str(error))
            raise
        view = _page_view(page.size, orientation)
        assessment = _assessment(page_number, view, orientation is TEXT_LAYER_UPRIGHT)
        with self._lock:
            self._views[page_number] = view
            self._assessments[page_number] = assessment
        with self._view_path(image_path, page, view) as path:
            regions = (
                self.reader.read(path, page_number, words=words.result())
                if words is not None and view.orientation.rotation == 0
                else self.reader.read(path, page_number)
            )
        annotated = [_annotate(region, assessment) for region in regions]
        return (
            annotated
            if self.defer_restore
            else self._restore(annotated, view, page_number)
        )

    @contextmanager
    def stage_view(self, image_path: Path, page_number: int) -> Iterator[Path]:
        """Expose the upright view while downstream stages run."""
        with self._lock:
            view = self._views.get(page_number)
        if not self.defer_restore or view is None:
            yield image_path
            return
        try:
            with Image.open(image_path) as source:
                page = ImageOps.exif_transpose(source).convert("RGB")
        except (OSError, UnidentifiedImageError, ValueError) as error:
            raise ReaderError("orientation_stage_image_failed", str(error)) from error
        with self._view_path(image_path, page, view) as path:
            yield path

    def restore_regions(
        self, regions: list[TextRegion], page_number: int
    ) -> list[TextRegion]:
        with self._lock:
            view = self._views.get(page_number)
        if not self.defer_restore or view is None:
            return regions
        return self._restore(regions, view, page_number)

    def page_needs_review(self, page_number: int) -> bool:
        with self._lock:
            assessment = self._assessments.get(page_number, {})
        return assessment.get("status") in {"review_recommended", "failed"}

    def coverage_assessment(self, page_count: int) -> dict[str, object]:
        with self._lock:
            pages = [
                dict(
                    self._assessments.get(
                        number, {"page_number": number, "status": "not_run"}
                    )
                )
                for number in range(1, page_count + 1)
            ]
        reviewed = sum(page["status"] == "review_recommended" for page in pages)
        failed = sum(page["status"] == "failed" for page in pages)
        if reviewed or failed:
            return {
                "status": "review_recommended",
                "message": (
                    f"Orientation review is recommended on {reviewed} page(s); "
                    f"{failed} page(s) failed orientation."
                ),
                "pages": pages,
            }
        if any(page["status"] == "orientation_confirmed" for page in pages):
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
    def _view_path(
        self, image_path: Path, page: Image.Image, view: _PageView
    ) -> Iterator[Path]:
        if view.orientation.rotation == 0:
            yield image_path
            return
        with tempfile.TemporaryDirectory(prefix="ocr-orientation-") as directory:
            path = Path(directory) / "page-upright.png"
            _render_view(page, view).save(path, format="PNG")
            yield path

    def _restore(
        self, regions: list[TextRegion], view: _PageView, page_number: int
    ) -> list[TextRegion]:
        if view.orientation.rotation == 0:
            return regions
        restored, outside = [], []
        for region in regions:
            try:
                restored.append(
                    replace(
                        region,
                        bounding_box=view.box_to_source(region.bounding_box),
                        structure=_restore_boxes(region.structure, view),
                        alternatives=list(region.alternatives),
                    )
                )
            except ReaderError:
                outside.append(region.id)
        if outside:
            # Only rotation padding under these boxes: no source pixels to report.
            with self._lock:
                assessment = self._assessments[page_number]
                assessment["status"] = "review_recommended"
                assessment["regions_outside_source"] = outside
        return restored

    def _fail(self, page_number: int, code: str, message: str) -> None:
        with self._lock:
            self._assessments[page_number] = {
                "page_number": page_number,
                "status": "failed",
                "failure": {"code": code, "message": message},
            }


def _assessment(
    page_number: int, view: _PageView, text_layer: bool = False
) -> dict[str, object]:
    orientation = view.orientation
    return {
        "page_number": page_number,
        "status": "orientation_confirmed"
        if orientation.confidence >= REVIEW_CONFIDENCE
        else "review_recommended",
        # A born-digital page was turned upright from its own text layer.
        **(
            {"method": "pdf_text_layer"}
            if text_layer
            else {"method": "doctr_page_and_tile_vote", "model": dict(DOCTR_MODEL)}
        ),
        "angle": orientation.angle,
        "skew": orientation.skew,
        "confidence": orientation.confidence,
        "probabilities": list(orientation.probabilities),
        "original_size": list(view.source_size),
        "rotated_size": list(view.view_size),
    }


def _annotate(region: TextRegion, assessment: dict[str, object]) -> TextRegion:
    provenance = dict(region.text_provenance or {})
    provenance["orientation"] = {
        key: assessment[key]
        for key in (
            "method",
            "angle",
            "skew",
            "confidence",
            "original_size",
            "rotated_size",
        )
    }
    return replace(
        region, text_provenance=provenance, alternatives=list(region.alternatives)
    )


def _restore_boxes(value: Any, view: _PageView) -> Any:
    if isinstance(value, list):
        return [_restore_boxes(item, view) for item in value]
    if not isinstance(value, dict):
        return value
    if set(value) == {"left", "top", "right", "bottom"} and all(
        isinstance(value[key], int) and not isinstance(value[key], bool)
        for key in value
    ):
        box = view.box_to_source(BoundingBox(**value))
        return {
            "left": box.left,
            "top": box.top,
            "right": box.right,
            "bottom": box.bottom,
        }
    return {key: _restore_boxes(item, view) for key, item in value.items()}
