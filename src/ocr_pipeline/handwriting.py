"""Conservative crop-level handwriting specialization."""

from __future__ import annotations

import copy
import math
import threading
import unicodedata
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from PIL import Image, UnidentifiedImageError

from .contracts import BoundingBox, TextAlternative, TextRegion
from .evidence_layout import refresh_layout_owners
from .providers import ReaderError

ABSTENTION_OUTPUTS = frozenset({"<no_handwriting>", "<unreadable>"})
# Large beats base on both held-out line splits: IAM 3.40% vs 4.72% CER over 2915
# lines, RIMES 22.97% vs 27.35% over 778, for +24ms per crop and +900 MiB.
TROCR_MODEL_ID = "microsoft/trocr-large-handwritten"
TROCR_MODEL_REVISION = "e68501f437cd2587ae5d68ee457964cac824ddee"
TROCR_MODEL_ORIGIN = "Microsoft"
TROCR_MODEL_LICENSE = "MIT"
TROCR_REQUIRED_FILES = ("config.json", "preprocessor_config.json")
TROCR_WEIGHT_FILES = (
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
)
ELIGIBLE_KINDS = frozenset({"handwriting", "text", "word"})
EXCLUDED_ROLES = frozenset(
    {
        "control",
        "coverage_risk",
        "footer",
        "header",
        "heading",
        "table",
        "table_candidate",
        "tiny_text_candidate",
        "title",
    }
)


class HandwritingCropReader(Protocol):
    name: str
    max_batch_items: int

    @property
    def provenance(self) -> dict[str, Any]: ...

    def transcribe_batch(self, images: Sequence[Image.Image]) -> list[str]: ...


class TrOCRHandwritingReader:
    """Lazy, local-only TrOCR reader for bounded single-line crops."""

    name = "trocr-handwriting"

    def __init__(
        self,
        *,
        model_name_or_path: str | Path = TROCR_MODEL_ID,
        model_revision: str = TROCR_MODEL_REVISION,
        device: str = "cuda:0",
        max_new_tokens: int = 128,
        max_batch_items: int = 16,
        batch_size: int = 4,
        binarize: bool = True,
        processor: object | None = None,
        model: object | None = None,
        torch_module: object | None = None,
    ) -> None:
        model_source = Path(model_name_or_path)
        if max_new_tokens <= 0:
            raise ValueError("TrOCR max_new_tokens must be positive")
        if max_batch_items <= 0:
            raise ValueError("TrOCR max_batch_items must be positive")
        if batch_size <= 0 or batch_size > max_batch_items:
            raise ValueError("TrOCR batch_size must fit within max_batch_items")
        if str(model_name_or_path) != TROCR_MODEL_ID and not model_source.is_dir():
            raise FileNotFoundError(f"Local TrOCR model was not found: {model_source}")
        injected = (processor, model, torch_module)
        if any(item is not None for item in injected) and not all(
            item is not None for item in injected
        ):
            raise ValueError("Injected TrOCR components must be provided together")
        self.model_name_or_path = str(model_name_or_path)
        self.model_revision = model_revision
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.max_batch_items = max_batch_items
        self.batch_size = batch_size
        self.binarize = binarize
        self._processor = processor
        self._model = model
        self._torch = torch_module
        self._lock = threading.Lock()

    @property
    def provenance(self) -> dict[str, Any]:
        official_model = self.model_name_or_path == TROCR_MODEL_ID
        identity_verified = (
            official_model and self.model_revision == TROCR_MODEL_REVISION
        )
        return {
            "id": TROCR_MODEL_ID
            if official_model
            else Path(self.model_name_or_path).name,
            "source": self.model_name_or_path if official_model else "local_directory",
            "revision": self.model_revision if official_model else "unverified",
            "origin": TROCR_MODEL_ORIGIN if official_model else "unverified",
            "license": TROCR_MODEL_LICENSE if official_model else "unverified",
            "identity_verified": identity_verified,
            "local_files_only": True,
            "scope": "single_text_line",
        }

    def transcribe_batch(self, images: Sequence[Image.Image]) -> list[str]:
        if not images:
            return []
        if len(images) > self.max_batch_items:
            raise ReaderError(
                "trocr_handwriting_batch_too_large",
                "TrOCR handwriting crop batch exceeds its configured limit",
            )
        with self._lock:
            processor, model, torch_module = self._initialize_components()
            prepared = (
                [_binarize(image) for image in images] if self.binarize else images
            )
            texts: list[str] = []
            for start in range(0, len(prepared), self.batch_size):
                batch = prepared[start : start + self.batch_size]
                try:
                    inputs = processor(images=list(batch), return_tensors="pt")
                    pixel_values = getattr(inputs, "pixel_values", None)
                    if pixel_values is None:
                        raise ValueError("processor returned no pixel values")
                    pixel_values = pixel_values.to(self.device)
                    with torch_module.inference_mode():
                        generated = model.generate(
                            pixel_values=pixel_values,
                            max_new_tokens=self.max_new_tokens,
                            do_sample=False,
                            num_beams=1,
                        )
                    decoded = processor.batch_decode(
                        generated,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                except Exception as error:
                    raise ReaderError(
                        "trocr_handwriting_inference_failed",
                        "TrOCR handwriting inference failed",
                    ) from error
                if len(decoded) != len(batch) or any(
                    not isinstance(text, str) for text in decoded
                ):
                    raise ReaderError(
                        "trocr_handwriting_output_failed",
                        "TrOCR handwriting output did not match the crop batch",
                    )
                texts.extend(decoded)
            return texts

    def check_health(self) -> None:
        """Load the requested local runtime before advertising it as ready."""
        try:
            import torch
            from transformers import (
                RobertaTokenizer,
                TrOCRProcessor,
                ViTImageProcessor,
                VisionEncoderDecoderModel,
            )

            _ = (
                RobertaTokenizer,
                TrOCRProcessor,
                ViTImageProcessor,
                VisionEncoderDecoderModel,
            )
        except Exception as error:
            raise ReaderError(
                "trocr_handwriting_unavailable",
                "TrOCR handwriting dependencies are unavailable",
            ) from error
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise ReaderError(
                "trocr_handwriting_unavailable",
                "TrOCR handwriting requires an available CUDA device",
            )

        files = TROCR_REQUIRED_FILES + TROCR_WEIGHT_FILES
        available = {
            filename: _cached_trocr_file(
                self.model_name_or_path,
                self.model_revision,
                filename,
            )
            for filename in files
        }
        if not all(available[name] for name in TROCR_REQUIRED_FILES) or not any(
            available[name] for name in TROCR_WEIGHT_FILES
        ):
            raise ReaderError(
                "trocr_handwriting_unavailable",
                "TrOCR handwriting model files are not available locally",
            )
        self._initialize_components()

    def _initialize_components(self) -> tuple[object, object, object]:
        if self._processor is not None:
            return self._processor, self._model, self._torch
        try:
            import torch
            from transformers import (
                RobertaTokenizer,
                TrOCRProcessor,
                ViTImageProcessor,
                VisionEncoderDecoderModel,
            )

            options: dict[str, object] = {"local_files_only": True}
            if self.model_name_or_path == TROCR_MODEL_ID:
                options["revision"] = self.model_revision
            image_processor = ViTImageProcessor.from_pretrained(
                self.model_name_or_path,
                **options,
            )
            tokenizer = RobertaTokenizer.from_pretrained(
                self.model_name_or_path,
                **options,
            )
            processor = TrOCRProcessor(
                image_processor=image_processor,
                tokenizer=tokenizer,
            )
            model = VisionEncoderDecoderModel.from_pretrained(
                self.model_name_or_path,
                **options,
            ).to(self.device)
            model.eval()
        except Exception as error:
            raise ReaderError(
                "trocr_handwriting_init_failed",
                "Cached TrOCR handwriting model could not be initialized",
            ) from error
        self._processor = processor
        self._model = model
        self._torch = torch
        return processor, model, torch


def _binarize(image: Image.Image) -> Image.Image:
    """Adaptive threshold to flatten page shading before the square resize.

    On by default. It cuts photographed-notebook prose CER from 17.3% to 13.1% on
    three reviewed lines, and is within noise on clean scans: IAM (2915 lines) 3.35%
    vs 3.40% CER, RIMES (778) 23.29% vs 22.97%. The one real cost is IAM exact-match
    lines falling from 1553 to 1539.
    """
    import cv2
    import numpy as np

    gray = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    mask = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        10,
    )
    return Image.fromarray(cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB))


def is_verified_trocr_model_provenance(value: Any) -> bool:
    """Return whether provenance identifies the pinned official TrOCR model."""
    return isinstance(value, dict) and all(
        (
            value.get("id") == TROCR_MODEL_ID,
            value.get("source") == TROCR_MODEL_ID,
            value.get("revision") == TROCR_MODEL_REVISION,
            value.get("origin") == TROCR_MODEL_ORIGIN,
            value.get("license") == TROCR_MODEL_LICENSE,
            value.get("identity_verified") is True,
            value.get("local_files_only") is True,
        )
    )


def _cached_trocr_file(
    model_name_or_path: str,
    model_revision: str,
    filename: str,
) -> bool:
    model_path = Path(model_name_or_path)
    if model_path.is_dir():
        return (model_path / filename).is_file()
    try:
        from transformers.utils import cached_file

        return (
            cached_file(
                model_name_or_path,
                filename,
                revision=model_revision,
                local_files_only=True,
            )
            is not None
        )
    except (OSError, RuntimeError, ValueError):
        return False


class HandwritingStage:
    """Reread explicitly marked, bounded handwriting regions using two crop views."""

    name = "handwriting"

    def __init__(
        self,
        reader: HandwritingCropReader,
        *,
        text_provider: str | None = None,
        confidence_threshold: float = 0.75,
        max_regions: int = 8,
        max_incumbent_characters: int = 64,
        max_candidate_characters: int = 96,
        context_padding: int = 12,
    ) -> None:
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be from 0 to 1")
        if max_regions <= 0:
            raise ValueError("max_regions must be positive")
        if max_incumbent_characters <= 0 or max_candidate_characters <= 0:
            raise ValueError("handwriting text limits must be positive")
        if context_padding <= 0:
            raise ValueError("context_padding must be positive")
        if reader.max_batch_items < max_regions * 2:
            raise ValueError("handwriting reader batch limit is too small")
        self.reader = reader
        self.text_provider = text_provider
        self.confidence_threshold = confidence_threshold
        self.max_regions = max_regions
        self.max_incumbent_characters = max_incumbent_characters
        self.max_candidate_characters = max_candidate_characters
        self.context_padding = context_padding

    def apply(
        self,
        image_path: Path,
        page_number: int,
        regions: list[TextRegion],
    ) -> list[TextRegion]:
        return self._apply(
            image_path,
            page_number,
            regions,
            raise_on_failure=False,
        )

    def _apply(
        self,
        image_path: Path,
        page_number: int,
        regions: list[TextRegion],
        *,
        raise_on_failure: bool,
    ) -> list[TextRegion]:
        selected = self._selected_indices(regions)
        if not selected:
            return regions

        crops: list[Image.Image] = []
        crop_boxes: list[tuple[BoundingBox, BoundingBox]] = []
        crop_indexes: list[tuple[int, int]] = []
        indexes_by_box: dict[tuple[int, int, int, int], int] = {}
        try:
            with Image.open(image_path) as opened:
                page = opened.convert("RGB")
            try:
                for index in selected:
                    tight, context = _crop_boxes(
                        regions[index].bounding_box,
                        page.size,
                        self.context_padding,
                    )
                    crop_boxes.append((tight, context))
                    indexes = []
                    for box in (tight, context):
                        key = _box_tuple(box)
                        if key not in indexes_by_box:
                            indexes_by_box[key] = len(crops)
                            crops.append(page.crop(key))
                        indexes.append(indexes_by_box[key])
                    crop_indexes.append((indexes[0], indexes[1]))
            finally:
                page.close()
        except (OSError, UnidentifiedImageError, ValueError) as error:
            for crop in crops:
                crop.close()
            failure = ReaderError(
                "handwriting_crop_failed",
                "Handwriting crops could not be prepared",
            )
            if raise_on_failure:
                raise failure from error
            self._record_failure(regions, selected, page_number, failure.code)
            return regions

        try:
            try:
                candidates = self.reader.transcribe_batch(crops)
                if len(candidates) != len(crops) or any(
                    not isinstance(candidate, str) for candidate in candidates
                ):
                    raise ReaderError(
                        "invalid_handwriting_output",
                        "Handwriting reader returned an invalid crop batch",
                    )
            except ReaderError as error:
                if raise_on_failure:
                    raise
                self._record_failure(regions, selected, page_number, error.code)
                return regions
        finally:
            for crop in crops:
                crop.close()

        model_provenance = self.reader.provenance
        owner_ids = {
            owner_id
            for index in selected
            if isinstance(
                owner_id := (regions[index].structure or {}).get("layout_owner_id"),
                str,
            )
        }
        for position, index in enumerate(selected):
            tight_index, context_index = crop_indexes[position]
            tight_text = candidates[tight_index].strip()
            context_text = candidates[context_index].strip()
            tight_box, context_box = crop_boxes[position]
            provenance = _crop_provenance(
                page_number,
                tight_box,
                context_box,
                model_provenance,
            )
            field_ownership = (regions[index].structure or {}).get("field_ownership")
            if isinstance(field_ownership, dict):
                provenance["field_ownership"] = copy.deepcopy(field_ownership)
            self._apply_candidate(
                regions[index],
                tight_text,
                context_text,
                provenance,
            )
        refresh_layout_owners(regions, owner_ids)
        return regions

    def _record_failure(
        self,
        regions: list[TextRegion],
        selected: list[int],
        page_number: int,
        reason: str,
    ) -> None:
        provenance = {
            "method": "bounded_handwriting_reread",
            "page_number": page_number,
            "model": self.reader.provenance,
        }
        for index in selected:
            _record_attempt(regions[index], "failed", reason, provenance)
            _mark_review(regions[index], reason, provenance)

    def review_region(
        self,
        image_path: Path,
        page_number: int,
        region: TextRegion,
    ) -> TextRegion:
        """Reread one user-selected region without enabling automatic routing."""
        candidate = copy.deepcopy(region)
        structure = dict(candidate.structure or {})
        structure["handwriting_candidate"] = True
        structure["handwriting_candidate_source"] = "manual"
        candidate.structure = structure
        if not self._is_eligible(candidate):
            raise ReaderError(
                "handwriting_region_ineligible",
                "The selected region is not a bounded text crop",
            )
        return self._apply(
            image_path,
            page_number,
            [candidate],
            raise_on_failure=True,
        )[0]

    def _selected_indices(self, regions: list[TextRegion]) -> list[int]:
        candidates = [
            index
            for index, region in enumerate(regions)
            if self._is_eligible(region) or _is_anchored_residual_proposal(region)
        ]
        candidates.sort(
            key=lambda index: (
                regions[index].confidence
                if regions[index].confidence is not None
                else -1.0,
                regions[index].reading_order,
                index,
            )
        )
        return candidates[: self.max_regions]

    def _is_eligible(self, region: TextRegion) -> bool:
        text = region.text.strip()
        structure = region.structure or {}
        manual = structure.get("handwriting_candidate_source") == "manual"
        semantic_labels = {
            str(structure.get(name, "")).strip().casefold().replace("-", "_")
            for name in (
                "role",
                "block_type",
                "semantic_class",
                "layout_owner_type",
            )
        }
        handwriting_signal = (
            region.kind == "handwriting"
            or structure.get("handwriting_candidate") is True
        )
        return (
            region.resolution == "resolved"
            and region.kind in ELIGIBLE_KINDS
            and semantic_labels.isdisjoint({"formula", "equation", "math"})
            and (manual or structure.get("role") not in EXCLUDED_ROLES)
            and handwriting_signal
            and bool(text)
            and "\n" not in text
            and len(text) <= self.max_incumbent_characters
            and region.confidence is not None
            and (manual or region.confidence < self.confidence_threshold)
            and (self.text_provider is None or region.provider == self.text_provider)
        )

    def _apply_candidate(
        self,
        region: TextRegion,
        tight_text: str,
        context_text: str,
        provenance: dict[str, Any],
    ) -> None:
        if tight_text != context_text:
            for view, text in (("tight", tight_text), ("context", context_text)):
                if (
                    text != region.text
                    and _literal_rejection(
                        text,
                        region.text,
                        self.max_candidate_characters,
                    )
                    is None
                ):
                    region.alternatives.append(
                        TextAlternative(
                            text=text,
                            confidence=None,
                            provider=f"{self.reader.name}:{view}",
                            text_provenance={**provenance, "view": view},
                        )
                    )
            _record_attempt(region, "unresolved", "crop_disagreement", provenance)
            _mark_review(region, "crop_disagreement", provenance)
            return

        rejection = _literal_rejection(
            tight_text,
            region.text,
            self.max_candidate_characters,
        )
        if rejection is not None:
            _record_attempt(region, "unresolved", rejection, provenance)
            _mark_review(region, rejection, provenance)
            return

        if tight_text == region.text:
            current = dict(region.text_provenance or {})
            current["handwriting_specialist"] = {
                **provenance,
                "decision": "corroborated",
            }
            region.text_provenance = current
            _record_attempt(
                region,
                "unchanged_after_reread",
                "specialist_matches_incumbent",
                provenance,
            )
            return

        if _is_anchored_residual_proposal(region) or _is_classifier_candidate(region):
            region.alternatives.append(
                TextAlternative(
                    text=tight_text,
                    confidence=None,
                    provider=self.reader.name,
                    text_provenance={**provenance, "view": "agreed"},
                )
            )
            _record_attempt(
                region,
                "candidate_pending",
                "awaiting_independent_validation",
                provenance,
            )
            _mark_review(region, "specialist_candidate", provenance)
            return

        support = _independent_support(
            region,
            tight_text,
            self.reader.name,
            provenance,
        )
        if support is not None:
            incumbent = TextAlternative(
                text=region.text,
                confidence=region.confidence,
                provider=region.provider,
                text_provenance=copy.deepcopy(region.text_provenance),
                decision_state="superseded",
            )
            history = [incumbent]
            for alternative in region.alternatives:
                historical = copy.deepcopy(alternative)
                if historical.decision_state == "pending":
                    historical.decision_state = (
                        "accepted" if alternative is support else "rejected"
                    )
                history.append(historical)
            region.text = tight_text
            region.confidence = None
            region.provider = self.reader.name
            region.text_provenance = {
                **provenance,
                "decision": "independently_corroborated_replacement",
                "supporting_provider": support.provider,
                "supporting_provenance": copy.deepcopy(support.text_provenance),
            }
            region.alternatives = history
            _record_attempt(
                region,
                "corrected",
                "independently_corroborated_replacement",
                provenance,
            )
            _mark_review(region, "corroborated_replacement", provenance)
            return

        region.alternatives.append(
            TextAlternative(
                text=tight_text,
                confidence=None,
                provider=self.reader.name,
                text_provenance={**provenance, "view": "agreed"},
            )
        )
        _record_attempt(
            region,
            "candidate_pending",
            "awaiting_independent_validation",
            provenance,
        )
        _mark_review(region, "specialist_candidate", provenance)


def _literal_rejection(
    candidate: str,
    incumbent: str,
    max_candidate_characters: int,
) -> str | None:
    if not candidate:
        return "empty_candidate"
    if candidate.casefold() in ABSTENTION_OUTPUTS:
        return "abstention_candidate"
    if len(candidate) > max_candidate_characters:
        return "candidate_too_long"
    incumbent = incumbent.strip()
    if incumbent:
        relative_limit = max(12, len(incumbent) * 3 + 8)
        if len(candidate) > relative_limit:
            return "candidate_expanded_context"
    if any(unicodedata.category(character).startswith("C") for character in candidate):
        return "candidate_control_characters"
    if "<|" in candidate or "|>" in candidate:
        return "candidate_control_tokens"
    return None


def _is_anchored_residual_proposal(region: TextRegion) -> bool:
    structure = region.structure or {}
    return (
        structure.get("handwriting_candidate_source") == "anchored_residual"
        and region.kind == "handwriting"
        and region.text == ""
        and region.confidence is None
        and region.resolution == "unreadable"
    )


def _is_classifier_candidate(region: TextRegion) -> bool:
    structure = region.structure or {}
    return (
        structure.get("handwriting_candidate_source") == "classifier"
        and structure.get("handwriting_candidate") is True
    )


def _independent_support(
    region: TextRegion,
    candidate: str,
    specialist_provider: str,
    specialist_provenance: dict[str, Any],
) -> TextAlternative | None:
    normalized = _normalized_text(candidate)
    specialist_identity = _model_identity(specialist_provenance)
    if specialist_identity is None:
        return None
    for alternative in region.alternatives:
        if alternative.decision_state != "pending":
            continue
        if alternative.provider.startswith(specialist_provider):
            continue
        supporting_identity = _model_identity(alternative.text_provenance)
        if supporting_identity is None or supporting_identity == specialist_identity:
            continue
        if _normalized_text(alternative.text) == normalized:
            return alternative
    return None


def _model_identity(provenance: dict[str, Any] | None) -> tuple[str, str] | None:
    model = (provenance or {}).get("model")
    if not isinstance(model, dict) or model.get("identity_verified") is not True:
        return None
    model_id = model.get("id")
    revision = model.get("revision")
    source = model.get("source")
    if not all(
        isinstance(value, str) and value.strip()
        for value in (model_id, revision, source)
    ):
        return None
    return model_id.strip(), revision.strip()


def _normalized_text(text: str) -> str:
    return " ".join(text.casefold().split())


def _crop_boxes(
    box: BoundingBox,
    page_size: tuple[int, int],
    context_padding: int,
) -> tuple[BoundingBox, BoundingBox]:
    width, height = page_size
    tight = BoundingBox(
        max(0, box.left),
        max(0, box.top),
        min(width, box.right),
        min(height, box.bottom),
    )
    if tight.right <= tight.left or tight.bottom <= tight.top:
        raise ValueError("handwriting region lies outside the page")
    region_height = tight.bottom - tight.top
    padding = max(context_padding, math.ceil(region_height / 2))
    context = BoundingBox(
        max(0, tight.left - padding),
        max(0, tight.top - padding),
        min(width, tight.right + padding),
        min(height, tight.bottom + padding),
    )
    return tight, context


def _crop_provenance(
    page_number: int,
    tight: BoundingBox,
    context: BoundingBox,
    model: dict[str, Any],
) -> dict[str, Any]:
    return {
        "method": "dual_crop_exact_agreement",
        "page_number": page_number,
        "crops": {
            "tight": {"bounding_box": list(_box_tuple(tight))},
            "context": {"bounding_box": list(_box_tuple(context))},
        },
        "model": model,
    }


def _mark_review(
    region: TextRegion,
    reason: str,
    provenance: dict[str, Any],
) -> None:
    structure = dict(region.structure or {})
    structure["handwriting_review"] = {
        "required": True,
        "reason": reason,
        "provenance": provenance,
    }
    region.structure = structure


def _record_attempt(
    region: TextRegion,
    outcome: str,
    reason: str,
    provenance: dict[str, Any],
) -> None:
    structure = dict(region.structure or {})
    structure["handwriting_attempt"] = {
        "outcome": outcome,
        "reason": reason,
        "provenance": provenance,
    }
    region.structure = structure


def _box_tuple(box: BoundingBox) -> tuple[int, int, int, int]:
    return box.left, box.top, box.right, box.bottom
