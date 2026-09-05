"""Local PILOT OCR reader backed by the pinned official release."""

from __future__ import annotations

import importlib.util
import re
import sys
import threading
from pathlib import Path
from types import ModuleType
from typing import Any

from PIL import Image

from .contracts import BoundingBox, TextRegion
from .providers import ReaderError

PILOT_MODEL_ID = "pilot_generic"
PILOT_SOURCE_REVISION = "3b616308bd6875ae41142a7a7405f353652b8ab8"
PILOT_REPOSITORY = "https://github.com/hamdilaziz/PILOT"
PILOT_WEIGHTS_DOI = "10.5281/zenodo.19681701"
PILOT_CHECKPOINT = "pilot_generic.pt"
PILOT_SOURCE_LICENSE = "Apache-2.0"
PILOT_WEIGHTS_LICENSE = "CC-BY-4.0"
PILOT_ORIGIN = "France"
PILOT_PROMPT = "<ocr_with_boxes>"
PILOT_COORD_BIN_SIZE = 10
PILOT_MAX_LENGTH = 2048

_LINE_PATTERN = re.compile(
    r"\s*<x_(\d+)><y_(\d+)>\s*(.*?)\s*<x_(\d+)><y_(\d+)>\s*",
    re.DOTALL,
)
_IMPORT_LOCK = threading.Lock()


class PilotOCRReader:
    """Run the official PILOT generic checkpoint from a local checkout."""

    name = "pilot-ocr"

    def __init__(
        self,
        pilot_root: Path,
        *,
        device: str | None = None,
        runtime: ModuleType | object | None = None,
    ) -> None:
        self.pilot_root = Path(pilot_root).resolve()
        self.config_path = self.pilot_root / "configs" / "pilot_generic.json"
        self.checkpoint_path = self.pilot_root / "checkpoints" / PILOT_CHECKPOINT
        self.tokenizer_path = self.pilot_root / "checkpoints" / "tokenizer"
        self.device_name = device
        self._runtime = runtime
        self._model: object | None = None
        self._tokenizer: object | None = None
        self._model_config: dict[str, Any] | None = None
        self._device: object | None = None
        self._lock = threading.Lock()
        self._validate_local_release()

    @property
    def provenance(self) -> dict[str, Any]:
        return _model_provenance()

    def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
        try:
            with Image.open(image_path) as source:
                image = source.convert("RGB")
                width, height = image.size
        except (OSError, ValueError) as error:
            raise ReaderError("pilot_image_failed", str(error)) from error

        raw_prediction = self._predict(image)
        return _parse_prediction(raw_prediction, page_number, width, height)

    def _validate_local_release(self) -> None:
        required_files = (
            self.pilot_root / "run_pilot.py",
            self.pilot_root / "pilot" / "__init__.py",
            self.config_path,
            self.checkpoint_path,
        )
        missing = [path for path in required_files if not path.is_file()]
        if not self.tokenizer_path.is_dir():
            missing.append(self.tokenizer_path)
        if missing:
            names = ", ".join(
                str(path.relative_to(self.pilot_root)) for path in missing
            )
            raise ValueError(f"PILOT local release is incomplete: {names}")

    def _predict(self, image: Image.Image) -> str:
        with self._lock:
            runtime, model, tokenizer, model_config, device = (
                self._initialize_components()
            )
            preprocessing = model_config["preprocessing"]
            try:
                batch = runtime.prepare_batch_for_inference(
                    image=image,
                    tokenizer=tokenizer,
                    prompt=PILOT_PROMPT,
                    mean=preprocessing["mean"],
                    std=preprocessing["std"],
                )
                batch["imgs"] = batch["imgs"].to(device)
                batch["token_prompt"] = batch["token_prompt"].to(device)
                result = model.predict(
                    batch,
                    use_amp=device.type == "cuda",
                    num_beams=1,
                    max_length=PILOT_MAX_LENGTH,
                )
            except Exception as error:
                raise ReaderError("pilot_predict_failed", str(error)) from error

        predictions = result.get("str_pred") if isinstance(result, dict) else None
        if (
            not isinstance(predictions, list)
            or len(predictions) != 1
            or not isinstance(predictions[0], str)
            or not predictions[0].strip()
        ):
            raise ReaderError(
                "pilot_output_failed",
                "PILOT returned an invalid prediction payload",
            )
        return predictions[0]

    def _initialize_components(
        self,
    ) -> tuple[object, object, object, dict[str, Any], object]:
        if (
            self._model is not None
            and self._tokenizer is not None
            and self._model_config is not None
            and self._device is not None
        ):
            return (
                self._runtime,
                self._model,
                self._tokenizer,
                self._model_config,
                self._device,
            )

        runtime = self._runtime or _load_official_runtime(self.pilot_root)
        try:
            device_name = self.device_name
            if device_name is None:
                device_name = "cuda" if runtime.torch.cuda.is_available() else "cpu"
            device = runtime.torch.device(device_name)
            model, tokenizer, model_config = runtime.load_pilot_model(
                config_path=self.config_path,
                device=device,
                checkpoint_path=self.checkpoint_path,
                tokenizer_path=self.tokenizer_path,
            )
            _validate_generic_config(model_config)
        except ReaderError:
            raise
        except Exception as error:
            raise ReaderError("pilot_init_failed", str(error)) from error

        self._runtime = runtime
        self._model = model
        self._tokenizer = tokenizer
        self._model_config = model_config
        self._device = device
        return runtime, model, tokenizer, model_config, device


def _load_official_runtime(pilot_root: Path) -> ModuleType:
    script_path = pilot_root / "run_pilot.py"
    spec = importlib.util.spec_from_file_location(
        "_ocr_pipeline_official_pilot_runtime",
        script_path,
    )
    if spec is None or spec.loader is None:
        raise ReaderError(
            "pilot_import_failed",
            f"Could not load the official PILOT runtime: {script_path}",
        )

    module = importlib.util.module_from_spec(spec)
    with _IMPORT_LOCK:
        sys.path.insert(0, str(pilot_root))
        try:
            spec.loader.exec_module(module)
        except Exception as error:
            raise ReaderError("pilot_import_failed", str(error)) from error
        finally:
            sys.path.pop(0)
    return module


def _validate_generic_config(config: object) -> None:
    if not isinstance(config, dict):
        raise ReaderError("pilot_init_failed", "PILOT config is not an object")
    preprocessing = config.get("preprocessing")
    decoder = config.get("decoder")
    supported_tasks = config.get("supported_tasks")
    if (
        config.get("name") != PILOT_MODEL_ID
        or not isinstance(preprocessing, dict)
        or preprocessing.get("coord_bin_size") != PILOT_COORD_BIN_SIZE
        or not isinstance(decoder, dict)
        or decoder.get("max_length") != PILOT_MAX_LENGTH
        or not isinstance(supported_tasks, list)
        or PILOT_PROMPT.removeprefix("<").removesuffix(">") not in supported_tasks
    ):
        raise ReaderError(
            "pilot_init_failed",
            "PILOT config does not match the released generic checkpoint",
        )


def _parse_prediction(
    prediction: str,
    page_number: int,
    width: int,
    height: int,
) -> list[TextRegion]:
    segments = prediction.split("<sep/>")
    if not segments or any(not segment.strip() for segment in segments):
        raise ReaderError("pilot_output_failed", "PILOT output has an empty line")

    regions = []
    for reading_order, segment in enumerate(segments, start=1):
        match = _LINE_PATTERN.fullmatch(segment)
        if match is None:
            raise ReaderError(
                "pilot_output_failed",
                "PILOT output is not a paired text-and-box sequence",
            )
        x1, y1, text, x2, y2 = match.groups()
        box = BoundingBox(
            int(x1) * PILOT_COORD_BIN_SIZE,
            int(y1) * PILOT_COORD_BIN_SIZE,
            int(x2) * PILOT_COORD_BIN_SIZE,
            int(y2) * PILOT_COORD_BIN_SIZE,
        )
        text = text.strip()
        if (
            not text
            or "<x_" in text
            or "<y_" in text
            or box.right <= box.left
            or box.bottom <= box.top
            or box.right > width
            or box.bottom > height
        ):
            raise ReaderError(
                "pilot_output_failed",
                "PILOT output contains invalid text or geometry",
            )
        regions.append(
            TextRegion(
                id=f"p{page_number}-line-{reading_order}",
                kind="line",
                text=text,
                confidence=None,
                bounding_box=box,
                reading_order=reading_order,
                provider=PilotOCRReader.name,
                text_provenance={
                    "method": "pilot_ocr_with_boxes",
                    "model": _model_provenance(),
                    "prompt": PILOT_PROMPT,
                    "generation": {
                        "max_length": PILOT_MAX_LENGTH,
                        "num_beams": 1,
                        "do_sample": False,
                    },
                    "geometry": {"coord_bin_size": PILOT_COORD_BIN_SIZE},
                },
            )
        )
    return regions


def _model_provenance() -> dict[str, Any]:
    return {
        "id": PILOT_MODEL_ID,
        "repository": PILOT_REPOSITORY,
        "source_revision": PILOT_SOURCE_REVISION,
        "source_license": PILOT_SOURCE_LICENSE,
        "weights_doi": PILOT_WEIGHTS_DOI,
        "weights_license": PILOT_WEIGHTS_LICENSE,
        "checkpoint": PILOT_CHECKPOINT,
        "origin": PILOT_ORIGIN,
        "local_files_only": True,
    }
