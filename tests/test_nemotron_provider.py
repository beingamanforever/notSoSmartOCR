from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

from PIL import Image

from ocr_pipeline.contracts import BoundingBox
from ocr_pipeline.pipeline import process_document
from ocr_pipeline.providers import NemotronOCRV2Reader


class FakePipeline:
    def __call__(self, image_path: str, *, merge_level: str) -> list[dict[str, object]]:
        assert Path(image_path).is_file()
        assert merge_level == "paragraph"
        return [
            {
                "text": " First block ",
                "confidence": 0.97,
                "left": 0.1,
                "upper": 0.4,
                "right": 0.8,
                "lower": 0.2,
            },
            {
                "text": "Second block",
                "confidence": 0.92,
                "left": 0.12,
                "upper": 0.75,
                "right": 0.9,
                "lower": 0.5,
            },
        ]


def test_nemotron_reader_preserves_regions_confidence_and_order(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (120, 100), "white").save(image_path)

    result = process_document(
        image_path,
        NemotronOCRV2Reader(batch_size=4, pipeline=FakePipeline()),
    )

    assert result.status == "success"
    assert result.pages[0].reader == "nemotron-ocr-v2"
    assert result.pages[0].text.value == "First block Second block"
    first, second = result.pages[0].regions
    assert first.id == "p1-block-1"
    assert first.text == "First block"
    assert first.confidence == 0.97
    assert first.bounding_box.left == 12
    assert first.bounding_box.top == 20
    assert first.bounding_box.right == 96
    assert first.bounding_box.bottom == 40
    assert first.text_provenance == {"merge_level": "paragraph"}
    assert second.reading_order == 2


def test_nemotron_readers_can_share_one_execution_lock(tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (120, 100), "white").save(image_path)

    class RecordingLock:
        def __init__(self) -> None:
            self.entries = 0
            self.active = False

        def __enter__(self) -> None:
            assert self.active is False
            self.active = True
            self.entries += 1

        def __exit__(self, *args: object) -> None:
            self.active = False

    lock = RecordingLock()

    class LockedPipeline(FakePipeline):
        def __call__(
            self,
            image_path: str,
            *,
            merge_level: str,
        ) -> list[dict[str, object]]:
            assert lock.active is True
            return super().__call__(image_path, merge_level=merge_level)

    pipeline = LockedPipeline()
    first = NemotronOCRV2Reader(pipeline=pipeline, execution_lock=lock)
    second = NemotronOCRV2Reader(pipeline=pipeline, execution_lock=lock)

    first.read(image_path, 1)
    second.read(image_path, 2)

    assert lock.entries == 2
    assert first._lock is second._lock


def test_nemotron_reader_uses_native_batch_and_preserves_page_geometry(
    tmp_path: Path,
) -> None:
    source = tmp_path / "two-pages.tiff"
    first = Image.new("RGB", (20, 10), "white")
    second = Image.new("RGB", (40, 30), "white")
    first.save(source, format="TIFF", save_all=True, append_images=[second])

    class BatchPipeline:
        def __init__(self) -> None:
            self.inputs: list[list[str]] = []

        def __call__(
            self, image_paths: list[str], *, merge_level: str
        ) -> list[list[dict[str, object]]]:
            self.inputs.append(image_paths)
            return [
                [
                    {
                        "text": f"page {index}",
                        "confidence": 0.9,
                        "left": 0.25,
                        "lower": 0.2,
                        "right": 0.75,
                        "upper": 0.8,
                    }
                ]
                for index, _ in enumerate(image_paths, start=1)
            ]

    pipeline = BatchPipeline()
    result = process_document(
        source,
        NemotronOCRV2Reader(batch_size=2, pipeline=pipeline),
    )

    assert len(pipeline.inputs) == 1
    assert all(isinstance(batch_input, str) for batch_input in pipeline.inputs[0])
    assert [page.text.value for page in result.pages] == ["page 1", "page 2"]
    assert [page.regions[0].id for page in result.pages] == [
        "p1-block-1",
        "p2-block-1",
    ]
    assert [page.regions[0].bounding_box for page in result.pages] == [
        BoundingBox(5, 2, 15, 8),
        BoundingBox(10, 6, 30, 24),
    ]


def test_nemotron_reader_chunks_native_batches(tmp_path: Path) -> None:
    image_paths = []
    for page_number in range(1, 4):
        image_path = tmp_path / f"page-{page_number}.png"
        Image.new("RGB", (10, 10), "white").save(image_path)
        image_paths.append(image_path)

    class ChunkPipeline:
        def __init__(self) -> None:
            self.batch_lengths: list[int] = []

        def __call__(
            self, image_paths: list[str], *, merge_level: str
        ) -> list[list[dict[str, object]]]:
            self.batch_lengths.append(len(image_paths))
            return [[] for _ in image_paths]

    pipeline = ChunkPipeline()
    results = NemotronOCRV2Reader(
        batch_size=2,
        pipeline=pipeline,
    ).read_batch(image_paths, [1, 2, 3])

    assert pipeline.batch_lengths == [2, 1]
    assert results == [[], [], []]


def test_nemotron_default_reads_multi_page_sources_one_page_at_a_time(
    tmp_path: Path,
) -> None:
    source = tmp_path / "two-pages.tiff"
    page = Image.new("RGB", (20, 10), "white")
    page.save(source, format="TIFF", save_all=True, append_images=[page])

    class SinglePipeline:
        def __init__(self) -> None:
            self.inputs: list[str] = []

        def __call__(
            self, image_path: str, *, merge_level: str
        ) -> list[dict[str, object]]:
            self.inputs.append(image_path)
            return [
                {
                    "text": Path(image_path).stem,
                    "confidence": 0.9,
                    "left": 0.0,
                    "lower": 0.0,
                    "right": 1.0,
                    "upper": 1.0,
                }
            ]

    pipeline = SinglePipeline()
    result = process_document(source, NemotronOCRV2Reader(pipeline=pipeline))

    assert len(pipeline.inputs) == 2
    assert [page.text.value for page in result.pages] == ["page-1", "page-2"]


def test_nemotron_batch_reports_invalid_output_for_only_affected_page(
    tmp_path: Path,
) -> None:
    source = tmp_path / "two-pages.tiff"
    page = Image.new("RGB", (20, 10), "white")
    page.save(source, format="TIFF", save_all=True, append_images=[page])

    class PartiallyInvalidPipeline:
        def __call__(
            self, image_paths: list[str], *, merge_level: str
        ) -> list[list[dict[str, object]]]:
            return [
                [
                    {
                        "text": "valid",
                        "confidence": 0.9,
                        "left": 0.0,
                        "lower": 0.0,
                        "right": 1.0,
                        "upper": 1.0,
                    }
                ],
                [{"text": "missing fields"}],
            ]

    result = process_document(
        source,
        NemotronOCRV2Reader(batch_size=2, pipeline=PartiallyInvalidPipeline()),
    )

    assert result.status == "partial"
    assert result.pages[0].failure_ids == []
    assert result.pages[1].failure_ids == ["failure-1"]
    assert result.failures[0].code == "invalid_reader_output"
    assert result.failures[0].page_number == 2


def test_nemotron_batch_failure_is_reported_for_each_page_without_retry(
    tmp_path: Path,
) -> None:
    source = tmp_path / "two-pages.tiff"
    page = Image.new("RGB", (20, 10), "white")
    page.save(source, format="TIFF", save_all=True, append_images=[page])

    class BrokenBatchPipeline:
        def __init__(self) -> None:
            self.inputs: list[object] = []

        def __call__(self, image_paths: object, *, merge_level: str) -> object:
            self.inputs.append(image_paths)
            raise RuntimeError("native batch failed")

    pipeline = BrokenBatchPipeline()
    result = process_document(
        source,
        NemotronOCRV2Reader(batch_size=2, pipeline=pipeline),
    )

    assert len(pipeline.inputs) == 1
    assert isinstance(pipeline.inputs[0], list)
    assert result.status == "failed"
    assert [failure.code for failure in result.failures] == [
        "nemotron_predict_failed",
        "nemotron_predict_failed",
    ]
    assert [failure.page_number for failure in result.failures] == [1, 2]


def test_nemotron_reader_can_rerun_with_a_different_merge_level(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (20, 20), "white").save(image_path)

    class MergePipeline:
        def __init__(self) -> None:
            self.merge_levels = []

        def __call__(
            self, image_path: str, *, merge_level: str
        ) -> list[dict[str, object]]:
            self.merge_levels.append(merge_level)
            return [
                {
                    "text": merge_level,
                    "confidence": 0.9,
                    "left": 0.0,
                    "lower": 0.0,
                    "right": 1.0,
                    "upper": 1.0,
                }
            ]

    pipeline = MergePipeline()
    reader = NemotronOCRV2Reader(merge_level="word", pipeline=pipeline)

    selected = reader.read(image_path, 1)
    final = reader.read_with_merge_level(image_path, 1, "paragraph")

    assert [region.text for region in selected + final] == ["word", "paragraph"]
    assert [region.text_provenance for region in selected + final] == [
        {"merge_level": "word"},
        {"merge_level": "paragraph"},
    ]
    assert pipeline.merge_levels == ["word", "paragraph"]


def test_nemotron_reader_reports_predict_and_invalid_output(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (10, 10), "white").save(image_path)

    class BrokenPipeline:
        def __call__(self, image_path: str, *, merge_level: str) -> object:
            raise RuntimeError("prediction failed")

    class InvalidPipeline:
        def __call__(self, image_path: str, *, merge_level: str) -> object:
            return [{"text": "bad"}]

    predict_result = process_document(
        image_path,
        NemotronOCRV2Reader(pipeline=BrokenPipeline()),
    )
    invalid_result = process_document(
        image_path,
        NemotronOCRV2Reader(pipeline=InvalidPipeline()),
    )

    assert predict_result.failures[0].code == "nemotron_predict_failed"
    assert invalid_result.failures[0].code == "invalid_reader_output"


def test_nemotron_reader_rejects_out_of_range_confidence(tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (10, 10), "white").save(image_path)

    class InvalidConfidencePipeline:
        def __init__(self, confidence: float) -> None:
            self.confidence = confidence

        def __call__(
            self, image_path: str, *, merge_level: str
        ) -> list[dict[str, object]]:
            return [
                {
                    "text": "bad confidence",
                    "confidence": self.confidence,
                    "left": 0.0,
                    "lower": 0.0,
                    "right": 1.0,
                    "upper": 1.0,
                }
            ]

    for confidence in (-0.1, 1.2):
        result = process_document(
            image_path,
            NemotronOCRV2Reader(pipeline=InvalidConfidencePipeline(confidence)),
        )
        assert result.failures[0].code == "invalid_reader_output"


def test_nemotron_reader_clips_image_anchored_detector_spill(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (100, 50), "white").save(image_path)

    class SpillPipeline:
        def __init__(self, box: tuple[float, float, float, float]) -> None:
            self.box = box

        def __call__(
            self, image_path: str, *, merge_level: str
        ) -> list[dict[str, object]]:
            left, lower, right, upper = self.box
            return [
                {
                    "text": "edge text",
                    "confidence": 0.9,
                    "left": left,
                    "lower": lower,
                    "right": right,
                    "upper": upper,
                }
            ]

    bottom_clipped = process_document(
        image_path,
        NemotronOCRV2Reader(pipeline=SpillPipeline((0.1, 0.75, 0.8, 1.0408))),
    )
    left_clipped = process_document(
        image_path,
        NemotronOCRV2Reader(pipeline=SpillPipeline((-0.03, 0.1, 0.8, 0.3))),
    )

    assert bottom_clipped.status == "success"
    assert bottom_clipped.pages[0].regions[0].bounding_box == BoundingBox(
        10, 37, 80, 50
    )
    assert left_clipped.status == "success"
    assert left_clipped.pages[0].regions[0].bounding_box == BoundingBox(0, 5, 80, 15)
    assert bottom_clipped.pages[0].regions[0].text_provenance == {
        "merge_level": "paragraph",
        "bounding_box_adjustment": {
            "method": "clip_to_image",
            "normalized_box": [0.1, 0.75, 0.8, 1.0408],
        },
    }


def test_nemotron_reader_rejects_detector_box_not_anchored_in_image(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (100, 50), "white").save(image_path)

    class OutsidePipeline:
        def __call__(
            self, image_path: str, *, merge_level: str
        ) -> list[dict[str, object]]:
            return [
                {
                    "text": "outside text",
                    "confidence": 0.9,
                    "left": 1.2,
                    "lower": 0.1,
                    "right": 1.5,
                    "upper": 0.3,
                }
            ]

    result = process_document(
        image_path,
        NemotronOCRV2Reader(pipeline=OutsidePipeline()),
    )

    assert result.failures[0].code == "invalid_reader_output"


def test_nemotron_reader_reports_import_and_init_failures(
    tmp_path: Path, monkeypatch
) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (10, 10), "white").save(image_path)

    missing_package = ModuleType("nemotron_ocr")
    monkeypatch.setitem(sys.modules, "nemotron_ocr", missing_package)
    monkeypatch.delitem(sys.modules, "nemotron_ocr.inference", raising=False)
    monkeypatch.delitem(
        sys.modules, "nemotron_ocr.inference.pipeline_v2", raising=False
    )
    import_result = process_document(image_path, NemotronOCRV2Reader())
    assert import_result.failures[0].code == "nemotron_import_failed"

    package = ModuleType("nemotron_ocr")
    inference = ModuleType("nemotron_ocr.inference")
    module = ModuleType("nemotron_ocr.inference.pipeline_v2")

    class FailingPipeline:
        def __init__(self, **options: object) -> None:
            raise RuntimeError("model unavailable")

    module.NemotronOCRV2 = FailingPipeline  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "nemotron_ocr", package)
    monkeypatch.setitem(sys.modules, "nemotron_ocr.inference", inference)
    monkeypatch.setitem(sys.modules, "nemotron_ocr.inference.pipeline_v2", module)
    init_result = process_document(image_path, NemotronOCRV2Reader())
    assert init_result.failures[0].code == "nemotron_init_failed"
