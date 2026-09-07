from __future__ import annotations

import importlib.util
import tempfile

import pytest
from PIL import Image

from ocr_pipeline.contracts import BoundingBox
from ocr_pipeline.providers import ReaderError
from ocr_pipeline.tableformer_structure import MODEL_ID, TableFormerStructure

DOCLING_IBM_MODELS_AVAILABLE = (
    importlib.util.find_spec("docling_ibm_models") is not None
)


def _raw_cell(
    *,
    left: float,
    top: float,
    right: float,
    bottom: float,
    row_start: int,
    row_end: int,
    column_start: int,
    column_end: int,
    column_header: bool = False,
    row_section: bool = False,
) -> dict:
    return {
        "cell_id": row_start * 10 + column_start,
        "bbox": {"l": left, "t": top, "r": right, "b": bottom, "token": ""},
        "row_span": row_end - row_start,
        "col_span": column_end - column_start,
        "start_row_offset_idx": row_start,
        "end_row_offset_idx": row_end,
        "start_col_offset_idx": column_start,
        "end_col_offset_idx": column_end,
        "indentation_level": 0,
        "text_cell_bboxes": [],
        "column_header": column_header,
        "row_header": False,
        "row_section": row_section,
    }


def test_predict_converts_geometry_and_spans() -> None:
    responses = [
        _raw_cell(
            left=1.4,
            top=2.2,
            right=49.6,
            bottom=19.8,
            row_start=0,
            row_end=1,
            column_start=0,
            column_end=1,
            column_header=True,
        ),
        _raw_cell(
            left=50.0,
            top=20.0,
            right=150.0,
            bottom=60.0,
            row_start=1,
            row_end=3,
            column_start=0,
            column_end=2,
            row_section=True,
        ),
    ]
    structure = TableFormerStructure(predictor=lambda crop: responses)

    cells = structure.predict(Image.new("RGB", (150, 60)))

    assert len(cells) == 2
    header, section = cells
    assert header.bounding_box == BoundingBox(1, 2, 50, 20)
    assert header.row_nums == (0,)
    assert header.column_nums == (0,)
    assert header.column_header is True
    assert header.projected_row_header is False
    assert section.bounding_box == BoundingBox(50, 20, 150, 60)
    assert section.row_nums == (1, 2)
    assert section.column_nums == (0, 1)
    assert section.column_header is False
    assert section.projected_row_header is True


def test_predict_drops_degenerate_and_malformed_boxes() -> None:
    responses = [
        _raw_cell(
            left=10.0,
            top=10.0,
            right=10.0,
            bottom=20.0,
            row_start=0,
            row_end=1,
            column_start=0,
            column_end=1,
        ),
        {"start_row_offset_idx": 0, "end_row_offset_idx": 1},  # no bbox at all
        _raw_cell(
            left=0.0,
            top=0.0,
            right=30.0,
            bottom=30.0,
            row_start=0,
            row_end=1,
            column_start=0,
            column_end=1,
        ),
    ]
    structure = TableFormerStructure(predictor=lambda crop: responses)

    cells = structure.predict(Image.new("RGB", (30, 30)))

    assert len(cells) == 1
    assert cells[0].bounding_box == BoundingBox(0, 0, 30, 30)


def test_predict_wraps_predictor_failure() -> None:
    def failing_predictor(crop: Image.Image) -> list[dict]:
        raise RuntimeError("boom")

    structure = TableFormerStructure(predictor=failing_predictor)

    with pytest.raises(ReaderError) as error:
        structure.predict(Image.new("RGB", (10, 10)))
    assert error.value.code == "tableformer_predict_failed"


def test_model_provenance() -> None:
    structure = TableFormerStructure(predictor=lambda crop: [])

    provenance = structure.model_provenance()

    assert provenance["id"] == MODEL_ID
    assert provenance["model"] == "TableFormer (TableModel04rs, OTSL)"
    assert provenance["code_license"] == "MIT"
    assert provenance["weights_license"] == "CDLA-Permissive-2.0"
    assert provenance["origin"] == "IBM Research"
    assert "issue 188" in provenance["note"]
    assert "MatchingPostProcessor" in provenance["note"]


@pytest.mark.skipif(
    not DOCLING_IBM_MODELS_AVAILABLE, reason="docling-ibm-models is not installed"
)
def test_tf_predictor_construction_needs_weights_offline() -> None:
    """Confirms the TFPredictor config contract without downloading any weights.

    TableModel04_rs only reads the "word_map_tag" entry of "dataset_wordmap",
    so a minimal wordmap is enough to build the model itself; the missing
    weights file is what should fail, and only that.
    """
    from docling_ibm_models.tableformer.data_management.tf_predictor import (
        TFPredictor,
    )

    with tempfile.TemporaryDirectory() as empty_save_dir:
        config = {
            "model": {
                "type": "TableModel04_rs",
                "save_dir": empty_save_dir,
                "backbone": "resnet18",
                "enc_image_size": 28,
                "tag_embed_dim": 16,
                "hidden_dim": 512,
                "tag_decoder_dim": 512,
                "bbox_embed_dim": 256,
                "tag_attention_dim": 256,
                "bbox_attention_dim": 512,
                "enc_layers": 1,
                "dec_layers": 1,
                "nheads": 8,
                "dropout": 0.1,
                "bbox_classes": 2,
            },
            "train": {"bbox": True},
            "predict": {
                "max_steps": 10,
                "beam_size": 1,
                "bbox": True,
                "pdf_cell_iou_thres": 0.05,
                "padding": False,
            },
            "dataset_wordmap": {
                "word_map_tag": {
                    "<pad>": 0,
                    "<unk>": 1,
                    "<start>": 2,
                    "<end>": 3,
                    "ecel": 4,
                    "fcel": 5,
                    "lcel": 6,
                    "ucel": 7,
                    "xcel": 8,
                    "nl": 9,
                    "ched": 10,
                    "rhed": 11,
                    "srow": 12,
                }
            },
        }
        with pytest.raises(ValueError, match="model file"):
            TFPredictor(config, device="cpu", num_threads=1)
