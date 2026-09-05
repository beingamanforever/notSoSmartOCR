from __future__ import annotations

import argparse

import pytest

from experiments.benchmark_unitable_tables import parse_crop, rescale_boxes


def test_parse_crop_accepts_non_empty_box() -> None:
    assert parse_crop("1,2,31,42") == (1, 2, 31, 42)


@pytest.mark.parametrize(
    "value",
    ["1,2,3", "-1,2,3,4", "2,2,2,4", "2,4,5,4"],
)
def test_parse_crop_rejects_invalid_box(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_crop(value)


def test_rescale_boxes_preserves_coordinate_axes() -> None:
    assert rescale_boxes([[1, 2, 3, 4]], (10, 20), (20, 80)) == [[2, 8, 6, 16]]
