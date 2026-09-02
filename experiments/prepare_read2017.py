"""Convert READ 2017 PAGE XML into line-level handwriting crops."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Sequence
import xml.etree.ElementTree as ET

from PIL import Image, UnidentifiedImageError


DATASET = "READ 2017 Train-A"
LICENSE = "CC BY 4.0"
BARE_AMPERSAND = re.compile(r"&(?!#\d+;|#x[0-9A-Fa-f]+;|[A-Za-z_:][\w:.-]*;)")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = prepare_read2017(
            args.source,
            args.output,
            max_lines_per_page=args.max_lines_per_page,
        )
    except (OSError, ValueError, ET.ParseError) as error:
        build_parser().error(str(error))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-lines-per-page", type=_positive_int, default=4)
    return parser


def prepare_read2017(
    source: Path,
    output: Path,
    *,
    max_lines_per_page: int = 4,
) -> dict[str, Any]:
    if not source.is_dir():
        raise FileNotFoundError(f"READ source was not found: {source}")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if max_lines_per_page < 1:
        raise ValueError("max lines per page must be positive")

    xml_paths = sorted(source.glob("*.xml"), key=lambda path: path.name.casefold())
    if not xml_paths:
        raise ValueError("READ source contains no PAGE XML files")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    records: list[dict[str, Any]] = []
    try:
        for xml_path in xml_paths:
            records.extend(
                _prepare_page(
                    source,
                    temporary,
                    xml_path,
                    max_lines=max_lines_per_page,
                )
            )
        if not records:
            raise ValueError("READ PAGE XML contains no usable text lines")
        _write_jsonl(temporary / "train.jsonl", records)
        summary = {
            "dataset": DATASET,
            "license": LICENSE,
            "pages": len({record["case_id"] for record in records}),
            "lines": len(records),
            "max_lines_per_page": max_lines_per_page,
            "split": "train_only",
        }
        (temporary / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return summary


def _prepare_page(
    source: Path,
    output: Path,
    xml_path: Path,
    *,
    max_lines: int,
) -> list[dict[str, Any]]:
    root = _parse_xml(xml_path)
    page = next((node for node in root.iter() if _local(node.tag) == "Page"), None)
    if page is None:
        raise ValueError(f"PAGE XML has no Page element: {xml_path}")
    filename = page.get("imageFilename")
    if not filename or Path(filename).name != filename:
        raise ValueError(f"PAGE XML has an unsafe image filename: {xml_path}")
    image_path = source / filename
    if not image_path.is_file():
        raise FileNotFoundError(f"READ image was not found: {image_path}")

    try:
        with Image.open(image_path) as image:
            image.load()
            width, height = image.size
            lines = _text_lines(page, width, height)
            selected = _even_sample(lines, max_lines)
            records = []
            for line_number, box, text in selected:
                field_id = f"READ2017-{xml_path.stem}-L{line_number:03d}"
                tight_path = Path("crops") / "tight" / f"{field_id}.png"
                padded_path = Path("crops") / "padded" / f"{field_id}.png"
                padded_box = _padded_box(box, (width, height))
                _save_crop(image, box, output / tight_path)
                _save_crop(image, padded_box, output / padded_path)
                records.append(
                    {
                        "field_id": field_id,
                        "case_id": f"READ2017-{xml_path.stem}",
                        "category_id": "READ2017",
                        "family_id": f"READ2017-{xml_path.stem}",
                        "reference": text,
                        "crop_path": tight_path.as_posix(),
                        "tight_crop_path": tight_path.as_posix(),
                        "padded_crop_path": padded_path.as_posix(),
                        "split": "train",
                        "data_origin": "public",
                        "target_state": "resolved",
                        "dataset": DATASET,
                        "license": LICENSE,
                    }
                )
            return records
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError(f"could not read READ image: {image_path}") from error


def _text_lines(
    page: ET.Element,
    width: int,
    height: int,
) -> list[tuple[int, tuple[int, int, int, int], str]]:
    lines = []
    for line_number, line in enumerate(
        (node for node in page.iter() if _local(node.tag) == "TextLine"),
        start=1,
    ):
        text = next(
            (
                node.text.strip()
                for node in line.iter()
                if _local(node.tag) == "Unicode" and node.text and node.text.strip()
            ),
            "",
        )
        coords = next(
            (node.get("points") for node in line if _local(node.tag) == "Coords"),
            None,
        )
        if not text or not coords:
            continue
        box = _points_box(coords, width, height)
        if box is not None:
            lines.append((line_number, box, text))
    return lines


def _points_box(
    value: str,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    try:
        points = [tuple(map(int, point.split(","))) for point in value.split()]
    except (TypeError, ValueError):
        return None
    if len(points) < 2 or any(len(point) != 2 for point in points):
        return None
    left = max(0, min(point[0] for point in points))
    top = max(0, min(point[1] for point in points))
    right = min(width, max(point[0] for point in points) + 1)
    bottom = min(height, max(point[1] for point in points) + 1)
    return (left, top, right, bottom) if right > left and bottom > top else None


def _even_sample(
    values: list[tuple[int, tuple[int, int, int, int], str]],
    limit: int,
) -> list[tuple[int, tuple[int, int, int, int], str]]:
    if len(values) <= limit:
        return values
    if limit == 1:
        return [values[len(values) // 2]]
    positions = [
        round(index * (len(values) - 1) / (limit - 1)) for index in range(limit)
    ]
    return [values[index] for index in positions]


def _padded_box(
    box: tuple[int, int, int, int],
    image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    padding = max(8, round((box[3] - box[1]) * 0.5))
    return (
        max(0, box[0] - padding),
        max(0, box[1] - padding),
        min(image_size[0], box[2] + padding),
        min(image_size[1], box[3] + padding),
    )


def _save_crop(
    image: Image.Image,
    box: tuple[int, int, int, int],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.crop(box).save(path, format="PNG")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _parse_xml(path: Path) -> ET.Element:
    try:
        return ET.parse(path).getroot()
    except ET.ParseError:
        raw = path.read_text(encoding="utf-8")
        return ET.fromstring(BARE_AMPERSAND.sub("&amp;", raw))


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return number


if __name__ == "__main__":
    raise SystemExit(main())
