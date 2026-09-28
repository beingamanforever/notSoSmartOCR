"""KVP10k to COCO: each linked key and value as one `key_value_region`, old classes from Heron.

KVP10k (IBM, CDLA-Permissive-2.0) links every value rectangle to its key. A field is the
union of the two; a key with no value (a blank field) is a field on its own. The original
Heron labels every other class; its own key-value and form boxes are dropped, since
those are what this data replaces.
"""

import json
import sys
from pathlib import Path

from classes import CLASSES, keep_old_label
from heron_labels import HeronLabeler
from PIL import Image

ROOT = Path(sys.argv[1])
REPLACED = {"key_value_region", "form"}
label = HeronLabeler()


def fields(item, width, height):
    boxes = {}
    for rect in item["annotations"]["rectangles"]:
        points = rect.get("coordinates") or []
        if len(points) == 4:
            xs, ys = [p["x"] * width for p in points], [p["y"] * height for p in points]
            boxes[rect["_id"]] = (rect, [min(xs), min(ys), max(xs), max(ys)])
    linked = {}
    for rect, box in boxes.values():
        key = ((rect.get("attributes") or {}).get("Linking") or {}).get("value")
        if key in boxes:
            linked.setdefault(key, []).append(box)
    for key, (rect, box) in boxes.items():
        if key in linked or rect.get("label") == "unvalued_key":
            parts = [box, *linked.get(key, [])]
            yield [
                min(p[0] for p in parts),
                min(p[1] for p in parts),
                max(p[2] for p in parts),
                max(p[3] for p in parts),
            ]


for split in ("train", "test"):
    folder = ROOT / "kvp10k" / split
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": n} for i, n in enumerate(CLASSES)],
    }
    for item_path in sorted((folder / "items").glob("*.json")):
        name = item_path.stem + ".png"
        with Image.open(folder / "images" / name) as source:
            image = source.convert("RGB")
        width, height = image.size
        truth = [
            ("key_value_region", box)
            for box in fields(json.loads(item_path.read_text()), width, height)
        ]
        if not truth:
            continue
        old = [
            (cls, box)
            for cls, box in label(image)
            if cls not in REPLACED and keep_old_label(cls, box, truth)
        ]
        image_id = len(coco["images"])
        coco["images"].append(
            {
                "id": image_id,
                "file_name": f"{split}/images/{name}",
                "width": width,
                "height": height,
            }
        )
        for cls, (x0, y0, x1, y1) in old + truth:
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(width, x1), min(height, y1)
            if x1 - x0 >= 2 and y1 - y0 >= 2:
                coco["annotations"].append(
                    {
                        "id": len(coco["annotations"]),
                        "image_id": image_id,
                        "category_id": CLASSES.index(cls),
                        "bbox": [x0, y0, x1 - x0, y1 - y0],
                        "area": (x1 - x0) * (y1 - y0),
                        "iscrowd": 0,
                    }
                )
    (ROOT / f"labels_kvp10k_{split}.json").write_text(json.dumps(coco))
    print(split, len(coco["images"]), len(coco["annotations"]), flush=True)
