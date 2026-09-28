"""DocLayNet replay: ground truth for its 11 classes, the original Heron for the 6 it lacks."""

import json
import random
import sys
import zipfile
from pathlib import Path

from classes import CLASSES
from heron_labels import HeronLabeler
from PIL import Image

ROOT = Path(sys.argv[1])
MISSING = {
    "document_index",
    "code",
    "checkbox_selected",
    "checkbox_unselected",
    "form",
    "key_value_region",
}
# val selects checkpoints; test is reported once.
SAMPLE = {"train": 3000, "val": 600, "test": 600}
archive = zipfile.ZipFile(ROOT / "data/DocLayNet_core.zip")
label = HeronLabeler()
out = ROOT / "doclaynet"
out.mkdir(exist_ok=True)
for split, count in SAMPLE.items():
    if (ROOT / f"labels_doclaynet_{split}.json").exists():
        continue
    coco = json.loads(archive.read(f"COCO/{split}.json"))
    source = {c["id"]: c["name"].lower().replace("-", "_") for c in coco["categories"]}
    by_image = {}
    for a in coco["annotations"]:
        by_image.setdefault(a["image_id"], []).append(a)
    result = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": n} for i, n in enumerate(CLASSES)],
    }
    # A per-split generator keeps each sample fixed whichever splits already exist.
    for image in random.Random(f"doclaynet-{split}").sample(coco["images"], count):
        with archive.open(f"PNG/{image['file_name']}") as handle:
            page = Image.open(handle).convert("RGB")
        page.save(out / image["file_name"])
        image_id = len(result["images"])
        result["images"].append(
            {
                "id": image_id,
                "file_name": image["file_name"],
                "width": page.width,
                "height": page.height,
            }
        )
        boxes = [
            (source[a["category_id"]], a["bbox"]) for a in by_image.get(image["id"], [])
        ]
        boxes += [
            (cls, [x0, y0, x1 - x0, y1 - y0])
            for cls, (x0, y0, x1, y1) in label(page, keep=MISSING)
        ]
        for cls, (x, y, w, h) in boxes:
            if w >= 2 and h >= 2:
                result["annotations"].append(
                    {
                        "id": len(result["annotations"]),
                        "image_id": image_id,
                        "category_id": CLASSES.index(cls),
                        "bbox": [x, y, w, h],
                        "area": w * h,
                        "iscrowd": 0,
                    }
                )
    (ROOT / f"labels_doclaynet_{split}.json").write_text(json.dumps(result))
    print(split, len(result["images"]), len(result["annotations"]), flush=True)
