"""SignverOD to COCO: signatures and initials as `signature`, dates as `text`, old classes from Heron."""

import hashlib
import io
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
from classes import CLASSES, iou, keep_old_label, keep_true_label
from heron_labels import HeronLabeler
from PIL import Image

ROOT = Path(sys.argv[1])
MAP = {1: "signature", 2: "signature", 4: "text"}  # 3 is redaction
label = HeronLabeler()
out = ROOT / "signverod"
out.mkdir(exist_ok=True)
splits = {name: {"images": [], "annotations": []} for name in ("train", "val")}
for part in sorted((ROOT / "data/signverod").glob("*.parquet")):
    for row in pq.read_table(part).to_pylist():
        image = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
        scale = min(1.0, 1650 / max(image.size))
        if scale < 1:
            image = image.resize(
                (round(image.width * scale), round(image.height * scale)), Image.LANCZOS
            )
        width, height = image.size
        name = f"sv_{row['image_id']}.png"
        image.save(out / name)
        objects = row["objects"]
        boxes = [
            [b[0] * scale, b[1] * scale, (b[0] + b[2]) * scale, (b[1] + b[3]) * scale]
            for b in objects["bbox"]
        ]
        truth = [(MAP[c], b) for b, c in zip(boxes, objects["category"]) if c in MAP]
        redactions = [b for b, c in zip(boxes, objects["category"]) if c == 3]
        # The original Heron has no signature class, so its boxes on marks are wrong.
        old = [
            (cls, b)
            for cls, b in label(image)
            if keep_old_label(cls, b, truth)
            and not any(iou(b, r) > 0.3 for r in redactions)
        ]
        split = (
            "val"
            if int(hashlib.sha1(name.encode()).hexdigest(), 16) % 10 == 0
            else "train"
        )
        coco = splits[split]
        image_id = len(coco["images"])
        coco["images"].append(
            {"id": image_id, "file_name": name, "width": width, "height": height}
        )
        truth = [(cls, b) for cls, b in truth if keep_true_label(cls, b, old)]
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
for split, coco in splits.items():
    coco["categories"] = [{"id": i, "name": n} for i, n in enumerate(CLASSES)]
    (ROOT / f"labels_signverod_{split}.json").write_text(json.dumps(coco))
    print(split, len(coco["images"]), len(coco["annotations"]), flush=True)
