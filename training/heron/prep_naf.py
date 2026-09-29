"""NAF (US National Archives forms, CDLA-Permissive-1.0) to COCO for the Heron fine-tune.

Filled-in fields are `text` (NAF isBlank 1 handwriting, 2 print), signatures `signature`,
checkboxes `checkbox_selected` / `checkbox_unselected` (by transcribed mark), and each
paired label and field one `key_value_region`. The original Heron labels the other
classes; its form and key-value boxes are replaced, and its `picture` boxes over
handwriting are dropped (keep_old_label).

A checkbox box spans the box and its paired label, as Heron draws one (the pipeline reads
the label from that crop); a checkbox with no paired label is left out. With
`checkboxes`, NAF gives only the checkboxes and Heron labels every other class
(self-distillation), so a fine-tune learns pen marks without moving the other classes.
"""

import json
import sys
from pathlib import Path

from classes import CLASSES, keep_old_label, keep_true_label
from heron_labels import HeronLabeler
from PIL import Image

ROOT = Path(sys.argv[1])
CHECKBOXES_ONLY = sys.argv[2:] == ["checkboxes"]
NAF = ROOT / "NAF_dataset"
MAX_SIDE = 1650
FILLED = {1: "text", 2: "text", 4: "signature"}
REPLACED = set() if CHECKBOXES_ONLY else {"key_value_region", "form"}
label = HeronLabeler()


def box(item, scale):
    xs, ys = zip(*item["poly_points"])
    return [min(xs) * scale, min(ys) * scale, max(xs) * scale, max(ys) * scale]


def union(a, b):
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def truth(annotation, scale):
    fields = {item["id"]: item for item in annotation["fieldBBs"]}
    texts = {item["id"]: item for item in annotation["textBBs"]}
    labels = {}
    for first, second in annotation["pairs"]:
        for field_id, text_id in ((first, second), (second, first)):
            if field_id in fields and text_id in texts:
                labels.setdefault(field_id, texts[text_id])
    for item in fields.values():
        if item["type"] == "fieldCheckBox":
            if item["id"] not in labels:
                continue
            # isBlank marks unticked boxes as filled too; the transcription ("X") is the state.
            mark = annotation.get("transcriptions", {}).get(item["id"])
            ticked = (
                bool(mark.strip()) if mark is not None else item.get("isBlank") != 3
            )
            yield (
                ("checkbox_selected" if ticked else "checkbox_unselected"),
                union(box(item, scale), box(labels[item["id"]], scale)),
            )
        elif CHECKBOXES_ONLY:
            continue
        elif item.get("isBlank") in FILLED:
            yield FILLED[item["isBlank"]], box(item, scale)
    for first, second in [] if CHECKBOXES_ONLY else annotation["pairs"]:
        label_box, field = (
            texts.get(first) or texts.get(second),
            fields.get(second) or fields.get(first),
        )
        if label_box and field and field["type"] in {"field", "fieldP"}:
            a, b = box(label_box, scale), box(field, scale)
            yield "key_value_region", union(a, b)


# Template-disjoint split: a form type is never in two splits.
splits = json.loads((NAF / "simple_train_valid_test_split.json").read_text())
out = ROOT / "naf"
out.mkdir(exist_ok=True)
for split, groups in splits.items():
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": n} for i, n in enumerate(CLASSES)],
    }
    for group, files in groups.items():
        for name in files:
            source = NAF / "groups" / group / name
            if not source.exists():
                continue
            annotation = json.loads(source.with_suffix(".json").read_text())
            image = Image.open(source).convert("RGB")
            scale = min(1.0, MAX_SIDE / max(image.size))
            if scale < 1:
                image = image.resize(
                    (round(image.width * scale), round(image.height * scale)),
                    Image.LANCZOS,
                )
            width, height = image.size
            image.save(out / name.replace(".jpg", ".png"))
            true = list(truth(annotation, scale))
            old = [
                (cls, b)
                for cls, b in label(image)
                if cls not in REPLACED and keep_old_label(cls, b, true)
            ]
            image_id = len(coco["images"])
            coco["images"].append(
                {
                    "id": image_id,
                    "file_name": name.replace(".jpg", ".png"),
                    "width": width,
                    "height": height,
                }
            )
            true = [(cls, b) for cls, b in true if keep_true_label(cls, b, old)]
            for cls, (x0, y0, x1, y1) in old + true:
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
    suffix = "_checkboxes" if CHECKBOXES_ONLY else ""
    (ROOT / f"labels_naf{suffix}_{split}.json").write_text(json.dumps(coco))
    print(split, len(coco["images"]), len(coco["annotations"]), flush=True)
