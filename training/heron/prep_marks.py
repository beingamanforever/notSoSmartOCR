"""Pen-marked checkboxes for the Heron fine-tune: CommonForms pages, Heron's own labels.

Heron does not propose a printed box once a pen tick crosses it (checked recall 0.064 on
the handwritten forms), and public data has few such boxes (NAF: 40 ticked). Each page is
labelled by the deployed path (page pass, 3x3 control tiles, Docling's postprocessor), so
the fine-tune keeps every class where Heron already puts it (self-distillation). Half of
the unselected checkboxes then get a synthetic pen mark over their glyph and become
`checkbox_selected`: copy-paste style augmentation of the one missing appearance
(Ghiasi et al., CVPR 2021). CommonForms is Apache-2.0 and shares no page with any
evaluation set.

Usage: prep_marks.py ROOT [PAGES]  (reads jbarrow/CommonForms `valid` from the HF cache)
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
from classes import CLASSES
from datasets import load_dataset
from PIL import ImageDraw

from ocr_pipeline.heron_layout import (
    CONTROL_LABELS,
    CONTROL_REGION_LABELS,
    HeronTableDetector,
    resolve_layout,
)

MARKED_SHARE = 0.5
heron = None  # the deployed detector, loaded by main


def labels(image):
    """What the service resolves on this page, controls from its 3x3 tiles included."""
    global heron
    heron = heron or HeronTableDetector(score_threshold=0.3)  # the service's threshold
    proposals = heron.detect_elements(image)
    regions = [p["box"] for p in proposals if p["label"] in CONTROL_REGION_LABELS]
    if regions:
        controls = [p for p in proposals if p["label"] in CONTROL_LABELS]
        proposals += heron.detect_sliced_controls(image, 3, regions, controls)
    return resolve_layout(proposals, *image.size)


def glyph(image, box):
    """The control's box glyph: its leftmost square ink component, else a left square."""
    import cv2

    left, top, right, bottom = (int(v) for v in box)
    gray = np.asarray(image.convert("L").crop((left, top, right, bottom)))
    side = bottom - top
    if gray.size:
        ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        _, _, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
        squares = [s for s in stats[1:] if 0.4 * side <= s[3] <= side and 0.75 <= s[2] / s[3] <= 1.33]
        if squares:
            x, y, w, h, _ = min(squares, key=lambda s: s[0])
            return left + x, top + y, w
    return left, top, side


def pen_mark(draw, x, y, s):
    """A hand-drawn tick, cross, slash, scribble or fill over an s-pixel box at (x, y)."""
    jitter = lambda: random.uniform(-0.1, 0.1) * s
    ink = tuple(random.randint(0, 60) for _ in range(2)) + (random.randint(0, 140),)
    width = max(1, round(s * random.uniform(0.1, 0.25)))
    kind = random.choice(["tick", "tick", "cross", "slash", "scribble", "fill"])
    if kind == "tick":  # short stroke down, long stroke up past the box, as on real forms
        low = (x + 0.45 * s + jitter(), y + 0.9 * s + jitter())
        strokes = [[(x + 0.1 * s + jitter(), y + 0.45 * s + jitter()), low,
                    (x + s * random.uniform(1.0, 1.8), y - s * random.uniform(0.2, 1.0))]]
    elif kind == "cross":
        over = s * random.uniform(0.0, 0.3)
        strokes = [[(x - over, y - over), (x + s + over, y + s + over)],
                   [(x + s + over, y - over), (x - over, y + s + over)]]
    elif kind == "slash":
        strokes = [[(x + jitter(), y + s + jitter()), (x + s + jitter(), y + jitter())]]
    elif kind == "scribble":
        strokes = [[(x + random.uniform(0.1, 0.9) * s, y + random.uniform(0.1, 0.9) * s) for _ in range(6)]]
    else:
        inset = s * random.uniform(0.15, 0.3)
        draw.rectangle((x + inset, y + inset, x + s - inset, y + s - inset), fill=ink)
        return
    for points in strokes:
        draw.line(points, fill=ink, width=width, joint="curve")


def mark_page(image, detections):
    draw = ImageDraw.Draw(image)
    for detection in detections:
        if detection["label"] == "checkbox_unselected" and random.random() < MARKED_SHARE:
            pen_mark(draw, *glyph(image, detection["box"]))
            detection["label"] = "checkbox_selected"
    return detections



def main(root, pages):
    out = root / "marks"
    out.mkdir(parents=True, exist_ok=True)
    random.seed(0)
    coco = {"images": [], "annotations": [], "categories": [{"id": i, "name": n} for i, n in enumerate(CLASSES)]}
    for row in load_dataset("jbarrow/CommonForms", split="valid", streaming=True):
        if len(coco["images"]) >= pages:
            break
        image = row["image"].convert("RGB")
        detections = labels(image)
        if sum(d["label"] == "checkbox_unselected" for d in detections) < 2:
            continue  # a page without controls teaches nothing new
        detections = mark_page(image, detections)
        image_id = len(coco["images"])
        name = f"{row['id']}.png"
        image.save(out / name)
        coco["images"].append({"id": image_id, "file_name": name, "width": image.width, "height": image.height})
        for d in detections:
            x0, y0, x1, y1 = d["box"]
            coco["annotations"].append({
                "id": len(coco["annotations"]), "image_id": image_id, "category_id": CLASSES.index(d["label"]),
                "bbox": [x0, y0, x1 - x0, y1 - y0], "area": (x1 - x0) * (y1 - y0), "iscrowd": 0,
            })
        if image_id % 100 == 0:
            print(image_id, "pages", len(coco["annotations"]), "boxes", flush=True)
    # Pages in order; the last tenth validates.
    cut = int(len(coco["images"]) * 0.9)
    for split, keep in (("train", lambda i: i < cut), ("valid", lambda i: i >= cut)):
        part = {**coco, "images": [i for i in coco["images"] if keep(i["id"])],
                "annotations": [a for a in coco["annotations"] if keep(a["image_id"])]}
        (root / f"labels_marks_{split}.json").write_text(json.dumps(part))
        print(split, len(part["images"]), "pages", len(part["annotations"]), "boxes")


if __name__ == "__main__":
    main(Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 3000)
