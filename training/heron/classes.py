"""Label space for the fine-tuned Heron: its 17 classes, then the added ones."""

OLD = [
    "caption",
    "footnote",
    "formula",
    "list_item",
    "page_footer",
    "page_header",
    "picture",
    "section_header",
    "table",
    "text",
    "title",
    "document_index",
    "code",
    "checkbox_selected",
    "checkbox_unselected",
    "form",
    "key_value_region",
]
CLASSES = OLD + ["signature", "hand_mark"]
HERON, REVISION = (
    "ds4sd/docling-layout-heron-101",
    "2e4993cf6bb211112084a2f80938f26138008917",
)


def iou(a, b) -> float:
    w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h
    return w * h / union if union > 0 else 0.0


def covered(inner, outer) -> float:
    """Share of `inner`'s area that lies inside `outer`."""
    w = max(0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    h = max(0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return w * h / area if area > 0 else 0.0


CONTAINERS = {"form", "key_value_region"}


TEXT_LIKE = {
    "caption",
    "footnote",
    "list_item",
    "page_footer",
    "page_header",
    "section_header",
    "text",
    "title",
}


def keep_true_label(cls, box, old) -> bool:
    """A true text box, unless it lies inside a line the original Heron already reads.

    Heron labels a whole "Name: value" line as one text element and Falcon reads the
    handwritten value with it; adding the value as a second, nested text box gives the
    set-prediction loss two targets for one element (14% of NAF text boxes were nested).
    """
    return not (
        cls == "text"
        and any(old_cls in TEXT_LIKE and covered(box, old_box) > 0.5 for old_cls, old_box in old)
    )


def keep_old_label(cls, box, truth) -> bool:
    """The original Heron's box for an old class, unless it mislabels a true box.

    Heron calls handwriting and signatures `picture`, so a picture holding most of a
    true (handwritten or signature) box is that mislabel; other boxes that overlap a
    true box are the same element seen differently. A true container (form, key-value
    field) holds text elements as in DocLayNet, so it never removes them.
    """
    return not any(
        iou(box, true_box) > 0.3 or (cls == "picture" and covered(true_box, box) > 0.5)
        for true_cls, true_box in truth
        if true_cls not in CONTAINERS
    )
