"""Old-class labels from the original Heron, resolved by Docling's official postprocessor."""

import torch
from classes import HERON, REVISION
from transformers import AutoImageProcessor, AutoModelForObjectDetection

from ocr_pipeline.heron_layout import resolve_layout


class HeronLabeler:
    def __init__(self, device: str = "cuda") -> None:
        self.processor = AutoImageProcessor.from_pretrained(HERON, revision=REVISION)
        self.model = (
            AutoModelForObjectDetection.from_pretrained(HERON, revision=REVISION)
            .to(device)
            .eval()
        )
        self.device = device

    def __call__(self, image, keep=None):
        width, height = image.size
        with torch.inference_mode():
            output = self.model(
                **self.processor(images=[image], return_tensors="pt").to(self.device)
            )
        probabilities = output.logits[0].sigmoid().float().cpu()
        scores, labels = probabilities.max(-1)
        centers, sizes = output.pred_boxes[0].float().cpu().split(2, -1)
        boxes = torch.cat(
            (centers - sizes / 2, centers + sizes / 2), -1
        ) * torch.tensor([width, height, width, height])
        names = self.model.config.id2label
        detections = [
            {
                "label": names[int(labels[q])],
                "score": float(scores[q]),
                "box": tuple(boxes[q].tolist()),
            }
            for q in range(len(scores))
            if scores[q] >= 0.2
        ]
        return [
            (d["label"], list(d["box"]))
            for d in resolve_layout(detections, width, height)
            if keep is None or d["label"] in keep
        ]
