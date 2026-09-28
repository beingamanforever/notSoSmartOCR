"""Score a Heron checkpoint, or the original via the output-preserving expansion, per eval set."""

import argparse
import json
from functools import partial
from pathlib import Path

import albumentations as A
from classes import CLASSES, HERON, REVISION
from run_object_detection import (
    augment_and_transform_batch,
    collate_fn,
    compute_metrics,
)
from train_heron import coco_dataset, expanded_model
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    Trainer,
    TrainingArguments,
)

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="original")
parser.add_argument("--eval", nargs="+", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
processor = AutoImageProcessor.from_pretrained(HERON, revision=REVISION)
model = (
    expanded_model()[0]
    if args.model == "original"
    else AutoModelForObjectDetection.from_pretrained(args.model)
)
plain = A.Compose(
    [A.NoOp()],
    bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True),
)
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="/tmp/heron-eval",
        # A partial last batch makes accelerate slice every label tensor (orig_size, boxes).
        per_device_eval_batch_size=1,
        eval_accumulation_steps=8,
        bf16=True,
        remove_unused_columns=False,
        eval_do_concat_batches=False,
        report_to=[],
        dataloader_num_workers=2,
    ),
    data_collator=collate_fn,
    processing_class=processor,
    # The official compute_metrics reads only outputs [1] (logits) and [2] (boxes); keeping
    # the encoder feature maps of every image ran the host out of memory.
    preprocess_logits_for_metrics=lambda outputs, labels: outputs[:3],
    compute_metrics=partial(
        compute_metrics,
        image_processor=processor,
        id2label=dict(enumerate(CLASSES)),
        threshold=0.0,
    ),
)
scores = {}
for spec in args.eval:
    coco, root = spec.split("=")
    data = coco_dataset(Path(coco), Path(root)).with_transform(
        partial(augment_and_transform_batch, transform=plain, image_processor=processor)
    )
    scores[Path(coco).stem] = trainer.evaluate(eval_dataset=data)
Path(args.output).write_text(json.dumps(scores, indent=1))
for name, s in scores.items():
    print(
        name,
        {
            k: v
            for k, v in s.items()
            if k
            in (
                "eval_map",
                "eval_map_50",
                "eval_map_signature",
                "eval_mar_100_signature",
                "eval_map_text",
                "eval_map_table",
            )
        },
    )
