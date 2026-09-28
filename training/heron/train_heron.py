"""Fine-tune Heron-101 with signature and hand_mark classes, reusing the official HF script.

The class heads grow from 17 to 19 rows so the original 17 outputs are unchanged at
initialization: old rows are copied, new rows start at zero weight and a low prior,
and the encoder score head gives new classes a -30 bias so query selection is intact.
`run_object_detection.py` is the official transformers example, fetched by setup.sh.
"""

import argparse
import json
from functools import partial
from pathlib import Path

import albumentations as A
import torch
from classes import CLASSES, HERON, OLD, REVISION
from datasets import Dataset, Features, Image, Sequence, Value, concatenate_datasets
from run_object_detection import augment_and_transform_batch, collate_fn
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForObjectDetection,
    Trainer,
    TrainingArguments,
)


def expanded_model():
    old = AutoModelForObjectDetection.from_pretrained(HERON, revision=REVISION)
    config = AutoConfig.from_pretrained(
        HERON,
        revision=REVISION,
        id2label=dict(enumerate(CLASSES)),
        label2id={n: i for i, n in enumerate(CLASSES)},
    )
    new = AutoModelForObjectDetection.from_config(config)
    shapes = new.state_dict()
    new.load_state_dict(
        {k: v for k, v in old.state_dict().items() if shapes[k].shape == v.shape},
        strict=False,
    )
    prior = -torch.log(torch.tensor((1 - 0.01) / 0.01))
    count = len(OLD)
    with torch.no_grad():
        for head, source in zip(
            [new.model.enc_score_head, *new.model.decoder.class_embed],
            [old.model.enc_score_head, *old.model.decoder.class_embed],
        ):
            head.weight.zero_()
            head.weight[:count] = source.weight
            head.bias.fill_(prior)
            head.bias[:count] = source.bias
        new.model.enc_score_head.bias[count:] = -30.0
        embed, source = (
            new.model.denoising_class_embed.weight,
            old.model.denoising_class_embed.weight,
        )
        embed[:count] = source[:count]
        embed[-1] = source[-1]
    return new, old


def check_preserved(new, old, processor, images) -> float:
    new.eval()
    old.eval()
    with torch.inference_mode():
        batch = processor(images=images, return_tensors="pt")
        return float(
            (new(**batch).logits[..., : len(OLD)] - old(**batch).logits).abs().max()
        )


def coco_dataset(coco_path: Path, image_root: Path) -> Dataset:
    coco = json.loads(coco_path.read_text())
    by_image = {}
    for a in coco["annotations"]:
        by_image.setdefault(a["image_id"], []).append(a)
    rows = [
        {
            "image_id": image["id"],
            "image": str(image_root / image["file_name"]),
            "objects": {
                key: [a[field] for a in by_image.get(image["id"], [])]
                for key, field in (
                    ("bbox", "bbox"),
                    ("category", "category_id"),
                    ("area", "area"),
                    ("id", "id"),
                )
            },
        }
        for image in coco["images"]
    ]
    features = Features(
        {
            "image_id": Value("int64"),
            "image": Image(),
            "objects": {
                "bbox": Sequence(Sequence(Value("float32"), length=4)),
                "category": Sequence(Value("int64")),
                "area": Sequence(Value("float32")),
                "id": Sequence(Value("int64")),
            },
        }
    )
    return Dataset.from_list(rows, features=features)


def load(specs):
    return concatenate_datasets(
        [coco_dataset(Path(c), Path(r)) for c, r in (s.split("=") for s in specs)]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train", nargs="+", required=True, help="coco.json=image_root pairs"
    )
    parser.add_argument(
        "--eval", nargs="+", required=True, help="coco.json=image_root pairs"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=float, default=12)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument(
        "--workers", type=int, default=2
    )  # each worker decodes pages; RAM is 30 GB
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    processor = AutoImageProcessor.from_pretrained(
        HERON, revision=REVISION
    )  # 640x640 stretch, as trained
    model, old = expanded_model()
    train, evaluation = load(args.train), load(args.eval)
    drift = check_preserved(
        model, old, processor, [evaluation[i]["image"].convert("RGB") for i in range(4)]
    )
    print("old-class logit drift at init", drift, flush=True)
    assert drift < 1e-4, "head expansion changed the original outputs"
    del old

    augment = A.Compose(
        [
            A.Affine(
                scale=(0.7, 1.0),
                translate_percent=(-0.05, 0.05),
                rotate=(-2, 2),
                fill=255,
                p=0.5,
            ),
            A.HorizontalFlip(p=0.5),
            A.OneOf([A.Blur(blur_limit=5), A.MotionBlur(blur_limit=5)], p=0.1),
            A.ImageCompression(quality_range=(40, 95), p=0.2),
            A.GaussNoise(p=0.1),
            A.RandomBrightnessContrast(p=0.3),
            A.ToGray(p=0.2),
        ],
        bbox_params=A.BboxParams(
            format="coco", label_fields=["category"], clip=True, min_area=4
        ),
    )
    train = train.with_transform(
        partial(
            augment_and_transform_batch, transform=augment, image_processor=processor
        )
    )
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.output,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch,
            learning_rate=args.lr,
            weight_decay=1e-4,
            warmup_steps=0.05,
            lr_scheduler_type="cosine",
            bf16=True,
            dataloader_num_workers=args.workers,
            remove_unused_columns=False,
            # In-loop COCO evaluation held every prediction in RAM and was OOM-killed; each
            # epoch's checkpoint is scored afterwards with eval_heron.py instead.
            eval_strategy="no",
            save_strategy="epoch",
            logging_steps=50,
            seed=args.seed,
            report_to=[],
        ),
        train_dataset=train,
        processing_class=processor,
        data_collator=collate_fn,
    )
    trainer.train()
    trainer.save_model(args.output + "/final")


if __name__ == "__main__":
    main()
