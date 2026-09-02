"""Train a MobileNetV3 handwriting proposal classifier in place."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

POSITIVE_STATES = frozenset({"resolved", "unreadable"})
NEGATIVE_STATE = "absent"


@dataclass(frozen=True)
class Sample:
    field_id: str
    family_id: str
    split: str
    view: str
    crop_path: Path
    label: int


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    train, dev = load_samples(args.dataset)
    summary = dataset_summary(train, dev)
    if args.validate_only:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    metrics = train_classifier(
        train,
        dev,
        args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        seed=args.seed,
        pretrained=args.pretrained,
    )
    print(json.dumps({**summary, **metrics}, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--epochs", type=_positive_int, default=12)
    parser.add_argument("--batch-size", type=_positive_int, default=32)
    parser.add_argument("--learning-rate", type=_positive_float, default=3e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser


def load_samples(root: Path) -> tuple[list[Sample], list[Sample]]:
    if not root.is_dir():
        raise FileNotFoundError(f"dataset not found: {root}")
    train = _load_split(root, "train")
    dev = _load_split(root, "dev")
    overlap = {sample.family_id for sample in train} & {
        sample.family_id for sample in dev
    }
    if overlap:
        raise ValueError("train and dev contain overlapping families")
    for split, samples in (("train", train), ("dev", dev)):
        if not samples:
            raise ValueError(f"{split} has no classifier samples")
        if {sample.label for sample in samples} != {0, 1}:
            raise ValueError(f"{split} requires positive and hard-negative samples")
    return train, dev


def dataset_summary(train: list[Sample], dev: list[Sample]) -> dict[str, int]:
    return {
        "train_samples": len(train),
        "train_families": len({sample.family_id for sample in train}),
        "train_positive": sum(sample.label for sample in train),
        "train_negative": sum(not sample.label for sample in train),
        "dev_samples": len(dev),
        "dev_families": len({sample.family_id for sample in dev}),
        "dev_positive": sum(sample.label for sample in dev),
        "dev_negative": sum(not sample.label for sample in dev),
    }


def train_classifier(
    train: list[Sample],
    dev: list[Sample],
    output: Path,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
    seed: int,
    pretrained: bool,
) -> dict[str, float | int]:
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    torch, models, transforms = _torch_stack()
    torch.manual_seed(seed)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, 1)
    model.to(device)
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomAffine(
                degrees=3, translate=(0.02, 0.02), scale=(0.9, 1.1)
            ),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        _Dataset(train, transform, torch),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    positive = sum(sample.label for sample in train)
    negative = len(train) - positive
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([negative / positive], device=device)
    )
    for _ in range(epochs):
        model.train()
        for images, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(images.to(device)).flatten()
            loss = loss_fn(logits, labels.to(device))
            loss.backward()
            optimizer.step()

    dev_loss, dev_accuracy = _evaluate(
        model,
        _Dataset(dev, eval_transform, torch),
        batch_size,
        device,
        torch,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "architecture": "torchvision/mobilenet_v3_small",
        },
        output,
    )
    return {
        "epochs": epochs,
        "dev_loss": round(dev_loss, 6),
        "dev_accuracy": round(dev_accuracy, 6),
    }


class _Dataset:
    def __init__(self, samples: list[Sample], transform: Any, torch: Any) -> None:
        self.samples = samples
        self.transform = transform
        self.torch = torch

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        sample = self.samples[index]
        with Image.open(sample.crop_path) as source:
            image = source.convert("RGB")
        return self.transform(image), self.torch.tensor(
            sample.label, dtype=self.torch.float32
        )


def _evaluate(
    model: Any,
    dataset: _Dataset,
    batch_size: int,
    device: str,
    torch: Any,
) -> tuple[float, float]:
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum")
    total_loss = 0.0
    correct = 0
    model.eval()
    with torch.inference_mode():
        for images, labels in loader:
            labels = labels.to(device)
            logits = model(images.to(device)).flatten()
            total_loss += float(loss_fn(logits, labels))
            correct += int(((logits >= 0).float() == labels).sum())
    return total_loss / len(dataset), correct / len(dataset)


def _load_split(root: Path, split: str) -> list[Sample]:
    path = root / f"{split}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"split not found: {path}")
    samples: list[Sample] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
            samples.extend(_row_samples(root, split, row, path, line_number))
    return samples


def _row_samples(
    root: Path,
    split: str,
    row: object,
    path: Path,
    line_number: int,
) -> list[Sample]:
    if not isinstance(row, dict):
        raise ValueError(f"row must be an object at {path}:{line_number}")
    required = ("field_id", "family_id", "case_id", "target_state")
    if any(not isinstance(row.get(key), str) or not row[key] for key in required):
        raise ValueError(f"row lacks identifiers at {path}:{line_number}")
    if row.get("split") != split:
        raise ValueError(f"row split mismatch at {path}:{line_number}")
    if row.get("category_id") == "C14" or row["case_id"].startswith("C14-"):
        raise ValueError(f"held-out C14 row is forbidden at {path}:{line_number}")
    state = row["target_state"]
    if state not in POSITIVE_STATES | {NEGATIVE_STATE}:
        raise ValueError(f"invalid target state at {path}:{line_number}")
    if state == NEGATIVE_STATE and not isinstance(row.get("abstention_subtype"), str):
        raise ValueError(f"hard negative lacks subtype at {path}:{line_number}")
    label = int(state in POSITIVE_STATES)
    samples = []
    for view, key in (("tight", "tight_crop_path"), ("context", "padded_crop_path")):
        value = row.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"row lacks {key} at {path}:{line_number}")
        crop_path = root / value
        if not crop_path.is_file():
            raise FileNotFoundError(f"crop not found: {crop_path}")
        samples.append(
            Sample(
                field_id=row["field_id"],
                family_id=row["family_id"],
                split=split,
                view=view,
                crop_path=crop_path,
                label=label,
            )
        )
    return samples


def _torch_stack() -> tuple[Any, Any, Any]:
    try:
        import torch
        from torchvision import models, transforms
    except ImportError as error:
        raise RuntimeError("training requires torch and torchvision") from error
    return torch, models, transforms


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
