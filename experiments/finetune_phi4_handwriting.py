"""Fine-tune Phi-4's existing vision decoder LoRA on handwriting crops."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Collection
from dataclasses import dataclass
from io import BytesIO
import json
import math
from pathlib import Path
import random
from typing import Any, Sequence
import unicodedata

from PIL import Image, ImageEnhance, ImageOps, ImageStat

from ocr_pipeline.verification import edit_counts


MODEL_ID = "microsoft/Phi-4-multimodal-instruct"
MODEL_REVISION = "93f923e1a7727d1c4f446756212d9d3e8fcc5d81"
PROMPT = "Transcribe only the handwritten text exactly. Do not correct or explain it."
IGNORE_INDEX = -100
MAX_TOKENS = 1024
MICROBATCH = 1
DEFAULT_EFFECTIVE_BATCH = 8
LORA_PARTS = ("lora_A.vision", "lora_B.vision")
LORA_TARGETS = ("qkv_proj", "o_proj", "gate_up_proj", "down_proj")
ADAPTER_FORMAT = "phi4_vision_decoder_lora_v2"
PUBLIC_OR_SYNTHETIC = frozenset({"public", "synthetic"})
NO_HANDWRITING = "<NO_HANDWRITING>"
UNREADABLE = "<UNREADABLE>"
TARGET_STATES = frozenset({"resolved", "absent", "unreadable"})
ABSTENTION_SUBTYPES = frozenset(
    {"absent", "blank", "printed_only", "stray_mark", "unreadable"}
)
HARD_STRATA = ("resolved", "blank", "printed_only", "stray_mark", "unreadable")
AUGMENTATION_POLICY = {
    "crop_views": {"tight": 0.5, "context": 0.5},
    "outcomes": {
        "identity": 0.5,
        "geometry": 0.25,
        "acquisition": 0.2,
        "geometry_acquisition": 0.05,
    },
    "geometry": {
        "rotation_degrees": [-1.5, 1.5],
        "translation_width_fraction": [-0.015, 0.015],
        "translation_height_fraction": [-0.03, 0.03],
        "isotropic_scale": [0.9, 1.05],
        "shear_degrees": [-3.0, 3.0],
    },
    "acquisition": {
        "gamma": [0.9, 1.1],
        "smooth_illumination_gain": [0.95, 1.05],
        "jpeg_quality": [75, 95],
        "downsample": [0.85, 1.0],
        "gaussian_noise_sigma": [0.003, 0.012],
    },
    "max_transforms": 2,
}


@dataclass(frozen=True)
class CropRecord:
    field_id: str
    case_id: str
    family_id: str
    reference: str
    crop_path: Path
    split: str
    data_origin: str | None
    tight_crop_path: Path | None = None
    padded_crop_path: Path | None = None
    target_state: str = "resolved"
    abstention_subtype: str | None = None
    reviewer_ids: tuple[str, ...] = ()
    split_group_id: str | None = None

    def __post_init__(self) -> None:
        if self.tight_crop_path is None:
            object.__setattr__(self, "tight_crop_path", self.crop_path)
        if self.padded_crop_path is None:
            object.__setattr__(self, "padded_crop_path", self.crop_path)
        if self.split_group_id is None:
            object.__setattr__(self, "split_group_id", self.family_id)


class CropDataset:
    def __init__(
        self,
        records: list[CropRecord],
        processor: Any,
        *,
        training: bool = False,
        seed: int = 17,
    ) -> None:
        self.records = records
        self.processor = processor
        self.training = training
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image = prepare_image(record, training=self.training, rng=self.rng)
        return encode_record(record, self.processor, image=image)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_runtime_choice(args.model, args.device)
        train, dev = load_splits(
            args.dataset,
            mode=args.mode,
            review_decisions=args.review_decision,
        )
        if args.validate_only:
            scheduled_train = materialize_training_schedule(
                train,
                hard_replay_ratio=args.hard_replay_ratio,
                hard_stratum_weights=args.hard_stratum_weights,
                epochs=args.epochs,
                seed=args.seed,
            )
            summary = dataset_summary(train, dev, args.mode)
            if (
                args.hard_replay_ratio is not None
                and args.hard_stratum_weights is not None
            ):
                summary["training_schedule"] = training_schedule_summary(
                    scheduled_train,
                    hard_replay_ratio=args.hard_replay_ratio,
                    hard_stratum_weights=args.hard_stratum_weights,
                )
            print(json.dumps(summary, sort_keys=True))
            return 0
        run_canary(
            train,
            dev,
            args.output,
            model_name=args.model,
            device=args.device,
            effective_batch=args.effective_batch,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            mode=args.mode,
            seed=args.seed,
            local_files_only=not args.allow_download,
            hard_replay_ratio=args.hard_replay_ratio,
            hard_stratum_weights=args.hard_stratum_weights,
        )
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--mode", choices=("canary", "infrastructure"), default="canary"
    )
    parser.add_argument("--effective-batch", type=_positive_int, default=8)
    parser.add_argument("--epochs", type=_positive_int, default=3)
    parser.add_argument("--learning-rate", type=_positive_float, default=5e-5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--hard-replay-ratio", type=_ratio)
    parser.add_argument("--hard-stratum-weights", type=_hard_weights)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument(
        "--review-decision",
        type=Path,
        action="append",
        default=[],
        help="review JSON; repeat for independent decisions",
    )
    parser.add_argument("--validate-only", action="store_true")
    return parser


def load_splits(
    root: Path,
    *,
    mode: str = "canary",
    review_decisions: Sequence[Path] = (),
) -> tuple[list[CropRecord], list[CropRecord]]:
    if mode not in {"canary", "infrastructure"}:
        raise ValueError(f"unsupported mode: {mode}")
    if review_decisions and len(review_decisions) < 2:
        raise ValueError("canary mode requires two agreeing review decision files")
    review_files = {path.resolve() for path in review_decisions}
    if len(review_files) != len(review_decisions):
        raise ValueError("review decision files must be distinct")
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root was not found: {root}")
    train = _load_split(root, "train", mode)
    dev = _load_split(root, "dev", mode)
    overlap = {row.family_id for row in train} & {row.family_id for row in dev}
    if overlap:
        raise ValueError("train and dev contain overlapping document families")
    _validate_split_groups(train + dev)
    if review_decisions:
        accepted = reviewed_field_ids(root, train + dev, review_decisions)
        train = [row for row in train if row.field_id in accepted]
        dev = [row for row in dev if row.field_id in accepted]
        if not train or not dev:
            raise ValueError("review decisions leave an empty train or dev split")
    elif mode == "canary" and any(
        row.data_origin not in PUBLIC_OR_SYNTHETIC and len(row.reviewer_ids) != 2
        for row in train + dev
    ):
        raise ValueError(
            "canary mode requires two agreeing review decision files or embedded independent reviews"
        )
    return train, dev


def reviewed_field_ids(
    root: Path, records: list[CropRecord], review_paths: Sequence[Path]
) -> set[str]:
    manifest_ids = {row.field_id for row in records}
    if len(manifest_ids) != len(records):
        raise ValueError("reviewed manifest contains duplicate field ids")

    agreed: set[str] | None = None
    reviewers: set[str] = set()
    for path in review_paths:
        decision = _read_review(path)
        reviewer = _reviewer_id(decision.get("reviewer_id"), str(path))
        if reviewer in reviewers:
            raise ValueError(
                "review decision files must have distinct reviewer identities"
            )
        reviewers.add(reviewer)
        if decision.get("reviewed_crop_set") != root.name:
            raise ValueError(f"review decision targets a different crop set: {path}")
        if decision.get("reviewed_field_count") != len(manifest_ids):
            raise ValueError(f"reviewed count does not match the manifest: {path}")
        rejected = _decision_ids(decision.get("rejected"), "rejected", path)
        if not rejected <= manifest_ids:
            raise ValueError(f"review identities do not match the manifest: {path}")
        uncertain_count = decision.get("uncertain_field_count")
        if uncertain_count != 0:
            raise ValueError(
                f"review decision must explicitly report zero uncertain fields: {path}"
            )

        accepted_value = decision.get("accepted")
        if accepted_value is None:
            accepted = manifest_ids - rejected
        else:
            accepted = _decision_ids(accepted_value, "accepted", path)
            if accepted | rejected != manifest_ids or accepted & rejected:
                raise ValueError(f"review identities do not match the manifest: {path}")
        if decision.get("accepted_field_count") != len(accepted):
            raise ValueError(f"accepted count does not match the manifest: {path}")
        if agreed is not None and accepted != agreed:
            raise ValueError("review decisions disagree on accepted field identities")
        agreed = accepted
    return agreed or set()


def dataset_summary(
    train: list[CropRecord], dev: list[CropRecord], mode: str
) -> dict[str, Any]:
    return {
        "mode": mode,
        "train_fields": len(train),
        "dev_fields": len(dev),
        "train_families": len({row.family_id for row in train}),
        "dev_families": len({row.family_id for row in dev}),
        "c14_fields": 0,
    }


def materialize_training_schedule(
    records: list[CropRecord],
    *,
    hard_replay_ratio: tuple[int, int] | None = None,
    hard_stratum_weights: dict[str, int] | None = None,
    epochs: int = 1,
    seed: int = 17,
) -> list[CropRecord]:
    if hard_replay_ratio is None and hard_stratum_weights is None:
        return records
    if hard_replay_ratio is None or hard_stratum_weights is None:
        raise ValueError(
            "hard replay ratio and hard stratum weights must be supplied together"
        )
    if not records:
        raise ValueError("cannot schedule an empty training split")
    if epochs <= 0:
        raise ValueError("training schedule epochs must be positive")
    if set(hard_stratum_weights) != set(HARD_STRATA):
        raise ValueError("hard stratum weights must name exactly the supported strata")
    if any(
        weight <= 0 for weight in (*hard_replay_ratio, *hard_stratum_weights.values())
    ):
        raise ValueError("training schedule weights must be positive")
    _validate_split_groups(records)

    hard = [row for row in records if row.data_origin not in PUBLIC_OR_SYNTHETIC]
    replay = [row for row in records if row.data_origin in PUBLIC_OR_SYNTHETIC]
    source_weights = {
        name: weight
        for name, weight, pool in (
            ("hard", hard_replay_ratio[0], hard),
            ("replay", hard_replay_ratio[1], replay),
        )
        if pool
    }
    schedule_length = len(records) * epochs
    source_counts = _weighted_counts(
        schedule_length,
        source_weights,
        {"hard": len(hard), "replay": len(replay)},
    )
    rng = random.Random(seed)
    scheduled = {
        "hard": _hard_schedule(
            hard, source_counts.get("hard", 0), hard_stratum_weights, rng
        ),
        "replay": _count_balanced_schedule(replay, source_counts.get("replay", 0), rng),
    }
    # Trainer randomizes dataset indices, so the schedule controls counts, not batches.
    return scheduled["hard"] + scheduled["replay"]


def training_schedule_summary(
    records: list[CropRecord],
    *,
    hard_replay_ratio: tuple[int, int],
    hard_stratum_weights: dict[str, int],
) -> dict[str, Any]:
    hard = [row for row in records if row.data_origin not in PUBLIC_OR_SYNTHETIC]
    return {
        "fields": len(records),
        "unique_fields": len({row.field_id for row in records}),
        "hard_fields": len(hard),
        "replay_fields": len(records) - len(hard),
        "hard_replay_ratio": {
            "hard": hard_replay_ratio[0],
            "replay": hard_replay_ratio[1],
        },
        "hard_stratum_weights": dict(sorted(hard_stratum_weights.items())),
        "hard_strata": dict(
            sorted(Counter(_hard_stratum(row) for row in hard).items())
        ),
    }


def _hard_schedule(
    records: list[CropRecord],
    count: int,
    weights: dict[str, int],
    rng: random.Random,
) -> list[CropRecord]:
    by_stratum = {
        stratum: [row for row in records if _hard_stratum(row) == stratum]
        for stratum in HARD_STRATA
    }
    available_weights = {
        stratum: weights[stratum] for stratum in HARD_STRATA if by_stratum[stratum]
    }
    counts = _weighted_counts(
        count,
        available_weights,
        {stratum: len(rows) for stratum, rows in by_stratum.items() if rows},
    )
    result = [
        record
        for stratum, amount in counts.items()
        for record in _count_balanced_schedule(by_stratum[stratum], amount, rng)
    ]
    return result


def _hard_stratum(record: CropRecord) -> str:
    if record.target_state == "resolved":
        return "resolved"
    if record.target_state == "unreadable":
        return "unreadable"
    if record.target_state == "absent" and record.abstention_subtype in HARD_STRATA:
        return str(record.abstention_subtype)
    if record.target_state == "absent":
        raise ValueError(
            "scheduled absent row must specify blank, printed_only, or stray_mark: "
            f"{record.field_id}"
        )
    raise ValueError(f"hard row has no supported stratum: {record.field_id}")


def _count_balanced_schedule(
    records: list[CropRecord], count: int, rng: random.Random
) -> list[CropRecord]:
    if not count:
        return []
    if count < len(records):
        raise ValueError("schedule cannot cover every accepted field")
    rows_by_family_group: dict[str, dict[str, list[CropRecord]]] = {}
    for row in records:
        family_groups = rows_by_family_group.setdefault(row.family_id, {})
        family_groups.setdefault(str(row.split_group_id), []).append(row)
    families = sorted(rows_by_family_group)
    rng.shuffle(families)
    groups_by_family = {
        family: sorted(rows_by_family_group[family]) for family in families
    }
    for groups in groups_by_family.values():
        rng.shuffle(groups)
    for family_groups in rows_by_family_group.values():
        for rows in family_groups.values():
            rng.shuffle(rows)

    result = list(records)
    family_counts = Counter(row.family_id for row in result)
    group_counts = Counter(str(row.split_group_id) for row in result)
    row_counts = Counter(row.field_id for row in result)
    for _ in range(count - len(records)):
        family = min(families, key=family_counts.__getitem__)
        group = min(groups_by_family[family], key=group_counts.__getitem__)
        row = min(
            rows_by_family_group[family][group],
            key=lambda item: row_counts[item.field_id],
        )
        result.append(row)
        family_counts[family] += 1
        group_counts[group] += 1
        row_counts[row.field_id] += 1
    return result


def _weighted_counts(
    total: int,
    weights: dict[str, int],
    minimums: dict[str, int],
) -> dict[str, int]:
    if not weights:
        return {}
    weight_sum = sum(weights.values())
    counts = {name: total * weight // weight_sum for name, weight in weights.items()}
    remainder = total - sum(counts.values())
    priority = sorted(
        weights,
        key=lambda name: (-(total * weights[name] % weight_sum), name),
    )
    for name in priority[:remainder]:
        counts[name] += 1
    unavailable = [
        name
        for name, minimum in minimums.items()
        if minimum and counts.get(name, 0) < minimum
    ]
    if unavailable:
        raise ValueError(
            "configured run cannot cover every accepted field within one-record "
            "rounding tolerance"
        )
    return counts


def run_canary(
    train: list[CropRecord],
    dev: list[CropRecord],
    output: Path,
    *,
    model_name: str = MODEL_ID,
    device: str = "cuda:0",
    effective_batch: int = DEFAULT_EFFECTIVE_BATCH,
    epochs: int = 3,
    learning_rate: float = 5e-5,
    mode: str = "canary",
    seed: int = 17,
    local_files_only: bool = True,
    hard_replay_ratio: tuple[int, int] | None = None,
    hard_stratum_weights: dict[str, int] | None = None,
) -> dict[str, Any]:
    validate_runtime_choice(model_name, device)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if effective_batch < MICROBATCH:
        raise ValueError("effective batch must be positive")
    if mode == "infrastructure" and not all(
        row.data_origin in PUBLIC_OR_SYNTHETIC for row in train + dev
    ):
        raise ValueError("infrastructure mode accepts only public or synthetic rows")

    scheduled_train = materialize_training_schedule(
        train,
        hard_replay_ratio=hard_replay_ratio,
        hard_stratum_weights=hard_stratum_weights,
        epochs=epochs,
        seed=seed,
    )

    torch, processor, model, trainer_types = load_runtime(
        model_name,
        device=device,
        seed=seed,
        local_files_only=local_files_only,
    )
    audit = enable_vision_decoder_lora(model)
    gradient_steps = effective_batch // MICROBATCH
    if gradient_steps * MICROBATCH != effective_batch:
        raise ValueError("effective batch must be divisible by the microbatch")

    output.mkdir(parents=True)
    before = predict_records(model, processor, dev, device=device, torch_module=torch)
    training_args = trainer_types[1](
        output_dir=str(output / "trainer"),
        per_device_train_batch_size=MICROBATCH,
        per_device_eval_batch_size=MICROBATCH,
        gradient_accumulation_steps=gradient_steps,
        num_train_epochs=1 if hard_replay_ratio is not None else epochs,
        max_steps=2 if mode == "infrastructure" else -1,
        learning_rate=learning_rate,
        warmup_ratio=0.03,
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.95,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        save_strategy="no",
        report_to="none",
        seed=seed,
        data_seed=seed,
    )
    trainer = trainer_types[0](
        model=model,
        args=training_args,
        train_dataset=CropDataset(scheduled_train, processor, training=True, seed=seed),
        data_collator=lambda batch: collate(batch, processor.tokenizer.pad_token_id),
    )
    trainer.train()
    after = predict_records(model, processor, dev, device=device, torch_module=torch)

    adapter_path = output / "vision_decoder_lora.pt"
    save_trainable_state(
        model,
        adapter_path,
        torch,
        training_family_ids={record.family_id for record in scheduled_train},
    )
    del trainer
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _, reload_processor, reloaded, _ = load_runtime(
        model_name,
        device=device,
        seed=seed,
        local_files_only=local_files_only,
    )
    enable_vision_decoder_lora(reloaded)
    load_trainable_state(reloaded, adapter_path, torch)
    reloaded_predictions = predict_records(
        reloaded, reload_processor, dev, device=device, torch_module=torch
    )
    if _prediction_text(after) != _prediction_text(reloaded_predictions):
        raise RuntimeError("saved adapter changed predictions after reload")

    before_metrics = validation_metrics(dev, before)
    after_metrics = validation_metrics(dev, after)
    result = {
        "status": "complete",
        "privacy": {
            "private_text_persisted": False,
            "prediction_text_persisted": False,
            "exception_messages_persisted": False,
            "case_identifiers_persisted": False,
        },
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "pinned": True,
        },
        "runtime": {
            "dtype": "bfloat16",
            "attention": "sdpa",
            "quantization": "none",
            "dynamic_hd": 1,
            "max_tokens": MAX_TOKENS,
            "microbatch": MICROBATCH,
            "effective_batch": effective_batch,
            "gradient_accumulation_steps": gradient_steps,
            "mode": mode,
            "seed": seed,
            "epochs": epochs,
            "trainer_epochs": 1 if hard_replay_ratio is not None else epochs,
            "max_steps": 2 if mode == "infrastructure" else None,
        },
        "dataset": dataset_summary(train, dev, mode),
        "augmentation": AUGMENTATION_POLICY,
        "trainable_audit": audit,
        "validation": {
            "before": before_metrics,
            "after": after_metrics,
        },
        "cases": safe_case_metadata(dev, before, after),
        "reload_prediction_parity": True,
    }
    if hard_replay_ratio is not None and hard_stratum_weights is not None:
        result["training_schedule"] = training_schedule_summary(
            scheduled_train,
            hard_replay_ratio=hard_replay_ratio,
            hard_stratum_weights=hard_stratum_weights,
        )
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def load_runtime(
    model_name: str,
    *,
    device: str,
    seed: int,
    local_files_only: bool,
) -> tuple[Any, Any, Any, tuple[Any, Any]]:
    validate_runtime_choice(model_name, device)
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoProcessor,
        Trainer,
        TrainingArguments,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("Phi-4 BF16 fine-tuning requires a CUDA GPU")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    options = {
        "revision": MODEL_REVISION,
        "trust_remote_code": True,
        "local_files_only": local_files_only,
    }
    processor = AutoProcessor.from_pretrained(model_name, dynamic_hd=1, **options)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        _attn_implementation="sdpa",
        **options,
    ).to(device)
    model.config.use_cache = False
    return torch, processor, model, (Trainer, TrainingArguments)


def validate_runtime_choice(model_name: str, device: str) -> None:
    if model_name != MODEL_ID:
        raise ValueError(f"model must be the pinned checkpoint: {MODEL_ID}")
    if device != "cuda:0":
        raise ValueError(
            "device must be cuda:0; select the physical GPU with CUDA_VISIBLE_DEVICES"
        )


def enable_vision_decoder_lora(model: Any) -> dict[str, Any]:
    model.set_lora_adapter("vision")
    for _, parameter in model.named_parameters():
        parameter.requires_grad = False
    for name, parameter in model.named_parameters():
        if any(part in name for part in LORA_PARTS):
            parameter.requires_grad = True

    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable:
        raise RuntimeError("Phi-4 exposes no existing vision decoder LoRA parameters")
    invalid = [
        name
        for name, _ in trainable
        if not any(part in name for part in LORA_PARTS)
        or not any(target in name for target in LORA_TARGETS)
    ]
    if invalid:
        raise RuntimeError(f"unexpected trainable parameters: {invalid}")
    return {
        "trainable_names": [name for name, _ in trainable],
        "trainable_parameters": sum(parameter.numel() for _, parameter in trainable),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "frozen_components": [
            "vision_encoder",
            "vision_projector",
            "audio",
            "decoder_base",
            "embeddings",
            "lm_head",
        ],
    }


def encode_record(
    record: CropRecord, processor: Any, *, image: Image.Image | None = None
) -> dict[str, Any]:
    import torch

    prompt = inference_prompt(processor)
    answer = f"{target_text(record)}<|end|><|endoftext|>"
    if image is None:
        with Image.open(record.crop_path) as source:
            image = source.convert("RGB")
    inputs = processor(prompt, images=[image.convert("RGB")], return_tensors="pt")
    answer_ids = processor.tokenizer(answer, return_tensors="pt").input_ids
    input_ids = torch.cat([inputs.input_ids, answer_ids], dim=1)
    if input_ids.size(1) > MAX_TOKENS:
        raise ValueError(
            f"training example exceeds {MAX_TOKENS} tokens: {record.field_id}"
        )
    labels = torch.tensor(
        assistant_labels(
            inputs.input_ids.size(1), answer_ids[0].tolist(), max_tokens=MAX_TOKENS
        ),
        dtype=input_ids.dtype,
    ).unsqueeze(0)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "input_image_embeds": inputs.input_image_embeds,
        "image_attention_mask": inputs.image_attention_mask,
        "image_sizes": inputs.image_sizes,
    }


def collate(batch: list[dict[str, Any]], pad_token_id: int | None) -> dict[str, Any]:
    import torch
    from torch.nn.utils.rnn import pad_sequence

    if pad_token_id is None:
        raise ValueError("Phi-4 tokenizer has no padding token")
    input_ids = pad_sequence(
        [item["input_ids"][0] for item in batch],
        batch_first=True,
        padding_value=pad_token_id,
    )
    labels = pad_sequence(
        [item["labels"][0] for item in batch],
        batch_first=True,
        padding_value=IGNORE_INDEX,
    )
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": input_ids.ne(pad_token_id).long(),
        "input_image_embeds": _cat_with_pad(
            [item["input_image_embeds"] for item in batch], dim=0
        ),
        "image_attention_mask": _cat_with_pad(
            [item["image_attention_mask"] for item in batch], dim=0
        ),
        "image_sizes": torch.cat([item["image_sizes"] for item in batch]),
        "input_mode": 1,
    }


def predict_records(
    model: Any,
    processor: Any,
    records: list[CropRecord],
    *,
    device: str,
    torch_module: Any,
) -> list[dict[str, Any]]:
    model.eval()
    predictions = []
    for record in records:
        prompt = inference_prompt(processor)
        with Image.open(record.crop_path) as image:
            inputs = processor(
                prompt, images=[image.convert("RGB")], return_tensors="pt"
            ).to(device)
        prompt_tokens = inputs.input_ids.size(1)
        generation_inputs = dict(inputs)
        generation_inputs.setdefault("input_mode", 1)
        with torch_module.inference_mode():
            output = model.generate(
                **generation_inputs,
                max_new_tokens=128,
                do_sample=False,
                num_beams=1,
            )
        prediction = processor.decode(
            output[0][prompt_tokens:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        predictions.append(
            {
                "field_id": record.field_id,
                "reference": target_text(record),
                "prediction": prediction,
                "exact": prediction == target_text(record),
            }
        )
    model.train()
    return predictions


def inference_prompt(processor: Any) -> str:
    message = {"role": "user", "content": f"<|image_1|>{PROMPT}"}
    return processor.tokenizer.apply_chat_template(
        [message], tokenize=False, add_generation_prompt=True
    )


def validation_metrics(
    records: list[CropRecord], predictions: list[dict[str, Any]]
) -> dict[str, Any]:
    record_ids = [record.field_id for record in records]
    prediction_ids = [row.get("field_id") for row in predictions]
    if len(set(record_ids)) != len(record_ids) or len(set(prediction_ids)) != len(
        prediction_ids
    ):
        raise ValueError("validation contains duplicate field ids")
    if set(record_ids) != set(prediction_ids):
        raise ValueError("validation predictions do not match the dev fields")

    by_id = {row["field_id"]: row for row in predictions}
    resolved = [record for record in records if record.target_state == "resolved"]
    edits = {"insertions": 0, "deletions": 0, "substitutions": 0}
    normalized_exact = 0
    reference_characters = 0
    for record in resolved:
        reference = normalize_text(record.reference)
        prediction = normalize_text(str(by_id[record.field_id]["prediction"]))
        normalized_exact += reference == prediction
        counts = edit_counts(prediction, reference)
        edits["insertions"] += counts.insertions
        edits["deletions"] += counts.deletions
        edits["substitutions"] += counts.substitutions
        reference_characters += len(reference)

    abstentions = [record for record in records if record.target_state != "resolved"]
    subtype_metrics = {}
    abstention_correct = 0
    for subtype in sorted({_abstention_subtype(record) for record in abstentions}):
        rows = [
            record for record in abstentions if _abstention_subtype(record) == subtype
        ]
        correct = sum(
            normalize_text(str(by_id[record.field_id]["prediction"]))
            == normalize_text(target_text(record))
            for record in rows
        )
        abstention_correct += correct
        subtype_metrics[subtype] = {
            "fields": len(rows),
            "correct": correct,
            "accuracy": _rate(correct, len(rows)),
        }

    abstention_fields = len(abstentions)
    total_edits = sum(edits.values())
    return {
        "fields": len(records),
        "resolved_fields": len(resolved),
        "normalized_exact": normalized_exact,
        "normalized_exact_rate": _rate(normalized_exact, len(resolved)),
        "reference_characters": reference_characters,
        "character_edit_counts": edits,
        "cer": _rate(total_edits, reference_characters),
        "missed_character_rate": _rate(edits["deletions"], reference_characters),
        "hallucinated_character_rate": _rate(edits["insertions"], reference_characters),
        "substitution_rate": _rate(edits["substitutions"], reference_characters),
        "abstention_fields": abstention_fields,
        "abstention_accuracy": _rate(abstention_correct, abstention_fields),
        "abstention_subtype_accuracy": subtype_metrics,
    }


def safe_case_metadata(
    records: list[CropRecord],
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    before_by_id = {row["field_id"]: row for row in before}
    after_by_id = {row["field_id"]: row for row in after}
    return [
        {
            "index": index,
            "split": record.split
            if record.split in {"train", "dev", "test"}
            else "unknown",
            "data_origin": (
                record.data_origin
                if record.data_origin in PUBLIC_OR_SYNTHETIC | {"private"}
                else "unknown"
            ),
            "target_state": (
                record.target_state
                if record.target_state in TARGET_STATES
                else "unknown"
            ),
            "abstention_subtype": (
                record.abstention_subtype
                if record.abstention_subtype in ABSTENTION_SUBTYPES
                else None
            ),
            "before_exact": normalize_text(
                str(before_by_id[record.field_id]["prediction"])
            )
            == normalize_text(target_text(record)),
            "after_exact": normalize_text(
                str(after_by_id[record.field_id]["prediction"])
            )
            == normalize_text(target_text(record)),
        }
        for index, record in enumerate(records)
    ]


def save_trainable_state(
    model: Any,
    path: Path,
    torch_module: Any,
    *,
    training_family_ids: Collection[str],
) -> None:
    state = {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if not state:
        raise RuntimeError("refusing to save an empty adapter")
    family_ids = sorted({_normalize_family_id(value) for value in training_family_ids})
    if not family_ids:
        raise ValueError("adapter provenance requires at least one training family")
    torch_module.save(
        {
            "format": ADAPTER_FORMAT,
            "state": state,
            "provenance": {
                "training_family_count": len(family_ids),
                "training_family_ids": family_ids,
            },
        },
        path,
    )


def load_trainable_state(
    model: Any,
    path: Path,
    torch_module: Any,
    *,
    evaluation_family_ids: Collection[str] = (),
) -> None:
    payload = torch_module.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("unsupported adapter format")
    if payload.get("format") == "phi4_vision_decoder_lora_v1":
        raise ValueError("adapter lacks training-family provenance")
    if payload.get("format") != ADAPTER_FORMAT:
        raise ValueError("unsupported adapter format")
    training_family_ids = _training_family_ids(payload.get("provenance"))
    normalized_evaluation_ids = {
        _normalize_family_id(value) for value in evaluation_family_ids
    }
    if training_family_ids & normalized_evaluation_ids:
        raise ValueError("adapter training families overlap evaluation families")
    state = payload.get("state")
    expected = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    if not isinstance(state, dict) or set(state) != expected:
        raise ValueError("saved adapter does not match the trainable parameter audit")
    load_result = model.load_state_dict(state, strict=False)
    if load_result.unexpected_keys:
        raise ValueError(
            f"adapter has unexpected parameters: {load_result.unexpected_keys}"
        )


def _training_family_ids(value: object) -> set[str]:
    if not isinstance(value, dict):
        raise ValueError("adapter lacks training-family provenance")
    family_ids = value.get("training_family_ids")
    count = value.get("training_family_count")
    if (
        not isinstance(family_ids, list)
        or not family_ids
        or not all(
            isinstance(item, str) and bool(item) and item == _normalize_family_id(item)
            for item in family_ids
        )
        or len(set(family_ids)) != len(family_ids)
        or count != len(family_ids)
    ):
        raise ValueError("adapter has invalid training-family provenance")
    return set(family_ids)


def _normalize_family_id(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("family id must be non-empty")
    return unicodedata.normalize("NFKC", value).strip().casefold()


def _read_review(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid review decision: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"review decision must be an object: {path}")
    return value


def _decision_ids(value: object, name: str, path: Path) -> set[str]:
    if not isinstance(value, list):
        raise ValueError(f"review decision lacks {name} identities: {path}")
    identities = []
    for row in value:
        field_id = row.get("field_id") if isinstance(row, dict) else row
        if not isinstance(field_id, str) or not field_id:
            raise ValueError(f"review decision has an invalid {name} identity: {path}")
        identities.append(field_id)
    if len(set(identities)) != len(identities):
        raise ValueError(f"review decision repeats a {name} identity: {path}")
    return set(identities)


def _load_split(root: Path, split: str, mode: str) -> list[CropRecord]:
    path = root / f"{split}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"split was not found: {path}")
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
            rows.append(_record(value, root, split, mode, path, line_number))
    if not rows:
        raise ValueError(f"split is empty: {path}")
    if len({row.field_id for row in rows}) != len(rows):
        raise ValueError(f"split contains duplicate field ids: {path}")
    return rows


def _validate_split_groups(records: list[CropRecord]) -> None:
    group_lineage: dict[str, tuple[str, str]] = {}
    for record in records:
        group = str(record.split_group_id)
        lineage = (record.family_id, record.split)
        previous = group_lineage.setdefault(group, lineage)
        if previous[0] != lineage[0]:
            raise ValueError("split group belongs to multiple document families")
        if previous[1] != lineage[1]:
            raise ValueError("split group belongs to multiple splits")


def _record(
    value: object,
    root: Path,
    split: str,
    mode: str,
    path: Path,
    line_number: int,
) -> CropRecord:
    if not isinstance(value, dict):
        raise ValueError(f"row must be an object at {path}:{line_number}")
    required = ("field_id", "case_id", "family_id", "split")
    if not all(
        isinstance(value.get(key), str) and value[key].strip() for key in required
    ):
        raise ValueError(f"row is missing required text at {path}:{line_number}")
    split_group_id = value.get("split_group_id", value["family_id"])
    if not isinstance(split_group_id, str) or not split_group_id.strip():
        raise ValueError(f"row has an invalid split group at {path}:{line_number}")
    identifiers = [value[key] for key in ("field_id", "case_id", "family_id")] + [
        split_group_id
    ]
    category = value.get("category_id")
    if (isinstance(category, str) and category.upper() == "C14") or any(
        identifier.upper() == "C14" or identifier.upper().startswith("C14-")
        for identifier in identifiers
    ):
        raise ValueError(f"held-out C14 data is forbidden at {path}:{line_number}")
    if value["split"] != split:
        raise ValueError(f"row has the wrong split at {path}:{line_number}")
    origin = value.get("data_origin")
    if origin is not None and origin not in PUBLIC_OR_SYNTHETIC | {"private"}:
        raise ValueError(f"invalid data origin at {path}:{line_number}")
    if mode == "infrastructure" and origin not in PUBLIC_OR_SYNTHETIC:
        raise ValueError(
            f"infrastructure row is not public or synthetic at {path}:{line_number}"
        )
    target_state = value.get("target_state", "resolved")
    if target_state not in TARGET_STATES:
        raise ValueError(f"row has an invalid target state at {path}:{line_number}")
    abstention_subtype = value.get("abstention_subtype")
    if abstention_subtype is not None and abstention_subtype not in ABSTENTION_SUBTYPES:
        raise ValueError(
            f"row has an invalid abstention subtype at {path}:{line_number}"
        )
    if target_state == "resolved" and abstention_subtype is not None:
        raise ValueError(
            f"resolved row cannot have an abstention subtype at {path}:{line_number}"
        )
    if target_state == "unreadable" and abstention_subtype not in {None, "unreadable"}:
        raise ValueError(
            f"unreadable row has the wrong abstention subtype at {path}:{line_number}"
        )
    if target_state == "absent" and abstention_subtype == "unreadable":
        raise ValueError(
            f"absent row has the wrong abstention subtype at {path}:{line_number}"
        )
    reference = value.get("reference", "")
    if not isinstance(reference, str):
        raise ValueError(f"row has an invalid reference at {path}:{line_number}")
    if target_state == "resolved" and not reference.strip():
        raise ValueError(f"resolved row lacks a literal at {path}:{line_number}")
    reviewer_ids = _embedded_reviewers(
        value,
        target_state,
        abstention_subtype,
        reference,
        path,
        line_number,
    )

    legacy_crop = value.get("crop_path")
    tight_value = value.get("tight_crop_path") or legacy_crop
    padded_value = value.get("padded_crop_path") or legacy_crop or tight_value
    tight_crop_path = _crop_path(tight_value, root, path, line_number, "tight")
    padded_crop_path = _crop_path(padded_value, root, path, line_number, "padded")
    return CropRecord(
        field_id=value["field_id"],
        case_id=value["case_id"],
        family_id=value["family_id"],
        reference=reference,
        crop_path=padded_crop_path,
        split=split,
        data_origin=origin,
        tight_crop_path=tight_crop_path,
        padded_crop_path=padded_crop_path,
        target_state=target_state,
        abstention_subtype=abstention_subtype,
        reviewer_ids=reviewer_ids,
        split_group_id=split_group_id,
    )


def _embedded_reviewers(
    value: dict[str, Any],
    target_state: str,
    abstention_subtype: object,
    reference: str,
    path: Path,
    line_number: int,
) -> tuple[str, ...]:
    reviewer_values = value.get("reviewer_ids")
    review_values = value.get("independent_reviews")
    if reviewer_values is None and review_values is None:
        return ()
    location = f"{path}:{line_number}"
    if (
        not isinstance(reviewer_values, list)
        or not isinstance(review_values, list)
        or len(reviewer_values) != 2
        or len(review_values) != 2
        or not all(isinstance(review, dict) for review in review_values)
    ):
        raise ValueError(f"invalid embedded review evidence at {location}")
    reviewers = tuple(sorted(_reviewer_id(item, location) for item in reviewer_values))
    evidence_reviewers = tuple(
        sorted(
            _reviewer_id(review.get("reviewer_id"), location)
            for review in review_values
        )
    )
    if len(set(reviewers)) != 2 or evidence_reviewers != reviewers:
        raise ValueError(f"embedded reviews are not independent at {location}")
    for review in review_values:
        transcription = review.get("transcription")
        legibility = review.get("legibility")
        if not isinstance(transcription, str):
            raise ValueError(f"invalid embedded review transcription at {location}")
        if target_state == "resolved" and (
            legibility != "legible"
            or normalize_text(transcription) != normalize_text(reference)
        ):
            raise ValueError(
                f"embedded reviewers disagree on the literal at {location}"
            )
        if target_state == "absent" and (
            transcription.strip() or review.get("region_type") != abstention_subtype
        ):
            raise ValueError(f"embedded negative review is inconsistent at {location}")
        if target_state == "unreadable" and (
            transcription.strip() or legibility != "unreadable"
        ):
            raise ValueError(
                f"embedded unreadable review is inconsistent at {location}"
            )
    return reviewers


def _reviewer_id(value: object, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"invalid embedded reviewer at {location}")
    return value.strip().casefold()


def target_text(record: CropRecord) -> str:
    if record.target_state == "resolved":
        if not record.reference.strip():
            raise ValueError(f"resolved row lacks a literal: {record.field_id}")
        return record.reference
    if record.target_state == "absent":
        return NO_HANDWRITING
    if record.target_state == "unreadable":
        return UNREADABLE
    raise ValueError(
        f"invalid target state for {record.field_id}: {record.target_state}"
    )


def _abstention_subtype(record: CropRecord) -> str:
    return record.abstention_subtype or record.target_state


def prepare_image(
    record: CropRecord, *, training: bool, rng: random.Random
) -> Image.Image:
    crop_path = record.crop_path
    if training:
        if rng.random() < 0.5:
            crop_path = record.tight_crop_path or record.crop_path
        else:
            crop_path = record.padded_crop_path or record.crop_path
    with Image.open(crop_path) as source:
        image = source.convert("RGB")
    return augment_image(image, rng) if training else image


def augmentation_kind(rng: random.Random) -> str:
    draw = rng.random()
    if draw < 0.5:
        return "identity"
    if draw < 0.75:
        return "geometry"
    if draw < 0.95:
        return "acquisition"
    return "geometry_acquisition"


def augment_image(image: Image.Image, rng: random.Random) -> Image.Image:
    kind = augmentation_kind(rng)
    if kind == "identity":
        return image.copy()
    result = image
    if kind in {"geometry", "geometry_acquisition"}:
        result = _geometry(result, rng)
    if kind in {"acquisition", "geometry_acquisition"}:
        result = _acquisition(result, rng)
    return result


def _geometry(image: Image.Image, rng: random.Random) -> Image.Image:
    width, height = image.size
    fill = tuple(round(value) for value in ImageStat.Stat(image).median)
    pad = max(2, math.ceil(height * 0.25), math.ceil(max(width, height) * 0.1))
    padded = ImageOps.expand(image, pad, fill=fill)
    transform = int(rng.random() * 4)
    if transform == 0:
        padded = padded.rotate(
            rng.uniform(-1.5, 1.5),
            resample=Image.Resampling.BICUBIC,
            expand=True,
            fillcolor=fill,
        )
    elif transform == 1:
        x_shift = rng.uniform(-0.015, 0.015) * width
        y_shift = rng.uniform(-0.03, 0.03) * height
        padded = padded.transform(
            padded.size,
            Image.Transform.AFFINE,
            (1, 0, -x_shift, 0, 1, -y_shift),
            resample=Image.Resampling.BICUBIC,
            fillcolor=fill,
        )
    elif transform == 2:
        scale = rng.uniform(0.9, 1.05)
        inverse = 1 / scale
        center_x, center_y = padded.width / 2, padded.height / 2
        padded = padded.transform(
            padded.size,
            Image.Transform.AFFINE,
            (
                inverse,
                0,
                center_x * (1 - inverse),
                0,
                inverse,
                center_y * (1 - inverse),
            ),
            resample=Image.Resampling.BICUBIC,
            fillcolor=fill,
        )
    else:
        shear = math.tan(math.radians(rng.uniform(-3.0, 3.0)))
        padded = padded.transform(
            padded.size,
            Image.Transform.AFFINE,
            (1, -shear, shear * padded.height / 2, 0, 1, 0),
            resample=Image.Resampling.BICUBIC,
            fillcolor=fill,
        )
    return padded


def _acquisition(image: Image.Image, rng: random.Random) -> Image.Image:
    transform = int(rng.random() * 5)
    if transform == 0:
        gamma = rng.uniform(0.9, 1.1)
        table = [round(255 * ((value / 255) ** gamma)) for value in range(256)]
        return image.point(table * len(image.getbands()))
    if transform == 1:
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=round(rng.uniform(75, 95)))
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            return compressed.convert("RGB")
    if transform == 2:
        scale = rng.uniform(0.85, 1.0)
        size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        return image.resize(size, Image.Resampling.LANCZOS).resize(
            image.size, Image.Resampling.BICUBIC
        )
    if transform == 3:
        sigma = rng.uniform(0.003, 0.012) * 255
        pixels = bytearray(image.tobytes())
        for index, value in enumerate(pixels):
            pixels[index] = min(255, max(0, round(value + rng.gauss(0, sigma))))
        return Image.frombytes(image.mode, image.size, bytes(pixels))

    gain = rng.uniform(0.95, 1.05)
    gradient = Image.linear_gradient("L")
    if rng.random() < 0.5:
        gradient = gradient.rotate(90)
    gradient = gradient.resize(image.size, Image.Resampling.BILINEAR)
    dim = ImageEnhance.Brightness(image).enhance(2 - gain)
    bright = ImageEnhance.Brightness(image).enhance(gain)
    return Image.composite(bright, dim, gradient)


def _crop_path(
    value: object,
    root: Path,
    manifest_path: Path,
    line_number: int,
    name: str,
) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"row lacks a {name} crop path at {manifest_path}:{line_number}"
        )
    crop_path = (root / value).resolve()
    if not crop_path.is_relative_to(root.resolve()):
        raise ValueError(
            f"crop path escapes the dataset at {manifest_path}:{line_number}"
        )
    if not crop_path.is_file():
        raise FileNotFoundError(f"crop was not found: {crop_path}")
    return crop_path


def assistant_labels(
    prompt_tokens: int, answer_ids: list[int], *, max_tokens: int = MAX_TOKENS
) -> list[int]:
    if prompt_tokens < 0 or not answer_ids:
        raise ValueError("prompt length and answer tokens must be nonempty")
    if prompt_tokens + len(answer_ids) > max_tokens:
        raise ValueError(f"training example exceeds {max_tokens} tokens")
    return [IGNORE_INDEX] * prompt_tokens + answer_ids


def _cat_with_pad(tensors: list[Any], *, dim: int) -> Any:
    shape = [
        max(tensor.shape[index] for tensor in tensors)
        for index in range(tensors[0].dim())
    ]
    shape[dim] = sum(tensor.shape[dim] for tensor in tensors)
    output = tensors[0].new_zeros(shape)
    offset = 0
    for tensor in tensors:
        slices = [slice(0, size) for size in tensor.shape]
        slices[dim] = slice(offset, offset + tensor.shape[dim])
        output[tuple(slices)] = tensor
        offset += tensor.shape[dim]
    return output


def _prediction_text(rows: list[dict[str, Any]]) -> list[str]:
    return [row["prediction"] for row in rows]


def normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _ratio(value: str) -> tuple[int, int]:
    parts = value.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("must be HARD:REPLAY")
    try:
        hard, replay = (int(part) for part in parts)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be HARD:REPLAY integers") from error
    if hard <= 0 or replay <= 0:
        raise argparse.ArgumentTypeError("ratio values must be positive")
    return hard, replay


def _hard_weights(value: str) -> dict[str, int]:
    weights: dict[str, int] = {}
    try:
        for item in value.split(","):
            name, weight = item.split("=", 1)
            if name in weights:
                raise ValueError
            weights[name] = int(weight)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "must be comma-separated STRATUM=WEIGHT integers"
        ) from error
    if set(weights) != set(HARD_STRATA):
        raise argparse.ArgumentTypeError(
            "must name exactly resolved, blank, printed_only, stray_mark, unreadable"
        )
    if any(weight <= 0 for weight in weights.values()):
        raise argparse.ArgumentTypeError("stratum weights must be positive")
    return weights


if __name__ == "__main__":
    raise SystemExit(main())
