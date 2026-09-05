from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
import json
from pathlib import Path
import random
from types import SimpleNamespace
import subprocess
import sys

from PIL import Image, ImageOps
import pytest
import torch

from experiments import finetune_phi4_handwriting as finetune


SCRIPT = Path(__file__).parents[1] / "experiments" / "finetune_phi4_handwriting.py"


class FakeParameter:
    def __init__(self, count: int) -> None:
        self.requires_grad = True
        self.count = count

    def numel(self) -> int:
        return self.count


class FakeModel:
    def __init__(self) -> None:
        self.adapter = None
        self.values = {
            "model.embed_tokens.weight": FakeParameter(10),
            "model.image_embed.proj.weight": FakeParameter(20),
            "model.layers.0.self_attn.qkv_proj.base_layer.weight": FakeParameter(30),
            "model.layers.0.self_attn.qkv_proj.lora_A.vision.weight": FakeParameter(4),
            "model.layers.0.self_attn.qkv_proj.lora_B.vision.weight": FakeParameter(4),
            "model.layers.0.mlp.down_proj.lora_A.speech.weight": FakeParameter(4),
            "lm_head.weight": FakeParameter(40),
        }

    def set_lora_adapter(self, name: str) -> None:
        self.adapter = name

    def named_parameters(self):
        return list(self.values.items())

    def parameters(self):
        return list(self.values.values())


def test_validate_only_consumes_compiler_jsonl_without_ml_dependencies(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset"
    _split(root, "train", "C08-D001-P001", "C08-D001")
    _split(root, "dev", "C08-D002-P001", "C08-D002")
    primary = _review(root, "primary.json", [], reviewed_count=2, reviewer_id="one")
    independent = _review(
        root, "independent.json", [], reviewed_count=2, reviewer_id="two"
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(root),
            str(tmp_path / "unused"),
            "--review-decision",
            str(primary),
            "--review-decision",
            str(independent),
            "--validate-only",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == {
        "c14_fields": 0,
        "dev_families": 1,
        "dev_fields": 1,
        "mode": "canary",
        "train_families": 1,
        "train_fields": 1,
    }
    assert not (tmp_path / "unused").exists()


def test_rejects_c14_family_leakage_and_private_infrastructure_rows(
    tmp_path: Path,
) -> None:
    c14 = tmp_path / "c14"
    _split(c14, "train", "C14-D001-P001", "C14-D001", origin="public")
    _split(c14, "dev", "C08-D002-P001", "C08-D002", origin="public")
    with pytest.raises(ValueError, match="C14"):
        finetune.load_splits(c14, mode="infrastructure")

    overlap = tmp_path / "overlap"
    _split(overlap, "train", "C08-D001-P001", "family", origin="public")
    _split(overlap, "dev", "C08-D001-P002", "family", origin="public")
    with pytest.raises(ValueError, match="overlapping"):
        finetune.load_splits(overlap, mode="infrastructure")

    private = tmp_path / "private"
    _split(private, "train", "C08-D001-P001", "train-family", origin="private")
    _split(private, "dev", "C08-D002-P001", "dev-family", origin="public")
    with pytest.raises(ValueError, match="public or synthetic"):
        finetune.load_splits(private, mode="infrastructure")


def test_review_decisions_filter_rejections_and_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "handwriting-crops-v2"
    _split(root, "train", "C08-D001-P001", "train-one")
    _split(root, "train", "C08-D002-P001", "train-two")
    _split(root, "dev", "C08-D003-P001", "dev-one")
    _split(root, "dev", "C08-D004-P001", "dev-two")
    rejected_id = "C08-D001-P001-H001"
    primary = _review(root, "primary.json", [rejected_id], reviewer_id="one")
    independent = _review(root, "independent.json", [rejected_id], reviewer_id="two")

    with pytest.raises(ValueError, match="two agreeing"):
        finetune.load_splits(root)
    with pytest.raises(ValueError, match="two agreeing"):
        finetune.load_splits(root, review_decisions=[primary])

    train, dev = finetune.load_splits(root, review_decisions=[primary, independent])

    assert {row.field_id for row in train} == {"C08-D002-P001-H001"}
    assert len(dev) == 2

    mismatch = json.loads(primary.read_text(encoding="utf-8"))
    mismatch["reviewed_field_count"] = 3
    primary.write_text(json.dumps(mismatch), encoding="utf-8")
    with pytest.raises(ValueError, match="reviewed count"):
        finetune.load_splits(root, review_decisions=[primary, independent])

    primary = _review(root, "primary.json", [rejected_id], reviewer_id="one")
    disagree = _review(root, "disagree.json", ["C08-D002-P001-H001"], reviewer_id="two")
    with pytest.raises(ValueError, match="disagree"):
        finetune.load_splits(root, review_decisions=[primary, disagree])

    missing_uncertainty = json.loads(primary.read_text(encoding="utf-8"))
    del missing_uncertainty["uncertain_field_count"]
    primary.write_text(json.dumps(missing_uncertainty), encoding="utf-8")
    with pytest.raises(ValueError, match="explicitly report zero"):
        finetune.load_splits(root, review_decisions=[primary, independent])


def test_review_decisions_require_distinct_reviewer_identities(tmp_path: Path) -> None:
    root = tmp_path / "reviews"
    _split(root, "train", "C08-D001-P001", "train")
    _split(root, "dev", "C08-D002-P001", "dev")
    first = _review(root, "first.json", [], reviewed_count=2, reviewer_id="Reviewer")
    copied = _review(
        root, "copied.json", [], reviewed_count=2, reviewer_id=" reviewer "
    )

    with pytest.raises(ValueError, match="distinct reviewer identities"):
        finetune.load_splits(root, review_decisions=[first, copied])

    missing = json.loads(copied.read_text(encoding="utf-8"))
    del missing["reviewer_id"]
    copied.write_text(json.dumps(missing), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid embedded reviewer"):
        finetune.load_splits(root, review_decisions=[first, copied])


def test_embedded_independent_reviews_satisfy_canary(tmp_path: Path) -> None:
    root = tmp_path / "embedded"
    _split(
        root,
        "train",
        "C08-D001-P001",
        "train-family",
        origin="private",
        embedded_reviews=True,
    )
    _split(
        root,
        "dev",
        "C08-D002-P001",
        "dev-family",
        origin="private",
        embedded_reviews=True,
    )

    train, dev = finetune.load_splits(root)

    assert train[0].reviewer_ids == ("one", "two")
    assert dev[0].reviewer_ids == ("one", "two")

    rows = [json.loads(line) for line in (root / "dev.jsonl").read_text().splitlines()]
    rows[0]["independent_reviews"][1]["transcription"] = "different"
    (root / "dev.jsonl").write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="disagree"):
        finetune.load_splits(root)


def test_enables_only_existing_vision_decoder_lora() -> None:
    model = FakeModel()

    audit = finetune.enable_vision_decoder_lora(model)

    assert model.adapter == "vision"
    assert audit["trainable_parameters"] == 8
    assert audit["total_parameters"] == 112
    assert audit["trainable_names"] == [
        "model.layers.0.self_attn.qkv_proj.lora_A.vision.weight",
        "model.layers.0.self_attn.qkv_proj.lora_B.vision.weight",
    ]
    assert all(
        parameter.requires_grad == (".vision." in name)
        for name, parameter in model.named_parameters()
    )


def test_parser_defaults_are_bf16_sdpa_canary_contract() -> None:
    args = finetune.build_parser().parse_args(["data", "output"])
    scheduled = finetune.build_parser().parse_args(
        [
            "data",
            "output",
            "--hard-replay-ratio",
            "1:1",
            "--hard-stratum-weights",
            "resolved=6,blank=1,printed_only=1,stray_mark=1,unreadable=1",
        ]
    )

    assert args.effective_batch == 8
    assert args.mode == "canary"
    assert args.epochs == 3
    assert args.hard_replay_ratio is None
    assert args.hard_stratum_weights is None
    assert scheduled.hard_replay_ratio == (1, 1)
    assert scheduled.hard_stratum_weights == {
        "resolved": 6,
        "blank": 1,
        "printed_only": 1,
        "stray_mark": 1,
        "unreadable": 1,
    }
    assert finetune.MAX_TOKENS == 1024
    assert finetune.MICROBATCH == 1
    assert finetune.MODEL_REVISION == "93f923e1a7727d1c4f446756212d9d3e8fcc5d81"


def test_split_group_lineage_and_legacy_family_fallback(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy-groups"
    _split(legacy, "train", "C08-D001-P001", "train", origin="public")
    _split(legacy, "dev", "C08-D002-P001", "dev", origin="public")

    train, _ = finetune.load_splits(legacy, mode="infrastructure")

    assert train[0].split_group_id == train[0].family_id

    invalid = tmp_path / "invalid-groups"
    _split(
        invalid,
        "train",
        "C08-D003-P001",
        "family-one",
        origin="public",
        split_group_id="shared",
    )
    _split(
        invalid,
        "train",
        "C08-D004-P001",
        "family-two",
        origin="public",
        split_group_id="shared",
    )
    _split(invalid, "dev", "C08-D005-P001", "dev", origin="public")
    with pytest.raises(ValueError, match="split group belongs to multiple"):
        finetune.load_splits(invalid, mode="infrastructure")


def test_materialized_schedule_is_deterministic_and_balanced() -> None:
    weights = {
        "resolved": 6,
        "blank": 1,
        "printed_only": 1,
        "stray_mark": 1,
        "unreadable": 1,
    }
    hard_counts = {
        "resolved": 134,
        "blank": 4,
        "printed_only": 3,
        "stray_mark": 10,
        "unreadable": 5,
    }
    records = [
        _schedule_record(
            f"{stratum}-{index}",
            f"hard-family-{index % 18}",
            f"hard-family-{index % 18}-group-{index % 2}",
            stratum=stratum,
        )
        for stratum, amount in hard_counts.items()
        for index in range(amount)
    ]
    records.extend(
        _schedule_record(
            f"replay-{index}",
            f"replay-family-{index % 20}",
            f"replay-family-{index % 20}-group-{index % 3}",
            origin="public",
        )
        for index in range(200)
    )

    first = finetune.materialize_training_schedule(
        records,
        hard_replay_ratio=(1, 1),
        hard_stratum_weights=weights,
        epochs=3,
        seed=17,
    )
    repeated = finetune.materialize_training_schedule(
        records,
        hard_replay_ratio=(1, 1),
        hard_stratum_weights=weights,
        epochs=3,
        seed=17,
    )
    changed_seed = finetune.materialize_training_schedule(
        records,
        hard_replay_ratio=(1, 1),
        hard_stratum_weights=weights,
        epochs=3,
        seed=19,
    )

    assert [row.field_id for row in first] == [row.field_id for row in repeated]
    assert [row.field_id for row in first] != [row.field_id for row in changed_seed]
    assert {row.field_id for row in first} == {row.field_id for row in records}
    summary = finetune.training_schedule_summary(
        first,
        hard_replay_ratio=(1, 1),
        hard_stratum_weights=weights,
    )
    assert summary["fields"] == len(records) * 3 == 1068
    assert summary["unique_fields"] == len(records) == 356
    assert summary["hard_fields"] == summary["replay_fields"] == 534
    assert summary["hard_replay_ratio"] == {"hard": 1, "replay": 1}
    assert summary["hard_stratum_weights"] == weights
    assert summary["hard_strata"] == {
        "blank": 54,
        "printed_only": 54,
        "resolved": 320,
        "stray_mark": 53,
        "unreadable": 53,
    }
    for stratum, weight in weights.items():
        expected = summary["hard_fields"] * weight / sum(weights.values())
        assert abs(summary["hard_strata"][stratum] - expected) < 1


@pytest.mark.parametrize("subtype", [None, "absent"])
def test_scheduled_mode_rejects_generic_absent_with_clear_error(
    subtype: str | None,
) -> None:
    records = [
        replace(
            _schedule_record("absent", "family", "group", stratum="blank"),
            abstention_subtype=subtype,
        )
    ]

    assert finetune.materialize_training_schedule(records) is records
    with pytest.raises(
        ValueError, match="must specify blank, printed_only, or stray_mark"
    ):
        finetune.materialize_training_schedule(
            records,
            hard_replay_ratio=(1, 1),
            hard_stratum_weights=dict.fromkeys(finetune.HARD_STRATA, 1),
            epochs=3,
        )


def test_materialized_schedule_falls_back_to_available_data() -> None:
    records = [
        _schedule_record(f"hard-{index}", "family", "group") for index in range(3)
    ]
    weights = dict.fromkeys(finetune.HARD_STRATA, 1)

    assert finetune.materialize_training_schedule(records) is records
    scheduled = finetune.materialize_training_schedule(
        records,
        hard_replay_ratio=(1, 1),
        hard_stratum_weights=weights,
        seed=17,
    )

    assert len(scheduled) == len(records)
    assert {row.target_state for row in scheduled} == {"resolved"}
    with pytest.raises(ValueError, match="supplied together"):
        finetune.materialize_training_schedule(
            records,
            hard_replay_ratio=(1, 1),
        )
    with pytest.raises(ValueError, match="exactly"):
        finetune.materialize_training_schedule(
            records,
            hard_replay_ratio=(1, 1),
            hard_stratum_weights={"resolved": 1},
        )


def test_adapter_provenance_rejects_overlap_and_accepts_train_only(
    tmp_path: Path,
) -> None:
    adapter = tmp_path / "adapter.pt"
    training_family = "PRIVATE-TRAIN-FAMILY"
    model = _AdapterModel()
    finetune.save_trainable_state(
        model,
        adapter,
        torch,
        training_family_ids={training_family},
    )

    payload = torch.load(adapter, map_location="cpu", weights_only=True)
    assert payload["provenance"]["training_family_count"] == 1
    assert payload["provenance"]["training_family_ids"] == [training_family.casefold()]

    finetune.load_trainable_state(
        _AdapterModel(),
        adapter,
        torch,
        evaluation_family_ids={"PRIVATE-EVALUATION-FAMILY"},
    )
    with pytest.raises(ValueError, match="overlap evaluation families"):
        finetune.load_trainable_state(
            _AdapterModel(),
            adapter,
            torch,
            evaluation_family_ids={training_family},
        )


def test_rejects_unpinned_model_and_nondefault_device_before_loading(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="pinned checkpoint"):
        finetune.run_canary([], [], tmp_path / "model", model_name="other/model")
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        finetune.run_canary([], [], tmp_path / "device", device="cuda:1")


def test_labels_mask_prompt_and_reject_target_truncation() -> None:
    assert finetune.assistant_labels(3, [41, 42], max_tokens=5) == [
        -100,
        -100,
        -100,
        41,
        42,
    ]
    assert finetune.IGNORE_INDEX == -100
    with pytest.raises(ValueError, match="exceeds"):
        finetune.assistant_labels(4, [41, 42], max_tokens=5)


@pytest.mark.parametrize(
    ("mode", "origin", "expected_max_steps"),
    [("infrastructure", "public", 2), ("canary", "private", -1)],
)
def test_training_output_excludes_private_literals_and_exception_messages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    origin: str,
    expected_max_steps: int,
) -> None:
    private_reference = "PRIVATE_REFERENCE_CANARY_8f1f"
    private_prediction = "PRIVATE_PREDICTION_CANARY_38ac"
    private_error = "PRIVATE_EXCEPTION_CANARY_6d2e"
    records = [
        finetune.CropRecord(
            field_id="public-H001",
            case_id="public",
            family_id="family",
            reference=private_reference,
            crop_path=tmp_path / "unused.png",
            split="train",
            data_origin=origin,
        )
    ]
    argument_calls = []
    train_calls = []
    train_datasets = []
    prediction_calls = []
    models = [object(), object()]

    class FakeTrainingArguments:
        def __init__(self, **values: object) -> None:
            self.values = values
            argument_calls.append(values)

    class FakeTrainer:
        def __init__(self, **values: object) -> None:
            self.values = values
            train_datasets.append(values["train_dataset"])

        def train(self) -> None:
            train_calls.append(True)

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

        @staticmethod
        def empty_cache() -> None:
            raise AssertionError("empty_cache should not run without CUDA")

    fake_torch = SimpleNamespace(cuda=FakeCuda())
    processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0))

    def handle_load(*args: object, **kwargs: object):
        return (
            fake_torch,
            processor,
            models.pop(0),
            (FakeTrainer, FakeTrainingArguments),
        )

    def handle_predict(*args: object, **kwargs: object):
        prediction_calls.append(True)
        return [
            {
                "field_id": "public-H001",
                "reference": private_reference,
                "prediction": (
                    private_prediction
                    if len(prediction_calls) == 1
                    else private_reference
                ),
                "error": private_error,
            }
        ]

    monkeypatch.setattr(finetune, "load_runtime", handle_load)
    monkeypatch.setattr(
        finetune,
        "enable_vision_decoder_lora",
        lambda model: {"trainable_parameters": 1},
    )
    monkeypatch.setattr(finetune, "predict_records", handle_predict)
    monkeypatch.setattr(
        finetune,
        "save_trainable_state",
        lambda model, path, torch_module, **kwargs: path.write_bytes(b"adapter"),
    )
    monkeypatch.setattr(
        finetune,
        "load_trainable_state",
        lambda model, path, torch_module: None,
    )

    schedule_options = {}
    if mode == "canary":
        schedule_options = {
            "hard_replay_ratio": (1, 1),
            "hard_stratum_weights": dict.fromkeys(finetune.HARD_STRATA, 1),
        }
    result = finetune.run_canary(
        records,
        records,
        tmp_path / "run",
        mode=mode,
        **schedule_options,
    )

    assert train_calls == [True]
    assert argument_calls[0]["max_steps"] == expected_max_steps
    assert argument_calls[0]["num_train_epochs"] == (1 if mode == "canary" else 3)
    assert len(train_datasets[0]) == (3 if mode == "canary" else 1)
    assert argument_calls[0]["per_device_train_batch_size"] == 1
    assert argument_calls[0]["gradient_accumulation_steps"] == 8
    assert argument_calls[0]["bf16"] is True
    assert argument_calls[0]["fp16"] is False
    assert result["model"] == {
        "id": finetune.MODEL_ID,
        "revision": finetune.MODEL_REVISION,
        "pinned": True,
    }
    assert result["reload_prediction_parity"] is True
    assert result["augmentation"] == finetune.AUGMENTATION_POLICY
    assert result["validation"]["after"]["normalized_exact_rate"] == 1.0
    assert result["validation"]["after"]["cer"] == 0.0
    assert result["privacy"] == {
        "case_identifiers_persisted": False,
        "exception_messages_persisted": False,
        "prediction_text_persisted": False,
        "private_text_persisted": False,
    }
    assert result["cases"] == [
        {
            "index": 0,
            "split": "train",
            "data_origin": origin,
            "target_state": "resolved",
            "abstention_subtype": None,
            "before_exact": False,
            "after_exact": True,
        }
    ]
    persisted = (tmp_path / "run" / "result.json").read_text(encoding="utf-8")
    assert private_reference not in persisted
    assert private_prediction not in persisted
    assert private_error not in persisted
    assert '"before": [' not in persisted
    assert '"after": [' not in persisted


def test_training_samples_both_crop_views_with_the_same_target(tmp_path: Path) -> None:
    tight = tmp_path / "tight.png"
    padded = tmp_path / "padded.png"
    Image.new("RGB", (20, 10), "black").save(tight)
    Image.new("RGB", (20, 10), "white").save(padded)
    record = _record(tmp_path, tight, padded)
    rng = SequenceRng([0.0, 0.0, 0.9, 0.0])

    first = finetune.prepare_image(record, training=True, rng=rng)
    second = finetune.prepare_image(record, training=True, rng=rng)

    assert first.getpixel((0, 0)) == (0, 0, 0)
    assert second.getpixel((0, 0)) == (255, 255, 255)
    assert finetune.target_text(record) == "literal"


def test_training_draws_repeat_by_seed_and_vary_across_accesses(tmp_path: Path) -> None:
    tight = tmp_path / "tight.png"
    padded = tmp_path / "padded.png"
    Image.new("RGB", (20, 10), "black").save(tight)
    Image.new("RGB", (20, 10), "white").save(padded)
    record = _record(tmp_path, tight, padded)

    def draws() -> list[bytes]:
        rng = random.Random(17)
        return [
            finetune.prepare_image(record, training=True, rng=rng).tobytes()
            for _ in range(16)
        ]

    first = draws()
    second = draws()

    assert first == second
    assert len(set(first)) > 1


def test_explicit_abstention_targets() -> None:
    base = {
        "field_id": "field",
        "case_id": "case",
        "family_id": "family",
        "reference": "",
        "crop_path": Path("unused.png"),
        "split": "train",
        "data_origin": "private",
    }

    assert (
        finetune.target_text(finetune.CropRecord(**base, target_state="absent"))
        == finetune.NO_HANDWRITING
    )
    assert (
        finetune.target_text(finetune.CropRecord(**base, target_state="unreadable"))
        == finetune.UNREADABLE
    )
    with pytest.raises(ValueError, match="lacks a literal"):
        finetune.target_text(finetune.CropRecord(**base, target_state="resolved"))


def test_validation_metrics_separate_transcription_and_abstention() -> None:
    records = [
        _metric_record("substitution", "abc"),
        _metric_record("deletion", "def"),
        _metric_record("insertion", "ghi"),
        _metric_record("normalized", "Dose 10"),
        _metric_record("blank", "", target_state="absent", subtype="blank"),
        _metric_record("printed", "", target_state="absent", subtype="printed_only"),
        _metric_record("stray", "", target_state="absent", subtype="stray_mark"),
        _metric_record(
            "unreadable", "", target_state="unreadable", subtype="unreadable"
        ),
    ]
    predictions = [
        {"field_id": "substitution", "prediction": "axc"},
        {"field_id": "deletion", "prediction": "df"},
        {"field_id": "insertion", "prediction": "ghxi"},
        {"field_id": "normalized", "prediction": " dose   10 "},
        {"field_id": "blank", "prediction": finetune.NO_HANDWRITING},
        {"field_id": "printed", "prediction": finetune.NO_HANDWRITING},
        {"field_id": "stray", "prediction": "copied print"},
        {"field_id": "unreadable", "prediction": finetune.NO_HANDWRITING},
    ]

    metrics = finetune.validation_metrics(records, predictions)

    assert metrics == {
        "fields": 8,
        "resolved_fields": 4,
        "normalized_exact": 1,
        "normalized_exact_rate": 0.25,
        "reference_characters": 16,
        "character_edit_counts": {
            "insertions": 1,
            "deletions": 1,
            "substitutions": 1,
        },
        "cer": 0.1875,
        "missed_character_rate": 0.0625,
        "hallucinated_character_rate": 0.0625,
        "substitution_rate": 0.0625,
        "abstention_fields": 4,
        "abstention_accuracy": 0.5,
        "abstention_subtype_accuracy": {
            "blank": {"fields": 1, "correct": 1, "accuracy": 1.0},
            "printed_only": {"fields": 1, "correct": 1, "accuracy": 1.0},
            "stray_mark": {"fields": 1, "correct": 0, "accuracy": 0.0},
            "unreadable": {"fields": 1, "correct": 0, "accuracy": 0.0},
        },
    }


def test_inference_prompt_contains_no_reference() -> None:
    calls = []

    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            calls.append((messages, kwargs))
            return "rendered prompt"

    prompt = finetune.inference_prompt(SimpleNamespace(tokenizer=FakeTokenizer()))

    assert prompt == "rendered prompt"
    assert calls == [
        (
            [{"role": "user", "content": f"<|image_1|>{finetune.PROMPT}"}],
            {"tokenize": False, "add_generation_prompt": True},
        )
    ]
    assert "reference" not in calls[0][0][0]["content"].casefold()


def test_prediction_reuses_processor_input_mode_without_duplicate_keyword(
    tmp_path: Path,
) -> None:
    crop = tmp_path / "crop.png"
    Image.new("RGB", (20, 10), "white").save(crop)
    record = _record(tmp_path, crop, crop)
    calls = []

    class FakeIds:
        def size(self, dimension: int) -> int:
            assert dimension == 1
            return 2

    class FakeInputs(dict):
        input_ids = FakeIds()

        def to(self, device: str):
            assert device == "cuda:0"
            return self

    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return "prompt"

    class FakeProcessor:
        tokenizer = FakeTokenizer()

        def __call__(self, prompt, *, images, return_tensors):
            assert prompt == "prompt"
            assert len(images) == 1
            assert return_tensors == "pt"
            return FakeInputs(input_ids=FakeIds(), input_mode=7)

        def decode(self, token_ids, **kwargs):
            assert token_ids == [2, 3]
            return "literal"

    class FakePredictionModel:
        def eval(self) -> None:
            return None

        def train(self) -> None:
            return None

        def generate(self, **kwargs):
            calls.append(kwargs)
            return [[0, 1, 2, 3]]

    predictions = finetune.predict_records(
        FakePredictionModel(),
        FakeProcessor(),
        [record],
        device="cuda:0",
        torch_module=SimpleNamespace(inference_mode=nullcontext),
    )

    assert predictions[0]["prediction"] == "literal"
    assert calls[0]["input_mode"] == 7


def test_abstention_manifest_rows_load_without_literals(tmp_path: Path) -> None:
    root = tmp_path / "abstention"
    _split(
        root,
        "train",
        "C08-D001-P001",
        "train",
        origin="public",
        reference="",
        target_state="absent",
        abstention_subtype="printed_only",
    )
    _split(
        root,
        "dev",
        "C08-D002-P001",
        "dev",
        origin="public",
        reference="",
        target_state="unreadable",
    )

    train, dev = finetune.load_splits(root, mode="infrastructure")

    assert finetune.target_text(train[0]) == finetune.NO_HANDWRITING
    assert train[0].abstention_subtype == "printed_only"
    assert finetune.target_text(dev[0]) == finetune.UNREADABLE


@pytest.mark.parametrize(
    ("draw", "expected"),
    [
        (0.0, "identity"),
        (0.499999, "identity"),
        (0.5, "geometry"),
        (0.749999, "geometry"),
        (0.75, "acquisition"),
        (0.949999, "acquisition"),
        (0.95, "geometry_acquisition"),
        (0.999999, "geometry_acquisition"),
    ],
)
def test_augmentation_policy_boundaries(draw: float, expected: str) -> None:
    assert finetune.augmentation_kind(SequenceRng([draw])) == expected


def test_combined_augmentation_runs_exactly_two_transforms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    monkeypatch.setattr(
        finetune, "_geometry", lambda image, rng: calls.append("geometry") or image
    )
    monkeypatch.setattr(
        finetune,
        "_acquisition",
        lambda image, rng: calls.append("acquisition") or image,
    )

    finetune.augment_image(Image.new("RGB", (20, 10)), SequenceRng([0.95]))

    assert calls == ["geometry", "acquisition"]


@pytest.mark.parametrize(
    ("transform_draw", "uniforms"),
    [
        (0.0, [1.5]),
        (0.25, [0.015, 0.03]),
        (0.5, [1.05]),
        (0.75, [3.0]),
    ],
)
def test_geometry_preserves_canvas_and_does_not_clip_content(
    transform_draw: float, uniforms: list[float]
) -> None:
    image = Image.new("RGB", (100, 40), "white")
    for x in (*range(4), *range(96, 100)):
        for y in range(5, 35):
            image.putpixel((x, y), (0, 0, 0))
    for x in range(5, 95):
        for y in (*range(4), *range(36, 40)):
            image.putpixel((x, y), (0, 0, 0))
    rng = SequenceRng([0.5, transform_draw], uniforms=uniforms)

    result = finetune.augment_image(image, rng)

    ink = ImageOps.invert(result.convert("L")).getbbox()
    assert ink is not None
    assert result.width > image.width
    assert result.height > image.height
    assert ink[0] > 0 and ink[1] > 0
    assert ink[2] < result.width and ink[3] < result.height


def test_smooth_illumination_is_bounded_and_nonuniform() -> None:
    image = Image.new("RGB", (64, 32), (100, 100, 100))
    rng = SequenceRng([0.99, 0.0], uniforms=[1.05])

    result = finetune._acquisition(image, rng)
    values = list(result.convert("L").getdata())

    assert result.size == image.size
    assert 94 <= min(values) < max(values) <= 105
    assert finetune.AUGMENTATION_POLICY["acquisition"]["smooth_illumination_gain"] == [
        0.95,
        1.05,
    ]


def test_dev_image_is_real_only_and_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    crop = tmp_path / "crop.png"
    Image.new("RGB", (20, 10), "white").save(crop)
    record = _record(tmp_path, crop, crop)
    monkeypatch.setattr(
        finetune,
        "augment_image",
        lambda image, rng: (_ for _ in ()).throw(AssertionError("augmented dev")),
    )

    first = finetune.prepare_image(record, training=False, rng=random.Random(1))
    second = finetune.prepare_image(record, training=False, rng=random.Random(2))

    assert first.tobytes() == second.tobytes()


def test_old_manifest_loads_as_resolved_with_both_views(tmp_path: Path) -> None:
    root = tmp_path / "legacy"
    _split(root, "train", "C08-D001-P001", "train", origin="public")
    _split(root, "dev", "C08-D002-P001", "dev", origin="public")

    train, _ = finetune.load_splits(root, mode="infrastructure")

    assert train[0].target_state == "resolved"
    assert train[0].tight_crop_path == train[0].crop_path
    assert train[0].padded_crop_path == train[0].crop_path


class SequenceRng:
    def __init__(
        self, draws: list[float], *, uniforms: list[float] | None = None
    ) -> None:
        self.draws = iter(draws)
        self.uniforms = iter(uniforms or [])

    def random(self) -> float:
        return next(self.draws)

    def uniform(self, start: float, end: float) -> float:
        value = next(self.uniforms)
        assert start <= value <= end
        return value

    def gauss(self, mean: float, sigma: float) -> float:
        return mean


class _AdapterModel:
    def __init__(self) -> None:
        self.parameter = torch.nn.Parameter(torch.tensor([1.0]))

    def named_parameters(self):
        return [("decoder.lora_A.vision.weight", self.parameter)]

    def load_state_dict(self, state: object, *, strict: bool):
        assert strict is False
        return SimpleNamespace(unexpected_keys=[])


def _record(root: Path, tight: Path, padded: Path) -> finetune.CropRecord:
    return finetune.CropRecord(
        field_id="field",
        case_id="case",
        family_id="family",
        reference="literal",
        crop_path=padded,
        split="train",
        data_origin="private",
        tight_crop_path=tight,
        padded_crop_path=padded,
    )


def _schedule_record(
    field_id: str,
    family_id: str,
    split_group_id: str,
    *,
    origin: str = "private",
    stratum: str = "resolved",
) -> finetune.CropRecord:
    target_state = "resolved" if stratum == "resolved" else "absent"
    if stratum == "unreadable":
        target_state = "unreadable"
    return finetune.CropRecord(
        field_id=field_id,
        case_id=field_id,
        family_id=family_id,
        split_group_id=split_group_id,
        reference="literal" if target_state == "resolved" else "",
        crop_path=Path("unused.png"),
        split="train",
        data_origin=origin,
        target_state=target_state,
        abstention_subtype=None if stratum == "resolved" else stratum,
    )


def _metric_record(
    field_id: str,
    reference: str,
    *,
    target_state: str = "resolved",
    subtype: str | None = None,
) -> finetune.CropRecord:
    return finetune.CropRecord(
        field_id=field_id,
        case_id="case",
        family_id="family",
        reference=reference,
        crop_path=Path("unused.png"),
        split="dev",
        data_origin="private",
        target_state=target_state,
        abstention_subtype=subtype,
    )


def _split(
    root: Path,
    split: str,
    case_id: str,
    family_id: str,
    *,
    origin: str | None = None,
    reference: str = "literal",
    target_state: str | None = None,
    abstention_subtype: str | None = None,
    embedded_reviews: bool = False,
    split_group_id: str | None = None,
) -> None:
    crop = root / "crops" / split / f"{case_id}.png"
    crop.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (20, 10), "white").save(crop)
    record = {
        "field_id": f"{case_id}-H001",
        "case_id": case_id,
        "category_id": case_id.split("-", 1)[0],
        "family_id": family_id,
        "reference": reference,
        "crop_path": crop.relative_to(root).as_posix(),
        "split": split,
    }
    if origin is not None:
        record["data_origin"] = origin
    if split_group_id is not None:
        record["split_group_id"] = split_group_id
    if target_state is not None:
        record["target_state"] = target_state
    if abstention_subtype is not None:
        record["abstention_subtype"] = abstention_subtype
    if embedded_reviews:
        record["reviewer_ids"] = ["one", "two"]
        record["independent_reviews"] = [
            {
                "reviewer_id": reviewer,
                "transcription": reference,
                "legibility": "legible",
                "region_type": "field",
            }
            for reviewer in ("one", "two")
        ]
    with (root / f"{split}.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def _review(
    root: Path,
    name: str,
    rejected_ids: list[str],
    *,
    reviewed_count: int = 4,
    reviewer_id: str = "reviewer",
) -> Path:
    path = root / name
    path.write_text(
        json.dumps(
            {
                "reviewer_id": reviewer_id,
                "reviewed_crop_set": root.name,
                "reviewed_field_count": reviewed_count,
                "accepted_field_count": reviewed_count - len(rejected_ids),
                "uncertain_field_count": 0,
                "rejected": [
                    {"field_id": field_id, "reason": "bad crop"}
                    for field_id in rejected_ids
                ],
            }
        ),
        encoding="utf-8",
    )
    return path
