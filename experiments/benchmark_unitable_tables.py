"""Run the official UniTable checkpoints on a cropped table image."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class DecodeResult:
    token_ids: Any
    generated_tokens: int
    terminated: bool


def parse_crop(value: str) -> tuple[int, int, int, int]:
    parts = tuple(int(part.strip()) for part in value.split(","))
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("crop must be left,top,right,bottom")
    left, top, right, bottom = parts
    if min(parts) < 0 or right <= left or bottom <= top:
        raise argparse.ArgumentTypeError("crop must be a positive non-empty box")
    return left, top, right, bottom


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unitable-root", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--crop", type=parse_crop)
    parser.add_argument("--cell-batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def autoregressive_decode(
    model: Any,
    image: Any,
    *,
    prefix: list[int],
    max_decode_len: int,
    eos_id: int,
    token_whitelist: list[int] | None = None,
    token_blacklist: list[int] | None = None,
) -> DecodeResult:
    import torch

    from src.utils import greedy_sampling, pred_token_within_range, subsequent_mask

    model.eval()
    device = image.device
    with torch.inference_mode():
        memory = model.encode(image)
        context = torch.tensor(prefix, dtype=torch.int32, device=device).repeat(
            image.shape[0], 1
        )

        for generated_tokens in range(1, max_decode_len + 1):
            if all(eos_id in row for row in context):
                return DecodeResult(context, generated_tokens - 1, True)
            causal_mask = subsequent_mask(context.shape[1]).to(device)
            logits = model.decode(
                memory,
                context,
                tgt_mask=causal_mask,
                tgt_padding_mask=None,
            )
            logits = pred_token_within_range(
                model.generator(logits)[:, -1, :],
                white_list=token_whitelist,
                black_list=token_blacklist,
            )
            _, next_tokens = greedy_sampling(logits)
            context = torch.cat([context, next_tokens], dim=1)

    return DecodeResult(context, max_decode_len, all(eos_id in row for row in context))


def image_to_tensor(image: Image.Image, size: tuple[int, int], device: Any) -> Any:
    from torchvision import transforms

    transform = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.86597056, 0.88463002, 0.87491087],
                std=[0.20686628, 0.18201602, 0.18485524],
            ),
        ]
    )
    return transform(image).to(device).unsqueeze(0)


def load_model(
    vocab_path: Path,
    model_weights: Path,
    *,
    max_seq_len: int,
    device: Any,
) -> tuple[Any, Any]:
    import tokenizers
    import torch
    from torch import nn

    from src.model import Decoder, Encoder, EncoderDecoder, ImgLinearBackbone

    d_model = 768
    dropout = 0.2
    vocab = tokenizers.Tokenizer.from_file(str(vocab_path))
    model = EncoderDecoder(
        backbone=ImgLinearBackbone(d_model=d_model, patch_size=16),
        encoder=Encoder(
            d_model=d_model,
            nhead=12,
            dropout=dropout,
            activation="gelu",
            norm_first=True,
            nlayer=12,
            ff_ratio=4,
        ),
        decoder=Decoder(
            d_model=d_model,
            nhead=12,
            dropout=dropout,
            activation="gelu",
            norm_first=True,
            nlayer=4,
            ff_ratio=4,
        ),
        vocab_size=vocab.get_vocab_size(),
        d_model=d_model,
        padding_idx=vocab.token_to_id("<pad>"),
        max_seq_len=max_seq_len,
        dropout=dropout,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    model.load_state_dict(torch.load(model_weights, map_location="cpu"))
    return vocab, model.to(device).eval()


def rescale_boxes(
    boxes: list[list[float]], source: tuple[int, int], target: tuple[int, int]
) -> list[list[int]]:
    ratio = [target[0] / source[0], target[1] / source[1]] * 2
    return [[round(value * scale) for value, scale in zip(box, ratio)] for box in boxes]


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    root = args.unitable_root.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from src.utils import (
        bbox_str_to_token_list,
        build_table_from_html_and_cell,
        cell_str_to_token_list,
        html_str_to_token_list,
        html_table_template,
    )
    from src.vocab import (
        BBOX_TOKENS,
        HTML_TOKENS,
        RESERVED_TOKENS,
        TASK_TOKENS,
    )

    valid_html_tokens = ["<eos>", *HTML_TOKENS]
    valid_bbox_tokens = ["<eos>", *BBOX_TOKENS[:448]]
    invalid_cell_tokens = [
        "<sos>",
        "<pad>",
        "<empty>",
        "<sep>",
        *TASK_TOKENS,
        *RESERVED_TOKENS,
    ]

    if not torch.cuda.is_available():
        raise RuntimeError("UniTable benchmark requires CUDA")
    if args.cell_batch_size <= 0:
        raise ValueError("cell batch size must be positive")

    device = torch.device("cuda:0")
    with Image.open(args.image) as source_image:
        page = source_image.convert("RGB")
    image = page.crop(args.crop) if args.crop else page
    image_size = image.size
    timings: dict[str, float] = {}

    started = time.perf_counter()
    html_vocab, html_model = load_model(
        root / "vocab/vocab_html.json",
        args.weights_dir / "unitable_large_structure.pt",
        max_seq_len=784,
        device=device,
    )
    html_decode = autoregressive_decode(
        html_model,
        image_to_tensor(image, (448, 448), device),
        prefix=[html_vocab.token_to_id("[html]")],
        max_decode_len=512,
        eos_id=html_vocab.token_to_id("<eos>"),
        token_whitelist=[html_vocab.token_to_id(token) for token in valid_html_tokens],
    )
    html_text = html_vocab.decode(
        html_decode.token_ids.detach().cpu().numpy()[0], skip_special_tokens=False
    )
    html_tokens = html_str_to_token_list(html_text)
    html_terminated = html_decode.terminated
    html_generated_tokens = html_decode.generated_tokens
    timings["structure_seconds"] = time.perf_counter() - started
    del html_decode, html_model
    torch.cuda.empty_cache()

    started = time.perf_counter()
    bbox_vocab, bbox_model = load_model(
        root / "vocab/vocab_bbox.json",
        args.weights_dir / "unitable_large_bbox.pt",
        max_seq_len=1024,
        device=device,
    )
    bbox_decode = autoregressive_decode(
        bbox_model,
        image_to_tensor(image, (448, 448), device),
        prefix=[bbox_vocab.token_to_id("[bbox]")],
        max_decode_len=1024,
        eos_id=bbox_vocab.token_to_id("<eos>"),
        token_whitelist=[bbox_vocab.token_to_id(token) for token in valid_bbox_tokens],
    )
    bbox_text = bbox_vocab.decode(
        bbox_decode.token_ids.detach().cpu().numpy()[0], skip_special_tokens=False
    )
    boxes = rescale_boxes(bbox_str_to_token_list(bbox_text), (448, 448), image_size)
    bbox_terminated = bbox_decode.terminated
    bbox_generated_tokens = bbox_decode.generated_tokens
    timings["bbox_seconds"] = time.perf_counter() - started
    del bbox_decode, bbox_model
    torch.cuda.empty_cache()

    started = time.perf_counter()
    cell_vocab, cell_model = load_model(
        root / "vocab/vocab_cell_6k.json",
        args.weights_dir / "unitable_large_content.pt",
        max_seq_len=200,
        device=device,
    )
    invalid_cell_ids = [cell_vocab.token_to_id(token) for token in invalid_cell_tokens]
    cells: list[str] = []
    cell_terminated: list[bool] = []
    for offset in range(0, len(boxes), args.cell_batch_size):
        batch_boxes = boxes[offset : offset + args.cell_batch_size]
        batch = torch.cat(
            [
                image_to_tensor(image.crop(tuple(box)), (112, 448), device)
                for box in batch_boxes
            ],
            dim=0,
        )
        decoded = autoregressive_decode(
            cell_model,
            batch,
            prefix=[cell_vocab.token_to_id("[cell]")],
            max_decode_len=200,
            eos_id=cell_vocab.token_to_id("<eos>"),
            token_blacklist=invalid_cell_ids,
        )
        decoded_text = cell_vocab.decode_batch(
            decoded.token_ids.detach().cpu().numpy(), skip_special_tokens=False
        )
        cells.extend(
            re.sub(r"(\d).\s+(\d)", r"\1.\2", cell_str_to_token_list(text))
            for text in decoded_text
        )
        cell_terminated.extend(
            cell_vocab.token_to_id("<eos>") in row
            for row in decoded.token_ids.detach().cpu().numpy()
        )
    timings["content_seconds"] = time.perf_counter() - started
    del cell_model
    torch.cuda.empty_cache()

    decoded_cells = list(cells)
    table = html_table_template(
        "".join(build_table_from_html_and_cell(html_tokens, list(cells)))
    )
    timings["total_seconds"] = sum(timings.values())
    return {
        "source": str(args.image),
        "crop": list(args.crop) if args.crop else None,
        "image_size": list(image_size),
        "structure": {
            "terminated": html_terminated,
            "generated_tokens": html_generated_tokens,
            "tokens": html_tokens,
        },
        "bbox": {
            "terminated": bbox_terminated,
            "generated_tokens": bbox_generated_tokens,
            "boxes": boxes,
        },
        "content": {
            "terminated_count": sum(cell_terminated),
            "cell_count": len(decoded_cells),
            "cells": decoded_cells,
        },
        "html": table,
        "timings": timings,
    }


def main() -> None:
    args = build_parser().parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["timings"], indent=2))
    print(
        json.dumps(
            {
                "cell_count": result["content"]["cell_count"],
                "structure_terminated": result["structure"]["terminated"],
                "bbox_terminated": result["bbox"]["terminated"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
