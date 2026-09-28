"""Fetch KVP10k pages with IBM's official helpers (github.com/IBM/KVP10k), without its OCR step.

Run from a KVP10k checkout: python download_kvp10k.py OUT_DIR. Writes images/<hash>.png
(long side at most 1650 px, as for SignverOD) and items/<hash>.json with the page's
annotation rectangles. Dead source URLs are counted and skipped.
"""

import json
import socket
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import datasets
from PIL import Image
from utils.huggingface_utils.download_and_prepare_dataset import get_image
from utils.huggingface_utils.parquets_to_jsons import parquets_to_jsons

MAX_SIDE = 1650
# The official get_image has no request timeout, and dead hosts hang every worker.
socket.setdefaulttimeout(30)


def fetch(out, item):
    name = item["hash_name"]
    if (out / "items" / f"{name}.json").exists():
        return "cached"
    try:
        image = get_image(item["image_url"], item["page_number"]).convert("RGB")
    except Exception as error:  # noqa: BLE001 - counted and reported, as IBM's downloader does
        return type(error).__name__
    image.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    image.save(out / "images" / f"{name}.png")
    (out / "items" / f"{name}.json").write_text(json.dumps(item))
    return "ok"


for split in ("train", "test"):
    out = Path(sys.argv[1]) / split
    for folder in ("images", "items"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    items = parquets_to_jsons(datasets.load_dataset("ibm-research/KVP10k", split=split))
    with ThreadPoolExecutor(
        32
    ) as pool:  # network-bound; dead hosts wait out the timeout
        counts = Counter(pool.map(partial(fetch, out), items))
    print(split, len(items), dict(counts), flush=True)
