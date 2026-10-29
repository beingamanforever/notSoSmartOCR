"""Where does page-preview time go before any GPU or model work?

Runs `pipeline._prepare_pages` (rendering, sidecars, PNG canonicalization) the way
`demo._render_previews` calls it, cold then warm, and reports pdftoppm alone. CPU
only: never loads a model or touches the GPU.

    profile_ocr_stages.py PDF_OR_IMAGE [--repeat 3]
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ocr_pipeline.pipeline import DEFAULT_PDF_DPI, _prepare_pages, _source_kind  # noqa: E402


def _run_once(source: Path) -> float:
    with tempfile.TemporaryDirectory(prefix="ocr-profile-") as raw_dir:
        started = time.perf_counter()
        _prepare_pages(
            source, _source_kind(source), Path(raw_dir), DEFAULT_PDF_DPI, "pdftoppm"
        )
        return time.perf_counter() - started


def _pdftoppm_alone(source: Path) -> float:
    with tempfile.TemporaryDirectory(prefix="ocr-profile-pdftoppm-") as raw_dir:
        started = time.perf_counter()
        subprocess.run(
            [
                "pdftoppm",
                "-cropbox",
                "-r",
                str(DEFAULT_PDF_DPI),
                "-png",
                str(source),
                str(Path(raw_dir) / "page"),
            ],
            capture_output=True,
            check=True,
        )
        return time.perf_counter() - started


def _profiled_run(source: Path) -> tuple[float, str]:
    profiler = cProfile.Profile()
    with tempfile.TemporaryDirectory(prefix="ocr-profile-cprofile-") as raw_dir:
        started = time.perf_counter()
        profiler.enable()
        _prepare_pages(
            source, _source_kind(source), Path(raw_dir), DEFAULT_PDF_DPI, "pdftoppm"
        )
        profiler.disable()
        elapsed = time.perf_counter() - started
    buffer = io.StringIO()
    pstats.Stats(profiler, stream=buffer).sort_stats("cumulative").print_stats(20)
    return elapsed, buffer.getvalue()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args(argv)

    if shutil.which("pdftoppm") is None and _source_kind(args.source) == "pdf":
        print("pdftoppm not found; run this on a host that has it.")
        return 1

    # The first call in the process pays for every lazy import (docling, PDFium, ...);
    # profile exactly that call so the cumulative breakdown explains the cold cost.
    cold_elapsed, profile_text = _profiled_run(args.source)
    print(f"cold _prepare_pages (profiled): {cold_elapsed:.3f}s")

    warm_times = [_run_once(args.source) for _ in range(args.repeat)]
    print(f"warm _prepare_pages (x{args.repeat}): " + ", ".join(f"{t:.3f}s" for t in warm_times))
    print(f"warm mean: {sum(warm_times) / len(warm_times):.3f}s")
    if warm_times and min(warm_times) > 0:
        print(f"cold / warm-min ratio: {cold_elapsed / min(warm_times):.1f}x")

    if _source_kind(args.source) == "pdf":
        pdftoppm_times = [_pdftoppm_alone(args.source) for _ in range(args.repeat)]
        print(
            "pdftoppm alone (x%d): " % args.repeat
            + ", ".join(f"{t:.3f}s" for t in pdftoppm_times)
        )

    print("\nTop 20 cumulative functions in the cold call:")
    print(profile_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
