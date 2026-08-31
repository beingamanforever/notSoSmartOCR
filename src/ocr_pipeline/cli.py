"""Command-line entry point for local OCR."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .pipeline import process_document
from .providers import TesseractReader


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract evidence-linked OCR JSON")
    parser.add_argument("input", type=Path, help="PDF or image to read")
    parser.add_argument("-o", "--output", type=Path, help="JSON output path")
    parser.add_argument("--language", default="eng", help="Tesseract language")
    parser.add_argument("--dpi", type=int, default=300, help="PDF render DPI")
    args = parser.parse_args(argv)

    result = process_document(
        args.input,
        TesseractReader(language=args.language),
        pdf_dpi=args.dpi,
    )
    payload = json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n"

    try:
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload, encoding="utf-8")
        else:
            sys.stdout.write(payload)
    except OSError as error:
        print(f"Could not write OCR result: {error}", file=sys.stderr)
        return 1

    return 0 if result.status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
