"""parse-bench CLI with the Not So Smart OCR provider registered.

Usage (from a ParseBench checkout): uv run python run.py run nssocr --max_concurrent 3
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import nssocr_parsebench  # noqa: E402,F401  registers the provider and pipeline
from parse_bench.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
