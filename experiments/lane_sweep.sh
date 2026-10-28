#!/usr/bin/env bash
# Falcon lane sweep on a dedicated GPU host; options and outputs: experiments/lane_sweep.py --help.
#   FALCON_PYTHON  interpreter with falcon-perception (and docTR, or pass --word-device none)
#   FALCON_ARGS    serve_falcon_layout.py flags for every lane; keep them equal to production,
#                  since batch size and KV pages change Falcon's tokens
set -euo pipefail
cd "$(dirname "$0")/.."
python="${FALCON_PYTHON:-python3}"
read -r -a lane_args <<< "${FALCON_ARGS:---dtype bfloat16 --max-batch-size 32 --n-pages 384}"
exec "$python" -u experiments/lane_sweep.py --python "$python" "$@" -- "${lane_args[@]}"
