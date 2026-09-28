#!/usr/bin/env bash
# Fetch the official transformers object-detection example matching the installed version,
# and install the training-only packages into a side directory.
set -e
cd "$(dirname "$0")"
PY=${PY:-/home/ubuntu/aman/ocr-demo/.venv-fp/bin/python}
VERSION=$($PY -c "import transformers; print(transformers.__version__)")
curl -sfL -o run_object_detection.py "https://raw.githubusercontent.com/huggingface/transformers/v$VERSION/examples/pytorch/object-detection/run_object_detection.py"
DEPS=${DEPS:-/opt/dlami/nvme/aman/heron-ft/deps}
UV="$HOME/.local/bin/uv pip install --python $PY --target $DEPS --no-deps"
$UV "albumentations==2.0.8" "albucore==0.0.24" torchmetrics lightning-utilities pycocotools "pydantic==2.13.5" "pydantic-core==2.46.5" \
    typing-inspection annotated-types simsimd stringzilla "accelerate>=1.1.0" psutil
