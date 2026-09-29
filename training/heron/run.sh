#!/usr/bin/env bash
# Usage: run.sh prep|train|select|eval  (ROOT holds data/, prepared images and labels)
set -e
cd "$(dirname "$0")"
ROOT=${ROOT:-/opt/dlami/nvme/aman/heron-ft}
W=/opt/dlami/nvme/aman/ocr-audit-20260927
export HF_HOME=$HOME/.cache/huggingface HF_HUB_OFFLINE=1 TMPDIR=/opt/dlami/nvme/aman/tmp
export SPATIALINDEX_C_LIBRARY=$(ls $W/pydeps/rtree.libs/libspatialindex-*.so | head -1)
export PYTHONPATH=$(pwd):$ROOT/deps:$W/src:$W/pydeps
PY=/home/ubuntu/aman/ocr-demo/.venv-fp/bin/python
case $1 in
  prep) $PY prep_signverod.py $ROOT && $PY prep_doclaynet.py $ROOT && $PY prep_naf.py $ROOT && $PY prep_kvp10k.py $ROOT ;;
  train) $PY train_heron.py --train $ROOT/labels_signverod_train.json=$ROOT/signverod $ROOT/labels_doclaynet_train.json=$ROOT/doclaynet \
           $ROOT/labels_naf_train.json=$ROOT/naf $ROOT/labels_naf_train.json=$ROOT/naf $ROOT/labels_naf_train.json=$ROOT/naf \
           $ROOT/labels_kvp10k_train.json=$ROOT/kvp10k \
           --eval $ROOT/labels_naf_valid.json=$ROOT/naf --output $ROOT/runs/ft-v4 "${@:2}" ;;
  # Checkpoints are chosen on validation splits only; `eval` (test) runs once, on the chosen one.
  select) $PY eval_heron.py --model ${2:-original} --eval $ROOT/labels_naf_valid.json=$ROOT/naf $ROOT/labels_signverod_val.json=$ROOT/signverod \
           $ROOT/labels_doclaynet_val.json=$ROOT/doclaynet --output $ROOT/select_${3:-original}.json ;;
  # Checkbox fine-tune: synthetic pen marks on CommonForms plus NAF's real ticked boxes (box
  # plus paired label, as Heron draws them); every other class from Heron itself.
  prep-checkboxes) HF_HUB_OFFLINE=0 $PY prep_marks.py $ROOT ${2:-3000} && $PY prep_naf.py $ROOT checkboxes ;;
  train-checkboxes) $PY train_heron.py --original-classes --decoder-only --tile-crops 550 \
           --train $ROOT/labels_marks_train.json=$ROOT/marks $ROOT/labels_naf_checkboxes_train.json=$ROOT/naf \
           --eval $ROOT/labels_marks_valid.json=$ROOT/marks --epochs 4 --lr 1e-5 --output $ROOT/runs/checkboxes-v1 "${@:3}" ;;
  eval) $PY eval_heron.py --model ${2:-original} --eval $ROOT/labels_naf_test.json=$ROOT/naf $ROOT/labels_signverod_val.json=$ROOT/signverod \
           $ROOT/labels_doclaynet_test.json=$ROOT/doclaynet $ROOT/labels_kvp10k_test.json=$ROOT/kvp10k --output $ROOT/eval_${3:-original}.json ;;
esac
