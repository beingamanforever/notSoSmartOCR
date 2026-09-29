#!/usr/bin/env bash
# Fetch the Heron fine-tune sources into ROOT: DocLayNet core, SignverOD, NAF, KVP10k.
# Usage: fetch_data.sh [naf]  (naf: only NAF, for the checkbox fine-tune)
set -e
ROOT=${ROOT:-/opt/dlami/nvme/aman/heron-ft}
BASE=$(dirname "$ROOT")
HERE=$(cd "$(dirname "$0")" && pwd)
export TMPDIR=${TMPDIR:-$BASE/tmp} UV_CACHE_DIR=$BASE/tmp/uv
mkdir -p $ROOT/data/signverod $TMPDIR
fetch_naf() {
  cd $ROOT
  [ -d NAF_dataset ] || git clone -q --depth 1 https://github.com/herobd/NAF_dataset.git
  cd NAF_dataset
  if [ ! -f groups/1/007182398_00026.jpg ]; then
    url=$(curl -s https://api.github.com/repos/herobd/NAF_dataset/releases | python3 -c "import json,sys; print([a['browser_download_url'] for x in json.load(sys.stdin) for a in x['assets'] if a['name']=='labeled_images.tar.gz'][0])")
    curl -sfL -o labeled_images.tar.gz "$url"
    sed "s#/tmp#$TMPDIR#g" move_images.sh | bash > /dev/null
  fi
}
[ "$1" = naf ] && { fetch_naf; exit 0; }
cd $ROOT/data
curl -sfL -C - -o DocLayNet_core.zip https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/dax-doclaynet/1.0.0/DocLayNet_core.zip &
for f in train-00000-of-00002.parquet train-00001-of-00002.parquet; do
  [ -s signverod/$f ] || curl -sfL -o signverod/$f https://huggingface.co/datasets/ondrs/signverod/resolve/main/data/$f
done
fetch_naf
cd $BASE
[ -d KVP10k ] || git clone -q --depth 1 https://github.com/IBM/KVP10k.git
[ -x kvp-venv/bin/python ] || { $HOME/.local/bin/uv venv -q -p 3.12 kvp-venv && $HOME/.local/bin/uv pip install -q --python kvp-venv/bin/python requests pdf2image tesserocr "datasets==5.0.1" pillow numpy pandas; }
(cd KVP10k && PYTHONPATH=$BASE/KVP10k HF_HOME=$TMPDIR/hf ../kvp-venv/bin/python $HERE/download_kvp10k.py $ROOT/kvp10k)
wait
