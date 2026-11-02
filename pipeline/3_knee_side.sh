#!/usr/bin/env bash
# The left / right knee classifier and the knee-side table (docs/knee_side.md). The side of each training study is the
# target of the orientation head (where "medial" points in a window); the model itself is never told the side and
# nothing at test time needs it. Needs the competition cache (pipeline/1_data.sh) and the released knee-side table
# (pipeline/2_release.sh): its tag, description and report sides are the training labels of the classifier, and its
# report sides are kept in the rebuilt table (the LLM reading of the reports is not part of this repository).
#   bash pipeline/3_knee_side.sh            # trains the classifier if data/models/laterality_r18.pt is absent, then the table
#   RETRAIN=1 bash pipeline/3_knee_side.sh  # trains it again (about 6 epochs on the middle coronal and axial slices)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export NUMPY_MADVISE_HUGEPAGE=0
if [ ! -f data/models/laterality_r18.pt ] || [ -n "${RETRAIN:-}" ]; then
    $PY -m knee.model.laterality_cnn --out data/models/laterality_r18.pt      # prints the held-out accuracy
fi
if [ ! -f data/models/laterality_r18_train_pred.parquet ]; then
    $PY -m knee.model.laterality_cnn --out data/models/laterality_r18.pt --predict_only   # the released model's predictions
fi
$PY -m knee.dataloading.laterality --out data/meta/laterality_train.parquet \
    --cnn_pred data/models/laterality_r18_train_pred.parquet            # report sides: from the existing table (default)
