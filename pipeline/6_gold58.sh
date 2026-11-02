#!/usr/bin/env bash
# The gold-58 check of a run or a checkpoint as the Kaggle path predicts it (docs/training.md).
#   bash pipeline/6_gold58.sh <config name>_f-1            # a run of pipeline/5_train.sh
#   bash pipeline/6_gold58.sh data/models/<file>.pt        # a released checkpoint
set -euo pipefail
cd "$(dirname "$0")/.."
export NUMPY_MADVISE_HUGEPAGE=0
case "$1" in
    *.pt) .venv/bin/python -m knee.eval.gold58_check --checkpoint "$1" ;;
    *)    .venv/bin/python -m knee.eval.gold58_check --run "$1" ;;
esac
