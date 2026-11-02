#!/usr/bin/env bash
# Environment of the pipeline (source env.sh): data folder, GPU, main uv environment.
# Every value can be overridden before sourcing.
#   data/raw/                    the competition zip and its extraction (step 1)
#   data/meta/                   DICOM header tables (step 2)
#   data/cache/v1/               canonical uint8 volumes of every series (step 2)
#   data/labels/                 the training label tables (release v76-data; docs/labels.md)
#   data/models/                 the released model weights (release v76-data)
#   experiments/                 training runs, MM_DD_HH_mm_<name>_f<fold>/ (docs/training.md)
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export KNEE_DATA="${KNEE_DATA:-$REPO/data}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"     # the GPU of every step (one GPU is enough)
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "$REPO/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$REPO/.venv/bin/activate"
fi
