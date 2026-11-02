#!/usr/bin/env bash
# The tissue segmentation model (docs/tissue_model.md): SKM-TEA download and preparation, training, then the tables it
# makes on the competition studies: the compartments found in every series, the study box of each compartment and the
# knee joint box of each study (the joint transformer's region). Needs the competition cache (pipeline/1_data.sh) and
# data/meta/laterality_train.parquet (pipeline/2_release.sh). With the release files every part is optional.
#   REDIVIS_API_TOKEN=<token> bash pipeline/4_tissue_model.sh      # a Redivis account with access to aimi.skm_tea
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export NUMPY_MADVISE_HUGEPAGE=0
if [ ! -f data/cache/v1/index_skm_tea.parquet ]; then
    [ -n "${REDIVIS_API_TOKEN:-}" ] || { echo "set REDIVIS_API_TOKEN (Redivis; the Stanford AIMI dataset aimi.skm_tea)" >&2; exit 1; }
    $PY -m knee.dataloading.skm_tea download
    $PY -m knee.dataloading.skm_tea prepare
fi
if [ ! -f data/models/skm_seg3_r50rad_288.pt ]; then
    run=$(ls -d experiments/*_seg_skm3_r50rad_288 2>/dev/null | tail -1 || true)
    if [ -n "$run" ] && [ -f "$run/checkpoints/last.pt" ]; then
        $PY -m knee.training.seg_pretrain --resume "$run"
    else
        $PY -m knee.training.seg_pretrain --config configs/seg_skm3_r50rad_288.yaml
    fi
fi
$PY -m knee.dataloading.tissue_rois --split train                 # -> data/meta/tissue_rois_train.parquet
$PY -m knee.dataloading.tissue_rois --split train --merge         # -> data/meta/tissue_rois_train_study.parquet
$PY -m knee.dataloading.tissue_rois --split train --joint_boxes   # -> data/meta/joint_box_train.parquet
