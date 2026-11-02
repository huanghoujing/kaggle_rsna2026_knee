#!/usr/bin/env bash
# The released files (docs/release.md): model weights, label tables, knee sides, joint boxes, tissue tables, folds.
# Downloaded into data/release/<tag>/ (curl; for a private repository the logged-in GitHub CLI `gh`) and copied to
# the paths the configs read (existing files are kept).
#   bash pipeline/2_release.sh [v76-data]
set -euo pipefail
cd "$(dirname "$0")/.."
TAG=${1:-v76-data}
URL=https://github.com/${KNEE_GITHUB_REPO:-huanghoujing/kaggle_rsna2026_knee}/releases/download/$TAG
R=data/release/$TAG
mkdir -p "$R" data/models data/labels data/meta
FILES="single_model_8_r50rad_288_spatial_os10_kd56_joint3d_orient_ft8_pl5m_bksyn_gold2_ft8.pt
v76_r50rad_288_spatial_os10_kd56_joint3d_orient_ft8_pl5m_bksyn_gold2_ft8_kd70f_ft3.pt
v76_r50rad_288_spatial_os10_kd56_joint3d_orient_ft8_pl5m_bksyn_gold2_ft8_kd70f_ft3_nojoint.pt
ablation_D_r50rad_288_spatial_os10_kd70fx_joint3d_orient_e48.pt
ablation_D_r50rad_288_spatial_os10_kd70fx_joint3d_orient_e48_nojoint.pt
skm_seg3_r50rad_288.pt
laterality_r18.pt
labels_targets_kd70f.parquet
labels_targets_kd70f_x.parquet
labels_targets_cx.parquet
meta_laterality_train.parquet
meta_joint_box_train.parquet
meta_tissue_rois_train.parquet
meta_tissue_rois_train_study.parquet
meta_series_geometry_train.parquet
folds_v1.csv
README.md"
for f in $FILES; do
    [ -s "$R/$f" ] && continue
    # a public repository: plain downloads; a private one (or no network to the CDN): the GitHub CLI, logged in
    curl -fL --retry 3 -o "$R/$f" "$URL/$f" 2>/dev/null || { rm -f "$R/$f"; gh release download "$TAG" -R "${KNEE_GITHUB_REPO:-huanghoujing/kaggle_rsna2026_knee}" -p "$f" -D "$R"; }
done
copy() { if [ -e "$2" ]; then echo "keep $2 (exists)"; else cp "$1" "$2"; echo "$2"; fi; }
for f in "$R"/*.pt; do copy "$f" "data/models/$(basename "$f")"; done
for f in "$R"/labels_*.parquet; do copy "$f" "data/labels/$(basename "${f#"$R"/labels_}")"; done
for f in "$R"/meta_*.parquet; do copy "$f" "data/meta/$(basename "${f#"$R"/meta_}")"; done
copy "$R/folds_v1.csv" data/folds_v1.csv
