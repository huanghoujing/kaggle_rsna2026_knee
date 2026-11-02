#!/usr/bin/env bash
# The competition data (docs/data.md): download the zip (265 GB; Kaggle login needed), extract it, the DICOM header
# table, the image cache (one uint8 volume per series, canonical orientation, 0.5 mm pixels) and the series geometry.
# Every step can be run again: finished parts are skipped.
#   bash pipeline/1_data.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p data/raw data/meta
ZIP=data/raw/rsna-knee-abnormality-detection.zip
[ -f "$ZIP" ] || .venv/bin/kaggle competitions download rsna-knee-abnormality-detection -p data/raw
$PY -m knee.dataloading.stream_unzip --zip "$ZIP" --out data/raw/extracted
$PY -m knee.dataloading.dicom_meta --root data/raw/extracted/train_series --out data/meta/train_dicom_meta.parquet
$PY -m knee.dataloading.build_cache --split train --version v1
$PY -m knee.dataloading.series_geometry --split train
