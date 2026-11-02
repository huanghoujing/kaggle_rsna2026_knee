# Data

`bash pipeline/1_data.sh` does the four steps below; each can be run again (finished parts are skipped).

| step | command | output |
|---|---|---|
| download | `kaggle competitions download rsna-knee-abnormality-detection -p data/raw` | `data/raw/rsna-knee-abnormality-detection.zip` (247 GB) |
| extract | `python -m knee.dataloading.stream_unzip --zip <zip> --out data/raw/extracted` | `data/raw/extracted/train_series/<study>/<series>/*.dcm`, `train.csv`, `train_series.csv` (533 GB); the script can follow a zip that is still downloading |
| DICOM headers | `python -m knee.dataloading.dicom_meta --root data/raw/extracted/train_series --out data/meta/train_dicom_meta.parquet` | one row per slice: orientation, position, spacing, laterality tag, description, scanner |
| image cache | `python -m knee.dataloading.build_cache --split train --version v1` | `data/cache/v1/train/<series>.npy` and `data/cache/v1/index_train.parquet` (294 GB) |
| series geometry | `python -m knee.dataloading.series_geometry --split train` | `data/meta/series_geometry_train.parquet`: the image center and the patient directions of the axes of every series (the joint transformer's positions) |

The competition data: 4,407 training studies with 24,371 DICOM series (sagittal, coronal, axial; with or without fat
suppression; about 5.5 series per study), a radiology report for every study and the 12 expert labels for 58 of them
(`train.csv`); a hidden test set of about 1,300 studies.

## The image cache

One uint8 volume per series, the same preparation in training and in the Kaggle notebook
(`knee.dataloading.build_cache.prepare_series`; the notebook reads only the 16 slices it uses):

1. The slices sorted by their position along the slice normal (the file order is not reliable).
2. A canonical display frame per plane from the DICOM orientation: sagittal columns anterior to posterior, rows
   superior to inferior, slices right to left; coronal columns right to left, rows superior to inferior, slices
   anterior to posterior; axial columns right to left, rows anterior to posterior, slices superior to inferior.
3. Resampled to 0.5 mm pixels and center-cropped or zero-padded to 352 x 352 px (a 176 mm field). The model crops
   160 mm of it and resizes to 288 px.
4. Slices thinner than 2 mm are averaged in groups to about 2 mm (773 of the 24,371 series).
5. Intensities scaled by the 0.5 / 99.5 percentiles of the series to 0-255.

Left knees are kept as acquired (the v76 model is never told the side). `index_train.parquet` holds one row per series
(plane, depth, slice spacing, native size, the image center, laterality tag, description, scanner).

## Folder layout

```
data/raw/                   the zip and its extraction (train_series/, train.csv, train_series.csv)
data/meta/                  DICOM header table, series geometry, knee sides (docs/knee_side.md), tissue tables, joint boxes
data/cache/v1/              the image cache (train/, skm_tea/ for the tissue model) and its index tables
data/labels/                the label tables (docs/labels.md)
data/models/                the released weights (docs/release.md); the tissue model and the knee-side classifier
data/folds_v1.csv           the folds (fold -1 = the 58 expert studies)
experiments/                training runs (docs/training.md)
kaggle/datasets/            the Kaggle dataset folders made by knee.submission.package (git-ignored contents)
```

`KNEE_DATA` (env.sh) moves `data/` elsewhere.
