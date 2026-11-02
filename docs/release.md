# Release v76-data

`bash pipeline/2_release.sh` downloads the files below from the GitHub release `v76-data` of this repository into
`data/release/v76-data/` and copies them to the paths the configs read (`data/models/`, `data/labels/`, `data/meta/`,
`data/folds_v1.csv`; existing files are kept).

The v76 weights are also on Kaggle, for notebooks without internet: the Model [houjinghuang/rsna-knee-v76](https://www.kaggle.com/models/houjinghuang/rsna-knee-v76) (the submitted float16 file and the full checkpoint, with a model card) and the dataset `houjinghuang/rsna-knee-v76-models` that the submission notebook loads (docs/submission.md).

| file | size | content | licence |
|---|---|---|---|
| `single_model_8_r50rad_288_spatial_os10_kd56_joint3d_orient_ft8_pl5m_bksyn_gold2_ft8.pt` | 124 MB | single model 8 (public LB 0.962 / private 0.951) with its joint transformer: the start of the v76 fine-tune | CC BY-NC-SA 4.0 |
| `v76_..._kd70f_ft3.pt` | 124 MB | v76 with its joint transformer (EMA weights, float32) | CC BY-NC-SA 4.0 |
| `v76_..._kd70f_ft3_nojoint.pt` | 106 MB | v76 as submitted: without the joint transformer | CC BY-NC-SA 4.0 |
| `ablation_D_r50rad_288_spatial_os10_kd70fx_joint3d_orient_e48.pt`, `..._nojoint.pt` | 124 + 106 MB | ablation run D (public LB 0.966 / private 0.955), with and without the joint transformer | CC BY-NC-SA 4.0 |
| `skm_seg3_r50rad_288.pt` | 99 MB | the tissue segmentation model (docs/tissue_model.md) | CC BY-NC-SA 4.0; trained on SKM-TEA (research use) |
| `laterality_r18.pt` | 45 MB | the left / right knee classifier (ResNet-18 on the middle coronal and axial slices): the side of the studies without a tag, description or report side (docs/knee_side.md) | CC BY-NC-SA 4.0 |
| `labels_targets_kd70f.parquet` (+ `.csv`) | 4,407 rows | the v76 training labels: the predictions of our v70f ensemble | competition data terms |
| `labels_targets_kd70f_x.parquet` (+ `.csv`) | 4,407 rows | the ablation D labels: 0.5 x `targets_kd70f` + 0.5 x the top teams' shared sets (docs/labels.md) | competition data terms; the shared sets' licences |
| `labels_targets_cx.parquet` (+ `.csv`) | 4,407 rows | our report label (validation reference; `is_gold` marks the 58 expert studies) | competition data terms |
| `meta_laterality_train.parquet` | 4,407 rows | the knee side of every study and its source | competition data terms |
| `meta_joint_box_train.parquet` | 4,407 rows | the knee joint box of every study (patient mm) | derived with the tissue model |
| `meta_tissue_rois_train.parquet`, `meta_tissue_rois_train_study.parquet` | 64,000 / 72,361 rows | the compartments found in every series; the study boxes projected into every series | derived with the tissue model |
| `meta_series_geometry_train.parquet` | 24,371 rows | the image center and axes of every series (also made by `pipeline/1_data.sh`) | competition DICOM headers |
| `folds_v1.csv` | 4,407 rows | the folds; fold -1 = the 58 expert studies | competition data terms |

The model weights derive from the RadImageNet ResNet-50 (CC BY-NC-SA 4.0), hence their licence. The label and side
tables are derived from the competition's reports and images and are shared for research on this competition, as the
other teams' label sets on Kaggle.

Not released: the six models of the v70f ensemble and the earlier label rounds (the chain that produced
`targets_kd70f` and single model 8), our LLM readings of the reports, and the SKM-TEA data (Stanford AIMI).
