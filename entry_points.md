# Entry points

Every step is a shell script in `pipeline/` that calls one or more `python -m knee...` modules of this repository; the
scripts run from the repository root with the environment of `install.sh` (`.venv/bin/python`). SETTINGS.json names the
folders they read and write (README.md, section "Settings").

| step | command | reads | writes |
|---|---|---|---|
| install | `bash install.sh && source env.sh` | requirements.txt (or requirements-lock.txt) | `.venv/` |
| prepare data | `bash pipeline/1_data.sh` | the competition zip (Kaggle login) | `data/raw/`, `data/cache/v1/`, `data/meta/` |
| release files | `bash pipeline/2_release.sh` | the GitHub release `v76-data` | `data/models/`, `data/labels/`, `data/meta/`, `data/folds_v1.csv` |
| knee side (optional) | `bash pipeline/3_knee_side.sh` | the cache, `data/meta/laterality_train.parquet` | `data/models/laterality_r18.pt`, `data/meta/laterality_train.parquet` |
| tissue model (optional) | `REDIVIS_API_TOKEN=<token> bash pipeline/4_tissue_model.sh` | SKM-TEA (Redivis), the cache | `data/cache/v1/skm_tea/`, `data/models/skm_seg3_r50rad_288.pt`, the tissue and joint-box tables in `data/meta/` |
| train v76 | `bash pipeline/5_train.sh configs/r50rad_288_spatial_os10_kd56_joint3d_orient_ft8_pl5m_bksyn_gold2_ft8_kd70f_ft3.yaml` | the cache, the release files | `experiments/<time>_<config>_f-1/` (checkpoints `final.pt`, `final_nojoint.pt`) |
| train ablation D | `bash pipeline/5_train.sh configs/r50rad_288_spatial_os10_kd70fx_joint3d_orient_e48.yaml` | the same | the same |
| check (predict the 58 expert studies) | `bash pipeline/6_gold58.sh <config name>_f-1` | the run folder, the cache | `gold58_k16.csv` in the run folder |
| predict a new test set locally | `python -m knee.inference.predict_efficient --series_root <dir>/test_series --series_csv <dir>/test_series.csv --checkpoints <final_nojoint.pt> --out predictions.csv --k_slices 16 --threads 12 --workers 6` | DICOM files | `predictions.csv` |
| predict on Kaggle (the submission) | `bash pipeline/7_submit.sh "<message>" <final_nojoint.pt>` (`SUBMIT=1` to submit) | the checkpoint | the Kaggle datasets, the notebook run, `submission.csv` |

Training and prediction are separate: the prediction scripts read only `final_nojoint.pt` (or the released weights)
and never touch the training data.
