# Submission

The Kaggle notebook runs `knee.inference.predict_efficient` on the hidden test set: the DICOM headers of every file
and the pixels of the 16 window slices of each series are read in loader workers (12 file threads each), the GPU does
the orientation, scaling, resampling and crop, one process per T4, fp16, no test-time augmentation. v76 took about
5 minutes for the ~1,300 test studies on the two T4 GPUs of a Kaggle notebook.

## The public notebook and datasets

| Kaggle | content |
|---|---|
| notebook [houjinghuang/rsna-knee-v76-inference](https://www.kaggle.com/code/houjinghuang/rsna-knee-v76-inference) | our final submission (v76): `kaggle/notebooks/knee-infer/infer.py` installs the codec wheels offline, runs the predictor, writes `submission.csv` |
| notebook [houjinghuang/rsna-knee-ablation-d-inference](https://www.kaggle.com/code/houjinghuang/rsna-knee-ablation-d-inference) | the same notebook with the ablation D weights (an after-deadline run, public LB 0.966 / private 0.955) |
| dataset `houjinghuang/rsna-knee-v76-models` | `00_v76_..._nojoint.pt` (the v76 weights in float16) and `models.txt` (the test-time settings) |
| Kaggle Model [houjinghuang/rsna-knee-v76](https://www.kaggle.com/models/houjinghuang/rsna-knee-v76) | the same v76 weights as a Kaggle Model with a model card: `v76_nojoint_fp16.pt` (byte-identical to the dataset's file) and `v76_joint_fp32.pt` (the full checkpoint with the joint transformer and the orientation head, for fine-tuning) |
| dataset `houjinghuang/rsna-knee-ablation-d-models` | the ablation D weights in float16 and their `models.txt` |
| dataset `houjinghuang/rsna-knee-v76-code` | the `knee` package of this repository |
| dataset `houjinghuang/rsna-knee-v76-wheels` | pylibjpeg, pylibjpeg-libjpeg, pylibjpeg-openjpeg, python-gdcm wheels for Python 3.11-3.13 |

`models.txt` lists the checkpoints (one per line, blended with equal weight) and the settings as comment lines:
`# k_slices: 16`, `# threads: 12`, `# workers: 6`, `# tta_mirror: 0`, `# gpus: 2`.

## Your own submission

```bash
bash pipeline/0_wheels.sh                                   # once: the codec wheels (pip download)
KNEE_KAGGLE_OWNER=<your Kaggle user> KNEE_KAGGLE_PREFIX=<slug prefix> KNEE_KAGGLE_KERNEL=<notebook slug> \
    bash pipeline/7_submit.sh "<version message>" <checkpoint.pt>          # package, upload, push, wait for the run
SUBMIT=1 ... bash pipeline/7_submit.sh ...                                  # also submit the finished run
PUBLIC=1 ... bash pipeline/7_submit.sh ...                                  # new datasets and the notebook public
```

The script packages the checkpoint with the v76 settings (`knee.submission.package --fp16 --k_slices 16
--threads 12 --workers 6 --gpus 2 --no_tta_mirror`), writes the notebook metadata for your account, creates the three
datasets (or uploads new versions), pushes the notebook (Kaggle runs it on the test set), waits for the run, and with
`SUBMIT=1` submits it. Without the environment variables it addresses our public datasets and notebook. A second
notebook can share the code and wheels datasets of the first: `KNEE_KAGGLE_CODE_PREFIX=<their prefix>` (our ablation D
notebook: `KNEE_KAGGLE_PREFIX=rsna-knee-ablation-d KNEE_KAGGLE_CODE_PREFIX=rsna-knee-v76
KNEE_KAGGLE_KERNEL=rsna-knee-ablation-d-inference PUBLIC=1 bash pipeline/7_submit.sh "ablation D"
data/models/ablation_D_r50rad_288_spatial_os10_kd70fx_joint3d_orient_e48_nojoint.pt`).

## Predicting locally

The same predictor on DICOM files of your own, here the training series:

```bash
.venv/bin/python -m knee.inference.predict_efficient --series_root data/raw/extracted/train_series \
    --series_csv data/raw/extracted/train_series.csv --studies_csv <csv with a StudyInstanceUID column> \
    --checkpoints data/models/v76_r50rad_288_spatial_os10_kd56_joint3d_orient_ft8_pl5m_bksyn_gold2_ft8_kd70f_ft3_nojoint.pt \
    --out predictions.csv --k_slices 16 --threads 12 --workers 6
```

The cache path (`pipeline/6_gold58.sh`) and the DICOM path agree to about 0.004 in probability on the same studies
(the DICOM path reads only the 16 slices and scales intensities from them).
