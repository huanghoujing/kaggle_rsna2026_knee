# RSNA 2026 Knee Abnormality Detection: the v76 solution (efficiency track)

Code, weights and tables to reproduce our efficiency-track submission to the Kaggle competition
[RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) (12 findings
per knee MRI study, macro AUC): **v76**, rank 3 of 6,299 teams on the efficiency track. One model of 26.4 M
parameters; public / private LB AUC 0.962 / 0.950; 285 s for the ~1,300 test studies on the two T4 GPUs of a Kaggle
notebook. The method is described in our writeup on the competition's writeups page. The v76 weights are also a
[Kaggle Model](https://www.kaggle.com/models/houjinghuang/rsna-knee-v76) with a model card.

The model: a ResNet-50 with RadImageNet weights encodes 16 slice windows of every series of a study; 12 label
queries (one per finding) attend over all windows. In training only: random flips on all three axes and any-angle
rotation with an orientation head (the model is never told whether a knee is left or right), and a 3D joint
transformer over the knee-joint region (dropped at test time). Labels: soft labels from LLM readings of the reports and
from image models, then the predictions of our ensembles, with the 58 expert-labelled studies.

What you can do with this repository:

| goal | what it needs | document |
|---|---|---|
| predict the test set on Kaggle with the released weights (the public notebooks of v76 and of ablation D) | nothing to train | [docs/submission.md](docs/submission.md) |
| fine-tune single model 8 into v76 (3 epochs on the predictions of our v70f ensemble) | the competition data, the release files | [docs/training.md](docs/training.md) |
| train ablation run D, the v76 recipe in one stage of 48 epochs from RadImageNet (public LB 0.966 / private 0.955) | the same | [docs/training.md](docs/training.md) |
| train the tissue segmentation model and rebuild the joint boxes | SKM-TEA (Stanford AIMI, registration) | [docs/tissue_model.md](docs/tissue_model.md) |
| train the left / right knee classifier (ResNet-18) and rebuild the knee-side table | the competition data, the release files | [docs/knee_side.md](docs/knee_side.md) |

## Quick start

```bash
bash install.sh && source env.sh                  # uv environment .venv (PyTorch, timm, pydicom + codecs); docs/install.md
bash pipeline/1_data.sh                           # competition data: download, extract, image cache, header tables; docs/data.md
bash pipeline/2_release.sh                        # release v76-data: weights, label tables, knee sides, joint boxes, folds; docs/release.md
bash pipeline/5_train.sh configs/r50rad_288_spatial_os10_kd56_joint3d_orient_ft8_pl5m_bksyn_gold2_ft8_kd70f_ft3.yaml   # v76 (3 epochs)
bash pipeline/5_train.sh configs/r50rad_288_spatial_os10_kd70fx_joint3d_orient_e48.yaml                                # ablation D (48 epochs)
bash pipeline/6_gold58.sh r50rad_288_spatial_os10_kd56_joint3d_orient_ft8_pl5m_bksyn_gold2_ft8_kd70f_ft3_f-1           # the 58 expert studies
bash pipeline/7_submit.sh "v76" experiments/*_r50rad_288_spatial_os10_kd56_joint3d_orient_ft8_pl5m_bksyn_gold2_ft8_kd70f_ft3_f-1/checkpoints/final_nojoint.pt
```

Optional: `REDIVIS_API_TOKEN=<token> bash pipeline/4_tissue_model.sh` trains the tissue segmentation model and remakes
the joint-box tables that the release already holds; `bash pipeline/3_knee_side.sh` trains the left / right knee
classifier and remakes the knee-side table.

## Documents

| document | content |
|---|---|
| [docs/install.md](docs/install.md) | the environment, the machine we used, the accounts you need |
| [docs/data.md](docs/data.md) | the competition data and the image cache |
| [docs/labels.md](docs/labels.md) | the label tables: what they are, how they were made, where they come from |
| [docs/knee_side.md](docs/knee_side.md) | the left / right knee classifier and the knee-side table |
| [docs/tissue_model.md](docs/tissue_model.md) | the tissue segmentation model (SKM-TEA) and the joint boxes |
| [docs/training.md](docs/training.md) | training v76 and ablation D, the outputs, the gold-58 check |
| [docs/submission.md](docs/submission.md) | packaging, the Kaggle datasets and notebook, submitting, predicting locally |
| [docs/release.md](docs/release.md) | the files of release `v76-data` and their licences |
| [configs/README.md](configs/README.md) | the three configs |

## Code

| package | content |
|---|---|
| `knee/dataloading/` | zip extraction, DICOM headers and pixels, canonical orientation, the image cache, series geometry, the study dataset, GPU augmentation, folds, knee side, SKM-TEA, the tissue tables |
| `knee/model/` | the 2.5D study model (`knee_net.py`: encoder, label queries, orientation head, joint transformer), the left / right knee classifier and its training (`laterality_cnn.py`) |
| `knee/training/` | resumable training (`train.py`), the tissue segmentation model and its training (`seg_pretrain.py`) |
| `knee/labels/` | the label table of the top teams' shared sets (`external_llm.py`, for ablation D) |
| `knee/eval/` | the competition metric, the gold-58 check |
| `knee/inference/` | prediction from DICOM files as on Kaggle (`predict_efficient.py`: partial reads, GPU preprocessing) |
| `knee/submission/` | the test-time copy without the joint transformer (`drop_joint.py`), the Kaggle package (`package.py`) |
| `kaggle/` | the notebook (`notebooks/knee-infer/infer.py`) and the dataset metadata |
| `pipeline/` | the steps as shell scripts, numbered in order |
| `figures/` | the figures of our Kaggle writeup |

## Settings, environment and assumptions

- [SETTINGS.json](SETTINGS.json) names the folders: the data folder (`DATA_DIR`, moved with `KNEE_DATA` in env.sh)
  and inside it the competition files, the image cache, the header and side tables, the labels, the models and the
  release download; the run folders (`EXPERIMENTS_DIR`), the logs and the Kaggle dataset folders. The configs name the
  files inside these folders (configs/README.md). [entry_points.md](entry_points.md) lists the commands of every step
  and what each reads and writes; [directory_structure.txt](directory_structure.txt) the folders.
- Hardware and software: one GPU (about 50 GB of memory at the default batch size), Ubuntu 26.04, Python 3.12, PyTorch
  2.13 (CUDA 12.9); the exact package versions are in [requirements-lock.txt](requirements-lock.txt); the Kaggle
  notebook runs on Kaggle's GPU image with two T4 GPUs (docs/install.md).
- Assumptions: the competition data as Kaggle provides it (`train_series/<study>/<series>/*.dcm`, `train_series.csv`,
  `train.csv`); the release files of `v76-data` for the labels, knee sides, joint boxes, folds and starting weights;
  about 1.1 TB of disk for the zip, the extracted files and the cache.
- Side effects: `pipeline/1_data.sh` writes under `data/` and uses a third of the CPUs; `3_knee_side.sh` and
  `4_tissue_model.sh` overwrite tables in `data/meta/` in place; `5_train.sh` creates a run folder under
  `experiments/`; `7_submit.sh` creates or updates datasets and a notebook on your Kaggle account (public only with
  `PUBLIC=1`) and submits only with `SUBMIT=1`.

## Licences and data sources

- Code: CC BY-NC 4.0 ([LICENSE](LICENSE)), the winner licence of the competition rules.
- Model weights: every encoder starts from the RadImageNet ResNet-50 (Mei X, et al., Radiology: AI 2022; weights
  `convergedmachine/RadImagenet` on Hugging Face, CC BY-NC-SA 4.0), so the released weights are shared under
  CC BY-NC-SA 4.0 (non-commercial).
- The tissue segmentation model was trained on SKM-TEA (Desai AD, et al. SKM-TEA [Dataset]. Center for Artificial
  Intelligence in Medicine and Imaging. https://doi.org/10.71718/2ghb-nv62; the paper: Desai AD, et al., NeurIPS
  Datasets and Benchmarks 2021; Stanford AIMI research use agreement). The dataset itself is not redistributed.
  This research used data provided by the Stanford Center for Artificial Intelligence in Medicine and Imaging
  (AIMI). AIMI curated a publicly available imaging data repository containing clinical imaging and data from
  Stanford Health Care, Stanford Medicine Children's Health, University HealthCare Alliance, and Packard Children's
  Health Alliance clinics, provisioned for research use by the Stanford Medicine Research Data Repository (STARR).
- The label tables derive from the competition's radiology reports and images (the competition rules) and, for
  `targets_kd70f_x`, from label sets other teams shared on Kaggle (docs/labels.md lists them with their licences).
