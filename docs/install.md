# Install

```bash
bash install.sh        # Python 3.12, PyTorch 2.13.0 (CUDA 12.9 build), timm, pydicom + JPEG codecs, this package -> ./.venv
source env.sh          # activates .venv, sets KNEE_DATA (the data folder, default ./data) and CUDA_VISIBLE_DEVICES (default 0)
```

`install.sh` needs [uv](https://docs.astral.sh/uv/). `CUDA=cu130 bash install.sh` picks another PyTorch build;
`VENV=/path bash install.sh` another environment folder; `REQUIREMENTS=requirements-lock.txt bash install.sh` installs
the exact package versions of our environment (`requirements-lock.txt`, from `uv pip freeze`). The script verifies the imports and runs one matrix product on
the GPU. Every `pipeline/*.sh` script uses `.venv/bin/python` and runs on the GPU named by `CUDA_VISIBLE_DEVICES`.

## Accounts and downloads

| need | for | how |
|---|---|---|
| Kaggle login (`~/.kaggle/kaggle.json` or `~/.kaggle/access_token`) | the competition data (`pipeline/1_data.sh`), the datasets and notebook of a submission (`pipeline/7_submit.sh`) | https://www.kaggle.com/settings |
| Hugging Face (no login) | the RadImageNet ResNet-50 weights, downloaded once by `huggingface_hub` into `~/.cache/huggingface` | automatic |
| GitHub CLI `gh`, logged in | the release files while this repository is private (`pipeline/2_release.sh` downloads with curl first and falls back to `gh`) | https://cli.github.com |
| Redivis account with access to the Stanford AIMI dataset `aimi.skm_tea` | the tissue segmentation data (`pipeline/4_tissue_model.sh`); optional: the release holds the model and its tables | https://redivis.com/datasets (SKM-TEA) |

## The machine we used

Ubuntu 26.04 LTS, NVIDIA driver 595.91, Python 3.12.13, PyTorch 2.13.0 (CUDA 12.9 build), torchvision 0.28.0,
timm 1.0.30, pydicom 3.0.2, numpy 2.5.4, pandas 3.0.6 (all versions: `requirements-lock.txt`). One run uses one GPU. Training v76 (3 epochs) took 22 minutes, ablation D (48 epochs) 5.8 hours and the left / right
knee classifier (6 epochs on the middle slices) under a minute on an RTX PRO 6000
(96 GB; about 50 GB of GPU memory at batch size 6; `train.batch_size` and `train.eval_batch_size` can be lowered). The
data loader uses 16 processes (`train.num_workers`). Disk: the competition zip 247 GB, the extracted DICOM files 533 GB,
the image cache 294 GB. The cache build and the DICOM header scan use one third of the CPUs (`--workers`).

## The Kaggle notebook

The submission runs on Kaggle's GPU notebook image (two Tesla T4, internet off; in our runs Python 3.13 and PyTorch
2.11.0 with CUDA 12.8, as the notebook log prints at the start). The notebook installs only the DICOM codec wheels
(docs/submission.md); everything else comes from the image.
