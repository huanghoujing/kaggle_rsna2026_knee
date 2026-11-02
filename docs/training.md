# Training

`bash pipeline/5_train.sh configs/<name>.yaml [key=value ...]` trains one config on all training studies on one GPU
(`CUDA_VISIBLE_DEVICES`), resumes an interrupted run when called again, and then writes the test-time copy of the model
without its joint transformer branch. It needs the competition cache (docs/data.md) and the release files
(`pipeline/2_release.sh`: labels, knee sides, joint boxes, folds and, for v76, the weights of single model 8).

| run | config | start | labels | epochs | time (one RTX PRO 6000) | our LB (public / private) |
|---|---|---|---|---|---|---|
| v76 (single model 9) | `r50rad_288_spatial_os10_kd56_joint3d_orient_ft8_pl5m_bksyn_gold2_ft8_kd70f_ft3` | `data/models/single_model_8_*.pt` | `targets_kd70f` + the 58 expert studies | 3 | 22 min | 0.962 / 0.950 |
| ablation D | `r50rad_288_spatial_os10_kd70fx_joint3d_orient_e48` | RadImageNet ResNet-50 | `targets_kd70f_x` + the 58 expert studies | 48 | 5.8 h | 0.966 / 0.955 |

Both use the same model and augmentation (random flips on all three axes, any-angle rotation, the nnU-Net-style
intensity and spatial transforms on the GPU, series dropout 0.1), the orientation head (loss weight 0.1) and the joint
transformer (2 layers over the stage-3 cells inside the joint box + 10 mm, up to 12,000 tokens). AdamW, cosine schedule
with 1 warm-up epoch, EMA 0.998, bf16, batch 6 studies, gradient clip 2, rarity sampling (studies with rare findings are
drawn more often; the expert studies at weight 0.8). The fine-tune uses a new AdamW at 0.25 x the first-stage rates
(backbone 2.5e-5, head 2.5e-4). The loss is binary cross-entropy against the soft labels (the expert labels are 0 / 1).

## Outputs

`experiments/MM_DD_HH_mm_<config name>_f-1/`:

| file | content |
|---|---|
| `config.yaml` | the config as run (with the overrides) |
| `steps.csv`, `log.csv`, `curves.png` | the training loss per step; the validation loss and the AUC of each finding on the 58 expert studies every `train.eval_every` epochs (`gold_auc`); the plots |
| `checkpoints/last.pt` | for resuming (saved every `train.ckpt_minutes` and at each epoch end) |
| `checkpoints/final.pt` | the EMA weights at the end, with the joint transformer: the start of a further fine-tune (`train.init_from`) |
| `checkpoints/final_nojoint.pt` | the same without the joint branch (`knee.submission.drop_joint`): the file that goes to Kaggle |
| `gold.parquet` | the final predictions of the 58 expert studies (with the joint branch, the training-time setting) |

## The gold-58 check

`bash pipeline/6_gold58.sh <config name>_f-1` (or `... data/models/<file>.pt`) predicts the 58 expert-labelled
studies as the Kaggle path does: `final_nojoint.pt`, 16 windows per series, one view, no augmentation, and prints the
AUC of each finding and the macro AUC (`knee.eval.gold58_check`). These studies are in training, so the number is
in-sample: it checks that the model and the path work, it does not compare models. Expected values of the released
weights: v76 0.9754, ablation D 0.9928. The gold-58 AUC does not follow the leaderboard closely (our Kaggle writeup,
section "External data"); we selected models on an external set that this repository does not include.

## Checks before a long run

A short run of the whole path (one epoch on 24 studies plus the 58 expert studies, a few minutes):

```bash
bash pipeline/5_train.sh configs/r50rad_288_spatial_os10_kd70fx_joint3d_orient_e48.yaml train.epochs=1 train.max_train_studies=24 train.eval_every=1
bash pipeline/6_gold58.sh r50rad_288_spatial_os10_kd70fx_joint3d_orient_e48_f-1
```

Delete the run folder afterwards, or the next call of `5_train.sh` for that config will report it as finished.

Run-to-run noise: GPU kernels are not bit-exact and the sampling is random (`train.seed` fixes the draw, not the
kernels), so a repeated run lands within about 0.001-0.002 of the LB scores above.
