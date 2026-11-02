# Configs

| file | what it trains | start weights | labels | epochs | result |
|---|---|---|---|---|---|
| `r50rad_288_spatial_os10_kd56_joint3d_orient_ft8_pl5m_bksyn_gold2_ft8_kd70f_ft3.yaml` | v76, the efficiency-track submission | single model 8 (release `single_model_8_*.pt`) | `targets_kd70f` + the 58 expert studies | 3 | public LB 0.962 / private 0.950 |
| `r50rad_288_spatial_os10_kd70fx_joint3d_orient_e48.yaml` | ablation run D: the v76 recipe in one stage | RadImageNet ResNet-50 | `targets_kd70f_x` + the 58 expert studies | 48 | public LB 0.966 / private 0.955 |
| `seg_skm3_r50rad_288.yaml` | the tissue segmentation model (SKM-TEA cartilage and menisci, three planes) | RadImageNet ResNet-50 | SKM-TEA masks | 24 | `data/models/skm_seg3_r50rad_288.pt` |

The config names are the run names of the original experiments (the run folder `experiments/MM_DD_HH_mm_<name>_f-1/`
and the checkpoint names carry them); the comments at the top of each file say what the run is. Keys: `data` (cache,
labels, folds), `input` (windows per series, series per study, augmentation), `model` (encoder, label queries, joint
transformer, orientation head), `train` (epochs, learning rates, EMA, sampling, the expert studies). Every key can be
overridden on the command line: `--set train.epochs=1`.
