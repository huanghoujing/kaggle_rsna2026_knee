# Labels

All tables have one row per training study (`StudyInstanceUID`) and the 12 findings as columns (`ACL`, `MCL`,
`Medial Meniscus`, `Lateral Meniscus`, `Medial OA`, `Lateral OA`, `PF OA`, `Effusion`, `Synovitis`, `Baker's`,
`Contusion`, `Fracture`), as probabilities 0-1 (soft targets). `pipeline/2_release.sh` puts them into `data/labels/`
and `data/meta/`. The 58 expert-labelled studies (`train.csv` of the competition) are in training with their 0 / 1
expert labels (`train.include_gold`, drawn with sampling weight 0.8), not with the table rows.

| file | trains | content | how it was made |
|---|---|---|---|
| `labels/targets_kd70f.parquet` | v76 | the predictions of our v70f ensemble (6 models of the AUC track, public LB 0.963) on every training study: distillation targets | the six full-data models predict the training studies with their Kaggle settings; equal mean (in-sample predictions, as in ensemble distillation) |
| `labels/targets_kd70f_x.parquet` | ablation D | 0.5 x `targets_kd70f` + 0.5 x the mean of six label sets that the top teams shared on Kaggle after the deadline (frontier LLMs reading the reports and the images) | `knee.labels.external_llm` from the sets below in `data/external_labels/` |
| `labels/targets_cx.parquet` | (validation reference) | our report label: for each finding the mean of LLM readings of the report (Claude Opus 5.5 for all findings; Gemma 4 12B added for effusion, Qwen3.5-35B for synovitis, a public label set for Baker's cyst); `is_gold` marks the 58 expert studies | the LLM runs are not part of this repository |
| `meta/laterality_train.parquet` | both (the orientation head's target) | the knee side of every study, `L` / `R`, and its source: the DICOM Laterality tag (2,178 studies), the series description (347), the report (606, read by an LLM), the image classifier `laterality_r18.pt` (1,276; 99.4 % held-out accuracy) | `pipeline/3_knee_side.sh` (docs/knee_side.md); the report sides come from our LLM reading of the reports, which is not part of this repository, so the released table carries them |
| `meta/joint_box_train.parquet` | both (the joint transformer's region) | the knee joint box of every study in patient mm (`lo_x` ... `hi_z`): the union of the medial, lateral and patellofemoral study boxes; the model adds a 10 mm margin | `knee.dataloading.tissue_rois --joint_boxes` (docs/tissue_model.md) |
| `meta/tissue_rois_train.parquet`, `meta/tissue_rois_train_study.parquet` | (inputs of the joint boxes) | the compartments the tissue model found in every series, and the study box of each compartment projected into every series | docs/tissue_model.md |
| `folds_v1.csv` | both | fold 0-4 of every study and fold -1 for the 58 expert studies; full-data runs (`--fold -1`) train on folds 0-4 (+ the expert studies) and validate on fold -1 | `knee.dataloading.folds` (multilabel-stratified on `targets_cx` and the scanner; the same report or patient stays in one fold) |

The label sets of the top teams in `targets_kd70f_x` (Kaggle datasets; each has its own licence, see the dataset page):
liushuzhi/rsna-knee-1st-place-training-supervision (Mix5), tascj0/rsna-2026-knee-agent-labels,
songqizhou/rsna-knee-llm-labels (file 9), chengtingyi/rsna-knee-llm-weak-labels, drizlo/rsna-knee-mri-llm-soft-labels,
matsurihime/rsna-knee-llm-readings. To rebuild the table: download them into `data/external_labels/` with the Kaggle
CLI and run `.venv/bin/python -m knee.labels.external_llm` (it also scores every set on the 58 expert studies).

## The labels behind single model 8 (not reproduced here)

v76 is the last step of a chain; the release holds the weights of the step before it (single model 8), so the chain
itself is not needed. For the record (our Kaggle writeup, section "Labels and training"): rounds 1-2
trained on public LLM label sets; rounds 3-5 on our own report readings (`targets_cx`) blended 50 / 50 with the
out-of-fold predictions of the previous round's models; single model 6 = the v76 recipe trained 48 epochs on the
round-5 labels; single model 7 = 8 more epochs on the predictions of our v56 ensemble; single model 8 = 8 more epochs on
the round-5 labels with updated Baker's cyst and synovitis labels and the 58 expert studies; v76 (single model 9) = 3
more epochs on `targets_kd70f` with the expert studies.
