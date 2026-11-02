# The tissue segmentation model and the joint boxes

The joint transformer of the v76 recipe attends, in training, to the stage-3 cells inside a box around the knee
joint. The box comes from a tissue segmentation model that finds cartilage and menisci in every series of every study.
The release holds the model (`skm_seg3_r50rad_288.pt`) and the tables it made (docs/release.md), so this part is
optional. `REDIVIS_API_TOKEN=<token> bash pipeline/4_tissue_model.sh` runs all of it:

| step | command | what it does |
|---|---|---|
| SKM-TEA download | `python -m knee.dataloading.skm_tea download` | the DICOM track of SKM-TEA from Redivis (`aimi.skm_tea`, https://doi.org/10.71718/2ghb-nv62; a Stanford AIMI research use agreement): 155 sagittal qDESS knee scans (0.31 x 0.31 x 0.8 mm voxels, two echoes), the masks of six tissues (patellar, femoral, medial and lateral tibial cartilage, medial and lateral meniscus), 19.9 GB |
| SKM-TEA preparation | `python -m knee.dataloading.skm_tea prepare` | cache split `skm_tea`: each scan as a sagittal series and as coronal and axial reformats in 3.5 mm slabs (the slice thickness of our coronal and axial series), with the masks, in the cache frame; left knees mirrored |
| training | `python -m knee.training.seg_pretrain --config configs/seg_skm3_r50rad_288.yaml` | the ResNet-50 encoder (RadImageNet weights) with a small U-Net decoder; 24 epochs of 250 steps, batch 64 windows, soft cross-entropy + Dice; the SKM-TEA train + val splits (119 scans) train, the test split (36 scans) gives the Dice. Writes `data/models/skm_seg3_r50rad_288.pt` (and the encoder alone, `r50rad_skm_seg3_288.pt`) |
| compartments per series | `python -m knee.dataloading.tissue_rois --split train` | the model labels every slice of every series; a compartment is where its tissues are (medial tibiofemoral = medial tibial cartilage + medial meniscus, lateral the same, patellofemoral = patellar cartilage) -> `data/meta/tissue_rois_train.parquet` |
| study boxes | `python -m knee.dataloading.tissue_rois --split train --merge` | the box of each compartment in patient mm, the median over the fat-suppressed series that found it, projected into every series -> `tissue_rois_train_study.parquet` |
| joint boxes | `python -m knee.dataloading.tissue_rois --split train --joint_boxes` | the union of the three study boxes of each study -> `joint_box_train.parquet` (median 74 x 71 x 56 mm) |

The model works in the right-knee frame (left knees are mirrored before it runs), so the last three steps read the knee
sides `data/meta/laterality_train.parquet` and the series geometry `data/meta/series_geometry_train.parquet`.
Rebuilding the study and joint boxes from the released per-series table gives the released tables exactly (checked:
the joint boxes of all 4,407 studies agree to 0.0 mm).

The segmentation masks themselves are not stored: they are the model's output on each slice, recomputed by
`tissue_rois`. The tables are what training needs.
