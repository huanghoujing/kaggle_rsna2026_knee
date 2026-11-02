# The left or right knee

v76 is never told whether a study shows a left or a right knee. The images are used as acquired, random mirroring is
part of the augmentation, and the orientation head learns for every window where inferior and where medial point.
"Medial" depends on the side, so the training target of the orientation head needs the side of every training study:
`data/meta/laterality_train.parquet` (columns `StudyInstanceUID`, `side` = `L` / `R`, `source`). Nothing at test time
needs the side. The release holds the table, so this part is optional: `bash pipeline/3_knee_side.sh` trains the
classifier (when `data/models/laterality_r18.pt` is absent, or with `RETRAIN=1`) and rebuilds the table.

Where the side of the 4,407 training studies comes from, in this order:

| source | studies | how |
|---|---|---|
| DICOM Laterality tag | 2,178 | the tag of the series, when every series of the study agrees |
| series description | 347 | `LT` / `RT` / `LEFT` / `RIGHT` in the series descriptions |
| report | 606 | the side an LLM read in the radiology report; our LLM run is not part of this repository, so these rows are copied from the released table |
| image classifier | 1,276 | the ResNet-18 below on the studies without a tag, description or report side |

Why a classifier: a rule on the knee position in the scanner (the knee center on the patient's left-right axis) is
98-99 % right for Siemens and Philips scanners and unreliable for GE, while the images themselves show the side (the
fibula lies on the lateral side). The classifier agrees with the position rule on most position-only studies and is
99.4 % right on held-out studies with a known side.

## The classifier

`python -m knee.model.laterality_cnn --out data/models/laterality_r18.pt`

- Input: the middle slice of every coronal and axial series of the study, from the image cache (docs/data.md): a
  320 px center crop (160 mm) resized to 224 px, the same slice in the three channels.
- Model: ResNet-18 (timm `resnet18.a1_in1k`, ImageNet weights), one logit (left).
- Labels: the studies with a tag, description or report side; 20 % of them (626 studies) held out (fixed draw).
- Training: 6 epochs, batch 64, AdamW 3e-4 with a one-cycle schedule, bf16; augmentation that keeps the side (a
  small rotation, scale and shift, an intensity factor); no mirroring, since mirroring changes the side.
- A study's prediction is the mean logit of its slices. Outputs: the model and `laterality_r18_train_pred.parquet`
  (`StudyInstanceUID`, `logit_left`, `pred`) for every training study; the script prints the held-out accuracy and the
  agreement with the position rule (when the input table has position-only rows). `--predict_only` writes the
  predictions of an existing model. Training takes under a minute on one GPU; the retrained classifier agrees with
  the released table on all 1,276 classifier-sourced studies.

## The table

`python -m knee.dataloading.laterality --out data/meta/laterality_train.parquet --cnn_pred data/models/laterality_r18_train_pred.parquet`

reads the DICOM header table (`data/meta/train_dicom_meta.parquet`) for the tag and the descriptions, the report sides
from an existing table (`--report_sides`, default the output path when it exists; the rows with `source` = `report`) or
from an LLM table (`--llm`, a column `laterality` with `left` / `right`), and fills the rest from the classifier's
predictions (without `--cnn_pred`: the position rule, `source` = `xpos`).
