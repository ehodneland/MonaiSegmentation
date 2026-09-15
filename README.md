# EC MonaiSegmentation

Active workflow for endometrial cancer tumor segmentation with MONAI-based 3D models and true nnU-Net v2.

For copy-paste command examples, see [workflow.txt](workflow.txt).

## Repository Layout

```text
tools/
  params.py
  datasetgenerator.py
  utils.py
  utils_finetuning.py
  upload_model_to_hf.py
  standardize_model_case_tables.py

training/
  train_UNet_Monai.py
  train_nnUNet_v2.py

prediction/
  predict_UNet_Monai.py
  predict_UNet_Monai_dataset.py
  evaluate_nnUNet_v2.py

models/
  <model_name>/
```

## Supported Models

`training/train_UNet_Monai.py` supports:

```text
UNet       MONAI 3D U-Net
SSM        custom state-space/Mamba-inspired 3D segmentation model
SwinUNETR  MONAI 3D SwinUNETR, with optional pretrained weights
nnUNet     nnU-Net v2-inspired model trained inside the MONAI/fastai workflow
```

`training/train_nnUNet_v2.py` supports:

```text
nnUNetv2   true nnU-Net v2 through the nnU-Net CLI
```

## Environments

Create environments:

```bash
conda env create -f environment_monai.yml
conda env create -f environment_nnunet.yml
```

Use `monai` for `UNet`, `SSM`, `SwinUNETR`, and MONAI-integrated `nnUNet`:

```bash
conda activate monai
```

Use `nnunet` for true `nnUNetv2`:

```bash
conda activate nnunet
```

## Data Configuration

Dataset paths, modality definitions, and valid-list CSV paths are configured in:

```text
tools/params.py
```

Supported project datasets:

```text
bergen
mont
hong_kong
```

Supported modalities are dataset-dependent, but current project keys are:

```text
vibe2min
T2
ADC
```

Multi-channel models use a semicolon-separated `imgpath` internally:

```text
/path/vibe2min.nii.gz;/path/T2.nii.gz;/path/ADC.nii.gz
```

The order must match the `modalities` list in `settings.json`.

## Training MONAI Models

Default training settings in `train_UNet_Monai.py` include:

```text
img_size: 192
epochs: 300
batch_size: 4
lr: 1e-3
dropout: 0.1
test_fraction: 0.2
validation_fraction: 0.1
```

Train a standard 3-channel UNet:

```bash
conda activate monai
python3 training/train_UNet_Monai.py \
  --modalities vibe2min T2 ADC \
  --reference vibe2min \
  --datasets bergen \
  --gpu 0 \
  --model_name UNet
```

Train SSM:

```bash
python3 training/train_UNet_Monai.py \
  --modalities vibe2min T2 ADC \
  --reference vibe2min \
  --datasets bergen \
  --gpu 0 \
  --model_name SSM
```

Train MONAI-integrated nnUNet:

```bash
python3 training/train_UNet_Monai.py \
  --modalities vibe2min T2 ADC \
  --reference vibe2min \
  --datasets bergen \
  --gpu 0 \
  --model_name nnUNet \
  --batch_size 3
```

## SwinUNETR

`SwinUNETR` is a 3D MONAI architecture. It is not pretrained by itself; use `--pretrained` to load pretrained weights.

Train SwinUNETR from scratch:

```bash
python3 training/train_UNet_Monai.py \
  --modalities vibe2min T2 ADC \
  --reference vibe2min \
  --datasets bergen \
  --gpu 0 \
  --model_name SwinUNETR \
  --swin_feature_size 48 \
  --batch_size 1 \
  --img_size 192 \
  --lr 1e-4
```

### BraTS21 MR-Pretrained SwinUNETR

MONAI provides SwinUNETR checkpoints trained on BraTS21 brain MRI:

```text
https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR/BRATS21
```

Source for the pretrained model used in this workflow:

```text
MONAI research-contributions / SwinUNETR / BRATS21
https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR/BRATS21
```

Direct fold 1 checkpoint archive:

```text
https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/fold1_f48_ep300_4gpu_dice0_9059.zip
```

Fold 1 can be downloaded with:

```bash
mkdir -p models/pretrained/swinunetr_brats21_fold1
wget -O /tmp/swinunetr_brats21_fold1.zip \
  https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/fold1_f48_ep300_4gpu_dice0_9059.zip
unzip /tmp/swinunetr_brats21_fold1.zip -d models/pretrained/swinunetr_brats21_fold1
find models/pretrained/swinunetr_brats21_fold1 -type f
```

The BraTS21 checkpoint uses 4 MR input channels and 3 output classes. This project may use 1-3 channels and 1 output class, so incompatible input/output weights are skipped while compatible weights are loaded.

Fine-tune from the BraTS21 checkpoint:

```bash
python3 training/train_UNet_Monai.py \
  --modalities vibe2min T2 ADC \
  --reference vibe2min \
  --datasets bergen \
  --gpu 0 \
  --model_name SwinUNETR \
  --swin_feature_size 48 \
  --pretrained models/pretrained/swinunetr_brats21_fold1 \
  --freeze_mode none \
  --batch_size 1 \
  --img_size 192 \
  --lr 3e-5 \
  --epochs 100 \
  --patience 25
```

For a conservative decoder-only test, use:

```bash
--freeze_mode decoder --lr 1e-5
```

If Dice stays near zero for many epochs, try `--freeze_mode none`; the BraTS domain is brain MRI, while this project is pelvic gynecologic MRI.

## Fine-Tuning MONAI Models

`--pretrained` accepts:

```text
local model folder
local checkpoint file
Hugging Face repo ID
Hugging Face URL
```

Available `--freeze_mode` values:

```text
none                train the whole model
head                train only final/head layers
decoder             train decoder layers
decoder_bottleneck  train decoder plus bottleneck/deep encoder stage
```

Fine-tune UNet from a local model:

```bash
python3 training/train_UNet_Monai.py \
  --modalities vibe2min T2 ADC \
  --reference vibe2min \
  --datasets mont \
  --gpu 0 \
  --model_name UNet \
  --pretrained models/UNet_vibe2min_T2_ADC_v01_bergen_20260319 \
  --freeze_mode decoder_bottleneck \
  --lr 1e-4
```

Fine-tune from Hugging Face:

```bash
python3 training/train_UNet_Monai.py \
  --modalities vibe2min T2 ADC \
  --reference vibe2min \
  --datasets mont \
  --gpu 0 \
  --model_name SSM \
  --pretrained ehodneland/SSM_vibe2min_T2_ADC_v01_bergen_20260520 \
  --freeze_mode decoder_bottleneck \
  --lr 1e-4
```

For private Hugging Face repositories, use:

```bash
--hf_token_file hf_token
```

## Training True nnU-Net v2

True nnU-Net v2 is trained through `training/train_nnUNet_v2.py` and requires the `nnunet` environment.

```bash
conda activate nnunet
python3 training/train_nnUNet_v2.py \
  --modalities vibe2min T2 ADC \
  --reference vibe2min \
  --datasets bergen \
  --gpu 0
```

The wrapper model folder contains:

```text
settings.json
case_table.csv
nnUNet_work/
  nnUNet_raw/
  nnUNet_preprocessed/
  nnUNet_results/
```

Fine-tune true nnU-Net v2 from a local or HF model with a new dataset ID:

```bash
python3 training/train_nnUNet_v2.py \
  --modalities vibe2min T2 ADC \
  --reference vibe2min \
  --datasets mont \
  --gpu 0 \
  --dataset_id 731 \
  --pretrained models/nnUNetv2_vibe2min_T2_ADC_v01_bergen_Dataset730_EC_20260518
```

True nnU-Net v2 fine-tuning uses nnU-Net's `-pretrained_weights` option and trains the full model; there is no `freeze_mode`.

## Training Outputs And Logs

Training outputs are written under:

```text
models/<model_name>/
```

For `UNet`, `SSM`, `SwinUNETR`, and MONAI-integrated `nnUNet`, typical files are:

```text
best_model.pth
settings.json
case_table.xlsx
case_table.csv
data_splits.xlsx
training.log
training_metrics.csv
training_curves.png
dice_summary.csv
```

`training.log` captures terminal output and traceback. It is overwritten when the same model folder is used again.

`training_metrics.csv` contains epoch-level values:

```text
epoch
train_loss
valid_loss
dice
```

For true `nnUNetv2`, the checkpoint is under:

```text
models/<model_name>/nnUNet_work/nnUNet_results/<DatasetID>/<trainer>__<plans>__<config>/fold_0/checkpoint_best.pth
```

## Dataset-Level Prediction And Evaluation

Use `prediction/predict_UNet_Monai_dataset.py` for dataset-level prediction/evaluation. It accepts local model folders, Hugging Face repo IDs, and Hugging Face URLs.

Choose cases with exactly one of:

```text
--datasets bergen mont
--case_table cases.xlsx
```

Local project data can usually use `--datasets`. External or portable evaluation should use `--case_table`.

The case table can be `.xlsx` or `.csv` and must contain at least:

```text
subj
```

If paths cannot be recovered from a local model case table, include:

```text
subj
imgpath
pathmask
```

For true `nnUNetv2` models read from HF, include `subj`, `imgpath`, and `pathmask`; `case_id` is optional and generated locally if missing.

Evaluate a local MONAI model:

```bash
conda activate monai
python3 prediction/predict_UNet_Monai_dataset.py \
  --model_dir models/UNet_vibe2min_T2_ADC_v01_bergen_20260319 \
  --datasets bergen \
  --gpu 0
```

Evaluate a Hugging Face model:

```bash
python3 prediction/predict_UNet_Monai_dataset.py \
  --model_dir ehodneland/UNet_vibe2min_T2_ADC_v01_bergen_20260319 \
  --case_table cases.xlsx \
  --gpu 0
```

Evaluate true nnU-Net v2:

```bash
conda activate nnunet
python3 prediction/predict_UNet_Monai_dataset.py \
  --model_dir ehodneland/nnUNetv2_vibe2min_T2_ADC_v01_bergen_Dataset730_EC_20260518 \
  --datasets bergen \
  --gpu 0
```

The wrapper detects `model_architecture: nnUNetv2` in `settings.json` and calls `prediction/evaluate_nnUNet_v2.py`.

Prediction outputs are written under:

```text
models/<model_name>/prediction/
```

Typical files:

```text
predict_UNet_Monai_dataset_results_YYYYMMDD.xlsx
predict_UNet_Monai_dataset_summary_YYYYMMDD.txt
```

The results Excel keeps input columns and overwrites/adds `dice`.

## Single-Case Prediction

Use `prediction/predict_UNet_Monai.py` for one case.

```bash
conda activate monai
python3 prediction/predict_UNet_Monai.py \
  --model_dir models/UNet_vibe2min_T2_ADC_v01_bergen_20260319 \
  --out_dir test_Bergen/ \
  --vibe2min /path/to/vibe2min.nii.gz \
  --T2 /path/to/T2.nii.gz \
  --ADC /path/to/ADC.nii.gz
```

The script reads modalities, reference modality, image size, architecture, and model file from `settings.json`.

## Hugging Face

Project models are commonly stored under:

```text
https://huggingface.co/ehodneland
```

For private repositories:

```bash
huggingface-cli login
```

or pass:

```bash
--hf_token_file hf_token
```

Upload with:

```bash
python3 tools/upload_model_to_hf.py \
  --model_dir models/UNet_vibe2min_T2_ADC_v01_bergen_20260319
```

Dry run:

```bash
python3 tools/upload_model_to_hf.py \
  --model_dir models/UNet_vibe2min_T2_ADC_v01_bergen_20260319 \
  --dry_run
```

If `--repo_id` is omitted, the repository name becomes:

```text
<authenticated-user>/<model-folder-name>
```

For `UNet`, `SSM`, `SwinUNETR`, and MONAI-integrated `nnUNet`, upload includes `settings.json` and the configured model checkpoint.

For true `nnUNetv2`, upload includes the wrapper metadata and nnU-Net model artefacts such as:

```text
settings.json
nnUNet_work/nnUNet_results/.../dataset.json
nnUNet_work/nnUNet_results/.../plans.json
nnUNet_work/nnUNet_results/.../fold_0/checkpoint_best.pth
```

Case tables are not uploaded by default for true `nnUNetv2`.

## Utility Scripts

Standardize older model case tables:

```bash
python3 tools/standardize_model_case_tables.py \
  --models_dir models
```

Upload a model:

```bash
python3 tools/upload_model_to_hf.py \
  --model_dir models/<model_name>
```

## Notes

`workflow.txt` is the quick operational command list. This README is the compact reference for supported functionality.

BraTS21 SwinUNETR pretraining is an experimental transfer-learning starting point. It is MR-based, but it is brain MRI, not pelvic gynecologic MRI. Always compare against at least:

```text
UNet from scratch
SwinUNETR from scratch
SwinUNETR with BraTS21 pretrained weights
```
