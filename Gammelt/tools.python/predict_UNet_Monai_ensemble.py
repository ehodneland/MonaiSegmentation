#!/usr/bin/env python3
#!/usr/bin/env python3
"""
predict_UNet_Monai_ensemble.py
==============================

Inference script for 3D UNet ensemble segmentation using MONAI.

This script:
- Loads an ensemble of trained 3D UNet models (fold-based)
- Automatically reads inference settings from each model folder
- Detects whether input images are DICOM folders or NIfTI files
- Co-registers all input modalities to a reference modality
- Applies the same preprocessing used during training (CropOrPad only)
- Runs ensemble inference (binary predictions)
- Saves:
    * One binary mask per ensemble member
    * One binary ensemble median mask
    * One vote-map (number of models voting foreground)
- Writes ALL outputs in the SAME FORMAT as the input (DICOM or NIfTI)
- Logs all steps to both screen and log file

IMPORTANT ASSUMPTIONS
---------------------
- All models were trained with TorchIO CropOrPad (no resampling)
- Modalities are listed in settings.csv under the column "Modalities"
- The FIRST modality is used as the registration reference
- Series.align(reference) returns a NEW Series object (not in-place)
- Voxel geometry is preserved throughout the pipeline

INPUTS
------
--model_root : Directory containing fold model subdirectories
--outdir     : Output directory (all results are written here)
--T2         : Path to T2 image (NIfTI file or DICOM folder)
--ADC        : Path to ADC image (optional, depends on model)
--vibe2min   : Path to VIBE image (optional, depends on model)

OUTPUTS
-------
For each ensemble member:
    <model_folder_name>_pred.(nii.gz or DICOM folder)

Ensemble outputs:
    median_mask.(nii.gz or DICOM folder)
    vote_map.(nii.gz or DICOM folder)

A log file is written to:
    <outdir>/ensemble_predict_YYYYMMDD_HHMMSS.log

USAGE EXAMPLE
-------------
NIfTI input:
    python predict_UNet_Monai_ensemble.py \
        --model_root /path/to/models \
        --outdir /path/to/output \
        --T2 image_T2.nii.gz \
        --ADC image_ADC.nii.gz \
        --vibe2min image_vibe.nii.gz

DICOM input:
    python predict_UNet_Monai_ensemble.py \
        --model_root /path/to/models \
        --outdir /path/to/output \
        --T2 /path/to/DICOM/T2 \
        --ADC /path/to/DICOM/ADC \
        --vibe2min /path/to/DICOM/VIBE

NOTES
-----
- This script is intended for inference only.
- No resampling or interpolation is performed.
- All masks are binary unless explicitly stated otherwise.
- The vote-map encodes agreement across ensemble members.

Author: Erlend Hodneland / MMIV
"""
import os
import argparse
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from imagedata.series import Series

from utils import (
    monai_unet_model,
    preprocess_images,
    inverse_crop_or_pad,
    load_modality,
    save_mask,
)

def detect_input_format(path):
    """
    Returnerer 'dicom' eller 'nifti'
    """
    if os.path.isdir(path):
        return "dicom"

    ext = os.path.splitext(path)[1].lower()
    if ext in [".nii", ".gz"]:
        return "nifti"

    raise RuntimeError(f"Ukjent input-format: {path}")

# ==========================================================
# Logger
# ==========================================================
def setup_logger(logfile):
    logger = logging.getLogger("ensemble_predict")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)

    fh = logging.FileHandler(logfile)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger


# ==========================================================
# Load ensemble models + settings
# ==========================================================
def load_ensemble(model_root, logger, csv_name="settings.csv", device="cpu"):
    logger.info("▶ Loading ensemble models")
    logger.info(f"Model root: {model_root}")

    model_dirs = sorted([
        os.path.join(model_root, d)
        for d in os.listdir(model_root)
        if os.path.isdir(os.path.join(model_root, d))
    ])

    if len(model_dirs) == 0:
        raise RuntimeError("Fant ingen modellmapper")

    logger.info(f"Found {len(model_dirs)} model folders")

    models = []
    model_names = []

    settings_ref = None
    modalities_ref = None
    img_size_ref = None
    dropout_ref = None

    for i, d in enumerate(model_dirs, start=1):
        logger.info(f"▶ Fold {i}/{len(model_dirs)}")

        model_path = os.path.join(d, "best_model.pth")
        csv_path = os.path.join(d, csv_name)

        if not os.path.exists(model_path):
            raise RuntimeError(f"Mangler best_model.pth i {d}")
        if not os.path.exists(csv_path):
            raise RuntimeError(f"Mangler {csv_name} i {d}")

        logger.info(f"  Model file: {model_path}")
        logger.info(f"  CSV file  : {csv_path}")

        settings_df = pd.read_csv(csv_path, sep=";")

        img_size = int(settings_df["Image Size"].values[0])
        dropout = float(settings_df["Dropout"].values[0])
        modalities = [m.strip() for m in settings_df["Modalities"].values[0].split(",")]
        in_channels = len(modalities)

        logger.info("  Settings:")
        logger.info(f"    Image Size : {img_size}")
        logger.info(f"    Dropout    : {dropout}")
        logger.info(f"    Modalities : {modalities}")

        if settings_ref is None:
            settings_ref = settings_df
            modalities_ref = modalities
            img_size_ref = img_size
            dropout_ref = dropout
        else:
            if modalities != modalities_ref:
                raise RuntimeError("Modalities mismatch mellom fold")
            if img_size != img_size_ref:
                raise RuntimeError("Image Size mismatch mellom fold")
            if dropout != dropout_ref:
                raise RuntimeError("Dropout mismatch mellom fold")

        model = monai_unet_model(
            in_channels=in_channels,
            dropout=dropout
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        models.append(model)
        model_names.append(os.path.basename(d))

        logger.info("  Model loaded")

    logger.info("Ensemble loading complete")
    logger.info(f"Effective settings:")
    logger.info(f"  Modalities : {modalities_ref}")
    logger.info(f"  Image Size : {img_size_ref}")
    logger.info(f"  Dropout    : {dropout_ref}")
    logger.info(f"  Ensemble   : {len(models)} models")

    return models, model_names, settings_ref, modalities_ref


# ==========================================================
# MAIN
# ==========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_root", required=True,
                        help="Folder containing fold model directories")
    parser.add_argument("--outdir", required=True,
                        help="Output directory (all files written here)")
    for m in ["vibe2min", "ADC", "T2"]:
        parser.add_argument(f"--{m}", help=f"Path to {m} image")

    args = parser.parse_args()

    # ------------------------------------------------------
    # Output dir + logger
    # ------------------------------------------------------
    os.makedirs(args.outdir, exist_ok=True)

    logfile = os.path.join(
        args.outdir,
        f"ensemble_predict_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    logger = setup_logger(logfile)

    logger.info("▶ Starting ensemble prediction")
    logger.info(f"Output directory: {args.outdir}")

    # ------------------------------------------------------
    # Load ensemble
    # ------------------------------------------------------
    models, model_names, settings_df, modalities = load_ensemble(
        args.model_root, logger
    )
    img_size = int(settings_df["Image Size"].values[0])

    # ------------------------------------------------------
    # Validate modality paths
    # ------------------------------------------------------
    logger.info("▶ Validating modality inputs")
    modality_paths = {}
    for m in modalities:
        path = getattr(args, m, None)
        if path is None:
            raise RuntimeError(f"Mangler --{m}")
        modality_paths[m] = path
        logger.info(f"  {m}: {path}")

    # ------------------------------------------------------
    # Detect input format (once)
    # ------------------------------------------------------
    image_format = detect_input_format(modality_paths[modalities[0]])    
    logger.info(f"Detected image format: {image_format}")
    
    # ------------------------------------------------------
    # Load modalities as Series
    # ------------------------------------------------------
    logger.info("▶ Loading modalities as Series")    
    series_dict = {}    
    for m in modalities:
        path = modality_paths[m]
        series = Series(path)
        series_dict[m] = series
        logger.info(f"  Loaded {m}: shape={series[:].shape}, spacing={series.header.spacing}")

    # Reference image is the first in the list
    ref_modality = modalities[0]
    im_ref = series_dict[ref_modality]
    logger.info(f"▶ Reference modality: {ref_modality}")

    # ------------------------------------------------------
    # Co-register images
    # ------------------------------------------------------
    logger.info("▶ Co-registering modalities")
    for m in modalities:
        if m == ref_modality:
            continue    
        logger.info(f"  Aligning {m} → {ref_modality}")    
        aligned = series_dict[m].align(im_ref, force=True)   # align RETURNERER ny Series
        series_dict[m] = aligned                 # <-- KRITISK LINJE    
        logger.info(
            f"    After align: shape={aligned[:].shape}, "
            f"spacing={aligned.header.spacing}"
        )


    
    # ------------------------------------------------------
    # Extract arrays
    # ------------------------------------------------------
    arrays = []
    spacings = []
    series_objects = []    
    for m in modalities:
        im = series_dict[m]
        arr = im[:]                          # (Z,Y,X)
        spacing = tuple(im.header.spacing)    
        arrays.append(arr)
        spacings.append(spacing)
        series_objects.append(im)
        logger.info(f"  Using {m}: shape={arr.shape}")

    # ------------------------------------------------------
    # Sanity check of dimensions
    # ------------------------------------------------------
    shapes = [im[:].shape for im in series_dict.values()]
    if len(set(shapes)) != 1:
        logger.warning("Shapes differ after alignment!")
    
    spacings = [tuple(im.header.spacing) for im in series_dict.values()]
    if len(set(spacings)) != 1:
        logger.warning("Spacing differ after alignment!")
    
    # ------------------------------------------------------
    # Preprocess
    # ------------------------------------------------------
    logger.info("▶ Preprocessing images")
    preprocessed, originals = preprocess_images(arrays, img_size)
    logger.info(f"  Preprocessed shape: {preprocessed.shape}")

    # preprocessed: (C, Z, Y, X)
    #import nibabel as nib
    #data_4d = np.transpose(preprocessed, (3, 2, 1, 0))  # -> (X, Y, Z, C)
    #print(data_4d.shape)    
    #nii = nib.Nifti1Image(data_4d.astype(np.float32), affine=np.eye(4))
    #outpath = os.path.join(args.outdir, "debug_preprocessed_4d.nii.gz")
    #nib.save(nii, outpath)    
    #logger.info(f"Saved 4D NIfTI: {outpath}")
    input_tensor = torch.tensor(preprocessed[None], dtype=torch.float32)

    view_imgs = False
    if view_imgs:
        import matplotlib.pyplot as plt    
        logger.info("▶ Plotting mid-slice from each channel (post-preprocess)")        
        C, Z, Y, X = preprocessed.shape
        z_mid = Z // 2        
        fig, axes = plt.subplots(1, C, figsize=(5 * C, 5))        
        if C == 1:
            axes = [axes]        
        for c in range(C):
            ax = axes[c]
            img = preprocessed[c, z_mid, :, :]        
            ax.imshow(img, cmap="gray")
            ax.set_title(f"Channel {c}: {modalities[c]}")
            ax.axis("off")    
        plt.tight_layout()        
        # Vis på skjerm
        plt.show()    
        logger.info("Channel slice plot shown")

    # ------------------------------------------------------
    # Ensemble prediction
    # ------------------------------------------------------
    logger.info("▶ Running ensemble prediction")

    preds = []
    with torch.no_grad():
        for i, model in enumerate(models, start=1):
            logger.info(f"  Forward pass model {i}/{len(models)}")
            p = torch.sigmoid(model(input_tensor))[0, 0]  # (D,H,W)
            preds.append(p)

    stacked = torch.stack(preds, dim=0)  # (N,D,H,W)

    # Binary predictions
    binary_preds = (stacked > 0.5).to(torch.uint8)

    # Vote-map
    vote_map = binary_preds.sum(dim=0)

    # Ensemble median (binary)
    median_prob = torch.median(stacked, dim=0).values
    median_mask = (median_prob > 0.5).to(torch.uint8)

    # ------------------------------------------------------
    # Resize to original geometry
    # ------------------------------------------------------
    logger.info("▶ Resizing predictions to original space")
    orig_shape = originals[0].shape

    median_resized = inverse_crop_or_pad(
        median_mask.cpu().numpy(),
        orig_shape,
    ).astype(np.uint8)

    vote_resized = inverse_crop_or_pad(
        vote_map.cpu().numpy(),
        orig_shape,
    ).astype(np.uint8)

    indiv_resized = []
    for i in range(len(models)):
        r = inverse_crop_or_pad(
            binary_preds[i].cpu().numpy(),
            orig_shape,
        ).astype(np.uint8)
        indiv_resized.append(r)

    # ------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------
    logger.info("▶ Saving outputs")
    
    # Individual models
    for name, mask in zip(model_names, indiv_resized):
        outpath = os.path.join(args.outdir, f"{name}_pred")
        save_mask(mask, series_objects[0], outpath, formats=[image_format])
        logger.info(f"  Saved individual mask: {outpath} (voxels={mask.sum()})")

    # Ensemble median
    median_path = os.path.join(args.outdir, "median_mask")
    save_mask(median_resized, series_objects[0], median_path, formats=[image_format])
    logger.info(f"  Saved ensemble median mask: {median_path} (voxels={median_resized.sum()})")

    # Vote-map
    vote_path = os.path.join(args.outdir, "vote_map")
    save_mask(vote_resized, series_objects[0], vote_path, formats=[image_format])
    logger.info(
        f"  Saved vote-map: {vote_path} "
        f"(range={vote_resized.min()}-{vote_resized.max()})"
    )

    logger.info("Ensemble prediction finished successfully")


if __name__ == "__main__":
    main()
