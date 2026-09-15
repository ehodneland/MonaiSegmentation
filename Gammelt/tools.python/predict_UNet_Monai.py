#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import ndimage

from utils import (
    monai_unet_model,
    predict_with_model_dir_nnUNet,
    preprocess_images_with_series,
    restore_prediction_to_reference,
    save_mask,
)

# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_dir",
        required=True,
        help="Path to model directory",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Output path (folder or file without extension)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Plot input tensor slices for debugging",
    )

    for m in ["T2", "ADC", "vibe2min"]:
        parser.add_argument(
            f"--{m}",
            help=f"Path to {m} MR series",
        )

    args = parser.parse_args()

    # ------------------------------------------------------
    # Load settings.json from model directory
    # ------------------------------------------------------
    settings_path = os.path.join(args.model_dir, "settings.json")
    if not os.path.exists(settings_path):
        raise FileNotFoundError(f"Fant ikke settings.json i {args.model_dir}")

    with open(settings_path) as f:
        meta = json.load(f)

    print("\n=== SETTINGS FROM JSON ===")
    for k, v in meta.items():
        print(f"{k:20}: {v}")
    print("==========================\n")

    modalities = meta["modalities"]
    model_architecture = meta.get("model_architecture", "UNet")
    reference = meta.get("reference", modalities[0])

    print(f"Forventede modaliteter: {modalities}")
    print(f"Referansemodalitet: {reference}")
    print(f"Modellarkitektur: {model_architecture}")

    # ------------------------------------------------------
    # Validate modality arguments
    # ------------------------------------------------------
    modality_paths = {}
    for m in modalities:
        path = getattr(args, m, None)
        if path is None:
            raise ValueError(f"Du må oppgi --{m} <sti>")
        modality_paths[m] = path

    print("\nModalitetsstier:")
    for m, p in modality_paths.items():
        print(f"  {m:8}: {p}")

    modality_path_list = [modality_paths[m] for m in modalities]
    ref_idx = modalities.index(reference)

    if model_architecture == "nnUNetv2":
        print("Bruker nnUNetv2_predict via settings.json")
        pred_path = predict_with_model_dir_nnUNet(
            model_dir=args.model_dir,
            modality_path_list=modality_path_list,
            out_dir=os.path.normpath(args.out_dir),
        )
        print(f"nnUNet prediksjon lagret til: {pred_path}")
        print("\nDone!\n")
        return

    img_size = int(meta["img_size"])
    dropout = float(meta.get("dropout", 0.0))
    model_file = meta.get("model_file", "best_model.pth")
    model_path = os.path.join(args.model_dir, model_file)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Fant ikke modellfil: {model_path}")

    print(f"Modellfil: {model_file}")

    # ------------------------------------------------------
    # Preprocess inputs using Series -> TorchIO
    # ------------------------------------------------------
    preprocessed, originals, spacings, series_objects, preproc_meta = \
        preprocess_images_with_series(
            modality_path_list,
            img_size,
            reference_idx=ref_idx,
        )

    input_tensor = torch.tensor(preprocessed[None], dtype=torch.float32)

    ref_spacing = spacings[ref_idx]
    ref_series = series_objects[ref_idx]
    ref_arr = originals[ref_idx]

    # ------------------------------------------------------
    # DEBUG visualization
    # ------------------------------------------------------
    if args.debug:
        print("\nDEBUG: plotting input tensor\n")

        data = input_tensor[0].numpy()   # shape (C, D, H, W)

        C, D, H, W = data.shape

        z = D // 2
        y = H // 2
        x = W // 2

        for c in range(C):
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))

            axes[0].imshow(data[c, z], cmap="gray")
            axes[0].set_title(f"{modalities[c]} axial")

            axes[1].imshow(data[c, :, y, :], cmap="gray")
            axes[1].set_title("coronal")

            axes[2].imshow(data[c, :, :, x], cmap="gray")
            axes[2].set_title("sagittal")

            for ax in axes:
                ax.axis("off")

            plt.tight_layout()
            plt.show()

    print("\n====================  DEBUG ORIENTATION  ====================\n")
    print(f"Reference modality: {reference}")
    print(f"Original reference shape: {ref_arr.shape}")
    print(f"Original reference spacing: {ref_spacing}")
    print(f"Preprocessed shape: {preprocessed.shape}")
    print(f"Isotropic reference shape before CropOrPad: {preproc_meta['ref_iso_shape']}")
    print("\n==============================================================\n")

    # ------------------------------------------------------
    # Load model
    # ------------------------------------------------------
    model = monai_unet_model(
        in_channels=len(modalities),
        dropout=dropout,
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # ------------------------------------------------------
    # Predict
    # ------------------------------------------------------
    with torch.no_grad():
        out = model(input_tensor)
        pred = torch.sigmoid(out)[0, 0].cpu().numpy()   # shape (D, H, W)

    # ------------------------------------------------------
    # Restore prediction back to original reference geometry
    # ------------------------------------------------------
    binary_mask = restore_prediction_to_reference(
        pred,
        meta=preproc_meta,
        threshold=0.5,
    )

    voxel_volume_mm3 = np.prod(ref_spacing)
    tumorvol_ml = voxel_volume_mm3 * binary_mask.sum() / 1000.0
    print(f"Tumor size of predicted mask (ml): {tumorvol_ml:.2f}")

    # ------------------------------------------------------
    # Connected components
    # ------------------------------------------------------
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    labeled_mask, num_objects = ndimage.label(binary_mask, structure=structure)

    print(f"Number of connected objects: {num_objects}")

    if num_objects > 0:
        object_sizes_vox = ndimage.sum(
            binary_mask,
            labeled_mask,
            index=np.arange(1, num_objects + 1),
        )
        object_sizes_ml = object_sizes_vox * voxel_volume_mm3 / 1000.0

        sorted_idx = np.argsort(object_sizes_ml)[::-1]

        print("Object sizes (ml):")
        for rank, idx in enumerate(sorted_idx, start=1):
            obj_label = idx + 1
            obj_size_ml = float(object_sizes_ml[idx])
            obj_size_vox = int(object_sizes_vox[idx])
            print(
                f"  Object {rank:2d} "
                f"(label={obj_label:2d}): "
                f"{obj_size_ml:.2f} ml "
                f"({obj_size_vox} voxels)"
            )
    else:
        print("No foreground objects found in predicted mask.")

    # ------------------------------------------------------
    # Save output mask
    # ------------------------------------------------------
    out_dir = os.path.normpath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "tumor_mask")

    if binary_mask.shape != ref_arr.shape:
        raise ValueError(
            f"Mask shape {binary_mask.shape} != reference shape {ref_arr.shape}"
        )

    save_mask(binary_mask, ref_series, out_path)

    print(f"Shape of predicted mask: {binary_mask.shape}")
    print("\nDone!\n")


if __name__ == "__main__":
    main()
