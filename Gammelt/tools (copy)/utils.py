# ==========================================================
# Standard library
# ==========================================================
import json
import os
import shutil
import subprocess

# ==========================================================
# Numerical / data handling
# ==========================================================
import numpy as np
import pandas as pd

# ==========================================================
# Medical image IO
# ==========================================================
import nibabel as nib
from nibabel.processing import resample_from_to
from imagedata.series import Series

# ==========================================================
# PyTorch core
# ==========================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ==========================================================
# TorchIO (medical image pipelines)
# ==========================================================
import torchio as tio

# ==========================================================
# MONAI (models & losses)
# ==========================================================
from monai.networks.nets import UNet
#from monai.losses import DiceLoss

# ==========================================================
# fastai (training loop, callbacks, metrics)
# ==========================================================
from fastai.data.core import DataLoaders
from fastai.learner import Learner, Metric
from fastai.callback.tracker import EarlyStoppingCallback
#from fastai.callback.core import Callback

# ==========================================================
# Plotting / visualization
# ==========================================================
import matplotlib.pyplot as plt
try:
    from plot_utils import plot_images_and_mask
except ImportError:
    def plot_images_and_mask(*args, **kwargs):
        raise ImportError(
            "plot_images_and_mask is only available if plot_utils.py is on PYTHONPATH."
        )

# ==========================================================
# Project-specific
# ==========================================================
try:
    from . import params
except ImportError:
    import params


def load_full_model_checkpoint(model, ckpt_path, device):
    print(f"Laster full modell-checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    missing, unexpected = model.load_state_dict(
        ckpt["model"], strict=False
    )

    print("Modellvekter lastet")
    if missing:
        print("  Mangler:", missing)
    if unexpected:
        print("  Uventede:", unexpected)

def evaluate_model_and_compute_dice(model, dataloader, device):
    model.eval()
    all_dice = []

    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)
            yb = yb.to(device)

            preds = model(xb)
            dices = compute_dice_score(preds, yb, apply_threshold=True)

            all_dice.extend(dices.cpu().tolist())

    return all_dice


def save_training_plot(df_metrics, outdir):
    """
    Lager og lagrer figur med train/val loss og Dice.
    """
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # --- Loss ---
    ax1.plot(df_metrics["epoch"], df_metrics["train_loss"], label="Train loss")
    ax1.plot(df_metrics["epoch"], df_metrics["valid_loss"], label="Val loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    # finn Dice-kolonne automatisk
    dice_cols = [c for c in df_metrics.columns if "dice" in c.lower()]
    
    if len(dice_cols) == 1:
        dice_col = dice_cols[0]
        ax2 = ax1.twinx()
        ax2.plot(
            df_metrics["epoch"],
            df_metrics[dice_col],
            linestyle="--",
            label=dice_col,
        )
        ax2.set_ylabel("Dice")

        # felles legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    plt.title("Training history")
    plt.tight_layout()

    outpath = os.path.join(outdir, "training_curves.png")
    plt.savefig(outpath, dpi=200)
    plt.close(fig)

    print(f"Treningsfigur lagret til: {outpath}")


def debug_plot_first_batch(
    dataloader,
    modality_names,
    subj_id=None,
    max_items=None,
):
    """
    Plotter alle kanaler + maske for HELE første batch i en dataloader.
    Ett plot per element i batchen.
    Stopper etter plotting.
    """
    print("DEBUG: henter én batch for visualisering")

    x, y = next(iter(dataloader))  # x: (B, C, Z, Y, X), y: (B, 1, Z, Y, X)

    B = x.shape[0]
    print(f"Batch size: {B}")

    if max_items is not None:
        B = min(B, max_items)
        print(f"Plotter kun første {B} element(er)")

    for i in range(B):
        print(f"\nPlotter batch-element {i}")

        xi = x[i]          # (C, Z, Y, X)
        yi = y[i, 0]       # (Z, Y, X)

        print(f"  x[{i}] shape: {xi.shape}")
        print(f"  y[{i}] shape: {yi.shape}")
        print(f"Unike verdier i maske: {yi.unique()}")
              
        # → numpy
        xi = xi.detach().cpu().numpy()
        yi = yi.detach().cpu().numpy()

        # Ett volum per kanal
        imagelist = [xi[c] for c in range(xi.shape[0])]
        imagenames = modality_names

        plot_images_and_mask(
            imagelist=imagelist,
            imagenames=imagenames,
            mask=yi,
            maskname="Tumor mask",
            slices_to_plot_mode="largest_mask",
            transpose_order=[2, 1, 0],
            alpha=0.15,
        )

    print("\nDEBUG ferdig - avslutter")


# ----------------------------------------------------------
# Dice score
# ----------------------------------------------------------
def compute_dice_score(
    preds,
    targets,
    apply_sigmoid=True,
    apply_threshold=False,
    threshold=0.5,
    eps=1e-8,
):
    """
    Universell Dice-funksjon.

    apply_threshold = False  → differentiable (LOSS)
    apply_threshold = True   → hard Dice (METRIC / EVAL)
    """
    if apply_sigmoid:
        preds = torch.sigmoid(preds)

    if apply_threshold:
        preds = (preds > threshold).float()

    preds = preds.view(preds.shape[0], -1)
    targets = targets.float().view(targets.shape[0], -1)

    inter = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1)

    dice = (2 * inter + eps) / (union + eps)
    return dice


def save_training_artifacts(
    save_dir,
    basename,
    timestamp,
    batch_size,
    img_size,
    modalities,
    reference,
    train_df,
    val_df,
    test_df,
    model_path,
    modelname,
    dropout,
    lr,
    epochs,
    datasets,
    pretrained_model_dir="",
    freeze_mode="none",
    fold=None,
    ensemble_id=None,
):

    os.makedirs(save_dir, exist_ok=True)

    # ======================================================
    # 1) Human-readable settings (CSV) - beholdes
    # ======================================================
 #   settings = {
 #       "Timestamp": timestamp,
 #       "Basename": basename,
 #       "Datasets Used": ", ".join(datasets),
 #       "Batch Size": batch_size,
 #       "Image Size": img_size,
 #       "Modalities": ", ".join(modalities),
 #       "Train Samples": len(train_df),
 #       "Validation Samples": len(val_df),
 #       "Test Samples": len(test_df) if test_df is not None else "N/A",
 #       "Model Architecture": modelname,
 #       "Spatial Dimensions": 3,
 #       "Dropout": dropout,
 #       "Loss Function": "DiceLoss(sigmoid=True)",
 #       "Metric": "ThresholdedDice",
 #       "Optimizer": "fit_one_cycle",
 #       "Epochs": epochs,
 #       "Learning Rate": lr,
 #       "Model Path": os.path.basename(model_path),
 #   }

 #   if fold is not None:
 #       settings["Fold"] = fold

#    if ensemble_id is not None:
 #       settings["Ensemble Member"] = ensemble_id

#    settings_path = os.path.join(save_dir, "settings.csv")
#    pd.DataFrame([settings]).to_csv(settings_path, index=False, sep=";")

    # ======================================================
    # 2) Machine-readable metadata (JSON) - NY
    # ======================================================
    model_meta = {
        "model_name": basename,
        "timestamp": timestamp,

        # Dette er kontrakten for inference
        "modalities": list(modalities),
        "num_channels": len(modalities),
        "reference": reference, 
        "img_size": int(img_size),
        "dropout": float(dropout),

        # Modellinfo
        "model_architecture": modelname,
        "spatial_dims": 3,
        "out_channels": 1,
        "is_finetuned": bool(pretrained_model_dir),
        "pretrained_model_dir": pretrained_model_dir or None,
        "pretrained_model_file": (
            os.path.join(pretrained_model_dir, "best_model.pth")
            if pretrained_model_dir
            else None
        ),
        "freeze_mode": freeze_mode if pretrained_model_dir else None,

        # Trening
        "datasets": list(datasets),
        "batch_size": int(batch_size),
        "learning_rate": float(lr),
        "epochs": int(epochs),

        # Data splits
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)) if test_df is not None else None,

        # Filer
        "model_file": os.path.basename(model_path),
        "data_splits_file": "data_splits.xlsx",
    }

    if fold is not None:
        model_meta["fold"] = int(fold)

    if ensemble_id is not None:
        model_meta["ensemble_id"] = int(ensemble_id)

    meta_path = os.path.join(save_dir, "settings.json")
    with open(meta_path, "w") as f:
        json.dump(model_meta, f, indent=2)

    print(f"Metadata lagret:")
    print(f"  - {meta_path}")

    # ======================================================
    # 3) Lagre DataFrames
    # ======================================================
    splits_path = os.path.join(save_dir, "data_splits.xlsx")
    with pd.ExcelWriter(splits_path, engine="openpyxl") as writer:
        train_df.to_excel(writer, sheet_name="train", index=False)
        val_df.to_excel(writer, sheet_name="val", index=False)

        if test_df is not None:
            test_df.to_excel(writer, sheet_name="test", index=False)

    print(f"Datasplitter lagret til:")
    print(f"  - {splits_path}")

    return meta_path



def get_dataloader(
    df,
    batch_size,
    img_size,
    selected_modalities,
    reference,
    augmentation="none",
    num_workers=4,
):
    """
    Returns PyTorch DataLoader yielding:
        x: (B, C, Z, Y, X)
        y: (B, 1, Z, Y, X)
    """
    n_channels = len(selected_modalities)

    if reference not in selected_modalities:
        raise ValueError(
            f"Reference '{reference}' må være en av {selected_modalities}"
        )

    reference_idx = selected_modalities.index(reference)

    AUGMENTATION_PRESETS = {
        "none": lambda img_size: tio.Compose([
            tio.ToCanonical(),

            tio.Resample(
                target="ref",
                include=["ref", "mask"] + [f"ch{i}" for i in range(n_channels)]
            ),

            tio.Resample((1, 1, 1)),

            tio.ZNormalization(),

            tio.CropOrPad(
                (img_size, img_size, img_size),
                mask_name="mask"
            ),
        ]),
        "basic": lambda img_size: multichannel_aug(img_size, n_channels),
        "robust": lambda img_size: multichannel_aug_robust(img_size, n_channels),
    }

    if augmentation not in AUGMENTATION_PRESETS:
        raise ValueError(
            f"Ukjent augmentation='{augmentation}'. "
            f"Gyldige: {list(AUGMENTATION_PRESETS.keys())}"
        )

    transforms = AUGMENTATION_PRESETS[augmentation](img_size)
    is_training = augmentation != "none"

    subjects = []

    for imgs, mask in zip(df.imgpath.values, df.pathmask.values):
        img_files = imgs.split(";")

        if len(img_files) != len(selected_modalities):
            raise ValueError(
                f"Forventer {len(selected_modalities)} modaliteter, "
                f"men fikk {len(img_files)} for: {imgs}"
            )

        subject_dict = {
            "ref": tio.ScalarImage(img_files[reference_idx]),
            "mask": tio.LabelMap(mask),
        }

        for i, p in enumerate(img_files):
            subject_dict[f"ch{i}"] = tio.ScalarImage(p)

        subjects.append(tio.Subject(subject_dict))

    dataset = tio.SubjectsDataset(subjects, transform=transforms)

    def collate_fn(batch):
        xs, ys = [], []

        for b in batch:
            chans = [b[f"ch{i}"][tio.DATA] for i in range(len(selected_modalities))]
            x = torch.cat(chans, dim=0)      # (C, Z, Y, X)
            y = b["mask"][tio.DATA].float()  # (1, Z, Y, X)

            xs.append(x)
            ys.append(y)

        return torch.stack(xs), torch.stack(ys)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=is_training,
    )


def multichannel_aug(img_size, n_channels):
    return tio.Compose([

        tio.ToCanonical(),
        
        tio.Resample(
            target="ref",
            include=["ref", "mask"] + [f"ch{i}" for i in range(n_channels)]
        ),
        
        # 3) Gjør voxler isotrope
        tio.Resample((1,1,1)),
        
        # 3. Intensity-only (trygt)
        tio.ZNormalization(),

        # 5. Geometriske augmenteringer (samme for alle kanaler + maske)
        tio.RandomFlip(axes=(0, 1, 2), p=0.5),
        tio.RandomAffine(
            scales=(0.9, 1.1),
            translation=2,
            degrees=5,
            p=0.25,
        ),


        # 6. Intensity-augmentering (kun på bilder)
        tio.RandomGamma(log_gamma=(-0.15, 0.15), p=0.15),
        tio.RandomNoise(std=0.02, p=0.1),

        # 4. Felles spatial crop/pad
        tio.CropOrPad(
            (img_size, img_size, img_size),
            mask_name="mask"
        ),

    ])




def monai_unet_model(in_channels, dropout=0):
    """
    Oppretter en 3D U-Net fra MONAI.
    """
    return UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        dropout=dropout,   # <- brukes i MONAI UNet-blokkene
    )


def nnunet_resenc_l_model(in_channels):
    """
    Oppretter en nnU-Net v2-inspirert 3D ResEnc L-modell.

    Dette bruker samme arkitekturpakke som nnU-Net v2
    (dynamic_network_architectures), men trenes i denne scriptets eksisterende
    fastai/TorchIO-løp. Full nnU-Net v2-pipeline med plans/preprocessing kjøres
    fortsatt best via nnUNetv2 CLI.
    """
    try:
        from dynamic_network_architectures.architectures.unet import (
            ResidualEncoderUNet,
        )
    except ImportError as exc:
        raise ImportError(
            "nnunet_resenc_l krever pakken 'dynamic_network_architectures'. "
            "I Python 3.8-miljøet bør du installere arkitekturpakken direkte, "
            "for eksempel: pip install dynamic-network-architectures timm"
        ) from exc

    return ResidualEncoderUNet(
        input_channels=in_channels,
        n_stages=6,
        features_per_stage=(32, 64, 128, 256, 320, 320),
        conv_op=torch.nn.Conv3d,
        kernel_sizes=((3, 3, 3),) * 6,
        strides=((1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)),
        n_blocks_per_stage=(1, 3, 4, 6, 6, 6),
        num_classes=1,
        n_conv_per_stage_decoder=(1, 1, 1, 1, 1),
        conv_bias=True,
        norm_op=torch.nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=torch.nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        deep_supervision=False,
    )


class ConvNormAct3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SSMBottleneckBlock3D(nn.Module):
    """
    Lightweight state-space-inspired bottleneck.

    Spatial voxels are treated as a sequence at the lowest-resolution feature
    map. A depthwise 1D convolution mixes neighboring states along the sequence,
    while a gated channel projection controls the update. This keeps the model
    dependency-free while giving it long-context sequence mixing.
    """

    def __init__(self, channels, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.in_proj = nn.Linear(channels, channels * 2)
        self.sequence_mixer = nn.Conv1d(
            channels,
            channels,
            kernel_size=9,
            padding=4,
            groups=channels,
        )
        self.out_proj = nn.Linear(channels, channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        residual = x
        b, c, d, h, w = x.shape

        seq = x.flatten(2).transpose(1, 2)  # (B, N, C)
        seq = self.norm(seq)
        values, gates = self.in_proj(seq).chunk(2, dim=-1)
        values = F.silu(values)
        gates = torch.sigmoid(gates)

        mixed = values.transpose(1, 2)
        mixed = self.sequence_mixer(mixed).transpose(1, 2)
        mixed = mixed * gates
        mixed = self.dropout(self.out_proj(mixed))

        mixed = mixed.transpose(1, 2).reshape(b, c, d, h, w)
        return residual + mixed


class UpBlockSSM3D(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, dropout=0.0):
        super().__init__()
        self.up = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )
        self.conv = ConvNormAct3D(
            out_channels + skip_channels,
            out_channels,
            dropout=dropout,
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(
                x,
                size=skip.shape[2:],
                mode="trilinear",
                align_corners=False,
            )
        return self.conv(torch.cat([x, skip], dim=1))


class SSMUNet3D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=1,
        features=(24, 48, 96, 192),
        dropout=0.1,
        ssm_depth=2,
    ):
        super().__init__()
        f0, f1, f2, f3 = features

        self.encoder = nn.ModuleDict(
            {
                "stage0": ConvNormAct3D(in_channels, f0, dropout=dropout),
                "stage1": ConvNormAct3D(f0, f1, stride=2, dropout=dropout),
                "stage2": ConvNormAct3D(f1, f2, stride=2, dropout=dropout),
                "stage3": ConvNormAct3D(f2, f3, stride=2, dropout=dropout),
            }
        )
        self.bottleneck = nn.Sequential(
            *[SSMBottleneckBlock3D(f3, dropout=dropout) for _ in range(ssm_depth)]
        )
        self.decoder = nn.ModuleDict(
            {
                "up2": UpBlockSSM3D(f3, f2, f2, dropout=dropout),
                "up1": UpBlockSSM3D(f2, f1, f1, dropout=dropout),
                "up0": UpBlockSSM3D(f1, f0, f0, dropout=dropout),
            }
        )
        self.head = nn.Conv3d(f0, out_channels, kernel_size=1)

    def forward(self, x):
        s0 = self.encoder["stage0"](x)
        s1 = self.encoder["stage1"](s0)
        s2 = self.encoder["stage2"](s1)
        x = self.encoder["stage3"](s2)
        x = self.bottleneck(x)
        x = self.decoder["up2"](x, s2)
        x = self.decoder["up1"](x, s1)
        x = self.decoder["up0"](x, s0)
        return self.head(x)


def ssm_model(in_channels, dropout=0.1):
    """
    Oppretter en lett 3D SSM/Mamba-inspirert segmenteringsmodell.

    Modellen bruker samme input/output-kontrakt som UNet i train_UNet_Monai.py,
    men har sekvens-/state-space-inspirert kontekstblanding i bottleneck.
    """
    return SSMUNet3D(
        in_channels=in_channels,
        out_channels=1,
        dropout=dropout,
    )

class DiceLossBinary(torch.nn.Module):
    def forward(self, preds, targets):
        dice = compute_dice_score(
            preds,
            targets,
            apply_threshold=False,   # 
        )
        return 1.0 - dice.mean()

class DiceBCELoss(torch.nn.Module):
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight

    def forward(self, preds, targets):
        dice = compute_dice_score(
            preds, 
            targets, 
            apply_threshold=False)
        dice_loss = 1.0 - dice.mean()

        bce = F.binary_cross_entropy_with_logits(
            preds, targets.float()
        )
        return dice_loss + self.bce_weight * bce
        
class DiceMetric(Metric):
    @property
    def name(self):
        return "dice"

    def reset(self):
        self.values = []

    def accumulate(self, learn):
        dices = compute_dice_score(
            learn.pred,
            learn.y,
            apply_threshold=True,   # 
        )
        self.values.extend(dices.detach().cpu().tolist())

    @property
    def value(self):
        return float(np.mean(self.values)) if self.values else 0.0


def compute_tumor_volume(mask_path, subj=None, source=None):
    """
    Leser en binær maskefil (.nii.gz) og returnerer tumorvolum i milliliter.
    Printer advarsel hvis masken er tom.
    """
    try:
        img = nib.load(mask_path)
        data = img.get_fdata()

        voxel_volume = np.prod(img.header.get_zooms())  # mm³ per voxel
        tumor_voxels = data.sum()
        tumor_volume_ml = tumor_voxels * voxel_volume / 1000.0

        if tumor_voxels == 0:
            print(
                f"TOM MASKE "
                f"| source={source} "
                f"| subj={subj} "
                f"| path={mask_path}"
            )

        return tumor_volume_ml

    except Exception as e:
        print(
            f"FEIL VED LESING AV MASKE "
            f"| source={source} "
            f"| subj={subj} "
            f"| path={mask_path} "
            f"| error={e}"
        )
        return np.nan
    

def save_mask(mask, series_obj, outpath, formats='nifti'):

    # Put back into right format
    #mask = np.transpose(mask, axes=(2,1,0))
    
    # Overwrite pixel data
    series_obj[:] = mask.astype(series_obj.dtype)

    # Write using Series — automatically chooses DICOM or NIfTI
    series_obj.write(outpath, formats=formats)
    print(f"Saved mask to {outpath}")
    
def build_image_path(
    subj,
    modalities,
    prepathnii,
    reference,   # beholdes for API-konsistens, ikke brukt her
    dataset,
):
    """
    Bygger semikolon-separert imgpath-streng.

    Alle modaliteter lastes fra:
        <prepathnii>/<SUBJ>/unregistered/<file>

    Filnavn hentes utelukkende fra params.DATASET_CONFIG.
    """

    # ------------------------------
    # Hent datasett-konfig
    # ------------------------------
    if dataset not in params.DATASET_CONFIG:
        raise ValueError(f"Ukjent datasett: {dataset}")

    cfg = params.DATASET_CONFIG[dataset]
    mod_info = cfg["modalities"]

    # ------------------------------
    # Bygg subject-mappenavn
    # ------------------------------
    # subj kan være int (bergen) eller str (mont)
    if isinstance(subj, (int, np.integer)) or str(subj).isdigit():
        subj_id = int(subj)
    else:
        subj_id = str(subj)

    subj_str = cfg["subject_format"].format(
        prefix=cfg.get("subject_prefix", ""),
        id=subj_id,
    )

    base = os.path.join(prepathnii, subj_str, "unregistered")

    # ------------------------------
    # Bygg paths per modalitet
    # ------------------------------
    paths = []

    for m in modalities:
        if m not in mod_info:
            raise ValueError(
                f"Modalitet '{m}' finnes ikke i datasett '{dataset}'"
            )

        fname = mod_info[m]["file"]   # <- HER kommer filnavnet fra params.py
        paths.append(os.path.join(base, fname))

    return ";".join(paths)


def build_mask_path(
    subj,
    maskname,
    prepathnii,
    dataset,
):
    """
    Bygger full path til maske.

    - Maskefilnavn brukes EXACT som i CSV
    - Masken ligger i <subject>/segmented/
    - Ingen suffix, ingen automatisk registreringslogikk
    """

    # ------------------------------
    # Datasett-konfig
    # ------------------------------
    if dataset not in params.DATASET_CONFIG:
        raise ValueError(f"Ukjent datasett: {dataset}")

    cfg = params.DATASET_CONFIG[dataset]

    # ------------------------------
    # Subject-mappenavn
    # ------------------------------
    if isinstance(subj, (int, np.integer)) or str(subj).isdigit():
        subj_id = int(subj)
    else:
        subj_id = str(subj)

    subj_str = cfg["subject_format"].format(
        prefix=cfg.get("subject_prefix", ""),
        id=subj_id,
    )

    # ------------------------------
    # Full path: legg til 'segmented'
    # ------------------------------
    mask_path = os.path.join(
        prepathnii,
        subj_str,
        "segmented",
        maskname,
    )

    return mask_path


def strip_nii_suffix_nnUNet(filename):
    name = os.path.basename(str(filename))
    if name.endswith(".nii.gz"):
        return name[:-7]
    return os.path.splitext(name)[0]


def subject_folder_nnUNet(subj, dataset):
    if dataset not in params.DATASET_CONFIG:
        raise ValueError(f"Ukjent datasett: {dataset}")

    cfg = params.DATASET_CONFIG[dataset]
    if isinstance(subj, (int, np.integer)) or str(subj).isdigit():
        subj_id = int(subj)
    else:
        subj_id = str(subj)

    return cfg["subject_format"].format(
        prefix=cfg.get("subject_prefix", ""),
        id=subj_id,
    )


def build_aligned_image_path_nnUNet(
    subj,
    modalities,
    prepathnii,
    reference,
    dataset,
):
    """
    Build semicolon-separated image paths for true nnU-Net training.

    Non-reference channels use pre-aligned files from:
        <prepathnii>/<SUBJ>/registered/<modality>-2-<reference>-header.nii.gz

    The reference channel itself uses:
        <prepathnii>/<SUBJ>/unregistered/<reference-file>
    """
    if dataset not in params.DATASET_CONFIG:
        raise ValueError(f"Ukjent datasett: {dataset}")

    cfg = params.DATASET_CONFIG[dataset]
    mod_info = cfg["modalities"]
    if reference not in mod_info:
        raise ValueError(f"Reference '{reference}' finnes ikke i datasett '{dataset}'")

    subj_str = subject_folder_nnUNet(subj, dataset)
    unregistered_dir = os.path.join(prepathnii, subj_str, "unregistered")
    registered_dir = os.path.join(prepathnii, subj_str, "registered")
    ref_stem = strip_nii_suffix_nnUNet(mod_info[reference]["file"])

    paths = []
    for modality in modalities:
        if modality not in mod_info:
            raise ValueError(
                f"Modalitet '{modality}' finnes ikke i datasett '{dataset}'"
            )

        filename = mod_info[modality]["file"]
        if modality == reference:
            path = os.path.join(unregistered_dir, filename)
        else:
            modality_stem = strip_nii_suffix_nnUNet(filename)
            path = os.path.join(
                registered_dir,
                f"{modality_stem}-2-{ref_stem}-header.nii.gz",
            )

        if not os.path.exists(path):
            raise FileNotFoundError(
                "Mangler aligned nnU-Net-inputfil: "
                f"subj={subj}, dataset={dataset}, modality={modality}, "
                f"reference={reference}, path={path}"
            )
        paths.append(path)

    return ";".join(paths)


def build_aligned_mask_path_nnUNet(
    subj,
    maskname,
    prepathnii,
    reference,
    dataset,
):
    """
    Prefer a registered mask aligned to reference, with segmented/ as fallback.
    """
    if dataset not in params.DATASET_CONFIG:
        raise ValueError(f"Ukjent datasett: {dataset}")

    subj_str = subject_folder_nnUNet(subj, dataset)
    mask_stem = strip_nii_suffix_nnUNet(maskname)
    reference_file = params.DATASET_CONFIG[dataset]["modalities"][reference]["file"]
    ref_stem = strip_nii_suffix_nnUNet(reference_file)

    aligned_path = os.path.join(
        prepathnii,
        subj_str,
        "registered",
        f"{mask_stem}-2-{ref_stem}-header.nii.gz",
    )
    if os.path.exists(aligned_path):
        return aligned_path

    fallback_path = os.path.join(prepathnii, subj_str, "segmented", maskname)
    if os.path.exists(fallback_path):
        return fallback_path

    raise FileNotFoundError(
        "Mangler maskefil for nnU-Net: "
        f"subj={subj}, dataset={dataset}, reference={reference}, "
        f"forsøkte {aligned_path} og {fallback_path}"
    )


def preprocess_images_with_series(modality_path_list, img_size, reference_idx=0):
    """
    Preprocess input modalities for inference.

    Pipeline:
      1) Read all modalities with Series
      2) Resample all to the reference geometry
      3) Resample reference space to isotropic 1x1x1 mm
      4) Apply centered crop/pad to (img_size, img_size, img_size)
      5) Return stacked numpy array + metadata needed for inverse restore

    Returns
    -------
    preprocessed : np.ndarray
        Shape (C, img_size, img_size, img_size), dtype float32
    originals : list[np.ndarray]
        Original arrays for each modality, shape (z,y,x)
    spacings : list[tuple]
        Original spacings for each modality
    series_objects : list[Series]
        Original Series objects
    preproc_meta : dict
        Metadata needed to restore prediction back to original reference grid
    """
    import numpy as np
    from scipy import ndimage
    from imagedata.series import Series

    # ------------------------------------------------------
    # Helpers
    # ------------------------------------------------------
    def _get_spacing_from_series(s):
        # Adjust if your Series uses another attribute name
        if hasattr(s, "spacing"):
            return tuple(float(x) for x in s.spacing)
        if hasattr(s, "voxelspacing"):
            return tuple(float(x) for x in s.voxelspacing)
        raise AttributeError("Could not find spacing on Series object")

    def _to_numpy_zyx(s):
        arr = np.asarray(s)
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D volume, got shape {arr.shape}")
        return arr

    def _resample_to_spacing(arr, old_spacing, new_spacing, order):
        zoom_factors = tuple(old_spacing[i] / new_spacing[i] for i in range(3))
        return ndimage.zoom(arr, zoom=zoom_factors, order=order)

    def _resample_to_shape(arr, target_shape, order):
        zoom_factors = tuple(target_shape[i] / arr.shape[i] for i in range(3))
        return ndimage.zoom(arr, zoom=zoom_factors, order=order)

    def _center_crop_or_pad(arr, target_shape):
        """
        Returns:
            out           : target array
            source_start  : start index in source arr
            source_end    : end index in source arr
            insert_start  : start index in target out
            insert_end    : end index in target out
        """
        out = np.zeros(target_shape, dtype=arr.dtype)

        source_start = []
        source_end = []
        insert_start = []
        insert_end = []

        for i in range(3):
            src = arr.shape[i]
            tgt = target_shape[i]

            if src >= tgt:
                s0 = (src - tgt) // 2
                s1 = s0 + tgt
                t0 = 0
                t1 = tgt
            else:
                s0 = 0
                s1 = src
                t0 = (tgt - src) // 2
                t1 = t0 + src

            source_start.append(s0)
            source_end.append(s1)
            insert_start.append(t0)
            insert_end.append(t1)

        out[
            insert_start[0]:insert_end[0],
            insert_start[1]:insert_end[1],
            insert_start[2]:insert_end[2],
        ] = arr[
            source_start[0]:source_end[0],
            source_start[1]:source_end[1],
            source_start[2]:source_end[2],
        ]

        return (
            out,
            tuple(source_start),
            tuple(source_end),
            tuple(insert_start),
            tuple(insert_end),
        )

    def _z_norm(x):
        x = x.astype(np.float32)
        m = x.mean()
        s = x.std()
        if s < 1e-8:
            return x - m
        return (x - m) / s

    # ------------------------------------------------------
    # Read originals
    # ------------------------------------------------------
    series_objects = [Series(p) for p in modality_path_list]
    originals = [_to_numpy_zyx(s) for s in series_objects]
    spacings = [_get_spacing_from_series(s) for s in series_objects]

    ref_arr = originals[reference_idx]
    ref_spacing = spacings[reference_idx]

    # ------------------------------------------------------
    # Build isotropic reference volume (1,1,1 mm)
    # ------------------------------------------------------
    iso_spacing = (1.0, 1.0, 1.0)
    ref_iso = _resample_to_spacing(
        ref_arr,
        old_spacing=ref_spacing,
        new_spacing=iso_spacing,
        order=1,
    )
    ref_iso_shape = tuple(int(x) for x in ref_iso.shape)

    # ------------------------------------------------------
    # Center crop/pad isotropic reference to model input size
    # ------------------------------------------------------
    target_shape = (img_size, img_size, img_size)
    ref_iso_cropped, source_start, source_end, insert_start, insert_end = _center_crop_or_pad(
        ref_iso,
        target_shape,
    )

    # ------------------------------------------------------
    # Process all modalities:
    #   1) resample to reference shape
    #   2) resample to isotropic ref_iso_shape
    #   3) apply same crop/pad window
    # ------------------------------------------------------
    processed_channels = []

    for arr, spacing in zip(originals, spacings):
        # First map modality to reference voxel grid
        if arr.shape != ref_arr.shape:
            arr_ref = _resample_to_shape(arr, ref_arr.shape, order=1)
        else:
            arr_ref = arr

        # Then map to isotropic reference grid
        arr_iso = _resample_to_spacing(
            arr_ref,
            old_spacing=ref_spacing,
            new_spacing=iso_spacing,
            order=1,
        )

        # Safety: force exact same isotropic shape as reference
        if arr_iso.shape != ref_iso_shape:
            arr_iso = _resample_to_shape(arr_iso, ref_iso_shape, order=1)

        # Apply same crop/pad as reference
        arr_crop = np.zeros(target_shape, dtype=np.float32)
        arr_crop[
            insert_start[0]:insert_end[0],
            insert_start[1]:insert_end[1],
            insert_start[2]:insert_end[2],
        ] = arr_iso[
            source_start[0]:source_end[0],
            source_start[1]:source_end[1],
            source_start[2]:source_end[2],
        ]

        arr_crop = _z_norm(arr_crop)
        processed_channels.append(arr_crop.astype(np.float32))

    preprocessed = np.stack(processed_channels, axis=0).astype(np.float32)

    preproc_meta = {
        "ref_original_shape": tuple(int(x) for x in ref_arr.shape),
        "ref_original_spacing": tuple(float(x) for x in ref_spacing),
        "ref_iso_shape": tuple(int(x) for x in ref_iso_shape),
        "iso_spacing": iso_spacing,
        "target_shape": target_shape,
        "source_start": tuple(int(x) for x in source_start),
        "source_end": tuple(int(x) for x in source_end),
        "insert_start": tuple(int(x) for x in insert_start),
        "insert_end": tuple(int(x) for x in insert_end),
    }

    return preprocessed, originals, spacings, series_objects, preproc_meta

def restore_prediction_to_reference(pred, meta, threshold=0.5):
    import numpy as np
    from scipy import ndimage

    pred_bin = (pred >= threshold).astype(np.uint8)

    ref_iso_shape = tuple(meta["ref_iso_shape"])
    ref_original_shape = tuple(meta["ref_original_shape"])

    source_start = tuple(meta["source_start"])
    source_end = tuple(meta["source_end"])
    insert_start = tuple(meta["insert_start"])
    insert_end = tuple(meta["insert_end"])

    # ------------------------------------------------------
    # Step 1: reconstruct full isotropic reference volume
    # ------------------------------------------------------
    full_iso = np.zeros(ref_iso_shape, dtype=np.uint8)

    full_iso[
        source_start[0]:source_end[0],
        source_start[1]:source_end[1],
        source_start[2]:source_end[2],
    ] = pred_bin[
        insert_start[0]:insert_end[0],
        insert_start[1]:insert_end[1],
        insert_start[2]:insert_end[2],
    ]

    # ------------------------------------------------------
    # Step 2: resample isotropic prediction back to original ref shape
    # ------------------------------------------------------
    zoom_factors = tuple(
        ref_original_shape[i] / ref_iso_shape[i] for i in range(3)
    )

    restored = ndimage.zoom(full_iso, zoom=zoom_factors, order=0)

    # Safety: force exact original shape
    restored_fixed = np.zeros(ref_original_shape, dtype=np.uint8)

    z = min(restored.shape[0], ref_original_shape[0])
    y = min(restored.shape[1], ref_original_shape[1])
    x = min(restored.shape[2], ref_original_shape[2])

    restored_fixed[:z, :y, :x] = restored[:z, :y, :x]

    return restored_fixed.astype(np.uint8)    


def format_dataset_name_nnUNet(dataset_id, name):
    return f"Dataset{int(dataset_id):03d}_{name}"


def derive_dataset_id_nnUNet(dataset_name, modalities, datasets, base_id=501):
    text = "|".join(
        [
            str(dataset_name),
            ",".join(sorted(str(m) for m in modalities)),
            ",".join(sorted(str(d) for d in datasets)),
        ]
    )
    offset = sum(ord(ch) for ch in text) % 400
    return int(base_id) + offset


def case_id_from_row_nnUNet(row, row_index):
    source = str(row.get("source", "case")).replace(" ", "_").replace("-", "_")
    subj = str(row["subj"]).replace(" ", "_").replace("-", "_")
    return f"{source}_{subj}_{int(row_index):04d}"


def get_nnunet_paths_nnUNet(base_dir):
    base_dir = os.path.abspath(base_dir)
    return {
        "nnUNet_raw": os.path.join(base_dir, "nnUNet_raw"),
        "nnUNet_preprocessed": os.path.join(base_dir, "nnUNet_preprocessed"),
        "nnUNet_results": os.path.join(base_dir, "nnUNet_results"),
    }


def get_nnunet_env_nnUNet(base_dir):
    paths = get_nnunet_paths_nnUNet(base_dir)
    env = os.environ.copy()
    env.update(paths)
    return env, paths


def require_nnunet_cli_nnUNet():
    missing = [
        cmd for cmd in (
            "nnUNetv2_plan_and_preprocess",
            "nnUNetv2_train",
            "nnUNetv2_predict",
        )
        if shutil.which(cmd) is None
    ]
    if missing:
        raise RuntimeError(
            "Fant ikke ekte nnU-Net v2 CLI i miljøet: "
            + ", ".join(missing)
            + ". Installer/kjør i et Python >=3.10-miljø med nnunetv2."
        )


def copy_or_link_file_nnUNet(src, dst, mode="symlink"):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.lexists(dst):
        os.remove(dst)

    if mode == "symlink":
        os.symlink(os.path.abspath(src), dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError("mode må være 'symlink' eller 'copy'")


def write_binary_label_nnUNet(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    img = nib.load(src)
    data = (img.get_fdata() > 0).astype(np.uint8)
    out = nib.Nifti1Image(data, img.affine, img.header)
    out.set_data_dtype(np.uint8)
    nib.save(out, dst)


def same_geometry_nnUNet(src, reference):
    src_img = nib.load(src)
    ref_img = nib.load(reference)
    return (
        tuple(src_img.shape[:3]) == tuple(ref_img.shape[:3])
        and np.allclose(src_img.affine, ref_img.affine, atol=1e-3)
    )


def resample_to_reference_nnUNet(src, dst, reference, is_label=False):
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    src_img = nib.load(src)
    ref_img = nib.load(reference)
    target = (ref_img.shape[:3], ref_img.affine)
    order = 0 if is_label else 1

    if same_geometry_nnUNet(src, reference):
        resampled = src_img
    else:
        resampled = resample_from_to(src_img, target, order=order)

    if is_label:
        data = (resampled.get_fdata() > 0).astype(np.uint8)
        out = nib.Nifti1Image(data, ref_img.affine, ref_img.header.copy())
        out.set_data_dtype(np.uint8)
    else:
        data = resampled.get_fdata(dtype=np.float32).astype(np.float32)
        out = nib.Nifti1Image(data, ref_img.affine, ref_img.header.copy())
        out.set_data_dtype(np.float32)

    out.set_qform(ref_img.affine, code=1)
    out.set_sform(ref_img.affine, code=1)
    nib.save(out, dst)


def write_dataset_json_nnUNet(dataset_dir, modalities, num_training):
    dataset_json = {
        "channel_names": {str(i): m for i, m in enumerate(modalities)},
        "labels": {
            "background": 0,
            "tumor": 1,
        },
        "numTraining": int(num_training),
        "file_ending": ".nii.gz",
    }
    outpath = os.path.join(dataset_dir, "dataset.json")
    with open(outpath, "w") as f:
        json.dump(dataset_json, f, indent=2)
    return outpath


def write_splits_final_nnUNet(preprocessed_dataset_dir, train_cases, val_cases):
    os.makedirs(preprocessed_dataset_dir, exist_ok=True)
    outpath = os.path.join(preprocessed_dataset_dir, "splits_final.json")
    with open(outpath, "w") as f:
        json.dump(
            [
                {
                    "train": list(train_cases),
                    "val": list(val_cases),
                }
            ],
            f,
            indent=2,
        )
    return outpath


def create_raw_dataset_nnUNet(
    dftrain,
    dftest,
    modalities,
    dataset_id,
    dataset_name,
    nnunet_base_dir,
    file_mode="symlink",
    reference_idx=0,
    resample_to_reference=True,
):
    paths = get_nnunet_paths_nnUNet(nnunet_base_dir)
    dataset_folder_name = format_dataset_name_nnUNet(dataset_id, dataset_name)
    dataset_dir = os.path.join(paths["nnUNet_raw"], dataset_folder_name)
    images_tr = os.path.join(dataset_dir, "imagesTr")
    labels_tr = os.path.join(dataset_dir, "labelsTr")
    images_ts = os.path.join(dataset_dir, "imagesTs")
    labels_ts = os.path.join(dataset_dir, "labelsTs")

    for d in (images_tr, labels_tr, images_ts, labels_ts):
        os.makedirs(d, exist_ok=True)

    records = []
    train_cases = []
    val_cases = []
    test_cases = []

    def add_case(row, row_index, split, images_dir, labels_dir):
        case_id = case_id_from_row_nnUNet(row, row_index)
        image_paths = str(row["imgpath"]).split(";")
        if len(image_paths) != len(modalities):
            raise ValueError(
                f"Forventer {len(modalities)} modaliteter for {case_id}, "
                f"men fikk {len(image_paths)}"
            )

        reference_src = image_paths[int(reference_idx)]
        if not os.path.exists(reference_src):
            raise FileNotFoundError(
                f"Mangler referansefil for {case_id}: {reference_src}"
            )

        for channel_idx, src in enumerate(image_paths):
            if not os.path.exists(src):
                raise FileNotFoundError(f"Mangler bildefil for {case_id}: {src}")
            dst = os.path.join(images_dir, f"{case_id}_{channel_idx:04d}.nii.gz")
            if resample_to_reference:
                resample_to_reference_nnUNet(
                    src,
                    dst,
                    reference=reference_src,
                    is_label=False,
                )
            else:
                copy_or_link_file_nnUNet(src, dst, mode=file_mode)

        mask_src = row["pathmask"]
        if not os.path.exists(mask_src):
            raise FileNotFoundError(f"Mangler maskefil for {case_id}: {mask_src}")
        label_dst = os.path.join(labels_dir, f"{case_id}.nii.gz")
        if resample_to_reference:
            resample_to_reference_nnUNet(
                mask_src,
                label_dst,
                reference=reference_src,
                is_label=True,
            )
        else:
            write_binary_label_nnUNet(mask_src, label_dst)

        records.append(
            {
                "case_id": case_id,
                "split": split,
                "source": row.get("source", ""),
                "subj": row["subj"],
                "imgpath": row["imgpath"],
                "pathmask": row["pathmask"],
            }
        )
        return case_id

    for idx, row in dftrain.reset_index(drop=True).iterrows():
        split = "val" if bool(row.get("isval", False)) else "train"
        case_id = add_case(row, idx, split, images_tr, labels_tr)
        if split == "val":
            val_cases.append(case_id)
        else:
            train_cases.append(case_id)

    for idx, row in dftest.reset_index(drop=True).iterrows():
        case_id = add_case(row, idx, "test", images_ts, labels_ts)
        test_cases.append(case_id)

    write_dataset_json_nnUNet(dataset_dir, modalities, len(train_cases) + len(val_cases))

    case_table = pd.DataFrame(records)
    case_table_path = os.path.join(dataset_dir, "case_table.csv")
    case_table.to_csv(case_table_path, index=False, sep=";")

    return {
        "dataset_folder_name": dataset_folder_name,
        "dataset_dir": dataset_dir,
        "case_table_path": case_table_path,
        "train_cases": train_cases,
        "val_cases": val_cases,
        "test_cases": test_cases,
        **paths,
    }


def run_command_nnUNet(cmd, env=None, cwd=None):
    print("Kjører:", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True, env=env, cwd=cwd)


def model_results_dir_nnUNet(meta):
    return os.path.join(
        meta["nnunet_results"],
        meta["nnunet_dataset_name"],
        f"{meta['nnunet_trainer']}__{meta['nnunet_plans']}__{meta['nnunet_configuration']}",
        f"fold_{meta['nnunet_fold']}",
    )


def resolve_pretrained_checkpoint_nnUNet(pretrained, checkpoint_name="checkpoint_best.pth"):
    pretrained = os.path.abspath(pretrained)

    if os.path.isfile(pretrained):
        return pretrained

    if not os.path.isdir(pretrained):
        raise ValueError(f"--pretrained må være en modellmappe eller checkpoint-fil: {pretrained}")

    direct_checkpoint = os.path.join(pretrained, checkpoint_name)
    if os.path.isfile(direct_checkpoint):
        return direct_checkpoint

    settings_path = os.path.join(pretrained, "settings.json")
    if os.path.isfile(settings_path):
        with open(settings_path) as f:
            meta = json.load(f)
        result_checkpoint = os.path.join(
            model_results_dir_nnUNet(meta),
            checkpoint_name,
        )
        if os.path.isfile(result_checkpoint):
            return result_checkpoint

    matches = []
    for root, _, files in os.walk(pretrained):
        if checkpoint_name in files:
            matches.append(os.path.join(root, checkpoint_name))

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(
            f"Fant flere '{checkpoint_name}' under {pretrained}. "
            "Pek --pretrained direkte til ønsket checkpoint-fil."
        )

    raise FileNotFoundError(
        f"Fant ikke '{checkpoint_name}' fra --pretrained={pretrained}. "
        "Pek til wrappermodellmappe med settings.json, nnU-Net resultmappe, "
        "eller direkte til checkpoint-filen."
    )


def write_settings_nnUNet(
    save_dir,
    model_name,
    dataset_id,
    dataset_folder_name,
    modalities,
    reference,
    datasets,
    configuration,
    planner,
    plans,
    trainer,
    fold,
    nnunet_base_dir,
    raw_dataset_dir,
    case_table_path,
    pretrained_model_dir="",
    pretrained_checkpoint="",
):
    os.makedirs(save_dir, exist_ok=True)
    env_paths = get_nnunet_paths_nnUNet(nnunet_base_dir)
    meta = {
        "model_name": model_name,
        "model_architecture": "nnUNetv2",
        "prediction_backend": "nnUNetv2",
        "modalities": list(modalities),
        "num_channels": len(modalities),
        "reference": reference,
        "datasets": list(datasets),
        "nnunet_dataset_id": int(dataset_id),
        "nnunet_dataset_name": dataset_folder_name,
        "nnunet_configuration": configuration,
        "nnunet_planner": planner,
        "nnunet_plans": plans,
        "nnunet_trainer": trainer,
        "nnunet_fold": str(fold),
        "nnunet_base_dir": os.path.abspath(nnunet_base_dir),
        "nnunet_raw": env_paths["nnUNet_raw"],
        "nnunet_preprocessed": env_paths["nnUNet_preprocessed"],
        "nnunet_results": env_paths["nnUNet_results"],
        "nnunet_raw_dataset_dir": raw_dataset_dir,
        "case_table_file": os.path.basename(case_table_path),
        "is_finetuned": bool(pretrained_checkpoint),
        "pretrained_model_dir": pretrained_model_dir or None,
        "pretrained_checkpoint": pretrained_checkpoint or None,
    }
    outpath = os.path.join(save_dir, "settings.json")
    with open(outpath, "w") as f:
        json.dump(meta, f, indent=2)

    shutil.copy2(case_table_path, os.path.join(save_dir, os.path.basename(case_table_path)))
    return outpath


def prepare_prediction_case_nnUNet(modality_path_list, input_dir, case_id="PRED_000", file_mode="symlink"):
    os.makedirs(input_dir, exist_ok=True)
    for channel_idx, src in enumerate(modality_path_list):
        if not os.path.isfile(src):
            raise FileNotFoundError(
                "Ekte nnU-Net-prediksjon krever NIfTI-filer som input. "
                f"Fant ikke fil: {src}"
            )
        dst = os.path.join(input_dir, f"{case_id}_{channel_idx:04d}.nii.gz")
        copy_or_link_file_nnUNet(src, dst, mode=file_mode)
    return case_id


def predict_with_model_dir_nnUNet(model_dir, modality_path_list, out_dir, file_mode="symlink"):
    import tempfile

    settings_path = os.path.join(model_dir, "settings.json")
    with open(settings_path) as f:
        meta = json.load(f)

    require_nnunet_cli_nnUNet()
    env = os.environ.copy()
    env["nnUNet_raw"] = meta["nnunet_raw"]
    env["nnUNet_preprocessed"] = meta["nnunet_preprocessed"]
    env["nnUNet_results"] = meta["nnunet_results"]

    os.makedirs(out_dir, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nnunet_predict_") as tmpdir:
        input_dir = os.path.join(tmpdir, "input")
        case_id = prepare_prediction_case_nnUNet(
            modality_path_list,
            input_dir=input_dir,
            case_id="PRED_000",
            file_mode=file_mode,
        )

        cmd = [
            "nnUNetv2_predict",
            "-i", input_dir,
            "-o", out_dir,
            "-d", str(meta["nnunet_dataset_id"]),
            "-c", meta["nnunet_configuration"],
            "-tr", meta["nnunet_trainer"],
            "-p", meta["nnunet_plans"],
            "-f", str(meta["nnunet_fold"]),
        ]
        run_command_nnUNet(cmd, env=env)

    pred_path = os.path.join(out_dir, f"{case_id}.nii.gz")
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"nnUNetv2_predict laget ikke forventet fil: {pred_path}")
    return pred_path
    
